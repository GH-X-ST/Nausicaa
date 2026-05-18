from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import (
    as_normalised_command_vector,
    clip_normalised_command,
    normalised_command_to_surface_rad,
    surface_rad_to_normalised_command,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from glide_primitive import build_glide_primitive_spec
from logging_contract import command_dataframe, metric_dataframe, trajectory_dataframe
from metric_contract import empty_metric_row, validate_metric_row
from primitive_contract import (
    PrimitiveEntrySet,
    PrimitiveExitCheck,
    PrimitiveSpec,
    validate_primitive_spec,
)
from primitive_interface import (
    EntryCheckResult,
    ExitCheckResult,
    evaluate_entry_set,
    evaluate_exit_checks,
)
from result_paths import make_result_tree
from rollout import RolloutResult, rk4_step
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector
from trim_solver import TrimResult, TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and data containers
# 2) State, trim, feedback, and handoff helpers
# 3) Bank case construction
# 4) Closed-loop bank rollout and batch execution
# 5) Output writing
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
PRIMITIVE_FAMILY = "bank"
BANK_CAMPAIGN = "06_bank_primitive"
BANK_NOTES = "bank_w0_mild_lateral_repositioning_feedback"
GLIDE_PROXY_SOURCE = "build_glide_primitive_spec + evaluate_entry_set"
INITIAL_POSITION_W_M = (1.30, 2.20, 1.80)
VALIDATION_COMMANDS = (
    "python -m py_compile "
    "03_Control/03_Primitives/bank_primitive.py "
    "03_Control/04_Scenarios/run_bank_primitive_w0.py",
    "python 03_Control/04_Scenarios/run_bank_primitive_w0.py "
    "--run-id 1 --overwrite",
    "python -m pytest -q "
    "03_Control/tests/test_bank_primitive.py "
    "03_Control/tests/test_bank_primitive_w0.py",
    "python -m pytest -q 03_Control/tests",
)


@dataclass(frozen=True)
class BankFeedbackGains:
    k_phi: float = 1.30
    k_p: float = 0.30
    k_theta: float = 0.45
    k_q: float = 0.15
    k_speed: float = 0.06
    k_r: float = 0.18
    k_beta: float = 0.60


@dataclass(frozen=True)
class BankPrimitiveConfig:
    dt_s: float = 0.02
    t_final_s: float = 0.60
    speed_m_s: float = 6.5
    altitude_m: float = 1.8
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    seed: int = 1
    scenario_name: str = "bank_w0_batch"
    target_bank_deg: float = 10.0
    min_lateral_displacement_m: float = 0.05


@dataclass(frozen=True)
class BankCaseSpec:
    name: str
    role: str
    description: str
    direction_sign: int
    x0: np.ndarray
    t_final_s: float


@dataclass(frozen=True)
class BankCheckResult:
    name: str
    value: float | bool | str
    limit: float | str
    pass_check: bool
    required: bool
    units: str
    reason: str


@dataclass(frozen=True)
class BankCaseResult:
    case_spec: BankCaseSpec
    primitive_spec: PrimitiveSpec
    time_s: np.ndarray
    x: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    entry_checks: tuple[EntryCheckResult, ...]
    exit_checks: tuple[ExitCheckResult, ...]
    bank_checks: tuple[BankCheckResult, ...]
    metrics: dict[str, object]
    success: bool
    failure_label: str
    notes: str


@dataclass(frozen=True)
class BankBatchResult:
    primitive_spec: PrimitiveSpec
    case_results: tuple[BankCaseResult, ...]
    overall_status: str
    required_case_success: bool
    optional_failures: tuple[str, ...]
    diagnostic_failures: tuple[str, ...]
    failure_label: str
    notes: str


# =============================================================================
# 2) State, Trim, Feedback, and Handoff Helpers
# =============================================================================
def _time_step_count(dt_s: float, t_final_s: float) -> int:
    ratio = float(t_final_s) / float(dt_s)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, rtol=1e-12, atol=1e-9):
        raise ValueError("t_final_s must be an integer multiple of dt_s.")
    return rounded


def _time_grid(dt_s: float, t_final_s: float) -> np.ndarray:
    step_count = _time_step_count(dt_s, t_final_s)
    return np.arange(step_count + 1, dtype=float) * float(dt_s)


def _validate_config(config: BankPrimitiveConfig) -> None:
    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("bank dt_s must be finite and positive.")
    if not np.isfinite(float(config.t_final_s)) or float(config.t_final_s) <= 0.0:
        raise ValueError("bank t_final_s must be finite and positive.")
    _time_step_count(config.dt_s, config.t_final_s)
    if not np.isfinite(float(config.speed_m_s)) or float(config.speed_m_s) <= 0.0:
        raise ValueError("bank speed_m_s must be finite and positive.")
    if not np.isfinite(float(config.altitude_m)):
        raise ValueError("bank altitude_m must be finite.")
    if config.wind_mode != "none":
        raise ValueError("W0 bank supports only wind_mode='none'.")
    if config.latency_case != "none":
        raise ValueError("W0 bank supports only latency_case='none'.")
    tau = np.asarray(config.actuator_tau_s, dtype=float)
    if tau.size != 3 or not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("actuator_tau_s must contain three finite positive values.")
    if not np.isfinite(float(config.target_bank_deg)) or config.target_bank_deg <= 0.0:
        raise ValueError("target_bank_deg must be finite and positive.")
    if not np.isfinite(float(config.min_lateral_displacement_m)) or (
        config.min_lateral_displacement_m <= 0.0
    ):
        raise ValueError("min_lateral_displacement_m must be finite and positive.")
    if not config.scenario_name:
        raise ValueError("scenario_name must be nonempty.")


def _default_aircraft() -> object:
    return adapt_glider(build_nausicaa_glider())


def _solve_trim(config: BankPrimitiveConfig, aircraft: object) -> TrimResult:
    return solve_straight_trim(
        aircraft,
        TrimTarget(
            speed_m_s=float(config.speed_m_s),
            altitude_m=float(config.altitude_m),
            wind_model=None,
            wind_mode="none",
            actuator_tau_s=config.actuator_tau_s,
        ),
    )


def _speed_alpha_beta_rad(x_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_b = np.asarray(x_log, dtype=float)[:, 6:9]
    speed = np.linalg.norm(velocity_b, axis=1)
    alpha_rad = np.arctan2(velocity_b[:, 2], velocity_b[:, 0])
    beta_rad = np.zeros_like(speed)
    valid = speed > 1e-9
    beta_rad[valid] = np.arcsin(np.clip(velocity_b[valid, 1] / speed[valid], -1.0, 1.0))
    return speed, alpha_rad, beta_rad


def _state_speed_beta(state: np.ndarray) -> tuple[float, float]:
    vector = as_state_vector(state)
    speed = float(np.linalg.norm(vector[6:9]))
    beta_rad = 0.0
    if speed > 1e-9:
        beta_rad = float(np.arcsin(np.clip(vector[STATE_INDEX["v"]] / speed, -1.0, 1.0)))
    return speed, beta_rad


def _set_velocity_from_speed_angles(
    state: np.ndarray,
    speed_m_s: float,
    alpha_rad: float,
    beta_rad: float,
) -> np.ndarray:
    updated = as_state_vector(state).copy()
    updated[STATE_INDEX["u"]] = speed_m_s * np.cos(alpha_rad) * np.cos(beta_rad)
    updated[STATE_INDEX["v"]] = speed_m_s * np.sin(beta_rad)
    updated[STATE_INDEX["w"]] = speed_m_s * np.sin(alpha_rad) * np.cos(beta_rad)
    return updated


def _safe_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _safe_min(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _margin_series(x_log: np.ndarray) -> dict[str, np.ndarray]:
    rows = [
        position_margin_m(position_w, TRUE_SAFE_BOUNDS)
        for position_w in np.asarray(x_log, dtype=float)[:, 0:3]
        if np.all(np.isfinite(position_w))
    ]
    if not rows:
        return {
            "min_wall_margin_m": np.array([np.nan]),
            "floor_margin_m": np.array([np.nan]),
            "ceiling_margin_m": np.array([np.nan]),
            "min_margin_m": np.array([np.nan]),
            "x_min_margin_m": np.array([np.nan]),
            "x_max_margin_m": np.array([np.nan]),
        }
    return {
        "min_wall_margin_m": np.array([row["min_wall_margin_m"] for row in rows]),
        "floor_margin_m": np.array([row["floor_margin_m"] for row in rows]),
        "ceiling_margin_m": np.array([row["ceiling_margin_m"] for row in rows]),
        "min_margin_m": np.array([row["min_margin_m"] for row in rows]),
        "x_min_margin_m": np.array([row["x_min_margin_m"] for row in rows]),
        "x_max_margin_m": np.array([row["x_max_margin_m"] for row in rows]),
    }


def _saturation_metrics(
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    dt_s: float,
) -> tuple[float, float]:
    clipped = np.any(
        np.abs(np.asarray(u_norm_requested) - np.asarray(u_norm_applied)) > 1e-12,
        axis=1,
    )
    if clipped.size <= 1:
        return 0.0, 0.0
    interval_clipped = clipped[:-1]
    count = int(np.count_nonzero(interval_clipped))
    return float(count / interval_clipped.size), float(count * dt_s)


def _glide_entry_failures(x_terminal: np.ndarray) -> tuple[EntryCheckResult, ...]:
    glide_spec = build_glide_primitive_spec()
    return tuple(
        check
        for check in evaluate_entry_set(as_state_vector(x_terminal), glide_spec.entry_set)
        if not check.pass_check
    )


def terminal_glide_entry_proxy(
    x_terminal: np.ndarray,
    glide_spec: PrimitiveSpec | None = None,
) -> bool:
    """Return exact terminal handoff compatibility with the W0 glide entry set."""

    try:
        state = as_state_vector(x_terminal)
    except ValueError:
        return False
    spec = build_glide_primitive_spec() if glide_spec is None else glide_spec
    try:
        checks = evaluate_entry_set(state, spec.entry_set)
    except ValueError:
        return False
    return bool(all(check.pass_check for check in checks))


def terminal_glide_entry_x_margin_m(
    x_terminal: np.ndarray,
    glide_spec: PrimitiveSpec | None = None,
) -> float:
    """Return terminal margin to the existing W0 glide x upper entry bound."""

    state = as_state_vector(x_terminal)
    spec = build_glide_primitive_spec() if glide_spec is None else glide_spec
    return float(spec.entry_set.upper["x_w"] - state[STATE_INDEX["x_w"]])


def terminal_true_safe_x_margin_m(x_terminal: np.ndarray) -> float:
    """Return terminal signed margin to the closest true-safe x face."""

    margins = position_margin_m(as_state_vector(x_terminal)[0:3], TRUE_SAFE_BOUNDS)
    return float(min(margins["x_min_margin_m"], margins["x_max_margin_m"]))


def bank_reference_bank_rad(
    t_s: float,
    case: BankCaseSpec,
    config: BankPrimitiveConfig,
) -> float:
    """Return the signed W0 bank-reference profile in radians."""

    _validate_config(config)
    t = float(t_s)
    if t <= 0.12:
        profile = 0.5 - 0.5 * np.cos(np.pi * t / 0.12)
    elif t <= 0.40:
        profile = 1.0
    elif t <= 0.60:
        profile = 0.5 + 0.5 * np.cos(np.pi * (t - 0.40) / 0.20)
    else:
        profile = 0.0
    target_bank_rad = np.deg2rad(float(config.target_bank_deg))
    return float(int(case.direction_sign) * target_bank_rad * profile)


def bank_feedback_command_norm(
    x: np.ndarray,
    x_ref: np.ndarray,
    bank_ref_rad: float,
    u_trim_norm: np.ndarray,
    gains: BankFeedbackGains,
) -> np.ndarray:
    """Return requested normalised aggregate command for W0 bank feedback."""

    state = as_state_vector(x)
    reference = as_state_vector(x_ref)
    trim_norm = as_normalised_command_vector(u_trim_norm)
    speed, beta_rad = _state_speed_beta(state)
    speed_ref, _ = _state_speed_beta(reference)
    correction = np.array(
        [
            gains.k_phi * (float(bank_ref_rad) - state[STATE_INDEX["phi"]])
            - gains.k_p * state[STATE_INDEX["p"]],
            -gains.k_theta * (state[STATE_INDEX["theta"]] - reference[STATE_INDEX["theta"]])
            - gains.k_q * state[STATE_INDEX["q"]]
            - gains.k_speed * (speed_ref - speed),
            -gains.k_r * state[STATE_INDEX["r"]] - gains.k_beta * beta_rad,
        ],
        dtype=float,
    )
    requested = trim_norm + correction
    if not np.all(np.isfinite(requested)):
        raise ValueError("bank feedback produced a nonfinite command.")
    return requested


# =============================================================================
# 3) Bank Case Construction
# =============================================================================
def build_bank_primitive_spec(
    config: BankPrimitiveConfig | None = None,
) -> PrimitiveSpec:
    """Return the W0 bank primitive metadata contract."""

    cfg = BankPrimitiveConfig() if config is None else config
    _validate_config(cfg)
    entry_set = PrimitiveEntrySet(
        name="bank_w0_entry",
        description="W0 no-wind bank entry set in SI units and radians",
        lower={
            "x_w": 1.25,
            "y_w": 0.3,
            "z_w": 0.8,
            "speed_m_s": 5.5,
            "alpha_rad": -0.15,
            "beta_rad": -0.18,
            "phi": -0.30,
            "theta": -0.25,
            "p": -1.0,
            "q": -1.0,
            "r": -1.0,
        },
        upper={
            "x_w": 1.35,
            "y_w": 4.1,
            "z_w": 2.6,
            "speed_m_s": 7.5,
            "alpha_rad": 0.25,
            "beta_rad": 0.18,
            "phi": 0.30,
            "theta": 0.25,
            "p": 1.0,
            "q": 1.0,
            "r": 1.0,
        },
    )
    exit_checks = (
        PrimitiveExitCheck(
            name="finite_state",
            description="full bank state history remains finite",
            required=True,
        ),
        PrimitiveExitCheck(
            name="true_safe_margin",
            description="minimum true-safety margin stays nonnegative",
            required=True,
        ),
        PrimitiveExitCheck(
            name="rollout_success",
            description="closed-loop bank rollout integrity metric reports success",
            required=True,
        ),
    )
    spec = PrimitiveSpec(
        name="bank_w0_nominal",
        family=PRIMITIVE_FAMILY,
        duration_s=float(cfg.t_final_s),
        entry_set=entry_set,
        exit_checks=exit_checks,
        metadata={
            "actual_bank_primitive_implemented": "true",
            "bank_updraft_encounter_role": "w0_lateral_repositioning_baseline_only",
            "terminal_glide_entry_proxy_source": GLIDE_PROXY_SOURCE,
            "wind_mode": cfg.wind_mode,
        },
    )
    validate_primitive_spec(spec)
    return spec


def _base_bank_state(trim_result: TrimResult, config: BankPrimitiveConfig) -> np.ndarray:
    if not trim_result.converged:
        raise ValueError("cannot build bank cases from nonconverged trim.")
    state = as_state_vector(trim_result.x_trim).copy()
    state[0:3] = [
        INITIAL_POSITION_W_M[0],
        INITIAL_POSITION_W_M[1],
        float(config.altitude_m),
    ]
    return state


def build_bank_cases(
    config: BankPrimitiveConfig | None = None,
    trim_result: TrimResult | None = None,
) -> tuple[BankCaseSpec, ...]:
    """Return deterministic W0 bank/updraft-encounter proxy case inventory."""

    cfg = BankPrimitiveConfig() if config is None else config
    _validate_config(cfg)
    trim = trim_result
    if trim is None:
        trim = _solve_trim(cfg, _default_aircraft())
    base = _base_bank_state(trim, cfg)

    left = base.copy()
    right = base.copy()

    left_sideslip = _set_velocity_from_speed_angles(base, 6.4, 0.07, -0.08)
    left_sideslip[STATE_INDEX["r"]] = -0.20
    right_sideslip = _set_velocity_from_speed_angles(base, 6.4, 0.07, 0.08)
    right_sideslip[STATE_INDEX["r"]] = 0.20

    boundary = right.copy()
    return (
        BankCaseSpec(
            name="bank_w0_left_mild",
            role="required",
            description="required W0 left-bank mild lateral repositioning case",
            direction_sign=-1,
            x0=left,
            t_final_s=float(cfg.t_final_s),
        ),
        BankCaseSpec(
            name="bank_w0_right_mild",
            role="required",
            description="required W0 right-bank mild lateral repositioning case",
            direction_sign=1,
            x0=right,
            t_final_s=float(cfg.t_final_s),
        ),
        BankCaseSpec(
            name="bank_w0_left_sideslip_entry",
            role="optional",
            description="optional left-bank case with modest sideslip entry",
            direction_sign=-1,
            x0=left_sideslip,
            t_final_s=float(cfg.t_final_s),
        ),
        BankCaseSpec(
            name="bank_w0_right_sideslip_entry",
            role="optional",
            description="optional right-bank case with modest sideslip entry",
            direction_sign=1,
            x0=right_sideslip,
            t_final_s=float(cfg.t_final_s),
        ),
        BankCaseSpec(
            name="bank_w0_0p80_handoff_boundary",
            role="diagnostic",
            description="diagnostic 0.80 s handoff-boundary case",
            direction_sign=1,
            x0=boundary,
            t_final_s=0.80,
        ),
    )


# =============================================================================
# 4) Closed-Loop Bank Rollout and Batch Execution
# =============================================================================
def _single_row_arrays(
    x0: np.ndarray,
    u_trim_norm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    requested = np.zeros(3) if u_trim_norm is None else as_normalised_command_vector(u_trim_norm)
    applied = clip_normalised_command(requested)
    command_rad = normalised_command_to_surface_rad(applied)
    return (
        np.array([0.0]),
        as_state_vector(x0).reshape(1, STATE_SIZE),
        requested.reshape(1, 3),
        applied.reshape(1, 3),
        command_rad.reshape(1, 3),
    )


def _rollout_metrics(
    spec: PrimitiveSpec,
    case: BankCaseSpec,
    config: BankPrimitiveConfig,
    time_s: np.ndarray,
    x_log: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    rollout_failure_label: str,
    final_failure_label: str,
    primitive_success: bool,
    notes: str,
) -> dict[str, object]:
    row = empty_metric_row(include_agile=True)
    speed, alpha_rad, beta_rad = _speed_alpha_beta_rad(x_log)
    margins = _margin_series(x_log)
    rate_norm = np.linalg.norm(x_log[:, 9:12], axis=1)
    saturation_fraction, saturation_time_s = _saturation_metrics(
        u_norm_requested,
        u_norm_applied,
        float(config.dt_s),
    )
    finite_state_success = bool(np.all(np.isfinite(x_log)))
    rollout_success = bool(
        finite_state_success
        and rollout_failure_label == "not_run"
        and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in x_log[:, 0:3])
    )
    row.update(
        {
            "run_id": "",
            "seed": int(config.seed),
            "primitive_name": spec.name,
            "primitive_family": spec.family,
            "scenario_name": case.name,
            "wind_mode": config.wind_mode,
            "latency_case": config.latency_case,
            "success": bool(primitive_success),
            "finite_state_success": finite_state_success,
            "rollout_success": rollout_success,
            "primitive_success": bool(primitive_success),
            "closed_loop_replay_success": bool(primitive_success),
            "failure_label": final_failure_label,
            "duration_s": float(time_s[-1] - time_s[0]) if time_s.size else 0.0,
            "initial_speed_m_s": float(speed[0]) if speed.size else float("nan"),
            "terminal_speed_m_s": float(speed[-1]) if speed.size else float("nan"),
            "height_change_m": float(x_log[-1, 2] - x_log[0, 2]),
            "min_true_wall_margin_m": _safe_min(margins["min_wall_margin_m"]),
            "min_floor_margin_m": _safe_min(margins["floor_margin_m"]),
            "min_ceiling_margin_m": _safe_min(margins["ceiling_margin_m"]),
            "max_alpha_deg": _safe_max(np.abs(np.rad2deg(alpha_rad))),
            "max_beta_deg": _safe_max(np.abs(np.rad2deg(beta_rad))),
            "max_bank_deg": _safe_max(np.abs(np.rad2deg(x_log[:, STATE_INDEX["phi"]]))),
            "max_pitch_deg": _safe_max(np.abs(np.rad2deg(x_log[:, STATE_INDEX["theta"]]))),
            "max_rate_rad_s": _safe_max(rate_norm),
            "saturation_fraction": saturation_fraction,
            "notes": notes,
            "saturation_time_s": saturation_time_s,
        }
    )
    validate_metric_row(row)
    return row


def _rollout_result(
    time_s: np.ndarray,
    x_log: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    delta_cmd_rad: np.ndarray,
    metrics: dict[str, object],
    rollout_failure_label: str,
    notes: str,
) -> RolloutResult:
    return RolloutResult(
        time_s=time_s,
        x=x_log,
        u_norm_requested=u_norm_requested,
        u_norm_applied=u_norm_applied,
        delta_cmd_rad=delta_cmd_rad,
        success=bool(metrics["rollout_success"]),
        failure_label=rollout_failure_label,
        metrics=metrics,
        notes=notes,
    )


def _check(
    name: str,
    value: float | bool | str,
    limit: float | str,
    pass_check: bool,
    units: str,
    reason: str,
    required: bool = True,
) -> BankCheckResult:
    return BankCheckResult(
        name=name,
        value=value,
        limit=limit,
        pass_check=bool(pass_check),
        required=bool(required),
        units=units,
        reason=reason,
    )


def _bank_checks(
    case: BankCaseSpec,
    config: BankPrimitiveConfig,
    x_log: np.ndarray,
    metrics: dict[str, object],
    glide_spec: PrimitiveSpec,
) -> tuple[BankCheckResult, ...]:
    speed, alpha_rad, beta_rad = _speed_alpha_beta_rad(x_log)
    rate_norm = np.linalg.norm(x_log[:, 9:12], axis=1)
    phi_deg = np.rad2deg(x_log[:, STATE_INDEX["phi"]])
    signed_peak_bank_deg = float(phi_deg[np.argmax(np.abs(phi_deg))])
    peak_bank_abs_deg = float(np.max(np.abs(phi_deg)))
    terminal_bank_abs_deg = float(abs(phi_deg[-1]))
    terminal_rate_rad_s = float(rate_norm[-1])
    lateral_displacement_m = float(x_log[-1, STATE_INDEX["y_w"]] - x_log[0, STATE_INDEX["y_w"]])
    expected_lateral_sign = int(case.direction_sign)
    terminal_x = float(x_log[-1, STATE_INDEX["x_w"]])
    true_x_margin = terminal_true_safe_x_margin_m(x_log[-1])
    x_margin = terminal_glide_entry_x_margin_m(x_log[-1], glide_spec)
    terminal_proxy = terminal_glide_entry_proxy(x_log[-1], glide_spec)
    margins = _margin_series(x_log)
    true_safe_pass = bool(_safe_min(margins["min_margin_m"]) >= 0.0)
    return (
        _check(
            "peak_bank_sign",
            signed_peak_bank_deg,
            expected_lateral_sign,
            np.sign(signed_peak_bank_deg) == expected_lateral_sign,
            "deg",
            "peak_bank_sign_matches_case_direction",
        ),
        _check(
            "peak_bank_min_abs_deg",
            peak_bank_abs_deg,
            5.0,
            peak_bank_abs_deg >= 5.0,
            "deg",
            "bank_response_reaches_minimum_proxy_magnitude",
        ),
        _check(
            "peak_bank_max_abs_deg",
            peak_bank_abs_deg,
            12.0,
            peak_bank_abs_deg <= 12.0,
            "deg",
            "bank_response_stays_mild",
        ),
        _check(
            "terminal_bank_abs_deg",
            terminal_bank_abs_deg,
            8.0,
            terminal_bank_abs_deg <= 8.0,
            "deg",
            "terminal_bank_unloaded_enough_for_glide_handoff_proxy",
        ),
        _check(
            "terminal_rate_boundary",
            terminal_rate_rad_s,
            0.7,
            terminal_rate_rad_s <= 0.7,
            "rad/s",
            "terminal_body_rate_norm_within_bank_handoff_bound",
        ),
        _check(
            "max_alpha_boundary",
            float(metrics["max_alpha_deg"]),
            20.0,
            float(metrics["max_alpha_deg"]) <= 20.0,
            "deg",
            "trajectory_alpha_within_w0_bank_bound",
        ),
        _check(
            "max_beta_boundary",
            float(metrics["max_beta_deg"]),
            15.0,
            float(metrics["max_beta_deg"]) <= 15.0,
            "deg",
            "trajectory_beta_within_w0_bank_bound",
        ),
        _check(
            "max_rate_boundary",
            float(metrics["max_rate_rad_s"]),
            2.0,
            float(metrics["max_rate_rad_s"]) <= 2.0,
            "rad/s",
            "trajectory_rate_within_w0_bank_bound",
        ),
        _check(
            "saturation_fraction",
            float(metrics["saturation_fraction"]),
            0.20,
            float(metrics["saturation_fraction"]) <= 0.20,
            "-",
            "requested_commands_do_not_excessively_clip",
        ),
        _check(
            "lateral_displacement_sign",
            lateral_displacement_m,
            expected_lateral_sign,
            np.sign(lateral_displacement_m) == expected_lateral_sign,
            "m",
            "terminal_lateral_displacement_matches_case_direction",
        ),
        _check(
            "lateral_displacement_min_abs_m",
            abs(lateral_displacement_m),
            float(config.min_lateral_displacement_m),
            abs(lateral_displacement_m) >= float(config.min_lateral_displacement_m),
            "m",
            "bank_proxy_repositions_laterally_by_minimum_amount",
        ),
        _check(
            "true_safe_trajectory",
            true_safe_pass,
            "true",
            true_safe_pass,
            "-",
            "full_trajectory_inside_true_safe_bounds",
        ),
        _check(
            "terminal_x_w_m",
            terminal_x,
            float(glide_spec.entry_set.upper["x_w"]),
            terminal_x <= float(glide_spec.entry_set.upper["x_w"]),
            "m",
            "terminal_x_against_existing_glide_entry_upper_bound",
        ),
        _check(
            "terminal_true_safe_x_margin_m",
            true_x_margin,
            0.0,
            true_x_margin >= 0.0,
            "m",
            "positive_margin_means_terminal_x_inside_true_safe_volume",
        ),
        _check(
            "terminal_glide_entry_x_margin_m",
            x_margin,
            0.0,
            x_margin >= 0.0,
            "m",
            "positive_margin_means_terminal_x_inside_existing_glide_entry",
        ),
        _check(
            "terminal_glide_entry_proxy",
            terminal_proxy,
            "true",
            terminal_proxy,
            "-",
            "terminal_state_matches_existing_glide_w0_nominal_entry_set",
        ),
        _check(
            "terminal_glide_entry_proxy_source",
            GLIDE_PROXY_SOURCE,
            GLIDE_PROXY_SOURCE,
            True,
            "-",
            "handoff_proxy_uses_existing_glide_spec_and_entry_evaluator",
        ),
        _check(
            "diagnostic_case",
            case.role == "diagnostic",
            "diagnostic",
            True,
            "-",
            "case_role_recorded_for_acceptance_gate",
            required=False,
        ),
    )


def _first_failed_bank_label(checks: tuple[BankCheckResult, ...]) -> str:
    mapping = {
        "max_alpha_boundary": "alpha_boundary",
        "max_beta_boundary": "beta_boundary",
        "terminal_rate_boundary": "rate_boundary",
        "max_rate_boundary": "rate_boundary",
        "saturation_fraction": "actuator_saturation_limited",
        "true_safe_trajectory": "true_safety_violation",
        "terminal_true_safe_x_margin_m": "true_safety_violation",
    }
    for check in checks:
        if check.required and not check.pass_check:
            return mapping.get(check.name, "terminal_recovery_limited")
    return "success"


def _boundary_x_limited_only(case: BankCaseSpec, x_log: np.ndarray) -> bool:
    if case.name != "bank_w0_0p80_handoff_boundary":
        return False
    finite = bool(np.all(np.isfinite(x_log)))
    true_safe = bool(all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in x_log[:, 0:3]))
    failures = _glide_entry_failures(x_log[-1])
    return bool(finite and true_safe and failures and {failure.variable for failure in failures} == {"x_w"})


def rollout_bank_case(
    case: BankCaseSpec,
    config: BankPrimitiveConfig | None = None,
    gains: BankFeedbackGains | None = None,
    aircraft: object | None = None,
    wind_model: object = None,
) -> BankCaseResult:
    """Run one W0 bank case with local feedback and exact glide handoff check."""

    cfg = BankPrimitiveConfig() if config is None else config
    _validate_config(cfg)
    if wind_model is not None:
        raise ValueError("W0 bank requires wind_model=None.")
    feedback_gains = BankFeedbackGains() if gains is None else gains
    aircraft_model = _default_aircraft() if aircraft is None else aircraft
    spec = build_bank_primitive_spec(cfg)
    state0 = as_state_vector(case.x0)
    entry_checks = evaluate_entry_set(state0, spec.entry_set)
    entry_pass = bool(all(check.pass_check for check in entry_checks))
    glide_spec = build_glide_primitive_spec()

    if not entry_pass:
        time_s, x_log, requested_log, applied_log, command_rad_log = _single_row_arrays(state0)
        metrics = _rollout_metrics(
            spec,
            case,
            cfg,
            time_s,
            x_log,
            requested_log,
            applied_log,
            "entry_set_violation",
            "entry_set_violation",
            False,
            "entry_set_violation",
        )
        return BankCaseResult(
            case_spec=case,
            primitive_spec=spec,
            time_s=time_s,
            x=x_log,
            u_norm_requested=requested_log,
            u_norm_applied=applied_log,
            delta_cmd_rad=command_rad_log,
            entry_checks=entry_checks,
            exit_checks=(),
            bank_checks=(),
            metrics=metrics,
            success=False,
            failure_label="entry_set_violation",
            notes="entry_set_violation",
        )

    trim = _solve_trim(cfg, aircraft_model)
    if not trim.converged:
        time_s, x_log, requested_log, applied_log, command_rad_log = _single_row_arrays(state0)
        metrics = _rollout_metrics(
            spec,
            case,
            cfg,
            time_s,
            x_log,
            requested_log,
            applied_log,
            "solver_failure",
            "solver_failure",
            False,
            "trim_solver_failure",
        )
        return BankCaseResult(
            case_spec=case,
            primitive_spec=spec,
            time_s=time_s,
            x=x_log,
            u_norm_requested=requested_log,
            u_norm_applied=applied_log,
            delta_cmd_rad=command_rad_log,
            entry_checks=entry_checks,
            exit_checks=(),
            bank_checks=(),
            metrics=metrics,
            success=False,
            failure_label="solver_failure",
            notes="trim_solver_failure",
        )

    x_ref = as_state_vector(trim.x_trim)
    x_ref[0:3] = state0[0:3]
    u_trim_norm = surface_rad_to_normalised_command(trim.u_cmd_trim)
    time_s = _time_grid(cfg.dt_s, case.t_final_s)
    sample_count = time_s.size
    x_log = np.empty((sample_count, STATE_SIZE), dtype=float)
    requested_log = np.empty((sample_count, 3), dtype=float)
    applied_log = np.empty((sample_count, 3), dtype=float)
    command_rad_log = np.empty((sample_count, 3), dtype=float)
    x_log[0] = state0
    rollout_failure_label = "not_run"
    notes = BANK_NOTES
    final_index = sample_count - 1

    for index in range(sample_count):
        bank_ref = bank_reference_bank_rad(float(time_s[index]), case, cfg)
        requested = bank_feedback_command_norm(
            x_log[index],
            x_ref,
            bank_ref,
            u_trim_norm,
            feedback_gains,
        )
        applied = clip_normalised_command(requested)
        command_rad = normalised_command_to_surface_rad(applied)
        requested_log[index] = requested
        applied_log[index] = applied
        command_rad_log[index] = command_rad
        if index == sample_count - 1:
            break

        next_state = rk4_step(
            x_log[index],
            command_rad,
            float(cfg.dt_s),
            aircraft_model,
            wind_model,
            cfg.wind_mode,
            cfg.actuator_tau_s,
        )
        x_log[index + 1] = next_state
        if not np.all(np.isfinite(next_state)):
            rollout_failure_label = "nonfinite_state"
            notes = "nonfinite_state"
            final_index = index + 1
            break
        if not inside_bounds(next_state[0:3], TRUE_SAFE_BOUNDS):
            rollout_failure_label = "true_safety_violation"
            notes = "true_safety_violation"
            final_index = index + 1
            break

    time_s = time_s[: final_index + 1]
    x_log = x_log[: final_index + 1]
    requested_log = requested_log[: final_index + 1]
    applied_log = applied_log[: final_index + 1]
    command_rad_log = command_rad_log[: final_index + 1]
    preliminary_metrics = _rollout_metrics(
        spec,
        case,
        cfg,
        time_s,
        x_log,
        requested_log,
        applied_log,
        rollout_failure_label,
        rollout_failure_label,
        False,
        notes,
    )
    rollout_evidence = _rollout_result(
        time_s,
        x_log,
        requested_log,
        applied_log,
        command_rad_log,
        preliminary_metrics,
        rollout_failure_label,
        notes,
    )
    exit_checks = evaluate_exit_checks(spec, rollout_evidence)
    exit_pass = bool(
        exit_checks
        and all(check.pass_check for check in exit_checks if check.required)
    )
    bank_checks = _bank_checks(case, cfg, x_log, preliminary_metrics, glide_spec)
    bank_pass = bool(all(check.pass_check for check in bank_checks if check.required))
    primitive_success = bool(entry_pass and exit_pass and bank_pass)
    if primitive_success:
        final_failure_label = "success"
        final_notes = BANK_NOTES
    elif rollout_failure_label != "not_run":
        final_failure_label = rollout_failure_label
        final_notes = rollout_failure_label
    elif _boundary_x_limited_only(case, x_log):
        final_failure_label = "terminal_recovery_limited"
        final_notes = "glide_entry_x_bound_limited"
    elif not exit_pass:
        final_failure_label = "terminal_recovery_limited"
        final_notes = "bank_exit_check_failure"
    else:
        final_failure_label = _first_failed_bank_label(bank_checks)
        final_notes = final_failure_label
    metrics = _rollout_metrics(
        spec,
        case,
        cfg,
        time_s,
        x_log,
        requested_log,
        applied_log,
        rollout_failure_label,
        final_failure_label,
        primitive_success,
        final_notes,
    )
    return BankCaseResult(
        case_spec=case,
        primitive_spec=spec,
        time_s=time_s,
        x=x_log,
        u_norm_requested=requested_log,
        u_norm_applied=applied_log,
        delta_cmd_rad=command_rad_log,
        entry_checks=entry_checks,
        exit_checks=exit_checks,
        bank_checks=bank_checks,
        metrics=metrics,
        success=primitive_success,
        failure_label=final_failure_label,
        notes=final_notes,
    )


def run_bank_batch(
    config: BankPrimitiveConfig | None = None,
    gains: BankFeedbackGains | None = None,
    aircraft: object | None = None,
) -> BankBatchResult:
    """Run the W0 bank case inventory and summarise acceptance status."""

    cfg = BankPrimitiveConfig() if config is None else config
    _validate_config(cfg)
    aircraft_model = _default_aircraft() if aircraft is None else aircraft
    trim = _solve_trim(cfg, aircraft_model)
    cases = build_bank_cases(cfg, trim)
    results = tuple(
        rollout_bank_case(
            case,
            config=cfg,
            gains=gains,
            aircraft=aircraft_model,
            wind_model=None,
        )
        for case in cases
    )
    required_results = [result for result in results if result.case_spec.role == "required"]
    required_success = bool(required_results and all(result.success for result in required_results))
    optional_failures = tuple(
        result.case_spec.name
        for result in results
        if result.case_spec.role == "optional" and not result.success
    )
    diagnostic_failures = tuple(
        result.case_spec.name
        for result in results
        if result.case_spec.role == "diagnostic" and not result.success
    )
    if not required_success:
        overall_status = "needs_review"
        failure_label = required_results[0].failure_label if required_results else "solver_failure"
        notes = "required_bank_case_failed"
    elif optional_failures:
        overall_status = "pass_with_optional_case_failures"
        failure_label = "success"
        notes = "required_bank_case_passed_with_optional_failures"
    else:
        overall_status = "pass"
        failure_label = "success"
        notes = "required_bank_case_passed"
    return BankBatchResult(
        primitive_spec=build_bank_primitive_spec(cfg),
        case_results=results,
        overall_status=overall_status,
        required_case_success=required_success,
        optional_failures=optional_failures,
        diagnostic_failures=diagnostic_failures,
        failure_label=failure_label,
        notes=notes,
    )


# =============================================================================
# 5) Output Writing
# =============================================================================
def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _entry_checks_dataframe(checks: tuple[EntryCheckResult, ...]) -> pd.DataFrame:
    return pd.DataFrame([check.__dict__ for check in checks])


def _exit_checks_dataframe(checks: tuple[ExitCheckResult, ...]) -> pd.DataFrame:
    return pd.DataFrame([check.__dict__ for check in checks])


def _bank_checks_dataframe(checks: tuple[BankCheckResult, ...]) -> pd.DataFrame:
    return pd.DataFrame([check.__dict__ for check in checks])


def _check_value(checks: tuple[BankCheckResult, ...], name: str) -> object:
    for check in checks:
        if check.name == name:
            return check.value
    return np.nan


def _case_summary_dataframe(batch: BankBatchResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in batch.case_results:
        rows.append(
            {
                "case_name": result.case_spec.name,
                "role": result.case_spec.role,
                "direction_sign": result.case_spec.direction_sign,
                "success": result.success,
                "failure_label": result.failure_label,
                "notes": result.notes,
                "duration_s": result.metrics["duration_s"],
                "lateral_displacement_m": float(
                    result.x[-1, STATE_INDEX["y_w"]] - result.x[0, STATE_INDEX["y_w"]]
                ),
                "terminal_x_w_m": float(result.x[-1, STATE_INDEX["x_w"]]),
                "terminal_true_safe_x_margin_m": terminal_true_safe_x_margin_m(result.x[-1]),
                "terminal_glide_entry_x_margin_m": terminal_glide_entry_x_margin_m(result.x[-1]),
                "terminal_glide_entry_proxy": terminal_glide_entry_proxy(result.x[-1]),
                "terminal_glide_entry_proxy_source": GLIDE_PROXY_SOURCE,
                "peak_bank_abs_deg": _check_value(result.bank_checks, "peak_bank_min_abs_deg"),
                "terminal_bank_abs_deg": _check_value(result.bank_checks, "terminal_bank_abs_deg"),
            }
        )
    return pd.DataFrame(rows)


def _manifest(
    batch: BankBatchResult,
    run_id: int,
    output_files: dict[str, Path],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": f"s{run_id:03d}",
        "overall_status": batch.overall_status,
        "required_case_success": batch.required_case_success,
        "optional_failures": list(batch.optional_failures),
        "diagnostic_failures": list(batch.diagnostic_failures),
        "failure_label": batch.failure_label,
        "notes": batch.notes,
        "case_outcomes": {
            result.case_spec.name: {
                "role": result.case_spec.role,
                "direction_sign": result.case_spec.direction_sign,
                "success": result.success,
                "failure_label": result.failure_label,
                "notes": result.notes,
                "terminal_x_w_m": float(result.x[-1, STATE_INDEX["x_w"]]),
                "terminal_true_safe_x_margin_m": terminal_true_safe_x_margin_m(result.x[-1]),
                "terminal_glide_entry_x_margin_m": terminal_glide_entry_x_margin_m(result.x[-1]),
                "terminal_glide_entry_proxy": terminal_glide_entry_proxy(result.x[-1]),
            }
            for result in batch.case_results
        },
        "primitive_interface_used": True,
        "primitive_implemented": True,
        "primitive_controller_implemented": True,
        "controller_implemented": True,
        "local_feedback_controller_implemented": True,
        "actual_glide_primitive_implemented": True,
        "actual_bank_primitive_implemented": True,
        "actual_recovery_primitive_implemented": True,
        "actual_agile_reversal_primitive_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "vicon_implemented": False,
        "hardware_implemented": False,
        "high_incidence_validation_claim": False,
        "updraft_validation_claim": False,
        "w1_w2_w3_updraft_validation_claim": False,
        "bank_updraft_encounter_role": "w0_lateral_repositioning_baseline_only",
        "trim_command_units": "rad",
        "feedback_command_units": "normalised",
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad",
        "state_derivative_command_input": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "terminal_glide_entry_proxy_source": GLIDE_PROXY_SOURCE,
        "terminal_glide_entry_proxy_is_governor_proof": False,
        "output_files": {
            name: _repo_relative(path)
            for name, path in output_files.items()
        },
        "validation_commands": list(VALIDATION_COMMANDS),
    }


def _write_report(
    path: Path,
    batch: BankBatchResult,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# W0 Bank Primitive Report",
        "",
        "This is the first actual W0 no-wind bank primitive and a baseline for",
        "later updraft-encounter use. It is not an updraft robustness result,",
        "not an agile reversal, not OCP, not TVLQR, not governor, not real",
        "flight, and not high-incidence validation.",
        "",
        "- Bank/updraft-encounter role: `w0_lateral_repositioning_baseline_only`",
        "",
        "## Batch Status",
        "",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Required case success: `{manifest['required_case_success']}`",
        f"- Optional failures: `{manifest['optional_failures']}`",
        f"- Diagnostic failures: `{manifest['diagnostic_failures']}`",
        f"- Failure label: `{manifest['failure_label']}`",
        f"- Notes: `{manifest['notes']}`",
        "",
        "## Geometry And Handoff Contract",
        "",
        f"- Default initial position: `{INITIAL_POSITION_W_M}` m",
        f"- Terminal glide-entry proxy source: `{GLIDE_PROXY_SOURCE}`",
        "- The diagnostic 0.80 s boundary is handoff-limited only when the",
        "  trajectory remains finite and true-safe while failing only the existing",
        "  glide-entry x bound.",
        "",
        "## Command Path",
        "",
        "- Trim command: physical radians from `solve_straight_trim()`.",
        "- Trim bridge: `surface_rad_to_normalised_command`.",
        "- Feedback correction: normalised command space only.",
        "- Applied bridge: `u_norm_requested -> u_norm_applied -> delta_cmd_rad`.",
        "- Plant input: `delta_cmd_rad`; raw normalised commands do not enter dynamics.",
        "",
        "## Case Outcomes",
        "",
    ]
    for result in batch.case_results:
        lines.extend(
            [
                f"### {result.case_spec.name}",
                "",
                f"- Role: `{result.case_spec.role}`",
                f"- Direction sign: `{result.case_spec.direction_sign}`",
                f"- Success: `{result.success}`",
                f"- Failure label: `{result.failure_label}`",
                f"- Notes: `{result.notes}`",
                f"- Duration: `{result.metrics['duration_s']}` s",
                f"- Lateral displacement: "
                f"`{float(result.x[-1, STATE_INDEX['y_w']] - result.x[0, STATE_INDEX['y_w']]):.6f}` m",
                f"- Terminal x_w: `{float(result.x[-1, STATE_INDEX['x_w']]):.6f}` m",
                "- Terminal true-safe x margin: "
                f"`{terminal_true_safe_x_margin_m(result.x[-1]):.6f}` m",
                "- Terminal glide-entry x margin: "
                f"`{terminal_glide_entry_x_margin_m(result.x[-1]):.6f}` m",
                "- Terminal glide-entry proxy: "
                f"`{terminal_glide_entry_proxy(result.x[-1])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Implementation Flags",
            "",
            f"- Actual bank primitive implemented: `{manifest['actual_bank_primitive_implemented']}`",
            f"- Actual recovery primitive implemented: `{manifest['actual_recovery_primitive_implemented']}`",
            f"- Actual glide primitive implemented: `{manifest['actual_glide_primitive_implemented']}`",
            f"- Actual agile reversal primitive implemented: `{manifest['actual_agile_reversal_primitive_implemented']}`",
            f"- Updraft validation claim: `{manifest['updraft_validation_claim']}`",
            f"- W1/W2/W3 updraft validation claim: `{manifest['w1_w2_w3_updraft_validation_claim']}`",
            f"- OCP implemented: `{manifest['ocp_implemented']}`",
            f"- TVLQR implemented: `{manifest['tvlqr_implemented']}`",
            f"- Governor implemented: `{manifest['governor_implemented']}`",
            f"- Outer loop implemented: `{manifest['outer_loop_implemented']}`",
            f"- High-incidence validation claim: `{manifest['high_incidence_validation_claim']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


def write_bank_outputs(
    result: BankBatchResult,
    result_root: Path,
    campaign: str,
    run_id: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write W0 bank primitive raw CSV/JSON/Markdown evidence."""

    paths = make_result_tree(Path(result_root), campaign, run_id, overwrite=overwrite)
    suffix = f"s{run_id:03d}"
    output_paths: dict[str, Path] = {
        "case_summary_csv": paths["metrics"] / f"bank_case_summary_{suffix}.csv",
        "manifest_json": paths["manifests"] / f"bank_primitive_manifest_{suffix}.json",
        "report_md": paths["reports"] / f"bank_primitive_report_{suffix}.md",
    }
    _case_summary_dataframe(result).to_csv(output_paths["case_summary_csv"], index=False)
    for case_result in result.case_results:
        case_name = case_result.case_spec.name
        output_paths[f"{case_name}_entry_checks_csv"] = (
            paths["metrics"] / f"{case_name}_entry_checks_{suffix}.csv"
        )
        output_paths[f"{case_name}_exit_checks_csv"] = (
            paths["metrics"] / f"{case_name}_exit_checks_{suffix}.csv"
        )
        output_paths[f"{case_name}_bank_checks_csv"] = (
            paths["metrics"] / f"{case_name}_bank_checks_{suffix}.csv"
        )
        output_paths[f"{case_name}_trajectory_csv"] = (
            paths["metrics"] / f"{case_name}_trajectory_{suffix}.csv"
        )
        output_paths[f"{case_name}_commands_csv"] = (
            paths["metrics"] / f"{case_name}_commands_{suffix}.csv"
        )
        output_paths[f"{case_name}_metrics_csv"] = (
            paths["metrics"] / f"{case_name}_metrics_{suffix}.csv"
        )
        _entry_checks_dataframe(case_result.entry_checks).to_csv(
            output_paths[f"{case_name}_entry_checks_csv"],
            index=False,
        )
        _exit_checks_dataframe(case_result.exit_checks).to_csv(
            output_paths[f"{case_name}_exit_checks_csv"],
            index=False,
        )
        _bank_checks_dataframe(case_result.bank_checks).to_csv(
            output_paths[f"{case_name}_bank_checks_csv"],
            index=False,
        )
        trajectory_dataframe(case_result.time_s, case_result.x).to_csv(
            output_paths[f"{case_name}_trajectory_csv"],
            index=False,
        )
        command_dataframe(
            case_result.time_s,
            case_result.u_norm_requested,
            case_result.u_norm_applied,
            case_result.delta_cmd_rad,
        ).to_csv(output_paths[f"{case_name}_commands_csv"], index=False)
        metric_row = dict(case_result.metrics)
        metric_row["run_id"] = suffix
        metric_dataframe(metric_row).to_csv(
            output_paths[f"{case_name}_metrics_csv"],
            index=False,
        )
    manifest = _manifest(result, run_id, output_paths)
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    _write_report(output_paths["report_md"], result, manifest)
    output_paths["root"] = paths["root"]
    return output_paths
