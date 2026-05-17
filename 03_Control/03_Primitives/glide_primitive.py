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
    state_entry_variables,
)
from result_paths import make_result_tree
from rollout import RolloutResult, rk4_step
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector
from trim_solver import TrimResult, TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and data containers
# 2) State, trim, and feedback helpers
# 3) Closed-loop glide rollout
# 4) Output writing
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
PRIMITIVE_FAMILY = "glide"
GLIDE_CAMPAIGN = "04_glide_primitive"
GLIDE_NOTES = "glide_w0_nominal_trim_hold_feedback"
INITIAL_POSITION_W_M = (2.0, 2.2, 1.6)
VALIDATION_COMMANDS = (
    "python -m py_compile "
    "03_Control/04_Scenarios/logging_contract.py "
    "03_Control/03_Primitives/primitive_interface.py "
    "03_Control/03_Primitives/glide_primitive.py "
    "03_Control/04_Scenarios/run_glide_primitive_w0.py",
    "python 03_Control/04_Scenarios/run_rollout_smoke.py --run-id 1 --overwrite",
    "python 03_Control/04_Scenarios/run_primitive_interface_smoke.py "
    "--run-id 1 --overwrite",
    "python 03_Control/04_Scenarios/run_glide_primitive_w0.py "
    "--run-id 1 --overwrite",
    "python -m pytest -q "
    "03_Control/tests/test_rollout_logging_contract.py "
    "03_Control/tests/test_rollout_smoke.py "
    "03_Control/tests/test_primitive_interface_smoke.py "
    "03_Control/tests/test_glide_primitive.py "
    "03_Control/tests/test_glide_primitive_w0.py",
    "python -m pytest -q 03_Control/tests",
)


@dataclass(frozen=True)
class GlideFeedbackGains:
    k_phi: float = 0.25
    k_p: float = 0.05
    k_theta: float = 0.20
    k_q: float = 0.05
    k_speed: float = 0.03
    k_r: float = 0.04
    k_beta: float = 0.15


@dataclass(frozen=True)
class GlidePrimitiveConfig:
    dt_s: float = 0.02
    t_final_s: float = 0.50
    speed_m_s: float = 6.5
    altitude_m: float = 1.6
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    seed: int = 1
    scenario_name: str = "glide_w0_nominal"


@dataclass(frozen=True)
class GlideCheckResult:
    name: str
    value: float | bool
    limit: float | str
    pass_check: bool
    required: bool
    units: str
    reason: str


@dataclass(frozen=True)
class GlidePrimitiveResult:
    primitive_spec: PrimitiveSpec
    time_s: np.ndarray
    x: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    entry_checks: tuple[EntryCheckResult, ...]
    exit_checks: tuple[ExitCheckResult, ...]
    glide_checks: tuple[GlideCheckResult, ...]
    metrics: dict[str, object]
    success: bool
    failure_label: str
    notes: str


# =============================================================================
# 2) State, Trim, and Feedback Helpers
# =============================================================================
def _time_step_count(dt_s: float, t_final_s: float) -> int:
    ratio = float(t_final_s) / float(dt_s)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, rtol=1e-12, atol=1e-9):
        raise ValueError("t_final_s must be an integer multiple of dt_s.")
    return rounded


def _time_grid(config: GlidePrimitiveConfig) -> np.ndarray:
    step_count = _time_step_count(config.dt_s, config.t_final_s)
    return np.arange(step_count + 1, dtype=float) * float(config.dt_s)


def _validate_config(config: GlidePrimitiveConfig) -> None:
    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("glide dt_s must be finite and positive.")
    if not np.isfinite(float(config.t_final_s)) or float(config.t_final_s) <= 0.0:
        raise ValueError("glide t_final_s must be finite and positive.")
    _time_step_count(config.dt_s, config.t_final_s)
    if not np.isfinite(float(config.speed_m_s)) or float(config.speed_m_s) <= 0.0:
        raise ValueError("glide speed_m_s must be finite and positive.")
    if not np.isfinite(float(config.altitude_m)):
        raise ValueError("glide altitude_m must be finite.")
    if config.wind_mode != "none":
        raise ValueError("W0 glide supports only wind_mode='none'.")
    if config.latency_case != "none":
        raise ValueError("W0 glide supports only latency_case='none'.")
    tau = np.asarray(config.actuator_tau_s, dtype=float)
    if tau.size != 3 or not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("actuator_tau_s must contain three finite positive values.")
    if not config.scenario_name:
        raise ValueError("scenario_name must be nonempty.")


def _speed_alpha_beta_rad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_b = np.asarray(x, dtype=float)[:, 6:9]
    speed = np.linalg.norm(velocity_b, axis=1)
    alpha_rad = np.arctan2(velocity_b[:, 2], velocity_b[:, 0])
    beta_rad = np.zeros_like(speed)
    valid = speed > 1e-9
    beta_rad[valid] = np.arcsin(np.clip(velocity_b[valid, 1] / speed[valid], -1.0, 1.0))
    return speed, alpha_rad, beta_rad


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
        }
    return {
        "min_wall_margin_m": np.array([row["min_wall_margin_m"] for row in rows]),
        "floor_margin_m": np.array([row["floor_margin_m"] for row in rows]),
        "ceiling_margin_m": np.array([row["ceiling_margin_m"] for row in rows]),
        "min_margin_m": np.array([row["min_margin_m"] for row in rows]),
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


def _default_aircraft() -> object:
    return adapt_glider(build_nausicaa_glider())


def _solve_trim(config: GlidePrimitiveConfig, aircraft: object) -> TrimResult:
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


def build_glide_primitive_spec(
    config: GlidePrimitiveConfig | None = None,
) -> PrimitiveSpec:
    """Return the W0 glide primitive metadata contract."""

    cfg = GlidePrimitiveConfig() if config is None else config
    _validate_config(cfg)
    entry_set = PrimitiveEntrySet(
        name="glide_w0_entry",
        description="first W0 glide primitive entry set in SI units and radians",
        lower={
            "x_w": 1.4,
            "y_w": 0.3,
            "z_w": 0.8,
            "speed_m_s": 5.5,
            "alpha_rad": -0.15,
            "beta_rad": -0.15,
            "phi": -0.25,
            "theta": -0.25,
            "p": -0.8,
            "q": -0.8,
            "r": -0.8,
        },
        upper={
            "x_w": 5.5,
            "y_w": 4.1,
            "z_w": 2.6,
            "speed_m_s": 7.5,
            "alpha_rad": 0.25,
            "beta_rad": 0.15,
            "phi": 0.25,
            "theta": 0.25,
            "p": 0.8,
            "q": 0.8,
            "r": 0.8,
        },
    )
    exit_checks = (
        PrimitiveExitCheck(
            name="finite_state",
            description="full glide state history remains finite",
            required=True,
        ),
        PrimitiveExitCheck(
            name="true_safe_margin",
            description="minimum true-safety margin stays nonnegative",
            required=True,
        ),
        PrimitiveExitCheck(
            name="rollout_success",
            description="closed-loop rollout integrity metric reports success",
            required=True,
        ),
    )
    spec = PrimitiveSpec(
        name="glide_w0_nominal",
        family=PRIMITIVE_FAMILY,
        duration_s=float(cfg.t_final_s),
        entry_set=entry_set,
        exit_checks=exit_checks,
        metadata={
            "actual_glide_primitive_implemented": "true",
            "controller_type": "trim_hold_pr_feedback",
            "wind_mode": cfg.wind_mode,
        },
    )
    validate_primitive_spec(spec)
    return spec


def initial_glide_state(
    config: GlidePrimitiveConfig,
    trim_result: TrimResult | None = None,
) -> np.ndarray:
    """Return the default W0 glide initial state from the straight trim result."""

    _validate_config(config)
    trim = trim_result
    if trim is None:
        trim = _solve_trim(config, _default_aircraft())
    if not trim.converged:
        raise ValueError("cannot build glide initial state from nonconverged trim.")
    state = as_state_vector(trim.x_trim)
    state[0] = INITIAL_POSITION_W_M[0]
    state[1] = INITIAL_POSITION_W_M[1]
    state[2] = float(config.altitude_m)
    return state


def glide_feedback_command_norm(
    x: np.ndarray,
    x_ref: np.ndarray,
    u_trim_norm: np.ndarray,
    gains: GlideFeedbackGains,
) -> np.ndarray:
    """Return requested normalised aggregate command for trim-hold glide."""

    state = as_state_vector(x)
    reference = as_state_vector(x_ref)
    trim_norm = as_normalised_command_vector(u_trim_norm)
    speed = float(np.linalg.norm(state[6:9]))
    speed_ref = float(np.linalg.norm(reference[6:9]))
    beta_rad = 0.0
    if speed > 1e-9:
        beta_rad = float(np.arcsin(np.clip(state[STATE_INDEX["v"]] / speed, -1.0, 1.0)))

    phi_error = float(state[STATE_INDEX["phi"]] - reference[STATE_INDEX["phi"]])
    theta_error = float(state[STATE_INDEX["theta"]] - reference[STATE_INDEX["theta"]])
    p_rate = float(state[STATE_INDEX["p"]])
    q_rate = float(state[STATE_INDEX["q"]])
    r_rate = float(state[STATE_INDEX["r"]])
    correction = np.array(
        [
            -gains.k_phi * phi_error - gains.k_p * p_rate,
            -gains.k_theta * theta_error
            - gains.k_q * q_rate
            - gains.k_speed * (speed_ref - speed),
            -gains.k_r * r_rate - gains.k_beta * beta_rad,
        ],
        dtype=float,
    )
    requested = trim_norm + correction
    if not np.all(np.isfinite(requested)):
        raise ValueError("glide feedback produced a nonfinite command.")
    return requested


def terminal_recoverable_proxy(
    x_terminal: np.ndarray,
    spec: PrimitiveSpec | None = None,
) -> bool:
    """Return True if the terminal state is plausible for future recovery/glide entry."""

    try:
        state = as_state_vector(x_terminal)
    except ValueError:
        return False
    if not inside_bounds(state[0:3], TRUE_SAFE_BOUNDS):
        return False
    variables = state_entry_variables(state)
    rate_norm = float(np.linalg.norm(state[9:12]))
    if not 5.0 <= variables["speed_m_s"] <= 8.0:
        return False
    if abs(variables["alpha_rad"]) > np.deg2rad(15.0):
        return False
    if abs(variables["beta_rad"]) > np.deg2rad(10.0):
        return False
    if rate_norm > 1.5:
        return False
    checked_spec = build_glide_primitive_spec() if spec is None else spec
    try:
        entry_checks = evaluate_entry_set(state, checked_spec.entry_set)
    except ValueError:
        return False
    return bool(all(check.pass_check for check in entry_checks))


# =============================================================================
# 3) Closed-Loop Glide Rollout
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
    config: GlidePrimitiveConfig,
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
            "scenario_name": config.scenario_name,
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
    value: float | bool,
    limit: float | str,
    pass_check: bool,
    units: str,
    reason: str,
) -> GlideCheckResult:
    return GlideCheckResult(
        name=name,
        value=value,
        limit=limit,
        pass_check=bool(pass_check),
        required=True,
        units=units,
        reason=reason,
    )


def _glide_checks(
    x_log: np.ndarray,
    metrics: dict[str, object],
    spec: PrimitiveSpec,
) -> tuple[GlideCheckResult, ...]:
    height_loss_m = -float(metrics["height_change_m"])
    terminal_speed_m_s = float(metrics["terminal_speed_m_s"])
    max_alpha_deg = float(metrics["max_alpha_deg"])
    max_beta_deg = float(metrics["max_beta_deg"])
    max_bank_deg = float(metrics["max_bank_deg"])
    max_pitch_deg = float(metrics["max_pitch_deg"])
    max_rate_rad_s = float(metrics["max_rate_rad_s"])
    saturation_fraction = float(metrics["saturation_fraction"])
    terminal_proxy = terminal_recoverable_proxy(x_log[-1], spec)
    return (
        _check(
            "terminal_speed_min",
            terminal_speed_m_s,
            5.0,
            terminal_speed_m_s >= 5.0,
            "m/s",
            "terminal_speed_above_minimum",
        ),
        _check(
            "terminal_speed_max",
            terminal_speed_m_s,
            8.0,
            terminal_speed_m_s <= 8.0,
            "m/s",
            "terminal_speed_below_maximum",
        ),
        _check(
            "height_loss",
            height_loss_m,
            0.40,
            height_loss_m <= 0.40,
            "m",
            "height_loss_within_w0_glide_bound",
        ),
        _check(
            "alpha_boundary",
            max_alpha_deg,
            15.0,
            max_alpha_deg <= 15.0,
            "deg",
            "alpha_within_low_incidence_glide_bound",
        ),
        _check(
            "beta_boundary",
            max_beta_deg,
            10.0,
            max_beta_deg <= 10.0,
            "deg",
            "beta_within_low_sideslip_glide_bound",
        ),
        _check(
            "bank_boundary",
            max_bank_deg,
            15.0,
            max_bank_deg <= 15.0,
            "deg",
            "bank_within_w0_glide_bound",
        ),
        _check(
            "pitch_boundary",
            max_pitch_deg,
            20.0,
            max_pitch_deg <= 20.0,
            "deg",
            "pitch_within_w0_glide_bound",
        ),
        _check(
            "rate_boundary",
            max_rate_rad_s,
            1.5,
            max_rate_rad_s <= 1.5,
            "rad/s",
            "body_rate_norm_within_w0_glide_bound",
        ),
        _check(
            "saturation_fraction",
            saturation_fraction,
            0.10,
            saturation_fraction <= 0.10,
            "-",
            "requested_commands_do_not_excessively_clip",
        ),
        _check(
            "terminal_recoverable_proxy",
            terminal_proxy,
            "true",
            terminal_proxy,
            "-",
            "terminal_state_plausible_for_later_recovery_or_glide_entry",
        ),
    )


def _first_failed_glide_label(checks: tuple[GlideCheckResult, ...]) -> str:
    mapping = {
        "terminal_speed_min": "speed_low",
        "terminal_speed_max": "speed_high",
        "alpha_boundary": "alpha_boundary",
        "beta_boundary": "beta_boundary",
        "rate_boundary": "rate_boundary",
        "saturation_fraction": "actuator_saturation_limited",
    }
    for check in checks:
        if not check.pass_check:
            return mapping.get(check.name, "terminal_recovery_limited")
    return "success"


def rollout_glide_primitive(
    x0: np.ndarray,
    config: GlidePrimitiveConfig | None = None,
    gains: GlideFeedbackGains | None = None,
    aircraft: object | None = None,
    wind_model: object = None,
) -> GlidePrimitiveResult:
    """Run the first actual W0 glide primitive with local trim-hold feedback."""

    cfg = GlidePrimitiveConfig() if config is None else config
    _validate_config(cfg)
    if wind_model is not None:
        raise ValueError("W0 glide requires wind_model=None.")
    feedback_gains = GlideFeedbackGains() if gains is None else gains
    aircraft_model = _default_aircraft() if aircraft is None else aircraft
    spec = build_glide_primitive_spec(cfg)
    state0 = as_state_vector(x0)
    entry_checks = evaluate_entry_set(state0, spec.entry_set)
    entry_pass = bool(all(check.pass_check for check in entry_checks))

    if not entry_pass:
        time_s, x_log, requested_log, applied_log, command_rad_log = _single_row_arrays(state0)
        metrics = _rollout_metrics(
            spec,
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
        return GlidePrimitiveResult(
            primitive_spec=spec,
            time_s=time_s,
            x=x_log,
            u_norm_requested=requested_log,
            u_norm_applied=applied_log,
            delta_cmd_rad=command_rad_log,
            entry_checks=entry_checks,
            exit_checks=(),
            glide_checks=(),
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
        return GlidePrimitiveResult(
            primitive_spec=spec,
            time_s=time_s,
            x=x_log,
            u_norm_requested=requested_log,
            u_norm_applied=applied_log,
            delta_cmd_rad=command_rad_log,
            entry_checks=entry_checks,
            exit_checks=(),
            glide_checks=(),
            metrics=metrics,
            success=False,
            failure_label="solver_failure",
            notes="trim_solver_failure",
        )

    x_ref = as_state_vector(trim.x_trim)
    x_ref[0:3] = state0[0:3]
    u_trim_norm = surface_rad_to_normalised_command(trim.u_cmd_trim)
    time_s = _time_grid(cfg)
    sample_count = time_s.size
    x_log = np.empty((sample_count, STATE_SIZE), dtype=float)
    requested_log = np.empty((sample_count, 3), dtype=float)
    applied_log = np.empty((sample_count, 3), dtype=float)
    command_rad_log = np.empty((sample_count, 3), dtype=float)
    x_log[0] = state0
    rollout_failure_label = "not_run"
    notes = GLIDE_NOTES
    final_index = sample_count - 1

    for index in range(sample_count):
        requested = glide_feedback_command_norm(
            x_log[index],
            x_ref,
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
    glide_checks = _glide_checks(x_log, preliminary_metrics, spec)
    glide_pass = bool(all(check.pass_check for check in glide_checks if check.required))
    primitive_success = bool(entry_pass and exit_pass and glide_pass)
    if primitive_success:
        final_failure_label = "success"
    elif rollout_failure_label != "not_run":
        final_failure_label = rollout_failure_label
    elif not exit_pass:
        final_failure_label = "terminal_recovery_limited"
    else:
        final_failure_label = _first_failed_glide_label(glide_checks)
    final_notes = GLIDE_NOTES if primitive_success else final_failure_label
    metrics = _rollout_metrics(
        spec,
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
    return GlidePrimitiveResult(
        primitive_spec=spec,
        time_s=time_s,
        x=x_log,
        u_norm_requested=requested_log,
        u_norm_applied=applied_log,
        delta_cmd_rad=command_rad_log,
        entry_checks=entry_checks,
        exit_checks=exit_checks,
        glide_checks=glide_checks,
        metrics=metrics,
        success=primitive_success,
        failure_label=final_failure_label,
        notes=final_notes,
    )


# =============================================================================
# 4) Output Writing
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


def _glide_checks_dataframe(checks: tuple[GlideCheckResult, ...]) -> pd.DataFrame:
    return pd.DataFrame([check.__dict__ for check in checks])


def _status(result: GlidePrimitiveResult) -> dict[str, bool | str]:
    entry_checks_pass = bool(all(check.pass_check for check in result.entry_checks))
    exit_checks_pass = bool(
        result.exit_checks
        and all(check.pass_check for check in result.exit_checks if check.required)
    )
    glide_checks_pass = bool(
        result.glide_checks
        and all(check.pass_check for check in result.glide_checks if check.required)
    )
    rollout_ran = bool(result.time_s.size > 1)
    interface_checks_pass = bool(entry_checks_pass and exit_checks_pass and rollout_ran)
    return {
        "overall_status": "pass" if result.success else "needs_review",
        "interface_checks_pass": interface_checks_pass,
        "entry_checks_pass": entry_checks_pass,
        "exit_checks_pass": exit_checks_pass,
        "glide_checks_pass": glide_checks_pass,
        "rollout_ran": rollout_ran,
    }


def _manifest(
    result: GlidePrimitiveResult,
    run_id: int,
    output_files: dict[str, Path],
) -> dict[str, Any]:
    status = _status(result)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": f"s{run_id:03d}",
        **status,
        "primitive_interface_used": True,
        "primitive_interface_implemented": True,
        "primitive_implemented": True,
        "primitive_controller_implemented": True,
        "controller_implemented": True,
        "local_feedback_controller_implemented": True,
        "controller_type": "trim_hold_pr_feedback",
        "actual_glide_primitive_implemented": True,
        "actual_bank_primitive_implemented": False,
        "actual_recovery_primitive_implemented": False,
        "actual_agile_reversal_primitive_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "vicon_implemented": False,
        "hardware_implemented": False,
        "high_incidence_validation_claim": False,
        "primitive_success": bool(result.metrics["primitive_success"]),
        "success": bool(result.metrics["success"]),
        "failure_label": result.failure_label,
        "notes": result.notes,
        "trim_command_units": "rad",
        "feedback_command_units": "normalised",
        "command_bridge": "normalised_command_to_surface_rad",
        "state_derivative_command_input": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "terminal_recoverable_proxy_is_recovery_proof": False,
        "metric_summary": {
            "duration_s": result.metrics["duration_s"],
            "terminal_speed_m_s": result.metrics["terminal_speed_m_s"],
            "height_change_m": result.metrics["height_change_m"],
            "max_alpha_deg": result.metrics["max_alpha_deg"],
            "max_beta_deg": result.metrics["max_beta_deg"],
            "max_rate_rad_s": result.metrics["max_rate_rad_s"],
            "saturation_fraction": result.metrics["saturation_fraction"],
        },
        "output_files": {
            name: _repo_relative(path)
            for name, path in output_files.items()
        },
        "validation_commands": list(VALIDATION_COMMANDS),
    }


def _write_report(
    path: Path,
    result: GlidePrimitiveResult,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# W0 Glide Primitive Report",
        "",
        "This is the first actual no-wind glide primitive. It uses the existing",
        "primitive-interface checks, the existing RK4 plant step, and a local",
        "trim-hold feedback law in normalised command space.",
        "",
        "It is not bank, recovery, agile reversal, OCP, TVLQR, governor, outer-loop,",
        "Vicon, hardware, real-flight, or high-incidence validation evidence.",
        "",
        "## Status",
        "",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Entry checks pass: `{manifest['entry_checks_pass']}`",
        f"- Exit checks pass: `{manifest['exit_checks_pass']}`",
        f"- Glide checks pass: `{manifest['glide_checks_pass']}`",
        f"- Rollout ran: `{manifest['rollout_ran']}`",
        f"- Primitive success: `{manifest['primitive_success']}`",
        f"- Failure label: `{manifest['failure_label']}`",
        f"- Notes: `{result.notes}`",
        "",
        "## Command Path",
        "",
        "- Trim command: physical radians from `solve_straight_trim()`.",
        "- Trim bridge: `surface_rad_to_normalised_command`.",
        "- Feedback correction: normalised command space only.",
        "- Applied bridge: `u_norm_requested -> u_norm_applied -> delta_cmd_rad`.",
        "- Plant input: `delta_cmd_rad`; raw normalised commands do not enter dynamics.",
        "",
        "## Observed Metrics",
        "",
        f"- Duration: `{result.metrics['duration_s']}` s",
        f"- Terminal speed: `{result.metrics['terminal_speed_m_s']}` m/s",
        f"- Height change: `{result.metrics['height_change_m']}` m",
        f"- Minimum wall margin: `{result.metrics['min_true_wall_margin_m']}` m",
        f"- Minimum floor margin: `{result.metrics['min_floor_margin_m']}` m",
        f"- Maximum alpha: `{result.metrics['max_alpha_deg']}` deg",
        f"- Maximum beta: `{result.metrics['max_beta_deg']}` deg",
        f"- Maximum rate norm: `{result.metrics['max_rate_rad_s']}` rad/s",
        f"- Saturation fraction: `{result.metrics['saturation_fraction']}`",
        "",
        "## Terminal Proxy",
        "",
        "`terminal_recoverable_proxy` is only a plausibility check for later",
        "primitive expansion. It does not claim that a recovery primitive exists.",
        "",
        "## Implementation Flags",
        "",
        f"- Actual glide primitive implemented: `{manifest['actual_glide_primitive_implemented']}`",
        f"- Local feedback controller implemented: `{manifest['local_feedback_controller_implemented']}`",
        f"- Actual bank primitive implemented: `{manifest['actual_bank_primitive_implemented']}`",
        f"- Actual recovery primitive implemented: `{manifest['actual_recovery_primitive_implemented']}`",
        f"- Actual agile reversal primitive implemented: `{manifest['actual_agile_reversal_primitive_implemented']}`",
        f"- OCP implemented: `{manifest['ocp_implemented']}`",
        f"- TVLQR implemented: `{manifest['tvlqr_implemented']}`",
        f"- Governor implemented: `{manifest['governor_implemented']}`",
        f"- Outer loop implemented: `{manifest['outer_loop_implemented']}`",
        f"- High-incidence validation claim: `{manifest['high_incidence_validation_claim']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def write_glide_outputs(
    result: GlidePrimitiveResult,
    result_root: Path,
    campaign: str,
    run_id: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write W0 glide primitive evidence."""

    paths = make_result_tree(Path(result_root), campaign, run_id, overwrite=overwrite)
    suffix = f"s{run_id:03d}"
    output_paths = {
        "entry_checks_csv": paths["metrics"] / f"entry_checks_{suffix}.csv",
        "exit_checks_csv": paths["metrics"] / f"exit_checks_{suffix}.csv",
        "glide_checks_csv": paths["metrics"] / f"glide_checks_{suffix}.csv",
        "trajectory_csv": paths["metrics"] / f"trajectory_{suffix}.csv",
        "commands_csv": paths["metrics"] / f"commands_{suffix}.csv",
        "metrics_csv": paths["metrics"] / f"glide_metrics_{suffix}.csv",
        "manifest_json": paths["manifests"] / f"glide_primitive_manifest_{suffix}.json",
        "report_md": paths["reports"] / f"glide_primitive_report_{suffix}.md",
    }
    _entry_checks_dataframe(result.entry_checks).to_csv(
        output_paths["entry_checks_csv"],
        index=False,
    )
    _exit_checks_dataframe(result.exit_checks).to_csv(
        output_paths["exit_checks_csv"],
        index=False,
    )
    _glide_checks_dataframe(result.glide_checks).to_csv(
        output_paths["glide_checks_csv"],
        index=False,
    )
    trajectory_dataframe(result.time_s, result.x).to_csv(
        output_paths["trajectory_csv"],
        index=False,
    )
    command_dataframe(
        result.time_s,
        result.u_norm_requested,
        result.u_norm_applied,
        result.delta_cmd_rad,
    ).to_csv(output_paths["commands_csv"], index=False)
    metric_row = dict(result.metrics)
    metric_row["run_id"] = suffix
    metric_dataframe(metric_row).to_csv(output_paths["metrics_csv"], index=False)
    manifest = _manifest(result, run_id, output_paths)
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    _write_report(output_paths["report_md"], result, manifest)
    output_paths["root"] = paths["root"]
    return output_paths
