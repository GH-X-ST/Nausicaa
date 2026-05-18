from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggressive_reversal_primitive import terminal_aggressive_recoverable_proxy
from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
)
from flight_dynamics import adapt_glider, build_symbolic_dynamics
from glider import build_nausicaa_glider
from metric_contract import FAILURE_LABELS
from rollout import rk4_step
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector
from trim_solver import TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and dataclasses
# 2) Seed families and phase profiles
# 3) Replay and metrics
# 4) Guided search and direct OCP attempts
# 5) Public solve workflow
# =============================================================================


# =============================================================================
# 1) Constants and Dataclasses
# =============================================================================
AGGRESSIVE_CAMPAIGN = "07_aggressive_reversal_ocp"
PHASES = (
    "entry",
    "pitch_brake",
    "slow_redirect",
    "heading_capture",
    "unload_descend",
    "exit_glide",
)
SEED_FAMILIES = (
    "short_perch_yaw_redirect",
    "long_perch_slow_redirect",
    "roll_dominant_banked_redirect",
    "split_pulse_redirect",
    "early_unload_descend_capture",
)
TARGET_HORIZONS_S = {
    15.0: 0.70,
    30.0: 0.86,
    60.0: 1.06,
    90.0: 1.20,
    120.0: 1.40,
    180.0: 1.60,
}
REPLAY_DEFECT_TOL = 1e-5
TARGET_TOKEN_WIDTH = 3


@dataclass(frozen=True)
class AggressiveReversalOcpConfig:
    dt_s: float = 0.02
    t_final_s: float = 1.20
    target_heading_deg: float = 15.0
    direction_sign: int = 1
    speed_m_s: float = 6.5
    altitude_m: float = 1.8
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    seed: int = 1
    max_ipopt_iter: int = 80
    use_nonlinear_ocp: bool = True
    use_phase_seed_fallback: bool = True
    phase_search_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    ocp_node_count: int = 2
    ocp_max_cpu_time_s: float = 0.75
    checkpoint_dir: str | None = None
    candidate_log_dir: str | None = None
    run_id: str = "s001"
    previous_solution_path: str | None = None


@dataclass(frozen=True)
class AggressiveReversalOcpResult:
    target_heading_deg: float
    direction_sign: int
    success: bool
    failure_label: str
    time_s: np.ndarray
    x_ref: np.ndarray
    u_ff_norm: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    phase: tuple[str, ...]
    metrics: dict[str, object]
    notes: str


@dataclass(frozen=True)
class CandidateResult:
    method: str
    family_name: str
    attempt_index: int
    result: AggressiveReversalOcpResult
    nlp_constructed: bool = False
    ipopt_called: bool = False
    solver_status: str = "not_run"
    solver_iter_count: int = 0
    solver_objective: float = np.nan
    constraint_residual_max: float = np.nan
    direct_ocp_attempted: bool = False
    direct_ocp_converged: bool = False
    replay_defect_max: float = np.nan
    limiting_mechanism: str = ""


# =============================================================================
# 2) Seed Families and Phase Profiles
# =============================================================================
def target_horizon_s(target_heading_deg: float) -> float:
    """Return the configured horizon for a target heading."""

    target = float(target_heading_deg)
    if target in TARGET_HORIZONS_S:
        return TARGET_HORIZONS_S[target]
    if target <= 30.0:
        return 0.85
    if target <= 90.0:
        return 1.20
    return 1.60


def _validate_config(config: AggressiveReversalOcpConfig) -> None:
    if not np.isfinite(config.dt_s) or config.dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive.")
    if not np.isfinite(config.t_final_s) or config.t_final_s <= 0.0:
        raise ValueError("t_final_s must be finite and positive.")
    ratio = config.t_final_s / config.dt_s
    if not np.isclose(ratio, round(ratio), rtol=1e-12, atol=1e-9):
        raise ValueError("t_final_s must be an integer multiple of dt_s.")
    if config.direction_sign not in (-1, 1):
        raise ValueError("direction_sign must be -1 or +1.")
    if config.wind_mode != "none":
        raise ValueError("aggressive reversal search currently supports only W0/no wind.")
    if config.latency_case != "none":
        raise ValueError("aggressive reversal search currently supports latency_case='none'.")
    if config.ocp_node_count < 1:
        raise ValueError("ocp_node_count must be positive.")


def target_config(
    target_heading_deg: float,
    base: AggressiveReversalOcpConfig | None = None,
) -> AggressiveReversalOcpConfig:
    """Return a config with target-specific horizon and target heading."""

    cfg = AggressiveReversalOcpConfig() if base is None else base
    return AggressiveReversalOcpConfig(
        dt_s=cfg.dt_s,
        t_final_s=target_horizon_s(float(target_heading_deg)),
        target_heading_deg=float(target_heading_deg),
        direction_sign=cfg.direction_sign,
        speed_m_s=cfg.speed_m_s,
        altitude_m=cfg.altitude_m,
        wind_mode=cfg.wind_mode,
        latency_case=cfg.latency_case,
        actuator_tau_s=cfg.actuator_tau_s,
        seed=cfg.seed,
        max_ipopt_iter=cfg.max_ipopt_iter,
        use_nonlinear_ocp=cfg.use_nonlinear_ocp,
        use_phase_seed_fallback=cfg.use_phase_seed_fallback,
        phase_search_scales=cfg.phase_search_scales,
        ocp_node_count=cfg.ocp_node_count,
        ocp_max_cpu_time_s=cfg.ocp_max_cpu_time_s,
        checkpoint_dir=cfg.checkpoint_dir,
        candidate_log_dir=cfg.candidate_log_dir,
        run_id=cfg.run_id,
        previous_solution_path=cfg.previous_solution_path,
    )


def _family_phase_edges(family_name: str, t_final_s: float) -> np.ndarray:
    fractions = {
        "short_perch_yaw_redirect": (0.08, 0.26, 0.56, 0.70, 0.86),
        "long_perch_slow_redirect": (0.08, 0.34, 0.64, 0.76, 0.90),
        "roll_dominant_banked_redirect": (0.08, 0.28, 0.58, 0.72, 0.88),
        "split_pulse_redirect": (0.08, 0.26, 0.50, 0.68, 0.86),
        "early_unload_descend_capture": (0.08, 0.24, 0.46, 0.62, 0.82),
    }
    if family_name not in fractions:
        raise ValueError(f"unknown seed family: {family_name}.")
    return np.array((0.0, *fractions[family_name], 1.0), dtype=float) * float(t_final_s)


def phase_labels_for_family(
    family_name: str,
    time_s: np.ndarray,
    t_final_s: float,
) -> tuple[str, ...]:
    """Return a phase label for each time sample."""

    edges = _family_phase_edges(family_name, t_final_s)
    labels: list[str] = []
    for time_value in np.asarray(time_s, dtype=float):
        index = int(np.searchsorted(edges, time_value, side="right") - 1)
        index = int(np.clip(index, 0, len(PHASES) - 1))
        labels.append(PHASES[index])
    return tuple(labels)


def _family_amplitudes(family_name: str) -> dict[str, tuple[float, float, float]]:
    return {
        "short_perch_yaw_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.10, 0.78, 0.10),
            "slow_redirect": (0.38, 0.55, 0.78),
            "heading_capture": (-0.22, -0.15, -0.25),
            "unload_descend": (0.0, -0.55, 0.0),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "long_perch_slow_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.12, 0.92, 0.12),
            "slow_redirect": (0.48, 0.72, 0.95),
            "heading_capture": (-0.26, -0.12, -0.32),
            "unload_descend": (0.0, -0.48, 0.0),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "roll_dominant_banked_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.25, 0.70, 0.12),
            "slow_redirect": (0.82, 0.45, 0.42),
            "heading_capture": (-0.36, -0.12, -0.20),
            "unload_descend": (-0.05, -0.45, 0.0),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "split_pulse_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.14, 0.88, 0.20),
            "slow_redirect": (0.58, 0.62, 0.92),
            "heading_capture": (-0.48, -0.20, -0.62),
            "unload_descend": (0.35, -0.50, 0.60),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "early_unload_descend_capture": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.10, 0.72, 0.10),
            "slow_redirect": (0.35, 0.38, 0.70),
            "heading_capture": (-0.22, -0.25, -0.30),
            "unload_descend": (0.0, -0.72, 0.0),
            "exit_glide": (0.0, -0.05, 0.0),
        },
    }[family_name]


def seed_family_inventory() -> tuple[str, ...]:
    """Return the fixed guided manoeuvre-family inventory."""

    return SEED_FAMILIES


def phase_seed_command_profile(
    config: AggressiveReversalOcpConfig,
    time_s: np.ndarray,
    family_name: str = "short_perch_yaw_redirect",
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    """Return a deterministic phase-structured normalised command seed."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    labels = phase_labels_for_family(family_name, time, config.t_final_s)
    amplitudes = _family_amplitudes(family_name)
    commands = np.zeros((time.size, 3), dtype=float)
    for index, label in enumerate(labels):
        aileron, elevator, rudder = amplitudes[label]
        commands[index, 0] = config.direction_sign * amplitude_scale * aileron
        commands[index, 1] = amplitude_scale * elevator
        commands[index, 2] = config.direction_sign * amplitude_scale * rudder
    return np.clip(commands, -1.0, 1.0)


def next_family_for_failure(failure_label: str, current_family: str) -> tuple[str, str]:
    """Return deterministic next family and reason for a failure label."""

    if failure_label == "under_turning":
        if current_family != "long_perch_slow_redirect":
            return "long_perch_slow_redirect", "under_turning_longer_perch"
        return "split_pulse_redirect", "under_turning_split_redirect"
    if failure_label == "speed_low":
        return "early_unload_descend_capture", "speed_low_earlier_unload"
    if failure_label in {"true_safety_violation", "wall_violation", "floor_violation", "ceiling_violation"}:
        return current_family, "safety_boundary_shorten_redirect_or_raise_penalty"
    if failure_label == "alpha_boundary":
        return current_family, "alpha_boundary_soften_pitch_brake"
    if failure_label == "actuator_saturation_limited":
        return current_family, "saturation_smooth_and_lengthen_phase"
    if failure_label == "terminal_recovery_limited":
        return "early_unload_descend_capture", "terminal_recovery_extend_exit_glide"
    if failure_label == "solver_failure":
        return current_family, "solver_failure_retry_best_finite_phase_search"
    if failure_label == "model_boundary_only":
        return current_family, "model_boundary_stop_family_escalation"
    return current_family, "no_family_change"


# =============================================================================
# 3) Replay and Metrics
# =============================================================================
def build_aggressive_initial_state(
    config: AggressiveReversalOcpConfig,
    aircraft: object | None = None,
) -> np.ndarray:
    """Return a safe trim-like initial state inside the true safety volume."""

    _validate_config(config)
    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    trim = solve_straight_trim(
        aircraft_model,
        TrimTarget(
            speed_m_s=float(config.speed_m_s),
            altitude_m=float(config.altitude_m),
            wind_model=None,
            wind_mode="none",
            actuator_tau_s=config.actuator_tau_s,
        ),
    )
    if not trim.converged:
        raise ValueError("cannot build aggressive reversal initial state from nonconverged trim.")
    state = as_state_vector(trim.x_trim).copy()
    state[0:3] = [1.25, 2.2, float(config.altitude_m)]
    return state


def _time_grid(config: AggressiveReversalOcpConfig) -> np.ndarray:
    step_count = int(round(config.t_final_s / config.dt_s))
    return np.arange(step_count + 1, dtype=float) * float(config.dt_s)


def _speed_alpha_beta(x_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_b = np.asarray(x_log, dtype=float)[:, 6:9]
    speed = np.linalg.norm(velocity_b, axis=1)
    alpha = np.arctan2(velocity_b[:, 2], velocity_b[:, 0])
    beta = np.zeros_like(speed)
    valid = speed > 1e-9
    beta[valid] = np.arcsin(np.clip(velocity_b[valid, 1] / speed[valid], -1.0, 1.0))
    return speed, alpha, beta


def unwrapped_signed_heading_change_deg(psi_rad: np.ndarray, direction_sign: int) -> float:
    """Return signed target-direction heading progress in degrees."""

    psi = np.unwrap(np.asarray(psi_rad, dtype=float).reshape(-1))
    if psi.size == 0:
        return float("nan")
    return float(np.rad2deg(int(direction_sign) * (psi[-1] - psi[0])))


def _margin_rows(x_log: np.ndarray) -> list[dict[str, float]]:
    return [
        position_margin_m(position, TRUE_SAFE_BOUNDS)
        for position in np.asarray(x_log, dtype=float)[:, 0:3]
        if np.all(np.isfinite(position))
    ]


def _min_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.min([row[key] for row in rows]))


def _saturation_metrics(u_requested: np.ndarray, u_applied: np.ndarray, dt_s: float) -> tuple[float, float]:
    clipped = np.any(np.abs(np.asarray(u_requested) - np.asarray(u_applied)) > 1e-12, axis=1)
    if clipped.size <= 1:
        return 0.0, 0.0
    count = int(np.count_nonzero(clipped[:-1]))
    return float(count / clipped[:-1].size), float(count * dt_s)


def replay_aggressive_candidate(
    x0: np.ndarray,
    u_norm: np.ndarray,
    time_s: np.ndarray,
    config: AggressiveReversalOcpConfig,
    aircraft: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay candidate commands using current plant and command bridge."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    requested = np.asarray(u_norm, dtype=float)
    if requested.shape != (time.size, 3):
        raise ValueError("u_norm must have shape (N, 3) matching time_s.")
    x_log = np.empty((time.size, STATE_SIZE), dtype=float)
    applied = np.empty((time.size, 3), dtype=float)
    command_rad = np.empty((time.size, 3), dtype=float)
    x_log[0] = as_state_vector(x0)
    for index in range(time.size):
        applied[index] = clip_normalised_command(requested[index])
        command_rad[index] = normalised_command_to_surface_rad(applied[index])
        if index == time.size - 1:
            break
        dt = float(time[index + 1] - time[index])
        x_log[index + 1] = rk4_step(
            x_log[index],
            command_rad[index],
            dt,
            aircraft,
            None,
            config.wind_mode,
            config.actuator_tau_s,
        )
        if not np.all(np.isfinite(x_log[index + 1])):
            x_log[index + 2 :] = np.nan
            break
    return x_log, applied, command_rad


def _energy_exploitation(x_log: np.ndarray, phase: tuple[str, ...]) -> bool:
    labels = np.asarray(phase)
    mask = labels == "pitch_brake"
    if np.count_nonzero(mask) < 2:
        return False
    indices = np.where(mask)[0]
    start = indices[0]
    stop = indices[-1]
    speed = np.linalg.norm(x_log[:, 6:9], axis=1)
    altitude_gain = float(x_log[stop, STATE_INDEX["z_w"]] - x_log[start, STATE_INDEX["z_w"]])
    speed_gain = float(speed[stop] - speed[start])
    return bool(altitude_gain > 0.02 and speed_gain > 0.05)


def metrics_for_candidate(
    config: AggressiveReversalOcpConfig,
    time_s: np.ndarray,
    x_log: np.ndarray,
    u_requested: np.ndarray,
    u_applied: np.ndarray,
    phase: tuple[str, ...],
) -> dict[str, object]:
    """Return scalar metrics and classification for one candidate."""

    finite = bool(np.all(np.isfinite(x_log)))
    speed, alpha, beta = _speed_alpha_beta(x_log)
    margins = _margin_rows(x_log)
    saturation_fraction, saturation_time_s = _saturation_metrics(
        u_requested,
        u_applied,
        config.dt_s,
    )
    actual_heading = unwrapped_signed_heading_change_deg(
        x_log[:, STATE_INDEX["psi"]],
        config.direction_sign,
    )
    terminal_recoverable = bool(finite and terminal_aggressive_recoverable_proxy(x_log[-1]))
    true_safe = bool(
        finite
        and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in x_log[:, 0:3])
    )
    replay_finite = bool(finite)
    primitive_success = bool(
        replay_finite
        and true_safe
        and actual_heading >= 0.80 * float(config.target_heading_deg)
        and float(speed[-1]) >= 5.0
        and terminal_recoverable
    )
    energy_exploitation = bool(finite and _energy_exploitation(x_log, phase))
    if primitive_success:
        failure_label = "success"
        notes = "target_achieved_and_recoverable"
    elif not replay_finite:
        failure_label = "nonfinite_state"
        notes = "replay_contains_nonfinite_state"
    elif not true_safe:
        failure_label = "true_safety_violation"
        notes = "trajectory_exits_true_safety_volume"
    elif energy_exploitation:
        failure_label = "model_boundary_only"
        notes = "w0_energy_exploitation"
    elif actual_heading < 0.80 * float(config.target_heading_deg):
        failure_label = "under_turning"
        notes = "heading_change_below_80_percent_target"
    elif float(speed[-1]) < 5.0:
        failure_label = "speed_low"
        notes = "terminal_speed_below_recoverability_threshold"
    elif np.nanmax(np.abs(np.rad2deg(beta))) > 35.0:
        failure_label = "beta_boundary"
        notes = "beta_exposure_exceeds_boundary"
    elif np.nanmax(np.abs(np.rad2deg(alpha))) > 55.0:
        failure_label = "alpha_boundary"
        notes = "alpha_exposure_exceeds_surrogate_boundary"
    elif np.nanmax(np.linalg.norm(x_log[:, 9:12], axis=1)) > 4.5:
        failure_label = "rate_boundary"
        notes = "body_rate_exceeds_boundary"
    elif saturation_fraction > 0.75:
        failure_label = "actuator_saturation_limited"
        notes = "commands_excessively_clipped"
    elif not terminal_recoverable:
        failure_label = "terminal_recovery_limited"
        notes = "terminal_state_not_glide_or_recovery_compatible"
    else:
        failure_label = "model_boundary_only"
        notes = "simulation_boundary_candidate"
    if failure_label not in FAILURE_LABELS:
        raise ValueError(f"unknown failure label: {failure_label}")
    return {
        "target_heading_deg": float(config.target_heading_deg),
        "actual_heading_change_deg": float(actual_heading),
        "heading_error_deg": float(config.target_heading_deg - actual_heading),
        "forward_travel_m": float(abs(x_log[-1, 0] - x_log[0, 0])),
        "turn_volume_proxy_m2": float(abs(x_log[-1, 0] - x_log[0, 0]) * abs(x_log[-1, 1] - x_log[0, 1])),
        "height_change_m": float(x_log[-1, 2] - x_log[0, 2]),
        "max_altitude_gain_m": float(np.nanmax(x_log[:, 2]) - x_log[0, 2]),
        "speed_min_m_s": float(np.nanmin(speed)),
        "terminal_speed_m_s": float(speed[-1]),
        "max_alpha_deg": float(np.nanmax(np.abs(np.rad2deg(alpha)))),
        "max_beta_deg": float(np.nanmax(np.abs(np.rad2deg(beta)))),
        "max_bank_deg": float(np.nanmax(np.abs(np.rad2deg(x_log[:, STATE_INDEX["phi"]])))),
        "max_pitch_deg": float(np.nanmax(np.abs(np.rad2deg(x_log[:, STATE_INDEX["theta"]])))),
        "max_rate_rad_s": float(np.nanmax(np.linalg.norm(x_log[:, 9:12], axis=1))),
        "min_true_wall_margin_m": _min_metric(margins, "min_wall_margin_m"),
        "min_floor_margin_m": _min_metric(margins, "floor_margin_m"),
        "min_ceiling_margin_m": _min_metric(margins, "ceiling_margin_m"),
        "saturation_fraction": saturation_fraction,
        "saturation_time_s": saturation_time_s,
        "finite_state_success": replay_finite,
        "rollout_success": bool(replay_finite and true_safe),
        "source_trajectory_success": replay_finite,
        "terminal_recoverable_proxy": terminal_recoverable,
        "recoverable": terminal_recoverable,
        "success": primitive_success,
        "primitive_success": primitive_success,
        "failure_label": failure_label,
        "notes": notes,
        "energy_exploitation": energy_exploitation,
    }


def _result_from_replay(
    config: AggressiveReversalOcpConfig,
    family_name: str,
    method: str,
    attempt_index: int,
    x0: np.ndarray,
    time_s: np.ndarray,
    u_requested: np.ndarray,
    aircraft: object,
) -> CandidateResult:
    x_log, u_applied, command_rad = replay_aggressive_candidate(
        x0,
        u_requested,
        time_s,
        config,
        aircraft,
    )
    phase = phase_labels_for_family(family_name, time_s, config.t_final_s)
    metrics = metrics_for_candidate(config, time_s, x_log, u_requested, u_applied, phase)
    result = AggressiveReversalOcpResult(
        target_heading_deg=float(config.target_heading_deg),
        direction_sign=int(config.direction_sign),
        success=bool(metrics["success"]),
        failure_label=str(metrics["failure_label"]),
        time_s=time_s,
        x_ref=x_log,
        u_ff_norm=u_requested,
        u_norm_applied=u_applied,
        delta_cmd_rad=command_rad,
        phase=phase,
        metrics=metrics,
        notes=str(metrics["notes"]),
    )
    return CandidateResult(
        method=method,
        family_name=family_name,
        attempt_index=attempt_index,
        result=result,
        direct_ocp_attempted=False,
        limiting_mechanism=limiting_mechanism(str(metrics["failure_label"])),
    )


# =============================================================================
# 4) Guided Search and Direct OCP Attempts
# =============================================================================
def _score_candidate(candidate: CandidateResult) -> tuple[int, int, float, float, float]:
    metrics = candidate.result.metrics
    return (
        int(bool(metrics["success"])),
        int(bool(metrics["recoverable"])),
        float(metrics["actual_heading_change_deg"]),
        float(metrics["terminal_speed_m_s"]),
        -float(metrics["saturation_fraction"]),
    )


def limiting_mechanism(failure_label: str) -> str:
    if failure_label == "success":
        return "none"
    if failure_label == "solver_failure":
        return "optimiser_convergence"
    if failure_label in {"true_safety_violation", "wall_violation", "floor_violation", "ceiling_violation"}:
        return "physical_safety_boundary"
    if failure_label == "model_boundary_only":
        return "model_boundary_behaviour"
    if failure_label in {"under_turning", "terminal_recovery_limited"}:
        return "insufficient_manoeuvre_seed"
    return "physical_boundary"


def _symbolic_command_to_rad(command_norm: ca.MX) -> ca.MX:
    from latency import AGGREGATE_LIMITS

    values = []
    for index, name in enumerate(("delta_a", "delta_e", "delta_r")):
        limit = AGGREGATE_LIMITS[name]
        positive = np.deg2rad(float(limit.positive_deg))
        negative = np.deg2rad(float(limit.negative_deg))
        value = command_norm[index]
        values.append(ca.if_else(value >= 0.0, value * positive, -value * negative))
    return ca.vertcat(*values)


def _downsample_indices(length: int, count: int) -> np.ndarray:
    return np.round(np.linspace(0, length - 1, count)).astype(int)


def _replay_defect(x_ref: np.ndarray, command_rad: np.ndarray, time_s: np.ndarray, config: AggressiveReversalOcpConfig, aircraft: object) -> float:
    if time_s.size < 2:
        return 0.0
    replay = np.empty_like(x_ref)
    replay[0] = x_ref[0]
    for index in range(time_s.size - 1):
        replay[index + 1] = rk4_step(
            replay[index],
            command_rad[index],
            float(time_s[index + 1] - time_s[index]),
            aircraft,
            None,
            config.wind_mode,
            config.actuator_tau_s,
        )
    return float(np.nanmax(np.abs(replay - x_ref)))


def direct_multiple_shooting_attempt(
    seed: CandidateResult,
    config: AggressiveReversalOcpConfig,
    aircraft: object,
    attempt_index: int,
) -> CandidateResult:
    """Attempt a real CasADi/IPOPT multiple-shooting refinement."""

    start = time.monotonic()
    nlp_constructed = False
    ipopt_called = False
    solver_status = "not_run"
    solver_iter_count = 0
    solver_objective = float("nan")
    constraint_residual_max = float("nan")
    replay_defect_max = float("nan")
    try:
        node_count = int(config.ocp_node_count)
        indices = _downsample_indices(seed.result.time_s.size, node_count + 1)
        seed_time = seed.result.time_s[indices]
        seed_state = seed.result.x_ref[indices]
        seed_command = seed.result.u_ff_norm[indices]
        dynamics = build_symbolic_dynamics(
            aircraft,
            actuator_tau_s=config.actuator_tau_s,
            wind_mode="none",
        )
        opti = ca.Opti()
        x_var = opti.variable(STATE_SIZE, node_count + 1)
        u_var = opti.variable(3, node_count + 1)
        opti.subject_to(x_var[:, 0] == seed_state[0])
        objective = 0.0
        defects = []
        for node in range(node_count):
            dt = float(seed_time[node + 1] - seed_time[node])
            u_rad = _symbolic_command_to_rad(u_var[:, node])
            f1 = dynamics.function(x_var[:, node], u_rad)
            f2 = dynamics.function(x_var[:, node] + 0.5 * dt * f1, u_rad)
            f3 = dynamics.function(x_var[:, node] + 0.5 * dt * f2, u_rad)
            f4 = dynamics.function(x_var[:, node] + dt * f3, u_rad)
            x_next = x_var[:, node] + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
            defect = x_var[:, node + 1] - x_next
            defects.append(defect)
            opti.subject_to(defect == 0)
            opti.subject_to(opti.bounded(-1.0, u_var[:, node], 1.0))
            opti.subject_to(opti.bounded(TRUE_SAFE_BOUNDS.x_w_m[0], x_var[0, node], TRUE_SAFE_BOUNDS.x_w_m[1]))
            opti.subject_to(opti.bounded(TRUE_SAFE_BOUNDS.y_w_m[0], x_var[1, node], TRUE_SAFE_BOUNDS.y_w_m[1]))
            opti.subject_to(opti.bounded(0.05, x_var[2, node], TRUE_SAFE_BOUNDS.z_w_m[1]))
            objective += 0.01 * ca.sumsqr(u_var[:, node] - seed_command[node])
            objective += 0.001 * ca.sumsqr(x_var[9:12, node])
        opti.subject_to(opti.bounded(-1.0, u_var[:, -1], 1.0))
        target_rad = np.deg2rad(float(config.target_heading_deg))
        heading_progress = int(config.direction_sign) * (x_var[STATE_INDEX["psi"], -1] - x_var[STATE_INDEX["psi"], 0])
        terminal_speed = ca.sqrt(ca.sumsqr(x_var[6:9, -1]) + 1e-12)
        objective += 25.0 * (heading_progress - target_rad) ** 2
        objective += 5.0 * ca.fmax(5.0 - terminal_speed, 0.0) ** 2
        objective += 0.1 * ca.sumsqr(x_var[9:12, -1])
        opti.minimize(objective)
        opti.set_initial(x_var, seed_state.T)
        opti.set_initial(u_var, seed_command.T)
        opti.solver(
            "ipopt",
            {
                "expand": False,
            },
            {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": int(config.max_ipopt_iter),
                "ipopt.max_cpu_time": float(config.ocp_max_cpu_time_s),
                "print_time": 0,
            },
        )
        nlp_constructed = True
        ipopt_called = True
        solution = opti.solve()
        solver_status = str(opti.stats().get("return_status", "unknown"))
        solver_iter_count = int(opti.stats().get("iter_count", 0))
        x_nodes = np.asarray(solution.value(x_var)).T
        u_nodes = np.asarray(solution.value(u_var)).T
        solver_objective = float(solution.value(objective))
        if defects:
            defect_values = [np.asarray(solution.value(defect)).reshape(-1) for defect in defects]
            constraint_residual_max = float(np.max(np.abs(np.concatenate(defect_values))))
        command_rad = np.vstack([normalised_command_to_surface_rad(row) for row in u_nodes])
        replay_defect_max = _replay_defect(x_nodes, command_rad, seed_time, config, aircraft)
        phase = phase_labels_for_family(seed.family_name, seed_time, float(seed_time[-1]))
        metrics = metrics_for_candidate(config, seed_time, x_nodes, u_nodes, np.clip(u_nodes, -1.0, 1.0), phase)
        if replay_defect_max > REPLAY_DEFECT_TOL:
            metrics["success"] = False
            metrics["primitive_success"] = False
            metrics["failure_label"] = "model_boundary_only"
            metrics["notes"] = "direct_ocp_replay_defect_exceeded"
        result = AggressiveReversalOcpResult(
            target_heading_deg=float(config.target_heading_deg),
            direction_sign=int(config.direction_sign),
            success=bool(metrics["success"]),
            failure_label=str(metrics["failure_label"]),
            time_s=seed_time,
            x_ref=x_nodes,
            u_ff_norm=u_nodes,
            u_norm_applied=np.clip(u_nodes, -1.0, 1.0),
            delta_cmd_rad=command_rad,
            phase=phase,
            metrics=metrics,
            notes=str(metrics["notes"]),
        )
        converged = bool(
            solver_status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
            and np.isfinite(replay_defect_max)
            and replay_defect_max <= REPLAY_DEFECT_TOL
        )
        return CandidateResult(
            method="direct_ocp",
            family_name=seed.family_name,
            attempt_index=attempt_index,
            result=result,
            nlp_constructed=nlp_constructed,
            ipopt_called=ipopt_called,
            solver_status=solver_status,
            solver_iter_count=solver_iter_count,
            solver_objective=solver_objective,
            constraint_residual_max=constraint_residual_max,
            direct_ocp_attempted=True,
            direct_ocp_converged=converged,
            replay_defect_max=replay_defect_max,
            limiting_mechanism=limiting_mechanism(str(metrics["failure_label"])),
        )
    except Exception as exc:
        metrics = dict(seed.result.metrics)
        metrics["success"] = False
        metrics["primitive_success"] = False
        metrics["failure_label"] = "solver_failure"
        metrics["notes"] = f"direct_ocp_failed:{type(exc).__name__}"
        result = AggressiveReversalOcpResult(
            target_heading_deg=seed.result.target_heading_deg,
            direction_sign=seed.result.direction_sign,
            success=False,
            failure_label="solver_failure",
            time_s=seed.result.time_s,
            x_ref=seed.result.x_ref,
            u_ff_norm=seed.result.u_ff_norm,
            u_norm_applied=seed.result.u_norm_applied,
            delta_cmd_rad=seed.result.delta_cmd_rad,
            phase=seed.result.phase,
            metrics=metrics,
            notes=str(metrics["notes"]),
        )
        return CandidateResult(
            method="direct_ocp",
            family_name=seed.family_name,
            attempt_index=attempt_index,
            result=result,
            nlp_constructed=nlp_constructed,
            ipopt_called=ipopt_called,
            solver_status=solver_status,
            solver_iter_count=solver_iter_count,
            solver_objective=solver_objective,
            constraint_residual_max=constraint_residual_max,
            direct_ocp_attempted=bool(nlp_constructed and ipopt_called),
            direct_ocp_converged=False,
            replay_defect_max=replay_defect_max,
            limiting_mechanism="optimiser_convergence",
        )
    finally:
        _ = time.monotonic() - start


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(item) for item in row) + "\n")


def _json_float(value: object) -> float | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def checkpoint_candidate(
    candidate: CandidateResult,
    config: AggressiveReversalOcpConfig,
    label: str,
) -> dict[str, str]:
    """Write compact checkpoint JSON plus CSV paths for one candidate."""

    if config.checkpoint_dir is None:
        return {}
    checkpoint_dir = Path(config.checkpoint_dir)
    candidate_dir = Path(config.candidate_log_dir or checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    target_token = f"{int(round(config.target_heading_deg)):0{TARGET_TOKEN_WIDTH}d}"
    stem = f"target_{target_token}_{label}_{candidate.attempt_index:03d}_{config.run_id}"
    trajectory_path = candidate_dir / f"{stem}_trajectory.csv"
    command_path = candidate_dir / f"{stem}_commands.csv"
    _write_csv(
        trajectory_path,
        ("time_s", *(f"x{i}" for i in range(STATE_SIZE))),
        [
            (candidate.result.time_s[index], *candidate.result.x_ref[index])
            for index in range(candidate.result.time_s.size)
        ],
    )
    _write_csv(
        command_path,
        (
            "time_s",
            "u_norm_requested_delta_a_norm",
            "u_norm_requested_delta_e_norm",
            "u_norm_requested_delta_r_norm",
            "u_norm_applied_delta_a_norm",
            "u_norm_applied_delta_e_norm",
            "u_norm_applied_delta_r_norm",
            "delta_cmd_rad_delta_a_cmd",
            "delta_cmd_rad_delta_e_cmd",
            "delta_cmd_rad_delta_r_cmd",
        ),
        [
            (
                candidate.result.time_s[index],
                *candidate.result.u_ff_norm[index],
                *candidate.result.u_norm_applied[index],
                *candidate.result.delta_cmd_rad[index],
            )
            for index in range(candidate.result.time_s.size)
        ],
    )
    checkpoint = {
        "target_heading_deg": float(config.target_heading_deg),
        "direction_sign": int(config.direction_sign),
        "method": candidate.method,
        "family_name": candidate.family_name,
        "attempt_index": int(candidate.attempt_index),
        "success": bool(candidate.result.success),
        "failure_label": candidate.result.failure_label,
        "notes": candidate.result.notes,
        "replay_finite": bool(candidate.result.metrics["finite_state_success"]),
        "recoverable": bool(candidate.result.metrics["recoverable"]),
        "primitive_success": bool(candidate.result.metrics["primitive_success"]),
        "actual_heading_change_deg": float(candidate.result.metrics["actual_heading_change_deg"]),
        "terminal_speed_m_s": float(candidate.result.metrics["terminal_speed_m_s"]),
        "nlp_constructed": bool(candidate.nlp_constructed),
        "ipopt_called": bool(candidate.ipopt_called),
        "solver_status": candidate.solver_status,
        "solver_iter_count": int(candidate.solver_iter_count),
        "solver_objective": _json_float(candidate.solver_objective),
        "constraint_residual_max": _json_float(candidate.constraint_residual_max),
        "replay_defect_max": _json_float(candidate.replay_defect_max),
        "direct_ocp_attempted": bool(candidate.direct_ocp_attempted),
        "direct_ocp_converged": bool(candidate.direct_ocp_converged),
        "limiting_mechanism": candidate.limiting_mechanism,
        "trajectory_csv": trajectory_path.as_posix(),
        "commands_csv": command_path.as_posix(),
    }
    checkpoint_path = checkpoint_dir / f"{stem}.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, allow_nan=False),
        encoding="ascii",
    )
    return {
        "checkpoint_json": checkpoint_path.as_posix(),
        "trajectory_csv": trajectory_path.as_posix(),
        "commands_csv": command_path.as_posix(),
    }


def guided_phase_search(
    config: AggressiveReversalOcpConfig,
    x0: np.ndarray,
    aircraft: object,
) -> list[CandidateResult]:
    """Run fixed manoeuvre-family seeds plus bounded amplitude variations."""

    time_s = _time_grid(config)
    candidates: list[CandidateResult] = []
    attempt_index = 0
    for family_name in seed_family_inventory():
        for scale in config.phase_search_scales:
            command = phase_seed_command_profile(config, time_s, family_name, scale)
            candidate = _result_from_replay(
                config,
                family_name,
                "phase_search",
                attempt_index,
                x0,
                time_s,
                command,
                aircraft,
            )
            checkpoint_candidate(candidate, config, "phase_search")
            candidates.append(candidate)
            attempt_index += 1
    return candidates


def _select_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("no aggressive reversal candidates were generated.")
    return max(candidates, key=_score_candidate)


def _best_by(candidates: list[CandidateResult], key: str) -> CandidateResult | None:
    filtered = [candidate for candidate in candidates if bool(candidate.result.metrics[key])]
    if not filtered:
        return None
    return max(filtered, key=_score_candidate)


# =============================================================================
# 5) Public Solve Workflow
# =============================================================================
def solve_aggressive_reversal_ocp(
    config: AggressiveReversalOcpConfig,
    x0: np.ndarray | None = None,
    aircraft: object | None = None,
) -> AggressiveReversalOcpResult:
    """Return a finite aggressive-reversal candidate or an honest failure result."""

    _validate_config(config)
    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    initial_state = build_aggressive_initial_state(config, aircraft_model) if x0 is None else as_state_vector(x0)
    candidates = guided_phase_search(config, initial_state, aircraft_model)
    phase_best = _select_candidate(candidates)
    next_family, reason = next_family_for_failure(
        phase_best.result.failure_label,
        phase_best.family_name,
    )
    if next_family != phase_best.family_name:
        time_s = _time_grid(config)
        command = phase_seed_command_profile(config, time_s, next_family, 1.0)
        followup = _result_from_replay(
            config,
            next_family,
            "phase_search_followup",
            len(candidates),
            initial_state,
            time_s,
            command,
            aircraft_model,
        )
        followup.result.metrics["next_family_reason"] = reason
        checkpoint_candidate(followup, config, "phase_followup")
        candidates.append(followup)
    if config.use_nonlinear_ocp:
        best_for_ocp = _select_candidate(candidates)
        ocp_candidate = direct_multiple_shooting_attempt(
            best_for_ocp,
            config,
            aircraft_model,
            len(candidates),
        )
        checkpoint_candidate(ocp_candidate, config, "direct_ocp")
        candidates.append(ocp_candidate)
    selected = _select_candidate(candidates)
    ocp_candidates = [
        candidate for candidate in candidates if candidate.method == "direct_ocp"
    ]
    ocp_evidence = ocp_candidates[-1] if ocp_candidates else None
    finite_best = _best_by(candidates, "finite_state_success")
    recoverable_best = _best_by(candidates, "recoverable")
    successful_best = _best_by(candidates, "success")
    selected.result.metrics.update(
        {
            "families_attempted": ";".join(seed_family_inventory()),
            "selected_family": selected.family_name,
            "selected_method": selected.method,
            "next_family_reason": reason,
            "limiting_mechanism": selected.limiting_mechanism,
            "best_finite_candidate": finite_best.family_name if finite_best else "",
            "best_recoverable_candidate": recoverable_best.family_name if recoverable_best else "",
            "best_successful_candidate": successful_best.family_name if successful_best else "",
            "nlp_constructed": bool(ocp_evidence and ocp_evidence.nlp_constructed),
            "ipopt_called": bool(ocp_evidence and ocp_evidence.ipopt_called),
            "solver_status": ocp_evidence.solver_status if ocp_evidence else "not_run",
            "solver_iter_count": ocp_evidence.solver_iter_count if ocp_evidence else 0,
            "solver_objective": ocp_evidence.solver_objective if ocp_evidence else np.nan,
            "constraint_residual_max": (
                ocp_evidence.constraint_residual_max if ocp_evidence else np.nan
            ),
            "replay_defect_max": ocp_evidence.replay_defect_max if ocp_evidence else np.nan,
            "direct_ocp_attempted": bool(
                ocp_evidence and ocp_evidence.direct_ocp_attempted
            ),
            "direct_ocp_converged": bool(
                ocp_evidence and ocp_evidence.direct_ocp_converged
            ),
            "phase_search_attempted": True,
            "replay_finite": selected.result.metrics["finite_state_success"],
        }
    )
    checkpoint_candidate(selected, config, "selected")
    return selected.result
