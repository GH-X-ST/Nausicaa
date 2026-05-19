from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
)
from flight_dynamics import adapt_glider, state_derivative
from glider import build_nausicaa_glider
from latency import (
    LatencyCaseConfig,
    actuator_tau_for_case,
    latency_adjusted_command_sample,
    latency_case_config,
)
from metric_contract import empty_metric_row, validate_metric_row
from scenario_contract import LATENCY_CASES, WIND_MODES
from state_contract import STATE_INDEX, STATE_SIZE


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data Containers
# 2) Timing, Schedule, and Configuration Validation
# 3) Integration
# 4) Rollout Metrics and Public Workflow
# =============================================================================


# =============================================================================
# 1) Data Containers
# =============================================================================
@dataclass(frozen=True)
class RolloutConfig:
    dt_s: float
    t_final_s: float
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    integrator: str = "rk4"


@dataclass(frozen=True)
class CommandSchedule:
    times_s: np.ndarray
    u_norm_requested: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    time_s: np.ndarray
    x: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    success: bool
    failure_label: str
    metrics: dict[str, object]
    notes: str


# =============================================================================
# 2) Timing, Schedule, and Configuration Validation
# =============================================================================
def _time_step_count(dt_s: float, t_final_s: float) -> int:
    ratio = float(t_final_s) / float(dt_s)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, rtol=1e-12, atol=1e-9):
        raise ValueError("t_final_s must be an integer multiple of dt_s.")
    return rounded


def _time_grid(config: RolloutConfig) -> np.ndarray:
    step_count = _time_step_count(config.dt_s, config.t_final_s)
    return np.arange(step_count + 1, dtype=float) * float(config.dt_s)


def validate_rollout_config(config: RolloutConfig) -> None:
    """Validate timing, wind/latency labels, actuator tau, and integrator."""

    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("rollout dt_s must be finite and positive.")
    if not np.isfinite(float(config.t_final_s)) or float(config.t_final_s) <= 0.0:
        raise ValueError("rollout t_final_s must be finite and positive.")
    _time_step_count(config.dt_s, config.t_final_s)
    if config.wind_mode not in WIND_MODES:
        raise ValueError(f"unknown wind_mode: {config.wind_mode}.")
    if config.latency_case not in LATENCY_CASES:
        raise ValueError(f"unknown latency_case: {config.latency_case}.")
    tau = np.asarray(config.actuator_tau_s, dtype=float)
    if tau.size != 3 or not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("actuator_tau_s must contain three finite positive values.")
    if config.integrator != "rk4":
        raise ValueError("rollout integrator must be 'rk4'.")


def _rollout_latency_case_config(config: RolloutConfig) -> LatencyCaseConfig:
    latency_config = latency_case_config(config.latency_case)
    if config.latency_case != "actuator_lag_only":
        return latency_config

    tau = tuple(
        float(value)
        for value in np.asarray(config.actuator_tau_s, dtype=float).reshape(3)
    )
    max_tau = max(tau)
    return LatencyCaseConfig(
        latency_case=latency_config.latency_case,
        state_feedback_delay_s=0.0,
        command_onset_delay_s=0.0,
        command_transport_delay_s=0.0,
        actuator_tau_s=tau,
        actuator_t50_s=float(max_tau * np.log(2.0)),
        actuator_t90_s=float(max_tau * np.log(10.0)),
        latency_jitter_s=0.0,
        timing_model_version=latency_config.timing_model_version,
        latency_pass_label=latency_config.latency_pass_label,
    )


def _actuator_tau_for_rollout(
    config: RolloutConfig,
    latency_config: LatencyCaseConfig,
) -> tuple[float, float, float]:
    return actuator_tau_for_case(latency_config, fallback_tau_s=config.actuator_tau_s)


def _validate_schedule(schedule: CommandSchedule) -> tuple[np.ndarray, np.ndarray]:
    times_s = np.asarray(schedule.times_s, dtype=float).reshape(-1)
    commands = np.asarray(schedule.u_norm_requested, dtype=float)
    if times_s.size == 0:
        raise ValueError("command schedule must contain at least one time sample.")
    if commands.shape != (times_s.size, 3):
        raise ValueError("command schedule u_norm_requested must have shape (N, 3).")
    if not np.all(np.isfinite(times_s)) or not np.all(np.isfinite(commands)):
        raise ValueError("command schedule must contain only finite values.")
    if np.any(np.diff(times_s) <= 0.0):
        raise ValueError("command schedule times_s must be strictly increasing.")
    return times_s, commands


def make_constant_command_schedule(
    u_norm: np.ndarray,
    t_final_s: float,
    dt_s: float,
) -> CommandSchedule:
    """Return a deterministic zero-order-hold requested command schedule."""

    step_count = _time_step_count(dt_s, t_final_s)
    command = np.asarray(u_norm, dtype=float)
    if command.size != 3:
        raise ValueError("u_norm must contain three normalised command values.")
    command = command.reshape(3)
    if not np.all(np.isfinite(command)):
        raise ValueError("u_norm must contain only finite values.")
    times_s = np.arange(step_count + 1, dtype=float) * float(dt_s)
    commands = np.repeat(command.reshape(1, 3), times_s.size, axis=0)
    return CommandSchedule(times_s=times_s, u_norm_requested=commands)


def sample_command_schedule(schedule: CommandSchedule, t_s: float) -> np.ndarray:
    """Return the requested normalised command at time t_s by zero-order hold."""

    times_s, commands = _validate_schedule(schedule)
    if not np.isfinite(float(t_s)):
        raise ValueError("sample time must be finite.")
    index = int(np.searchsorted(times_s, float(t_s), side="right") - 1)
    index = int(np.clip(index, 0, times_s.size - 1))
    return commands[index].copy()


# =============================================================================
# 3) Integration
# =============================================================================
def rk4_step(
    x: np.ndarray,
    delta_cmd_rad: np.ndarray,
    dt_s: float,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    actuator_tau_s: tuple[float, float, float],
) -> np.ndarray:
    """Advance one RK4 step using physical radian surface targets."""

    state = np.asarray(x, dtype=float).reshape(STATE_SIZE)
    command_rad = np.asarray(delta_cmd_rad, dtype=float).reshape(3)
    dt = float(dt_s)
    # The plant contract is physical radian surface targets, not normalised
    # command requests. This is the only signal passed to state_derivative.
    k1 = state_derivative(
        state,
        command_rad,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )
    k2 = state_derivative(
        state + 0.5 * dt * k1,
        command_rad,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )
    k3 = state_derivative(
        state + 0.5 * dt * k2,
        command_rad,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )
    k4 = state_derivative(
        state + dt * k3,
        command_rad,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =============================================================================
# 4) Rollout Metrics and Public Workflow
# =============================================================================
def _speed_alpha_beta(x_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_b = np.asarray(x_log[:, 6:9], dtype=float)
    speed = np.linalg.norm(velocity_b, axis=1)
    alpha_rad = np.arctan2(velocity_b[:, 2], velocity_b[:, 0])
    beta_rad = np.zeros_like(speed)
    valid = speed > 1e-9
    beta_rad[valid] = np.arcsin(np.clip(velocity_b[valid, 1] / speed[valid], -1.0, 1.0))
    return speed, np.rad2deg(alpha_rad), np.rad2deg(beta_rad)


def _safe_nanmax(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _safe_nanmin(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _margin_series(x_log: np.ndarray) -> dict[str, np.ndarray]:
    rows = [
        position_margin_m(position_w, TRUE_SAFE_BOUNDS)
        for position_w in np.asarray(x_log[:, 0:3], dtype=float)
        if np.all(np.isfinite(position_w))
    ]
    if not rows:
        return {
            "min_wall_margin_m": np.array([np.nan]),
            "floor_margin_m": np.array([np.nan]),
            "ceiling_margin_m": np.array([np.nan]),
        }
    return {
        "min_wall_margin_m": np.array([row["min_wall_margin_m"] for row in rows]),
        "floor_margin_m": np.array([row["floor_margin_m"] for row in rows]),
        "ceiling_margin_m": np.array([row["ceiling_margin_m"] for row in rows]),
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
    # Saturation duration is counted over integration intervals; the terminal
    # command row is logged for audit but does not create an extra time interval.
    interval_clipped = clipped[:-1]
    count = int(np.count_nonzero(interval_clipped))
    return float(count / interval_clipped.size), float(count * dt_s)


def _metric_row(
    time_s: np.ndarray,
    x_log: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    config: RolloutConfig,
    seed: int,
    scenario_name: str,
    failure_label: str,
    notes: str,
    saturation_reference: np.ndarray | None = None,
) -> dict[str, object]:
    metrics = empty_metric_row(include_agile=True)
    speed, alpha_deg, beta_deg = _speed_alpha_beta(x_log)
    margins = _margin_series(x_log)
    rate_norm = np.linalg.norm(x_log[:, 9:12], axis=1)
    saturation_requested = (
        u_norm_requested if saturation_reference is None else saturation_reference
    )
    saturation_fraction, saturation_time_s = _saturation_metrics(
        saturation_requested,
        u_norm_applied,
        float(config.dt_s),
    )
    finite_state_success = bool(np.all(np.isfinite(x_log)))
    rollout_success = bool(
        finite_state_success
        and failure_label == "not_run"
        and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in x_log[:, 0:3])
    )
    metrics.update(
        {
            "run_id": "",
            "seed": int(seed),
            "primitive_name": "none",
            "primitive_family": "none",
            "scenario_name": scenario_name,
            "wind_mode": config.wind_mode,
            "latency_case": config.latency_case,
            "success": False,
            "finite_state_success": finite_state_success,
            "rollout_success": rollout_success,
            "primitive_success": False,
            "closed_loop_replay_success": False,
            "failure_label": failure_label,
            "duration_s": float(time_s[-1] - time_s[0]) if time_s.size else 0.0,
            "initial_speed_m_s": float(speed[0]) if speed.size else float("nan"),
            "terminal_speed_m_s": float(speed[-1]) if speed.size else float("nan"),
            "height_change_m": float(x_log[-1, 2] - x_log[0, 2]),
            "min_true_wall_margin_m": _safe_nanmin(margins["min_wall_margin_m"]),
            "min_floor_margin_m": _safe_nanmin(margins["floor_margin_m"]),
            "min_ceiling_margin_m": _safe_nanmin(margins["ceiling_margin_m"]),
            "max_alpha_deg": _safe_nanmax(np.abs(alpha_deg)),
            "max_beta_deg": _safe_nanmax(np.abs(beta_deg)),
            "max_bank_deg": _safe_nanmax(np.abs(np.rad2deg(x_log[:, 3]))),
            "max_pitch_deg": _safe_nanmax(np.abs(np.rad2deg(x_log[:, 4]))),
            "max_rate_rad_s": _safe_nanmax(rate_norm),
            "saturation_fraction": saturation_fraction,
            "notes": notes,
            "saturation_time_s": saturation_time_s,
        }
    )
    validate_metric_row(metrics)
    return metrics


def _single_row_result(
    x0: np.ndarray,
    command_schedule: CommandSchedule,
    config: RolloutConfig,
    seed: int,
    scenario_name: str,
    failure_label: str,
    notes: str,
) -> RolloutResult:
    requested = sample_command_schedule(command_schedule, 0.0).reshape(1, 3)
    applied = clip_normalised_command(requested[0]).reshape(1, 3)
    command_rad = normalised_command_to_surface_rad(applied[0]).reshape(1, 3)
    time_s = np.array([0.0])
    x_log = np.asarray(x0, dtype=float).reshape(1, STATE_SIZE)
    metrics = _metric_row(
        time_s,
        x_log,
        requested,
        applied,
        config,
        seed,
        scenario_name,
        failure_label,
        notes,
        saturation_reference=applied,
    )
    return RolloutResult(
        time_s=time_s,
        x=x_log,
        u_norm_requested=requested,
        u_norm_applied=applied,
        delta_cmd_rad=command_rad,
        success=False,
        failure_label=failure_label,
        metrics=metrics,
        notes=notes,
    )


def rollout_open_loop_normalised(
    x0: np.ndarray,
    command_schedule: CommandSchedule,
    config: RolloutConfig,
    aircraft: object | None = None,
    wind_model: object = None,
    seed: int = 1,
    scenario_name: str = "rollout_smoke",
) -> RolloutResult:
    """Roll out the plant with requested normalised commands converted to radians."""

    validate_rollout_config(config)
    _validate_schedule(command_schedule)
    state0 = np.asarray(x0, dtype=float)
    if state0.size != STATE_SIZE:
        raise ValueError(f"initial state must contain {STATE_SIZE} values.")
    state0 = state0.reshape(STATE_SIZE)
    if not np.all(np.isfinite(state0)):
        return _single_row_result(
            state0,
            command_schedule,
            config,
            seed,
            scenario_name,
            "nonfinite_state",
            "nonfinite_initial_state",
        )
    if not inside_bounds(state0[0:3], TRUE_SAFE_BOUNDS):
        return _single_row_result(
            state0,
            command_schedule,
            config,
            seed,
            scenario_name,
            "true_safety_violation",
            "initial_state_outside_true_safe_bounds",
        )

    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    latency_config = _rollout_latency_case_config(config)
    actuator_tau = _actuator_tau_for_rollout(config, latency_config)
    schedule_times_s, schedule_commands = _validate_schedule(command_schedule)
    time_s = _time_grid(config)
    sample_count = time_s.size
    x_log = np.empty((sample_count, STATE_SIZE), dtype=float)
    requested_log = np.empty((sample_count, 3), dtype=float)
    applied_log = np.empty((sample_count, 3), dtype=float)
    effective_log = np.empty((sample_count, 3), dtype=float)
    command_rad_log = np.empty((sample_count, 3), dtype=float)
    x_log[0] = state0
    failure_label = "not_run"
    notes = "rollout_smoke_no_primitive"
    final_index = sample_count - 1

    for index in range(sample_count):
        requested = sample_command_schedule(command_schedule, time_s[index])
        delayed_requested = latency_adjusted_command_sample(
            schedule_times_s,
            schedule_commands,
            time_s[index],
            latency_config,
        )
        applied = clip_normalised_command(delayed_requested)
        command_rad = normalised_command_to_surface_rad(applied)
        if config.latency_case == "none":
            # Ideal timing is a rollout-level ablation: surfaces track the
            # current command target instantly before the pure plant step.
            x_log[index, 12:15] = command_rad
        requested_log[index] = requested
        effective_log[index] = delayed_requested
        applied_log[index] = applied
        command_rad_log[index] = command_rad
        if index == sample_count - 1:
            break

        next_state = rk4_step(
            x_log[index],
            command_rad,
            float(config.dt_s),
            aircraft_model,
            wind_model,
            config.wind_mode,
            actuator_tau,
        )
        x_log[index + 1] = next_state
        if not np.all(np.isfinite(next_state)):
            failure_label = "nonfinite_state"
            notes = "nonfinite_state"
            final_index = index + 1
            break
        if not inside_bounds(next_state[0:3], TRUE_SAFE_BOUNDS):
            failure_label = "true_safety_violation"
            notes = "true_safety_violation"
            final_index = index + 1
            break

    time_s = time_s[: final_index + 1]
    x_log = x_log[: final_index + 1]
    requested_log = requested_log[: final_index + 1]
    applied_log = applied_log[: final_index + 1]
    effective_log = effective_log[: final_index + 1]
    command_rad_log = command_rad_log[: final_index + 1]
    metrics = _metric_row(
        time_s,
        x_log,
        requested_log,
        applied_log,
        config,
        seed,
        scenario_name,
        failure_label,
        notes,
        saturation_reference=effective_log,
    )
    return RolloutResult(
        time_s=time_s,
        x=x_log,
        u_norm_requested=requested_log,
        u_norm_applied=applied_log,
        delta_cmd_rad=command_rad_log,
        success=False,
        failure_label=failure_label,
        metrics=metrics,
        notes=notes,
    )
