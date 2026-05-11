from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena import ArenaConfig, safety_margins
from flight_dynamics import AircraftModel, evaluate_state, state_derivative
from latency import (
    CommandToSurfaceLayer,
    feedback_delay_s,
    half_response_s,
    latency_audit_fields,
    latency_range_label,
)
from linearisation import INPUT_NAMES, STATE_INDEX, STATE_NAMES
from metrics import rollout_metrics
from primitive import FlightPrimitive, PrimitiveContext


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Rollout dataclasses
# 2) Integration and safety helpers
# 3) Primitive rollout loop
# 4) Log writer
# =============================================================================

# =============================================================================
# 1) Rollout Dataclasses
# =============================================================================
# Rollout containers keep simulation settings, logs, and safety flags reproducible.
@dataclass(frozen=True)
class RolloutConfig:
    dt_s: float = 0.02
    rho_kg_m3: float = 1.225
    min_altitude_m: float = 0.25
    speed_bounds_m_s: tuple[float, float] = (2.0, 10.5)
    max_abs_phi_rad: float = np.deg2rad(85.0)
    max_abs_theta_rad: float = np.deg2rad(65.0)
    max_abs_alpha_rad: float = np.deg2rad(28.0)


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    termination_reason: str
    times_s: np.ndarray
    states: np.ndarray
    desired_commands: np.ndarray
    target_commands: np.ndarray
    metrics: dict[str, float | str | bool | int]
    log_rows: tuple[dict[str, float | str | bool], ...]


# =============================================================================
# 2) Integration and Safety Helpers
# =============================================================================
def rk4_step(
    x: np.ndarray,
    u_cmd: np.ndarray,
    dt_s: float,
    aircraft: AircraftModel,
    wind_model: object,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
    wind_mode: str,
) -> np.ndarray:
    # RK4 samples the plant with fixed command and wind over one integration step
    kwargs = {
        "u_cmd": u_cmd,
        "aircraft": aircraft,
        "wind_model": wind_model,
        "rho": rho_kg_m3,
        "actuator_tau_s": actuator_tau_s,
        "wind_mode": wind_mode,
    }
    k1 = state_derivative(x, **kwargs)
    k2 = state_derivative(x + 0.5 * dt_s * k1, **kwargs)
    k3 = state_derivative(x + 0.5 * dt_s * k2, **kwargs)
    k4 = state_derivative(x + dt_s * k3, **kwargs)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _wrap_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def primitive_tracking_error_rms(
    primitive: FlightPrimitive,
    context: PrimitiveContext,
    times_s: np.ndarray,
    states: np.ndarray,
) -> float:
    time_arr = np.asarray(times_s, dtype=float)
    state_arr = np.asarray(states, dtype=float)
    if time_arr.size == 0 or state_arr.size == 0:
        return float("nan")

    phi_ref_fn = getattr(primitive, "phi_ref", None)
    if callable(phi_ref_fn):
        phi_ref = np.asarray([float(phi_ref_fn(float(t_s))) for t_s in time_arr], dtype=float)
    else:
        phi_ref = np.zeros_like(time_arr, dtype=float)

    # Tracking is reduced to the attitude references used by the primitive layer.
    phi_error = _wrap_angle_rad(state_arr[:, STATE_INDEX["phi"]] - phi_ref)
    theta_error = _wrap_angle_rad(
        state_arr[:, STATE_INDEX["theta"]] - float(context.theta_trim_rad)
    )
    return float(np.sqrt(np.mean(0.5 * (phi_error**2 + theta_error**2))))


def violation_reason(
    x: np.ndarray,
    loads: dict[str, object],
    rollout_config: RolloutConfig,
    arena_config: ArenaConfig,
) -> str | None:
    if not np.all(np.isfinite(x)):
        return "non-finite state"
    # Safe-volume checks use the configured indoor arena margins
    margins = safety_margins(x, arena_config)
    if not bool(margins["inside_safe_volume"]):
        return "state outside safe volume"
    altitude = float(x[STATE_INDEX["z_w"]])
    if altitude < rollout_config.min_altitude_m:
        return f"altitude below floor: {altitude:.3f} m"
    speed = float(loads["speed_m_s"])
    low, high = rollout_config.speed_bounds_m_s
    if speed < low or speed > high:
        return f"speed out of bounds: {speed:.3f} m/s"
    if abs(float(x[STATE_INDEX["phi"]])) > rollout_config.max_abs_phi_rad:
        return "bank angle exceeded bound"
    if abs(float(x[STATE_INDEX["theta"]])) > rollout_config.max_abs_theta_rad:
        return "pitch angle exceeded bound"
    if abs(float(loads["alpha_rad"])) > rollout_config.max_abs_alpha_rad:
        return "angle of attack exceeded bound"
    return None


def feedback_state(
    t_s: float,
    current_x: np.ndarray,
    initial_x: np.ndarray,
    times_s: np.ndarray,
    states: np.ndarray,
    step: int,
    delay_s: float,
) -> np.ndarray:
    if delay_s <= 0.0:
        return np.asarray(current_x, dtype=float)
    target_t_s = float(t_s) - float(delay_s)
    if target_t_s <= 0.0 or step == 0:
        return np.asarray(initial_x, dtype=float)

    history_t = times_s[:step]
    history_x = states[:step]
    if target_t_s >= float(history_t[-1]):
        return history_x[-1].copy()

    right = int(np.searchsorted(history_t, target_t_s, side="right"))
    left = max(right - 1, 0)
    right = min(right, history_t.size - 1)
    t0 = float(history_t[left])
    t1 = float(history_t[right])
    if t1 <= t0:
        return history_x[left].copy()
    frac = (target_t_s - t0) / (t1 - t0)
    return history_x[left] + frac * (history_x[right] - history_x[left])


# =============================================================================
# 3) Primitive Rollout Loop
# =============================================================================
def simulate_primitive(
    scenario_id: str,
    seed: int,
    primitive: FlightPrimitive,
    x0: np.ndarray,
    context: PrimitiveContext,
    aircraft: AircraftModel,
    wind_model: object,
    wind_model_name: str,
    wind_mode: str,
    command_layer: CommandToSurfaceLayer,
    log_path: Path,
    repo_root: Path,
    rollout_config: RolloutConfig | None = None,
    arena_config: ArenaConfig | None = None,
    wind_param_label: str = "",
    selected_primitive_name: str | None = None,
    governor_rejection_reason: str = "",
    candidate_count: int = 0,
    rejected_count: int = 0,
) -> RolloutResult:
    rollout_config = rollout_config or RolloutConfig()
    arena_config = arena_config or ArenaConfig()
    x = np.asarray(x0, dtype=float).reshape(15).copy()
    x_initial = x.copy()
    command_layer.reset(context.u_trim)
    state_delay_s = feedback_delay_s(command_layer.config, command_layer.envelope)
    # Rollout length follows the primitive duration at fixed simulation dt
    steps = int(np.ceil(float(primitive.duration_s) / rollout_config.dt_s))

    times = np.empty(steps + 1, dtype=float)
    states = np.empty((steps + 1, 15), dtype=float)
    desired_commands = np.empty((steps + 1, 3), dtype=float)
    target_commands = np.empty((steps + 1, 3), dtype=float)
    rows: list[dict[str, float | str | bool]] = []
    success = True
    termination_reason = ""
    last_idx = 0

    for step in range(steps + 1):
        t_s = min(step * rollout_config.dt_s, float(primitive.duration_s))
        # Controller feedback uses the measured Vicon/filter delay, while safety and plant
        # propagation still use the true simulated state.
        x_feedback = feedback_state(
            t_s=t_s,
            current_x=x,
            initial_x=x_initial,
            times_s=times,
            states=states,
            step=step,
            delay_s=state_delay_s,
        )
        desired = np.asarray(
            primitive.command(t_s, x_feedback, context),
            dtype=float,
        ).reshape(3)
        # Desired commands pass through quantisation, latency, and surface limits
        target = command_layer.apply(desired)
        loads = evaluate_state(
            x=x,
            u_cmd=target,
            aircraft=aircraft,
            wind_model=wind_model,
            rho=rollout_config.rho_kg_m3,
            actuator_tau_s=command_layer.actuator_tau_vector_s,
            wind_mode=wind_mode,
        )
        times[step] = t_s
        states[step] = x
        desired_commands[step] = desired
        target_commands[step] = target
        row = {
            "scenario_id": scenario_id,
            "primitive": primitive.name,
            "phase": primitive.target_label(t_s),
            "t_s": float(t_s),
        }
        row.update({name: float(x[idx]) for idx, name in enumerate(STATE_NAMES)})
        for idx, name in enumerate(INPUT_NAMES):
            row[f"desired_{name}_rad"] = float(desired[idx])
            row[f"target_{name}_rad"] = float(target[idx])
        row.update(command_layer.log_fields())
        row.update(
            {
                "speed_m_s": float(loads["speed_m_s"]),
                "alpha_rad": float(loads["alpha_rad"]),
                "beta_rad": float(loads["beta_rad"]),
                "gamma_rad": float(loads["gamma_rad"]),
                "sink_rate_m_s": float(loads["sink_rate_m_s"]),
                "wind_model": wind_model_name,
                "wind_mode": wind_mode,
                "wind_param_label": wind_param_label,
            }
        )
        row.update(safety_margins(x, arena_config))
        rows.append(row)
        last_idx = step

        reason = violation_reason(x, loads, rollout_config, arena_config)
        if reason is not None:
            success = False
            termination_reason = reason
            break
        if step == steps:
            break
        x = rk4_step(
            x=x,
            u_cmd=target,
            dt_s=rollout_config.dt_s,
            aircraft=aircraft,
            wind_model=wind_model,
            rho_kg_m3=rollout_config.rho_kg_m3,
            actuator_tau_s=command_layer.actuator_tau_vector_s,
            wind_mode=wind_mode,
        )

    times = times[: last_idx + 1]
    states = states[: last_idx + 1]
    desired_commands = desired_commands[: last_idx + 1]
    target_commands = target_commands[: last_idx + 1]
    lower = np.deg2rad([-26.0, -30.0, -35.0])
    upper = np.deg2rad([22.0, 22.0, 28.0])
    # Saturation is evaluated on aggregate surface targets in radians
    saturated = np.isclose(target_commands, lower, atol=1e-12) | np.isclose(
        target_commands,
        upper,
        atol=1e-12,
    )
    sat_fraction = float(np.mean(saturated))
    sat_time_s = float(np.mean(np.any(saturated, axis=1)) * times[-1]) if times.size else 0.0
    tracking_error_rms = primitive_tracking_error_rms(
        primitive=primitive,
        context=context,
        times_s=times,
        states=states,
    )
    metrics = rollout_metrics(
        scenario_id=scenario_id,
        seed=seed,
        wind_model=wind_model_name,
        wind_mode=wind_mode,
        wind_param_label=wind_param_label,
        latency_mode=command_layer.config.mode,
        latency_s=half_response_s(command_layer.config, command_layer.envelope),
        latency_range_s=latency_range_label(command_layer.config, command_layer.envelope),
        **latency_audit_fields(command_layer.config, command_layer.envelope),
        primitive_selected=primitive.name,
        selected_primitive=selected_primitive_name or primitive.name,
        success=success,
        termination_reason=termination_reason,
        states=states,
        log_path=log_path,
        repo_root=repo_root,
        arena_config=arena_config,
        saturation_fraction=sat_fraction,
        saturation_time_s=sat_time_s,
        tracking_error_rms=tracking_error_rms,
        duration_s=float(times[-1]) if times.size else 0.0,
        governor_rejection_reason=governor_rejection_reason,
        candidate_count=candidate_count,
        rejected_count=rejected_count,
    )
    return RolloutResult(
        success=success,
        termination_reason=termination_reason,
        times_s=times,
        states=states,
        desired_commands=desired_commands,
        target_commands=target_commands,
        metrics=metrics,
        log_rows=tuple(rows),
    )


# =============================================================================
# 4) Log Writer
# =============================================================================
# Logs store state and command channels in canonical order for post-run audit scripts.
def write_log(result: RolloutResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not result.log_rows:
        return
    fieldnames = list(result.log_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.log_rows)
