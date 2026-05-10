from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flight_dynamics import AircraftModel, evaluate_state, state_derivative
from implementation_wrappers import ImplementationCommandWrapper
from linearisation import INPUT_NAMES, STATE_INDEX, STATE_NAMES
from primitives import FlightPrimitive, PrimitiveContext


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float = 0.02
    rho_kg_m3: float = 1.225
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    min_altitude_m: float = 0.05
    speed_bounds_m_s: tuple[float, float] = (2.0, 10.5)
    max_abs_phi_rad: float = np.deg2rad(85.0)
    max_abs_theta_rad: float = np.deg2rad(65.0)
    max_abs_alpha_rad: float = np.deg2rad(28.0)


@dataclass(frozen=True)
class SimulationResult:
    scenario_name: str
    primitive_name: str
    status: str
    termination_reason: str
    times_s: np.ndarray
    states: np.ndarray
    ideal_commands: np.ndarray
    wrapped_commands: np.ndarray
    metrics: dict[str, float | str]
    log_rows: tuple[dict[str, float | str], ...]


def _rk4_step(
    x: np.ndarray,
    u_cmd: np.ndarray,
    dt_s: float,
    aircraft: AircraftModel,
    wind_model: object,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
    wind_mode: str,
) -> np.ndarray:
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


def _violation_reason(
    x: np.ndarray,
    loads: dict[str, object],
    config: SimulationConfig,
) -> str | None:
    if not np.all(np.isfinite(x)):
        return "non-finite state"
    altitude = float(x[STATE_INDEX["z_w"]])
    if altitude < config.min_altitude_m:
        return f"altitude below simulation floor: {altitude:.3f} m"
    speed = float(loads["speed_m_s"])
    low, high = config.speed_bounds_m_s
    if speed < low or speed > high:
        return f"speed out of simulation bounds: {speed:.3f} m/s"
    if abs(float(x[STATE_INDEX["phi"]])) > config.max_abs_phi_rad:
        return "bank angle exceeded simulation bound"
    if abs(float(x[STATE_INDEX["theta"]])) > config.max_abs_theta_rad:
        return "pitch angle exceeded simulation bound"
    if abs(float(loads["alpha_rad"])) > config.max_abs_alpha_rad:
        return "angle of attack exceeded simulation bound"
    return None


def _log_row(
    scenario_name: str,
    primitive: FlightPrimitive,
    t_s: float,
    x: np.ndarray,
    ideal_command: np.ndarray,
    wrapped_command: np.ndarray,
    loads: dict[str, object],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "scenario": scenario_name,
        "primitive": primitive.name,
        "phase": primitive.target_label(t_s),
        "t_s": float(t_s),
    }
    row.update({name: float(x[idx]) for idx, name in enumerate(STATE_NAMES)})
    for idx, name in enumerate(INPUT_NAMES):
        row[f"ideal_{name}_rad"] = float(ideal_command[idx])
        row[f"wrapped_{name}_rad"] = float(wrapped_command[idx])
    row.update(
        {
            "speed_m_s": float(loads["speed_m_s"]),
            "alpha_rad": float(loads["alpha_rad"]),
            "beta_rad": float(loads["beta_rad"]),
            "gamma_rad": float(loads["gamma_rad"]),
            "sink_rate_m_s": float(loads["sink_rate_m_s"]),
            "wind_cg_x_m_s": float(np.asarray(loads["wind_cg_w"])[0]),
            "wind_cg_y_m_s": float(np.asarray(loads["wind_cg_w"])[1]),
            "wind_cg_z_m_s": float(np.asarray(loads["wind_cg_w"])[2]),
        }
    )
    return row


def _metrics_from_logs(
    scenario_name: str,
    primitive_name: str,
    status: str,
    termination_reason: str,
    rows: list[dict[str, float | str]],
    states: np.ndarray,
    wrapped_commands: np.ndarray,
) -> dict[str, float | str]:
    speeds = np.asarray([row["speed_m_s"] for row in rows], dtype=float)
    alpha = np.asarray([row["alpha_rad"] for row in rows], dtype=float)
    beta = np.asarray([row["beta_rad"] for row in rows], dtype=float)
    sink = np.asarray([row["sink_rate_m_s"] for row in rows], dtype=float)
    altitude = states[:, STATE_INDEX["z_w"]]
    phi = states[:, STATE_INDEX["phi"]]
    theta = states[:, STATE_INDEX["theta"]]
    p = states[:, STATE_INDEX["p"]]
    q = states[:, STATE_INDEX["q"]]
    r = states[:, STATE_INDEX["r"]]
    return {
        "scenario": scenario_name,
        "primitive": primitive_name,
        "status": status,
        "termination_reason": termination_reason,
        "duration_completed_s": float(rows[-1]["t_s"]) if rows else 0.0,
        "start_altitude_m": float(altitude[0]),
        "final_altitude_m": float(altitude[-1]),
        "altitude_loss_m": float(altitude[0] - altitude[-1]),
        "min_altitude_m": float(np.min(altitude)),
        "final_speed_m_s": float(speeds[-1]),
        "min_speed_m_s": float(np.min(speeds)),
        "max_speed_m_s": float(np.max(speeds)),
        "mean_sink_rate_m_s": float(np.mean(sink)),
        "max_abs_phi_deg": float(np.rad2deg(np.max(np.abs(phi)))),
        "final_phi_deg": float(np.rad2deg(phi[-1])),
        "max_abs_theta_deg": float(np.rad2deg(np.max(np.abs(theta)))),
        "max_abs_p_deg_s": float(np.rad2deg(np.max(np.abs(p)))),
        "max_abs_q_deg_s": float(np.rad2deg(np.max(np.abs(q)))),
        "max_abs_r_deg_s": float(np.rad2deg(np.max(np.abs(r)))),
        "max_abs_alpha_deg": float(np.rad2deg(np.max(np.abs(alpha)))),
        "max_abs_beta_deg": float(np.rad2deg(np.max(np.abs(beta)))),
        "max_abs_wrapped_command_deg": float(
            np.rad2deg(np.max(np.abs(wrapped_commands)))
        ),
    }


def simulate_primitive(
    scenario_name: str,
    primitive: FlightPrimitive,
    x0: np.ndarray,
    context: PrimitiveContext,
    aircraft: AircraftModel,
    wind_model: object,
    wind_mode: str,
    wrapper: ImplementationCommandWrapper,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    config = config or SimulationConfig()
    x = np.asarray(x0, dtype=float).reshape(15).copy()
    wrapper.reset(dt_s=config.dt_s, initial_command_rad=context.u_trim)

    times: list[float] = []
    states: list[np.ndarray] = []
    ideal_commands: list[np.ndarray] = []
    wrapped_commands: list[np.ndarray] = []
    rows: list[dict[str, float | str]] = []
    status = "completed"
    termination_reason = ""
    steps = int(np.ceil(float(primitive.duration_s) / config.dt_s))

    for step in range(steps + 1):
        t_s = min(step * config.dt_s, float(primitive.duration_s))
        ideal = np.asarray(primitive.command(t_s, x, context), dtype=float).reshape(3)
        wrapped = wrapper.apply(ideal)
        loads = evaluate_state(
            x=x,
            u_cmd=wrapped,
            aircraft=aircraft,
            wind_model=wind_model,
            rho=config.rho_kg_m3,
            actuator_tau_s=config.actuator_tau_s,
            wind_mode=wind_mode,
        )
        times.append(t_s)
        states.append(x.copy())
        ideal_commands.append(ideal.copy())
        wrapped_commands.append(wrapped.copy())
        rows.append(
            _log_row(
                scenario_name=scenario_name,
                primitive=primitive,
                t_s=t_s,
                x=x,
                ideal_command=ideal,
                wrapped_command=wrapped,
                loads=loads,
            )
        )
        reason = _violation_reason(x=x, loads=loads, config=config)
        if reason is not None:
            status = "terminated"
            termination_reason = reason
            break
        if step == steps:
            break
        x = _rk4_step(
            x=x,
            u_cmd=wrapped,
            dt_s=config.dt_s,
            aircraft=aircraft,
            wind_model=wind_model,
            rho_kg_m3=config.rho_kg_m3,
            actuator_tau_s=config.actuator_tau_s,
            wind_mode=wind_mode,
        )

    state_arr = np.vstack(states)
    wrapped_arr = np.vstack(wrapped_commands)
    metrics = _metrics_from_logs(
        scenario_name=scenario_name,
        primitive_name=primitive.name,
        status=status,
        termination_reason=termination_reason,
        rows=rows,
        states=state_arr,
        wrapped_commands=wrapped_arr,
    )
    return SimulationResult(
        scenario_name=scenario_name,
        primitive_name=primitive.name,
        status=status,
        termination_reason=termination_reason,
        times_s=np.asarray(times, dtype=float),
        states=state_arr,
        ideal_commands=np.vstack(ideal_commands),
        wrapped_commands=wrapped_arr,
        metrics=metrics,
        log_rows=tuple(rows),
    )


def write_result_log(result: SimulationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not result.log_rows:
        return
    fieldnames = list(result.log_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.log_rows)
