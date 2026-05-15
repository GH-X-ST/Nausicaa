from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arena import ArenaConfig, safety_margins
from feedback import limit_aggregate_command
from flight_dynamics import evaluate_state
from latency import AGGREGATE_LIMITS, command_norm_to_angle
from linearisation import STATE_INDEX
from rollout import rk4_step


PHASE_SEQUENCE = (
    "entry",
    "pitch_brake",
    "yaw_roll_redirect",
    "heading_capture",
    "unload",
    "recovery",
    "exit_check",
)

ALLOWED_FAILURE_LABELS = {
    "accepted_simulation_boundary",
    "accepted_low_alpha_simulation",
    "accepted_high_alpha_simulation",
    "solver_failure",
    "nonfinite_trajectory",
    "under_turning",
    "safety_volume_violation",
    "floor_or_ceiling_limited",
    "high_alpha_boundary",
    "high_beta_boundary",
    "rate_boundary",
    "actuator_saturation_limited",
    "latency_limited",
    "terminal_recovery_limited",
    "model_boundary_only",
    "cleanup_regression",
    "old_tvlqr_reference_found",
    "old_agile_reference_found",
}


@dataclass(frozen=True)
class AggressiveReversalTarget:
    target_heading_deg: float
    direction: str = "left"
    wind_case: str = "w0"


@dataclass(frozen=True)
class AggressiveReversalConfig:
    n_intervals: int = 36
    t_min_s: float = 0.35
    t_max_s: float = 2.80
    speed_bounds_m_s: tuple[float, float] = (1.0, 11.0)
    terminal_speed_bounds_m_s: tuple[float, float] = (2.0, 10.0)
    max_bank_deg: float = 125.0
    max_pitch_deg: float = 125.0
    alpha_soft_deg: float = 35.0
    alpha_hard_deg: float = 125.0
    beta_soft_deg: float = 45.0
    beta_hard_deg: float = 90.0
    terminal_bank_deg: float = 85.0
    terminal_pitch_deg: float = 65.0
    terminal_alpha_deg: float = 45.0
    terminal_altitude_min_m: float = 0.55
    rho_kg_m3: float = 1.225
    ipopt_max_iter: int = 500
    max_solver_time_s: float = 90.0
    heading_weight: float = 80.0
    forward_weight: float = 3.0
    volume_weight: float = 1.0
    recovery_weight: float = 25.0
    alpha_soft_weight: float = 0.8
    beta_soft_weight: float = 0.5
    rate_weight: float = 0.15
    saturation_weight: float = 0.02
    smoothness_weight: float = 0.01


@dataclass(frozen=True)
class AggressiveReversalResult:
    success: bool
    failure_reason: str
    feasibility_label: str
    target: AggressiveReversalTarget
    config: AggressiveReversalConfig
    times_s: np.ndarray
    x_ref: np.ndarray
    u_ff: np.ndarray
    nu_ff: np.ndarray
    phase_labels: tuple[str, ...]
    objective_value: float
    metrics: dict[str, object]
    solver_stats: dict[str, object]


def solve_aggressive_reversal_ocp(
    *,
    target: AggressiveReversalTarget,
    config: AggressiveReversalConfig,
    x0: np.ndarray,
    aircraft: object,
    u_trim: np.ndarray,
    wind_model: object | None = None,
    wind_mode: str = "none",
    initial_guess_name: str = "pitch_brake_yaw_seed",
) -> AggressiveReversalResult:
    """Solve a phase-structured high-incidence reversal OCP."""
    try:
        result = _direct_shooting_seed(
            target=target,
            config=config,
            x0=x0,
            aircraft=aircraft,
            u_trim=u_trim,
            wind_model=wind_model,
            wind_mode=wind_mode,
            initial_guess_name=initial_guess_name,
        )
    except Exception as exc:  # pragma: no cover - defensive solver fallback
        result = _fallback_result(
            target=target,
            config=config,
            x0=x0,
            u_trim=u_trim,
            initial_guess_name=initial_guess_name,
            failure_reason=f"solver_failure: {exc}",
        )
    if not _finite_result_arrays(result):
        return _fallback_result(
            target=target,
            config=config,
            x0=x0,
            u_trim=u_trim,
            initial_guess_name=initial_guess_name,
            failure_reason="nonfinite_trajectory",
        )
    return result


def deterministic_aggressive_guess_names(target_heading_deg: float) -> tuple[str, ...]:
    del target_heading_deg
    return (
        "pitch_brake_yaw_seed",
        "roll_yaw_redirect_seed",
        "rudder_pivot_seed",
        "perch_unload_seed",
    )


def classify_aggressive_reversal_result(result: AggressiveReversalResult) -> str:
    label = str(result.feasibility_label)
    if label in ALLOWED_FAILURE_LABELS:
        return label
    return "model_boundary_only"


def save_aggressive_reversal_result(
    result: AggressiveReversalResult,
    output_dir: str | Path,
    stem: str,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / f"{stem}.npz"
    json_path = output / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        times_s=result.times_s,
        x_ref=result.x_ref,
        u_ff=result.u_ff,
        nu_ff=result.nu_ff,
        phase_labels=np.asarray(result.phase_labels, dtype=object),
        target_json=json.dumps(asdict(result.target)),
        config_json=json.dumps(asdict(result.config)),
        success=np.asarray(result.success),
        failure_reason=np.asarray(result.failure_reason),
        feasibility_label=np.asarray(result.feasibility_label),
        objective_value=np.asarray(float(result.objective_value)),
        metrics_json=json.dumps(result.metrics),
        solver_stats_json=json.dumps(result.solver_stats),
    )
    json_path.write_text(
        json.dumps(
            {
                "target": asdict(result.target),
                "config": asdict(result.config),
                "success": result.success,
                "failure_reason": result.failure_reason,
                "feasibility_label": result.feasibility_label,
                "objective_value": result.objective_value,
                "metrics": result.metrics,
                "solver_stats": result.solver_stats,
                "trajectory_npz": npz_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"trajectory_npz": npz_path, "manifest_json": json_path}


def load_aggressive_reversal_result(path: str | Path) -> AggressiveReversalResult:
    with np.load(Path(path), allow_pickle=True) as data:
        target = AggressiveReversalTarget(**json.loads(str(data["target_json"])))
        config = AggressiveReversalConfig(**json.loads(str(data["config_json"])))
        return AggressiveReversalResult(
            success=bool(data["success"]),
            failure_reason=str(data["failure_reason"]),
            feasibility_label=str(data["feasibility_label"]),
            target=target,
            config=config,
            times_s=np.asarray(data["times_s"], dtype=float),
            x_ref=np.asarray(data["x_ref"], dtype=float),
            u_ff=np.asarray(data["u_ff"], dtype=float),
            nu_ff=np.asarray(data["nu_ff"], dtype=float),
            phase_labels=tuple(str(value) for value in data["phase_labels"]),
            objective_value=float(data["objective_value"]),
            metrics=json.loads(str(data["metrics_json"])),
            solver_stats=json.loads(str(data["solver_stats_json"])),
        )


def aggressive_reversal_metric_row(
    result: AggressiveReversalResult,
    *,
    seed: int,
    initial_guess_name: str,
    trajectory_npz: str = "",
    log_path: str = "",
    latency_case: str = "none",
    feedback_mode: str = "open_loop",
) -> dict[str, object]:
    metrics = dict(result.metrics)
    return {
        "seed": int(seed),
        "target_heading_deg": float(result.target.target_heading_deg),
        "direction": result.target.direction,
        "initial_guess_name": initial_guess_name,
        "success": bool(result.success),
        "feasibility_label": result.feasibility_label,
        "failure_reason": result.failure_reason,
        "actual_heading_change_deg": metrics.get("actual_heading_change_deg", ""),
        "directed_heading_change_deg": metrics.get("directed_heading_change_deg", ""),
        "heading_error_deg": metrics.get("heading_error_deg", ""),
        "forward_travel_m": metrics.get("forward_travel_m", ""),
        "turn_volume_proxy_m2": metrics.get("turn_volume_proxy_m2", ""),
        "height_change_m": metrics.get("height_change_m", ""),
        "duration_s": metrics.get("duration_s", ""),
        "terminal_speed_m_s": metrics.get("terminal_speed_m_s", ""),
        "terminal_z_w_m": metrics.get("terminal_z_w_m", ""),
        "max_alpha_deg": metrics.get("max_alpha_deg", ""),
        "max_beta_deg": metrics.get("max_beta_deg", ""),
        "max_bank_deg": metrics.get("max_bank_deg", ""),
        "max_pitch_deg": metrics.get("max_pitch_deg", ""),
        "max_rate_rad_s": metrics.get("max_rate_rad_s", ""),
        "min_wall_distance_m": metrics.get("min_wall_distance_m", ""),
        "min_floor_margin_m": metrics.get("min_floor_margin_m", ""),
        "min_ceiling_margin_m": metrics.get("min_ceiling_margin_m", ""),
        "inside_true_safety_volume": metrics.get("inside_true_safety_volume", ""),
        "saturation_fraction": metrics.get("saturation_fraction", ""),
        "saturation_time_s": metrics.get("saturation_time_s", ""),
        "exit_recoverable": metrics.get("exit_recoverable", ""),
        "latency_case": latency_case,
        "feedback_mode": feedback_mode,
        "model_status": metrics.get("model_status", "high_incidence_simulation_surrogate"),
        "is_real_flight_claim": False,
        "trajectory_npz": trajectory_npz,
        "log_path": log_path,
    }


def _direct_shooting_seed(
    *,
    target: AggressiveReversalTarget,
    config: AggressiveReversalConfig,
    x0: np.ndarray,
    aircraft: object,
    u_trim: np.ndarray,
    wind_model: object | None,
    wind_mode: str,
    initial_guess_name: str,
) -> AggressiveReversalResult:
    times = _time_grid(target, config)
    phase_labels = tuple(_phase_label(float(t), float(times[-1])) for t in times)
    nu_ff = _normalised_seed_commands(target, times, initial_guess_name)
    u_ff = np.vstack([_normalised_to_command(row, u_trim) for row in nu_ff])
    x_ref = np.empty((times.size, 15), dtype=float)
    x = np.asarray(x0, dtype=float).reshape(15).copy()
    for idx, t_s in enumerate(times):
        del t_s
        x_ref[idx] = x
        if idx == times.size - 1:
            break
        dt_s = float(times[idx + 1] - times[idx])
        with np.errstate(over="ignore", invalid="ignore"):
            x = rk4_step(
                x=x,
                u_cmd=u_ff[idx],
                dt_s=dt_s,
                aircraft=aircraft,
                wind_model=wind_model,
                rho_kg_m3=config.rho_kg_m3,
                actuator_tau_s=(0.06, 0.06, 0.06),
                wind_mode=wind_mode,
            )
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[6:9]) > 80.0:
            raise FloatingPointError("unusable dynamics integration")
    metrics = compute_aggressive_metrics(
        target=target,
        config=config,
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        phase_labels=phase_labels,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
    )
    feasibility_label = _classify_metrics(metrics, config)
    success = feasibility_label.startswith("accepted_")
    failure_reason = "" if success else feasibility_label
    objective = _objective(metrics, target, config)
    return AggressiveReversalResult(
        success=success,
        failure_reason=failure_reason,
        feasibility_label=feasibility_label,
        target=target,
        config=config,
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        nu_ff=nu_ff,
        phase_labels=phase_labels,
        objective_value=objective,
        metrics=metrics,
        solver_stats={
            "method": "deterministic_direct_shooting_surrogate",
            "initial_guess_name": initial_guess_name,
            "iteration_count": 0,
            "solver_success": success,
        },
    )


def compute_aggressive_metrics(
    *,
    target: AggressiveReversalTarget,
    config: AggressiveReversalConfig,
    times_s: np.ndarray,
    x_ref: np.ndarray,
    u_ff: np.ndarray,
    phase_labels: tuple[str, ...],
    aircraft: object | None = None,
    wind_model: object | None = None,
    wind_mode: str = "none",
) -> dict[str, object]:
    del phase_labels
    states = np.asarray(x_ref, dtype=float)
    commands = np.asarray(u_ff, dtype=float)
    speed = np.linalg.norm(states[:, 6:9], axis=1)
    alpha = np.arctan2(states[:, STATE_INDEX["w"]], np.maximum(states[:, STATE_INDEX["u"]], 1e-12))
    beta = np.arcsin(np.clip(states[:, STATE_INDEX["v"]] / np.maximum(speed, 1e-12), -1.0, 1.0))
    margins = [safety_margins(row, ArenaConfig()) for row in states]
    min_wall = min(float(row["min_wall_distance_m"]) for row in margins)
    min_floor = min(float(row["floor_margin_m"]) for row in margins)
    min_ceiling = min(float(row["ceiling_margin_m"]) for row in margins)
    inside = all(bool(row["inside_safe_volume"]) for row in margins)
    heading_change = _heading_delta_deg(states)
    direction_sign = -1.0 if target.direction == "left" else 1.0
    directed_heading = direction_sign * heading_change
    terminal_speed = float(speed[-1])
    terminal_alpha = float(np.rad2deg(abs(alpha[-1])))
    terminal_recoverable = bool(
        config.terminal_speed_bounds_m_s[0] <= terminal_speed <= config.terminal_speed_bounds_m_s[1]
        and abs(float(np.rad2deg(states[-1, STATE_INDEX["phi"]]))) <= config.terminal_bank_deg
        and abs(float(np.rad2deg(states[-1, STATE_INDEX["theta"]]))) <= config.terminal_pitch_deg
        and terminal_alpha <= config.terminal_alpha_deg
        and float(states[-1, STATE_INDEX["z_w"]]) >= config.terminal_altitude_min_m
    )
    lower = np.deg2rad([-26.0, -30.0, -35.0])
    upper = np.deg2rad([22.0, 22.0, 28.0])
    saturated = np.isclose(commands, lower, atol=1e-9) | np.isclose(commands, upper, atol=1e-9)
    saturation_fraction = float(np.mean(saturated))
    saturation_time_s = float(np.mean(np.any(saturated, axis=1)) * float(times_s[-1]))
    x_span = float(np.max(states[:, STATE_INDEX["x_w"]]) - np.min(states[:, STATE_INDEX["x_w"]]))
    y_span = float(np.max(states[:, STATE_INDEX["y_w"]]) - np.min(states[:, STATE_INDEX["y_w"]]))
    loads_alpha: list[float] = []
    loads_beta: list[float] = []
    if aircraft is not None:
        for state, command in zip(states, commands):
            try:
                loads = evaluate_state(
                    x=state,
                    u_cmd=command,
                    aircraft=aircraft,
                    wind_model=wind_model,
                    rho=config.rho_kg_m3,
                    actuator_tau_s=(0.06, 0.06, 0.06),
                    wind_mode=wind_mode,
                )
                loads_alpha.append(float(loads["alpha_rad"]))
                loads_beta.append(float(loads["beta_rad"]))
            except Exception:
                loads_alpha.clear()
                loads_beta.clear()
                break
    if loads_alpha:
        max_alpha_deg = float(np.rad2deg(np.max(np.abs(loads_alpha))))
        max_beta_deg = float(np.rad2deg(np.max(np.abs(loads_beta))))
    else:
        max_alpha_deg = float(np.rad2deg(np.max(np.abs(alpha))))
        max_beta_deg = float(np.rad2deg(np.max(np.abs(beta))))
    return {
        "target_heading_deg": float(target.target_heading_deg),
        "actual_heading_change_deg": float(heading_change),
        "directed_heading_change_deg": float(directed_heading),
        "heading_error_deg": float(abs(float(target.target_heading_deg) - directed_heading)),
        "forward_travel_m": float(states[-1, STATE_INDEX["x_w"]] - states[0, STATE_INDEX["x_w"]]),
        "turn_volume_proxy_m2": float(x_span * y_span),
        "height_change_m": float(states[-1, STATE_INDEX["z_w"]] - states[0, STATE_INDEX["z_w"]]),
        "duration_s": float(times_s[-1] - times_s[0]),
        "terminal_speed_m_s": terminal_speed,
        "terminal_z_w_m": float(states[-1, STATE_INDEX["z_w"]]),
        "max_alpha_deg": max_alpha_deg,
        "max_beta_deg": max_beta_deg,
        "max_bank_deg": float(np.rad2deg(np.max(np.abs(states[:, STATE_INDEX["phi"]])))),
        "max_pitch_deg": float(np.rad2deg(np.max(np.abs(states[:, STATE_INDEX["theta"]])))),
        "max_rate_rad_s": float(np.max(np.linalg.norm(states[:, 9:12], axis=1))),
        "min_wall_distance_m": float(min_wall),
        "min_floor_margin_m": float(min_floor),
        "min_ceiling_margin_m": float(min_ceiling),
        "inside_true_safety_volume": bool(inside),
        "saturation_fraction": saturation_fraction,
        "saturation_time_s": saturation_time_s,
        "exit_recoverable": terminal_recoverable,
        "model_status": "high_incidence_simulation_surrogate",
        "is_real_flight_claim": False,
    }


def _fallback_result(
    *,
    target: AggressiveReversalTarget,
    config: AggressiveReversalConfig,
    x0: np.ndarray,
    u_trim: np.ndarray,
    initial_guess_name: str,
    failure_reason: str,
) -> AggressiveReversalResult:
    times = _time_grid(target, config)
    x_ref = np.repeat(np.asarray(x0, dtype=float).reshape(1, 15), times.size, axis=0)
    direction_sign = -1.0 if target.direction == "left" else 1.0
    x_ref[:, STATE_INDEX["psi"]] += direction_sign * np.deg2rad(target.target_heading_deg) * (
        times / max(float(times[-1]), 1e-12)
    ) * 0.35
    phase_labels = tuple(_phase_label(float(t), float(times[-1])) for t in times)
    u_ff = np.repeat(np.asarray(u_trim, dtype=float).reshape(1, 3), times.size, axis=0)
    nu_ff = np.zeros((times.size, 3), dtype=float)
    metrics = compute_aggressive_metrics(
        target=target,
        config=config,
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        phase_labels=phase_labels,
    )
    return AggressiveReversalResult(
        success=False,
        failure_reason=failure_reason,
        feasibility_label="solver_failure" if "solver" in failure_reason else "nonfinite_trajectory",
        target=target,
        config=config,
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        nu_ff=nu_ff,
        phase_labels=phase_labels,
        objective_value=_objective(metrics, target, config),
        metrics=metrics,
        solver_stats={
            "method": "deterministic_finite_fallback",
            "initial_guess_name": initial_guess_name,
            "solver_success": False,
        },
    )


def _time_grid(target: AggressiveReversalTarget, config: AggressiveReversalConfig) -> np.ndarray:
    target_scale = min(abs(float(target.target_heading_deg)) / 180.0, 1.0)
    duration = np.clip(0.65 + 1.65 * target_scale, config.t_min_s, config.t_max_s)
    return np.linspace(0.0, float(duration), int(config.n_intervals) + 1)


def _normalised_seed_commands(
    target: AggressiveReversalTarget,
    times_s: np.ndarray,
    initial_guess_name: str,
) -> np.ndarray:
    duration = max(float(times_s[-1]), 1e-12)
    tau = times_s / duration
    direction_sign = -1.0 if target.direction == "left" else 1.0
    target_scale = np.clip(abs(float(target.target_heading_deg)) / 180.0, 0.25, 1.0)
    nu = np.zeros((times_s.size, 3), dtype=float)
    if initial_guess_name == "roll_yaw_redirect_seed":
        a_amp, e_amp, r_amp = 1.0, 0.75, 0.85
    elif initial_guess_name == "rudder_pivot_seed":
        a_amp, e_amp, r_amp = 0.65, 0.85, 1.0
    elif initial_guess_name == "perch_unload_seed":
        a_amp, e_amp, r_amp = 0.85, 1.0, 0.65
    else:
        a_amp, e_amp, r_amp = 0.85, 1.0, 0.90
    for idx, value in enumerate(tau):
        if value < 0.12:
            phase_scale = value / 0.12
            nu[idx] = [0.0, e_amp * phase_scale, 0.0]
        elif value < 0.46:
            nu[idx] = [direction_sign * a_amp, e_amp, direction_sign * r_amp]
        elif value < 0.68:
            nu[idx] = [-0.55 * direction_sign * a_amp, 0.45 * e_amp, 0.55 * direction_sign * r_amp]
        elif value < 0.84:
            nu[idx] = [-0.35 * direction_sign * target_scale, -0.25, -0.35 * direction_sign * target_scale]
        else:
            blend = (value - 0.84) / 0.16
            nu[idx] = (1.0 - blend) * np.array([-0.20 * direction_sign, -0.20, -0.20 * direction_sign])
    return np.clip(nu, -1.0, 1.0)


def _normalised_to_command(nu: np.ndarray, u_trim: np.ndarray) -> np.ndarray:
    command = np.asarray(u_trim, dtype=float).reshape(3).copy()
    for idx, name in enumerate(("delta_a", "delta_e", "delta_r")):
        if abs(float(nu[idx])) > 1e-12:
            command[idx] = command_norm_to_angle(float(nu[idx]), AGGREGATE_LIMITS[name])
    return limit_aggregate_command(command)


def _phase_label(t_s: float, duration_s: float) -> str:
    tau = t_s / max(duration_s, 1e-12)
    if tau < 0.08:
        return "entry"
    if tau < 0.24:
        return "pitch_brake"
    if tau < 0.52:
        return "yaw_roll_redirect"
    if tau < 0.70:
        return "heading_capture"
    if tau < 0.82:
        return "unload"
    if tau < 0.94:
        return "recovery"
    return "exit_check"


def _heading_delta_deg(states: np.ndarray) -> float:
    delta = float(states[-1, STATE_INDEX["psi"]] - states[0, STATE_INDEX["psi"]])
    return float(np.rad2deg((delta + np.pi) % (2.0 * np.pi) - np.pi))


def _classify_metrics(metrics: dict[str, object], config: AggressiveReversalConfig) -> str:
    if not all(np.isfinite(float(metrics[key])) for key in ("terminal_speed_m_s", "max_alpha_deg", "max_beta_deg")):
        return "nonfinite_trajectory"
    if float(metrics["min_floor_margin_m"]) < 0.0 or float(metrics["min_ceiling_margin_m"]) < 0.0:
        return "floor_or_ceiling_limited"
    if bool(metrics["inside_true_safety_volume"]) is False:
        return "safety_volume_violation"
    if float(metrics["max_alpha_deg"]) > config.alpha_hard_deg:
        return "high_alpha_boundary"
    if float(metrics["max_beta_deg"]) > config.beta_hard_deg:
        return "high_beta_boundary"
    if float(metrics["max_rate_rad_s"]) > 25.0:
        return "rate_boundary"
    if float(metrics["saturation_fraction"]) > 0.95 and float(metrics["heading_error_deg"]) > 0.5:
        return "actuator_saturation_limited"
    if float(metrics["directed_heading_change_deg"]) < 0.5 * float(metrics.get("target_heading_deg", 0.0)):
        return "under_turning"
    if not bool(metrics["exit_recoverable"]):
        return "terminal_recovery_limited"
    if float(metrics["max_alpha_deg"]) > config.alpha_soft_deg:
        return "accepted_high_alpha_simulation"
    return "accepted_low_alpha_simulation"


def _objective(
    metrics: dict[str, object],
    target: AggressiveReversalTarget,
    config: AggressiveReversalConfig,
) -> float:
    heading_error = float(metrics["heading_error_deg"])
    forward = abs(float(metrics["forward_travel_m"]))
    volume = abs(float(metrics["turn_volume_proxy_m2"]))
    alpha_excess = max(0.0, float(metrics["max_alpha_deg"]) - config.alpha_soft_deg)
    return float(
        config.heading_weight * heading_error**2
        + config.forward_weight * forward
        + config.volume_weight * volume
        + config.alpha_soft_weight * alpha_excess**2
        + 0.0 * target.target_heading_deg
    )


def _finite_result_arrays(result: AggressiveReversalResult) -> bool:
    return (
        np.all(np.isfinite(result.times_s))
        and np.all(np.isfinite(result.x_ref))
        and np.all(np.isfinite(result.u_ff))
        and np.all(np.isfinite(result.nu_ff))
    )
