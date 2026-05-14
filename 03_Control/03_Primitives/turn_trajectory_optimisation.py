from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import casadi as ca
import numpy as np

from arena import ArenaConfig, safety_margins, safe_bounds
from flight_dynamics import AircraftModel, build_symbolic_dynamics
from latency import AGGREGATE_LIMITS, CommandToSurfaceLayer, command_norm_to_angle
from linearisation import STATE_INDEX
from primitive import PrimitiveContext
from rollout import rk4_step
from trajectory_primitive import TrajectoryEntryLimits, TrajectoryPrimitive
from tvlqr import TVLQRConfig, linearise_trajectory_finite_difference, solve_discrete_tvlqr


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Public containers and labels
# 2) Command mapping and heading helpers
# 3) Initial guesses and numeric diagnostics
# 4) Direct multiple-shooting solve
# 5) Primitive conversion and persistence
# =============================================================================

# =============================================================================
# 1) Public Containers and Labels
# =============================================================================
ACCEPTED_LABELS = (
    "accepted_low_alpha",
    "accepted_moderate_alpha",
    "accepted_high_alpha_sim_only",
)


@dataclass(frozen=True)
class TurnTarget:
    target_heading_deg: float
    direction: str
    wind_case: str = "w0"
    allow_high_alpha: bool = False
    allow_safety_slack: bool = False


@dataclass(frozen=True)
class TurnOptimisationConfig:
    n_intervals: int = 18
    t_min_s: float = 0.35
    t_max_s: float = 1.40
    speed_bounds_m_s: tuple[float, float] = (2.5, 9.5)
    terminal_speed_bounds_m_s: tuple[float, float] = (2.5, 9.5)
    max_bank_deg: float = 85.0
    max_pitch_deg: float = 65.0
    terminal_bank_deg: float = 65.0
    terminal_pitch_deg: float = 40.0
    max_alpha_deg: float = 28.0
    max_beta_deg: float = 35.0
    min_wall_margin_m: float = 0.0
    terminal_wall_margin_m: float = 0.10
    terminal_altitude_min_m: float = 0.75
    rho_kg_m3: float = 1.225
    actuator_tau_s: tuple[float, float, float] = (0.05049432643111373,) * 3
    solver_name: str = "casadi_ipopt"
    ipopt_max_iter: int = 220
    max_solver_time_s: float = 30.0
    dynamics_defect_tolerance: float = 5e-4
    heading_weight: float = 20.0
    recovery_weight: float = 8.0
    smoothness_weight: float = 0.08
    saturation_weight: float = 0.04
    energy_loss_weight: float = 0.02
    slack_weight: float = 1.0e5


@dataclass(frozen=True)
class OptimisedTurnResult:
    success: bool
    failure_reason: str
    feasibility_label: str
    target: TurnTarget
    config: TurnOptimisationConfig
    times_s: np.ndarray
    x_ref: np.ndarray
    u_ff: np.ndarray
    objective_value: float
    metrics: dict[str, object]
    solver_stats: dict[str, object]
    nu_ff: np.ndarray | None = None


# =============================================================================
# 2) Command Mapping and Heading Helpers
# =============================================================================
def normalised_command_to_radians(command_norm: np.ndarray) -> np.ndarray:
    """Map normalised commands to aggregate surface radians using latency.py limits."""
    norm = np.asarray(command_norm, dtype=float).reshape(-1, 3)
    out = np.empty_like(norm)
    for col, name in enumerate(("delta_a", "delta_e", "delta_r")):
        out[:, col] = [
            command_norm_to_angle(float(value), AGGREGATE_LIMITS[name])
            for value in norm[:, col]
        ]
    return out.reshape(np.asarray(command_norm).shape)


def normalised_command_to_radians_ca(command_norm: ca.MX | ca.SX) -> ca.MX | ca.SX:
    """CasADi equivalent of normalised_command_to_radians."""
    values = []
    for idx, name in enumerate(("delta_a", "delta_e", "delta_r")):
        limit = AGGREGATE_LIMITS[name]
        value = command_norm[idx]
        deg = ca.if_else(
            value >= 0.0,
            value * float(limit.positive_deg),
            (-value) * float(limit.negative_deg),
        )
        values.append((np.pi / 180.0) * deg)
    return ca.vertcat(*values)


def heading_direction_sign(direction: str) -> float:
    text = str(direction).lower()
    if text == "left":
        return -1.0
    if text == "right":
        return 1.0
    raise ValueError("direction must be 'left' or 'right'.")


def accepted_heading_threshold_deg(target_heading_deg: float) -> float:
    return 0.8 * abs(float(target_heading_deg))


def _phase_metadata(duration_s: float) -> dict[str, dict[str, float]]:
    edges = (
        ("entry", 0.00, 0.10),
        ("brake_or_pitch", 0.10, 0.25),
        ("roll_yaw_redirect", 0.25, 0.55),
        ("turn_hold_or_heading_capture", 0.55, 0.75),
        ("recover", 0.75, 0.95),
        ("exit_check", 0.95, 1.00),
    )
    return {
        name: {
            "start_s": float(duration_s * start),
            "end_s": float(duration_s * end),
            "duration_s": float(duration_s * (end - start)),
        }
        for name, start, end in edges
    }


# =============================================================================
# 3) Initial Guesses and Numeric Diagnostics
# =============================================================================
def deterministic_initial_guess_names(target_heading_deg: float) -> tuple[str, ...]:
    if abs(float(target_heading_deg)) <= 1e-12:
        return ("trim_glide",)
    return ("template_seed", "bank_yaw_seed")


def build_initial_guess(
    *,
    guess_name: str,
    target: TurnTarget,
    config: TurnOptimisationConfig,
    x0: np.ndarray,
    u_trim: np.ndarray,
    aircraft: AircraftModel,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x0, dtype=float).reshape(15).copy()
    trim_command = np.asarray(u_trim, dtype=float).reshape(3)
    interval_count = int(config.n_intervals)
    target_abs = abs(float(target.target_heading_deg))
    duration = np.clip(
        0.45 + 0.012 * target_abs,
        float(config.t_min_s),
        float(config.t_max_s),
    )
    times = np.linspace(0.0, float(duration), interval_count + 1)
    nu = np.zeros((interval_count, 3), dtype=float)
    direction = heading_direction_sign(target.direction)

    if guess_name == "template_seed" and target_abs > 0.0:
        split = max(1, interval_count // 4)
        end = max(split + 1, int(0.78 * interval_count))
        nu[:split, 1] = -0.8
        nu[split:end, 0] = -direction
        nu[split:end, 2] = -direction
        nu[end:, 0] = 0.45 * direction
        nu[end:, 2] = 0.25 * direction
    elif guess_name == "bank_yaw_seed" and target_abs > 0.0:
        mid = max(1, int(0.70 * interval_count))
        nu[:mid, 0] = -direction
        nu[:mid, 2] = -0.75 * direction
        nu[mid:, 0] = 0.55 * direction
        nu[mid:, 2] = 0.35 * direction

    x_guess = np.empty((interval_count + 1, 15), dtype=float)
    x_guess[0] = x
    for idx in range(interval_count):
        command = (
            trim_command
            if guess_name == "trim_glide"
            else normalised_command_to_radians(nu[idx])
        )
        dt_s = float(times[idx + 1] - times[idx])
        x = rk4_step(
            x=x,
            u_cmd=command,
            dt_s=dt_s,
            aircraft=aircraft,
            wind_model=None,
            rho_kg_m3=float(config.rho_kg_m3),
            actuator_tau_s=config.actuator_tau_s,
            wind_mode="none",
        )
        x_guess[idx + 1] = x
    if guess_name == "trim_glide":
        nu = command_radians_to_normalised(trim_command, interval_count)
    return x_guess, nu, float(duration)


def command_radians_to_normalised(command_rad: np.ndarray, interval_count: int) -> np.ndarray:
    command = np.asarray(command_rad, dtype=float).reshape(3)
    out = np.zeros((int(interval_count), 3), dtype=float)
    for idx, name in enumerate(("delta_a", "delta_e", "delta_r")):
        limit = AGGREGATE_LIMITS[name]
        angle_deg = float(np.rad2deg(command[idx]))
        if abs(angle_deg) <= 1e-12:
            norm = 0.0
        elif angle_deg >= 0.0 and abs(limit.positive_deg) > 1e-12:
            norm = angle_deg / float(limit.positive_deg)
        elif angle_deg < 0.0 and abs(limit.negative_deg) > 1e-12:
            norm = -angle_deg / float(limit.negative_deg)
        else:
            norm = 0.0
        out[:, idx] = float(np.clip(norm, -1.0, 1.0))
    return out


def dynamics_defect_max(
    x_ref: np.ndarray,
    u_ff: np.ndarray,
    times_s: np.ndarray,
    aircraft: AircraftModel,
    config: TurnOptimisationConfig,
) -> float:
    x_arr = np.asarray(x_ref, dtype=float)
    u_arr = np.asarray(u_ff, dtype=float)
    t_arr = np.asarray(times_s, dtype=float)
    if x_arr.shape[0] < 2:
        return float("inf")
    defects = []
    for idx in range(x_arr.shape[0] - 1):
        x_next = rk4_step(
            x=x_arr[idx],
            u_cmd=u_arr[idx],
            dt_s=float(t_arr[idx + 1] - t_arr[idx]),
            aircraft=aircraft,
            wind_model=None,
            rho_kg_m3=float(config.rho_kg_m3),
            actuator_tau_s=config.actuator_tau_s,
            wind_mode="none",
        )
        defects.append(float(np.linalg.norm(x_next - x_arr[idx + 1], ord=np.inf)))
    return float(max(defects, default=float("inf")))


def turn_result_metrics(
    x_ref: np.ndarray,
    u_ff: np.ndarray,
    nu_ff: np.ndarray,
    times_s: np.ndarray,
    target: TurnTarget,
    config: TurnOptimisationConfig,
    slack_max: float,
    dynamics_defect: float,
    objective_value: float,
) -> dict[str, object]:
    x_arr = np.asarray(x_ref, dtype=float)
    speed = np.linalg.norm(x_arr[:, 6:9], axis=1)
    alpha = np.arctan2(x_arr[:, STATE_INDEX["w"]], np.maximum(x_arr[:, STATE_INDEX["u"]], 1e-12))
    beta = np.arcsin(np.clip(x_arr[:, STATE_INDEX["v"]] / np.maximum(speed, 1e-12), -1.0, 1.0))
    margins = [safety_margins(row, ArenaConfig()) for row in x_arr]
    raw_heading = float(np.rad2deg(x_arr[-1, STATE_INDEX["psi"]] - x_arr[0, STATE_INDEX["psi"]]))
    wrapped = float((raw_heading + 180.0) % 360.0 - 180.0)
    directed = float(heading_direction_sign(target.direction) * raw_heading)
    terminal = x_arr[-1]
    terminal_speed = float(speed[-1])
    target_abs = abs(float(target.target_heading_deg))
    threshold = accepted_heading_threshold_deg(target_abs)
    return {
        "target_heading_deg": target_abs,
        "direction": target.direction,
        "wind_case": target.wind_case,
        "allow_high_alpha": bool(target.allow_high_alpha),
        "allow_safety_slack": bool(target.allow_safety_slack),
        "duration_s": float(times_s[-1] - times_s[0]),
        "sample_count": int(x_arr.shape[0]),
        "objective_value": float(objective_value),
        "dynamics_defect_max": float(dynamics_defect),
        "slack_max": float(slack_max),
        "slack_nonzero": bool(slack_max > 1e-8),
        "unwrapped_heading_change_deg": raw_heading,
        "actual_heading_change_deg": wrapped,
        "directed_heading_change_deg": directed,
        "heading_threshold_deg": threshold,
        "heading_gate_passed": bool(abs(wrapped) >= threshold if target_abs > 0.0 else True),
        "min_wall_distance_m": float(min(row["min_wall_distance_m"] for row in margins)),
        "min_floor_margin_m": float(min(row["floor_margin_m"] for row in margins)),
        "min_ceiling_margin_m": float(min(row["ceiling_margin_m"] for row in margins)),
        "inside_true_safety_volume": bool(all(row["inside_safe_volume"] for row in margins)),
        "terminal_z_w_m": float(terminal[STATE_INDEX["z_w"]]),
        "terminal_speed_m_s": terminal_speed,
        "terminal_abs_phi_deg": float(abs(np.rad2deg(terminal[STATE_INDEX["phi"]]))),
        "terminal_abs_theta_deg": float(abs(np.rad2deg(terminal[STATE_INDEX["theta"]]))),
        "max_abs_phi_deg": float(np.max(np.abs(np.rad2deg(x_arr[:, STATE_INDEX["phi"]])))),
        "max_abs_theta_deg": float(np.max(np.abs(np.rad2deg(x_arr[:, STATE_INDEX["theta"]])))),
        "max_alpha_deg": float(np.max(np.abs(np.rad2deg(alpha)))),
        "max_beta_deg": float(np.max(np.abs(np.rad2deg(beta)))),
        "saturation_fraction": float(np.mean(np.isclose(np.abs(nu_ff), 1.0, atol=1e-9))),
        "command_norm_max_abs": float(np.max(np.abs(nu_ff))) if nu_ff.size else 0.0,
        "exit_recoverable_gate": bool(
            terminal_speed >= config.terminal_speed_bounds_m_s[0]
            and terminal_speed <= config.terminal_speed_bounds_m_s[1]
            and abs(float(terminal[STATE_INDEX["phi"]])) <= np.deg2rad(config.terminal_bank_deg)
            and abs(float(terminal[STATE_INDEX["theta"]])) <= np.deg2rad(config.terminal_pitch_deg)
            and float(terminal[STATE_INDEX["z_w"]]) >= float(config.terminal_altitude_min_m)
        ),
    }


# =============================================================================
# 4) Direct Multiple-Shooting Solve
# =============================================================================
def solve_turn_ocp(
    target: TurnTarget,
    config: TurnOptimisationConfig,
    x0: np.ndarray,
    aircraft: AircraftModel,
    wind_model: object | None = None,
    wind_mode: str = "none",
    u_trim: np.ndarray | None = None,
    initial_guess_name: str = "trim_glide",
) -> OptimisedTurnResult:
    del wind_model
    if config.solver_name != "casadi_ipopt":
        raise ValueError("Phase 1 supports solver_name='casadi_ipopt' only.")
    if wind_mode != "none" or target.wind_case != "w0":
        raise ValueError("Phase 1 OCP optimisation is W0/no-wind only.")
    x_initial = np.asarray(x0, dtype=float).reshape(15)
    trim_command = (
        np.asarray(u_trim, dtype=float).reshape(3)
        if u_trim is not None
        else x_initial[12:15].copy()
    )
    interval_count = int(config.n_intervals)
    if interval_count < 2:
        raise ValueError("n_intervals must be at least 2.")

    guess_x, guess_nu, guess_duration = build_initial_guess(
        guess_name=initial_guess_name,
        target=target,
        config=config,
        x0=x_initial,
        u_trim=trim_command,
        aircraft=aircraft,
    )
    if abs(float(target.target_heading_deg)) <= 1e-12 and not target.allow_safety_slack:
        return _smoke_result_from_initial_guess(
            target=target,
            config=config,
            aircraft=aircraft,
            guess_x=guess_x,
            guess_nu=guess_nu,
            guess_duration=guess_duration,
            initial_guess_name=initial_guess_name,
        )
    opti = ca.Opti()
    x_var = opti.variable(15, interval_count + 1)
    nu_var = opti.variable(3, interval_count)
    t_var = opti.variable()
    psi_unwrapped = opti.variable(interval_count + 1)
    slack_var = None
    if target.allow_safety_slack:
        slack_var = opti.variable(15, interval_count + 1)
        opti.subject_to(slack_var >= 0.0)

    opti.subject_to(opti.bounded(float(config.t_min_s), t_var, float(config.t_max_s)))
    opti.subject_to(x_var[:, 0] == x_initial)
    opti.subject_to(opti.bounded(-1.0, nu_var, 1.0))
    opti.subject_to(psi_unwrapped[0] == x_var[STATE_INDEX["psi"], 0])

    dynamics = build_symbolic_dynamics(
        aircraft=aircraft,
        actuator_tau_s=config.actuator_tau_s,
        wind_mode="none",
    )
    dt_s = t_var / float(interval_count)
    objective = 0.0
    smoothness = 0.0
    slack_cost = 0.0

    for idx in range(interval_count):
        u_cmd = normalised_command_to_radians_ca(nu_var[:, idx])
        x_next = _rk4_symbolic(dynamics.function, x_var[:, idx], u_cmd, dt_s)
        opti.subject_to(x_var[:, idx + 1] == x_next)
        opti.subject_to(
            psi_unwrapped[idx + 1] - psi_unwrapped[idx]
            == x_var[STATE_INDEX["psi"], idx + 1] - x_var[STATE_INDEX["psi"], idx]
        )
        _add_path_constraints(opti, x_var[:, idx], config, slack_var, idx)
        if idx > 0:
            smoothness += ca.sumsqr(nu_var[:, idx] - nu_var[:, idx - 1])
    _add_path_constraints(opti, x_var[:, interval_count], config, slack_var, interval_count)
    _add_terminal_constraints(opti, x_var[:, interval_count], config, slack_var, interval_count)

    direction = heading_direction_sign(target.direction)
    heading_change_rad = direction * (psi_unwrapped[-1] - psi_unwrapped[0])
    target_rad = np.deg2rad(abs(float(target.target_heading_deg)))
    threshold_rad = np.deg2rad(accepted_heading_threshold_deg(target.target_heading_deg))
    if abs(float(target.target_heading_deg)) > 1e-12 and not target.allow_safety_slack:
        opti.subject_to(heading_change_rad >= threshold_rad)
    heading_error = heading_change_rad - target_rad
    objective += float(config.heading_weight) * heading_error**2
    objective += float(config.smoothness_weight) * smoothness
    objective += float(config.saturation_weight) * ca.sumsqr(nu_var)
    objective += _recovery_objective(x_var[:, interval_count], config)
    objective += _energy_objective(x_var[:, 0], x_var[:, interval_count], config)
    if slack_var is not None:
        slack_cost = ca.sumsqr(slack_var)
        objective += float(config.slack_weight) * slack_cost
    opti.minimize(objective)

    opti.set_initial(x_var, guess_x.T)
    opti.set_initial(nu_var, guess_nu.T)
    opti.set_initial(t_var, guess_duration)
    opti.set_initial(psi_unwrapped, guess_x[:, STATE_INDEX["psi"]])
    if slack_var is not None:
        opti.set_initial(slack_var, 0.05)

    opti.solver(
        "ipopt",
        {"print_time": False, "expand": True},
        {
            "print_level": 0,
            "max_iter": int(config.ipopt_max_iter),
            "max_cpu_time": float(config.max_solver_time_s),
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
        },
    )

    try:
        solution = opti.solve()
        x_sol = np.asarray(solution.value(x_var), dtype=float).T
        nu_sol = np.asarray(solution.value(nu_var), dtype=float).T
        duration = float(solution.value(t_var))
        objective_value = float(solution.value(objective))
        slack_max = (
            0.0
            if slack_var is None
            else float(np.max(np.asarray(solution.value(slack_var), dtype=float)))
        )
        stats = dict(opti.stats())
        status = str(stats.get("return_status", "Solve_Succeeded"))
        failure_reason = ""
    except Exception as exc:  # pragma: no cover - exercised when local IPOPT fails
        x_sol = guess_x
        nu_sol = guess_nu
        duration = guess_duration
        objective_value = float("inf")
        slack_max = float("inf") if target.allow_safety_slack else 0.0
        status = "solver_exception"
        stats = {"return_status": status, "exception": str(exc)}
        failure_reason = f"solver_failure: {exc}"

    times = np.linspace(0.0, float(duration), interval_count + 1)
    u_ff = normalised_command_to_radians(nu_sol)
    dynamics_defect = dynamics_defect_max(x_sol, u_ff, times, aircraft, config)
    metrics = turn_result_metrics(
        x_ref=x_sol,
        u_ff=u_ff,
        nu_ff=nu_sol,
        times_s=times,
        target=target,
        config=config,
        slack_max=slack_max,
        dynamics_defect=dynamics_defect,
        objective_value=objective_value,
    )
    if not failure_reason:
        failure_reason = _failure_reason_from_metrics(metrics, target, config)
    label = classify_turn_metrics(
        metrics=metrics,
        target=target,
        config=config,
        solver_status=status,
        failure_reason=failure_reason,
    )
    success = label in ACCEPTED_LABELS
    return OptimisedTurnResult(
        success=bool(success),
        failure_reason="" if success else failure_reason,
        feasibility_label=label,
        target=target,
        config=config,
        times_s=times,
        x_ref=x_sol,
        u_ff=u_ff,
        objective_value=objective_value,
        metrics=metrics,
        solver_stats={
            "solver_name": config.solver_name,
            "initial_guess_name": initial_guess_name,
            "return_status": status,
            **{str(key): _json_safe(value) for key, value in stats.items()},
        },
        nu_ff=nu_sol,
    )


def _rk4_symbolic(dyn_fun: ca.Function, x: ca.MX, u_cmd: ca.MX, dt_s: ca.MX) -> ca.MX:
    k1 = dyn_fun(x, u_cmd)
    k2 = dyn_fun(x + 0.5 * dt_s * k1, u_cmd)
    k3 = dyn_fun(x + 0.5 * dt_s * k2, u_cmd)
    k4 = dyn_fun(x + dt_s * k3, u_cmd)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _smoke_result_from_initial_guess(
    *,
    target: TurnTarget,
    config: TurnOptimisationConfig,
    aircraft: AircraftModel,
    guess_x: np.ndarray,
    guess_nu: np.ndarray,
    guess_duration: float,
    initial_guess_name: str,
) -> OptimisedTurnResult:
    times = np.linspace(0.0, float(guess_duration), int(config.n_intervals) + 1)
    u_ff = normalised_command_to_radians(guess_nu)
    dynamics_defect = dynamics_defect_max(guess_x, u_ff, times, aircraft, config)
    metrics = turn_result_metrics(
        x_ref=guess_x,
        u_ff=u_ff,
        nu_ff=guess_nu,
        times_s=times,
        target=target,
        config=config,
        slack_max=0.0,
        dynamics_defect=dynamics_defect,
        objective_value=0.0,
    )
    failure_reason = _failure_reason_from_metrics(metrics, target, config)
    label = classify_turn_metrics(
        metrics=metrics,
        target=target,
        config=config,
        solver_status="Smoke_Succeeded",
        failure_reason=failure_reason,
    )
    success = label in ACCEPTED_LABELS
    return OptimisedTurnResult(
        success=bool(success),
        failure_reason="" if success else failure_reason,
        feasibility_label=label,
        target=target,
        config=config,
        times_s=times,
        x_ref=guess_x,
        u_ff=u_ff,
        objective_value=0.0,
        metrics=metrics,
        solver_stats={
            "solver_name": config.solver_name,
            "initial_guess_name": initial_guess_name,
            "return_status": "Smoke_Succeeded",
            "smoke_case": True,
        },
        nu_ff=guess_nu,
    )


def _add_path_constraints(
    opti: ca.Opti,
    x_node: ca.MX,
    config: TurnOptimisationConfig,
    slack_var: ca.MX | None,
    node_idx: int,
) -> None:
    bounds = safe_bounds(ArenaConfig())
    slack = None if slack_var is None else slack_var[:, node_idx]
    _bounded(opti, x_node[STATE_INDEX["x_w"]], bounds["x_w"][0], bounds["x_w"][1], slack, 0)
    _bounded(opti, x_node[STATE_INDEX["y_w"]], bounds["y_w"][0], bounds["y_w"][1], slack, 2)
    _bounded(opti, x_node[STATE_INDEX["z_w"]], max(bounds["z_w"][0], 0.25), bounds["z_w"][1], slack, 4)
    speed = _speed_ca(x_node)
    opti.subject_to(speed >= float(config.speed_bounds_m_s[0]))
    opti.subject_to(speed <= float(config.speed_bounds_m_s[1]))
    opti.subject_to(ca.fabs(x_node[STATE_INDEX["phi"]]) <= np.deg2rad(config.max_bank_deg))
    opti.subject_to(ca.fabs(x_node[STATE_INDEX["theta"]]) <= np.deg2rad(config.max_pitch_deg))
    alpha = ca.atan2(x_node[STATE_INDEX["w"]], x_node[STATE_INDEX["u"]])
    beta_limit = np.sin(np.deg2rad(config.max_beta_deg))
    opti.subject_to(ca.fabs(alpha) <= np.deg2rad(config.max_alpha_deg))
    opti.subject_to(ca.fabs(x_node[STATE_INDEX["v"]]) <= beta_limit * speed)


def _add_terminal_constraints(
    opti: ca.Opti,
    x_terminal: ca.MX,
    config: TurnOptimisationConfig,
    slack_var: ca.MX | None,
    node_idx: int,
) -> None:
    bounds = safe_bounds(ArenaConfig())
    slack = None if slack_var is None else slack_var[:, node_idx]
    wall = float(config.terminal_wall_margin_m)
    _lower(opti, x_terminal[STATE_INDEX["x_w"]], bounds["x_w"][0] + wall, slack, 6)
    _upper(opti, x_terminal[STATE_INDEX["x_w"]], bounds["x_w"][1] - wall, slack, 7)
    _lower(opti, x_terminal[STATE_INDEX["y_w"]], bounds["y_w"][0] + wall, slack, 8)
    _upper(opti, x_terminal[STATE_INDEX["y_w"]], bounds["y_w"][1] - wall, slack, 9)
    _lower(opti, x_terminal[STATE_INDEX["z_w"]], config.terminal_altitude_min_m, slack, 10)
    terminal_speed = _speed_ca(x_terminal)
    _lower(opti, terminal_speed, config.terminal_speed_bounds_m_s[0], slack, 11)
    _upper(opti, terminal_speed, config.terminal_speed_bounds_m_s[1], slack, 12)
    phi_limit = np.deg2rad(config.terminal_bank_deg)
    theta_limit = np.deg2rad(config.terminal_pitch_deg)
    opti.subject_to(
        ca.fabs(x_terminal[STATE_INDEX["phi"]])
        <= (phi_limit if slack is None else phi_limit + slack[13])
    )
    opti.subject_to(
        ca.fabs(x_terminal[STATE_INDEX["theta"]])
        <= (theta_limit if slack is None else theta_limit + slack[14])
    )


def _bounded(
    opti: ca.Opti,
    value: ca.MX,
    low: float,
    high: float,
    slack: ca.MX | None,
    offset: int,
) -> None:
    _lower(opti, value, low, slack, offset)
    _upper(opti, value, high, slack, offset + 1)


def _lower(opti: ca.Opti, value: ca.MX, low: float, slack: ca.MX | None, idx: int) -> None:
    opti.subject_to(value >= float(low) if slack is None else value >= float(low) - slack[idx])


def _upper(opti: ca.Opti, value: ca.MX, high: float, slack: ca.MX | None, idx: int) -> None:
    opti.subject_to(value <= float(high) if slack is None else value <= float(high) + slack[idx])


def _speed_ca(x_node: ca.MX) -> ca.MX:
    return ca.sqrt(
        x_node[STATE_INDEX["u"]] ** 2
        + x_node[STATE_INDEX["v"]] ** 2
        + x_node[STATE_INDEX["w"]] ** 2
        + 1e-12
    )


def _recovery_objective(x_terminal: ca.MX, config: TurnOptimisationConfig) -> ca.MX:
    speed = _speed_ca(x_terminal)
    speed_mid = 0.5 * (
        float(config.terminal_speed_bounds_m_s[0]) + float(config.terminal_speed_bounds_m_s[1])
    )
    return float(config.recovery_weight) * (
        0.2 * (speed - speed_mid) ** 2
        + x_terminal[STATE_INDEX["phi"]] ** 2
        + x_terminal[STATE_INDEX["theta"]] ** 2
    )


def _energy_objective(
    x_start: ca.MX,
    x_terminal: ca.MX,
    config: TurnOptimisationConfig,
) -> ca.MX:
    g_m_s2 = 9.80665
    e_start = g_m_s2 * x_start[STATE_INDEX["z_w"]] + 0.5 * _speed_ca(x_start) ** 2
    e_terminal = g_m_s2 * x_terminal[STATE_INDEX["z_w"]] + 0.5 * _speed_ca(x_terminal) ** 2
    loss = ca.fmax(0.0, e_start - e_terminal)
    return float(config.energy_loss_weight) * loss**2


def _failure_reason_from_metrics(
    metrics: dict[str, object],
    target: TurnTarget,
    config: TurnOptimisationConfig,
) -> str:
    if float(metrics["dynamics_defect_max"]) > float(config.dynamics_defect_tolerance):
        return "dynamics defect above tolerance"
    if bool(metrics["slack_nonzero"]):
        return "nonzero safety or recovery slack"
    if not bool(metrics["inside_true_safety_volume"]):
        return "trajectory violates true safety volume"
    if not bool(metrics["heading_gate_passed"]):
        return "heading target threshold not achieved"
    if not bool(metrics["exit_recoverable_gate"]):
        return "terminal state outside recoverable exit bounds"
    if (
        float(metrics["max_alpha_deg"]) > float(config.max_alpha_deg)
        and not bool(target.allow_high_alpha)
    ):
        return "normal alpha bound exceeded"
    return ""


def classify_turn_metrics(
    *,
    metrics: dict[str, object],
    target: TurnTarget,
    config: TurnOptimisationConfig,
    solver_status: str,
    failure_reason: str,
) -> str:
    reason = str(failure_reason).lower()
    status = str(solver_status).lower()
    if "solver" in reason or "exception" in status or "failed" in status:
        return "solver_failure"
    if bool(metrics["slack_nonzero"]):
        return "boundary_only_with_slack"
    if float(metrics["dynamics_defect_max"]) > float(config.dynamics_defect_tolerance):
        return "solver_failure"
    if not bool(metrics["inside_true_safety_volume"]):
        if float(metrics["min_floor_margin_m"]) < 0.0:
            return "floor_limited"
        if float(metrics["min_ceiling_margin_m"]) < 0.0:
            return "ceiling_limited"
        return "wall_limited"
    if not bool(metrics["heading_gate_passed"]):
        return "under_turning"
    if not bool(metrics["exit_recoverable_gate"]):
        return "unrecoverable_exit"
    max_alpha = float(metrics["max_alpha_deg"])
    if max_alpha > float(config.max_alpha_deg) and not bool(target.allow_high_alpha):
        return "alpha_model_limited"
    if max_alpha > float(config.max_alpha_deg) and bool(target.allow_high_alpha):
        return "accepted_high_alpha_sim_only"
    if max_alpha > 20.0:
        return "accepted_moderate_alpha"
    return "accepted_low_alpha"


def classify_turn_result(result: OptimisedTurnResult) -> str:
    return classify_turn_metrics(
        metrics=result.metrics,
        target=result.target,
        config=result.config,
        solver_status=str(result.solver_stats.get("return_status", "")),
        failure_reason=result.failure_reason,
    )


# =============================================================================
# 5) Primitive Conversion and Persistence
# =============================================================================
def build_turn_trajectory_primitive(
    result: OptimisedTurnResult,
    context: PrimitiveContext,
    aircraft: AircraftModel,
    wind_model: object | None = None,
    wind_mode: str = "none",
    command_layer: CommandToSurfaceLayer | None = None,
) -> TrajectoryPrimitive:
    if wind_mode != "none":
        raise ValueError("Phase 2 TVLQR conversion supports W0/no-wind only.")
    command_layer = command_layer or CommandToSurfaceLayer()
    dt_s = float(np.median(np.diff(result.times_s)))
    u_nodes = _node_aligned_u_ff(result)
    a_mats, b_mats = linearise_trajectory_finite_difference(
        x_ref=result.x_ref,
        u_ff=u_nodes,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
        rho_kg_m3=float(result.config.rho_kg_m3),
        actuator_tau_s=command_layer.actuator_tau_vector_s,
    )
    k_lqr, s_mats = solve_discrete_tvlqr(
        a_mats=a_mats,
        b_mats=b_mats,
        dt_s=dt_s,
        config=_turn_tvlqr_config(),
    )
    target_tag = f"{int(round(abs(float(result.target.target_heading_deg)))):03d}"
    metadata = {
        "target_label": f"turn_ocp_ref_target_{target_tag}",
        "primitive_family": "turn_ocp",
        "candidate_id": str(result.solver_stats.get("initial_guess_name", "ocp")),
        "is_full_turn_claim": False,
        "target_heading_deg": float(result.target.target_heading_deg),
        "direction": result.target.direction,
        "phase_metadata": _phase_metadata(float(result.times_s[-1])),
        "command_domain": "aggregate surface radians from calibrated [-1,+1] normalised OCP",
        "terminal_altitude_min_m": float(result.config.terminal_altitude_min_m),
        "ocp_metrics": dict(result.metrics),
        "context_theta_trim_rad": float(context.theta_trim_rad),
    }
    return TrajectoryPrimitive(
        name=f"turn_ocp_{result.target.direction}_target_{target_tag}",
        times_s=result.times_s,
        x_ref=result.x_ref,
        u_ff=u_nodes,
        k_lqr=k_lqr,
        a_mats=a_mats,
        b_mats=b_mats,
        s_mats=s_mats,
        entry_limits=TrajectoryEntryLimits(
            max_position_error_m=0.24,
            max_attitude_error_rad=np.deg2rad(10.0),
            max_surface_error_rad=np.deg2rad(10.0),
        ),
        metadata=metadata,
    )


def _node_aligned_u_ff(result: OptimisedTurnResult) -> np.ndarray:
    u_arr = np.asarray(result.u_ff, dtype=float)
    n_nodes = int(np.asarray(result.times_s).size)
    if u_arr.shape == (n_nodes, 3):
        return u_arr
    if u_arr.shape == (n_nodes - 1, 3):
        return np.vstack([u_arr, u_arr[-1]])
    raise ValueError("u_ff must have shape (N, 3) or (N+1, 3) relative to times_s.")


def primitive_open_loop_copy(primitive: TrajectoryPrimitive) -> TrajectoryPrimitive:
    return replace(
        primitive,
        name=f"{primitive.name}_open_loop",
        k_lqr=np.zeros_like(primitive.k_lqr),
    )


def save_turn_result(
    result: OptimisedTurnResult,
    output_dir: str | Path,
    stem: str,
    primitive: TrajectoryPrimitive | None = None,
) -> dict[str, Path]:
    root = Path(output_dir)
    traj_dir = root / "trajectories"
    manifest_dir = root / "manifests"
    traj_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    npz_path = traj_dir / f"{stem}.npz"
    arrays = {
        "times_s": result.times_s,
        "x_ref": result.x_ref,
        "u_ff": result.u_ff,
        "nu_ff": np.zeros((0, 3)) if result.nu_ff is None else result.nu_ff,
    }
    if primitive is not None:
        arrays.update(
            {
                "a_mats": primitive.a_mats,
                "b_mats": primitive.b_mats,
                "k_lqr": primitive.k_lqr,
                "s_mats": primitive.s_mats,
            }
        )
    np.savez_compressed(npz_path, **arrays)
    manifest_path = manifest_dir / f"{stem}.json"
    manifest = {
        "target": asdict(result.target),
        "config": asdict(result.config),
        "success": bool(result.success),
        "failure_reason": result.failure_reason,
        "feasibility_label": result.feasibility_label,
        "metrics": result.metrics,
        "solver_stats": result.solver_stats,
        "trajectory_npz": str(npz_path).replace("\\", "/"),
        "has_tvlqr": primitive is not None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"trajectory": npz_path, "manifest": manifest_path}


def load_selected_turn_primitive(path: str | Path) -> TrajectoryPrimitive:
    npz_path = Path(path)
    data = np.load(npz_path, allow_pickle=False)
    required = ("times_s", "x_ref", "u_ff", "a_mats", "b_mats", "k_lqr", "s_mats")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"saved turn primitive missing arrays: {', '.join(missing)}")
    return TrajectoryPrimitive(
        name=npz_path.stem,
        times_s=data["times_s"],
        x_ref=data["x_ref"],
        u_ff=data["u_ff"],
        k_lqr=data["k_lqr"],
        a_mats=data["a_mats"],
        b_mats=data["b_mats"],
        s_mats=data["s_mats"],
        metadata={"loaded_from": str(npz_path).replace("\\", "/")},
    )


def _turn_tvlqr_config() -> TVLQRConfig:
    return TVLQRConfig(
        q_diag=(
            0.08,
            0.25,
            0.10,
            2.20,
            1.60,
            1.30,
            0.25,
            0.35,
            0.35,
            0.70,
            0.70,
            0.70,
            0.08,
            0.08,
            0.08,
        ),
        r_diag=(55.0, 55.0, 55.0),
        qf_diag=(
            0.20,
            0.50,
            0.20,
            3.20,
            2.60,
            2.20,
            0.40,
            0.45,
            0.45,
            0.90,
            0.90,
            0.90,
            0.10,
            0.10,
            0.10,
        ),
    )


def _json_safe(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
