from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m  # noqa: E402
from command_contract import clip_normalised_command, normalised_command_to_surface_rad  # noqa: E402
from primitive_library_generators import generate_command_profile  # noqa: E402
from primitive_library_schema import PrimitiveCandidateSpec  # noqa: E402
from rollout import rk4_step  # noqa: E402
from state_contract import STATE_SIZE, as_state_vector  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Public Data Containers
# 2) Optimisation Entry Point
# 3) Rollout and Metric Helpers
# 4) Request/Result Conversion Helpers
# =============================================================================


# =============================================================================
# 1) Public Data Containers
# =============================================================================
AGILE_TRAJECTORY_GENERATION_METHOD = "slsqp_direct_shooting_reference"
COMMAND_TEMPLATE_ABLATION_METHOD = "command_template_initial_guess_or_ablation"
AGILE_FAMILIES = (
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
AGILE_TARGETS_BY_FAMILY = {
    "canyon_steep_bank": (15.0, 30.0, 45.0),
    "wingover_lite": (30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0),
    "bank_yaw_energy_retaining": (15.0, 30.0, 45.0),
}
DEFAULT_HEADING_SUCCESS_TOLERANCE_DEG = 5.0


@dataclass(frozen=True)
class AgileTrajectoryRequest:
    trajectory_id: str
    family: str
    target_heading_deg: float
    direction_sign: int
    x0: np.ndarray
    layout_branch_id: str = ""
    fan_layout: str = ""
    test_environment_mode: str = ""
    updraft_model_id: str = ""
    sample_id: str = ""
    dt_s: float = 0.02
    horizon_s: float = 0.95
    command_knot_count: int = 5
    max_iterations: int = 20
    heading_success_tolerance_deg: float = DEFAULT_HEADING_SUCCESS_TOLERANCE_DEG
    minimum_terminal_speed_m_s: float = 3.5
    maximum_recoverable_height_loss_m: float = 1.4
    random_seed: int = 20260530


@dataclass(frozen=True)
class AgileTrajectoryResult:
    request: AgileTrajectoryRequest
    trajectory_id: str
    trajectory_generation_method: str
    optimizer_status: str
    optimizer_message: str
    optimizer_success: bool
    optimizer_iterations: int
    optimizer_wall_time_s: float
    objective_cost: float
    heading_cost: float
    speed_loss_cost: float
    height_loss_cost: float
    saturation_cost: float
    safety_status: str
    time_s: np.ndarray
    x_ref: np.ndarray
    u_norm_ref: np.ndarray
    delta_cmd_ref_rad: np.ndarray
    achieved_heading_deg: float
    terminal_heading_error_deg: float
    terminal_speed_m_s: float
    height_loss_m: float
    minimum_safety_margin_m: float
    actuator_saturation_fraction: float
    open_loop_success_flag: bool
    failure_label: str


# =============================================================================
# 2) Optimisation Entry Point
# =============================================================================
def optimise_agile_trajectory(
    request: AgileTrajectoryRequest,
    *,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
) -> AgileTrajectoryResult:
    """Optimise and store a target-specific open-loop reference trajectory."""

    active_request = _validate_request(request)
    time_s = time_grid(active_request.dt_s, active_request.horizon_s)
    initial = command_template_initial_guess(active_request, time_s)
    knot_times = np.linspace(float(time_s[0]), float(time_s[-1]), active_request.command_knot_count)
    initial_knots = _sample_commands(time_s, initial, knot_times).reshape(-1)

    def unpack(values: np.ndarray) -> np.ndarray:
        knots = np.asarray(values, dtype=float).reshape(active_request.command_knot_count, 3)
        return _commands_from_knots(knot_times, knots, time_s)

    def objective(values: np.ndarray) -> float:
        replay = integrate_reference(
            x0=active_request.x0,
            time_s=time_s,
            u_norm_ref=unpack(values),
            dt_s=active_request.dt_s,
            aircraft=aircraft,
            wind_model=wind_model,
            wind_mode=wind_mode,
        )
        metrics = trajectory_metrics(
            replay["time_s"],
            replay["x_ref"],
            replay["u_norm_ref"],
            active_request,
        )
        safety_penalty = 0.0
        if metrics["minimum_safety_margin_m"] < 0.0:
            safety_penalty = 1.0e6 * abs(float(metrics["minimum_safety_margin_m"]))
        return float(
            metrics["heading_cost"]
            + metrics["speed_loss_cost"]
            + metrics["height_loss_cost"]
            + metrics["saturation_cost"]
            + safety_penalty
        )

    constraints = (
        {
            "type": "ineq",
            "fun": lambda values: _constraint_minimum_safety_margin(
                values,
                unpack=unpack,
                request=active_request,
                time_s=time_s,
                aircraft=aircraft,
                wind_model=wind_model,
                wind_mode=wind_mode,
            ),
        },
    )

    started = time.perf_counter()
    try:
        result = minimize(
            objective,
            initial_knots,
            method="SLSQP",
            bounds=[(-1.0, 1.0)] * initial_knots.size,
            constraints=constraints,
            options={
                "maxiter": int(active_request.max_iterations),
                "ftol": 1.0e-4,
                "disp": False,
            },
        )
        optimizer_message = str(result.message)
        optimizer_success = bool(result.success)
        values = np.asarray(result.x, dtype=float)
        iterations = int(getattr(result, "nit", 0))
    except Exception as exc:  # pragma: no cover - exercised through failure tests via monkeypatch
        optimizer_message = f"{type(exc).__name__}: {exc}"
        optimizer_success = False
        values = initial_knots
        iterations = 0
    elapsed = time.perf_counter() - started

    u_ref = unpack(values)
    replay = integrate_reference(
        x0=active_request.x0,
        time_s=time_s,
        u_norm_ref=u_ref,
        dt_s=active_request.dt_s,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
    )
    metrics = trajectory_metrics(
        replay["time_s"],
        replay["x_ref"],
        replay["u_norm_ref"],
        active_request,
    )
    failure_label = _failure_label(metrics, active_request)
    open_loop_success = failure_label == "success" and optimizer_success
    optimizer_status = "success" if optimizer_success and failure_label == "success" else "failed"
    if optimizer_success and failure_label != "success":
        optimizer_status = "optimizer_converged_reference_failed_acceptance"

    return AgileTrajectoryResult(
        request=active_request,
        trajectory_id=str(active_request.trajectory_id),
        trajectory_generation_method=AGILE_TRAJECTORY_GENERATION_METHOD,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        optimizer_success=optimizer_success,
        optimizer_iterations=iterations,
        optimizer_wall_time_s=float(elapsed),
        objective_cost=float(metrics["objective_cost"]),
        heading_cost=float(metrics["heading_cost"]),
        speed_loss_cost=float(metrics["speed_loss_cost"]),
        height_loss_cost=float(metrics["height_loss_cost"]),
        saturation_cost=float(metrics["saturation_cost"]),
        safety_status="pass" if metrics["minimum_safety_margin_m"] >= 0.0 else "failed_hard_safety",
        time_s=replay["time_s"],
        x_ref=replay["x_ref"],
        u_norm_ref=replay["u_norm_ref"],
        delta_cmd_ref_rad=replay["delta_cmd_ref_rad"],
        achieved_heading_deg=float(metrics["achieved_heading_deg"]),
        terminal_heading_error_deg=float(metrics["terminal_heading_error_deg"]),
        terminal_speed_m_s=float(metrics["terminal_speed_m_s"]),
        height_loss_m=float(metrics["height_loss_m"]),
        minimum_safety_margin_m=float(metrics["minimum_safety_margin_m"]),
        actuator_saturation_fraction=float(metrics["actuator_saturation_fraction"]),
        open_loop_success_flag=bool(open_loop_success),
        failure_label=failure_label,
    )


def command_template_reference(
    request: AgileTrajectoryRequest,
    *,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
) -> AgileTrajectoryResult:
    """Return a labelled command-template ablation, not D2 success evidence."""

    active_request = _validate_request(request)
    time_s = time_grid(active_request.dt_s, active_request.horizon_s)
    command = command_template_initial_guess(active_request, time_s)
    replay = integrate_reference(
        x0=active_request.x0,
        time_s=time_s,
        u_norm_ref=command,
        dt_s=active_request.dt_s,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
    )
    metrics = trajectory_metrics(
        replay["time_s"],
        replay["x_ref"],
        replay["u_norm_ref"],
        active_request,
    )
    return AgileTrajectoryResult(
        request=active_request,
        trajectory_id=str(active_request.trajectory_id),
        trajectory_generation_method=COMMAND_TEMPLATE_ABLATION_METHOD,
        optimizer_status="not_optimised_command_template_ablation",
        optimizer_message="raw command template used only as initial guess or ablation",
        optimizer_success=False,
        optimizer_iterations=0,
        optimizer_wall_time_s=0.0,
        objective_cost=float(metrics["objective_cost"]),
        heading_cost=float(metrics["heading_cost"]),
        speed_loss_cost=float(metrics["speed_loss_cost"]),
        height_loss_cost=float(metrics["height_loss_cost"]),
        saturation_cost=float(metrics["saturation_cost"]),
        safety_status="pass" if metrics["minimum_safety_margin_m"] >= 0.0 else "failed_hard_safety",
        time_s=replay["time_s"],
        x_ref=replay["x_ref"],
        u_norm_ref=replay["u_norm_ref"],
        delta_cmd_ref_rad=replay["delta_cmd_ref_rad"],
        achieved_heading_deg=float(metrics["achieved_heading_deg"]),
        terminal_heading_error_deg=float(metrics["terminal_heading_error_deg"]),
        terminal_speed_m_s=float(metrics["terminal_speed_m_s"]),
        height_loss_m=float(metrics["height_loss_m"]),
        minimum_safety_margin_m=float(metrics["minimum_safety_margin_m"]),
        actuator_saturation_fraction=float(metrics["actuator_saturation_fraction"]),
        open_loop_success_flag=False,
        failure_label=_failure_label(metrics, active_request),
    )


# =============================================================================
# 3) Rollout and Metric Helpers
# =============================================================================
def time_grid(dt_s: float, horizon_s: float) -> np.ndarray:
    dt = float(dt_s)
    horizon = float(horizon_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be positive and finite.")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_s must be positive and finite.")
    count = int(math.floor(horizon / dt + 0.5)) + 1
    return np.linspace(0.0, dt * (count - 1), count, dtype=float)


def command_template_initial_guess(
    request: AgileTrajectoryRequest,
    time_s: np.ndarray,
) -> np.ndarray:
    spec = PrimitiveCandidateSpec(
        primitive_id=str(request.trajectory_id),
        parent_primitive_id=str(request.family),
        variant_id=str(request.trajectory_id),
        family=str(request.family),
        target_heading_deg=float(request.target_heading_deg),
        updraft_config=str(request.updraft_model_id or "none"),
        wind_fidelity="W0" if str(request.test_environment_mode).startswith("W0") else "W1",
        start_condition="d2_boundary_refinement",
        direction_sign=int(request.direction_sign),
        horizon_s=float(request.horizon_s),
    )
    command, _ = generate_command_profile(spec, np.asarray(time_s, dtype=float))
    return np.asarray(command, dtype=float)


def integrate_reference(
    *,
    x0: np.ndarray,
    time_s: np.ndarray,
    u_norm_ref: np.ndarray,
    dt_s: float,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
) -> dict[str, np.ndarray]:
    time = np.asarray(time_s, dtype=float).reshape(-1)
    command = np.asarray(u_norm_ref, dtype=float)
    if command.shape != (time.size, 3):
        raise ValueError("u_norm_ref must have shape (N, 3).")
    x_log = np.empty((time.size, STATE_SIZE), dtype=float)
    u_applied = np.empty((time.size, 3), dtype=float)
    delta_cmd = np.empty((time.size, 3), dtype=float)
    x_log[0] = as_state_vector(x0)
    final_index = time.size - 1

    for index in range(time.size):
        u_applied[index] = clip_normalised_command(command[index])
        delta_cmd[index] = normalised_command_to_surface_rad(u_applied[index])
        if index == time.size - 1:
            break
        x_log[index + 1] = rk4_step(
            x_log[index],
            delta_cmd[index],
            float(dt_s),
            aircraft,
            wind_model,
            str(wind_mode),
            actuator_tau_s,
        )
        if not np.all(np.isfinite(x_log[index + 1])):
            final_index = index + 1
            u_applied[final_index] = u_applied[index]
            delta_cmd[final_index] = delta_cmd[index]
            break
        if not inside_bounds(x_log[index + 1, 0:3], TRUE_SAFE_BOUNDS):
            final_index = index + 1
            u_applied[final_index] = u_applied[index]
            delta_cmd[final_index] = delta_cmd[index]
            break

    terminal = final_index + 1
    return {
        "time_s": time[:terminal],
        "x_ref": x_log[:terminal],
        "u_norm_ref": command[:terminal],
        "u_norm_applied": u_applied[:terminal],
        "delta_cmd_ref_rad": delta_cmd[:terminal],
    }


def trajectory_metrics(
    time_s: np.ndarray,
    x_ref: np.ndarray,
    u_norm_ref: np.ndarray,
    request: AgileTrajectoryRequest,
) -> dict[str, float]:
    time = np.asarray(time_s, dtype=float).reshape(-1)
    state = np.asarray(x_ref, dtype=float)
    command = np.asarray(u_norm_ref, dtype=float)
    if state.ndim != 2 or state.shape[1] != STATE_SIZE:
        raise ValueError("x_ref must have shape (N, 15).")
    if command.shape != (state.shape[0], 3):
        raise ValueError("u_norm_ref must have shape (N, 3).")
    if time.size != state.shape[0]:
        raise ValueError("time_s and x_ref must share row count.")

    yaw_deg = np.rad2deg(np.unwrap(state[:, 5]))
    achieved = float(int(np.sign(request.direction_sign) or 1) * (yaw_deg[-1] - yaw_deg[0]))
    heading_error = float(abs(achieved - float(request.target_heading_deg)))
    speed = np.linalg.norm(state[:, 6:9], axis=1)
    terminal_speed = float(speed[-1])
    speed_loss = max(float(speed[0] - terminal_speed), 0.0)
    height_loss = max(float(state[0, 2] - state[-1, 2]), 0.0)
    margins = [position_margin_m(position, TRUE_SAFE_BOUNDS)["min_margin_m"] for position in state[:, 0:3]]
    minimum_margin = float(min(margins)) if margins else float("nan")
    saturation_fraction = float(np.mean(np.abs(command) >= 1.0 - 1.0e-12))
    heading_cost = heading_error * heading_error
    speed_loss_cost = 4.0 * max(0.0, float(request.minimum_terminal_speed_m_s) - terminal_speed) ** 2
    height_loss_cost = 2.0 * max(0.0, height_loss - float(request.maximum_recoverable_height_loss_m)) ** 2
    saturation_cost = 0.1 * saturation_fraction
    objective = heading_cost + speed_loss_cost + height_loss_cost + saturation_cost
    return {
        "duration_s": float(time[-1] - time[0]) if time.size else 0.0,
        "achieved_heading_deg": achieved,
        "terminal_heading_error_deg": heading_error,
        "terminal_speed_m_s": terminal_speed,
        "speed_loss_m_s": speed_loss,
        "height_loss_m": height_loss,
        "minimum_safety_margin_m": minimum_margin,
        "actuator_saturation_fraction": saturation_fraction,
        "heading_cost": float(heading_cost),
        "speed_loss_cost": float(speed_loss_cost),
        "height_loss_cost": float(height_loss_cost),
        "saturation_cost": float(saturation_cost),
        "objective_cost": float(objective),
    }


# =============================================================================
# 4) Request/Result Conversion Helpers
# =============================================================================
def result_index_row(result: AgileTrajectoryResult) -> dict[str, object]:
    request = result.request
    return {
        "trajectory_id": str(result.trajectory_id),
        "trajectory_generation_method": str(result.trajectory_generation_method),
        "optimizer_status": str(result.optimizer_status),
        "optimizer_message": str(result.optimizer_message),
        "optimizer_success": bool(result.optimizer_success),
        "optimizer_iterations": int(result.optimizer_iterations),
        "optimizer_wall_time_s": float(result.optimizer_wall_time_s),
        "objective_cost": float(result.objective_cost),
        "heading_cost": float(result.heading_cost),
        "speed_loss_cost": float(result.speed_loss_cost),
        "height_loss_cost": float(result.height_loss_cost),
        "saturation_cost": float(result.saturation_cost),
        "safety_status": str(result.safety_status),
        "layout_branch_id": str(request.layout_branch_id),
        "fan_layout": str(request.fan_layout),
        "test_environment_mode": str(request.test_environment_mode),
        "updraft_model_id": str(request.updraft_model_id),
        "sample_id": str(request.sample_id),
        "family": str(request.family),
        "target_heading_deg": float(request.target_heading_deg),
        "direction_sign": int(request.direction_sign),
        "reference_horizon_s": float(result.time_s[-1] - result.time_s[0]),
        "terminal_heading_target_deg": float(request.target_heading_deg),
        "achieved_heading_deg": float(result.achieved_heading_deg),
        "terminal_heading_error_deg": float(result.terminal_heading_error_deg),
        "terminal_speed_m_s": float(result.terminal_speed_m_s),
        "height_loss_m": float(result.height_loss_m),
        "minimum_safety_margin_m": float(result.minimum_safety_margin_m),
        "actuator_saturation_fraction": float(result.actuator_saturation_fraction),
        "open_loop_success_flag": bool(result.open_loop_success_flag),
        "failure_label": str(result.failure_label),
        "aggressive_turn_speed_height_loss_recoverable": bool(
            result.terminal_speed_m_s >= request.minimum_terminal_speed_m_s
            and result.height_loss_m <= request.maximum_recoverable_height_loss_m
        ),
    }


def result_sample_rows(
    result: AgileTrajectoryResult,
    *,
    controller_config_id: str = "",
    k_feedback: np.ndarray | None = None,
) -> list[dict[str, object]]:
    gains = (
        np.zeros((result.time_s.size, 3, STATE_SIZE), dtype=float)
        if k_feedback is None
        else np.asarray(k_feedback, dtype=float)
    )
    if gains.shape != (result.time_s.size, 3, STATE_SIZE):
        raise ValueError("k_feedback must have shape (N, 3, 15).")
    rows: list[dict[str, object]] = []
    for index, sample_time in enumerate(result.time_s):
        row: dict[str, object] = {
            "trajectory_id": str(result.trajectory_id),
            "controller_config_id": str(controller_config_id),
            "sample_index": int(index),
            "time_s": float(sample_time),
        }
        for state_index in range(STATE_SIZE):
            row[f"x_ref_{state_index:02d}"] = float(result.x_ref[index, state_index])
        for command_index, name in enumerate(("delta_a", "delta_e", "delta_r")):
            row[f"u_norm_ref_{name}"] = float(result.u_norm_ref[index, command_index])
            row[f"delta_cmd_ref_rad_{name}"] = float(
                result.delta_cmd_ref_rad[index, command_index]
            )
        for command_index in range(3):
            for state_index in range(STATE_SIZE):
                row[f"k_{command_index}_{state_index:02d}"] = float(
                    gains[index, command_index, state_index]
                )
        rows.append(row)
    return rows


def _validate_request(request: AgileTrajectoryRequest) -> AgileTrajectoryRequest:
    if str(request.family) not in AGILE_FAMILIES:
        raise ValueError(f"unknown D2 agile family: {request.family!r}")
    target = float(request.target_heading_deg)
    allowed_targets = AGILE_TARGETS_BY_FAMILY[str(request.family)]
    if target not in allowed_targets:
        raise ValueError(
            f"target_heading_deg={target:g} is not enabled for {request.family}."
        )
    direction = int(np.sign(int(request.direction_sign)) or 1)
    if direction not in (-1, 1):
        raise ValueError("direction_sign must be -1 or +1.")
    x0 = as_state_vector(request.x0)
    if not inside_bounds(x0[0:3], TRUE_SAFE_BOUNDS):
        raise ValueError("initial state must be inside the true safe bounds.")
    if int(request.command_knot_count) < 2:
        raise ValueError("command_knot_count must be at least 2.")
    if int(request.max_iterations) <= 0:
        raise ValueError("max_iterations must be positive.")
    return replace(request, direction_sign=direction, x0=x0)


def _sample_commands(
    source_time_s: np.ndarray,
    commands: np.ndarray,
    query_time_s: np.ndarray,
) -> np.ndarray:
    source_time = np.asarray(source_time_s, dtype=float).reshape(-1)
    command = np.asarray(commands, dtype=float)
    query = np.asarray(query_time_s, dtype=float).reshape(-1)
    return np.column_stack(
        [np.interp(query, source_time, command[:, column]) for column in range(3)]
    )


def _commands_from_knots(
    knot_times_s: np.ndarray,
    knot_commands: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    command = np.column_stack(
        [
            np.interp(time_s, knot_times_s, knot_commands[:, column])
            for column in range(3)
        ]
    )
    return np.clip(command, -1.0, 1.0)


def _failure_label(
    metrics: dict[str, float],
    request: AgileTrajectoryRequest,
) -> str:
    if metrics["minimum_safety_margin_m"] < 0.0:
        return "true_safety_violation"
    if metrics["terminal_heading_error_deg"] > float(request.heading_success_tolerance_deg):
        return "target_miss"
    if metrics["terminal_speed_m_s"] < float(request.minimum_terminal_speed_m_s):
        return "terminal_speed_below_recoverable_limit"
    if metrics["height_loss_m"] > float(request.maximum_recoverable_height_loss_m):
        return "height_loss_above_recoverable_limit"
    return "success"


def _constraint_minimum_safety_margin(
    values: np.ndarray,
    *,
    unpack,
    request: AgileTrajectoryRequest,
    time_s: np.ndarray,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
) -> float:
    replay = integrate_reference(
        x0=request.x0,
        time_s=time_s,
        u_norm_ref=unpack(values),
        dt_s=request.dt_s,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
    )
    metrics = trajectory_metrics(
        replay["time_s"],
        replay["x_ref"],
        replay["u_norm_ref"],
        request,
    )
    return float(metrics["minimum_safety_margin_m"])
