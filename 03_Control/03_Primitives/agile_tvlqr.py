from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agile_trajectory_optimisation import (  # noqa: E402
    AGILE_TRAJECTORY_GENERATION_METHOD,
    COMMAND_TEMPLATE_ABLATION_METHOD,
    AgileTrajectoryResult,
    trajectory_metrics,
)
from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds  # noqa: E402
from command_contract import clip_normalised_command, normalised_command_to_surface_rad  # noqa: E402
from latency import (  # noqa: E402
    actuator_tau_for_case,
    delayed_state_sample,
    latency_adjusted_command_sample,
    latency_audit_fields_from_case,
    latency_case_config,
    latency_pass_label_for_single_run,
)
from rollout import rk4_step  # noqa: E402
from state_contract import STATE_SIZE, as_state_vector  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Public Data Containers
# 2) Public Synthesis Entry Point
# 3) Linearisation and Riccati Helpers
# 4) Closed-Loop Rollout Helpers
# =============================================================================


# =============================================================================
# 1) Public Data Containers
# =============================================================================
TVLQR_STATUS_SUCCESS = "tvlqr_synthesised"
TVLQR_STATUS_FALLBACK = "local_feedback_approx"
TVLQR_STATUS_NOT_APPLICABLE = "not_applicable_command_template_ablation"
EVIDENCE_TVLQR_NOMINAL = "trajectory_optimised_tvlqr_nominal_latency"
EVIDENCE_LOCAL_FEEDBACK_NOMINAL = "local_feedback_approx_nominal_latency"
EVIDENCE_COMMAND_TEMPLATE_BASELINE = "command_template_baseline"


@dataclass(frozen=True)
class TVLQRResult:
    trajectory_id: str
    controller_config_id: str
    tvlqr_status: str
    agile_evidence_class: str
    synthesis_wall_time_s: float
    k_feedback: np.ndarray
    time_s: np.ndarray
    x_closed_loop: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_effective_target: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    latency_case: str
    latency_pass_label: str
    closed_loop_success_flag: bool
    closed_loop_failure_label: str
    closed_loop_tracking_error_rms: float
    achieved_heading_deg: float
    terminal_heading_error_deg: float
    terminal_speed_m_s: float
    height_loss_m: float
    minimum_safety_margin_m: float
    actuator_saturation_fraction: float


# =============================================================================
# 2) Public Synthesis Entry Point
# =============================================================================
def synthesize_tvlqr_for_trajectory(
    trajectory: AgileTrajectoryResult,
    *,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    dt_s: float,
) -> TVLQRResult:
    """Synthesize TVLQR for one reference and evaluate nominal-latency closed loop."""

    started = time.perf_counter()
    if trajectory.trajectory_generation_method == COMMAND_TEMPLATE_ABLATION_METHOD:
        gains = np.zeros((trajectory.time_s.size, 3, STATE_SIZE), dtype=float)
        rollout = rollout_reference_feedback(
            trajectory,
            gains,
            aircraft=aircraft,
            wind_model=wind_model,
            wind_mode=wind_mode,
            dt_s=dt_s,
            latency_case="nominal",
        )
        return _result_from_rollout(
            trajectory,
            gains,
            rollout,
            tvlqr_status=TVLQR_STATUS_NOT_APPLICABLE,
            agile_evidence_class=EVIDENCE_COMMAND_TEMPLATE_BASELINE,
            synthesis_wall_time_s=time.perf_counter() - started,
        )

    try:
        gains = _finite_difference_tvlqr(
            trajectory,
            aircraft=aircraft,
            wind_model=wind_model,
            wind_mode=wind_mode,
            dt_s=dt_s,
        )
        tvlqr_status = TVLQR_STATUS_SUCCESS
        evidence_class = EVIDENCE_TVLQR_NOMINAL
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        gains = local_feedback_approximation(trajectory)
        tvlqr_status = TVLQR_STATUS_FALLBACK
        evidence_class = EVIDENCE_LOCAL_FEEDBACK_NOMINAL

    rollout = rollout_reference_feedback(
        trajectory,
        gains,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
        dt_s=dt_s,
        latency_case="nominal",
    )
    return _result_from_rollout(
        trajectory,
        gains,
        rollout,
        tvlqr_status=tvlqr_status,
        agile_evidence_class=evidence_class,
        synthesis_wall_time_s=time.perf_counter() - started,
    )


# =============================================================================
# 3) Linearisation and Riccati Helpers
# =============================================================================
def _finite_difference_tvlqr(
    trajectory: AgileTrajectoryResult,
    *,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    dt_s: float,
) -> np.ndarray:
    state = np.asarray(trajectory.x_ref, dtype=float)
    command = np.asarray(trajectory.u_norm_ref, dtype=float)
    if state.shape[0] < 2:
        raise ValueError("trajectory must contain at least two samples.")
    a_mats: list[np.ndarray] = []
    b_mats: list[np.ndarray] = []
    for index in range(state.shape[0] - 1):
        a, b = _linearise_discrete_step(
            state[index],
            command[index],
            aircraft=aircraft,
            wind_model=wind_model,
            wind_mode=wind_mode,
            dt_s=dt_s,
        )
        a_mats.append(a)
        b_mats.append(b)

    q = np.diag(
        [
            0.5,
            0.5,
            0.8,
            2.0,
            2.0,
            12.0,
            0.4,
            0.4,
            0.4,
            1.0,
            1.0,
            1.0,
            0.05,
            0.05,
            0.05,
        ]
    )
    q_final = 8.0 * q
    r = np.diag([0.20, 0.25, 0.20])
    p = q_final.copy()
    gains = np.zeros((state.shape[0], 3, STATE_SIZE), dtype=float)
    for rev_index in range(len(a_mats) - 1, -1, -1):
        a = a_mats[rev_index]
        b = b_mats[rev_index]
        system = r + b.T @ p @ b
        rhs = b.T @ p @ a
        gain = np.linalg.solve(system, rhs)
        if not np.all(np.isfinite(gain)):
            raise FloatingPointError("non-finite TVLQR gain")
        gains[rev_index] = gain
        p = q + a.T @ p @ (a - b @ gain)
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("non-finite Riccati matrix")
    return np.clip(gains, -8.0, 8.0)


def _linearise_discrete_step(
    x_ref: np.ndarray,
    u_norm_ref: np.ndarray,
    *,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = as_state_vector(x_ref)
    u = clip_normalised_command(u_norm_ref)
    eps_x = np.array(
        [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        dtype=float,
    )
    eps_u = np.array([1e-4, 1e-4, 1e-4], dtype=float)
    a = np.empty((STATE_SIZE, STATE_SIZE), dtype=float)
    b = np.empty((STATE_SIZE, 3), dtype=float)
    for column in range(STATE_SIZE):
        step = np.zeros(STATE_SIZE, dtype=float)
        step[column] = eps_x[column]
        a[:, column] = (
            _discrete_step(x + step, u, aircraft, wind_model, wind_mode, dt_s)
            - _discrete_step(x - step, u, aircraft, wind_model, wind_mode, dt_s)
        ) / (2.0 * eps_x[column])
    for column in range(3):
        step = np.zeros(3, dtype=float)
        step[column] = eps_u[column]
        b[:, column] = (
            _discrete_step(x, u + step, aircraft, wind_model, wind_mode, dt_s)
            - _discrete_step(x, u - step, aircraft, wind_model, wind_mode, dt_s)
        ) / (2.0 * eps_u[column])
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise FloatingPointError("non-finite finite-difference linearisation")
    return a, b


def _discrete_step(
    x: np.ndarray,
    u_norm: np.ndarray,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    dt_s: float,
) -> np.ndarray:
    command_rad = normalised_command_to_surface_rad(clip_normalised_command(u_norm))
    return rk4_step(
        x,
        command_rad,
        float(dt_s),
        aircraft,
        wind_model,
        str(wind_mode),
        (0.06, 0.06, 0.06),
    )


def local_feedback_approximation(trajectory: AgileTrajectoryResult) -> np.ndarray:
    """Return a labelled stabilising approximation if full TVLQR fails."""

    gains = np.zeros((trajectory.time_s.size, 3, STATE_SIZE), dtype=float)
    direction = float(np.sign(trajectory.request.direction_sign) or 1.0)
    for index in range(trajectory.time_s.size):
        gains[index, 0, 3] = 0.35
        gains[index, 0, 9] = 0.12
        gains[index, 0, 5] = -0.04 * direction
        gains[index, 1, 4] = 0.30
        gains[index, 1, 8] = -0.04
        gains[index, 1, 10] = 0.10
        gains[index, 2, 5] = -0.10 * direction
        gains[index, 2, 11] = 0.16
    return gains


# =============================================================================
# 4) Closed-Loop Rollout Helpers
# =============================================================================
def rollout_reference_feedback(
    trajectory: AgileTrajectoryResult,
    k_feedback: np.ndarray,
    *,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    dt_s: float,
    latency_case: str = "nominal",
    x0: np.ndarray | None = None,
) -> dict[str, np.ndarray | dict[str, object]]:
    time_s = np.asarray(trajectory.time_s, dtype=float)
    x_ref = np.asarray(trajectory.x_ref, dtype=float)
    u_ref = np.asarray(trajectory.u_norm_ref, dtype=float)
    gains = np.asarray(k_feedback, dtype=float)
    if gains.shape != (time_s.size, 3, STATE_SIZE):
        raise ValueError("k_feedback must have shape (N, 3, 15).")
    latency_config = latency_case_config(latency_case)
    actuator_tau = actuator_tau_for_case(latency_config)
    x_log = np.empty_like(x_ref)
    u_requested = np.empty((time_s.size, 3), dtype=float)
    u_effective = np.empty((time_s.size, 3), dtype=float)
    u_applied = np.empty((time_s.size, 3), dtype=float)
    delta_cmd = np.empty((time_s.size, 3), dtype=float)
    x_log[0] = x_ref[0] if x0 is None else as_state_vector(x0)
    final_index = time_s.size - 1

    for index, sample_time in enumerate(time_s):
        if index == 0:
            observed = x_log[0]
            observed_ref = x_ref[0]
        else:
            delayed_time = float(sample_time) - float(latency_config.state_feedback_delay_s)
            observed = delayed_state_sample(time_s[: index + 1], x_log[: index + 1], delayed_time)
            observed_ref = _sample_state(time_s, x_ref, delayed_time)
        error = observed - observed_ref
        u_requested[index] = clip_normalised_command(u_ref[index] - gains[index] @ error)
        u_effective[index] = latency_adjusted_command_sample(
            time_s[: index + 1],
            u_requested[: index + 1],
            float(sample_time),
            latency_config,
        )
        u_applied[index] = clip_normalised_command(u_effective[index])
        delta_cmd[index] = normalised_command_to_surface_rad(u_applied[index])
        if index == time_s.size - 1:
            break
        x_log[index + 1] = rk4_step(
            x_log[index],
            delta_cmd[index],
            float(dt_s),
            aircraft,
            wind_model,
            str(wind_mode),
            actuator_tau,
        )
        if not np.all(np.isfinite(x_log[index + 1])):
            final_index = index + 1
            u_requested[final_index] = u_requested[index]
            u_effective[final_index] = u_effective[index]
            u_applied[final_index] = u_applied[index]
            delta_cmd[final_index] = delta_cmd[index]
            break
        if not inside_bounds(x_log[index + 1, 0:3], TRUE_SAFE_BOUNDS):
            final_index = index + 1
            u_requested[final_index] = u_requested[index]
            u_effective[final_index] = u_effective[index]
            u_applied[final_index] = u_applied[index]
            delta_cmd[final_index] = delta_cmd[index]
            break

    terminal = final_index + 1
    latency_fields = latency_audit_fields_from_case(
        latency_config,
        active_actuator_tau_s=actuator_tau,
    )
    latency_fields["state_feedback_delay_applied"] = bool(
        latency_config.state_feedback_delay_s > 0.0
    )
    return {
        "time_s": time_s[:terminal],
        "x_closed_loop": x_log[:terminal],
        "u_norm_requested": u_requested[:terminal],
        "u_norm_effective_target": u_effective[:terminal],
        "u_norm_applied": u_applied[:terminal],
        "delta_cmd_rad": delta_cmd[:terminal],
        "latency_fields": latency_fields,
    }


def _result_from_rollout(
    trajectory: AgileTrajectoryResult,
    gains: np.ndarray,
    rollout: dict[str, np.ndarray | dict[str, object]],
    *,
    tvlqr_status: str,
    agile_evidence_class: str,
    synthesis_wall_time_s: float,
) -> TVLQRResult:
    time_s = np.asarray(rollout["time_s"], dtype=float)
    x_closed_loop = np.asarray(rollout["x_closed_loop"], dtype=float)
    u_requested = np.asarray(rollout["u_norm_requested"], dtype=float)
    metrics = trajectory_metrics(
        time_s,
        x_closed_loop,
        u_requested,
        trajectory.request,
    )
    failure_label = _closed_loop_failure_label(metrics, trajectory)
    success = failure_label == "success"
    latency_case = str(rollout["latency_fields"]["latency_case"])  # type: ignore[index]
    latency_pass = latency_pass_label_for_single_run(latency_case, success)
    tracking_error = _tracking_error_rms(time_s, x_closed_loop, trajectory.time_s, trajectory.x_ref)
    return TVLQRResult(
        trajectory_id=str(trajectory.trajectory_id),
        controller_config_id=f"{trajectory.trajectory_id}_{tvlqr_status}",
        tvlqr_status=str(tvlqr_status),
        agile_evidence_class=str(agile_evidence_class),
        synthesis_wall_time_s=float(synthesis_wall_time_s),
        k_feedback=np.asarray(gains, dtype=float),
        time_s=time_s,
        x_closed_loop=x_closed_loop,
        u_norm_requested=u_requested,
        u_norm_effective_target=np.asarray(rollout["u_norm_effective_target"], dtype=float),
        u_norm_applied=np.asarray(rollout["u_norm_applied"], dtype=float),
        delta_cmd_rad=np.asarray(rollout["delta_cmd_rad"], dtype=float),
        latency_case=latency_case,
        latency_pass_label=latency_pass,
        closed_loop_success_flag=bool(success),
        closed_loop_failure_label=failure_label,
        closed_loop_tracking_error_rms=float(tracking_error),
        achieved_heading_deg=float(metrics["achieved_heading_deg"]),
        terminal_heading_error_deg=float(metrics["terminal_heading_error_deg"]),
        terminal_speed_m_s=float(metrics["terminal_speed_m_s"]),
        height_loss_m=float(metrics["height_loss_m"]),
        minimum_safety_margin_m=float(metrics["minimum_safety_margin_m"]),
        actuator_saturation_fraction=float(metrics["actuator_saturation_fraction"]),
    )


def _sample_state(time_s: np.ndarray, states: np.ndarray, query_time_s: float) -> np.ndarray:
    return delayed_state_sample(np.asarray(time_s, dtype=float), np.asarray(states, dtype=float), query_time_s)


def _tracking_error_rms(
    time_s: np.ndarray,
    states: np.ndarray,
    reference_time_s: np.ndarray,
    reference_states: np.ndarray,
) -> float:
    if len(time_s) == 0:
        return float("nan")
    errors = []
    for time_value, state in zip(time_s, states, strict=True):
        ref = _sample_state(reference_time_s, reference_states, float(time_value))
        errors.append(float(np.linalg.norm(state - ref)))
    return float(np.sqrt(np.mean(np.square(errors))))


def _closed_loop_failure_label(
    metrics: dict[str, float],
    trajectory: AgileTrajectoryResult,
) -> str:
    if metrics["minimum_safety_margin_m"] < 0.0:
        return "true_safety_violation"
    if metrics["terminal_heading_error_deg"] > trajectory.request.heading_success_tolerance_deg:
        return "target_miss"
    if metrics["terminal_speed_m_s"] < trajectory.request.minimum_terminal_speed_m_s:
        return "terminal_speed_below_recoverable_limit"
    if metrics["height_loss_m"] > trajectory.request.maximum_recoverable_height_loss_m:
        return "height_loss_above_recoverable_limit"
    return "success"
