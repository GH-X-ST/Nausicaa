from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m
from command_contract import normalised_command_to_surface_rad
from env_ctx import (
    EnvironmentContext,
    context_feature_vector_json,
    environment_context_row,
)
from flight_dynamics import adapt_glider, state_derivative
from glider import build_nausicaa_glider
from latency import (
    actuator_tau_for_case,
    delayed_state_sample,
    latency_adjusted_command_sample,
    latency_case_config,
    latency_execution_status,
    latency_mechanism_flags_from_case,
    latency_pass_label_for_single_run,
)
from prim_cat import PrimitiveDefinition, primitive_parameters_json
from prim_ctrl import PrimitiveControlContext, primitive_command_norm
from state_contract import STATE_INDEX, STATE_SIZE
from state_contract import as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Rollout schema
# 2) Public rollout evaluation
# 3) Smoke rollout evaluation
# 4) Dynamics rollout evaluation
# 5) Serialisation helpers
# =============================================================================


# =============================================================================
# 1) Rollout Schema
# =============================================================================
OUTCOME_CLASSES = (
    "accepted",
    "weak",
    "failed",
    "rejected",
    "boundary_terminal",
    "blocked",
)
ROLLOUT_BACKENDS = (
    "smoke_only",
    "model_backed_command_template",
    "model_backed_feedback",
    "blocked_feedback",
)
EVIDENCE_ROLE_BY_BACKEND = {
    "smoke_only": "interface_smoke",
    "model_backed_command_template": "diagnostic_model_rollout",
    "model_backed_feedback": "feedback_rollout_candidate",
    "blocked_feedback": "blocked_feedback",
}
ROLLOUT_EVIDENCE_COLUMNS = (
    "rollout_id",
    "episode_id",
    "environment_id",
    "W_layer",
    "initial_state_vector",
    "context_feature_vector",
    "primitive_id",
    "primitive_family",
    "primitive_parameters",
    "controller_mode",
    "feedback_mode",
    "latency_case",
    "state_feedback_delay_applied",
    "command_delay_applied",
    "actuator_lag_applied",
    "latency_execution_status",
    "latency_pass_label",
    "timing_model_version",
    "rollout_backend",
    "evidence_role",
    "surrogate_binding_status",
    "trajectory_integrity_status",
    "entry_check_status",
    "exit_check_status",
    "continuation_status",
    "episode_terminal_status",
    "episode_utility_label",
    "terminal_use_trainable",
    "boundary_use_class",
    "accepted",
    "outcome_class",
    "energy_residual_m",
    "lift_dwell_time_s",
    "minimum_wall_margin_m",
    "floor_margin_m",
    "ceiling_margin_m",
    "minimum_speed_m_s",
    "saturation_count",
    "saturation_fraction",
    "max_abs_command_norm",
    "max_abs_surface_rad",
    "exit_state_vector",
    "termination_cause",
    "failure_label",
    "claim_status",
)


@dataclass(frozen=True)
class RolloutConfig:
    W_layer: str = "W0"
    dt_s: float = 0.02
    minimum_speed_m_s: float = 3.0
    wall_margin_reserve_m: float = 0.20
    rollout_backend: str = "smoke_only"
    wind_mode: str = "panel"


@dataclass(frozen=True)
class RolloutEvidence:
    rollout_id: str
    episode_id: str
    environment_id: str
    W_layer: str
    initial_state_vector: str
    context_feature_vector: str
    primitive_id: str
    primitive_family: str
    primitive_parameters: str
    controller_mode: str
    feedback_mode: str
    latency_case: str
    state_feedback_delay_applied: bool
    command_delay_applied: bool
    actuator_lag_applied: bool
    latency_execution_status: str
    latency_pass_label: str
    timing_model_version: str
    rollout_backend: str
    evidence_role: str
    surrogate_binding_status: str
    trajectory_integrity_status: str
    entry_check_status: str
    exit_check_status: str
    continuation_status: str
    episode_terminal_status: str
    episode_utility_label: str
    terminal_use_trainable: bool
    boundary_use_class: str
    accepted: bool
    outcome_class: str
    energy_residual_m: float
    lift_dwell_time_s: float
    minimum_wall_margin_m: float
    floor_margin_m: float
    ceiling_margin_m: float
    minimum_speed_m_s: float
    saturation_count: int
    saturation_fraction: float
    max_abs_command_norm: float
    max_abs_surface_rad: float
    exit_state_vector: str
    termination_cause: str
    failure_label: str
    claim_status: str


# =============================================================================
# 2) Public Rollout Evaluation
# =============================================================================
def simulate_primitive_rollout(
    *,
    rollout_id: str,
    episode_id: str | None = None,
    initial_state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    config: RolloutConfig | None = None,
    wind_field: object | None = None,
) -> RolloutEvidence:
    """Return one primitive rollout row using smoke-only or model-backed backend."""

    cfg = config or RolloutConfig()
    state = _initial_state_for_rollout(initial_state)
    if primitive.claim_status != "simulation_only":
        raise ValueError("rollout evidence only supports simulation_only primitives.")
    if float(primitive.finite_horizon_s) <= 0.0:
        raise ValueError("primitive finite_horizon_s must be positive.")
    if cfg.rollout_backend not in ROLLOUT_BACKENDS:
        raise ValueError("rollout_backend must be one of the retained rollout backends.")
    if cfg.rollout_backend == "smoke_only":
        return _simulate_smoke_rollout(
            rollout_id=rollout_id,
            episode_id="" if episode_id is None else str(episode_id),
            state=state,
            context=context,
            primitive=primitive,
            config=cfg,
        )
    if cfg.rollout_backend in {"model_backed_command_template", "model_backed_feedback"}:
        return _simulate_dynamics_rollout(
            rollout_id=rollout_id,
            episode_id="" if episode_id is None else str(episode_id),
            state=state,
            context=context,
            primitive=primitive,
            config=cfg,
            wind_field=wind_field,
        )
    return blocked_rollout_evidence(
        rollout_id=rollout_id,
        episode_id=episode_id,
        initial_state=state,
        context=context,
        primitive=primitive,
        config=cfg,
        failure_label="blocked_feedback_requested",
    )


def blocked_rollout_evidence(
    *,
    rollout_id: str,
    episode_id: str | None = None,
    initial_state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    config: RolloutConfig | None = None,
    failure_label: str = "surrogate_binding_blocked",
) -> RolloutEvidence:
    """Return a retained blocked row when a strict surrogate cannot be loaded."""

    cfg = config or RolloutConfig()
    state = as_state_vector(initial_state)
    margins = position_margin_m(state[:3], TRUE_SAFE_BOUNDS)
    latency_fields = _latency_field_values(
        latency_case=context.latency_case,
        accepted=False,
        state_delay=False,
        command_delay=False,
        actuator_lag=False,
        execution_status="blocked_before_simulation",
    )
    return RolloutEvidence(
        rollout_id=str(rollout_id),
        episode_id="" if episode_id is None else str(episode_id),
        environment_id=context.environment_id,
        W_layer=str(cfg.W_layer),
        initial_state_vector=_vector_json(state),
        context_feature_vector=context_feature_vector_json(context),
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        primitive_parameters=primitive_parameters_json(primitive),
        controller_mode=primitive.controller_mode,
        feedback_mode=primitive.feedback_mode,
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend="blocked_feedback",
        evidence_role=EVIDENCE_ROLE_BY_BACKEND["blocked_feedback"],
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status="blocked_before_simulation",
        entry_check_status=str(failure_label),
        exit_check_status="not_evaluated",
        continuation_status="blocked",
        episode_terminal_status="not_terminal",
        episode_utility_label="blocked",
        terminal_use_trainable=False,
        boundary_use_class="blocked",
        accepted=False,
        outcome_class="blocked",
        energy_residual_m=0.0,
        lift_dwell_time_s=0.0,
        minimum_wall_margin_m=float(margins["min_wall_margin_m"]),
        floor_margin_m=float(margins["floor_margin_m"]),
        ceiling_margin_m=float(margins["ceiling_margin_m"]),
        minimum_speed_m_s=float(np.linalg.norm(state[6:9])),
        saturation_count=0,
        saturation_fraction=0.0,
        max_abs_command_norm=0.0,
        max_abs_surface_rad=0.0,
        exit_state_vector=_vector_json(state),
        termination_cause="surrogate_binding_blocked",
        failure_label=str(failure_label),
        claim_status=primitive.claim_status,
    )


# =============================================================================
# 3) Smoke Rollout Evaluation
# =============================================================================
def _simulate_smoke_rollout(
    *,
    rollout_id: str,
    episode_id: str,
    state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    config: RolloutConfig,
) -> RolloutEvidence:
    speed_m_s = float(np.linalg.norm(state[6:9]))
    energy_residual_m = _smoke_energy_residual(context, primitive)
    lift_dwell_time_s = (
        float(primitive.finite_horizon_s)
        if context.w_wing_mean_m_s > 0.05 and context.wall_margin_m > 0.0
        else 0.0
    )
    minimum_wall_margin_m = float(
        context.wall_margin_m - _primitive_wall_reserve_m(primitive, config)
    )
    minimum_speed_m_s = float(speed_m_s + 0.20 * energy_residual_m)
    outcome_class, termination_cause, failure_label = _classify_smoke_outcome(
        context=context,
        energy_residual_m=energy_residual_m,
        minimum_wall_margin_m=minimum_wall_margin_m,
        minimum_speed_m_s=minimum_speed_m_s,
        minimum_speed_required_m_s=float(config.minimum_speed_m_s),
    )
    exit_state = state.copy()
    exit_state[0] += max(speed_m_s, 0.0) * float(primitive.finite_horizon_s) * 0.08
    exit_state[2] += energy_residual_m
    accepted = outcome_class == "accepted"
    latency_fields = _latency_field_values(
        latency_case=context.latency_case,
        accepted=accepted,
        state_delay=False,
        command_delay=False,
        actuator_lag=False,
        execution_status="interface_smoke_not_integrated",
    )
    return RolloutEvidence(
        rollout_id=str(rollout_id),
        episode_id=str(episode_id),
        environment_id=context.environment_id,
        W_layer=str(config.W_layer),
        initial_state_vector=_vector_json(state),
        context_feature_vector=context_feature_vector_json(context),
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        primitive_parameters=primitive_parameters_json(primitive),
        controller_mode=primitive.controller_mode,
        feedback_mode=primitive.feedback_mode,
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend="smoke_only",
        evidence_role=EVIDENCE_ROLE_BY_BACKEND["smoke_only"],
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status="smoke_only_not_integrated",
        entry_check_status="interface_smoke_not_evaluated",
        exit_check_status=_exit_check_status(outcome_class, failure_label),
        continuation_status=_continuation_status(outcome_class),
        episode_terminal_status=_episode_terminal_status(outcome_class),
        episode_utility_label=_episode_utility_label(
            outcome_class=outcome_class,
            energy_residual_m=energy_residual_m,
            lift_dwell_time_s=lift_dwell_time_s,
        ),
        terminal_use_trainable=_terminal_use_trainable(outcome_class, trajectory_status="smoke"),
        boundary_use_class=_boundary_use_class(outcome_class),
        accepted=accepted,
        outcome_class=outcome_class,
        energy_residual_m=energy_residual_m,
        lift_dwell_time_s=lift_dwell_time_s,
        minimum_wall_margin_m=minimum_wall_margin_m,
        floor_margin_m=float(context.floor_margin_m),
        ceiling_margin_m=float(context.ceiling_margin_m),
        minimum_speed_m_s=minimum_speed_m_s,
        saturation_count=0,
        saturation_fraction=0.0,
        max_abs_command_norm=0.0,
        max_abs_surface_rad=0.0,
        exit_state_vector=_vector_json(exit_state),
        termination_cause=termination_cause,
        failure_label=failure_label,
        claim_status=primitive.claim_status,
    )


# =============================================================================
# 4) Model-Backed Rollout Evaluation
# =============================================================================
def _simulate_dynamics_rollout(
    *,
    rollout_id: str,
    episode_id: str,
    state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    config: RolloutConfig,
    wind_field: object | None,
) -> RolloutEvidence:
    if not np.all(np.isfinite(state)):
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0),
            context=context,
            primitive=primitive,
            config=config,
            failure_label="nonfinite_initial_state",
        )
    initial_margins = position_margin_m(state[:3], TRUE_SAFE_BOUNDS)
    if float(initial_margins["floor_margin_m"]) < 0.0:
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label="initial_floor_violation",
        )
    if float(initial_margins["ceiling_margin_m"]) < 0.0:
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label="initial_ceiling_violation",
        )
    if float(initial_margins["min_wall_margin_m"]) < 0.0:
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label="physically_impossible_initial_state",
        )
    if _attitude_is_physically_impossible(state):
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label="physically_impossible_initial_state",
        )
    initial_speed_m_s = float(np.linalg.norm(state[6:9]))
    if initial_speed_m_s < float(config.minimum_speed_m_s):
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label="speed_low",
        )

    command_template_norm = _primitive_command_template_norm(primitive)
    command_template = normalised_command_to_surface_rad(command_template_norm)
    latency = latency_case_config(context.latency_case)
    mechanism_flags = latency_mechanism_flags_from_case(
        context.latency_case,
        state_feedback_delay_applied=context.latency_case in {"nominal", "conservative"},
    )
    tau_s = (
        actuator_tau_for_case(latency)
        if mechanism_flags["actuator_lag_applied"]
        else (1.0, 1.0, 1.0)
    )
    wind_mode = _wind_mode_for_rollout(context=context, config=config, wind_field=wind_field)
    aircraft = _aircraft_model()
    x = state.copy()
    min_wall_margin_m = float("inf")
    min_floor_margin_m = float("inf")
    min_ceiling_margin_m = float("inf")
    min_speed_m_s = initial_speed_m_s
    lift_dwell_time_s = 0.0
    trajectory_status = "finite_model_backed"
    termination_cause = "controlled_finish"
    failure_label = "success"
    feedback_mode = (
        "command_template_diagnostic"
        if config.rollout_backend == "model_backed_command_template"
        else primitive.feedback_mode
    )
    steps = max(1, int(np.ceil(float(primitive.finite_horizon_s) / float(config.dt_s))))
    times_s = [0.0]
    states = [x.copy()]
    if config.rollout_backend == "model_backed_feedback":
        initial_command = primitive_command_norm(
            primitive,
            PrimitiveControlContext(
                state_vector=x,
                environment_context=context,
                time_in_primitive_s=0.0,
            ),
        )
        command_norm_history = [np.asarray(initial_command.command_norm, dtype=float)]
        saturation_count = int(initial_command.saturation_applied)
        max_abs_command_norm = float(np.max(np.abs(command_norm_history[-1])))
        max_abs_surface_rad = float(np.max(np.abs(initial_command.command_rad)))
    else:
        command_norm_history = [command_template_norm.copy()]
        saturation_count = 0
        max_abs_command_norm = float(np.max(np.abs(command_template_norm)))
        max_abs_surface_rad = float(np.max(np.abs(command_template)))

    for step_index in range(steps):
        time_s = float(step_index) * float(config.dt_s)
        margins = position_margin_m(x[:3], TRUE_SAFE_BOUNDS)
        min_wall_margin_m = min(min_wall_margin_m, float(margins["min_wall_margin_m"]))
        min_floor_margin_m = min(min_floor_margin_m, float(margins["floor_margin_m"]))
        min_ceiling_margin_m = min(min_ceiling_margin_m, float(margins["ceiling_margin_m"]))
        min_speed_m_s = min(min_speed_m_s, float(np.linalg.norm(x[6:9])))
        if min_floor_margin_m < 0.0:
            termination_cause = "floor_margin_stop"
            failure_label = "floor_violation"
            break
        if min_ceiling_margin_m < 0.0:
            termination_cause = "ceiling_margin_stop"
            failure_label = "ceiling_violation"
            break
        if min_wall_margin_m < 0.0:
            termination_cause = "wall_boundary_exit_retained"
            failure_label = "wall_violation"
            break
        if min_speed_m_s < float(config.minimum_speed_m_s):
            termination_cause = "speed_floor"
            failure_label = "speed_low"
            break
        if context.w_wing_mean_m_s > 0.05:
            lift_dwell_time_s += float(config.dt_s)
        if config.rollout_backend == "model_backed_feedback":
            if mechanism_flags["state_feedback_delay_applied"]:
                x_control = delayed_state_sample(
                    np.asarray(times_s, dtype=float),
                    np.asarray(states, dtype=float),
                    time_s - float(latency.state_feedback_delay_s),
                )
            else:
                x_control = x
            control_command = primitive_command_norm(
                primitive,
                PrimitiveControlContext(
                    state_vector=x_control,
                    environment_context=context,
                    time_in_primitive_s=time_s,
                ),
            )
            desired_command_norm = np.asarray(control_command.command_norm, dtype=float)
            if time_s > times_s[-1]:
                times_s.append(time_s)
                states.append(x.copy())
                command_norm_history.append(desired_command_norm.copy())
            else:
                command_norm_history[-1] = desired_command_norm.copy()
            if mechanism_flags["command_delay_applied"]:
                applied_norm = latency_adjusted_command_sample(
                    np.asarray(times_s, dtype=float),
                    np.asarray(command_norm_history, dtype=float),
                    time_s,
                    latency,
                )
            else:
                applied_norm = desired_command_norm
            command = normalised_command_to_surface_rad(applied_norm)
            feedback_mode = control_command.feedback_mode
            saturation_count += int(control_command.saturation_applied)
            max_abs_command_norm = max(max_abs_command_norm, float(np.max(np.abs(applied_norm))))
            max_abs_surface_rad = max(max_abs_surface_rad, float(np.max(np.abs(command))))
        else:
            command = command_template
            if time_s > times_s[-1]:
                times_s.append(time_s)
                states.append(x.copy())
                command_norm_history.append(command_template_norm.copy())
        if not mechanism_flags["actuator_lag_applied"]:
            x[12:15] = command
        x = _rk4_step(
            x=x,
            command=command,
            aircraft=aircraft,
            wind_field=wind_field,
            wind_mode=wind_mode,
            actuator_tau_s=tau_s,
            dt_s=float(config.dt_s),
        )
        if not np.all(np.isfinite(x)):
            trajectory_status = "nonfinite_model_backed"
            termination_cause = "nonfinite_trajectory"
            failure_label = "nonfinite_trajectory"
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            break
        next_time_s = time_s + float(config.dt_s)
        if next_time_s > times_s[-1]:
            times_s.append(next_time_s)
            states.append(x.copy())
            command_norm_history.append(command_norm_history[-1].copy())

    margins = position_margin_m(x[:3], TRUE_SAFE_BOUNDS)
    min_wall_margin_m = min(min_wall_margin_m, float(margins["min_wall_margin_m"]))
    min_floor_margin_m = min(min_floor_margin_m, float(margins["floor_margin_m"]))
    min_ceiling_margin_m = min(min_ceiling_margin_m, float(margins["ceiling_margin_m"]))
    min_speed_m_s = min(min_speed_m_s, float(np.linalg.norm(x[6:9])))
    energy_residual_m = float(x[STATE_INDEX["z_w"]] - state[STATE_INDEX["z_w"]])
    outcome_class, termination_cause, failure_label = _classify_model_backed_outcome(
        energy_residual_m=energy_residual_m,
        minimum_wall_margin_m=min_wall_margin_m,
        floor_margin_m=min_floor_margin_m,
        ceiling_margin_m=min_ceiling_margin_m,
        minimum_speed_m_s=min_speed_m_s,
        minimum_speed_required_m_s=float(config.minimum_speed_m_s),
        trajectory_status=trajectory_status,
        current_termination=termination_cause,
        current_failure=failure_label,
    )
    accepted = outcome_class == "accepted"
    latency_fields = _latency_field_values(
        latency_case=context.latency_case,
        accepted=accepted,
        state_delay=mechanism_flags["state_feedback_delay_applied"],
        command_delay=mechanism_flags["command_delay_applied"],
        actuator_lag=mechanism_flags["actuator_lag_applied"],
        execution_status=latency_execution_status(
            latency_case=context.latency_case,
            state_feedback_delay_applied=mechanism_flags["state_feedback_delay_applied"],
            command_delay_applied=mechanism_flags["command_delay_applied"],
            actuator_lag_applied=mechanism_flags["actuator_lag_applied"],
        ),
    )
    return RolloutEvidence(
        rollout_id=str(rollout_id),
        episode_id=str(episode_id),
        environment_id=context.environment_id,
        W_layer=str(config.W_layer),
        initial_state_vector=_vector_json(state),
        context_feature_vector=context_feature_vector_json(context),
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        primitive_parameters=primitive_parameters_json(primitive),
        controller_mode=_controller_mode_for_backend(config.rollout_backend, primitive),
        feedback_mode=feedback_mode,
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend=config.rollout_backend,
        evidence_role=_evidence_role_for_backend(config.rollout_backend),
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status=trajectory_status,
        entry_check_status="passed",
        exit_check_status=_exit_check_status(outcome_class, failure_label),
        continuation_status=_continuation_status(outcome_class),
        episode_terminal_status=_episode_terminal_status(outcome_class),
        episode_utility_label=_episode_utility_label(
            outcome_class=outcome_class,
            energy_residual_m=energy_residual_m,
            lift_dwell_time_s=lift_dwell_time_s,
        ),
        terminal_use_trainable=_terminal_use_trainable(
            outcome_class,
            trajectory_status=trajectory_status,
        ),
        boundary_use_class=_boundary_use_class(outcome_class),
        accepted=accepted,
        outcome_class=outcome_class,
        energy_residual_m=energy_residual_m,
        lift_dwell_time_s=float(lift_dwell_time_s),
        minimum_wall_margin_m=float(min_wall_margin_m),
        floor_margin_m=float(min_floor_margin_m),
        ceiling_margin_m=float(min_ceiling_margin_m),
        minimum_speed_m_s=float(min_speed_m_s),
        saturation_count=int(saturation_count),
        saturation_fraction=float(saturation_count / max(1, steps)),
        max_abs_command_norm=float(max_abs_command_norm),
        max_abs_surface_rad=float(max_abs_surface_rad),
        exit_state_vector=_vector_json(x),
        termination_cause=termination_cause,
        failure_label=failure_label,
        claim_status=primitive.claim_status,
    )


def _blocked_from_state(
    *,
    rollout_id: str,
    episode_id: str,
    state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    config: RolloutConfig,
    failure_label: str,
) -> RolloutEvidence:
    margins = position_margin_m(state[:3], TRUE_SAFE_BOUNDS)
    latency_fields = _latency_field_values(
        latency_case=context.latency_case,
        accepted=False,
        state_delay=False,
        command_delay=False,
        actuator_lag=False,
        execution_status="blocked_before_simulation",
    )
    return RolloutEvidence(
        rollout_id=str(rollout_id),
        episode_id=str(episode_id),
        environment_id=context.environment_id,
        W_layer=str(config.W_layer),
        initial_state_vector=_vector_json(state),
        context_feature_vector=context_feature_vector_json(context),
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        primitive_parameters=primitive_parameters_json(primitive),
        controller_mode=_controller_mode_for_backend(config.rollout_backend, primitive),
        feedback_mode=(
            "blocked_feedback"
            if config.rollout_backend == "model_backed_feedback"
            else primitive.feedback_mode
        ),
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend=config.rollout_backend,
        evidence_role=_evidence_role_for_backend(config.rollout_backend),
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status="blocked_before_simulation",
        entry_check_status=str(failure_label),
        exit_check_status="not_evaluated",
        continuation_status="blocked",
        episode_terminal_status="not_terminal",
        episode_utility_label="blocked",
        terminal_use_trainable=False,
        boundary_use_class="blocked",
        accepted=False,
        outcome_class="blocked",
        energy_residual_m=0.0,
        lift_dwell_time_s=0.0,
        minimum_wall_margin_m=float(margins["min_wall_margin_m"]),
        floor_margin_m=float(margins["floor_margin_m"]),
        ceiling_margin_m=float(margins["ceiling_margin_m"]),
        minimum_speed_m_s=float(np.linalg.norm(state[6:9])),
        saturation_count=0,
        saturation_fraction=0.0,
        max_abs_command_norm=0.0,
        max_abs_surface_rad=0.0,
        exit_state_vector=_vector_json(state),
        termination_cause=str(failure_label),
        failure_label=str(failure_label),
        claim_status=primitive.claim_status,
    )


@lru_cache(maxsize=1)
def _aircraft_model():
    return adapt_glider(build_nausicaa_glider())


def _rk4_step(
    *,
    x: np.ndarray,
    command: np.ndarray,
    aircraft,
    wind_field: object | None,
    wind_mode: str,
    actuator_tau_s: tuple[float, float, float],
    dt_s: float,
) -> np.ndarray:
    k1 = state_derivative(
        x,
        command,
        aircraft,
        wind_model=wind_field,
        wind_mode=wind_mode,
        actuator_tau_s=actuator_tau_s,
    )
    k2 = state_derivative(
        x + 0.5 * dt_s * k1,
        command,
        aircraft,
        wind_model=wind_field,
        wind_mode=wind_mode,
        actuator_tau_s=actuator_tau_s,
    )
    k3 = state_derivative(
        x + 0.5 * dt_s * k2,
        command,
        aircraft,
        wind_model=wind_field,
        wind_mode=wind_mode,
        actuator_tau_s=actuator_tau_s,
    )
    k4 = state_derivative(
        x + dt_s * k3,
        command,
        aircraft,
        wind_model=wind_field,
        wind_mode=wind_mode,
        actuator_tau_s=actuator_tau_s,
    )
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _primitive_command_template_norm(primitive: PrimitiveDefinition) -> np.ndarray:
    command_by_id = {
        "glide": (0.0, -0.12, 0.0),
        "recovery": (0.0, 0.08, 0.0),
        "lift_entry": (0.05, 0.03, 0.0),
        "lift_dwell_arc": (0.22, 0.02, 0.0),
        "mild_turn_left": (-0.22, 0.0, -0.04),
        "mild_turn_right": (0.22, 0.0, 0.04),
        "energy_retaining_bank": (0.16, -0.03, 0.0),
        "safe_exit_or_recovery_handoff": (0.0, 0.10, 0.0),
    }
    return np.asarray(command_by_id.get(primitive.primitive_id, (0.0, 0.0, 0.0)), dtype=float)


def _initial_state_for_rollout(initial_state: np.ndarray) -> np.ndarray:
    state = np.asarray(initial_state, dtype=float)
    if state.size != STATE_SIZE:
        raise ValueError(f"state vector must contain {STATE_SIZE} values.")
    return state.reshape(STATE_SIZE).copy()


def _attitude_is_physically_impossible(state: np.ndarray) -> bool:
    phi_rad = abs(float(state[STATE_INDEX["phi"]]))
    theta_rad = abs(float(state[STATE_INDEX["theta"]]))
    return phi_rad > np.deg2rad(135.0) or theta_rad > np.deg2rad(85.0)


def _evidence_role_for_backend(backend: str) -> str:
    return EVIDENCE_ROLE_BY_BACKEND.get(str(backend), "blocked_feedback")


def _controller_mode_for_backend(
    backend: str,
    primitive: PrimitiveDefinition,
) -> str:
    if backend == "model_backed_feedback":
        return "bounded_local_feedback"
    if backend == "model_backed_command_template":
        return "command_template_diagnostic"
    return primitive.controller_mode


def _exit_check_status(outcome_class: str, failure_label: str) -> str:
    if outcome_class == "boundary_terminal":
        return "boundary_terminal_retained"
    if outcome_class in {"accepted", "weak"}:
        return "continuation_exit_valid"
    if outcome_class == "blocked":
        return "blocked"
    if failure_label in {
        "floor_violation",
        "ceiling_violation",
        "nonfinite_trajectory",
        "corrupt_integration",
    }:
        return "hard_failure"
    return "failed_or_rejected"


def _continuation_status(outcome_class: str) -> str:
    if outcome_class == "accepted":
        return "continuation_success"
    if outcome_class == "weak":
        return "continuation_weak"
    if outcome_class == "boundary_terminal":
        return "not_continuation_valid"
    if outcome_class == "blocked":
        return "blocked"
    return "continuation_failed"


def _episode_terminal_status(outcome_class: str) -> str:
    if outcome_class == "boundary_terminal":
        return "boundary_terminal"
    return "not_terminal"


def _episode_utility_label(
    *,
    outcome_class: str,
    energy_residual_m: float,
    lift_dwell_time_s: float,
) -> str:
    if outcome_class == "boundary_terminal":
        if energy_residual_m >= 0.0 or lift_dwell_time_s >= 0.20:
            return "terminal_useful"
        return "terminal_low_utility"
    if outcome_class in {"accepted", "weak"}:
        return "continuation_useful"
    if outcome_class == "blocked":
        return "blocked"
    return "not_useful"


def _terminal_use_trainable(outcome_class: str, *, trajectory_status: str) -> bool:
    return outcome_class == "boundary_terminal" and trajectory_status == "finite_model_backed"


def _boundary_use_class(outcome_class: str) -> str:
    if outcome_class == "boundary_terminal":
        return "episode_terminal_useful_or_trainable"
    if outcome_class in {"accepted", "weak"}:
        return "continuation_valid"
    if outcome_class == "blocked":
        return "blocked"
    return "hard_failure_or_rejected"


def _latency_field_values(
    *,
    latency_case: str,
    accepted: bool,
    state_delay: bool,
    command_delay: bool,
    actuator_lag: bool,
    execution_status: str,
) -> dict[str, object]:
    config = latency_case_config(latency_case)
    return {
        "state_feedback_delay_applied": bool(state_delay),
        "command_delay_applied": bool(command_delay),
        "actuator_lag_applied": bool(actuator_lag),
        "latency_execution_status": str(execution_status),
        "latency_pass_label": latency_pass_label_for_single_run(latency_case, accepted),
        "timing_model_version": str(config.timing_model_version),
    }


def _wind_mode_for_rollout(
    *,
    context: EnvironmentContext,
    config: RolloutConfig,
    wind_field: object | None,
) -> str:
    if wind_field is None:
        return "none"
    mode = str(context.wind_mode or config.wind_mode)
    return mode if mode in {"cg", "panel"} else "panel"


def _classify_model_backed_outcome(
    *,
    energy_residual_m: float,
    minimum_wall_margin_m: float,
    floor_margin_m: float,
    ceiling_margin_m: float,
    minimum_speed_m_s: float,
    minimum_speed_required_m_s: float,
    trajectory_status: str,
    current_termination: str,
    current_failure: str,
) -> tuple[str, str, str]:
    if trajectory_status != "finite_model_backed":
        return "blocked", current_termination, current_failure
    if floor_margin_m < 0.0:
        return "rejected", "floor_margin_stop", "floor_violation"
    if ceiling_margin_m < 0.0:
        return "rejected", "ceiling_margin_stop", "ceiling_violation"
    if minimum_wall_margin_m < 0.0:
        return "boundary_terminal", "wall_boundary_exit_retained", "xy_boundary_terminal"
    if minimum_speed_m_s < minimum_speed_required_m_s:
        return "failed", "speed_floor", "speed_low"
    if energy_residual_m >= 0.02:
        return "accepted", "controlled_finish", "success"
    if energy_residual_m >= -0.08:
        return "weak", "weak_energy_result", "model_boundary_only"
    return "failed", "terminal_recovery_limited", "terminal_recovery_limited"


def _smoke_energy_residual(
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
) -> float:
    family_bias = {
        "glide": -0.05,
        "recovery": 0.02,
        "lift_entry": 0.04,
        "lift_dwell": 0.06,
        "mild_turn": -0.02,
        "energy_retaining_bank": 0.01,
        "safe_exit": -0.01,
    }.get(primitive.primitive_family, -0.03)
    value = (
        0.35 * float(context.w_wing_mean_m_s)
        + 0.20 * float(context.lift_score)
        - 0.08
        + family_bias
    ) * float(primitive.finite_horizon_s)
    return float(value)


def _primitive_wall_reserve_m(
    primitive: PrimitiveDefinition,
    config: RolloutConfig,
) -> float:
    reserve = float(config.wall_margin_reserve_m)
    if primitive.primitive_family in {"lift_dwell", "mild_turn", "energy_retaining_bank"}:
        reserve += 0.10
    return reserve


def _classify_smoke_outcome(
    *,
    context: EnvironmentContext,
    energy_residual_m: float,
    minimum_wall_margin_m: float,
    minimum_speed_m_s: float,
    minimum_speed_required_m_s: float,
) -> tuple[str, str, str]:
    if context.floor_margin_m < 0.0 or context.ceiling_margin_m < 0.0:
        return "rejected", "safety_volume_exit", "true_safety_violation"
    if minimum_wall_margin_m < 0.0:
        return "boundary_terminal", "wall_boundary_exit_retained", "xy_boundary_terminal"
    if minimum_speed_m_s < minimum_speed_required_m_s:
        return "blocked", "low_speed", "speed_low"
    if energy_residual_m >= 0.05:
        return "accepted", "controlled_finish", "success"
    if energy_residual_m >= -0.03:
        return "weak", "weak_energy_result", "model_boundary_only"
    return "failed", "terminal_recovery_limited", "terminal_recovery_limited"


# =============================================================================
# 5) Serialisation Helpers
# =============================================================================
def rollout_evidence_row(evidence: RolloutEvidence) -> dict[str, object]:
    """Return one CSV-ready rollout evidence row."""

    row = asdict(evidence)
    return {name: row[name] for name in ROLLOUT_EVIDENCE_COLUMNS}


def rollout_with_context_row(evidence: RolloutEvidence, context: EnvironmentContext) -> dict[str, object]:
    """Return evidence plus expanded context fields for archive partitions."""

    row = rollout_evidence_row(evidence)
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    return row


def _vector_json(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))
