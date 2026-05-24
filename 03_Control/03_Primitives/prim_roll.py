from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
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
from implementation_instance import (
    ImplementationInstance,
    adjusted_actuator_tau_s,
    apply_surface_implementation,
    implementation_instance_for_layer,
)
from plant_instance import PlantInstance, apply_plant_instance_to_aircraft, plant_instance_for_layer
from prim_cat import PrimitiveDefinition, primitive_parameters_json
from lqr_controller import (
    LQR_SYNTHESIS_SOLVED,
    LQRController,
    controller_is_executable_lqr,
    lqr_controller_for_primitive_id,
    lqr_rollout_metadata,
)
from prim_ctrl import PrimitiveControlContext, primitive_lqr_command
from primitive_evidence_schema import (
    BOUNDARY_USE_CLASSES,
    OUTCOME_CLASSES,
    evidence_use_labels,
    terminal_evidence_is_useful,
)
from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE
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
ROLLOUT_BACKENDS = (
    "smoke_only",
    "model_backed_lqr",
    "blocked_lqr",
)
EVIDENCE_ROLE_BY_BACKEND = {
    "smoke_only": "interface_smoke",
    "model_backed_lqr": "lqr_rollout_candidate",
    "blocked_lqr": "blocked_lqr_synthesis",
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
    "controller_family",
    "controller_id",
    "lqr_reference_id",
    "linearisation_id",
    "linearisation_source",
    "reduced_order_lqr",
    "lqr_state_mask_json",
    "zero_position_gain_expansion_status",
    "lqr_Q_weights_json",
    "lqr_R_weights_json",
    "lqr_gain_checksum",
    "lqr_synthesis_status",
    "lqr_blocked_reason",
    "lqr_closed_loop_eigenvalue_summary",
    "care_residual_norm",
    "sampled_data_check_status",
    "sampled_data_spectral_radius",
    "command_clip_check_status",
    "saturation_summary",
    "latency_actuator_survival_status",
    "tuning_stage",
    "controller_claim_status",
    "controller_selection_status",
    "controller_executable",
    "controller_evidence_status",
    "candidate_index",
    "candidate_weight_label",
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
    "entry_rejection_class",
    "exit_check_status",
    "continuation_valid",
    "episode_terminal_useful",
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
    "archive_evidence_status",
    "evidence_eligibility_reason",
    "claim_status",
)
ROLLOUT_EVIDENCE_ALIAS_VALUES = (
    "not_continuation_valid",
    "terminal_useful",
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
    controller_family: str
    controller_id: str
    lqr_reference_id: str
    linearisation_id: str
    linearisation_source: str
    reduced_order_lqr: bool
    lqr_state_mask_json: str
    zero_position_gain_expansion_status: str
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    lqr_synthesis_status: str
    lqr_blocked_reason: str
    lqr_closed_loop_eigenvalue_summary: str
    care_residual_norm: float
    sampled_data_check_status: str
    sampled_data_spectral_radius: float
    command_clip_check_status: str
    saturation_summary: str
    latency_actuator_survival_status: str
    tuning_stage: str
    controller_claim_status: str
    controller_selection_status: str
    controller_executable: bool
    controller_evidence_status: str
    candidate_index: int | str
    candidate_weight_label: str
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
    entry_rejection_class: str
    exit_check_status: str
    continuation_valid: bool
    episode_terminal_useful: bool
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
    archive_evidence_status: str
    evidence_eligibility_reason: str
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
    implementation_instance: ImplementationInstance | None = None,
    plant_instance: PlantInstance | None = None,
    controller: LQRController | None = None,
    controller_selection_status: str | None = None,
    candidate_index: int | str = "",
    candidate_weight_label: str = "",
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
            controller=controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
        )
    if cfg.rollout_backend == "model_backed_lqr":
        return _simulate_dynamics_rollout(
            rollout_id=rollout_id,
            episode_id="" if episode_id is None else str(episode_id),
            state=state,
            context=context,
            primitive=primitive,
            config=cfg,
            wind_field=wind_field,
            implementation_instance=implementation_instance,
            plant_instance=plant_instance,
            controller=controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
        )
    return blocked_rollout_evidence(
        rollout_id=rollout_id,
        episode_id=episode_id,
        initial_state=state,
        context=context,
        primitive=primitive,
        config=cfg,
        failure_label="blocked_lqr_requested",
        controller=controller,
        controller_selection_status=controller_selection_status,
        candidate_index=candidate_index,
        candidate_weight_label=candidate_weight_label,
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
    controller: LQRController | None = None,
    controller_selection_status: str | None = None,
    candidate_index: int | str = "",
    candidate_weight_label: str = "",
    termination_cause: str | None = None,
) -> RolloutEvidence:
    """Return a retained blocked row when a strict surrogate cannot be loaded."""

    cfg = config or RolloutConfig()
    state = as_state_vector(initial_state)
    resolved_termination_cause = _blocked_termination_cause(failure_label, termination_cause)
    margins = position_margin_m(state[:3], TRUE_SAFE_BOUNDS)
    latency_fields = _latency_field_values(
        latency_case=context.latency_case,
        accepted=False,
        state_delay=False,
        command_delay=False,
        actuator_lag=False,
        execution_status="blocked_before_simulation",
    )
    resolved_controller = controller or lqr_controller_for_primitive_id(primitive.primitive_id)
    labels = evidence_use_labels(
        outcome_class="blocked",
        failure_label=failure_label,
        termination_cause=resolved_termination_cause,
        energy_residual_m=0.0,
        lift_dwell_time_s=0.0,
        trajectory_status="blocked_before_simulation",
    )
    selection_status = _controller_selection_status(
        controller=controller,
        explicit_status=controller_selection_status,
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
        **_lqr_metadata_for_evidence(
            controller=controller,
            fallback_controller=resolved_controller,
            primitive=primitive,
            controller_selection_status=selection_status,
        ),
        controller_selection_status=selection_status,
        controller_executable=_controller_executable(controller, selection_status),
        controller_evidence_status=_controller_evidence_status(controller, selection_status),
        candidate_index=candidate_index,
        candidate_weight_label=str(candidate_weight_label),
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend="blocked_lqr",
        evidence_role=EVIDENCE_ROLE_BY_BACKEND["blocked_lqr"],
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status="blocked_before_simulation",
        entry_check_status=str(failure_label),
        entry_rejection_class=_entry_rejection_class(failure_label),
        exit_check_status=labels.exit_check_status,
        continuation_valid=labels.continuation_valid,
        episode_terminal_useful=labels.episode_terminal_useful,
        continuation_status=labels.continuation_status,
        episode_terminal_status=labels.episode_terminal_status,
        episode_utility_label=labels.episode_utility_label,
        terminal_use_trainable=labels.terminal_use_trainable,
        boundary_use_class=labels.boundary_use_class,
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
        termination_cause=resolved_termination_cause,
        failure_label=str(failure_label),
        archive_evidence_status="blocked",
        evidence_eligibility_reason=str(failure_label),
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
    controller: LQRController | None,
    controller_selection_status: str | None,
    candidate_index: int | str,
    candidate_weight_label: str,
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
        lift_dwell_time_s=lift_dwell_time_s,
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
    resolved_controller = controller or lqr_controller_for_primitive_id(primitive.primitive_id)
    labels = evidence_use_labels(
        outcome_class=outcome_class,
        failure_label=failure_label,
        termination_cause=termination_cause,
        energy_residual_m=energy_residual_m,
        lift_dwell_time_s=lift_dwell_time_s,
        trajectory_status="smoke_only_not_integrated",
    )
    selection_status = _controller_selection_status(
        controller=controller,
        explicit_status=controller_selection_status,
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
        **_lqr_metadata_for_evidence(
            controller=controller,
            fallback_controller=resolved_controller,
            primitive=primitive,
            controller_selection_status=selection_status,
        ),
        controller_selection_status=selection_status,
        controller_executable=_controller_executable(controller, selection_status),
        controller_evidence_status=_controller_evidence_status(controller, selection_status),
        candidate_index=candidate_index,
        candidate_weight_label=str(candidate_weight_label),
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend="smoke_only",
        evidence_role=EVIDENCE_ROLE_BY_BACKEND["smoke_only"],
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status="smoke_only_not_integrated",
        entry_check_status="interface_smoke_not_evaluated",
        entry_rejection_class=(
            _entry_rejection_class(failure_label)
            if outcome_class == "blocked"
            else "not_rejected"
        ),
        exit_check_status=labels.exit_check_status,
        continuation_valid=labels.continuation_valid,
        episode_terminal_useful=labels.episode_terminal_useful,
        continuation_status=labels.continuation_status,
        episode_terminal_status=labels.episode_terminal_status,
        episode_utility_label=labels.episode_utility_label,
        terminal_use_trainable=labels.terminal_use_trainable,
        boundary_use_class=labels.boundary_use_class,
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
        archive_evidence_status="smoke_incomplete",
        evidence_eligibility_reason="debug_smoke_incomplete",
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
    implementation_instance: ImplementationInstance | None = None,
    plant_instance: PlantInstance | None = None,
    controller: LQRController | None = None,
    controller_selection_status: str | None = None,
    candidate_index: int | str = "",
    candidate_weight_label: str = "",
) -> RolloutEvidence:
    if controller is None:
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label="missing_explicit_lqr_controller",
            controller=None,
            controller_selection_status=controller_selection_status
            or "missing_explicit_lqr_controller",
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
        )
    resolved_controller = controller
    if not np.all(np.isfinite(state)):
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0),
            context=context,
            primitive=primitive,
            config=config,
            failure_label="nonfinite_initial_state",
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
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
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
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
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
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
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
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
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
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
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
        )

    controller_ok, controller_reason = controller_is_executable_lqr(resolved_controller)
    if not controller_ok:
        return _blocked_from_state(
            rollout_id=rollout_id,
            episode_id=episode_id,
            state=state,
            context=context,
            primitive=primitive,
            config=config,
            failure_label=controller_reason,
            controller=resolved_controller,
            controller_selection_status=controller_selection_status,
            candidate_index=candidate_index,
            candidate_weight_label=candidate_weight_label,
        )
    implementation = implementation_instance or implementation_instance_for_layer(
        config.W_layer,
        0,
        latency_case=context.latency_case,
    )
    latency = _latency_for_implementation(context.latency_case, implementation)
    mechanism_flags = latency_mechanism_flags_from_case(
        context.latency_case,
        state_feedback_delay_applied=context.latency_case in {"nominal", "conservative"},
    )
    base_tau_s = (
        actuator_tau_for_case(latency)
        if mechanism_flags["actuator_lag_applied"]
        else (1.0, 1.0, 1.0)
    )
    tau_s = adjusted_actuator_tau_s(base_tau_s, implementation)
    wind_mode = _wind_mode_for_rollout(context=context, config=config, wind_field=wind_field)
    base_aircraft = _aircraft_model()
    plant = plant_instance or plant_instance_for_layer(
        config.W_layer,
        0,
        baseline_mass_kg=float(base_aircraft.mass_kg),
    )
    aircraft = apply_plant_instance_to_aircraft(base_aircraft, plant)
    x = state.copy()
    min_wall_margin_m = float("inf")
    min_floor_margin_m = float("inf")
    min_ceiling_margin_m = float("inf")
    min_speed_m_s = initial_speed_m_s
    lift_dwell_time_s = 0.0
    trajectory_status = "finite_model_backed"
    termination_cause = "controlled_finish"
    failure_label = "success"
    feedback_mode = resolved_controller.feedback_mode
    steps = max(1, int(np.ceil(float(primitive.finite_horizon_s) / float(config.dt_s))))
    times_s = [0.0]
    states = [x.copy()]
    command_delay_s = float(latency.command_onset_delay_s + latency.command_transport_delay_s)
    if config.rollout_backend == "model_backed_lqr":
        initial_command = primitive_lqr_command(
            primitive,
            PrimitiveControlContext(
                state_vector=x,
                environment_context=context,
                time_in_primitive_s=0.0,
            ),
            resolved_controller,
        )
        if mechanism_flags["command_delay_applied"]:
            command_times_s = [-(command_delay_s + 1e-9)]
            command_norm_history = [np.zeros(3, dtype=float)]
        else:
            command_times_s = [0.0]
            command_norm_history = [np.asarray(initial_command.command_norm, dtype=float)]
        saturation_count = int(initial_command.saturation_applied)
        max_abs_command_norm = float(np.max(np.abs(initial_command.command_norm)))
        max_abs_surface_rad = float(np.max(np.abs(initial_command.command_rad)))
    else:
        raise ValueError("model-backed rollout requires rollout_backend='model_backed_lqr'.")

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
        if config.rollout_backend == "model_backed_lqr":
            if mechanism_flags["state_feedback_delay_applied"]:
                x_control = delayed_state_sample(
                    np.asarray(times_s, dtype=float),
                    np.asarray(states, dtype=float),
                    time_s - float(latency.state_feedback_delay_s),
                )
            else:
                x_control = x
            control_command = primitive_lqr_command(
                primitive,
                PrimitiveControlContext(
                    state_vector=x_control,
                    environment_context=context,
                    time_in_primitive_s=time_s,
                ),
                resolved_controller,
            )
            desired_command_norm = np.asarray(control_command.command_norm, dtype=float)
            if time_s > command_times_s[-1]:
                command_times_s.append(time_s)
                command_norm_history.append(desired_command_norm.copy())
            else:
                command_norm_history[-1] = desired_command_norm.copy()
            if mechanism_flags["command_delay_applied"]:
                applied_norm = latency_adjusted_command_sample(
                    np.asarray(command_times_s, dtype=float),
                    np.asarray(command_norm_history, dtype=float),
                    time_s,
                    latency,
                )
            else:
                applied_norm = desired_command_norm
            command = apply_surface_implementation(
                normalised_command_to_surface_rad(applied_norm),
                implementation,
            )
            feedback_mode = control_command.feedback_mode
            saturation_count += int(control_command.saturation_applied)
            max_abs_command_norm = max(max_abs_command_norm, float(np.max(np.abs(applied_norm))))
            max_abs_surface_rad = max(max_abs_surface_rad, float(np.max(np.abs(command))))
        else:
            raise ValueError("model-backed rollout requires rollout_backend='model_backed_lqr'.")
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

    margins = position_margin_m(x[:3], TRUE_SAFE_BOUNDS)
    min_wall_margin_m = min(min_wall_margin_m, float(margins["min_wall_margin_m"]))
    min_floor_margin_m = min(min_floor_margin_m, float(margins["floor_margin_m"]))
    min_ceiling_margin_m = min(min_ceiling_margin_m, float(margins["ceiling_margin_m"]))
    min_speed_m_s = min(min_speed_m_s, float(np.linalg.norm(x[6:9])))
    energy_residual_m = float(x[STATE_INDEX["z_w"]] - state[STATE_INDEX["z_w"]])
    outcome_class, termination_cause, failure_label = _classify_model_backed_outcome(
        energy_residual_m=energy_residual_m,
        lift_dwell_time_s=lift_dwell_time_s,
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
    labels = evidence_use_labels(
        outcome_class=outcome_class,
        failure_label=failure_label,
        termination_cause=termination_cause,
        energy_residual_m=energy_residual_m,
        lift_dwell_time_s=lift_dwell_time_s,
        trajectory_status=trajectory_status,
    )
    selection_status = _controller_selection_status(
        controller=controller,
        explicit_status=controller_selection_status,
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
        **_lqr_metadata_for_evidence(
            controller=controller,
            fallback_controller=resolved_controller,
            primitive=primitive,
            controller_selection_status=selection_status,
        ),
        controller_selection_status=selection_status,
        controller_executable=_controller_executable(controller, selection_status),
        controller_evidence_status=_controller_evidence_status(controller, selection_status),
        candidate_index=candidate_index,
        candidate_weight_label=str(candidate_weight_label),
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend=config.rollout_backend,
        evidence_role=_evidence_role_for_backend(config.rollout_backend),
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status=trajectory_status,
        entry_check_status="passed",
        entry_rejection_class="not_rejected",
        exit_check_status=labels.exit_check_status,
        continuation_valid=labels.continuation_valid,
        episode_terminal_useful=labels.episode_terminal_useful,
        continuation_status=labels.continuation_status,
        episode_terminal_status=labels.episode_terminal_status,
        episode_utility_label=labels.episode_utility_label,
        terminal_use_trainable=labels.terminal_use_trainable,
        boundary_use_class=labels.boundary_use_class,
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
        archive_evidence_status="blocked" if outcome_class == "blocked" else "smoke_incomplete",
        evidence_eligibility_reason=(
            "blocked_rollout" if outcome_class == "blocked" else "debug_registry_status_not_assessed"
        ),
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
    controller: LQRController | None = None,
    controller_selection_status: str | None = None,
    candidate_index: int | str = "",
    candidate_weight_label: str = "",
    termination_cause: str | None = None,
) -> RolloutEvidence:
    margins = position_margin_m(state[:3], TRUE_SAFE_BOUNDS)
    resolved_termination_cause = _blocked_termination_cause(failure_label, termination_cause)
    latency_fields = _latency_field_values(
        latency_case=context.latency_case,
        accepted=False,
        state_delay=False,
        command_delay=False,
        actuator_lag=False,
        execution_status="blocked_before_simulation",
    )
    resolved_controller = controller or lqr_controller_for_primitive_id(primitive.primitive_id)
    labels = evidence_use_labels(
        outcome_class="blocked",
        failure_label=failure_label,
        termination_cause=resolved_termination_cause,
        energy_residual_m=0.0,
        lift_dwell_time_s=0.0,
        trajectory_status="blocked_before_simulation",
    )
    selection_status = _controller_selection_status(
        controller=controller,
        explicit_status=controller_selection_status,
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
            "blocked_lqr"
            if config.rollout_backend == "model_backed_lqr"
            else primitive.feedback_mode
        ),
        **_lqr_metadata_for_evidence(
            controller=controller,
            fallback_controller=resolved_controller,
            primitive=primitive,
            controller_selection_status=selection_status,
        ),
        controller_selection_status=selection_status,
        controller_executable=_controller_executable(controller, selection_status),
        controller_evidence_status=_controller_evidence_status(controller, selection_status),
        candidate_index=candidate_index,
        candidate_weight_label=str(candidate_weight_label),
        latency_case=context.latency_case,
        **latency_fields,
        rollout_backend=config.rollout_backend,
        evidence_role=_evidence_role_for_backend(config.rollout_backend),
        surrogate_binding_status=context.surrogate_binding_status,
        trajectory_integrity_status="blocked_before_simulation",
        entry_check_status=str(failure_label),
        entry_rejection_class=_entry_rejection_class(failure_label),
        exit_check_status=labels.exit_check_status,
        continuation_valid=labels.continuation_valid,
        episode_terminal_useful=labels.episode_terminal_useful,
        continuation_status=labels.continuation_status,
        episode_terminal_status=labels.episode_terminal_status,
        episode_utility_label=labels.episode_utility_label,
        terminal_use_trainable=labels.terminal_use_trainable,
        boundary_use_class=labels.boundary_use_class,
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
        termination_cause=resolved_termination_cause,
        failure_label=str(failure_label),
        archive_evidence_status="blocked",
        evidence_eligibility_reason=str(failure_label),
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
    return EVIDENCE_ROLE_BY_BACKEND.get(str(backend), "blocked_lqr_synthesis")


def _controller_mode_for_backend(
    backend: str,
    primitive: PrimitiveDefinition,
) -> str:
    if backend == "model_backed_lqr":
        return "lqr_local_feedback"
    return primitive.controller_mode


def _lqr_metadata_for_primitive(primitive: PrimitiveDefinition) -> dict[str, object]:
    controller = lqr_controller_for_primitive_id(primitive.primitive_id)
    return _lqr_metadata_for_controller(controller)


def _lqr_metadata_for_controller(controller: LQRController) -> dict[str, object]:
    return lqr_rollout_metadata(controller)


def _lqr_metadata_for_evidence(
    *,
    controller: LQRController | None,
    fallback_controller: LQRController,
    primitive: PrimitiveDefinition,
    controller_selection_status: str,
) -> dict[str, object]:
    if controller is None and _controller_selection_is_missing(controller_selection_status):
        metadata = lqr_rollout_metadata(fallback_controller)
        metadata.update(
            {
                "controller_id": f"blocked_missing_explicit_lqr_controller_{primitive.primitive_id}",
                "lqr_reference_id": "",
                "linearisation_id": "",
                "linearisation_source": "",
                "lqr_Q_weights_json": "",
                "lqr_R_weights_json": "",
                "lqr_gain_checksum": "",
                "lqr_synthesis_status": str(controller_selection_status),
                "lqr_blocked_reason": str(controller_selection_status),
                "lqr_closed_loop_eigenvalue_summary": "not_evaluated_missing_explicit_lqr_controller",
                "care_residual_norm": float("inf"),
                "sampled_data_check_status": "not_evaluated_missing_explicit_lqr_controller",
                "sampled_data_spectral_radius": float("inf"),
                "command_clip_check_status": "not_evaluated_missing_explicit_lqr_controller",
                "saturation_summary": "not_evaluated_missing_explicit_lqr_controller",
                "latency_actuator_survival_status": "not_evaluated_missing_explicit_lqr_controller",
                "controller_claim_status": "simulation_only_blocked",
            }
        )
        return metadata
    return lqr_rollout_metadata(controller or fallback_controller)


def _controller_executable(controller: LQRController | None, controller_selection_status: str) -> bool:
    if controller is None or _controller_selection_is_missing(controller_selection_status):
        return False
    ok, _ = controller_is_executable_lqr(controller)
    return bool(ok)


def _controller_evidence_status(controller: LQRController | None, controller_selection_status: str) -> str:
    selection_status = str(controller_selection_status)
    if controller is None and _controller_selection_is_missing(controller_selection_status):
        return "blocked_missing_explicit_lqr_controller"
    if controller is None:
        return "nominal_debug_smoke"
    ok, reason = controller_is_executable_lqr(controller)
    if not ok:
        return f"blocked_{reason}"
    if selection_status in {
        "W01_variant_registry_candidate",
        "W2_fixed_lqr_survival_replay",
        "W3_fixed_lqr_survival_replay",
    }:
        return "registry_backed_executable"
    if selection_status == "W01_variant_registry_candidate":
        return "candidate_executable_lqr"
    if selection_status == "nominal_debug_smoke":
        return "nominal_debug_smoke_executable"
    return "executable_lqr"


def _controller_selection_is_missing(controller_selection_status: str) -> bool:
    text = str(controller_selection_status)
    return "missing" in text or "invalid_source_controller" in text


def _controller_selection_status(
    *,
    controller: LQRController | None,
    explicit_status: str | None,
) -> str:
    if explicit_status:
        return str(explicit_status)
    if controller is None:
        return "nominal_debug_smoke"
    return "explicit_lqr_unverified"


def _blocked_termination_cause(failure_label: str, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    label = str(failure_label)
    if "surrogate_binding_blocked" in label:
        return "surrogate_binding_blocked"
    if label in {
        "nonfinite_initial_state",
        "initial_floor_violation",
        "initial_ceiling_violation",
        "physically_impossible_initial_state",
        "z_boundary_exit",
    }:
        return label
    if label == "speed_low":
        return "speed_gate_blocked"
    if (
        "controller" in label
        or "registry" in label
        or label.startswith("missing_explicit_lqr_controller")
        or label.startswith("blocked_")
    ):
        return "controller_blocked"
    if label == "blocked_lqr_requested":
        return "controller_blocked"
    return "blocked_before_simulation"


def _entry_rejection_class(failure_label: str) -> str:
    label = str(failure_label)
    if label in {
        "nonfinite_initial_state",
        "initial_floor_violation",
        "initial_ceiling_violation",
        "physically_impossible_initial_state",
        "z_boundary_exit",
    }:
        return "physical_hard_failure"
    if label == "speed_low":
        return "speed_gate_blocked"
    if "surrogate_binding_blocked" in label:
        return "surrogate_blocked"
    if "controller" in label or "registry" in label or label == "blocked_lqr_requested":
        return "controller_blocked"
    return "other_blocked"


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


def _latency_for_implementation(latency_case: str, implementation: ImplementationInstance):
    base = latency_case_config(latency_case)
    return replace(
        base,
        state_feedback_delay_s=float(base.state_feedback_delay_s)
        * float(implementation.state_feedback_delay_scale)
        + float(implementation.latency_jitter_s),
        command_onset_delay_s=float(base.command_onset_delay_s)
        * float(implementation.command_onset_delay_scale),
        command_transport_delay_s=float(base.command_transport_delay_s)
        * float(implementation.command_transport_delay_scale)
        + float(implementation.latency_jitter_s),
        latency_jitter_s=float(base.latency_jitter_s) + float(implementation.latency_jitter_s),
    )


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
    lift_dwell_time_s: float,
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
        return "failed", "floor_margin_stop", "floor_violation"
    if ceiling_margin_m < 0.0:
        return "failed", "ceiling_margin_stop", "ceiling_violation"
    if minimum_wall_margin_m < 0.0:
        outcome = (
            "weak"
            if terminal_evidence_is_useful(
                energy_residual_m=energy_residual_m,
                lift_dwell_time_s=lift_dwell_time_s,
            )
            else "failed"
        )
        return outcome, "wall_boundary_exit_retained", "xy_boundary_terminal"
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
    lift_dwell_time_s: float,
    minimum_wall_margin_m: float,
    minimum_speed_m_s: float,
    minimum_speed_required_m_s: float,
) -> tuple[str, str, str]:
    if context.floor_margin_m < 0.0 or context.ceiling_margin_m < 0.0:
        return "failed", "safety_volume_exit", "true_safety_violation"
    if minimum_wall_margin_m < 0.0:
        outcome = (
            "weak"
            if terminal_evidence_is_useful(
                energy_residual_m=energy_residual_m,
                lift_dwell_time_s=lift_dwell_time_s,
            )
            else "failed"
        )
        return outcome, "wall_boundary_exit_retained", "xy_boundary_terminal"
    if minimum_speed_m_s < minimum_speed_required_m_s:
        return "blocked", "speed_gate_blocked", "speed_low"
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
    result = {name: row[name] for name in ROLLOUT_EVIDENCE_COLUMNS}
    state = np.asarray(json.loads(evidence.initial_state_vector), dtype=float).reshape(STATE_SIZE)
    result.update(
        {
            f"initial_{name}": float(state[index])
            for index, name in enumerate(STATE_NAMES)
        }
    )
    return result


def rollout_with_context_row(evidence: RolloutEvidence, context: EnvironmentContext) -> dict[str, object]:
    """Return evidence plus expanded context fields for archive partitions."""

    row = rollout_evidence_row(evidence)
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    return row


def _vector_json(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))
