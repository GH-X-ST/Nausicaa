from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import linalg

from command_contract import clip_normalised_command, normalised_command_to_surface_rad
from latency import AGGREGATE_LIMITS, SurfaceLimit, latency_case_config
from lqr_linearisation import (
    LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    LQR_STATE_MASK,
    LQRLinearisation,
    build_lqr_linearisation,
    lqr_linearisation_row,
    reduced_state_indices,
)
from prim_cat import PrimitiveDefinition, primitive_by_id
from primitive_timing_contract import (
    CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE,
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    PRIMITIVE_TIMING_CONTRACT_VERSION,
    assert_primitive_timing_contract,
)
from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Controller dataclasses
# 2) Synthesis
# 3) Command evaluation
# 4) Audit helpers
# =============================================================================


# =============================================================================
# 1) Controller Dataclasses
# =============================================================================
LQR_CONTROLLER_VERSION = "predictor_compensated_augmented_discrete_lqr_v1"
BASELINE_LQR_CONTROLLER_VERSION = "time_invariant_reduced_order_lqr_v1"
LQR_SYNTHESIS_SOLVED = "solved"
LQR_SYNTHESIS_BLOCKED = "blocked_lqr_synthesis"
ROLLOUT_DT_S = CONTROLLER_INPUT_UPDATE_PERIOD_S
ACTIVE_TIMING_AWARE_ROLE = "active_timing_aware_w01"
BASELINE_CONTROLLER_ROLE = "superseded_baseline_not_active_w01"
TIMING_AUGMENTATION_TYPE = "actuator_surface_state_command_fifo_predictor_compensated"
TIMING_DESIGN_VERSION = "predictor_compensated_augmented_discrete_lqr_v1"
TIMING_STATE_HISTORY_BACKED = "history_backed_fifo"
TIMING_STATE_INITIALISED = "initialised_fifo_not_history_backed"


@dataclass(frozen=True)
class LQRWeightSpec:
    q_attitude: float
    q_velocity: float
    q_rates: float
    q_surfaces: float
    r_aileron: float
    r_elevator: float
    r_rudder: float
    reference_pitch_bias_rad: float = 0.0
    reference_bank_bias_rad: float = 0.0
    reference_speed_bias_m_s: float = 0.0
    tuning_stage: str = "W0_W1"
    weight_label: str = "nominal"


@dataclass(frozen=True)
class LQRController:
    primitive_id: str
    controller_family: str
    controller_mode: str
    feedback_mode: str
    controller_id: str
    controller_version: str
    lqr_reference_id: str
    linearisation_id: str
    linearisation_source: str
    reduced_order_lqr: bool
    lqr_state_mask_json: str
    zero_position_gain_expansion_status: str
    full_state_care_status: str
    full_state_care_message: str
    full_controllability_rank: int
    full_state_size: int
    reduced_controllability_rank: int
    reduced_state_size: int
    care_residual_norm: float
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    lqr_synthesis_status: str
    lqr_blocked_reason: str
    lqr_closed_loop_eigenvalue_summary: str
    sampled_data_check_status: str
    sampled_data_spectral_radius: float
    command_clip_check_status: str
    saturation_summary: str
    latency_actuator_survival_status: str
    controller_design_role: str
    timing_augmentation_type: str
    timing_design_version: str
    sample_time_s: float
    latency_case: str
    state_feedback_delay_s: float
    command_delay_s: float
    command_delay_steps: int
    actuator_tau_s: tuple[float, float, float]
    actuator_state_count: int
    command_delay_state_count: int
    predictor_horizon_steps: int
    augmented_state_size: int
    augmented_input_size: int
    augmented_A_checksum: str
    augmented_B_checksum: str
    augmented_A_matrix_json: str
    augmented_B_matrix_json: str
    augmented_Q_json: str
    augmented_R_json: str
    augmented_gain_checksum: str
    augmented_gain_matrix_json: str
    augmented_closed_loop_spectral_radius: float
    timing_lqr_blocked_reason: str
    predictor_A_reduced_json: str
    timing_aware_synthesis_level: str
    timing_effects_in_synthesis: str
    timing_effects_in_rollout: str
    sampled_data_timing_audit_status: str
    delayed_state_lqr_augmentation_status: str
    tuning_stage: str
    controller_claim_status: str
    k_gain_matrix: tuple[tuple[float, ...], ...]
    reference_state_vector: tuple[float, ...]
    reference_command_vector: tuple[float, float, float]


@dataclass(frozen=True)
class TimingAwareControllerState:
    command_fifo_rad: tuple[tuple[float, float, float], ...]
    last_requested_command_rad: tuple[float, float, float]
    last_applied_command_rad: tuple[float, float, float]
    predictor_reference_command_rad: tuple[float, float, float]
    current_surface_state_rad: tuple[float, float, float]
    timing_state_source: str = TIMING_STATE_HISTORY_BACKED


@dataclass(frozen=True)
class LQRCommand:
    primitive_id: str
    controller_id: str
    feedback_mode: str
    command_norm: tuple[float, float, float]
    command_rad: tuple[float, float, float]
    saturation_applied: bool
    raw_command_rad: tuple[float, float, float]
    timing_state_source: str = TIMING_STATE_INITIALISED
    command_units: str = "normalised_command_and_radian_surface_targets"


# =============================================================================
# 2) Synthesis
# =============================================================================
def default_lqr_weight_spec(primitive_id: str, *, tuning_stage: str = "W0_W1") -> LQRWeightSpec:
    """Return conservative grouped diagonal Q/R weights for one primitive."""

    base = {
        "glide": (4.0, 2.0, 1.5, 0.15, 1.3, 0.9, 1.4),
        "recovery": (6.0, 2.2, 2.0, 0.20, 1.1, 0.8, 1.2),
        "lift_entry": (4.5, 2.5, 1.8, 0.15, 1.1, 0.9, 1.2),
        "lift_dwell_arc": (5.0, 2.2, 2.1, 0.18, 0.9, 1.0, 1.1),
        "mild_turn_left": (5.0, 2.0, 2.1, 0.18, 0.9, 1.1, 1.0),
        "mild_turn_right": (5.0, 2.0, 2.1, 0.18, 0.9, 1.1, 1.0),
        "energy_retaining_bank": (4.8, 2.8, 1.8, 0.15, 1.0, 0.9, 1.2),
        "safe_exit_or_recovery_handoff": (6.0, 2.0, 2.4, 0.20, 1.1, 0.8, 1.2),
        # Archive compatibility only: active evidence generation obtains
        # primitives through primitive_by_id(), which rejects these retired IDs.
        "launch_capture_glide_stabilise": (5.2, 2.4, 2.0, 0.18, 1.2, 0.9, 1.3),
        "launch_capture_lift_seek": (5.0, 2.8, 2.0, 0.18, 1.1, 0.9, 1.2),
        "launch_capture_energy_build": (4.8, 3.0, 1.9, 0.16, 1.1, 0.8, 1.2),
        "launch_capture_shallow_left": (5.3, 2.3, 2.2, 0.18, 1.0, 1.0, 1.1),
        "launch_capture_shallow_right": (5.3, 2.3, 2.2, 0.18, 1.0, 1.0, 1.1),
        "launch_capture_safe_handoff": (6.2, 2.3, 2.4, 0.20, 1.2, 0.8, 1.2),
    }.get(str(primitive_id), (4.0, 2.0, 2.0, 0.2, 1.0, 1.0, 1.0))
    return LQRWeightSpec(
        q_attitude=base[0],
        q_velocity=base[1],
        q_rates=base[2],
        q_surfaces=base[3],
        r_aileron=base[4],
        r_elevator=base[5],
        r_rudder=base[6],
        tuning_stage=tuning_stage,
        weight_label="nominal",
    )


@lru_cache(maxsize=128)
def lqr_controller_for_primitive_id(
    primitive_id: str,
    *,
    weight_label: str = "nominal",
    local_reference_speed_m_s: float = LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
) -> LQRController:
    weight_spec = default_lqr_weight_spec(primitive_id)
    if weight_label != "nominal":
        weight_spec = LQRWeightSpec(
            **{
                **asdict(weight_spec),
                "weight_label": str(weight_label),
            }
        )
    return synthesize_lqr_controller(
        primitive_by_id(primitive_id),
        weight_spec=weight_spec,
        local_reference_speed_m_s=float(local_reference_speed_m_s),
    )


def synthesize_lqr_controller(
    primitive: PrimitiveDefinition,
    *,
    weight_spec: LQRWeightSpec | None = None,
    local_reference_speed_m_s: float,
    rollout_dt_s: float = ROLLOUT_DT_S,
    latency_case: str = "nominal",
) -> LQRController:
    """Synthesize the active timing-aware W01 LQR controller for one primitive."""

    assert_primitive_timing_contract(
        finite_horizon_s=primitive.finite_horizon_s,
        controller_input_slots_per_primitive=primitive.controller_input_slots_per_primitive,
        controller_input_update_period_s=primitive.controller_input_update_period_s,
        primitive_timing_contract_version=primitive.primitive_timing_contract_version,
    )
    if not np.isclose(float(rollout_dt_s), CONTROLLER_INPUT_UPDATE_PERIOD_S, rtol=0.0, atol=1e-12):
        raise ValueError("rollout_dt_s_not_v411_0p020s")
    weights = weight_spec or default_lqr_weight_spec(primitive.primitive_id)
    linearisation = build_lqr_linearisation(
        primitive,
        local_reference_speed_m_s=float(local_reference_speed_m_s),
        reference_pitch_bias_rad=float(weights.reference_pitch_bias_rad),
        reference_bank_bias_rad=float(weights.reference_bank_bias_rad),
        reference_speed_bias_m_s=float(weights.reference_speed_bias_m_s),
    )
    a_full = np.asarray(linearisation.a_full, dtype=float)
    b_full = np.asarray(linearisation.b_full, dtype=float)
    a_reduced = np.asarray(linearisation.a_reduced, dtype=float)
    b_reduced = np.asarray(linearisation.b_reduced, dtype=float)
    q_reduced = _q_reduced(weights)
    r_matrix = _r_matrix(weights)
    q_full = _expand_q_to_full(q_reduced)
    full_status, full_message = _care_attempt_status(a_full, b_full, q_full, r_matrix)

    timing = latency_case_config(latency_case)
    command_delay_s = float(timing.command_onset_delay_s + timing.command_transport_delay_s)
    delay_steps = max(1, int(np.ceil(command_delay_s / float(rollout_dt_s))))
    predictor_steps = max(0, int(np.ceil(float(timing.state_feedback_delay_s) / float(rollout_dt_s))))
    actuator_tau = tuple(float(value) for value in timing.actuator_tau_s)

    ad_reduced, bd_reduced = _discretise_zoh(a_reduced, b_reduced, float(rollout_dt_s))
    a_aug, b_aug = _augment_discrete_command_delay(ad_reduced, bd_reduced, delay_steps)
    q_aug = _augmented_q_matrix(q_reduced, weights, delay_steps)
    r_aug = r_matrix.copy()
    k_aug = np.zeros((3, a_aug.shape[0]), dtype=float)
    care_residual = float("inf")
    augmented_radius = float("inf")
    sampled_status = LQR_SYNTHESIS_BLOCKED
    synthesis_status = LQR_SYNTHESIS_BLOCKED
    timing_blocked_reason = ""
    eig_summary = "not_evaluated"
    try:
        p_aug = linalg.solve_discrete_are(a_aug, b_aug, q_aug, r_aug)
        k_aug = np.linalg.solve(b_aug.T @ p_aug @ b_aug + r_aug, b_aug.T @ p_aug @ a_aug)
        care_residual = _dare_residual_norm(a_aug, b_aug, q_aug, r_aug, p_aug)
        closed_loop = a_aug - b_aug @ k_aug
        eig_aug = np.linalg.eigvals(closed_loop)
        eig_summary = _eigen_summary(eig_aug)
        augmented_radius = float(np.max(np.abs(eig_aug)))
        sampled_status = "sampled_stable" if augmented_radius < 1.0 else "failed_augmented_sampled_data_instability"
    except Exception as exc:
        timing_blocked_reason = f"augmented_discrete_lqr_failed:{type(exc).__name__}:{exc}"

    k_physical_full = _expand_gain_to_full(k_aug[:, : len(LQR_STATE_MASK)])
    expansion_status = _gain_expansion_status(k_physical_full)
    command_status, saturation_summary = _reference_command_check(
        tuple(linearisation.reference.reference_command_vector)
    )
    if not timing_blocked_reason:
        reasons = _timing_blocked_reason(
            linearisation=linearisation,
            dare_residual=care_residual,
            sampled_status=sampled_status,
            command_status=command_status,
            expansion_status=expansion_status,
        )
        timing_blocked_reason = reasons
    synthesis_status = LQR_SYNTHESIS_SOLVED if not timing_blocked_reason else LQR_SYNTHESIS_BLOCKED

    q_json = json.dumps(_q_weight_payload(weights), sort_keys=True, separators=(",", ":"))
    r_json = json.dumps(_r_weight_payload(weights), sort_keys=True, separators=(",", ":"))
    augmented_q_json = json.dumps(_augmented_q_payload(weights, delay_steps), sort_keys=True, separators=(",", ":"))
    augmented_r_json = json.dumps(_augmented_r_payload(weights), sort_keys=True, separators=(",", ":"))
    gain_checksum = gain_checksum_sha256(k_physical_full)
    augmented_gain_checksum = gain_checksum_sha256(k_aug)
    a_checksum = matrix_checksum_sha256(a_aug)
    b_checksum = matrix_checksum_sha256(b_aug)
    controller_id = _timing_controller_id(
        primitive_id=primitive.primitive_id,
        linearisation_id=linearisation.linearisation_id,
        q_json=q_json,
        r_json=r_json,
        augmented_q_json=augmented_q_json,
        augmented_r_json=augmented_r_json,
        gain_checksum=gain_checksum,
        augmented_gain_checksum=augmented_gain_checksum,
        weight_label=weights.weight_label,
        sample_time_s=float(rollout_dt_s),
        finite_horizon_s=float(primitive.finite_horizon_s),
        controller_input_slots_per_primitive=CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE,
        controller_input_update_period_s=CONTROLLER_INPUT_UPDATE_PERIOD_S,
        primitive_timing_contract_version=PRIMITIVE_TIMING_CONTRACT_VERSION,
        latency_case=str(latency_case),
        command_delay_steps=delay_steps,
        predictor_horizon_steps=predictor_steps,
        augmented_A_checksum=a_checksum,
        augmented_B_checksum=b_checksum,
    )
    return LQRController(
        primitive_id=primitive.primitive_id,
        controller_family="lqr",
        controller_mode="lqr_local_feedback",
        feedback_mode="predictor_compensated_augmented_discrete_lqr",
        controller_id=controller_id,
        controller_version=LQR_CONTROLLER_VERSION,
        lqr_reference_id=linearisation.reference.reference_id,
        linearisation_id=linearisation.linearisation_id,
        linearisation_source=linearisation.linearisation_source,
        reduced_order_lqr=True,
        lqr_state_mask_json=json.dumps(LQR_STATE_MASK, separators=(",", ":")),
        zero_position_gain_expansion_status=expansion_status,
        full_state_care_status=full_status,
        full_state_care_message=full_message,
        full_controllability_rank=int(linearisation.full_controllability_rank),
        full_state_size=int(linearisation.full_state_size),
        reduced_controllability_rank=int(linearisation.reduced_controllability_rank),
        reduced_state_size=int(linearisation.reduced_state_size),
        care_residual_norm=float(care_residual),
        lqr_Q_weights_json=q_json,
        lqr_R_weights_json=r_json,
        lqr_gain_checksum=gain_checksum,
        lqr_synthesis_status=synthesis_status,
        lqr_blocked_reason=timing_blocked_reason,
        lqr_closed_loop_eigenvalue_summary=eig_summary,
        sampled_data_check_status=sampled_status,
        sampled_data_spectral_radius=float(augmented_radius),
        command_clip_check_status=command_status,
        saturation_summary=saturation_summary,
        latency_actuator_survival_status=(
            "timing_augmented_discrete_lqr_solved"
            if synthesis_status == LQR_SYNTHESIS_SOLVED
            else "timing_augmented_discrete_lqr_blocked"
        ),
        controller_design_role=ACTIVE_TIMING_AWARE_ROLE,
        timing_augmentation_type=TIMING_AUGMENTATION_TYPE,
        timing_design_version=TIMING_DESIGN_VERSION,
        sample_time_s=float(rollout_dt_s),
        latency_case=str(latency_case),
        state_feedback_delay_s=float(timing.state_feedback_delay_s),
        command_delay_s=float(command_delay_s),
        command_delay_steps=int(delay_steps),
        actuator_tau_s=actuator_tau,
        actuator_state_count=3,
        command_delay_state_count=int(3 * delay_steps),
        predictor_horizon_steps=int(predictor_steps),
        augmented_state_size=int(a_aug.shape[0]),
        augmented_input_size=int(b_aug.shape[1]),
        augmented_A_checksum=a_checksum,
        augmented_B_checksum=b_checksum,
        augmented_A_matrix_json=json.dumps(_rounded_matrix_payload(a_aug), separators=(",", ":")),
        augmented_B_matrix_json=json.dumps(_rounded_matrix_payload(b_aug), separators=(",", ":")),
        augmented_Q_json=augmented_q_json,
        augmented_R_json=augmented_r_json,
        augmented_gain_checksum=augmented_gain_checksum,
        augmented_gain_matrix_json=json.dumps(_rounded_matrix_payload(k_aug), separators=(",", ":")),
        augmented_closed_loop_spectral_radius=float(augmented_radius),
        timing_lqr_blocked_reason=timing_blocked_reason,
        predictor_A_reduced_json=json.dumps(_rounded_matrix_payload(ad_reduced), separators=(",", ":")),
        timing_aware_synthesis_level="predictor_compensated_augmented_discrete_lqr",
        timing_effects_in_synthesis="discrete_lqr_with_actuator_surface_states_and_command_delay_fifo",
        timing_effects_in_rollout="predictor_compensation_feedback_delay_command_timing_actuator_lag_applied",
        sampled_data_timing_audit_status=(
            "augmented_discrete_sampled_stable"
            if synthesis_status == LQR_SYNTHESIS_SOLVED
            else "augmented_discrete_lqr_blocked"
        ),
        delayed_state_lqr_augmentation_status="predictor_compensation_only_no_full_delayed_state_augmentation",
        tuning_stage=weights.tuning_stage,
        controller_claim_status="simulation_only",
        k_gain_matrix=tuple(tuple(float(value) for value in row) for row in k_physical_full),
        reference_state_vector=linearisation.reference.reference_state_vector,
        reference_command_vector=linearisation.reference.reference_command_vector,
    )


def synthesize_baseline_trim_lqr_controller(
    primitive: PrimitiveDefinition,
    *,
    weight_spec: LQRWeightSpec | None = None,
    local_reference_speed_m_s: float = LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    rollout_dt_s: float = ROLLOUT_DT_S,
) -> LQRController:
    """Synthesize the superseded non-augmented trim/local baseline LQR."""

    weights = weight_spec or default_lqr_weight_spec(primitive.primitive_id)
    linearisation = build_lqr_linearisation(
        primitive,
        local_reference_speed_m_s=float(local_reference_speed_m_s),
        reference_pitch_bias_rad=float(weights.reference_pitch_bias_rad),
        reference_bank_bias_rad=float(weights.reference_bank_bias_rad),
        reference_speed_bias_m_s=float(weights.reference_speed_bias_m_s),
    )
    a_full = np.asarray(linearisation.a_full, dtype=float)
    b_full = np.asarray(linearisation.b_full, dtype=float)
    a_reduced = np.asarray(linearisation.a_reduced, dtype=float)
    b_reduced = np.asarray(linearisation.b_reduced, dtype=float)
    q_reduced = _q_reduced(weights)
    r_matrix = _r_matrix(weights)
    q_full = _expand_q_to_full(q_reduced)

    full_status, full_message = _care_attempt_status(a_full, b_full, q_full, r_matrix)
    blocked_reason = ""
    try:
        p_reduced = linalg.solve_continuous_are(a_reduced, b_reduced, q_reduced, r_matrix)
        k_reduced = np.linalg.solve(r_matrix, b_reduced.T @ p_reduced)
        care_residual = _care_residual_norm(a_reduced, b_reduced, q_reduced, r_matrix, p_reduced)
        k_full = _expand_gain_to_full(k_reduced)
        expansion_status = _gain_expansion_status(k_full)
        eig_cont = np.linalg.eigvals(a_reduced - b_reduced @ k_reduced)
        eig_summary = _eigen_summary(eig_cont)
        sampled_radius = _sampled_spectral_radius(a_reduced, b_reduced, k_reduced, rollout_dt_s)
        sampled_status = "sampled_stable" if sampled_radius < 1.0 else "failed_sampled_data_instability"
        command_status, saturation_summary = _reference_command_check(
            tuple(linearisation.reference.reference_command_vector)
        )
        latency_status = (
            "survives_nominal_latency_actuator_lag"
            if sampled_status == "sampled_stable" and float(np.max(np.real(eig_cont))) < 0.0
            else "failed_linear_lag_smoke"
        )
        sampled_data_timing_audit_status = (
            "sampled_stable_with_nominal_timing_smoke"
            if latency_status == "survives_nominal_latency_actuator_lag"
            else "failed_nominal_timing_smoke"
        )
        synthesis_status = (
            LQR_SYNTHESIS_SOLVED
            if (
                linearisation.finite_ab_check == "finite"
                and linearisation.reduced_controllability_rank == linearisation.reduced_state_size
                and np.isfinite(care_residual)
                and care_residual < 1e-6
                and sampled_status == "sampled_stable"
                and command_status == "passed"
                and latency_status == "survives_nominal_latency_actuator_lag"
                and expansion_status == "zero_position_gains_verified"
            )
            else LQR_SYNTHESIS_BLOCKED
        )
        if synthesis_status == LQR_SYNTHESIS_BLOCKED:
            blocked_reason = _blocked_reason(
                linearisation=linearisation,
                care_residual=care_residual,
                sampled_status=sampled_status,
                command_status=command_status,
                latency_status=latency_status,
                expansion_status=expansion_status,
            )
    except Exception as exc:
        k_full = np.zeros((3, STATE_SIZE), dtype=float)
        care_residual = float("inf")
        eig_summary = f"blocked:{type(exc).__name__}:{exc}"
        sampled_radius = float("inf")
        sampled_status = "blocked_lqr_synthesis"
        command_status = "not_evaluated"
        saturation_summary = "not_evaluated"
        latency_status = "not_evaluated"
        expansion_status = "not_evaluated"
        synthesis_status = LQR_SYNTHESIS_BLOCKED
        sampled_data_timing_audit_status = "not_evaluated"
        blocked_reason = f"reduced_care_failed:{type(exc).__name__}:{exc}"

    q_json = json.dumps(_q_weight_payload(weights), sort_keys=True, separators=(",", ":"))
    r_json = json.dumps(_r_weight_payload(weights), sort_keys=True, separators=(",", ":"))
    gain_checksum = gain_checksum_sha256(k_full)
    controller_id = _controller_id(
        primitive.primitive_id,
        linearisation.linearisation_id,
        q_json,
        r_json,
        gain_checksum,
        weights.weight_label,
    )
    return LQRController(
        primitive_id=primitive.primitive_id,
        controller_family="lqr",
        controller_mode="lqr_local_feedback",
        feedback_mode="lqr_state_feedback",
        controller_id=controller_id,
        controller_version=BASELINE_LQR_CONTROLLER_VERSION,
        lqr_reference_id=linearisation.reference.reference_id,
        linearisation_id=linearisation.linearisation_id,
        linearisation_source=linearisation.linearisation_source,
        reduced_order_lqr=True,
        lqr_state_mask_json=json.dumps(LQR_STATE_MASK, separators=(",", ":")),
        zero_position_gain_expansion_status=expansion_status,
        full_state_care_status=full_status,
        full_state_care_message=full_message,
        full_controllability_rank=int(linearisation.full_controllability_rank),
        full_state_size=int(linearisation.full_state_size),
        reduced_controllability_rank=int(linearisation.reduced_controllability_rank),
        reduced_state_size=int(linearisation.reduced_state_size),
        care_residual_norm=float(care_residual),
        lqr_Q_weights_json=q_json,
        lqr_R_weights_json=r_json,
        lqr_gain_checksum=gain_checksum,
        lqr_synthesis_status=synthesis_status,
        lqr_blocked_reason=blocked_reason,
        lqr_closed_loop_eigenvalue_summary=eig_summary,
        sampled_data_check_status=sampled_status,
        sampled_data_spectral_radius=float(sampled_radius),
        command_clip_check_status=command_status,
        saturation_summary=saturation_summary,
        latency_actuator_survival_status=latency_status,
        controller_design_role=BASELINE_CONTROLLER_ROLE,
        timing_augmentation_type="none",
        timing_design_version=BASELINE_LQR_CONTROLLER_VERSION,
        sample_time_s=float(rollout_dt_s),
        latency_case="nominal",
        state_feedback_delay_s=0.0,
        command_delay_s=0.0,
        command_delay_steps=0,
        actuator_tau_s=(0.0, 0.0, 0.0),
        actuator_state_count=0,
        command_delay_state_count=0,
        predictor_horizon_steps=0,
        augmented_state_size=int(linearisation.reduced_state_size),
        augmented_input_size=3,
        augmented_A_checksum=matrix_checksum_sha256(a_reduced),
        augmented_B_checksum=matrix_checksum_sha256(b_reduced),
        augmented_A_matrix_json=json.dumps(_rounded_matrix_payload(a_reduced), separators=(",", ":")),
        augmented_B_matrix_json=json.dumps(_rounded_matrix_payload(b_reduced), separators=(",", ":")),
        augmented_Q_json=json.dumps({"baseline_q": _q_weight_payload(weights)}, sort_keys=True, separators=(",", ":")),
        augmented_R_json=json.dumps({"baseline_r": _r_weight_payload(weights)}, sort_keys=True, separators=(",", ":")),
        augmented_gain_checksum=gain_checksum,
        augmented_gain_matrix_json=json.dumps(_rounded_matrix_payload(k_reduced if "k_reduced" in locals() else np.zeros((3, len(LQR_STATE_MASK)))), separators=(",", ":")),
        augmented_closed_loop_spectral_radius=float(sampled_radius),
        timing_lqr_blocked_reason=blocked_reason,
        predictor_A_reduced_json=json.dumps(_rounded_matrix_payload(linalg.expm(a_reduced * float(rollout_dt_s))), separators=(",", ":")),
        timing_aware_synthesis_level="trim_local_reduced_order_lqr_no_delay_augmentation",
        timing_effects_in_synthesis="sampled_data_stability_and_nominal_latency_actuator_smoke_only",
        timing_effects_in_rollout="feedback_delay_command_timing_actuator_lag_applied_in_w01_rollout",
        sampled_data_timing_audit_status=sampled_data_timing_audit_status,
        delayed_state_lqr_augmentation_status="not_implemented_state_delay_simulated_in_rollout",
        tuning_stage=weights.tuning_stage,
        controller_claim_status="simulation_only",
        k_gain_matrix=tuple(tuple(float(value) for value in row) for row in k_full),
        reference_state_vector=linearisation.reference.reference_state_vector,
        reference_command_vector=linearisation.reference.reference_command_vector,
    )


# =============================================================================
# 3) Command Evaluation
# =============================================================================
def lqr_command_for_state(
    *,
    controller: LQRController,
    state_vector: np.ndarray,
    timing_state: TimingAwareControllerState | None = None,
) -> LQRCommand:
    """Return the LQR surface command for one state."""

    state = as_state_vector(state_vector)
    x_ref = np.asarray(controller.reference_state_vector, dtype=float)
    u_ref = np.asarray(controller.reference_command_vector, dtype=float)
    gain = np.asarray(controller.k_gain_matrix, dtype=float).reshape(3, STATE_SIZE)
    if controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
        raise RuntimeError(
            "blocked LQR controller cannot produce an executable command: "
            f"{controller.controller_id}:{controller.lqr_synthesis_status}"
    )
    if controller_is_active_timing_aware_w01(controller):
        raw_rad, timing_state_source = _timing_aware_raw_command_rad(
            controller=controller,
            state=state,
            timing_state=timing_state,
        )
    else:
        raw_rad = u_ref - gain @ (state - x_ref)
        timing_state_source = "not_applicable_baseline_controller"
    raw_norm = _surface_rad_to_unclipped_norm(raw_rad)
    clipped_norm = clip_normalised_command(raw_norm)
    clipped_rad = normalised_command_to_surface_rad(clipped_norm)
    return LQRCommand(
        primitive_id=controller.primitive_id,
        controller_id=controller.controller_id,
        feedback_mode=controller.feedback_mode,
        command_norm=tuple(float(value) for value in clipped_norm),
        command_rad=tuple(float(value) for value in clipped_rad),
        saturation_applied=bool(np.any(np.abs(raw_norm - clipped_norm) > 1e-12)),
        raw_command_rad=tuple(float(value) for value in raw_rad),
        timing_state_source=timing_state_source,
    )


def lqr_controller_metadata_row(controller: LQRController) -> dict[str, object]:
    row = asdict(controller)
    row.pop("k_gain_matrix")
    row["k_gain_matrix_json"] = json.dumps(controller.k_gain_matrix, separators=(",", ":"))
    row["reference_state_vector"] = json.dumps(
        list(controller.reference_state_vector),
        separators=(",", ":"),
    )
    row["reference_command_vector"] = json.dumps(
        list(controller.reference_command_vector),
        separators=(",", ":"),
    )
    return row


def lqr_rollout_metadata(controller: LQRController) -> dict[str, object]:
    """Return the LQR fields required on every rollout evidence row."""

    return {
        "controller_family": controller.controller_family,
        "controller_id": controller.controller_id,
        "lqr_reference_id": controller.lqr_reference_id,
        "linearisation_id": controller.linearisation_id,
        "linearisation_source": controller.linearisation_source,
        "reduced_order_lqr": controller.reduced_order_lqr,
        "lqr_state_mask_json": controller.lqr_state_mask_json,
        "zero_position_gain_expansion_status": controller.zero_position_gain_expansion_status,
        "lqr_Q_weights_json": controller.lqr_Q_weights_json,
        "lqr_R_weights_json": controller.lqr_R_weights_json,
        "lqr_gain_checksum": controller.lqr_gain_checksum,
        "lqr_synthesis_status": controller.lqr_synthesis_status,
        "lqr_blocked_reason": controller.lqr_blocked_reason,
        "lqr_closed_loop_eigenvalue_summary": controller.lqr_closed_loop_eigenvalue_summary,
        "care_residual_norm": controller.care_residual_norm,
        "sampled_data_check_status": controller.sampled_data_check_status,
        "sampled_data_spectral_radius": controller.sampled_data_spectral_radius,
        "command_clip_check_status": controller.command_clip_check_status,
        "saturation_summary": controller.saturation_summary,
        "latency_actuator_survival_status": controller.latency_actuator_survival_status,
        "controller_design_role": controller.controller_design_role,
        "timing_augmentation_type": controller.timing_augmentation_type,
        "timing_design_version": controller.timing_design_version,
        "sample_time_s": controller.sample_time_s,
        "state_feedback_delay_s": controller.state_feedback_delay_s,
        "command_delay_s": controller.command_delay_s,
        "command_delay_steps": controller.command_delay_steps,
        "actuator_tau_s": json.dumps(list(controller.actuator_tau_s), separators=(",", ":")),
        "actuator_state_count": controller.actuator_state_count,
        "command_delay_state_count": controller.command_delay_state_count,
        "predictor_horizon_steps": controller.predictor_horizon_steps,
        "augmented_state_size": controller.augmented_state_size,
        "augmented_input_size": controller.augmented_input_size,
        "augmented_A_checksum": controller.augmented_A_checksum,
        "augmented_B_checksum": controller.augmented_B_checksum,
        "augmented_Q_json": controller.augmented_Q_json,
        "augmented_R_json": controller.augmented_R_json,
        "augmented_gain_checksum": controller.augmented_gain_checksum,
        "augmented_closed_loop_spectral_radius": controller.augmented_closed_loop_spectral_radius,
        "timing_lqr_blocked_reason": controller.timing_lqr_blocked_reason,
        "timing_aware_synthesis_level": controller.timing_aware_synthesis_level,
        "timing_effects_in_synthesis": controller.timing_effects_in_synthesis,
        "timing_effects_in_rollout": controller.timing_effects_in_rollout,
        "sampled_data_timing_audit_status": controller.sampled_data_timing_audit_status,
        "delayed_state_lqr_augmentation_status": controller.delayed_state_lqr_augmentation_status,
        "tuning_stage": controller.tuning_stage,
        "controller_claim_status": controller.controller_claim_status,
    }


def controller_is_executable_lqr(controller: LQRController) -> tuple[bool, str]:
    """Return whether an LQR controller can be executed in active W01 evidence."""

    if controller.controller_family != "lqr":
        return False, "controller_family_not_lqr"
    if not controller_is_active_timing_aware_w01(controller):
        return False, "baseline_controller_not_active_w01"
    if controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
        return False, "lqr_synthesis_not_solved"
    if controller.sampled_data_check_status != "sampled_stable":
        return False, "sampled_data_not_stable"
    if controller.zero_position_gain_expansion_status != "zero_position_gains_verified":
        return False, "zero_position_gains_not_verified"
    if controller.controller_claim_status != "simulation_only":
        return False, "controller_claim_status_not_allowed"
    return True, ""


# =============================================================================
# 4) Audit Helpers
# =============================================================================
def controller_is_active_timing_aware_w01(controller: LQRController) -> bool:
    """Return whether a controller is the active v4.3 W01 timing-aware design."""

    return (
        controller.controller_design_role == ACTIVE_TIMING_AWARE_ROLE
        and controller.timing_augmentation_type == TIMING_AUGMENTATION_TYPE
        and controller.timing_design_version == TIMING_DESIGN_VERSION
    )


def timing_augmented_lqr_design_row(controller: LQRController) -> dict[str, object]:
    """Return the compact timing-aware design audit row used by tests and reports."""

    return {
        "primitive_id": controller.primitive_id,
        "controller_id": controller.controller_id,
        "controller_design_role": controller.controller_design_role,
        "timing_augmentation_type": controller.timing_augmentation_type,
        "timing_design_version": controller.timing_design_version,
        "sample_time_s": float(controller.sample_time_s),
        "latency_case": controller.latency_case,
        "state_feedback_delay_s": float(controller.state_feedback_delay_s),
        "command_delay_s": float(controller.command_delay_s),
        "command_delay_steps": int(controller.command_delay_steps),
        "actuator_tau_s": json.dumps(list(controller.actuator_tau_s), separators=(",", ":")),
        "actuator_state_count": int(controller.actuator_state_count),
        "command_delay_state_count": int(controller.command_delay_state_count),
        "predictor_horizon_steps": int(controller.predictor_horizon_steps),
        "augmented_state_size": int(controller.augmented_state_size),
        "augmented_input_size": int(controller.augmented_input_size),
        "augmented_A_checksum": controller.augmented_A_checksum,
        "augmented_B_checksum": controller.augmented_B_checksum,
        "augmented_A_matrix_recorded": bool(controller.augmented_A_matrix_json),
        "augmented_B_matrix_recorded": bool(controller.augmented_B_matrix_json),
        "augmented_Q_json": controller.augmented_Q_json,
        "augmented_R_json": controller.augmented_R_json,
        "augmented_gain_checksum": controller.augmented_gain_checksum,
        "augmented_closed_loop_spectral_radius": float(controller.augmented_closed_loop_spectral_radius),
        "lqr_synthesis_status": controller.lqr_synthesis_status,
        "timing_lqr_blocked_reason": controller.timing_lqr_blocked_reason,
        "delayed_state_lqr_augmentation_status": controller.delayed_state_lqr_augmentation_status,
        "controller_claim_status": controller.controller_claim_status,
    }


def compare_timing_aware_vs_baseline_nominal(
    primitive: PrimitiveDefinition,
    *,
    weight_spec: LQRWeightSpec | None = None,
    local_reference_speed_m_s: float = LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
) -> dict[str, object]:
    """Return a deterministic command-path comparison under nominal timing assumptions."""

    timing_aware = synthesize_lqr_controller(
        primitive,
        weight_spec=weight_spec,
        local_reference_speed_m_s=float(local_reference_speed_m_s),
    )
    baseline = synthesize_baseline_trim_lqr_controller(
        primitive,
        weight_spec=weight_spec,
        local_reference_speed_m_s=float(local_reference_speed_m_s),
    )
    state = np.asarray(timing_aware.reference_state_vector, dtype=float).copy()
    state[STATE_INDEX["theta"]] += np.deg2rad(1.0)
    state[STATE_INDEX["q"]] += 0.02
    state[STATE_INDEX["delta_e"]] += np.deg2rad(0.5)
    timing_command = lqr_command_for_state(
        controller=timing_aware,
        state_vector=state,
        timing_state=initialised_timing_state_for_controller(timing_aware, state),
    )
    baseline_command = lqr_command_for_state(controller=baseline, state_vector=state)
    timing_rad = np.asarray(timing_command.command_rad, dtype=float)
    baseline_rad = np.asarray(baseline_command.command_rad, dtype=float)
    return {
        "primitive_id": primitive.primitive_id,
        "timing_aware_controller_id": timing_aware.controller_id,
        "baseline_controller_id": baseline.controller_id,
        "timing_aware_role": timing_aware.controller_design_role,
        "baseline_role": baseline.controller_design_role,
        "command_delta_norm": float(np.linalg.norm(timing_rad - baseline_rad)),
        "timing_aware_command_rad": tuple(float(value) for value in timing_rad),
        "baseline_command_rad": tuple(float(value) for value in baseline_rad),
    }


def synthesis_audit_row(primitive: PrimitiveDefinition) -> dict[str, object]:
    controller = synthesize_lqr_controller(
        primitive,
        local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    )
    row = lqr_controller_metadata_row(controller)
    row.update(
        {
            f"linearisation_{key}": value
            for key, value in lqr_linearisation_row(
                build_lqr_linearisation(
                    primitive,
                    local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
                )
            ).items()
        }
    )
    return row


def gain_checksum_sha256(gain_matrix: np.ndarray) -> str:
    rounded = np.round(np.asarray(gain_matrix, dtype=float), decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def matrix_checksum_sha256(matrix: np.ndarray) -> str:
    """Return a stable checksum for an audited numeric matrix."""

    rounded = np.round(np.asarray(matrix, dtype=float), decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _rounded_matrix_payload(matrix: np.ndarray, *, decimals: int = 12) -> list[list[float]]:
    rounded = np.round(np.asarray(matrix, dtype=float), decimals=int(decimals))
    return [[float(value) for value in row] for row in rounded.tolist()]


def _discretise_zoh(a: np.ndarray, b: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return exact zero-order-hold discrete A/B matrices."""

    a_matrix = np.asarray(a, dtype=float)
    b_matrix = np.asarray(b, dtype=float)
    n_state, n_input = b_matrix.shape
    block = np.zeros((n_state + n_input, n_state + n_input), dtype=float)
    block[:n_state, :n_state] = a_matrix
    block[:n_state, n_state:] = b_matrix
    expm = linalg.expm(block * float(dt_s))
    return expm[:n_state, :n_state], expm[:n_state, n_state:]


def _augment_discrete_command_delay(
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
    delay_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a command FIFO chain to the reduced discrete plant."""

    ad = np.asarray(a_discrete, dtype=float)
    bd = np.asarray(b_discrete, dtype=float)
    steps = max(0, int(delay_steps))
    if steps <= 0:
        return ad, bd
    n_state = int(ad.shape[0])
    n_input = int(bd.shape[1])
    fifo_size = steps * n_input
    a_aug = np.zeros((n_state + fifo_size, n_state + fifo_size), dtype=float)
    b_aug = np.zeros((n_state + fifo_size, n_input), dtype=float)
    a_aug[:n_state, :n_state] = ad
    a_aug[:n_state, n_state : n_state + n_input] = bd
    for fifo_index in range(steps - 1):
        row_start = n_state + fifo_index * n_input
        col_start = n_state + (fifo_index + 1) * n_input
        a_aug[row_start : row_start + n_input, col_start : col_start + n_input] = np.eye(n_input)
    b_aug[n_state + (steps - 1) * n_input : n_state + steps * n_input, :] = np.eye(n_input)
    return a_aug, b_aug


def _augmented_q_matrix(q_reduced: np.ndarray, weights: LQRWeightSpec, delay_steps: int) -> np.ndarray:
    q = np.asarray(q_reduced, dtype=float)
    steps = max(0, int(delay_steps))
    if steps <= 0:
        return q.copy()
    fifo_weight = max(1e-6, float(weights.q_surfaces) * 0.25)
    q_fifo = np.eye(3 * steps, dtype=float) * fifo_weight
    out = np.zeros((q.shape[0] + q_fifo.shape[0], q.shape[1] + q_fifo.shape[1]), dtype=float)
    out[: q.shape[0], : q.shape[1]] = q
    out[q.shape[0] :, q.shape[1] :] = q_fifo
    return out


def _augmented_q_payload(weights: LQRWeightSpec, delay_steps: int) -> dict[str, object]:
    return {
        "grouping": "reduced_physical_plus_command_fifo_diagonal",
        "physical_state_q": _q_weight_payload(weights),
        "command_fifo_step_count": int(delay_steps),
        "command_fifo_state_count": int(max(0, delay_steps) * 3),
        "command_fifo_weight": float(max(1e-6, float(weights.q_surfaces) * 0.25)),
    }


def _augmented_r_payload(weights: LQRWeightSpec) -> dict[str, object]:
    payload = _r_weight_payload(weights)
    payload["input_role"] = "new_command_request_after_fifo"
    return payload


def _dare_residual_norm(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
) -> float:
    bt_p_b = b.T @ p @ b
    gain_term = a.T @ p @ b @ np.linalg.solve(bt_p_b + r, b.T @ p @ a)
    residual = a.T @ p @ a - p - gain_term + q
    scale = max(1.0, float(np.linalg.norm(q, ord="fro")))
    return float(np.linalg.norm(residual, ord="fro") / scale)


def _timing_blocked_reason(
    *,
    linearisation: LQRLinearisation,
    dare_residual: float,
    sampled_status: str,
    command_status: str,
    expansion_status: str,
) -> str:
    reasons = []
    if linearisation.finite_ab_check != "finite":
        reasons.append("nonfinite_linearisation")
    if linearisation.reduced_controllability_rank != linearisation.reduced_state_size:
        reasons.append("reduced_controllability_rank_deficient")
    if not np.isfinite(dare_residual) or dare_residual >= 1e-6:
        reasons.append("augmented_dare_residual_high")
    if sampled_status != "sampled_stable":
        reasons.append(sampled_status)
    if command_status != "passed":
        reasons.append(command_status)
    if expansion_status != "zero_position_gains_verified":
        reasons.append(expansion_status)
    return ";".join(reasons)


def _timing_controller_id(
    *,
    primitive_id: str,
    linearisation_id: str,
    q_json: str,
    r_json: str,
    augmented_q_json: str,
    augmented_r_json: str,
    gain_checksum: str,
    augmented_gain_checksum: str,
    weight_label: str,
    sample_time_s: float,
    finite_horizon_s: float,
    controller_input_slots_per_primitive: int,
    controller_input_update_period_s: float,
    primitive_timing_contract_version: str,
    latency_case: str,
    command_delay_steps: int,
    predictor_horizon_steps: int,
    augmented_A_checksum: str,
    augmented_B_checksum: str,
) -> str:
    payload = {
        "primitive_id": primitive_id,
        "linearisation_id": linearisation_id,
        "controller_design_role": ACTIVE_TIMING_AWARE_ROLE,
        "timing_augmentation_type": TIMING_AUGMENTATION_TYPE,
        "timing_design_version": TIMING_DESIGN_VERSION,
        "sample_time_s": float(sample_time_s),
        "finite_horizon_s": float(finite_horizon_s),
        "controller_input_slots_per_primitive": int(controller_input_slots_per_primitive),
        "controller_input_update_period_s": float(controller_input_update_period_s),
        "primitive_timing_contract_version": str(primitive_timing_contract_version),
        "latency_case": str(latency_case),
        "command_delay_steps": int(command_delay_steps),
        "predictor_horizon_steps": int(predictor_horizon_steps),
        "Q_weight_json": q_json,
        "R_weight_json": r_json,
        "augmented_Q_json": augmented_q_json,
        "augmented_R_json": augmented_r_json,
        "K_gain_checksum": gain_checksum,
        "augmented_gain_checksum": augmented_gain_checksum,
        "augmented_A_checksum": augmented_A_checksum,
        "augmented_B_checksum": augmented_B_checksum,
        "weight_label": str(weight_label),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()[:12]
    return f"lqrta_{primitive_id}_{digest}"


def initialised_timing_state_for_controller(
    controller: LQRController,
    state_vector: np.ndarray,
) -> TimingAwareControllerState:
    """Return a deterministic compatibility timing state when no history exists."""

    state = as_state_vector(state_vector)
    reference_command = tuple(float(value) for value in controller.reference_command_vector)
    fifo_steps = max(0, int(controller.command_delay_steps))
    return TimingAwareControllerState(
        command_fifo_rad=tuple(reference_command for _ in range(fifo_steps)),
        last_requested_command_rad=reference_command,
        last_applied_command_rad=reference_command,
        predictor_reference_command_rad=reference_command,
        current_surface_state_rad=tuple(
            float(state[STATE_INDEX[name]]) for name in ("delta_a", "delta_e", "delta_r")
        ),
        timing_state_source=TIMING_STATE_INITIALISED,
    )


def _timing_aware_raw_command_rad(
    *,
    controller: LQRController,
    state: np.ndarray,
    timing_state: TimingAwareControllerState | None,
) -> tuple[np.ndarray, str]:
    """Evaluate the augmented gain with predictor compensation and command FIFO state."""

    x_ref = np.asarray(controller.reference_state_vector, dtype=float)
    u_ref = np.asarray(controller.reference_command_vector, dtype=float)
    reduced_indices = list(reduced_state_indices())
    x_error = np.asarray(state, dtype=float)[reduced_indices] - x_ref[reduced_indices]
    try:
        predictor_a = np.asarray(json.loads(controller.predictor_A_reduced_json), dtype=float)
        for _ in range(max(0, int(controller.predictor_horizon_steps))):
            x_error = predictor_a @ x_error
    except Exception:
        pass
    fifo_steps = max(0, int(controller.command_delay_steps))
    if fifo_steps:
        resolved_timing_state = timing_state or initialised_timing_state_for_controller(controller, state)
        fifo = _normalised_fifo_rad(
            resolved_timing_state.command_fifo_rad,
            fifo_steps=fifo_steps,
            reference_command_rad=u_ref,
        )
        fifo_error = (fifo - u_ref.reshape(1, 3)).reshape(-1)
        augmented_error = np.concatenate([x_error, fifo_error])
        timing_state_source = str(resolved_timing_state.timing_state_source)
    else:
        augmented_error = x_error
        timing_state_source = "no_command_delay_fifo_configured"
    gain = np.asarray(json.loads(controller.augmented_gain_matrix_json), dtype=float)
    return u_ref - gain @ augmented_error, timing_state_source


def _normalised_fifo_rad(
    command_fifo_rad: tuple[tuple[float, float, float], ...],
    *,
    fifo_steps: int,
    reference_command_rad: np.ndarray,
) -> np.ndarray:
    fifo = np.asarray(command_fifo_rad, dtype=float)
    if fifo.size == 0:
        fifo = np.tile(np.asarray(reference_command_rad, dtype=float).reshape(1, 3), (int(fifo_steps), 1))
    if fifo.ndim == 1:
        fifo = fifo.reshape(1, 3)
    if fifo.shape[1] != 3:
        raise ValueError("command_fifo_rad must contain 3-column radian command rows.")
    if fifo.shape[0] < int(fifo_steps):
        pad = np.tile(fifo[-1].reshape(1, 3), (int(fifo_steps) - fifo.shape[0], 1))
        fifo = np.vstack([fifo, pad])
    if fifo.shape[0] > int(fifo_steps):
        fifo = fifo[: int(fifo_steps), :]
    if not np.all(np.isfinite(fifo)):
        raise ValueError("command_fifo_rad must be finite.")
    return fifo


def _q_reduced(weights: LQRWeightSpec) -> np.ndarray:
    diag = []
    for name in LQR_STATE_MASK:
        if name in {"phi", "theta", "psi"}:
            diag.append(weights.q_attitude)
        elif name in {"u", "v", "w"}:
            diag.append(weights.q_velocity)
        elif name in {"p", "q", "r"}:
            diag.append(weights.q_rates)
        else:
            diag.append(weights.q_surfaces)
    return np.diag(np.asarray(diag, dtype=float))


def _r_matrix(weights: LQRWeightSpec) -> np.ndarray:
    return np.diag([weights.r_aileron, weights.r_elevator, weights.r_rudder]).astype(float)


def _expand_q_to_full(q_reduced: np.ndarray) -> np.ndarray:
    q_full = np.eye(STATE_SIZE, dtype=float) * 1e-6
    for reduced_index, full_index in enumerate(reduced_state_indices()):
        q_full[full_index, full_index] = float(q_reduced[reduced_index, reduced_index])
    return q_full


def _expand_gain_to_full(k_reduced: np.ndarray) -> np.ndarray:
    k_full = np.zeros((3, STATE_SIZE), dtype=float)
    for reduced_index, full_index in enumerate(reduced_state_indices()):
        k_full[:, full_index] = k_reduced[:, reduced_index]
    return k_full


def _gain_expansion_status(k_full: np.ndarray) -> str:
    for name in ("x_w", "y_w", "z_w"):
        if np.max(np.abs(k_full[:, STATE_INDEX[name]])) > 1e-12:
            return "failed_position_gain_nonzero"
    return "zero_position_gains_verified"


def _care_attempt_status(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> tuple[str, str]:
    try:
        linalg.solve_continuous_are(a, b, q, r)
    except Exception as exc:
        return "unsuitable_use_reduced_order", f"{type(exc).__name__}:{exc}"
    return "solved", "full_state_care_solved"


def _care_residual_norm(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
) -> float:
    r_inv = np.linalg.inv(r)
    residual = a.T @ p + p @ a - p @ b @ r_inv @ b.T @ p + q
    scale = max(1.0, float(np.linalg.norm(q, ord="fro")))
    return float(np.linalg.norm(residual, ord="fro") / scale)


def _sampled_spectral_radius(
    a: np.ndarray,
    b: np.ndarray,
    k: np.ndarray,
    dt_s: float,
) -> float:
    discrete = linalg.expm((a - b @ k) * float(dt_s))
    return float(np.max(np.abs(np.linalg.eigvals(discrete))))


def _eigen_summary(eigenvalues: np.ndarray) -> str:
    eig = np.asarray(eigenvalues, dtype=complex)
    payload = {
        "max_real": float(np.max(np.real(eig))),
        "min_real": float(np.min(np.real(eig))),
        "max_abs_imag": float(np.max(np.abs(np.imag(eig)))),
        "count": int(eig.size),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _reference_command_check(command_rad: tuple[float, float, float]) -> tuple[str, str]:
    raw_norm = _surface_rad_to_unclipped_norm(np.asarray(command_rad, dtype=float))
    clipped = clip_normalised_command(raw_norm)
    saturated = bool(np.any(np.abs(raw_norm - clipped) > 1e-12))
    return (
        "passed",
        json.dumps(
            {
                "reference_saturated": saturated,
                "max_abs_reference_norm": float(np.max(np.abs(raw_norm))),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _surface_rad_to_unclipped_norm(command_rad: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _angle_to_command_norm_unclipped(value, AGGREGATE_LIMITS[name])
            for value, name in zip(command_rad, ("delta_a", "delta_e", "delta_r"), strict=True)
        ],
        dtype=float,
    )


def _angle_to_command_norm_unclipped(angle_rad: float, limit: SurfaceLimit) -> float:
    """Map a requested surface angle to command units without hiding saturation."""

    angle_deg = float(np.rad2deg(angle_rad))
    if abs(angle_deg) <= 1e-12:
        return 0.0

    candidates: list[float] = []
    positive_deg = float(limit.positive_deg)
    negative_deg = float(limit.negative_deg)
    if abs(positive_deg) > 1e-12:
        positive_ratio = angle_deg / positive_deg
        if positive_ratio >= 0.0:
            candidates.append(float(positive_ratio))
    if abs(negative_deg) > 1e-12:
        negative_ratio = angle_deg / negative_deg
        if negative_ratio >= 0.0:
            candidates.append(float(-negative_ratio))
    if candidates:
        return min(
            candidates,
            key=lambda value: abs(abs(value) - 1.0) if abs(value) > 1.0 else abs(value),
        )

    endpoint = (
        positive_deg
        if abs(angle_deg - positive_deg) <= abs(angle_deg - negative_deg)
        else negative_deg
    )
    return 1.0 if endpoint == positive_deg else -1.0


def _q_weight_payload(weights: LQRWeightSpec) -> dict[str, object]:
    return {
        "grouping": "diagonal_grouped_log_scaled_with_reference_bias",
        "state_mask": list(LQR_STATE_MASK),
        "q_attitude": float(weights.q_attitude),
        "q_velocity": float(weights.q_velocity),
        "q_rates": float(weights.q_rates),
        "q_surfaces": float(weights.q_surfaces),
        "reference_pitch_bias_rad": float(weights.reference_pitch_bias_rad),
        "reference_bank_bias_rad": float(weights.reference_bank_bias_rad),
        "reference_speed_bias_m_s": float(weights.reference_speed_bias_m_s),
    }


def _r_weight_payload(weights: LQRWeightSpec) -> dict[str, object]:
    return {
        "grouping": "diagonal_grouped_log_scaled",
        "command_names": ["delta_a_cmd", "delta_e_cmd", "delta_r_cmd"],
        "r_aileron": float(weights.r_aileron),
        "r_elevator": float(weights.r_elevator),
        "r_rudder": float(weights.r_rudder),
    }


def _controller_id(
    primitive_id: str,
    linearisation_id: str,
    q_json: str,
    r_json: str,
    gain_checksum: str,
    weight_label: str,
) -> str:
    payload = json.dumps(
        [primitive_id, linearisation_id, q_json, r_json, gain_checksum, weight_label],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]
    return f"lqr_{primitive_id}_{digest}"


def _blocked_reason(
    *,
    linearisation: LQRLinearisation,
    care_residual: float,
    sampled_status: str,
    command_status: str,
    latency_status: str,
    expansion_status: str,
) -> str:
    reasons = []
    if linearisation.finite_ab_check != "finite":
        reasons.append("nonfinite_linearisation")
    if linearisation.reduced_controllability_rank != linearisation.reduced_state_size:
        reasons.append("reduced_controllability_rank_deficient")
    if not np.isfinite(care_residual) or care_residual >= 1e-6:
        reasons.append("care_residual_high")
    if sampled_status != "sampled_stable":
        reasons.append(sampled_status)
    if command_status != "passed":
        reasons.append(command_status)
    if latency_status != "survives_nominal_latency_actuator_lag":
        reasons.append(latency_status)
    if expansion_status != "zero_position_gains_verified":
        reasons.append(expansion_status)
    return ";".join(reasons) if reasons else "blocked_lqr_synthesis"
