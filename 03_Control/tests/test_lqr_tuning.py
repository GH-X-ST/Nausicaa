from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lqr_controller import lqr_command_for_state, synthesize_lqr_controller
from lqr_linearisation import LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S
from lqr_tuning import W01_TUNING_METHOD_VERSION, candidate_weight_specs, tuning_candidates_for_primitive
from prim_cat import ACTIVE_PRIMITIVE_IDS, primitive_by_id
from primitive_variant_registry import (
    ENTRY_ROLE_BY_PRIMITIVE_ID,
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
    TRANSITION_OBJECT_ENTRY_ROLE,
    entry_role_rejection_fields,
    primitive_controller_variant,
    start_family_is_compatible,
    validate_variant_controller_match,
)


def test_w01_candidate_generation_keeps_multiple_variants_per_primitive() -> None:
    primitive = primitive_by_id("glide")
    candidates = tuning_candidates_for_primitive(primitive, candidate_count=3)

    assert len(candidates) == 3
    assert len({candidate.controller_id for candidate in candidates}) == 3
    assert {candidate.tuning_stage for candidate in candidates} == {"W01"}


def test_w01_qr_generator_is_structured_32_candidate_transition_training() -> None:
    assert W01_TUNING_METHOD_VERSION == "w01_transition_robust_reference_v8"
    for primitive_id in ACTIVE_PRIMITIVE_IDS:
        specs = candidate_weight_specs(primitive_id=primitive_id, candidate_count=32)
        labels = [spec.weight_label for spec in specs]

        assert len(specs) == 32
        assert labels[0].endswith("_robust_anchor_nominal_ref_nominal_000")
        assert labels[1].endswith("_robust_anchor_attitude_heavy_ref_pitch_up_001")
        assert labels[7].endswith("_robust_anchor_balanced_agile_ref_right_bias_007")
        assert all("_robust_lhs_logqr_refbias_" in label for label in labels[8:])
        assert not any("launch_capture" in label for label in labels)
        assert len({spec.weight_label for spec in specs}) == 32
        assert specs[0].reference_pitch_bias_rad == 0.0
        assert specs[0].reference_bank_bias_rad == 0.0
        assert specs[0].reference_roll_rate_bias_rad_s == 0.0
        assert specs[0].reference_speed_bias_m_s == 0.0
        assert all(spec.reference_speed_bias_m_s == 0.0 for spec in specs)
        assert any(
            abs(spec.reference_pitch_bias_rad) > 0.0
            or abs(spec.reference_bank_bias_rad) > 0.0
            for spec in specs[1:]
        )


def test_mild_turn_reference_biases_are_not_active_turn_expression_objectives() -> None:
    left_specs = candidate_weight_specs(primitive_id="mild_turn_left", candidate_count=32)
    right_specs = candidate_weight_specs(primitive_id="mild_turn_right", candidate_count=32)
    glide_specs = candidate_weight_specs(primitive_id="glide", candidate_count=32)

    assert left_specs[0].reference_bank_bias_rad == 0.0
    assert right_specs[0].reference_bank_bias_rad == 0.0
    assert any(spec.reference_bank_bias_rad < 0.0 for spec in left_specs)
    assert any(spec.reference_bank_bias_rad > 0.0 for spec in left_specs)
    assert any(spec.reference_bank_bias_rad < 0.0 for spec in right_specs)
    assert any(spec.reference_bank_bias_rad > 0.0 for spec in right_specs)
    assert any(spec.reference_bank_bias_rad < 0.0 for spec in glide_specs)
    assert any(spec.reference_bank_bias_rad > 0.0 for spec in glide_specs)
    assert all(
        spec.reference_roll_rate_bias_rad_s == 0.0
        for spec in [*left_specs, *right_specs, *glide_specs]
    )
    assert all(spec.reference_speed_bias_m_s == 0.0 for spec in [*left_specs, *right_specs])


def test_reference_bias_changes_controller_identity_and_is_serialised() -> None:
    primitive = primitive_by_id("glide")
    nominal = synthesize_lqr_controller(
        primitive,
        weight_spec=candidate_weight_specs(primitive_id="glide", candidate_count=1)[0],
        local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    )
    biased_spec = candidate_weight_specs(primitive_id="glide", candidate_count=2)[1]
    biased = synthesize_lqr_controller(
        primitive,
        weight_spec=biased_spec,
        local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    )
    q_payload = json.loads(biased.lqr_Q_weights_json)

    assert nominal.controller_id != biased.controller_id
    assert nominal.linearisation_id != biased.linearisation_id
    assert q_payload["reference_pitch_bias_rad"] == biased_spec.reference_pitch_bias_rad
    assert q_payload["reference_bank_bias_rad"] == biased_spec.reference_bank_bias_rad
    assert q_payload["reference_roll_rate_bias_rad_s"] == biased_spec.reference_roll_rate_bias_rad_s
    assert q_payload["reference_speed_bias_m_s"] == biased_spec.reference_speed_bias_m_s
    assert q_payload["longitudinal_speed_error_policy"] == "passive_u_error_zeroed_speed_is_scheduling_only_v1"


def test_lqr_command_does_not_chase_longitudinal_speed_error_directly() -> None:
    primitive = primitive_by_id("glide")
    controller = synthesize_lqr_controller(
        primitive,
        weight_spec=candidate_weight_specs(primitive_id="glide", candidate_count=1)[0],
        local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    )
    state = list(controller.reference_state_vector)
    slow = state.copy()
    fast = state.copy()
    slow[6] -= 1.0
    fast[6] += 1.0

    slow_command = lqr_command_for_state(controller=controller, state_vector=slow)
    fast_command = lqr_command_for_state(controller=controller, state_vector=fast)

    assert slow_command.raw_command_rad == pytest.approx(fast_command.raw_command_rad)


def test_variant_registry_schema_stable_ids_and_checksum_validation() -> None:
    primitive = primitive_by_id("glide")
    controller = synthesize_lqr_controller(
        primitive,
        weight_spec=candidate_weight_specs(primitive_id="glide", candidate_count=1)[0],
        local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    )
    variant_a = primitive_controller_variant(primitive=primitive, controller=controller)
    variant_b = primitive_controller_variant(primitive=primitive, controller=controller)
    changed = replace(controller, lqr_gain_checksum="0" * 64)

    assert variant_a.primitive_variant_id == variant_b.primitive_variant_id
    assert variant_a.primitive_variant_id.startswith("primvar_glide_transition_object_")
    assert variant_a.K_gain_checksum == controller.lqr_gain_checksum
    assert variant_a.local_lqr_reference_speed_m_s == pytest.approx(LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S)
    assert variant_a.local_lqr_speed_bin_id == "speed_bin_5p0_m_s"
    assert "9.0" in variant_a.local_lqr_speed_grid_m_s
    assert variant_a.controller_design_role == "active_timing_aware_w01"
    assert variant_a.timing_augmentation_type == "actuator_surface_state_command_fifo_predictor_compensated"
    assert variant_a.timing_design_version == "predictor_compensated_augmented_discrete_lqr_v1"
    assert variant_a.command_delay_steps >= 1
    assert variant_a.actuator_state_count == 3
    assert variant_a.command_delay_state_count == 3 * variant_a.command_delay_steps
    assert variant_a.augmented_state_size > 12
    assert variant_a.augmented_A_checksum
    assert variant_a.augmented_B_checksum
    assert variant_a.augmented_gain_checksum
    assert variant_a.timing_aware_synthesis_level == "predictor_compensated_augmented_discrete_lqr"
    assert variant_a.delayed_state_lqr_augmentation_status == (
        "predictor_compensation_only_no_full_delayed_state_augmentation"
    )
    validate_variant_controller_match(variant_a, controller)
    with pytest.raises(ValueError, match="gain checksum"):
        validate_variant_controller_match(variant_a, changed)


def test_transition_object_entry_role_accepts_all_active_start_families() -> None:
    assert {ENTRY_ROLE_BY_PRIMITIVE_ID[primitive_id] for primitive_id in ACTIVE_PRIMITIVE_IDS} == {
        TRANSITION_OBJECT_ENTRY_ROLE
    }
    for family in (
        "launch_gate",
        "inflight_nominal",
        "inflight_lift_region",
        "inflight_boundary_near",
        "inflight_recovery_edge",
    ):
        assert start_family_is_compatible(entry_role=TRANSITION_OBJECT_ENTRY_ROLE, start_state_family=family)

    fields = entry_role_rejection_fields(entry_role="launch_capable", start_state_family="inflight_nominal")
    assert fields["entry_check_status"] == ENTRY_ROLE_REJECTION_STATUS
    assert fields["failure_label"] == ENTRY_ROLE_REJECTION_LABEL
    assert fields["outcome_class"] == "rejected"


def test_blocked_variant_metadata_is_retained() -> None:
    primitive = primitive_by_id("glide")
    controller = synthesize_lqr_controller(
        primitive,
        local_reference_speed_m_s=LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
    )
    blocked = replace(
        controller,
        lqr_synthesis_status="blocked_lqr_synthesis",
        lqr_blocked_reason="unit_test_blocked_reason",
    )
    variant = primitive_controller_variant(primitive=primitive, controller=blocked)

    assert variant.lqr_synthesis_status == "blocked_lqr_synthesis"
    assert variant.lqr_blocked_reason == "unit_test_blocked_reason"
