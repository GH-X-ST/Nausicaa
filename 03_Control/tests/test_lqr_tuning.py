from __future__ import annotations

from dataclasses import replace

import pytest

from lqr_controller import synthesize_lqr_controller
from lqr_tuning import candidate_weight_specs, tuning_candidates_for_primitive
from prim_cat import ACTIVE_PRIMITIVE_IDS, primitive_by_id
from primitive_variant_registry import (
    ENTRY_ROLE_BY_PRIMITIVE_ID,
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
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


def test_variant_registry_schema_stable_ids_and_checksum_validation() -> None:
    primitive = primitive_by_id("glide")
    controller = synthesize_lqr_controller(
        primitive,
        weight_spec=candidate_weight_specs(primitive_id="glide", candidate_count=1)[0],
    )
    variant_a = primitive_controller_variant(primitive=primitive, controller=controller)
    variant_b = primitive_controller_variant(primitive=primitive, controller=controller)
    changed = replace(controller, lqr_gain_checksum="0" * 64)

    assert variant_a.primitive_variant_id == variant_b.primitive_variant_id
    assert variant_a.primitive_variant_id.startswith("primvar_glide_launch_capable_")
    assert variant_a.K_gain_checksum == controller.lqr_gain_checksum
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


def test_correct_first_pass_entry_roles_and_launch_gate_rejections() -> None:
    assert ENTRY_ROLE_BY_PRIMITIVE_ID == {
        "glide": "launch_capable",
        "lift_entry": "inflight_only",
        "lift_dwell_arc": "inflight_only",
        "mild_turn_left": "inflight_only",
        "mild_turn_right": "inflight_only",
        "energy_retaining_bank": "inflight_only",
        "recovery": "terminal_or_recovery",
        "safe_exit_or_recovery_handoff": "terminal_or_recovery",
    }
    assert set(ENTRY_ROLE_BY_PRIMITIVE_ID) == set(ACTIVE_PRIMITIVE_IDS)
    assert start_family_is_compatible(entry_role="launch_capable", start_state_family="launch_gate")
    assert not start_family_is_compatible(entry_role="inflight_only", start_state_family="launch_gate")
    assert not start_family_is_compatible(entry_role="terminal_or_recovery", start_state_family="launch_gate")

    fields = entry_role_rejection_fields(entry_role="inflight_only", start_state_family="launch_gate")
    assert fields["entry_check_status"] == ENTRY_ROLE_REJECTION_STATUS
    assert fields["failure_label"] == ENTRY_ROLE_REJECTION_LABEL
    assert fields["outcome_class"] == "rejected"


def test_blocked_variant_metadata_is_retained() -> None:
    primitive = primitive_by_id("glide")
    controller = synthesize_lqr_controller(primitive)
    blocked = replace(
        controller,
        lqr_synthesis_status="blocked_lqr_synthesis",
        lqr_blocked_reason="unit_test_blocked_reason",
    )
    variant = primitive_controller_variant(primitive=primitive, controller=blocked)

    assert variant.lqr_synthesis_status == "blocked_lqr_synthesis"
    assert variant.lqr_blocked_reason == "unit_test_blocked_reason"
