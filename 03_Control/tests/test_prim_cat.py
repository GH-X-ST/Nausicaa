from __future__ import annotations

import json

from prim_cat import (
    ACTIVE_PRIMITIVE_IDS,
    PrimitiveDefinition,
    active_primitive_catalogue,
    primitive_by_id,
    primitive_parameters_json,
)


def test_active_primitive_catalogue_has_unique_expected_ids() -> None:
    catalogue = active_primitive_catalogue()
    ids = tuple(primitive.primitive_id for primitive in catalogue)

    assert ids == ACTIVE_PRIMITIVE_IDS
    assert len(set(ids)) == len(ids) == 8


def test_every_primitive_has_required_schema_and_claim_boundary() -> None:
    for primitive in active_primitive_catalogue():
        assert isinstance(primitive, PrimitiveDefinition)
        assert primitive.primitive_family
        assert primitive.parameters
        assert primitive.entry_conditions
        assert primitive.exit_checks
        assert primitive.metrics_to_record
        assert primitive.failure_labels
        assert primitive.finite_horizon_s > 0.0
        assert primitive.claim_status == "simulation_only"
        assert primitive.controller_family == "lqr"
        assert primitive.controller_id == f"lqrta_{primitive.primitive_id}_resolved_by_synthesis"
        assert primitive.controller_id_policy == "resolved_by_lqr_controller"
        assert primitive.nominal_controller_label == f"lqrta_{primitive.primitive_id}_timing_aware_nominal"
        assert primitive.controller_mode == "lqr_local_feedback"
        assert primitive.feedback_mode == "predictor_compensated_augmented_discrete_lqr"


def test_primitive_lookup_and_parameter_json_are_stable() -> None:
    primitive = primitive_by_id("lift_dwell_arc")
    payload = json.loads(primitive_parameters_json(primitive))

    assert primitive.primitive_family == "lift_dwell"
    assert "arc_bank_rad" in payload
    assert payload["arc_bank_rad"]["unit"] == "rad"


def test_active_primitive_text_has_no_environment_algorithm_split() -> None:
    text = " ".join(str(primitive) for primitive in active_primitive_catalogue()).lower()

    assert "single_fan" not in text
    assert "four_fan" not in text
    assert "reachable" not in text
    assert "final dependency" not in text
