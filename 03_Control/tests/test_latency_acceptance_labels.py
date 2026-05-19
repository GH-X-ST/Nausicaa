from __future__ import annotations

import numpy as np
import pytest

import scenario_contract
from latency import (
    LATENCY_CASES,
    LATENCY_PASS_LABELS,
    format_actuator_tau_s,
    latency_acceptance_scope,
    latency_audit_fields_from_case,
    latency_case_config,
    latency_pass_label_for_paired_comparison,
    latency_pass_label_for_single_run,
)
from primitive_library_generators import primitive_candidate_inventory
from primitive_library_schema import PrimitiveLibraryConfig


EXPECTED_LATENCY_CASES = ("none", "actuator_lag_only", "nominal", "conservative")
EXPECTED_AUDIT_FIELDS = {
    "latency_case",
    "state_feedback_delay_s",
    "command_onset_delay_s",
    "command_transport_delay_s",
    "actuator_tau_s",
    "actuator_t50_s",
    "actuator_t90_s",
    "latency_jitter_s",
    "timing_model_version",
    "latency_pass_label",
}


def test_canonical_latency_labels_are_exact() -> None:
    assert LATENCY_CASES == EXPECTED_LATENCY_CASES
    assert scenario_contract.LATENCY_CASES == EXPECTED_LATENCY_CASES
    assert LATENCY_PASS_LABELS == (
        "ideal_only",
        "nominal_pass",
        "nominal_fail",
        "conservative_pass",
        "conservative_fail",
        "ideal_only_latency_failed",
    )


def test_latency_case_configs_and_audit_fields() -> None:
    expected_pass_labels = {
        "none": "ideal_only",
        "actuator_lag_only": "ideal_only",
        "nominal": "nominal_pass",
        "conservative": "conservative_pass",
    }
    for latency_case in EXPECTED_LATENCY_CASES:
        config = latency_case_config(latency_case)
        fields = latency_audit_fields_from_case(config)

        assert set(fields) == EXPECTED_AUDIT_FIELDS
        assert fields["latency_case"] == latency_case
        assert fields["latency_jitter_s"] == 0.0
        assert fields["latency_pass_label"] == expected_pass_labels[latency_case]
        assert fields["actuator_tau_s"] == format_actuator_tau_s(config.actuator_tau_s)

    nominal = latency_case_config("nominal")
    conservative = latency_case_config("conservative")
    assert nominal.state_feedback_delay_s == pytest.approx(0.0229)
    assert nominal.command_onset_delay_s == pytest.approx(0.073)
    assert nominal.command_transport_delay_s == 0.0
    assert nominal.actuator_t50_s == pytest.approx(0.108)
    assert nominal.actuator_t90_s == pytest.approx(0.130)
    assert conservative.command_onset_delay_s == pytest.approx(0.073)
    assert conservative.command_transport_delay_s == 0.0
    assert conservative.actuator_t50_s == pytest.approx(0.151)
    assert conservative.actuator_tau_s[0] == pytest.approx((0.151 - 0.073) / np.log(2.0))


def test_latency_audit_override_matches_active_tau_semantics() -> None:
    active_tau = (0.06, 0.06, 0.06)

    none_fields = latency_audit_fields_from_case(
        latency_case_config("none"),
        active_actuator_tau_s=active_tau,
    )
    assert none_fields["actuator_tau_s"] == "0.000000000;0.000000000;0.000000000"
    assert none_fields["actuator_t50_s"] == 0.0
    assert none_fields["actuator_t90_s"] == 0.0

    lag_fields = latency_audit_fields_from_case(
        latency_case_config("actuator_lag_only"),
        active_actuator_tau_s=active_tau,
    )
    assert lag_fields["actuator_tau_s"] == format_actuator_tau_s(active_tau)
    assert lag_fields["actuator_t50_s"] == pytest.approx(0.06 * np.log(2.0))
    assert lag_fields["actuator_t90_s"] == pytest.approx(0.06 * np.log(10.0))

    nominal = latency_case_config("nominal")
    nominal_fields = latency_audit_fields_from_case(
        nominal,
        active_actuator_tau_s=active_tau,
    )
    assert nominal_fields["actuator_tau_s"] == format_actuator_tau_s(active_tau)
    assert nominal_fields["actuator_t50_s"] == pytest.approx(nominal.actuator_t50_s)
    assert nominal_fields["actuator_t90_s"] == pytest.approx(nominal.actuator_t90_s)

    conservative = latency_case_config("conservative")
    conservative_fields = latency_audit_fields_from_case(
        conservative,
        active_actuator_tau_s=conservative.actuator_tau_s,
    )
    assert conservative_fields["actuator_tau_s"] == format_actuator_tau_s(
        conservative.actuator_tau_s
    )
    assert conservative_fields["actuator_t50_s"] == pytest.approx(
        conservative.actuator_t50_s
    )
    assert conservative_fields["actuator_t90_s"] == pytest.approx(
        conservative.actuator_t90_s
    )


def test_single_run_latency_labels_and_acceptance_scopes_are_separate() -> None:
    assert latency_acceptance_scope("none") == "ideal_ablation_only"
    assert latency_acceptance_scope("actuator_lag_only") == "actuator_lag_only_ablation"
    assert (
        latency_acceptance_scope("nominal")
        == "command_path_nominal_no_feedback_controller"
    )
    assert (
        latency_acceptance_scope("conservative")
        == "command_path_conservative_no_feedback_controller"
    )
    assert latency_pass_label_for_single_run("none", False) == "ideal_only"
    assert latency_pass_label_for_single_run("actuator_lag_only", True) == "ideal_only"
    assert latency_pass_label_for_single_run("nominal", True) == "nominal_pass"
    assert latency_pass_label_for_single_run("nominal", False) == "nominal_fail"
    assert latency_pass_label_for_single_run("conservative", True) == "conservative_pass"
    assert latency_pass_label_for_single_run("conservative", False) == "conservative_fail"
    assert {
        latency_pass_label_for_single_run(case, accepted)
        for case in EXPECTED_LATENCY_CASES
        for accepted in (False, True)
    }.isdisjoint({"ideal_only_latency_failed"})
    assert (
        latency_pass_label_for_paired_comparison(True, False)
        == "ideal_only_latency_failed"
    )


def test_unknown_latency_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="latency_case"):
        latency_case_config("robust_upper")


def test_baseline_primitive_library_config_remains_backward_compatible() -> None:
    config = PrimitiveLibraryConfig(
        run_id=2,
        targets_deg=(15.0, 30.0),
        wind_fidelities=("W0",),
        updraft_configs=("none",),
        start_conditions=("favourable", "mid_arena"),
        direction_signs=(1,),
    )
    candidates = primitive_candidate_inventory(config)

    assert candidates
    assert {candidate.wind_fidelity for candidate in candidates} == {"W0"}
    assert {candidate.start_condition for candidate in candidates} == {
        "favourable",
        "mid_arena",
    }


def test_run_002_to_run_006_modules_still_import() -> None:
    import run_primitive_library_outer_loop  # noqa: F401
    import run_primitive_library_pass  # noqa: F401
    import run_primitive_library_shortlist  # noqa: F401
    import run_primitive_library_w3_stress  # noqa: F401
    import run_primitive_library_governor_seed  # noqa: F401

    assert np.isfinite(latency_case_config("nominal").state_feedback_delay_s)
