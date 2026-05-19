from __future__ import annotations

import numpy as np

from arena import ArenaConfig, safe_bounds
from arena_contract import TRUE_SAFE_BOUNDS
from primitive_library_schema import (
    CANDIDATE_CLASSES,
    COVERAGE_STATUSES,
    ENVELOPE_STATUSES,
    ENTRY_ENVELOPE_STATUSES,
    EVALUATION_STATUSES,
    EVIDENCE_SOURCES,
    PRIMITIVE_FAMILIES,
    RECOVERY_BASIS_VALUES,
    TARGET_LADDER_DEG,
    TRUE_SAFE_BOUNDS_M,
    UPDRAFT_CONFIGS,
    WIND_FIDELITIES,
    Z_OUTLET_M,
    PrimitiveEvidenceRow,
    PrimitiveLibraryConfig,
    classify_candidate,
    classify_candidate_semantics,
    classify_wind_query_region,
    entry_clearance_metrics,
    path_metrics,
    target_heading_band_deg,
)


def test_library_constants_and_true_safe_bounds_are_locked() -> None:
    assert TARGET_LADDER_DEG == (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
    assert PRIMITIVE_FAMILIES == (
        "glide",
        "recovery",
        "mild_bank",
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    )
    assert UPDRAFT_CONFIGS == ("none", "U1_single_fan", "U4_four_fan")
    assert WIND_FIDELITIES == ("W0", "W1", "W2", "W3")
    assert EVALUATION_STATUSES == ("evaluated", "not_evaluated_model_missing", "model_unavailable")
    assert "deterministic_seed_replay" in EVIDENCE_SOURCES
    assert "dry_recoverable" in RECOVERY_BASIS_VALUES
    assert "outside_entry_envelope_governor_reject" in ENTRY_ENVELOPE_STATUSES
    assert "candidate_family_needs_refinement" in ENVELOPE_STATUSES
    assert "uncovered_needs_refinement" in COVERAGE_STATUSES
    assert "boundary_evidence" in CANDIDATE_CLASSES
    assert Z_OUTLET_M == 0.330

    assert TRUE_SAFE_BOUNDS_M["z_w"] == (0.4, 3.5)
    assert TRUE_SAFE_BOUNDS.z_w_m == (0.4, 3.5)
    assert safe_bounds(ArenaConfig())["z_w"] == (0.4, 3.5)


def test_heading_bands_and_path_clearance_metrics() -> None:
    assert target_heading_band_deg(15.0) == (13.0, 17.0)
    assert target_heading_band_deg(30.0) == (27.0, 33.0)

    positions = np.array(
        [
            [1.3, 2.2, 1.8],
            [2.0, 2.5, 1.9],
            [2.6, 2.1, 1.7],
        ],
        dtype=float,
    )
    path = path_metrics(positions)
    clearance = entry_clearance_metrics(positions)

    assert path["path_length_xy_m"] > 0.0
    assert path["path_length_3d_m"] >= path["path_length_xy_m"]
    assert path["forward_displacement_m"] > 0.0
    assert clearance["entry_clearance_required_x_plus_m"] > 0.0
    assert clearance["floor_margin_required_m"] > 0.0
    assert clearance["ceiling_margin_required_m"] > 0.0


def test_model_missing_is_not_physical_boundary_evidence() -> None:
    candidate_class, failure_label, limiter = classify_candidate(
        {
            "evaluation_status": "not_evaluated_model_missing",
            "target_heading_deg": 30.0,
        }
    )

    assert candidate_class == "not_evaluated"
    assert failure_label == "not_evaluated_model_missing"
    assert limiter == "model_unavailable"
    semantics = classify_candidate_semantics({"evaluation_status": "not_evaluated_model_missing"})
    assert semantics["library_growth_trigger"] is False
    assert semantics["coverage_status"] == "not_evaluated_model_unavailable"


def test_candidate_classification_for_wrong_heading_and_updraft_pending() -> None:
    wrong_heading_class, wrong_heading_failure, _ = classify_candidate(
        {
            "evaluation_status": "evaluated",
            "target_heading_deg": 30.0,
            "heading_band_pass": False,
            "true_safe_trajectory": True,
            "terminal_speed_m_s": 5.5,
            "speed_min_m_s": 4.5,
            "alpha_max_deg": 20.0,
            "beta_max_deg": 10.0,
            "rate_max_rad_s": 2.0,
            "saturation_fraction": 0.1,
            "wind_fidelity": "W0",
            "recovery_class": "dry_recoverable",
        }
    )
    assert wrong_heading_class == "boundary_evidence"
    assert wrong_heading_failure == "target_miss"

    pending_class, pending_failure, limiter = classify_candidate(
        {
            "evaluation_status": "evaluated",
            "target_heading_deg": 30.0,
            "heading_band_pass": True,
            "true_safe_trajectory": True,
            "terminal_speed_m_s": 4.3,
            "speed_min_m_s": 3.7,
            "alpha_max_deg": 25.0,
            "beta_max_deg": 12.0,
            "rate_max_rad_s": 2.5,
            "saturation_fraction": 0.2,
            "wind_fidelity": "W0",
            "recovery_class": "updraft_pending",
        }
    )
    assert pending_class == "w0_updraft_pending_target_candidate"
    assert pending_failure == "dry_recovery_pending"
    assert limiter == "updraft_condition_required"


def test_w1_w2_dry_recoverable_updraft_environment_is_commandable() -> None:
    semantics = classify_candidate_semantics(
        {
            "evaluation_status": "evaluated",
            "target_heading_deg": 30.0,
            "heading_band_pass": True,
            "true_safe_trajectory": True,
            "terminal_speed_m_s": 5.8,
            "speed_min_m_s": 4.6,
            "alpha_max_deg": 25.0,
            "beta_max_deg": 10.0,
            "rate_max_rad_s": 2.0,
            "saturation_fraction": 0.1,
            "wind_fidelity": "W2",
            "recovery_class": "dry_recoverable",
            "lift_belief_condition": "U4_four_fan_W2_available",
        }
    )

    assert semantics["candidate_class"] == "updraft_assisted_commandable"
    assert semantics["failure_label"] == "success"
    assert semantics["recovery_basis"] == "dry_recoverable"
    assert semantics["evaluated_under_updraft_environment"] is True
    assert semantics["coverage_status"] == "covered_by_existing_envelope"
    assert semantics["library_growth_trigger"] is False


def test_envelope_status_for_entry_reject_and_target_miss() -> None:
    entry_reject = classify_candidate_semantics(
        {
            "evaluation_status": "evaluated",
            "target_heading_deg": 30.0,
            "heading_band_pass": True,
            "true_safe_trajectory": False,
            "terminal_speed_m_s": 5.8,
            "speed_min_m_s": 4.6,
            "alpha_max_deg": 25.0,
            "beta_max_deg": 10.0,
            "rate_max_rad_s": 2.0,
            "saturation_fraction": 0.1,
            "wind_fidelity": "W0",
            "recovery_class": "dry_recoverable",
            "start_condition": "mid_arena",
            "margin_consumption_x_m": 1.2,
        }
    )
    assert entry_reject["envelope_status"] == "outside_entry_envelope_governor_reject"
    assert entry_reject["coverage_status"] == "uncovered_governor_reject"
    assert entry_reject["library_growth_trigger"] is False
    assert entry_reject["growth_reason"] == "entry_clearance_insufficient"

    target_miss = classify_candidate_semantics(
        {
            "evaluation_status": "evaluated",
            "target_heading_deg": 30.0,
            "heading_band_pass": False,
            "true_safe_trajectory": True,
            "terminal_speed_m_s": 5.8,
            "speed_min_m_s": 4.6,
            "alpha_max_deg": 25.0,
            "beta_max_deg": 10.0,
            "rate_max_rad_s": 2.0,
            "saturation_fraction": 0.1,
            "wind_fidelity": "W0",
            "recovery_class": "dry_recoverable",
        }
    )
    assert target_miss["envelope_status"] == "candidate_family_needs_refinement"
    assert target_miss["coverage_status"] == "uncovered_needs_refinement"
    assert target_miss["library_growth_trigger"] is False


def test_wind_query_region_uses_outlet_relative_height() -> None:
    z_axis = np.array([0.0, 0.5, 1.0, 1.5])
    z_w = np.array([0.5, 0.9, 1.2])

    assert classify_wind_query_region(z_w, z_axis) == "measured"
    assert classify_wind_query_region(np.array([3.0]), z_axis) == "extrapolated"
    assert classify_wind_query_region(np.array([0.2, 2.0]), z_axis) == "clipped"
    assert classify_wind_query_region(z_w, None) == "unknown"


def test_primitive_library_config_and_evidence_latency_fields_are_present() -> None:
    config = PrimitiveLibraryConfig()
    latency_fields = {
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
        "state_feedback_delay_applied",
        "command_delay_applied",
        "actuator_lag_applied",
        "latency_acceptance_scope",
    }

    assert config.latency_case == "actuator_lag_only"
    assert latency_fields.issubset(PrimitiveEvidenceRow.__dataclass_fields__)
