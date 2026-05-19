from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from primitive_library_schema import PrimitiveLibraryConfig
from primitive_library_w3_stress import (
    ALLOWED_CANDIDATE_W3_STATUSES,
    apply_command_latency,
    build_candidate_summary,
    build_coverage_update,
    build_w3_trial_table,
    classify_candidate_w3_status,
    evaluate_w3_trial,
    load_source_evidence,
    load_w3_plan,
)
from run_primitive_library_w3_stress import run_primitive_library_w3_stress


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library"
ARCHIVE_ROOT = (
    REPO_ROOT
    / "03_Control"
    / "05_Results"
    / ("07_" + "aggressive_" + "reversal_" + "ocp")
    / "002"
)


def test_source_loading_fails_loudly_for_missing_w3_plan(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing selected W3 plan"):
        load_w3_plan(tmp_path, source_run_id=3)


def test_trial_table_has_nominal_seed_and_is_reproducible() -> None:
    plan = load_w3_plan(RESULT_ROOT, source_run_id=3)
    table_a = build_w3_trial_table(plan, seeds_per_candidate=3, random_seed=123)
    table_b = build_w3_trial_table(plan, seeds_per_candidate=3, random_seed=123)

    pd.testing.assert_frame_equal(table_a, table_b)
    assert len(table_a) == len(plan) * 3
    assert set(table_a.groupby("source_primitive_id")["stress_seed_index"].min()) == {0}
    nominal = table_a[table_a["stress_seed_index"] == 0]
    assert (nominal["start_dx_m"] == 0.0).all()
    assert (nominal["command_latency_s"] == 0.0).all()


def test_trial_count_follows_csv_and_can_be_capped() -> None:
    plan = load_w3_plan(RESULT_ROOT, source_run_id=3)
    production = build_w3_trial_table(plan, random_seed=123)
    capped = build_w3_trial_table(plan, seeds_per_candidate=2, random_seed=123)

    assert len(production) == int(plan["stress_seed_count"].sum())
    assert len(capped) == len(plan) * 2


def test_invalid_perturbed_entry_is_marked_not_clipped() -> None:
    plan = load_w3_plan(RESULT_ROOT, source_run_id=3)
    _, evidence = load_source_evidence(RESULT_ROOT, evidence_run_id=2)
    trial = build_w3_trial_table(plan.iloc[[0]], seeds_per_candidate=1).iloc[0].to_dict()
    trial["start_dz_m"] = -5.0

    result = evaluate_w3_trial(trial, evidence, PrimitiveLibraryConfig(run_id=4))

    assert result.summary["trial_evaluation_status"] == "entry_state_outside_true_safety"
    assert result.summary["failure_label"] == "entry_invalid"
    assert bool(result.summary["trial_success"]) is False
    assert result.x_ref[0, 2] < 0.4


def test_command_latency_preserves_shape_and_bridge_columns() -> None:
    requested = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
        ]
    )
    delayed = apply_command_latency(requested, dt_s=0.02, command_latency_s=0.04)

    assert delayed.shape == requested.shape
    np.testing.assert_allclose(delayed[0], requested[0])
    np.testing.assert_allclose(delayed[1], requested[0])
    np.testing.assert_allclose(delayed[2], requested[0])
    np.testing.assert_allclose(delayed[3], requested[1])


def test_candidate_status_thresholds_and_nominal_requirement() -> None:
    assert (
        classify_candidate_w3_status(
            trial_count=25,
            evaluated_trial_count=25,
            entry_invalid_count=0,
            trial_success_fraction=0.80,
            nominal_trial_success=True,
            hard_true_safety_violation_count=0,
            dominant_failure_label="target_miss",
            dominant_limiting_mechanism="turn_authority_limited",
        )
        == "w3_supported"
    )
    assert (
        classify_candidate_w3_status(
            trial_count=25,
            evaluated_trial_count=25,
            entry_invalid_count=0,
            trial_success_fraction=0.96,
            nominal_trial_success=True,
            hard_true_safety_violation_count=1,
            dominant_failure_label="success",
            dominant_limiting_mechanism="none",
        )
        == "w3_marginal"
    )
    assert (
        classify_candidate_w3_status(
            trial_count=25,
            evaluated_trial_count=25,
            entry_invalid_count=0,
            trial_success_fraction=0.92,
            nominal_trial_success=False,
            hard_true_safety_violation_count=0,
            dominant_failure_label="success",
            dominant_limiting_mechanism="none",
        )
        == "w3_marginal"
    )
    assert (
        classify_candidate_w3_status(
            trial_count=25,
            evaluated_trial_count=3,
            entry_invalid_count=22,
            trial_success_fraction=0.0,
            nominal_trial_success=False,
            hard_true_safety_violation_count=0,
            dominant_failure_label="entry_invalid",
            dominant_limiting_mechanism="entry_envelope_limited",
        )
        == "w3_rejected_entry_envelope"
    )
    assert (
        classify_candidate_w3_status(
            trial_count=25,
            evaluated_trial_count=25,
            entry_invalid_count=0,
            trial_success_fraction=0.0,
            nominal_trial_success=False,
            hard_true_safety_violation_count=12,
            dominant_failure_label="true_safety_violation",
            dominant_limiting_mechanism="safety_limited",
        )
        == "w3_rejected_safety_or_recovery"
    )


def test_candidate_summary_and_coverage_mapping_use_aggregation() -> None:
    rows = []
    for index in range(5):
        rows.append(
            _trial_row(
                seed=index,
                success=index == 0,
                failure_label="success" if index == 0 else "target_miss",
                mechanism="none" if index == 0 else "turn_authority_limited",
            )
        )
    trial_summary = pd.DataFrame(rows)
    candidate = build_candidate_summary(trial_summary).iloc[0]
    coverage = build_coverage_update(pd.DataFrame([candidate])).iloc[0]

    assert candidate["trial_count"] == 5
    assert candidate["evaluated_trial_count"] == 5
    assert candidate["trial_success_count"] == 1
    assert bool(candidate["nominal_trial_success"]) is True
    assert candidate["candidate_w3_status"] == "w3_marginal"
    assert coverage["coverage_status_s004"] == "w3_marginal_needs_refinement"


def test_runner_writes_required_run_004_files_and_manifest_flags() -> None:
    protected = _protected_hashes()
    paths = run_primitive_library_w3_stress(
        source_run_id=3,
        evidence_run_id=2,
        run_id=4,
        overwrite=True,
        random_seed=20260526,
        max_seeds_per_candidate=3,
        write_trial_logs="selected",
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="ascii"))
    trial_summary = pd.read_csv(paths["trial_summary_csv"])
    candidate_summary = pd.read_csv(paths["candidate_summary_csv"])
    coverage = pd.read_csv(paths["coverage_update_csv"])

    for path in paths.values():
        assert path.exists()
    assert len(trial_summary) == 15
    assert len(candidate_summary) == 5
    assert len(coverage) == 5
    assert set(candidate_summary["candidate_w3_status"]).issubset(ALLOWED_CANDIDATE_W3_STATUSES)
    assert manifest["w3_stress_implemented"] is True
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["real_flight_validation_claim"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["candidate_count"] == 5
    assert manifest["trial_count"] == 15
    assert _protected_hashes() == protected
    for relative_path in manifest["output_files"].values():
        assert not Path(relative_path).is_absolute()


def test_trial_summary_contains_required_columns_after_runner() -> None:
    paths = run_primitive_library_w3_stress(
        source_run_id=3,
        evidence_run_id=2,
        run_id=4,
        overwrite=True,
        max_seeds_per_candidate=2,
        write_trial_logs="none",
    )
    trial_summary = pd.read_csv(paths["trial_summary_csv"])
    required = {
        "w3_plan_id",
        "source_primitive_id",
        "stress_seed_index",
        "trial_evaluation_status",
        "trial_success",
        "failure_label",
        "finite_replay",
        "true_safe_trajectory",
        "heading_band_pass",
        "terminal_heading_change_deg",
        "speed_min_m_s",
        "terminal_speed_m_s",
        "energy_residual_m",
        "alpha_max_deg",
        "beta_max_deg",
        "rate_max_rad_s",
        "saturation_fraction",
        "updraft_strength_scale",
        "updraft_center_shift_x_m",
        "updraft_width_scale",
        "command_latency_s",
        "command_bridge_verified",
    }
    assert required.issubset(trial_summary.columns)


def _trial_row(
    *,
    seed: int,
    success: bool,
    failure_label: str,
    mechanism: str,
) -> dict[str, object]:
    return {
        "w3_plan_id": "w3_s003_01",
        "source_primitive_id": "candidate_a",
        "w3_role": "target_steering",
        "family": "bank_yaw_energy_retaining",
        "target_heading_deg": 15.0,
        "updraft_config": "U1_single_fan",
        "wind_fidelity": "W1",
        "start_condition": "favourable",
        "stress_seed_index": seed,
        "random_seed": 1,
        "trial_evaluation_status": "evaluated",
        "trial_success": success,
        "failure_label": failure_label,
        "active_limiting_mechanism": mechanism,
        "finite_replay": True,
        "true_safe_trajectory": True,
        "heading_band_pass": success,
        "terminal_heading_change_deg": 15.0 if success else 10.0,
        "terminal_heading_error_deg": 0.0 if success else 5.0,
        "path_length_xy_m": 3.0,
        "path_length_3d_m": 3.1,
        "forward_displacement_m": 3.0,
        "lateral_displacement_m": 0.2,
        "xy_bounding_box_area_m2": 0.5,
        "turn_footprint_proxy_m2": 0.5,
        "entry_clearance_required_x_plus_m": 3.0,
        "entry_clearance_required_y_plus_m": 0.2,
        "margin_consumption_x_m": 0.5,
        "margin_consumption_y_m": 0.1,
        "speed_min_m_s": 5.0,
        "terminal_speed_m_s": 6.0,
        "specific_energy_initial_m": 4.0,
        "specific_energy_terminal_m": 4.1,
        "energy_residual_m": 0.1,
        "alpha_max_deg": 20.0,
        "beta_max_deg": 5.0,
        "rate_max_rad_s": 1.0,
        "saturation_fraction": 0.0,
        "min_true_margin_m": 0.2,
        "floor_margin_min_m": 1.0,
        "ceiling_margin_min_m": 1.0,
        "wind_query_region": "measured",
        "updraft_strength_scale": 1.0,
        "updraft_center_shift_x_m": 0.0,
        "updraft_center_shift_y_m": 0.0,
        "updraft_width_scale": 1.0,
        "width_scale_applied": True,
        "width_scale_note": "selected_w3_wrapper",
        "command_latency_s": 0.0,
        "start_dx_m": 0.0,
        "start_dy_m": 0.0,
        "start_dz_m": 0.0,
        "speed_perturbation_m_s": 0.0,
        "phi_perturbation_deg": 0.0,
        "theta_perturbation_deg": 0.0,
        "psi_perturbation_deg": 0.0,
        "command_bridge_verified": True,
        "coverage_region_id": "target_015|favourable|U1_single_fan|W1|dp1",
        "coverage_status_s002": "covered_by_existing_envelope",
        "candidate_class_s002": "updraft_assisted_commandable",
    }


def _protected_hashes() -> dict[str, str]:
    paths = [
        RESULT_ROOT / "003" / "metrics" / "w3_stress_plan_s003.csv",
        RESULT_ROOT / "003" / "metrics" / "higher_target_growth_request_s003.csv",
        RESULT_ROOT / "002" / "metrics" / "primitive_evidence_library_s002.csv",
    ]
    if ARCHIVE_ROOT.exists():
        paths.extend(sorted(path for path in ARCHIVE_ROOT.rglob("*") if path.is_file())[:3])
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()}
