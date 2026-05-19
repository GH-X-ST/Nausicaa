from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from primitive_library_selection import (
    HIGHER_TARGET_REQUEST_STATUSES,
    SELECTION_STATUSES,
    build_coverage_decision_summary,
    build_higher_target_growth_request,
    classify_selection_status,
)
from run_primitive_library_shortlist import run_primitive_library_shortlist


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "003"


def _load_run_003() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (RESULT_ROOT / "manifests" / "primitive_library_shortlist_manifest_s003.json").exists():
        run_primitive_library_shortlist(source_run_id=2, run_id=3, overwrite=True)
    manifest = json.loads(
        (RESULT_ROOT / "manifests" / "primitive_library_shortlist_manifest_s003.json").read_text(
            encoding="ascii"
        )
    )
    shortlist = pd.read_csv(RESULT_ROOT / "metrics" / "candidate_shortlist_s003.csv")
    coverage = pd.read_csv(RESULT_ROOT / "metrics" / "coverage_decision_summary_s003.csv")
    w3_plan = pd.read_csv(RESULT_ROOT / "metrics" / "w3_stress_plan_s003.csv")
    higher = pd.read_csv(RESULT_ROOT / "metrics" / "higher_target_growth_request_s003.csv")
    return manifest, shortlist, coverage, w3_plan, higher


def test_shortlist_runner_writes_required_run_003_outputs() -> None:
    paths = run_primitive_library_shortlist(source_run_id=2, run_id=3, overwrite=True)
    manifest, shortlist, coverage, w3_plan, higher = _load_run_003()

    for path in paths.values():
        assert path.exists()
    assert not shortlist.empty
    assert not coverage.empty
    assert set(shortlist["selection_status"]).issubset(SELECTION_STATUSES)
    assert len(w3_plan) <= manifest["max_w3_candidates"]
    if not w3_plan.empty:
        assert w3_plan["not_implemented_in_this_pass"].astype(bool).all()
    assert set(higher["requested_target_deg"].astype(float)) == {45.0, 60.0, 90.0, 120.0, 150.0, 180.0}
    assert set(higher["request_status"]).issubset(HIGHER_TARGET_REQUEST_STATUSES)


def test_manifest_is_run_002_only_and_no_overclaiming() -> None:
    manifest, _, _, _, _ = _load_run_003()

    assert manifest["source_run_id"] == "s002"
    assert manifest["run_002_only_source"] is True
    assert manifest["planning_only"] is True
    assert manifest["no_replay_performed"] is True
    assert manifest["coverage_driven_higher_target_logic"] is True
    assert manifest["thirty_deg_uncovered_does_not_auto_recommend_45_60"] is True
    assert manifest["w3_stress_implemented"] is False
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["real_flight_validation_claim"] is False
    assert manifest["high_incidence_validation_claim"] is False
    for relative_path in manifest["output_files"].values():
        assert not Path(relative_path).is_absolute()


def test_coverage_decision_includes_run_002_regions_and_governor_rejects() -> None:
    _, _, coverage, _, _ = _load_run_003()
    source_coverage = pd.read_csv(
        REPO_ROOT
        / "03_Control"
        / "05_Results"
        / "09_primitive_library"
        / "002"
        / "metrics"
        / "primitive_coverage_region_summary_s002.csv"
    )

    assert set(coverage["coverage_region_id"]) == set(source_coverage["coverage_region_id"])
    governor = coverage[coverage["coverage_status_s002"] == "uncovered_governor_reject"]
    assert not governor.empty
    assert set(governor["coverage_decision_s003"]) == {"entry_envelope_reject"}
    assert not governor["library_growth_trigger_s003"].astype(bool).any()
    assert governor["needs_governor_reject"].astype(bool).all()


def test_30_deg_boundary_or_entry_failure_does_not_recommend_45_60() -> None:
    _, _, coverage, _, higher = _load_run_003()
    thirty = coverage[coverage["target_heading_deg"].astype(float) == 30.0]

    assert not thirty.empty
    assert set(thirty["coverage_status_s002"]).issubset({"uncovered_boundary", "uncovered_governor_reject"})
    future_45_60 = higher[higher["requested_target_deg"].astype(float).isin([45.0, 60.0])]
    assert not future_45_60.empty
    assert "recommended_next" not in set(future_45_60["request_status"])
    assert set(future_45_60["request_status"]) == {"defer_boundary_only"}


def test_refinement_failure_prefers_refine_30_seed() -> None:
    coverage = pd.DataFrame(
        [
            {
                "coverage_region_id": "target_030|favourable|none|W0|dp1",
                "coverage_decision_s003": "generator_refinement_needed",
                "coverage_status_s002": "uncovered_needs_refinement",
                "reason": "refine_existing_seed_before_higher_target_request",
                "target_heading_deg": 30.0,
            }
        ]
    )
    higher = build_higher_target_growth_request(coverage)
    future_45_60 = higher[higher["requested_target_deg"].astype(float).isin([45.0, 60.0])]

    assert set(future_45_60["request_status"]) == {"refine_30_seed"}


def test_mission_critical_gap_can_request_future_45_60() -> None:
    coverage = pd.DataFrame(
        [
            {
                "coverage_region_id": "target_030|favourable|none|W0|dp1",
                "coverage_decision_s003": "boundary_keep_for_discussion",
                "coverage_status_s002": "uncovered_boundary",
                "reason": "boundary_or_true_safety_limited_not_commandable",
                "target_heading_deg": 30.0,
            }
        ]
    )
    mission = pd.DataFrame(
        [
            {
                "requested_target_deg": 45.0,
                "coverage_region_id": "mission_lift_exploitation_sector_A",
                "mission_critical_region_present": True,
                "safe_15_or_30_alternative_present": False,
                "plausible_shorter_footprint_entry_envelope": True,
                "current_failure_mode": "library_growth_gap",
                "recommended_families": "canyon_steep_bank,bank_yaw_energy_retaining",
            }
        ]
    )
    higher = build_higher_target_growth_request(coverage, mission)
    row_45 = higher[higher["requested_target_deg"].astype(float) == 45.0].iloc[0]

    assert row_45["request_status"] == "recommended_next"
    assert bool(row_45["mission_critical_region_present"]) is True
    assert bool(row_45["safe_15_or_30_alternative_present"]) is False
    assert bool(row_45["plausible_shorter_footprint_entry_envelope"]) is True


def test_selection_status_classifier_is_stable() -> None:
    assert (
        classify_selection_status(
            {
                "candidate_class": "updraft_assisted_commandable",
                "coverage_status": "covered_by_existing_envelope",
                "entry_envelope_status": "inside_entry_envelope",
                "envelope_status": "widening_existing_envelope",
                "evaluation_status": "evaluated",
            }
        )
        == "selected_for_w3_stress"
    )
    assert (
        classify_selection_status(
            {
                "candidate_class": "boundary_evidence",
                "coverage_status": "uncovered_governor_reject",
                "entry_envelope_status": "outside_entry_envelope_governor_reject",
                "envelope_status": "outside_entry_envelope_governor_reject",
                "evaluation_status": "evaluated",
            }
        )
        == "governor_reject_entry_envelope"
    )


def test_coverage_decision_can_be_built_from_synthetic_rows() -> None:
    coverage = pd.DataFrame(
        [
            {
                "coverage_region_id": "target_015|favourable|U1_single_fan|W1|dp1",
                "row_count": 1,
                "best_primitive_id": "candidate_a",
                "best_family": "bank_yaw_energy_retaining",
                "best_candidate_class": "updraft_assisted_commandable",
                "coverage_status": "covered_by_existing_envelope",
                "best_heading_error_deg": 1.0,
                "best_path_length_xy_m": 3.0,
                "best_footprint_m2": 0.8,
                "best_terminal_speed_m_s": 6.0,
                "best_energy_residual_m": 0.1,
                "best_min_true_margin_m": 0.2,
            }
        ]
    )
    shortlist = pd.DataFrame(
        [
            {
                "primitive_id": "candidate_a",
                "selection_status": "selected_for_w3_stress",
            }
        ]
    )
    decision = build_coverage_decision_summary(coverage, shortlist).iloc[0]

    assert decision["coverage_decision_s003"] == "covered_send_to_w3"
    assert bool(decision["needs_w3"]) is True
