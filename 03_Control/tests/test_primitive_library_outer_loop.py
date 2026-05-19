from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from primitive_library_outer_loop import (
    build_candidate_cases_for_state,
    build_outer_loop_initial_state,
    build_outer_loop_scenarios,
    load_outer_loop_sources,
    score_outer_loop_candidates,
    select_governor_approved_candidate,
)
from run_primitive_library_outer_loop import run_primitive_library_outer_loop


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library"
ARCHIVE_ROOT = (
    REPO_ROOT
    / "03_Control"
    / "05_Results"
    / ("07_" + "aggressive_" + "reversal_" + "ocp")
    / "002"
)
EXPECTED_ACTIVE_SEEDS = {
    "glide_none_favourable_U4_four_fan_W2_dp1",
    "recovery_none_favourable_U4_four_fan_W1_dp1",
    "mild_bank_none_favourable_U1_single_fan_W2_dp1",
    "glide_none_favourable_U1_single_fan_W2_dp1",
}
TARGET_STEERING = "bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1"


def test_source_loading_fails_loudly_for_missing_run005(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing run-005 outer-loop source files"):
        load_outer_loop_sources(tmp_path, governor_run_id=5)


def test_loads_four_active_governor_seeds_and_excludes_target() -> None:
    sources = load_outer_loop_sources(RESULT_ROOT, governor_run_id=5)
    active = sources["active_seed_table"]
    target = sources["target_steering_table"]

    assert set(active["source_primitive_id"]) == EXPECTED_ACTIVE_SEEDS
    assert len(active) == 4
    assert len(target) == 1
    assert target.iloc[0]["source_primitive_id"] == TARGET_STEERING
    assert TARGET_STEERING not in set(active["source_primitive_id"])


def test_governor_query_preserves_numerical_clearance_contract() -> None:
    sources = load_outer_loop_sources(RESULT_ROOT, governor_run_id=5)
    scenario = [item for item in build_outer_loop_scenarios() if item.scenario_id.startswith("U1_")][0]
    state = build_outer_loop_initial_state(scenario)
    cases = build_candidate_cases_for_state(state, sources["active_seed_table"], scenario)
    scored = score_outer_loop_candidates(cases, sources["governor_decisions"], sources["active_seed_table"], scenario)

    assert "clearance_min_margin_m" in scored.columns
    assert (scored["clearance_min_margin_m"].notna()).all()
    assert scored["clearance_check_pass"].astype(bool).any()
    assert scored["accepted"].astype(bool).any()
    selected = select_governor_approved_candidate(scored)
    assert selected is not None
    assert selected["source_primitive_id"] in EXPECTED_ACTIVE_SEEDS
    assert selected["source_primitive_id"] != TARGET_STEERING


def test_low_lift_and_clearance_scenarios_reject_as_expected() -> None:
    sources = load_outer_loop_sources(RESULT_ROOT, governor_run_id=5)
    scenarios = {item.scenario_id: item for item in build_outer_loop_scenarios()}

    low_lift = scenarios["low_lift_confidence_rejection"]
    low_state = build_outer_loop_initial_state(low_lift)
    low_cases = build_candidate_cases_for_state(low_state, sources["active_seed_table"], low_lift)
    low_scored = score_outer_loop_candidates(low_cases, sources["governor_decisions"], sources["active_seed_table"], low_lift)
    assert not low_scored["accepted"].astype(bool).any()
    assert set(low_scored["governor_decision_status"]) == {"rejected_lift_belief"}

    clearance = scenarios["clearance_limited_no_go"]
    clearance_state = build_outer_loop_initial_state(clearance)
    clearance_cases = build_candidate_cases_for_state(clearance_state, sources["active_seed_table"], clearance)
    clearance_scored = score_outer_loop_candidates(
        clearance_cases,
        sources["governor_decisions"],
        sources["active_seed_table"],
        clearance,
    )
    assert not clearance_scored["accepted"].astype(bool).any()
    assert "rejected_clearance" in set(clearance_scored["governor_decision_status"])
    assert (clearance_scored["clearance_min_margin_m"] < 0.0).any()


def test_runner_writes_outputs_flags_and_preserves_sources() -> None:
    protected = _protected_hashes()
    paths = run_primitive_library_outer_loop(
        source_governor_run_id=5,
        run_id=6,
        overwrite=True,
        max_steps=8,
        dt_s=0.02,
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    summary = pd.read_csv(paths["mission_summary_csv"])
    step_log = pd.read_csv(paths["step_log_csv"])
    candidate_log = pd.read_csv(paths["candidate_decision_log_csv"])
    energy = pd.read_csv(paths["energy_trace_csv"])
    lift = pd.read_csv(paths["lift_dwell_summary_csv"])
    gaps = pd.read_csv(paths["coverage_gap_summary_csv"])

    for path in paths.values():
        assert path.exists()
    assert manifest["run_id"] == "s006"
    assert manifest["source_governor_run"] == "s005"
    assert manifest["accepted_seed_candidate_count"] == 4
    assert manifest["outer_loop_implemented"] is True
    assert manifest["outer_loop_mission_simulation_implemented"] is True
    assert manifest["governor_implemented"] is True
    assert manifest["governor_online_flight_ready"] is False
    assert manifest["target_steering_used"] is False
    assert manifest["target_steering_governor_allowed"] is False
    assert manifest["higher_target_primitives_added"] is False
    assert manifest["new_primitive_generated"] is False
    assert manifest["real_flight_validation_claim"] is False
    assert manifest["hardware_implemented"] is False
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["outer_loop_governor_query_demonstrated"] is True
    assert manifest["partial_transit_clearance_limited_count"] == 2
    assert manifest["sustained_outer_loop_mission_success_count"] == 0
    assert manifest["energy_gain_scenario_count"] == 0
    assert manifest["autonomous_thermal_exploitation_claim"] is False
    assert manifest["continuous_flight_claim"] is False
    assert manifest["mission_claim_boundary"] == (
        "short governor-mediated transit/rejection evidence only; "
        "no sustained thermal exploitation claim"
    )

    assert set(summary["scenario_id"]) == {
        "U1_lift_sector_governed_transit",
        "U4_lift_sector_governed_transit",
        "low_lift_confidence_rejection",
        "clearance_limited_no_go",
    }
    assert "completed_with_governor_evidence" not in set(summary["mission_success_label"])
    transit = summary[summary["scenario_id"].isin({
        "U1_lift_sector_governed_transit",
        "U4_lift_sector_governed_transit",
    })]
    assert set(transit["mission_success_label"]) == {"partial_governed_transit_then_clearance_limited"}
    assert transit["governor_evidence_obtained"].astype(bool).all()
    assert transit["short_transit_supported"].astype(bool).all()
    assert transit["clearance_limited_after_first_step"].astype(bool).all()
    assert not transit["energy_gain_demonstrated"].astype(bool).any()
    assert set(transit["energy_delta_sign"]) == {"negative"}
    assert not summary["sustained_outer_loop_mission_success"].astype(bool).any()
    assert not summary["sustained_lift_exploitation_claim"].astype(bool).any()
    assert not summary["continuous_flight_claim"].astype(bool).any()

    no_go = dict(zip(summary["scenario_id"], summary["mission_success_label"]))
    assert no_go["low_lift_confidence_rejection"] == "no_go_lift_belief_rejection"
    assert no_go["clearance_limited_no_go"] == "no_go_clearance_limited"
    assert int(summary["steps_accepted"].sum()) >= 1
    assert not step_log.empty
    assert not candidate_log.empty
    assert not energy.empty
    assert not lift.empty
    assert not gaps.empty
    assert TARGET_STEERING not in set(candidate_log["source_primitive_id"])
    assert "clearance_min_margin_m" in candidate_log.columns
    assert "specific_energy_height_m" in energy.columns
    assert "higher_target_request_status" in gaps.columns
    transit_gaps = gaps[gaps["scenario_id"].isin({
        "U1_lift_sector_governed_transit",
        "U4_lift_sector_governed_transit",
    })]
    assert set(transit_gaps["coverage_gap_type"]) == {"partial_short_mission_clearance_limited"}
    assert set(transit_gaps["recommended_next_action"]) == {"proceed_to_ablation_with_clearance_limitation"}
    assert set(transit_gaps["higher_target_request_status"]) == {
        "not_requested_current_library_sufficient_for_short_governor_test_only"
    }
    assert "not_requested_current_library_sufficient_for_test" not in set(gaps["higher_target_request_status"])
    assert "short_mission_supported_without_target_steering" not in set(gaps["coverage_gap_type"])
    for relative_path in manifest["output_files"].values():
        assert not Path(relative_path).is_absolute()
    assert _protected_hashes() == protected


def _protected_hashes() -> dict[str, str]:
    paths = [
        RESULT_ROOT / "002" / "metrics" / "primitive_evidence_library_s002.csv",
        RESULT_ROOT / "003" / "metrics" / "w3_stress_plan_s003.csv",
        RESULT_ROOT / "004" / "metrics" / "w3_stress_candidate_summary_s004.csv",
        RESULT_ROOT / "004" / "metrics" / "w3_stress_coverage_update_s004.csv",
        RESULT_ROOT / "005" / "metrics" / "governor_seed_candidate_table_s005.csv",
        RESULT_ROOT / "005" / "metrics" / "governor_coverage_update_s005.csv",
        RESULT_ROOT / "005" / "manifests" / "governor_seed_manifest_s005.json",
    ]
    if ARCHIVE_ROOT.exists():
        paths.extend(sorted(path for path in ARCHIVE_ROOT.rglob("*") if path.is_file())[:3])
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()}
