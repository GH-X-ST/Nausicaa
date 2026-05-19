from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from primitive_library_governor import (
    GOVERNOR_DECISION_STATUSES,
    build_governor_coverage_update,
    build_governor_decision_cases,
    build_governor_rejection_summary,
    build_governor_seed_table,
    evaluate_governor_cases,
    load_governor_sources,
    load_w3_supported_candidates,
)
from run_primitive_library_governor_seed import run_primitive_library_governor_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library"
ARCHIVE_ROOT = (
    REPO_ROOT
    / "03_Control"
    / "05_Results"
    / ("07_" + "aggressive_" + "reversal_" + "ocp")
    / "002"
)
EXPECTED_SEEDS = {
    "glide_none_favourable_U4_four_fan_W2_dp1",
    "recovery_none_favourable_U4_four_fan_W1_dp1",
    "mild_bank_none_favourable_U1_single_fan_W2_dp1",
    "glide_none_favourable_U1_single_fan_W2_dp1",
}
TARGET_STEERING = "bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1"


def test_source_loading_fails_loudly_for_missing_candidate_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing run-004 W3 source evidence"):
        load_w3_supported_candidates(tmp_path, run_id=4)


def test_seed_table_includes_exact_supported_candidates_and_excludes_target() -> None:
    sources = load_governor_sources(RESULT_ROOT)
    seed_table = build_governor_seed_table(
        sources["candidate_summary"],
        sources["coverage_update"],
        sources["manifest"],
    )

    accepted = seed_table[seed_table["governor_seed_candidate"].astype(bool)]
    excluded = seed_table[seed_table["seed_table_status"] == "excluded_marginal_target_steering"]
    assert set(accepted["source_primitive_id"]) == EXPECTED_SEEDS
    assert len(accepted) == 4
    assert len(excluded) == 1
    assert excluded.iloc[0]["source_primitive_id"] == TARGET_STEERING
    assert bool(excluded.iloc[0]["governor_seed_candidate"]) is False


def test_decision_cases_have_expected_inventory_and_nominal_cases_accept() -> None:
    seed_table, cases, decisions = _build_decision_tables()

    assert len(cases) == 33
    assert set(cases["case_kind"]) == {
        "nominal_valid_case",
        "wrong_updraft_config_case",
        "wrong_wind_fidelity_case",
        "low_lift_confidence_case",
        "insufficient_clearance_case",
        "entry_outside_true_safety_case",
        "recovery_unavailable_case",
        "unsupported_model_region_case",
        "target_steering_marginal_rejected_case",
    }
    nominal = decisions[decisions["case_kind"] == "nominal_valid_case"]
    assert set(nominal["source_primitive_id"]) == EXPECTED_SEEDS
    assert nominal["accepted"].astype(bool).all()
    assert set(nominal["governor_decision_status"]) == {"accepted_governor_seed"}
    assert TARGET_STEERING not in set(nominal["source_primitive_id"])


def test_planned_rejection_cases_fail_with_specific_statuses() -> None:
    _, _, decisions = _build_decision_tables()
    expected = {
        "wrong_updraft_config_case": "rejected_lift_belief",
        "wrong_wind_fidelity_case": "rejected_wind_fidelity",
        "low_lift_confidence_case": "rejected_lift_belief",
        "insufficient_clearance_case": "rejected_clearance",
        "entry_outside_true_safety_case": "rejected_entry_envelope",
        "recovery_unavailable_case": "rejected_recovery_class",
        "unsupported_model_region_case": "rejected_model_region",
        "target_steering_marginal_rejected_case": "rejected_target_steering_marginal",
    }
    for case_kind, status in expected.items():
        rows = decisions[decisions["case_kind"] == case_kind]
        assert not rows.empty
        assert set(rows["governor_decision_status"]) == {status}
        assert not rows["accepted"].astype(bool).any()


def test_rejection_summary_and_coverage_update_match_decisions() -> None:
    seed_table, _, decisions = _build_decision_tables()
    summary = build_governor_rejection_summary(decisions)
    coverage = build_governor_coverage_update(seed_table, decisions)

    assert set(summary["governor_decision_status"]) == set(GOVERNOR_DECISION_STATUSES)
    assert int(summary["case_count"].sum()) == len(decisions)
    assert int(summary["accepted_count"].sum()) == int(decisions["accepted"].astype(bool).sum())
    accepted = coverage[coverage["governor_coverage_status_s005"] == "governor_seed_available"]
    target = coverage[coverage["source_primitive_id"] == TARGET_STEERING].iloc[0]
    assert set(accepted["source_primitive_id"]) == EXPECTED_SEEDS
    assert target["governor_coverage_status_s005"] == "governor_seed_rejected_refine_first"
    assert target["recommended_next_step_s005"] == "refine_seed_before_governor"


def test_runner_writes_outputs_manifest_flags_and_preserves_sources() -> None:
    protected = _protected_hashes()
    paths = run_primitive_library_governor_seed(
        source_w3_run_id=4,
        source_plan_run_id=3,
        source_evidence_run_id=2,
        run_id=5,
        overwrite=True,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="ascii"))
    seed_table = pd.read_csv(paths["seed_table_csv"])
    decisions = pd.read_csv(paths["decisions_csv"])
    coverage = pd.read_csv(paths["coverage_update_csv"])

    for path in paths.values():
        assert path.exists()
    assert manifest["run_id"] == "s005"
    assert manifest["source_w3_run"] == "s004"
    assert manifest["source_w3_plan_run"] == "s003"
    assert manifest["source_evidence_run"] == "s002"
    assert manifest["governor_seed_implemented"] is True
    assert manifest["governor_query_implemented"] is True
    assert manifest["governor_implemented"] is True
    assert manifest["governor_online_flight_ready"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["real_flight_validation_claim"] is False
    assert manifest["hardware_implemented"] is False
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["accepted_seed_candidate_count"] == 4
    assert manifest["excluded_target_steering_count"] == 1
    assert manifest["target_steering_governor_allowed"] is False
    assert manifest["target_steering_next_action"] == "refine_seed_before_governor"
    assert set(seed_table[seed_table["governor_seed_candidate"].astype(bool)]["source_primitive_id"]) == EXPECTED_SEEDS
    assert int(decisions["accepted"].astype(bool).sum()) == 4
    assert set(coverage["governor_coverage_status_s005"]) == {
        "governor_seed_available",
        "governor_seed_rejected_refine_first",
    }
    for relative_path in manifest["output_files"].values():
        assert not Path(relative_path).is_absolute()
    assert _protected_hashes() == protected


def _build_decision_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = load_governor_sources(RESULT_ROOT)
    seed_table = build_governor_seed_table(
        sources["candidate_summary"],
        sources["coverage_update"],
        sources["manifest"],
    )
    cases = build_governor_decision_cases(seed_table, sources["source_evidence"])
    decisions = evaluate_governor_cases(cases, seed_table)
    return seed_table, cases, decisions


def _protected_hashes() -> dict[str, str]:
    paths = [
        RESULT_ROOT / "002" / "metrics" / "primitive_evidence_library_s002.csv",
        RESULT_ROOT / "003" / "metrics" / "w3_stress_plan_s003.csv",
        RESULT_ROOT / "004" / "metrics" / "w3_stress_candidate_summary_s004.csv",
        RESULT_ROOT / "004" / "metrics" / "w3_stress_coverage_update_s004.csv",
        RESULT_ROOT / "004" / "manifests" / "w3_stress_manifest_s004.json",
    ]
    if ARCHIVE_ROOT.exists():
        paths.extend(sorted(path for path in ARCHIVE_ROOT.rglob("*") if path.is_file())[:3])
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()}
