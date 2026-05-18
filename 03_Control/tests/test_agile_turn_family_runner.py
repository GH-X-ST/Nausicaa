from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agile_turn_family_comparison import FAMILY_NAMES, TARGET_HORIZON_GRID_S
from run_agile_turn_family_comparison import run_comparison


@pytest.fixture(scope="module")
def agile_turn_outputs(tmp_path_factory):
    import run_agile_turn_family_comparison as runner

    output_root = tmp_path_factory.mktemp("agile_turn_outputs")
    original_root = runner.DEFAULT_RESULTS_ROOT
    runner.DEFAULT_RESULTS_ROOT = output_root
    try:
        outputs = run_comparison(run_id=1, targets_deg=(15.0, 30.0), overwrite=False)
    finally:
        runner.DEFAULT_RESULTS_ROOT = original_root
    return outputs


def test_runner_writes_all_family_target_horizon_evidence(agile_turn_outputs) -> None:
    root = agile_turn_outputs["root"]
    candidate_summary = pd.read_csv(agile_turn_outputs["candidate_summary_csv"])

    assert root.name == "001"
    assert (root / "metrics").exists()
    assert agile_turn_outputs["target_summary_csv"].exists()
    assert agile_turn_outputs["family_summary_csv"].exists()
    assert agile_turn_outputs["cleanup_manifest_json"].exists()
    for target in (15.0, 30.0):
        rows = candidate_summary[candidate_summary["target_heading_deg"] == target]
        assert set(rows["family_name"]) == set(FAMILY_NAMES)
        for family_name in FAMILY_NAMES:
            family_rows = rows[rows["family_name"] == family_name]
            assert set(family_rows["horizon_s"]) == set(TARGET_HORIZON_GRID_S[target])


def test_command_logs_preserve_requested_applied_and_radian_columns(agile_turn_outputs) -> None:
    root = agile_turn_outputs["root"]
    command_file = next((root / "metrics").glob("*_commands.csv"))
    commands = pd.read_csv(command_file)

    for column in (
        "u_norm_requested_delta_a_norm",
        "u_norm_applied_delta_a_norm",
        "delta_cmd_rad_delta_a_cmd",
    ):
        assert column in commands.columns


def test_manifest_report_flags_and_cleanup_manifest(agile_turn_outputs) -> None:
    manifest = json.loads(agile_turn_outputs["manifest_json"].read_text(encoding="ascii"))
    cleanup = json.loads(agile_turn_outputs["cleanup_manifest_json"].read_text(encoding="ascii"))
    report = agile_turn_outputs["report_md"].read_text(encoding="ascii")

    assert manifest["campaign"] == "08_agile_turn_family_comparison"
    assert manifest["fixed_target_ladder_deg"] == [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    assert manifest["targets_run_deg"] == [15.0, 30.0]
    assert manifest["active_family_inventory"] == list(FAMILY_NAMES)
    assert manifest["no_20deg_bin"] is True
    assert manifest["old_perch_like_branch_active"] is False
    assert manifest["archived_perch_reference_preserved"] is True
    assert cleanup["archive_preserved_byte_for_byte"] is True
    assert "Archived high-alpha/perch-like boundary reference" in report
    for key in (
        "actual_agile_reversal_primitive_implemented",
        "updraft_validation_claim",
        "w1_w2_w3_updraft_validation_claim",
        "real_flight_validation_claim",
        "ocp_implemented",
        "tvlqr_implemented",
        "governor_implemented",
        "outer_loop_implemented",
        "vicon_implemented",
        "hardware_implemented",
        "high_incidence_validation_claim",
        "raw_normalised_commands_enter_state_derivative",
    ):
        assert manifest[key] is False
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())


def test_active_sources_do_not_reference_removed_branch_tokens() -> None:
    source_paths = (
        Path("03_Control/03_Primitives/agile_turn_family_comparison.py"),
        Path("03_Control/04_Scenarios/run_agile_turn_family_comparison.py"),
        Path("03_Control/tests/test_agile_turn_family_profiles.py"),
        Path("03_Control/tests/test_agile_turn_family_runner.py"),
    )
    combined = "\n".join(path.read_text(encoding="ascii") for path in source_paths)
    tokens = (
        "aggressive" + "_reversal" + "_ocp",
        "run_" + "aggressive" + "_reversal" + "_search",
        "dive_" + "perch" + "_redirect_30",
        "reduced_" + "perch" + "_redirect_30",
        "early_unload_" + "recovery_30",
        "speed_" + "collapse" + "_pitch_redirect",
    )
    for token in tokens:
        assert token not in combined


def test_target_summary_handles_empty_commandable_selection(agile_turn_outputs) -> None:
    summary = pd.read_csv(agile_turn_outputs["target_summary_csv"])

    for _, row in summary.iterrows():
        if not bool(row["commandable_target_found"]):
            assert pd.isna(row["selected_candidate_id"]) or row["selected_candidate_id"] == ""
            assert pd.isna(row["selected_family"]) or row["selected_family"] == ""
            assert row["escalation_allowed"] in (False, "False")
        assert "best_safe_partial_candidate_id" in row
        assert "best_accurate_boundary_candidate_id" in row


def test_candidate_summary_contains_target_accuracy_fields(agile_turn_outputs) -> None:
    candidate_summary = pd.read_csv(agile_turn_outputs["candidate_summary_csv"])

    for column in (
        "terminal_heading_change_deg",
        "peak_heading_change_deg",
        "heading_error_deg",
        "absolute_heading_error_deg",
        "heading_band_pass",
        "overshoot_deg",
        "undershoot_deg",
        "candidate_class",
        "active_limiting_mechanism",
    ):
        assert column in candidate_summary.columns


def test_no_figure_files_created(agile_turn_outputs) -> None:
    root = agile_turn_outputs["root"]

    assert not list((root / "figures").glob("*"))
    assert not list(root.rglob("*.png"))
    assert not list(root.rglob("*.pdf"))
    assert not list(root.rglob("*.svg"))
