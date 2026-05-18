from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agile_turn_family_comparison import FAMILY_NAMES, candidate_ranking_key
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
    for target in (15.0, 30.0):
        rows = candidate_summary[candidate_summary["target_heading_deg"] == target]
        assert set(rows["family_name"]) == set(FAMILY_NAMES)
        for family_name in FAMILY_NAMES:
            family_rows = rows[rows["family_name"] == family_name]
            assert len(set(family_rows["horizon_s"])) == 3


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


def test_manifest_report_flags_and_boundary_note(agile_turn_outputs) -> None:
    manifest = json.loads(agile_turn_outputs["manifest_json"].read_text(encoding="ascii"))
    report = agile_turn_outputs["report_md"].read_text(encoding="ascii")

    assert manifest["campaign"] == "08_agile_turn_family_comparison"
    assert manifest["targets_run_deg"] == [15.0, 30.0]
    assert manifest["active_family_inventory"] == list(FAMILY_NAMES)
    assert manifest["retired_high_alpha_branch_active"] is False
    assert manifest["boundary_reference_note"] in report
    assert "archived boundary evidence only" in report
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


def test_runner_source_does_not_import_retired_aggressive_reversal_branch() -> None:
    source_paths = (
        Path("03_Control/03_Primitives/agile_turn_family_comparison.py"),
        Path("03_Control/04_Scenarios/run_agile_turn_family_comparison.py"),
    )
    combined = "\n".join(path.read_text(encoding="ascii") for path in source_paths)

    assert "aggressive_reversal_ocp" not in combined
    assert "run_aggressive_reversal_search" not in combined


def test_recoverable_partial_heading_ranks_above_unrecoverable_full_heading() -> None:
    base = {
        "strict_family_success": False,
        "useful_recoverable_candidate": False,
        "terminal_speed_m_s": 5.1,
        "min_speed_m_s": 4.1,
        "energy_lost_per_deg_m_per_deg": 0.02,
        "max_alpha_deg": 20.0,
        "max_beta_deg": 5.0,
        "max_rate_rad_s": 2.0,
        "min_true_margin_m": 0.5,
        "saturation_fraction": 0.0,
    }
    recoverable_partial = {
        **base,
        "recoverable": True,
        "actual_heading_change_deg": 22.0,
    }
    unrecoverable_full = {
        **base,
        "recoverable": False,
        "actual_heading_change_deg": 30.0,
    }

    assert candidate_ranking_key(recoverable_partial) > candidate_ranking_key(unrecoverable_full)


def test_candidate_summary_contains_limiter_diagnostics(agile_turn_outputs) -> None:
    candidate_summary = pd.read_csv(agile_turn_outputs["candidate_summary_csv"])

    for column in (
        "horizon_limited",
        "turn_authority_limited",
        "energy_limited",
        "safety_limited",
        "exposure_limited",
    ):
        assert column in candidate_summary.columns


def test_escalation_remains_default_blocked_or_not_requested(agile_turn_outputs) -> None:
    manifest = json.loads(agile_turn_outputs["manifest_json"].read_text(encoding="ascii"))

    assert manifest["escalation_requested"] is False
    assert manifest["escalation_targets_run_deg"] == []
