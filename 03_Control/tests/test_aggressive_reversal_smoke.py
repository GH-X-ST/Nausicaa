from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aggressive_reversal_ocp import AGGRESSIVE_CAMPAIGN, BASE_SEED_FAMILIES, SEED_FAMILIES
from run_aggressive_reversal_search import run_search


@pytest.fixture(scope="module")
def aggressive_one_target_outputs(tmp_path_factory):
    import run_aggressive_reversal_search as runner

    output_root = tmp_path_factory.mktemp("aggressive_one_target")
    original_root = runner.DEFAULT_RESULTS_ROOT
    runner.DEFAULT_RESULTS_ROOT = output_root
    try:
        outputs = run_search(
            run_id=1,
            overwrite=False,
            targets_deg=(15.0,),
            max_ipopt_iter=1,
            ocp_max_cpu_time_s=0.05,
            ocp_node_count=1,
        )
    finally:
        runner.DEFAULT_RESULTS_ROOT = original_root
    return outputs


def test_aggressive_runner_writes_core_outputs(aggressive_one_target_outputs) -> None:
    root = aggressive_one_target_outputs["root"]
    required = (
        root / "metrics" / "aggressive_reversal_target_summary_s001.csv",
        root / "metrics" / "aggressive_reversal_target_015_trajectory_s001.csv",
        root / "metrics" / "aggressive_reversal_target_015_commands_s001.csv",
        root / "metrics" / "aggressive_reversal_target_015_metrics_s001.csv",
        root / "metrics" / "aggressive_reversal_target_015_phase_audit_s001.csv",
        root / "manifests" / "aggressive_reversal_manifest_s001.json",
        root / "reports" / "aggressive_reversal_report_s001.md",
    )

    assert root.parent.name == AGGRESSIVE_CAMPAIGN
    for path in required:
        assert path.exists()


def test_aggressive_manifest_is_audit_focused(aggressive_one_target_outputs) -> None:
    manifest = json.loads(
        aggressive_one_target_outputs["manifest_json"].read_text(encoding="ascii")
    )

    assert manifest["run_id"] == "s001"
    assert manifest["targets_completed_deg"] == [15.0]
    assert manifest["seed_families"] == list(SEED_FAMILIES)
    assert manifest["command_bridge"] == "u_norm_requested -> u_norm_applied -> delta_cmd_rad"
    assert manifest["state_derivative_command_input"] == "delta_cmd_rad"
    assert manifest["raw_normalised_commands_enter_state_derivative"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["high_incidence_real_validation_claim"] is False
    assert not any(
        Path(value).is_absolute() for value in manifest["output_files"].values()
    )


def test_aggressive_summary_records_family_and_direct_ocp_attempt(
    aggressive_one_target_outputs,
) -> None:
    summary = pd.read_csv(aggressive_one_target_outputs["summary_csv"])
    row = summary.iloc[0]

    assert row["run_id"] == "s001"
    assert row["target_heading_deg"] == 15.0
    assert set(row["families_attempted"].split(";")) == set(BASE_SEED_FAMILIES)
    assert row["selected_family"] in SEED_FAMILIES
    assert row["selected_method"] in {
        "phase_search",
        "phase_search_followup",
        "direct_ocp",
    }
    assert bool(row["phase_search_attempted"]) is True
    assert bool(row["direct_ocp_attempted"]) is True
    assert bool(row["nlp_constructed"]) is True
    assert bool(row["ipopt_called"]) is True
    assert row["failure_label"] in {
        "success",
        "under_turning",
        "speed_low",
        "true_safety_violation",
        "alpha_boundary",
        "beta_boundary",
        "rate_boundary",
        "actuator_saturation_limited",
        "terminal_recovery_limited",
        "solver_failure",
        "model_boundary_only",
    }


def test_aggressive_command_log_preserves_command_bridge(
    aggressive_one_target_outputs,
) -> None:
    commands = pd.read_csv(
        aggressive_one_target_outputs["root"]
        / "metrics"
        / "aggressive_reversal_target_015_commands_s001.csv"
    )

    for column in (
        "u_norm_requested_delta_a_norm",
        "u_norm_applied_delta_a_norm",
        "delta_cmd_rad_delta_a_cmd",
        "u_norm_requested_delta_e_norm",
        "u_norm_applied_delta_e_norm",
        "delta_cmd_rad_delta_e_cmd",
        "u_norm_requested_delta_r_norm",
        "u_norm_applied_delta_r_norm",
        "delta_cmd_rad_delta_r_cmd",
    ):
        assert column in commands.columns


def test_aggressive_checkpoints_are_compact(aggressive_one_target_outputs) -> None:
    checkpoint_files = sorted(
        (aggressive_one_target_outputs["root"] / "logs" / "checkpoints").glob("*.json")
    )
    direct_files = [path for path in checkpoint_files if "direct_ocp" in path.name]

    assert checkpoint_files
    assert direct_files
    for path in checkpoint_files[:5] + direct_files[:1]:
        checkpoint = json.loads(path.read_text(encoding="ascii"))
        assert "x_ref" not in checkpoint
        assert "u_ff_norm" not in checkpoint
        assert checkpoint["trajectory_csv"].endswith(".csv")
        assert checkpoint["commands_csv"].endswith(".csv")


def test_aggressive_report_lists_ocp_diagnostics(aggressive_one_target_outputs) -> None:
    report = aggressive_one_target_outputs["report_md"].read_text(encoding="ascii")

    assert "nlp_constructed" in report
    assert "ipopt_called" in report
    assert "direct_ocp_attempted" in report
    assert "not high-incidence validation" in report
