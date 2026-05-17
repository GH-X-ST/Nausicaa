from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from glide_primitive import GLIDE_CAMPAIGN
from run_glide_primitive_w0 import run_smoke


def test_w0_glide_runner_writes_required_outputs(monkeypatch, tmp_path) -> None:
    import run_glide_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    root = tmp_path / GLIDE_CAMPAIGN / "001"

    required = (
        root / "metrics" / "entry_checks_s001.csv",
        root / "metrics" / "exit_checks_s001.csv",
        root / "metrics" / "glide_checks_s001.csv",
        root / "metrics" / "trajectory_s001.csv",
        root / "metrics" / "commands_s001.csv",
        root / "metrics" / "glide_metrics_s001.csv",
        root / "manifests" / "glide_primitive_manifest_s001.json",
        root / "reports" / "glide_primitive_report_s001.md",
    )
    for path in required:
        assert path.exists()
    assert outputs["root"] == root


def test_w0_glide_manifest_flags_and_paths(monkeypatch, tmp_path) -> None:
    import run_glide_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))

    assert manifest["overall_status"] == "pass"
    assert manifest["run_id"] == "s001"
    assert manifest["primitive_interface_used"] is True
    assert manifest["primitive_implemented"] is True
    assert manifest["controller_implemented"] is True
    assert manifest["local_feedback_controller_implemented"] is True
    assert manifest["actual_glide_primitive_implemented"] is True
    for key in (
        "actual_bank_primitive_implemented",
        "actual_recovery_primitive_implemented",
        "actual_agile_reversal_primitive_implemented",
        "ocp_implemented",
        "tvlqr_implemented",
        "governor_implemented",
        "outer_loop_implemented",
        "vicon_implemented",
        "hardware_implemented",
        "high_incidence_validation_claim",
        "raw_normalised_commands_enter_state_derivative",
        "terminal_recoverable_proxy_is_recovery_proof",
    ):
        assert manifest[key] is False
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())


def test_w0_glide_metric_and_command_csv_contract(monkeypatch, tmp_path) -> None:
    import run_glide_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    metrics = pd.read_csv(outputs["metrics_csv"])
    commands = pd.read_csv(outputs["commands_csv"])
    row = metrics.iloc[0]

    assert row["run_id"] == "s001"
    assert str(row["finite_state_success"]) == "True"
    assert str(row["rollout_success"]) == "True"
    assert str(row["primitive_success"]) == "True"
    assert str(row["success"]) == "True"
    assert row["failure_label"] == "success"
    assert row["terminal_speed_m_s"] >= 5.0
    assert -row["height_change_m"] <= 0.40
    for column in (
        "u_norm_requested_delta_a_norm",
        "u_norm_applied_delta_a_norm",
        "delta_cmd_rad_delta_a_cmd",
    ):
        assert column in commands.columns
