from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recovery_primitive import RECOVERY_CAMPAIGN
from run_recovery_primitive_w0 import run_smoke


def _assert_csv_bool(value: object, expected: bool) -> None:
    if isinstance(value, np.bool_):
        assert bool(value) is expected
        return
    if isinstance(value, bool):
        assert value is expected
        return
    if isinstance(value, str) and value.strip() in {"True", "False"}:
        assert (value.strip() == "True") is expected
        return
    raise AssertionError(f"unexpected CSV boolean value: {value!r}")


def test_w0_recovery_runner_writes_raw_outputs(monkeypatch, tmp_path) -> None:
    import run_recovery_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    root = tmp_path / RECOVERY_CAMPAIGN / "001"

    required = (
        root / "metrics" / "recovery_case_summary_s001.csv",
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_entry_checks_s001.csv",
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_exit_checks_s001.csv",
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_recovery_checks_s001.csv",
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_trajectory_s001.csv",
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_commands_s001.csv",
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_metrics_s001.csv",
        root / "manifests" / "recovery_primitive_manifest_s001.json",
        root / "reports" / "recovery_primitive_report_s001.md",
    )
    for path in required:
        assert path.exists()
    assert outputs["root"] == root


def test_w0_recovery_manifest_flags_and_relative_paths(monkeypatch, tmp_path) -> None:
    import run_recovery_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))

    assert manifest["overall_status"] == "pass"
    assert manifest["run_id"] == "s001"
    assert manifest["required_case_success"] is True
    assert manifest["diagnostic_failures"] == ["recovery_w0_0p80_boundary"]
    assert manifest["actual_recovery_primitive_implemented"] is True
    assert manifest["actual_glide_primitive_implemented"] is True
    assert manifest["primitive_implemented"] is True
    assert manifest["controller_implemented"] is True
    for key in (
        "actual_bank_primitive_implemented",
        "actual_agile_reversal_primitive_implemented",
        "ocp_implemented",
        "tvlqr_implemented",
        "governor_implemented",
        "outer_loop_implemented",
        "vicon_implemented",
        "hardware_implemented",
        "high_incidence_validation_claim",
        "raw_normalised_commands_enter_state_derivative",
        "terminal_glide_entry_proxy_is_governor_proof",
    ):
        assert manifest[key] is False
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())


def test_w0_recovery_metrics_run_id_and_command_columns(monkeypatch, tmp_path) -> None:
    import run_recovery_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    root = outputs["root"]
    metric_files = sorted((root / "metrics").glob("*_metrics_s001.csv"))
    commands = pd.read_csv(
        root
        / "metrics"
        / "recovery_w0_moderate_attitude_rate_commands_s001.csv"
    )

    assert metric_files
    for path in metric_files:
        row = pd.read_csv(path).iloc[0]
        assert row["run_id"] == "s001"
    for column in (
        "u_norm_requested_delta_a_norm",
        "u_norm_applied_delta_a_norm",
        "delta_cmd_rad_delta_a_cmd",
    ):
        assert column in commands.columns


def test_w0_recovery_no_figure_files_created(monkeypatch, tmp_path) -> None:
    import run_recovery_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    figure_dir = outputs["root"] / "figures"

    assert figure_dir.exists()
    assert not list(figure_dir.glob("*"))
    assert not list(outputs["root"].rglob("*.png"))
    assert not list(outputs["root"].rglob("*.pdf"))
    assert not list(outputs["root"].rglob("*.svg"))


def test_w0_recovery_case_summary_records_handoff_boundary(monkeypatch, tmp_path) -> None:
    import run_recovery_primitive_w0 as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    summary = pd.read_csv(outputs["case_summary_csv"])
    rows = {row["case_name"]: row for _, row in summary.iterrows()}

    _assert_csv_bool(rows["recovery_w0_moderate_attitude_rate"]["success"], True)
    _assert_csv_bool(
        rows["recovery_w0_moderate_attitude_rate"]["terminal_glide_entry_proxy"],
        True,
    )
    boundary = rows["recovery_w0_0p80_boundary"]
    assert boundary["role"] == "diagnostic"
    _assert_csv_bool(boundary["success"], False)
    assert boundary["failure_label"] == "terminal_recovery_limited"
    assert boundary["notes"] == "glide_entry_x_bound_limited"
    assert boundary["terminal_glide_entry_x_margin_m"] < 0.0
