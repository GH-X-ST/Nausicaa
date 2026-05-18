from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bank_primitive import BANK_CAMPAIGN
from run_bank_primitive_w0 import run_smoke


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


@pytest.fixture(scope="module")
def bank_outputs(tmp_path_factory):
    import run_bank_primitive_w0 as runner

    output_root = tmp_path_factory.mktemp("bank_outputs")
    original_root = runner.DEFAULT_RESULTS_ROOT
    runner.DEFAULT_RESULTS_ROOT = output_root
    try:
        outputs = run_smoke(run_id=1, overwrite=False)
    finally:
        runner.DEFAULT_RESULTS_ROOT = original_root
    return outputs


def test_w0_bank_runner_writes_raw_outputs(bank_outputs) -> None:
    root = bank_outputs["root"]
    required = (
        root / "metrics" / "bank_case_summary_s001.csv",
        root / "metrics" / "bank_w0_left_mild_entry_checks_s001.csv",
        root / "metrics" / "bank_w0_left_mild_exit_checks_s001.csv",
        root / "metrics" / "bank_w0_left_mild_bank_checks_s001.csv",
        root / "metrics" / "bank_w0_left_mild_trajectory_s001.csv",
        root / "metrics" / "bank_w0_left_mild_commands_s001.csv",
        root / "metrics" / "bank_w0_left_mild_metrics_s001.csv",
        root / "manifests" / "bank_primitive_manifest_s001.json",
        root / "reports" / "bank_primitive_report_s001.md",
    )

    assert root == bank_outputs["root"].parent / "001"
    assert root.parent.name == BANK_CAMPAIGN
    for path in required:
        assert path.exists()


def test_w0_bank_manifest_flags_and_relative_paths(bank_outputs) -> None:
    manifest = json.loads(bank_outputs["manifest_json"].read_text(encoding="ascii"))

    assert manifest["overall_status"] == "pass"
    assert manifest["run_id"] == "s001"
    assert manifest["required_case_success"] is True
    assert manifest["diagnostic_failures"] == ["bank_w0_0p80_handoff_boundary"]
    assert manifest["actual_bank_primitive_implemented"] is True
    assert manifest["actual_glide_primitive_implemented"] is True
    assert manifest["actual_recovery_primitive_implemented"] is True
    assert manifest["primitive_implemented"] is True
    assert manifest["controller_implemented"] is True
    assert manifest["bank_updraft_encounter_role"] == "w0_lateral_repositioning_baseline_only"
    for key in (
        "actual_agile_reversal_primitive_implemented",
        "updraft_validation_claim",
        "w1_w2_w3_updraft_validation_claim",
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


def test_w0_bank_metrics_run_id_and_command_columns(bank_outputs) -> None:
    root = bank_outputs["root"]
    metric_files = sorted((root / "metrics").glob("*_metrics_s001.csv"))
    commands = pd.read_csv(root / "metrics" / "bank_w0_right_mild_commands_s001.csv")

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


def test_w0_bank_no_figure_files_created(bank_outputs) -> None:
    figure_dir = bank_outputs["root"] / "figures"

    assert figure_dir.exists()
    assert not list(figure_dir.glob("*"))
    assert not list(bank_outputs["root"].rglob("*.png"))
    assert not list(bank_outputs["root"].rglob("*.pdf"))
    assert not list(bank_outputs["root"].rglob("*.svg"))


def test_w0_bank_case_summary_records_required_and_boundary(bank_outputs) -> None:
    summary = pd.read_csv(bank_outputs["case_summary_csv"])
    rows = {row["case_name"]: row for _, row in summary.iterrows()}

    for name in ("bank_w0_left_mild", "bank_w0_right_mild"):
        _assert_csv_bool(rows[name]["success"], True)
        _assert_csv_bool(rows[name]["terminal_glide_entry_proxy"], True)
        assert rows[name]["terminal_glide_entry_x_margin_m"] > 0.0
        assert rows[name]["terminal_true_safe_x_margin_m"] > 0.0
    assert rows["bank_w0_left_mild"]["lateral_displacement_m"] < -0.05
    assert rows["bank_w0_right_mild"]["lateral_displacement_m"] > 0.05

    boundary = rows["bank_w0_0p80_handoff_boundary"]
    assert boundary["role"] == "diagnostic"
    _assert_csv_bool(boundary["success"], False)
    _assert_csv_bool(boundary["terminal_glide_entry_proxy"], False)
    assert boundary["failure_label"] == "terminal_recovery_limited"
    assert boundary["notes"] == "glide_entry_x_bound_limited"
    assert boundary["terminal_true_safe_x_margin_m"] > 0.0
    assert boundary["terminal_glide_entry_x_margin_m"] < 0.0


def test_w0_bank_report_states_no_updraft_validation(bank_outputs) -> None:
    report = bank_outputs["report_md"].read_text(encoding="ascii")

    assert "not an updraft robustness result" in report
    assert "not high-incidence validation" in report
    assert "w0_lateral_repositioning_baseline_only" in report
