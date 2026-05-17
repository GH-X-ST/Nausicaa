from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from primitive_interface import (
    PRIMITIVE_INTERFACE_CAMPAIGN,
    PrimitiveExecutionConfig,
    build_interface_smoke_spec,
    execute_open_loop_primitive_interface,
    write_primitive_interface_outputs,
)
from rollout import make_constant_command_schedule
from run_primitive_interface_smoke import run_smoke


def _smoke_state() -> np.ndarray:
    state = np.zeros(15)
    state[0:3] = [2.5, 2.2, 1.5]
    state[6] = 6.5
    return state


def _execute_smoke():
    config = PrimitiveExecutionConfig(dt_s=0.02, t_final_s=0.24)
    schedule = make_constant_command_schedule(np.zeros(3), 0.24, 0.02)
    return execute_open_loop_primitive_interface(
        build_interface_smoke_spec(),
        _smoke_state(),
        schedule,
        config,
        aircraft=None,
        wind_model=None,
    )


def test_smoke_runner_writes_required_files_and_relative_manifest_paths(tmp_path) -> None:
    result = _execute_smoke()
    outputs = write_primitive_interface_outputs(
        result,
        tmp_path,
        PRIMITIVE_INTERFACE_CAMPAIGN,
        1,
    )
    root = tmp_path / PRIMITIVE_INTERFACE_CAMPAIGN / "001"

    required = (
        root / "metrics" / "entry_checks_s001.csv",
        root / "metrics" / "exit_checks_s001.csv",
        root / "metrics" / "trajectory_s001.csv",
        root / "metrics" / "commands_s001.csv",
        root / "metrics" / "primitive_interface_metrics_s001.csv",
        root / "manifests" / "primitive_interface_manifest_s001.json",
        root / "reports" / "primitive_interface_report_s001.md",
    )
    for path in required:
        assert path.exists()
    assert outputs["root"] == root

    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    assert manifest["overall_status"] == "pass"
    assert manifest["interface_checks_pass"] is True
    assert manifest["entry_checks_pass"] is True
    assert manifest["exit_checks_pass"] is True
    assert manifest["rollout_ran"] is True
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())
    assert "manifest_json" in manifest["output_files"]
    assert "report_md" in manifest["output_files"]


def test_smoke_manifest_flags_remain_audit_only(tmp_path) -> None:
    outputs = write_primitive_interface_outputs(
        _execute_smoke(),
        tmp_path,
        PRIMITIVE_INTERFACE_CAMPAIGN,
        1,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))

    assert manifest["primitive_interface_implemented"] is True
    for key in (
        "primitive_implemented",
        "primitive_controller_implemented",
        "controller_implemented",
        "actual_glide_primitive_implemented",
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
        "primitive_success",
        "success",
        "raw_normalised_commands_enter_state_derivative",
    ):
        assert manifest[key] is False
    assert manifest["command_bridge"] == "normalised_command_to_surface_rad"
    assert manifest["state_derivative_command_input"] == "delta_cmd_rad"


def test_smoke_csvs_are_nonempty_and_metric_row_has_contract_fields(tmp_path) -> None:
    outputs = write_primitive_interface_outputs(
        _execute_smoke(),
        tmp_path,
        PRIMITIVE_INTERFACE_CAMPAIGN,
        1,
    )
    entry = pd.read_csv(outputs["entry_checks_csv"])
    exit_checks = pd.read_csv(outputs["exit_checks_csv"])
    trajectory = pd.read_csv(outputs["trajectory_csv"])
    commands = pd.read_csv(outputs["commands_csv"])
    metrics = pd.read_csv(outputs["metrics_csv"])
    row = metrics.iloc[0]

    assert not entry.empty
    assert not exit_checks.empty
    assert not trajectory.empty
    assert not commands.empty
    for column in (
        "primitive_name",
        "primitive_family",
        "finite_state_success",
        "rollout_success",
        "primitive_success",
        "success",
        "failure_label",
        "notes",
    ):
        assert column in metrics.columns
    assert str(row["finite_state_success"]) == "True"
    assert str(row["rollout_success"]) == "True"
    assert str(row["primitive_success"]) == "False"
    assert str(row["success"]) == "False"
    assert row["failure_label"] == "not_run"
    assert row["notes"] == "primitive_interface_smoke_no_controller"
    assert row["run_id"] == "s001"


def test_entry_failure_skips_rollout_and_does_not_write_fake_logs(tmp_path) -> None:
    config = PrimitiveExecutionConfig(dt_s=0.02, t_final_s=0.24)
    schedule = make_constant_command_schedule(np.zeros(3), 0.24, 0.02)
    state = _smoke_state()
    state[0] = 7.0
    result = execute_open_loop_primitive_interface(
        build_interface_smoke_spec(),
        state,
        schedule,
        config,
        aircraft=object(),
        wind_model=None,
    )

    outputs = write_primitive_interface_outputs(
        result,
        tmp_path,
        PRIMITIVE_INTERFACE_CAMPAIGN,
        1,
    )
    root = tmp_path / PRIMITIVE_INTERFACE_CAMPAIGN / "001"
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    metrics = pd.read_csv(outputs["metrics_csv"])
    row = metrics.iloc[0]

    assert result.rollout_result is None
    assert manifest["overall_status"] == "needs_review"
    assert manifest["entry_checks_pass"] is False
    assert manifest["exit_checks_pass"] is False
    assert manifest["rollout_ran"] is False
    assert row["failure_label"] == "entry_set_violation"
    assert row["notes"] == "entry_set_violation"
    assert not (root / "metrics" / "trajectory_s001.csv").exists()
    assert not (root / "metrics" / "commands_s001.csv").exists()
    assert not (root / "metrics" / "exit_checks_s001.csv").exists()


def test_run_smoke_uses_default_result_tree(monkeypatch, tmp_path) -> None:
    import run_primitive_interface_smoke as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_smoke(run_id=1, overwrite=False)
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))

    assert outputs["root"] == tmp_path / PRIMITIVE_INTERFACE_CAMPAIGN / "001"
    assert manifest["overall_status"] == "pass"
    assert manifest["rollout_ran"] is True
