from __future__ import annotations

import json

import pandas as pd

from run_control_contract_audit import run_control_contract_audit


def test_audit_runner_writes_required_outputs(tmp_path) -> None:
    outputs = run_control_contract_audit(root=tmp_path, run_id=1)
    root = tmp_path / "00_contracts" / "001"

    assert outputs["root"] == root
    required = [
        root / "metrics" / "state_command_contract_audit.csv",
        root / "metrics" / "arena_contract_audit.csv",
        root / "metrics" / "primitive_contract_audit.csv",
        root / "metrics" / "metric_schema_audit.csv",
        root / "metrics" / "scenario_contract_audit.csv",
        root / "manifests" / "control_contract_manifest.json",
        root / "reports" / "control_contract_report.md",
    ]
    for path in required:
        assert path.exists()


def test_audit_manifest_contains_required_flags(tmp_path) -> None:
    outputs = run_control_contract_audit(root=tmp_path, run_id=1)
    manifest = json.loads(
        outputs["control_contract_manifest_json"].read_text(encoding="ascii")
    )

    for key in (
        "state_order_pass",
        "command_order_pass",
        "control_sign_convention_recorded",
        "arena_bounds_pass",
        "true_safe_and_tracker_separate",
        "primitive_contract_pass",
        "metric_schema_pass",
        "scenario_contract_pass",
        "result_path_contract_pass",
        "normalised_to_radian_command_bridge_recorded",
    ):
        assert manifest[key] is True
    for key in (
        "high_incidence_validation_claim",
        "controller_implemented",
        "ocp_implemented",
        "tvlqr_implemented",
        "governor_implemented",
        "outer_loop_implemented",
        "raw_normalised_commands_enter_state_derivative",
    ):
        assert manifest[key] is False
    assert manifest["state_derivative_command_input"] == "delta_cmd_rad"
    assert manifest["validation_commands"] == [
        "python 03_Control/04_Scenarios/run_control_contract_audit.py --overwrite",
        "python -m pytest -q tests/test_control_contract_state_command.py "
        "tests/test_control_contract_arena.py "
        "tests/test_control_contract_primitive_metric.py "
        "tests/test_control_contract_scenario_paths.py "
        "tests/test_control_contract_audit_runner.py",
    ]


def test_audit_report_states_contract_only(tmp_path) -> None:
    outputs = run_control_contract_audit(root=tmp_path, run_id=1)
    report = outputs["control_contract_report_md"].read_text(encoding="ascii")

    assert "contracts only" in report
    assert "does not implement a controller" in report
    assert "normalised_command_to_surface_rad" in report
    assert "delta_cmd_rad" in report
    assert "never raw normalised commands" in report
    assert "High-incidence validation claim: `False`" in report
    assert "python 03_Control/04_Scenarios/run_control_contract_audit.py --overwrite" in (
        report
    )


def test_audit_csv_records_command_bridge(tmp_path) -> None:
    outputs = run_control_contract_audit(root=tmp_path, run_id=1)
    table = pd.read_csv(outputs["state_command_contract_audit_csv"])
    row = table.iloc[0]

    assert row["command_units"] == "rad"
    assert row["command_interface_to_state_derivative"] == "delta_cmd_rad"
    assert row["normalised_to_radian_bridge"] == "normalised_command_to_surface_rad"
    assert str(row["raw_normalised_commands_enter_state_derivative"]) == "False"
    assert "delta_a" in row["aggregate_limits"]
