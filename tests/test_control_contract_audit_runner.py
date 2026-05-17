from __future__ import annotations

import json

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
    ):
        assert manifest[key] is True
    for key in (
        "high_incidence_validation_claim",
        "controller_implemented",
        "ocp_implemented",
        "tvlqr_implemented",
        "governor_implemented",
        "outer_loop_implemented",
    ):
        assert manifest[key] is False


def test_audit_report_states_contract_only(tmp_path) -> None:
    outputs = run_control_contract_audit(root=tmp_path, run_id=1)
    report = outputs["control_contract_report_md"].read_text(encoding="ascii")

    assert "contracts only" in report
    assert "does not implement a controller" in report
    assert "High-incidence validation claim: `False`" in report
