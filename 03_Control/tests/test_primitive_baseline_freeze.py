from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from run_freeze_primitive_baseline import (
    BLOCKER_COLUMNS,
    FREEZE_DIR_NAME,
    REQUIRED_BASELINE_RUNS,
    REQUIRED_EXTERNAL_TESTS,
    update_stage0_validation_status,
    FreezeOutputs,
    freeze_primitive_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library"


def test_freeze_writes_strict_audit_outputs_and_preserves_source_hashes() -> None:
    paths = _stage0_outputs()
    source_hashes = _baseline_source_hashes(RESULT_ROOT)

    for key, path in paths.items():
        assert path.exists(), key
        if key != "root":
            assert FREEZE_DIR_NAME in path.parts

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    blockers = pd.read_csv(paths["blocker_csv"])
    plot_ready = pd.read_csv(paths["plot_ready_summary_csv"])
    blocker_counts = pd.read_csv(paths["plot_ready_blocker_counts_csv"])

    assert manifest["freeze_gate_status"] == "passed"
    assert manifest["external_validation_required"] is True
    assert manifest["external_validation_status"] in {"pending", "passed"}
    assert manifest["overall_stage0_gate_status"] in {"pending_external_validation", "passed"}
    assert manifest["source_hashes_unchanged_after_writing"] is True
    assert manifest["source_hash_count"] > 0
    assert manifest["rendered_plots_status"] == "deferred"
    assert manifest["plot_ready_csvs_sufficient_for_freeze_audit_only"] is True
    assert manifest["plot_ready_csvs_sufficient_for_final_phase_a_writing_package"] is False
    assert manifest["phase_b_implementation_allowed"] is False
    assert not manifest["freeze_gate_failures"]
    if manifest["external_validation_status"] == "passed":
        assert manifest["overall_stage0_gate_status"] == "passed"
        assert manifest["phase_a_stage0_complete"] is True
    else:
        assert manifest["phase_a_stage0_complete"] is False

    required_runs = {f"{run_id:03d}" for run_id in REQUIRED_BASELINE_RUNS}
    assert {row["run_id"] for row in manifest["baseline_run_inventory"]} == required_runs
    assert all(row["exists"] for row in manifest["baseline_run_inventory"])
    assert all(int(row["hashable_file_count"]) > 0 for row in manifest["baseline_run_inventory"])
    assert all(FREEZE_DIR_NAME not in row["path"] for row in manifest["source_hashes_before"])
    assert all(row["source_run_id"] in required_runs for row in manifest["source_hashes_before"])
    if manifest["external_validation_status"] == "passed":
        assert manifest["source_hash_count"] == len(source_hashes)
        assert {
            row["path"]: row["sha256"]
            for row in manifest["source_hashes_before"]
        } == source_hashes
    assert all((REPO_ROOT / item["path"]).exists() for item in manifest["required_external_validation_tests"])
    assert {item["path"] for item in manifest["required_external_validation_tests"]} == set(REQUIRED_EXTERNAL_TESTS)

    assert set(BLOCKER_COLUMNS).issubset(blockers.columns)
    assert not blockers.empty
    assert not plot_ready.empty
    assert not blocker_counts.empty
    assert "baseline_file_inventory" in set(plot_ready["plot_group"])


def test_blocker_table_contains_future_not_evaluated_gap_contract() -> None:
    paths = _stage0_outputs()
    blockers = pd.read_csv(paths["blocker_csv"])
    future = blockers[blockers["evidence_status"] == "not_evaluated"]
    future_archive = future[future["blocker_scope"] == "future_archive_gap"]

    assert {45.0, 60.0, 90.0, 120.0, 150.0, 180.0}.issubset(
        set(future_archive["target_heading_deg"].dropna().astype(float))
    )
    assert {-1, 1}.issubset(set(future_archive["direction_sign"].dropna().astype(str).astype(int)))
    assert {"lift_sector", "random_stress"}.issubset(set(future_archive["start_condition"]))
    assert {
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    }.issubset(set(future_archive["family"]))
    assert "objective_one_sustained_operation" in set(future["objective"])
    assert "objective_two_volume_coverage" in set(future["objective"])
    assert "real_flight_transfer" in set(future["objective"])
    assert "future_archive_gap" in set(future["blocker_scope"])
    assert "future_mission_gap" in set(future["blocker_scope"])
    assert "future_transfer_gap" in set(future["blocker_scope"])


def test_run003_selected_shortlist_statuses_are_not_blockers() -> None:
    paths = _stage0_outputs()
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    blockers = pd.read_csv(paths["blocker_csv"])
    shortlist_blockers = blockers[blockers["blocker_scope"] == "shortlist_selection"]

    assert "selected_for_w3_stress" not in set(shortlist_blockers["evidence_status"])
    assert "selected_for_governor_seed" not in set(shortlist_blockers["evidence_status"])
    assert {"needs_seed_refinement", "governor_reject_entry_envelope", "boundary_only"}.issubset(
        set(shortlist_blockers["evidence_status"])
    )
    assert manifest["run003_shortlist_blocking_row_count"] == len(shortlist_blockers)
    assert manifest["run003_shortlist_non_blocking_row_count"] == 16

    plot_ready = pd.read_csv(paths["plot_ready_summary_csv"])
    status_counts = plot_ready[plot_ready["plot_group"] == "run003_shortlist_blocking_status"]
    assert dict(zip(status_counts["metric_name"], status_counts["metric_value"])) == {
        "blocking_shortlist_rows": len(shortlist_blockers),
        "non_blocking_shortlist_rows": 16,
    }


def test_claim_boundary_forbids_overclaims_and_widening_growth_answer() -> None:
    paths = _stage0_outputs()
    claim = paths["claim_boundary_md"].read_text(encoding="ascii")
    summary = paths["baseline_summary_md"].read_text(encoding="ascii")

    required_phrases = (
        "deterministic primitive evidence",
        "short governed-transit and rejection evidence",
        "does not answer the final widening-versus-growth research question",
        "sustained updraft exploitation",
        "prolonged confined arena operation",
        "objective-one sustained operation",
        "objective-two volume coverage",
        "volume coverage mission completion",
        "successful real flight transfer",
        "full target ladder evidence",
        "high angle reversal transfer evidence",
    )
    for phrase in required_phrases:
        assert phrase in claim
    assert "Plot-ready CSVs: `written_for_freeze_audit_only`" in summary
    assert "sustained updraft exploitation" in summary
    assert "Phase B implementation allowed: `false`" in summary


def test_manifest_records_phase_b_blocked_and_validation_fields() -> None:
    manifest = json.loads(_stage0_outputs()["manifest_json"].read_text(encoding="ascii"))

    required_fields = {
        "external_validation_status",
        "external_validation_exit_code",
        "external_validation_command",
        "external_validation_test_paths",
        "external_validation_missing_tests",
        "external_validation_stdout_tail",
        "external_validation_stderr_tail",
        "overall_stage0_gate_status",
        "phase_a_stage0_complete",
        "phase_b_implementation_allowed",
        "phase_b_blocked_reason",
        "recommended_next_branch_after_stage0",
        "forbidden_scope_this_pass",
    }
    assert required_fields.issubset(manifest)
    assert manifest["external_validation_test_paths"] == list(REQUIRED_EXTERNAL_TESTS)
    assert manifest["phase_b_implementation_allowed"] is False
    assert manifest["recommended_next_branch_after_stage0"] == "rewrite/phase-b-dense-archive-planning"
    assert "target expansion" in manifest["forbidden_scope_this_pass"]
    assert "dense archive work must start in a separate task" in manifest["phase_b_blocked_reason"]


def test_missing_runs_in_tmp_root_write_outputs_and_fail_freeze_gate(tmp_path: Path) -> None:
    result_root = tmp_path / "09_primitive_library"
    partial = result_root / "002" / "metrics"
    partial.mkdir(parents=True)
    (partial / "primitive_evidence_library_s002.csv").write_text(
        "family,target_heading_deg,direction_sign,start_condition,updraft_config,wind_fidelity,"
        "evaluation_status,failure_label,active_limiting_mechanism,candidate_class,"
        "heading_band_pass,true_safe_trajectory\n"
        "glide,,1,favourable,none,W0,evaluated,success,none,w0_standalone_commandable,True,True\n",
        encoding="ascii",
    )

    paths = freeze_primitive_baseline(
        result_root=result_root,
        baseline_runs=REQUIRED_BASELINE_RUNS,
        run_id=0,
        overwrite=True,
        allow_missing_runs_for_tests=True,
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))

    for key, path in paths.items():
        assert path.exists(), key
    assert manifest["freeze_gate_status"] == "failed"
    assert manifest["overall_stage0_gate_status"] == "failed"
    assert "missing required baseline runs: 003, 004, 005, 006" in manifest["freeze_gate_failures"]
    assert manifest["source_hash_count"] == 1
    assert all(FREEZE_DIR_NAME not in row["path"] for row in manifest["source_hashes_before"])
    assert "Missing required run `003`." in paths["claim_boundary_md"].read_text(encoding="ascii")


def test_cli_does_not_expose_missing_run_tolerance() -> None:
    script = (REPO_ROOT / "03_Control" / "04_Scenarios" / "run_freeze_primitive_baseline.py").read_text(
        encoding="ascii"
    )
    assert "--allow-missing" not in script
    assert "allow_missing_runs_for_tests" in script


def test_validation_update_helper_recomputes_overall_gate(tmp_path: Path) -> None:
    output_root = tmp_path / FREEZE_DIR_NAME
    outputs = FreezeOutputs(
        root=output_root,
        manifest_json=output_root / "manifests" / "frozen_baseline_manifest_s000.json",
        blocker_csv=output_root / "metrics" / "baseline_blockers_s000.csv",
        plot_ready_summary_csv=output_root / "metrics" / "baseline_plot_ready_summary_s000.csv",
        plot_ready_blocker_counts_csv=output_root / "metrics" / "baseline_plot_ready_blocker_counts_s000.csv",
        claim_boundary_md=output_root / "reports" / "claim_boundary_s000.md",
        baseline_summary_md=output_root / "reports" / "baseline_summary_s000.md",
    )
    for path in (outputs.manifest_json.parent, outputs.baseline_summary_md.parent):
        path.mkdir(parents=True)
    outputs.manifest_json.write_text(
        json.dumps(
            {
                "freeze_gate_status": "passed",
                "external_validation_required": True,
                "external_validation_status": "pending",
                "overall_stage0_gate_status": "pending_external_validation",
                "source_hash_count": 1,
                "blocker_row_count": 2,
                "rendered_plots_status": "deferred",
                "baseline_run_inventory": [],
                "allowed_stage0_claim": "narrow baseline",
                "forbidden_claims": ["real flight transfer"],
            }
        ),
        encoding="ascii",
    )
    update_stage0_validation_status(
        outputs=outputs,
        validation={
            "external_validation_status": "passed",
            "external_validation_exit_code": 0,
            "external_validation_command": ["python", "-m", "pytest"],
            "external_validation_test_paths": list(REQUIRED_EXTERNAL_TESTS),
            "external_validation_missing_tests": [],
            "external_validation_stdout_tail": "tests passed",
            "external_validation_stderr_tail": "",
        },
    )
    manifest = json.loads(outputs.manifest_json.read_text(encoding="ascii"))

    assert manifest["external_validation_status"] == "passed"
    assert manifest["overall_stage0_gate_status"] == "passed"
    assert manifest["phase_a_stage0_complete"] is True
    assert manifest["phase_b_implementation_allowed"] is False


def _baseline_source_hashes(result_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for run_id in REQUIRED_BASELINE_RUNS:
        run_dir = result_root / f"{run_id:03d}"
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
                hashes[path.relative_to(REPO_ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def _stage0_outputs() -> dict[str, Path]:
    root = RESULT_ROOT / FREEZE_DIR_NAME
    return {
        "root": root,
        "manifest_json": root / "manifests" / "frozen_baseline_manifest_s000.json",
        "blocker_csv": root / "metrics" / "baseline_blockers_s000.csv",
        "plot_ready_summary_csv": root / "metrics" / "baseline_plot_ready_summary_s000.csv",
        "plot_ready_blocker_counts_csv": root / "metrics" / "baseline_plot_ready_blocker_counts_s000.csv",
        "claim_boundary_md": root / "reports" / "claim_boundary_s000.md",
        "baseline_summary_md": root / "reports" / "baseline_summary_s000.md",
    }
