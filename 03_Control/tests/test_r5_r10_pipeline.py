from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_r5_r10_pipeline as pipeline


def test_docs_alignment_guard_repeats_and_writes_audit(tmp_path: Path) -> None:
    run_root = tmp_path / "pipeline" / "001"
    (run_root / "metrics").mkdir(parents=True)
    guard = pipeline.DocsAlignmentGuard(run_root)
    pipeline._ACTIVE_DOCS_ALIGNMENT_GUARD = guard
    try:
        first = pipeline.read_docs_alignment_snapshot("pipeline_start", "PIPELINE", "baseline")
        second = pipeline.read_docs_alignment_snapshot("before_stage_preflight", "R5", "preflight")
    finally:
        pipeline._ACTIVE_DOCS_ALIGNMENT_GUARD = None

    audit = pd.read_csv(run_root / "metrics" / "docs_alignment_audit.csv")
    assert first.status == "baseline_recorded"
    assert second.status == "aligned"
    assert len(audit) == 2 * len(pipeline.CONTROLLING_DOCS)
    assert {
        "path",
        "readable",
        "byte_count",
        "sha256",
        "modified_time_utc",
        "checkpoint",
        "timestamp_utc",
        "docs_alignment_status",
    }.issubset(audit.columns)
    assert "docs/Glider_Control_Project_Plan.md" in second.controlling_hashes


def test_terminate_blocked_decision_includes_docs_alignment_fields(tmp_path: Path) -> None:
    run_root = tmp_path / "pipeline" / "001"
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "manifests").mkdir()
    (run_root / "reports").mkdir()
    guard = pipeline.DocsAlignmentGuard(run_root)
    config = pipeline.R5R10PipelineConfig(output_root=tmp_path / "pipeline", run_id=1)
    context: dict[str, object] = {"run_id": 1, "run_root": run_root.as_posix(), "stages": {}, "status": "running"}
    pipeline._ACTIVE_DOCS_ALIGNMENT_GUARD = guard
    try:
        pipeline.read_docs_alignment_snapshot("pipeline_start", "PIPELINE", "baseline")
        result = pipeline._terminate_blocked(
            run_root=run_root,
            config=config,
            context=context,
            guard=guard,
            decision_rows=[],
            stage_summary_rows=[],
            reason="unit_test_block",
            stage_id="R5",
            checkpoint_context="unit_test",
        )
    finally:
        pipeline._ACTIVE_DOCS_ALIGNMENT_GUARD = None

    decision = pd.read_csv(run_root / "metrics" / "decision_log.csv")
    assert result["status"] == "blocked"
    assert decision["docs_alignment_status"].iloc[-1] == "aligned"
    assert decision["docs_changed_during_run"].iloc[-1] in (False, "False")
    assert decision["docs_alignment_checkpoint"].iloc[-1] == "before_terminate_blocked"
    assert "controlling_hashes" in decision.columns


def test_official_run_label_targets_direct_sibling_stage_roots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "W01_OUTPUT_ROOT", tmp_path / "w01_dense")
    monkeypatch.setattr(pipeline, "W3_OUTPUT_ROOT", tmp_path / "w3_survival")
    monkeypatch.setattr(pipeline, "POST_W3_OUTPUT_ROOT", tmp_path / "post_w3_library_size_study")
    monkeypatch.setattr(pipeline, "OUTCOME_OUTPUT_ROOT", tmp_path / "outcome_model")
    monkeypatch.setattr(pipeline, "DEFAULT_CHANGED_OUTPUT_ROOT", tmp_path / "changed_case_validation")
    monkeypatch.setattr(pipeline, "DEFAULT_HELDOUT_CHANGED_OUTPUT_ROOT", tmp_path / "heldout_changed_case_validation")

    config = pipeline.R5R10PipelineConfig(
        output_root=tmp_path / "r5_r11_pipeline",
        run_id=1,
        run_label="A01",
        stage_run_label="A01",
        start_stage="R5",
        stop_after_stage="R11",
    )
    run_root = tmp_path / "r5_r11_pipeline" / "A01"
    expected_roots = {
        run_root,
        tmp_path / "w01_dense" / "A01",
        tmp_path / "w3_survival" / "A01",
        tmp_path / "post_w3_library_size_study" / "A01",
        tmp_path / "outcome_model" / "A01",
        tmp_path / "changed_case_validation" / "A01",
        tmp_path / "heldout_changed_case_validation" / "A01",
    }
    for root in expected_roots:
        root.mkdir(parents=True)

    conflicts = set(pipeline._official_run_label_conflicts(config=config, run_id=1, run_root=run_root))
    assert conflicts == expected_roots
    assert pipeline._stage_run_id(config, tmp_path / "w01_dense") == 1
