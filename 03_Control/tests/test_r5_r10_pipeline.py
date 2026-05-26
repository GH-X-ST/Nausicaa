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
    assert "docs/R5_R10_Full_Evidence_Execution_Plan.md" in second.controlling_hashes


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
