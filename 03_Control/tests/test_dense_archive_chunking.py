from __future__ import annotations

from dense_archive_chunking import progress_manifest_payload
from dense_archive_runtime import RUNTIME_CORE_VERSION, STORAGE_CONTRACT_VERSION, WorkerCountDecision


def test_progress_manifest_payload_preserves_contextual_runtime_metadata() -> None:
    decision = WorkerCountDecision(
        requested=8,
        selected_worker_count=8,
        max_workers=8,
        os_cpu_count=20,
        memory_total_gb=32.0,
        memory_safety_margin_gb=8.0,
        estimated_worker_memory_gb=2.0,
        fallback_reason="none",
    )

    payload = progress_manifest_payload(
        status="dry_run",
        run_id=1,
        source_run_id=0,
        run_stage="contextual_foundation",
        worker_decision=decision,
        storage_format="csv_gz",
        latency_case="nominal",
        resume=True,
        repair_incomplete=False,
        continue_on_chunk_failure=False,
        chunks=[],
        recommended_command="python run_ctx_archive.py --dry-run-schedule",
        context_decision_scope="context_features_only",
        failures=[],
    )

    assert payload["runtime_core_version"] == RUNTIME_CORE_VERSION
    assert payload["storage_contract_version"] == STORAGE_CONTRACT_VERSION
    assert payload["selected_worker_count"] == 8
    assert payload["storage_format"] == "csv_gz"
    assert payload["context_decision_scope"] == "context_features_only"
