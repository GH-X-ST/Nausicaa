from __future__ import annotations

from dense_archive_chunking import progress_manifest_payload
from dense_archive_runtime import RUNTIME_CORE_VERSION, STORAGE_CONTRACT_VERSION, WorkerCountDecision


def test_progress_manifest_payload_preserves_runtime_and_worker_metadata() -> None:
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
        run_id=14,
        planning_run_id=13,
        simulation_stage="paired_w0_w1_proof",
        worker_decision=decision,
        storage_format="csv_gz",
        latency_case="nominal",
        resume=True,
        repair_incomplete=False,
        continue_on_chunk_failure=False,
        chunks=[],
        recommended_command="python proof command",
        branch_decision_scope="branch_local_only_no_cross_layout_decision_transfer",
        failures=[],
    )

    assert payload["runtime_core_version"] == RUNTIME_CORE_VERSION
    assert payload["storage_contract_version"] == STORAGE_CONTRACT_VERSION
    assert payload["selected_worker_count"] == 8
    assert payload["storage_format"] == "csv_gz"
