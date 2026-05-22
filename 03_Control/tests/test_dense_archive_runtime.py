from __future__ import annotations

import dense_archive_runtime as runtime


def test_runtime_manifest_fields_preserve_contextual_storage_contract() -> None:
    payload = runtime.dense_run_manifest_fields(
        run_stage="contextual_foundation",
        environment_context="local_flow_features",
    )

    assert payload["runtime_core_version"] == runtime.RUNTIME_CORE_VERSION
    assert payload["storage_contract_version"] == runtime.STORAGE_CONTRACT_VERSION
    assert payload["max_generated_file_size_mb"] == 100.0
    assert "chunked" in runtime.RUNTIME_STORAGE_CONTRACT
    assert "resumable" in runtime.RUNTIME_STORAGE_CONTRACT
    assert "compressed" in runtime.RUNTIME_STORAGE_CONTRACT
    assert "worker-enabled" in runtime.RUNTIME_STORAGE_CONTRACT
    assert "checksum-manifested" in runtime.RUNTIME_STORAGE_CONTRACT


def test_worker_count_decision_caps_and_records_memory_guardrail() -> None:
    decision = runtime.worker_count_decision(
        16,
        logical_cpu_count=20,
        memory_total_gb=16.0,
        estimated_worker_memory_gb=3.0,
        max_workers=8,
    )

    assert 1 <= decision.selected_worker_count <= 8
    assert "capped_by_max_workers_8" in decision.fallback_reason
