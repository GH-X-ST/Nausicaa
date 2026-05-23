from __future__ import annotations

from pathlib import Path

from dense_archive_chunking import ContextChunkSpec, contextual_table_paths, progress_manifest_payload
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


def test_contextual_chunk_paths_use_short_future_layout() -> None:
    spec = ContextChunkSpec(
        run_id=646,
        source_run_id=64,
        result_root=None,
        context_id="W1",
        environment_id="gaussian_single",
        chunk_index=12,
        chunk_count=80,
        chunk_size=1000,
        storage_format="csv_gz",
    )

    paths = contextual_table_paths(
        spec,
        run_root=Path("03_Control/05_Results/feedback_contextual_v1_4/r6/r646"),
    )
    path_text = paths.partition_path.as_posix()

    assert path_text.endswith("tables/contextual_rows/c00012_W1_gaussian-single.csv.gz")
    assert "context_id=" not in path_text
    assert "environment_id=" not in path_text
    assert "chunk_index=" not in path_text
    assert "part-00000" not in path_text
    assert "run_64" not in path_text
    assert len(path_text) <= 120


def test_contextual_chunk_paths_cap_long_audit_tokens() -> None:
    spec = ContextChunkSpec(
        run_id=646,
        source_run_id=64,
        result_root=None,
        context_id="W3",
        environment_id="randomised_gp_corrected_annular_gaussian_single_fan_shift_power_scale",
        chunk_index=99,
        chunk_count=120,
        chunk_size=1000,
        storage_format="csv_gz",
    )

    paths = contextual_table_paths(
        spec,
        run_root=Path("03_Control/05_Results/feedback_contextual_v1_4/r9/w3_649"),
    )
    file_name = paths.partition_path.name

    assert file_name.startswith("c00099_W3_")
    assert file_name.endswith(".csv.gz")
    assert len(file_name) <= len("c00099_W3_") + 28 + len(".csv.gz")
