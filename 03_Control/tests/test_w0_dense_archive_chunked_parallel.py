from __future__ import annotations

import json
from pathlib import Path

import run_w0_dense_archive_chunked as chunked


def test_auto_worker_selection_uses_8_on_i9_32gb() -> None:
    decision = chunked.worker_count_decision(
        "auto",
        logical_cpu_count=20,
        memory_total_gb=32.0,
        memory_safety_margin_gb=8.0,
        estimated_worker_memory_gb=2.0,
        max_workers=8,
    )

    assert decision.selected_worker_count == 8
    assert decision.fallback_reason == "none"


def test_memory_guardrail_records_fallback() -> None:
    decision = chunked.worker_count_decision(
        "auto",
        logical_cpu_count=20,
        memory_total_gb=32.0,
        memory_safety_margin_gb=8.0,
        estimated_worker_memory_gb=4.0,
        max_workers=8,
    )

    assert decision.selected_worker_count == 6
    assert "memory_guardrail" in decision.fallback_reason


def test_explicit_8_worker_schedule_has_unique_paths(tmp_path: Path) -> None:
    config = chunked.W0ChunkedRunConfig(
        run_id=13,
        planning_run_id=12,
        result_root=tmp_path / "11_w0_dense_archive",
        target_trials_total=16,
        target_trials_per_branch=8,
        chunk_size=2,
        workers=8,
        max_workers=8,
    )

    schedule = chunked.build_chunk_schedule(config)
    paths = [
        chunked.output_paths(item).partition_path
        for item in schedule
    ]

    assert len(schedule) == 8
    assert len(paths) == len(set(paths))
    assert chunked.resolve_worker_count(
        8,
        logical_cpu_count=20,
        memory_total_gb=32.0,
        estimated_worker_memory_gb=2.0,
        max_workers=8,
    ) == 8


def test_dry_run_manifest_records_worker_memory_and_gpu_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(chunked.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(chunked, "_memory_total_gb", lambda: 32.0)

    paths = chunked.run_w0_dense_archive_chunked(
        run_id=99,
        planning_run_id=98,
        result_root=tmp_path / "11_w0_dense_archive",
        target_trials_total=16,
        target_trials_per_branch=8,
        chunk_size=2,
        workers=8,
        max_workers=8,
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    assert manifest["selected_worker_count"] == 8
    assert manifest["max_workers"] == 8
    assert manifest["memory_safety_margin_gb"] == 8.0
    assert manifest["estimated_worker_memory_gb"] == 2.0
    assert "GPU acceleration is deferred" in manifest["gpu_acceleration_assessment"]
    assert "--workers 8 --max-workers 8" in manifest["recommended_production_command"]
