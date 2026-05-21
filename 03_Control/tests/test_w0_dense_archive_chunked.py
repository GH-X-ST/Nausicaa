from __future__ import annotations

import json
from pathlib import Path

import run_w0_dense_archive_chunked as chunked


def test_chunked_dry_run_stop_after_chunks_records_pending_subset(tmp_path: Path) -> None:
    paths = chunked.run_w0_dense_archive_chunked(
        run_id=13,
        planning_run_id=12,
        result_root=tmp_path / "11_w0_dense_archive",
        target_trials_total=8,
        target_trials_per_branch=4,
        chunk_size=1,
        workers=1,
        max_workers=1,
        stop_after_chunks=3,
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    assert manifest["scheduled_chunk_count"] == 8
    assert manifest["pending_chunk_count"] == 8
    assert manifest["selected_worker_count"] == 1
    assert len(manifest["chunks"]) == 8
    assert manifest["chunks"][0]["status"] == "pending"
    assert manifest["chunks"][0]["manifest_path"].endswith("chunk-00000.json")
