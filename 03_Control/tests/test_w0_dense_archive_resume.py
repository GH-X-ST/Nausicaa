from __future__ import annotations

from pathlib import Path

import pytest

import run_w0_dense_archive_chunk as chunk
import run_w0_dense_archive_chunked as chunked


def test_corrupt_chunks_rejected_by_default_and_repaired_when_requested(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    cfg = chunk.W0ChunkConfig(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        layout_branch_id="single_fan_branch",
        chunk_index=0,
        chunk_count=1,
        chunk_size=2,
        storage_format="csv_gz",
    )
    paths = chunk.output_paths(cfg)
    paths.manifest_json.parent.mkdir(parents=True)
    paths.manifest_json.write_text("{\"status\":\"incomplete\"}\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="corrupt/incomplete"):
        chunked.run_w0_dense_archive_chunked(
            run_id=13,
            planning_run_id=12,
            result_root=result_root,
            target_trials_total=4,
            target_trials_per_branch=2,
            chunk_size=2,
            workers=1,
            max_workers=1,
            dry_run_schedule=True,
            resume=True,
        )

    chunked.run_w0_dense_archive_chunked(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        target_trials_total=4,
        target_trials_per_branch=2,
        chunk_size=2,
        workers=1,
        max_workers=1,
        dry_run_schedule=True,
        resume=True,
        repair_incomplete=True,
    )

    assert not paths.manifest_json.exists()
