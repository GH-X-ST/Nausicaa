from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dense_archive_table_io import load_table_manifest, read_table_partition
from run_lqr_w01_dense_chunked import W01DenseRunConfig, run_lqr_w01_dense_chunked


def test_w01_resume_skips_complete_chunks_and_repair_regenerates_corrupt_chunks(tmp_path: Path) -> None:
    config = W01DenseRunConfig(
        run_id=4,
        output_root=tmp_path,
        rows=18,
        seed=4,
        candidate_chunk_size=9,
        workers=1,
        max_workers=1,
        storage_format="csv_gz",
        compression_level=1,
        candidate_count=1,
    )
    result = run_lqr_w01_dense_chunked(config)
    run_root = Path(result["run_root"])
    first_partition = run_root / "tables" / "w01_primitive_rows" / "c00000.csv.gz"
    first_manifest = run_root / "chunk_manifests" / "w01_primitive_rows" / "c00000.json"
    before = (first_partition.stat().st_mtime_ns, first_manifest.stat().st_mtime_ns)

    with pytest.raises(RuntimeError, match="already exists|already contains"):
        run_lqr_w01_dense_chunked(W01DenseRunConfig(**{**config.__dict__, "resume": False}))

    resumed = run_lqr_w01_dense_chunked(W01DenseRunConfig(**{**config.__dict__, "resume": True}))
    after = (first_partition.stat().st_mtime_ns, first_manifest.stat().st_mtime_ns)
    resumed_summary = pd.read_csv(Path(resumed["run_root"]) / "metrics" / "chunk_summary.csv")
    assert before == after
    assert "skipped" in set(resumed_summary["status"])

    with first_partition.open("ab") as handle:
        handle.write(b"corrupt")
    repaired = run_lqr_w01_dense_chunked(W01DenseRunConfig(**{**config.__dict__, "repair_incomplete": True}))
    repaired_manifest = load_table_manifest(Path(repaired["run_root"]) / "manifests" / "table_manifest.json")
    repaired_frame = pd.concat(
        [
            read_table_partition(Path(repaired["run_root"]) / "tables" / part.relative_path, storage_format=part.storage_format)
            for part in repaired_manifest.tables
        ],
        ignore_index=True,
    )
    assert len(repaired_frame) == 18


def test_w01_worker_chunks_match_serial_chunks(tmp_path: Path) -> None:
    common = {
        "run_id": 5,
        "rows": 24,
        "seed": 5,
        "candidate_chunk_size": 12,
        "storage_format": "csv_gz",
        "compression_level": 1,
        "candidate_count": 2,
    }
    serial = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(output_root=tmp_path / "serial", workers=1, max_workers=1, **common)
    )
    parallel = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(output_root=tmp_path / "parallel", workers=2, max_workers=2, **common)
    )
    serial_summary = pd.read_csv(Path(serial["run_root"]) / "metrics" / "chunk_summary.csv")
    parallel_summary = pd.read_csv(Path(parallel["run_root"]) / "metrics" / "chunk_summary.csv")

    assert set(parallel_summary["status"]) == {"complete"}
    assert serial_summary["checksum_sha256"].tolist() == parallel_summary["checksum_sha256"].tolist()
