from __future__ import annotations

from pathlib import Path

import pandas as pd

from dense_archive_table_io import load_table_manifest
from run_lqr_w01_dense_chunked import W01DenseRunConfig, run_lqr_w01_dense_chunked


def test_w01_dry_run_writes_compact_manifests_and_no_partitions(tmp_path: Path) -> None:
    result = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(
            run_id=1,
            output_root=tmp_path,
            rows=60,
            seed=1,
            candidate_chunk_size=20,
            workers=1,
            max_workers=1,
            dry_run_schedule=True,
            candidate_count=1,
        )
    )
    run_root = Path(result["run_root"])
    table_manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    chunk_summary = pd.read_csv(run_root / "metrics" / "chunk_summary.csv")

    assert table_manifest.tables == ()
    assert not (run_root / "tables" / "w01_primitive_rows").exists()
    assert set(chunk_summary["status"]) == {"scheduled"}
    assert (run_root / "manifests" / "primitive_variant_registry.json").is_file()
