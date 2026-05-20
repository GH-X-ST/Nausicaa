from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dense_archive_table_io import (
    TableManifest,
    load_table_manifest,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
    read_table_partition,
)


def test_csv_gz_round_trip_and_checksum(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "tables" / "example" / "part-00000.csv.gz"

    partition = write_table_partition(frame, path, storage_format="csv_gz")
    loaded = read_table_partition(path)

    pd.testing.assert_frame_equal(loaded, frame)
    assert partition.table_name == "example"
    assert partition.storage_format == "csv_gz"
    assert partition.row_count == 2
    assert partition.byte_count > 0
    assert len(partition.checksum_sha256) == 64


def test_auto_falls_back_to_csv_gz_without_parquet_engine() -> None:
    assert resolve_storage_format("auto") == "csv_gz"
    with pytest.raises(RuntimeError, match="pyarrow or fastparquet"):
        resolve_storage_format("parquet")


def test_table_manifest_write_load(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1]})
    partition = write_table_partition(
        frame,
        tmp_path / "tables" / "example" / "part-00000.csv.gz",
        storage_format="csv_gz",
    )
    manifest = TableManifest(
        run_id=13,
        root="root",
        storage_format="csv_gz",
        tables=(partition,),
    )
    path = tmp_path / "manifest.json"

    write_table_manifest(path, manifest)
    loaded = load_table_manifest(path)

    assert loaded == manifest
