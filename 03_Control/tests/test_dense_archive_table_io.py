from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dense_archive_table_io import (
    TableManifest,
    csv_gz_text,
    list_table_partitions,
    load_table_manifest,
    file_sha256,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
    read_table_partition,
)


def test_csv_gz_round_trip_and_checksum(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "tables" / "example" / "part-00000.csv.gz"

    partition = write_table_partition(
        frame,
        path,
        storage_format="csv_gz",
        compression_level=1,
    )
    loaded = read_table_partition(path)

    pd.testing.assert_frame_equal(loaded, frame)
    assert partition.table_name == "example"
    assert partition.storage_format == "csv_gz"
    assert partition.row_count == 2
    assert partition.byte_count > 0
    assert len(partition.checksum_sha256) == 64
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_csv_gz_is_deterministic_and_tmp_files_are_ignored(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path_a = tmp_path / "tables" / "example" / "part-00000.csv.gz"
    path_b = tmp_path / "tables" / "example" / "part-00001.csv.gz"

    write_table_partition(frame, path_a, storage_format="csv_gz", compression_level=1)
    write_table_partition(frame, path_b, storage_format="csv_gz", compression_level=1)
    path_a.with_name(f"{path_a.name}.tmp").write_text("ignored", encoding="ascii")

    assert file_sha256(path_a) == file_sha256(path_b)
    assert csv_gz_text(path_a) == "a,b\n1,x\n2,y\n"
    assert path_a.with_name(f"{path_a.name}.tmp") not in list_table_partitions(
        tmp_path,
        "example",
    )


def test_invalid_compression_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="compression_level"):
        write_table_partition(
            pd.DataFrame({"a": [1]}),
            tmp_path / "tables" / "example" / "part-00000.csv.gz",
            storage_format="csv_gz",
            compression_level=10,
        )


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
