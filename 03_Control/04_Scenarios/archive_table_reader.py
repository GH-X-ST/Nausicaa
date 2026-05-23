from __future__ import annotations

from pathlib import Path

import pandas as pd

from dense_archive_table_io import (
    filesystem_path,
    load_table_manifest,
    read_table_partition,
)


def read_archive_table(source: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read archive rows from a manifest, run root, or table partition."""

    path = Path(source)
    fs_path = filesystem_path(path)
    if fs_path.is_dir():
        manifest = path / "manifests" / "table_manifest.json"
        if filesystem_path(manifest).is_file():
            return _read_manifest(manifest, max_rows=max_rows)
        partitions = sorted(
            item for item in fs_path.rglob("*") if item.is_file() and _is_partition_name(item.name)
        )
        return _concat_partitions(partitions, max_rows=max_rows)
    if path.name == "table_manifest.json":
        return _read_manifest(path, max_rows=max_rows)
    if _is_partition_name(path.name):
        frame = read_table_partition(path)
        return frame.head(max_rows) if max_rows is not None else frame
    raise ValueError(f"unsupported archive table source: {path}")


def _read_manifest(path: Path, *, max_rows: int | None) -> pd.DataFrame:
    manifest = load_table_manifest(path)
    root = Path(manifest.root)
    frames = []
    remaining = None if max_rows is None else int(max_rows)
    for partition in manifest.tables:
        partition_path = root / "tables" / partition.relative_path
        frame = read_table_partition(partition_path, storage_format=partition.storage_format)
        if remaining is not None:
            frame = frame.head(remaining)
            remaining -= len(frame)
        frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _concat_partitions(paths: list[Path], *, max_rows: int | None) -> pd.DataFrame:
    frames = []
    remaining = None if max_rows is None else int(max_rows)
    for path in paths:
        frame = read_table_partition(Path(path))
        if remaining is not None:
            frame = frame.head(remaining)
            remaining -= len(frame)
        frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _is_partition_name(name: str) -> bool:
    lower = str(name).lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz") or lower.endswith(".parquet")
