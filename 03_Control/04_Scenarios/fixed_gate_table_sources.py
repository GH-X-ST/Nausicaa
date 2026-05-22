from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"

for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import (  # noqa: E402
    file_sha256,
    filesystem_path,
    list_table_partitions,
    load_table_manifest,
    read_table_partition,
)


DEFAULT_ARCHIVE_PASS_DIR = "003_fixed_gate_w0_w1_proof_archive"


def read_fixed_gate_table_source(source: Path, *, table_name: str) -> pd.DataFrame:
    """Read a fixed-gate table from a legacy CSV, table manifest, or run root."""

    source_path = Path(source)
    if filesystem_path(source_path).is_file():
        if source_path.name == "table_manifest.json":
            return _read_table_manifest(source_path, table_name=table_name)
        return read_table_partition(source_path)

    manifest_path = _find_table_manifest(source_path)
    if manifest_path is not None:
        return _read_table_manifest(manifest_path, table_name=table_name)

    partitions = list_table_partitions(source_path, table_name)
    if partitions:
        return _concat_partitions(partitions)
    raise FileNotFoundError(f"could not find table {table_name!r} in source {source_path}")


def _find_table_manifest(source: Path) -> Path | None:
    candidates = (
        source / "manifests" / "table_manifest.json",
        source / DEFAULT_ARCHIVE_PASS_DIR / "manifests" / "table_manifest.json",
        source / "table_manifest.json",
    )
    for candidate in candidates:
        if filesystem_path(candidate).exists():
            return candidate
    return None


def _read_table_manifest(manifest_path: Path, *, table_name: str) -> pd.DataFrame:
    manifest = load_table_manifest(manifest_path)
    root = Path(manifest.root)
    if not root.is_absolute():
        root = manifest_path.parents[1]
    frames = []
    for partition in manifest.tables:
        if str(partition.table_name) != str(table_name):
            continue
        path = root / "tables" / partition.relative_path
        if file_sha256(path) != partition.checksum_sha256:
            raise RuntimeError(f"checksum mismatch for table partition: {path}")
        frames.append(read_table_partition(path, storage_format=partition.storage_format))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _concat_partitions(partitions: list[Path]) -> pd.DataFrame:
    frames = [read_table_partition(path) for path in partitions]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
