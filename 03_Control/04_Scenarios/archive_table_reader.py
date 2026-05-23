from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dense_archive_table_io import (
    filesystem_path,
    load_table_manifest,
    read_table_partition,
)


@dataclass(frozen=True)
class ArchiveTableSourceInfo:
    source_path: str
    source_kind: str
    manifest_path: str
    run_manifest_path: str
    row_count_loaded: int
    row_count_manifested: int
    storage_format: str
    run_stage: str
    claim_status: str
    rollout_backend: str
    rows_requested: int
    evidence_eligible: bool


def read_archive_table(source: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read archive rows from a manifest, run root, or table partition."""

    frame, _ = read_archive_table_with_info(source, max_rows=max_rows)
    return frame


def read_archive_table_with_info(
    source: Path,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, ArchiveTableSourceInfo]:
    """Read archive rows and return compact source metadata."""

    path = Path(source)
    fs_path = filesystem_path(path)
    if fs_path.is_dir():
        manifest = path / "manifests" / "table_manifest.json"
        if filesystem_path(manifest).is_file():
            frame = _read_manifest(manifest, max_rows=max_rows)
            return frame, _source_info(
                source=path,
                source_kind="run_root",
                manifest_path=manifest,
                frame=frame,
            )
        partitions = sorted(
            item for item in fs_path.rglob("*") if item.is_file() and _is_partition_name(item.name)
        )
        frame = _concat_partitions(partitions, max_rows=max_rows)
        return frame, _source_info(
            source=path,
            source_kind="partition_directory",
            manifest_path=None,
            frame=frame,
        )
    if path.name == "table_manifest.json":
        frame = _read_manifest(path, max_rows=max_rows)
        return frame, _source_info(
            source=path,
            source_kind="table_manifest",
            manifest_path=path,
            frame=frame,
        )
    if _is_partition_name(path.name):
        frame = read_table_partition(path)
        frame = frame.head(max_rows) if max_rows is not None else frame
        return frame, _source_info(
            source=path,
            source_kind="single_partition",
            manifest_path=None,
            frame=frame,
        )
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


def _source_info(
    *,
    source: Path,
    source_kind: str,
    manifest_path: Path | None,
    frame: pd.DataFrame,
) -> ArchiveTableSourceInfo:
    run_manifest_path = _run_manifest_path(manifest_path)
    run_manifest = _read_json(run_manifest_path)
    table_manifest = load_table_manifest(manifest_path) if manifest_path is not None else None
    row_count_manifested = (
        sum(int(partition.row_count) for partition in table_manifest.tables)
        if table_manifest is not None
        else int(len(frame))
    )
    storage_format = (
        str(table_manifest.storage_format)
        if table_manifest is not None
        else _storage_format_from_frame_source(source)
    )
    rows_requested = int(run_manifest.get("rows_requested", row_count_manifested))
    claim_status = str(run_manifest.get("claim_status", "unknown"))
    rollout_backend = str(run_manifest.get("rollout_backend", "unknown"))
    run_stage = str(run_manifest.get("run_stage", "unknown"))
    evidence_eligible = bool(
        manifest_path is not None
        and rows_requested >= 40_000
        and rollout_backend == "model_backed_lqr"
        and not bool(run_manifest.get("dry_run_schedule", False))
    )
    return ArchiveTableSourceInfo(
        source_path=Path(source).as_posix(),
        source_kind=str(source_kind),
        manifest_path="" if manifest_path is None else Path(manifest_path).as_posix(),
        run_manifest_path="" if run_manifest_path is None else Path(run_manifest_path).as_posix(),
        row_count_loaded=int(len(frame)),
        row_count_manifested=int(row_count_manifested),
        storage_format=storage_format,
        run_stage=run_stage,
        claim_status=claim_status,
        rollout_backend=rollout_backend,
        rows_requested=rows_requested,
        evidence_eligible=evidence_eligible,
    )


def _run_manifest_path(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None
    candidate = Path(manifest_path).with_name("run_manifest.json")
    return candidate if filesystem_path(candidate).is_file() else None


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None or not filesystem_path(path).is_file():
        return {}
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _storage_format_from_frame_source(source: Path) -> str:
    lower = Path(source).name.lower()
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".csv.gz"):
        return "csv_gz"
    if lower.endswith(".csv"):
        return "csv"
    return "unknown"
