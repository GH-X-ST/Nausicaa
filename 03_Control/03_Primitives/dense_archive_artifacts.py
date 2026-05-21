from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from dense_archive_runtime import GOVERNOR_PACKAGE_SCHEMA_VERSION
from dense_archive_table_io import (
    TablePartition,
    file_sha256,
    table_extension,
    write_table_partition,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Artifact Guardrails
# 2) Upload Package Helpers
# 3) Governor Package Helpers
# 4) Diagnostic Slice Helpers
# =============================================================================


# =============================================================================
# 1) Artifact Guardrails
# =============================================================================
UPLOAD_PACKAGE_MAX_BYTES = 25 * 1024 * 1024


def ensure_no_raw_tables(package_root: Path) -> None:
    offenders = [
        path
        for path in Path(package_root).rglob("*")
        if path.is_file() and "tables" in path.parts
    ]
    if offenders:
        raise RuntimeError("artifact package contains raw tables content.")


def check_package_file_sizes(
    package_root: Path,
    *,
    max_bytes: int = UPLOAD_PACKAGE_MAX_BYTES,
) -> None:
    oversized = [
        path
        for path in Path(package_root).rglob("*")
        if path.is_file() and path.stat().st_size > int(max_bytes)
    ]
    if oversized:
        names = ", ".join(path.as_posix() for path in oversized)
        raise RuntimeError(f"artifact package file exceeds size limit: {names}")


def reset_package_dir(path: Path) -> Path:
    package = Path(path)
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    return package


# =============================================================================
# 2) Upload Package Helpers
# =============================================================================
def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )


def finalize_upload_package(package_root: Path) -> None:
    ensure_no_raw_tables(package_root)
    check_package_file_sizes(package_root)


# =============================================================================
# 3) Governor Package Helpers
# =============================================================================
def write_governor_branch_package(
    *,
    root: Path,
    fan_layout: str,
    environment_mode: str,
    envelope_cells: pd.DataFrame,
    candidate_representatives: pd.DataFrame,
    viability_thresholds: pd.DataFrame,
    latency_metadata: pd.DataFrame,
    model_ids: pd.DataFrame,
    worker_profile_metadata: dict[str, object],
    storage_format: str,
    compression_level: int = 1,
) -> dict[str, object]:
    fan_root = Path(root) / str(fan_layout)
    fan_root.mkdir(parents=True, exist_ok=True)
    extension = table_extension(storage_format)
    partitions: list[TablePartition] = []
    partitions.append(
        write_table_partition(
            envelope_cells,
            fan_root / f"{environment_mode}_envelopes.{extension}",
            storage_format=storage_format,
            compression_level=int(compression_level),
        )
    )
    partitions.append(
        write_table_partition(
            candidate_representatives,
            fan_root / f"{environment_mode}_candidates.{extension}",
            storage_format=storage_format,
            compression_level=int(compression_level),
        )
    )
    viability_path = fan_root / f"{environment_mode}_viability_thresholds.csv"
    latency_path = fan_root / f"{environment_mode}_latency_metadata.csv"
    model_path = fan_root / f"{environment_mode}_model_ids.csv"
    viability_thresholds.to_csv(viability_path, index=False)
    latency_metadata.to_csv(latency_path, index=False)
    model_ids.to_csv(model_path, index=False)
    metadata = {
        "governor_package_schema_version": GOVERNOR_PACKAGE_SCHEMA_VERSION,
        "fan_layout": str(fan_layout),
        "environment_mode": str(environment_mode),
        "raw_tables_included": False,
        "worker_profile_metadata": worker_profile_metadata,
        "files": {
            "envelope_cells": partitions[0].relative_path,
            "candidate_representatives": partitions[1].relative_path,
            "viability_thresholds": viability_path.name,
            "latency_metadata": latency_path.name,
            "model_ids": model_path.name,
        },
        "checksums": {
            "envelope_cells": partitions[0].checksum_sha256,
            "candidate_representatives": partitions[1].checksum_sha256,
            "viability_thresholds": file_sha256(viability_path),
            "latency_metadata": file_sha256(latency_path),
            "model_ids": file_sha256(model_path),
        },
    }
    write_json(fan_root / f"{environment_mode}_governor_metadata.json", metadata)
    ensure_no_raw_tables(fan_root)
    return metadata


# =============================================================================
# 4) Diagnostic Slice Helpers
# =============================================================================
def export_diagnostic_slice(
    descriptors: pd.DataFrame,
    *,
    output_path: Path,
    layout_branch_id: str | None = None,
    test_environment_mode: str | None = None,
    failure_label: str | None = None,
    max_rows: int = 5000,
) -> Path:
    frame = descriptors.copy()
    if layout_branch_id:
        frame = frame[frame["layout_branch_id"].astype(str).eq(str(layout_branch_id))]
    if test_environment_mode:
        frame = frame[
            frame["test_environment_mode"].astype(str).eq(str(test_environment_mode))
        ]
    if failure_label:
        frame = frame[frame["failure_label"].astype(str).eq(str(failure_label))]
    frame = frame.head(int(max_rows))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
