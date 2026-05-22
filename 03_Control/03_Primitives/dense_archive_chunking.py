from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dense_archive_runtime import (
    RUNTIME_CORE_VERSION,
    STORAGE_CONTRACT_VERSION,
    WorkerCountDecision,
)
from dense_archive_table_io import (
    ensure_directory,
    file_sha256,
    filesystem_path,
    partition_row_count,
    resolve_storage_format,
    table_extension,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data containers and path builders
# 2) Chunk status and repair helpers
# 3) Schedule and progress helpers
# 4) Manifest validation
# =============================================================================


# =============================================================================
# 1) Data Containers and Path Builders
# =============================================================================
TIMING_FIELDS = (
    "planning_read_s",
    "selection_s",
    "simulation_s",
    "descriptor_build_s",
    "write_s",
    "total_s",
)


@dataclass(frozen=True)
class ContextChunkSpec:
    run_id: int
    source_run_id: int
    result_root: Path | None
    context_id: str
    environment_id: str
    chunk_index: int
    chunk_count: int
    chunk_size: int
    storage_format: str = "auto"
    compression_level: int = 1
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    run_stage: str = "contextual_foundation"


@dataclass(frozen=True)
class ContextChunkPaths:
    root: Path
    partition_path: Path
    manifest_json: Path
    log_path: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "partition_path": self.partition_path,
            "manifest_json": self.manifest_json,
            "log_path": self.log_path,
        }


# Backward-compatible names for retained import paths.
GenericChunkSpec = ContextChunkSpec
GenericChunkPaths = ContextChunkPaths


def contextual_table_paths(spec: ContextChunkSpec, *, run_root: Path) -> ContextChunkPaths:
    partition = partition_path(
        run_root,
        table_name="contextual_rows",
        context_id=spec.context_id,
        environment_id=spec.environment_id,
        chunk_index=int(spec.chunk_index),
        storage_format=spec.storage_format,
    )
    manifest = (
        run_root
        / "chunk_manifests"
        / f"context_id={spec.context_id}"
        / f"environment_id={spec.environment_id}"
        / f"chunk-{int(spec.chunk_index):05d}.json"
    )
    log_path = (
        run_root
        / "chunk_logs"
        / f"context_id={spec.context_id}"
        / f"environment_id={spec.environment_id}"
        / f"chunk-{int(spec.chunk_index):05d}.log"
    )
    return ContextChunkPaths(
        root=run_root,
        partition_path=partition,
        manifest_json=manifest,
        log_path=log_path,
    )


def trial_outcome_paths(spec: ContextChunkSpec, *, run_root: Path) -> ContextChunkPaths:
    """Backward-compatible wrapper for retained tests."""

    return contextual_table_paths(spec, run_root=run_root)


def partition_path(
    root: Path,
    *,
    table_name: str,
    context_id: str,
    environment_id: str,
    chunk_index: int,
    storage_format: str,
) -> Path:
    return (
        Path(root)
        / "tables"
        / table_name
        / f"context_id={context_id}"
        / f"environment_id={environment_id}"
        / f"chunk_index={int(chunk_index):05d}"
        / f"part-00000.{table_extension(storage_format)}"
    )


def chunk_key(spec_or_row: ContextChunkSpec | dict[str, object]) -> tuple[str, str, int]:
    if isinstance(spec_or_row, ContextChunkSpec):
        return (
            str(spec_or_row.context_id),
            str(spec_or_row.environment_id),
            int(spec_or_row.chunk_index),
        )
    return (
        str(spec_or_row["context_id"]),
        str(spec_or_row["environment_id"]),
        int(spec_or_row["chunk_index"]),
    )


def chunk_payload(spec: ContextChunkSpec) -> dict[str, object]:
    payload = asdict(spec)
    payload["result_root"] = None if spec.result_root is None else str(spec.result_root)
    return payload


# =============================================================================
# 2) Chunk Status and Repair Helpers
# =============================================================================
def chunk_status(
    spec: ContextChunkSpec,
    *,
    run_root: Path,
) -> str:
    paths = contextual_table_paths(spec, run_root=run_root)
    manifest_exists = filesystem_path(paths.manifest_json).exists()
    partition_exists = filesystem_path(paths.partition_path).exists()
    if not manifest_exists and not partition_exists:
        return "missing"
    if manifest_exists != partition_exists:
        return "corrupt"
    try:
        validate_chunk_manifest(spec, run_root=run_root)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError, OSError):
        return "corrupt"
    return "complete"


def remove_chunk_outputs(spec: ContextChunkSpec, *, run_root: Path) -> None:
    paths = contextual_table_paths(spec, run_root=run_root)
    for path in (
        paths.partition_path,
        paths.partition_path.with_name(f"{paths.partition_path.name}.tmp"),
        paths.manifest_json,
        paths.manifest_json.with_name(f"{paths.manifest_json.name}.tmp"),
        paths.log_path,
        paths.log_path.with_name(f"{paths.log_path.name}.tmp"),
    ):
        fs_path = filesystem_path(path)
        if fs_path.exists():
            fs_path.unlink()


# =============================================================================
# 3) Schedule and Progress Helpers
# =============================================================================
def assert_unique_output_paths(schedule: list[ContextChunkSpec], *, run_root: Path) -> None:
    paths = [
        contextual_table_paths(chunk, run_root=run_root).partition_path.resolve()
        for chunk in schedule
    ]
    if len(paths) != len(set(paths)):
        raise RuntimeError("chunk schedule contains duplicate output partition paths.")


def progress_chunk_records(
    *,
    schedule: list[ContextChunkSpec],
    run_root: Path,
    pending: list[ContextChunkSpec],
    completed: list[dict[str, object]],
    skipped: list[ContextChunkSpec],
    failed: list[dict[str, object]],
    corrupt: list[ContextChunkSpec],
    path_text,
) -> list[dict[str, object]]:
    completed_keys = {chunk_key(row) for row in completed}
    skipped_keys = {chunk_key(chunk) for chunk in skipped}
    pending_keys = {chunk_key(chunk) for chunk in pending}
    corrupt_keys = {chunk_key(chunk) for chunk in corrupt}
    failed_by_key = {chunk_key(row): row for row in failed}
    records: list[dict[str, object]] = []
    for chunk in schedule:
        key = chunk_key(chunk)
        if key in failed_by_key:
            records.append(
                _chunk_record(
                    chunk,
                    run_root=run_root,
                    status="failed",
                    failure=failed_by_key[key],
                    path_text=path_text,
                )
            )
        elif key in completed_keys or key in skipped_keys:
            records.append(
                _chunk_record(chunk, run_root=run_root, status="complete", path_text=path_text)
            )
        elif key in pending_keys:
            records.append(
                _chunk_record(chunk, run_root=run_root, status="pending", path_text=path_text)
            )
        elif key in corrupt_keys:
            records.append(
                _chunk_record(chunk, run_root=run_root, status="corrupt", path_text=path_text)
            )
        else:
            records.append(
                _chunk_record(chunk, run_root=run_root, status="pending", path_text=path_text)
            )
    return records


def chunk_status_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_progress_manifest(
    *,
    path: Path,
    report_path: Path,
    payload: dict[str, object],
) -> None:
    ensure_directory(path.parent)
    ensure_directory(report_path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    filesystem_path(tmp_path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )
    filesystem_path(tmp_path).replace(filesystem_path(path))
    filesystem_path(report_path).write_text(_progress_report(payload), encoding="ascii")


def progress_manifest_payload(
    *,
    status: str,
    run_id: int,
    source_run_id: int | None = None,
    planning_run_id: int | None = None,
    run_stage: str | None = None,
    simulation_stage: str | None = None,
    worker_decision: WorkerCountDecision,
    storage_format: str,
    latency_case: str,
    resume: bool,
    repair_incomplete: bool,
    continue_on_chunk_failure: bool,
    chunks: list[dict[str, object]],
    recommended_command: str,
    context_decision_scope: str | None = None,
    failures: list[dict[str, object]],
    profiling_rows_per_second: dict[str, float] | None = None,
) -> dict[str, object]:
    counts = chunk_status_counts(chunks)
    source = int(source_run_id if source_run_id is not None else planning_run_id or 0)
    stage = str(run_stage if run_stage is not None else simulation_stage or "contextual_foundation")
    payload: dict[str, object] = {
        "status": str(status),
        "runtime_core_version": RUNTIME_CORE_VERSION,
        "storage_contract_version": STORAGE_CONTRACT_VERSION,
        "run_id": int(run_id),
        "source_run_id": source,
        "run_stage": stage,
        "scheduled_chunk_count": int(len(chunks)),
        "pending_chunk_count": int(counts.get("pending", 0)),
        "completed_chunk_count": int(counts.get("complete", 0)),
        "failed_chunk_count": int(counts.get("failed", 0)),
        "corrupt_chunk_count": int(counts.get("corrupt", 0)),
        "selected_worker_count": int(worker_decision.selected_worker_count),
        "max_workers": worker_decision.max_workers,
        "os_cpu_count": worker_decision.os_cpu_count,
        "memory_total_gb": worker_decision.memory_total_gb,
        "memory_safety_margin_gb": worker_decision.memory_safety_margin_gb,
        "estimated_worker_memory_gb": worker_decision.estimated_worker_memory_gb,
        "worker_fallback_reason": worker_decision.fallback_reason,
        "storage_format": resolve_storage_format(storage_format),
        "latency_case": str(latency_case),
        "resume": bool(resume),
        "repair_incomplete": bool(repair_incomplete),
        "continue_on_chunk_failure": bool(continue_on_chunk_failure),
        "context_decision_scope": str(context_decision_scope or "context_features_only"),
        "recommended_command": str(recommended_command),
        "failures": failures,
        "chunks": chunks,
    }
    if profiling_rows_per_second is not None:
        payload["rows_per_second_by_worker_count"] = dict(profiling_rows_per_second)
    return payload


def _chunk_record(
    spec: ContextChunkSpec,
    *,
    run_root: Path,
    status: str,
    path_text,
    failure: dict[str, object] | None = None,
) -> dict[str, object]:
    paths = contextual_table_paths(spec, run_root=run_root)
    row_count: int | None = None
    checksum = ""
    if status == "complete" and filesystem_path(paths.manifest_json).exists():
        try:
            payload = json.loads(filesystem_path(paths.manifest_json).read_text(encoding="ascii"))
            row_count = int(payload.get("row_count", 0))
            checksum = str(payload.get("checksum_sha256", ""))
        except (OSError, ValueError, json.JSONDecodeError):
            status = "corrupt"
    return {
        "context_id": str(spec.context_id),
        "environment_id": str(spec.environment_id),
        "chunk_index": int(spec.chunk_index),
        "chunk_count": int(spec.chunk_count),
        "chunk_size": int(spec.chunk_size),
        "status": str(status),
        "partition_path": path_text(paths.partition_path),
        "manifest_path": path_text(paths.manifest_json),
        "row_count": row_count,
        "checksum_sha256": checksum,
        "error_type": "" if failure is None else str(failure.get("error_type", "")),
        "error": "" if failure is None else str(failure.get("error", "")),
    }


def _progress_report(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# {payload['run_stage']} Chunked Progress",
            "",
            f"- Status: `{payload['status']}`",
            f"- Selected worker count: `{payload['selected_worker_count']}`",
            f"- Worker fallback reason: `{payload['worker_fallback_reason']}`",
            f"- Scheduled chunks: `{payload['scheduled_chunk_count']}`",
            f"- Completed chunks: `{payload['completed_chunk_count']}`",
            f"- Failed chunks: `{payload['failed_chunk_count']}`",
            "",
            "Recommended command:",
            "",
            f"```powershell\n{payload['recommended_command']}\n```",
            "",
        ]
    )


# =============================================================================
# 4) Manifest Validation
# =============================================================================
def validate_chunk_manifest(spec: ContextChunkSpec, *, run_root: Path) -> dict[str, object]:
    paths = contextual_table_paths(spec, run_root=run_root)
    if (
        not filesystem_path(paths.manifest_json).exists()
        or not filesystem_path(paths.partition_path).exists()
    ):
        raise FileNotFoundError("missing chunk manifest or partition")
    manifest = json.loads(filesystem_path(paths.manifest_json).read_text(encoding="ascii"))
    expected = {
        "run_id": int(spec.run_id),
        "source_run_id": int(spec.source_run_id),
        "context_id": str(spec.context_id),
        "environment_id": str(spec.environment_id),
        "chunk_index": int(spec.chunk_index),
        "chunk_count": int(spec.chunk_count),
        "chunk_size": int(spec.chunk_size),
        "storage_format": resolve_storage_format(spec.storage_format),
        "latency_case": str(spec.latency_case),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"chunk manifest mismatch for {key}")
    for key, value in (("dt_s", spec.dt_s), ("horizon_s", spec.horizon_s)):
        if not math.isclose(float(manifest.get(key)), float(value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"chunk manifest mismatch for {key}")
    if manifest.get("status") != "complete":
        raise ValueError("chunk manifest is not complete")
    if int(manifest["row_count"]) != partition_row_count(paths.partition_path):
        raise ValueError("chunk row count mismatch")
    if str(manifest["checksum_sha256"]) != file_sha256(paths.partition_path):
        raise ValueError("chunk checksum mismatch")
    for field in TIMING_FIELDS:
        value = float(manifest[field])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"chunk timing field is not finite: {field}")
    return manifest
