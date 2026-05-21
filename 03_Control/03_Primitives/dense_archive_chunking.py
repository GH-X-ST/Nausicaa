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
# 1) Data Containers and Path Builders
# 2) Chunk Status and Repair Helpers
# 3) Schedule and Progress Helpers
# 4) Manifest Validation
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
class GenericChunkSpec:
    run_id: int
    planning_run_id: int
    result_root: Path | None
    layout_branch_id: str
    test_environment_mode: str
    chunk_index: int
    chunk_count: int
    chunk_size: int
    storage_format: str = "auto"
    compression_level: int = 1
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    simulation_stage: str = "paired_w0_w1_proof"


@dataclass(frozen=True)
class GenericChunkPaths:
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


def trial_outcome_paths(spec: GenericChunkSpec, *, run_root: Path) -> GenericChunkPaths:
    partition = partition_path(
        run_root,
        table_name="trial_outcomes",
        layout_branch_id=spec.layout_branch_id,
        test_environment_mode=spec.test_environment_mode,
        chunk_index=int(spec.chunk_index),
        storage_format=spec.storage_format,
    )
    manifest = (
        run_root
        / "chunk_manifests"
        / f"layout_branch_id={spec.layout_branch_id}"
        / f"test_environment_mode={spec.test_environment_mode}"
        / f"chunk-{int(spec.chunk_index):05d}.json"
    )
    log_path = (
        run_root
        / "chunk_logs"
        / f"layout_branch_id={spec.layout_branch_id}"
        / f"test_environment_mode={spec.test_environment_mode}"
        / f"chunk-{int(spec.chunk_index):05d}.log"
    )
    return GenericChunkPaths(
        root=run_root,
        partition_path=partition,
        manifest_json=manifest,
        log_path=log_path,
    )


def partition_path(
    root: Path,
    *,
    table_name: str,
    layout_branch_id: str,
    test_environment_mode: str,
    chunk_index: int,
    storage_format: str,
) -> Path:
    return (
        Path(root)
        / "tables"
        / table_name
        / f"layout_branch_id={layout_branch_id}"
        / f"test_environment_mode={test_environment_mode}"
        / f"archive_chunk_index={int(chunk_index):05d}"
        / f"part-00000.{table_extension(storage_format)}"
    )


def chunk_key(spec_or_row: GenericChunkSpec | dict[str, object]) -> tuple[str, str, int]:
    if isinstance(spec_or_row, GenericChunkSpec):
        return (
            str(spec_or_row.layout_branch_id),
            str(spec_or_row.test_environment_mode),
            int(spec_or_row.chunk_index),
        )
    return (
        str(spec_or_row["layout_branch_id"]),
        str(spec_or_row["test_environment_mode"]),
        int(spec_or_row["chunk_index"]),
    )


def chunk_payload(spec: GenericChunkSpec) -> dict[str, object]:
    payload = asdict(spec)
    payload["result_root"] = None if spec.result_root is None else str(spec.result_root)
    return payload


# =============================================================================
# 2) Chunk Status and Repair Helpers
# =============================================================================
def chunk_status(
    spec: GenericChunkSpec,
    *,
    run_root: Path,
) -> str:
    paths = trial_outcome_paths(spec, run_root=run_root)
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


def remove_chunk_outputs(spec: GenericChunkSpec, *, run_root: Path) -> None:
    paths = trial_outcome_paths(spec, run_root=run_root)
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
def assert_unique_output_paths(schedule: list[GenericChunkSpec], *, run_root: Path) -> None:
    paths = [
        trial_outcome_paths(chunk, run_root=run_root).partition_path.resolve()
        for chunk in schedule
    ]
    if len(paths) != len(set(paths)):
        raise RuntimeError("chunk schedule contains duplicate output partition paths.")


def progress_chunk_records(
    *,
    schedule: list[GenericChunkSpec],
    run_root: Path,
    pending: list[GenericChunkSpec],
    completed: list[dict[str, object]],
    skipped: list[GenericChunkSpec],
    failed: list[dict[str, object]],
    corrupt: list[GenericChunkSpec],
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
    planning_run_id: int,
    simulation_stage: str,
    worker_decision: WorkerCountDecision,
    storage_format: str,
    latency_case: str,
    resume: bool,
    repair_incomplete: bool,
    continue_on_chunk_failure: bool,
    chunks: list[dict[str, object]],
    recommended_command: str,
    branch_decision_scope: str,
    failures: list[dict[str, object]],
    profiling_rows_per_second: dict[str, float] | None = None,
) -> dict[str, object]:
    counts = chunk_status_counts(chunks)
    payload: dict[str, object] = {
        "status": str(status),
        "runtime_core_version": RUNTIME_CORE_VERSION,
        "storage_contract_version": STORAGE_CONTRACT_VERSION,
        "run_id": int(run_id),
        "planning_run_id": int(planning_run_id),
        "simulation_stage": str(simulation_stage),
        "environment_mode": "multiple",
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
        "branch_decision_scope": str(branch_decision_scope),
        "recommended_command": str(recommended_command),
        "failures": failures,
        "chunks": chunks,
    }
    if profiling_rows_per_second is not None:
        payload["rows_per_second_by_worker_count"] = dict(profiling_rows_per_second)
    return payload


def _chunk_record(
    spec: GenericChunkSpec,
    *,
    run_root: Path,
    status: str,
    path_text,
    failure: dict[str, object] | None = None,
) -> dict[str, object]:
    paths = trial_outcome_paths(spec, run_root=run_root)
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
        "layout_branch_id": str(spec.layout_branch_id),
        "test_environment_mode": str(spec.test_environment_mode),
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
            f"# {payload['simulation_stage']} Chunked Progress",
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
def validate_chunk_manifest(spec: GenericChunkSpec, *, run_root: Path) -> dict[str, object]:
    paths = trial_outcome_paths(spec, run_root=run_root)
    if (
        not filesystem_path(paths.manifest_json).exists()
        or not filesystem_path(paths.partition_path).exists()
    ):
        raise FileNotFoundError("missing chunk manifest or partition")
    manifest = json.loads(filesystem_path(paths.manifest_json).read_text(encoding="ascii"))
    expected = {
        "run_id": int(spec.run_id),
        "planning_run_id": int(spec.planning_run_id),
        "layout_branch_id": str(spec.layout_branch_id),
        "test_environment_mode": str(spec.test_environment_mode),
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
