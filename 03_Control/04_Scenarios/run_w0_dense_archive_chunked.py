from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import resolve_storage_format  # noqa: E402
from run_w0_dense_archive_chunk import (  # noqa: E402
    W0ChunkConfig,
    active_result_root,
    chunk_status,
    output_paths,
    remove_chunk_outputs,
    run_w0_dense_archive_chunk,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Configuration and Worker Policy
# 2) Chunk Scheduling and Progress Manifests
# 3) Parallel Execution
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Configuration and Worker Policy
# =============================================================================
W0_BRANCH_IDS = ("single_fan_branch", "four_fan_branch")
GPU_ACCELERATION_ASSESSMENT = (
    "GPU acceleration is deferred in this pass because it would require a "
    "batched dynamics refactor, new dependencies, and numerical-equivalence "
    "validation. CPU chunk parallelism preserves RK4, state_derivative, plant "
    "dynamics, latency constants, command semantics, and acceptance metrics."
)
PRODUCTION_COMMAND = (
    "python 03_Control/04_Scenarios/run_w0_dense_archive_chunked.py "
    "--run-id 13 --planning-run-id 12 --target-trials-total 500000 "
    "--target-trials-per-branch 250000 --chunk-size 2500 --workers 8 "
    "--max-workers 8 --latency-case nominal --storage-format auto "
    "--compression-level 1 --resume"
)


@dataclass(frozen=True)
class WorkerCountDecision:
    requested: str | int
    selected_worker_count: int
    max_workers: int | None
    os_cpu_count: int | None
    memory_total_gb: float | None
    memory_safety_margin_gb: float
    estimated_worker_memory_gb: float | None
    fallback_reason: str


@dataclass(frozen=True)
class W0ChunkedRunConfig:
    run_id: int = 13
    planning_run_id: int = 12
    target_trials_total: int = 500000
    target_trials_per_branch: int = 250000
    chunk_size: int = 2500
    workers: str | int = "auto"
    max_workers: int | None = 8
    memory_safety_margin_gb: float = 8.0
    storage_format: str = "auto"
    compression_level: int = 1
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    resume: bool = True
    repair_incomplete: bool = False
    stop_after_chunks: int | None = None
    result_root: Path | None = None
    profile_first: bool = False
    continue_on_chunk_failure: bool = False
    random_seed: int = 20260525


def resolve_worker_count(
    requested: str | int,
    *,
    logical_cpu_count: int | None = None,
    memory_total_gb: float | None = None,
    memory_safety_margin_gb: float = 8.0,
    estimated_worker_memory_gb: float | None = None,
    max_workers: int | None = 8,
) -> int:
    """Return the selected worker count for the local W0 chunked run."""

    return worker_count_decision(
        requested,
        logical_cpu_count=logical_cpu_count,
        memory_total_gb=memory_total_gb,
        memory_safety_margin_gb=memory_safety_margin_gb,
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_workers=max_workers,
    ).selected_worker_count


def worker_count_decision(
    requested: str | int,
    *,
    logical_cpu_count: int | None = None,
    memory_total_gb: float | None = None,
    memory_safety_margin_gb: float = 8.0,
    estimated_worker_memory_gb: float | None = None,
    max_workers: int | None = 8,
) -> WorkerCountDecision:
    """Return the selected worker count and recorded fallback reason."""

    cpu_count = os.cpu_count() if logical_cpu_count is None else logical_cpu_count
    memory_gb = _memory_total_gb() if memory_total_gb is None else memory_total_gb
    fallback_reasons: list[str] = []
    requested_value = requested
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        if (cpu_count is None or int(cpu_count) >= 12) and (
            memory_gb is None or float(memory_gb) >= 31.0
        ):
            candidate = 8
        else:
            candidate = 6
    else:
        try:
            candidate = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError("workers must be 'auto' or a positive integer.") from exc
        if candidate < 1:
            raise ValueError("workers must be at least 1.")

    if max_workers is not None and candidate > int(max_workers):
        fallback_reasons.append(f"capped_by_max_workers_{int(max_workers)}")
        candidate = int(max_workers)
    if cpu_count is not None:
        cpu_cap = max(1, int(cpu_count) - 2)
        if candidate > cpu_cap:
            fallback_reasons.append(f"capped_by_cpu_count_minus_2_{cpu_cap}")
            candidate = cpu_cap
    if (
        estimated_worker_memory_gb is not None
        and memory_gb is not None
        and float(estimated_worker_memory_gb) > 0.0
    ):
        while (
            candidate > 1
            and candidate * float(estimated_worker_memory_gb)
            + float(memory_safety_margin_gb)
            > float(memory_gb)
        ):
            candidate -= 1
        if candidate < (8 if str(requested).lower() == "auto" else int(requested)):
            fallback_reasons.append("reduced_by_memory_guardrail")

    return WorkerCountDecision(
        requested=requested_value,
        selected_worker_count=max(1, int(candidate)),
        max_workers=None if max_workers is None else int(max_workers),
        os_cpu_count=None if cpu_count is None else int(cpu_count),
        memory_total_gb=None if memory_gb is None else float(memory_gb),
        memory_safety_margin_gb=float(memory_safety_margin_gb),
        estimated_worker_memory_gb=(
            None
            if estimated_worker_memory_gb is None
            else float(estimated_worker_memory_gb)
        ),
        fallback_reason="none" if not fallback_reasons else ";".join(fallback_reasons),
    )


def _memory_total_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.ullTotalPhys) / float(1024**3)


# =============================================================================
# 2) Chunk Scheduling and Progress Manifests
# =============================================================================
def _validate_config(config: W0ChunkedRunConfig) -> None:
    if int(config.target_trials_total) != 2 * int(config.target_trials_per_branch):
        raise ValueError("target_trials_total must equal two target_trials_per_branch.")
    if int(config.target_trials_per_branch) <= 0:
        raise ValueError("target_trials_per_branch must be positive.")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if int(config.compression_level) < 0:
        raise ValueError("compression_level must be nonnegative.")
    if config.stop_after_chunks is not None and int(config.stop_after_chunks) < 0:
        raise ValueError("stop_after_chunks must be nonnegative when supplied.")
    resolve_storage_format(config.storage_format)


def _chunk_count_per_branch(config: W0ChunkedRunConfig) -> int:
    return int(math.ceil(float(config.target_trials_per_branch) / float(config.chunk_size)))


def build_chunk_schedule(config: W0ChunkedRunConfig) -> list[W0ChunkConfig]:
    """Return the deterministic branch/chunk schedule for a W0 archive run."""

    chunk_count = _chunk_count_per_branch(config)
    schedule: list[W0ChunkConfig] = []
    for branch_id in W0_BRANCH_IDS:
        for chunk_index in range(chunk_count):
            schedule.append(
                W0ChunkConfig(
                    run_id=int(config.run_id),
                    planning_run_id=int(config.planning_run_id),
                    result_root=config.result_root,
                    layout_branch_id=branch_id,
                    chunk_index=int(chunk_index),
                    chunk_count=int(chunk_count),
                    chunk_size=int(config.chunk_size),
                    latency_case=str(config.latency_case),
                    dt_s=float(config.dt_s),
                    horizon_s=float(config.horizon_s),
                    storage_format=str(config.storage_format),
                    resume=bool(config.resume),
                    overwrite_chunk=False,
                    random_seed=int(config.random_seed),
                )
            )
    _assert_unique_output_paths(schedule)
    return schedule


def _assert_unique_output_paths(schedule: list[W0ChunkConfig]) -> None:
    paths = [output_paths(chunk).partition_path.resolve() for chunk in schedule]
    if len(paths) != len(set(paths)):
        raise RuntimeError("chunk schedule contains duplicate output partition paths.")


def _progress_paths(config: W0ChunkedRunConfig) -> tuple[Path, Path]:
    root = active_result_root(
        W0ChunkConfig(run_id=config.run_id, planning_run_id=config.planning_run_id, result_root=config.result_root)
    ) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    return (
        root / "manifests" / f"w0_dense_archive_progress_{suffix}.json",
        root / "reports" / f"w0_dense_archive_progress_{suffix}.md",
    )


def _write_progress(
    *,
    config: W0ChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    schedule: list[W0ChunkConfig],
    pending: list[W0ChunkConfig],
    completed: list[dict[str, object]],
    skipped: list[W0ChunkConfig],
    failed: list[dict[str, object]],
    corrupt: list[W0ChunkConfig],
    dry_run: bool = False,
) -> None:
    manifest_path, report_path = _progress_paths(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "dry_run"
        if dry_run
        else ("failed" if failed else ("complete" if not pending else "in_progress")),
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "target_trials_total": int(config.target_trials_total),
        "target_trials_per_branch": int(config.target_trials_per_branch),
        "chunk_size": int(config.chunk_size),
        "chunk_count_per_branch": _chunk_count_per_branch(config),
        "scheduled_chunk_count": int(len(schedule)),
        "pending_chunk_count": int(len(pending)),
        "completed_chunk_count": int(len(completed)),
        "skipped_complete_chunk_count": int(len(skipped)),
        "failed_chunk_count": int(len(failed)),
        "corrupt_chunk_count": int(len(corrupt)),
        "workers_requested": str(config.workers),
        "selected_worker_count": int(worker_decision.selected_worker_count),
        "max_workers": worker_decision.max_workers,
        "os_cpu_count": worker_decision.os_cpu_count,
        "memory_total_gb": worker_decision.memory_total_gb,
        "memory_safety_margin_gb": worker_decision.memory_safety_margin_gb,
        "estimated_worker_memory_gb": worker_decision.estimated_worker_memory_gb,
        "worker_fallback_reason": worker_decision.fallback_reason,
        "storage_format": resolve_storage_format(config.storage_format),
        "latency_case": str(config.latency_case),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "continue_on_chunk_failure": bool(config.continue_on_chunk_failure),
        "gpu_acceleration_assessment": GPU_ACCELERATION_ASSESSMENT,
        "recommended_production_command": PRODUCTION_COMMAND,
        "branch_decision_scope": "branch_local_only_no_cross_layout_decision_transfer",
        "failures": failed,
    }
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")
    tmp_path.replace(manifest_path)
    report_path.write_text(_progress_report(payload), encoding="ascii")


def _progress_report(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# W0 Dense Archive Chunked Progress",
            "",
            f"- Status: `{payload['status']}`",
            f"- Selected worker count: `{payload['selected_worker_count']}`",
            f"- Worker fallback reason: `{payload['worker_fallback_reason']}`",
            f"- Scheduled chunks: `{payload['scheduled_chunk_count']}`",
            f"- Completed chunks: `{payload['completed_chunk_count']}`",
            f"- Failed chunks: `{payload['failed_chunk_count']}`",
            f"- GPU assessment: {payload['gpu_acceleration_assessment']}",
            "",
            "Recommended production command:",
            "",
            f"```powershell\n{payload['recommended_production_command']}\n```",
            "",
        ]
    )


# =============================================================================
# 3) Parallel Execution
# =============================================================================
def _worker_run_chunk(payload: dict[str, object]) -> dict[str, object]:
    result_root = payload.get("result_root")
    paths = run_w0_dense_archive_chunk(
        run_id=int(payload["run_id"]),
        planning_run_id=int(payload["planning_run_id"]),
        result_root=None if result_root in {None, ""} else Path(str(result_root)),
        layout_branch_id=str(payload["layout_branch_id"]),
        chunk_index=int(payload["chunk_index"]),
        chunk_count=int(payload["chunk_count"]),
        chunk_size=int(payload["chunk_size"]),
        latency_case=str(payload["latency_case"]),
        dt_s=float(payload["dt_s"]),
        horizon_s=float(payload["horizon_s"]),
        storage_format=str(payload["storage_format"]),
        resume=bool(payload["resume"]),
        overwrite_chunk=bool(payload["overwrite_chunk"]),
        random_seed=int(payload["random_seed"]),
    )
    return {
        "layout_branch_id": str(payload["layout_branch_id"]),
        "chunk_index": int(payload["chunk_index"]),
        "status": "complete",
        "manifest_json": str(paths["manifest_json"]),
        "partition_path": str(paths["partition_path"]),
    }


def _chunk_payload(chunk: W0ChunkConfig) -> dict[str, object]:
    payload = asdict(chunk)
    payload["result_root"] = None if chunk.result_root is None else str(chunk.result_root)
    return payload


def _prepare_pending_chunks(
    config: W0ChunkedRunConfig,
    schedule: list[W0ChunkConfig],
) -> tuple[list[W0ChunkConfig], list[W0ChunkConfig], list[W0ChunkConfig]]:
    pending: list[W0ChunkConfig] = []
    skipped: list[W0ChunkConfig] = []
    corrupt: list[W0ChunkConfig] = []
    for chunk in schedule:
        status = chunk_status(chunk)
        if status == "complete" and bool(config.resume):
            skipped.append(chunk)
            continue
        if status == "corrupt":
            corrupt.append(chunk)
            if not bool(config.repair_incomplete):
                continue
            remove_chunk_outputs(chunk)
        pending.append(chunk)
    if corrupt and not bool(config.repair_incomplete):
        bad = ", ".join(f"{c.layout_branch_id}:{c.chunk_index}" for c in corrupt[:5])
        raise RuntimeError(
            "corrupt/incomplete chunks found; rerun with --repair-incomplete "
            f"to remove only those chunk outputs. Examples: {bad}"
        )
    if config.stop_after_chunks is not None:
        pending = pending[: int(config.stop_after_chunks)]
    return pending, skipped, corrupt


def _run_pending_chunks(
    *,
    config: W0ChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    schedule: list[W0ChunkConfig],
    pending: list[W0ChunkConfig],
    skipped: list[W0ChunkConfig],
    corrupt: list[W0ChunkConfig],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    _write_progress(
        config=config,
        worker_decision=worker_decision,
        schedule=schedule,
        pending=pending,
        completed=completed,
        skipped=skipped,
        failed=failed,
        corrupt=corrupt,
    )
    if not pending:
        return completed, failed

    if int(worker_decision.selected_worker_count) == 1:
        for chunk in pending:
            try:
                completed.append(_worker_run_chunk(_chunk_payload(chunk)))
            except Exception as exc:  # pragma: no cover - exercised by integration failures.
                failed.append(_failure_row(chunk, exc))
                if not bool(config.continue_on_chunk_failure):
                    _write_progress(
                        config=config,
                        worker_decision=worker_decision,
                        schedule=schedule,
                        pending=pending,
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                        corrupt=corrupt,
                    )
                    raise
            _write_progress(
                config=config,
                worker_decision=worker_decision,
                schedule=schedule,
                pending=pending,
                completed=completed,
                skipped=skipped,
                failed=failed,
                corrupt=corrupt,
            )
        return completed, failed

    with ProcessPoolExecutor(max_workers=int(worker_decision.selected_worker_count)) as executor:
        futures = {
            executor.submit(_worker_run_chunk, _chunk_payload(chunk)): chunk
            for chunk in pending
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failed.append(_failure_row(chunk, exc))
                if not bool(config.continue_on_chunk_failure):
                    _write_progress(
                        config=config,
                        worker_decision=worker_decision,
                        schedule=schedule,
                        pending=pending,
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                        corrupt=corrupt,
                    )
                    raise
            _write_progress(
                config=config,
                worker_decision=worker_decision,
                schedule=schedule,
                pending=pending,
                completed=completed,
                skipped=skipped,
                failed=failed,
                corrupt=corrupt,
            )
    return completed, failed


def _failure_row(chunk: W0ChunkConfig, exc: BaseException) -> dict[str, object]:
    return {
        "layout_branch_id": chunk.layout_branch_id,
        "chunk_index": int(chunk.chunk_index),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_w0_dense_archive_chunked(
    *,
    run_id: int = 13,
    planning_run_id: int = 12,
    target_trials_total: int = 500000,
    target_trials_per_branch: int = 250000,
    chunk_size: int = 2500,
    workers: str | int = "auto",
    max_workers: int | None = 8,
    memory_safety_margin_gb: float = 8.0,
    storage_format: str = "auto",
    compression_level: int = 1,
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    resume: bool = True,
    repair_incomplete: bool = False,
    stop_after_chunks: int | None = None,
    result_root: Path | None = None,
    profile_first: bool = False,
    continue_on_chunk_failure: bool = False,
    dry_run_schedule: bool = False,
    random_seed: int = 20260525,
) -> dict[str, Path]:
    """Run or schedule the local process-parallel W0 dense archive chunks."""

    config = W0ChunkedRunConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        target_trials_total=int(target_trials_total),
        target_trials_per_branch=int(target_trials_per_branch),
        chunk_size=int(chunk_size),
        workers=workers,
        max_workers=None if max_workers is None else int(max_workers),
        memory_safety_margin_gb=float(memory_safety_margin_gb),
        storage_format=str(storage_format),
        compression_level=int(compression_level),
        latency_case=str(latency_case),
        dt_s=float(dt_s),
        horizon_s=float(horizon_s),
        resume=bool(resume),
        repair_incomplete=bool(repair_incomplete),
        stop_after_chunks=stop_after_chunks,
        result_root=result_root,
        profile_first=bool(profile_first),
        continue_on_chunk_failure=bool(continue_on_chunk_failure),
        random_seed=int(random_seed),
    )
    _validate_config(config)
    estimated_worker_memory_gb = 2.0
    if config.profile_first:
        from profile_w0_dense_archive import profile_w0_dense_archive

        profile_paths = profile_w0_dense_archive(
            planning_run_id=int(config.planning_run_id),
            result_root=active_result_root(
                W0ChunkConfig(
                    run_id=config.run_id,
                    planning_run_id=config.planning_run_id,
                    result_root=config.result_root,
                )
            ),
            sample_trials=min(2000, int(config.target_trials_total)),
            storage_format=str(config.storage_format),
            latency_case=str(config.latency_case),
            dt_s=float(config.dt_s),
            horizon_s=float(config.horizon_s),
            workers=config.workers,
            memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        )
        profile_payload = json.loads(
            profile_paths["profile_json"].read_text(encoding="ascii")
        )
        estimated_worker_memory_gb = float(
            profile_payload.get("estimated_worker_memory_gb", estimated_worker_memory_gb)
        )
    worker_decision = worker_count_decision(
        config.workers,
        memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_workers=config.max_workers,
    )
    schedule = build_chunk_schedule(config)
    pending, skipped, corrupt = _prepare_pending_chunks(config, schedule)
    if dry_run_schedule:
        _write_progress(
            config=config,
            worker_decision=worker_decision,
            schedule=schedule,
            pending=pending,
            completed=[],
            skipped=skipped,
            failed=[],
            corrupt=corrupt,
            dry_run=True,
        )
        return _run_paths(config)
    completed, failed = _run_pending_chunks(
        config=config,
        worker_decision=worker_decision,
        schedule=schedule,
        pending=pending,
        skipped=skipped,
        corrupt=corrupt,
    )
    _write_progress(
        config=config,
        worker_decision=worker_decision,
        schedule=schedule,
        pending=[],
        completed=completed,
        skipped=skipped,
        failed=failed,
        corrupt=corrupt,
    )
    return _run_paths(config)


def _run_paths(config: W0ChunkedRunConfig) -> dict[str, Path]:
    manifest_path, report_path = _progress_paths(config)
    root = manifest_path.parents[1]
    return {
        "root": root,
        "progress_manifest_json": manifest_path,
        "progress_report_md": report_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=13)
    parser.add_argument("--planning-run-id", type=int, default=12)
    parser.add_argument("--target-trials-total", type=int, default=500000)
    parser.add_argument("--target-trials-per-branch", type=int, default=250000)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--memory-safety-margin-gb", type=float, default=8.0)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--profile-first", action="store_true")
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workers: str | int
    workers = int(args.workers) if str(args.workers).isdigit() else str(args.workers)
    paths = run_w0_dense_archive_chunked(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        target_trials_total=args.target_trials_total,
        target_trials_per_branch=args.target_trials_per_branch,
        chunk_size=args.chunk_size,
        workers=workers,
        max_workers=args.max_workers,
        memory_safety_margin_gb=args.memory_safety_margin_gb,
        storage_format=args.storage_format,
        compression_level=args.compression_level,
        latency_case=args.latency_case,
        dt_s=args.dt_s,
        horizon_s=args.horizon_s,
        resume=args.resume,
        repair_incomplete=args.repair_incomplete,
        stop_after_chunks=args.stop_after_chunks,
        result_root=args.result_root,
        profile_first=args.profile_first,
        continue_on_chunk_failure=args.continue_on_chunk_failure,
        dry_run_schedule=args.dry_run_schedule,
        random_seed=args.random_seed,
    )
    print(f"w0_dense_archive_chunked_outputs={paths['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
