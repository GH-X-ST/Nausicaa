from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_chunking import (  # noqa: E402
    assert_unique_output_paths,
    chunk_payload,
    progress_chunk_records,
    progress_manifest_payload,
    write_progress_manifest,
)
from dense_archive_runtime import (  # noqa: E402
    DEFAULT_ESTIMATED_WORKER_MEMORY_GB,
    GPU_ACCELERATION_ASSESSMENT,
    WorkerCountDecision,
    worker_count_decision,
)
from dense_archive_schema import BRANCH_DECISION_SCOPE  # noqa: E402
from dense_archive_table_io import list_table_partitions, resolve_storage_format  # noqa: E402
from run_paired_w0_w1_archive_chunk import (  # noqa: E402
    PairedChunkConfig,
    active_result_root,
    archive_run_root,
    chunk_status,
    generic_spec,
    remove_chunk_outputs,
    run_paired_w0_w1_archive_chunk,
)
from run_paired_w0_w1_partitioned_planning import (  # noqa: E402
    PAIRED_ENVIRONMENT_MODES,
    SIMULATION_STAGE,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Configuration and Paths
# 2) Schedule and Progress
# 3) Parallel Execution
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Configuration and Paths
# =============================================================================
RECOMMENDED_PAIRED_PROOF_COMMAND = (
    "python 03_Control/04_Scenarios/run_paired_w0_w1_archive_chunked.py "
    "--run-id 14 --planning-run-id 13 --workers 8 --max-workers 8 --resume"
)


@dataclass(frozen=True)
class PairedChunkedRunConfig:
    run_id: int = 14
    planning_run_id: int = 13
    result_root: Path | None = None
    environment_modes: tuple[str, ...] = PAIRED_ENVIRONMENT_MODES
    workers: str | int = "auto"
    max_workers: int | None = 8
    memory_safety_margin_gb: float = 8.0
    chunk_size: int = 2500
    storage_format: str = "auto"
    compression_level: int = 1
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    profile_first: bool = False
    dry_run_schedule: bool = False
    resume: bool = True
    repair_incomplete: bool = False
    stop_after_chunks: int | None = None
    continue_on_chunk_failure: bool = False
    random_seed: int = 20260526


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _planning_root(config: PairedChunkedRunConfig) -> Path:
    return (
        active_result_root(
            PairedChunkConfig(
                run_id=config.run_id,
                planning_run_id=config.planning_run_id,
                result_root=config.result_root,
                layout_branch_id="single_fan_branch",
                test_environment_mode="W1_single_fan",
            )
        ).parent
        / "10_dense_archive_planning"
        / f"{int(config.planning_run_id):03d}"
    )


def _progress_paths(config: PairedChunkedRunConfig) -> tuple[Path, Path]:
    root = archive_run_root(
        PairedChunkConfig(
            run_id=config.run_id,
            planning_run_id=config.planning_run_id,
            result_root=config.result_root,
            layout_branch_id="single_fan_branch",
            test_environment_mode="W1_single_fan",
        )
    )
    suffix = f"s{int(config.run_id):03d}"
    return (
        root / "manifests" / f"paired_w0_w1_progress_{suffix}.json",
        root / "reports" / f"paired_w0_w1_progress_{suffix}.md",
    )


def _run_paths(config: PairedChunkedRunConfig) -> dict[str, Path]:
    manifest_path, report_path = _progress_paths(config)
    return {
        "root": manifest_path.parents[1],
        "progress_manifest_json": manifest_path,
        "progress_report_md": report_path,
    }


# =============================================================================
# 2) Schedule and Progress
# =============================================================================
def _validate_config(config: PairedChunkedRunConfig) -> None:
    if int(config.run_id) <= 0 or int(config.planning_run_id) <= 0:
        raise ValueError("run_id and planning_run_id must be positive.")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if int(config.compression_level) < 0 or int(config.compression_level) > 9:
        raise ValueError("compression_level must be in [0, 9].")
    if config.stop_after_chunks is not None and int(config.stop_after_chunks) < 0:
        raise ValueError("stop_after_chunks must be nonnegative.")
    unknown = set(config.environment_modes).difference(PAIRED_ENVIRONMENT_MODES)
    if unknown:
        raise ValueError(f"unknown paired environment modes: {sorted(unknown)}")
    resolve_storage_format(config.storage_format)


def build_paired_chunk_schedule(config: PairedChunkedRunConfig) -> list[PairedChunkConfig]:
    root = _planning_root(config)
    paths = list_table_partitions(root, "candidate_index")
    rows: list[tuple[str, str, int]] = []
    for path in paths:
        parsed = _parse_planning_partition_path(path)
        if parsed is None:
            continue
        branch_id, environment_mode, chunk_index = parsed
        if environment_mode not in config.environment_modes:
            continue
        rows.append((branch_id, environment_mode, chunk_index))
    if not rows:
        raise FileNotFoundError(f"missing paired planning partitions under {root}")

    counts: dict[tuple[str, str], int] = {}
    for branch_id, environment_mode, _chunk_index in rows:
        counts[(branch_id, environment_mode)] = counts.get((branch_id, environment_mode), 0) + 1

    schedule: list[PairedChunkConfig] = []
    for branch_id, environment_mode, chunk_index in sorted(rows):
        schedule.append(
            PairedChunkConfig(
                run_id=int(config.run_id),
                planning_run_id=int(config.planning_run_id),
                result_root=config.result_root,
                layout_branch_id=branch_id,
                test_environment_mode=environment_mode,
                chunk_index=int(chunk_index),
                chunk_count=int(counts[(branch_id, environment_mode)]),
                chunk_size=int(config.chunk_size),
                latency_case=str(config.latency_case),
                dt_s=float(config.dt_s),
                horizon_s=float(config.horizon_s),
                storage_format=str(config.storage_format),
                compression_level=int(config.compression_level),
                resume=bool(config.resume),
                repair_incomplete=bool(config.repair_incomplete),
                random_seed=int(config.random_seed),
            )
        )
    run_root = archive_run_root(schedule[0])
    assert_unique_output_paths(
        [generic_spec(chunk) for chunk in schedule],
        run_root=run_root,
    )
    return schedule


def _parse_planning_partition_path(path: Path) -> tuple[str, str, int] | None:
    branch_id = None
    environment_mode = None
    chunk_index = None
    for part in Path(path).parts:
        if part.startswith("layout_branch_id="):
            branch_id = part.split("=", 1)[1]
        elif part.startswith("test_environment_mode="):
            environment_mode = part.split("=", 1)[1]
        elif part.startswith("archive_chunk_index="):
            chunk_index = int(part.split("=", 1)[1])
    if branch_id is None or environment_mode is None or chunk_index is None:
        return None
    return branch_id, environment_mode, chunk_index


def _prepare_pending_chunks(
    config: PairedChunkedRunConfig,
    schedule: list[PairedChunkConfig],
) -> tuple[list[PairedChunkConfig], list[PairedChunkConfig], list[PairedChunkConfig]]:
    pending: list[PairedChunkConfig] = []
    skipped: list[PairedChunkConfig] = []
    corrupt: list[PairedChunkConfig] = []
    for chunk in schedule:
        status = chunk_status(chunk)
        if status == "complete" and config.resume:
            skipped.append(chunk)
            continue
        if status == "corrupt":
            corrupt.append(chunk)
            if not config.repair_incomplete:
                continue
            remove_chunk_outputs(chunk)
        pending.append(chunk)
    if corrupt and not config.repair_incomplete:
        bad = ", ".join(
            f"{item.layout_branch_id}:{item.test_environment_mode}:{item.chunk_index}"
            for item in corrupt[:5]
        )
        raise RuntimeError(f"corrupt/incomplete paired chunks found: {bad}")
    if config.stop_after_chunks is not None:
        pending = pending[: int(config.stop_after_chunks)]
    return pending, skipped, corrupt


def _write_progress(
    *,
    config: PairedChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    schedule: list[PairedChunkConfig],
    pending: list[PairedChunkConfig],
    completed: list[dict[str, object]],
    skipped: list[PairedChunkConfig],
    failed: list[dict[str, object]],
    corrupt: list[PairedChunkConfig],
    dry_run: bool = False,
    profiling_rows_per_second: dict[str, float] | None = None,
) -> None:
    run_root = archive_run_root(schedule[0]) if schedule else _run_paths(config)["root"]
    chunk_records = progress_chunk_records(
        schedule=[generic_spec(chunk) for chunk in schedule],
        run_root=run_root,
        pending=[generic_spec(chunk) for chunk in pending],
        completed=completed,
        skipped=[generic_spec(chunk) for chunk in skipped],
        failed=failed,
        corrupt=[generic_spec(chunk) for chunk in corrupt],
        path_text=_path_text,
    )
    status = "dry_run" if dry_run else (
        "failed" if failed else ("complete" if not pending else "in_progress")
    )
    payload = progress_manifest_payload(
        status=status,
        run_id=int(config.run_id),
        planning_run_id=int(config.planning_run_id),
        simulation_stage=SIMULATION_STAGE,
        worker_decision=worker_decision,
        storage_format=str(config.storage_format),
        latency_case=str(config.latency_case),
        resume=bool(config.resume),
        repair_incomplete=bool(config.repair_incomplete),
        continue_on_chunk_failure=bool(config.continue_on_chunk_failure),
        chunks=chunk_records,
        recommended_command=RECOMMENDED_PAIRED_PROOF_COMMAND,
        branch_decision_scope=BRANCH_DECISION_SCOPE,
        failures=failed,
        profiling_rows_per_second=profiling_rows_per_second,
    )
    payload["gpu_acceleration_assessment"] = GPU_ACCELERATION_ASSESSMENT
    manifest_path, report_path = _progress_paths(config)
    write_progress_manifest(path=manifest_path, report_path=report_path, payload=payload)


# =============================================================================
# 3) Parallel Execution
# =============================================================================
def _worker_run_chunk(payload: dict[str, object]) -> dict[str, object]:
    result_root = payload.get("result_root")
    paths = run_paired_w0_w1_archive_chunk(
        run_id=int(payload["run_id"]),
        planning_run_id=int(payload["planning_run_id"]),
        result_root=None if result_root in {None, ""} else Path(str(result_root)),
        layout_branch_id=str(payload["layout_branch_id"]),
        test_environment_mode=str(payload["test_environment_mode"]),
        chunk_index=int(payload["chunk_index"]),
        chunk_count=int(payload["chunk_count"]),
        chunk_size=int(payload["chunk_size"]),
        latency_case=str(payload["latency_case"]),
        dt_s=float(payload["dt_s"]),
        horizon_s=float(payload["horizon_s"]),
        storage_format=str(payload["storage_format"]),
        compression_level=int(payload["compression_level"]),
        resume=bool(payload["resume"]),
        repair_incomplete=bool(payload["repair_incomplete"]),
        random_seed=int(payload["random_seed"]),
    )
    return {
        "layout_branch_id": str(payload["layout_branch_id"]),
        "test_environment_mode": str(payload["test_environment_mode"]),
        "chunk_index": int(payload["chunk_index"]),
        "status": "complete",
        "manifest_json": str(paths["manifest_json"]),
        "partition_path": str(paths["partition_path"]),
    }


def _chunk_payload(chunk: PairedChunkConfig) -> dict[str, object]:
    payload = asdict(chunk)
    payload["result_root"] = None if chunk.result_root is None else str(chunk.result_root)
    return payload


def _run_pending_chunks(
    *,
    config: PairedChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    schedule: list[PairedChunkConfig],
    pending: list[PairedChunkConfig],
    skipped: list[PairedChunkConfig],
    corrupt: list[PairedChunkConfig],
    profiling_rows_per_second: dict[str, float] | None,
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
        profiling_rows_per_second=profiling_rows_per_second,
    )
    if not pending:
        return completed, failed
    if int(worker_decision.selected_worker_count) == 1:
        for chunk in pending:
            try:
                completed.append(_worker_run_chunk(_chunk_payload(chunk)))
            except Exception as exc:
                failed.append(_failure_row(chunk, exc))
                if not config.continue_on_chunk_failure:
                    _write_progress(
                        config=config,
                        worker_decision=worker_decision,
                        schedule=schedule,
                        pending=pending,
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                        corrupt=corrupt,
                        profiling_rows_per_second=profiling_rows_per_second,
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
                profiling_rows_per_second=profiling_rows_per_second,
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
                if not config.continue_on_chunk_failure:
                    _write_progress(
                        config=config,
                        worker_decision=worker_decision,
                        schedule=schedule,
                        pending=pending,
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                        corrupt=corrupt,
                        profiling_rows_per_second=profiling_rows_per_second,
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
                profiling_rows_per_second=profiling_rows_per_second,
            )
    return completed, failed


def _failure_row(chunk: PairedChunkConfig, exc: BaseException) -> dict[str, object]:
    return {
        "layout_branch_id": str(chunk.layout_branch_id),
        "test_environment_mode": str(chunk.test_environment_mode),
        "chunk_index": int(chunk.chunk_index),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_paired_w0_w1_archive_chunked(
    *,
    run_id: int = 14,
    planning_run_id: int = 13,
    result_root: Path | None = None,
    environment_modes: tuple[str, ...] = PAIRED_ENVIRONMENT_MODES,
    workers: str | int = "auto",
    max_workers: int | None = 8,
    memory_safety_margin_gb: float = 8.0,
    chunk_size: int = 2500,
    storage_format: str = "auto",
    compression_level: int = 1,
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    profile_first: bool = False,
    dry_run_schedule: bool = False,
    resume: bool = True,
    repair_incomplete: bool = False,
    stop_after_chunks: int | None = None,
    continue_on_chunk_failure: bool = False,
    random_seed: int = 20260526,
) -> dict[str, Path]:
    config = PairedChunkedRunConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        result_root=result_root,
        environment_modes=tuple(environment_modes),
        workers=workers,
        max_workers=None if max_workers is None else int(max_workers),
        memory_safety_margin_gb=float(memory_safety_margin_gb),
        chunk_size=int(chunk_size),
        storage_format=str(storage_format),
        compression_level=int(compression_level),
        latency_case=str(latency_case),
        dt_s=float(dt_s),
        horizon_s=float(horizon_s),
        profile_first=bool(profile_first),
        dry_run_schedule=bool(dry_run_schedule),
        resume=bool(resume),
        repair_incomplete=bool(repair_incomplete),
        stop_after_chunks=stop_after_chunks,
        continue_on_chunk_failure=bool(continue_on_chunk_failure),
        random_seed=int(random_seed),
    )
    _validate_config(config)
    estimated_worker_memory_gb = DEFAULT_ESTIMATED_WORKER_MEMORY_GB
    profiling_rows_per_second: dict[str, float] | None = None
    if config.profile_first:
        from profile_paired_w0_w1_archive import profile_paired_w0_w1_archive

        profile_paths = profile_paired_w0_w1_archive(
            planning_run_id=int(config.planning_run_id),
            result_root=active_result_root(
                PairedChunkConfig(
                    run_id=config.run_id,
                    planning_run_id=config.planning_run_id,
                    result_root=config.result_root,
                    layout_branch_id="single_fan_branch",
                    test_environment_mode="W1_single_fan",
                )
            ),
            sample_trials=min(2000, int(config.chunk_size) * 4),
            storage_format=str(config.storage_format),
            latency_case=str(config.latency_case),
            workers=config.workers,
            memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        )
        profile = json.loads(profile_paths["profile_json"].read_text(encoding="ascii"))
        estimated_worker_memory_gb = float(
            profile.get("estimated_worker_memory_gb", estimated_worker_memory_gb)
        )
        profiling_rows_per_second = profile.get("rows_per_second_by_worker_count", None)
    worker_decision = worker_count_decision(
        config.workers,
        memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_workers=config.max_workers,
    )
    schedule = build_paired_chunk_schedule(config)
    pending, skipped, corrupt = _prepare_pending_chunks(config, schedule)
    if config.dry_run_schedule:
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
            profiling_rows_per_second=profiling_rows_per_second,
        )
        return _run_paths(config)
    completed, failed = _run_pending_chunks(
        config=config,
        worker_decision=worker_decision,
        schedule=schedule,
        pending=pending,
        skipped=skipped,
        corrupt=corrupt,
        profiling_rows_per_second=profiling_rows_per_second,
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
        profiling_rows_per_second=profiling_rows_per_second,
    )
    return _run_paths(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=14)
    parser.add_argument("--planning-run-id", type=int, default=13)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--environment-modes", nargs="*", default=list(PAIRED_ENVIRONMENT_MODES))
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--memory-safety-margin-gb", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--profile-first", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260526)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workers: str | int = int(args.workers) if str(args.workers).isdigit() else str(args.workers)
    paths = run_paired_w0_w1_archive_chunked(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        result_root=args.result_root,
        environment_modes=tuple(args.environment_modes),
        workers=workers,
        max_workers=args.max_workers,
        memory_safety_margin_gb=args.memory_safety_margin_gb,
        chunk_size=args.chunk_size,
        storage_format=args.storage_format,
        compression_level=args.compression_level,
        latency_case=args.latency_case,
        dt_s=args.dt_s,
        horizon_s=args.horizon_s,
        profile_first=args.profile_first,
        dry_run_schedule=args.dry_run_schedule,
        resume=args.resume,
        repair_incomplete=args.repair_incomplete,
        stop_after_chunks=args.stop_after_chunks,
        continue_on_chunk_failure=args.continue_on_chunk_failure,
        random_seed=args.random_seed,
    )
    print(f"paired_w0_w1_archive_chunked_outputs={_path_text(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
