"""Stage-isolated v1.4 driver for feedback contextual primitive evidence.

This runner hardens the v1.3 overnight path without changing the method:
R6 is a W0/W1 contextual archive, R7 is a selector report, R8 is W2
model-backed replay, and R9 is optional W3 generalisation.  The driver is
safe to use for dry schedules and tiny tests; it does not erase earlier stage
outputs when later stages block.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import py_compile
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:  # pragma: no cover - exercised by package import tests
    from .env_surrogate import resolve_surrogate_binding, validate_surrogate_ladder
    from .evidence_stage_utils import (
        StageEvidenceStatus,
        stage_status,
        write_evidence_status_manifest,
    )
    from .run_ctx_archive import ContextArchiveConfig as ArchiveRunConfig
    from .run_ctx_archive import run_contextual_archive_preflight
    from .run_primitive_selector_report import (
        SelectorReportConfig,
        run_primitive_selector_report,
    )
    from .run_w2_replay import W2ReplayConfig, run_w2_replay
    from .run_w3_generalisation import W3GeneralisationConfig, run_w3_generalisation
except ImportError:  # pragma: no cover - script execution fallback
    from env_surrogate import resolve_surrogate_binding, validate_surrogate_ladder
    from evidence_stage_utils import StageEvidenceStatus, stage_status, write_evidence_status_manifest
    from run_ctx_archive import ContextArchiveConfig as ArchiveRunConfig
    from run_ctx_archive import run_contextual_archive_preflight
    from run_primitive_selector_report import (
        SelectorReportConfig,
        run_primitive_selector_report,
    )
    from run_w2_replay import W2ReplayConfig, run_w2_replay
    from run_w3_generalisation import W3GeneralisationConfig, run_w3_generalisation


STAGES = ("R6", "R7", "R8", "R9")
STATUS_MANIFEST_NAME = "feedback_contextual_primitive_v1_4_status.json"
PREFERRED_PARTITION_BYTES = 75 * 1024 * 1024
HARD_PARTITION_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class CommandResult:
    """Small serialisable record for preflight subprocesses."""

    label: str
    command: tuple[str, ...]
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True)
class ProjectionResult:
    """Stage-specific first-chunk projection."""

    stage: str
    target_rows: int
    fallback_rows: int
    selected_rows: int
    projected_wall_time_s: float
    first_chunk_seconds: float
    actual_worker_count: int
    total_chunks: int
    first_partition_bytes: int
    selected_chunk_size: int
    fallback_reason: str
    blocked: bool = False


@dataclass(frozen=True)
class OvernightV14Config:
    """Configuration for the v1.4 hardened stage driver."""

    run_id: str = "v1_4_smoke"
    output_root: Path = Path("03_Control/05_Results/feedback_contextual_v1_4")
    r6_target_rows: int = 80_000
    r6_fallback_rows: int = 40_000
    r8_target_rows: int = 15_000
    r8_fallback_rows: int = 2_000
    r9_target_rows: int = 30_000
    r9_fallback_rows: int = 5_000
    candidate_chunk_size: int = 1000
    workers: int = 8
    max_workers: int = 8
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    stop_after_stage: str | None = None
    skip_r9: bool = False
    run_preflight_checks: bool = True
    preflight_commands: tuple[tuple[str, ...], ...] | None = None
    r6_stage_time_budget_s: float = 12 * 60 * 60
    r8_stage_time_budget_s: float = 4 * 60 * 60
    r9_stage_time_budget_s: float = 6 * 60 * 60
    r7_evaluation_max_rows: int = 4096
    status_manifest_name: str = STATUS_MANIFEST_NAME
    extra_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StageDriverResult:
    """Return value for tests and callers."""

    run_root: Path
    status_manifest_path: Path
    stage_statuses: tuple[StageEvidenceStatus, ...]
    preflight_results: tuple[CommandResult, ...]
    projections: tuple[ProjectionResult, ...]


def run_feedback_contextual_v1_4_overnight(config: OvernightV14Config) -> StageDriverResult:
    """Run the hardened stage driver without deleting prior evidence roots."""

    run_root = Path(config.output_root) / f"run_{config.run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    statuses: list[StageEvidenceStatus] = []
    projections: list[ProjectionResult] = []

    preflight_results = tuple(_run_preflight(config, run_root))
    r6_surrogate = _surrogate_status(("W0", "W1"))
    preflight_failure = _preflight_failure(preflight_results, r6_surrogate)
    if preflight_failure:
        statuses.append(
            stage_status(
                stage="R6",
                status="blocked",
                row_count=0,
                table_manifest_path="",
                fallback_reason=preflight_failure,
                coverage_status="blocked",
                surrogate_status=r6_surrogate,
                claim_status="no_evidence_claim",
            )
        )
        _extend_deferred(statuses, "R6", "preflight_failed")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if _should_stop(config, "preflight"):
        statuses = _with_deferred(statuses, reason="stopped_after_preflight")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    if config.dry_run_schedule:
        _run_r6_dry_schedule(config, run_root)
        statuses = _with_deferred(statuses, reason="dry_run_schedule_only")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r6_projection = _run_r6_projection(config, run_root)
    projections.append(r6_projection)
    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if r6_projection.blocked:
        statuses.append(_blocked_from_projection("R6", r6_projection, r6_surrogate))
        _extend_deferred(statuses, "R6", "r6_projection_blocked")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))
    if _should_stop(config, "r6_projection"):
        statuses = _with_deferred(statuses, reason="stopped_after_r6_projection")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r6_result = _run_r6_archive(config, run_root, r6_projection)
    r6_status = _stage_status_from_archive_result(
        stage="R6",
        result=r6_result,
        target_rows=config.r6_target_rows,
        fallback_rows=config.r6_fallback_rows,
        fallback_reason=r6_projection.fallback_reason,
        surrogate_status=r6_surrogate,
    )
    statuses.append(r6_status)
    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if _should_stop(config, "r6") or r6_status.status not in {"complete", "fallback"}:
        statuses = _with_deferred(statuses, reason="stopped_or_r6_not_complete")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r7_result = _run_r7_report(config, run_root, Path(str(r6_status.table_manifest_path)))
    r7_status = _stage_status_from_selector_result(r7_result)
    statuses.append(r7_status)
    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if _should_stop(config, "r7") or r7_status.status in {"blocked", "deferred"}:
        statuses = _with_deferred(statuses, reason="stopped_or_r7_not_complete")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r8_surrogate = _surrogate_status(("W2",))
    if "ready" not in r8_surrogate:
        statuses.append(
            stage_status(
                stage="R8",
                status="blocked",
                row_count=0,
                table_manifest_path="",
                fallback_reason="missing_w2_surrogate_support",
                coverage_status="blocked",
                surrogate_status=r8_surrogate,
                claim_status="no_w2_replay_claim",
            )
        )
        _extend_deferred(statuses, "R8", "r8_blocked")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r8_projection = _run_r8_projection(config, run_root, Path(str(r6_status.table_manifest_path)))
    projections.append(r8_projection)
    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if r8_projection.blocked:
        statuses.append(_blocked_from_projection("R8", r8_projection, r8_surrogate))
        _extend_deferred(statuses, "R8", "r8_projection_blocked")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))
    if _should_stop(config, "r8_projection"):
        statuses = _with_deferred(statuses, reason="stopped_after_r8_projection")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r8_result = _run_r8_replay(config, run_root, Path(str(r6_status.table_manifest_path)), r8_projection)
    r8_status = _stage_status_from_replay_result(
        stage="R8",
        result=r8_result,
        target_rows=config.r8_target_rows,
        fallback_rows=config.r8_fallback_rows,
        fallback_reason=r8_projection.fallback_reason,
        surrogate_status=r8_surrogate,
        claim_status="simulation_only_w2_replay_evidence",
    )
    statuses.append(r8_status)
    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if _should_stop(config, "r8") or r8_status.status not in {"complete", "fallback"}:
        statuses = _with_deferred(statuses, reason="stopped_or_r8_not_complete")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    if config.skip_r9:
        statuses = _with_deferred(statuses, reason="r9_skipped")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r9_surrogate = _surrogate_status(("W3",))
    if "ready" not in r9_surrogate:
        statuses.append(
            stage_status(
                stage="R9",
                status="blocked",
                row_count=0,
                table_manifest_path="",
                fallback_reason="missing_w3_surrogate_support",
                coverage_status="blocked",
                surrogate_status=r9_surrogate,
                claim_status="no_w3_generalisation_claim",
            )
        )
        status_path = _write_statuses(run_root, _with_deferred(statuses), config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r9_projection = _run_r9_projection(config, run_root, Path(str(r8_status.table_manifest_path)))
    projections.append(r9_projection)
    _write_statuses(run_root, _with_deferred(statuses), config, projections)
    if r9_projection.blocked:
        statuses.append(_blocked_from_projection("R9", r9_projection, r9_surrogate))
        status_path = _write_statuses(run_root, _with_deferred(statuses), config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))
    if _should_stop(config, "r9_projection"):
        statuses = _with_deferred(statuses, reason="stopped_after_r9_projection")
        status_path = _write_statuses(run_root, statuses, config, projections)
        return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))

    r9_result = _run_r9_generalisation(config, run_root, Path(str(r8_status.table_manifest_path)), r9_projection)
    r9_status = _stage_status_from_replay_result(
        stage="R9",
        result=r9_result,
        target_rows=config.r9_target_rows,
        fallback_rows=config.r9_fallback_rows,
        fallback_reason=r9_projection.fallback_reason,
        surrogate_status=r9_surrogate,
        claim_status="simulation_only_w3_generalisation_evidence",
    )
    statuses.append(r9_status)
    status_path = _write_statuses(run_root, _with_deferred(statuses), config, projections)
    return StageDriverResult(run_root, status_path, tuple(statuses), preflight_results, tuple(projections))


def _run_preflight(config: OvernightV14Config, run_root: Path) -> list[CommandResult]:
    if not config.run_preflight_checks:
        return [
            CommandResult(
                label="preflight_checks",
                command=("skipped",),
                returncode=0,
                stdout_tail="preflight checks skipped by config",
            )
        ]

    results: list[CommandResult] = []
    commands = config.preflight_commands
    if commands is not None:
        for index, command in enumerate(commands):
            results.append(_run_command(f"preflight_command_{index}", command, cwd=Path.cwd()))
        return results

    results.append(_run_py_compile())
    if results[-1].returncode != 0:
        return results
    results.append(_run_command("pytest", (sys.executable, "-m", "pytest", "-q", "03_Control/tests"), Path.cwd()))
    if results[-1].returncode != 0:
        return results
    results.append(_run_command("git_diff_check", ("git", "diff", "--check"), Path.cwd()))
    if results[-1].returncode != 0:
        return results
    results.append(_result_root_policy_check(run_root))
    return results


def _run_py_compile() -> CommandResult:
    roots = [Path("02_Glider_Design"), Path("03_Control"), Path("04_Simulation")]
    py_files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    errors: list[str] = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{path}: {exc.msg}")
            if len(errors) >= 8:
                break
    if errors:
        return CommandResult(
            label="py_compile",
            command=(sys.executable, "-m", "py_compile", "<retained importable modules>"),
            returncode=1,
            stderr_tail="\n".join(errors),
        )
    return CommandResult(
        label="py_compile",
        command=(sys.executable, "-m", "py_compile", "<retained importable modules>"),
        returncode=0,
        stdout_tail=f"compiled_py_files={len(py_files)}",
    )


def _run_command(label: str, command: Sequence[str], cwd: Path) -> CommandResult:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(
        label=label,
        command=tuple(command),
        returncode=int(completed.returncode),
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def _result_root_policy_check(run_root: Path) -> CommandResult:
    results_root = Path("03_Control/05_Results").resolve()
    resolved = run_root.resolve()
    if results_root in (resolved, *resolved.parents):
        return CommandResult(
            label="result_root_policy",
            command=("result-root-cleanliness",),
            returncode=0,
            stdout_tail=f"run_root={run_root}",
        )
    if "pytest-" in str(resolved) or "Temp" in str(resolved) or "tmp" in str(resolved).lower():
        return CommandResult(
            label="result_root_policy",
            command=("result-root-cleanliness",),
            returncode=0,
            stdout_tail=f"temp_run_root={run_root}",
        )
    return CommandResult(
        label="result_root_policy",
        command=("result-root-cleanliness",),
        returncode=1,
        stderr_tail=f"output root must be under {results_root} or a temp test directory: {run_root}",
    )


def _run_r6_dry_schedule(config: OvernightV14Config, run_root: Path) -> dict[str, object]:
    return run_contextual_archive_preflight(
        ArchiveRunConfig(
            run_id=_numeric_run_id(config, 1),
            rows=config.r6_target_rows,
            seed=_numeric_run_id(config, 1),
            w_layers=("W0", "W1"),
            env_modes=_r6_environment_modes(),
            candidate_chunk_size=config.candidate_chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=True,
            repair_incomplete=False,
            dry_run_schedule=True,
            stop_after_chunks=None,
            continue_on_chunk_failure=False,
            output_root=run_root / "dry_run_schedule" / "r6_w0_w1",
            rollout_backend="model_backed_feedback",
        )
    )


def _run_r6_projection(config: OvernightV14Config, run_root: Path) -> ProjectionResult:
    chunk_size = max(1, min(config.candidate_chunk_size, config.r6_target_rows))
    output_root = run_root / "first_chunk_projection" / "r6_w0_w1"
    started = time.perf_counter()
    result = run_contextual_archive_preflight(
        ArchiveRunConfig(
            run_id=_numeric_run_id(config, 2),
            rows=chunk_size,
            seed=_numeric_run_id(config, 2),
            w_layers=("W0", "W1"),
            env_modes=_r6_environment_modes(),
            candidate_chunk_size=chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=False,
            repair_incomplete=True,
            dry_run_schedule=False,
            stop_after_chunks=1,
            continue_on_chunk_failure=False,
            output_root=output_root,
            rollout_backend="model_backed_feedback",
        )
    )
    elapsed = max(0.001, time.perf_counter() - started)
    return _projection_from_result(
        stage="R6",
        result=result,
        target_rows=config.r6_target_rows,
        fallback_rows=config.r6_fallback_rows,
        chunk_size=chunk_size,
        first_chunk_seconds=elapsed,
        workers=config.workers,
        time_budget_s=config.r6_stage_time_budget_s,
    )


def _run_r8_projection(config: OvernightV14Config, run_root: Path, r6_manifest: Path) -> ProjectionResult:
    chunk_size = max(1, min(config.candidate_chunk_size, config.r8_target_rows))
    output_root = run_root / "first_chunk_projection" / "r8_w2_replay"
    started = time.perf_counter()
    result = run_w2_replay(
        W2ReplayConfig(
            source_archive=r6_manifest,
            target_rows=chunk_size,
            output_root=output_root,
            run_id=_numeric_run_id(config, 8),
            chunk_size=chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=False,
            repair_incomplete=True,
            stop_after_chunks=1,
        )
    )
    elapsed = max(0.001, time.perf_counter() - started)
    return _projection_from_result(
        stage="R8",
        result=result,
        target_rows=config.r8_target_rows,
        fallback_rows=config.r8_fallback_rows,
        chunk_size=chunk_size,
        first_chunk_seconds=elapsed,
        workers=config.workers,
        time_budget_s=config.r8_stage_time_budget_s,
    )


def _run_r9_projection(config: OvernightV14Config, run_root: Path, r8_manifest: Path) -> ProjectionResult:
    chunk_size = max(1, min(config.candidate_chunk_size, config.r9_target_rows))
    output_root = run_root / "first_chunk_projection" / "r9_w3_generalisation"
    started = time.perf_counter()
    result = run_w3_generalisation(
        W3GeneralisationConfig(
            source_replay=r8_manifest,
            target_rows=chunk_size,
            output_root=output_root,
            run_id=_numeric_run_id(config, 9),
            chunk_size=chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=False,
            repair_incomplete=True,
            stop_after_chunks=1,
        )
    )
    elapsed = max(0.001, time.perf_counter() - started)
    return _projection_from_result(
        stage="R9",
        result=result,
        target_rows=config.r9_target_rows,
        fallback_rows=config.r9_fallback_rows,
        chunk_size=chunk_size,
        first_chunk_seconds=elapsed,
        workers=config.workers,
        time_budget_s=config.r9_stage_time_budget_s,
    )


def _run_r6_archive(config: OvernightV14Config, run_root: Path, projection: ProjectionResult) -> dict[str, object]:
    return run_contextual_archive_preflight(
        ArchiveRunConfig(
            run_id=_numeric_run_id(config, 6),
            rows=projection.selected_rows,
            seed=_numeric_run_id(config, 6),
            w_layers=("W0", "W1"),
            env_modes=_r6_environment_modes(),
            candidate_chunk_size=projection.selected_chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=config.resume,
            repair_incomplete=config.repair_incomplete,
            dry_run_schedule=False,
            stop_after_chunks=None,
            continue_on_chunk_failure=False,
            output_root=run_root / "r6_w0_w1_archive",
            rollout_backend="model_backed_feedback",
        )
    )


def _run_r7_report(config: OvernightV14Config, run_root: Path, r6_manifest: Path) -> dict[str, object]:
    return run_primitive_selector_report(
        SelectorReportConfig(
            archive_table=r6_manifest,
            output_root=run_root / "r7_selector_report",
            run_id=_numeric_run_id(config, 7),
            evaluation_max_rows=config.r7_evaluation_max_rows,
        )
    )


def _run_r8_replay(
    config: OvernightV14Config,
    run_root: Path,
    r6_manifest: Path,
    projection: ProjectionResult,
) -> dict[str, object]:
    return run_w2_replay(
        W2ReplayConfig(
            source_archive=r6_manifest,
            target_rows=projection.selected_rows,
            output_root=run_root / "r8_w2_replay",
            run_id=_numeric_run_id(config, 8),
            chunk_size=projection.selected_chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=config.resume,
            repair_incomplete=config.repair_incomplete,
        )
    )


def _run_r9_generalisation(
    config: OvernightV14Config,
    run_root: Path,
    r8_manifest: Path,
    projection: ProjectionResult,
) -> dict[str, object]:
    return run_w3_generalisation(
        W3GeneralisationConfig(
            source_replay=r8_manifest,
            target_rows=projection.selected_rows,
            output_root=run_root / "r9_w3_generalisation",
            run_id=_numeric_run_id(config, 9),
            chunk_size=projection.selected_chunk_size,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=config.resume,
            repair_incomplete=config.repair_incomplete,
        )
    )


def _projection_from_result(
    *,
    stage: str,
    result: dict[str, object],
    target_rows: int,
    fallback_rows: int,
    chunk_size: int,
    first_chunk_seconds: float,
    workers: int,
    time_budget_s: float,
) -> ProjectionResult:
    first_partition_bytes = _first_partition_bytes(result)
    selected_chunk_size = 500 if first_partition_bytes > PREFERRED_PARTITION_BYTES and chunk_size > 500 else chunk_size
    actual_workers = max(1, int(workers))
    selected_rows = target_rows
    total_chunks = max(1, math.ceil(target_rows / max(1, selected_chunk_size)))
    projected = math.ceil(total_chunks / actual_workers) * first_chunk_seconds * 1.25
    fallback_reason = ""
    blocked = False
    if first_partition_bytes > HARD_PARTITION_BYTES:
        blocked = True
        fallback_reason = "first_partition_exceeds_100mb_hard_gate"
        selected_rows = 0
    elif projected > time_budget_s:
        selected_rows = fallback_rows
        total_chunks = max(1, math.ceil(fallback_rows / max(1, selected_chunk_size)))
        projected = math.ceil(total_chunks / actual_workers) * first_chunk_seconds * 1.25
        fallback_reason = _fallback_reason_for_stage(stage)
    elif selected_chunk_size != chunk_size:
        fallback_reason = "chunk_size_reduced_1000_to_500_partition_projection"
    return ProjectionResult(
        stage=stage,
        target_rows=int(target_rows),
        fallback_rows=int(fallback_rows),
        selected_rows=int(selected_rows),
        projected_wall_time_s=float(projected),
        first_chunk_seconds=float(first_chunk_seconds),
        actual_worker_count=int(actual_workers),
        total_chunks=int(total_chunks),
        first_partition_bytes=int(first_partition_bytes),
        selected_chunk_size=int(selected_chunk_size),
        fallback_reason=fallback_reason,
        blocked=blocked,
    )


def _stage_status_from_archive_result(
    *,
    stage: str,
    result: dict[str, object],
    target_rows: int,
    fallback_rows: int,
    fallback_reason: str,
    surrogate_status: str,
) -> StageEvidenceStatus:
    manifest = str(result.get("table_manifest", ""))
    rows = int(result.get("row_count", 0) or 0)
    if rows <= 0 and manifest:
        rows = _row_count_from_table_manifest(Path(manifest))
    status = "complete" if rows >= target_rows else "fallback" if rows >= fallback_rows else "partial"
    if not manifest:
        status = "blocked"
    return stage_status(
        stage=stage,
        status=status,
        run_root=str(result.get("run_root", "")),
        row_count=rows,
        table_manifest_path=manifest,
        fallback_reason=fallback_reason,
        blocked_ratio=float(result.get("blocked_ratio", 0.0) or 0.0),
        file_size_status=str(result.get("file_size_status", "unknown")),
        coverage_status=str(result.get("coverage_status", "observed_summary_written")),
        surrogate_status=surrogate_status,
        runtime_projection_s=0.0,
        claim_status="simulation_only_contextual_archive_evidence" if status in {"complete", "fallback"} else "no_archive_claim",
    )


def _stage_status_from_selector_result(result: dict[str, object]) -> StageEvidenceStatus:
    manifest_payload: dict[str, object] = {}
    if result.get("manifest"):
        try:
            manifest_payload = json.loads(Path(str(result["manifest"])).read_text(encoding="ascii"))
        except (OSError, json.JSONDecodeError):
            manifest_payload = {}
    merged = {**manifest_payload, **result}
    status = str(merged.get("stage_status", "partial"))
    return stage_status(
        stage="R7",
        status=status,
        run_root=str(merged.get("run_root", "")),
        row_count=int(merged.get("evaluation_row_count", merged.get("candidate_row_count", 0)) or 0),
        table_manifest_path=str(merged.get("source_manifest_path", merged.get("archive_table", ""))),
        fallback_reason=str(merged.get("bounded_evaluation_reason", "")),
        file_size_status=str(merged.get("file_size_status", "unknown")),
        coverage_status=str(merged.get("evaluation_strategy", "unknown")),
        surrogate_status="not_applicable_selector_report",
        runtime_projection_s=0.0,
        claim_status="simulation_only_selector_report" if status in {"complete", "fallback"} else "no_selector_claim",
    )


def _stage_status_from_replay_result(
    *,
    stage: str,
    result: dict[str, object],
    target_rows: int,
    fallback_rows: int,
    fallback_reason: str,
    surrogate_status: str,
    claim_status: str,
) -> StageEvidenceStatus:
    manifest_payload: dict[str, object] = {}
    if result.get("manifest"):
        try:
            manifest_payload = json.loads(Path(str(result["manifest"])).read_text(encoding="ascii"))
        except (OSError, json.JSONDecodeError):
            manifest_payload = {}
    merged = {**manifest_payload, **result}
    rows = int(merged.get("row_count", merged.get("replayed_row_count", merged.get("case_count", 0))) or 0)
    manifest = str(result.get("table_manifest", ""))
    copied_label_block = not bool(merged.get("actual_model_backed_replay", merged.get("rollout_generated", True)))
    status = str(merged.get("replay_status", merged.get("generalisation_status", "")))
    if status not in {"complete", "fallback", "partial", "blocked", "deferred"}:
        status = "complete" if rows >= target_rows else "fallback" if rows >= fallback_rows else "partial"
    if copied_label_block or not manifest:
        status = "blocked"
        fallback_reason = fallback_reason or "missing_actual_rollout_rows"
    return stage_status(
        stage=stage,
        status=status,
        run_root=str(result.get("run_root", "")),
        row_count=rows,
        table_manifest_path=manifest,
        fallback_reason=fallback_reason,
        blocked_ratio=float(merged.get("blocked_ratio", 0.0) or 0.0),
        approximate_ratio=float(merged.get("approximate_ratio", 0.0) or 0.0),
        file_size_status=str(merged.get("file_size_status", "unknown")),
        coverage_status=str(merged.get("coverage_status", "unknown")),
        surrogate_status=surrogate_status,
        runtime_projection_s=0.0,
        claim_status=claim_status if status in {"complete", "fallback"} else "no_replay_claim",
    )


def _blocked_from_projection(stage: str, projection: ProjectionResult, surrogate_status: str) -> StageEvidenceStatus:
    return stage_status(
        stage=stage,
        status="blocked",
        row_count=0,
        table_manifest_path="",
        fallback_reason=projection.fallback_reason,
        coverage_status="projection_blocked",
        surrogate_status=surrogate_status,
        runtime_projection_s=projection.projected_wall_time_s,
        claim_status="no_evidence_claim",
    )


def _write_statuses(
    run_root: Path,
    statuses: Sequence[StageEvidenceStatus],
    config: OvernightV14Config,
    projections: Sequence[ProjectionResult],
) -> Path:
    status_root = run_root / "evidence_status"
    status_root.mkdir(parents=True, exist_ok=True)
    path = status_root / config.status_manifest_name
    metadata = {
        "version": "feedback_contextual_primitive_v1_4",
        "run_id": config.run_id,
        "driver": Path(__file__).name,
        "dry_run_schedule": config.dry_run_schedule,
        "resume": config.resume,
        "repair_incomplete": config.repair_incomplete,
        "candidate_chunk_size": config.candidate_chunk_size,
        "workers": config.workers,
        "max_workers": config.max_workers,
        "storage_format": config.storage_format,
        "compression_level": config.compression_level,
        "projection_records": [projection.__dict__ for projection in projections],
        "claim_boundary": (
            "No controller-performance, mission-success, hardware-readiness, real-flight-transfer, "
            "full W2 survival, W3 robustness, or environment-generalisation claim is made unless "
            "the corresponding evidence table and manifest exist."
        ),
        **config.extra_metadata,
    }
    write_evidence_status_manifest(
        path,
        statuses=list(statuses),
        run_label="feedback_contextual_primitive_v1_4",
        run_root=run_root,
        metadata=metadata,
    )
    return path


def _with_deferred(
    statuses: Sequence[StageEvidenceStatus],
    reason: str = "not_yet_run",
) -> list[StageEvidenceStatus]:
    out = list(statuses)
    seen = {status.stage for status in out}
    for stage in STAGES:
        if stage not in seen:
            out.append(
                stage_status(
                    stage=stage,
                    status="deferred",
                    row_count=0,
                    table_manifest_path="",
                    fallback_reason=reason,
                    coverage_status="not_run",
                    surrogate_status="stage_local_check_pending",
                    claim_status="no_evidence_claim",
                )
            )
    return out


def _extend_deferred(statuses: list[StageEvidenceStatus], after_stage: str, reason: str) -> None:
    seen = {status.stage for status in statuses}
    start = STAGES.index(after_stage) + 1 if after_stage in STAGES else 0
    for stage in STAGES[start:]:
        if stage not in seen:
            statuses.append(
                stage_status(
                    stage=stage,
                    status="deferred",
                    row_count=0,
                    table_manifest_path="",
                    fallback_reason=reason,
                    coverage_status="not_run",
                    surrogate_status="stage_local_check_pending",
                    claim_status="no_evidence_claim",
                )
            )


def _preflight_failure(results: Sequence[CommandResult], surrogate_status: str) -> str:
    failing = [result for result in results if result.returncode != 0]
    if failing:
        first = failing[0]
        return f"{first.label}_failed_returncode_{first.returncode}"
    if "ready" not in surrogate_status:
        return "missing_w0_w1_surrogate_support"
    return ""


def _surrogate_status(layers: Iterable[str]) -> str:
    layer_tuple = tuple(str(layer).upper() for layer in layers)
    rows: list[str] = []
    for layer in layer_tuple:
        metadata = _surrogate_metadata_for_layer(layer)
        binding = resolve_surrogate_binding(layer, metadata)
        if binding.surrogate_binding_status != "ready":
            rows.append(f"{layer}:blocked:{binding.blocked_reason}")
            continue
        try:
            validate_surrogate_ladder(binding)
        except ValueError as exc:
            rows.append(f"{layer}:blocked:{exc}")
            continue
        rows.append(f"{layer}:ready:{binding.updraft_model_id}")
    if all(":ready:" in row for row in rows):
        return "ready:" + ",".join(rows)
    return "blocked:" + ",".join(rows)


def _surrogate_metadata_for_layer(layer: str):
    try:
        from .env_ctx import EnvironmentMetadata
    except ImportError:  # pragma: no cover - script execution fallback
        from env_ctx import EnvironmentMetadata

    if layer == "W0":
        return EnvironmentMetadata(
            environment_id="w0_dry_air",
            fan_count=0,
            updraft_model_id="dry_air_zero_wind",
            W_layer="W0",
            wind_mode="none",
            environment_mode="dry_air",
        )
    if layer == "W1":
        return EnvironmentMetadata(
            environment_id="w1_gaussian_single",
            fan_count=1,
            updraft_model_id="single_gaussian_var",
            W_layer="W1",
            wind_mode="panel",
            environment_mode="gaussian_single",
        )
    if layer == "W2":
        return EnvironmentMetadata(
            environment_id="w2_annular_gp_single",
            fan_count=1,
            updraft_model_id="single_annular_gp_grid",
            W_layer="W2",
            wind_mode="panel",
            environment_mode="annular_gp_single",
        )
    return EnvironmentMetadata(
        environment_id="w3_randomised_annular_gp_single",
        fan_count=1,
        updraft_model_id="single_annular_gp_grid",
        W_layer="W3",
        wind_mode="panel",
        environment_mode="randomised_annular_gp_single",
        randomisation_seed=0,
    )


def _first_partition_bytes(result: dict[str, object]) -> int:
    candidates = [
        result.get("replay_table"),
        result.get("archive_table"),
        result.get("table_manifest"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file() and path.name != "table_manifest.json":
            return path.stat().st_size
        if path.is_file() and path.name == "table_manifest.json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return 0
            root = Path(str(payload.get("root", path.parent.parent)))
            for table in payload.get("tables", []):
                table_path = root / "tables" / str(table.get("relative_path", table.get("path", "")))
                if table_path.is_file():
                    return table_path.stat().st_size
    return 0


def _row_count_from_table_manifest(path: Path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(sum(int(row.get("row_count", 0) or 0) for row in payload.get("tables", [])))


def _fallback_reason_for_stage(stage: str) -> str:
    if stage == "R6":
        return "fallback_80k_to_40k_runtime_or_partition_limit"
    if stage == "R8":
        return "fallback_15k_to_2k_runtime_or_blocked_ratio_limit"
    if stage == "R9":
        return "fallback_30k_to_5k_runtime_or_blocked_approximate_limit"
    return "fallback_runtime_or_partition_limit"


def _numeric_run_id(config: OvernightV14Config, offset: int) -> int:
    text = str(config.run_id)
    digits = "".join(character for character in text if character.isdigit())
    base = int(digits) if digits else sum(ord(character) for character in text) % 9000 + 1
    return int(base * 10 + int(offset))


def _r6_environment_modes() -> tuple[str, ...]:
    return ("dry_air", "gaussian_single", "gaussian_four", "fan_shift", "power_scale")


def _should_stop(config: OvernightV14Config, stage: str) -> bool:
    if not config.stop_after_stage:
        return False
    return config.stop_after_stage.strip().lower() == stage.strip().lower()


def _tail(text: str, limit: int = 2400) -> str:
    text = text or ""
    return text[-limit:]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v1_4_smoke")
    parser.add_argument("--output-root", type=Path, default=Path("03_Control/05_Results/feedback_contextual_v1_4"))
    parser.add_argument("--r6-target-rows", type=int, default=80_000)
    parser.add_argument("--r6-fallback-rows", type=int, default=40_000)
    parser.add_argument("--r8-target-rows", type=int, default=15_000)
    parser.add_argument("--r8-fallback-rows", type=int, default=2_000)
    parser.add_argument("--r9-target-rows", type=int, default=30_000)
    parser.add_argument("--r9-fallback-rows", type=int, default=5_000)
    parser.add_argument("--candidate-chunk-size", "--chunk-size", dest="candidate_chunk_size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-stage", default=None)
    parser.add_argument("--skip-r9", action="store_true")
    parser.add_argument("--skip-preflight-checks", action="store_true")
    parser.add_argument("--r7-evaluation-max-rows", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_feedback_contextual_v1_4_overnight(
        OvernightV14Config(
            run_id=args.run_id,
            output_root=args.output_root,
            r6_target_rows=args.r6_target_rows,
            r6_fallback_rows=args.r6_fallback_rows,
            r8_target_rows=args.r8_target_rows,
            r8_fallback_rows=args.r8_fallback_rows,
            r9_target_rows=args.r9_target_rows,
            r9_fallback_rows=args.r9_fallback_rows,
            candidate_chunk_size=args.candidate_chunk_size,
            workers=args.workers,
            max_workers=args.max_workers,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            resume=args.resume,
            repair_incomplete=args.repair_incomplete,
            dry_run_schedule=args.dry_run_schedule,
            stop_after_stage=args.stop_after_stage,
            skip_r9=args.skip_r9,
            run_preflight_checks=not args.skip_preflight_checks,
            r7_evaluation_max_rows=args.r7_evaluation_max_rows,
        )
    )
    print(result.status_manifest_path)
    blocking = [status for status in result.stage_statuses if status.status == "blocked"]
    return 2 if blocking else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
