from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_runtime import (  # noqa: E402
    DEFAULT_ESTIMATED_WORKER_MEMORY_GB,
    DEFAULT_MEMORY_SAFETY_MARGIN_GB,
    GPU_ACCELERATION_ASSESSMENT,
    WorkerCountDecision,
    runtime_manifest_fields,
    worker_count_decision,
)
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    file_sha256,
    filesystem_path,
    read_table_partition,
    resolve_storage_format,
    table_extension,
    write_table_manifest,
    write_table_partition,
)
from fixed_gate_code_path_map import active_code_path_text, code_path_map_frame  # noqa: E402
from fixed_gate_primitive_rollout import (  # noqa: E402
    FixedGatePrimitiveRolloutConfig,
    build_archive_move_on_gates,
    build_rollout_outcome_summary,
    build_w0_w1_pairing_audit,
    run_fixed_gate_primitive_rollouts,
)
from fixed_gate_sampling import (  # noqa: E402
    build_fixed_gate_w0_w1_candidate_rows,
    validate_w1_independent_of_w0,
)
from primitive_roles import active_mission_primitive_families  # noqa: E402
from run_fixed_gate_w0_w1_archive import (  # noqa: E402
    BRANCHES,
    CAMPAIGN,
    PASS_NAME,
    RESULT_ROOT,
    _branch_coverage_report,
    _branch_coverage_summary,
    _branches,
    _layers,
    _limit_candidate_rows,
    _read_optional_reachable_sources,
    _role_rows,
    _sample_archive_states,
)


ARCHIVE_PASS_DIR = "003_fixed_gate_w0_w1_proof_archive"
CHUNKED_PASS_NAME = "fixed_gate_w0_w1_chunked_primitive_rollout_archive"
V11_4_EXECUTION_GUIDANCE = "Nausicaa_CODEX_v11_4_execute_official_chunked_W0_W1_archive_guidance.md"
OFFICIAL_PRIMARY_RUN_ID = 6
PREFLIGHT_RUN_ID = 999


@dataclass(frozen=True)
class FixedGateChunkedRunConfig:
    run_id: int
    rows_per_branch: int = 200
    seed: int = 20260522
    fan_branch: str = "all"
    w_layers: str = "W0,W1"
    latency_case: str = "nominal"
    controller_mode: str = "both"
    candidate_chunk_size: int = 2500
    workers: str | int = 8
    max_workers: int | None = 8
    memory_safety_margin_gb: float = DEFAULT_MEMORY_SAFETY_MARGIN_GB
    storage_format: str = "csv_gz"
    compression_level: int = 1
    reachable_source_csv: Path | None = None
    result_root: Path | None = None
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    stop_after_chunks: int | None = None
    continue_on_chunk_failure: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class FixedGateChunkSpec:
    chunk_index: int
    chunk_count: int
    fan_branch: str
    candidate_rows: pd.DataFrame
    sample_rows: pd.DataFrame


def run_fixed_gate_w0_w1_archive_chunked(
    *,
    run_id: int,
    rows_per_branch: int = 200,
    seed: int = 20260522,
    fan_branch: str = "all",
    w_layers: str = "W0,W1",
    latency_case: str = "nominal",
    controller_mode: str = "both",
    candidate_chunk_size: int = 2500,
    workers: str | int = 8,
    max_workers: int | None = 8,
    memory_safety_margin_gb: float = DEFAULT_MEMORY_SAFETY_MARGIN_GB,
    storage_format: str = "csv_gz",
    compression_level: int = 1,
    reachable_source_csv: Path | None = None,
    result_root: Path | None = None,
    resume: bool = True,
    repair_incomplete: bool = False,
    dry_run_schedule: bool = False,
    stop_after_chunks: int | None = None,
    continue_on_chunk_failure: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    config = FixedGateChunkedRunConfig(
        run_id=int(run_id),
        rows_per_branch=int(rows_per_branch),
        seed=int(seed),
        fan_branch=str(fan_branch),
        w_layers=str(w_layers),
        latency_case=str(latency_case),
        controller_mode=str(controller_mode),
        candidate_chunk_size=int(candidate_chunk_size),
        workers=workers,
        max_workers=max_workers,
        memory_safety_margin_gb=float(memory_safety_margin_gb),
        storage_format=str(storage_format),
        compression_level=int(compression_level),
        reachable_source_csv=reachable_source_csv,
        result_root=result_root,
        resume=bool(resume),
        repair_incomplete=bool(repair_incomplete),
        dry_run_schedule=bool(dry_run_schedule),
        stop_after_chunks=stop_after_chunks,
        continue_on_chunk_failure=bool(continue_on_chunk_failure),
        overwrite=bool(overwrite),
    )
    _validate_config(config)
    root = _archive_root(config.run_id, config.result_root)
    execution_context = _execution_context(config, root)
    paths = _prepare_tree(root, config)
    worker_decision = worker_count_decision(
        config.workers,
        memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        estimated_worker_memory_gb=DEFAULT_ESTIMATED_WORKER_MEMORY_GB,
        max_workers=config.max_workers,
    )
    _write_build_note(paths, config, worker_decision, execution_context)
    _write_progress(
        paths=paths,
        config=config,
        worker_decision=worker_decision,
        schedule=[],
        pending=[],
        skipped=[],
        failed=[],
        corrupt=[],
        status="building_schedule",
    )

    samples, candidates, schedule = build_fixed_gate_chunk_schedule(config)

    pending, skipped, corrupt = _prepare_pending_chunks(root, schedule, config)
    if config.stop_after_chunks is not None:
        pending = pending[: int(config.stop_after_chunks)]
    _write_progress(
        paths=paths,
        config=config,
        worker_decision=worker_decision,
        schedule=schedule,
        pending=pending,
        skipped=skipped,
        failed=[],
        corrupt=corrupt,
        status="dry_run" if config.dry_run_schedule else "scheduled",
    )
    if config.dry_run_schedule:
        _write_schedule(paths, schedule)
        return _output_paths(paths)

    completed, failed = _run_pending_chunks(
        config=config,
        root=root,
        pending=pending,
        worker_decision=worker_decision,
    )
    _write_progress(
        paths=paths,
        config=config,
        worker_decision=worker_decision,
        schedule=schedule,
        pending=[] if not failed else pending,
        skipped=skipped,
        failed=failed,
        corrupt=corrupt,
        status="failed" if failed else "complete",
    )
    if failed and not config.continue_on_chunk_failure:
        raise RuntimeError(f"fixed-gate chunked archive failed for {len(failed)} chunk(s)")

    _write_final_outputs(
        paths=paths,
        config=config,
        worker_decision=worker_decision,
        samples=samples,
        candidates=candidates,
        schedule=schedule,
        completed=completed,
        skipped=skipped,
        failed=failed,
        corrupt=corrupt,
        execution_context=execution_context,
    )
    return _output_paths(paths)


def build_fixed_gate_chunk_schedule(
    config: FixedGateChunkedRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FixedGateChunkSpec]]:
    branches = _branches(config.fan_branch)
    layers = _layers(config.w_layers)
    reachable_sources = _read_optional_reachable_sources(config.reachable_source_csv)
    samples = _sample_archive_states(
        rows_per_branch=int(config.rows_per_branch),
        seed=int(config.seed),
        branches=branches,
        reachable_sources=reachable_sources,
    )
    candidates = build_fixed_gate_w0_w1_candidate_rows(
        samples,
        primitive_families=active_mission_primitive_families(),
    )
    candidates = candidates[candidates["W_layer"].astype(str).isin(layers)].copy()
    candidates = _limit_candidate_rows(candidates, rows_per_branch=int(config.rows_per_branch), branches=branches)
    validate_w1_independent_of_w0(candidates) if set(layers) == {"W0", "W1"} else None
    chunks: list[tuple[str, pd.DataFrame]] = []
    group_keys = ["fan_branch", "paired_sample_key", "primitive_family"]
    for branch in branches:
        branch_rows = candidates[candidates["fan_branch"].astype(str).eq(branch)].copy()
        current: list[pd.DataFrame] = []
        current_count = 0
        for _, group in branch_rows.groupby(group_keys, sort=True, dropna=False):
            current.append(group)
            current_count += int(len(group))
            if current_count >= int(config.candidate_chunk_size):
                chunks.append((branch, pd.concat(current, ignore_index=True)))
                current = []
                current_count = 0
        if current:
            chunks.append((branch, pd.concat(current, ignore_index=True)))
    chunk_count = len(chunks)
    schedule = []
    for chunk_index, (branch, chunk_candidates) in enumerate(chunks):
        sample_ids = set(chunk_candidates["sample_id"].astype(str))
        sample_rows = samples[samples["sample_id"].astype(str).isin(sample_ids)].copy()
        schedule.append(
            FixedGateChunkSpec(
                chunk_index=int(chunk_index),
                chunk_count=int(chunk_count),
                fan_branch=str(branch),
                candidate_rows=chunk_candidates.reset_index(drop=True),
                sample_rows=sample_rows.reset_index(drop=True),
            )
        )
    return samples, candidates, schedule


def _validate_config(config: FixedGateChunkedRunConfig) -> None:
    if int(config.run_id) <= 0:
        raise ValueError("run_id must be positive.")
    if int(config.rows_per_branch) <= 0:
        raise ValueError("rows_per_branch must be positive.")
    if int(config.candidate_chunk_size) <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    if int(config.compression_level) < 0 or int(config.compression_level) > 9:
        raise ValueError("compression_level must be in [0, 9].")
    if config.stop_after_chunks is not None and int(config.stop_after_chunks) < 0:
        raise ValueError("stop_after_chunks must be nonnegative.")
    _branches(config.fan_branch)
    _layers(config.w_layers)
    resolve_storage_format(config.storage_format)


def _archive_root(run_id: int, result_root: Path | None) -> Path:
    return (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / ARCHIVE_PASS_DIR


def _prepare_tree(root: Path, config: FixedGateChunkedRunConfig) -> dict[str, Path]:
    root_fs = filesystem_path(root)
    if root_fs.exists() and any(root_fs.iterdir()):
        if config.overwrite:
            if not _is_scratch_root(root):
                raise RuntimeError(f"--overwrite is allowed only for scratch/preflight roots, got {root}")
            _safe_remove_tree(root)
        elif not config.resume:
            raise RuntimeError(f"result tree already exists; rerun with --resume or use a new run id: {root}")
    paths = {
        "root": root,
        "tables": root / "tables",
        "metrics": root / "metrics",
        "manifests": root / "manifests",
        "reports": root / "reports",
        "chunk_manifests": root / "chunk_manifests",
    }
    for path in paths.values():
        filesystem_path(path).mkdir(parents=True, exist_ok=True)
    return paths


def _is_scratch_root(root: Path) -> bool:
    text = root.as_posix().lower()
    return "preflight" in text or "scratch" in text


def _safe_remove_tree(root: Path) -> None:
    def onexc(function: object, path: str, excinfo: object) -> None:
        del excinfo
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            pass
        function(path)

    shutil.rmtree(filesystem_path(root), onexc=onexc)


def _prepare_pending_chunks(
    root: Path,
    schedule: list[FixedGateChunkSpec],
    config: FixedGateChunkedRunConfig,
) -> tuple[list[FixedGateChunkSpec], list[FixedGateChunkSpec], list[FixedGateChunkSpec]]:
    pending: list[FixedGateChunkSpec] = []
    skipped: list[FixedGateChunkSpec] = []
    corrupt: list[FixedGateChunkSpec] = []
    for chunk in schedule:
        status = _chunk_status(root, chunk.chunk_index)
        if status == "complete" and config.resume:
            skipped.append(chunk)
            continue
        if status == "corrupt":
            corrupt.append(chunk)
            if not config.repair_incomplete:
                continue
            _remove_chunk_outputs(root, chunk.chunk_index)
        pending.append(chunk)
    if corrupt and not config.repair_incomplete:
        bad = ", ".join(f"{item.fan_branch}:{item.chunk_index}" for item in corrupt[:5])
        raise RuntimeError(f"corrupt/incomplete fixed-gate chunks found: {bad}; rerun with --repair-incomplete")
    return pending, skipped, corrupt


def _run_pending_chunks(
    *,
    config: FixedGateChunkedRunConfig,
    root: Path,
    pending: list[FixedGateChunkSpec],
    worker_decision: WorkerCountDecision,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    if not pending:
        return completed, failed
    payloads = [_chunk_payload(config, root, chunk) for chunk in pending]
    if int(worker_decision.selected_worker_count) <= 1:
        for payload in payloads:
            try:
                completed.append(_worker_run_chunk(payload))
            except Exception as exc:
                failed.append(_failure_record(payload, exc))
                if not config.continue_on_chunk_failure:
                    break
        return completed, failed
    with ProcessPoolExecutor(max_workers=int(worker_decision.selected_worker_count)) as executor:
        future_map = {executor.submit(_worker_run_chunk, payload): payload for payload in payloads}
        for future in as_completed(future_map):
            payload = future_map[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failed.append(_failure_record(payload, exc))
                if not config.continue_on_chunk_failure:
                    for item in future_map:
                        item.cancel()
                    break
    completed.sort(key=lambda item: int(item["chunk_index"]))
    failed.sort(key=lambda item: int(item["chunk_index"]))
    return completed, failed


def _worker_run_chunk(payload: dict[str, object]) -> dict[str, object]:
    root = Path(str(payload["root"]))
    chunk_index = int(payload["chunk_index"])
    chunk_count = int(payload["chunk_count"])
    fan_branch = str(payload["fan_branch"])
    storage_format = str(payload["storage_format"])
    compression_level = int(payload["compression_level"])
    candidates = payload["candidate_rows"]
    samples = payload["sample_rows"]
    if not isinstance(candidates, pd.DataFrame) or not isinstance(samples, pd.DataFrame):
        raise TypeError("chunk payload candidate_rows and sample_rows must be DataFrames")
    candidates = _with_chunk_columns(candidates, chunk_index=chunk_index, chunk_count=chunk_count)
    samples = _with_chunk_columns(samples, chunk_index=chunk_index, chunk_count=chunk_count)
    rollout_rows = run_fixed_gate_primitive_rollouts(
        candidates,
        FixedGatePrimitiveRolloutConfig(
            latency_case=str(payload["latency_case"]),
            random_seed=int(payload["seed"]) + chunk_index,
            controller_mode=str(payload["controller_mode"]),
            feedback_mode="instant_state_feedback" if str(payload["controller_mode"]) in {"both", "feedback_stabilised_primitive"} else "open_loop",
            allow_open_loop_diagnostic=True,
        ),
    )
    rollout_rows = _with_chunk_columns(rollout_rows, chunk_index=chunk_index, chunk_count=chunk_count)
    pairing_audit = _with_chunk_columns(
        build_w0_w1_pairing_audit(candidates, rollout_rows),
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    outcome_summary = _with_chunk_columns(
        build_rollout_outcome_summary(rollout_rows),
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    move_on_gates = build_archive_move_on_gates(rollout_rows)
    branch_coverage = _with_chunk_columns(
        _branch_coverage_summary(rollout_rows, move_on_gates),
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    partitions = [
        _write_chunk_partition(root, "fixed_gate_samples", samples, fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "candidate_index", candidates, fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "primitive_rollout_rows", rollout_rows, fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "diagnostic_rows", _role_rows(rollout_rows, {"ablation_diagnostic", "boundary_diagnostic"}), fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "partial_feedback_rows", _role_rows(rollout_rows, {"partial_feedback", "blocked_partial"}), fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "mission_candidate_rows", _role_rows(rollout_rows, {"mission_candidate"}), fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "chunk_branch_coverage", branch_coverage, fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "pairing_audit", pairing_audit, fan_branch, chunk_index, storage_format, compression_level),
        _write_chunk_partition(root, "rollout_outcome_summary", outcome_summary, fan_branch, chunk_index, storage_format, compression_level),
    ]
    summary = _chunk_count_summary(rollout_rows, move_on_gates)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "fan_branch": fan_branch,
        "input_candidate_row_count": int(len(candidates)),
        "output_rollout_row_count": int(len(rollout_rows)),
        "summary": summary,
        "table_partitions": [_partition_payload(partition) for partition in partitions],
    }
    manifest_path = _chunk_manifest_path(root, chunk_index)
    filesystem_path(manifest_path.parent).mkdir(parents=True, exist_ok=True)
    _write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return {
        "chunk_index": chunk_index,
        "fan_branch": fan_branch,
        "status": "complete",
        "manifest_json": str(manifest_path),
        "input_candidate_row_count": int(len(candidates)),
        "output_rollout_row_count": int(len(rollout_rows)),
    }


def _chunk_payload(config: FixedGateChunkedRunConfig, root: Path, chunk: FixedGateChunkSpec) -> dict[str, object]:
    return {
        "root": str(root),
        "chunk_index": int(chunk.chunk_index),
        "chunk_count": int(chunk.chunk_count),
        "fan_branch": str(chunk.fan_branch),
        "candidate_rows": chunk.candidate_rows,
        "sample_rows": chunk.sample_rows,
        "seed": int(config.seed),
        "latency_case": str(config.latency_case),
        "controller_mode": str(config.controller_mode),
        "storage_format": str(config.storage_format),
        "compression_level": int(config.compression_level),
    }


def _failure_record(payload: dict[str, object], exc: Exception) -> dict[str, object]:
    return {
        "chunk_index": int(payload["chunk_index"]),
        "fan_branch": str(payload["fan_branch"]),
        "status": "failed",
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
    }


def _write_chunk_partition(
    root: Path,
    table_name: str,
    frame: pd.DataFrame,
    fan_branch: str,
    chunk_index: int,
    storage_format: str,
    compression_level: int,
) -> TablePartition:
    extension = table_extension(storage_format)
    path = (
        root
        / "tables"
        / table_name
        / f"fan_branch={fan_branch}"
        / f"archive_chunk_index={int(chunk_index):05d}"
        / f"part-00000.{extension}"
    )
    return write_table_partition(
        frame,
        path,
        storage_format=storage_format,
        compression_level=int(compression_level),
    )


def _with_chunk_columns(frame: pd.DataFrame, *, chunk_index: int, chunk_count: int) -> pd.DataFrame:
    output = frame.copy()
    output["archive_chunk_index"] = int(chunk_index)
    output["archive_chunk_count"] = int(chunk_count)
    return output


def _chunk_status(root: Path, chunk_index: int) -> str:
    manifest_path = _chunk_manifest_path(root, chunk_index)
    if not filesystem_path(manifest_path).exists():
        return "pending"
    try:
        payload = json.loads(filesystem_path(manifest_path).read_text(encoding="ascii"))
        if payload.get("status") != "complete":
            return "corrupt"
        for partition in payload.get("table_partitions", []):
            path = root / "tables" / str(partition["relative_path"])
            if not filesystem_path(path).exists() or file_sha256(path) != str(partition["checksum_sha256"]):
                return "corrupt"
        return "complete"
    except (OSError, json.JSONDecodeError, KeyError):
        return "corrupt"


def _chunk_manifest_path(root: Path, chunk_index: int) -> Path:
    return root / "chunk_manifests" / f"chunk_{int(chunk_index):05d}.json"


def _remove_chunk_outputs(root: Path, chunk_index: int) -> None:
    manifest_path = _chunk_manifest_path(root, chunk_index)
    if filesystem_path(manifest_path).exists():
        try:
            payload = json.loads(filesystem_path(manifest_path).read_text(encoding="ascii"))
            for partition in payload.get("table_partitions", []):
                path = root / "tables" / str(partition["relative_path"])
                if filesystem_path(path).exists():
                    filesystem_path(path).unlink()
        finally:
            filesystem_path(manifest_path).unlink(missing_ok=True)


def _write_final_outputs(
    *,
    paths: dict[str, Path],
    config: FixedGateChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    execution_context: dict[str, object],
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    schedule: list[FixedGateChunkSpec],
    completed: list[dict[str, object]],
    skipped: list[FixedGateChunkSpec],
    failed: list[dict[str, object]],
    corrupt: list[FixedGateChunkSpec],
) -> None:
    root = paths["root"]
    for name in ("metrics", "manifests", "reports"):
        filesystem_path(paths[name]).mkdir(parents=True, exist_ok=True)
    partitions = _complete_partitions(root)
    table_manifest_path = paths["manifests"] / "table_manifest.json"
    write_table_manifest(
        table_manifest_path,
        TableManifest(
            run_id=int(config.run_id),
            root=root.as_posix(),
            storage_format=resolve_storage_format(config.storage_format),
            tables=tuple(partitions),
        ),
    )
    chunk_summaries = _complete_chunk_summaries(root)
    move_on_gates = _aggregate_move_on_gates(chunk_summaries)
    branch_coverage = _read_partition_table(root, "chunk_branch_coverage")
    outcome_summary = _aggregate_outcome_summary(_read_partition_table(root, "rollout_outcome_summary"))
    pairing_summary = _pairing_summary(_read_partition_table(root, "pairing_audit"))
    move_on_gates.update(_pairing_move_on_fields(pairing_summary))
    accepted_partial_counts = _aggregate_accepted_partial_counts(branch_coverage)
    code_path_map = code_path_map_frame()

    branch_coverage.to_csv(filesystem_path(paths["metrics"] / "fixed_gate_w0_w1_branch_coverage_summary.csv"), index=False)
    outcome_summary.to_csv(filesystem_path(paths["metrics"] / "fixed_gate_w0_w1_outcome_summary.csv"), index=False)
    pairing_summary.to_csv(filesystem_path(paths["metrics"] / "fixed_gate_w0_w1_pairing_audit_summary.csv"), index=False)
    code_path_map.to_csv(filesystem_path(paths["metrics"] / "active_deprecated_code_path_map.csv"), index=False)
    runtime_summary = _runtime_summary(config, worker_decision, schedule, completed, skipped, failed, corrupt)
    runtime_summary.to_csv(filesystem_path(paths["metrics"] / "runtime_summary.csv"), index=False)
    _write_text(
        paths["reports"] / "fixed_gate_w0_w1_branch_coverage_report.md",
        _branch_coverage_report(branch_coverage, move_on_gates),
    )
    manifest = _final_manifest(
        config=config,
        worker_decision=worker_decision,
        execution_context=execution_context,
        samples=samples,
        candidates=candidates,
        schedule=schedule,
        completed=completed,
        skipped=skipped,
        failed=failed,
        corrupt=corrupt,
        move_on_gates=move_on_gates,
        accepted_partial_counts=accepted_partial_counts,
        paths=paths,
    )
    _write_text(
        paths["manifests"] / "fixed_gate_w0_w1_chunked_archive_manifest.json",
        json.dumps(manifest, indent=2) + "\n",
    )
    _write_text(
        paths["reports"] / "fixed_gate_w0_w1_chunked_archive_report.md",
        _final_report(manifest),
    )
    _write_execution_note(paths, manifest)


def _complete_partitions(root: Path) -> list[TablePartition]:
    partitions: list[TablePartition] = []
    for manifest_path in sorted(filesystem_path(root / "chunk_manifests").glob("chunk_*.json")):
        payload = json.loads(manifest_path.read_text(encoding="ascii"))
        if payload.get("status") != "complete":
            continue
        for row in payload.get("table_partitions", []):
            partitions.append(
                TablePartition(
                    table_name=str(row["table_name"]),
                    relative_path=str(row["relative_path"]),
                    storage_format=str(row["storage_format"]),
                    row_count=int(row["row_count"]),
                    byte_count=int(row["byte_count"]),
                    columns=tuple(str(column) for column in row["columns"]),
                    checksum_sha256=str(row["checksum_sha256"]),
                )
            )
    return partitions


def _complete_chunk_summaries(root: Path) -> list[dict[str, object]]:
    rows = []
    for manifest_path in sorted(filesystem_path(root / "chunk_manifests").glob("chunk_*.json")):
        payload = json.loads(manifest_path.read_text(encoding="ascii"))
        if payload.get("status") == "complete":
            rows.append(dict(payload.get("summary", {})))
    return rows


def _read_partition_table(root: Path, table_name: str) -> pd.DataFrame:
    frames = []
    for partition in _complete_partitions(root):
        if partition.table_name != table_name:
            continue
        frames.append(read_table_partition(root / "tables" / partition.relative_path, storage_format=partition.storage_format))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _chunk_count_summary(rollout_rows: pd.DataFrame, move_on_gates: dict[str, object]) -> dict[str, object]:
    accepted_mission_or_partial = rollout_rows[
        rollout_rows["accepted"].astype(bool)
        & rollout_rows["evidence_role"].astype(str).isin({"mission_candidate", "partial_feedback"})
    ].copy() if not rollout_rows.empty else pd.DataFrame()
    accepted_by_branch_layer = (
        accepted_mission_or_partial.groupby(["fan_branch", "W_layer"]).size().to_dict()
        if not accepted_mission_or_partial.empty
        else {}
    )
    return {
        "candidate_row_count": int(len(rollout_rows.drop_duplicates(subset=["sample_id", "W_layer", "primitive_family"], keep="first"))) if not rollout_rows.empty else 0,
        "rollout_row_count": int(len(rollout_rows)),
        "w0_row_count_by_branch": move_on_gates["w0_row_count_by_branch"],
        "w1_row_count_by_branch": move_on_gates["w1_row_count_by_branch"],
        "w1_measured_updraft_row_count_by_branch": move_on_gates["w1_measured_updraft_row_count_by_branch"],
        "mission_candidate_row_count": int(move_on_gates["mission_candidate_row_count"]),
        "partial_feedback_row_count": int(move_on_gates["partial_feedback_row_count"]),
        "accepted_w0_partial_feedback_row_count": int(move_on_gates["accepted_w0_partial_feedback_row_count"]),
        "accepted_w1_partial_feedback_row_count": int(move_on_gates["accepted_w1_partial_feedback_row_count"]),
        "blocked_partial_row_count": int(move_on_gates["blocked_partial_row_count"]),
        "ablation_diagnostic_row_count": int(move_on_gates["ablation_diagnostic_row_count"]),
        "accepted_mission_or_partial_by_branch_layer": {
            f"{key[0]}|{key[1]}": int(value)
            for key, value in accepted_by_branch_layer.items()
        },
        "code_ready": str(move_on_gates["code_ready"]),
    }


def _aggregate_move_on_gates(chunk_summaries: list[dict[str, object]]) -> dict[str, object]:
    w0_by_branch = _sum_branch_dict(chunk_summaries, "w0_row_count_by_branch")
    w1_by_branch = _sum_branch_dict(chunk_summaries, "w1_row_count_by_branch")
    w1_measured_by_branch = _sum_branch_dict(chunk_summaries, "w1_measured_updraft_row_count_by_branch")
    accepted_by_branch_layer = _sum_flat_dict(chunk_summaries, "accepted_mission_or_partial_by_branch_layer")
    branches = set(BRANCHES)
    paired = all(w0_by_branch.get(branch, 0) > 0 and w1_by_branch.get(branch, 0) > 0 for branch in branches)
    measured = all(w1_measured_by_branch.get(branch, 0) > 0 for branch in branches)
    mission_ready = all(accepted_by_branch_layer.get(f"{branch}|{layer}", 0) > 0 for branch in branches for layer in ("W0", "W1"))
    code_ready = all(str(row.get("code_ready")) == "ready" for row in chunk_summaries) if chunk_summaries else False
    return {
        "code_ready": "ready" if code_ready else "blocked_schema_or_promoted_diagnostic_rows",
        "archive_prepared": "ready" if paired and measured else "blocked_missing_branch_layer_rows_or_w1_measured_updraft_rows",
        "mission_evidence_ready": "ready" if mission_ready else "blocked_no_mission_or_partial_feedback_rows_for_both_branches",
        "w0_row_count_by_branch": w0_by_branch,
        "w1_row_count_by_branch": w1_by_branch,
        "w1_measured_updraft_row_count": int(sum(w1_measured_by_branch.values())),
        "w1_measured_updraft_row_count_by_branch": w1_measured_by_branch,
        "mission_candidate_row_count": _sum_scalar(chunk_summaries, "mission_candidate_row_count"),
        "partial_feedback_row_count": _sum_scalar(chunk_summaries, "partial_feedback_row_count"),
        "accepted_w0_partial_feedback_row_count": _sum_scalar(chunk_summaries, "accepted_w0_partial_feedback_row_count"),
        "accepted_w1_partial_feedback_row_count": _sum_scalar(chunk_summaries, "accepted_w1_partial_feedback_row_count"),
        "blocked_partial_row_count": _sum_scalar(chunk_summaries, "blocked_partial_row_count"),
        "ablation_diagnostic_row_count": _sum_scalar(chunk_summaries, "ablation_diagnostic_row_count"),
    }


def _pairing_move_on_fields(pairing_summary: pd.DataFrame) -> dict[str, object]:
    if pairing_summary.empty:
        return {
            "w1_scheduled_independently_of_w0_success": False,
            "pairing_group_count": 0,
        }
    row = pairing_summary.iloc[0]
    return {
        "w1_scheduled_independently_of_w0_success": bool(row.get("w1_independent", False)),
        "pairing_group_count": int(row.get("pairing_group_count", 0)),
    }


def _sum_branch_dict(chunk_summaries: list[dict[str, object]], field: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for summary in chunk_summaries:
        for key, value in dict(summary.get(field, {})).items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def _sum_flat_dict(chunk_summaries: list[dict[str, object]], field: str) -> dict[str, int]:
    return _sum_branch_dict(chunk_summaries, field)


def _sum_scalar(chunk_summaries: list[dict[str, object]], field: str) -> int:
    return int(sum(int(summary.get(field, 0)) for summary in chunk_summaries))


def _aggregate_outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["fan_branch", "W_layer", "evidence_role", "outcome_class", "row_count"])
    return (
        frame.groupby(["fan_branch", "W_layer", "evidence_role", "outcome_class"], dropna=False)["row_count"]
        .sum()
        .reset_index()
    )


def _aggregate_accepted_partial_counts(branch_coverage: pd.DataFrame) -> list[dict[str, object]]:
    if branch_coverage.empty:
        return []
    required = {"summary_section", "fan_branch", "W_layer", "primitive_family", "row_count"}
    if not required.issubset(branch_coverage.columns):
        return []
    rows = branch_coverage[
        branch_coverage["summary_section"].astype(str).eq("accepted_partial_feedback_by_branch_layer_primitive")
    ].copy()
    if rows.empty:
        return []
    grouped = (
        rows.groupby(["fan_branch", "W_layer", "primitive_family"], dropna=False)["row_count"]
        .sum()
        .reset_index()
    )
    return [
        {
            "fan_branch": str(record["fan_branch"]),
            "W_layer": str(record["W_layer"]),
            "primitive_family": str(record["primitive_family"]),
            "accepted_partial_feedback_row_count": int(record["row_count"]),
        }
        for record in grouped.to_dict(orient="records")
    ]


def _branch_coverage_conclusion(move_on_gates: dict[str, object]) -> str:
    if str(move_on_gates.get("archive_prepared")) != "ready":
        return "blocked_archive_not_prepared"
    if str(move_on_gates.get("mission_evidence_ready")) != "ready":
        return "archive_prepared_but_mission_evidence_blocked"
    return "ready_for_downstream_non_diagnostic_evidence"


def _pairing_summary(pairing_audit: pd.DataFrame) -> pd.DataFrame:
    if pairing_audit.empty:
        return pd.DataFrame(
            [{"pairing_group_count": 0, "all_have_W0": False, "all_have_W1": False, "w1_independent": False}]
        )
    return pd.DataFrame(
        [
            {
                "pairing_group_count": int(len(pairing_audit)),
                "all_have_W0": bool(pairing_audit["has_W0"].astype(bool).all()),
                "all_have_W1": bool(pairing_audit["has_W1"].astype(bool).all()),
                "w1_independent": bool(pairing_audit["w1_scheduled_independent_of_w0_success"].astype(bool).all()),
            }
        ]
    )


def _runtime_summary(
    config: FixedGateChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    schedule: list[FixedGateChunkSpec],
    completed: list[dict[str, object]],
    skipped: list[FixedGateChunkSpec],
    failed: list[dict[str, object]],
    corrupt: list[FixedGateChunkSpec],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": int(config.run_id),
                "rows_per_branch": int(config.rows_per_branch),
                "candidate_chunk_size": int(config.candidate_chunk_size),
                "chunk_count": int(len(schedule)),
                "completed_chunk_count": int(len(completed)),
                "skipped_chunk_count": int(len(skipped)),
                "failed_chunk_count": int(len(failed)),
                "corrupt_chunk_count": int(len(corrupt)),
                "selected_worker_count": int(worker_decision.selected_worker_count),
                "worker_fallback_reason": worker_decision.fallback_reason,
                "storage_format": resolve_storage_format(config.storage_format),
                "compression_level": int(config.compression_level),
                "full_rollout_metrics_csv_written": False,
            }
        ]
    )


def _write_schedule(paths: dict[str, Path], schedule: list[FixedGateChunkSpec]) -> None:
    rows = [
        {
            "chunk_index": int(chunk.chunk_index),
            "chunk_count": int(chunk.chunk_count),
            "fan_branch": chunk.fan_branch,
            "candidate_row_count": int(len(chunk.candidate_rows)),
            "sample_row_count": int(len(chunk.sample_rows)),
        }
        for chunk in schedule
    ]
    pd.DataFrame(rows).to_csv(filesystem_path(paths["metrics"] / "chunk_schedule.csv"), index=False)


def _write_progress(
    *,
    paths: dict[str, Path],
    config: FixedGateChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    schedule: list[FixedGateChunkSpec],
    pending: list[FixedGateChunkSpec],
    skipped: list[FixedGateChunkSpec],
    failed: list[dict[str, object]],
    corrupt: list[FixedGateChunkSpec],
    status: str,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "run_id": int(config.run_id),
        "campaign": CAMPAIGN,
        "pass_name": CHUNKED_PASS_NAME,
        "chunk_count": int(len(schedule)),
        "pending_chunk_count": int(len(pending)),
        "skipped_chunk_count": int(len(skipped)),
        "failed_chunk_count": int(len(failed)),
        "corrupt_chunk_count": int(len(corrupt)),
        "worker_count_decision": worker_decision.as_manifest_fields(),
        "chunks": [
            {
                "chunk_index": int(chunk.chunk_index),
                "fan_branch": chunk.fan_branch,
                "candidate_row_count": int(len(chunk.candidate_rows)),
                "status": _chunk_status(paths["root"], chunk.chunk_index),
            }
            for chunk in schedule
        ],
        "failures": failed,
    }
    path = paths["manifests"] / "fixed_gate_w0_w1_chunked_progress.json"
    _write_text(path, json.dumps(payload, indent=2) + "\n")
    _write_text(
        paths["reports"] / "fixed_gate_w0_w1_chunked_progress.md",
        "\n".join(
            [
                "# Fixed-Gate W0/W1 Chunked Progress",
                "",
                f"- Status: `{status}`",
                f"- Chunk count: `{len(schedule)}`",
                f"- Pending chunks: `{len(pending)}`",
                f"- Skipped chunks: `{len(skipped)}`",
                f"- Failed chunks: `{len(failed)}`",
                f"- Selected workers: `{worker_decision.selected_worker_count}`",
                "",
            ]
        ),
    )


def _execution_context(config: FixedGateChunkedRunConfig, root: Path) -> dict[str, object]:
    run_scope = _run_scope(config, root)
    official_manifest = (
        RESULT_ROOT
        / f"{OFFICIAL_PRIMARY_RUN_ID:03d}"
        / ARCHIVE_PASS_DIR
        / "manifests"
        / "fixed_gate_w0_w1_chunked_archive_manifest.json"
    )
    preflight_manifest = (
        CONTROL_DIR
        / "05_Results"
        / "11_fixed_gate_repeated_launch_chunked_preflight"
        / f"{PREFLIGHT_RUN_ID:03d}"
        / ARCHIVE_PASS_DIR
        / "manifests"
        / "fixed_gate_w0_w1_chunked_archive_manifest.json"
    )
    official_root = RESULT_ROOT / f"{OFFICIAL_PRIMARY_RUN_ID:03d}"
    return {
        "run_scope": run_scope,
        "execution_command": _current_command(),
        "head_before_run": _git_output(["git", "rev-parse", "--short", "HEAD"]),
        "git_status_short_before_run": _git_status_short(),
        "official_run_root_existed_before_run": filesystem_path(official_root).exists(),
        "official_manifest_status_before_run": _manifest_status(official_manifest),
        "preflight_manifest_status": _manifest_status(preflight_manifest),
        "git_fetch_all": _git_fetch_all_status(run_scope),
        "preflight_root_ignore_status": _git_ignore_status(preflight_manifest),
        "official_compact_manifest_ignore_status": _git_ignore_status(official_manifest),
        "official_tables_ignore_status": _git_ignore_status(
            RESULT_ROOT / f"{OFFICIAL_PRIMARY_RUN_ID:03d}" / ARCHIVE_PASS_DIR / "tables" / "primitive_rollout_rows"
        ),
        "result_root": str(root),
    }


def _run_scope(config: FixedGateChunkedRunConfig, root: Path) -> str:
    if _is_scratch_root(root) or int(config.run_id) == PREFLIGHT_RUN_ID:
        return "preflight_run"
    if config.result_root is None and int(config.run_id) == OFFICIAL_PRIMARY_RUN_ID:
        return "official_archive_run"
    if config.result_root is None and int(config.rows_per_branch) >= 60_000:
        return "fallback_archive_run"
    return "archive_run"


def _current_command() -> str:
    if Path(sys.argv[0]).name:
        return " ".join(str(item) for item in ["python", *sys.argv])
    return "python_api_invocation"


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def _git_fetch_all_status(run_scope: str) -> dict[str, object]:
    if run_scope != "official_archive_run":
        return {
            "attempted": False,
            "succeeded": False,
            "return_code": None,
            "summary": "not_attempted_for_non_official_run",
        }
    try:
        completed = subprocess.run(
            ["git", "fetch", "--all"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "attempted": True,
            "succeeded": False,
            "return_code": None,
            "summary": f"{type(exc).__name__}:{exc}",
        }
    text = (completed.stdout + "\n" + completed.stderr).strip()
    return {
        "attempted": True,
        "succeeded": completed.returncode == 0,
        "return_code": int(completed.returncode),
        "summary": text[-1000:] if text else "",
    }


def _git_ignore_status(path: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-v", str(path)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ignored": False, "rule": "", "error": f"{type(exc).__name__}:{exc}"}
    return {
        "ignored": completed.returncode == 0,
        "rule": completed.stdout.strip(),
        "error": "" if completed.returncode in {0, 1} else completed.stderr.strip(),
    }


def _manifest_status(path: Path) -> dict[str, object]:
    if not filesystem_path(path).exists():
        return {"exists": False, "status": "absent", "path": str(path)}
    try:
        payload = json.loads(filesystem_path(path).read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"exists": True, "status": f"unreadable:{type(exc).__name__}", "path": str(path)}
    return {
        "exists": True,
        "status": str(payload.get("status", payload.get("run_scope", "present"))),
        "archive_prepared": str(payload.get("move_on_gates", {}).get("archive_prepared", "")),
        "mission_evidence_ready": str(payload.get("move_on_gates", {}).get("mission_evidence_ready", "")),
        "chunk_count": payload.get("chunk_count"),
        "completed_chunk_count": payload.get("completed_chunk_count"),
        "path": str(path),
    }


def _write_build_note(
    paths: dict[str, Path],
    config: FixedGateChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    execution_context: dict[str, object],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "fixed_gate_chunked_runtime_build_note",
        "execution_guidance": V11_4_EXECUTION_GUIDANCE,
        "execution_context": execution_context,
        "head_before_run": execution_context["head_before_run"],
        "git_status_short_before_run": execution_context["git_status_short_before_run"],
        "official_run_root_existed_before_run": execution_context["official_run_root_existed_before_run"],
        "preflight_manifest_status": execution_context["preflight_manifest_status"],
        "git_fetch_all": execution_context["git_fetch_all"],
        "config": _config_payload(config),
        "worker_count_decision": worker_decision.as_manifest_fields(),
        "dense_runner_preflight_checklist": {
            "chunked_runtime_reused": "dense_archive_runtime.worker_count_decision plus fixed-gate chunk scheduler",
            "compressed_table_writer_reused": "dense_archive_table_io.write_table_partition/write_table_manifest",
            "worker_count_decision_recorded": "manifests/fixed_gate_w0_w1_chunked_progress.json and final manifest",
            "resume_repair_implemented": "chunk_manifests/chunk_*.json with --resume and --repair-incomplete",
            "partition_checksums_written": "manifests/table_manifest.json and per-chunk manifests",
            "why_new_runner_not_old_wrapper": "the legacy fixed-gate runner is smoke-scale only and writes full in-memory tables",
        },
    }
    path = paths["manifests"] / "fixed_gate_chunked_build_note.json"
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _final_manifest(
    *,
    config: FixedGateChunkedRunConfig,
    worker_decision: WorkerCountDecision,
    execution_context: dict[str, object],
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    schedule: list[FixedGateChunkSpec],
    completed: list[dict[str, object]],
    skipped: list[FixedGateChunkSpec],
    failed: list[dict[str, object]],
    corrupt: list[FixedGateChunkSpec],
    move_on_gates: dict[str, object],
    accepted_partial_counts: list[dict[str, object]],
    paths: dict[str, Path],
) -> dict[str, object]:
    run_scope = str(execution_context["run_scope"])
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(config.run_id),
        "campaign": CAMPAIGN,
        "pass_name": CHUNKED_PASS_NAME,
        "execution_guidance": V11_4_EXECUTION_GUIDANCE,
        "run_scope": run_scope,
        "official_archive_run": run_scope == "official_archive_run",
        "preflight_run": run_scope == "preflight_run",
        "fallback_archive_run": run_scope == "fallback_archive_run",
        "active_mission_path": active_code_path_text(),
        "simple_runner_dense_guard_active": True,
        "official_run_006_launched_by_this_pass": run_scope == "official_archive_run" and int(config.run_id) == OFFICIAL_PRIMARY_RUN_ID,
        "execution_context": execution_context,
        "sample_row_count": int(len(samples)),
        "candidate_row_count": int(len(candidates)),
        "rollout_row_count": _sum_scalar(_complete_chunk_summaries(paths["root"]), "rollout_row_count"),
        "chunk_count": int(len(schedule)),
        "completed_chunk_count": int(len(completed)),
        "skipped_chunk_count": int(len(skipped)),
        "failed_chunk_count": int(len(failed)),
        "corrupt_chunk_count": int(len(corrupt)),
        "rows_per_branch_requested": int(config.rows_per_branch),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "controller_mode": str(config.controller_mode),
        "latency_case": str(config.latency_case),
        "storage_format": resolve_storage_format(config.storage_format),
        "compression_level": int(config.compression_level),
        "worker_count_decision": worker_decision.as_manifest_fields(),
        "gpu_acceleration_assessment": GPU_ACCELERATION_ASSESSMENT,
        "move_on_gates": move_on_gates,
        "accepted_partial_feedback_counts_by_branch_layer_primitive": accepted_partial_counts,
        "branch_coverage_conclusion": _branch_coverage_conclusion(move_on_gates),
        "downstream_execution_status": "not_run_by_archive_runner",
        "full_rollout_metrics_csv_written": False,
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Fixed-gate W0/W1 chunked runtime evidence only; no real-flight transfer, mission success, "
            "same-flight recapture, perching, all-arena validity, hardware-ready agile-turn, or full "
            "delayed-state-feedback claim is made."
        ),
        "reported_commands": _reported_commands(),
        "output_files": {key: str(path) for key, path in _output_paths(paths).items()},
    }
    payload.update(
        runtime_manifest_fields(
            simulation_stage=CHUNKED_PASS_NAME,
            environment_mode="fixed_gate_w0_w1",
            branch_decision_scope="branch_local_only_no_cross_branch_decision_transfer",
            worker_decision=worker_decision,
        )
    )
    return payload


def _final_report(manifest: dict[str, object]) -> str:
    gates = manifest["move_on_gates"]
    return "\n".join(
        [
            "# Fixed-Gate W0/W1 Chunked Archive",
            "",
            f"- Run scope: `{manifest['run_scope']}`",
            f"- Official run 006 launched by this pass: `{manifest['official_run_006_launched_by_this_pass']}`",
            f"- Candidate rows: `{manifest['candidate_row_count']}`",
            f"- Rollout rows: `{manifest['rollout_row_count']}`",
            f"- Chunk count: `{manifest['chunk_count']}`",
            f"- Completed chunks: `{manifest['completed_chunk_count']}`",
            f"- Failed chunks: `{manifest['failed_chunk_count']}`",
            f"- Selected workers: `{manifest['worker_count_decision']['selected_worker_count']}`",
            f"- Storage format: `{manifest['storage_format']}`",
            f"- Full rollout metrics CSV written: `{manifest['full_rollout_metrics_csv_written']}`",
            f"- Archive-prepared status: `{gates['archive_prepared']}`",
            f"- Mission-evidence-ready status: `{gates['mission_evidence_ready']}`",
            f"- Branch coverage conclusion: `{manifest['branch_coverage_conclusion']}`",
            "",
            "Use the reported commands in the manifest for the official 120k run, repair retry, or 60k fallback.",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, hardware-ready agile-turn, or full delayed-state-feedback claim is made.",
            "",
        ]
    )


def _write_execution_note(paths: dict[str, Path], manifest: dict[str, object]) -> None:
    note = _execution_note_payload(manifest)
    _write_text(
        paths["manifests"] / "fixed_gate_w0_w1_v11_4_execution_note.json",
        json.dumps(note, indent=2) + "\n",
    )
    _write_text(
        paths["reports"] / "fixed_gate_w0_w1_v11_4_execution_note.md",
        _execution_note_report(note),
    )


def _execution_note_payload(manifest: dict[str, object]) -> dict[str, object]:
    gates = manifest["move_on_gates"]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execution_guidance": V11_4_EXECUTION_GUIDANCE,
        "command_executed": manifest["execution_context"]["execution_command"],
        "run_id": manifest["run_id"],
        "run_scope": manifest["run_scope"],
        "worker_count_selected": manifest["worker_count_decision"]["selected_worker_count"],
        "candidate_chunk_size": manifest["candidate_chunk_size"],
        "storage_format": manifest["storage_format"],
        "compression_level": manifest["compression_level"],
        "completed_chunk_count": manifest["completed_chunk_count"],
        "skipped_chunk_count": manifest["skipped_chunk_count"],
        "failed_chunk_count": manifest["failed_chunk_count"],
        "corrupt_chunk_count": manifest["corrupt_chunk_count"],
        "candidate_row_count": manifest["candidate_row_count"],
        "rollout_row_count": manifest["rollout_row_count"],
        "w0_row_count_by_branch": gates["w0_row_count_by_branch"],
        "w1_row_count_by_branch": gates["w1_row_count_by_branch"],
        "w1_measured_updraft_row_count_by_branch": gates["w1_measured_updraft_row_count_by_branch"],
        "w1_scheduled_independently_of_w0_success": gates.get("w1_scheduled_independently_of_w0_success", False),
        "pairing_group_count": gates.get("pairing_group_count", 0),
        "accepted_partial_feedback_counts_by_branch_layer_primitive": manifest[
            "accepted_partial_feedback_counts_by_branch_layer_primitive"
        ],
        "branch_coverage_conclusion": manifest["branch_coverage_conclusion"],
        "downstream_execution_status": manifest["downstream_execution_status"],
        "git_fetch_all": manifest["execution_context"]["git_fetch_all"],
        "next_commands": manifest["reported_commands"],
        "claim_status": manifest["claim_status"],
        "claim_boundary": manifest["claim_boundary"],
    }


def _execution_note_report(note: dict[str, object]) -> str:
    return "\n".join(
        [
            "# v11.4 Fixed-Gate W0/W1 Execution Note",
            "",
            f"- Run ID: `{note['run_id']}`",
            f"- Run scope: `{note['run_scope']}`",
            f"- Worker count selected: `{note['worker_count_selected']}`",
            f"- Chunk size: `{note['candidate_chunk_size']}`",
            f"- Storage/compression: `{note['storage_format']}`, level `{note['compression_level']}`",
            f"- Chunks completed/skipped/failed/corrupt: `{note['completed_chunk_count']}` / `{note['skipped_chunk_count']}` / `{note['failed_chunk_count']}` / `{note['corrupt_chunk_count']}`",
            f"- Candidate rows: `{note['candidate_row_count']}`",
            f"- Rollout rows: `{note['rollout_row_count']}`",
            f"- W0 rows by branch: `{note['w0_row_count_by_branch']}`",
            f"- W1 rows by branch: `{note['w1_row_count_by_branch']}`",
            f"- W1 measured rows by branch: `{note['w1_measured_updraft_row_count_by_branch']}`",
            f"- W1 scheduled independently of W0 success: `{note['w1_scheduled_independently_of_w0_success']}`",
            f"- Branch coverage conclusion: `{note['branch_coverage_conclusion']}`",
            f"- Downstream status: `{note['downstream_execution_status']}`",
            "",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, hardware-ready agile-turn, true delayed-state-feedback, full W2/W3 robustness, or real repeated-launch validation claim is made.",
            "",
        ]
    )


def _output_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "root": paths["root"],
        "table_manifest_json": paths["manifests"] / "table_manifest.json",
        "build_note_json": paths["manifests"] / "fixed_gate_chunked_build_note.json",
        "progress_manifest_json": paths["manifests"] / "fixed_gate_w0_w1_chunked_progress.json",
        "manifest_json": paths["manifests"] / "fixed_gate_w0_w1_chunked_archive_manifest.json",
        "execution_note_json": paths["manifests"] / "fixed_gate_w0_w1_v11_4_execution_note.json",
        "execution_note_md": paths["reports"] / "fixed_gate_w0_w1_v11_4_execution_note.md",
        "report_md": paths["reports"] / "fixed_gate_w0_w1_chunked_archive_report.md",
        "branch_coverage_summary_csv": paths["metrics"] / "fixed_gate_w0_w1_branch_coverage_summary.csv",
        "runtime_summary_csv": paths["metrics"] / "runtime_summary.csv",
    }


def _partition_payload(partition: TablePartition) -> dict[str, object]:
    return {
        "table_name": partition.table_name,
        "relative_path": partition.relative_path,
        "storage_format": partition.storage_format,
        "row_count": int(partition.row_count),
        "byte_count": int(partition.byte_count),
        "columns": list(partition.columns),
        "checksum_sha256": partition.checksum_sha256,
    }


def _config_payload(config: FixedGateChunkedRunConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["reachable_source_csv"] = "" if config.reachable_source_csv is None else str(config.reachable_source_csv)
    payload["result_root"] = "" if config.result_root is None else str(config.result_root)
    return payload


def _git_status_short() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _write_text(path: Path, text: str) -> None:
    filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(text, encoding="ascii")


def _reported_commands() -> dict[str, str]:
    return {
        "official_120k_command": (
            "python 03_Control/04_Scenarios/run_fixed_gate_w0_w1_archive_chunked.py --run-id 006 "
            "--rows-per-branch 120000 --seed 20260522 --fan-branch all --w-layers W0,W1 "
            "--latency-case nominal --controller-mode both --candidate-chunk-size 2500 --workers 8 "
            "--max-workers 8 --memory-safety-margin-gb 8 --storage-format csv_gz --compression-level 1 --resume"
        ),
        "repair_retry_command": (
            "python 03_Control/04_Scenarios/run_fixed_gate_w0_w1_archive_chunked.py --run-id 006 "
            "--rows-per-branch 120000 --seed 20260522 --fan-branch all --w-layers W0,W1 "
            "--latency-case nominal --controller-mode both --candidate-chunk-size 1000 --workers 6 "
            "--max-workers 8 --memory-safety-margin-gb 8 --storage-format csv_gz --compression-level 1 "
            "--resume --repair-incomplete"
        ),
        "fallback_60k_command": (
            "python 03_Control/04_Scenarios/run_fixed_gate_w0_w1_archive_chunked.py --run-id 007 "
            "--rows-per-branch 60000 --seed 20260522 --fan-branch all --w-layers W0,W1 "
            "--latency-case nominal --controller-mode both --candidate-chunk-size 2500 --workers 8 "
            "--max-workers 8 --memory-safety-margin-gb 8 --storage-format csv_gz --compression-level 1 --resume"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--rows-per-branch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--fan-branch", choices=("single_fan_branch", "four_fan_branch", "all"), default="all")
    parser.add_argument("--w-layers", default="W0,W1")
    parser.add_argument("--latency-case", choices=("none", "actuator_lag_only", "nominal", "conservative"), default="nominal")
    parser.add_argument("--controller-mode", choices=("open_loop_rollout", "feedback_stabilised_primitive", "both"), default="both")
    parser.add_argument("--candidate-chunk-size", type=int, default=2500)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--memory-safety-margin-gb", type=float, default=DEFAULT_MEMORY_SAFETY_MARGIN_GB)
    parser.add_argument("--storage-format", default="csv_gz")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--reachable-source-csv", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_fixed_gate_w0_w1_archive_chunked(
        run_id=args.run_id,
        rows_per_branch=args.rows_per_branch,
        seed=args.seed,
        fan_branch=args.fan_branch,
        w_layers=args.w_layers,
        latency_case=args.latency_case,
        controller_mode=args.controller_mode,
        candidate_chunk_size=args.candidate_chunk_size,
        workers=args.workers,
        max_workers=args.max_workers,
        memory_safety_margin_gb=args.memory_safety_margin_gb,
        storage_format=args.storage_format,
        compression_level=args.compression_level,
        reachable_source_csv=args.reachable_source_csv,
        result_root=args.result_root,
        resume=args.resume,
        repair_incomplete=args.repair_incomplete,
        dry_run_schedule=args.dry_run_schedule,
        stop_after_chunks=args.stop_after_chunks,
        continue_on_chunk_failure=args.continue_on_chunk_failure,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
