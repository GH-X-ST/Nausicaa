from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_chunking import (  # noqa: E402
    ContextChunkSpec,
    chunk_status,
    contextual_table_paths,
    remove_chunk_outputs,
    validate_chunk_manifest,
)
from dense_archive_runtime import (  # noqa: E402
    MAX_GENERATED_FILE_SIZE_MB,
    dense_run_manifest_fields,
    worker_count_decision,
)
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    file_sha256,
    filesystem_path,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import EnvironmentMetadata, build_environment_context  # noqa: E402
from prim_cat import active_primitive_catalogue  # noqa: E402
from prim_roll import RolloutConfig, rollout_with_context_row, simulate_primitive_rollout  # noqa: E402
from updraft_models import SINGLE_FAN_CENTER_XY, load_updraft_model  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Archive config and CLI
# 2) Schedule and row generation
# 3) Chunk execution and output manifests
# =============================================================================


# =============================================================================
# 1) Archive Config and CLI
# =============================================================================
@dataclass(frozen=True)
class ContextArchiveConfig:
    run_id: int
    rows: int
    seed: int
    w_layers: tuple[str, ...]
    env_modes: tuple[str, ...]
    candidate_chunk_size: int
    workers: str | int
    max_workers: int
    storage_format: str
    compression_level: int
    resume: bool
    repair_incomplete: bool
    dry_run_schedule: bool
    stop_after_chunks: int | None
    continue_on_chunk_failure: bool
    output_root: Path


def parse_args(argv: list[str] | None = None) -> ContextArchiveConfig:
    parser = argparse.ArgumentParser(
        description="Run a temp/local contextual primitive archive preflight."
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--w-layers", default="W0,W1")
    parser.add_argument("--env-modes", default="dry_air,measured_updraft")
    parser.add_argument("--candidate-chunk-size", type=int, default=125)
    parser.add_argument("--workers", default="8")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    return ContextArchiveConfig(
        run_id=int(args.run_id),
        rows=int(args.rows),
        seed=int(args.seed),
        w_layers=_split_csv(args.w_layers),
        env_modes=_split_csv(args.env_modes),
        candidate_chunk_size=int(args.candidate_chunk_size),
        workers=args.workers,
        max_workers=int(args.max_workers),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
        resume=bool(args.resume),
        repair_incomplete=bool(args.repair_incomplete),
        dry_run_schedule=bool(args.dry_run_schedule),
        stop_after_chunks=args.stop_after_chunks,
        continue_on_chunk_failure=bool(args.continue_on_chunk_failure),
        output_root=Path(args.output_root),
    )


def _split_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must contain at least one value.")
    return values


# =============================================================================
# 2) Schedule and Row Generation
# =============================================================================
def run_contextual_archive_preflight(config: ContextArchiveConfig) -> dict[str, object]:
    """Run a chunked contextual archive preflight under the provided output root."""

    _validate_config(config)
    run_root = Path(config.output_root) / f"run_{config.run_id:03d}"
    filesystem_path(run_root).mkdir(parents=True, exist_ok=True)
    storage_format = resolve_storage_format(config.storage_format)
    worker_decision = worker_count_decision(
        config.workers,
        max_workers=config.max_workers,
    )
    schedule = _build_schedule(config, run_root=run_root, storage_format=storage_format)
    if config.dry_run_schedule:
        return _write_run_outputs(
            config=config,
            run_root=run_root,
            worker_fields=dense_run_manifest_fields(
                run_stage="contextual_archive_preflight",
                environment_context="schedule_only",
                worker_decision=worker_decision,
            ),
            partitions=[],
            outcome_rows=[],
            failures=[],
            dry_run=True,
        )

    partitions: list[TablePartition] = []
    outcome_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    chunks_to_run = schedule
    if config.stop_after_chunks is not None:
        chunks_to_run = chunks_to_run[: int(config.stop_after_chunks)]
    for spec in chunks_to_run:
        started = time.perf_counter()
        try:
            status = chunk_status(spec, run_root=run_root)
            if status == "complete" and config.resume:
                paths = contextual_table_paths(spec, run_root=run_root)
                manifest = validate_chunk_manifest(spec, run_root=run_root)
                partitions.append(
                    TablePartition(
                        table_name="contextual_rows",
                        relative_path=paths.partition_path.relative_to(
                            run_root / "tables"
                        ).as_posix(),
                        storage_format=storage_format,
                        row_count=int(manifest["row_count"]),
                        byte_count=int(filesystem_path(paths.partition_path).stat().st_size),
                        columns=tuple(str(value) for value in manifest["columns"]),
                        checksum_sha256=str(manifest["checksum_sha256"]),
                    )
                )
                continue
            if status == "corrupt" and config.repair_incomplete:
                remove_chunk_outputs(spec, run_root=run_root)
            rows = _chunk_rows(config=config, spec=spec)
            frame = pd.DataFrame(rows)
            paths = contextual_table_paths(spec, run_root=run_root)
            partition = write_table_partition(
                frame,
                paths.partition_path,
                storage_format=storage_format,
                compression_level=config.compression_level,
            )
            elapsed = time.perf_counter() - started
            _write_chunk_manifest(
                spec=spec,
                paths=paths,
                partition=partition,
                columns=tuple(str(column) for column in frame.columns),
                elapsed_s=elapsed,
            )
            partitions.append(partition)
            outcome_rows.extend(
                {
                    "W_layer": row["W_layer"],
                    "environment_id": row["environment_id"],
                    "outcome_class": row["outcome_class"],
                    "accepted": bool(row["accepted"]),
                }
                for row in rows
            )
        except Exception as exc:
            failures.append(
                {
                    "chunk_index": int(spec.chunk_index),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not config.continue_on_chunk_failure:
                raise
    return _write_run_outputs(
        config=config,
        run_root=run_root,
        worker_fields=dense_run_manifest_fields(
            run_stage="contextual_archive_preflight",
            environment_context="W0_W1_context_smoke",
            worker_decision=worker_decision,
        ),
        partitions=partitions,
        outcome_rows=outcome_rows,
        failures=failures,
        dry_run=False,
    )


def _validate_config(config: ContextArchiveConfig) -> None:
    if config.rows <= 0:
        raise ValueError("rows must be positive.")
    if config.rows > 10_000:
        raise ValueError("R2-R5 preflight is capped at 10k rows.")
    if config.candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    if not {"W0", "W1"}.issubset(set(config.w_layers)):
        raise ValueError("R2-R5 preflight requires W0 and W1.")
    unknown = set(config.w_layers) - {"W0", "W1", "W2", "W3"}
    if unknown:
        raise ValueError(f"unknown W layer labels: {sorted(unknown)}")


def _build_schedule(
    config: ContextArchiveConfig,
    *,
    run_root: Path,
    storage_format: str,
) -> list[ContextChunkSpec]:
    chunk_count = int(np.ceil(config.rows / config.candidate_chunk_size))
    schedule = []
    for chunk_index in range(chunk_count):
        w_layer = config.w_layers[chunk_index % len(config.w_layers)]
        env_mode = config.env_modes[chunk_index % len(config.env_modes)]
        schedule.append(
            ContextChunkSpec(
                run_id=config.run_id,
                source_run_id=0,
                result_root=run_root,
                context_id=w_layer,
                environment_id=env_mode,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                chunk_size=min(
                    config.candidate_chunk_size,
                    config.rows - chunk_index * config.candidate_chunk_size,
                ),
                storage_format=storage_format,
                compression_level=config.compression_level,
                latency_case="nominal",
                dt_s=0.02,
                horizon_s=0.80,
                run_stage="contextual_archive_preflight",
            )
        )
    return schedule


def _chunk_rows(
    *,
    config: ContextArchiveConfig,
    spec: ContextChunkSpec,
) -> list[dict[str, object]]:
    primitives = active_primitive_catalogue()
    wind = None if spec.context_id == "W0" else load_updraft_model("single_gaussian_var")
    rows: list[dict[str, object]] = []
    for offset in range(spec.chunk_size):
        row_index = int(spec.chunk_index * config.candidate_chunk_size + offset)
        state = _state_for_row(row_index)
        metadata = _metadata_for_row(
            w_layer=spec.context_id,
            environment_mode=spec.environment_id,
            seed=config.seed + row_index,
            wind_name="dry_air_zero_wind" if wind is None else wind.name,
            wind_source="not_applicable" if wind is None else wind.source,
        )
        context = build_environment_context(
            state,
            wind_field=wind,
            metadata=metadata,
            latency_case="none" if spec.context_id == "W0" else "nominal",
            actuator_case="nominal",
        )
        primitive = primitives[row_index % len(primitives)]
        evidence = simulate_primitive_rollout(
            rollout_id=f"r{config.run_id:03d}_c{spec.chunk_index:05d}_{offset:05d}",
            initial_state=state,
            context=context,
            primitive=primitive,
            config=RolloutConfig(W_layer=spec.context_id),
        )
        rows.append(rollout_with_context_row(evidence, context))
    return rows


def _state_for_row(row_index: int) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = 2.0 + 0.01 * float(row_index % 20)
    state[1] = 2.0 + 0.02 * float((row_index // 3) % 10)
    state[2] = 1.45 + 0.01 * float(row_index % 15)
    state[3] = np.deg2rad(float((row_index % 5) - 2) * 2.0)
    state[4] = np.deg2rad(float((row_index % 7) - 3) * 1.5)
    state[5] = 0.0
    state[6] = 5.5 + 0.05 * float(row_index % 10)
    return state


def _metadata_for_row(
    *,
    w_layer: str,
    environment_mode: str,
    seed: int,
    wind_name: str,
    wind_source: str,
) -> EnvironmentMetadata:
    fan_count = 0 if w_layer == "W0" else 1
    return EnvironmentMetadata(
        environment_id=f"{w_layer}_{environment_mode}",
        fan_count=fan_count,
        fan_positions_m=() if fan_count == 0 else (SINGLE_FAN_CENTER_XY,),
        fan_power_scales=() if fan_count == 0 else (1.0,),
        updraft_model_id=wind_name,
        updraft_amplitude_scale=1.0,
        updraft_width_scale=1.0,
        updraft_centre_shift_m=(0.0, 0.0),
        residual_field_id="none",
        randomisation_seed=seed,
        model_source=wind_source,
    )


# =============================================================================
# 3) Chunk Execution and Output Manifests
# =============================================================================
def _write_chunk_manifest(
    *,
    spec: ContextChunkSpec,
    paths,
    partition: TablePartition,
    columns: tuple[str, ...],
    elapsed_s: float,
) -> None:
    payload = {
        "status": "complete",
        "run_id": int(spec.run_id),
        "source_run_id": int(spec.source_run_id),
        "context_id": str(spec.context_id),
        "environment_id": str(spec.environment_id),
        "chunk_index": int(spec.chunk_index),
        "chunk_count": int(spec.chunk_count),
        "chunk_size": int(spec.chunk_size),
        "storage_format": str(partition.storage_format),
        "latency_case": str(spec.latency_case),
        "dt_s": float(spec.dt_s),
        "horizon_s": float(spec.horizon_s),
        "row_count": int(partition.row_count),
        "columns": list(columns),
        "checksum_sha256": file_sha256(paths.partition_path),
        "planning_read_s": 0.0,
        "selection_s": 0.0,
        "simulation_s": float(elapsed_s),
        "descriptor_build_s": 0.0,
        "write_s": 0.0,
        "total_s": float(elapsed_s),
    }
    filesystem_path(paths.manifest_json).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(paths.manifest_json).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )


def _write_run_outputs(
    *,
    config: ContextArchiveConfig,
    run_root: Path,
    worker_fields: dict[str, object],
    partitions: list[TablePartition],
    outcome_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    dry_run: bool,
) -> dict[str, object]:
    manifest_dir = run_root / "manifests"
    metrics_dir = run_root / "metrics"
    reports_dir = run_root / "reports"
    for directory in (manifest_dir, metrics_dir, reports_dir):
        filesystem_path(directory).mkdir(parents=True, exist_ok=True)

    run_manifest = {
        **worker_fields,
        "run_id": int(config.run_id),
        "rows_requested": int(config.rows),
        "seed": int(config.seed),
        "w_layers": list(config.w_layers),
        "env_modes": list(config.env_modes),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "storage_format": resolve_storage_format(config.storage_format),
        "compression_level": int(config.compression_level),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "dry_run_schedule": bool(dry_run),
        "claim_status": "simulation_only_preflight",
        "blocked_claims": [
            "real_flight_transfer",
            "hardware_readiness",
            "mission_success",
            "W2_W3_robustness",
        ],
        "failures": failures,
    }
    (manifest_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="ascii",
    )
    write_table_manifest(
        manifest_dir / "table_manifest.json",
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=resolve_storage_format(config.storage_format),
            tables=tuple(partitions),
        ),
    )
    _write_runtime_summary(metrics_dir / "runtime_summary.csv", run_manifest, partitions)
    _write_outcome_summary(metrics_dir / "outcome_summary.csv", outcome_rows)
    file_audit = _write_file_size_audit(metrics_dir / "file_size_audit.csv", run_root)
    _write_report(
        reports_dir / "run_report.md",
        run_manifest=run_manifest,
        partition_count=len(partitions),
        file_audit=file_audit,
    )
    return {
        "run_root": run_root,
        "run_manifest": manifest_dir / "run_manifest.json",
        "table_manifest": manifest_dir / "table_manifest.json",
        "partition_count": len(partitions),
        "file_size_audit": file_audit,
    }


def _write_runtime_summary(
    path: Path,
    run_manifest: dict[str, object],
    partitions: list[TablePartition],
) -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": run_manifest["run_id"],
                "selected_worker_count": run_manifest.get("selected_worker_count", 0),
                "storage_format": run_manifest["storage_format"],
                "partition_count": len(partitions),
                "row_count": sum(partition.row_count for partition in partitions),
                "claim_status": run_manifest["claim_status"],
            }
        ]
    )
    frame.to_csv(filesystem_path(path), index=False)


def _write_outcome_summary(path: Path, outcome_rows: list[dict[str, object]]) -> None:
    if not outcome_rows:
        frame = pd.DataFrame(
            columns=["W_layer", "environment_id", "outcome_class", "row_count", "accepted_count"]
        )
    else:
        frame = (
            pd.DataFrame(outcome_rows)
            .groupby(["W_layer", "environment_id", "outcome_class"], dropna=False)
            .agg(row_count=("outcome_class", "size"), accepted_count=("accepted", "sum"))
            .reset_index()
        )
    frame.to_csv(filesystem_path(path), index=False)


def _write_file_size_audit(path: Path, run_root: Path) -> list[dict[str, object]]:
    rows = []
    for item in sorted(run_root.rglob("*")):
        if item.is_file():
            byte_count = int(filesystem_path(item).stat().st_size)
            rows.append(
                {
                    "path": item.relative_to(run_root).as_posix(),
                    "byte_count": byte_count,
                    "under_100mb": bool(byte_count <= MAX_GENERATED_FILE_SIZE_MB * 1024 * 1024),
                }
            )
    pd.DataFrame(rows).to_csv(filesystem_path(path), index=False)
    return rows


def _write_report(
    path: Path,
    *,
    run_manifest: dict[str, object],
    partition_count: int,
    file_audit: list[dict[str, object]],
) -> None:
    oversized = [row for row in file_audit if not row["under_100mb"]]
    text = "\n".join(
        [
            "# Contextual Archive Preflight Report",
            "",
            f"- Run ID: `{run_manifest['run_id']}`",
            f"- Rows requested: `{run_manifest['rows_requested']}`",
            f"- Partitions written: `{partition_count}`",
            f"- Storage format: `{run_manifest['storage_format']}`",
            f"- File-size failures: `{len(oversized)}`",
            f"- Claim status: `{run_manifest['claim_status']}`",
            "",
            "This preflight writes schema and runtime evidence only. It does not make a controller, transfer, hardware-readiness, or robustness claim.",
            "",
        ]
    )
    filesystem_path(path).write_text(text, encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    run_contextual_archive_preflight(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
