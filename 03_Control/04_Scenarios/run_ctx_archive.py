from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from controller_registry import load_selected_controller_records  # noqa: E402
from evidence_status import registry_claim_status_for  # noqa: E402
from evidence_stage_utils import (  # noqa: E402
    write_blocked_approximate_ratio_summary,
    write_claim_boundary_report,
    write_coverage_summary,
    write_file_size_audit as write_stage_file_size_audit,
)
from env_ctx import EnvironmentMetadata, build_environment_context  # noqa: E402
from env_instance import (  # noqa: E402
    environment_instance_for_mode,
    environment_instance_row,
    environment_metadata_from_instance,
)
from env_surrogate import (  # noqa: E402
    READY_STATUS,
    resolve_surrogate_binding,
    surrogate_binding_row,
    wind_field_for_binding,
)
from implementation_instance import (  # noqa: E402
    implementation_instance_for_layer,
    implementation_instance_row,
)
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from prim_cat import active_primitive_catalogue  # noqa: E402
from prim_features import primitive_feature_record, primitive_feature_row  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    blocked_rollout_evidence,
    rollout_with_context_row,
    simulate_primitive_rollout,
)
from state_sampling import (  # noqa: E402
    archive_state_sample_for_row,
    archive_state_sample_row,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Archive config and CLI
# 2) Schedule and row generation
# 3) Worker execution
# 4) Output manifests
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
    rollout_backend: str = "model_backed_lqr"
    selected_controller_registry: Path | None = None


def parse_args(argv: list[str] | None = None) -> ContextArchiveConfig:
    parser = argparse.ArgumentParser(
        description="Run a temp/local contextual primitive archive preflight."
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--w-layers", default="W0,W1")
    parser.add_argument("--env-modes", default="dry_air,gaussian_single")
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
    parser.add_argument("--selected-controller-registry", type=Path, default=None)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Use the deterministic interface smoke backend; not thesis archive evidence.",
    )
    parser.add_argument(
        "--rollout-backend",
        choices=(
            "smoke_only",
            "model_backed_lqr",
        ),
        default="model_backed_lqr",
        help="Rollout backend. LQR is the default R6 evidence path.",
    )
    args = parser.parse_args(argv)
    rollout_backend = str(args.rollout_backend)
    if bool(args.smoke_only):
        rollout_backend = "smoke_only"
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
        rollout_backend=rollout_backend,
        selected_controller_registry=None
        if args.selected_controller_registry is None
        else Path(args.selected_controller_registry),
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
    run_root = Path(config.output_root) / f"r{config.run_id:03d}"
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

    chunks_to_run = schedule
    if config.stop_after_chunks is not None:
        chunks_to_run = chunks_to_run[: int(config.stop_after_chunks)]
    execution = _execute_chunks(
        config=config,
        run_root=run_root,
        schedule=chunks_to_run,
        storage_format=storage_format,
        selected_worker_count=worker_decision.selected_worker_count,
    )
    return _write_run_outputs(
        config=config,
        run_root=run_root,
        worker_fields=dense_run_manifest_fields(
            run_stage="contextual_archive_preflight",
            environment_context="strict_surrogate_lqr_backed",
            worker_decision=worker_decision,
        ),
        partitions=execution["partitions"],
        outcome_rows=execution["outcome_rows"],
        failures=execution["failures"],
        worker_execution=execution["worker_execution"],
        dry_run=False,
    )


def _validate_config(config: ContextArchiveConfig) -> None:
    if config.rows <= 0:
        raise ValueError("rows must be positive.")
    if config.rollout_backend == "smoke_only" and config.rows > 10_000:
        raise ValueError("smoke-only preflight is capped at 10k rows.")
    if config.candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    if not {"W0", "W1"}.issubset(set(config.w_layers)):
        raise ValueError("R6 LQR contextual archive requires W0 and W1 coverage.")
    unknown = set(config.w_layers) - {"W0", "W1"}
    if unknown:
        raise ValueError(
            "R6 LQR contextual archive is W0/W1 only; "
            f"reserve {sorted(unknown)} for R8/R9 replay stages."
        )
    if "dry_air" not in set(config.env_modes):
        raise ValueError("R6 contextual archive requires dry_air for W0.")
    if not any(mode != "dry_air" for mode in config.env_modes):
        raise ValueError("R6 contextual archive requires at least one W1 Gaussian environment mode.")
    if config.rollout_backend not in {
        "model_backed_lqr",
        "smoke_only",
    }:
        raise ValueError("rollout_backend must be a retained LQR backend.")


def _build_schedule(
    config: ContextArchiveConfig,
    *,
    run_root: Path,
    storage_format: str,
) -> list[ContextChunkSpec]:
    chunk_count = int(np.ceil(config.rows / config.candidate_chunk_size))
    pairs = _r6_schedule_pairs(config)
    schedule = []
    for chunk_index in range(chunk_count):
        w_layer, env_mode = pairs[chunk_index % len(pairs)]
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


def _r6_schedule_pairs(config: ContextArchiveConfig) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    if "W0" in config.w_layers:
        pairs.append(("W0", "dry_air"))
    if "W1" in config.w_layers:
        for mode in config.env_modes:
            if mode != "dry_air":
                pairs.append(("W1", mode))
    return tuple(pairs)


def _chunk_rows(
    *,
    config: ContextArchiveConfig,
    spec: ContextChunkSpec,
) -> list[dict[str, object]]:
    primitives = active_primitive_catalogue()
    controller_registry = load_selected_controller_records(config.selected_controller_registry)
    rows: list[dict[str, object]] = []
    for offset in range(spec.chunk_size):
        row_index = int(spec.chunk_index * config.candidate_chunk_size + offset)
        state_sample = archive_state_sample_for_row(
            row_index,
            seed=config.seed,
            W_layer=spec.context_id,
            environment_mode=spec.environment_id,
        )
        state = state_sample.state_vector
        instance = environment_instance_for_mode(
            spec.context_id,
            spec.environment_id,
            config.seed + row_index,
        )
        implementation = implementation_instance_for_layer(
            spec.context_id,
            config.seed + row_index,
            latency_case="nominal",
        )
        plant = plant_instance_for_layer(
            spec.context_id,
            config.seed + row_index,
        )
        metadata = environment_metadata_from_instance(instance)
        binding = resolve_surrogate_binding(
            spec.context_id,
            metadata,
            randomisation_seed=config.seed + row_index,
        )
        wind = wind_field_for_binding(binding)
        context = build_environment_context(
            state,
            wind_field=wind,
            metadata=metadata,
            latency_case=implementation.latency_case,
            actuator_case="nominal",
            surrogate_binding=binding,
        )
        primitive = primitives[row_index % len(primitives)]
        selected_record = controller_registry.get(primitive.primitive_id) if controller_registry else None
        selected_controller = selected_record.controller if selected_record is not None else None
        controller_status = (
            "W0_W1_registry_selected"
            if selected_controller is not None
            else (
                "nominal_unselected_smoke"
                if config.rollout_backend == "smoke_only"
                else "missing_selected_registry_entry"
            )
        )
        rollout_id = f"r{config.run_id:03d}_c{spec.chunk_index:05d}_{offset:05d}"
        rollout_config = RolloutConfig(
            W_layer=spec.context_id,
            dt_s=float(spec.dt_s),
            rollout_backend=config.rollout_backend,
            wind_mode=binding.wind_mode,
        )
        if binding.surrogate_binding_status != READY_STATUS:
            evidence = blocked_rollout_evidence(
                rollout_id=rollout_id,
                episode_id=f"episode_{row_index:07d}",
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                failure_label="surrogate_binding_blocked",
                controller=selected_controller,
                controller_selection_status=controller_status,
                candidate_index=selected_record.candidate_index if selected_record is not None else "",
                candidate_weight_label=selected_record.candidate_weight_label if selected_record is not None else "",
            )
        elif config.selected_controller_registry is not None and selected_controller is None:
            evidence = blocked_rollout_evidence(
                rollout_id=rollout_id,
                episode_id=f"episode_{row_index:07d}",
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                failure_label="controller_missing_from_selected_registry",
                controller_selection_status="missing_selected_registry_entry",
                candidate_index="",
                candidate_weight_label="",
            )
        else:
            evidence = simulate_primitive_rollout(
                rollout_id=rollout_id,
                episode_id=f"episode_{row_index:07d}",
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                wind_field=wind,
                implementation_instance=implementation,
                plant_instance=plant,
                controller=selected_controller,
                controller_selection_status=controller_status,
                candidate_index=selected_record.candidate_index if selected_record is not None else "",
                candidate_weight_label=selected_record.candidate_weight_label if selected_record is not None else "",
            )
        row = rollout_with_context_row(evidence, context)
        _apply_registry_row_status(
            row,
            selected_record=selected_record,
            config=config,
            outcome_class=str(row.get("outcome_class", "")),
        )
        row.update({f"surrogate_{key}": value for key, value in surrogate_binding_row(binding).items()})
        row.update({f"environment_instance_{key}": value for key, value in environment_instance_row(instance).items()})
        row["environment_adjustment_status"] = _wind_adjustment_field(
            wind,
            "environment_adjustment_status",
            "dry_air_or_blocked",
        )
        row["environment_adjustment_limitations"] = _wind_adjustment_field(
            wind,
            "environment_adjustment_limitations",
            "",
        )
        row.update({f"implementation_instance_{key}": value for key, value in implementation_instance_row(implementation).items()})
        row.update({f"plant_instance_{key}": value for key, value in plant_instance_row(plant).items()})
        row.update(archive_state_sample_row(state_sample))
        row.update(
            primitive_feature_row(
                primitive_feature_record(
                    state=state,
                    context=context,
                    primitive=primitive,
                    start_state_family=state_sample.start_state_family,
                    previous_primitive_status=state_sample.previous_primitive_status,
                    synthetic_time_since_launch_s=state_sample.synthetic_time_since_launch_s,
                    controller_id=str(row.get("controller_id", primitive.controller_id)),
                    linearisation_id=str(row.get("linearisation_id", primitive.linearisation_source)),
                    lqr_synthesis_status=str(row.get("lqr_synthesis_status", "solved")),
                )
            )
        )
        rows.append(row)
    return rows


def _state_for_row(row_index: int) -> np.ndarray:
    return archive_state_sample_for_row(
        row_index,
        seed=0,
        W_layer="W0",
        environment_mode="dry_air",
    ).state_vector


def _metadata_for_row(
    *,
    w_layer: str,
    environment_mode: str,
    seed: int,
) -> EnvironmentMetadata:
    instance = environment_instance_for_mode(
        w_layer,
        environment_mode,
        seed,
    )
    return environment_metadata_from_instance(instance)


def _wind_adjustment_field(wind: object | None, name: str, default: str) -> str:
    if wind is None:
        return str(default)
    if hasattr(wind, name):
        return str(getattr(wind, name))
    base = getattr(wind, "base", None)
    if base is not None and hasattr(base, name):
        return str(getattr(base, name))
    return str(default)


def _apply_registry_row_status(
    row: dict[str, object],
    *,
    selected_record,
    config: ContextArchiveConfig,
    outcome_class: str,
) -> None:
    if selected_record is not None:
        row.update(selected_record.row_metadata())
        row["source_controller_selection_status"] = "W0_W1_registry_selected"
        row["archive_evidence_status"] = (
            "blocked" if outcome_class == "blocked" else selected_record.registry_status
        )
        row["evidence_eligibility_reason"] = (
            "blocked_registry_backed_row"
            if outcome_class == "blocked"
            else f"eligible_registry_backed_{selected_record.registry_status}"
        )
        return

    if config.rollout_backend == "smoke_only":
        row["registry_status"] = "smoke_incomplete"
        row["registry_claim_status"] = registry_claim_status_for("smoke_incomplete")
        row["archive_evidence_status"] = "smoke_incomplete"
        row["evidence_eligibility_reason"] = "debug_smoke_incomplete"
    else:
        row["registry_status"] = "blocked"
        row["registry_claim_status"] = registry_claim_status_for("blocked")
        row["archive_evidence_status"] = "blocked"
        row["evidence_eligibility_reason"] = "blocked_missing_selected_registry"
    row["registry_path"] = (
        "" if config.selected_controller_registry is None else Path(config.selected_controller_registry).as_posix()
    )
    row["selected_controller_status"] = ""
    row["selected_controller_reason"] = ""
    row["source_controller_selection_status"] = ""


# =============================================================================
# 3) Worker Execution
# =============================================================================
def _execute_chunks(
    *,
    config: ContextArchiveConfig,
    run_root: Path,
    schedule: list[ContextChunkSpec],
    storage_format: str,
    selected_worker_count: int,
) -> dict[str, object]:
    partitions: list[TablePartition] = []
    outcome_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    pending: list[ContextChunkSpec] = []

    for spec in schedule:
        try:
            status = chunk_status(spec, run_root=run_root)
            if status == "complete" and config.resume:
                partitions.append(_partition_from_existing_manifest(spec, run_root=run_root, storage_format=storage_format))
                continue
            if status == "corrupt" and config.repair_incomplete:
                remove_chunk_outputs(spec, run_root=run_root)
            pending.append(spec)
        except Exception as exc:
            failures.append(_failure_row(spec, exc))
            if not config.continue_on_chunk_failure:
                raise

    worker_count = min(max(1, int(selected_worker_count)), max(1, len(pending) or 1))
    if worker_count > 1 and pending:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_single_chunk,
                    config=config,
                    spec=spec,
                    run_root=run_root,
                    storage_format=storage_format,
                ): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                    partitions.append(result["partition"])
                    outcome_rows.extend(result["outcome_rows"])
                except Exception as exc:
                    failures.append(_failure_row(spec, exc))
                    if not config.continue_on_chunk_failure:
                        raise
    else:
        for spec in pending:
            try:
                result = _run_single_chunk(
                    config=config,
                    spec=spec,
                    run_root=run_root,
                    storage_format=storage_format,
                )
                partitions.append(result["partition"])
                outcome_rows.extend(result["outcome_rows"])
            except Exception as exc:
                failures.append(_failure_row(spec, exc))
                if not config.continue_on_chunk_failure:
                    raise

    return {
        "partitions": sorted(partitions, key=lambda item: item.relative_path),
        "outcome_rows": outcome_rows,
        "failures": failures,
        "worker_execution": {
            "chunk_execution_backend": "process_pool" if worker_count > 1 and pending else "single_worker",
            "actual_worker_count": int(worker_count),
            "parallel_chunk_count": int(len(pending)),
            "worker_enabled": bool(worker_count > 1 and pending),
        },
    }


def _run_single_chunk(
    *,
    config: ContextArchiveConfig,
    spec: ContextChunkSpec,
    run_root: Path,
    storage_format: str,
) -> dict[str, object]:
    started = time.perf_counter()
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
    return {
        "partition": partition,
        "outcome_rows": [
            {
                "W_layer": row["W_layer"],
                "environment_id": row["environment_id"],
                "outcome_class": row["outcome_class"],
                "accepted": bool(row["accepted"]),
                "rollout_backend": row["rollout_backend"],
                "evidence_role": row["evidence_role"],
                "controller_id": row["controller_id"],
                "controller_selection_status": row.get("controller_selection_status", ""),
                "controller_evidence_status": row.get("controller_evidence_status", ""),
                "controller_executable": bool(row.get("controller_executable", False)),
                "candidate_index": row.get("candidate_index", ""),
                "candidate_weight_label": row.get("candidate_weight_label", ""),
                "selected_controller_status": row.get("selected_controller_status", ""),
                "selected_controller_reason": row.get("selected_controller_reason", ""),
                "registry_status": row.get("registry_status", ""),
                "registry_claim_status": row.get("registry_claim_status", ""),
                "registry_path": row.get("registry_path", ""),
                "archive_evidence_status": row.get("archive_evidence_status", ""),
                "evidence_eligibility_reason": row.get("evidence_eligibility_reason", ""),
                "continuation_valid": bool(row["continuation_valid"]),
                "episode_terminal_useful": bool(row["episode_terminal_useful"]),
                "continuation_status": row["continuation_status"],
                "episode_terminal_status": row["episode_terminal_status"],
                "episode_utility_label": row["episode_utility_label"],
                "terminal_use_trainable": bool(row["terminal_use_trainable"]),
                "boundary_use_class": row["boundary_use_class"],
                "start_state_family": row["start_state_family"],
                "state_sample_source": row["state_sample_source"],
                "state_envelope_label": row["state_envelope_label"],
                "previous_primitive_status": row["previous_primitive_status"],
                "primitive_id": row["primitive_id"],
                "latency_case": row["latency_case"],
                "surrogate_binding_status": row["surrogate_binding_status"],
                "environment_instance_environment_id": row.get("environment_instance_environment_id", ""),
            }
            for row in rows
        ],
    }


def _partition_from_existing_manifest(
    spec: ContextChunkSpec,
    *,
    run_root: Path,
    storage_format: str,
) -> TablePartition:
    paths = contextual_table_paths(spec, run_root=run_root)
    manifest = validate_chunk_manifest(spec, run_root=run_root)
    return TablePartition(
        table_name="contextual_rows",
        relative_path=paths.partition_path.relative_to(run_root / "tables").as_posix(),
        storage_format=storage_format,
        row_count=int(manifest["row_count"]),
        byte_count=int(filesystem_path(paths.partition_path).stat().st_size),
        columns=tuple(str(value) for value in manifest["columns"]),
        checksum_sha256=str(manifest["checksum_sha256"]),
    )


def _failure_row(spec: ContextChunkSpec, exc: Exception) -> dict[str, object]:
    return {
        "chunk_index": int(spec.chunk_index),
        "context_id": str(spec.context_id),
        "environment_id": str(spec.environment_id),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


# =============================================================================
# 4) Output Manifests
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


def _archive_evidence_status_for_run(
    *,
    config: ContextArchiveConfig,
    outcome_frame: pd.DataFrame,
    dry_run: bool,
) -> tuple[str, str]:
    if dry_run:
        return "smoke_incomplete", "debug_smoke_incomplete"
    if config.rollout_backend == "smoke_only":
        return "smoke_incomplete", "debug_smoke_incomplete"
    if config.selected_controller_registry is None:
        return "blocked", "blocked_missing_selected_registry"
    if outcome_frame.empty:
        return "blocked", "blocked_no_rows"
    registry_backed = outcome_frame.get("controller_selection_status", pd.Series(dtype=str)).astype(str).eq(
        "W0_W1_registry_selected"
    )
    nonblocked = outcome_frame.get("outcome_class", pd.Series(dtype=str)).astype(str).ne("blocked")
    candidate_label = outcome_frame.get("candidate_weight_label", pd.Series(dtype=str)).astype(str)
    q_json = outcome_frame.get("lqr_Q_weights_json", pd.Series(dtype=str)).astype(str)
    r_json = outcome_frame.get("lqr_R_weights_json", pd.Series(dtype=str)).astype(str)
    checksum = outcome_frame.get("lqr_gain_checksum", pd.Series(dtype=str)).astype(str)
    linearisation = outcome_frame.get("linearisation_id", pd.Series(dtype=str)).astype(str)
    complete_metadata = (
        _nonmissing_series(candidate_label)
        & _nonmissing_series(q_json)
        & _nonmissing_series(r_json)
        & _nonmissing_series(checksum)
        & _nonmissing_series(linearisation)
    )
    missing_controller = outcome_frame.get("controller_evidence_status", pd.Series(dtype=str)).astype(str).str.contains(
        "missing",
        na=False,
    )
    missing_ratio = float(missing_controller.mean()) if len(outcome_frame) else 1.0
    blocked_ratio = float((~nonblocked).mean()) if len(outcome_frame) else 1.0
    if not bool((registry_backed & nonblocked).any()):
        return "blocked", "blocked_no_registry_backed_nonblocked_rows"
    if not bool((registry_backed & nonblocked & complete_metadata).any()):
        return "blocked", "blocked_missing_candidate_metadata"
    if missing_ratio > 0.05:
        return "blocked", "blocked_high_missing_controller_ratio"
    if blocked_ratio > 0.60:
        return "blocked", "blocked_high_blocked_ratio"
    statuses = set(outcome_frame.get("registry_status", pd.Series(dtype=str)).astype(str))
    if "complete" in statuses:
        return "complete", "eligible_registry_backed_complete"
    if "accepted_fallback" in statuses:
        return "accepted_fallback", "eligible_registry_backed_accepted_fallback"
    if "smoke_incomplete" in statuses:
        return "smoke_incomplete", "debug_smoke_incomplete"
    return "blocked", "blocked_registry_status_not_eligible"


def _nonmissing_series(series: pd.Series) -> pd.Series:
    normalised = series.astype(str).str.strip().str.lower()
    return normalised.ne("") & ~normalised.isin(["nan", "none", "null"])


def _write_run_outputs(
    *,
    config: ContextArchiveConfig,
    run_root: Path,
    worker_fields: dict[str, object],
    partitions: list[TablePartition],
    outcome_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    dry_run: bool,
    worker_execution: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest_dir = run_root / "manifests"
    metrics_dir = run_root / "metrics"
    reports_dir = run_root / "reports"
    for directory in (manifest_dir, metrics_dir, reports_dir):
        filesystem_path(directory).mkdir(parents=True, exist_ok=True)

    outcome_frame = pd.DataFrame(outcome_rows)
    archive_evidence_status, evidence_eligibility_reason = _archive_evidence_status_for_run(
        config=config,
        outcome_frame=outcome_frame,
        dry_run=dry_run,
    )
    run_manifest = {
        **worker_fields,
        **(worker_execution or {"chunk_execution_backend": "not_started", "worker_enabled": False}),
        "run_id": int(config.run_id),
        "rows_requested": int(config.rows),
        "seed": int(config.seed),
        "w_layers": list(config.w_layers),
        "env_modes": list(config.env_modes),
        "rollout_backend": str(config.rollout_backend),
        "selected_controller_registry": ""
        if config.selected_controller_registry is None
        else Path(config.selected_controller_registry).as_posix(),
        "selected_controller_registry_required": bool(config.rollout_backend == "model_backed_lqr"),
        "archive_evidence_status": archive_evidence_status,
        "evidence_eligibility_reason": evidence_eligibility_reason,
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "storage_format": resolve_storage_format(config.storage_format),
        "compression_level": int(config.compression_level),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "dry_run_schedule": bool(dry_run),
        "claim_status": "simulation_only_lqr_backed_preflight",
        "blocked_claims": [
            "real_flight_transfer",
            "hardware_readiness",
            "mission_success",
            "W2_W3_robustness",
            "controller_performance",
        ],
        "failures": failures,
        "official_deferred_commands": _official_deferred_commands(),
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
    write_coverage_summary(
        metrics_dir / "coverage_summary.csv",
        outcome_frame,
        columns=(
            "start_state_family",
            "primitive_id",
            "controller_id",
            "environment_id",
            "environment_instance_environment_id",
            "W_layer",
            "latency_case",
            "boundary_use_class",
            "archive_evidence_status",
            "evidence_eligibility_reason",
            "registry_status",
            "registry_claim_status",
            "controller_evidence_status",
            "continuation_valid",
            "episode_terminal_useful",
            "continuation_status",
            "episode_terminal_status",
            "outcome_class",
        ),
    )
    ratio_summary = write_blocked_approximate_ratio_summary(
        metrics_dir / "blocked_or_approximate_ratio_summary.csv",
        outcome_frame,
    )
    _write_failure_taxonomy(metrics_dir / "failure_taxonomy.csv", outcome_frame)
    _write_governor_rejection_table(metrics_dir / "governor_rejection_table.csv", outcome_frame)
    _write_medoid_report(reports_dir / "medoid_cluster_report.md", outcome_frame)
    _write_hardware_shortlist(metrics_dir / "hardware_shortlist.csv", outcome_frame)
    _write_figure_source_manifest(manifest_dir / "figure_source_manifest.json", run_root, outcome_frame)
    file_audit, _ = write_stage_file_size_audit(
        run_root,
        metrics_dir / "file_size_audit.csv",
    )
    write_claim_boundary_report(
        reports_dir / "claim_boundary_report.md",
        stage="R6 W0/W1 contextual archive",
        status="dry_run" if dry_run else "local_archive_written",
        claim_status=str(run_manifest["claim_status"]),
        blocked_claims=tuple(str(item) for item in run_manifest["blocked_claims"]),
    )
    _write_report(
        reports_dir / "run_report.md",
        run_manifest=run_manifest,
        partition_count=len(partitions),
        file_audit=file_audit,
        blocked_ratio=float(ratio_summary["blocked_ratio"]),
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
                "rollout_backend": run_manifest["rollout_backend"],
                "worker_enabled": bool(run_manifest.get("worker_enabled", False)),
            }
        ]
    )
    frame.to_csv(filesystem_path(path), index=False)


def _write_outcome_summary(path: Path, outcome_rows: list[dict[str, object]]) -> None:
    if not outcome_rows:
        frame = pd.DataFrame(
            columns=[
                "W_layer",
                "environment_id",
                "rollout_backend",
                "evidence_role",
                "surrogate_binding_status",
                "outcome_class",
                "continuation_status",
                "episode_terminal_status",
                "start_state_family",
                "state_envelope_label",
                "previous_primitive_status",
                "primitive_id",
                "latency_case",
                "boundary_use_class",
                "archive_evidence_status",
                "evidence_eligibility_reason",
                "registry_status",
                "registry_claim_status",
                "controller_evidence_status",
                "continuation_valid",
                "episode_terminal_useful",
                "row_count",
                "accepted_count",
                "continuation_valid_count",
                "episode_terminal_useful_count",
                "terminal_use_trainable_count",
            ]
        )
    else:
        frame = (
            pd.DataFrame(outcome_rows)
            .groupby(
                [
                    "W_layer",
                    "environment_id",
                    "rollout_backend",
                    "evidence_role",
                    "surrogate_binding_status",
                    "outcome_class",
                    "continuation_status",
                    "episode_terminal_status",
                    "start_state_family",
                    "state_envelope_label",
                    "previous_primitive_status",
                    "primitive_id",
                    "latency_case",
                    "boundary_use_class",
                    "archive_evidence_status",
                    "evidence_eligibility_reason",
                    "registry_status",
                    "registry_claim_status",
                    "controller_evidence_status",
                    "continuation_valid",
                    "episode_terminal_useful",
                ],
                dropna=False,
            )
            .agg(
                row_count=("outcome_class", "size"),
                accepted_count=("accepted", "sum"),
                continuation_valid_count=("continuation_valid", "sum"),
                episode_terminal_useful_count=("episode_terminal_useful", "sum"),
                terminal_use_trainable_count=("terminal_use_trainable", "sum"),
            )
            .reset_index()
        )
    frame.to_csv(filesystem_path(path), index=False)


def _write_file_size_audit(path: Path, run_root: Path) -> list[dict[str, object]]:
    rows = []
    for item in sorted(run_root.rglob("*")):
        if item.is_file():
            byte_count = int(filesystem_path(item).stat().st_size)
            size_mb = float(byte_count) / (1024.0 * 1024.0)
            rows.append(
                {
                    "path": item.relative_to(run_root).as_posix(),
                    "byte_count": byte_count,
                    "size_mb": size_mb,
                    "above_75mb": bool(size_mb > 75.0),
                    "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                    "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                    "under_100mb": bool(byte_count <= MAX_GENERATED_FILE_SIZE_MB * 1024 * 1024),
                }
            )
    pd.DataFrame(rows).to_csv(filesystem_path(path), index=False)
    return rows


def _write_failure_taxonomy(path: Path, frame: pd.DataFrame) -> None:
    columns = ["primitive_id", "controller_id", "W_layer", "outcome_class", "failure_label", "termination_cause"]
    if frame.empty:
        pd.DataFrame(columns=[*columns, "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    use_columns = [column for column in columns if column in frame.columns]
    summary = frame.groupby(use_columns, dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_governor_rejection_table(path: Path, frame: pd.DataFrame) -> None:
    columns = ["primitive_id", "controller_id", "W_layer", "entry_check_status", "outcome_class"]
    if frame.empty:
        pd.DataFrame(columns=[*columns, "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    use_columns = [column for column in columns if column in frame.columns]
    summary = frame.groupby(use_columns, dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_medoid_report(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# LQR Medoid Cluster Report",
        "",
        "Rows are replayable archive rows. LQR gains are never averaged.",
        "",
    ]
    if not frame.empty:
        strata = [
            column
            for column in (
                "W_layer",
                "primitive_id",
                "controller_id",
                "latency_case",
                "start_state_family",
                "outcome_class",
                "continuation_valid",
                "episode_terminal_useful",
                "boundary_use_class",
            )
            if column in frame.columns
        ]
        for values, group in frame.groupby(strata, dropna=False):
            medoid = group.iloc[len(group) // 2]
            key = values if isinstance(values, tuple) else (values,)
            lines.append(f"- {dict(zip(strata, key, strict=True))}: `{medoid.get('rollout_id', '')}`")
    filesystem_path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_hardware_shortlist(path: Path, frame: pd.DataFrame) -> None:
    columns = [
        "primitive_id",
        "controller_id",
        "accepted",
        "minimum_wall_margin_m",
        "minimum_speed_m_s",
        "archive_evidence_status",
        "evidence_eligibility_reason",
    ]
    if frame.empty or not {"primitive_id", "controller_id"}.issubset(frame.columns):
        pd.DataFrame(columns=["primitive_id", "controller_id", "row_count", "hardware_shortlist_status"]).to_csv(
            filesystem_path(path),
            index=False,
        )
        return
    eligible = frame[
        frame.get("archive_evidence_status", pd.Series([""] * len(frame))).astype(str).isin(
            ["complete", "accepted_fallback"]
        )
    ].copy()
    if eligible.empty:
        pd.DataFrame(
            [
                {
                    "primitive_id": "",
                    "controller_id": "",
                    "row_count": 0,
                    "hardware_shortlist_status": "not_thesis_eligible",
                    "archive_evidence_status": _dominant_status(frame, "archive_evidence_status"),
                    "evidence_eligibility_reason": _dominant_status(frame, "evidence_eligibility_reason"),
                }
            ]
        ).to_csv(filesystem_path(path), index=False)
        return
    summary = (
        eligible.groupby(["primitive_id", "controller_id"], dropna=False)
        .agg(
            row_count=("primitive_id", "size"),
            accepted_count=("accepted", "sum"),
            min_wall_margin_m=("minimum_wall_margin_m", "min"),
            min_speed_m_s=("minimum_speed_m_s", "min"),
            archive_evidence_status=("archive_evidence_status", "first"),
            evidence_eligibility_reason=("evidence_eligibility_reason", "first"),
        )
        .reset_index()
        .sort_values(["accepted_count", "min_wall_margin_m"], ascending=[False, False])
        .head(10)
    )
    summary["hardware_shortlist_status"] = "simulation_only_candidate_not_hardware_ready"
    summary.to_csv(filesystem_path(path), index=False)


def _dominant_status(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    values = frame[column].astype(str)
    if values.empty:
        return ""
    return str(values.mode().iloc[0]) if not values.mode().empty else str(values.iloc[0])


def _write_figure_source_manifest(path: Path, run_root: Path, frame: pd.DataFrame) -> None:
    payload = {
        "figure_source_manifest_version": "lqr_dense_v1",
        "run_root": run_root.as_posix(),
        "row_count": int(len(frame)),
        "sources": [
            "metrics/outcome_summary.csv",
            "metrics/coverage_summary.csv",
            "metrics/failure_taxonomy.csv",
            "reports/medoid_cluster_report.md",
            "metrics/hardware_shortlist.csv",
        ],
        "claim_boundary": "simulation_only_until_matched_real_flight_replay",
    }
    filesystem_path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _write_report(
    path: Path,
    *,
    run_manifest: dict[str, object],
    partition_count: int,
    file_audit: list[dict[str, object]],
    blocked_ratio: float,
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
            f"- Rollout backend: `{run_manifest['rollout_backend']}`",
            f"- Worker enabled: `{run_manifest.get('worker_enabled', False)}`",
            f"- File-size failures: `{len(oversized)}`",
            f"- Blocked row ratio: `{blocked_ratio:.6f}`",
            f"- Claim status: `{run_manifest['claim_status']}`",
            "",
            "This preflight writes schema/runtime evidence and retained terminal-boundary labels only. It does not make a controller-performance, transfer, hardware-readiness, mission-success, or robustness claim.",
            "",
        ]
    )
    filesystem_path(path).write_text(text, encoding="ascii")


def _official_deferred_commands() -> dict[str, str]:
    return {
        "r6_lqr_20k_temp_or_local": (
            "python 03_Control/04_Scenarios/run_lqr_contextual_archive.py --run-id 60 "
            "--rows 20000 --seed 60 --w-layers W0,W1 "
            "--env-modes dry_air,gaussian_single,gaussian_four,fan_shift,power_scale "
            "--candidate-chunk-size 1000 --workers 8 --max-workers 8 "
            "--storage-format auto --compression-level 1 --resume --repair-incomplete "
            "--rollout-backend model_backed_lqr "
            "--selected-controller-registry 03_Control/05_Results/lqr_contextual_v1_0/r6/tune_100/metrics/selected_lqr_controllers.csv "
            "--output-root 03_Control/05_Results/lqr_contextual_v1_0/r6_lqr_20k"
        ),
        "r6_lqr_40k_temp_or_local": (
            "python 03_Control/04_Scenarios/run_lqr_contextual_archive.py --run-id 61 "
            "--rows 40000 --seed 61 --w-layers W0,W1 "
            "--env-modes dry_air,gaussian_single,gaussian_four,fan_shift,power_scale "
            "--candidate-chunk-size 1000 --workers 8 --max-workers 8 "
            "--storage-format auto --compression-level 1 --resume --repair-incomplete "
            "--rollout-backend model_backed_lqr "
            "--selected-controller-registry 03_Control/05_Results/lqr_contextual_v1_0/r6/tune_100/metrics/selected_lqr_controllers.csv "
            "--output-root 03_Control/05_Results/lqr_contextual_v1_0/r6_lqr_40k"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    run_contextual_archive_preflight(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
