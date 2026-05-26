from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import (  # noqa: E402
    MAX_GENERATED_FILE_SIZE_MB,
    RUNTIME_STORAGE_CONTRACT,
    worker_count_decision,
)
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    file_sha256,
    filesystem_path,
    read_table_partition,
    resolve_storage_format,
    table_extension,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import build_environment_context, environment_context_row  # noqa: E402
from env_instance import environment_instance_for_mode, environment_instance_row, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, surrogate_binding_row, wind_field_for_binding  # noqa: E402
from implementation_instance import implementation_instance_for_layer, implementation_instance_row  # noqa: E402
from frozen_w01_controller_bundle import load_frozen_w01_controller_bundle, write_frozen_w01_controller_bundle  # noqa: E402
from latency import DEFAULT_LATENCY_ENVELOPE, latency_case_config  # noqa: E402
from lqr_controller import ACTIVE_TIMING_AWARE_ROLE, controller_is_active_timing_aware_w01, synthesize_lqr_controller  # noqa: E402
from lqr_tuning import candidate_weight_specs, lqr_tuning_schedule, tuning_schedule_row  # noqa: E402
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from prim_cat import ACTIVE_PRIMITIVE_IDS, active_primitive_catalogue  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    blocked_rollout_evidence,
    rollout_evidence_row,
    simulate_primitive_rollout,
)
from primitive_variant_registry import (  # noqa: E402
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
    PrimitiveControllerVariant,
    primitive_controller_variant,
    start_family_is_compatible,
    variant_row,
    write_variant_registry,
)
from primitive_timing_contract import (  # noqa: E402
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    primitive_timing_contract_row,
)
from state_sampling import archive_state_sample_for_family, archive_state_sample_for_row, archive_state_sample_row, start_state_family_for_row  # noqa: E402


W01_RUNNER_VERSION = "run_lqr_w01_dense_chunked_v411"
PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.11"
W01_TABLE_NAME = "w01_primitive_rows"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w01_dense")
L6_FALLBACK_ROW_COUNT = 19_200
L6_RICH_SIDE_ROW_COUNT = 76_800
L6_RICH_SIDE_CANDIDATE_COUNT = 32
L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE = 100
CROSS_LAYER_START_FAMILY_CYCLE = 20
CROSS_LAYER_SMOKE_READY = "cross_layer_smoke_start_family_complete"
CROSS_LAYER_SMOKE_INCOMPLETE = "artifact_smoke_only_start_family_incomplete"
OFFICIAL_W01_ENVIRONMENT_CASES = (
    ("W0", "dry_air"),
    ("W1", "gaussian_single"),
    ("W1", "gaussian_four"),
)
CLAIM_BOUNDARY = (
    "cleaned_and_restructured_for_corrected_W0_W1_rich_primitive_controller_dense_generation_readiness_only"
)
BLOCKED_CLAIMS = (
    "W0_W1_dense_evidence_complete",
    "W2_survival_complete",
    "W3_robustness_complete",
    "post_W3_library_size_study_ready",
    "governor_validation",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
)
START_FAMILY_MIX = {
    "launch_gate": 0.40,
    "inflight_nominal": 0.25,
    "inflight_lift_region": 0.15,
    "inflight_boundary_near": 0.10,
    "inflight_recovery_edge": 0.10,
}
BALANCED_SCHEDULE_MODE = "balanced_paired"
SMOKE_SCHEDULE_MODE = "smoke_row_index"


@dataclass(frozen=True)
class W01DenseRunConfig:
    run_id: int
    output_root: Path = DEFAULT_OUTPUT_ROOT
    rows: int = 400
    seed: int = 1
    candidate_chunk_size: int = 100
    workers: str | int = 8
    max_workers: int | None = 8
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    stop_after_chunks: int | None = None
    continue_on_chunk_failure: bool = False
    candidate_count: int = 16
    paired_tests_per_candidate: int | None = None
    latency_case: str = "nominal"
    rollout_dt_s: float = CONTROLLER_INPUT_UPDATE_PERIOD_S
    schedule_mode: str = BALANCED_SCHEDULE_MODE


@dataclass(frozen=True)
class W01ChunkSpec:
    run_id: int
    chunk_index: int
    chunk_count: int
    row_start: int
    row_stop: int
    storage_format: str
    compression_level: int


@dataclass(frozen=True)
class W01RowSchedule:
    row_index: int
    primitive_id: str
    candidate_index: int
    W_layer: str
    environment_mode: str
    start_state_family: str
    paired_start_key: str
    paired_start_index: int
    schedule_mode: str


def run_lqr_w01_dense_chunked(config: W01DenseRunConfig) -> dict[str, object]:
    """Run or schedule the corrected W0/W1 primitive-controller dense preflight."""

    if int(config.rows) <= 0:
        raise ValueError("rows must be positive.")
    if int(config.candidate_chunk_size) <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    if int(config.candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")
    if not math.isclose(float(config.rollout_dt_s), CONTROLLER_INPUT_UPDATE_PERIOD_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rollout_dt_s_not_v411_0p020s")
    if config.paired_tests_per_candidate is not None and int(config.paired_tests_per_candidate) <= 0:
        raise ValueError("paired_tests_per_candidate must be positive when provided.")
    if str(config.schedule_mode) not in {BALANCED_SCHEDULE_MODE, SMOKE_SCHEDULE_MODE}:
        raise ValueError(f"schedule_mode must be {BALANCED_SCHEDULE_MODE!r} or {SMOKE_SCHEDULE_MODE!r}.")

    storage_format = resolve_storage_format(config.storage_format)
    run_root = _run_root(config)
    _ensure_run_root(config, run_root)
    for subdir in ("manifests", "metrics", "reports", "chunk_manifests", "tables"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    worker_decision = worker_count_decision(
        config.workers,
        max_workers=config.max_workers,
    )
    variants_by_key, variants = _build_variant_registry(config.candidate_count)
    write_variant_registry(
        variants=variants,
        csv_path=run_root / "metrics" / "primitive_variant_registry.csv",
        json_path=run_root / "manifests" / "primitive_variant_registry.json",
    )

    schedule = _chunk_schedule(config, storage_format=storage_format)
    if config.stop_after_chunks is not None:
        schedule = schedule[: max(0, int(config.stop_after_chunks))]

    run_manifest = _run_manifest(config, run_root, worker_decision, storage_format, schedule, variants)
    _write_json(run_root / "manifests" / "run_manifest.json", run_manifest)

    if config.dry_run_schedule:
        chunk_rows = [_scheduled_chunk_row(chunk, run_root) for chunk in schedule]
        pd.DataFrame(chunk_rows).to_csv(filesystem_path(run_root / "metrics" / "chunk_summary.csv"), index=False)
        _write_empty_table_manifest(run_root, config.run_id, storage_format)
        _write_runtime_summary(run_root, config, row_count=0, status="dry_run_schedule", started=time.time(), ended=time.time())
        _write_empty_metrics(run_root)
        write_frozen_w01_controller_bundle(
            run_root=run_root,
            source_records=_frozen_bundle_source_records(variants_by_key, variants),
        )
        _write_file_size_audit(run_root)
        _write_reports(run_root, status="dry_run_schedule", row_count=0)
        _write_file_size_audit(run_root)
        return _result_payload(run_root, status="dry_run_schedule")

    started = time.time()
    completed_partitions = []
    chunk_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    pending_chunks: list[W01ChunkSpec] = []
    for chunk in schedule:
        try:
            status = _existing_chunk_status(chunk, run_root=run_root)
            if status == "complete":
                if config.resume or config.repair_incomplete:
                    partition = _partition_from_existing(chunk, run_root)
                    completed_partitions.append(partition)
                    chunk_records.append(_completed_chunk_row(chunk, run_root, status="skipped"))
                    continue
                raise RuntimeError(f"complete chunk already exists: c{chunk.chunk_index:05d}; use --resume")
            if status == "corrupt":
                if not config.repair_incomplete:
                    raise RuntimeError(f"corrupt or incomplete chunk exists: c{chunk.chunk_index:05d}; use --repair-incomplete")
                _remove_chunk_files(chunk, run_root)
            pending_chunks.append(chunk)
        except Exception as exc:
            failure = {
                "chunk_index": int(chunk.chunk_index),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            chunk_records.append({**_scheduled_chunk_row(chunk, run_root), **failure})
            if not config.continue_on_chunk_failure:
                _write_chunk_summary(run_root, chunk_records)
                raise

    executed_partitions, executed_records, executed_failures = _execute_pending_chunks(
        pending_chunks,
        run_root=run_root,
        config=config,
        variants_by_key=variants_by_key,
        selected_worker_count=int(worker_decision.selected_worker_count),
    )
    completed_partitions.extend(executed_partitions)
    chunk_records.extend(executed_records)
    failures.extend(executed_failures)
    if executed_failures and not config.continue_on_chunk_failure:
        _write_chunk_summary(run_root, sorted(chunk_records, key=lambda item: int(item.get("chunk_index", -1))))
        first = executed_failures[0]
        raise RuntimeError(f"W01 chunk failed: c{int(first['chunk_index']):05d}: {first.get('error', '')}")

    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=storage_format,
            tables=tuple(completed_partitions),
        ),
    )
    write_frozen_w01_controller_bundle(
        run_root=run_root,
        source_records=_frozen_bundle_source_records(variants_by_key, variants),
    )
    ended = time.time()
    row_count = _write_dense_metrics_from_partitions(run_root, completed_partitions, storage_format)
    _write_chunk_summary(run_root, sorted(chunk_records, key=lambda item: int(item.get("chunk_index", -1))))
    _write_runtime_summary(
        run_root,
        config,
        row_count=row_count,
        status="complete" if not failures else "partial_failed",
        started=started,
        ended=ended,
    )
    _write_file_size_audit(run_root)
    _write_reports(run_root, status="complete" if not failures else "partial_failed", row_count=row_count)
    _write_file_size_audit(run_root)
    return _result_payload(run_root, status="complete" if not failures else "partial_failed")


def _build_variant_registry(
    candidate_count: int,
) -> tuple[dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]], tuple[PrimitiveControllerVariant, ...]]:
    variants_by_key: dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]] = {}
    variants: list[PrimitiveControllerVariant] = []
    for primitive in active_primitive_catalogue():
        specs = candidate_weight_specs(primitive_id=primitive.primitive_id, candidate_count=candidate_count)
        for candidate_index, weight_spec in enumerate(specs):
            controller = synthesize_lqr_controller(primitive, weight_spec=weight_spec)
            variant = primitive_controller_variant(
                primitive=primitive,
                controller=controller,
                candidate_index=int(candidate_index),
                candidate_weight_label=weight_spec.weight_label,
            )
            variants_by_key[(primitive.primitive_id, int(candidate_index))] = (
                primitive,
                controller,
                variant,
                weight_spec.weight_label,
            )
            variants.append(variant)
    return variants_by_key, tuple(variants)


def _frozen_bundle_source_records(
    variants_by_key: dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]],
    variants: tuple[PrimitiveControllerVariant, ...],
):
    controller_by_variant_id = {
        variant.primitive_variant_id: controller
        for _, controller, variant, _ in variants_by_key.values()
    }
    return tuple(
        (variant, controller_by_variant_id[variant.primitive_variant_id])
        for variant in variants
    )


def _write_chunk(
    chunk: W01ChunkSpec,
    *,
    run_root: Path,
    config: W01DenseRunConfig,
    variants_by_key: dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]],
):
    started = time.time()
    rows = [
        _row_for_index(
            row_index=row_index,
            config=config,
            variants_by_key=variants_by_key,
        )
        for row_index in range(int(chunk.row_start), int(chunk.row_stop))
    ]
    frame = pd.DataFrame(rows)
    partition_path = _partition_path(chunk, run_root)
    partition = write_table_partition(
        frame,
        partition_path,
        storage_format=chunk.storage_format,
        compression_level=chunk.compression_level,
    )
    ended = time.time()
    manifest = {
        "runner_version": W01_RUNNER_VERSION,
        "status": "complete",
        "run_id": int(chunk.run_id),
        "chunk_index": int(chunk.chunk_index),
        "chunk_count": int(chunk.chunk_count),
        "row_start": int(chunk.row_start),
        "row_stop": int(chunk.row_stop),
        "row_count": int(len(frame)),
        "storage_format": str(partition.storage_format),
        "compression_level": int(chunk.compression_level),
        "partition_path": partition.relative_path,
        "byte_count": int(partition.byte_count),
        "checksum_sha256": partition.checksum_sha256,
        "duration_s": float(ended - started),
        "table_name": W01_TABLE_NAME,
        "primitive_timing_contract": primitive_timing_contract_row(),
    }
    _write_json(_chunk_manifest_path(chunk, run_root), manifest)
    record = {
        "chunk_index": int(chunk.chunk_index),
        "status": "complete",
        "row_start": int(chunk.row_start),
        "row_stop": int(chunk.row_stop),
        "row_count": int(len(frame)),
        "partition_path": partition.relative_path,
        "manifest_path": _chunk_manifest_path(chunk, run_root).relative_to(run_root).as_posix(),
        "byte_count": int(partition.byte_count),
        "checksum_sha256": partition.checksum_sha256,
        "duration_s": float(ended - started),
    }
    return partition, record


def _write_chunk_worker(payload):
    chunk, run_root, config, variants_by_key = payload
    return _write_chunk(
        chunk,
        run_root=Path(run_root),
        config=config,
        variants_by_key=variants_by_key,
    )


def _execute_pending_chunks(
    chunks: list[W01ChunkSpec],
    *,
    run_root: Path,
    config: W01DenseRunConfig,
    variants_by_key: dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]],
    selected_worker_count: int,
):
    if not chunks:
        return [], [], []
    partitions = []
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    worker_count = max(1, int(selected_worker_count))
    if worker_count <= 1 or len(chunks) <= 1:
        for chunk in chunks:
            try:
                partition, record = _write_chunk(
                    chunk,
                    run_root=run_root,
                    config=config,
                    variants_by_key=variants_by_key,
                )
                partitions.append(partition)
                records.append(record)
            except Exception as exc:
                failure = _chunk_failure_row(chunk, exc)
                failures.append(failure)
                records.append({**_scheduled_chunk_row(chunk, run_root), **failure})
                if not config.continue_on_chunk_failure:
                    break
        return partitions, records, failures

    payloads = [(chunk, run_root, config, variants_by_key) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_chunk = {
            executor.submit(_write_chunk_worker, payload): payload[0]
            for payload in payloads
        }
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                partition, record = future.result()
                partitions.append(partition)
                records.append(record)
            except Exception as exc:
                failure = _chunk_failure_row(chunk, exc)
                failures.append(failure)
                records.append({**_scheduled_chunk_row(chunk, run_root), **failure})
                if not config.continue_on_chunk_failure:
                    for pending in future_to_chunk:
                        pending.cancel()
                    break
    return partitions, records, failures


def _chunk_failure_row(chunk: W01ChunkSpec, exc: Exception) -> dict[str, object]:
    return {
        "chunk_index": int(chunk.chunk_index),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _row_for_index(
    *,
    row_index: int,
    config: W01DenseRunConfig,
    variants_by_key: dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]],
) -> dict[str, object]:
    schedule = _row_schedule_for_index(row_index, config)
    primitive_id = schedule.primitive_id
    candidate_index = int(schedule.candidate_index)
    primitive, controller, variant, weight_label = variants_by_key[(primitive_id, int(candidate_index))]
    W_layer = schedule.W_layer
    environment_mode = schedule.environment_mode
    if schedule.schedule_mode == SMOKE_SCHEDULE_MODE:
        sample = archive_state_sample_for_row(
            int(row_index),
            seed=int(config.seed),
            W_layer=W_layer,
            environment_mode=environment_mode,
        )
    else:
        sample = archive_state_sample_for_family(
            start_state_family=schedule.start_state_family,
            paired_start_key=schedule.paired_start_key,
            sample_index=int(schedule.paired_start_index),
            seed=int(config.seed),
            W_layer=W_layer,
            environment_mode=environment_mode,
        )
    environment = environment_instance_for_mode(W_layer, environment_mode, int(config.seed) + int(row_index))
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding(
        W_layer,
        metadata,
        randomisation_seed=int(config.seed) + int(row_index),
    )
    wind_field = wind_field_for_binding(binding)
    context = build_environment_context(
        sample.state_vector,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=str(config.latency_case),
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    rollout_config = RolloutConfig(
        W_layer=W_layer,
        dt_s=float(config.rollout_dt_s),
        rollout_backend="model_backed_lqr",
        wind_mode="panel",
    )
    compatible = start_family_is_compatible(
        entry_role=variant.entry_role,
        start_state_family=sample.start_state_family,
    )
    if binding.surrogate_binding_status != "ready":
        evidence = blocked_rollout_evidence(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w01_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=rollout_config,
            failure_label="surrogate_binding_blocked",
            controller=controller,
            controller_selection_status="W01_variant_registry_candidate",
            candidate_index=candidate_index,
            candidate_weight_label=str(weight_label),
            termination_cause=str(binding.blocked_reason),
        )
        row = rollout_evidence_row(evidence)
    elif not compatible:
        evidence = blocked_rollout_evidence(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w01_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=RolloutConfig(
                W_layer=W_layer,
                dt_s=float(config.rollout_dt_s),
                rollout_backend="blocked_lqr",
                wind_mode="panel",
            ),
            failure_label=ENTRY_ROLE_REJECTION_LABEL,
            controller=controller,
            controller_selection_status="W01_variant_registry_candidate",
            candidate_index=candidate_index,
            candidate_weight_label=str(weight_label),
            termination_cause=ENTRY_ROLE_REJECTION_STATUS,
        )
        row = rollout_evidence_row(evidence)
        row.update(
            {
                "entry_check_status": ENTRY_ROLE_REJECTION_STATUS,
                "entry_rejection_class": ENTRY_ROLE_REJECTION_LABEL,
                "outcome_class": "rejected",
                "failure_label": ENTRY_ROLE_REJECTION_LABEL,
                "archive_evidence_status": "rejected",
                "evidence_eligibility_reason": ENTRY_ROLE_REJECTION_LABEL,
                "continuation_valid": False,
                "episode_terminal_useful": False,
                "termination_cause": ENTRY_ROLE_REJECTION_STATUS,
            }
        )
    else:
        implementation = implementation_instance_for_layer(
            W_layer,
            int(config.seed) + int(row_index),
            latency_case=str(config.latency_case),
        )
        plant = plant_instance_for_layer(W_layer, int(config.seed) + int(row_index))
        evidence = simulate_primitive_rollout(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w01_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=rollout_config,
            wind_field=wind_field,
            implementation_instance=implementation,
            plant_instance=plant,
            controller=controller,
            controller_selection_status="W01_variant_registry_candidate",
            candidate_index=candidate_index,
            candidate_weight_label=str(weight_label),
        )
        row = rollout_evidence_row(evidence)
        row["implementation_instance_id"] = implementation.implementation_instance_id
        row["plant_instance_id"] = plant.plant_instance_id

    row.update(archive_state_sample_row(sample))
    row.update(_variant_prefix_row(variant))
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    row.update({f"surrogate_{key}": value for key, value in surrogate_binding_row(binding).items()})
    row.update({f"environment_{key}": value for key, value in environment_instance_row(environment).items()})
    if compatible and binding.surrogate_binding_status == "ready":
        row.update({f"implementation_{key}": value for key, value in implementation_instance_row(implementation).items()})
        row.update({f"plant_{key}": value for key, value in plant_instance_row(plant).items()})
        implementation_audit_status = "applied_in_rollout"
        plant_audit_status = "applied_in_rollout"
    else:
        row.update(_blocked_instance_prefix_rows("implementation"))
        row.update(_blocked_instance_prefix_rows("plant"))
        implementation_audit_status = "not_applied_blocked_before_rollout"
        plant_audit_status = "not_applied_blocked_before_rollout"
    row.update(
        {
            "runner_version": W01_RUNNER_VERSION,
            "run_stage": "W01_dense_primitive_variant_generation",
            "row_index": int(row_index),
            "schedule_mode": schedule.schedule_mode,
            "paired_start_policy": "common_random_start_key_reused_across_primitives_candidates_and_w01_environments",
            "schedule_paired_start_index": int(schedule.paired_start_index),
            "primitive_variant_id": variant.primitive_variant_id,
            "entry_role": variant.entry_role,
            "entry_role_compatible": bool(compatible),
            "candidate_index": int(candidate_index),
            "candidate_weight_label": weight_label,
            "environment_mode": environment_mode,
            "environment_instance_id": environment.environment_id,
            "surrogate_blocked_reason": binding.blocked_reason,
            "implementation_instance_status": implementation_audit_status,
            "plant_instance_status": plant_audit_status,
            "W_layer_official_role": "W0_dry_air" if W_layer == "W0" else "W1_gaussian_preflight",
            "small_library_selection_allowed": False,
            "clustering_before_w2_w3_allowed": False,
            "w2_w3_replay_only": True,
            "panelwise_glider_dynamics_active": True,
            "state_feedback_latency_lag_active": True,
            "command_timing_active": True,
            "actuator_lag_active": True,
            "pd_pid_fallback_allowed": False,
            "timing_aware_synthesis_level": controller.timing_aware_synthesis_level,
            "timing_effects_in_synthesis": controller.timing_effects_in_synthesis,
            "timing_effects_in_rollout": controller.timing_effects_in_rollout,
            "sampled_data_timing_audit_status": controller.sampled_data_timing_audit_status,
            "delayed_state_lqr_augmentation_status": controller.delayed_state_lqr_augmentation_status,
            "active_timing_aware_controller_used": controller_is_active_timing_aware_w01(controller),
            "baseline_controller_active": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return row


def _row_schedule_for_index(row_index: int, config: W01DenseRunConfig) -> W01RowSchedule:
    if str(config.schedule_mode) == SMOKE_SCHEDULE_MODE:
        primitive_id = ACTIVE_PRIMITIVE_IDS[int(row_index) % len(ACTIVE_PRIMITIVE_IDS)]
        candidate_index = (int(row_index) // len(ACTIVE_PRIMITIVE_IDS)) % int(config.candidate_count)
        W_layer, environment_mode = OFFICIAL_W01_ENVIRONMENT_CASES[int(row_index) % len(OFFICIAL_W01_ENVIRONMENT_CASES)]
        family = start_state_family_for_row(row_index)
        return W01RowSchedule(
            row_index=int(row_index),
            primitive_id=str(primitive_id),
            candidate_index=int(candidate_index),
            W_layer=str(W_layer),
            environment_mode=str(environment_mode),
            start_state_family=str(family),
            paired_start_key=f"smoke_start_{int(row_index) // 2:07d}",
            paired_start_index=int(row_index) // 2,
            schedule_mode=SMOKE_SCHEDULE_MODE,
        )

    if config.paired_tests_per_candidate is not None:
        primitive_index = int(row_index) % len(ACTIVE_PRIMITIVE_IDS)
        primitive_id = ACTIVE_PRIMITIVE_IDS[primitive_index]
        grouped_index = int(row_index) // len(ACTIVE_PRIMITIVE_IDS)
        candidate_environment_count = int(config.candidate_count) * len(OFFICIAL_W01_ENVIRONMENT_CASES)
        paired_start_index = grouped_index // max(1, candidate_environment_count)
        candidate_environment_index = grouped_index % max(1, candidate_environment_count)
        candidate_index = candidate_environment_index // len(OFFICIAL_W01_ENVIRONMENT_CASES)
        W_layer, environment_mode = OFFICIAL_W01_ENVIRONMENT_CASES[
            candidate_environment_index % len(OFFICIAL_W01_ENVIRONMENT_CASES)
        ]
        family = start_state_family_for_row(paired_start_index)
        return W01RowSchedule(
            row_index=int(row_index),
            primitive_id=str(primitive_id),
            candidate_index=int(candidate_index),
            W_layer=str(W_layer),
            environment_mode=str(environment_mode),
            start_state_family=str(family),
            paired_start_key=f"paired_{int(paired_start_index):07d}_{family}",
            paired_start_index=int(paired_start_index),
            schedule_mode=BALANCED_SCHEDULE_MODE,
        )

    primitive_index = int(row_index) % len(ACTIVE_PRIMITIVE_IDS)
    primitive_id = ACTIVE_PRIMITIVE_IDS[primitive_index]
    grouped_index = int(row_index) // len(ACTIVE_PRIMITIVE_IDS)
    candidate_index = grouped_index % int(config.candidate_count)
    W_layer, environment_mode = OFFICIAL_W01_ENVIRONMENT_CASES[grouped_index % len(OFFICIAL_W01_ENVIRONMENT_CASES)]
    family = start_state_family_for_row(row_index)
    full_cycle = len(ACTIVE_PRIMITIVE_IDS) * int(config.candidate_count) * len(OFFICIAL_W01_ENVIRONMENT_CASES)
    cycle_index = int(row_index) // max(1, full_cycle)
    family_slot = int(row_index) % 20
    paired_start_index = int(cycle_index * 20 + family_slot)
    paired_start_key = f"paired_{paired_start_index:07d}_{family}"
    return W01RowSchedule(
        row_index=int(row_index),
        primitive_id=str(primitive_id),
        candidate_index=int(candidate_index),
        W_layer=str(W_layer),
        environment_mode=str(environment_mode),
        start_state_family=str(family),
        paired_start_key=paired_start_key,
        paired_start_index=paired_start_index,
        schedule_mode=BALANCED_SCHEDULE_MODE,
    )


def _blocked_instance_prefix_rows(prefix: str) -> dict[str, object]:
    if prefix == "implementation":
        return {
            "implementation_implementation_instance_id": "",
            "implementation_W_layer": "",
            "implementation_latency_case": "",
            "implementation_implementation_adjustment_status": "not_applied_blocked_before_rollout",
            "implementation_implementation_adjustment_limitations": "blocked_before_rollout_simulation",
        }
    return {
        "plant_plant_instance_id": "",
        "plant_W_layer": "",
        "plant_plant_adjustment_status": "not_applied_blocked_before_rollout",
        "plant_plant_adjustment_limitations": "blocked_before_rollout_simulation",
    }


def _variant_prefix_row(variant: PrimitiveControllerVariant) -> dict[str, object]:
    return {f"variant_{key}": value for key, value in variant_row(variant).items()}


def _chunk_schedule(config: W01DenseRunConfig, *, storage_format: str) -> list[W01ChunkSpec]:
    chunk_count = int(math.ceil(int(config.rows) / int(config.candidate_chunk_size)))
    chunks: list[W01ChunkSpec] = []
    for chunk_index in range(chunk_count):
        row_start = chunk_index * int(config.candidate_chunk_size)
        row_stop = min(int(config.rows), row_start + int(config.candidate_chunk_size))
        chunks.append(
            W01ChunkSpec(
                run_id=int(config.run_id),
                chunk_index=int(chunk_index),
                chunk_count=int(chunk_count),
                row_start=int(row_start),
                row_stop=int(row_stop),
                storage_format=storage_format,
                compression_level=int(config.compression_level),
            )
        )
    return chunks


def _existing_chunk_status(chunk: W01ChunkSpec, *, run_root: Path) -> str:
    partition_path = _partition_path(chunk, run_root)
    manifest_path = _chunk_manifest_path(chunk, run_root)
    partition_exists = filesystem_path(partition_path).is_file()
    manifest_exists = filesystem_path(manifest_path).is_file()
    if not partition_exists and not manifest_exists:
        return "missing"
    if partition_exists != manifest_exists:
        return "corrupt"
    try:
        manifest = json.loads(filesystem_path(manifest_path).read_text(encoding="ascii"))
        frame = read_table_partition(partition_path, storage_format=chunk.storage_format)
        if manifest.get("status") != "complete":
            return "corrupt"
        if int(manifest.get("row_count", -1)) != len(frame):
            return "corrupt"
        if str(manifest.get("checksum_sha256", "")) != file_sha256(partition_path):
            return "corrupt"
    except Exception:
        return "corrupt"
    return "complete"


def _remove_chunk_files(chunk: W01ChunkSpec, run_root: Path) -> None:
    for path in (_partition_path(chunk, run_root), _chunk_manifest_path(chunk, run_root)):
        fs_path = filesystem_path(path)
        if fs_path.exists():
            fs_path.unlink()


def _partition_from_existing(chunk: W01ChunkSpec, run_root: Path):
    manifest = json.loads(filesystem_path(_chunk_manifest_path(chunk, run_root)).read_text(encoding="ascii"))
    from dense_archive_table_io import TablePartition

    frame = read_table_partition(_partition_path(chunk, run_root), storage_format=chunk.storage_format)
    return TablePartition(
        table_name=W01_TABLE_NAME,
        relative_path=str(manifest["partition_path"]),
        storage_format=chunk.storage_format,
        row_count=len(frame),
        byte_count=int(filesystem_path(_partition_path(chunk, run_root)).stat().st_size),
        columns=tuple(str(column) for column in frame.columns),
        checksum_sha256=file_sha256(_partition_path(chunk, run_root)),
    )


def _partition_path(chunk: W01ChunkSpec, run_root: Path) -> Path:
    return run_root / "tables" / W01_TABLE_NAME / f"c{int(chunk.chunk_index):05d}.{table_extension(chunk.storage_format)}"


def _chunk_manifest_path(chunk: W01ChunkSpec, run_root: Path) -> Path:
    return run_root / "chunk_manifests" / W01_TABLE_NAME / f"c{int(chunk.chunk_index):05d}.json"


def _scheduled_chunk_row(chunk: W01ChunkSpec, run_root: Path) -> dict[str, object]:
    return {
        "chunk_index": int(chunk.chunk_index),
        "status": "scheduled",
        "row_start": int(chunk.row_start),
        "row_stop": int(chunk.row_stop),
        "row_count": int(chunk.row_stop - chunk.row_start),
        "partition_path": _partition_path(chunk, run_root).relative_to(run_root).as_posix(),
        "manifest_path": _chunk_manifest_path(chunk, run_root).relative_to(run_root).as_posix(),
        "byte_count": 0,
        "checksum_sha256": "",
        "duration_s": 0.0,
    }


def _completed_chunk_row(chunk: W01ChunkSpec, run_root: Path, *, status: str) -> dict[str, object]:
    manifest = json.loads(filesystem_path(_chunk_manifest_path(chunk, run_root)).read_text(encoding="ascii"))
    return {
        "chunk_index": int(chunk.chunk_index),
        "status": status,
        "row_start": int(manifest["row_start"]),
        "row_stop": int(manifest["row_stop"]),
        "row_count": int(manifest["row_count"]),
        "partition_path": str(manifest["partition_path"]),
        "manifest_path": _chunk_manifest_path(chunk, run_root).relative_to(run_root).as_posix(),
        "byte_count": int(manifest["byte_count"]),
        "checksum_sha256": str(manifest["checksum_sha256"]),
        "duration_s": 0.0,
    }


def _write_dense_metrics_from_partitions(run_root: Path, partitions: Iterable[object], storage_format: str) -> int:
    counters: dict[str, Counter] = defaultdict(Counter)
    grouped: dict[str, Counter] = defaultdict(Counter)
    row_count = 0
    coverage_columns = (
        "primitive_id",
        "entry_role",
        "candidate_index",
        "start_state_family",
        "W_layer",
        "environment_mode",
        "lqr_synthesis_status",
        "controller_design_role",
        "timing_state_source",
        "active_timing_aware_controller_used",
        "baseline_controller_active",
        "outcome_class",
        "boundary_use_class",
    )
    for partition in partitions:
        frame = read_table_partition(run_root / "tables" / partition.relative_path, storage_format=storage_format)
        row_count += int(len(frame))
        for column in ("outcome_class", "failure_label", "boundary_use_class"):
            _counter_update(counters[column], frame, column)
        for column in coverage_columns:
            _counter_update(counters[f"coverage:{column}"], frame, column)
        _group_counter_update(
            grouped["variant_synthesis"],
            frame,
            (
                "primitive_id",
                "entry_role",
                "candidate_index",
                "controller_design_role",
                "timing_augmentation_type",
                "timing_design_version",
                "lqr_synthesis_status",
                "sampled_data_check_status",
                "sampled_data_timing_audit_status",
            ),
        )
        _group_counter_update(
            grouped["start_family"],
            frame,
            ("start_state_family", "primitive_id", "entry_role", "W_layer"),
        )
        _group_counter_update(grouped["environment"], frame, ("W_layer", "environment_mode"))
        _group_counter_update(
            grouped["entry_role_rejection"],
            frame,
            ("entry_role", "start_state_family", "entry_check_status", "failure_label"),
        )
        _group_counter_update(
            grouped["entry_role_compatibility"],
            frame,
            ("primitive_id", "entry_role", "entry_role_compatible"),
        )
        _group_counter_update(
            grouped["boundary_use"],
            frame,
            ("boundary_use_class", "continuation_valid", "episode_terminal_useful", "outcome_class"),
        )

    _counter_frame(counters["outcome_class"]).to_csv(filesystem_path(run_root / "metrics" / "outcome_summary.csv"), index=False)
    _counter_frame(counters["failure_label"]).to_csv(filesystem_path(run_root / "metrics" / "failure_summary.csv"), index=False)
    _counter_frame(counters["boundary_use_class"]).to_csv(filesystem_path(run_root / "metrics" / "boundary_summary.csv"), index=False)
    _counter_frame(counters["coverage:timing_state_source"]).to_csv(
        filesystem_path(run_root / "metrics" / "timing_state_summary.csv"),
        index=False,
    )
    coverage_rows = []
    for column in coverage_columns:
        coverage_rows.extend(
            _counter_frame(counters[f"coverage:{column}"]).assign(coverage_axis=column).to_dict(orient="records")
        )
    pd.DataFrame(coverage_rows).to_csv(filesystem_path(run_root / "metrics" / "coverage_summary.csv"), index=False)
    _group_counter_frame(
        grouped["variant_synthesis"],
        (
            "primitive_id",
            "entry_role",
            "candidate_index",
            "controller_design_role",
            "timing_augmentation_type",
            "timing_design_version",
            "lqr_synthesis_status",
            "sampled_data_check_status",
            "sampled_data_timing_audit_status",
        ),
    ).to_csv(filesystem_path(run_root / "metrics" / "variant_synthesis_summary.csv"), index=False)
    _group_counter_frame(
        grouped["start_family"],
        ("start_state_family", "primitive_id", "entry_role", "W_layer"),
    ).to_csv(filesystem_path(run_root / "metrics" / "start_family_coverage.csv"), index=False)
    _group_counter_frame(grouped["environment"], ("W_layer", "environment_mode")).to_csv(
        filesystem_path(run_root / "metrics" / "environment_coverage.csv"),
        index=False,
    )
    rejection = _entry_role_rejection_summary(grouped["entry_role_rejection"])
    rejection.to_csv(filesystem_path(run_root / "metrics" / "entry_role_rejection_summary.csv"), index=False)
    _group_counter_frame(
        grouped["entry_role_compatibility"],
        ("primitive_id", "entry_role", "entry_role_compatible"),
    ).to_csv(filesystem_path(run_root / "metrics" / "entry_role_compatibility_by_primitive.csv"), index=False)
    _group_counter_frame(
        grouped["boundary_use"],
        ("boundary_use_class", "continuation_valid", "episode_terminal_useful", "outcome_class"),
    ).to_csv(filesystem_path(run_root / "metrics" / "boundary_use_summary.csv"), index=False)
    return int(row_count)


def _write_empty_metrics(run_root: Path) -> None:
    for name in (
        "outcome_summary.csv",
        "coverage_summary.csv",
        "failure_summary.csv",
        "boundary_summary.csv",
        "variant_synthesis_summary.csv",
        "start_family_coverage.csv",
        "environment_coverage.csv",
        "entry_role_rejection_summary.csv",
        "entry_role_compatibility_by_primitive.csv",
        "boundary_use_summary.csv",
        "timing_state_summary.csv",
    ):
        pd.DataFrame().to_csv(filesystem_path(run_root / "metrics" / name), index=False)


def _value_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame([{"value": "missing_or_empty", "row_count": 0}])
    counts = frame[column].fillna("").astype(str).value_counts(dropna=False)
    return pd.DataFrame(
        [{"value": str(value), "row_count": int(count)} for value, count in counts.items()]
    )


def _counter_update(counter: Counter, frame: pd.DataFrame, column: str) -> None:
    if frame.empty or column not in frame.columns:
        counter["missing_or_empty"] += 0
        return
    counter.update(frame[column].fillna("").astype(str).tolist())


def _counter_frame(counter: Counter) -> pd.DataFrame:
    if not counter:
        return pd.DataFrame([{"value": "missing_or_empty", "row_count": 0}])
    return pd.DataFrame(
        [{"value": str(value), "row_count": int(count)} for value, count in counter.items()]
    ).sort_values(["value"]).reset_index(drop=True)


def _group_counter_update(counter: Counter, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if frame.empty or any(column not in frame.columns for column in columns):
        return
    values = frame.loc[:, list(columns)].fillna("").astype(str)
    counter.update(tuple(row) for row in values.itertuples(index=False, name=None))


def _group_counter_frame(counter: Counter, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key, count in counter.items():
        rows.append({**{column: key[index] for index, column in enumerate(columns)}, "row_count": int(count)})
    if not rows:
        return pd.DataFrame([{**{column: "" for column in columns}, "row_count": 0}])
    return pd.DataFrame(rows).sort_values(list(columns)).reset_index(drop=True)


def _entry_role_rejection_summary(counter: Counter) -> pd.DataFrame:
    rows = []
    totals: Counter = Counter()
    rejected: Counter = Counter()
    for key, count in counter.items():
        entry_role, start_family, entry_status, failure_label = key
        group = (entry_role, start_family)
        totals[group] += int(count)
        if entry_status == "entry_role_incompatible_start" or failure_label == "entry_role_not_launch_capable":
            rejected[group] += int(count)
    for (entry_role, start_family), total in totals.items():
        rejection_count = int(rejected[(entry_role, start_family)])
        rows.append(
            {
                "entry_role": entry_role,
                "start_state_family": start_family,
                "row_count": int(total),
                "entry_role_rejection_count": rejection_count,
                "entry_role_rejection_rate": float(rejection_count) / float(total) if total else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(
            [
                {
                    "entry_role": "",
                    "start_state_family": "",
                    "row_count": 0,
                    "entry_role_rejection_count": 0,
                    "entry_role_rejection_rate": 0.0,
                }
            ]
        )
    return pd.DataFrame(rows).sort_values(["entry_role", "start_state_family"]).reset_index(drop=True)


def _write_runtime_summary(
    run_root: Path,
    config: W01DenseRunConfig,
    *,
    row_count: int,
    status: str,
    started: float,
    ended: float,
) -> None:
    pd.DataFrame(
        [
            {
                "run_id": int(config.run_id),
                "status": status,
                "rows_requested": int(config.rows),
                "row_count": int(row_count),
                "duration_s": float(ended - started),
                "runtime_storage_contract": RUNTIME_STORAGE_CONTRACT,
                "max_generated_file_size_mb": float(MAX_GENERATED_FILE_SIZE_MB),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ]
    ).to_csv(filesystem_path(run_root / "metrics" / "runtime_summary.csv"), index=False)


def _write_chunk_summary(run_root: Path, records: list[dict[str, object]]) -> None:
    pd.DataFrame(records).to_csv(filesystem_path(run_root / "metrics" / "chunk_summary.csv"), index=False)


def _write_file_size_audit(run_root: Path) -> None:
    rows = []
    for path in sorted(filesystem_path(run_root).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(filesystem_path(run_root)).as_posix()
        byte_count = int(path.stat().st_size)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": float(byte_count) / (1024.0 * 1024.0),
                "above_75mb": bool(byte_count > 75.0 * 1024.0 * 1024.0),
                "above_100mb": bool(byte_count > MAX_GENERATED_FILE_SIZE_MB * 1024.0 * 1024.0),
                "push_allowed": bool(byte_count <= MAX_GENERATED_FILE_SIZE_MB * 1024.0 * 1024.0),
                "dense_table_partition": rel.startswith(f"tables/{W01_TABLE_NAME}/"),
            }
        )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def _write_reports(run_root: Path, *, status: str, row_count: int) -> None:
    blocked_claims = "\n".join(f"- `{claim}`" for claim in BLOCKED_CLAIMS)
    start_family_counts = _start_family_counts_for_report(run_root, row_count=row_count)
    cross_layer_status, cross_layer_blockers = _cross_layer_smoke_status_from_counts(
        start_family_counts,
        row_count=_planned_or_written_row_count(run_root, row_count),
    )
    entry_role_counts = _entry_role_compatibility_counts_for_report(run_root)
    history_backed_fifo_count = _timing_state_count_for_report(run_root, "history_backed_fifo")
    ready_frozen_controller_count = _ready_frozen_controller_count(run_root)
    claim_report = "\n".join(
        [
            "# W01 Claim Boundary",
            "",
            f"- Status: `{status}`",
            f"- Row count: `{int(row_count)}`",
            f"- Claim boundary: `{CLAIM_BOUNDARY}`",
            "",
            "Blocked claims:",
            "",
            blocked_claims,
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "claim_boundary_report.md").write_text(claim_report, encoding="ascii")
    run_class, move_on_blockers = _l6_move_on_status(run_root=run_root, status=status, row_count=row_count)
    run_report = "\n".join(
        [
            "# W01 Dense Preflight Readiness Run",
            "",
            f"- Status: `{status}`",
            f"- Rows written: `{int(row_count)}`",
            f"- Run class: `{run_class}`",
            f"- Cross-layer smoke status: `{cross_layer_status}`",
            f"- Start-family counts: `{json.dumps(start_family_counts, sort_keys=True)}`",
            f"- Start-family mix exact or blocked: `{not cross_layer_blockers}`",
            f"- Entry-role compatibility by primitive: `{json.dumps(entry_role_counts, sort_keys=True)}`",
            f"- History-backed FIFO count: `{history_backed_fifo_count}`",
            f"- Ready frozen controller count: `{ready_frozen_controller_count}`",
            "- Retired controller-picking, compact-library, integration, hardware, transfer, and mission-success claims remain blocked.",
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "run_report.md").write_text(run_report, encoding="ascii")
    l6_report = "\n".join(
        [
            "# L6 Move-On Check",
            "",
            f"- Status: `{status}`",
            f"- Run class: `{run_class}`",
            f"- Rows written: `{int(row_count)}`",
            "- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`",
            "- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`",
            "- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`",
            "- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`",
            f"- Cross-layer smoke status: `{cross_layer_status}`",
            f"- Start-family counts: `{json.dumps(start_family_counts, sort_keys=True)}`",
            f"- Start-family mix exact or blocked: `{not cross_layer_blockers}`",
            f"- Entry-role compatibility by primitive: `{json.dumps(entry_role_counts, sort_keys=True)}`",
            f"- History-backed FIFO count: `{history_backed_fifo_count}`",
            f"- Ready frozen controller count: `{ready_frozen_controller_count}`",
            f"- Rich-side W01 fixed-library cleared for W2 planning: `{not move_on_blockers and run_class == 'rich_side_l6_candidate'}`",
            "",
            "Blockers before heavy W01:",
            "",
            *[f"- `{item}`" for item in (move_on_blockers + cross_layer_blockers or ["none"])],
            "",
            "Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.",
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "l6_move_on_check.md").write_text(l6_report, encoding="ascii")
    timing_report = "\n".join(
        [
            "# Timing Synthesis Boundary",
            "",
            f"- Project title version: `{PROJECT_TITLE_VERSION}`",
            "- Active W01 controller: `predictor_compensated_augmented_discrete_lqr_v1`",
            "- Augmentation: actuator surface states and command-delay FIFO states in a discrete-time LQR model.",
            "- State-feedback delay treatment: predictor compensation using the local reduced-order model and nominal delay horizon.",
            "- Rollout timing: panel-wise wind, feedback latency, command timing, actuator lag, implementation instance, and plant instance remain applied in evidence rows.",
            "- Not claimed: true full delayed-state-feedback augmentation, hardware readiness, transfer, or mission success.",
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "timing_synthesis_boundary.md").write_text(timing_report, encoding="ascii")
    _write_timing_contract_audit(run_root=run_root, status=status, row_count=row_count)
    _write_l7_completeness_audit(
        run_root=run_root,
        status=status,
        row_count=row_count,
        run_class=run_class,
        move_on_blockers=move_on_blockers,
    )


def _write_l7_completeness_audit(
    *,
    run_root: Path,
    status: str,
    row_count: int,
    run_class: str,
    move_on_blockers: list[str],
) -> None:
    manifest_path = filesystem_path(run_root / "manifests" / "run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="ascii")) if manifest_path.is_file() else {}
    largest = _largest_file_audit_row(run_root)
    primitive_counts = _coverage_counts(run_root, "primitive_id")
    candidate_counts = _coverage_counts(run_root, "candidate_index")
    start_counts = _coverage_counts(run_root, "start_state_family")
    environment_counts = _coverage_counts(run_root, "environment_mode")
    boundary_counts = _coverage_counts(run_root, "boundary_use_class")
    synthesis_rows = _read_metric_preview(run_root / "metrics" / "variant_synthesis_summary.csv", limit=12)
    cleared = bool(not move_on_blockers and run_class == "rich_side_l6_candidate")
    report = "\n".join(
        [
            "# L7 W01 Completeness Audit",
            "",
            f"- Status: `{status}`",
            f"- Run class: `{run_class}`",
            f"- Rows written: `{int(row_count)}`",
            f"- Worker count: `{manifest.get('selected_worker_count', '')}`",
            f"- Chunk count: `{manifest.get('chunk_count', '')}`",
            f"- Chunk size: `{manifest.get('candidate_chunk_size', '')}`",
            f"- Storage format: `{manifest.get('storage_format', '')}`",
            f"- Candidate count requested: `{manifest.get('candidate_count', '')}`",
            f"- Paired tests per candidate: `{manifest.get('paired_tests_per_candidate', '')}`",
            f"- Largest file: `{largest.get('relative_path', '')}` at `{largest.get('size_mb', '')}` MB",
            f"- Above 75 MB present: `{largest.get('above_75mb', '')}`",
            f"- Above 100 MB present: `{largest.get('above_100mb', '')}`",
            f"- W1 single/four mixed in one root: `{ {'gaussian_single', 'gaussian_four'}.issubset(set(environment_counts)) }`",
            f"- Fixed W01 library cleared for future W2 fixed-LQR replay: `{cleared}`",
            "",
            "Coverage summaries:",
            "",
            f"- Primitives: `{json.dumps(primitive_counts, sort_keys=True)}`",
            f"- Candidate indices present: `{len(candidate_counts)}`",
            f"- Start families: `{json.dumps(start_counts, sort_keys=True)}`",
            f"- Environments: `{json.dumps(environment_counts, sort_keys=True)}`",
            f"- Boundary use: `{json.dumps(boundary_counts, sort_keys=True)}`",
            "",
            "Timing-aware synthesis preview:",
            "",
            *[f"- `{row}`" for row in synthesis_rows],
            "",
            "Blockers:",
            "",
            *[f"- `{item}`" for item in (move_on_blockers or ["none"])],
            "",
            "Blocked claims remain W2 execution, W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.",
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "l7_w01_completeness_audit.md").write_text(report, encoding="ascii")


def _planned_or_written_row_count(run_root: Path, row_count: int) -> int:
    if int(row_count) > 0:
        return int(row_count)
    manifest_path = filesystem_path(run_root / "manifests" / "run_manifest.json")
    if not manifest_path.is_file():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        return int(manifest.get("rows_requested", 0))
    except Exception:
        return 0


def _start_family_counts_for_report(run_root: Path, *, row_count: int) -> dict[str, int]:
    counts = _coverage_counts(run_root, "start_state_family") if int(row_count) > 0 else {}
    if counts:
        return counts
    manifest_path = filesystem_path(run_root / "manifests" / "run_manifest.json")
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        return {str(key): int(value) for key, value in dict(manifest.get("per_start_family_row_counts", {})).items()}
    except Exception:
        return {}


def _cross_layer_smoke_status_from_counts(start_family_counts: dict[str, int], *, row_count: int) -> tuple[str, list[str]]:
    blockers: list[str] = []
    expected = {
        family: int(round(float(row_count) * proportion))
        for family, proportion in START_FAMILY_MIX.items()
    }
    missing = [family for family in START_FAMILY_MIX if int(start_family_counts.get(family, 0)) <= 0]
    if missing:
        blockers.append("cross_layer_smoke_missing_start_families:" + ",".join(missing))
    if int(row_count) > 0 and start_family_counts != expected:
        blockers.append("cross_layer_start_family_mix_not_exact_40_25_15_10_10")
    status = CROSS_LAYER_SMOKE_READY if not blockers else CROSS_LAYER_SMOKE_INCOMPLETE
    return status, blockers


def _entry_role_compatibility_counts_for_report(run_root: Path) -> dict[str, dict[str, int]]:
    path = filesystem_path(run_root / "metrics" / "entry_role_compatibility_by_primitive.csv")
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    rows: dict[str, dict[str, int]] = {}
    for row in frame.to_dict(orient="records"):
        primitive_id = str(row.get("primitive_id", ""))
        compatible = str(row.get("entry_role_compatible", "")).lower() in {"true", "1"}
        key = "compatible" if compatible else "incompatible"
        rows.setdefault(primitive_id, {"compatible": 0, "incompatible": 0})
        rows[primitive_id][key] += int(row.get("row_count", 0))
    return rows


def _timing_state_count_for_report(run_root: Path, value: str) -> int:
    path = filesystem_path(run_root / "metrics" / "timing_state_summary.csv")
    if not path.is_file():
        return 0
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return 0
    if frame.empty or "value" not in frame.columns or "row_count" not in frame.columns:
        return 0
    rows = frame[frame["value"].astype(str) == str(value)]
    return int(rows["row_count"].astype(int).sum()) if not rows.empty else 0


def _ready_frozen_controller_count(run_root: Path) -> int:
    bundle_path = filesystem_path(run_root / "manifests" / "frozen_w01_controller_bundle.json")
    if not bundle_path.is_file():
        return 0
    try:
        bundle = load_frozen_w01_controller_bundle(bundle_path)
    except Exception:
        return 0
    return int(bundle.ready_count)


def _write_timing_contract_audit(*, run_root: Path, status: str, row_count: int) -> None:
    registry_path = filesystem_path(run_root / "metrics" / "primitive_variant_registry.csv")
    timing_state_path = filesystem_path(run_root / "metrics" / "timing_state_summary.csv")
    registry = _read_csv_or_empty(registry_path)
    timing_state = _read_csv_or_empty(timing_state_path)
    manifest_path = filesystem_path(run_root / "manifests" / "run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="ascii")) if manifest_path.is_file() else {}
    latency_case = str(manifest.get("latency_case", "nominal"))
    latency = latency_case_config(latency_case)
    timing_state_sources = (
        ",".join(sorted(timing_state["value"].dropna().astype(str).unique()))
        if not timing_state.empty and "value" in timing_state.columns
        else ""
    )
    active = registry[registry["controller_design_role"].astype(str) == ACTIVE_TIMING_AWARE_ROLE] if not registry.empty and "controller_design_role" in registry.columns else pd.DataFrame()
    row = {
        "status": status,
        "row_count": int(row_count),
        "active_timing_aware_variant_count": int(len(active)),
        "state_feedback_delay_s_values": _unique_csv(active, "state_feedback_delay_s"),
        "vicon_latency_nominal_s": float(DEFAULT_LATENCY_ENVELOPE.vicon_latency_nominal_s),
        "vicon_filter_delay_s": float(DEFAULT_LATENCY_ENVELOPE.vicon_filter_delay_s),
        "command_delay_s_values": _unique_csv(active, "command_delay_s"),
        "command_onset_delay_s": float(latency.command_onset_delay_s),
        "command_transport_delay_s": float(latency.command_transport_delay_s),
        "actuator_tau_s_values": _unique_csv(active, "actuator_tau_s"),
        "actuator_t50_s": float(latency.actuator_t50_s),
        "actuator_t90_s": float(latency.actuator_t90_s),
        "latency_jitter_s": float(latency.latency_jitter_s),
        "command_delay_steps_values": _unique_csv(active, "command_delay_steps"),
        "predictor_horizon_steps_values": _unique_csv(active, "predictor_horizon_steps"),
        "timing_state_sources": timing_state_sources,
        "history_backed_fifo_count": _timing_state_count_for_report(run_root, "history_backed_fifo"),
        "latency_case": latency_case,
        "state_delay_affects_predictor_horizon": bool(latency.state_feedback_delay_s > 0.0),
        "command_delay_affects_fifo_length": bool(latency.command_onset_delay_s + latency.command_transport_delay_s > 0.0),
        "actuator_lag_affects_rollout_state_propagation": bool(max(latency.actuator_tau_s) > 0.0),
        "W3_randomisation_changes_timing_terms_where_configured": "not_applicable_W01",
        "history_backed_fifo_required_for_l6": True,
        "full_delayed_state_feedback_claim": False,
    }
    pd.DataFrame([row]).to_csv(filesystem_path(run_root / "metrics" / "timing_contract_audit.csv"), index=False)
    report = "\n".join(
        [
            "# Timing Contract Audit",
            "",
            f"- Status: `{status}`",
            f"- Rows written: `{int(row_count)}`",
            f"- Active timing-aware variants: `{row['active_timing_aware_variant_count']}`",
            f"- State feedback delay values: `{row['state_feedback_delay_s_values']}`",
            f"- Vicon nominal latency: `{row['vicon_latency_nominal_s']}`",
            f"- Vicon/filter delay: `{row['vicon_filter_delay_s']}`",
            f"- Command delay values: `{row['command_delay_s_values']}`",
            f"- Command onset delay: `{row['command_onset_delay_s']}`",
            f"- Command transport delay: `{row['command_transport_delay_s']}`",
            f"- Actuator tau values: `{row['actuator_tau_s_values']}`",
            f"- Actuator t50/t90: `{row['actuator_t50_s']}` / `{row['actuator_t90_s']}`",
            f"- Latency jitter: `{row['latency_jitter_s']}`",
            f"- Command FIFO lengths: `{row['command_delay_steps_values']}`",
            f"- Predictor horizons: `{row['predictor_horizon_steps_values']}`",
            f"- Timing state sources: `{timing_state_sources}`",
            f"- History-backed FIFO count: `{row['history_backed_fifo_count']}`",
            f"- State delay affects predictor horizon: `{row['state_delay_affects_predictor_horizon']}`",
            f"- Command delay affects FIFO length: `{row['command_delay_affects_fifo_length']}`",
            f"- Actuator lag affects rollout state propagation: `{row['actuator_lag_affects_rollout_state_propagation']}`",
            "- Vicon/filter delay, command delay, actuator lag/t50/t90, and jitter are represented by the active latency/implementation instances in rollout evidence.",
            "- Not claimed: true full delayed-state-feedback validation, hardware readiness, transfer, or mission success.",
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "timing_contract_audit.md").write_text(report, encoding="ascii")


def _unique_csv(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    return ",".join(sorted({str(value) for value in frame[column].dropna().tolist()}))


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not filesystem_path(path).is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(filesystem_path(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _largest_file_audit_row(run_root: Path) -> dict[str, object]:
    path = filesystem_path(run_root / "metrics" / "file_size_audit.csv")
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "size_mb" not in frame.columns:
        return {}
    frame["size_mb"] = frame["size_mb"].astype(float)
    row = frame.sort_values("size_mb", ascending=False).iloc[0].to_dict()
    return {str(key): value for key, value in row.items()}


def _read_metric_preview(path: Path, *, limit: int) -> list[str]:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return ["missing"]
    try:
        frame = pd.read_csv(fs_path)
    except pd.errors.EmptyDataError:
        return ["empty"]
    if frame.empty:
        return ["empty"]
    columns = [
        column
        for column in (
            "primitive_id",
            "candidate_index",
            "controller_design_role",
            "lqr_synthesis_status",
            "row_count",
        )
        if column in frame.columns
    ]
    return [
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in frame.loc[:, columns].head(int(limit)).to_dict(orient="records")
    ]


def _l6_move_on_status(*, run_root: Path, status: str, row_count: int) -> tuple[str, list[str]]:
    blockers = _rich_side_gate_blockers(run_root=run_root, row_count=row_count)
    if status == "dry_run_schedule":
        return "dry_run_schedule", ["no_rollout_evidence_written", *blockers]
    if int(row_count) < L6_FALLBACK_ROW_COUNT:
        return "preflight", ["below_19200_fallback_scale_threshold", *blockers]
    if int(row_count) < L6_RICH_SIDE_ROW_COUNT:
        return "fallback_scale_only", ["below_76800_rich_side_threshold", *blockers]
    return "rich_side_l6_candidate", blockers


def _rich_side_gate_blockers(*, run_root: Path, row_count: int) -> list[str]:
    blockers: list[str] = []
    registry_path = filesystem_path(run_root / "metrics" / "primitive_variant_registry.csv")
    if not registry_path.is_file():
        blockers.append("missing_primitive_variant_registry")
    else:
        registry = pd.read_csv(registry_path)
        required_columns = {"primitive_id", "candidate_index", "controller_id", "controller_design_role", "lqr_synthesis_status"}
        if not required_columns.issubset(set(registry.columns)):
            blockers.append("registry_missing_timing_aware_controller_columns")
        else:
            for primitive_id in ACTIVE_PRIMITIVE_IDS:
                primitive_rows = registry[registry["primitive_id"].astype(str) == str(primitive_id)]
                active_rows = primitive_rows[
                    (primitive_rows["controller_design_role"].astype(str) == ACTIVE_TIMING_AWARE_ROLE)
                    & (
                        primitive_rows["lqr_synthesis_status"]
                        .astype(str)
                        .isin(["solved", "blocked_lqr_synthesis"])
                    )
                ]
                if active_rows.empty:
                    blockers.append(f"missing_timing_aware_solved_or_blocked_variant_for_{primitive_id}")
            candidate_values = {
                int(float(value))
                for value in registry["candidate_index"].dropna().tolist()
                if str(value) != ""
            }
            expected_candidates = set(range(L6_RICH_SIDE_CANDIDATE_COUNT))
            if int(row_count) >= L6_RICH_SIDE_ROW_COUNT and candidate_values != expected_candidates:
                blockers.append("registry_missing_32_rich_side_candidate_indices")
    if int(row_count) <= 0:
        return blockers
    summary_path = filesystem_path(run_root / "metrics" / "variant_synthesis_summary.csv")
    if not summary_path.is_file():
        blockers.append("missing_variant_synthesis_summary")
    else:
        summary = pd.read_csv(summary_path)
        if "controller_design_role" not in summary.columns:
            blockers.append("variant_synthesis_summary_missing_controller_design_role")
        else:
            active_count = int(
                summary.loc[
                    summary["controller_design_role"].astype(str) == ACTIVE_TIMING_AWARE_ROLE,
                    "row_count",
                ].sum()
            )
            non_active_count = int(
                summary.loc[
                    summary["controller_design_role"].astype(str) != ACTIVE_TIMING_AWARE_ROLE,
                    "row_count",
                ].sum()
            )
            if active_count <= 0:
                blockers.append("w01_rows_missing_active_timing_aware_controller_ids")
            if non_active_count > 0:
                blockers.append("w01_rows_include_superseded_baseline_controller_ids")
    blockers.extend(_coverage_gate_blockers(run_root=run_root, row_count=row_count))
    blockers.extend(_active_timing_state_gate_blockers(run_root=run_root))
    blockers.extend(_frozen_bundle_gate_blockers(run_root=run_root))
    blockers.extend(_file_size_gate_blockers(run_root=run_root))
    blockers.extend(_manifest_checksum_gate_blockers(run_root=run_root))
    blockers.extend(_contract_audit_gate_blockers())
    return blockers


def _coverage_gate_blockers(*, run_root: Path, row_count: int) -> list[str]:
    blockers: list[str] = []
    primitive_values = set(_coverage_values(run_root, "primitive_id"))
    if primitive_values and primitive_values != set(ACTIVE_PRIMITIVE_IDS):
        blockers.append("missing_active_primitive_coverage")
    candidate_values = {int(value) for value in _coverage_values(run_root, "candidate_index") if str(value).isdigit()}
    if int(row_count) >= L6_RICH_SIDE_ROW_COUNT and candidate_values != set(range(L6_RICH_SIDE_CANDIDATE_COUNT)):
        blockers.append("missing_32_candidate_row_coverage")
    environment_values = set(_coverage_values(run_root, "environment_mode"))
    if not {"dry_air", "gaussian_single", "gaussian_four"}.issubset(environment_values):
        blockers.append("missing_w01_dry_single_four_environment_mix")
    w_values = set(_coverage_values(run_root, "W_layer"))
    if not {"W0", "W1"}.issubset(w_values):
        blockers.append("missing_w0_w1_layer_coverage")
    start_counts = _coverage_counts(run_root, "start_state_family")
    expected = {
        family: int(round(float(row_count) * proportion))
        for family, proportion in START_FAMILY_MIX.items()
    }
    if int(row_count) > 0:
        _, cross_layer_blockers = _cross_layer_smoke_status_from_counts(start_counts, row_count=int(row_count))
        blockers.extend(cross_layer_blockers)
    if int(row_count) >= L6_RICH_SIDE_ROW_COUNT and start_counts != expected:
        blockers.append("start_family_mix_not_exact_40_25_15_10_10")
    return blockers


def _coverage_values(run_root: Path, axis: str) -> list[str]:
    return list(_coverage_counts(run_root, axis).keys())


def _coverage_counts(run_root: Path, axis: str) -> dict[str, int]:
    path = filesystem_path(run_root / "metrics" / "coverage_summary.csv")
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if frame.empty or "coverage_axis" not in frame.columns:
        return {}
    rows = frame[frame["coverage_axis"].astype(str) == str(axis)]
    return {
        str(row["value"]): int(row["row_count"])
        for row in rows.to_dict(orient="records")
        if str(row["value"]) != "missing_or_empty"
    }


def _active_timing_state_gate_blockers(*, run_root: Path) -> list[str]:
    manifest_path = filesystem_path(run_root / "manifests" / "table_manifest.json")
    if not manifest_path.is_file():
        return ["missing_table_manifest_for_timing_state_gate"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        storage_format = str(manifest["storage_format"])
        non_history_count = 0
        active_row_count = 0
        for table in manifest.get("tables", []):
            frame = read_table_partition(
                run_root / "tables" / str(table["relative_path"]),
                storage_format=storage_format,
            )
            if frame.empty or "controller_design_role" not in frame.columns:
                continue
            active = frame[
                (frame["controller_design_role"].astype(str) == ACTIVE_TIMING_AWARE_ROLE)
                & (frame["rollout_backend"].astype(str) == "model_backed_lqr")
            ]
            active_row_count += int(len(active))
            if "timing_state_source" not in active.columns:
                non_history_count += int(len(active))
            else:
                non_history_count += int((active["timing_state_source"].astype(str) != "history_backed_fifo").sum())
    except Exception as exc:
        return [f"timing_state_gate_unreadable:{type(exc).__name__}"]
    blockers = []
    if active_row_count <= 0:
        blockers.append("no_active_model_backed_timing_rows")
    if non_history_count > 0:
        blockers.append("active_timing_rows_not_history_backed_fifo")
    return blockers


def _frozen_bundle_gate_blockers(*, run_root: Path) -> list[str]:
    bundle_path = filesystem_path(run_root / "manifests" / "frozen_w01_controller_bundle.json")
    if not bundle_path.is_file():
        return ["missing_frozen_w01_controller_bundle"]
    try:
        bundle = load_frozen_w01_controller_bundle(bundle_path)
    except Exception as exc:
        return [f"frozen_w01_controller_bundle_unloadable:{type(exc).__name__}"]
    blockers: list[str] = []
    if int(bundle.variant_count) <= 0:
        blockers.append("frozen_w01_controller_bundle_empty")
    if int(bundle.blocked_count) > 0:
        blockers.append("frozen_w01_controller_bundle_has_incomplete_payloads")
    if int(bundle.ready_count) != int(bundle.variant_count):
        blockers.append("frozen_w01_controller_bundle_not_fully_ready")
    for record in bundle.records:
        if record.bundle_status != "ready":
            continue
        controller = record.controller
        if not controller.augmented_gain_matrix_json:
            blockers.append("frozen_bundle_missing_augmented_gain_matrix_json")
            break
        if not controller.predictor_A_reduced_json:
            blockers.append("frozen_bundle_missing_predictor_A_reduced_json")
            break
        if not controller.augmented_A_matrix_json:
            blockers.append("frozen_bundle_missing_augmented_A_matrix_json")
            break
        if not controller.augmented_B_matrix_json:
            blockers.append("frozen_bundle_missing_augmented_B_matrix_json")
            break
    return blockers


def _file_size_gate_blockers(*, run_root: Path) -> list[str]:
    path = filesystem_path(run_root / "metrics" / "file_size_audit.csv")
    if not path.is_file():
        return ["missing_file_size_audit"]
    frame = pd.read_csv(path)
    if "above_100mb" not in frame.columns:
        return ["file_size_audit_missing_above_100mb"]
    if frame["above_100mb"].astype(str).str.lower().isin({"true", "1"}).any():
        return ["file_size_audit_above_100mb"]
    return []


def _manifest_checksum_gate_blockers(*, run_root: Path) -> list[str]:
    manifest_path = filesystem_path(run_root / "manifests" / "table_manifest.json")
    if not manifest_path.is_file():
        return ["missing_table_manifest"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        storage_format = str(manifest["storage_format"])
        for table in manifest.get("tables", []):
            partition_path = run_root / "tables" / str(table["relative_path"])
            if file_sha256(partition_path) != str(table["checksum_sha256"]):
                return ["table_manifest_partition_checksum_mismatch"]
            frame = read_table_partition(partition_path, storage_format=storage_format)
            if int(table["row_count"]) != int(len(frame)):
                return ["table_manifest_partition_row_count_mismatch"]
            chunk_stem = Path(str(table["relative_path"])).name
            if chunk_stem.endswith(".csv.gz"):
                chunk_stem = chunk_stem[: -len(".csv.gz")]
            else:
                chunk_stem = Path(chunk_stem).stem
            chunk_manifest = filesystem_path(run_root / "chunk_manifests" / W01_TABLE_NAME / f"{chunk_stem}.json")
            if not chunk_manifest.is_file():
                return ["missing_chunk_manifest"]
            chunk_payload = json.loads(chunk_manifest.read_text(encoding="ascii"))
            if str(chunk_payload.get("checksum_sha256", "")) != str(table["checksum_sha256"]):
                return ["chunk_manifest_checksum_mismatch"]
    except Exception as exc:
        return [f"manifest_checksum_gate_unreadable:{type(exc).__name__}"]
    return []


def _contract_audit_gate_blockers() -> list[str]:
    try:
        from run_w01_w2_w3_contract_audit import run_w01_w2_w3_contract_audit

        findings = run_w01_w2_w3_contract_audit(Path("."))
    except Exception as exc:
        return [f"contract_audit_not_passed:{type(exc).__name__}"]
    if findings:
        return ["contract_audit_not_passed"]
    return []


def _write_empty_table_manifest(run_root: Path, run_id: int, storage_format: str) -> None:
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(run_id=int(run_id), root=run_root.as_posix(), storage_format=storage_format, tables=()),
    )


def _read_completed_frame(run_root: Path, partitions: Iterable[object], storage_format: str) -> pd.DataFrame:
    frames = [
        read_table_partition(run_root / "tables" / partition.relative_path, storage_format=storage_format)
        for partition in partitions
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_manifest(
    config: W01DenseRunConfig,
    run_root: Path,
    worker_decision,
    storage_format: str,
    schedule: list[W01ChunkSpec],
    variants: tuple[PrimitiveControllerVariant, ...],
) -> dict[str, object]:
    paired_tests = (
        int(config.paired_tests_per_candidate)
        if config.paired_tests_per_candidate is not None
        else max(1, int(config.rows) // max(1, len(variants) * len(OFFICIAL_W01_ENVIRONMENT_CASES)))
    )
    schedule_row = tuning_schedule_row(
        lqr_tuning_schedule(
            candidate_count=int(config.candidate_count),
            paired_tests_per_candidate=paired_tests,
        )
    )
    schedule_counts = _schedule_count_summary(config)
    cross_layer_status, cross_layer_blockers = _cross_layer_smoke_status_from_counts(
        schedule_counts["start_state_family"],
        row_count=int(config.rows),
    )
    return {
        "runner_version": W01_RUNNER_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "run_stage": "W01_dense_primitive_variant_generation",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "rows_requested": int(config.rows),
        "seed": int(config.seed),
        "latency_case": str(config.latency_case),
        "rollout_dt_s": float(config.rollout_dt_s),
        "primitive_timing_contract": primitive_timing_contract_row(),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "candidate_count": int(config.candidate_count),
        "paired_tests_per_candidate": (
            int(config.paired_tests_per_candidate)
            if config.paired_tests_per_candidate is not None
            else paired_tests
        ),
        "rich_side_l6_row_threshold": int(L6_RICH_SIDE_ROW_COUNT),
        "fallback_l6_row_threshold": int(L6_FALLBACK_ROW_COUNT),
        "storage_format": storage_format,
        "compression_level": int(config.compression_level),
        "dry_run_schedule": bool(config.dry_run_schedule),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "selected_worker_count": int(worker_decision.selected_worker_count),
        "worker_decision": asdict(worker_decision),
        "chunk_count": int(len(schedule)),
        "schedule_mode": str(config.schedule_mode),
        "paired_start_policy": "common_random_start_key_reused_across_primitives_candidates_and_w01_environments",
        "per_primitive_row_counts": schedule_counts["primitive_id"],
        "per_candidate_row_counts": schedule_counts["candidate_index"],
        "per_W_layer_row_counts": schedule_counts["W_layer"],
        "per_environment_mode_row_counts": schedule_counts["environment_mode"],
        "per_start_family_row_counts": schedule_counts["start_state_family"],
        "cross_layer_smoke_status": cross_layer_status,
        "cross_layer_smoke_blockers": cross_layer_blockers,
        "cross_layer_minimum_paired_start_cycle": int(CROSS_LAYER_START_FAMILY_CYCLE),
        "start_family_mix_exact_or_blocked": bool(not cross_layer_blockers),
        "official_W_layers": {"W0": ["dry_air"], "W1": ["gaussian_single", "gaussian_four"]},
        "W2_generated": False,
        "W3_generated": False,
        "mixed_primitive_start_sampling": START_FAMILY_MIX,
        "no_small_library_selection": True,
        "no_clustering_before_W2_W3": True,
        "W2_W3_replay_only": True,
        "panelwise_timing_actuator_effects_active_from_W01": True,
        "PD_PID_fallback_allowed": False,
        "timing_aware_synthesis_level": "predictor_compensated_augmented_discrete_lqr",
        "active_controller_design_role": ACTIVE_TIMING_AWARE_ROLE,
        "timing_augmentation_type": "actuator_surface_state_command_fifo_predictor_compensated",
        "timing_effects_in_rollout": "panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances",
        "claim_status": CLAIM_BOUNDARY,
        "blocked_claims": list(BLOCKED_CLAIMS),
        "tuning_schedule": schedule_row,
    }


def _schedule_count_summary(config: W01DenseRunConfig) -> dict[str, dict[str, int]]:
    counters = {
        "primitive_id": Counter(),
        "candidate_index": Counter(),
        "W_layer": Counter(),
        "environment_mode": Counter(),
        "start_state_family": Counter(),
    }
    for row_index in range(int(config.rows)):
        row = _row_schedule_for_index(row_index, config)
        counters["primitive_id"][row.primitive_id] += 1
        counters["candidate_index"][str(row.candidate_index)] += 1
        counters["W_layer"][row.W_layer] += 1
        counters["environment_mode"][row.environment_mode] += 1
        counters["start_state_family"][row.start_state_family] += 1
    return {
        key: {str(item): int(count) for item, count in sorted(counter.items())}
        for key, counter in counters.items()
    }


def _ensure_run_root(config: W01DenseRunConfig, run_root: Path) -> None:
    if config.dry_run_schedule:
        return
    if filesystem_path(run_root).exists() and not (config.resume or config.repair_incomplete):
        runtime_summary = run_root / "metrics" / "runtime_summary.csv"
        table_manifest = run_root / "manifests" / "table_manifest.json"
        if filesystem_path(runtime_summary).is_file() or filesystem_path(table_manifest).is_file():
            raise RuntimeError(f"W01 run root already exists: {run_root}; use --resume, --repair-incomplete, or a new run id.")
    tables = run_root / "tables" / W01_TABLE_NAME
    if filesystem_path(tables).exists() and not (config.resume or config.repair_incomplete):
        existing = list(filesystem_path(tables).glob("*"))
        if existing:
            raise RuntimeError(f"W01 run root already contains table partitions: {run_root}; use --resume or --repair-incomplete.")


def _run_root(config: W01DenseRunConfig) -> Path:
    return Path(config.output_root) / f"{int(config.run_id):03d}"


def _rollout_id(run_id: int, row_index: int) -> str:
    return f"w01_{int(run_id):03d}_{int(row_index):07d}"


def _result_payload(run_root: Path, *, status: str) -> dict[str, object]:
    return {
        "status": status,
        "run_root": run_root.as_posix(),
        "run_manifest": (run_root / "manifests" / "run_manifest.json").as_posix(),
        "table_manifest": (run_root / "manifests" / "table_manifest.json").as_posix(),
        "variant_registry_csv": (run_root / "metrics" / "primitive_variant_registry.csv").as_posix(),
        "variant_registry_json": (run_root / "manifests" / "primitive_variant_registry.json").as_posix(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the W01 primitive-controller dense preflight.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", "--chunk-size", dest="candidate_chunk_size", type=int, default=100)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", choices=("auto", "parquet", "csv_gz", "csv"), default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--paired-tests-per-candidate", type=int, default=None)
    parser.add_argument("--schedule-mode", choices=(BALANCED_SCHEDULE_MODE, SMOKE_SCHEDULE_MODE), default=BALANCED_SCHEDULE_MODE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = W01DenseRunConfig(
        run_id=args.run_id,
        output_root=args.output_root,
        rows=args.rows,
        seed=args.seed,
        candidate_chunk_size=args.candidate_chunk_size,
        workers=args.workers,
        max_workers=args.max_workers,
        storage_format=args.storage_format,
        compression_level=args.compression_level,
        resume=args.resume,
        repair_incomplete=args.repair_incomplete,
        dry_run_schedule=args.dry_run_schedule,
        stop_after_chunks=args.stop_after_chunks,
        continue_on_chunk_failure=args.continue_on_chunk_failure,
        candidate_count=args.candidate_count,
        paired_tests_per_candidate=args.paired_tests_per_candidate,
        schedule_mode=args.schedule_mode,
    )
    result = run_lqr_w01_dense_chunked(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
