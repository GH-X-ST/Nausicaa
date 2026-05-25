from __future__ import annotations

import argparse
import json
import math
import shutil
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
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import (  # noqa: E402
    MAX_GENERATED_FILE_SIZE_MB,
    PREFERRED_GENERATED_FILE_SIZE_MB,
    worker_count_decision,
)
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    file_sha256,
    filesystem_path,
    load_table_manifest,
    read_table_partition,
    resolve_storage_format,
    table_extension,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import build_environment_context, environment_context_row  # noqa: E402
from env_instance import (  # noqa: E402
    environment_instance_for_mode,
    environment_instance_row,
    environment_metadata_from_instance,
)
from env_surrogate import resolve_surrogate_binding, surrogate_binding_row, wind_field_for_binding  # noqa: E402
from frozen_w01_controller_bundle import (  # noqa: E402
    FROZEN_CONTROLLER_READY,
    FROZEN_W01_CONTROLLER_BUNDLE_VERSION,
    FrozenW01ControllerBundle,
    FrozenW01ControllerRecord,
    frozen_bundle_record_row,
    load_frozen_w01_controller_bundle,
)
from implementation_instance import implementation_instance_for_layer, implementation_instance_row  # noqa: E402
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from prim_cat import primitive_by_id  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    blocked_rollout_evidence,
    rollout_evidence_row,
    simulate_primitive_rollout,
)
from primitive_variant_registry import (  # noqa: E402
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
    start_family_is_compatible,
    variant_row,
)
from state_sampling import (  # noqa: E402
    archive_state_sample_for_family,
    archive_state_sample_row,
    start_state_family_for_row,
)


W2_SURVIVAL_VERSION = "w2_fixed_lqr_survival_replay_v1"
PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.5"
DEFAULT_W01_INPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w01_dense/012")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w2_survival")
W2_TABLE_NAME = "w2_survival_rows"
W2_ENVIRONMENT_CASES = ("annular_gp_single", "annular_gp_four")
W2_STATUS_VOCABULARY = ("survived", "downgraded", "eliminated", "blocked", "not_run")
CLAIM_BOUNDARY = (
    "simulation_only_W2_fixed_LQR_survival_replay_for_frozen_W01_timing_aware_primitive_library"
)
BLOCKED_CLAIMS = (
    "W3_robustness_complete",
    "post_W3_compact_library_ready",
    "governor_validation",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "formal_LQR_tree_funnel_region_of_attraction",
)


@dataclass(frozen=True)
class W2SurvivalConfig:
    run_id: int
    input_root: Path = DEFAULT_W01_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    seed: int = 10
    paired_tests_per_variant: int = 100
    candidate_chunk_size: int = 800
    workers: int | str = 8
    max_workers: int | None = 8
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    stop_after_chunks: int | None = None
    continue_on_chunk_failure: bool = False
    rollout_dt_s: float = 0.02
    latency_case: str = "nominal"


@dataclass(frozen=True)
class W2ChunkSpec:
    run_id: int
    chunk_index: int
    chunk_count: int
    row_start: int
    row_stop: int
    storage_format: str
    compression_level: int


@dataclass(frozen=True)
class W2RowSchedule:
    row_index: int
    primitive_variant_id: str
    variant_index: int
    environment_mode: str
    paired_start_index: int
    start_state_family: str
    paired_start_key: str


def run_w2_survival(config: W2SurvivalConfig) -> dict[str, object]:
    """Run W2 fixed-LQR survival replay from a frozen W01 controller bundle."""

    if int(config.paired_tests_per_variant) <= 0:
        raise ValueError("paired_tests_per_variant must be positive.")
    if int(config.candidate_chunk_size) <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    storage_format = resolve_storage_format(config.storage_format)
    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports", "chunk_manifests", "tables"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    source_bundle_path = Path(config.input_root) / "manifests" / "frozen_w01_controller_bundle.json"
    bundle_path = run_root / "manifests" / "frozen_w01_controller_bundle.json"
    if filesystem_path(source_bundle_path).is_file():
        shutil.copyfile(filesystem_path(source_bundle_path), filesystem_path(bundle_path))
        bundle = load_frozen_w01_controller_bundle(source_bundle_path)
    else:
        bundle = _missing_source_bundle(Path(config.input_root), "missing_W01_frozen_controller_bundle")
        _write_json(bundle_path, _blocked_bundle_payload(bundle, "missing_W01_frozen_controller_bundle"))
    if int(bundle.variant_count) <= 0:
        _write_blocked_run(
            run_root=run_root,
            config=config,
            bundle=bundle,
            storage_format=storage_format,
            blocker="missing_W01_frozen_controller_bundle",
        )
        return _result_payload(run_root, "blocked")

    row_count = _planned_row_count(bundle=bundle, config=config)
    schedule = _chunk_schedule(config, row_count=row_count, storage_format=storage_format)
    if config.stop_after_chunks is not None:
        schedule = schedule[: max(0, int(config.stop_after_chunks))]
    selected_workers = worker_count_decision(config.workers, max_workers=config.max_workers)
    _protect_existing_run(run_root, config)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        bundle=bundle,
        row_count=row_count,
        schedule=schedule,
        storage_format=storage_format,
        selected_worker_count=selected_workers.selected_worker_count,
        worker_decision=selected_workers.as_manifest_fields(),
        status="dry_run_schedule" if config.dry_run_schedule else "running",
    )

    if config.dry_run_schedule:
        _write_chunk_summary(run_root, [_scheduled_chunk_row(chunk, run_root) for chunk in schedule])
        _write_empty_table_manifest(run_root, config.run_id, storage_format)
        _write_empty_metrics(run_root, bundle=bundle, row_count=row_count)
        _write_empty_survivor_registry(run_root, bundle=bundle, status="dry_run_schedule")
        _write_file_size_audit(run_root)
        _write_reports(run_root=run_root, config=config, bundle=bundle, status="dry_run_schedule", row_count=0)
        return _result_payload(run_root, "dry_run_schedule")

    partitions = []
    chunk_records: list[dict[str, object]] = []
    pending_chunks: list[W2ChunkSpec] = []
    for chunk in schedule:
        status = _existing_chunk_status(chunk, run_root=run_root)
        if status == "complete":
            if config.resume or config.repair_incomplete:
                partitions.append(_partition_from_existing(chunk, run_root))
                chunk_records.append(_completed_chunk_row(chunk, run_root, status="skipped"))
                continue
            raise RuntimeError(f"complete W2 chunk already exists: c{chunk.chunk_index:05d}; use --resume")
        if status == "corrupt":
            if not config.repair_incomplete:
                raise RuntimeError(f"corrupt W2 chunk exists: c{chunk.chunk_index:05d}; use --repair-incomplete")
            _remove_chunk_files(chunk, run_root)
        pending_chunks.append(chunk)

    executed_partitions, executed_records, failures = _execute_pending_chunks(
        pending_chunks,
        run_root=run_root,
        config=config,
        bundle_path=bundle_path,
        selected_worker_count=selected_workers.selected_worker_count,
    )
    partitions.extend(executed_partitions)
    chunk_records.extend(executed_records)
    if failures and not config.continue_on_chunk_failure:
        _write_chunk_summary(run_root, sorted(chunk_records, key=lambda item: int(item.get("chunk_index", -1))))
        raise RuntimeError(f"W2 chunk failed: c{int(failures[0]['chunk_index']):05d}: {failures[0].get('error', '')}")

    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=storage_format,
            tables=tuple(sorted(partitions, key=lambda item: item.relative_path)),
        ),
    )
    _write_chunk_summary(run_root, sorted(chunk_records, key=lambda item: int(item.get("chunk_index", -1))))
    rows_written, variant_summary = _write_dense_metrics_from_partitions(run_root, partitions)
    _write_survivor_registry(run_root, bundle=bundle, variant_summary=variant_summary)
    _write_file_size_audit(run_root)
    status, blockers = _w2_move_on_status(run_root=run_root, bundle=bundle, row_count=rows_written, variant_summary=variant_summary)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        bundle=bundle,
        row_count=rows_written,
        schedule=schedule,
        storage_format=storage_format,
        selected_worker_count=selected_workers.selected_worker_count,
        worker_decision=selected_workers.as_manifest_fields(),
        status=status,
        blockers=blockers,
    )
    _write_reports(run_root=run_root, config=config, bundle=bundle, status=status, row_count=rows_written, blockers=blockers)
    return _result_payload(run_root, status)


def _write_chunk(
    chunk: W2ChunkSpec,
    *,
    run_root: Path,
    config: W2SurvivalConfig,
    bundle_path: Path,
):
    started = time.time()
    bundle = load_frozen_w01_controller_bundle(bundle_path)
    records = tuple(bundle.records)
    rows = [
        _row_for_index(
            row_index=row_index,
            config=config,
            bundle=bundle,
            records=records,
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
        "runner_version": W2_SURVIVAL_VERSION,
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
        "table_name": W2_TABLE_NAME,
    }
    _write_json(_chunk_manifest_path(chunk, run_root), manifest)
    return partition, {
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


def _write_chunk_worker(payload):
    chunk, run_root, config, bundle_path = payload
    return _write_chunk(
        chunk,
        run_root=Path(run_root),
        config=config,
        bundle_path=Path(bundle_path),
    )


def _execute_pending_chunks(
    chunks: list[W2ChunkSpec],
    *,
    run_root: Path,
    config: W2SurvivalConfig,
    bundle_path: Path,
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
                partition, record = _write_chunk(chunk, run_root=run_root, config=config, bundle_path=bundle_path)
                partitions.append(partition)
                records.append(record)
            except Exception as exc:
                failure = _chunk_failure_row(chunk, exc)
                failures.append(failure)
                records.append({**_scheduled_chunk_row(chunk, run_root), **failure})
                if not config.continue_on_chunk_failure:
                    break
        return partitions, records, failures

    payloads = [(chunk, run_root, config, bundle_path) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_chunk = {executor.submit(_write_chunk_worker, payload): payload[0] for payload in payloads}
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


def _row_for_index(
    *,
    row_index: int,
    config: W2SurvivalConfig,
    bundle: FrozenW01ControllerBundle,
    records: tuple[FrozenW01ControllerRecord, ...],
) -> dict[str, object]:
    schedule = _row_schedule_for_index(row_index=row_index, records=records, config=config)
    record = records[int(schedule.variant_index)]
    variant = record.variant
    controller = record.controller
    primitive = primitive_by_id(variant.primitive_id)
    sample = archive_state_sample_for_family(
        start_state_family=schedule.start_state_family,
        paired_start_key=schedule.paired_start_key,
        sample_index=int(schedule.paired_start_index),
        seed=int(config.seed),
        W_layer="W2",
        environment_mode=schedule.environment_mode,
    )
    environment = environment_instance_for_mode("W2", schedule.environment_mode, int(config.seed) + int(row_index))
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding("W2", metadata, randomisation_seed=int(config.seed) + int(row_index))
    wind_field = wind_field_for_binding(binding)
    context = build_environment_context(
        sample.state_vector,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=str(config.latency_case),
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    compatible = start_family_is_compatible(
        entry_role=variant.entry_role,
        start_state_family=sample.start_state_family,
    )
    rollout_config = RolloutConfig(
        W_layer="W2",
        dt_s=float(config.rollout_dt_s),
        rollout_backend="model_backed_lqr",
        wind_mode="panel",
    )
    controller_status = "W2_fixed_lqr_replay_frozen_w01_bundle"
    if not compatible:
        evidence = blocked_rollout_evidence(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w2_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=RolloutConfig(W_layer="W2", dt_s=float(config.rollout_dt_s), rollout_backend="blocked_lqr", wind_mode="panel"),
            failure_label=ENTRY_ROLE_REJECTION_LABEL,
            controller=controller,
            controller_selection_status=controller_status,
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
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
        implementation = None
        plant = None
        implementation_audit_status = "not_applied_entry_role_incompatible"
        plant_audit_status = "not_applied_entry_role_incompatible"
    elif record.bundle_status != FROZEN_CONTROLLER_READY:
        evidence = blocked_rollout_evidence(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w2_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=RolloutConfig(W_layer="W2", dt_s=float(config.rollout_dt_s), rollout_backend="blocked_lqr", wind_mode="panel"),
            failure_label="w2_controller_reconstruction_failed",
            controller=controller,
            controller_selection_status=controller_status,
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
            termination_cause=record.blocked_reason,
        )
        row = rollout_evidence_row(evidence)
        row["controller_executable"] = False
        row["controller_evidence_status"] = "blocked_W2_frozen_controller_reconstruction"
        implementation = None
        plant = None
        implementation_audit_status = "not_applied_controller_reconstruction_blocked"
        plant_audit_status = "not_applied_controller_reconstruction_blocked"
    elif binding.surrogate_binding_status != "ready":
        evidence = blocked_rollout_evidence(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w2_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=RolloutConfig(W_layer="W2", dt_s=float(config.rollout_dt_s), rollout_backend="blocked_lqr", wind_mode="panel"),
            failure_label="w2_surrogate_binding_blocked",
            controller=controller,
            controller_selection_status=controller_status,
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
            termination_cause=str(binding.blocked_reason),
        )
        row = rollout_evidence_row(evidence)
        implementation = None
        plant = None
        implementation_audit_status = "not_applied_surrogate_blocked"
        plant_audit_status = "not_applied_surrogate_blocked"
    else:
        implementation = implementation_instance_for_layer("W2", int(config.seed) + int(row_index), latency_case=str(config.latency_case))
        plant = plant_instance_for_layer("W2", int(config.seed) + int(row_index))
        evidence = simulate_primitive_rollout(
            rollout_id=_rollout_id(config.run_id, row_index),
            episode_id=f"w2_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=rollout_config,
            wind_field=wind_field,
            implementation_instance=implementation,
            plant_instance=plant,
            controller=controller,
            controller_selection_status=controller_status,
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
        )
        row = rollout_evidence_row(evidence)
        row["implementation_instance_id"] = implementation.implementation_instance_id
        row["plant_instance_id"] = plant.plant_instance_id
        implementation_audit_status = "applied_in_rollout"
        plant_audit_status = "applied_in_rollout"

    row.update(archive_state_sample_row(sample))
    row.update({f"variant_{key}": value for key, value in variant_row(variant).items()})
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    row.update({f"surrogate_{key}": value for key, value in surrogate_binding_row(binding).items()})
    row.update({f"environment_{key}": value for key, value in environment_instance_row(environment).items()})
    if implementation is None or plant is None:
        row.update(_blocked_instance_prefix_rows("implementation"))
        row.update(_blocked_instance_prefix_rows("plant"))
    else:
        row.update({f"implementation_{key}": value for key, value in implementation_instance_row(implementation).items()})
        row.update({f"plant_{key}": value for key, value in plant_instance_row(plant).items()})
    row_status, row_reason = _row_survival_status(row=row, entry_role_compatible=compatible, record=record)
    row.update(
        {
            "runner_version": W2_SURVIVAL_VERSION,
            "project_title_version": PROJECT_TITLE_VERSION,
            "run_stage": "W2_fixed_LQR_survival_replay",
            "row_index": int(row_index),
            "schedule_mode": "w2_fixed_variant_paired_survival",
            "paired_start_policy": "common_random_start_key_reused_across_w2_environments_and_variants",
            "schedule_paired_start_index": int(schedule.paired_start_index),
            "source_w01_run_id": bundle.source_w01_run_id,
            "source_w01_root": bundle.source_w01_root,
            "source_w01_registry_sha256": bundle.source_registry_sha256,
            "source_w01_table_manifest_sha256": bundle.source_table_manifest_sha256,
            "primitive_variant_id": variant.primitive_variant_id,
            "entry_role": variant.entry_role,
            "entry_role_compatible": bool(compatible),
            "candidate_index": variant.candidate_index,
            "candidate_weight_label": variant.candidate_weight_label,
            "environment_mode": schedule.environment_mode,
            "environment_instance_id": environment.environment_id,
            "annular_gp_model_id": environment.updraft_model_id,
            "fan_count": int(environment.fan_count),
            "surrogate_blocked_reason": binding.blocked_reason,
            "implementation_instance_status": implementation_audit_status,
            "plant_instance_status": plant_audit_status,
            "controller_bundle_status": record.bundle_status,
            "controller_bundle_blocked_reason": record.blocked_reason,
            "w2_survival_status": row_status,
            "w2_survival_reason": row_reason,
            "fixed_lqr_replay_only": True,
            "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
            "w2_environment_contract": "single_and_four_annular_gp_survival_replay_only",
            "panelwise_glider_dynamics_active": True,
            "state_feedback_latency_lag_active": True,
            "command_timing_active": True,
            "actuator_lag_active": True,
            "pd_pid_fallback_allowed": False,
            "baseline_controller_active": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return row


def _row_survival_status(
    *,
    row: dict[str, object],
    entry_role_compatible: bool,
    record: FrozenW01ControllerRecord,
) -> tuple[str, str]:
    if not entry_role_compatible:
        return "blocked", ENTRY_ROLE_REJECTION_LABEL
    if record.bundle_status != FROZEN_CONTROLLER_READY:
        return "blocked", "w2_controller_reconstruction_failed"
    outcome = str(row.get("outcome_class", ""))
    failure = str(row.get("failure_label", ""))
    if outcome == "blocked":
        return "blocked", failure
    if bool(row.get("continuation_valid", False)):
        return "survived", "continuation_valid"
    if bool(row.get("episode_terminal_useful", False)):
        return "downgraded", "terminal_useful_boundary_evidence_retained"
    if outcome == "accepted":
        return "survived", "accepted"
    return "eliminated", failure or str(row.get("termination_cause", ""))


def _row_schedule_for_index(
    *,
    row_index: int,
    records: tuple[FrozenW01ControllerRecord, ...],
    config: W2SurvivalConfig,
) -> W2RowSchedule:
    variant_count = max(1, len(records))
    variant_index = int(row_index) % variant_count
    grouped_index = int(row_index) // variant_count
    environment_index = grouped_index % len(W2_ENVIRONMENT_CASES)
    paired_start_index = grouped_index // len(W2_ENVIRONMENT_CASES)
    family = start_state_family_for_row(paired_start_index)
    return W2RowSchedule(
        row_index=int(row_index),
        primitive_variant_id=records[variant_index].primitive_variant_id,
        variant_index=int(variant_index),
        environment_mode=W2_ENVIRONMENT_CASES[environment_index],
        paired_start_index=int(paired_start_index),
        start_state_family=str(family),
        paired_start_key=f"w2_paired_{int(paired_start_index):07d}_{family}",
    )


def _planned_row_count(*, bundle: FrozenW01ControllerBundle, config: W2SurvivalConfig) -> int:
    return int(bundle.variant_count) * len(W2_ENVIRONMENT_CASES) * int(config.paired_tests_per_variant)


def _chunk_schedule(config: W2SurvivalConfig, *, row_count: int, storage_format: str) -> list[W2ChunkSpec]:
    chunk_count = int(math.ceil(int(row_count) / int(config.candidate_chunk_size)))
    return [
        W2ChunkSpec(
            run_id=int(config.run_id),
            chunk_index=int(chunk_index),
            chunk_count=int(chunk_count),
            row_start=int(chunk_index) * int(config.candidate_chunk_size),
            row_stop=min(int(row_count), (int(chunk_index) + 1) * int(config.candidate_chunk_size)),
            storage_format=storage_format,
            compression_level=int(config.compression_level),
        )
        for chunk_index in range(chunk_count)
    ]


def _existing_chunk_status(chunk: W2ChunkSpec, *, run_root: Path) -> str:
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


def _remove_chunk_files(chunk: W2ChunkSpec, run_root: Path) -> None:
    for path in (_partition_path(chunk, run_root), _chunk_manifest_path(chunk, run_root)):
        fs_path = filesystem_path(path)
        if fs_path.exists():
            fs_path.unlink()


def _partition_from_existing(chunk: W2ChunkSpec, run_root: Path):
    from dense_archive_table_io import TablePartition

    manifest = json.loads(filesystem_path(_chunk_manifest_path(chunk, run_root)).read_text(encoding="ascii"))
    frame = read_table_partition(_partition_path(chunk, run_root), storage_format=chunk.storage_format)
    return TablePartition(
        table_name=W2_TABLE_NAME,
        relative_path=str(manifest["partition_path"]),
        storage_format=chunk.storage_format,
        row_count=len(frame),
        byte_count=int(filesystem_path(_partition_path(chunk, run_root)).stat().st_size),
        columns=tuple(str(column) for column in frame.columns),
        checksum_sha256=file_sha256(_partition_path(chunk, run_root)),
    )


def _partition_path(chunk: W2ChunkSpec, run_root: Path) -> Path:
    return run_root / "tables" / W2_TABLE_NAME / f"c{int(chunk.chunk_index):05d}.{table_extension(chunk.storage_format)}"


def _chunk_manifest_path(chunk: W2ChunkSpec, run_root: Path) -> Path:
    return run_root / "chunk_manifests" / W2_TABLE_NAME / f"c{int(chunk.chunk_index):05d}.json"


def _scheduled_chunk_row(chunk: W2ChunkSpec, run_root: Path) -> dict[str, object]:
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


def _completed_chunk_row(chunk: W2ChunkSpec, run_root: Path, *, status: str) -> dict[str, object]:
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


def _chunk_failure_row(chunk: W2ChunkSpec, exc: Exception) -> dict[str, object]:
    return {
        "chunk_index": int(chunk.chunk_index),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_dense_metrics_from_partitions(run_root: Path, partitions: Iterable[object]) -> tuple[int, pd.DataFrame]:
    counters: dict[str, Counter] = defaultdict(Counter)
    grouped: dict[str, Counter] = defaultdict(Counter)
    variant_rows: list[dict[str, object]] = []
    row_count = 0
    for partition in partitions:
        frame = read_table_partition(run_root / "tables" / partition.relative_path, storage_format=partition.storage_format)
        row_count += int(len(frame))
        for column in ("w2_survival_status", "failure_label", "boundary_use_class", "environment_mode", "start_state_family"):
            _counter_update(counters[column], frame, column)
        _group_counter_update(grouped["environment"], frame, ("W_layer", "environment_mode", "fan_count", "annular_gp_model_id"))
        _group_counter_update(grouped["failure"], frame, ("w2_survival_status", "failure_label", "termination_cause"))
        _group_counter_update(grouped["boundary"], frame, ("boundary_use_class", "continuation_valid", "episode_terminal_useful", "w2_survival_status"))
        _group_counter_update(grouped["variant"], frame, ("primitive_variant_id", "primitive_id", "candidate_index", "environment_mode", "w2_survival_status", "entry_role_compatible"))
        variant_rows.extend(
            frame[
                [
                    "primitive_variant_id",
                    "primitive_id",
                    "candidate_index",
                    "environment_mode",
                    "w2_survival_status",
                    "entry_role_compatible",
                    "boundary_use_class",
                    "episode_terminal_useful",
                    "continuation_valid",
                ]
            ].to_dict(orient="records")
        )
    _counter_frame(counters["w2_survival_status"]).to_csv(filesystem_path(run_root / "metrics" / "w2_survival_summary.csv"), index=False)
    _group_counter_frame(grouped["environment"], ("W_layer", "environment_mode", "fan_count", "annular_gp_model_id")).to_csv(
        filesystem_path(run_root / "metrics" / "w2_environment_coverage.csv"),
        index=False,
    )
    _group_counter_frame(grouped["failure"], ("w2_survival_status", "failure_label", "termination_cause")).to_csv(
        filesystem_path(run_root / "metrics" / "w2_failure_summary.csv"),
        index=False,
    )
    _group_counter_frame(grouped["boundary"], ("boundary_use_class", "continuation_valid", "episode_terminal_useful", "w2_survival_status")).to_csv(
        filesystem_path(run_root / "metrics" / "w2_boundary_use_summary.csv"),
        index=False,
    )
    variant_summary = _variant_survival_summary(pd.DataFrame(variant_rows))
    variant_summary.to_csv(filesystem_path(run_root / "metrics" / "w2_variant_survival_summary.csv"), index=False)
    pd.DataFrame(
        [
            {"coverage_axis": key, "value": value, "row_count": count}
            for key, counter in counters.items()
            for value, count in sorted(counter.items())
        ]
    ).to_csv(filesystem_path(run_root / "metrics" / "w2_coverage_summary.csv"), index=False)
    return int(row_count), variant_summary


def _variant_survival_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "primitive_variant_id",
                "primitive_id",
                "candidate_index",
                "compatible_row_count",
                "survived_environment_count",
                "downgraded_row_count",
                "eliminated_row_count",
                "blocked_row_count",
                "w2_variant_status",
                "eligible_for_w3",
            ]
        )
    rows = []
    for variant_id, group in frame.groupby("primitive_variant_id", dropna=False):
        compatible = group[group["entry_role_compatible"].astype(bool)]
        survived_envs = set(compatible.loc[compatible["w2_survival_status"].eq("survived"), "environment_mode"].astype(str))
        downgraded_count = int(compatible["w2_survival_status"].eq("downgraded").sum())
        eliminated_count = int(compatible["w2_survival_status"].eq("eliminated").sum())
        blocked_count = int(compatible["w2_survival_status"].eq("blocked").sum())
        if len(survived_envs) == len(W2_ENVIRONMENT_CASES):
            status = "survived"
        elif survived_envs or downgraded_count:
            status = "downgraded"
        elif blocked_count and not eliminated_count:
            status = "blocked"
        elif eliminated_count:
            status = "eliminated"
        else:
            status = "not_run"
        rows.append(
            {
                "primitive_variant_id": str(variant_id),
                "primitive_id": str(group["primitive_id"].iloc[0]),
                "candidate_index": group["candidate_index"].iloc[0],
                "compatible_row_count": int(len(compatible)),
                "survived_environment_count": int(len(survived_envs)),
                "downgraded_row_count": downgraded_count,
                "eliminated_row_count": eliminated_count,
                "blocked_row_count": blocked_count,
                "w2_variant_status": status,
                "eligible_for_w3": bool(status == "survived"),
            }
        )
    return pd.DataFrame(rows).sort_values(["w2_variant_status", "primitive_id", "candidate_index", "primitive_variant_id"])


def _write_survivor_registry(
    run_root: Path,
    *,
    bundle: FrozenW01ControllerBundle,
    variant_summary: pd.DataFrame,
) -> None:
    summary_by_id = {
        str(row["primitive_variant_id"]): row
        for row in variant_summary.to_dict(orient="records")
    }
    records = []
    for record in bundle.records:
        summary = summary_by_id.get(record.primitive_variant_id, {})
        if str(summary.get("w2_variant_status", "")) == "survived":
            payload = variant_row(record.variant)
            payload.update(
                {
                    "w2_variant_status": "survived",
                    "eligible_for_w3": True,
                    "source_w01_run_id": bundle.source_w01_run_id,
                    "source_w01_registry_sha256": bundle.source_registry_sha256,
                }
            )
            records.append(payload)
    payload = {
        "survivor_registry_version": "w2_survivor_registry_v1",
        "status": "survived_variants_available" if records else "blocked_no_w2_survivors",
        "survivor_count": len(records),
        "source_w01_run_id": bundle.source_w01_run_id,
        "survivors": records,
    }
    _write_json(run_root / "manifests" / "w2_survivor_registry.json", payload)


def _write_empty_metrics(run_root: Path, *, bundle: FrozenW01ControllerBundle, row_count: int) -> None:
    pd.DataFrame([{"w2_survival_status": "not_run", "row_count": 0}]).to_csv(
        filesystem_path(run_root / "metrics" / "w2_survival_summary.csv"),
        index=False,
    )
    pd.DataFrame([frozen_bundle_record_row(record) for record in bundle.records]).to_csv(
        filesystem_path(run_root / "metrics" / "w2_controller_bundle_summary.csv"),
        index=False,
    )
    pd.DataFrame(
        [
            {
                "planned_row_count": int(row_count),
                "variant_count": int(bundle.variant_count),
                "ready_controller_count": int(bundle.ready_count),
                "blocked_controller_count": int(bundle.blocked_count),
            }
        ]
    ).to_csv(filesystem_path(run_root / "metrics" / "w2_schedule_summary.csv"), index=False)


def _missing_source_bundle(input_root: Path, blocked_reason: str) -> FrozenW01ControllerBundle:
    return FrozenW01ControllerBundle(
        bundle_version=FROZEN_W01_CONTROLLER_BUNDLE_VERSION,
        source_w01_root=Path(input_root).as_posix(),
        source_w01_run_id=_run_id_from_root(input_root),
        source_registry_sha256="",
        source_table_manifest_sha256="",
        source_run_manifest_sha256="",
        variant_count=0,
        ready_count=0,
        blocked_count=0,
        records=(),
    )


def _blocked_bundle_payload(bundle: FrozenW01ControllerBundle, blocked_reason: str) -> dict[str, object]:
    return {
        "bundle_version": bundle.bundle_version,
        "source_w01_root": bundle.source_w01_root,
        "source_w01_run_id": bundle.source_w01_run_id,
        "source_registry_sha256": bundle.source_registry_sha256,
        "source_table_manifest_sha256": bundle.source_table_manifest_sha256,
        "source_run_manifest_sha256": bundle.source_run_manifest_sha256,
        "variant_count": 0,
        "ready_count": 0,
        "blocked_count": 0,
        "exact_replay_policy": "blocked_missing_w01_emitted_bundle_no_controller_design",
        "physical_K_only_active_replay_allowed": False,
        "blocked_reason": blocked_reason,
        "records": [],
    }


def _run_id_from_root(root: Path) -> int | str:
    try:
        return int(Path(root).name)
    except ValueError:
        return Path(root).name


def _write_blocked_run(
    *,
    run_root: Path,
    config: W2SurvivalConfig,
    bundle: FrozenW01ControllerBundle,
    storage_format: str,
    blocker: str = "missing_or_empty_frozen_w01_controller_bundle",
) -> None:
    _write_empty_table_manifest(run_root, config.run_id, storage_format)
    _write_empty_metrics(run_root, bundle=bundle, row_count=0)
    _write_file_size_audit(run_root)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        bundle=bundle,
        row_count=0,
        schedule=[],
        storage_format=storage_format,
        selected_worker_count=0,
        worker_decision={},
        status="blocked",
        blockers=[blocker],
    )
    _write_json(
        run_root / "manifests" / "w2_survivor_registry.json",
        {
            "survivor_registry_version": "w2_survivor_registry_v1",
            "status": "blocked_no_w2_survivors",
            "survivor_count": 0,
            "survivors": [],
        },
    )
    _write_reports(
        run_root=run_root,
        config=config,
        bundle=bundle,
        status="blocked",
        row_count=0,
        blockers=[blocker],
    )


def _write_empty_survivor_registry(
    run_root: Path,
    *,
    bundle: FrozenW01ControllerBundle,
    status: str,
) -> None:
    _write_json(
        run_root / "manifests" / "w2_survivor_registry.json",
        {
            "survivor_registry_version": "w2_survivor_registry_v1",
            "status": status,
            "survivor_count": 0,
            "source_w01_run_id": bundle.source_w01_run_id,
            "survivors": [],
        },
    )


def _w2_move_on_status(
    *,
    run_root: Path,
    bundle: FrozenW01ControllerBundle,
    row_count: int,
    variant_summary: pd.DataFrame,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    expected_rows = int(bundle.variant_count) * len(W2_ENVIRONMENT_CASES) * 100
    if int(row_count) < expected_rows:
        blockers.append("below_51200_w2_survival_row_threshold")
    if int(bundle.variant_count) != 256:
        blockers.append("variant_count_not_256")
    if int(bundle.ready_count) <= 0:
        blockers.append("no_ready_frozen_timing_aware_controllers")
    if variant_summary.empty or not variant_summary["eligible_for_w3"].astype(bool).any():
        blockers.append("no_w2_survived_variants_available")
    blockers.extend(_manifest_checksum_gate_blockers(run_root))
    blockers.extend(_file_size_gate_blockers(run_root))
    status = "survived_variants_available" if not blockers else "blocked"
    return status, blockers


def _manifest_checksum_gate_blockers(run_root: Path) -> list[str]:
    manifest_path = filesystem_path(run_root / "manifests" / "table_manifest.json")
    if not manifest_path.is_file():
        return ["missing_table_manifest"]
    try:
        manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
        for table in manifest.tables:
            partition_path = run_root / "tables" / table.relative_path
            if file_sha256(partition_path) != str(table.checksum_sha256):
                return ["table_manifest_partition_checksum_mismatch"]
            if len(read_table_partition(partition_path, storage_format=table.storage_format)) != int(table.row_count):
                return ["table_manifest_partition_row_count_mismatch"]
        return []
    except Exception as exc:
        return [f"manifest_checksum_gate_unreadable:{type(exc).__name__}"]


def _file_size_gate_blockers(run_root: Path) -> list[str]:
    path = filesystem_path(run_root / "metrics" / "file_size_audit.csv")
    if not path.is_file():
        return ["missing_file_size_audit"]
    frame = pd.read_csv(path)
    if "above_100mb" not in frame.columns:
        return ["file_size_audit_missing_above_100mb"]
    if frame["above_100mb"].astype(bool).any():
        return ["file_size_audit_above_100mb"]
    return []


def _write_run_manifest(
    *,
    run_root: Path,
    config: W2SurvivalConfig,
    bundle: FrozenW01ControllerBundle,
    row_count: int,
    schedule: list[W2ChunkSpec],
    storage_format: str,
    selected_worker_count: int,
    worker_decision: dict[str, object],
    status: str,
    blockers: list[str] | None = None,
) -> None:
    manifest = {
        "runner_version": W2_SURVIVAL_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "source_w01_root": Path(config.input_root).as_posix(),
        "source_w01_run_id": bundle.source_w01_run_id,
        "source_w01_registry_sha256": bundle.source_registry_sha256,
        "source_w01_table_manifest_sha256": bundle.source_table_manifest_sha256,
        "rows_written_or_planned": int(row_count),
        "variant_count": int(bundle.variant_count),
        "ready_controller_count": int(bundle.ready_count),
        "blocked_controller_count": int(bundle.blocked_count),
        "w2_environment_modes": list(W2_ENVIRONMENT_CASES),
        "paired_tests_per_variant": int(config.paired_tests_per_variant),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "chunk_count": int(len(schedule)),
        "storage_format": storage_format,
        "compression_level": int(config.compression_level),
        "dry_run_schedule": bool(config.dry_run_schedule),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "selected_worker_count": int(selected_worker_count),
        "worker_decision": worker_decision,
        "fixed_lqr_replay_only": True,
        "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
        "status_vocabulary": list(W2_STATUS_VOCABULARY),
        "controller_bundle_source_policy": "load_only_w01_emitted_frozen_bundle_no_materialisation_no_design",
        "source_controller_bundle_path": (Path(config.input_root) / "manifests" / "frozen_w01_controller_bundle.json").as_posix(),
        "controller_bundle_path": (run_root / "manifests" / "frozen_w01_controller_bundle.json").as_posix(),
        "table_manifest": (run_root / "manifests" / "table_manifest.json").as_posix(),
        "survivor_registry": (run_root / "manifests" / "w2_survivor_registry.json").as_posix(),
        "move_on_blockers": blockers or [],
        "blocked_claims": list(BLOCKED_CLAIMS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(run_root / "manifests" / "w2_survival_manifest.json", manifest)


def _write_reports(
    *,
    run_root: Path,
    config: W2SurvivalConfig,
    bundle: FrozenW01ControllerBundle,
    status: str,
    row_count: int,
    blockers: list[str] | None = None,
) -> None:
    blockers = blockers or []
    report_lines = [
        "# W2 Fixed-LQR Survival Replay",
        "",
        f"- Status: `{status}`",
        f"- Source W01 root: `{Path(config.input_root).as_posix()}`",
        f"- Rows written/planned: `{int(row_count)}`",
        f"- Variant count: `{int(bundle.variant_count)}`",
        f"- Ready frozen controllers: `{int(bundle.ready_count)}`",
        f"- Blocked frozen controllers: `{int(bundle.blocked_count)}`",
        "- W2 environments: `annular_gp_single`, `annular_gp_four`",
        "- Fixed replay only: `True`",
        "- Q/R, K, reference, horizon, entry role, controller ID, and variant ID mutation: `False`",
        "- Boundary terminal-useful evidence is retained.",
        "",
        "Blocked claims remain W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal ROA guarantees.",
        "",
    ]
    filesystem_path(run_root / "reports" / "w2_survival_report.md").write_text("\n".join(report_lines), encoding="ascii")
    gate_lines = [
        "# L8 W2 Move-On Check",
        "",
        f"- Status: `{status}`",
        f"- Rows: `{int(row_count)}`",
        f"- Survived variants available: `{status == 'survived_variants_available'}`",
        "",
        "Blockers:",
        "",
    ]
    if blockers:
        gate_lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        gate_lines.append("- `none`")
    filesystem_path(run_root / "reports" / "l8_w2_move_on_check.md").write_text("\n".join(gate_lines), encoding="ascii")


def _write_file_size_audit(run_root: Path) -> None:
    rows = []
    root_fs = filesystem_path(run_root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_fs).as_posix()
        byte_count = int(path.stat().st_size)
        size_mb = float(byte_count) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > PREFERRED_GENERATED_FILE_SIZE_MB),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                "dense_table_partition": rel.startswith(f"tables/{W2_TABLE_NAME}/"),
            }
        )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def _write_empty_table_manifest(run_root: Path, run_id: int, storage_format: str) -> None:
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(run_id=int(run_id), root=run_root.as_posix(), storage_format=storage_format, tables=()),
    )


def _write_chunk_summary(run_root: Path, records: list[dict[str, object]]) -> None:
    pd.DataFrame(records).to_csv(filesystem_path(run_root / "metrics" / "chunk_summary.csv"), index=False)


def _protect_existing_run(run_root: Path, config: W2SurvivalConfig) -> None:
    if config.dry_run_schedule or config.resume or config.repair_incomplete:
        return
    table_manifest = filesystem_path(run_root / "manifests" / "table_manifest.json")
    if table_manifest.is_file():
        raise RuntimeError("existing W2 run root refuses overwrite without resume or repair.")


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


def _counter_update(counter: Counter, frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        return
    for value, count in frame[column].astype(str).value_counts(dropna=False).items():
        counter[str(value)] += int(count)


def _group_counter_update(counter: Counter, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return
    grouped = frame.groupby(list(columns), dropna=False).size().reset_index(name="row_count")
    for row in grouped.to_dict(orient="records"):
        key = tuple(str(row[column]) for column in columns)
        counter[key] += int(row["row_count"])


def _counter_frame(counter: Counter) -> pd.DataFrame:
    return pd.DataFrame(
        [{"value": value, "row_count": count} for value, count in sorted(counter.items())]
    )


def _group_counter_frame(counter: Counter, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key, count in sorted(counter.items()):
        row = {column: value for column, value in zip(columns, key)}
        row["row_count"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def _rollout_id(run_id: int, row_index: int) -> str:
    return f"w2r{int(run_id):03d}_{int(row_index):07d}"


def _result_payload(run_root: Path, status: str) -> dict[str, object]:
    return {
        "status": status,
        "run_root": run_root.as_posix(),
        "manifest": (run_root / "manifests" / "w2_survival_manifest.json").as_posix(),
        "survivor_registry": (run_root / "manifests" / "w2_survivor_registry.json").as_posix(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W2 fixed-LQR survival replay from frozen W01 variants.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_W01_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--paired-tests-per-variant", type=int, default=100)
    parser.add_argument("--candidate-chunk-size", "--chunk-size", dest="candidate_chunk_size", type=int, default=800)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_w2_survival(
        W2SurvivalConfig(
            run_id=args.run_id,
            input_root=args.input_root,
            output_root=args.output_root,
            seed=args.seed,
            paired_tests_per_variant=args.paired_tests_per_variant,
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
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
