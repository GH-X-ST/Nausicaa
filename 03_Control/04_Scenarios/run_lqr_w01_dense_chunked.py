from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
from env_ctx import build_environment_context  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from implementation_instance import implementation_instance_for_layer  # noqa: E402
from lqr_controller import synthesize_lqr_controller  # noqa: E402
from lqr_tuning import candidate_weight_specs, lqr_tuning_schedule, tuning_schedule_row  # noqa: E402
from plant_instance import plant_instance_for_layer  # noqa: E402
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
from state_sampling import archive_state_sample_for_row, archive_state_sample_row  # noqa: E402


W01_RUNNER_VERSION = "run_lqr_w01_dense_chunked_v1"
W01_TABLE_NAME = "w01_primitive_rows"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w01_dense")
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
    "post_W3_compact_library_ready",
    "governor_validation",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
)


@dataclass(frozen=True)
class W01DenseRunConfig:
    run_id: int
    output_root: Path = DEFAULT_OUTPUT_ROOT
    rows: int = 400
    seed: int = 1
    candidate_chunk_size: int = 100
    workers: str | int = 1
    max_workers: int | None = 1
    storage_format: str = "csv_gz"
    compression_level: int = 1
    resume: bool = False
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    stop_after_chunks: int | None = None
    continue_on_chunk_failure: bool = False
    candidate_count: int = 16
    latency_case: str = "nominal"
    rollout_dt_s: float = 0.02


@dataclass(frozen=True)
class W01ChunkSpec:
    run_id: int
    chunk_index: int
    chunk_count: int
    row_start: int
    row_stop: int
    storage_format: str
    compression_level: int


def run_lqr_w01_dense_chunked(config: W01DenseRunConfig) -> dict[str, object]:
    """Run or schedule the corrected W0/W1 primitive-controller dense preflight."""

    if int(config.rows) <= 0:
        raise ValueError("rows must be positive.")
    if int(config.candidate_chunk_size) <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    if int(config.candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")

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
        _write_reports(run_root, status="dry_run_schedule", row_count=0)
        _write_file_size_audit(run_root)
        return _result_payload(run_root, status="dry_run_schedule")

    started = time.time()
    completed_partitions = []
    chunk_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
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
            partition, record = _write_chunk(
                chunk,
                run_root=run_root,
                config=config,
                variants_by_key=variants_by_key,
            )
            completed_partitions.append(partition)
            chunk_records.append(record)
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

    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=storage_format,
            tables=tuple(completed_partitions),
        ),
    )
    ended = time.time()
    frame = _read_completed_frame(run_root, completed_partitions, storage_format)
    _write_chunk_summary(run_root, chunk_records)
    _write_dense_metrics(run_root, frame)
    _write_runtime_summary(
        run_root,
        config,
        row_count=len(frame),
        status="complete" if not failures else "partial_failed",
        started=started,
        ended=ended,
    )
    _write_reports(run_root, status="complete" if not failures else "partial_failed", row_count=len(frame))
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
            variant = primitive_controller_variant(primitive=primitive, controller=controller)
            variants_by_key[(primitive.primitive_id, int(candidate_index))] = (
                primitive,
                controller,
                variant,
                weight_spec.weight_label,
            )
            variants.append(variant)
    return variants_by_key, tuple(variants)


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


def _row_for_index(
    *,
    row_index: int,
    config: W01DenseRunConfig,
    variants_by_key: dict[tuple[str, int], tuple[object, object, PrimitiveControllerVariant, str]],
) -> dict[str, object]:
    primitive_id = ACTIVE_PRIMITIVE_IDS[int(row_index) % len(ACTIVE_PRIMITIVE_IDS)]
    candidate_index = (int(row_index) // len(ACTIVE_PRIMITIVE_IDS)) % int(config.candidate_count)
    primitive, controller, variant, weight_label = variants_by_key[(primitive_id, int(candidate_index))]
    W_layer, environment_mode = OFFICIAL_W01_ENVIRONMENT_CASES[int(row_index) % len(OFFICIAL_W01_ENVIRONMENT_CASES)]
    sample = archive_state_sample_for_row(
        int(row_index),
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
            candidate_weight_label=str(controller.tuning_stage),
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
            candidate_weight_label=str(controller.tuning_stage),
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
            candidate_weight_label=str(controller.tuning_stage),
        )
        row = rollout_evidence_row(evidence)
        row["implementation_instance_id"] = implementation.implementation_instance_id
        row["plant_instance_id"] = plant.plant_instance_id

    row.update(archive_state_sample_row(sample))
    row.update(_variant_prefix_row(variant))
    row.update(
        {
            "runner_version": W01_RUNNER_VERSION,
            "run_stage": "W01_dense_primitive_variant_generation",
            "row_index": int(row_index),
            "primitive_variant_id": variant.primitive_variant_id,
            "entry_role": variant.entry_role,
            "entry_role_compatible": bool(compatible),
            "candidate_index": int(candidate_index),
            "candidate_weight_label": weight_label,
            "environment_mode": environment_mode,
            "environment_instance_id": environment.environment_id,
            "surrogate_blocked_reason": binding.blocked_reason,
            "W_layer_official_role": "W0_dry_air" if W_layer == "W0" else "W1_gaussian_preflight",
            "small_library_selection_allowed": False,
            "clustering_before_w2_w3_allowed": False,
            "w2_w3_replay_only": True,
            "panelwise_glider_dynamics_active": True,
            "state_feedback_latency_lag_active": True,
            "command_timing_active": True,
            "actuator_lag_active": True,
            "pd_pid_fallback_allowed": False,
            "delayed_state_lqr_augmentation_status": "not_implemented_state_delay_simulated_in_rollout",
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return row


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


def _write_dense_metrics(run_root: Path, frame: pd.DataFrame) -> None:
    _value_counts(frame, "outcome_class").to_csv(filesystem_path(run_root / "metrics" / "outcome_summary.csv"), index=False)
    _value_counts(frame, "failure_label").to_csv(filesystem_path(run_root / "metrics" / "failure_summary.csv"), index=False)
    _value_counts(frame, "boundary_use_class").to_csv(filesystem_path(run_root / "metrics" / "boundary_summary.csv"), index=False)
    coverage_columns = (
        "primitive_id",
        "entry_role",
        "start_state_family",
        "W_layer",
        "environment_mode",
        "lqr_synthesis_status",
    )
    coverage_rows = []
    for column in coverage_columns:
        coverage_rows.extend(_value_counts(frame, column).assign(coverage_axis=column).to_dict(orient="records"))
    pd.DataFrame(coverage_rows).to_csv(filesystem_path(run_root / "metrics" / "coverage_summary.csv"), index=False)


def _write_empty_metrics(run_root: Path) -> None:
    for name in ("outcome_summary.csv", "coverage_summary.csv", "failure_summary.csv", "boundary_summary.csv"):
        pd.DataFrame().to_csv(filesystem_path(run_root / "metrics" / name), index=False)


def _value_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame([{"value": "missing_or_empty", "row_count": 0}])
    counts = frame[column].fillna("").astype(str).value_counts(dropna=False)
    return pd.DataFrame(
        [{"value": str(value), "row_count": int(count)} for value, count in counts.items()]
    )


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
                "under_100mb": bool(byte_count <= MAX_GENERATED_FILE_SIZE_MB * 1024.0 * 1024.0),
                "dense_table_partition": rel.startswith(f"tables/{W01_TABLE_NAME}/"),
            }
        )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def _write_reports(run_root: Path, *, status: str, row_count: int) -> None:
    blocked_claims = "\n".join(f"- `{claim}`" for claim in BLOCKED_CLAIMS)
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
    run_report = "\n".join(
        [
            "# W01 Dense Preflight Readiness Run",
            "",
            f"- Status: `{status}`",
            f"- Rows written: `{int(row_count)}`",
            "- Retired controller-picking, compact-library, integration, hardware, transfer, and mission-success claims remain blocked.",
            "",
        ]
    )
    filesystem_path(run_root / "reports" / "run_report.md").write_text(run_report, encoding="ascii")


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
    schedule_row = tuning_schedule_row(
        lqr_tuning_schedule(
            candidate_count=int(config.candidate_count),
            paired_tests_per_candidate=max(1, int(config.rows) // max(1, len(variants))),
        )
    )
    return {
        "runner_version": W01_RUNNER_VERSION,
        "run_stage": "W01_dense_primitive_variant_generation",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "rows_requested": int(config.rows),
        "seed": int(config.seed),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "candidate_count": int(config.candidate_count),
        "storage_format": storage_format,
        "compression_level": int(config.compression_level),
        "dry_run_schedule": bool(config.dry_run_schedule),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "selected_worker_count": int(worker_decision.selected_worker_count),
        "worker_decision": asdict(worker_decision),
        "chunk_count": int(len(schedule)),
        "official_W_layers": {"W0": ["dry_air"], "W1": ["gaussian_single", "gaussian_four"]},
        "W2_generated": False,
        "W3_generated": False,
        "mixed_primitive_start_sampling": {
            "launch_gate": 0.40,
            "inflight_nominal": 0.25,
            "inflight_lift_region": 0.15,
            "inflight_boundary_near": 0.10,
            "inflight_recovery_edge": 0.10,
        },
        "no_small_library_selection": True,
        "no_clustering_before_W2_W3": True,
        "W2_W3_replay_only": True,
        "panelwise_timing_actuator_effects_active_from_W01": True,
        "PD_PID_fallback_allowed": False,
        "claim_status": CLAIM_BOUNDARY,
        "blocked_claims": list(BLOCKED_CLAIMS),
        "tuning_schedule": schedule_row,
    }


def _ensure_run_root(config: W01DenseRunConfig, run_root: Path) -> None:
    if config.dry_run_schedule:
        return
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
    parser.add_argument("--workers", default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--storage-format", choices=("auto", "parquet", "csv_gz", "csv"), default="csv_gz")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-count", type=int, default=16)
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
    )
    result = run_lqr_w01_dense_chunked(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
