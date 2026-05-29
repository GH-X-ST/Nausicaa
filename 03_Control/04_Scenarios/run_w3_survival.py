from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
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

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB, worker_count_decision  # noqa: E402
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
    EnvironmentRandomisationConfig,
    environment_instance_for_mode,
    environment_instance_row,
    environment_metadata_from_instance,
)
from env_surrogate import resolve_surrogate_binding, surrogate_binding_row, wind_field_for_binding  # noqa: E402
from frozen_w01_controller_bundle import FROZEN_CONTROLLER_READY, FrozenW01ControllerRecord, load_frozen_w01_controller_bundle  # noqa: E402
from implementation_instance import implementation_instance_for_layer, implementation_instance_row  # noqa: E402
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from prim_cat import primitive_by_id  # noqa: E402
from prim_roll import RolloutConfig, blocked_rollout_evidence, rollout_evidence_row, simulate_primitive_rollout  # noqa: E402
from primitive_variant_registry import (  # noqa: E402
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
    start_family_for_entry_role_index,
    start_family_is_compatible,
    variant_row,
)
from primitive_timing_contract import (  # noqa: E402
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    PRIMITIVE_TIMING_CONTRACT_VERSION,
    primitive_timing_contract_row,
)
from state_sampling import archive_state_sample_for_family, archive_state_sample_row  # noqa: E402
from transition_labels import transition_contract_row, transition_row_fields, turn_intent_row_fields  # noqa: E402


W3_SURVIVAL_VERSION = "r7_fixed_lqr_survival_replay_v415_includes_dry_air"
PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.20"
DEFAULT_R5_DISCOVERY_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w01_dense")
DEFAULT_W2_DISCOVERY_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w2_survival")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w3_survival")
W3_TABLE_NAME = "w3_survival_rows"
W3_ENVIRONMENT_CASES = ("dry_air", "w3_randomised_single", "w3_randomised_four")
W3_ACTIVE_FAN_COUNT_SEQUENCE = (1, 2, 3, 4)
REQUIRED_R5_ANNULAR_GP_TRAINING_CASES = (
    "w1_annular_gp_randomised_single",
    "w1_annular_gp_randomised_four",
)
ACCEPTED_W2_SOURCE_STATUSES = ("w2_dense_survival_pass",)
R5_INPUT_KIND = "r5_frozen_bundle_direct"
R5_SELECTED_FOR_R7_CSV = "r5_transition_selected_for_r7.csv"
R5_TRANSITION_TRAINING_MANIFEST_JSON = "r5_transition_training_manifest.json"
LEGACY_W2_INPUT_KIND = "legacy_w2_survivor_registry_diagnostic"
UNKNOWN_INPUT_KIND = "unknown_input_root"
SURVIVAL_STATUS_VOCABULARY = ("blocked", "ready_for_fixed_lqr_replay", "complete", "not_run")
TEST_FIXTURE_LABEL = "test_fixture_not_method_evidence"
BLOCKED_CLAIMS = (
    "W3_robustness_complete",
    "post_W3_library_size_study_ready",
    "governor_validation",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
)


@dataclass(frozen=True)
class W3SurvivalConfig:
    run_id: int
    input_root: Path | None = None
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_label: str = ""
    rows: int | None = None
    seed: int = 30
    paired_tests_per_variant: int = 20
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
    rollout_dt_s: float = CONTROLLER_INPUT_UPDATE_PERIOD_S
    latency_case: str = "nominal"


@dataclass(frozen=True)
class W3ChunkSpec:
    run_id: int
    chunk_index: int
    chunk_count: int
    row_start: int
    row_stop: int
    storage_format: str
    compression_level: int


@dataclass(frozen=True)
class W3ReplayRecord:
    record: FrozenW01ControllerRecord
    transition_entry_class: str = ""
    r5_selection_row: dict[str, object] = field(default_factory=dict)


def discover_latest_r5_root_for_w3(discovery_root: Path = DEFAULT_R5_DISCOVERY_ROOT) -> Path | None:
    root = filesystem_path(discovery_root)
    if not root.is_dir():
        return None
    candidates = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            numeric_id = int(path.name)
        except ValueError:
            continue
        if _r5_root_is_default_w3_eligible(Path(path)):
            candidates.append((numeric_id, Path(path)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def discover_latest_w2_root_for_w3(discovery_root: Path = DEFAULT_W2_DISCOVERY_ROOT) -> Path | None:
    root = filesystem_path(discovery_root)
    if not root.is_dir():
        return None
    candidates = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            numeric_id = int(path.name)
        except ValueError:
            continue
        if _w2_root_is_default_w3_eligible(Path(path)):
            candidates.append((numeric_id, Path(path)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def write_w3_fixture_survivor_root(*, fixture_root: Path, source_w01_root: Path) -> Path:
    """Create a non-method W2 survivor fixture for W3 plumbing tests only."""

    root = filesystem_path(fixture_root)
    if "03_Control" in root.parts and "05_Results" in root.parts:
        raise ValueError("W3 fixture survivor roots must not be written under method result roots.")
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    source_bundle_path = filesystem_path(source_w01_root / "manifests" / "frozen_w01_controller_bundle.json")
    if not source_bundle_path.is_file():
        raise FileNotFoundError(f"missing source frozen bundle: {source_bundle_path}")
    bundle = load_frozen_w01_controller_bundle(source_bundle_path)
    ready = [record for record in bundle.records if record.bundle_status == FROZEN_CONTROLLER_READY]
    if not ready:
        raise ValueError("source W01 bundle has no ready frozen controllers for fixture plumbing.")
    shutil.copyfile(source_bundle_path, manifests / "frozen_w01_controller_bundle.json")
    record = ready[0]
    survivor = variant_row(record.variant)
    survivor.update({"w2_variant_status": "survived", "eligible_for_w3": True})
    _write_json(
        Path(manifests / "w2_survival_manifest.json"),
        {
            "status": "w2_artifact_smoke_pass",
            "fixture_evidence_label": TEST_FIXTURE_LABEL,
            "source_w01_root": Path(source_w01_root).as_posix(),
            "fixed_lqr_replay_only": True,
            "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
        },
    )
    _write_json(
        Path(manifests / "w2_survivor_registry.json"),
        {
            "survivor_registry_version": "w2_survivor_registry_v1",
            "status": "survived_variants_available",
            "survivor_count": 1,
            "fixture_evidence_label": TEST_FIXTURE_LABEL,
            "survivors": [survivor],
        },
    )
    return Path(root)


def run_w3_survival(config: W3SurvivalConfig) -> dict[str, object]:
    """Run chunked fixed-LQR W3 replay from frozen R5 controllers, or block cleanly."""

    config = replace(config, input_root=_resolve_w3_input_root(config.input_root))
    if not math.isclose(float(config.rollout_dt_s), CONTROLLER_INPUT_UPDATE_PERIOD_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rollout_dt_s_not_v411_0p020s")
    storage_format = resolve_storage_format(config.storage_format)
    run_root = Path(config.output_root) / _run_folder_name(config.run_id, config.run_label)
    for subdir in ("manifests", "metrics", "reports", "chunk_manifests", "tables"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    blocked_reason = _input_blocked_reason(Path(config.input_root))
    if blocked_reason:
        _write_blocked_outputs(run_root=run_root, config=config, storage_format=storage_format, blocked_reason=blocked_reason)
        return _result_payload(run_root, "blocked")

    bundle_path = Path(config.input_root) / "manifests" / "frozen_w01_controller_bundle.json"
    bundle = load_frozen_w01_controller_bundle(bundle_path)
    records = _input_records(Path(config.input_root), bundle)
    if not records:
        _write_blocked_outputs(
            run_root=run_root,
            config=config,
            storage_format=storage_format,
            blocked_reason="missing_reconstructable_R5_or_legacy_W2_frozen_records",
        )
        return _result_payload(run_root, "blocked")

    row_count = int(config.rows) if config.rows is not None else len(records) * len(W3_ENVIRONMENT_CASES) * int(config.paired_tests_per_variant)
    schedule = _chunk_schedule(config, row_count=row_count, storage_format=storage_format)
    if config.stop_after_chunks is not None:
        schedule = schedule[: max(0, int(config.stop_after_chunks))]
    worker_decision = worker_count_decision(config.workers, max_workers=config.max_workers)
    fixture_label = _fixture_label(Path(config.input_root))
    _write_run_manifest(
        run_root=run_root,
        config=config,
        storage_format=storage_format,
        status="ready_for_fixed_lqr_replay" if config.dry_run_schedule else "running",
        row_count=0 if config.dry_run_schedule else row_count,
        survivor_count=len(records),
        blocked_reason="",
        schedule=schedule,
        selected_worker_count=int(worker_decision.selected_worker_count),
        worker_decision=worker_decision.as_manifest_fields(),
        fixture_label=fixture_label,
    )
    if config.dry_run_schedule:
        _write_empty_table_manifest(run_root, config.run_id, storage_format)
        _write_chunk_summary(run_root, [_scheduled_chunk_row(chunk, run_root) for chunk in schedule])
        _write_empty_metrics(run_root, row_count=row_count, status="ready_for_fixed_lqr_replay", fixture_label=fixture_label)
        _write_file_size_audit(run_root)
        _write_reports(run_root=run_root, status="ready_for_fixed_lqr_replay", row_count=0, blocked_reason="", fixture_label=fixture_label)
        return _result_payload(run_root, "ready_for_fixed_lqr_replay")

    partitions = []
    chunk_records: list[dict[str, object]] = []
    pending_chunks: list[W3ChunkSpec] = []
    for chunk in schedule:
        status = _existing_chunk_status(chunk, run_root=run_root)
        if status == "complete":
            if config.resume or config.repair_incomplete:
                partitions.append(_partition_from_existing(chunk, run_root))
                chunk_records.append(_completed_chunk_row(chunk, run_root, status="skipped"))
                continue
            raise RuntimeError(f"complete W3 chunk already exists: c{chunk.chunk_index:05d}; use --resume")
        if status == "corrupt":
            if not config.repair_incomplete:
                raise RuntimeError(f"corrupt W3 chunk exists: c{chunk.chunk_index:05d}; use --repair-incomplete")
            _remove_chunk_files(chunk, run_root)
        pending_chunks.append(chunk)

    executed_partitions, executed_records, failures = _execute_pending_chunks(
        pending_chunks,
        run_root=run_root,
        config=config,
        records=records,
        fixture_label=fixture_label,
        selected_worker_count=int(worker_decision.selected_worker_count),
    )
    partitions.extend(executed_partitions)
    chunk_records.extend(executed_records)
    if failures and not config.continue_on_chunk_failure:
        _write_chunk_summary(run_root, sorted(chunk_records, key=lambda item: int(item.get("chunk_index", -1))))
        raise RuntimeError(f"W3 chunk failed: c{int(failures[0]['chunk_index']):05d}: {failures[0].get('error', '')}")

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
    rows_written = _write_metrics_from_partitions(run_root, partitions)
    _write_randomisation_manifest(run_root=run_root, config=config, row_count=rows_written)
    _write_file_size_audit(run_root)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        storage_format=storage_format,
        status="complete",
        row_count=rows_written,
        survivor_count=len(records),
        blocked_reason="",
        schedule=schedule,
        selected_worker_count=int(worker_decision.selected_worker_count),
        worker_decision=worker_decision.as_manifest_fields(),
        fixture_label=fixture_label,
    )
    _write_reports(run_root=run_root, status="complete", row_count=rows_written, blocked_reason="", fixture_label=fixture_label)
    return _result_payload(run_root, "complete")


def _resolve_w3_input_root(input_root: Path | None) -> Path:
    if input_root is not None:
        return Path(input_root)
    discovered_r5 = discover_latest_r5_root_for_w3()
    if discovered_r5 is not None:
        return discovered_r5
    discovered = discover_latest_w2_root_for_w3()
    if discovered is not None:
        return discovered
    return DEFAULT_R5_DISCOVERY_ROOT / "__missing_eligible_r5_root__"


def _r5_root_is_default_w3_eligible(root: Path) -> bool:
    manifest_path = filesystem_path(root / "manifests" / "run_manifest.json")
    bundle_path = filesystem_path(root / "manifests" / "frozen_w01_controller_bundle.json")
    training_path = filesystem_path(root / "manifests" / R5_TRANSITION_TRAINING_MANIFEST_JSON)
    selected_path = filesystem_path(root / "metrics" / R5_SELECTED_FOR_R7_CSV)
    if not manifest_path.is_file() or not bundle_path.is_file() or not training_path.is_file() or not selected_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        training = json.loads(training_path.read_text(encoding="ascii"))
        bundle = load_frozen_w01_controller_bundle(bundle_path)
        selected = pd.read_csv(selected_path)
    except Exception:
        return False
    timing = manifest.get("primitive_timing_contract", {})
    official_w1 = set(manifest.get("official_W_layers", {}).get("W1", []))
    ready_count = sum(1 for record in bundle.records if record.bundle_status == FROZEN_CONTROLLER_READY)
    return (
        str(manifest.get("project_title_version", "")) == PROJECT_TITLE_VERSION
        and str(timing.get("primitive_timing_contract_version", "")) == PRIMITIVE_TIMING_CONTRACT_VERSION
        and bool(manifest.get("w01_dense_evidence_complete", False))
        and str(manifest.get("method_evidence_level", "")) == "w01_dense_evidence_complete"
        and str(manifest.get("R7_W3_direct_source", "")) == "r5_transition_selected_for_r7_frozen_controller_bundle"
        and str(training.get("status", "")) == "passed"
        and set(REQUIRED_R5_ANNULAR_GP_TRAINING_CASES).issubset(official_w1)
        and ready_count > 0
        and not selected.empty
    )


def _w2_root_is_default_w3_eligible(root: Path) -> bool:
    manifest_path = filesystem_path(root / "manifests" / "w2_survival_manifest.json")
    survivor_path = filesystem_path(root / "manifests" / "w2_survivor_registry.json")
    bundle_path = filesystem_path(root / "manifests" / "frozen_w01_controller_bundle.json")
    if not manifest_path.is_file() or not survivor_path.is_file() or not bundle_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        survivors = json.loads(survivor_path.read_text(encoding="ascii"))
    except Exception:
        return False
    timing = manifest.get("primitive_timing_contract", {})
    return (
        str(manifest.get("status", "")) in ACCEPTED_W2_SOURCE_STATUSES
        and str(manifest.get("project_title_version", "")) == PROJECT_TITLE_VERSION
        and str(timing.get("primitive_timing_contract_version", "")) == PRIMITIVE_TIMING_CONTRACT_VERSION
        and str(survivors.get("status", "")) == "survived_variants_available"
        and int(survivors.get("survivor_count", 0)) > 0
        and str(survivors.get("fixture_evidence_label", "")) != TEST_FIXTURE_LABEL
    )


def _source_input_kind(input_root: Path) -> str:
    r5_manifest = filesystem_path(input_root / "manifests" / "run_manifest.json")
    bundle_path = filesystem_path(input_root / "manifests" / "frozen_w01_controller_bundle.json")
    if r5_manifest.is_file() and bundle_path.is_file():
        return R5_INPUT_KIND
    w2_manifest = filesystem_path(input_root / "manifests" / "w2_survival_manifest.json")
    survivor_registry = filesystem_path(input_root / "manifests" / "w2_survivor_registry.json")
    if w2_manifest.is_file() and survivor_registry.is_file() and bundle_path.is_file():
        return LEGACY_W2_INPUT_KIND
    return UNKNOWN_INPUT_KIND


def _input_blocked_reason(input_root: Path) -> str:
    input_kind = _source_input_kind(input_root)
    if input_kind == R5_INPUT_KIND:
        r5_manifest = Path(input_root) / "manifests" / "run_manifest.json"
        bundle_path = Path(input_root) / "manifests" / "frozen_w01_controller_bundle.json"
        training_path = Path(input_root) / "manifests" / R5_TRANSITION_TRAINING_MANIFEST_JSON
        selected_path = Path(input_root) / "metrics" / R5_SELECTED_FOR_R7_CSV
        if not filesystem_path(r5_manifest).is_file():
            return "missing_R5_run_manifest"
        if not filesystem_path(bundle_path).is_file():
            return "missing_frozen_w01_controller_bundle_from_R5_root"
        if not filesystem_path(training_path).is_file():
            return "missing_r5_transition_training_manifest"
        if not filesystem_path(selected_path).is_file():
            return "missing_r5_transition_selected_for_r7"
        source = json.loads(filesystem_path(r5_manifest).read_text(encoding="ascii"))
        training = json.loads(filesystem_path(training_path).read_text(encoding="ascii"))
        timing = source.get("primitive_timing_contract", {})
        if str(source.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
            return "R5_source_not_v5_project_title"
        if str(timing.get("primitive_timing_contract_version", "")) != PRIMITIVE_TIMING_CONTRACT_VERSION:
            return "R5_source_missing_v411_timing_contract"
        if not bool(source.get("w01_dense_evidence_complete", False)):
            return "R5_source_not_w01_dense_evidence_complete"
        if str(source.get("method_evidence_level", "")) != "w01_dense_evidence_complete":
            return "R5_source_method_evidence_level_not_dense"
        if bool(source.get("W2_required_for_move_on", False)):
            return "R5_source_still_requires_archived_W2"
        if str(source.get("R7_W3_direct_source", "")) != "r5_transition_selected_for_r7_frozen_controller_bundle":
            return "R5_source_missing_direct_W3_contract"
        if str(training.get("status", "")) != "passed":
            return "R5_transition_training_not_passed"
        try:
            selected = pd.read_csv(filesystem_path(selected_path))
        except pd.errors.EmptyDataError:
            return "R5_transition_selection_empty"
        if selected.empty:
            return "R5_transition_selection_empty"
        required_columns = {"primitive_variant_id", "transition_entry_class", "selected_for_r7"}
        if not required_columns.issubset(set(selected.columns)):
            return "R5_transition_selection_missing_required_columns"
        official_w1 = set(source.get("official_W_layers", {}).get("W1", []))
        if not set(REQUIRED_R5_ANNULAR_GP_TRAINING_CASES).issubset(official_w1):
            return "R5_source_missing_annular_gp_randomised_W1_training_cases"
        return ""
    if input_kind == UNKNOWN_INPUT_KIND:
        return "missing_R5_frozen_bundle_or_legacy_W2_survivor_root"
    w2_manifest = Path(input_root) / "manifests" / "w2_survival_manifest.json"
    survivor_registry = Path(input_root) / "manifests" / "w2_survivor_registry.json"
    bundle_path = Path(input_root) / "manifests" / "frozen_w01_controller_bundle.json"
    if not filesystem_path(w2_manifest).is_file():
        return "missing_W2_survival_manifest"
    if not filesystem_path(survivor_registry).is_file():
        return "missing_W2_survivor_registry"
    if not filesystem_path(bundle_path).is_file():
        return "missing_frozen_w01_controller_bundle_from_W2_root"
    source = json.loads(filesystem_path(w2_manifest).read_text(encoding="ascii"))
    survivors = json.loads(filesystem_path(survivor_registry).read_text(encoding="ascii"))
    timing = source.get("primitive_timing_contract", {})
    is_fixture = str(survivors.get("fixture_evidence_label", "")) == TEST_FIXTURE_LABEL
    if not is_fixture and str(source.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "W2_source_not_v5_project_title"
    if not is_fixture and str(timing.get("primitive_timing_contract_version", "")) != PRIMITIVE_TIMING_CONTRACT_VERSION:
        return "W2_source_missing_v411_timing_contract"
    if not is_fixture and source.get("status") not in ACCEPTED_W2_SOURCE_STATUSES:
        return "W2_survival_status_not_accepted_for_W3_fixed_replay"
    if is_fixture and source.get("status") not in {"w2_artifact_smoke_pass", "survived_variants_available"}:
        return "W2_fixture_status_not_accepted_for_W3_plumbing"
    if not is_fixture and not bool(source.get("w2_dense_survival_evidence_complete", False)):
        return "W2_source_not_dense_survival_evidence_complete"
    if survivors.get("status") != "survived_variants_available":
        return "W2_survivor_registry_not_available"
    if int(survivors.get("survivor_count", 0)) <= 0:
        return "missing_W2_surviving_variants"
    return ""


def _input_records(input_root: Path, bundle) -> tuple[W3ReplayRecord, ...]:
    if _source_input_kind(input_root) == R5_INPUT_KIND:
        return _selected_r5_replay_records(input_root, bundle)
    return tuple(W3ReplayRecord(record=record) for record in _surviving_records(input_root, bundle))


def _selected_r5_replay_records(input_root: Path, bundle) -> tuple[W3ReplayRecord, ...]:
    selected = pd.read_csv(filesystem_path(input_root / "metrics" / R5_SELECTED_FOR_R7_CSV))
    if "selected_for_r7" in selected.columns:
        selected = selected[selected["selected_for_r7"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    by_variant_id = bundle.records_by_variant_id
    records: list[W3ReplayRecord] = []
    for row in selected.sort_values(["primitive_id", "transition_entry_class", "selected_rank", "candidate_index"]).to_dict(orient="records"):
        variant_id = str(row.get("primitive_variant_id", ""))
        record = by_variant_id.get(variant_id)
        if record is None or record.bundle_status != FROZEN_CONTROLLER_READY:
            continue
        records.append(
            W3ReplayRecord(
                record=record,
                transition_entry_class=str(row.get("transition_entry_class", "")),
                r5_selection_row=dict(row),
            )
        )
    return tuple(records)


def _surviving_records(input_root: Path, bundle) -> tuple[FrozenW01ControllerRecord, ...]:
    survivor_registry = json.loads(
        filesystem_path(input_root / "manifests" / "w2_survivor_registry.json").read_text(encoding="ascii")
    )
    survivor_ids = {
        str(row.get("primitive_variant_id", ""))
        for row in survivor_registry.get("survivors", [])
    }
    records = []
    by_variant_id = bundle.records_by_variant_id
    for variant_id in sorted(survivor_ids):
        record = by_variant_id.get(variant_id)
        if record is not None and record.bundle_status == FROZEN_CONTROLLER_READY:
            records.append(record)
    return tuple(records)


def _write_chunk(
    chunk: W3ChunkSpec,
    *,
    run_root: Path,
    config: W3SurvivalConfig,
    records: tuple[W3ReplayRecord, ...],
    fixture_label: str,
):
    started = time.time()
    rows = [
        _row_for_index(row_index=row_index, config=config, records=records, fixture_label=fixture_label)
        for row_index in range(int(chunk.row_start), int(chunk.row_stop))
    ]
    frame = pd.DataFrame(rows)
    partition_path = _partition_path(chunk, run_root)
    partition = write_table_partition(frame, partition_path, storage_format=chunk.storage_format, compression_level=chunk.compression_level)
    ended = time.time()
    manifest = {
        "runner_version": W3_SURVIVAL_VERSION,
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
        "table_name": W3_TABLE_NAME,
        "primitive_timing_contract": primitive_timing_contract_row(),
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
    chunk, run_root, config, records, fixture_label = payload
    return _write_chunk(chunk, run_root=Path(run_root), config=config, records=records, fixture_label=str(fixture_label))


def _execute_pending_chunks(
    chunks: list[W3ChunkSpec],
    *,
    run_root: Path,
    config: W3SurvivalConfig,
    records: tuple[W3ReplayRecord, ...],
    fixture_label: str,
    selected_worker_count: int,
):
    if not chunks:
        return [], [], []
    partitions = []
    chunk_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    worker_count = max(1, int(selected_worker_count))
    if worker_count <= 1 or len(chunks) <= 1:
        for chunk in chunks:
            try:
                partition, record = _write_chunk(chunk, run_root=run_root, config=config, records=records, fixture_label=fixture_label)
                partitions.append(partition)
                chunk_records.append(record)
            except Exception as exc:
                failure = _chunk_failure_row(chunk, exc)
                failures.append(failure)
                chunk_records.append({**_scheduled_chunk_row(chunk, run_root), **failure})
                if not config.continue_on_chunk_failure:
                    break
        return partitions, chunk_records, failures
    payloads = [(chunk, run_root, config, records, fixture_label) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_chunk = {executor.submit(_write_chunk_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                partition, record = future.result()
                partitions.append(partition)
                chunk_records.append(record)
            except Exception as exc:
                failure = _chunk_failure_row(chunk, exc)
                failures.append(failure)
                chunk_records.append({**_scheduled_chunk_row(chunk, run_root), **failure})
                if not config.continue_on_chunk_failure:
                    for pending in future_to_chunk:
                        pending.cancel()
                    break
    return partitions, chunk_records, failures


def _row_for_index(
    *,
    row_index: int,
    config: W3SurvivalConfig,
    records: tuple[W3ReplayRecord, ...],
    fixture_label: str,
) -> dict[str, object]:
    replay_record = records[int(row_index) % len(records)]
    record = replay_record.record
    grouped_index = int(row_index) // len(records)
    environment_mode = W3_ENVIRONMENT_CASES[grouped_index % len(W3_ENVIRONMENT_CASES)]
    environment_layer = _environment_layer_for_mode(environment_mode)
    paired_start_index = grouped_index // len(W3_ENVIRONMENT_CASES)
    start_family = _start_family_for_r5_selected_entry_class(
        transition_entry_class=replay_record.transition_entry_class,
        fallback_entry_role=record.variant.entry_role,
        paired_start_index=int(paired_start_index),
    )
    paired_start_key = f"w3_paired_{int(paired_start_index):07d}_{start_family}"
    scheduled_active_fan_count = _scheduled_active_fan_count(
        environment_mode=environment_mode,
        paired_start_index=int(paired_start_index),
    )
    sample = archive_state_sample_for_family(
        start_state_family=start_family,
        paired_start_key=paired_start_key,
        sample_index=int(paired_start_index),
        seed=int(config.seed),
        W_layer=environment_layer,
        environment_mode=environment_mode,
    )
    environment = environment_instance_for_mode(
        environment_layer,
        environment_mode,
        int(config.seed) + int(row_index),
        randomisation_config=EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count if environment_mode == "w3_randomised_four" else None
        ),
    )
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding(environment_layer, metadata, randomisation_seed=int(config.seed) + int(row_index))
    wind_field = wind_field_for_binding(binding)
    context = build_environment_context(
        sample.state_vector,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=str(config.latency_case),
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    variant = record.variant
    primitive = primitive_by_id(variant.primitive_id)
    compatible = start_family_is_compatible(entry_role=variant.entry_role, start_state_family=start_family)
    rollout_config = RolloutConfig(
        W_layer=environment_layer,
        dt_s=float(config.rollout_dt_s),
        rollout_backend="model_backed_lqr",
        wind_mode="none" if environment_layer == "W0" else "panel",
    )
    if not compatible or binding.surrogate_binding_status != "ready":
        failure_label = ENTRY_ROLE_REJECTION_LABEL if not compatible else "w3_surrogate_binding_blocked"
        termination = ENTRY_ROLE_REJECTION_STATUS if not compatible else str(binding.blocked_reason)
        evidence = blocked_rollout_evidence(
            rollout_id=f"w3r{int(config.run_id):03d}_{int(row_index):07d}",
            episode_id=f"w3_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=RolloutConfig(
                W_layer=environment_layer,
                dt_s=float(config.rollout_dt_s),
                rollout_backend="blocked_lqr",
                wind_mode="none" if environment_layer == "W0" else "panel",
            ),
            failure_label=failure_label,
            controller=record.controller,
            controller_selection_status="W3_fixed_lqr_survival_replay",
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
            termination_cause=termination,
        )
        row = rollout_evidence_row(evidence)
        implementation = None
        plant = None
    else:
        implementation = implementation_instance_for_layer("W3", int(config.seed) + int(row_index), latency_case=str(config.latency_case))
        plant = plant_instance_for_layer("W3", int(config.seed) + int(row_index))
        evidence = simulate_primitive_rollout(
            rollout_id=f"w3r{int(config.run_id):03d}_{int(row_index):07d}",
            episode_id=f"w3_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=rollout_config,
            wind_field=wind_field,
            implementation_instance=implementation,
            plant_instance=plant,
            controller=record.controller,
            controller_selection_status="W3_fixed_lqr_survival_replay",
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
        )
        row = rollout_evidence_row(evidence)
    row.update(archive_state_sample_row(sample))
    row.update({f"variant_{key}": value for key, value in variant_row(variant).items()})
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    row.update({f"surrogate_{key}": value for key, value in surrogate_binding_row(binding).items()})
    row.update({f"environment_{key}": value for key, value in environment_instance_row(environment).items()})
    row.update(
        transition_row_fields(
            row,
            entry_role=variant.entry_role,
            start_state_family=start_family,
            primitive_step_index=0 if start_family == "launch_gate" else None,
        )
    )
    row.update(turn_intent_row_fields(row))
    if implementation is None or plant is None:
        row.update(_blocked_instance_prefix_rows("implementation"))
        row.update(_blocked_instance_prefix_rows("plant"))
    else:
        row.update({f"implementation_{key}": value for key, value in implementation_instance_row(implementation).items()})
        row.update({f"plant_{key}": value for key, value in plant_instance_row(plant).items()})
    row.update(
        {
            "runner_version": W3_SURVIVAL_VERSION,
            "project_title_version": PROJECT_TITLE_VERSION,
            "run_stage": "W3_fixed_LQR_randomised_survival_replay",
            "row_index": int(row_index),
            "source_input_kind": _source_input_kind(Path(config.input_root)),
            "source_input_root": Path(config.input_root).as_posix(),
            "source_w01_root": Path(config.input_root).as_posix()
            if _source_input_kind(Path(config.input_root)) == R5_INPUT_KIND
            else "",
            "source_w2_root": Path(config.input_root).as_posix()
            if _source_input_kind(Path(config.input_root)) == LEGACY_W2_INPUT_KIND
            else "",
            "source_evidence_label": fixture_label,
            "r5_selected_transition_entry_class": str(replay_record.transition_entry_class),
            "r5_transition_training_score": _selection_float(replay_record, "r5_transition_training_score"),
            "r5_transition_success_lcb": _selection_float(replay_record, "transition_success_lcb"),
            "r5_transition_hard_failure_ucb": _selection_float(replay_record, "hard_failure_ucb"),
            "r5_turn_intent_success_lcb": _selection_float(replay_record, "turn_intent_success_lcb"),
            "r5_turn_intent_score_mean": _selection_float(replay_record, "turn_intent_score_mean"),
            "r5_turn_intent_roll_rate_score_mean": _selection_float(replay_record, "turn_intent_roll_rate_score_mean"),
            "r5_signed_turn_bank_delta_mean_rad": _selection_float(replay_record, "signed_turn_bank_delta_mean_rad"),
            "r5_signed_turn_exit_roll_rate_mean_rad_s": _selection_float(replay_record, "signed_turn_exit_roll_rate_mean_rad_s"),
            "r5_selected_rank": str(replay_record.r5_selection_row.get("selected_rank", "")),
            "primitive_variant_id": variant.primitive_variant_id,
            "entry_role": variant.entry_role,
            "entry_role_compatible": bool(compatible),
            "environment_mode": environment_mode,
            "environment_validation_layer": environment_layer,
            "scheduled_active_fan_count": int(sum(bool(value) for value in environment.active_fan_mask)),
            "fixed_lqr_replay_only": True,
            "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
            "w3_environment_contract": "dry_air_plus_randomised_single_and_four_annular_gp_survival_replay_only",
            "baseline_controller_active": False,
            "claim_boundary": (
                "test_fixture_not_method_evidence"
                if fixture_label == TEST_FIXTURE_LABEL
                else "simulation_only_W3_fixed_LQR_randomised_replay_smoke_no_robustness_claim"
            ),
        }
    )
    return row


def _start_family_for_r5_selected_entry_class(
    *,
    transition_entry_class: str,
    fallback_entry_role: str,
    paired_start_index: int,
) -> str:
    entry_class = str(transition_entry_class)
    if entry_class == "launch_gate":
        return "launch_gate"
    if entry_class == "inflight_stable":
        return ("inflight_nominal", "inflight_lift_region")[int(paired_start_index) % 2]
    if entry_class == "boundary_near":
        return "inflight_boundary_near"
    if entry_class == "recoverable_degraded":
        return "inflight_recovery_edge"
    return start_family_for_entry_role_index(entry_role=fallback_entry_role, index=int(paired_start_index))


def _selection_float(replay_record: W3ReplayRecord, key: str) -> float:
    try:
        return float(replay_record.r5_selection_row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _scheduled_active_fan_count(*, environment_mode: str, paired_start_index: int) -> int:
    if str(environment_mode) == "dry_air":
        return 0
    if str(environment_mode) == "w3_randomised_four":
        return int(W3_ACTIVE_FAN_COUNT_SEQUENCE[int(paired_start_index) % len(W3_ACTIVE_FAN_COUNT_SEQUENCE)])
    return 1


def _environment_layer_for_mode(environment_mode: str) -> str:
    if str(environment_mode) == "dry_air":
        return "W0"
    return "W3"


def _chunk_schedule(config: W3SurvivalConfig, *, row_count: int, storage_format: str) -> list[W3ChunkSpec]:
    chunk_count = int(math.ceil(int(row_count) / int(config.candidate_chunk_size)))
    return [
        W3ChunkSpec(
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


def _existing_chunk_status(chunk: W3ChunkSpec, *, run_root: Path) -> str:
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


def _remove_chunk_files(chunk: W3ChunkSpec, run_root: Path) -> None:
    for path in (_partition_path(chunk, run_root), _chunk_manifest_path(chunk, run_root)):
        fs_path = filesystem_path(path)
        if fs_path.exists():
            fs_path.unlink()


def _partition_from_existing(chunk: W3ChunkSpec, run_root: Path):
    from dense_archive_table_io import TablePartition

    manifest = json.loads(filesystem_path(_chunk_manifest_path(chunk, run_root)).read_text(encoding="ascii"))
    frame = read_table_partition(_partition_path(chunk, run_root), storage_format=chunk.storage_format)
    return TablePartition(
        table_name=W3_TABLE_NAME,
        relative_path=str(manifest["partition_path"]),
        storage_format=chunk.storage_format,
        row_count=len(frame),
        byte_count=int(filesystem_path(_partition_path(chunk, run_root)).stat().st_size),
        columns=tuple(str(column) for column in frame.columns),
        checksum_sha256=file_sha256(_partition_path(chunk, run_root)),
    )


def _partition_path(chunk: W3ChunkSpec, run_root: Path) -> Path:
    return run_root / "tables" / W3_TABLE_NAME / f"c{int(chunk.chunk_index):05d}.{table_extension(chunk.storage_format)}"


def _chunk_manifest_path(chunk: W3ChunkSpec, run_root: Path) -> Path:
    return run_root / "chunk_manifests" / W3_TABLE_NAME / f"c{int(chunk.chunk_index):05d}.json"


def _scheduled_chunk_row(chunk: W3ChunkSpec, run_root: Path) -> dict[str, object]:
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


def _completed_chunk_row(chunk: W3ChunkSpec, run_root: Path, *, status: str) -> dict[str, object]:
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


def _chunk_failure_row(chunk: W3ChunkSpec, exc: Exception) -> dict[str, object]:
    return {
        "chunk_index": int(chunk.chunk_index),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_blocked_outputs(
    *,
    run_root: Path,
    config: W3SurvivalConfig,
    storage_format: str,
    blocked_reason: str,
) -> None:
    _write_empty_table_manifest(run_root, config.run_id, storage_format)
    _write_empty_metrics(run_root, row_count=0, status="blocked", fixture_label="")
    _write_file_size_audit(run_root)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        storage_format=storage_format,
        status="blocked",
        row_count=0,
        survivor_count=0,
        blocked_reason=blocked_reason,
        schedule=[],
        selected_worker_count=0,
        worker_decision={},
        fixture_label="",
    )
    _write_reports(run_root=run_root, status="blocked", row_count=0, blocked_reason=blocked_reason, fixture_label="")


def _write_run_manifest(
    *,
    run_root: Path,
    config: W3SurvivalConfig,
    storage_format: str,
    status: str,
    row_count: int,
    survivor_count: int,
    blocked_reason: str,
    schedule: list[W3ChunkSpec],
    selected_worker_count: int,
    worker_decision: dict[str, object],
    fixture_label: str,
) -> None:
    input_root = Path(config.input_root)
    source_input_kind = _source_input_kind(input_root)
    source_r5_manifest = _read_json_or_empty(input_root / "manifests" / "run_manifest.json")
    source_r5_training_manifest = _read_json_or_empty(input_root / "manifests" / R5_TRANSITION_TRAINING_MANIFEST_JSON)
    source_w2_manifest = _read_json_or_empty(input_root / "manifests" / "w2_survival_manifest.json")
    source_w01_root = input_root.as_posix() if source_input_kind == R5_INPUT_KIND else str(source_w2_manifest.get("source_w01_root", ""))
    source_w2_root = input_root.as_posix() if source_input_kind == LEGACY_W2_INPUT_KIND else ""
    manifest = {
        "version": W3_SURVIVAL_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_label": str(config.run_label).strip(),
        "input_root": input_root.as_posix(),
        "input_contract": "R5 selected transition-object frozen controller bundle for W3 held-out validation; legacy W2 survivor roots are diagnostic fixtures only",
        "source_input_kind": source_input_kind,
        "source_w01_root": source_w01_root,
        "source_w2_root": source_w2_root,
        "source_r5_status": str(source_r5_manifest.get("status", "")),
        "source_r5_method_evidence_level": str(source_r5_manifest.get("method_evidence_level", "")),
        "source_r5_w01_dense_evidence_complete": bool(source_r5_manifest.get("w01_dense_evidence_complete", False)),
        "source_r5_transition_training_status": str(source_r5_training_manifest.get("status", "")),
        "source_r5_selected_transition_object_count": int(
            source_r5_training_manifest.get("selected_transition_object_count", 0) or 0
        ),
        "source_r5_selected_for_r7_csv": f"metrics/{R5_SELECTED_FOR_R7_CSV}" if source_input_kind == R5_INPUT_KIND else "",
        "source_w2_status": str(source_w2_manifest.get("status", "")),
        "source_w2_dense_survival_evidence_complete": bool(
            source_w2_manifest.get("w2_dense_survival_evidence_complete", False)
        ),
        "source_w2_required_for_move_on": False,
        "row_count": int(row_count),
        "survivor_count": int(survivor_count),
        "storage_format": storage_format,
        "compression_level": int(config.compression_level),
        "rollout_dt_s": float(config.rollout_dt_s),
        "primitive_timing_contract": primitive_timing_contract_row(),
        "primitive_timing_contract_status": "compliant",
        "method_evidence_level": (
            "w3_dense_survival_pass"
            if status == "complete" and fixture_label != TEST_FIXTURE_LABEL
            else "w3_fixture_or_smoke_only"
            if fixture_label == TEST_FIXTURE_LABEL
            else status
        ),
        "paired_tests_per_variant": int(config.paired_tests_per_variant),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "chunk_count": int(len(schedule)),
        "selected_worker_count": int(selected_worker_count),
        "worker_decision": worker_decision,
        "dry_run_schedule": bool(config.dry_run_schedule),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "source_evidence_label": fixture_label,
        "test_fixture_not_method_evidence": bool(fixture_label == TEST_FIXTURE_LABEL),
        "fixed_lqr_replay_only": True,
        "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
        "entry_role_regime_separation_policy": "role_aware_start_family_schedule_launch_inflight_recovery_not_mixed",
        "transition_contract": transition_contract_row(),
        "transition_success_policy": "R7_survival_requires_transition_compatibility_not_local_rollout_success_only",
        "forbidden_mutations": "Q/R,K,reference,horizon,entry_set,entry_role",
        "status_vocabulary": list(SURVIVAL_STATUS_VOCABULARY),
        "redesign_policy": "new_ids_return_to_W01",
        "blocked_reason": blocked_reason,
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "w3_survival_manifest.json", manifest)


def _write_randomisation_manifest(*, run_root: Path, config: W3SurvivalConfig, row_count: int) -> None:
    _write_json(
        run_root / "manifests" / "w3_randomisation_manifest.json",
        {
            "version": "w3_randomisation_manifest_v1",
            "row_count": int(row_count),
            "environment_modes": list(W3_ENVIRONMENT_CASES),
            "seed": int(config.seed),
            "randomisation_source": "env_instance_W3_single_layer_composed_annular_gp_modes",
            "active_fan_count_policy": "dry_air_zero_fans_w3_randomised_single_fixed_one_w3_randomised_four_balanced_1_2_3_4",
            "w3_randomised_four_active_fan_count_sequence": list(W3_ACTIVE_FAN_COUNT_SEQUENCE),
            "duplicate_strength_width_centre_wrapper": "disabled",
            "active_strength_source": "fan_power_scales",
            "spatial_shift_source": "fan_positions_m",
            "shape_spread_source": "updraft_width_scale",
            "W3_randomisation_changes_timing_terms_where_configured": True,
        },
    )


def _write_metrics_from_partitions(run_root: Path, partitions: Iterable[object]) -> int:
    counters: dict[str, dict[str, int]] = {}
    active_fan_counts: dict[tuple[str, int], int] = {}
    row_count = 0
    for partition in partitions:
        frame = read_table_partition(run_root / "tables" / partition.relative_path, storage_format=partition.storage_format)
        row_count += int(len(frame))
        if "scheduled_active_fan_count" in frame.columns and "environment_mode" in frame.columns:
            active_counts = frame[["environment_mode", "scheduled_active_fan_count"]].copy()
            active_counts["environment_mode"] = active_counts["environment_mode"].fillna("").astype(str)
            active_counts["scheduled_active_fan_count"] = pd.to_numeric(
                active_counts["scheduled_active_fan_count"],
                errors="coerce",
            ).fillna(0).astype(int)
            for (mode, active_count), count in active_counts.value_counts(dropna=False).items():
                key = (str(mode), int(active_count))
                active_fan_counts[key] = active_fan_counts.get(key, 0) + int(count)
        for column in (
            "outcome_class",
            "failure_label",
            "boundary_use_class",
            "environment_mode",
            "start_state_family",
            "timing_state_source",
            "source_evidence_label",
            "scheduled_active_fan_count",
            "transition_entry_class",
            "transition_exit_class",
            "transition_pair",
            "transition_chain_compatible",
            "transition_failure_reason",
            "turn_intent_label",
            "turn_intent_status",
            "turn_intent_correct_sign",
        ):
            if column not in frame.columns:
                continue
            values = frame[column].fillna("").astype(str).value_counts(dropna=False)
            axis = counters.setdefault(column, {})
            for value, count in values.items():
                axis[str(value)] = axis.get(str(value), 0) + int(count)
    rows = [
        {"coverage_axis": axis, "value": value, "row_count": count}
        for axis, values in counters.items()
        for value, count in sorted(values.items())
    ]
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "w3_survival_summary.csv"), index=False)
    active_rows = [
        {
            "environment_mode": mode,
            "active_fan_count": active_count,
            "row_count": count,
        }
        for (mode, active_count), count in sorted(active_fan_counts.items())
    ]
    pd.DataFrame(active_rows).to_csv(filesystem_path(run_root / "metrics" / "w3_active_fan_count_audit.csv"), index=False)
    return int(row_count)


def _write_empty_metrics(run_root: Path, *, row_count: int, status: str, fixture_label: str) -> None:
    pd.DataFrame(
        [
            {
                "coverage_axis": "status",
                "value": status,
                "row_count": int(row_count),
                "source_evidence_label": fixture_label,
            }
        ]
    ).to_csv(filesystem_path(run_root / "metrics" / "w3_survival_summary.csv"), index=False)
    pd.DataFrame(
        columns=["environment_mode", "active_fan_count", "row_count"]
    ).to_csv(filesystem_path(run_root / "metrics" / "w3_active_fan_count_audit.csv"), index=False)


def _write_empty_table_manifest(run_root: Path, run_id: int, storage_format: str) -> None:
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(run_id=int(run_id), root=run_root.as_posix(), storage_format=storage_format, tables=()),
    )


def _run_folder_name(run_id: int, run_label: str = "") -> str:
    label = str(run_label).strip()
    return label if label else f"{int(run_id):03d}"


def _write_chunk_summary(run_root: Path, records: list[dict[str, object]]) -> None:
    pd.DataFrame(records).to_csv(filesystem_path(run_root / "metrics" / "chunk_summary.csv"), index=False)


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
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                "dense_table_partition": rel.startswith(f"tables/{W3_TABLE_NAME}/"),
            }
        )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def _write_reports(*, run_root: Path, status: str, row_count: int, blocked_reason: str, fixture_label: str) -> None:
    lines = [
        "# W3 Fixed-LQR Survival Replay",
        "",
        f"- Status: `{status}`",
        f"- Rows written: `{int(row_count)}`",
        f"- Blocked reason: `{blocked_reason}`",
        "- Input contract: `R5 frozen bundle direct; R6/W2 archived diagnostic only`",
        f"- Source evidence label: `{fixture_label}`",
        f"- Test fixture not method evidence: `{fixture_label == TEST_FIXTURE_LABEL}`",
        "- Fixed-LQR replay only: `True`",
        "- Q/R, K, reference, horizon, entry set, and entry role mutation: `False`",
        "- R7 environment cases: `dry_air`, `w3_randomised_single`, `w3_randomised_four`",
        "- W3 active-fan-count policy: `dry_air=0`, `w3_randomised_single=1`, `w3_randomised_four=balanced 1/2/3/4`",
        "- Chunked/resumable dense runtime contract: `True`",
        "- W3 robustness, post-W3 library-size readiness, hardware readiness, transfer, and mission success remain blocked.",
        "",
    ]
    filesystem_path(run_root / "reports" / "w3_survival_report.md").write_text("\n".join(lines), encoding="ascii")
    filesystem_path(run_root / "reports" / "l9_w3_move_on_check.md").write_text("\n".join(lines), encoding="ascii")


def _fixture_label(input_root: Path) -> str:
    path = filesystem_path(input_root / "manifests" / "w2_survivor_registry.json")
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except Exception:
        return ""
    return str(payload.get("fixture_evidence_label", ""))


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _read_json_or_empty(path: Path) -> dict[str, object]:
    try:
        return json.loads(filesystem_path(path).read_text(encoding="ascii"))
    except Exception:
        return {}


def _result_payload(run_root: Path, status: str) -> dict[str, object]:
    return {
        "status": status,
        "run_root": run_root.as_posix(),
        "manifest": (run_root / "manifests" / "w3_survival_manifest.json").as_posix(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W3 fixed-LQR survival replay from a frozen R5 controller bundle.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--paired-tests-per-variant", type=int, default=20)
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
    result = run_w3_survival(
        W3SurvivalConfig(
            run_id=args.run_id,
            input_root=args.input_root,
            output_root=args.output_root,
            rows=args.rows,
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
