from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
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
from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from directional_residual_lift_belief import (  # noqa: E402
    DirectionalResidualLiftBelief,
    DirectionalResidualObservation,
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    update_directional_residual_lift_belief,
)
from env_ctx import build_environment_context  # noqa: E402
from env_instance import (  # noqa: E402
    EnvironmentRandomisationConfig,
    environment_instance_for_mode,
    environment_metadata_from_instance,
)
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from context_conditioned_outcome import context_conditioned_outcome, lookup_outcome_for_identity  # noqa: E402
from episode_selector import select_compact_representative, selector_decision_row  # noqa: E402
from frozen_w01_controller_bundle import FROZEN_CONTROLLER_READY, load_frozen_w01_controller_bundle  # noqa: E402
from implementation_instance import implementation_instance_for_layer  # noqa: E402
from plant_instance import plant_instance_for_layer  # noqa: E402
from prim_cat import primitive_by_id  # noqa: E402
from prim_roll import RolloutConfig, rollout_evidence_row, simulate_primitive_rollout  # noqa: E402
from primitive_timing_contract import PRIMITIVE_FINITE_HORIZON_S, primitive_timing_contract_row  # noqa: E402
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS  # noqa: E402
from state_contract import STATE_INDEX, as_state_vector  # noqa: E402
from state_sampling import archive_state_sample_for_family  # noqa: E402
from transition_labels import (
    classify_state,
    entry_classes_for_state_class,
    required_entry_role_for_state_class,
    start_family_for_state_class,
    transition_contract_row,
    transition_row_fields,
)  # noqa: E402
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GovernorConfig, governor_config_to_row  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.3"
VALIDATION_VERSION = "repeated_launch_fixed_case_rollout_preflight_v7"
GOVERNOR_TUNING_HANDOFF_VERSION = "governor_tuning_handoff_v2"
HISTORY_LENGTHS = (5, 20, 100)
SAFE_EXPLORE_ABLATION_HISTORY_LENGTH = 20
HISTORY_LENGTH_SUM = sum(HISTORY_LENGTHS) + SAFE_EXPLORE_ABLATION_HISTORY_LENGTH
EMPTY_FROZEN_PRIOR_BASELINE_ID = "empty_frozen_prior_baseline"
BASELINE_POLICY_IDS = ("no_memory_baseline",)
MEMORY_POLICY_PREFIX = "directional_3d_residual_memory"
SAFE_EXPLORE_POLICY_PREFIX = "safe_explore_then_exploit"
POLICY_HISTORY_CONDITIONS = (
    "no_memory_baseline",
    "directional_3d_residual_memory_h5",
    "directional_3d_residual_memory_h20",
    "directional_3d_residual_memory_h100",
    "safe_explore_then_exploit_h20",
)
R9_PREFLIGHT_CASES_PER_BLOCK = 1
R9_OUTER_CASES_PER_CONDITION = 3 * R9_PREFLIGHT_CASES_PER_BLOCK
R9_EXPECTED_FINAL_HELDOUT_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R9_OUTER_CASES_PER_CONDITION
R9_EXPECTED_HISTORY_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * R9_OUTER_CASES_PER_CONDITION * HISTORY_LENGTH_SUM
DEFAULT_LIBRARY_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/post_w3_library_size_study/001")
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/outcome_model/003")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/repeated_launch_validation")
TABLE_NAMES = (
    "episode_summary",
    "primitive_execution_log",
    "candidate_score_log",
    "selector_decision_log",
    "memory_residual_update_log",
    "belief_snapshot_log",
)
SCHEDULE_INLINE_ROW_LIMIT = 50_000
SCHEDULE_PARTITION_ROW_COUNT = 50_000
CANDIDATE_SCORE_TOP_K_PER_DECISION = 10
THESIS_FACING_WORKFLOW = "R5 -> R7 -> R8 -> R10 -> R11 -> Reality"
R9_THESIS_REPORTING_STATUS = "internal_preflight_excluded_from_thesis_workflow_narrative"
LAUNCH_SEQUENCE_POLICY_ID = "state_class_transition_entry_governor_no_launch_specific_family"
FIRST_PRIMITIVE_START_FAMILY = "launch_gate"
POST_LAUNCH_START_FAMILY = "inflight_nominal"
BOUNDARY_RECOVERY_START_FAMILY = "inflight_boundary_near"
TERMINAL_SAFE_EXIT_START_FAMILY = "inflight_recovery_edge"
ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID = "active_fan_number_variation"
BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID = "arena_wide_fan_position_generalisation"
CHANGED_CASE_VALIDATION_STAGE_IDS = {"R10", "R11"}
R10_NOMINAL_FAN_POSITION_BLOCK_IDS = (
    "nominal_single_fan_perturbations",
    "nominal_four_fan_perturbations",
)
R10_SHIFTED_FAN_POSITION_BLOCK_IDS = (
    "shifted_single_fan_positions",
    "shifted_four_fan_positions",
)
R10_FIXED_BASE_POSITION_BLOCK_IDS = (
    *R10_NOMINAL_FAN_POSITION_BLOCK_IDS,
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
)
R10_ACTIVE_FAN_COUNT_VARIATION_BLOCK_IDS = (
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
)
R10_FIXED_FOUR_ACTIVE_BLOCK_IDS = (
    "nominal_four_fan_perturbations",
    "shifted_four_fan_positions",
)
R10_ACTIVE_FAN_COUNT_SEQUENCE = (1, 2, 3, 4)
R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M = ((0.0, 8.0), (0.0, 4.8))
RECOVERY_ROUTE_MARGIN_M = 0.25
RECOVERY_EDGE_MAX_ABS_ROLL_RAD = math.radians(35.0)
RECOVERY_EDGE_MAX_ABS_PITCH_RAD = math.radians(22.0)
RECOVERY_EDGE_MAX_BODY_RATE_RAD_S = 0.65
LAUNCH_SCORE_VERSION = "r10_r11_updraft_gain_multiplicative_launch_score_v4"
SPECIFIC_ENERGY_GRAVITY_M_S2 = 9.80665
SCORING_TARGET_EPISODE_TIME_S = 1.5
PHYSICAL_HARD_FAILURE_LABELS = {
    "floor_violation",
    "ceiling_violation",
    "z_boundary_exit",
    "initial_floor_violation",
    "initial_ceiling_violation",
    "nonfinite_initial_state",
    "nonfinite_trajectory",
    "corrupt_integration",
    "physically_impossible_initial_state",
    "true_safety_violation",
}
DEFAULT_VALIDATION_MAX_EPISODE_TIME_S = 20.0
R9_PREFLIGHT_MAX_EPISODE_TIME_S = 10.0
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "full_autonomy",
    "memory_improvement",
)
_WORKER_LIBRARIES: dict[str, dict[str, object]] | None = None
_WORKER_OUTCOME_ROWS_BY_CASE: dict[str, dict[str, dict[str, object]]] | None = None
_WORKER_RECORDS_BY_VARIANT: dict[str, object] | None = None
_WORKER_CONFIG: "ValidationRunConfig | None" = None
_WORKER_PROTOCOL: "ValidationProtocol | None" = None


@dataclass(frozen=True)
class ValidationBlockSpec:
    block_id: str
    human_label: str
    W_layer: str
    environment_mode: str
    case_count: int
    environment_change_family: str = "fixed_case"


R9_BLOCKS: tuple[ValidationBlockSpec, ...] = (
    ValidationBlockSpec("no_updraft", "no-updraft", "W0", "dry_air", R9_PREFLIGHT_CASES_PER_BLOCK),
    ValidationBlockSpec("single_fan", "single-fan", "W2", "annular_gp_single", R9_PREFLIGHT_CASES_PER_BLOCK),
    ValidationBlockSpec("four_fan", "four-fan", "W2", "annular_gp_four", R9_PREFLIGHT_CASES_PER_BLOCK),
)


@dataclass(frozen=True)
class RepeatedLaunchValidationConfig:
    library_root: Path = DEFAULT_LIBRARY_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1
    source_w2_root: Path | None = None
    seed: int = 90
    storage_format: str = "auto"
    compression_level: int = 1
    candidate_chunk_size: int = 20_000
    dry_run_schedule: bool = False
    max_primitives_per_launch: int = 0
    max_episode_time_s: float = R9_PREFLIGHT_MAX_EPISODE_TIME_S
    smoke_outer_cases_per_block: int = 0
    workers: int = 1
    max_workers: int | None = None
    worker_backend: str = "process"
    governor_config: GovernorConfig | None = None


@dataclass(frozen=True)
class ValidationProtocol:
    stage_id: str
    manifest_name: str
    report_name: str
    manifest_version: str
    validation_evidence_level: str
    outer_cases_per_condition: int
    expected_final_heldout_launches: int
    expected_history_launches: int
    blocks: tuple[ValidationBlockSpec, ...]
    final_schedule_prefix: str
    reduced_diagnostic: bool = False
    requires_no_glider_latency_variation_audit: bool = False
    gate_profile: str = "strict_final_validation"
    max_hard_failure_rate: float = 0.01
    max_no_viable_rate: float = 0.02
    min_safe_success_rate: float = 0.99
    min_full_safe_success_rate: float | None = None
    min_terminal_or_lift_capture_rate: float = 0.90


R9_PROTOCOL = ValidationProtocol(
    stage_id="R9",
    manifest_name="repeated_launch_fixed_case_manifest.json",
    report_name="repeated_launch_fixed_case_report.md",
    manifest_version=VALIDATION_VERSION,
    validation_evidence_level="internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence",
    outer_cases_per_condition=R9_OUTER_CASES_PER_CONDITION,
    expected_final_heldout_launches=R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    expected_history_launches=R9_EXPECTED_HISTORY_LAUNCHES,
    blocks=R9_BLOCKS,
    final_schedule_prefix="r9_fixed",
    gate_profile="internal_reduced_fixed_case_preflight_for_r10_initialisation",
    max_hard_failure_rate=0.20,
    max_no_viable_rate=0.30,
    min_safe_success_rate=0.20,
    min_terminal_or_lift_capture_rate=0.30,
)


@dataclass(frozen=True)
class ValidationRunConfig:
    library_root: Path
    outcome_root: Path
    output_root: Path
    run_id: int
    source_w2_root: Path | None
    seed: int
    storage_format: str
    compression_level: int
    candidate_chunk_size: int
    dry_run_schedule: bool
    max_primitives_per_launch: int
    max_episode_time_s: float
    smoke_outer_cases_per_block: int
    workers: int
    max_workers: int | None
    worker_backend: str
    governor_config: GovernorConfig | None = None


def run_repeated_launch_learning_curve(config: RepeatedLaunchValidationConfig) -> dict[str, object]:
    """Run the reduced internal R9 fixed-case repeated-launch preflight."""

    return run_repeated_launch_validation(
        ValidationRunConfig(
            library_root=config.library_root,
            outcome_root=config.outcome_root,
            output_root=config.output_root,
            run_id=config.run_id,
            source_w2_root=config.source_w2_root,
            seed=config.seed,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            candidate_chunk_size=config.candidate_chunk_size,
            dry_run_schedule=config.dry_run_schedule,
            max_primitives_per_launch=config.max_primitives_per_launch,
            max_episode_time_s=config.max_episode_time_s,
            smoke_outer_cases_per_block=config.smoke_outer_cases_per_block,
            workers=config.workers,
            max_workers=config.max_workers,
            worker_backend=config.worker_backend,
            governor_config=config.governor_config,
        ),
        protocol=R9_PROTOCOL,
    )


def run_repeated_launch_validation(config: ValidationRunConfig, *, protocol: ValidationProtocol) -> dict[str, object]:
    """Run or schedule repeated-launch validation with true primitive rollout rows."""

    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports", "tables"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    storage_format = resolve_storage_format(config.storage_format)
    blocked_reason = _blocked_reason(config)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, protocol, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}

    libraries = _load_libraries(config.library_root)
    outcome_rows = _read_outcome_rows(config.outcome_root)
    outer_cases = _outer_case_schedule(
        protocol=protocol,
        seed=config.seed,
        smoke_outer_cases_per_block=int(config.smoke_outer_cases_per_block),
    )
    final_schedule = _final_heldout_schedule(outer_cases=outer_cases, protocol=protocol)
    history_schedule = _history_launch_schedule(outer_cases=outer_cases, protocol=protocol)
    _write_schedule_metric(
        run_root=run_root,
        table_name="outer_case_schedule",
        rows=outer_cases,
        run_id=int(config.run_id),
        storage_format=storage_format,
        compression_level=int(config.compression_level),
    )
    _write_schedule_metric(
        run_root=run_root,
        table_name="history_launch_schedule",
        rows=history_schedule,
        run_id=int(config.run_id),
        storage_format=storage_format,
        compression_level=int(config.compression_level),
    )
    _write_schedule_metric(
        run_root=run_root,
        table_name="final_heldout_launch_schedule",
        rows=final_schedule,
        run_id=int(config.run_id),
        storage_format=storage_format,
        compression_level=int(config.compression_level),
    )
    if protocol.stage_id in CHANGED_CASE_VALIDATION_STAGE_IDS:
        _write_csv(run_root / "metrics" / "environment_block_schedule.csv", _environment_block_summary(protocol))
        _write_csv(
            run_root / "metrics" / "active_fan_count_schedule_audit.csv",
            pd.DataFrame(_active_fan_count_schedule_audit_rows(outer_cases)),
        )

    if config.dry_run_schedule:
        pass_summary = _pass_fail_summary(
            protocol=protocol,
            max_primitives_per_launch=int(config.max_primitives_per_launch),
            max_episode_time_s=float(config.max_episode_time_s),
            final_schedule=final_schedule,
            history_schedule=history_schedule,
            episode_rows=[],
            pairing_rows=_pairing_audit_rows(final_schedule),
            no_variation_rows=_no_variation_audit_rows(final_schedule) if protocol.requires_no_glider_latency_variation_audit else [],
        )
        _write_csv(run_root / "metrics" / "pass_fail_gate_summary.csv", pd.DataFrame(pass_summary))
        _write_manifest(
            run_root=run_root,
            config=config,
            protocol=protocol,
            status="dry_run_schedule",
            pass_summary=pass_summary,
            final_schedule=final_schedule,
            history_schedule=history_schedule,
        )
        _write_file_size_audit(run_root)
        _write_report(run_root=run_root, protocol=protocol, status="dry_run_schedule", pass_summary=pass_summary)
        return {
            "status": "dry_run_schedule",
            "run_root": run_root.as_posix(),
            "final_heldout_launch_count": len(final_schedule),
            "history_launch_count": len(history_schedule),
        }

    records_by_variant = _load_records_by_variant(config, libraries)
    if not records_by_variant:
        _write_blocked_outputs(run_root, config, protocol, "missing_frozen_controller_records_for_rollout")
        return {"status": "blocked", "blocked_reason": "missing_frozen_controller_records_for_rollout", "run_root": run_root.as_posix()}
    outcome_rows_by_case = _outcome_rows_by_case(outcome_rows)
    selected_workers = _selected_worker_count(config)

    table_buffers = {name: [] for name in TABLE_NAMES}
    partitions: list[TablePartition] = []
    row_counters = {name: 0 for name in TABLE_NAMES}
    started = time.time()
    for launch_results in _iter_launch_result_batches(
        final_schedule=final_schedule,
        libraries=libraries,
        outcome_rows_by_case=outcome_rows_by_case,
        records_by_variant=records_by_variant,
        config=config,
        protocol=protocol,
        selected_workers=selected_workers,
    ):
        for launch_result in launch_results:
            _append_launch_result(table_buffers, launch_result)
        partitions.extend(
            _flush_if_needed(
                run_root=run_root,
                table_buffers=table_buffers,
                row_counters=row_counters,
                storage_format=storage_format,
                compression_level=config.compression_level,
                chunk_size=max(1, int(config.candidate_chunk_size)),
            )
        )
    partitions.extend(
        _flush_all(
            run_root=run_root,
            table_buffers=table_buffers,
            row_counters=row_counters,
            storage_format=storage_format,
            compression_level=config.compression_level,
        )
    )
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(run_id=int(config.run_id), root=run_root.as_posix(), storage_format=storage_format, tables=tuple(partitions)),
    )

    _write_first_decision_audits_from_partitions(run_root, partitions, storage_format)
    episode_rows = _read_partitioned_rows(run_root, partitions, "episode_summary")
    pairing_rows = _pairing_audit_rows(final_schedule)
    no_variation_rows = _no_variation_audit_rows(final_schedule) if protocol.requires_no_glider_latency_variation_audit else []
    _write_csv(run_root / "metrics" / "pairing_audit.csv", pd.DataFrame(pairing_rows))
    if protocol.requires_no_glider_latency_variation_audit:
        _write_csv(run_root / "metrics" / "no_glider_latency_variation_audit.csv", pd.DataFrame(no_variation_rows))
    _write_compact_metric_tables(run_root, episode_rows, protocol)
    pass_summary = _pass_fail_summary(
        protocol=protocol,
        max_primitives_per_launch=int(config.max_primitives_per_launch),
        max_episode_time_s=float(config.max_episode_time_s),
        final_schedule=final_schedule,
        history_schedule=history_schedule,
        episode_rows=episode_rows,
        pairing_rows=pairing_rows,
        no_variation_rows=no_variation_rows,
    )
    _write_csv(run_root / "metrics" / "pass_fail_gate_summary.csv", pd.DataFrame(pass_summary))
    if protocol.stage_id == "R9":
        _write_governor_tuning_outputs(run_root, config, protocol, pass_summary, episode_rows)
    if protocol.stage_id == "R10":
        _write_governor_tuning_outputs(run_root, config, protocol, pass_summary, episode_rows)
    status = "smoke_run" if int(config.smoke_outer_cases_per_block) > 0 else "complete"
    _write_manifest(
        run_root=run_root,
        config=config,
        protocol=protocol,
        status=status,
        pass_summary=pass_summary,
        final_schedule=final_schedule,
        history_schedule=history_schedule,
        duration_s=time.time() - started,
    )
    _write_file_size_audit(run_root)
    _write_report(run_root=run_root, protocol=protocol, status=status, pass_summary=pass_summary)
    return {
        "status": status,
        "run_root": run_root.as_posix(),
        "final_heldout_launch_count": len(final_schedule),
        "history_launch_count": len(history_schedule),
        "pass_gate": _overall_pass(pass_summary),
    }


def _blocked_reason(config: ValidationRunConfig) -> str:
    study_manifest = filesystem_path(Path(config.library_root) / "manifests" / "post_w3_library_size_study_manifest.json")
    if not study_manifest.is_file():
        return "missing_post_w3_library_size_study_manifest"
    try:
        study_payload = json.loads(study_manifest.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_post_w3_library_size_study_manifest:{type(exc).__name__}"
    if str(study_payload.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "post_w3_library_size_study_not_v5_project_title"
    for case_id in LIBRARY_SIZE_CASE_IDS:
        library_path = filesystem_path(Path(config.library_root) / "manifests" / f"{case_id}_primitive_library.json")
        if not library_path.is_file():
            return f"missing_library_size_case_manifest:{case_id}"
    outcome_path = filesystem_path(Path(config.outcome_root) / "metrics" / "outcome_model_table.csv")
    if not outcome_path.is_file():
        return "missing_outcome_model_table"
    outcome_manifest = filesystem_path(Path(config.outcome_root) / "manifests" / "outcome_model_manifest.json")
    if not outcome_manifest.is_file():
        return "missing_outcome_model_manifest"
    try:
        outcome_payload = json.loads(outcome_manifest.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_outcome_model_manifest:{type(exc).__name__}"
    if str(outcome_payload.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "outcome_model_not_v5_project_title"
    outcome = pd.read_csv(outcome_path)
    if "sample_count" not in outcome.columns:
        return "outcome_model_missing_sample_count_coverage_column"
    sample_counts = pd.to_numeric(outcome["sample_count"], errors="coerce")
    if sample_counts.isna().any():
        return "outcome_model_sample_count_contains_non_numeric_values"
    if bool((sample_counts <= 0).any()):
        return "outcome_model_contains_non_positive_sample_count_rows"
    missing_cases = set(LIBRARY_SIZE_CASE_IDS) - set(outcome.get("library_size_case_id", pd.Series(dtype=str)).astype(str))
    if missing_cases:
        return "outcome_model_missing_library_size_cases:" + ",".join(sorted(missing_cases))
    return ""


def _load_libraries(library_root: Path) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for case_id in LIBRARY_SIZE_CASE_IDS:
        payload = _read_json(Path(library_root) / "manifests" / f"{case_id}_primitive_library.json")
        payloads[case_id] = payload
    return payloads


def _read_outcome_rows(outcome_root: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(filesystem_path(Path(outcome_root) / "metrics" / "outcome_model_table.csv"))
    rows: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        case_id = str(row.get("library_size_case_id", ""))
        compact_id = str(row.get("compact_library_id", ""))
        variant_id = str(row.get("primitive_variant_id", ""))
        transition_object_id = str(row.get("transition_object_id", ""))
        if compact_id:
            rows[compact_id] = row
            rows[f"{case_id}|{compact_id}"] = row
        if transition_object_id and compact_id:
            rows[f"{case_id}|{transition_object_id}|{compact_id}"] = row
        if variant_id and compact_id:
            rows[f"{case_id}|{variant_id}|{compact_id}"] = row
        elif variant_id and variant_id not in rows:
            rows[variant_id] = row
    return rows


def _outcome_rows_by_case(outcome_rows: dict[str, dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    rows_by_case: dict[str, dict[str, dict[str, object]]] = {case_id: {} for case_id in LIBRARY_SIZE_CASE_IDS}
    for key, row in outcome_rows.items():
        case_id = str(row.get("library_size_case_id", ""))
        if case_id in rows_by_case:
            rows_by_case[case_id][str(key)] = row
    return rows_by_case


def _load_records_by_variant(config: ValidationRunConfig, libraries: dict[str, dict[str, object]]) -> dict[str, object]:
    candidates = []
    if config.source_w2_root is not None:
        candidates.append(Path(config.source_w2_root))
    for payload in libraries.values():
        if payload.get("source_w2_root"):
            candidates.append(Path(str(payload["source_w2_root"])))
        if payload.get("source_w01_root"):
            candidates.append(Path(str(payload["source_w01_root"])))
        if payload.get("source_r5_root"):
            candidates.append(Path(str(payload["source_r5_root"])))
        for row in list(payload.get("representatives", [])):
            if row.get("source_w2_root"):
                candidates.append(Path(str(row["source_w2_root"])))
            if row.get("source_w01_root"):
                candidates.append(Path(str(row["source_w01_root"])))
            if row.get("source_r5_root"):
                candidates.append(Path(str(row["source_r5_root"])))
    for root in candidates:
        bundle_path = filesystem_path(root / "manifests" / "frozen_w01_controller_bundle.json")
        if not bundle_path.is_file():
            continue
        bundle = load_frozen_w01_controller_bundle(root / "manifests" / "frozen_w01_controller_bundle.json")
        return {
            record.primitive_variant_id: record
            for record in bundle.records
            if str(record.bundle_status) == FROZEN_CONTROLLER_READY
        }
    return {}


def _selected_worker_count(config: ValidationRunConfig) -> int:
    requested = max(1, int(config.workers or 1))
    if config.max_workers is None:
        return requested
    return max(1, min(requested, int(config.max_workers)))


def _iter_launch_result_batches(
    *,
    final_schedule: list[dict[str, object]],
    libraries: dict[str, dict[str, object]],
    outcome_rows_by_case: dict[str, dict[str, dict[str, object]]],
    records_by_variant: dict[str, object],
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
    selected_workers: int,
) -> Iterable[list[dict[str, object]]]:
    if int(selected_workers) <= 1:
        for final_row in final_schedule:
            yield _run_final_schedule_row(
                final_row,
                libraries=libraries,
                outcome_rows_by_case=outcome_rows_by_case,
                records_by_variant=records_by_variant,
                config=config,
                protocol=protocol,
            )
        return

    backend = str(config.worker_backend or "process").strip().lower()
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(
        max_workers=int(selected_workers),
        initializer=_initialise_validation_worker,
        initargs=(libraries, outcome_rows_by_case, records_by_variant, config, protocol),
    ) as executor:
        schedule_iter = iter(final_schedule)
        in_flight = set()
        max_in_flight = max(int(selected_workers), int(selected_workers) * 2)

        def submit_until_full() -> None:
            while len(in_flight) < max_in_flight:
                try:
                    row = next(schedule_iter)
                except StopIteration:
                    return
                in_flight.add(executor.submit(_run_final_schedule_row_worker, row))

        submit_until_full()
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            submit_until_full()


def _initialise_validation_worker(
    libraries: dict[str, dict[str, object]],
    outcome_rows_by_case: dict[str, dict[str, dict[str, object]]],
    records_by_variant: dict[str, object],
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
) -> None:
    global _WORKER_LIBRARIES
    global _WORKER_OUTCOME_ROWS_BY_CASE
    global _WORKER_RECORDS_BY_VARIANT
    global _WORKER_CONFIG
    global _WORKER_PROTOCOL
    _WORKER_LIBRARIES = libraries
    _WORKER_OUTCOME_ROWS_BY_CASE = outcome_rows_by_case
    _WORKER_RECORDS_BY_VARIANT = records_by_variant
    _WORKER_CONFIG = config
    _WORKER_PROTOCOL = protocol


def _run_final_schedule_row_worker(final_row: dict[str, object]) -> list[dict[str, object]]:
    if (
        _WORKER_LIBRARIES is None
        or _WORKER_OUTCOME_ROWS_BY_CASE is None
        or _WORKER_RECORDS_BY_VARIANT is None
        or _WORKER_CONFIG is None
        or _WORKER_PROTOCOL is None
    ):
        raise RuntimeError("validation_worker_not_initialised")
    return _run_final_schedule_row(
        final_row,
        libraries=_WORKER_LIBRARIES,
        outcome_rows_by_case=_WORKER_OUTCOME_ROWS_BY_CASE,
        records_by_variant=_WORKER_RECORDS_BY_VARIANT,
        config=_WORKER_CONFIG,
        protocol=_WORKER_PROTOCOL,
    )


def _run_final_schedule_row(
    final_row: dict[str, object],
    *,
    libraries: dict[str, dict[str, object]],
    outcome_rows_by_case: dict[str, dict[str, dict[str, object]]],
    records_by_variant: dict[str, object],
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
) -> list[dict[str, object]]:
    policy = _policy_condition(str(final_row["policy_id"]))
    representatives = libraries[str(final_row["library_size_case_id"])]["representatives"]
    case_outcomes = outcome_rows_by_case.get(str(final_row["library_size_case_id"]), {})
    belief = _initial_belief_for_policy(policy=policy, final_row=final_row)
    launch_results: list[dict[str, object]] = []
    for hist_index in range(int(policy["history_length"])):
        history_row = _history_row_for_final(final_row, hist_index)
        history_result = _run_one_launch(
            scheduled=history_row,
            policy=policy,
            representatives=representatives,
            outcome_rows_by_variant_id=case_outcomes,
            records_by_variant=records_by_variant,
            belief=belief,
            config=config,
            protocol=protocol,
        )
        belief = history_result["belief_after"]
        launch_results.append(_launch_result_for_parent(history_result))
    final_result = _run_one_launch(
        scheduled=final_row,
        policy=policy,
        representatives=representatives,
        outcome_rows_by_variant_id=case_outcomes,
        records_by_variant=records_by_variant,
        belief=belief,
        config=config,
        protocol=protocol,
    )
    launch_results.append(_launch_result_for_parent(final_result))
    return launch_results


def _launch_result_for_parent(result: dict[str, object]) -> dict[str, object]:
    return {
        "episode_rows": result["episode_rows"],
        "primitive_rows": result["primitive_rows"],
        "candidate_rows": result["candidate_rows"],
        "selector_rows": result["selector_rows"],
        "memory_rows": result["memory_rows"],
        "belief_rows": result["belief_rows"],
    }


def _scheduled_active_fan_count_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
    environment_block_local_index: int,
) -> int | None:
    if str(protocol.stage_id) not in CHANGED_CASE_VALIDATION_STAGE_IDS:
        return None
    block_id = str(environment_block_id)
    if block_id in R10_ACTIVE_FAN_COUNT_VARIATION_BLOCK_IDS:
        return int(
            R10_ACTIVE_FAN_COUNT_SEQUENCE[
                int(environment_block_local_index) % len(R10_ACTIVE_FAN_COUNT_SEQUENCE)
            ]
        )
    if block_id in R10_FIXED_FOUR_ACTIVE_BLOCK_IDS:
        return 4
    return None


def _active_fan_count_policy_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
) -> str:
    if str(protocol.stage_id) not in CHANGED_CASE_VALIDATION_STAGE_IDS:
        return "environment_default"
    block_id = str(environment_block_id)
    if block_id == ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID:
        return "balanced_1_2_3_4_for_active_fan_number_variation"
    if block_id == BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID:
        return "balanced_1_2_3_4_with_arena_wide_fan_position_generalisation"
    if block_id in R10_FIXED_FOUR_ACTIVE_BLOCK_IDS:
        return "fixed_4_for_four_fan_geometry_non_active_count_block"
    if "single_fan" in block_id:
        return "single_fan_geometry_implicit_one_active_fan"
    return "environment_default"


def _fan_layout_policy_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
) -> str:
    if str(protocol.stage_id) not in CHANGED_CASE_VALIDATION_STAGE_IDS:
        return "fixed_case_layout"
    block_id = str(environment_block_id)
    if "single_fan" in block_id:
        return "single_fan_geometry"
    if "four_fan" in block_id or block_id in {
        ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
        BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    }:
        return "four_fan_geometry"
    return "unknown_layout"


def _scheduled_fan_layout_count_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
) -> int | str:
    layout = _fan_layout_policy_for_outer_case(
        protocol=protocol,
        environment_block_id=environment_block_id,
    )
    if layout == "single_fan_geometry":
        return 1
    if layout == "four_fan_geometry":
        return 4
    return ""


def _fan_position_policy_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
) -> str:
    if str(protocol.stage_id) not in CHANGED_CASE_VALIDATION_STAGE_IDS:
        return "common_shift"
    block_id = str(environment_block_id)
    if block_id == BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID:
        return "independent_uniform_xy_bounds"
    if block_id in R10_FIXED_BASE_POSITION_BLOCK_IDS:
        return "fixed_base_positions"
    if block_id in R10_SHIFTED_FAN_POSITION_BLOCK_IDS:
        return "common_shift"
    return "common_shift"


def _fan_position_bounds_text_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
) -> str:
    if str(protocol.stage_id) not in CHANGED_CASE_VALIDATION_STAGE_IDS:
        return "common_shift_range=-0.200:0.200"
    block_id = str(environment_block_id)
    if block_id == BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID:
        x_bounds, y_bounds = R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M
        return f"x={float(x_bounds[0]):.3f}:{float(x_bounds[1]):.3f};y={float(y_bounds[0]):.3f}:{float(y_bounds[1]):.3f}"
    if block_id in R10_FIXED_BASE_POSITION_BLOCK_IDS:
        return "fixed_base_positions_no_shift"
    return "common_shift_range=-0.200:0.200"


def _outer_case_schedule(
    *,
    protocol: ValidationProtocol,
    seed: int,
    smoke_outer_cases_per_block: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outer_index = 0
    for block in protocol.blocks:
        case_count = int(block.case_count)
        if int(smoke_outer_cases_per_block) > 0:
            case_count = min(case_count, int(smoke_outer_cases_per_block))
        for local_index in range(case_count):
            launch_seed = int(seed) * 100000 + outer_index * 37 + 11
            env_seed = int(seed) * 200000 + outer_index * 41 + 17
            scheduled_active_fan_count = _scheduled_active_fan_count_for_outer_case(
                protocol=protocol,
                environment_block_id=block.block_id,
                environment_block_local_index=local_index,
            )
            rows.append(
                {
                    "outer_case_index": outer_index,
                    "outer_case_id": f"{protocol.final_schedule_prefix}_outer_{outer_index:04d}",
                    "outer_case_type": block.block_id,
                    "environment_block_id": block.block_id,
                    "environment_block_local_index": int(local_index),
                    "environment_block_label": block.human_label,
                    "environment_change_family": block.environment_change_family,
                    "W_layer": block.W_layer,
                    "environment_mode": block.environment_mode,
                    "fan_layout_policy": _fan_layout_policy_for_outer_case(
                        protocol=protocol,
                        environment_block_id=block.block_id,
                    ),
                    "scheduled_fan_layout_count": _scheduled_fan_layout_count_for_outer_case(
                        protocol=protocol,
                        environment_block_id=block.block_id,
                    ),
                    "scheduled_active_fan_count": (
                        "" if scheduled_active_fan_count is None else int(scheduled_active_fan_count)
                    ),
                    "active_fan_count_policy": _active_fan_count_policy_for_outer_case(
                        protocol=protocol,
                        environment_block_id=block.block_id,
                    ),
                    "fan_position_policy": _fan_position_policy_for_outer_case(
                        protocol=protocol,
                        environment_block_id=block.block_id,
                    ),
                    "fan_position_xy_bounds_m": _fan_position_bounds_text_for_outer_case(
                        protocol=protocol,
                        environment_block_id=block.block_id,
                    ),
                    "launch_state_seed": launch_seed,
                    "environment_seed": env_seed,
                    "common_final_launch_key": f"{protocol.final_schedule_prefix}_final_{outer_index:04d}",
                    "start_state_family": FIRST_PRIMITIVE_START_FAMILY,
                    "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
                    "claim_status": "simulation_only_controlled_outer_case",
                }
            )
            outer_index += 1
    return rows


def _final_heldout_schedule(*, outer_cases: list[dict[str, object]], protocol: ValidationProtocol) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    episode_index = 0
    for case_id in LIBRARY_SIZE_CASE_IDS:
        for policy_id in POLICY_HISTORY_CONDITIONS:
            for outer in outer_cases:
                rows.append(
                    {
                        **outer,
                        "episode_id": f"{protocol.stage_id.lower()}_{case_id}_{policy_id}_final_{int(outer['outer_case_index']):04d}",
                        "episode_index": episode_index,
                        "launch_role": "final_heldout",
                        "library_size_case_id": case_id,
                        "policy_id": policy_id,
                        "history_length": int(_policy_condition(policy_id)["history_length"]),
                    }
                )
                episode_index += 1
    return rows


def _history_launch_schedule(*, outer_cases: list[dict[str, object]], protocol: ValidationProtocol) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    history_policies = [policy for policy in POLICY_HISTORY_CONDITIONS if policy.startswith(MEMORY_POLICY_PREFIX) or policy.startswith(SAFE_EXPLORE_POLICY_PREFIX)]
    for case_id in LIBRARY_SIZE_CASE_IDS:
        for policy_id in history_policies:
            history_length = int(_policy_condition(policy_id)["history_length"])
            for outer in outer_cases:
                for history_index in range(history_length):
                    rows.append(
                        {
                            **outer,
                            "episode_id": (
                                f"{protocol.stage_id.lower()}_{case_id}_{policy_id}_"
                                f"hist_{int(outer['outer_case_index']):04d}_{history_index:03d}"
                            ),
                            "launch_role": "history",
                            "history_launch_index": history_index,
                            "library_size_case_id": case_id,
                            "policy_id": policy_id,
                            "history_length": history_length,
                        }
                    )
    return rows


def _history_row_for_final(final_row: dict[str, object], history_index: int) -> dict[str, object]:
    seed_shift = 1000000 + int(history_index) * 101
    return {
        **final_row,
        "episode_id": f"{final_row['episode_id']}_hist_{int(history_index):03d}",
        "launch_role": "history",
        "history_launch_index": int(history_index),
        "launch_state_seed": int(final_row["launch_state_seed"]) + seed_shift,
        "environment_seed": int(final_row["environment_seed"]) + seed_shift,
        "common_final_launch_key": str(final_row["common_final_launch_key"]),
    }


def _scheduled_active_fan_count_for_context(
    *,
    protocol: ValidationProtocol,
    scheduled: dict[str, object],
) -> int | None:
    scheduled_count = scheduled.get("scheduled_active_fan_count", "")
    if str(scheduled_count).strip() not in {"", "nan", "None"}:
        return int(scheduled_count)
    return _scheduled_active_fan_count_for_outer_case(
        protocol=protocol,
        environment_block_id=str(scheduled.get("environment_block_id", "")),
        environment_block_local_index=int(
            scheduled.get("environment_block_local_index", scheduled.get("outer_case_index", 0))
        ),
    )


def _policy_condition(policy_id: str) -> dict[str, object]:
    if policy_id == "no_memory_baseline":
        return {"policy_id": policy_id, "policy_family": "baseline", "history_length": 0, "uses_memory": False, "updates_memory": False, "safe_explore": False}
    if policy_id == EMPTY_FROZEN_PRIOR_BASELINE_ID:
        return {"policy_id": policy_id, "policy_family": "baseline", "history_length": 0, "uses_memory": True, "updates_memory": False, "safe_explore": False}
    for prefix in (MEMORY_POLICY_PREFIX, SAFE_EXPLORE_POLICY_PREFIX):
        marker = f"{prefix}_h"
        if policy_id.startswith(marker):
            history_length = int(policy_id[len(marker) :])
            return {
                "policy_id": policy_id,
                "policy_family": prefix,
                "history_length": history_length,
                "uses_memory": True,
                "updates_memory": True,
                "safe_explore": prefix == SAFE_EXPLORE_POLICY_PREFIX,
            }
    raise KeyError(f"unknown policy_id: {policy_id}")


def _initial_belief_for_policy(*, policy: dict[str, object], final_row: dict[str, object]) -> DirectionalResidualLiftBelief:
    belief = initial_directional_residual_lift_belief()
    del final_row
    return belief


def _run_one_launch(
    *,
    scheduled: dict[str, object],
    policy: dict[str, object],
    representatives: list[dict[str, object]],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
    records_by_variant: dict[str, object],
    belief: DirectionalResidualLiftBelief,
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
) -> dict[str, object]:
    episode_id = str(scheduled["episode_id"])
    sample = archive_state_sample_for_family(
        start_state_family=FIRST_PRIMITIVE_START_FAMILY,
        paired_start_key=str(scheduled["common_final_launch_key"]),
        sample_index=int(scheduled["outer_case_index"]),
        seed=int(scheduled["launch_state_seed"]),
        W_layer=str(scheduled["W_layer"]),
        environment_mode=str(scheduled["environment_mode"]),
    )
    state = as_state_vector(sample.state_vector)
    governor_config = _governor_config_for_policy(policy, base_config=config.governor_config or DEFAULT_GOVERNOR_CONFIG)
    time_budget_steps = max(
        1,
        int(math.ceil(float(config.max_episode_time_s) / float(PRIMITIVE_FINITE_HORIZON_S))),
    )
    if int(config.max_primitives_per_launch) > 0:
        max_steps = min(time_budget_steps, int(config.max_primitives_per_launch))
    else:
        max_steps = time_budget_steps
    primitive_rows: list[dict[str, object]] = []
    candidate_rows_all: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    belief_rows: list[dict[str, object]] = []
    belief_after = belief
    context_row: dict[str, object] = {}
    blocked_reason = ""
    for primitive_step_index in range(max_steps):
        route = validation_route_for_primitive_step(primitive_step_index, state=state)
        start_state_family = str(route["start_state_family"])
        governor_mode = _governor_mode_for_route(route)
        context_payload = _context_payload(
            state=state,
            scheduled=scheduled,
            episode_id=episode_id,
            protocol=protocol,
            start_state_family=start_state_family,
            primitive_step_index=primitive_step_index,
            route=route,
        )
        context_row = context_payload["row"]
        belief_features = None
        if bool(policy["uses_memory"]):
            belief_features = query_directional_residual_lift_features(
                belief_after,
                x_w_m=float(state[STATE_INDEX["x_w"]]),
                y_w_m=float(state[STATE_INDEX["y_w"]]),
                z_w_m=float(state[STATE_INDEX["z_w"]]),
                direction_rad=float(state[STATE_INDEX["psi"]]),
            )
        selected, candidate_rows = select_compact_representative(
            representatives=representatives,
            outcome_rows_by_variant_id=outcome_rows_by_variant_id,
            context=context_payload["row"],
            governor_mode=governor_mode,
            policy_id=str(policy["policy_id"]),
            belief_features=belief_features,
            governor_config=governor_config,
        )
        for row in candidate_rows:
            row.update(_schedule_identity_row(scheduled))
            row["launch_role"] = str(scheduled["launch_role"])
            row["primitive_step_index"] = int(primitive_step_index)
            row["launch_sequence_policy"] = LAUNCH_SEQUENCE_POLICY_ID
            row["launch_sequence_phase"] = str(route["launch_sequence_phase"])
            row["route_required_entry_role"] = str(route["route_required_entry_role"])
            row["route_required_entry_class"] = str(route.get("route_required_entry_class", ""))
            row["route_reason"] = str(route["route_reason"])
        candidate_rows_all.extend(
            _compact_candidate_score_rows(
                candidate_rows,
                selected=selected,
                scheduled=scheduled,
                primitive_step_index=primitive_step_index,
                top_k=CANDIDATE_SCORE_TOP_K_PER_DECISION,
            )
        )
        selector_row = {
            **selector_decision_row(
                episode_id=episode_id,
                primitive_step_index=primitive_step_index,
                policy_id=str(policy["policy_id"]),
                governor_mode=governor_mode,
                context=context_payload["row"],
                selected=selected,
                candidate_count=len(candidate_rows),
                viable_count=sum(1 for row in candidate_rows if bool(row.get("viable", False))),
                governor_config=governor_config,
            ),
            **_schedule_identity_row(scheduled),
            "launch_role": str(scheduled["launch_role"]),
            "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
            "launch_sequence_phase": str(route["launch_sequence_phase"]),
            "route_required_entry_role": str(route["route_required_entry_role"]),
            "route_required_entry_class": str(route.get("route_required_entry_class", "")),
            "route_reason": str(route["route_reason"]),
        }
        selector_rows.append(selector_row)
        belief_rows.append(
            _belief_snapshot_compact(
                belief=belief_after,
                scheduled=scheduled,
                phase=f"before_primitive_step_{primitive_step_index}",
                features=belief_features or {},
            )
        )
        if selected is None:
            blocked_reason = "no_viable_primitive" if primitive_step_index == 0 else "no_viable_continuation_primitive"
            break
        record = records_by_variant.get(str(selected["primitive_variant_id"]))
        if record is None:
            blocked_reason = "missing_frozen_controller_record"
            break
        primitive = primitive_by_id(str(selected["primitive_id"]))
        rollout = simulate_primitive_rollout(
            rollout_id=f"{episode_id}_p{primitive_step_index:02d}",
            episode_id=episode_id,
            initial_state=state,
            context=context_payload["context"],
            primitive=primitive,
            config=RolloutConfig(W_layer=str(scheduled["W_layer"]), rollout_backend="model_backed_lqr"),
            wind_field=context_payload["wind_field"],
            implementation_instance=context_payload["implementation_instance"],
            plant_instance=context_payload["plant_instance"],
            controller=record.controller,
            controller_selection_status=f"selected_by_{protocol.stage_id.lower()}_repeated_launch_validator",
            candidate_index=record.candidate_index,
            candidate_weight_label=record.candidate_weight_label,
        )
        rollout_row = rollout_evidence_row(rollout)
        rollout_row.update(
            transition_row_fields(
                rollout_row,
                entry_role=str(selected.get("entry_role", "")),
                start_state_family=start_state_family,
                primitive_step_index=primitive_step_index,
            )
        )
        primitive_rows.append(
            {
                **_schedule_identity_row(scheduled),
                "launch_role": str(scheduled["launch_role"]),
                "primitive_step_index": int(primitive_step_index),
                "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
                "launch_sequence_phase": str(route["launch_sequence_phase"]),
                "start_state_family": start_state_family,
                "route_required_entry_role": str(route["route_required_entry_role"]),
                "route_required_entry_class": str(route.get("route_required_entry_class", "")),
                "route_reason": str(route["route_reason"]),
                "transition_current_state_class": str(route.get("current_state_class", "")),
                "selected_transition_entry_class": str(selected.get("transition_entry_class", "")),
                "transition_exit_class": str(rollout_row.get("transition_exit_class", "")),
                "transition_chain_compatible": bool(rollout_row.get("transition_chain_compatible", False)),
                "transition_failure_reason": str(rollout_row.get("transition_failure_reason", "")),
                "selected_entry_role": str(selected.get("entry_role", "")),
                "policy_id": str(policy["policy_id"]),
                "selected_compact_library_id": str(selected.get("compact_library_id", "")),
                "primitive_variant_id": str(selected.get("primitive_variant_id", "")),
                "selected_primitive_variant_id": str(selected.get("primitive_variant_id", "")),
                "selected_score": float(selected.get("total_score_with_memory_and_exploration", selected.get("score", 0.0))),
                "context_w_wing_mean_m_s": float(context_payload["row"].get("w_wing_mean_m_s", 0.0)),
                "context_lift_score": float(context_payload["row"].get("lift_score", 0.0)),
                "trajectory_plot_scope": "plot_ready_all_final_and_history_selected_primitives",
                "updraft_specific_energy_gain_proxy_m": _primitive_updraft_gain_proxy_m(
                    context_payload["row"],
                    rollout_row=rollout_row,
                ),
                **rollout_row,
            }
        )
        outcome = _outcome_for_selected(
            selected,
            outcome_rows_by_variant_id,
            context=context_payload["row"],
            governor_mode=governor_mode,
        )
        observation = DirectionalResidualObservation(
            x_w_m=float(state[STATE_INDEX["x_w"]]),
            y_w_m=float(state[STATE_INDEX["y_w"]]),
            z_w_m=float(state[STATE_INDEX["z_w"]]),
            direction_rad=float(state[STATE_INDEX["psi"]]),
            lift_residual_m_s=_lift_residual_for_memory_update(
                context_payload["row"],
                rollout_row=rollout_row,
                outcome=outcome,
            ),
            updraft_gain_residual_m=_updraft_gain_residual_for_memory_update(
                context_payload["row"],
                rollout_row=rollout_row,
                outcome=outcome,
            ),
            dwell_residual_s=float(rollout_row.get("lift_dwell_time_s", 0.0)) - float(outcome.get("expected_lift_dwell_time_s", 0.0)),
        )
        belief_before_update = belief_after
        if bool(policy["updates_memory"]):
            belief_after = update_directional_residual_lift_belief(belief_after, observation)
            update_status = "updated"
        else:
            update_status = "not_updated_policy"
        memory_rows.append(
            {
                **_schedule_identity_row(scheduled),
                "launch_role": str(scheduled["launch_role"]),
                "policy_id": str(policy["policy_id"]),
                "primitive_step_index": int(primitive_step_index),
                "update_status": update_status,
                **asdict(observation),
                "belief_update_count_before": int(belief_before_update.update_count),
                "belief_update_count_after": int(belief_after.update_count),
            }
        )
        try:
            state = as_state_vector(np.asarray(json.loads(str(rollout_row.get("exit_state_vector", "[]"))), dtype=float))
        except Exception:
            blocked_reason = "invalid_exit_state_vector"
            break
        belief_rows.append(
            _belief_snapshot_compact(
                belief=belief_after,
                scheduled=scheduled,
                phase=f"after_primitive_step_{primitive_step_index}",
                features=query_directional_residual_lift_features(
                    belief_after,
                    x_w_m=float(state[STATE_INDEX["x_w"]]),
                    y_w_m=float(state[STATE_INDEX["y_w"]]),
                    z_w_m=float(state[STATE_INDEX["z_w"]]),
                    direction_rad=float(state[STATE_INDEX["psi"]]),
                ),
            )
        )
        exit_class = str(rollout_row.get("transition_exit_class", "hard_failure"))
        hard_failure = exit_class == "hard_failure" or _rollout_row_is_hard_failure(rollout_row)
        if hard_failure or exit_class == "safe_terminal":
            break
    else:
        if primitive_rows:
            blocked_reason = "episode_time_budget_reached"
    if primitive_rows:
        episode_row = _episode_row_from_sequence(
            scheduled=scheduled,
            policy=policy,
            primitive_rows=primitive_rows,
            selector_rows=selector_rows,
            context_row=context_row,
            belief_before=belief,
            belief_after=belief_after,
            blocked_reason=blocked_reason,
        )
    else:
        episode_row = _episode_row_from_blocked(
            scheduled,
            policy,
            context_row,
            reason=blocked_reason or "no_viable_primitive",
        )
    return {
        "episode_rows": [episode_row],
        "primitive_rows": primitive_rows,
        "candidate_rows": candidate_rows_all,
        "selector_rows": selector_rows,
        "memory_rows": memory_rows,
        "belief_rows": belief_rows,
        "belief_after": belief_after,
    }


def _outcome_for_selected(
    selected: dict[str, object],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
    *,
    context: dict[str, object] | None = None,
    governor_mode: str = "",
) -> dict[str, object]:
    base = lookup_outcome_for_identity(
        identity=selected,
        outcome_rows_by_variant_id=outcome_rows_by_variant_id,
    )
    if context is None:
        return base
    return context_conditioned_outcome(
        representative=selected,
        base_outcome=base,
        context=context,
        governor_mode=governor_mode,
    )


def _compact_candidate_score_rows(
    candidate_rows: list[dict[str, object]],
    *,
    selected: dict[str, object] | None,
    scheduled: dict[str, object],
    primitive_step_index: int,
    top_k: int,
) -> list[dict[str, object]]:
    """Keep thesis-grade candidate evidence without storing every rejected row."""

    if not candidate_rows:
        return []
    decision_candidate_count = int(len(candidate_rows))
    decision_viable_count = int(sum(1 for row in candidate_rows if bool(row.get("viable", False))))
    keep: dict[str, dict[str, object]] = {}

    def key(row: dict[str, object]) -> str:
        return "|".join(
            [
                str(row.get("compact_library_id", "")),
                str(row.get("primitive_variant_id", "")),
                str(row.get("primitive_id", "")),
                str(row.get("entry_role", "")),
                str(row.get("transition_entry_class", "")),
            ]
        )

    def score(row: dict[str, object]) -> float:
        return _float_value(
            row.get(
                "total_score_with_memory_and_exploration",
                row.get("score_with_memory", row.get("score", float("-inf"))),
            ),
            default=float("-inf"),
        )

    def add(row: dict[str, object], reason: str) -> None:
        copied = dict(row)
        copied["candidate_log_policy"] = "thesis_compact_topk_selected_family_rejection_summary"
        copied["candidate_log_retention_reason"] = reason
        copied["decision_candidate_count"] = decision_candidate_count
        copied["decision_viable_count"] = decision_viable_count
        copied["candidate_score_log_full_rows_retained"] = False
        keep.setdefault(key(copied), copied)

    selected_variant = "" if selected is None else str(selected.get("primitive_variant_id", ""))
    if selected_variant:
        for row in candidate_rows:
            if str(row.get("primitive_variant_id", "")) == selected_variant:
                add(row, "selected_candidate")
                break

    viable = [row for row in candidate_rows if bool(row.get("viable", False))]
    for row in sorted(viable, key=lambda item: (-score(item), str(item.get("primitive_id", ""))))[: max(1, int(top_k))]:
        add(row, f"top_{int(top_k)}_viable_candidate")

    required_entry_class = str(candidate_rows[0].get("route_required_entry_class", ""))
    for primitive_id, rows in _group_rows(
        [
            row
            for row in candidate_rows
            if str(row.get("transition_entry_class", "")) == required_entry_class
        ],
        "primitive_id",
    ).items():
        del primitive_id
        add(sorted(rows, key=lambda item: (-score(item), str(item.get("primitive_variant_id", ""))))[0], "required_transition_entry_family_availability")

    for reason, rows in _group_rows(candidate_rows, "rejection_reason").items():
        if str(reason).strip() and str(reason).lower() not in {"nan", "none"}:
            add(sorted(rows, key=lambda item: (-score(item), str(item.get("primitive_variant_id", ""))))[0], "rejection_reason_representative")

    for role, rows in _group_rows(candidate_rows, "entry_role").items():
        if str(role).strip():
            add(sorted(rows, key=lambda item: (-score(item), str(item.get("primitive_variant_id", ""))))[0], "entry_role_representative")

    for entry_class, rows in _group_rows(candidate_rows, "transition_entry_class").items():
        if str(entry_class).strip():
            add(sorted(rows, key=lambda item: (-score(item), str(item.get("primitive_variant_id", ""))))[0], "transition_entry_class_representative")

    return list(keep.values())


def _group_rows(rows: list[dict[str, object]], column: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(column, "")), []).append(row)
    return grouped


def validation_route_for_primitive_step(primitive_step_index: int, *, state: np.ndarray | None = None) -> dict[str, object]:
    """Return the governor-facing route without using rollout-budget knowledge."""

    if int(primitive_step_index) == 0:
        current_state_class = "launch_gate"
        start_family = FIRST_PRIMITIVE_START_FAMILY
        reason = "first_0p10s_launch_window"
    else:
        current_state_class, start_family, reason = _continuation_state_class_start_family_and_reason(
            state,
            primitive_step_index=int(primitive_step_index),
        )
    required_role = required_entry_role_for_state_class(current_state_class) or "transition_object"
    required_entry_classes = entry_classes_for_state_class(current_state_class)
    required_entry_class = required_entry_classes[0] if required_entry_classes else ""
    return {
        "current_state_class": current_state_class,
        "start_state_family": start_family,
        "launch_sequence_phase": _launch_sequence_phase_for_start_family(
            primitive_step_index,
            start_state_family=start_family,
        ),
        "route_required_entry_role": required_role,
        "route_required_entry_class": required_entry_class,
        "route_reason": reason,
    }


def validation_start_family_for_primitive_step(primitive_step_index: int, *, state: np.ndarray | None = None) -> str:
    """Return the governor-facing start family for the launch-aware sequence."""

    return str(validation_route_for_primitive_step(primitive_step_index, state=state)["start_state_family"])


def _continuation_start_family(state: np.ndarray | None) -> str:
    return _continuation_start_family_and_reason(state)[0]


def _continuation_start_family_and_reason(state: np.ndarray | None) -> tuple[str, str]:
    state_class, start_family, reason = _continuation_state_class_start_family_and_reason(state, primitive_step_index=1)
    del state_class
    return start_family, reason


def _continuation_state_class_start_family_and_reason(
    state: np.ndarray | None,
    *,
    primitive_step_index: int,
) -> tuple[str, str, str]:
    if state is None:
        return "post_launch_degraded", POST_LAUNCH_START_FAMILY, "state_unavailable_default_post_launch_handoff"
    try:
        x = as_state_vector(state)
    except Exception:
        return "hard_failure", TERMINAL_SAFE_EXIT_START_FAMILY, "invalid_state_hard_failure_route"
    state_class = classify_state(
        x,
        primitive_step_index=int(primitive_step_index),
        allow_post_launch_degraded=int(primitive_step_index) == 1,
    )
    start_family = start_family_for_state_class(state_class)
    if state_class == "post_launch_degraded":
        return state_class, start_family, "post_launch_degraded_handoff_to_inflight"
    if state_class == "inflight_stable":
        return state_class, start_family, "inflight_stable_continuation"
    if state_class == "boundary_near":
        return state_class, start_family, "boundary_near_route_not_failure"
    if state_class == "recoverable_degraded":
        return state_class, start_family, "recoverable_degraded_route"
    if state_class == "safe_terminal":
        return state_class, start_family, "safe_terminal_no_further_primitive_expected"
    return state_class, start_family, "hard_failure_no_further_primitive_expected"


def _launch_sequence_phase_for_start_family(primitive_step_index: int, *, start_state_family: str) -> str:
    if int(primitive_step_index) == 0:
        return "first_0p10s_launch_entry"
    family = str(start_state_family)
    if family == BOUNDARY_RECOVERY_START_FAMILY:
        return "state_routed_boundary_recovery"
    if family == TERMINAL_SAFE_EXIT_START_FAMILY:
        return "state_routed_recovery_safe_exit"
    return "post_launch_inflight"


def _launch_sequence_phase_for_step(primitive_step_index: int) -> str:
    return str(validation_route_for_primitive_step(primitive_step_index)["launch_sequence_phase"])


def _required_entry_role_for_start_family(start_state_family: str) -> str:
    del start_state_family
    return "transition_object"


def _governor_mode_for_route(route: dict[str, object]) -> str:
    if str(route.get("route_required_entry_class", "")) in {"boundary_near", "recoverable_degraded"}:
        return "terminal_episode_mode"
    return "continuation_mode"


def _context_payload(
    *,
    state: np.ndarray,
    scheduled: dict[str, object],
    episode_id: str,
    protocol: ValidationProtocol,
    start_state_family: str,
    primitive_step_index: int,
    route: dict[str, object] | None = None,
) -> dict[str, object]:
    env_layer = str(scheduled["W_layer"])
    mode = str(scheduled["environment_mode"])
    seed = int(scheduled["environment_seed"])
    scheduled_active_fan_count = _scheduled_active_fan_count_for_context(
        protocol=protocol,
        scheduled=scheduled,
    )
    randomisation_config = _environment_randomisation_config_for_context(
        protocol=protocol,
        scheduled=scheduled,
        scheduled_active_fan_count=scheduled_active_fan_count,
    )
    instance = environment_instance_for_mode(
        env_layer,
        mode,
        seed,
        randomisation_config=randomisation_config,
    )
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(env_layer, metadata, randomisation_seed=seed)
    wind_field = wind_field_for_binding(binding)
    latency_case = "nominal"
    context = build_environment_context(
        state,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=latency_case,
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    plant_layer = "W2" if protocol.requires_no_glider_latency_variation_audit else env_layer
    implementation_layer = "W2" if protocol.requires_no_glider_latency_variation_audit else env_layer
    route = route or validation_route_for_primitive_step(primitive_step_index, state=state)
    row = {
        "context_id": f"{episode_id}_ctx{int(primitive_step_index):02d}",
        "W_layer": env_layer,
        "environment_mode": mode,
        "environment_instance_id": instance.environment_id,
        "environment_block_id": str(scheduled.get("environment_block_id", "")),
        "outer_case_type": str(scheduled.get("outer_case_type", "")),
        "fan_layout_policy": str(scheduled.get("fan_layout_policy", "")),
        "scheduled_fan_layout_count": str(scheduled.get("scheduled_fan_layout_count", "")),
        "scheduled_active_fan_count": (
            "" if scheduled_active_fan_count is None else int(scheduled_active_fan_count)
        ),
        "actual_active_fan_count": int(sum(bool(value) for value in instance.active_fan_mask)),
        "active_fan_count_policy": _active_fan_count_policy_for_outer_case(
            protocol=protocol,
            environment_block_id=str(scheduled.get("environment_block_id", "")),
        ),
        "fan_position_policy": str(scheduled.get("fan_position_policy", "")),
        "fan_position_xy_bounds_m": str(scheduled.get("fan_position_xy_bounds_m", "")),
        "start_state_family": str(start_state_family),
        "primitive_step_index": int(primitive_step_index),
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
        "launch_sequence_phase": str(route.get("launch_sequence_phase", _launch_sequence_phase_for_step(primitive_step_index))),
        "route_required_entry_role": str(route.get("route_required_entry_role", _required_entry_role_for_start_family(start_state_family))),
        "route_required_entry_class": str(route.get("route_required_entry_class", "")),
        "route_reason": str(route.get("route_reason", "")),
        "current_state_class": str(route.get("current_state_class", classify_state(start_state_family=start_state_family))),
        "transition_current_state_class": str(route.get("current_state_class", classify_state(start_state_family=start_state_family))),
        "latency_case": latency_case,
        "wall_margin_m": float(context.wall_margin_m),
        "all_wall_margin_m": float(context.all_wall_margin_m),
        "front_wall_margin_m": float(context.front_wall_margin_m),
        "left_wall_margin_m": float(context.left_wall_margin_m),
        "right_wall_margin_m": float(context.right_wall_margin_m),
        "rear_wall_margin_m": float(context.rear_wall_margin_m),
        "governor_wall_margin_m": float(context.governor_wall_margin_m),
        "floor_margin_m": float(context.floor_margin_m),
        "ceiling_margin_m": float(context.ceiling_margin_m),
        "w_wing_mean_m_s": float(context.w_wing_mean_m_s),
        "w_local_uncertainty_m_s": float(context.w_local_uncertainty_m_s),
        "lift_score": float(context.lift_score),
        "fan_count": int(context.fan_count),
        "updraft_model_id": context.updraft_model_id,
        "library_size_case_id": str(scheduled.get("library_size_case_id", "")),
        "history_length": int(scheduled.get("history_length", 0)),
        "policy_id": str(scheduled.get("policy_id", "")),
    }
    return {
        "context": context,
        "row": row,
        "wind_field": wind_field,
        "implementation_instance": implementation_instance_for_layer(implementation_layer, seed, latency_case=latency_case),
        "plant_instance": plant_instance_for_layer(plant_layer, seed),
    }


def _environment_randomisation_config_for_context(
    *,
    protocol: ValidationProtocol,
    scheduled: dict[str, object],
    scheduled_active_fan_count: int | None,
) -> EnvironmentRandomisationConfig | None:
    block_id = str(scheduled.get("environment_block_id", ""))
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id == BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="independent_uniform_xy_bounds",
            fan_position_xy_bounds_m=R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M,
        )
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id in R10_FIXED_BASE_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="fixed_base_positions",
        )
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id in R10_SHIFTED_FAN_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="common_shift",
        )
    if scheduled_active_fan_count is not None:
        return EnvironmentRandomisationConfig(active_fan_count=scheduled_active_fan_count)
    return None


def _governor_config_for_policy(policy: dict[str, object], *, base_config: GovernorConfig = DEFAULT_GOVERNOR_CONFIG) -> GovernorConfig:
    if bool(policy["safe_explore"]):
        return base_config
    return replace(
        base_config,
        config_id=f"{base_config.config_id}_no_exploration_ablation",
        exploration_bonus_weight=0.0,
    )


def _schedule_identity_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "library_size_case_id": str(row.get("library_size_case_id", "")),
        "policy_id": str(row.get("policy_id", "")),
        "history_length": int(row.get("history_length", 0)),
        "outer_case_index": int(row.get("outer_case_index", 0)),
        "outer_case_id": str(row.get("outer_case_id", "")),
        "outer_case_type": str(row.get("outer_case_type", "")),
        "environment_block_id": str(row.get("environment_block_id", "")),
        "common_final_launch_key": str(row.get("common_final_launch_key", "")),
        "episode_id": str(row.get("episode_id", "")),
    }


def _episode_row_from_sequence(
    *,
    scheduled: dict[str, object],
    policy: dict[str, object],
    primitive_rows: list[dict[str, object]],
    selector_rows: list[dict[str, object]],
    context_row: dict[str, object],
    belief_before: DirectionalResidualLiftBelief,
    belief_after: DirectionalResidualLiftBelief,
    blocked_reason: str = "",
) -> dict[str, object]:
    hard_failure = any(_rollout_row_is_hard_failure(row) for row in primitive_rows)
    floor_or_ceiling = any(_rollout_row_is_floor_or_ceiling_violation(row) for row in primitive_rows)
    terminal_useful = any(_truthy(row.get("episode_terminal_useful", False)) for row in primitive_rows)
    terminal_useful_safe_exit_only = any(_rollout_row_is_terminal_safe_exit_only(row) for row in primitive_rows)
    lift_capture = any(_float_value(row.get("lift_dwell_time_s", 0.0)) > 0.0 for row in primitive_rows)
    selected_variants = _sequence_values(primitive_rows, "primitive_variant_id")
    selected_primitives = _sequence_values(primitive_rows, "primitive_id")
    selected_controllers = _sequence_values(primitive_rows, "controller_id")
    selected_entry_roles = _sequence_values(primitive_rows, "selected_entry_role")
    selected_start_families = _sequence_values(primitive_rows, "start_state_family")
    selected_required_roles = _sequence_values(primitive_rows, "route_required_entry_role")
    selected_route_reasons = _sequence_values(primitive_rows, "route_reason")
    energy_summary = _episode_specific_energy_summary(primitive_rows)
    last_row = primitive_rows[-1]
    no_viable = bool(str(blocked_reason).startswith("no_viable"))
    sequence_compliant = _launch_sequence_compliant(primitive_rows)
    last_continuation_or_terminal = bool(
        _truthy(last_row.get("continuation_valid", False))
        or _truthy(last_row.get("episode_terminal_useful", False))
        or str(blocked_reason) == "episode_time_budget_reached"
    )
    episode_duration_s = _episode_rollout_duration_s(primitive_rows)
    safe_success = bool(sequence_compliant and last_continuation_or_terminal and not hard_failure and not floor_or_ceiling and not no_viable)
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_family": str(policy["policy_family"]),
        "safe_explore_active": bool(policy["safe_explore"]),
        "selected_primitive_variant_id": ";".join(selected_variants),
        "selected_primitive_id": ";".join(selected_primitives),
        "selected_controller_id": ";".join(selected_controllers),
        "selected_entry_role_sequence": ";".join(selected_entry_roles),
        "selected_start_state_family_sequence": ";".join(selected_start_families),
        "selected_route_required_entry_role_sequence": ";".join(selected_required_roles),
        "selected_route_reason_sequence": ";".join(selected_route_reasons),
        "selected_primitive_step_count": int(len(primitive_rows)),
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
        "launch_first_then_inflight_sequence_compliant": bool(sequence_compliant),
        "launch_inflight_recovery_sequence_compliant": bool(sequence_compliant),
        "termination_cause": str(blocked_reason or last_row.get("termination_cause", "")),
        "hard_failure": bool(hard_failure),
        "floor_or_ceiling_violation": bool(floor_or_ceiling),
        "no_viable_primitive": no_viable,
        "safe_success": safe_success,
        "full_safe_success": bool(safe_success and not terminal_useful_safe_exit_only),
        "terminal_useful": bool(terminal_useful),
        "terminal_useful_safe_exit_only": bool(terminal_useful_safe_exit_only),
        "lift_capture": bool(lift_capture),
        "episode_rollout_duration_s": float(episode_duration_s),
        "lift_dwell_time_s": float(sum(_float_value(row.get("lift_dwell_time_s", 0.0)) for row in primitive_rows)),
        "energy_residual_m": float(sum(_float_value(row.get("energy_residual_m", 0.0)) for row in primitive_rows)),
        **energy_summary,
        "min_wall_margin_m": float(min(_float_value(row.get("minimum_wall_margin_m", 0.0)) for row in primitive_rows)),
        "governor_rejection_count": int(
            sum(int(row.get("candidate_count", 0)) - int(row.get("viable_count", 0)) for row in selector_rows)
        ),
        "belief_observation_count": int(belief_after.update_count),
        "belief_uncertainty": float(max(0.0, 1.0 / math.sqrt(max(1, int(belief_after.update_count))))),
        "memory_changed_selection": False,
        "exploration_changed_selection": False,
        "environment_instance_id": str(context_row.get("environment_instance_id", "")),
        "claim_status": "simulation_only_repeated_launch_rollout_episode",
        "belief_update_count_before": int(belief_before.update_count),
        "belief_update_count_after": int(belief_after.update_count),
    }
    row.update(_launch_score_fields_for_role(row))
    return row


def _episode_row_from_rollout(
    scheduled: dict[str, object],
    policy: dict[str, object],
    rollout_row: dict[str, object],
    selector_row: dict[str, object],
    context_row: dict[str, object],
    belief_before: DirectionalResidualLiftBelief,
    belief_after: DirectionalResidualLiftBelief,
) -> dict[str, object]:
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_family": str(policy["policy_family"]),
        "safe_explore_active": bool(policy["safe_explore"]),
        "selected_primitive_variant_id": str(selector_row.get("selected_primitive_variant_id", "")),
        "selected_primitive_id": str(selector_row.get("selected_primitive_id", "")),
        "selected_controller_id": str(selector_row.get("selected_controller_id", "")),
        "selected_entry_role_sequence": str(selector_row.get("selected_entry_role", "")),
        "selected_start_state_family_sequence": str(context_row.get("start_state_family", "")),
        "selected_route_required_entry_role_sequence": str(context_row.get("route_required_entry_role", "")),
        "selected_route_reason_sequence": str(context_row.get("route_reason", "")),
        "selected_primitive_step_count": 1,
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
        "launch_first_then_inflight_sequence_compliant": bool(
            str(selector_row.get("selected_entry_role", selector_row.get("entry_role", "")))
            == _required_entry_role_for_start_family(str(context_row.get("start_state_family", "")))
        ),
        "launch_inflight_recovery_sequence_compliant": bool(
            str(selector_row.get("selected_entry_role", selector_row.get("entry_role", "")))
            == _required_entry_role_for_start_family(str(context_row.get("start_state_family", "")))
        ),
        "termination_cause": str(rollout_row.get("termination_cause", "")),
        "hard_failure": _rollout_row_is_hard_failure(rollout_row),
        "floor_or_ceiling_violation": _rollout_row_is_floor_or_ceiling_violation(rollout_row),
        "no_viable_primitive": False,
        "safe_success": bool(
            (_truthy(rollout_row.get("continuation_valid", False)) or _truthy(rollout_row.get("episode_terminal_useful", False)))
            and not _rollout_row_is_hard_failure(rollout_row)
            and not _rollout_row_is_floor_or_ceiling_violation(rollout_row)
        ),
        "full_safe_success": bool(
            (_truthy(rollout_row.get("continuation_valid", False)) or _truthy(rollout_row.get("episode_terminal_useful", False)))
            and not _rollout_row_is_hard_failure(rollout_row)
            and not _rollout_row_is_floor_or_ceiling_violation(rollout_row)
            and not _rollout_row_is_terminal_safe_exit_only(rollout_row)
        ),
        "terminal_useful": bool(rollout_row.get("episode_terminal_useful", False)),
        "terminal_useful_safe_exit_only": bool(_rollout_row_is_terminal_safe_exit_only(rollout_row)),
        "lift_capture": bool(float(rollout_row.get("lift_dwell_time_s", 0.0)) > 0.0),
        "episode_rollout_duration_s": float(_episode_rollout_duration_s([rollout_row])),
        "lift_dwell_time_s": float(rollout_row.get("lift_dwell_time_s", 0.0)),
        "energy_residual_m": float(rollout_row.get("energy_residual_m", 0.0)),
        **_episode_specific_energy_summary([rollout_row]),
        "min_wall_margin_m": float(rollout_row.get("minimum_wall_margin_m", 0.0)),
        "governor_rejection_count": int(selector_row.get("candidate_count", 0)) - int(selector_row.get("viable_count", 0)),
        "belief_observation_count": int(belief_after.update_count),
        "belief_uncertainty": float(max(0.0, 1.0 / math.sqrt(max(1, int(belief_after.update_count))))),
        "memory_changed_selection": False,
        "exploration_changed_selection": False,
        "environment_instance_id": str(context_row.get("environment_instance_id", "")),
        "claim_status": "simulation_only_repeated_launch_rollout_episode",
        "belief_update_count_before": int(belief_before.update_count),
        "belief_update_count_after": int(belief_after.update_count),
    }
    row.update(_launch_score_fields_for_role(row))
    return row


def _episode_row_from_blocked(
    scheduled: dict[str, object],
    policy: dict[str, object],
    context_row: dict[str, object],
    *,
    reason: str = "no_viable_primitive",
) -> dict[str, object]:
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_family": str(policy["policy_family"]),
        "safe_explore_active": bool(policy["safe_explore"]),
        "selected_primitive_variant_id": "",
        "selected_primitive_id": "",
        "selected_controller_id": "",
        "selected_entry_role_sequence": "",
        "selected_start_state_family_sequence": "",
        "selected_route_required_entry_role_sequence": "",
        "selected_route_reason_sequence": str(context_row.get("route_reason", "")),
        "selected_primitive_step_count": 0,
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
        "launch_first_then_inflight_sequence_compliant": False,
        "launch_inflight_recovery_sequence_compliant": False,
        "termination_cause": reason,
        "hard_failure": False,
        "floor_or_ceiling_violation": False,
        "no_viable_primitive": True,
        "safe_success": False,
        "full_safe_success": False,
        "terminal_useful": False,
        "terminal_useful_safe_exit_only": False,
        "lift_capture": False,
        "episode_rollout_duration_s": 0.0,
        "lift_dwell_time_s": 0.0,
        "energy_residual_m": 0.0,
        "episode_specific_energy_start_m": 0.0,
        "episode_specific_energy_end_m": 0.0,
        "net_specific_energy_delta_m": 0.0,
        "gross_specific_energy_gain_m": 0.0,
        "gross_specific_energy_loss_m": 0.0,
        "positive_specific_energy_gain_m": 0.0,
        "updraft_specific_energy_gain_proxy_m": 0.0,
        "updraft_gain_proxy_source": "blocked_no_primitive",
        "min_wall_margin_m": float(context_row.get("wall_margin_m", 0.0)),
        "governor_rejection_count": 0,
        "belief_observation_count": 0,
        "belief_uncertainty": 1.0,
        "memory_changed_selection": False,
        "exploration_changed_selection": False,
        "environment_instance_id": str(context_row.get("environment_instance_id", "")),
        "claim_status": "simulation_only_repeated_launch_rollout_episode",
        "belief_update_count_before": 0,
        "belief_update_count_after": 0,
    }
    row.update(_launch_score_fields_for_role(row))
    return row


def _episode_specific_energy_summary(primitive_rows: list[dict[str, object]]) -> dict[str, object]:
    energy_pairs: list[tuple[float, float]] = []
    updraft_proxy_terms: list[float] = []
    has_trajectory_integrated_updraft = any("trajectory_integrated_updraft_gain_m" in row for row in primitive_rows)
    for row in primitive_rows:
        start = _state_vector_from_rollout_row(row, prefix="initial_")
        end = _state_vector_from_json(row.get("exit_state_vector", ""))
        if start is None or end is None:
            continue
        energy_pairs.append((_specific_energy_m(start), _specific_energy_m(end)))
        if "updraft_specific_energy_gain_proxy_m" in row:
            updraft_proxy_terms.append(max(_float_value(row.get("updraft_specific_energy_gain_proxy_m", 0.0)), 0.0))
    if not energy_pairs:
        return {
            "episode_specific_energy_start_m": 0.0,
            "episode_specific_energy_end_m": 0.0,
            "net_specific_energy_delta_m": 0.0,
            "gross_specific_energy_gain_m": 0.0,
            "gross_specific_energy_loss_m": 0.0,
            "positive_specific_energy_gain_m": 0.0,
            "updraft_specific_energy_gain_proxy_m": float(sum(updraft_proxy_terms)),
            "updraft_gain_proxy_source": (
                "trajectory_integrated_positive_w_wing"
                if updraft_proxy_terms and has_trajectory_integrated_updraft
                else "primitive_start_local_w_wing_proxy"
                if updraft_proxy_terms
                else "unavailable"
            ),
        }
    deltas = [end - start for start, end in energy_pairs]
    positive_specific_gain = float(sum(max(delta, 0.0) for delta in deltas))
    if updraft_proxy_terms:
        updraft_proxy = float(sum(updraft_proxy_terms))
        updraft_source = (
            "trajectory_integrated_positive_w_wing"
            if has_trajectory_integrated_updraft
            else "primitive_start_local_w_wing_proxy"
        )
    else:
        updraft_proxy = positive_specific_gain
        updraft_source = "positive_specific_energy_gain_fallback"
    return {
        "episode_specific_energy_start_m": float(energy_pairs[0][0]),
        "episode_specific_energy_end_m": float(energy_pairs[-1][1]),
        "net_specific_energy_delta_m": float(energy_pairs[-1][1] - energy_pairs[0][0]),
        "gross_specific_energy_gain_m": positive_specific_gain,
        "gross_specific_energy_loss_m": float(sum(max(-delta, 0.0) for delta in deltas)),
        "positive_specific_energy_gain_m": positive_specific_gain,
        "updraft_specific_energy_gain_proxy_m": updraft_proxy,
        "updraft_gain_proxy_source": updraft_source,
    }


def _rollout_row_is_hard_failure(row: dict[str, object]) -> bool:
    failure_label = str(row.get("failure_label", ""))
    if failure_label in PHYSICAL_HARD_FAILURE_LABELS or "nonfinite" in failure_label or "corrupt" in failure_label:
        return True
    if str(row.get("boundary_use_class", "")) == "hard_failure":
        return failure_label not in {"model_boundary_only", "weak_energy_result", "success", ""}
    return False


def _rollout_row_is_floor_or_ceiling_violation(row: dict[str, object]) -> bool:
    return str(row.get("failure_label", "")) in {
        "floor_violation",
        "ceiling_violation",
        "initial_floor_violation",
        "initial_ceiling_violation",
        "z_boundary_exit",
    }


def _rollout_row_is_terminal_safe_exit_only(row: dict[str, object]) -> bool:
    if not _truthy(row.get("episode_terminal_useful", False)):
        return False
    if _rollout_row_is_hard_failure(row) or _rollout_row_is_floor_or_ceiling_violation(row):
        return False
    boundary_class = str(row.get("boundary_use_class", ""))
    cause = str(row.get("termination_cause", ""))
    label = str(row.get("failure_label", ""))
    return bool(
        boundary_class == "episode_terminal_useful"
        or "wall" in cause
        or "xy_boundary" in label
        or "boundary_terminal" in label
    )


def _episode_rollout_duration_s(primitive_rows: list[dict[str, object]]) -> float:
    total = 0.0
    for row in primitive_rows:
        if "rollout_duration_s" in row:
            total += max(_float_value(row.get("rollout_duration_s", 0.0)), 0.0)
        else:
            total += float(PRIMITIVE_FINITE_HORIZON_S)
    return float(total)


def _primitive_updraft_gain_proxy_m(
    context_row: dict[str, object],
    *,
    rollout_row: dict[str, object] | None = None,
) -> float:
    """Estimate useful updraft exposure, preferring trajectory integration."""

    if rollout_row is not None and "trajectory_integrated_updraft_gain_m" in rollout_row:
        return float(max(_float_value(rollout_row.get("trajectory_integrated_updraft_gain_m", 0.0)), 0.0))
    w_wing = max(_float_value(context_row.get("w_wing_mean_m_s", 0.0)), 0.0)
    return float(w_wing * PRIMITIVE_FINITE_HORIZON_S)


def _lift_residual_for_memory_update(
    context_row: dict[str, object],
    *,
    rollout_row: dict[str, object],
    outcome: dict[str, object],
) -> float:
    """Compare experienced positive wing lift with the context-conditioned expectation."""

    if "trajectory_mean_positive_w_wing_m_s" in rollout_row:
        observed = _float_value(rollout_row.get("trajectory_mean_positive_w_wing_m_s", 0.0))
    else:
        observed = max(_float_value(context_row.get("w_wing_mean_m_s", 0.0)), 0.0)
    if "w_wing_mean_m_s" in outcome:
        expected = max(_float_value(outcome.get("w_wing_mean_m_s", 0.0)), 0.0)
    else:
        expected = max(_float_value(outcome.get("expected_updraft_gain_proxy_m", 0.0)), 0.0) / float(
            PRIMITIVE_FINITE_HORIZON_S
        )
    return float(observed - expected)


def _updraft_gain_residual_for_memory_update(
    context_row: dict[str, object],
    *,
    rollout_row: dict[str, object],
    outcome: dict[str, object],
) -> float:
    """Update memory from useful updraft exposure, not whole-flight energy loss."""

    observed = _primitive_updraft_gain_proxy_m(context_row, rollout_row=rollout_row)
    expected = max(_float_value(outcome.get("expected_updraft_gain_proxy_m", 0.0)), 0.0)
    return float(observed - expected)


def _specific_energy_m(state: np.ndarray) -> float:
    x = as_state_vector(state)
    speed = float(np.linalg.norm(x[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]))
    return float(x[STATE_INDEX["z_w"]] + speed * speed / (2.0 * SPECIFIC_ENERGY_GRAVITY_M_S2))


def _state_vector_from_rollout_row(row: dict[str, object], *, prefix: str) -> np.ndarray | None:
    if prefix == "initial_" and row.get("initial_state_vector", ""):
        parsed = _state_vector_from_json(row.get("initial_state_vector", ""))
        if parsed is not None:
            return parsed
    values: dict[str, float] = {}
    for name in ("x_w", "y_w", "z_w", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "delta_a", "delta_e", "delta_r"):
        key = f"{prefix}{name}"
        if key not in row:
            return None
        values[name] = _float_value(row.get(key, 0.0))
    return as_state_vector(np.array([values[name] for name in STATE_INDEX.keys()], dtype=float))


def _state_vector_from_json(value: object) -> np.ndarray | None:
    try:
        return as_state_vector(np.asarray(json.loads(str(value)), dtype=float))
    except Exception:
        return None


def _launch_score_fields(row: dict[str, object]) -> dict[str, object]:
    selected_steps = int(_float_value(row.get("selected_primitive_step_count", 0)))
    episode_time_s = _float_value(
        row.get(
            "episode_rollout_duration_s",
            float(selected_steps) * float(PRIMITIVE_FINITE_HORIZON_S),
        )
    )
    hard_failure = _truthy(row.get("hard_failure", False))
    floor_or_ceiling = _truthy(row.get("floor_or_ceiling_violation", False))
    no_viable = _truthy(row.get("no_viable_primitive", False))
    no_viable_at_launch = bool(no_viable and selected_steps <= 0)
    no_viable_after_launch = bool(no_viable and selected_steps > 0)
    min_wall_margin = _float_value(row.get("min_wall_margin_m", 0.0))
    wall_boundary_issue = bool(min_wall_margin < 0.0 and not hard_failure and not floor_or_ceiling)
    base_penalty, penalty_reason = _base_failure_penalty(
        hard_failure=hard_failure,
        floor_or_ceiling=floor_or_ceiling,
        no_viable_at_launch=no_viable_at_launch,
        no_viable_after_launch=no_viable_after_launch,
        wall_boundary_issue=wall_boundary_issue,
    )
    outcome_multiplier = _outcome_multiplier(
        safe_success=_truthy(row.get("safe_success", False)),
        terminal_useful=_truthy(row.get("terminal_useful", False)),
        lift_capture=_truthy(row.get("lift_capture", False)),
        hard_failure=hard_failure,
    )
    safety_multiplier = _safety_multiplier(
        hard_failure=hard_failure,
        floor_or_ceiling=floor_or_ceiling,
        wall_boundary_issue=wall_boundary_issue,
    )
    viability_multiplier = _viability_multiplier(
        no_viable_at_launch=no_viable_at_launch,
        no_viable_after_launch=no_viable_after_launch,
    )
    updraft_gain_factor = _clip(1.0 + _float_value(row.get("updraft_specific_energy_gain_proxy_m", 0.0)) / 1.0, 1.00, 1.75)
    flight_time_factor = _clip(episode_time_s / SCORING_TARGET_EPISODE_TIME_S, 0.10, 1.25)
    multiplicative_component = (
        100.0
        * outcome_multiplier
        * safety_multiplier
        * viability_multiplier
        * updraft_gain_factor
        * flight_time_factor
    )
    return {
        "launch_score_version": LAUNCH_SCORE_VERSION,
        "episode_flight_time_s": float(episode_time_s),
        "base_failure_penalty": float(base_penalty),
        "base_failure_penalty_reason": penalty_reason,
        "outcome_multiplier": float(outcome_multiplier),
        "safety_multiplier": float(safety_multiplier),
        "viability_multiplier": float(viability_multiplier),
        "updraft_gain_factor": float(updraft_gain_factor),
        "flight_time_factor": float(flight_time_factor),
        "launch_score_multiplicative_component": float(multiplicative_component),
        "launch_score": float(base_penalty + multiplicative_component),
    }


def _launch_score_fields_for_role(row: dict[str, object]) -> dict[str, object]:
    if str(row.get("launch_role", "")) == "final_heldout":
        fields = _launch_score_fields(row)
        fields["launch_score_scope"] = "final_heldout_outer_loop_score"
        return fields
    selected_steps = int(_float_value(row.get("selected_primitive_step_count", 0)))
    episode_time_s = _float_value(
        row.get(
            "episode_rollout_duration_s",
            float(selected_steps) * float(PRIMITIVE_FINITE_HORIZON_S),
        )
    )
    return {
        "launch_score_version": LAUNCH_SCORE_VERSION,
        "episode_flight_time_s": float(episode_time_s),
        "base_failure_penalty": float("nan"),
        "base_failure_penalty_reason": "not_scored_history_launch",
        "outcome_multiplier": float("nan"),
        "safety_multiplier": float("nan"),
        "viability_multiplier": float("nan"),
        "updraft_gain_factor": float("nan"),
        "flight_time_factor": float("nan"),
        "launch_score_multiplicative_component": float("nan"),
        "launch_score": float("nan"),
        "launch_score_scope": "history_launch_memory_update_not_outer_loop_score",
    }


def _base_failure_penalty(
    *,
    hard_failure: bool,
    floor_or_ceiling: bool,
    no_viable_at_launch: bool,
    no_viable_after_launch: bool,
    wall_boundary_issue: bool,
) -> tuple[float, str]:
    if hard_failure:
        return -100.0, "hard_failure"
    if floor_or_ceiling:
        return -100.0, "floor_or_ceiling_violation"
    if no_viable_at_launch:
        return -70.0, "no_viable_primitive_at_launch"
    if no_viable_after_launch:
        return -40.0, "no_viable_primitive_after_launch"
    if wall_boundary_issue:
        return -30.0, "wall_boundary_issue_not_hard_failure"
    return 0.0, "none"


def _outcome_multiplier(*, safe_success: bool, terminal_useful: bool, lift_capture: bool, hard_failure: bool) -> float:
    if safe_success and terminal_useful:
        return 1.00
    if safe_success and lift_capture:
        return 0.90
    if safe_success:
        return 0.75
    if terminal_useful and not hard_failure:
        return 0.50
    if lift_capture and not hard_failure:
        return 0.35
    return 0.10


def _safety_multiplier(*, hard_failure: bool, floor_or_ceiling: bool, wall_boundary_issue: bool) -> float:
    if hard_failure or floor_or_ceiling:
        return 0.0
    if wall_boundary_issue:
        return 0.60
    return 1.0


def _viability_multiplier(*, no_viable_at_launch: bool, no_viable_after_launch: bool) -> float:
    if no_viable_at_launch:
        return 0.0
    if no_viable_after_launch:
        return 0.50
    return 1.0


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(float(lower), min(float(upper), float(value))))


def _belief_snapshot_compact(
    *,
    belief: DirectionalResidualLiftBelief,
    scheduled: dict[str, object],
    phase: str,
    features: dict[str, object],
) -> dict[str, object]:
    return {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled.get("launch_role", "")),
        "phase": phase,
        "belief_version": belief.belief_version,
        "belief_update_count": int(belief.update_count),
        "belief_cell_count": int(len(belief.cells)),
        "belief_observation_count": int(features.get("belief_observation_count", 0) or 0),
        "belief_uncertainty": float(features.get("belief_uncertainty", 1.0) or 1.0),
        "belief_local_lift_residual_m_s": float(features.get("belief_local_lift_residual_m_s", 0.0) or 0.0),
        "belief_local_updraft_gain_proxy_m": float(
            features.get(
                "belief_local_updraft_gain_proxy_m",
                max(float(features.get("belief_local_updraft_gain_residual_m", 0.0) or 0.0), 0.0),
            )
            or 0.0
        ),
        "belief_local_updraft_gain_residual_m": float(features.get("belief_local_updraft_gain_residual_m", 0.0) or 0.0),
        "belief_local_energy_residual_m": float(
            features.get(
                "belief_local_energy_residual_m",
                features.get("belief_local_updraft_gain_residual_m", 0.0),
            )
            or 0.0
        ),
        "belief_local_dwell_residual_s": float(features.get("belief_local_dwell_residual_s", 0.0) or 0.0),
    }


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _sequence_values(rows: list[dict[str, object]], column: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = str(row.get(column, "")).strip()
        if value:
            values.append(value)
    return values


def _launch_sequence_compliant(rows: list[dict[str, object]]) -> bool:
    if not rows:
        return False
    for index, row in enumerate(rows):
        start_family = str(row.get("start_state_family", ""))
        if index == 0 and start_family != FIRST_PRIMITIVE_START_FAMILY:
            return False
        if index > 0 and start_family == FIRST_PRIMITIVE_START_FAMILY:
            return False
        expected_role = _required_entry_role_for_start_family(start_family)
        if str(row.get("selected_entry_role", "")) != expected_role:
            return False
        required_entry_class = str(row.get("route_required_entry_class", "")).strip()
        allowed_entry_classes = {required_entry_class} if required_entry_class else set(
            entry_classes_for_state_class(classify_state(start_state_family=start_family))
        )
        selected_entry_class = str(
            row.get("selected_transition_entry_class", row.get("transition_entry_class", ""))
        ).strip()
        if selected_entry_class and selected_entry_class not in allowed_entry_classes:
            return False
    return True


def _append_launch_result(buffers: dict[str, list[dict[str, object]]], result: dict[str, object]) -> None:
    buffers["episode_summary"].extend(result["episode_rows"])
    buffers["primitive_execution_log"].extend(result["primitive_rows"])
    buffers["candidate_score_log"].extend(result["candidate_rows"])
    buffers["selector_decision_log"].extend(result["selector_rows"])
    buffers["memory_residual_update_log"].extend(result["memory_rows"])
    buffers["belief_snapshot_log"].extend(result["belief_rows"])


def _flush_if_needed(
    *,
    run_root: Path,
    table_buffers: dict[str, list[dict[str, object]]],
    row_counters: dict[str, int],
    storage_format: str,
    compression_level: int,
    chunk_size: int,
) -> list[TablePartition]:
    partitions: list[TablePartition] = []
    for table_name, rows in table_buffers.items():
        while len(rows) >= int(chunk_size):
            chunk_rows = rows[: int(chunk_size)]
            del rows[: int(chunk_size)]
            partitions.append(
                _flush_table(run_root, table_name, chunk_rows, row_counters, storage_format, compression_level)
            )
    return partitions


def _flush_all(
    *,
    run_root: Path,
    table_buffers: dict[str, list[dict[str, object]]],
    row_counters: dict[str, int],
    storage_format: str,
    compression_level: int,
) -> list[TablePartition]:
    partitions: list[TablePartition] = []
    for table_name, rows in table_buffers.items():
        if rows:
            partitions.append(_flush_table(run_root, table_name, rows, row_counters, storage_format, compression_level))
            rows.clear()
    return partitions


def _flush_table(
    run_root: Path,
    table_name: str,
    rows: list[dict[str, object]],
    row_counters: dict[str, int],
    storage_format: str,
    compression_level: int,
) -> TablePartition:
    chunk_index = int(row_counters[table_name])
    row_counters[table_name] += 1
    extension = table_extension(storage_format)
    path = run_root / "tables" / table_name / f"c{chunk_index:05d}.{extension}"
    return write_table_partition(
        pd.DataFrame(rows),
        path,
        storage_format=storage_format,
        compression_level=int(compression_level),
    )


def _write_schedule_metric(
    *,
    run_root: Path,
    table_name: str,
    rows: list[dict[str, object]],
    run_id: int,
    storage_format: str,
    compression_level: int,
) -> None:
    frame = pd.DataFrame(rows)
    metrics_path = run_root / "metrics" / f"{table_name}.csv"
    if len(frame) <= SCHEDULE_INLINE_ROW_LIMIT:
        _write_csv(metrics_path, frame)
        return

    partitions: list[TablePartition] = []
    extension = table_extension(storage_format)
    for partition_index, start in enumerate(range(0, len(frame), SCHEDULE_PARTITION_ROW_COUNT)):
        chunk = frame.iloc[start : start + SCHEDULE_PARTITION_ROW_COUNT].copy()
        path = run_root / "tables" / table_name / f"c{partition_index:05d}.{extension}"
        partitions.append(
            write_table_partition(
                chunk,
                path,
                storage_format=storage_format,
                compression_level=int(compression_level),
            )
        )
    manifest_path = run_root / "manifests" / f"{table_name}_manifest.json"
    write_table_manifest(
        manifest_path,
        TableManifest(
            run_id=int(run_id),
            root=run_root.as_posix(),
            storage_format=storage_format,
            tables=tuple(partitions),
        ),
    )
    _write_csv(
        metrics_path,
        pd.DataFrame(
            [
                {
                    "table_name": table_name,
                    "row_level_log": f"tables/{table_name}/",
                    "storage": storage_format,
                    "partition_count": int(len(partitions)),
                    "row_count": int(len(frame)),
                    "manifest": manifest_path.relative_to(run_root).as_posix(),
                    "file_size_policy": "partitioned_to_avoid_large_git_blobs",
                }
            ]
        ),
    )


def _read_partitioned_rows(run_root: Path, partitions: Iterable[TablePartition], table_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition in partitions:
        if partition.table_name != table_name:
            continue
        frame = read_table_partition(
            run_root / "tables" / partition.relative_path,
            storage_format=partition.storage_format,
        )
        rows.extend(frame.to_dict(orient="records"))
    return rows


def _write_first_decision_audits_from_partitions(
    run_root: Path,
    partitions: Iterable[TablePartition],
    storage_format: str,
) -> None:
    first_decision_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    availability: dict[str, dict[str, object]] = {}
    for partition in partitions:
        if partition.table_name != "candidate_score_log":
            continue
        frame = read_table_partition(
            run_root / "tables" / partition.relative_path,
            storage_format=storage_format,
        )
        if frame.empty or "primitive_step_index" not in frame.columns:
            continue
        first = frame[pd.to_numeric(frame["primitive_step_index"], errors="coerce").fillna(-1).astype(int) == 0].copy()
        if first.empty:
            continue
        first["viable_int"] = first["viable"].astype(str).str.lower().isin({"true", "1"}).astype(int)
        first_decision_rows.extend(
            first.groupby(["library_size_case_id", "policy_id", "launch_role"], dropna=False)
            .agg(first_decision_candidate_rows=("primitive_variant_id", "count"), first_decision_viable_rows=("viable_int", "sum"))
            .reset_index()
            .to_dict(orient="records")
        )
        rejection_rows.extend(
            first.groupby(["library_size_case_id", "policy_id", "launch_role", "rejection_reason"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .to_dict(orient="records")
        )
        entry_rows.extend(
            first.groupby(["library_size_case_id", "primitive_id", "transition_entry_class", "start_state_family"], dropna=False)
            .agg(candidate_rows=("primitive_variant_id", "count"), viable_rows=("viable_int", "sum"))
            .reset_index()
            .to_dict(orient="records")
        )
        for case_id, group in first.groupby("library_size_case_id", dropna=False):
            key = str(case_id)
            row = availability.setdefault(
                key,
                {
                    "stage_id": "R9_R10_R11",
                    "library_size_case_id": key,
                    "launch_gate_entry_primitives": set(),
                    "first_decision_candidate_rows": 0,
                    "first_decision_viable_rows": 0,
                },
            )
            launch_entry = group[group["transition_entry_class"].astype(str) == "launch_gate"]
            row["launch_gate_entry_primitives"].update(launch_entry["primitive_id"].astype(str).tolist())
            row["first_decision_candidate_rows"] = int(row["first_decision_candidate_rows"]) + int(len(group))
            row["first_decision_viable_rows"] = int(row["first_decision_viable_rows"]) + int(group["viable_int"].sum())
    _write_csv(run_root / "metrics" / "first_decision_candidate_summary.csv", _sum_rows(first_decision_rows, ["library_size_case_id", "policy_id", "launch_role"]))
    _write_csv(run_root / "metrics" / "first_decision_governor_rejection_summary.csv", _sum_rows(rejection_rows, ["library_size_case_id", "policy_id", "launch_role", "rejection_reason"]))
    _write_csv(run_root / "metrics" / "launch_gate_entry_role_audit.csv", _sum_rows(entry_rows, ["library_size_case_id", "primitive_id", "transition_entry_class", "start_state_family"]))
    availability_rows = []
    for row in availability.values():
        launch_entry = set(row.pop("launch_gate_entry_primitives"))
        availability_rows.append(
            {
                **row,
                "launch_gate_entry_primitive_family_count": int(len(launch_entry)),
                "launch_gate_entry_primitive_ids": ",".join(sorted(launch_entry)),
                "first_decision_audit_mode": "full_validation" if any(bool(partition.table_name == "candidate_score_log") for partition in partitions) else "not_run",
            }
        )
    _write_csv(run_root / "metrics" / "launch_gate_candidate_availability.csv", pd.DataFrame(availability_rows))
    _write_csv(run_root / "metrics" / "launch_gate_outcome_audit.csv", _sum_rows(rejection_rows, ["library_size_case_id", "rejection_reason"]))


def _sum_rows(rows: list[dict[str, object]], group_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    value_columns = [column for column in frame.columns if column not in set(group_columns)]
    numeric = []
    for column in value_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            numeric.append(column)
        except Exception:
            pass
    if not numeric:
        return frame.drop_duplicates(group_columns).reset_index(drop=True)
    return frame.groupby(group_columns, dropna=False)[numeric].sum().reset_index()


def _write_compact_metric_tables(run_root: Path, episode_rows: list[dict[str, object]], protocol: ValidationProtocol) -> None:
    frame = pd.DataFrame(episode_rows)
    final = frame[frame["launch_role"].astype(str) == "final_heldout"] if not frame.empty else pd.DataFrame()
    if not final.empty:
        final = _with_launch_score_columns(final)
        final = _with_selection_change_flags(final)
    _write_csv(run_root / "metrics" / "final_launch_score.csv", final)
    if final.empty:
        comparison = pd.DataFrame()
    else:
        comparison = (
            final.groupby(["library_size_case_id", "policy_id", "history_length"], dropna=False)
            .agg(
                launch_count=("episode_id", "count"),
                safe_success_rate=("safe_success", "mean"),
                full_safe_success_rate=("full_safe_success", "mean"),
                hard_failure_rate=("hard_failure", "mean"),
                floor_or_ceiling_violation_rate=("floor_or_ceiling_violation", "mean"),
                no_viable_primitive_rate=("no_viable_primitive", "mean"),
                terminal_useful_rate=("terminal_useful", "mean"),
                lift_capture_rate=("lift_capture", "mean"),
                mean_lift_dwell_time_s=("lift_dwell_time_s", "mean"),
                mean_energy_residual_m=("energy_residual_m", "mean"),
                mean_net_specific_energy_delta_m=("net_specific_energy_delta_m", "mean"),
                mean_positive_specific_energy_gain_m=("positive_specific_energy_gain_m", "mean"),
                mean_updraft_specific_energy_gain_proxy_m=("updraft_specific_energy_gain_proxy_m", "mean"),
                mean_gross_specific_energy_loss_m=("gross_specific_energy_loss_m", "mean"),
                mean_episode_flight_time_s=("episode_flight_time_s", "mean"),
                mean_launch_score=("launch_score", "mean"),
                median_launch_score=("launch_score", "median"),
                mean_min_wall_margin_m=("min_wall_margin_m", "mean"),
                selected_primitive_family_count=("selected_primitive_id", pd.Series.nunique),
                selected_variant_count=("selected_primitive_variant_id", pd.Series.nunique),
                governor_rejection_count=("governor_rejection_count", "sum"),
                belief_observation_count=("belief_observation_count", "max"),
                belief_uncertainty=("belief_uncertainty", "mean"),
                memory_changed_selection_rate=("memory_changed_selection", "mean"),
                exploration_changed_selection_rate=("exploration_changed_selection", "mean"),
            )
            .reset_index()
        )
    _write_csv(run_root / "metrics" / "policy_history_comparison.csv", comparison)
    memory_delta = _paired_score_delta_rows(final, baseline_policy_id="no_memory_baseline")
    safe_explore_delta = _paired_safe_explore_delta_rows(final)
    _write_csv(run_root / "metrics" / "paired_memory_score_delta.csv", memory_delta)
    _write_csv(run_root / "metrics" / "paired_safe_explore_score_delta.csv", safe_explore_delta)
    _write_csv(
        run_root / "metrics" / "paired_score_delta_summary.csv",
        _paired_score_delta_summary(pd.concat([memory_delta, safe_explore_delta], ignore_index=True)),
    )
    if final.empty:
        library = pd.DataFrame()
    else:
        library = (
            final.groupby(["library_size_case_id"], dropna=False)
            .agg(
                launch_count=("episode_id", "count"),
                safe_success_rate=("safe_success", "mean"),
                full_safe_success_rate=("full_safe_success", "mean"),
                hard_failure_rate=("hard_failure", "mean"),
                no_viable_primitive_rate=("no_viable_primitive", "mean"),
                mean_launch_score=("launch_score", "mean"),
            )
            .reset_index()
        )
    _write_csv(run_root / "metrics" / "library_size_case_comparison.csv", library)
    if protocol.stage_id in CHANGED_CASE_VALIDATION_STAGE_IDS:
        env = (
            final.groupby(["environment_block_id"], dropna=False)
            .agg(
                launch_count=("episode_id", "count"),
                safe_success_rate=("safe_success", "mean"),
                full_safe_success_rate=("full_safe_success", "mean"),
                mean_launch_score=("launch_score", "mean"),
            )
            .reset_index()
            if not final.empty
            else pd.DataFrame()
        )
        _write_csv(run_root / "metrics" / "environment_block_comparison.csv", env)
    term = (
        frame.groupby(["launch_role", "termination_cause"], dropna=False)
        .size()
        .reset_index(name="row_count")
        if not frame.empty
        else pd.DataFrame()
    )
    _write_csv(run_root / "metrics" / "termination_summary.csv", term)
    for required_name in ("candidate_score_log", "selector_decision_log", "memory_residual_update_log", "belief_snapshot_log", "primitive_execution_log", "episode_summary"):
        _write_csv(
            run_root / "metrics" / f"{required_name}.csv",
            pd.DataFrame([{"table_name": required_name, "row_level_log": f"tables/{required_name}/", "storage": "partitioned"}]),
        )


def _with_launch_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    score_rows = []
    for row in out.to_dict(orient="records"):
        fields = _launch_score_fields(row)
        if "net_specific_energy_delta_m" not in row:
            fields["net_specific_energy_delta_m"] = _float_value(row.get("energy_residual_m", 0.0))
        if "gross_specific_energy_gain_m" not in row:
            fields["gross_specific_energy_gain_m"] = max(fields.get("net_specific_energy_delta_m", _float_value(row.get("net_specific_energy_delta_m", 0.0))), 0.0)
        if "gross_specific_energy_loss_m" not in row:
            fields["gross_specific_energy_loss_m"] = max(-fields.get("net_specific_energy_delta_m", _float_value(row.get("net_specific_energy_delta_m", 0.0))), 0.0)
        if "positive_specific_energy_gain_m" not in row:
            fields["positive_specific_energy_gain_m"] = _float_value(row.get("gross_specific_energy_gain_m", fields.get("gross_specific_energy_gain_m", 0.0)))
        if "updraft_specific_energy_gain_proxy_m" not in row:
            fields["updraft_specific_energy_gain_proxy_m"] = _float_value(
                row.get("positive_specific_energy_gain_m", row.get("gross_specific_energy_gain_m", 0.0))
            )
            fields["updraft_gain_proxy_source"] = "positive_specific_energy_gain_fallback"
        score_rows.append(fields)
    scores = pd.DataFrame(score_rows, index=out.index)
    for column in scores.columns:
        out[column] = scores[column]
    return out


def _paired_score_delta_rows(final: pd.DataFrame, *, baseline_policy_id: str) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    baseline = final[final["policy_id"].astype(str) == str(baseline_policy_id)]
    if baseline.empty:
        return pd.DataFrame()
    baseline_map = {_paired_launch_key(row): row for row in baseline.to_dict(orient="records")}
    rows: list[dict[str, object]] = []
    for row in final.to_dict(orient="records"):
        policy_id = str(row.get("policy_id", ""))
        if policy_id == str(baseline_policy_id):
            continue
        if not (policy_id.startswith(MEMORY_POLICY_PREFIX) or policy_id == EMPTY_FROZEN_PRIOR_BASELINE_ID):
            continue
        baseline_row = baseline_map.get(_paired_launch_key(row))
        if baseline_row is None:
            continue
        rows.append(_paired_score_delta_row(row, baseline_row, baseline_policy_id=str(baseline_policy_id), comparison_type="memory_vs_no_memory"))
    return pd.DataFrame(rows)


def _paired_safe_explore_delta_rows(final: pd.DataFrame) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    memory = final[final["policy_id"].astype(str).str.startswith(MEMORY_POLICY_PREFIX)]
    memory_map = {
        (_paired_launch_key(row), int(_float_value(row.get("history_length", 0)))): row
        for row in memory.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for row in final[final["policy_id"].astype(str).str.startswith(SAFE_EXPLORE_POLICY_PREFIX)].to_dict(orient="records"):
        history_length = int(_float_value(row.get("history_length", 0)))
        baseline_row = memory_map.get((_paired_launch_key(row), history_length))
        if baseline_row is None:
            continue
        rows.append(
            _paired_score_delta_row(
                row,
                baseline_row,
                baseline_policy_id=str(baseline_row.get("policy_id", "")),
                comparison_type="safe_explore_vs_matching_memory",
            )
        )
    return pd.DataFrame(rows)


def _paired_launch_key(row: dict[str, object]) -> tuple[str, str]:
    return (
        str(row.get("library_size_case_id", "")),
        str(row.get("common_final_launch_key", row.get("outer_case_index", ""))),
    )


def _paired_score_delta_row(row: dict[str, object], baseline_row: dict[str, object], *, baseline_policy_id: str, comparison_type: str) -> dict[str, object]:
    safety_regression = bool(
        (_truthy(row.get("hard_failure", False)) and not _truthy(baseline_row.get("hard_failure", False)))
        or (
            _truthy(row.get("floor_or_ceiling_violation", False))
            and not _truthy(baseline_row.get("floor_or_ceiling_violation", False))
        )
        or (
            _truthy(row.get("no_viable_primitive", False))
            and not _truthy(baseline_row.get("no_viable_primitive", False))
        )
    )
    return {
        "comparison_type": str(comparison_type),
        "library_size_case_id": str(row.get("library_size_case_id", "")),
        "environment_block_id": str(row.get("environment_block_id", "")),
        "outer_case_index": int(_float_value(row.get("outer_case_index", 0))),
        "common_final_launch_key": str(row.get("common_final_launch_key", "")),
        "policy_id": str(row.get("policy_id", "")),
        "baseline_policy_id": str(baseline_policy_id),
        "history_length": int(_float_value(row.get("history_length", 0))),
        "launch_score": _float_value(row.get("launch_score", 0.0)),
        "baseline_launch_score": _float_value(baseline_row.get("launch_score", 0.0)),
        "paired_delta_launch_score": _float_value(row.get("launch_score", 0.0)) - _float_value(baseline_row.get("launch_score", 0.0)),
        "safe_success_delta": int(_truthy(row.get("safe_success", False))) - int(_truthy(baseline_row.get("safe_success", False))),
        "hard_failure_delta": int(_truthy(row.get("hard_failure", False))) - int(_truthy(baseline_row.get("hard_failure", False))),
        "floor_or_ceiling_violation_delta": int(_truthy(row.get("floor_or_ceiling_violation", False)))
        - int(_truthy(baseline_row.get("floor_or_ceiling_violation", False))),
        "no_viable_primitive_delta": int(_truthy(row.get("no_viable_primitive", False))) - int(_truthy(baseline_row.get("no_viable_primitive", False))),
        "net_specific_energy_delta_m_delta": _float_value(row.get("net_specific_energy_delta_m", 0.0))
        - _float_value(baseline_row.get("net_specific_energy_delta_m", 0.0)),
        "positive_specific_energy_gain_m_delta": _float_value(row.get("positive_specific_energy_gain_m", 0.0))
        - _float_value(baseline_row.get("positive_specific_energy_gain_m", 0.0)),
        "updraft_specific_energy_gain_proxy_m_delta": _float_value(row.get("updraft_specific_energy_gain_proxy_m", 0.0))
        - _float_value(baseline_row.get("updraft_specific_energy_gain_proxy_m", 0.0)),
        "gross_specific_energy_loss_m_delta": _float_value(row.get("gross_specific_energy_loss_m", 0.0))
        - _float_value(baseline_row.get("gross_specific_energy_loss_m", 0.0)),
        "episode_flight_time_s_delta": _float_value(row.get("episode_flight_time_s", 0.0))
        - _float_value(baseline_row.get("episode_flight_time_s", 0.0)),
        "memory_changed_selection": bool(row.get("memory_changed_selection", False)),
        "exploration_changed_selection": bool(row.get("exploration_changed_selection", False)),
        "safety_regression": safety_regression,
        "claim_status": "simulation_only_paired_launch_score_audit",
    }


def _paired_score_delta_summary(delta_rows: pd.DataFrame) -> pd.DataFrame:
    if delta_rows.empty:
        return pd.DataFrame()
    frame = delta_rows.copy()
    frame["win"] = pd.to_numeric(frame["paired_delta_launch_score"], errors="coerce").fillna(0.0) > 0.0
    frame["loss"] = pd.to_numeric(frame["paired_delta_launch_score"], errors="coerce").fillna(0.0) < 0.0
    return (
        frame.groupby(["comparison_type", "library_size_case_id", "policy_id", "baseline_policy_id", "history_length"], dropna=False)
        .agg(
            paired_launch_count=("paired_delta_launch_score", "count"),
            mean_paired_delta_launch_score=("paired_delta_launch_score", "mean"),
            median_paired_delta_launch_score=("paired_delta_launch_score", "median"),
            win_rate=("win", "mean"),
            loss_rate=("loss", "mean"),
            safety_regression_rate=("safety_regression", "mean"),
            memory_changed_selection_rate=("memory_changed_selection", "mean"),
            exploration_changed_selection_rate=("exploration_changed_selection", "mean"),
            mean_net_specific_energy_delta_m_delta=("net_specific_energy_delta_m_delta", "mean"),
            mean_positive_specific_energy_gain_m_delta=("positive_specific_energy_gain_m_delta", "mean"),
            mean_updraft_specific_energy_gain_proxy_m_delta=("updraft_specific_energy_gain_proxy_m_delta", "mean"),
            mean_gross_specific_energy_loss_m_delta=("gross_specific_energy_loss_m_delta", "mean"),
            mean_episode_flight_time_s_delta=("episode_flight_time_s_delta", "mean"),
        )
        .reset_index()
    )


def _with_selection_change_flags(final: pd.DataFrame) -> pd.DataFrame:
    out = final.copy()
    out["selection_signature"] = out["selected_primitive_variant_id"].fillna("").astype(str)
    out["memory_changed_selection"] = False
    out["exploration_changed_selection"] = False
    baseline = out[out["policy_id"].astype(str) == "no_memory_baseline"]
    baseline_map = {
        (str(row["library_size_case_id"]), int(row["outer_case_index"])): str(row["selection_signature"])
        for row in baseline.to_dict(orient="records")
    }
    memory_signatures = {
        (str(row["library_size_case_id"]), int(row["outer_case_index"]), int(row["history_length"])): str(row["selection_signature"])
        for row in out[out["policy_id"].astype(str).str.startswith(MEMORY_POLICY_PREFIX)].to_dict(orient="records")
    }
    for index, row in out.iterrows():
        policy_id = str(row["policy_id"])
        key = (str(row["library_size_case_id"]), int(row["outer_case_index"]))
        signature = str(row["selection_signature"])
        if policy_id.startswith(MEMORY_POLICY_PREFIX) or policy_id == EMPTY_FROZEN_PRIOR_BASELINE_ID:
            out.at[index, "memory_changed_selection"] = signature != baseline_map.get(key, signature)
        if policy_id.startswith(SAFE_EXPLORE_POLICY_PREFIX):
            memory_key = (str(row["library_size_case_id"]), int(row["outer_case_index"]), int(row["history_length"]))
            out.at[index, "exploration_changed_selection"] = signature != memory_signatures.get(memory_key, signature)
    return out


def _pass_fail_summary(
    *,
    protocol: ValidationProtocol,
    max_primitives_per_launch: int,
    max_episode_time_s: float,
    final_schedule: list[dict[str, object]],
    history_schedule: list[dict[str, object]],
    episode_rows: list[dict[str, object]],
    pairing_rows: list[dict[str, object]],
    no_variation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    final_rows = [row for row in episode_rows if str(row.get("launch_role", "")) == "final_heldout"]
    rows = [
        _gate_row("final_heldout_launch_count", len(final_schedule) == protocol.expected_final_heldout_launches, len(final_schedule), protocol.expected_final_heldout_launches),
        _gate_row("history_launch_count", len(history_schedule) == protocol.expected_history_launches, len(history_schedule), protocol.expected_history_launches),
        _gate_row("library_size_case_count", set(row["library_size_case_id"] for row in final_schedule) == set(LIBRARY_SIZE_CASE_IDS), len(set(row["library_size_case_id"] for row in final_schedule)), len(LIBRARY_SIZE_CASE_IDS)),
        _gate_row("policy_history_condition_count", set(row["policy_id"] for row in final_schedule) == set(POLICY_HISTORY_CONDITIONS), len(set(row["policy_id"] for row in final_schedule)), len(POLICY_HISTORY_CONDITIONS)),
        _gate_row("pairing_audit", all(bool(row["pairing_passed"]) for row in pairing_rows), sum(bool(row["pairing_passed"]) for row in pairing_rows), len(pairing_rows)),
        _gate_row("primitive_count_cap_disabled_for_full_validation", int(max_primitives_per_launch) <= 0, int(max_primitives_per_launch), "0_or_negative_disabled"),
        _gate_row("max_episode_time_budget_positive", float(max_episode_time_s) >= float(PRIMITIVE_FINITE_HORIZON_S), float(max_episode_time_s), f">={PRIMITIVE_FINITE_HORIZON_S}"),
    ]
    if no_variation_rows:
        rows.append(_gate_row("no_glider_latency_variation_audit", all(bool(row["variation_audit_passed"]) for row in no_variation_rows), sum(bool(row["variation_audit_passed"]) for row in no_variation_rows), len(no_variation_rows)))
    if protocol.reduced_diagnostic:
        rows.append(
            _gate_row(
                "reduced_diagnostic_not_target_R10",
                False,
                "reduced_diagnostic_not_target_R10",
                "full_R10_validation_required",
            )
        )
    if final_rows:
        selected_primitives = _selected_set(final_rows, "selected_primitive_id")
        selected_variants = _selected_set(final_rows, "selected_primitive_variant_id")
        sequence_compliant_count = sum(
            1 for row in episode_rows if _truthy(row.get("launch_inflight_recovery_sequence_compliant", False))
        )
        rows.extend(
            [
                _gate_row(
                    "launch_inflight_recovery_sequence_policy",
                    sequence_compliant_count == len(episode_rows),
                    sequence_compliant_count,
                    len(episode_rows),
                ),
                _gate_row(
                    "hard_failure_rate_within_stage_profile",
                    _mean_bool(final_rows, "hard_failure") <= float(protocol.max_hard_failure_rate),
                    _mean_bool(final_rows, "hard_failure"),
                    protocol.max_hard_failure_rate,
                ),
                _gate_row("floor_or_ceiling_violation_rate_zero", _mean_bool(final_rows, "floor_or_ceiling_violation") == 0.0, _mean_bool(final_rows, "floor_or_ceiling_violation"), 0.0),
                _gate_row(
                    "no_viable_primitive_rate_within_stage_profile",
                    _mean_bool(final_rows, "no_viable_primitive") <= float(protocol.max_no_viable_rate),
                    _mean_bool(final_rows, "no_viable_primitive"),
                    protocol.max_no_viable_rate,
                ),
                _gate_row(
                    "safe_success_rate_within_stage_profile",
                    _mean_bool(final_rows, "safe_success") >= float(protocol.min_safe_success_rate),
                    _mean_bool(final_rows, "safe_success"),
                    protocol.min_safe_success_rate,
                ),
                _gate_row(
                    "terminal_or_lift_capture_within_stage_profile",
                    max(_mean_bool(final_rows, "terminal_useful"), _mean_bool(final_rows, "lift_capture"))
                    >= float(protocol.min_terminal_or_lift_capture_rate),
                    max(_mean_bool(final_rows, "terminal_useful"), _mean_bool(final_rows, "lift_capture")),
                    protocol.min_terminal_or_lift_capture_rate,
                ),
                _gate_row("selected_primitive_family_count_ge_5", len(selected_primitives) >= 5, len(selected_primitives), 5),
                _gate_row("selected_variant_count_ge_10", len(selected_variants) >= 10, len(selected_variants), 10),
            ]
        )
        if protocol.min_full_safe_success_rate is not None:
            rows.append(
                _gate_row(
                    "full_safe_success_rate_within_stage_profile",
                    _mean_bool(final_rows, "full_safe_success") >= float(protocol.min_full_safe_success_rate),
                    _mean_bool(final_rows, "full_safe_success"),
                    protocol.min_full_safe_success_rate,
                )
            )
    else:
        rows.append(_gate_row("final_rollout_rows_present", False, 0, protocol.expected_final_heldout_launches))
    return rows


def _gate_row(gate_id: str, passed: bool, observed: object, required: object) -> dict[str, object]:
    return {"gate_id": gate_id, "passed": bool(passed), "observed": observed, "required": required}


def _overall_pass(rows: list[dict[str, object]]) -> bool:
    return bool(rows) and all(bool(row.get("passed", False)) for row in rows)


def _mean_bool(rows: list[dict[str, object]], column: str) -> float:
    return float(sum(1 for row in rows if _truthy(row.get(column, False))) / max(1, len(rows)))


def _selected_set(rows: list[dict[str, object]], column: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        for item in str(row.get(column, "")).split(";"):
            value = item.strip()
            if value and value.lower() != "nan":
                values.add(value)
    return values


def _pairing_audit_rows(final_schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    frame = pd.DataFrame(final_schedule)
    if frame.empty:
        return [{"audit_id": "empty_final_schedule", "pairing_passed": False, "detail": "no final schedule rows"}]
    for key, group in frame.groupby("outer_case_index"):
        rows.append(
            {
                "outer_case_index": int(key),
                "pairing_passed": bool(
                    group["common_final_launch_key"].nunique() == 1
                    and group["launch_state_seed"].nunique() == 1
                    and group["environment_seed"].nunique() == 1
                ),
                "row_count": int(len(group)),
                "library_case_count": int(group["library_size_case_id"].nunique()),
                "policy_count": int(group["policy_id"].nunique()),
            }
        )
    return rows


def _no_variation_audit_rows(final_schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for key, group in pd.DataFrame(final_schedule).groupby("outer_case_index"):
        rows.append(
            {
                "outer_case_index": int(key),
                "variation_audit_passed": True,
                "glider_model_fixed": True,
                "latency_model_fixed": True,
                "actuator_model_fixed": True,
                "mass_cg_inertia_surface_calibration_not_varied": True,
                "row_count": int(len(group)),
            }
        )
    return rows


def _environment_block_summary(protocol: ValidationProtocol) -> pd.DataFrame:
    return pd.DataFrame([asdict(block) for block in protocol.blocks])


def _active_fan_count_schedule_audit_rows(outer_cases: list[dict[str, object]]) -> list[dict[str, object]]:
    if not outer_cases:
        return [
            {
                "environment_block_id": ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
                "scheduled_active_fan_count": "",
                "outer_case_count": 0,
                "policy": "balanced_1_2_3_4_for_active_fan_number_variation",
                "audit_passed": False,
            }
        ]
    frame = pd.DataFrame(outer_cases)
    if "scheduled_active_fan_count" not in frame.columns:
        return []
    frame["scheduled_active_fan_count"] = pd.to_numeric(
        frame["scheduled_active_fan_count"],
        errors="coerce",
    )
    rows: list[dict[str, object]] = []
    for block_id in R10_ACTIVE_FAN_COUNT_VARIATION_BLOCK_IDS:
        active = frame[frame["environment_block_id"].eq(block_id)].copy()
        if active.empty:
            continue
        counts = active.groupby("scheduled_active_fan_count", dropna=False).size()
        expected_counts = {
            value: sum(
                1
                for index in range(int(len(active)))
                if R10_ACTIVE_FAN_COUNT_SEQUENCE[index % len(R10_ACTIVE_FAN_COUNT_SEQUENCE)] == value
            )
            for value in R10_ACTIVE_FAN_COUNT_SEQUENCE
        }
        policy = str(
            active["active_fan_count_policy"].iloc[0]
            if "active_fan_count_policy" in active.columns and not active.empty
            else "balanced_1_2_3_4"
        )
        for active_count, row_count in counts.items():
            count_value = "" if pd.isna(active_count) else int(active_count)
            expected_count = expected_counts.get(count_value, 0) if isinstance(count_value, int) else 0
            rows.append(
                {
                    "environment_block_id": block_id,
                    "scheduled_active_fan_count": count_value,
                    "outer_case_count": int(row_count),
                    "expected_outer_case_count": expected_count,
                    "expected_active_fan_counts": ";".join(str(value) for value in R10_ACTIVE_FAN_COUNT_SEQUENCE),
                    "policy": policy,
                    "audit_passed": bool(
                        count_value in set(R10_ACTIVE_FAN_COUNT_SEQUENCE)
                        and int(row_count) == expected_count
                    ),
                }
            )
    return rows


def _write_governor_tuning_outputs(
    run_root: Path,
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
    pass_summary: list[dict[str, object]],
    episode_rows: list[dict[str, object]],
) -> None:
    final = pd.DataFrame([row for row in episode_rows if str(row.get("launch_role", "")) == "final_heldout"])
    metrics = {
        "final_launch_count": int(len(final)),
        "hard_failure_rate": _mean_bool(final.to_dict(orient="records"), "hard_failure") if not final.empty else 1.0,
        "no_viable_primitive_rate": _mean_bool(final.to_dict(orient="records"), "no_viable_primitive") if not final.empty else 1.0,
        "safe_success_rate": _mean_bool(final.to_dict(orient="records"), "safe_success") if not final.empty else 0.0,
        "full_safe_success_rate": _mean_bool(final.to_dict(orient="records"), "full_safe_success") if not final.empty else 0.0,
        "terminal_or_lift_capture_rate": max(
            _mean_bool(final.to_dict(orient="records"), "terminal_useful") if not final.empty else 0.0,
            _mean_bool(final.to_dict(orient="records"), "lift_capture") if not final.empty else 0.0,
        ),
    }
    base_config = config.governor_config or DEFAULT_GOVERNOR_CONFIG
    selected_config, tuning_rows = _tuned_governor_config_from_metrics(
        base_config=base_config,
        metrics=metrics,
        protocol=protocol,
    )
    metrics["input_governor_config_id"] = base_config.config_id
    metrics["governor_config_id"] = selected_config.config_id
    if protocol.stage_id == "R9":
        output_name = "initial_governor_config_for_r10.json"
        status = "selected_for_r10_initialisation" if _overall_pass(pass_summary) else "not_selected_r9_preflight_gate_failed"
        selection_policy = "internal_r9_preflight_initialises_r10_only_after_reduced_gate_pass"
        target_stage = "R10"
        thesis_status = R9_THESIS_REPORTING_STATUS
    else:
        output_name = "frozen_governor_config_for_r11.json"
        status = "selected_for_r11" if _overall_pass(pass_summary) else "not_selected_r10_gate_failed"
        selection_policy = "robust_first_freeze_only_after_full_r10_pass_gate"
        target_stage = "R11"
        thesis_status = "changed_case_governor_tuning_not_final_claim_gate"
    payload = {
        "manifest_version": GOVERNOR_TUNING_HANDOFF_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "stage_id": protocol.stage_id,
        "target_stage": target_stage,
        "selection_policy": selection_policy,
        "source_run_root": run_root.as_posix(),
        "governor_config": governor_config_to_row(selected_config),
        "selection_metrics": metrics,
        "tuning_decisions": tuning_rows,
        "controller_mutation_allowed": False,
        "primitive_retuning_allowed": False,
        "thesis_facing_workflow": THESIS_FACING_WORKFLOW,
        "thesis_reporting_status": thesis_status,
        "claim_status": "simulation_only_governor_tuning_handoff_not_memory_improvement_claim",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / output_name, payload)
    _write_csv(
        run_root / "metrics" / "governor_config_selection.csv",
        pd.DataFrame([metrics | {"status": status, "target_stage": target_stage, "selection_policy": selection_policy}]),
    )
    _write_csv(run_root / "metrics" / "governor_config_tuning_decisions.csv", pd.DataFrame(tuning_rows))


def _tuned_governor_config_from_metrics(
    *,
    base_config: GovernorConfig,
    metrics: dict[str, object],
    protocol: ValidationProtocol,
) -> tuple[GovernorConfig, list[dict[str, object]]]:
    values = asdict(base_config)
    decisions: list[dict[str, object]] = []

    def update(key: str, value: float, reason: str) -> None:
        old = float(values[key])
        new = float(value)
        if abs(old - new) <= 1e-12:
            return
        values[key] = new
        decisions.append(
            {
                "parameter": key,
                "old_value": old,
                "new_value": new,
                "reason": reason,
                "stage_id": protocol.stage_id,
            }
        )

    hard_failure_rate = _float_metric(metrics.get("hard_failure_rate", 1.0), default=1.0)
    no_viable_rate = _float_metric(metrics.get("no_viable_primitive_rate", 1.0), default=1.0)
    safe_success_rate = _float_metric(metrics.get("safe_success_rate", 0.0), default=0.0)
    terminal_or_lift_rate = _float_metric(metrics.get("terminal_or_lift_capture_rate", 0.0), default=0.0)

    if hard_failure_rate > float(protocol.max_hard_failure_rate):
        update(
            "maximum_hard_failure_risk",
            max(0.45, float(values["maximum_hard_failure_risk"]) - 0.10),
            "hard_failure_rate_above_stage_profile_tighten_admission",
        )
        update(
            "hard_failure_weight",
            min(-0.20, float(values["hard_failure_weight"]) * 1.25),
            "hard_failure_rate_above_stage_profile_penalise_risk_more",
        )
        update(
            "terminal_hard_failure_weight",
            min(-0.20, float(values["terminal_hard_failure_weight"]) * 1.25),
            "hard_failure_rate_above_stage_profile_penalise_terminal_risk_more",
        )
        update(
            "exploration_bonus_weight",
            max(0.0, float(values["exploration_bonus_weight"]) * 0.50),
            "hard_failure_rate_above_stage_profile_reduce_exploration_bonus",
        )
    elif no_viable_rate > float(protocol.max_no_viable_rate):
        update(
            "maximum_hard_failure_risk",
            min(0.90, float(values["maximum_hard_failure_risk"]) + 0.05),
            "no_viable_rate_above_stage_profile_relax_admission_without_removing_safety_gate",
        )
        update(
            "continuation_weight",
            float(values["continuation_weight"]) + 0.05,
            "no_viable_rate_above_stage_profile_prefer_continuation_candidates",
        )
        update(
            "terminal_continuation_weight",
            float(values["terminal_continuation_weight"]) + 0.05,
            "no_viable_rate_above_stage_profile_keep_terminal_mode_from_dead_ending",
        )

    if terminal_or_lift_rate < float(protocol.min_terminal_or_lift_capture_rate):
        update(
            "updraft_gain_weight",
            float(values["updraft_gain_weight"]) + 0.01,
            "terminal_or_lift_capture_below_stage_profile_increase_updraft_gain_preference",
        )
        update(
            "terminal_updraft_gain_weight",
            float(values["terminal_updraft_gain_weight"]) + 0.01,
            "terminal_or_lift_capture_below_stage_profile_increase_terminal_updraft_gain_preference",
        )
        update(
            "lift_dwell_weight",
            float(values["lift_dwell_weight"]) + 0.005,
            "terminal_or_lift_capture_below_stage_profile_increase_lift_dwell_preference",
        )

    if safe_success_rate < float(protocol.min_safe_success_rate) and hard_failure_rate <= float(protocol.max_hard_failure_rate):
        update(
            "belief_weight",
            min(0.20, float(values["belief_weight"]) + 0.01),
            "safe_success_below_stage_profile_increase_memory_residual_sensitivity",
        )

    values["minimum_wall_margin_m"] = float(base_config.minimum_wall_margin_m)
    values["config_id"] = f"v53_{protocol.stage_id.lower()}_tuned_viability_governor"
    if not decisions:
        decisions.append(
            {
                "parameter": "none",
                "old_value": "",
                "new_value": "",
                "reason": "base_governor_config_retained_metrics_within_stage_profile",
                "stage_id": protocol.stage_id,
            }
        )
    return GovernorConfig(**values), decisions


def _float_metric(value: object, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _write_manifest(
    *,
    run_root: Path,
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
    status: str,
    pass_summary: list[dict[str, object]],
    final_schedule: list[dict[str, object]],
    history_schedule: list[dict[str, object]],
    duration_s: float = 0.0,
) -> None:
    payload = {
        "manifest_version": protocol.manifest_version,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "stage_id": protocol.stage_id,
        "library_root": Path(config.library_root).as_posix(),
        "outcome_root": Path(config.outcome_root).as_posix(),
        "source_w2_root": "" if config.source_w2_root is None else Path(config.source_w2_root).as_posix(),
        "history_lengths": list(HISTORY_LENGTHS),
        "policy_history_conditions": list(POLICY_HISTORY_CONDITIONS),
        "policy_history_condition_count": len(POLICY_HISTORY_CONDITIONS),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "outer_cases_per_condition": int(protocol.outer_cases_per_condition),
        "expected_final_heldout_launches": int(protocol.expected_final_heldout_launches),
        "actual_final_heldout_launches": int(len(final_schedule)),
        "expected_history_launches": int(protocol.expected_history_launches),
        "actual_history_launches": int(len(history_schedule)),
        "primitive_timing_contract": primitive_timing_contract_row(),
        "validation_protocol": protocol.validation_evidence_level,
        "validation_gate_profile": protocol.gate_profile,
        "max_hard_failure_rate": float(protocol.max_hard_failure_rate),
        "max_no_viable_rate": float(protocol.max_no_viable_rate),
        "min_safe_success_rate": float(protocol.min_safe_success_rate),
        "min_full_safe_success_rate": (
            None if protocol.min_full_safe_success_rate is None else float(protocol.min_full_safe_success_rate)
        ),
        "min_terminal_or_lift_capture_rate": float(protocol.min_terminal_or_lift_capture_rate),
        "dry_run_schedule": bool(config.dry_run_schedule),
        "storage_format": resolve_storage_format(config.storage_format),
        "compression_level": int(config.compression_level),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "requested_workers": int(config.workers),
        "max_workers": None if config.max_workers is None else int(config.max_workers),
        "selected_workers": int(_selected_worker_count(config)),
        "worker_backend": str(config.worker_backend),
        "parallel_execution_policy": "parallelise_across_independent_final_schedule_rows_history_sequential_inside_worker_parent_writes_partitions",
        "thesis_facing_workflow": THESIS_FACING_WORKFLOW,
        "governor_config_override_active": config.governor_config is not None,
        "governor_config": governor_config_to_row(config.governor_config or DEFAULT_GOVERNOR_CONFIG),
        "r9_initial_governor_config_for_r10": (
            (run_root / "manifests" / "initial_governor_config_for_r10.json").as_posix()
            if protocol.stage_id == "R9"
            else ""
        ),
        "r10_frozen_governor_config_for_r11": (
            (run_root / "manifests" / "frozen_governor_config_for_r11.json").as_posix()
            if protocol.stage_id == "R10"
            else ""
        ),
        "thesis_reporting_status": R9_THESIS_REPORTING_STATUS
        if protocol.stage_id == "R9"
        else "claim_bearing_stage_only_if_final_gate_passes",
        "max_primitives_per_launch": int(config.max_primitives_per_launch),
        "primitive_count_cap_status": "disabled" if int(config.max_primitives_per_launch) <= 0 else "diagnostic_cap_enabled",
        "max_episode_time_s": float(config.max_episode_time_s),
        "max_episode_steps_from_time_budget": int(
            math.ceil(float(config.max_episode_time_s) / float(PRIMITIVE_FINITE_HORIZON_S))
        ),
        "candidate_score_log_policy": "compact_topk_selected_family_rejection_summary",
        "candidate_score_top_k_per_decision": int(CANDIDATE_SCORE_TOP_K_PER_DECISION),
        "full_candidate_score_log_default": False,
        "history_launch_plot_evidence_retained": True,
        "history_launch_retained_tables": [
            "episode_summary",
            "primitive_execution_log",
            "selector_decision_log",
            "memory_residual_update_log",
            "belief_snapshot_log",
        ],
        "final_launch_plot_evidence_retained": True,
        "smoke_outer_cases_per_block": int(config.smoke_outer_cases_per_block),
        "smoke_run_not_full_gate_evidence": bool(int(config.smoke_outer_cases_per_block) > 0),
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
        "first_primitive_start_state_family": FIRST_PRIMITIVE_START_FAMILY,
        "post_launch_start_state_family": POST_LAUNCH_START_FAMILY,
        "boundary_recovery_start_state_family": BOUNDARY_RECOVERY_START_FAMILY,
        "terminal_safe_exit_start_state_family": TERMINAL_SAFE_EXIT_START_FAMILY,
        "first_primitive_required_entry_class": "launch_gate",
        "post_launch_required_entry_class": "inflight_stable",
        "boundary_recovery_required_entry_class": "boundary_near_or_recoverable_degraded",
        "terminal_safe_exit_required_entry_class": "recoverable_degraded",
        "transition_contract": transition_contract_row(),
        "active_governor_path": "transition_viability_governor_v1",
        "boundary_near_status": "route_state_not_automatic_failure",
        "changed_case_active_fan_count_policy": "balanced_1_2_3_4_for_active_fan_number_variation"
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else "not_applicable",
        "changed_case_active_fan_count_sequence": list(R10_ACTIVE_FAN_COUNT_SEQUENCE)
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else [],
        "changed_case_arena_wide_fan_position_block_id": BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else "not_applicable",
        "changed_case_arena_wide_fan_position_xy_bounds_m": R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else (),
        "legacy_recovery_threshold_alias_status": "superseded_by_transition_labels_contract",
        "launch_score_version": LAUNCH_SCORE_VERSION,
        "launch_score_target_episode_time_s": float(SCORING_TARGET_EPISODE_TIME_S),
        "launch_score_gravity_m_s2": float(SPECIFIC_ENERGY_GRAVITY_M_S2),
        "duration_s": float(duration_s),
        "pass_gate": _overall_pass(pass_summary),
        "claim_status": "simulation_only_repeated_launch_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / protocol.manifest_name, payload)


def _write_blocked_outputs(
    run_root: Path,
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
    blocked_reason: str,
) -> None:
    manifest = {
        "manifest_version": protocol.manifest_version,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "stage_id": protocol.stage_id,
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "blocked_reason": blocked_reason,
        "pass_gate": False,
        "claim_status": "simulation_only_blocked_repeated_launch_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / protocol.manifest_name, manifest)
    _write_csv(run_root / "metrics" / "pass_fail_gate_summary.csv", pd.DataFrame([_gate_row("blocked_before_execution", False, blocked_reason, "unblocked_inputs")]))
    _write_file_size_audit(run_root)
    _write_report(run_root=run_root, protocol=protocol, status="blocked", pass_summary=[{"gate_id": "blocked", "passed": False, "observed": blocked_reason, "required": "unblocked"}])


def _write_report(*, run_root: Path, protocol: ValidationProtocol, status: str, pass_summary: list[dict[str, object]]) -> None:
    lines = [
        f"# {protocol.stage_id} Repeated-Launch Validation",
        "",
        f"- Status: `{status}`",
        f"- Pass gate: `{_overall_pass(pass_summary)}`",
        f"- Expected final held-out launches: `{protocol.expected_final_heldout_launches}`",
        f"- Expected history launches: `{protocol.expected_history_launches}`",
        f"- Gate profile: `{protocol.gate_profile}`",
        f"- Safety thresholds: hard failure <= `{protocol.max_hard_failure_rate}`, no-viable <= `{protocol.max_no_viable_rate}`, safe success >= `{protocol.min_safe_success_rate}`, full safe success >= `{protocol.min_full_safe_success_rate}`, terminal/lift >= `{protocol.min_terminal_or_lift_capture_rate}`.",
        f"- Launch sequence policy: `{LAUNCH_SEQUENCE_POLICY_ID}`",
        "- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, updraft gain, flight time, and residual memory.",
        "- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.",
        f"- Launch score: `{LAUNCH_SCORE_VERSION}`; rewards safe valid flight time and updraft-gain proxy, while net/gross energy drift remains audit-only.",
        "- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.",
        "",
        "Gate summary:",
        "",
    ]
    lines.extend(f"- `{row['gate_id']}`: `{row['passed']}` observed `{row['observed']}` required `{row['required']}`" for row in pass_summary)
    filesystem_path(run_root / "reports" / protocol.report_name).write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        size_mb = float(path.stat().st_size) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": path.relative_to(root_fs).as_posix(),
                "byte_count": int(path.stat().st_size),
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reduced internal R9 fixed-case repeated-launch preflight.")
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--source-w2-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--storage-format", choices=("auto", "parquet", "csv_gz", "csv"), default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=20_000)
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument(
        "--max-primitives-per-launch",
        type=int,
        default=0,
        help="Optional diagnostic primitive-count cap. Use 0 to disable the cap for full validation.",
    )
    parser.add_argument("--max-episode-time-s", type=float, default=R9_PREFLIGHT_MAX_EPISODE_TIME_S)
    parser.add_argument("--smoke-outer-cases-per-block", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="process")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_repeated_launch_learning_curve(
        RepeatedLaunchValidationConfig(
            library_root=args.library_root,
            outcome_root=args.outcome_root,
            source_w2_root=args.source_w2_root,
            output_root=args.output_root,
            run_id=args.run_id,
            seed=args.seed,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            candidate_chunk_size=args.candidate_chunk_size,
            dry_run_schedule=args.dry_run_schedule,
            max_primitives_per_launch=args.max_primitives_per_launch,
            max_episode_time_s=args.max_episode_time_s,
            smoke_outer_cases_per_block=args.smoke_outer_cases_per_block,
            workers=args.workers,
            max_workers=args.max_workers,
            worker_backend=args.worker_backend,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "blocked", "dry_run_schedule", "smoke_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
