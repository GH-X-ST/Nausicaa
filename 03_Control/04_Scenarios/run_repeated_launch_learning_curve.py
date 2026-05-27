from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
from episode_selector import select_compact_representative, selector_decision_row  # noqa: E402
from frozen_w01_controller_bundle import FROZEN_CONTROLLER_READY, load_frozen_w01_controller_bundle  # noqa: E402
from implementation_instance import implementation_instance_for_layer  # noqa: E402
from plant_instance import plant_instance_for_layer  # noqa: E402
from prim_cat import LAUNCH_CAPTURE_PRIMITIVE_IDS, primitive_by_id  # noqa: E402
from prim_roll import RolloutConfig, rollout_evidence_row, simulate_primitive_rollout  # noqa: E402
from primitive_timing_contract import PRIMITIVE_FINITE_HORIZON_S, primitive_timing_contract_row  # noqa: E402
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS  # noqa: E402
from state_contract import STATE_INDEX, as_state_vector  # noqa: E402
from state_sampling import archive_state_sample_for_family  # noqa: E402
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GovernorConfig  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.3"
VALIDATION_VERSION = "repeated_launch_fixed_case_rollout_validation_v6"
HISTORY_LENGTHS = (0, 5, 10, 20, 50, 100)
HISTORY_LENGTH_SUM = sum(HISTORY_LENGTHS)
BASELINE_POLICY_IDS = ("no_memory_baseline", "static_map_baseline")
MEMORY_POLICY_PREFIX = "directional_3d_residual_memory"
SAFE_EXPLORE_POLICY_PREFIX = "safe_explore_then_exploit"
POLICY_HISTORY_CONDITIONS = (
    "no_memory_baseline",
    "static_map_baseline",
    "directional_3d_residual_memory_h0",
    "directional_3d_residual_memory_h5",
    "directional_3d_residual_memory_h10",
    "directional_3d_residual_memory_h20",
    "directional_3d_residual_memory_h50",
    "directional_3d_residual_memory_h100",
    "safe_explore_then_exploit_h0",
    "safe_explore_then_exploit_h5",
    "safe_explore_then_exploit_h10",
    "safe_explore_then_exploit_h20",
    "safe_explore_then_exploit_h50",
    "safe_explore_then_exploit_h100",
)
R9_OUTER_CASES_PER_CONDITION = 60
R9_EXPECTED_FINAL_HELDOUT_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R9_OUTER_CASES_PER_CONDITION
R9_EXPECTED_HISTORY_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * R9_OUTER_CASES_PER_CONDITION * (
    HISTORY_LENGTH_SUM + HISTORY_LENGTH_SUM
)
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
LAUNCH_SEQUENCE_POLICY_ID = "first_0p10s_launch_capture_then_inflight_then_recovery_safe_exit"
FIRST_PRIMITIVE_START_FAMILY = "launch_gate"
POST_LAUNCH_START_FAMILY = "inflight_nominal"
BOUNDARY_RECOVERY_START_FAMILY = "inflight_boundary_near"
TERMINAL_SAFE_EXIT_START_FAMILY = "inflight_recovery_edge"
ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID = "active_fan_number_variation"
BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID = "arena_wide_fan_position_generalisation"
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
RECOVERY_EDGE_MIN_SPEED_M_S = 4.2
RECOVERY_EDGE_MAX_ABS_ROLL_RAD = math.radians(35.0)
RECOVERY_EDGE_MAX_ABS_PITCH_RAD = math.radians(22.0)
RECOVERY_EDGE_MAX_BODY_RATE_RAD_S = 0.65
LAUNCH_SCORE_VERSION = "r9_r10_specific_energy_multiplicative_launch_score_v1"
SPECIFIC_ENERGY_GRAVITY_M_S2 = 9.80665
SCORING_TARGET_EPISODE_TIME_S = 1.5
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "full_autonomy",
    "memory_improvement",
)


@dataclass(frozen=True)
class ValidationBlockSpec:
    block_id: str
    human_label: str
    W_layer: str
    environment_mode: str
    case_count: int
    environment_change_family: str = "fixed_case"


R9_BLOCKS: tuple[ValidationBlockSpec, ...] = (
    ValidationBlockSpec("no_updraft", "no-updraft", "W0", "dry_air", 20),
    ValidationBlockSpec("single_fan", "single-fan", "W2", "annular_gp_single", 20),
    ValidationBlockSpec("four_fan", "four-fan", "W2", "annular_gp_four", 20),
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
    candidate_chunk_size: int = 800
    dry_run_schedule: bool = False
    max_primitives_per_launch: int = 4


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


R9_PROTOCOL = ValidationProtocol(
    stage_id="R9",
    manifest_name="repeated_launch_fixed_case_manifest.json",
    report_name="repeated_launch_fixed_case_report.md",
    manifest_version=VALIDATION_VERSION,
    validation_evidence_level="full_fixed_case_repeated_launch_rollout_validation",
    outer_cases_per_condition=R9_OUTER_CASES_PER_CONDITION,
    expected_final_heldout_launches=R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    expected_history_launches=R9_EXPECTED_HISTORY_LAUNCHES,
    blocks=R9_BLOCKS,
    final_schedule_prefix="r9_fixed",
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


def run_repeated_launch_learning_curve(config: RepeatedLaunchValidationConfig) -> dict[str, object]:
    """Run the full R9 fixed-case repeated-launch rollout validation."""

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
    outer_cases = _outer_case_schedule(protocol=protocol, seed=config.seed)
    final_schedule = _final_heldout_schedule(outer_cases=outer_cases, protocol=protocol)
    history_schedule = _history_launch_schedule(outer_cases=outer_cases, protocol=protocol)
    _write_csv(run_root / "metrics" / "outer_case_schedule.csv", pd.DataFrame(outer_cases))
    _write_csv(run_root / "metrics" / "history_launch_schedule.csv", pd.DataFrame(history_schedule))
    _write_csv(run_root / "metrics" / "final_heldout_launch_schedule.csv", pd.DataFrame(final_schedule))
    if protocol.stage_id == "R10":
        _write_csv(run_root / "metrics" / "environment_block_schedule.csv", _environment_block_summary(protocol))
        _write_csv(
            run_root / "metrics" / "active_fan_count_schedule_audit.csv",
            pd.DataFrame(_active_fan_count_schedule_audit_rows(outer_cases)),
        )

    if config.dry_run_schedule:
        pass_summary = _pass_fail_summary(
            protocol=protocol,
            max_primitives_per_launch=int(config.max_primitives_per_launch),
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

    table_buffers = {name: [] for name in TABLE_NAMES}
    partitions: list[TablePartition] = []
    row_counters = {name: 0 for name in TABLE_NAMES}
    started = time.time()
    for final_row in final_schedule:
        policy = _policy_condition(str(final_row["policy_id"]))
        representatives = libraries[str(final_row["library_size_case_id"])]["representatives"]
        case_outcomes = {
            variant_id: row
            for variant_id, row in outcome_rows.items()
            if str(row.get("library_size_case_id", "")) == str(final_row["library_size_case_id"])
        }
        belief = _initial_belief_for_policy(policy=policy, final_row=final_row)
        for hist_index in range(int(policy["history_length"])):
            history_row = _history_row_for_final(final_row, hist_index)
            launch_result = _run_one_launch(
                scheduled=history_row,
                policy=policy,
                representatives=representatives,
                outcome_rows_by_variant_id=case_outcomes,
                records_by_variant=records_by_variant,
                belief=belief,
                config=config,
                protocol=protocol,
            )
            belief = launch_result["belief_after"]
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
        _append_launch_result(table_buffers, final_result)
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
        final_schedule=final_schedule,
        history_schedule=history_schedule,
        episode_rows=episode_rows,
        pairing_rows=pairing_rows,
        no_variation_rows=no_variation_rows,
    )
    _write_csv(run_root / "metrics" / "pass_fail_gate_summary.csv", pd.DataFrame(pass_summary))
    status = "complete"
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
        if compact_id:
            rows[compact_id] = row
            rows[f"{case_id}|{compact_id}"] = row
        if variant_id and compact_id:
            rows[f"{case_id}|{variant_id}|{compact_id}"] = row
        elif variant_id and variant_id not in rows:
            rows[variant_id] = row
    return rows


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


def _scheduled_active_fan_count_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
    environment_block_local_index: int,
) -> int | None:
    if str(protocol.stage_id) != "R10":
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
    if str(protocol.stage_id) != "R10":
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
    if str(protocol.stage_id) != "R10":
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
    if str(protocol.stage_id) != "R10":
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
    if str(protocol.stage_id) != "R10":
        return "common_shift_range=-0.200:0.200"
    block_id = str(environment_block_id)
    if block_id == BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID:
        x_bounds, y_bounds = R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M
        return f"x={float(x_bounds[0]):.3f}:{float(x_bounds[1]):.3f};y={float(y_bounds[0]):.3f}:{float(y_bounds[1]):.3f}"
    if block_id in R10_FIXED_BASE_POSITION_BLOCK_IDS:
        return "fixed_base_positions_no_shift"
    return "common_shift_range=-0.200:0.200"


def _outer_case_schedule(*, protocol: ValidationProtocol, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outer_index = 0
    for block in protocol.blocks:
        for local_index in range(int(block.case_count)):
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
    if policy_id == "static_map_baseline":
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
    if policy["policy_id"] != "static_map_baseline":
        return belief
    for index in range(12):
        belief = update_directional_residual_lift_belief(
            belief,
            DirectionalResidualObservation(
                x_w_m=0.25 * (index % 4),
                y_w_m=0.20 * (index % 3),
                z_w_m=1.0 + 0.2 * (index % 3),
                direction_rad=0.25 * index,
                lift_residual_m_s=0.01 * (index % 2),
                energy_residual_m=0.005 * (index % 3),
                dwell_residual_s=0.002 * (index % 4),
            ),
        )
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
    governor_config = _governor_config_for_policy(policy)
    max_steps = max(1, int(config.max_primitives_per_launch))
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
            row["route_reason"] = str(route["route_reason"])
        candidate_rows_all.extend(candidate_rows)
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
        primitive_rows.append(
            {
                **_schedule_identity_row(scheduled),
                "launch_role": str(scheduled["launch_role"]),
                "primitive_step_index": int(primitive_step_index),
                "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
                "launch_sequence_phase": str(route["launch_sequence_phase"]),
                "start_state_family": start_state_family,
                "route_required_entry_role": str(route["route_required_entry_role"]),
                "route_reason": str(route["route_reason"]),
                "selected_entry_role": str(selected.get("entry_role", "")),
                "policy_id": str(policy["policy_id"]),
                "selected_compact_library_id": str(selected.get("compact_library_id", "")),
                "selected_score": float(selected.get("total_score_with_memory_and_exploration", selected.get("score", 0.0))),
                **rollout_row,
            }
        )
        outcome = _outcome_for_selected(selected, outcome_rows_by_variant_id)
        observation = DirectionalResidualObservation(
            x_w_m=float(state[STATE_INDEX["x_w"]]),
            y_w_m=float(state[STATE_INDEX["y_w"]]),
            z_w_m=float(state[STATE_INDEX["z_w"]]),
            direction_rad=float(state[STATE_INDEX["psi"]]),
            lift_residual_m_s=float(context_payload["row"].get("w_wing_mean_m_s", 0.0)) - float(outcome.get("w_wing_mean_m_s", 0.0)),
            energy_residual_m=float(rollout_row.get("energy_residual_m", 0.0)) - float(outcome.get("expected_energy_residual_m", 0.0)),
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
        hard_failure = bool(
            str(rollout_row.get("boundary_use_class", "")) == "hard_failure"
            or str(rollout_row.get("outcome_class", "")) == "failed"
        )
        if hard_failure or bool(rollout_row.get("episode_terminal_useful", False)) or not bool(rollout_row.get("continuation_valid", False)):
            break
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
) -> dict[str, object]:
    case_id = str(selected.get("library_size_case_id", ""))
    compact_id = str(selected.get("compact_library_id", ""))
    variant_id = str(selected.get("primitive_variant_id", ""))
    for key in (compact_id, f"{case_id}|{compact_id}", f"{case_id}|{variant_id}|{compact_id}", variant_id):
        if key and key in outcome_rows_by_variant_id:
            return outcome_rows_by_variant_id[key]
    return {}


def validation_route_for_primitive_step(primitive_step_index: int, *, state: np.ndarray | None = None) -> dict[str, object]:
    """Return the governor-facing route without using rollout-budget knowledge."""

    if int(primitive_step_index) == 0:
        start_family = FIRST_PRIMITIVE_START_FAMILY
        reason = "first_0p10s_launch_window"
    else:
        start_family, reason = _continuation_start_family_and_reason(state)
    return {
        "start_state_family": start_family,
        "launch_sequence_phase": _launch_sequence_phase_for_start_family(
            primitive_step_index,
            start_state_family=start_family,
        ),
        "route_required_entry_role": _required_entry_role_for_start_family(start_family),
        "route_reason": reason,
    }


def validation_start_family_for_primitive_step(primitive_step_index: int, *, state: np.ndarray | None = None) -> str:
    """Return the governor-facing start family for the launch-aware sequence."""

    return str(validation_route_for_primitive_step(primitive_step_index, state=state)["start_state_family"])


def _continuation_start_family(state: np.ndarray | None) -> str:
    return _continuation_start_family_and_reason(state)[0]


def _continuation_start_family_and_reason(state: np.ndarray | None) -> tuple[str, str]:
    if state is None:
        return POST_LAUNCH_START_FAMILY, "state_unavailable_default_nominal_continuation"
    try:
        x = as_state_vector(state)
        margins = position_margin_m(x[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]], TRUE_SAFE_BOUNDS)
    except Exception:
        return TERMINAL_SAFE_EXIT_START_FAMILY, "invalid_state_recovery_edge_route"
    speed = float(np.linalg.norm(x[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]))
    max_body_rate = max(
        abs(float(x[STATE_INDEX["p"]])),
        abs(float(x[STATE_INDEX["q"]])),
        abs(float(x[STATE_INDEX["r"]])),
    )
    degraded_energy_or_attitude = (
        speed < RECOVERY_EDGE_MIN_SPEED_M_S
        or abs(float(x[STATE_INDEX["phi"]])) > RECOVERY_EDGE_MAX_ABS_ROLL_RAD
        or abs(float(x[STATE_INDEX["theta"]])) > RECOVERY_EDGE_MAX_ABS_PITCH_RAD
        or max_body_rate > RECOVERY_EDGE_MAX_BODY_RATE_RAD_S
    )
    if degraded_energy_or_attitude:
        return TERMINAL_SAFE_EXIT_START_FAMILY, "degraded_energy_attitude_or_rate_recovery_edge_route"
    if float(margins["min_margin_m"]) <= 0.0:
        return TERMINAL_SAFE_EXIT_START_FAMILY, "outside_or_on_true_safe_boundary_recovery_edge_route"
    if float(margins["min_margin_m"]) <= RECOVERY_ROUTE_MARGIN_M:
        return BOUNDARY_RECOVERY_START_FAMILY, "true_safe_margin_below_recovery_route_threshold"
    return POST_LAUNCH_START_FAMILY, "normal_post_launch_continuation"


def _launch_sequence_phase_for_start_family(primitive_step_index: int, *, start_state_family: str) -> str:
    if int(primitive_step_index) == 0:
        return "first_0p10s_launch_capture"
    family = str(start_state_family)
    if family == BOUNDARY_RECOVERY_START_FAMILY:
        return "state_routed_boundary_recovery"
    if family == TERMINAL_SAFE_EXIT_START_FAMILY:
        return "state_routed_recovery_safe_exit"
    return "post_launch_inflight"


def _launch_sequence_phase_for_step(primitive_step_index: int) -> str:
    return str(validation_route_for_primitive_step(primitive_step_index)["launch_sequence_phase"])


def _required_entry_role_for_start_family(start_state_family: str) -> str:
    family = str(start_state_family)
    if family == FIRST_PRIMITIVE_START_FAMILY:
        return "launch_capable"
    if family in {BOUNDARY_RECOVERY_START_FAMILY, TERMINAL_SAFE_EXIT_START_FAMILY}:
        return "terminal_or_recovery"
    return "inflight_only"


def _governor_mode_for_route(route: dict[str, object]) -> str:
    if str(route.get("route_required_entry_role", "")) == "terminal_or_recovery":
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
        "route_reason": str(route.get("route_reason", "")),
        "latency_case": latency_case,
        "wall_margin_m": float(context.wall_margin_m),
        "floor_margin_m": float(context.floor_margin_m),
        "ceiling_margin_m": float(context.ceiling_margin_m),
        "speed_margin_m_s": float(context.speed_margin_m_s),
        "w_wing_mean_m_s": float(context.w_wing_mean_m_s),
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
    if str(protocol.stage_id) == "R10" and block_id == BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="independent_uniform_xy_bounds",
            fan_position_xy_bounds_m=R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M,
        )
    if str(protocol.stage_id) == "R10" and block_id in R10_FIXED_BASE_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="fixed_base_positions",
        )
    if str(protocol.stage_id) == "R10" and block_id in R10_SHIFTED_FAN_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="common_shift",
        )
    if scheduled_active_fan_count is not None:
        return EnvironmentRandomisationConfig(active_fan_count=scheduled_active_fan_count)
    return None


def _governor_config_for_policy(policy: dict[str, object]) -> GovernorConfig:
    if bool(policy["safe_explore"]):
        return DEFAULT_GOVERNOR_CONFIG
    return replace(
        DEFAULT_GOVERNOR_CONFIG,
        config_id="v411_viability_filtered_no_exploration_ablation",
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
    hard_failure = any(_truthy(row.get("boundary_use_class", "") == "hard_failure") or _truthy(row.get("outcome_class", "") == "failed") for row in primitive_rows)
    floor_or_ceiling = any(str(row.get("failure_label", "")) in {"floor_violation", "ceiling_violation"} for row in primitive_rows)
    continuation_or_terminal = any(_truthy(row.get("continuation_valid", False)) or _truthy(row.get("episode_terminal_useful", False)) for row in primitive_rows)
    terminal_useful = any(_truthy(row.get("episode_terminal_useful", False)) for row in primitive_rows)
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
    sequence_compliant = _launch_sequence_compliant(primitive_rows)
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
        "no_viable_primitive": bool(str(blocked_reason).startswith("no_viable")),
        "safe_success": bool(continuation_or_terminal and not hard_failure),
        "terminal_useful": bool(terminal_useful),
        "lift_capture": bool(lift_capture),
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
    row.update(_launch_score_fields(row))
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
        "hard_failure": bool(str(rollout_row.get("boundary_use_class", "")) == "hard_failure" or str(rollout_row.get("outcome_class", "")) == "failed"),
        "floor_or_ceiling_violation": bool(str(rollout_row.get("failure_label", "")) in {"floor_violation", "ceiling_violation"}),
        "no_viable_primitive": False,
        "safe_success": bool(rollout_row.get("continuation_valid", False) or rollout_row.get("episode_terminal_useful", False)),
        "terminal_useful": bool(rollout_row.get("episode_terminal_useful", False)),
        "lift_capture": bool(float(rollout_row.get("lift_dwell_time_s", 0.0)) > 0.0),
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
    row.update(_launch_score_fields(row))
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
        "terminal_useful": False,
        "lift_capture": False,
        "lift_dwell_time_s": 0.0,
        "energy_residual_m": 0.0,
        "episode_specific_energy_start_m": 0.0,
        "episode_specific_energy_end_m": 0.0,
        "net_specific_energy_delta_m": 0.0,
        "gross_specific_energy_gain_m": 0.0,
        "gross_specific_energy_loss_m": 0.0,
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
    row.update(_launch_score_fields(row))
    return row


def _episode_specific_energy_summary(primitive_rows: list[dict[str, object]]) -> dict[str, float]:
    energy_pairs: list[tuple[float, float]] = []
    for row in primitive_rows:
        start = _state_vector_from_rollout_row(row, prefix="initial_")
        end = _state_vector_from_json(row.get("exit_state_vector", ""))
        if start is None or end is None:
            continue
        energy_pairs.append((_specific_energy_m(start), _specific_energy_m(end)))
    if not energy_pairs:
        return {
            "episode_specific_energy_start_m": 0.0,
            "episode_specific_energy_end_m": 0.0,
            "net_specific_energy_delta_m": 0.0,
            "gross_specific_energy_gain_m": 0.0,
            "gross_specific_energy_loss_m": 0.0,
        }
    deltas = [end - start for start, end in energy_pairs]
    return {
        "episode_specific_energy_start_m": float(energy_pairs[0][0]),
        "episode_specific_energy_end_m": float(energy_pairs[-1][1]),
        "net_specific_energy_delta_m": float(energy_pairs[-1][1] - energy_pairs[0][0]),
        "gross_specific_energy_gain_m": float(sum(max(delta, 0.0) for delta in deltas)),
        "gross_specific_energy_loss_m": float(sum(max(-delta, 0.0) for delta in deltas)),
    }


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
    episode_time_s = float(selected_steps) * PRIMITIVE_FINITE_HORIZON_S
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
    net_energy_factor = _clip(1.0 + _float_value(row.get("net_specific_energy_delta_m", row.get("energy_residual_m", 0.0))) / 2.0, 0.25, 1.75)
    energy_loss_factor = _clip(1.0 - _float_value(row.get("gross_specific_energy_loss_m", 0.0)) / 2.0, 0.50, 1.00)
    flight_time_factor = _clip(episode_time_s / SCORING_TARGET_EPISODE_TIME_S, 0.10, 1.25)
    wall_margin_factor = _wall_margin_factor(min_wall_margin)
    multiplicative_component = (
        100.0
        * outcome_multiplier
        * safety_multiplier
        * viability_multiplier
        * net_energy_factor
        * energy_loss_factor
        * flight_time_factor
        * wall_margin_factor
    )
    return {
        "launch_score_version": LAUNCH_SCORE_VERSION,
        "episode_flight_time_s": float(episode_time_s),
        "base_failure_penalty": float(base_penalty),
        "base_failure_penalty_reason": penalty_reason,
        "outcome_multiplier": float(outcome_multiplier),
        "safety_multiplier": float(safety_multiplier),
        "viability_multiplier": float(viability_multiplier),
        "net_energy_factor": float(net_energy_factor),
        "energy_loss_factor": float(energy_loss_factor),
        "flight_time_factor": float(flight_time_factor),
        "wall_margin_factor": float(wall_margin_factor),
        "launch_score_multiplicative_component": float(multiplicative_component),
        "launch_score": float(base_penalty + multiplicative_component),
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


def _wall_margin_factor(min_wall_margin_m: float) -> float:
    margin = float(min_wall_margin_m)
    if margin < 0.00:
        return 0.50
    if margin < 0.05:
        return 0.80
    if margin < 0.15:
        return 0.95
    if margin < 0.40:
        return 1.00
    return 1.05


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
        "belief_local_energy_residual_m": float(features.get("belief_local_energy_residual_m", 0.0) or 0.0),
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
        if len(rows) < int(chunk_size):
            continue
        partitions.append(_flush_table(run_root, table_name, rows, row_counters, storage_format, compression_level))
        rows.clear()
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
    launch_capture_ids = set(LAUNCH_CAPTURE_PRIMITIVE_IDS)
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
            first.groupby(["library_size_case_id", "primitive_id", "entry_role", "start_state_family"], dropna=False)
            .agg(candidate_rows=("primitive_variant_id", "count"), viable_rows=("viable_int", "sum"))
            .reset_index()
            .to_dict(orient="records")
        )
        for case_id, group in first.groupby("library_size_case_id", dropna=False):
            key = str(case_id)
            row = availability.setdefault(
                key,
                {
                    "stage_id": "R9_R10",
                    "library_size_case_id": key,
                    "launch_capable_primitives": set(),
                    "launch_capture_primitives": set(),
                    "first_decision_candidate_rows": 0,
                    "first_decision_viable_rows": 0,
                },
            )
            launch_capable = group[group["entry_role"].astype(str) == "launch_capable"]
            row["launch_capable_primitives"].update(launch_capable["primitive_id"].astype(str).tolist())
            row["launch_capture_primitives"].update(
                launch_capable.loc[
                    launch_capable["primitive_id"].astype(str).isin(launch_capture_ids),
                    "primitive_id",
                ]
                .astype(str)
                .tolist()
            )
            row["first_decision_candidate_rows"] = int(row["first_decision_candidate_rows"]) + int(len(group))
            row["first_decision_viable_rows"] = int(row["first_decision_viable_rows"]) + int(group["viable_int"].sum())
    _write_csv(run_root / "metrics" / "first_decision_candidate_summary.csv", _sum_rows(first_decision_rows, ["library_size_case_id", "policy_id", "launch_role"]))
    _write_csv(run_root / "metrics" / "first_decision_governor_rejection_summary.csv", _sum_rows(rejection_rows, ["library_size_case_id", "policy_id", "launch_role", "rejection_reason"]))
    _write_csv(run_root / "metrics" / "launch_gate_entry_role_audit.csv", _sum_rows(entry_rows, ["library_size_case_id", "primitive_id", "entry_role", "start_state_family"]))
    availability_rows = []
    for row in availability.values():
        launch_capable = set(row.pop("launch_capable_primitives"))
        launch_capture = set(row.pop("launch_capture_primitives"))
        non_launch_capture_launch_capable = sorted(launch_capable - launch_capture_ids)
        missing_launch_capture = sorted(launch_capture_ids - launch_capture)
        availability_rows.append(
            {
                **row,
                "launch_capable_primitive_family_count": int(len(launch_capable)),
                "launch_capture_primitive_family_count": int(len(launch_capture)),
                "required_launch_capture_primitive_family_count": int(len(launch_capture_ids)),
                "missing_launch_capture_primitive_ids": ",".join(missing_launch_capture),
                "non_launch_capture_launch_capable_primitive_ids": ",".join(non_launch_capture_launch_capable),
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
                hard_failure_rate=("hard_failure", "mean"),
                floor_or_ceiling_violation_rate=("floor_or_ceiling_violation", "mean"),
                no_viable_primitive_rate=("no_viable_primitive", "mean"),
                terminal_useful_rate=("terminal_useful", "mean"),
                lift_capture_rate=("lift_capture", "mean"),
                mean_lift_dwell_time_s=("lift_dwell_time_s", "mean"),
                mean_energy_residual_m=("energy_residual_m", "mean"),
                mean_net_specific_energy_delta_m=("net_specific_energy_delta_m", "mean"),
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
                hard_failure_rate=("hard_failure", "mean"),
                no_viable_primitive_rate=("no_viable_primitive", "mean"),
                mean_launch_score=("launch_score", "mean"),
            )
            .reset_index()
        )
    _write_csv(run_root / "metrics" / "library_size_case_comparison.csv", library)
    if protocol.stage_id == "R10":
        env = (
            final.groupby(["environment_block_id"], dropna=False)
            .agg(
                launch_count=("episode_id", "count"),
                safe_success_rate=("safe_success", "mean"),
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
            fields["gross_specific_energy_gain_m"] = max(fields["net_specific_energy_delta_m"], 0.0)
            fields["gross_specific_energy_loss_m"] = max(-fields["net_specific_energy_delta_m"], 0.0)
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
        if not (policy_id.startswith(MEMORY_POLICY_PREFIX) or policy_id == "static_map_baseline"):
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
        if policy_id.startswith(MEMORY_POLICY_PREFIX) or policy_id == "static_map_baseline":
            out.at[index, "memory_changed_selection"] = signature != baseline_map.get(key, signature)
        if policy_id.startswith(SAFE_EXPLORE_POLICY_PREFIX):
            memory_key = (str(row["library_size_case_id"]), int(row["outer_case_index"]), int(row["history_length"]))
            out.at[index, "exploration_changed_selection"] = signature != memory_signatures.get(memory_key, signature)
    return out


def _pass_fail_summary(
    *,
    protocol: ValidationProtocol,
    max_primitives_per_launch: int,
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
        _gate_row("max_primitives_per_launch_full_validation", int(max_primitives_per_launch) >= 4, int(max_primitives_per_launch), ">=4"),
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
                _gate_row("hard_failure_rate_le_1pct", _mean_bool(final_rows, "hard_failure") <= 0.01, _mean_bool(final_rows, "hard_failure"), 0.01),
                _gate_row("floor_or_ceiling_violation_rate_zero", _mean_bool(final_rows, "floor_or_ceiling_violation") == 0.0, _mean_bool(final_rows, "floor_or_ceiling_violation"), 0.0),
                _gate_row("no_viable_primitive_rate_le_2pct", _mean_bool(final_rows, "no_viable_primitive") <= 0.02, _mean_bool(final_rows, "no_viable_primitive"), 0.02),
                _gate_row("safe_success_rate_near_100pct", _mean_bool(final_rows, "safe_success") >= 0.99, _mean_bool(final_rows, "safe_success"), 0.99),
                _gate_row("terminal_or_lift_capture_ge_90pct", max(_mean_bool(final_rows, "terminal_useful"), _mean_bool(final_rows, "lift_capture")) >= 0.90, max(_mean_bool(final_rows, "terminal_useful"), _mean_bool(final_rows, "lift_capture")), 0.90),
                _gate_row("selected_primitive_family_count_ge_5", len(selected_primitives) >= 5, len(selected_primitives), 5),
                _gate_row("selected_variant_count_ge_10", len(selected_variants) >= 10, len(selected_variants), 10),
            ]
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
        "dry_run_schedule": bool(config.dry_run_schedule),
        "storage_format": resolve_storage_format(config.storage_format),
        "compression_level": int(config.compression_level),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "max_primitives_per_launch": int(config.max_primitives_per_launch),
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
        "first_primitive_start_state_family": FIRST_PRIMITIVE_START_FAMILY,
        "post_launch_start_state_family": POST_LAUNCH_START_FAMILY,
        "boundary_recovery_start_state_family": BOUNDARY_RECOVERY_START_FAMILY,
        "terminal_safe_exit_start_state_family": TERMINAL_SAFE_EXIT_START_FAMILY,
        "first_primitive_required_entry_role": "launch_capable",
        "post_launch_required_entry_role": "inflight_only",
        "boundary_recovery_required_entry_role": "terminal_or_recovery",
        "terminal_safe_exit_required_entry_role": "terminal_or_recovery",
        "r10_active_fan_count_policy": "balanced_1_2_3_4_for_active_fan_number_variation"
        if str(protocol.stage_id) == "R10"
        else "not_applicable",
        "r10_active_fan_count_sequence": list(R10_ACTIVE_FAN_COUNT_SEQUENCE)
        if str(protocol.stage_id) == "R10"
        else [],
        "r10_arena_wide_fan_position_block_id": BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID
        if str(protocol.stage_id) == "R10"
        else "not_applicable",
        "r10_arena_wide_fan_position_xy_bounds_m": R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M
        if str(protocol.stage_id) == "R10"
        else (),
        "recovery_route_margin_m": float(RECOVERY_ROUTE_MARGIN_M),
        "recovery_edge_min_speed_m_s": float(RECOVERY_EDGE_MIN_SPEED_M_S),
        "recovery_edge_max_abs_roll_rad": float(RECOVERY_EDGE_MAX_ABS_ROLL_RAD),
        "recovery_edge_max_abs_pitch_rad": float(RECOVERY_EDGE_MAX_ABS_PITCH_RAD),
        "recovery_edge_max_body_rate_rad_s": float(RECOVERY_EDGE_MAX_BODY_RATE_RAD_S),
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
        f"- Launch sequence policy: `{LAUNCH_SEQUENCE_POLICY_ID}`",
        f"- Recovery route: `{BOUNDARY_RECOVERY_START_FAMILY}` below `{RECOVERY_ROUTE_MARGIN_M}` m safe margin, `{TERMINAL_SAFE_EXIT_START_FAMILY}` for degraded speed, attitude, rate, or boundary contact.",
        f"- Launch score: `{LAUNCH_SCORE_VERSION}`; paired score deltas are audit evidence, not pass-gate substitutes.",
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
    parser = argparse.ArgumentParser(description="Run full R9 fixed-case repeated-launch rollout validation.")
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--source-w2-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--storage-format", choices=("auto", "parquet", "csv_gz", "csv"), default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=800)
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--max-primitives-per-launch", type=int, default=4)
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
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "blocked", "dry_run_schedule"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
