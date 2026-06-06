from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

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
    DirectionalResidualCell,
    DirectionalResidualLiftBelief,
    DirectionalResidualObservation,
    FLOW_BELIEF_GRID_RESOLUTION_M,
    FLOW_BELIEF_QUERY_RADIUS_M,
    directional_residual_lift_cell_lookup,
    directional_residual_lift_spatial_cell_lookup,
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    query_spatial_flow_belief_features,
    query_spatial_flow_belief_features_fast,
    update_directional_residual_lift_belief_batch,
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
from primitive_timing_contract import (  # noqa: E402
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    LAUNCH_HANDOFF_DURATION_S,
    PRIMITIVE_FINITE_HORIZON_S,
    primitive_timing_contract_row,
)
from lqr_linearisation import local_speed_from_state_vector, lqr_speed_bin_id  # noqa: E402
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
from viability_governor import (  # noqa: E402
    DEFAULT_GOVERNOR_CONFIG,
    GovernorConfig,
    calibrated_regime_risk_features,
    governor_config_to_row,
)


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.20"
VALIDATION_VERSION = "repeated_launch_fixed_case_rollout_preflight_v7"
GOVERNOR_TUNING_HANDOFF_VERSION = "governor_tuning_handoff_v4"
HISTORY_LENGTHS = (3, 10, 30)
SAFE_EXPLORE_ABLATION_HISTORY_LENGTH = 10
HISTORY_LENGTH_SUM = sum(HISTORY_LENGTHS)
OPEN_LOOP_COMPARISON_POLICY_ID = "open_loop_zero_command_baseline"
EMPTY_FROZEN_PRIOR_BASELINE_ID = "empty_frozen_prior_baseline"
BASELINE_POLICY_IDS = ("no_memory_baseline",)
MEMORY_POLICY_PREFIX = "spatial_flow_belief_memory"
SAFE_EXPLORE_POLICY_PREFIX = "safe_explore_then_exploit"
POLICY_HISTORY_CONDITIONS = (
    "no_memory_baseline",
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
)
R11_POLICY_HISTORY_CONDITIONS = (
    OPEN_LOOP_COMPARISON_POLICY_ID,
    *POLICY_HISTORY_CONDITIONS,
)
LAUNCH_SPEED_BIN_DEFINITIONS = (
    ("v0_lt_4_0_m_s", None, 4.0, "initial_launch_speed_m_s < 4.0"),
    ("v0_4_0_to_5_0_m_s", 4.0, 5.0, "4.0 <= initial_launch_speed_m_s < 5.0"),
    ("v0_5_0_to_6_0_m_s", 5.0, 6.0, "5.0 <= initial_launch_speed_m_s < 6.0"),
    ("v0_6_0_to_7_0_m_s", 6.0, 7.0, "6.0 <= initial_launch_speed_m_s < 7.0"),
    ("v0_ge_7_0_m_s", 7.0, None, "initial_launch_speed_m_s >= 7.0"),
)
START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S = 5.0
LOW_START_ENERGY_GROUP_ID = "low_start_energy_v0_lt_5_0_m_s"
HIGH_START_ENERGY_GROUP_ID = "high_start_energy_v0_ge_5_0_m_s"
R9_POLICY_HISTORY_CONDITIONS = (
    "no_memory_baseline",
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
)
R9_HISTORY_LENGTH_SUM = sum(HISTORY_LENGTHS)
R9_PREFLIGHT_CASES_PER_BLOCK = 1
R9_OUTER_CASES_PER_CONDITION = 3 * R9_PREFLIGHT_CASES_PER_BLOCK
R9_EXPECTED_FINAL_HELDOUT_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * len(R9_POLICY_HISTORY_CONDITIONS) * R9_OUTER_CASES_PER_CONDITION
R9_EXPECTED_HISTORY_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * R9_OUTER_CASES_PER_CONDITION * R9_HISTORY_LENGTH_SUM
DEFAULT_LIBRARY_ROOT = Path("03_Control/05_Results/R8_library_size_study/A01")
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/R8_outcome/A01")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/R9_test")
LEGACY_RESULT_ROOT_NAMES = {"lqr_contextual_v1_0", "lqr_contextual_v_1_0"}
RENAMED_STAGE_ROOTS = {
    "w01_dense": "R5_dense",
    "w2_survival": "R6_archived",
    "w3_survival": "R7_survival",
    "outcome_model": "R8_outcome",
    "post_w3_library_size_study": "R8_library_size_study",
    "changed_case_validation": "R10_learn",
    "repeated_launch_validation": "R11_validation",
}
TABLE_NAMES = (
    "episode_summary",
    "primitive_execution_log",
    "history_plot_trace",
    "history_memory_trace",
    "history_selector_summary",
    "candidate_score_log",
    "selector_decision_log",
    "memory_residual_update_log",
    "belief_snapshot_log",
)
SCHEDULE_INLINE_ROW_LIMIT = 50_000
SCHEDULE_PARTITION_ROW_COUNT = 50_000
MEMORY_OPPORTUNITY_DECISION_LOG_INLINE_ROW_LIMIT = 50_000
MEMORY_OPPORTUNITY_DECISION_LOG_PARTITION_ROW_COUNT = 25_000
CANDIDATE_SCORE_TOP_K_PER_DECISION = 10
HISTORY_LOG_MODES = ("auto", "plot_summary", "sampled_debug", "full_debug")
DEFAULT_HISTORY_DEBUG_SAMPLE_STRIDE = 10
REAL_TIME_OUTER_LOOP_SCHEDULER_VERSION = "predictive_next_primitive_scheduler_profile_v1"
REAL_TIME_PREFERRED_DECISION_BUDGET_S = CONTROLLER_INPUT_UPDATE_PERIOD_S
REAL_TIME_HARD_DECISION_BUDGET_S = PRIMITIVE_FINITE_HORIZON_S
OUTER_LOOP_MEMORY_POLICY_VERSION = "outer_loop_cost_benefit_spatial_flow_memory_v4_1"
OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION = (
    "case_local_online_memory_plus_r10_global_deterministic_calibration_v1"
)
ONLINE_MEMORY_SCOPE = "case_local_reset_per_final_schedule_row"
R10_GLOBAL_CALIBRATION_SCOPE = "aggregate_all_r10_final_heldout_rows_and_selector_opportunity_diagnostics"
R11_GOVERNOR_HANDOFF_SCOPE = "single_frozen_r10_governor_config_used_for_r11_validation"
GOVERNOR_CALIBRATION_SEARCH_POLICY = "deterministic_bounded_rule_update_no_profile_ladder_no_black_box_search"
CANDIDATE_PATH_MEMORY_LOOKAHEAD_S = 5.0 * PRIMITIVE_FINITE_HORIZON_S
CANDIDATE_PATH_MEMORY_RESIDUAL_CAP_M = 0.75
CANDIDATE_PATH_MEMORY_SPECIFIC_ENERGY_RESIDUAL_CAP_M = 1.00
CANDIDATE_PATH_MEMORY_UTILITY_SPECIFIC_ENERGY_WEIGHT = 0.75
CANDIDATE_PATH_MEMORY_UTILITY_UPDRAFT_WEIGHT = 0.25
CANDIDATE_PATH_MEMORY_FULL_CONFIDENCE_OBSERVATIONS = 3.0
RESIDUAL_MEMORY_LAUNCH_RECENCY_HALF_LIFE = 4.0
CANDIDATE_PATH_MEMORY_HEADING_OFFSET_CAP_RAD = math.radians(35.0)
CANDIDATE_PATH_MEMORY_PROBES = (
    (0.0, 0.08),
    (0.17, 0.12),
    (0.33, 0.16),
    (0.50, 0.20),
    (0.67, 0.18),
    (0.83, 0.14),
    (1.0, 0.12),
)
REAL_TIME_FLOW_BELIEF_QUERY_RADIUS_M = FLOW_BELIEF_GRID_RESOLUTION_M
REAL_TIME_CANDIDATE_PATH_MEMORY_PROBES = (
    (0.0, 0.18),
    (0.50, 0.42),
    (1.0, 0.40),
)
FLOW_BELIEF_REACHABLE_ATTRACTION_LOOKAHEAD_M = 0.80
FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD = math.radians(35.0)
FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD = math.radians(20.0)
FLOW_BELIEF_REACHABLE_ATTRACTION_HALF_ANGLE_RAD = FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD
FLOW_BELIEF_REACHABLE_ATTRACTION_CAP_M = 0.25
FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES = (
    (0.35, -math.radians(35.0), -math.radians(20.0), 0.040),
    (0.35, -math.radians(35.0), 0.0, 0.055),
    (0.35, -math.radians(35.0), math.radians(20.0), 0.040),
    (0.35, 0.0, -math.radians(20.0), 0.055),
    (0.35, 0.0, 0.0, 0.085),
    (0.35, 0.0, math.radians(20.0), 0.055),
    (0.35, math.radians(35.0), -math.radians(20.0), 0.040),
    (0.35, math.radians(35.0), 0.0, 0.055),
    (0.35, math.radians(35.0), math.radians(20.0), 0.040),
    (0.80, -math.radians(35.0), -math.radians(20.0), 0.030),
    (0.80, -math.radians(35.0), 0.0, 0.040),
    (0.80, -math.radians(35.0), math.radians(20.0), 0.030),
    (0.80, 0.0, -math.radians(20.0), 0.040),
    (0.80, 0.0, 0.0, 0.060),
    (0.80, 0.0, math.radians(20.0), 0.040),
    (0.80, math.radians(35.0), -math.radians(20.0), 0.030),
    (0.80, math.radians(35.0), 0.0, 0.040),
    (0.80, math.radians(35.0), math.radians(20.0), 0.030),
)
REAL_TIME_FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES = (
    (0.35, 0.0, 0.0, 0.24),
    (0.80, -math.radians(35.0), 0.0, 0.16),
    (0.80, 0.0, -math.radians(20.0), 0.14),
    (0.80, 0.0, 0.0, 0.26),
    (0.80, 0.0, math.radians(20.0), 0.14),
    (0.80, math.radians(35.0), 0.0, 0.16),
)
FLOW_BELIEF_ROUTE_PROBE_FRACTIONS = (
    (0.25, 0.0, 0.0, 0.26),
    (0.50, -math.radians(18.0), 0.0, 0.12),
    (0.50, 0.0, 0.0, 0.22),
    (0.50, math.radians(18.0), 0.0, 0.12),
    (0.75, 0.0, -math.radians(12.0), 0.08),
    (0.75, 0.0, 0.0, 0.12),
    (0.75, 0.0, math.radians(12.0), 0.08),
)
REAL_TIME_FLOW_BELIEF_ROUTE_PROBE_FRACTIONS = (
    (0.50, 0.0, 0.0, 0.46),
    (0.75, -math.radians(18.0), 0.0, 0.18),
    (0.75, 0.0, 0.0, 0.18),
    (0.75, math.radians(18.0), 0.0, 0.18),
)
FLOW_BELIEF_HISTORY_UPDATE_SPACING_M = FLOW_BELIEF_GRID_RESOLUTION_M
FLOW_BELIEF_HISTORY_UPDATE_MAX_SAMPLES_PER_PRIMITIVE = 128
FLOW_BELIEF_HISTORY_UPDATE_POLICY = "dense_executed_segment_samples_at_0p1m_grid_spacing_with_launch_recency_decay"
THESIS_FACING_WORKFLOW = "R5 -> R7 -> R8 -> R10 -> R11 -> Reality"
R9_THESIS_REPORTING_STATUS = "internal_preflight_excluded_from_thesis_workflow_narrative"
REAL_FLIGHT_REQUIRED_LIBRARY_CASE_IDS = ("heavy_cluster", "balanced_cluster")
REAL_FLIGHT_OPTIONAL_LIBRARY_CASE_IDS = ("light_cluster", "super_light_cluster")
OFFLINE_UNRESTRICTED_LIBRARY_CASE_IDS = ("no_cluster_no_merge",)
LAUNCH_SEQUENCE_POLICY_ID = "state_class_transition_entry_governor_no_launch_specific_family"
FIRST_PRIMITIVE_START_FAMILY = "launch_gate"
POST_LAUNCH_START_FAMILY = "inflight_nominal"
BOUNDARY_RECOVERY_START_FAMILY = "inflight_boundary_near"
TERMINAL_SAFE_EXIT_START_FAMILY = "inflight_recovery_edge"
ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID = "active_fan_number_variation"
BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID = "arena_wide_fan_position_generalisation"
NO_UPDRAFT_CHANGED_CASE_BLOCK_ID = "no_updraft_dry_air_generalisation"
R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID = "r10_l7_full_domain_randomisation_arena_wide_training"
R11_L0_DRY_AIR_FIXED_BLOCK_ID = "r11_l0_dry_air_fixed"
R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID = "r11_l1_single_fan_fixed_nominal"
R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID = "r11_l2_four_fan_fixed_nominal"
R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID = "r11_l3_fan_parameter_uncertainty"
R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID = "r11_l4_local_fan_position_uncertainty"
R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID = "r11_l5_active_fan_count_uncertainty"
R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID = "r11_l6_environment_only_full_uncertainty"
R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID = "r11_l7_full_domain_randomisation_arena_wide"
TARGETED_MEMORY_OPPORTUNITY_BLOCK_ID = "targeted_memory_opportunity_arena_wide_four_fan"
R11_FIDELITY_LADDER_BLOCK_IDS = (
    R11_L0_DRY_AIR_FIXED_BLOCK_ID,
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
)
CHANGED_CASE_VALIDATION_STAGE_IDS = {"R10", "R11"}
R10_R11_REALISTIC_REPEATED_LAUNCH_BLOCK_IDS = (
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    TARGETED_MEMORY_OPPORTUNITY_BLOCK_ID,
    *R11_FIDELITY_LADDER_BLOCK_IDS,
)
R10_FULL_DOMAIN_RANDOMISATION_BLOCK_IDS = (
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
)
R10_SINGLE_FAN_GEOMETRY_BLOCK_IDS = (
    "nominal_single_fan_perturbations",
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
)
R10_FOUR_FAN_GEOMETRY_BLOCK_IDS = (
    "nominal_four_fan_perturbations",
    "shifted_four_fan_positions",
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    TARGETED_MEMORY_OPPORTUNITY_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
)
R10_FIXED_BASE_POSITION_BLOCK_IDS = (
    "nominal_single_fan_perturbations",
    "nominal_four_fan_perturbations",
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
)
R10_SHIFTED_FAN_POSITION_BLOCK_IDS = (
    "shifted_single_fan_positions",
    "shifted_four_fan_positions",
    R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
)
R10_ACTIVE_FAN_COUNT_VARIATION_BLOCK_IDS = (
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
)
R10_FIXED_FOUR_ACTIVE_BLOCK_IDS = (
    "nominal_four_fan_perturbations",
    "shifted_four_fan_positions",
    TARGETED_MEMORY_OPPORTUNITY_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
)
R10_NO_UPDRAFT_BLOCK_IDS = (
    NO_UPDRAFT_CHANGED_CASE_BLOCK_ID,
    R11_L0_DRY_AIR_FIXED_BLOCK_ID,
)
DRY_AIR_ENERGY_DEPLETION_BLOCK_IDS = (
    "no_updraft",
    NO_UPDRAFT_CHANGED_CASE_BLOCK_ID,
    R11_L0_DRY_AIR_FIXED_BLOCK_ID,
)
R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS = (
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    TARGETED_MEMORY_OPPORTUNITY_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
)
R10_FIXED_ENVIRONMENT_BETWEEN_HISTORY_BLOCK_IDS = (
    R11_L0_DRY_AIR_FIXED_BLOCK_ID,
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
)
R10_UPDRAFT_PARAMETER_VARIATION_BETWEEN_HISTORY_BLOCK_IDS = (
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
)
R10_NOMINAL_FAN_PARAMETER_BLOCK_IDS = (
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
)
R10_ACTIVE_FAN_COUNT_SEQUENCE = (0, 1, 2, 3, 4)
R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M = ((0.0, 8.0), (0.0, 4.8))
R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M = 0.5
RECOVERY_ROUTE_MARGIN_M = 0.25
RECOVERY_EDGE_MAX_ABS_ROLL_RAD = math.radians(35.0)
RECOVERY_EDGE_MAX_ABS_PITCH_RAD = math.radians(22.0)
RECOVERY_EDGE_MAX_BODY_RATE_RAD_S = 0.65
LAUNCH_SCORE_VERSION = "r10_r11_front_wall_mission_updraft_terminal_energy_score_v2"
SPECIFIC_ENERGY_GRAVITY_M_S2 = 9.80665
MISSION_FRONT_WALL_X_TOL_M = 1e-6
MISSION_COMPLETION_SCORE = 100.0
MISSION_LIFT_CAPTURE_BASE_SCORE = 30.0
MISSION_SAFE_ROLLOUT_BASE_SCORE = 10.0
UPDRAFT_GAIN_SCORE_PER_M = 20.0
UPDRAFT_GAIN_SCORE_CAP = 40.0
LIFT_DWELL_SCORE_PER_S = 5.0
LIFT_DWELL_SCORE_CAP = 20.0
TERMINAL_SPECIFIC_ENERGY_REFERENCE_M = TRUE_SAFE_BOUNDS.z_w_m[0]
TERMINAL_SPECIFIC_ENERGY_SCORE_PER_M = 10.0
TERMINAL_SPECIFIC_ENERGY_SCORE_CAP = 20.0
WRONG_WALL_EXIT_PENALTY = -50.0
LOW_LAUNCH_SPEED_DRY_AIR_THRESHOLD_M_S = 5.0
DRY_AIR_ENERGY_DEPLETION_MIN_FLIGHT_TIME_S = 0.5
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
    "uncontrolled_xy_boundary_exit",
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
    run_label: str = ""
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
    history_log_mode: str = "auto"
    history_debug_sample_stride: int = DEFAULT_HISTORY_DEBUG_SAMPLE_STRIDE


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
    policy_history_conditions: tuple[str, ...] = POLICY_HISTORY_CONDITIONS
    reduced_diagnostic: bool = False
    requires_no_glider_latency_variation_audit: bool = False
    gate_profile: str = "strict_final_validation"
    max_hard_failure_rate: float = 0.01
    max_floor_or_ceiling_violation_rate: float = 0.0
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
    policy_history_conditions=R9_POLICY_HISTORY_CONDITIONS,
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
    run_label: str
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
    history_log_mode: str = "auto"
    history_debug_sample_stride: int = DEFAULT_HISTORY_DEBUG_SAMPLE_STRIDE


@dataclass(frozen=True)
class FrozenControllerRecordLoadResult:
    records_by_variant: dict[str, object]
    resolved_root: Path | None
    resolved_reason: str
    attempted_roots: tuple[Path, ...]
    attempted_reasons: tuple[str, ...]
    record_count: int
    bundle_variant_count: int
    bundle_ready_count: int


def run_repeated_launch_learning_curve(config: RepeatedLaunchValidationConfig) -> dict[str, object]:
    """Run the reduced internal R9 fixed-case repeated-launch preflight."""

    return run_repeated_launch_validation(
        ValidationRunConfig(
            library_root=config.library_root,
            outcome_root=config.outcome_root,
            output_root=config.output_root,
            run_id=config.run_id,
            run_label=config.run_label,
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
            history_log_mode=config.history_log_mode,
            history_debug_sample_stride=config.history_debug_sample_stride,
        ),
        protocol=R9_PROTOCOL,
    )


def run_repeated_launch_validation(config: ValidationRunConfig, *, protocol: ValidationProtocol) -> dict[str, object]:
    """Run or schedule repeated-launch validation with true primitive rollout rows."""

    run_root = Path(config.output_root) / _run_folder_name(config.run_id, config.run_label)
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
        _write_real_time_scheduler_audit_from_partitions(run_root, [], storage_format)
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

    source_record_load = _load_records_by_variant(config, libraries)
    records_by_variant = source_record_load.records_by_variant
    if not records_by_variant:
        _write_blocked_outputs(
            run_root,
            config,
            protocol,
            "missing_frozen_controller_records_for_rollout",
            source_record_load=source_record_load,
        )
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
    _write_real_time_scheduler_audit_from_partitions(run_root, partitions, storage_format)
    _write_memory_opportunity_audit_from_partitions(
        run_root,
        partitions,
        storage_format,
        run_id=int(config.run_id),
        compression_level=int(config.compression_level),
    )
    episode_rows = _read_partitioned_rows(run_root, partitions, "episode_summary")
    selector_rows = _read_partitioned_rows(run_root, partitions, "selector_decision_log")
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
        _write_governor_tuning_outputs(run_root, config, protocol, pass_summary, episode_rows, selector_rows)
    if protocol.stage_id == "R10":
        _write_governor_tuning_outputs(run_root, config, protocol, pass_summary, episode_rows, selector_rows)
    status = "smoke_run" if int(config.smoke_outer_cases_per_block) > 0 else "complete"
    _write_manifest(
        run_root=run_root,
        config=config,
        protocol=protocol,
        status=status,
        pass_summary=pass_summary,
        final_schedule=final_schedule,
        history_schedule=history_schedule,
        source_record_load=source_record_load,
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


def _load_records_by_variant(config: ValidationRunConfig, libraries: dict[str, dict[str, object]]) -> FrozenControllerRecordLoadResult:
    candidates = _frozen_controller_source_candidates(config, libraries)
    attempted_roots = tuple(root for root, _reason in candidates)
    attempted_reasons = tuple(reason for _root, reason in candidates)
    for root, reason in candidates:
        bundle_path = filesystem_path(root / "manifests" / "frozen_w01_controller_bundle.json")
        if not bundle_path.is_file():
            continue
        bundle = load_frozen_w01_controller_bundle(root / "manifests" / "frozen_w01_controller_bundle.json")
        records = {
            record.primitive_variant_id: record
            for record in bundle.records
            if str(record.bundle_status) == FROZEN_CONTROLLER_READY
        }
        return FrozenControllerRecordLoadResult(
            records_by_variant=records,
            resolved_root=Path(root),
            resolved_reason=reason,
            attempted_roots=attempted_roots,
            attempted_reasons=attempted_reasons,
            record_count=len(records),
            bundle_variant_count=int(bundle.variant_count),
            bundle_ready_count=int(bundle.ready_count),
        )
    return FrozenControllerRecordLoadResult(
        records_by_variant={},
        resolved_root=None,
        resolved_reason="",
        attempted_roots=attempted_roots,
        attempted_reasons=attempted_reasons,
        record_count=0,
        bundle_variant_count=0,
        bundle_ready_count=0,
    )


def _frozen_controller_source_candidates(
    config: ValidationRunConfig,
    libraries: dict[str, dict[str, object]],
) -> tuple[tuple[Path, str], ...]:
    roots: list[tuple[Path, str]] = []

    def append_root(value: object, reason: str) -> None:
        text = str(value).strip() if value is not None else ""
        if not text:
            return
        root = Path(text)
        roots.append((root, reason))
        relocated = _relocated_result_root(root)
        if relocated != root:
            roots.append((relocated, f"{reason}:relocated_result_root"))

    if config.source_w2_root is not None:
        append_root(config.source_w2_root, "config_source_w2_root")
    for payload in libraries.values():
        append_root(payload.get("source_w2_root"), "library_source_w2_root")
        append_root(payload.get("source_w01_root"), "library_source_w01_root")
        append_root(payload.get("source_r5_root"), "library_source_r5_root")
        for row in list(payload.get("representatives", [])):
            append_root(row.get("source_w2_root"), "representative_source_w2_root")
            append_root(row.get("source_w01_root"), "representative_source_w01_root")
            append_root(row.get("source_r5_root"), "representative_source_r5_root")

    deduped: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for root, reason in roots:
        key = root.as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((root, reason))
    return tuple(deduped)


def _relocated_result_root(root: Path) -> Path:
    parts = list(root.parts)
    lowered = [part.lower() for part in parts]
    for index, part in enumerate(lowered):
        if part not in LEGACY_RESULT_ROOT_NAMES:
            continue
        if index + 1 >= len(parts):
            return root
        replacement = RENAMED_STAGE_ROOTS.get(lowered[index + 1])
        if replacement is None:
            return root
        return Path(*parts[:index]) / replacement / Path(*parts[index + 2 :])
    return root


def _selected_worker_count(config: ValidationRunConfig) -> int:
    requested = max(1, int(config.workers or 1))
    if config.max_workers is None:
        return requested
    return max(1, min(requested, int(config.max_workers)))


def _resolved_history_log_mode(config: ValidationRunConfig, protocol: ValidationProtocol) -> str:
    requested = str(config.history_log_mode or "auto").strip().lower()
    if requested not in HISTORY_LOG_MODES:
        raise ValueError(f"history_log_mode must be one of: {', '.join(HISTORY_LOG_MODES)}")
    if requested != "auto":
        return requested
    if str(protocol.stage_id) == "R9" or bool(protocol.reduced_diagnostic) or int(config.smoke_outer_cases_per_block) > 0:
        return "full_debug"
    return "plot_summary"


def _history_debug_log_retained(
    scheduled: dict[str, object],
    *,
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
) -> bool:
    if str(scheduled.get("launch_role", "")) != "history":
        return True
    mode = _resolved_history_log_mode(config, protocol)
    if mode == "full_debug":
        return True
    if mode != "sampled_debug":
        return False
    stride = max(1, int(config.history_debug_sample_stride or DEFAULT_HISTORY_DEBUG_SAMPLE_STRIDE))
    history_index = int(scheduled.get("history_launch_index", 0))
    history_length = max(1, int(scheduled.get("history_length", 1)))
    return history_index == 0 or history_index == history_length - 1 or history_index % stride == 0


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
        _warm_up_realtime_context_cache(protocol)
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
    _warm_up_realtime_context_cache(protocol)


def _warm_up_realtime_context_cache(protocol: ValidationProtocol) -> None:
    """Prepare one context path outside the profiled primitive-boundary budget."""

    try:
        sample = archive_state_sample_for_family(
            start_state_family=FIRST_PRIMITIVE_START_FAMILY,
            paired_start_key=f"{protocol.stage_id.lower()}_scheduler_warmup",
            sample_index=0,
            seed=0,
            W_layer="W2",
            environment_mode="annular_gp_single",
        )
        scheduled = {
            "W_layer": "W2",
            "environment_mode": "annular_gp_single",
            "environment_seed": 0,
            "plant_implementation_seed": 0,
            "environment_block_id": "scheduler_warmup",
            "outer_case_type": "scheduler_warmup",
            "scheduled_fan_layout_count": "",
            "fan_layout_policy": "scheduler_warmup",
            "fan_position_policy": "scheduler_warmup",
            "fan_position_xy_bounds_m": "",
            "fan_position_safety_radius_m": "",
            "library_size_case_id": "scheduler_warmup",
            "history_length": 0,
            "adaptation_launch_index": 0,
            "policy_id": "scheduler_warmup",
        }
        _context_payload(
            state=as_state_vector(sample.state_vector),
            scheduled=scheduled,
            episode_id=f"{protocol.stage_id.lower()}_scheduler_warmup",
            protocol=protocol,
            start_state_family=FIRST_PRIMITIVE_START_FAMILY,
            primitive_step_index=0,
            route=validation_route_for_primitive_step(0, state=as_state_vector(sample.state_vector)),
        )
    except Exception:
        return


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
    base_governor_config = config.governor_config or DEFAULT_GOVERNOR_CONFIG
    belief = _initial_belief_for_policy(
        policy=policy,
        final_row=final_row,
        governor_config=base_governor_config,
    )
    launch_results: list[dict[str, object]] = []
    for hist_index in range(int(policy["history_length"])):
        history_row = _history_row_for_final(final_row, hist_index)
        retain_history_debug = _history_debug_log_retained(history_row, config=config, protocol=protocol)
        history_result = _run_one_launch(
            scheduled=history_row,
            policy=policy,
            representatives=representatives,
            outcome_rows_by_variant_id=case_outcomes,
            records_by_variant=records_by_variant,
            belief=belief,
            config=config,
            protocol=protocol,
            retain_debug_logs=retain_history_debug,
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
        retain_debug_logs=True,
    )
    launch_results.append(_launch_result_for_parent(final_result))
    return launch_results


def _launch_result_for_parent(result: dict[str, object]) -> dict[str, object]:
    debug_log_retained = bool(result.get("debug_log_retained", True))
    is_history = _result_launch_role(result) == "history"
    return {
        "episode_rows": result["episode_rows"],
        "primitive_rows": result["primitive_rows"],
        "history_plot_rows": _history_plot_trace_rows(result) if is_history else [],
        "history_memory_rows": _history_memory_trace_rows(result) if is_history else [],
        "history_selector_rows": _history_selector_summary_rows(result) if is_history else [],
        "candidate_rows": result["candidate_rows"] if debug_log_retained else [],
        "selector_rows": result["selector_rows"] if debug_log_retained else [],
        "memory_rows": result["memory_rows"] if debug_log_retained else [],
        "belief_rows": result["belief_rows"] if debug_log_retained else [],
    }


def _result_launch_role(result: dict[str, object]) -> str:
    scheduled = dict(result.get("scheduled", {}))
    if scheduled.get("launch_role"):
        return str(scheduled.get("launch_role", ""))
    for row in list(result.get("episode_rows", [])):
        if row.get("launch_role"):
            return str(row.get("launch_role", ""))
    return ""


def _history_trace_identity(result: dict[str, object]) -> dict[str, object]:
    scheduled = dict(result.get("scheduled", {}))
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": "history",
        "history_launch_index": int(_float_value(scheduled.get("history_launch_index", 0))),
        "launch_state_seed": int(_float_value(scheduled.get("launch_state_seed", 0))),
        "environment_seed": int(_float_value(scheduled.get("environment_seed", 0))),
        "environment_layout_seed": int(_float_value(scheduled.get("environment_layout_seed", scheduled.get("environment_seed", 0)))),
        "environment_active_fan_seed": int(_float_value(scheduled.get("environment_active_fan_seed", scheduled.get("environment_seed", 0)))),
        "environment_parameter_seed": int(_float_value(scheduled.get("environment_parameter_seed", scheduled.get("environment_seed", 0)))),
        "plant_implementation_seed": int(_float_value(scheduled.get("plant_implementation_seed", 0))),
        "environment_block_local_index": int(
            _float_value(scheduled.get("environment_block_local_index", scheduled.get("outer_case_index", 0)))
        ),
        "scheduled_active_fan_count": int(_float_value(scheduled["scheduled_active_fan_count"]))
        if str(scheduled.get("scheduled_active_fan_count", "")).strip() not in {"", "nan", "None"}
        else "",
        "history_log_mode": str(result.get("history_log_mode", "")),
        "history_debug_log_retained": bool(result.get("debug_log_retained", True)),
    }
    return row


def _history_plot_trace_rows(result: dict[str, object]) -> list[dict[str, object]]:
    episode_rows = list(result.get("episode_rows", []))
    primitive_rows = list(result.get("primitive_rows", []))
    if not episode_rows:
        return []
    episode = dict(episode_rows[0])
    return [
        {
            **_history_trace_identity(result),
            "plot_trace_policy": "all_history_selected_primitives_retained_in_primitive_execution_log",
            "primitive_execution_log_scope": "selected_primitives_only_plot_ready",
            "selected_primitive_step_count": int(episode.get("selected_primitive_step_count", len(primitive_rows))),
            "selected_primitive_id_sequence": str(episode.get("selected_primitive_id", "")),
            "selected_primitive_variant_id_sequence": str(episode.get("selected_primitive_variant_id", "")),
            "selected_entry_role_sequence": str(episode.get("selected_entry_role_sequence", "")),
            "selected_start_state_family_sequence": str(episode.get("selected_start_state_family_sequence", "")),
            "selected_route_reason_sequence": str(episode.get("selected_route_reason_sequence", "")),
            "transition_exit_class_sequence": ";".join(_sequence_values(primitive_rows, "transition_exit_class")),
            "termination_cause": str(episode.get("termination_cause", "")),
            "safe_success": bool(_truthy(episode.get("safe_success", False))),
            "full_safe_success": bool(_truthy(episode.get("full_safe_success", False))),
            "terminal_useful": bool(_truthy(episode.get("terminal_useful", False))),
            "lift_capture": bool(_truthy(episode.get("lift_capture", False))),
            "episode_rollout_duration_s": float(_float_value(episode.get("episode_rollout_duration_s", 0.0))),
            "lift_dwell_time_s": float(_float_value(episode.get("lift_dwell_time_s", 0.0))),
            "energy_residual_m": float(_float_value(episode.get("energy_residual_m", 0.0))),
            "positive_specific_energy_gain_m": float(_float_value(episode.get("positive_specific_energy_gain_m", 0.0))),
            "updraft_specific_energy_gain_proxy_m": float(_float_value(episode.get("updraft_specific_energy_gain_proxy_m", 0.0))),
            "min_wall_margin_m": float(_float_value(episode.get("min_wall_margin_m", 0.0))),
        }
    ]


def _history_memory_trace_rows(result: dict[str, object]) -> list[dict[str, object]]:
    episode_rows = list(result.get("episode_rows", []))
    memory_rows = list(result.get("memory_rows", []))
    if not episode_rows:
        return []
    episode = dict(episode_rows[0])
    updated = [row for row in memory_rows if str(row.get("update_status", "")) == "updated"]
    dense_sample_count = int(sum(int(_float_value(row.get("memory_path_sample_count", 1))) for row in memory_rows))
    observation_weight_sum = _memory_row_weight_sum(memory_rows)
    return [
        {
            **_history_trace_identity(result),
            "memory_trace_policy": "one_compact_row_per_history_launch_with_weighted_update_aggregates",
            "memory_update_row_count": int(len(memory_rows)),
            "memory_update_log_scope_sequence": ";".join(_sequence_values(memory_rows, "memory_update_log_scope")),
            "memory_dense_sample_count_total": int(dense_sample_count),
            "memory_observation_weight_sum": float(observation_weight_sum),
            "memory_updated_count": int(len(updated)),
            "memory_not_updated_count": int(len(memory_rows) - len(updated)),
            "belief_update_count_before": int(episode.get("belief_update_count_before", 0)),
            "belief_update_count_after": int(episode.get("belief_update_count_after", 0)),
            "belief_observation_count": int(episode.get("belief_observation_count", 0)),
            "belief_uncertainty": float(_float_value(episode.get("belief_uncertainty", 1.0))),
            "lift_residual_sum_m_s": _memory_weighted_sum(memory_rows, "lift_residual_m_s", "lift_residual_weighted_sum_m_s"),
            "lift_residual_mean_m_s": _memory_weighted_mean(memory_rows, "lift_residual_m_s", "lift_residual_weighted_sum_m_s"),
            "updraft_gain_residual_sum_m": _memory_weighted_sum(memory_rows, "updraft_gain_residual_m", "updraft_gain_residual_weighted_sum_m"),
            "updraft_gain_residual_mean_m": _memory_weighted_mean(memory_rows, "updraft_gain_residual_m", "updraft_gain_residual_weighted_sum_m"),
            "specific_energy_residual_sum_m": _memory_weighted_sum(
                memory_rows,
                "specific_energy_residual_m",
                "specific_energy_residual_weighted_sum_m",
            ),
            "specific_energy_residual_mean_m": _memory_weighted_mean(
                memory_rows,
                "specific_energy_residual_m",
                "specific_energy_residual_weighted_sum_m",
            ),
            "dwell_residual_sum_s": _memory_weighted_sum(memory_rows, "dwell_residual_s", "dwell_residual_weighted_sum_s"),
            "dwell_residual_mean_s": _memory_weighted_mean(memory_rows, "dwell_residual_s", "dwell_residual_weighted_sum_s"),
        }
    ]


def _memory_row_weight(row: dict[str, object]) -> float:
    return _float_value(row.get("observation_weight_sum", row.get("observation_weight", 1.0)), default=1.0)


def _memory_row_weight_sum(rows: list[dict[str, object]]) -> float:
    return float(sum(_memory_row_weight(row) for row in rows))


def _memory_weighted_sum(rows: list[dict[str, object]], value_key: str, weighted_key: str) -> float:
    if not rows:
        return 0.0
    if any(str(row.get(weighted_key, "")).strip() not in {"", "nan", "None"} for row in rows):
        return float(sum(_float_value(row.get(weighted_key, 0.0)) for row in rows))
    return float(sum(_float_value(row.get(value_key, 0.0)) * _memory_row_weight(row) for row in rows))


def _memory_weighted_mean(rows: list[dict[str, object]], value_key: str, weighted_key: str) -> float:
    weight_sum = _memory_row_weight_sum(rows)
    if weight_sum <= 0.0:
        return 0.0
    return float(_memory_weighted_sum(rows, value_key, weighted_key) / weight_sum)


def _history_selector_summary_rows(result: dict[str, object]) -> list[dict[str, object]]:
    episode_rows = list(result.get("episode_rows", []))
    selector_rows = list(result.get("selector_rows", []))
    if not episode_rows:
        return []
    duration = [_float_value(row.get("decision_total_duration_s", 0.0)) for row in selector_rows]
    candidate_count = [int(_float_value(row.get("candidate_count", row.get("decision_candidate_count", 0)))) for row in selector_rows]
    viable_count = [int(_float_value(row.get("viable_count", row.get("decision_viable_count", 0)))) for row in selector_rows]
    return [
        {
            **_history_trace_identity(result),
            "selector_summary_policy": "launch_level_counts_for_history_debug_compaction",
            "selector_decision_count": int(len(selector_rows)),
            "candidate_count_total": int(sum(candidate_count)),
            "viable_count_total": int(sum(viable_count)),
            "governor_rejection_count_total": int(sum(max(0, c - v) for c, v in zip(candidate_count, viable_count))),
            "blocked_no_viable_decision_count": int(sum(str(row.get("decision_status", "")) == "blocked_no_viable_representative" for row in selector_rows)),
            "prepared_before_boundary_count": int(sum(_truthy(row.get("scheduler_prepared_before_primitive_boundary", False)) for row in selector_rows)),
            "preferred_20ms_slot_met_count": int(sum(_truthy(row.get("preferred_20ms_slot_met", False)) for row in selector_rows)),
            "hard_100ms_boundary_met_count": int(sum(_truthy(row.get("hard_100ms_boundary_met", False)) for row in selector_rows)),
            "decision_total_duration_sum_s": float(sum(duration)),
            "decision_total_duration_mean_s": float(sum(duration) / max(1, len(duration))),
            "decision_total_duration_max_s": float(max(duration) if duration else 0.0),
        }
    ]


def _scheduled_active_fan_count_for_outer_case(
    *,
    protocol: ValidationProtocol,
    environment_block_id: str,
    environment_block_local_index: int,
) -> int | None:
    if str(protocol.stage_id) not in CHANGED_CASE_VALIDATION_STAGE_IDS:
        return None
    return _scheduled_active_fan_count_for_block_id(
        environment_block_id=environment_block_id,
        environment_block_local_index=environment_block_local_index,
    )


def _scheduled_active_fan_count_for_block_id(
    *,
    environment_block_id: str,
    environment_block_local_index: int,
) -> int | None:
    block_id = str(environment_block_id)
    if block_id in R10_NO_UPDRAFT_BLOCK_IDS:
        return 0
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
    if block_id in R10_NO_UPDRAFT_BLOCK_IDS:
        return "zero_active_fans_no_updraft"
    if block_id in {ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID, R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID}:
        return "balanced_0_1_2_3_4_for_active_fan_number_variation"
    if block_id == R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID:
        return "balanced_0_1_2_3_4_for_environment_only_full_uncertainty"
    if block_id in R10_FIXED_FOUR_ACTIVE_BLOCK_IDS:
        return "fixed_4_for_four_fan_geometry_non_active_count_block"
    if block_id in R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS:
        return "balanced_0_1_2_3_4_with_arena_wide_fan_position_generalisation"
    if block_id in R10_SINGLE_FAN_GEOMETRY_BLOCK_IDS:
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
    if block_id in R10_NO_UPDRAFT_BLOCK_IDS:
        return "zero_fan_no_updraft"
    if block_id in R10_SINGLE_FAN_GEOMETRY_BLOCK_IDS:
        return "single_fan_geometry"
    if block_id in R10_FOUR_FAN_GEOMETRY_BLOCK_IDS:
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
    if layout == "zero_fan_no_updraft":
        return 0
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
    if block_id in R10_NO_UPDRAFT_BLOCK_IDS:
        return "no_fan_positions"
    if block_id in R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS:
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
    if block_id in R10_NO_UPDRAFT_BLOCK_IDS:
        return "no_fan_positions"
    if block_id in R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS:
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
            paired_start_index = _paired_start_condition_index(
                protocol=protocol,
                environment_block_local_index=local_index,
                outer_case_index=outer_index,
            )
            launch_seed = int(seed) * 100000 + paired_start_index * 37 + 11
            env_seed = int(seed) * 200000 + outer_index * 41 + 17
            environment_layout_seed = env_seed + 1
            environment_active_fan_seed = env_seed + 2
            environment_parameter_seed = env_seed + 3
            plant_implementation_seed = int(seed) * 300000 + outer_index * 43 + 23
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
                    "fan_position_safety_radius_m": (
                        R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M
                        if block.block_id in R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS
                        else ""
                    ),
                    "launch_state_seed": launch_seed,
                    "environment_seed": environment_parameter_seed,
                    "environment_layout_seed": environment_layout_seed,
                    "environment_active_fan_seed": environment_active_fan_seed,
                    "environment_parameter_seed": environment_parameter_seed,
                    "plant_implementation_seed": plant_implementation_seed,
                    "paired_start_condition_index": int(paired_start_index),
                    "paired_start_condition_key": (
                        f"{protocol.final_schedule_prefix}_paired_start_{int(paired_start_index):04d}"
                    ),
                    "paired_start_condition_policy": _paired_start_condition_policy(protocol=protocol),
                    "between_episode_environment_variation": _block_varies_environment_between_history_episodes(block.block_id),
                    "between_episode_environment_parameter_variation": _block_varies_environment_parameters_between_history_episodes(block.block_id),
                    "between_episode_fan_layout_variation": False,
                    "between_episode_active_fan_count_variation": False,
                    "plant_implementation_variation_scope": (
                        "fixed_per_outer_case"
                        if block.block_id in R10_FULL_DOMAIN_RANDOMISATION_BLOCK_IDS
                        else "fixed_nominal_or_deterministic_for_block"
                    ),
                    "common_final_launch_key": f"{protocol.final_schedule_prefix}_final_{outer_index:04d}",
                    "start_state_family": FIRST_PRIMITIVE_START_FAMILY,
                    "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
                    "claim_status": "simulation_only_controlled_outer_case",
                }
            )
            outer_index += 1
    return rows


def _paired_start_condition_index(
    *,
    protocol: ValidationProtocol,
    environment_block_local_index: int,
    outer_case_index: int,
) -> int:
    if str(protocol.stage_id) == "R11":
        block_count = max(1, int(protocol.outer_cases_per_condition) // max(1, len(protocol.blocks)))
        if block_count < 50:
            return int(round(float(environment_block_local_index) * 49.0 / float(max(1, block_count - 1))))
        return int(environment_block_local_index)
    return int(outer_case_index)


def _paired_start_condition_policy(*, protocol: ValidationProtocol) -> str:
    if str(protocol.stage_id) == "R11":
        if int(protocol.outer_cases_per_condition) // max(1, len(protocol.blocks)) < 50:
            return "stratified_local_case_index_spread_over_50_launch_seed_grid_reused_across_l0_l7_library_tiers_and_memory_policies"
        return "same_local_case_index_launch_seed_reused_across_l0_l7_library_tiers_and_memory_policies"
    return "outer_case_unique_launch_seed"


def _final_heldout_schedule(*, outer_cases: list[dict[str, object]], protocol: ValidationProtocol) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    episode_index = 0
    for case_id in LIBRARY_SIZE_CASE_IDS:
        for policy_id in protocol.policy_history_conditions:
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
                        "adaptation_launch_index": int(_policy_condition(policy_id)["history_length"]),
                    }
                )
                episode_index += 1
    return rows


def _history_launch_schedule(*, outer_cases: list[dict[str, object]], protocol: ValidationProtocol) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    history_policies = [
        policy
        for policy in protocol.policy_history_conditions
        if policy.startswith(MEMORY_POLICY_PREFIX) or policy.startswith(SAFE_EXPLORE_POLICY_PREFIX)
    ]
    for case_id in LIBRARY_SIZE_CASE_IDS:
        for policy_id in history_policies:
            history_length = int(_policy_condition(policy_id)["history_length"])
            for outer in outer_cases:
                for history_index in range(history_length):
                    final_like_row = {
                        **outer,
                        "episode_id": (
                            f"{protocol.stage_id.lower()}_{case_id}_{policy_id}_"
                            f"final_{int(outer['outer_case_index']):04d}"
                        ),
                        "launch_role": "final_heldout",
                        "library_size_case_id": case_id,
                        "policy_id": policy_id,
                        "history_length": history_length,
                        "adaptation_launch_index": history_length,
                    }
                    rows.append(_history_row_for_final(final_like_row, history_index))
    return rows


def _history_row_for_final(final_row: dict[str, object], history_index: int) -> dict[str, object]:
    seed_shift = 1000000 + int(history_index) * 101
    block_id = str(final_row.get("environment_block_id", ""))
    environment_seed = int(final_row["environment_seed"])
    if _block_varies_environment_between_history_episodes(block_id):
        environment_seed += seed_shift
    row = {
        **final_row,
        "episode_id": f"{final_row['episode_id']}_hist_{int(history_index):03d}",
        "launch_role": "history",
        "history_launch_index": int(history_index),
        "adaptation_launch_index": int(history_index),
        "launch_state_seed": int(final_row["launch_state_seed"]) + seed_shift,
        "environment_seed": environment_seed,
        "environment_layout_seed": int(final_row.get("environment_layout_seed", final_row["environment_seed"])),
        "environment_active_fan_seed": int(final_row.get("environment_active_fan_seed", final_row["environment_seed"])),
        "environment_parameter_seed": environment_seed,
        "between_episode_environment_variation": _block_varies_environment_between_history_episodes(block_id),
        "between_episode_environment_parameter_variation": _block_varies_environment_parameters_between_history_episodes(block_id),
        "between_episode_fan_layout_variation": False,
        "between_episode_active_fan_count_variation": False,
        "common_final_launch_key": str(final_row["common_final_launch_key"]),
    }
    return row


def _block_varies_environment_between_history_episodes(environment_block_id: str) -> bool:
    block_id = str(environment_block_id)
    if block_id in R10_R11_REALISTIC_REPEATED_LAUNCH_BLOCK_IDS:
        return _block_varies_environment_parameters_between_history_episodes(block_id)
    return block_id not in R10_FIXED_ENVIRONMENT_BETWEEN_HISTORY_BLOCK_IDS


def _block_varies_environment_parameters_between_history_episodes(environment_block_id: str) -> bool:
    return str(environment_block_id) in R10_UPDRAFT_PARAMETER_VARIATION_BETWEEN_HISTORY_BLOCK_IDS


def _block_varies_active_fan_count_between_history_episodes(environment_block_id: str) -> bool:
    del environment_block_id
    return False


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
    if policy_id == OPEN_LOOP_COMPARISON_POLICY_ID:
        return {
            "policy_id": policy_id,
            "policy_family": "open_loop_comparison",
            "history_length": 0,
            "uses_memory": False,
            "updates_memory": False,
            "safe_explore": False,
            "open_loop": True,
            "comparison_only": True,
        }
    if policy_id == "no_memory_baseline":
        return {"policy_id": policy_id, "policy_family": "baseline", "history_length": 0, "uses_memory": False, "updates_memory": False, "safe_explore": False, "open_loop": False, "comparison_only": False}
    if policy_id == EMPTY_FROZEN_PRIOR_BASELINE_ID:
        return {"policy_id": policy_id, "policy_family": "baseline", "history_length": 0, "uses_memory": True, "updates_memory": False, "safe_explore": False, "open_loop": False, "comparison_only": False}
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
                "safe_explore": True,
                "open_loop": False,
                "comparison_only": False,
            }
    raise KeyError(f"unknown policy_id: {policy_id}")


def _initial_belief_for_policy(
    *,
    policy: dict[str, object],
    final_row: dict[str, object],
    governor_config: GovernorConfig | None = None,
) -> DirectionalResidualLiftBelief:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    belief = initial_directional_residual_lift_belief(
        launch_recency_half_life=float(cfg.residual_memory_launch_recency_half_life),
    )
    del final_row
    return belief


def _rollout_backend_for_policy(policy: dict[str, object]) -> str:
    if bool(policy.get("open_loop", False)):
        return "model_backed_open_loop_zero_command"
    return "model_backed_lqr"


def _controller_selection_status_for_policy(policy: dict[str, object], *, protocol: ValidationProtocol) -> str:
    if bool(policy.get("open_loop", False)):
        return f"selected_by_{protocol.stage_id.lower()}_open_loop_zero_command_comparison"
    return f"selected_by_{protocol.stage_id.lower()}_repeated_launch_validator"


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
    retain_debug_logs: bool = True,
) -> dict[str, object]:
    episode_id = str(scheduled["episode_id"])
    sample = archive_state_sample_for_family(
        start_state_family=FIRST_PRIMITIVE_START_FAMILY,
        paired_start_key=str(scheduled.get("paired_start_condition_key", scheduled["common_final_launch_key"])),
        sample_index=int(scheduled.get("paired_start_condition_index", scheduled["outer_case_index"])),
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
    episode_absolute_time_s = 0.0
    episode_command_history_times_s_json = ""
    episode_command_norm_history_json = ""
    prepared_decision: dict[str, object] | None = None
    for primitive_step_index in range(max_steps):
        if (
            prepared_decision is not None
            and int(prepared_decision.get("primitive_step_index", -1)) == int(primitive_step_index)
        ):
            decision = prepared_decision
            prepared_decision = None
            _mark_prepared_decision_committed(decision)
        else:
            decision = _prepare_realtime_governor_decision(
                state=state,
                scheduled=scheduled,
                episode_id=episode_id,
                protocol=protocol,
                primitive_step_index=primitive_step_index,
                policy=policy,
                belief=belief_after,
                representatives=representatives,
                outcome_rows_by_variant_id=outcome_rows_by_variant_id,
                governor_config=governor_config,
                scheduler_decision_source=(
                    "initial_launch_precomputed_before_release"
                    if int(primitive_step_index) == 0
                    else "boundary_compute_no_prepared_decision"
                ),
            )
        route = dict(decision["route"])
        start_state_family = str(decision["start_state_family"])
        governor_mode = str(decision["governor_mode"])
        context_payload = dict(decision["context_payload"])
        context_row = context_payload["row"]
        belief_features = decision["belief_features"]
        selected = decision["selected"]
        candidate_rows = list(decision["candidate_rows"])
        scheduler_fields = dict(decision["scheduler_fields"])
        for row in candidate_rows:
            row.update(_schedule_identity_row(scheduled))
            row.update(scheduler_fields)
            row["launch_role"] = str(scheduled["launch_role"])
            row["primitive_step_index"] = int(primitive_step_index)
            row["launch_sequence_policy"] = LAUNCH_SEQUENCE_POLICY_ID
            row["launch_sequence_phase"] = str(route["launch_sequence_phase"])
            row["route_required_entry_role"] = str(route["route_required_entry_role"])
            row["route_required_entry_class"] = str(route.get("route_required_entry_class", ""))
            row["route_reason"] = str(route["route_reason"])
        if retain_debug_logs:
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
            **scheduler_fields,
            "launch_role": str(scheduled["launch_role"]),
            "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
            "launch_sequence_phase": str(route["launch_sequence_phase"]),
            "route_required_entry_role": str(route["route_required_entry_role"]),
            "route_required_entry_class": str(route.get("route_required_entry_class", "")),
            "route_reason": str(route["route_reason"]),
        }
        selector_rows.append(selector_row)
        if retain_debug_logs:
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
            config=RolloutConfig(
                W_layer=str(scheduled["W_layer"]),
                rollout_backend=_rollout_backend_for_policy(policy),
                absolute_start_time_s=float(episode_absolute_time_s),
                preserve_command_timing_state=True,
                initial_command_history_times_s_json=episode_command_history_times_s_json,
                initial_command_norm_history_json=episode_command_norm_history_json,
                launch_handoff_duration_s=(
                    LAUNCH_HANDOFF_DURATION_S if primitive_step_index == 0 else 0.0
                ),
            ),
            wind_field=context_payload["wind_field"],
            implementation_instance=context_payload["implementation_instance"],
            plant_instance=context_payload["plant_instance"],
            controller=record.controller,
            controller_selection_status=_controller_selection_status_for_policy(policy, protocol=protocol),
            candidate_index=record.candidate_index,
            candidate_weight_label=record.candidate_weight_label,
        )
        rollout_row = rollout_evidence_row(rollout)
        episode_absolute_time_s = float(
            rollout_row.get(
                "rollout_absolute_end_time_s",
                episode_absolute_time_s + float(rollout_row.get("rollout_duration_s", PRIMITIVE_FINITE_HORIZON_S)),
            )
        )
        episode_command_history_times_s_json = str(rollout_row.get("command_history_times_s_json", ""))
        episode_command_norm_history_json = str(rollout_row.get("command_norm_history_json", ""))
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
                **scheduler_fields,
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
        try:
            exit_state_for_memory = as_state_vector(
                np.asarray(json.loads(str(rollout_row.get("exit_state_vector", "[]"))), dtype=float)
            )
        except Exception:
            exit_state_for_memory = None
        observations = _memory_observations_for_rollout_segment(
            start_state=state,
            exit_state=exit_state_for_memory,
            scheduled=scheduled,
            context_row=context_payload["row"],
            rollout_row=rollout_row,
            outcome=outcome,
        )
        belief_before_update = belief_after
        update_status = "not_updated_no_observations" if not observations else "not_updated_policy"
        if bool(policy["updates_memory"]) and observations:
            belief_after = update_directional_residual_lift_belief_batch(
                belief_after,
                (row["observation"] for row in observations),
            )
            update_status = "updated"
        if retain_debug_logs:
            memory_rows.extend(
                _memory_update_detail_rows(
                    scheduled=scheduled,
                    policy=policy,
                    primitive_step_index=primitive_step_index,
                    observations=observations,
                    update_status=update_status,
                    belief_before_update=belief_before_update,
                    belief_after_update=belief_after,
                )
            )
        elif observations:
            memory_rows.append(
                _memory_update_compact_row(
                    scheduled=scheduled,
                    policy=policy,
                    primitive_step_index=primitive_step_index,
                    observations=observations,
                    update_status=update_status,
                    belief_before_update=belief_before_update,
                    belief_after_update=belief_after,
                )
            )
        if exit_state_for_memory is None:
            blocked_reason = "invalid_exit_state_vector"
            break
        state = exit_state_for_memory
        if retain_debug_logs:
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
                        current_history_launch_index=_adaptation_launch_index(scheduled),
                    ),
                )
            )
        exit_class = str(rollout_row.get("transition_exit_class", "hard_failure"))
        hard_failure = exit_class == "hard_failure" or _rollout_row_is_hard_failure(rollout_row)
        if hard_failure or exit_class == "safe_terminal":
            break
        next_step_index = int(primitive_step_index) + 1
        if next_step_index < int(max_steps):
            prepared_decision = _prepare_realtime_governor_decision(
                state=state,
                scheduled=scheduled,
                episode_id=episode_id,
                protocol=protocol,
                primitive_step_index=next_step_index,
                policy=policy,
                belief=belief_after,
                representatives=representatives,
                outcome_rows_by_variant_id=outcome_rows_by_variant_id,
                governor_config=governor_config,
                scheduler_decision_source="prepared_during_previous_primitive_window",
            )
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
        "scheduled": scheduled,
        "debug_log_retained": bool(retain_debug_logs),
        "history_log_mode": _resolved_history_log_mode(config, protocol),
    }


def _candidate_path_belief_features_fn(
    *,
    belief: DirectionalResidualLiftBelief,
    state: np.ndarray,
    current_history_launch_index: int,
    use_residual_memory: bool,
    governor_config: GovernorConfig | None = None,
    real_time_controller_mode: bool = False,
) -> Callable[[dict[str, object], dict[str, object]], dict[str, object]]:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    state_vector = as_state_vector(state)
    cell_lookup = directional_residual_lift_cell_lookup(belief) if use_residual_memory else {}
    spatial_cell_lookup = directional_residual_lift_spatial_cell_lookup(belief) if use_residual_memory else {}
    query_cache: dict[tuple[int, int, int, int, int, int], dict[str, object]] = {}

    def features_for_candidate(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
        memory_query_mode = str(representative.get("__memory_query_mode", "full"))
        full_memory_query = bool(use_residual_memory and memory_query_mode == "full")
        return _candidate_path_belief_features(
            belief=belief,
            cell_lookup=cell_lookup,
            spatial_cell_lookup=spatial_cell_lookup,
            query_cache=query_cache,
            state=state_vector,
            representative=representative,
            outcome=outcome,
            current_history_launch_index=current_history_launch_index,
            use_residual_memory=full_memory_query,
            governor_config=cfg,
            real_time_controller_mode=real_time_controller_mode,
        )

    return features_for_candidate


def _memory_observations_for_rollout_segment(
    *,
    start_state: np.ndarray,
    exit_state: np.ndarray | None,
    scheduled: dict[str, object],
    context_row: dict[str, object],
    rollout_row: dict[str, object],
    outcome: dict[str, object],
) -> list[dict[str, object]]:
    """Distribute one primitive-level utility residual densely along the executed 3D segment."""

    start = as_state_vector(start_state)
    if exit_state is None:
        return []
    exit_vector = as_state_vector(exit_state)
    start_xyz = np.array(
        [start[STATE_INDEX["x_w"]], start[STATE_INDEX["y_w"]], start[STATE_INDEX["z_w"]]],
        dtype=float,
    )
    exit_xyz = np.array(
        [exit_vector[STATE_INDEX["x_w"]], exit_vector[STATE_INDEX["y_w"]], exit_vector[STATE_INDEX["z_w"]]],
        dtype=float,
    )
    distance_m = float(np.linalg.norm(exit_xyz - start_xyz))
    if distance_m <= 1e-6:
        sample_count = 1
    else:
        sample_count = int(
            max(
                2,
                min(
                    FLOW_BELIEF_HISTORY_UPDATE_MAX_SAMPLES_PER_PRIMITIVE,
                    math.ceil(distance_m / max(FLOW_BELIEF_HISTORY_UPDATE_SPACING_M, 1e-9)) + 1,
                ),
            )
        )
    lift_residual = _lift_residual_for_memory_update(context_row, rollout_row=rollout_row, outcome=outcome)
    updraft_residual = _updraft_gain_residual_for_memory_update(context_row, rollout_row=rollout_row, outcome=outcome)
    dwell_residual = float(rollout_row.get("lift_dwell_time_s", 0.0)) - float(outcome.get("expected_lift_dwell_time_s", 0.0))
    specific_energy_residual = _specific_energy_residual_for_memory_update(rollout_row=rollout_row, outcome=outcome)
    if distance_m > 1e-6:
        segment_direction = math.atan2(float(exit_xyz[1] - start_xyz[1]), float(exit_xyz[0] - start_xyz[0]))
    else:
        segment_direction = float(start[STATE_INDEX["psi"]])
    rows: list[dict[str, object]] = []
    for sample_index in range(sample_count):
        fraction = 0.0 if sample_count <= 1 else float(sample_index) / float(sample_count - 1)
        xyz = (1.0 - fraction) * start_xyz + fraction * exit_xyz
        rows.append(
            {
                "sample_index": int(sample_index),
                "sample_count": int(sample_count),
                "sample_fraction": float(fraction),
                "sample_source": "executed_primitive_start_to_exit_segment_dense_arc_length",
                "observation": DirectionalResidualObservation(
                    x_w_m=float(xyz[0]),
                    y_w_m=float(xyz[1]),
                    z_w_m=float(xyz[2]),
                    direction_rad=float(segment_direction),
                    lift_residual_m_s=float(lift_residual),
                    updraft_gain_residual_m=float(updraft_residual),
                    dwell_residual_s=float(dwell_residual),
                    specific_energy_residual_m=float(specific_energy_residual),
                    observation_weight=1.0 / float(sample_count),
                    history_launch_index=_adaptation_launch_index(scheduled),
                ),
            }
        )
    return rows


def _memory_update_detail_rows(
    *,
    scheduled: dict[str, object],
    policy: dict[str, object],
    primitive_step_index: int,
    observations: list[dict[str, object]],
    update_status: str,
    belief_before_update: DirectionalResidualLiftBelief,
    belief_after_update: DirectionalResidualLiftBelief,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start_update_count = int(belief_before_update.update_count)
    updated = str(update_status) == "updated"
    for local_index, observation_row in enumerate(observations):
        observation = observation_row["observation"]
        if not isinstance(observation, DirectionalResidualObservation):
            continue
        before_count = start_update_count + int(local_index) if updated else start_update_count
        after_count = start_update_count + int(local_index) + 1 if updated else start_update_count
        rows.append(
            {
                **_memory_update_common_fields(
                    scheduled=scheduled,
                    policy=policy,
                    primitive_step_index=primitive_step_index,
                    update_status=update_status,
                    belief_before_update_count=before_count,
                    belief_after_update_count=after_count,
                ),
                "memory_update_log_scope": "dense_sample_debug",
                "memory_dense_sample_rows_retained": True,
                "memory_path_sample_index": int(observation_row["sample_index"]),
                "memory_path_sample_count": int(observation_row["sample_count"]),
                "memory_path_sample_fraction": float(observation_row["sample_fraction"]),
                "memory_path_sample_spacing_m": float(FLOW_BELIEF_HISTORY_UPDATE_SPACING_M),
                "memory_path_sample_max_per_primitive": int(FLOW_BELIEF_HISTORY_UPDATE_MAX_SAMPLES_PER_PRIMITIVE),
                "memory_path_sample_source": str(observation_row["sample_source"]),
                **asdict(observation),
                **_memory_observation_weighted_fields(observation),
            }
        )
    if rows:
        rows[-1]["belief_update_count_after_batch"] = int(belief_after_update.update_count)
    return rows


def _memory_update_compact_row(
    *,
    scheduled: dict[str, object],
    policy: dict[str, object],
    primitive_step_index: int,
    observations: list[dict[str, object]],
    update_status: str,
    belief_before_update: DirectionalResidualLiftBelief,
    belief_after_update: DirectionalResidualLiftBelief,
) -> dict[str, object]:
    observation_values = [
        row["observation"]
        for row in observations
        if isinstance(row.get("observation"), DirectionalResidualObservation)
    ]
    weight_sum = float(sum(float(obs.observation_weight) for obs in observation_values))
    weight_denominator = max(1e-9, weight_sum)
    sample_fractions = [float(row.get("sample_fraction", 0.0)) for row in observations]
    sample_sources = sorted({str(row.get("sample_source", "")) for row in observations if str(row.get("sample_source", ""))})
    return {
        **_memory_update_common_fields(
            scheduled=scheduled,
            policy=policy,
            primitive_step_index=primitive_step_index,
            update_status=update_status,
            belief_before_update_count=int(belief_before_update.update_count),
            belief_after_update_count=int(belief_after_update.update_count),
        ),
        "memory_update_log_scope": "compact_primitive_summary_no_dense_sample_rows",
        "memory_dense_sample_rows_retained": False,
        "memory_dense_sample_logging_policy": "dense_rows_suppressed_for_history_plot_summary",
        "memory_path_sample_index": -1,
        "memory_path_sample_count": int(len(observation_values)),
        "memory_path_sample_fraction": float(sum(sample_fractions) / max(1, len(sample_fractions))),
        "memory_path_sample_fraction_min": float(min(sample_fractions) if sample_fractions else 0.0),
        "memory_path_sample_fraction_max": float(max(sample_fractions) if sample_fractions else 0.0),
        "memory_path_sample_spacing_m": float(FLOW_BELIEF_HISTORY_UPDATE_SPACING_M),
        "memory_path_sample_max_per_primitive": int(FLOW_BELIEF_HISTORY_UPDATE_MAX_SAMPLES_PER_PRIMITIVE),
        "memory_path_sample_source": ";".join(sample_sources),
        "observation_weight_sum": float(weight_sum),
        "history_launch_index": int(observation_values[-1].history_launch_index) if observation_values else 0,
        "lift_residual_m_s": float(
            sum(float(obs.lift_residual_m_s) * float(obs.observation_weight) for obs in observation_values)
            / weight_denominator
        ),
        "updraft_gain_residual_m": float(
            sum(float(obs.updraft_gain_residual_m) * float(obs.observation_weight) for obs in observation_values)
            / weight_denominator
        ),
        "dwell_residual_s": float(
            sum(float(obs.dwell_residual_s) * float(obs.observation_weight) for obs in observation_values)
            / weight_denominator
        ),
        "specific_energy_residual_m": float(
            sum(float(obs.specific_energy_residual_m) * float(obs.observation_weight) for obs in observation_values)
            / weight_denominator
        ),
        "lift_residual_weighted_sum_m_s": float(
            sum(float(obs.lift_residual_m_s) * float(obs.observation_weight) for obs in observation_values)
        ),
        "updraft_gain_residual_weighted_sum_m": float(
            sum(float(obs.updraft_gain_residual_m) * float(obs.observation_weight) for obs in observation_values)
        ),
        "dwell_residual_weighted_sum_s": float(
            sum(float(obs.dwell_residual_s) * float(obs.observation_weight) for obs in observation_values)
        ),
        "specific_energy_residual_weighted_sum_m": float(
            sum(float(obs.specific_energy_residual_m) * float(obs.observation_weight) for obs in observation_values)
        ),
        "x_w_m_min": float(min((obs.x_w_m for obs in observation_values), default=0.0)),
        "x_w_m_max": float(max((obs.x_w_m for obs in observation_values), default=0.0)),
        "y_w_m_min": float(min((obs.y_w_m for obs in observation_values), default=0.0)),
        "y_w_m_max": float(max((obs.y_w_m for obs in observation_values), default=0.0)),
        "z_w_m_min": float(min((obs.z_w_m for obs in observation_values), default=0.0)),
        "z_w_m_max": float(max((obs.z_w_m for obs in observation_values), default=0.0)),
        "direction_rad_mean": float(
            sum(float(obs.direction_rad) * float(obs.observation_weight) for obs in observation_values)
            / weight_denominator
        ),
    }


def _memory_update_common_fields(
    *,
    scheduled: dict[str, object],
    policy: dict[str, object],
    primitive_step_index: int,
    update_status: str,
    belief_before_update_count: int,
    belief_after_update_count: int,
) -> dict[str, object]:
    return {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_id": str(policy["policy_id"]),
        "primitive_step_index": int(primitive_step_index),
        "update_status": str(update_status),
        "memory_update_policy": FLOW_BELIEF_HISTORY_UPDATE_POLICY,
        "belief_update_count_before": int(belief_before_update_count),
        "belief_update_count_after": int(belief_after_update_count),
    }


def _memory_observation_weighted_fields(observation: DirectionalResidualObservation) -> dict[str, float]:
    weight = float(observation.observation_weight)
    return {
        "lift_residual_weighted_sum_m_s": float(observation.lift_residual_m_s) * weight,
        "updraft_gain_residual_weighted_sum_m": float(observation.updraft_gain_residual_m) * weight,
        "dwell_residual_weighted_sum_s": float(observation.dwell_residual_s) * weight,
        "specific_energy_residual_weighted_sum_m": float(observation.specific_energy_residual_m) * weight,
    }


def _candidate_path_belief_features(
    *,
    belief: DirectionalResidualLiftBelief,
    cell_lookup: dict[tuple[int, int, int, int], DirectionalResidualCell],
    spatial_cell_lookup: dict[tuple[int, int, int], tuple[DirectionalResidualCell, ...]] | None = None,
    query_cache: dict[tuple[int, int, int, int, int, int], dict[str, object]] | None = None,
    state: np.ndarray,
    representative: dict[str, object],
    outcome: dict[str, object],
    current_history_launch_index: int,
    use_residual_memory: bool,
    governor_config: GovernorConfig | None = None,
    real_time_controller_mode: bool = False,
) -> dict[str, object]:
    """Return candidate path geometry plus optional spatial flow-belief correction."""

    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    path_probes = REAL_TIME_CANDIDATE_PATH_MEMORY_PROBES if real_time_controller_mode else CANDIDATE_PATH_MEMORY_PROBES
    query_radius_m = REAL_TIME_FLOW_BELIEF_QUERY_RADIUS_M if real_time_controller_mode else FLOW_BELIEF_QUERY_RADIUS_M
    reachable_probe_set = (
        REAL_TIME_FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES
        if real_time_controller_mode
        else FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES
    )
    route_probe_set = (
        REAL_TIME_FLOW_BELIEF_ROUTE_PROBE_FRACTIONS
        if real_time_controller_mode
        else FLOW_BELIEF_ROUTE_PROBE_FRACTIONS
    )
    reference_state = _candidate_reference_state_vector(representative, outcome)
    path_speed_m_s = _candidate_path_speed_m_s(state=state, representative=representative, reference_state=reference_state)
    vertical_speed_m_s = _candidate_path_vertical_speed_m_s(state=state, reference_state=reference_state)
    calibrated_regime = (
        calibrated_regime_risk_features(
            u_m_s=reference_state[STATE_INDEX["u"]],
            w_m_s=reference_state[STATE_INDEX["w"]],
            governor_config=cfg,
            alpha_source="reference_state_vector",
        )
        if reference_state
        else calibrated_regime_risk_features(
            path_speed_m_s=path_speed_m_s,
            vertical_speed_m_s=vertical_speed_m_s,
            governor_config=cfg,
            alpha_source="candidate_path_vertical_speed_over_speed",
        )
    )
    reference_bank_rad = _candidate_reference_bank_rad(representative=representative, reference_state=reference_state)
    heading_offset_rad = _candidate_path_heading_offset_rad(
        reference_bank_rad=reference_bank_rad,
        path_speed_m_s=path_speed_m_s,
    )
    x0 = float(state[STATE_INDEX["x_w"]])
    y0 = float(state[STATE_INDEX["y_w"]])
    z0 = float(state[STATE_INDEX["z_w"]])
    psi0 = float(state[STATE_INDEX["psi"]])
    weighted_lift = 0.0
    weighted_updraft = 0.0
    weighted_specific_energy = 0.0
    weighted_dwell = 0.0
    confidence = 0.0
    weighted_uncertainty = 0.0
    uncertainty_weight_sum = 0.0
    low_confidence_probe_count = 0
    observation_count = 0
    last_features: dict[str, object] = {}
    exit_x = x0
    exit_y = y0
    exit_z = z0
    exit_direction = psi0
    for fraction, weight in path_probes:
        fraction = float(fraction)
        weight = float(weight)
        probe_time_s = float(CANDIDATE_PATH_MEMORY_LOOKAHEAD_S) * fraction
        probe_direction = psi0 + heading_offset_rad * fraction
        displacement_direction = psi0 + 0.5 * heading_offset_rad * fraction
        distance_m = path_speed_m_s * probe_time_s
        x_w_m = _clamp_to_bounds(x0 + math.cos(displacement_direction) * distance_m, TRUE_SAFE_BOUNDS.x_w_m)
        y_w_m = _clamp_to_bounds(y0 + math.sin(displacement_direction) * distance_m, TRUE_SAFE_BOUNDS.y_w_m)
        z_w_m = _clamp_to_bounds(z0 + vertical_speed_m_s * probe_time_s, TRUE_SAFE_BOUNDS.z_w_m)
        if use_residual_memory:
            last_features = _cached_spatial_flow_belief_features(
                belief,
                x_w_m=x_w_m,
                y_w_m=y_w_m,
                z_w_m=z_w_m,
                direction_rad=probe_direction,
                cell_lookup=cell_lookup,
                spatial_cell_lookup=spatial_cell_lookup,
                current_history_launch_index=current_history_launch_index,
                query_radius_m=query_radius_m,
                query_cache=query_cache,
            )
            probe_confidence = _candidate_path_probe_confidence(last_features, governor_config=cfg)
            weighted_lift += weight * probe_confidence * _float_value(last_features.get("belief_local_lift_residual_m_s", 0.0))
            weighted_updraft += weight * probe_confidence * _float_value(last_features.get("belief_local_updraft_gain_residual_m", 0.0))
            weighted_specific_energy += weight * probe_confidence * _float_value(
                last_features.get(
                    "belief_local_specific_energy_residual_m",
                    last_features.get("belief_local_energy_residual_m", 0.0),
                )
            )
            weighted_dwell += weight * probe_confidence * _float_value(last_features.get("belief_local_dwell_residual_s", 0.0))
            confidence += weight * probe_confidence
            weighted_uncertainty += weight * _clip(1.0 - probe_confidence, 0.0, 1.0)
            uncertainty_weight_sum += weight
            if probe_confidence < 0.5:
                low_confidence_probe_count += 1
            observation_count += int(_float_value(last_features.get("belief_observation_count", 0)))
        exit_x = x_w_m
        exit_y = y_w_m
        exit_z = z_w_m
        exit_direction = probe_direction
    capped_updraft = _clip(
        weighted_updraft,
        -float(cfg.candidate_path_memory_residual_cap_m),
        float(cfg.candidate_path_memory_residual_cap_m),
    )
    capped_lift = _clip(
        weighted_lift,
        -float(cfg.candidate_path_memory_residual_cap_m),
        float(cfg.candidate_path_memory_residual_cap_m),
    )
    capped_dwell = _clip(weighted_dwell, -1.0, 1.0)
    capped_specific_energy = _clip(
        weighted_specific_energy,
        -float(cfg.candidate_path_memory_specific_energy_residual_cap_m),
        float(cfg.candidate_path_memory_specific_energy_residual_cap_m),
    )
    specific_energy_weight = float(cfg.candidate_path_memory_utility_specific_energy_weight)
    updraft_weight = float(cfg.candidate_path_memory_utility_updraft_weight)
    weight_sum = max(1e-9, abs(specific_energy_weight) + abs(updraft_weight))
    specific_energy_weight = specific_energy_weight / weight_sum
    updraft_weight = updraft_weight / weight_sum
    path_memory_utility = (
        float(specific_energy_weight) * float(capped_specific_energy)
        + float(updraft_weight) * float(capped_updraft)
    )
    confidence = _clip(confidence, 0.0, 1.0)
    path_uncertainty = (
        _clip(weighted_uncertainty / max(1e-9, uncertainty_weight_sum), 0.0, 1.0)
        if use_residual_memory
        else 0.0
    )
    reachable_attraction = (
        _candidate_reachable_flow_attraction(
            belief=belief,
            cell_lookup=cell_lookup,
            spatial_cell_lookup=spatial_cell_lookup,
            query_cache=query_cache,
            exit_x=exit_x,
            exit_y=exit_y,
            exit_z=exit_z,
            exit_direction=exit_direction,
            current_history_launch_index=current_history_launch_index,
            use_residual_memory=use_residual_memory,
            governor_config=cfg,
            specific_energy_weight=specific_energy_weight,
            updraft_weight=updraft_weight,
            query_radius_m=query_radius_m,
            probe_set=reachable_probe_set,
        )
        if use_residual_memory
        else _empty_reachable_flow_attraction()
    )
    route_value = (
        _candidate_route_flow_value(
            belief=belief,
            cell_lookup=cell_lookup,
            spatial_cell_lookup=spatial_cell_lookup,
            query_cache=query_cache,
            x0=x0,
            exit_x=exit_x,
            exit_y=exit_y,
            exit_z=exit_z,
            exit_direction=exit_direction,
            path_speed_m_s=path_speed_m_s,
            current_history_launch_index=current_history_launch_index,
            use_residual_memory=use_residual_memory,
            governor_config=cfg,
            specific_energy_weight=specific_energy_weight,
            updraft_weight=updraft_weight,
            query_radius_m=query_radius_m,
            probe_set=route_probe_set,
        )
        if use_residual_memory
        else _empty_route_flow_value()
    )
    memory_utility = _clip(
        float(path_memory_utility)
        + float(reachable_attraction["capped_attraction_m"])
        + float(route_value["exploitation_m"]),
        -float(cfg.candidate_path_memory_specific_energy_residual_cap_m),
        float(cfg.candidate_path_memory_specific_energy_residual_cap_m),
    )
    map_guided_uncertainty = _memory_guided_exploration_uncertainty(
        confidence=confidence,
        x0=x0,
        exit_x=exit_x,
        exit_y=exit_y,
        exit_z=exit_z,
    )
    information_gain = _flow_map_information_gain(
        path_uncertainty=path_uncertainty,
        reachable_uncertainty=max(
            float(reachable_attraction["mean_uncertainty"]),
            float(route_value["information_gain"]),
        ),
        x0=x0,
        exit_x=exit_x,
        exit_y=exit_y,
        exit_z=exit_z,
    )
    map_guided_uncertainty = max(float(map_guided_uncertainty), float(information_gain["information_gain"]))
    exit_margins = position_margin_m(np.array([exit_x, exit_y, exit_z], dtype=float), TRUE_SAFE_BOUNDS)
    return {
        "belief_version": (
            f"{belief.belief_version}+{OUTER_LOOP_MEMORY_POLICY_VERSION}"
            if use_residual_memory
            else "candidate_path_geometry_only_no_residual_memory"
        ),
        "belief_local_lift_m_s": float(capped_lift),
        "belief_local_lift_residual_m_s": float(capped_lift),
        "belief_local_updraft_gain_proxy_m": max(float(capped_updraft), 0.0),
        "belief_local_updraft_gain_residual_m": float(capped_updraft),
        "belief_local_energy_residual_m": float(memory_utility),
        "belief_local_specific_energy_residual_m": float(capped_specific_energy),
        "belief_local_dwell_residual_s": float(capped_dwell),
        "belief_uncertainty": float(map_guided_uncertainty),
        "belief_observation_count": int(observation_count),
        "belief_effective_observation_count": float(observation_count) * float(confidence),
        "belief_recency_weight": float(last_features.get("belief_recency_weight", 0.0) or 0.0),
        "belief_observation_age": int(_float_value(last_features.get("belief_observation_age", 0))),
        "belief_launch_recency_weight": float(last_features.get("belief_launch_recency_weight", 0.0) or 0.0),
        "belief_history_launch_age": int(_float_value(last_features.get("belief_history_launch_age", 0))),
        "belief_last_history_launch_index": int(_float_value(last_features.get("belief_last_history_launch_index", -1))),
        "belief_launch_recency_half_life": float(cfg.residual_memory_launch_recency_half_life),
        "belief_direction_bin": int(_float_value(last_features.get("belief_direction_bin", 0))),
        "belief_z_bin": int(_float_value(last_features.get("belief_z_bin", 0))),
        "belief_update_count": int(belief.update_count),
        "belief_current_history_launch_index": int(current_history_launch_index),
        "belief_memory_policy_version": OUTER_LOOP_MEMORY_POLICY_VERSION if use_residual_memory else "",
        "belief_candidate_path_residual_memory_active": bool(use_residual_memory),
        "belief_candidate_path_probe_count": int(len(path_probes)),
        "belief_candidate_path_lookahead_s": float(CANDIDATE_PATH_MEMORY_LOOKAHEAD_S),
        "belief_candidate_path_confidence": float(confidence),
        "belief_candidate_path_updraft_residual_uncapped_m": float(weighted_updraft),
        "belief_candidate_path_updraft_residual_cap_m": float(cfg.candidate_path_memory_residual_cap_m),
        "belief_candidate_path_specific_energy_residual_uncapped_m": float(weighted_specific_energy),
        "belief_candidate_path_specific_energy_residual_cap_m": float(
            cfg.candidate_path_memory_specific_energy_residual_cap_m
        ),
        "belief_candidate_path_memory_utility_without_attraction_m": float(path_memory_utility),
        "belief_candidate_path_memory_utility_m": float(memory_utility),
        "belief_candidate_path_memory_utility_policy": (
            f"spatial_flow_map_specific_energy_{specific_energy_weight:.2f}_plus_updraft_{updraft_weight:.2f}_plus_reachable_attraction"
        ),
        "belief_flow_map_grid_resolution_m": float(FLOW_BELIEF_GRID_RESOLUTION_M),
        "belief_flow_map_query_radius_m": float(query_radius_m),
        "belief_flow_map_controller_query_mode": (
            "real_time_full_memory_probe_set"
            if real_time_controller_mode
            else "full_online_and_diagnostic_probe_set"
        ),
        "belief_flow_map_controller_path_probe_count": int(len(path_probes)),
        "belief_flow_map_controller_reachable_probe_count": int(len(reachable_probe_set)),
        "belief_flow_map_controller_route_probe_count": int(len(route_probe_set)),
        "belief_flow_map_reachable_attraction_m": float(reachable_attraction["capped_attraction_m"]),
        "belief_flow_map_reachable_attraction_raw_m": float(reachable_attraction["raw_attraction_m"]),
        "belief_flow_map_reachable_attraction_cap_m": float(FLOW_BELIEF_REACHABLE_ATTRACTION_CAP_M),
        "belief_flow_map_reachable_attraction_confidence": float(reachable_attraction["confidence"]),
        "belief_flow_map_reachable_attraction_query_count": int(reachable_attraction["query_count"]),
        "belief_flow_map_reachable_attraction_observation_count": int(reachable_attraction["observation_count"]),
        "belief_flow_map_reachable_attraction_best_x_w_m": float(reachable_attraction["best_x_w_m"]),
        "belief_flow_map_reachable_attraction_best_y_w_m": float(reachable_attraction["best_y_w_m"]),
        "belief_flow_map_reachable_attraction_best_z_w_m": float(reachable_attraction["best_z_w_m"]),
        "belief_flow_map_reachable_attraction_lookahead_m": float(FLOW_BELIEF_REACHABLE_ATTRACTION_LOOKAHEAD_M),
        "belief_flow_map_reachable_attraction_half_angle_rad": float(FLOW_BELIEF_REACHABLE_ATTRACTION_HALF_ANGLE_RAD),
        "belief_flow_map_reachable_attraction_azimuth_half_angle_rad": float(
            FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD
        ),
        "belief_flow_map_reachable_attraction_elevation_half_angle_rad": float(
            FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD
        ),
        "belief_flow_map_reachable_attraction_geometry": (
            "compact_real_time_sparse_3d_cone" if real_time_controller_mode else "full_sparse_3d_cone_2_range_3_azimuth_3_elevation_stencil"
        ),
        "belief_flow_map_candidate_path_uncertainty": float(path_uncertainty),
        "belief_flow_map_memory_guided_exploration_uncertainty": float(map_guided_uncertainty),
        "belief_flow_map_information_gain": float(information_gain["information_gain"]),
        "belief_flow_map_information_gain_path_uncertainty": float(path_uncertainty),
        "belief_flow_map_information_gain_reachable_uncertainty": float(reachable_attraction["mean_uncertainty"]),
        "belief_flow_map_information_gain_progress_gate": float(information_gain["progress_gate"]),
        "belief_flow_map_information_gain_safe_gate": float(information_gain["safe_gate"]),
        "belief_flow_map_information_gain_query_count": int(
            (len(path_probes) + int(reachable_attraction["query_count"]))
            if use_residual_memory
            else 0
        ),
        "belief_flow_map_information_gain_low_confidence_query_count": int(
            (low_confidence_probe_count + int(reachable_attraction["low_confidence_query_count"]))
            if use_residual_memory
            else 0
        ),
        "belief_flow_map_exploration_scale": 1.0,
        "belief_flow_map_policy": "0p1m_spatial_updraft_utility_map_with_path_updates_reachable_attraction_and_information_gain",
        "belief_flow_map_route_policy": (
            "bounded_short_horizon_route_value_from_candidate_exit_sparse_flow_map_probes"
        ),
        "belief_flow_map_route_horizon_primitives": int(route_value["horizon_primitives"]),
        "belief_flow_map_route_probe_count": int(route_value["probe_count"]),
        "belief_flow_map_route_exploitation_m": float(route_value["exploitation_m"]),
        "belief_flow_map_route_information_gain": float(route_value["information_gain"]),
        "belief_flow_map_route_confidence": float(route_value["confidence"]),
        "belief_flow_map_route_uncertainty": float(route_value["uncertainty"]),
        "belief_flow_map_route_front_progress": float(route_value["front_progress"]),
        "belief_flow_map_route_safe_fraction": float(route_value["safe_fraction"]),
        "belief_flow_map_route_best_x_w_m": float(route_value["best_x_w_m"]),
        "belief_flow_map_route_best_y_w_m": float(route_value["best_y_w_m"]),
        "belief_flow_map_route_best_z_w_m": float(route_value["best_z_w_m"]),
        "belief_candidate_path_reference_bank_rad": float(reference_bank_rad),
        "belief_candidate_path_heading_offset_rad": float(heading_offset_rad),
        "belief_candidate_path_speed_m_s": float(path_speed_m_s),
        "belief_candidate_path_vertical_speed_m_s": float(vertical_speed_m_s),
        "belief_candidate_path_alpha_proxy_deg": float(calibrated_regime["calibrated_regime_alpha_proxy_deg"]),
        "belief_candidate_path_alpha_abs_deg": float(calibrated_regime["calibrated_regime_alpha_abs_deg"]),
        "belief_candidate_path_calibrated_regime_transition_start_alpha_deg": float(
            calibrated_regime["calibrated_regime_transition_start_alpha_deg"]
        ),
        "belief_candidate_path_calibrated_regime_post_stall_alpha_deg": float(
            calibrated_regime["calibrated_regime_post_stall_alpha_deg"]
        ),
        "belief_candidate_path_calibrated_regime_source_calibration_id": str(
            calibrated_regime["calibrated_regime_source_calibration_id"]
        ),
        "belief_candidate_path_calibrated_regime_label": str(calibrated_regime["calibrated_regime_label"]),
        "belief_candidate_path_calibrated_transition_activation": float(
            calibrated_regime["calibrated_transition_activation"]
        ),
        "belief_candidate_path_calibrated_post_stall_activation": float(
            calibrated_regime["calibrated_post_stall_activation"]
        ),
        "belief_candidate_path_calibrated_regime_mismatch_risk": float(
            calibrated_regime["calibrated_regime_mismatch_risk"]
        ),
        "belief_candidate_path_exit_x_w_m": float(exit_x),
        "belief_candidate_path_exit_y_w_m": float(exit_y),
        "belief_candidate_path_exit_z_w_m": float(exit_z),
        "belief_candidate_path_exit_direction_rad": float(exit_direction),
        "belief_candidate_path_exit_wall_margin_m": float(exit_margins["min_wall_margin_m"]),
        "belief_candidate_path_exit_min_margin_m": float(exit_margins["min_margin_m"]),
    }


def _candidate_reference_state_vector(
    representative: dict[str, object],
    outcome: dict[str, object],
) -> list[float]:
    for source in (representative, outcome):
        raw = source.get("reference_state_vector", "")
        if raw in ("", None):
            continue
        try:
            values = json.loads(str(raw)) if isinstance(raw, str) else list(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if len(vector) > max(STATE_INDEX["r"], STATE_INDEX["w"]):
            return vector
    return []


def _candidate_path_speed_m_s(
    *,
    state: np.ndarray,
    representative: dict[str, object],
    reference_state: list[float],
) -> float:
    reference_speed = _float_value(representative.get("local_lqr_reference_speed_m_s", float("nan")), default=float("nan"))
    if math.isfinite(reference_speed) and reference_speed > 0.0:
        return _clip(reference_speed, 1.5, 9.0)
    if reference_state:
        u_ref = reference_state[STATE_INDEX["u"]]
        v_ref = reference_state[STATE_INDEX["v"]]
        w_ref = reference_state[STATE_INDEX["w"]]
        speed = math.sqrt(u_ref * u_ref + v_ref * v_ref + w_ref * w_ref)
        if math.isfinite(speed) and speed > 0.0:
            return _clip(speed, 1.5, 9.0)
    u = float(state[STATE_INDEX["u"]])
    v = float(state[STATE_INDEX["v"]])
    w = float(state[STATE_INDEX["w"]])
    return _clip(math.sqrt(u * u + v * v + w * w), 1.5, 9.0)


def _candidate_path_vertical_speed_m_s(*, state: np.ndarray, reference_state: list[float]) -> float:
    if reference_state:
        return _clip(reference_state[STATE_INDEX["w"]], -0.8, 0.8)
    return _clip(float(state[STATE_INDEX["w"]]), -0.8, 0.8)


def _candidate_reference_bank_rad(
    *,
    representative: dict[str, object],
    reference_state: list[float],
) -> float:
    if reference_state:
        bank = float(reference_state[STATE_INDEX["phi"]])
        if abs(bank) > 1e-6:
            return _clip(bank, -0.45, 0.45)
    primitive_id = str(representative.get("primitive_id", ""))
    default_bank = {
        "mild_turn_left": -0.20,
        "mild_turn_right": 0.20,
        "lift_dwell_arc": 0.22,
        "energy_retaining_bank": 0.16,
    }.get(primitive_id, 0.0)
    return float(default_bank)


def _candidate_path_heading_offset_rad(*, reference_bank_rad: float, path_speed_m_s: float) -> float:
    if abs(float(reference_bank_rad)) < 1e-9:
        return 0.0
    yaw_rate_rad_s = SPECIFIC_ENERGY_GRAVITY_M_S2 * math.tan(_clip(reference_bank_rad, -0.45, 0.45)) / max(1.5, float(path_speed_m_s))
    return _clip(
        yaw_rate_rad_s * float(CANDIDATE_PATH_MEMORY_LOOKAHEAD_S),
        -float(CANDIDATE_PATH_MEMORY_HEADING_OFFSET_CAP_RAD),
        float(CANDIDATE_PATH_MEMORY_HEADING_OFFSET_CAP_RAD),
    )


def _cached_spatial_flow_belief_features(
    belief: DirectionalResidualLiftBelief,
    *,
    x_w_m: float,
    y_w_m: float,
    z_w_m: float,
    direction_rad: float,
    cell_lookup: dict[tuple[int, int, int, int], DirectionalResidualCell] | dict[tuple[int, int, int, int], object],
    spatial_cell_lookup: dict[tuple[int, int, int], tuple[DirectionalResidualCell, ...]] | None = None,
    current_history_launch_index: int,
    query_radius_m: float,
    query_cache: dict[tuple[int, int, int, int, int, int], dict[str, object]] | None,
) -> dict[str, object]:
    if query_cache is None:
        return query_spatial_flow_belief_features_fast(
            belief,
            x_w_m=x_w_m,
            y_w_m=y_w_m,
            z_w_m=z_w_m,
            direction_rad=direction_rad,
            cell_lookup=cell_lookup,  # type: ignore[arg-type]
            spatial_cell_lookup=spatial_cell_lookup,
            current_history_launch_index=current_history_launch_index,
            query_radius_m=query_radius_m,
        )
    resolution = max(1e-9, float(FLOW_BELIEF_GRID_RESOLUTION_M))
    key = (
        int(round(float(x_w_m) / resolution)),
        int(round(float(y_w_m) / resolution)),
        int(round(float(z_w_m) / resolution)),
        int(round(float(direction_rad) / 0.08726646259971647)),
        int(current_history_launch_index),
        int(round(float(query_radius_m) / resolution)),
    )
    cached = query_cache.get(key)
    if cached is not None:
        return dict(cached)
    features = query_spatial_flow_belief_features_fast(
        belief,
        x_w_m=x_w_m,
        y_w_m=y_w_m,
        z_w_m=z_w_m,
        direction_rad=direction_rad,
        cell_lookup=cell_lookup,  # type: ignore[arg-type]
        spatial_cell_lookup=spatial_cell_lookup,
        current_history_launch_index=current_history_launch_index,
        query_radius_m=query_radius_m,
    )
    query_cache[key] = dict(features)
    return features


def _candidate_reachable_flow_attraction(
    *,
    belief: DirectionalResidualLiftBelief,
    cell_lookup: dict[tuple[int, int, int, int], object],
    spatial_cell_lookup: dict[tuple[int, int, int], tuple[DirectionalResidualCell, ...]] | None = None,
    query_cache: dict[tuple[int, int, int, int, int, int], dict[str, object]] | None = None,
    exit_x: float,
    exit_y: float,
    exit_z: float,
    exit_direction: float,
    current_history_launch_index: int,
    use_residual_memory: bool,
    governor_config: GovernorConfig,
    specific_energy_weight: float,
    updraft_weight: float,
    query_radius_m: float = FLOW_BELIEF_QUERY_RADIUS_M,
    probe_set: tuple[tuple[float, float, float, float], ...] = FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES,
) -> dict[str, float | int]:
    """Return a bounded attraction toward confident useful flow just beyond the candidate exit."""

    if not use_residual_memory or not cell_lookup:
        return _empty_reachable_flow_attraction()
    best_score = 0.0
    best_confidence = 0.0
    best_observation_count = 0
    best_x = float(exit_x)
    best_y = float(exit_y)
    best_z = float(exit_z)
    query_count = 0
    uncertainty_weighted_sum = 0.0
    uncertainty_weight_sum = 0.0
    low_confidence_query_count = 0
    for distance_m, azimuth_offset_rad, elevation_offset_rad, probe_weight in probe_set:
        distance_m = float(distance_m)
        if distance_m > float(FLOW_BELIEF_REACHABLE_ATTRACTION_LOOKAHEAD_M):
            continue
        azimuth_offset_rad = _clip(
            float(azimuth_offset_rad),
            -float(FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD),
            float(FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD),
        )
        elevation_offset_rad = _clip(
            float(elevation_offset_rad),
            -float(FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD),
            float(FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD),
        )
        probe_direction = float(exit_direction) + float(azimuth_offset_rad)
        horizontal_distance_m = float(distance_m) * math.cos(float(elevation_offset_rad))
        probe_x = float(exit_x) + math.cos(probe_direction) * horizontal_distance_m
        probe_y = float(exit_y) + math.sin(probe_direction) * horizontal_distance_m
        probe_z = float(exit_z) + math.sin(float(elevation_offset_rad)) * float(distance_m)
        if not _point_inside_true_safe_bounds(probe_x, probe_y, probe_z):
            continue
        features = _cached_spatial_flow_belief_features(
            belief,
            x_w_m=probe_x,
            y_w_m=probe_y,
            z_w_m=probe_z,
            direction_rad=probe_direction,
            cell_lookup=cell_lookup,
            spatial_cell_lookup=spatial_cell_lookup,
            current_history_launch_index=current_history_launch_index,
            query_radius_m=query_radius_m,
            query_cache=query_cache,
        )
        query_count += 1
        confidence = _candidate_path_probe_confidence(features, governor_config=governor_config)
        distance_discount = 1.0 - 0.35 * _clip(
            distance_m / max(1e-9, float(FLOW_BELIEF_REACHABLE_ATTRACTION_LOOKAHEAD_M)),
            0.0,
            1.0,
        )
        azimuth_discount = max(0.0, math.cos(float(azimuth_offset_rad)))
        elevation_discount = max(0.0, math.cos(float(elevation_offset_rad)))
        information_weight = (
            float(probe_weight)
            * float(distance_discount)
            * float(azimuth_discount)
            * float(elevation_discount)
        )
        if information_weight > 0.0:
            uncertainty_weighted_sum += information_weight * _clip(1.0 - confidence, 0.0, 1.0)
            uncertainty_weight_sum += information_weight
            if confidence < 0.5:
                low_confidence_query_count += 1
        if confidence <= 0.0:
            continue
        specific_energy = _clip(
            _float_value(
                features.get(
                    "belief_local_specific_energy_residual_m",
                    features.get("belief_local_energy_residual_m", 0.0),
                )
            ),
            -float(governor_config.candidate_path_memory_specific_energy_residual_cap_m),
            float(governor_config.candidate_path_memory_specific_energy_residual_cap_m),
        )
        updraft = _clip(
            _float_value(features.get("belief_local_updraft_gain_residual_m", 0.0)),
            -float(governor_config.candidate_path_memory_residual_cap_m),
            float(governor_config.candidate_path_memory_residual_cap_m),
        )
        utility = float(specific_energy_weight) * float(specific_energy) + float(updraft_weight) * float(updraft)
        if utility <= 0.0:
            continue
        score = (
            float(utility)
            * float(confidence)
            * float(distance_discount)
            * float(azimuth_discount)
            * float(elevation_discount)
            * float(probe_weight)
        )
        if score > best_score:
            best_score = float(score)
            best_confidence = float(confidence)
            best_observation_count = int(_float_value(features.get("belief_observation_count", 0.0)))
            best_x = float(probe_x)
            best_y = float(probe_y)
            best_z = float(probe_z)
    capped_score = _clip(float(best_score), 0.0, float(FLOW_BELIEF_REACHABLE_ATTRACTION_CAP_M))
    mean_uncertainty = _clip(
        uncertainty_weighted_sum / max(1e-9, uncertainty_weight_sum),
        0.0,
        1.0,
    )
    return {
        "raw_attraction_m": float(best_score),
        "capped_attraction_m": float(capped_score),
        "confidence": float(best_confidence),
        "query_count": int(query_count),
        "observation_count": int(best_observation_count),
        "mean_uncertainty": float(mean_uncertainty),
        "low_confidence_query_count": int(low_confidence_query_count),
        "best_x_w_m": float(best_x),
        "best_y_w_m": float(best_y),
        "best_z_w_m": float(best_z),
    }


def _empty_reachable_flow_attraction() -> dict[str, float | int]:
    return {
        "raw_attraction_m": 0.0,
        "capped_attraction_m": 0.0,
        "confidence": 0.0,
        "query_count": 0,
        "observation_count": 0,
        "mean_uncertainty": 0.0,
        "low_confidence_query_count": 0,
        "best_x_w_m": 0.0,
        "best_y_w_m": 0.0,
        "best_z_w_m": 0.0,
    }


def _candidate_route_flow_value(
    *,
    belief: DirectionalResidualLiftBelief,
    cell_lookup: dict[tuple[int, int, int, int], object],
    spatial_cell_lookup: dict[tuple[int, int, int], tuple[DirectionalResidualCell, ...]] | None = None,
    query_cache: dict[tuple[int, int, int, int, int, int], dict[str, object]] | None = None,
    x0: float,
    exit_x: float,
    exit_y: float,
    exit_z: float,
    exit_direction: float,
    path_speed_m_s: float,
    current_history_launch_index: int,
    use_residual_memory: bool,
    governor_config: GovernorConfig,
    specific_energy_weight: float,
    updraft_weight: float,
    query_radius_m: float = FLOW_BELIEF_QUERY_RADIUS_M,
    probe_set: tuple[tuple[float, float, float, float], ...] = FLOW_BELIEF_ROUTE_PROBE_FRACTIONS,
) -> dict[str, float | int]:
    """Estimate short-horizon route value beyond the first primitive with sparse map probes."""

    if not use_residual_memory or not cell_lookup:
        return _empty_route_flow_value(horizon_primitives=governor_config.memory_route_horizon_primitives)
    horizon = max(1, int(round(float(governor_config.memory_route_horizon_primitives))))
    route_distance_m = _clip(
        float(path_speed_m_s) * float(PRIMITIVE_FINITE_HORIZON_S) * float(horizon),
        0.20,
        2.20,
    )
    discount = _clip(float(governor_config.memory_route_discount), 0.0, 1.0)
    weighted_utility = 0.0
    weighted_confidence = 0.0
    weighted_uncertainty = 0.0
    weighted_safe = 0.0
    weight_sum = 0.0
    best_score = float("-inf")
    best_x = float(exit_x)
    best_y = float(exit_y)
    best_z = float(exit_z)
    probe_count = 0
    for fraction, azimuth_offset_rad, elevation_offset_rad, probe_weight in probe_set:
        fraction = _clip(float(fraction), 0.0, 1.0)
        azimuth_offset_rad = _clip(
            float(azimuth_offset_rad),
            -float(FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD),
            float(FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD),
        )
        elevation_offset_rad = _clip(
            float(elevation_offset_rad),
            -float(FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD),
            float(FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD),
        )
        distance_m = float(route_distance_m) * float(fraction)
        route_direction = float(exit_direction) + float(azimuth_offset_rad)
        horizontal_distance_m = float(distance_m) * math.cos(float(elevation_offset_rad))
        probe_x = float(exit_x) + math.cos(route_direction) * horizontal_distance_m
        probe_y = float(exit_y) + math.sin(route_direction) * horizontal_distance_m
        probe_z = float(exit_z) + math.sin(float(elevation_offset_rad)) * float(distance_m)
        safe = 1.0 if _point_inside_true_safe_bounds(probe_x, probe_y, probe_z) else 0.0
        if safe <= 0.0:
            weight_sum += float(probe_weight)
            continue
        features = _cached_spatial_flow_belief_features(
            belief,
            x_w_m=probe_x,
            y_w_m=probe_y,
            z_w_m=probe_z,
            direction_rad=route_direction,
            cell_lookup=cell_lookup,
            spatial_cell_lookup=spatial_cell_lookup,
            current_history_launch_index=current_history_launch_index,
            query_radius_m=query_radius_m,
            query_cache=query_cache,
        )
        probe_count += 1
        confidence = _candidate_path_probe_confidence(features, governor_config=governor_config)
        specific_energy = _clip(
            _float_value(
                features.get(
                    "belief_local_specific_energy_residual_m",
                    features.get("belief_local_energy_residual_m", 0.0),
                )
            ),
            -float(governor_config.candidate_path_memory_specific_energy_residual_cap_m),
            float(governor_config.candidate_path_memory_specific_energy_residual_cap_m),
        )
        updraft = _clip(
            _float_value(features.get("belief_local_updraft_gain_residual_m", 0.0)),
            -float(governor_config.candidate_path_memory_residual_cap_m),
            float(governor_config.candidate_path_memory_residual_cap_m),
        )
        utility = float(specific_energy_weight) * float(specific_energy) + float(updraft_weight) * float(updraft)
        step_index = max(1, int(math.ceil(float(fraction) * float(horizon))))
        step_discount = float(discount) ** float(step_index - 1)
        progress_gate = _clip((float(probe_x) - float(x0)) / max(0.2, float(TRUE_SAFE_BOUNDS.x_w_m[1]) - float(x0)), 0.0, 1.0)
        weight = float(probe_weight) * float(step_discount) * (0.35 + 0.65 * float(progress_gate))
        weight_sum += float(weight)
        weighted_utility += float(weight) * float(confidence) * float(utility)
        weighted_confidence += float(weight) * float(confidence)
        weighted_uncertainty += float(weight) * _clip(1.0 - float(confidence), 0.0, 1.0)
        weighted_safe += float(weight) * float(safe)
        score = float(confidence) * float(utility) * float(progress_gate)
        if score > best_score:
            best_score = float(score)
            best_x = float(probe_x)
            best_y = float(probe_y)
            best_z = float(probe_z)
    denominator = max(1e-9, float(weight_sum))
    front_progress = _clip(
        (max(float(exit_x), float(best_x)) - float(x0))
        / max(0.2, float(TRUE_SAFE_BOUNDS.x_w_m[1]) - float(x0)),
        0.0,
        1.0,
    )
    confidence = _clip(float(weighted_confidence) / denominator, 0.0, 1.0)
    uncertainty = _clip(float(weighted_uncertainty) / denominator, 0.0, 1.0)
    safe_fraction = _clip(float(weighted_safe) / denominator, 0.0, 1.0)
    exploitation = _clip(
        float(weighted_utility) / denominator,
        -float(governor_config.candidate_path_memory_specific_energy_residual_cap_m),
        float(governor_config.candidate_path_memory_specific_energy_residual_cap_m),
    )
    information_gain = float(uncertainty) * (0.25 + 0.75 * float(front_progress)) * float(safe_fraction)
    return {
        "horizon_primitives": int(horizon),
        "probe_count": int(probe_count),
        "exploitation_m": float(exploitation),
        "information_gain": float(information_gain),
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "front_progress": float(front_progress),
        "safe_fraction": float(safe_fraction),
        "best_x_w_m": float(best_x),
        "best_y_w_m": float(best_y),
        "best_z_w_m": float(best_z),
    }


def _empty_route_flow_value(horizon_primitives: float = 0.0) -> dict[str, float | int]:
    return {
        "horizon_primitives": int(round(float(horizon_primitives))) if horizon_primitives else 0,
        "probe_count": 0,
        "exploitation_m": 0.0,
        "information_gain": 0.0,
        "confidence": 0.0,
        "uncertainty": 0.0,
        "front_progress": 0.0,
        "safe_fraction": 0.0,
        "best_x_w_m": 0.0,
        "best_y_w_m": 0.0,
        "best_z_w_m": 0.0,
    }


def _flow_map_information_gain(
    *,
    path_uncertainty: float,
    reachable_uncertainty: float,
    x0: float,
    exit_x: float,
    exit_y: float,
    exit_z: float,
) -> dict[str, float]:
    """Return bounded information value for entering under-observed safe map regions."""

    path_term = _clip(float(path_uncertainty), 0.0, 1.0)
    reachable_term = _clip(float(reachable_uncertainty), 0.0, 1.0)
    uncertainty = 0.45 * float(path_term) + 0.55 * float(reachable_term)
    front_x = float(TRUE_SAFE_BOUNDS.x_w_m[1])
    remaining_x = max(0.2, front_x - float(x0))
    progress_gate = _clip((float(exit_x) - float(x0)) / remaining_x, 0.0, 1.0)
    safe_gate = 1.0 if _point_inside_true_safe_bounds(exit_x, exit_y, exit_z) else 0.0
    mission_gate = 0.25 + 0.75 * float(progress_gate)
    return {
        "information_gain": float(uncertainty) * float(mission_gate) * float(safe_gate),
        "progress_gate": float(progress_gate),
        "safe_gate": float(safe_gate),
    }


def _point_inside_true_safe_bounds(x_w_m: float, y_w_m: float, z_w_m: float) -> bool:
    return (
        float(TRUE_SAFE_BOUNDS.x_w_m[0]) <= float(x_w_m) <= float(TRUE_SAFE_BOUNDS.x_w_m[1])
        and float(TRUE_SAFE_BOUNDS.y_w_m[0]) <= float(y_w_m) <= float(TRUE_SAFE_BOUNDS.y_w_m[1])
        and float(TRUE_SAFE_BOUNDS.z_w_m[0]) <= float(z_w_m) <= float(TRUE_SAFE_BOUNDS.z_w_m[1])
    )


def _candidate_path_probe_confidence(
    features: dict[str, object],
    *,
    governor_config: GovernorConfig | None = None,
) -> float:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    effective_observation_count = _float_value(
        features.get("belief_effective_observation_count", features.get("belief_observation_count", 0.0))
    )
    return _clip(
        effective_observation_count / max(1e-9, float(cfg.candidate_path_memory_full_confidence_observations)),
        0.0,
        1.0,
    )


def _memory_guided_exploration_uncertainty(
    *,
    confidence: float,
    x0: float,
    exit_x: float,
    exit_y: float,
    exit_z: float,
) -> float:
    unknown = _clip(1.0 - float(confidence), 0.0, 1.0)
    front_x = float(TRUE_SAFE_BOUNDS.x_w_m[1])
    remaining_x = max(0.2, float(front_x) - float(x0))
    progress_gate = _clip((float(exit_x) - float(x0)) / remaining_x, 0.0, 1.0)
    mission_relevance_gate = 0.25 + 0.75 * float(progress_gate)
    inside_safe_box = (
        float(TRUE_SAFE_BOUNDS.x_w_m[0]) <= float(exit_x) <= float(TRUE_SAFE_BOUNDS.x_w_m[1])
        and float(TRUE_SAFE_BOUNDS.y_w_m[0]) <= float(exit_y) <= float(TRUE_SAFE_BOUNDS.y_w_m[1])
        and float(TRUE_SAFE_BOUNDS.z_w_m[0]) <= float(exit_z) <= float(TRUE_SAFE_BOUNDS.z_w_m[1])
    )
    safety_gate = 1.0 if inside_safe_box else 0.0
    return float(unknown) * float(mission_relevance_gate) * float(safety_gate)


def _clamp_to_bounds(value: float, bounds: tuple[float, float]) -> float:
    return _clip(float(value), float(bounds[0]), float(bounds[1]))


def _prepare_realtime_governor_decision(
    *,
    state: np.ndarray,
    scheduled: dict[str, object],
    episode_id: str,
    protocol: ValidationProtocol,
    primitive_step_index: int,
    policy: dict[str, object],
    belief: DirectionalResidualLiftBelief,
    representatives: list[dict[str, object]],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
    governor_config: GovernorConfig,
    scheduler_decision_source: str,
) -> dict[str, object]:
    """Prepare one primitive decision and profile whether it fits the real-time budget."""

    context_started = time.perf_counter()
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
    context_build_duration_s = time.perf_counter() - context_started

    belief_started = time.perf_counter()
    belief_features = None
    current_history_launch_index = _adaptation_launch_index(scheduled)
    if bool(policy["uses_memory"]):
        belief_features = query_spatial_flow_belief_features(
            belief,
            x_w_m=float(state[STATE_INDEX["x_w"]]),
            y_w_m=float(state[STATE_INDEX["y_w"]]),
            z_w_m=float(state[STATE_INDEX["z_w"]]),
            direction_rad=float(state[STATE_INDEX["psi"]]),
            current_history_launch_index=current_history_launch_index,
            query_radius_m=FLOW_BELIEF_QUERY_RADIUS_M,
        )
    candidate_belief_features = _candidate_path_belief_features_fn(
        belief=belief,
        state=state,
        current_history_launch_index=current_history_launch_index,
        use_residual_memory=bool(policy["uses_memory"]),
        governor_config=governor_config,
        real_time_controller_mode=False,
    )
    belief_query_duration_s = time.perf_counter() - belief_started

    selection_started = time.perf_counter()
    selected, candidate_rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows_by_variant_id,
        context=context_payload["row"],
        governor_mode=governor_mode,
        policy_id=str(policy["policy_id"]),
        belief_features=belief_features,
        candidate_belief_features=candidate_belief_features,
        adaptive_memory_active=bool(policy["uses_memory"]),
        governor_config=governor_config,
        candidate_row_mode="controller",
    )
    selection_duration_s = time.perf_counter() - selection_started
    prefilter_source = selected if selected is not None else (candidate_rows[0] if candidate_rows else {})

    scheduler_fields = _real_time_scheduler_decision_fields(
        primitive_step_index=primitive_step_index,
        scheduler_decision_source=scheduler_decision_source,
        context_build_duration_s=context_build_duration_s,
        belief_query_duration_s=belief_query_duration_s,
        selection_duration_s=selection_duration_s,
        candidate_count=len(candidate_rows),
        viable_count=sum(1 for row in candidate_rows if bool(row.get("viable", False))),
    )
    scheduler_fields.update(
        {
            "decision_total_library_candidate_count": int(
                prefilter_source.get("selector_total_candidate_count", len(representatives))
            ),
            "decision_entry_filtered_candidate_count": int(
                prefilter_source.get("selector_entry_filtered_candidate_count", len(candidate_rows))
            ),
            "decision_prefilter_skipped_candidate_count": int(
                prefilter_source.get("selector_skipped_candidate_count", max(0, len(representatives) - len(candidate_rows)))
            ),
            "decision_prefilter_status": str(prefilter_source.get("selector_prefilter_status", "")),
            "decision_prefilter_version": str(prefilter_source.get("selector_prefilter_version", "")),
            "decision_allowed_entry_classes": str(prefilter_source.get("selector_allowed_entry_classes", "")),
            "decision_allowed_speed_bins": str(prefilter_source.get("selector_allowed_speed_bins", "")),
        }
    )
    for row in candidate_rows:
        row.update(scheduler_fields)
    return {
        "primitive_step_index": int(primitive_step_index),
        "route": route,
        "start_state_family": start_state_family,
        "governor_mode": governor_mode,
        "context_payload": context_payload,
        "belief_features": belief_features,
        "selected": selected,
        "candidate_rows": candidate_rows,
        "scheduler_fields": scheduler_fields,
    }


def _real_time_scheduler_decision_fields(
    *,
    primitive_step_index: int,
    scheduler_decision_source: str,
    context_build_duration_s: float,
    belief_query_duration_s: float,
    selection_duration_s: float,
    candidate_count: int,
    viable_count: int,
) -> dict[str, object]:
    total_duration_s = float(context_build_duration_s) + float(belief_query_duration_s) + float(selection_duration_s)
    preferred_margin_s = float(REAL_TIME_PREFERRED_DECISION_BUDGET_S) - float(total_duration_s)
    hard_margin_s = float(REAL_TIME_HARD_DECISION_BUDGET_S) - float(total_duration_s)
    prepared_before_boundary = str(scheduler_decision_source) in {
        "initial_launch_precomputed_before_release",
        "prepared_during_previous_primitive_window",
    }
    return {
        "real_time_outer_loop_scheduler_version": REAL_TIME_OUTER_LOOP_SCHEDULER_VERSION,
        "real_time_claim_status": "controller_compute_profile_excludes_table_flush_and_posthoc_diagnostics",
        "scheduler_policy": "prepare_next_decision_before_primitive_boundary_with_full_memory_query_controller_row_no_table_flush",
        "scheduler_decision_source": str(scheduler_decision_source),
        "scheduler_commit_status": "prepared_pending_commit" if prepared_before_boundary else "computed_at_boundary",
        "scheduler_prepared_before_primitive_boundary": bool(prepared_before_boundary),
        "decision_context_build_duration_s": float(context_build_duration_s),
        "decision_belief_query_duration_s": float(belief_query_duration_s),
        "decision_selection_duration_s": float(selection_duration_s),
        "decision_total_duration_s": float(total_duration_s),
        "decision_controller_compute_duration_s": float(total_duration_s),
        "decision_diagnostic_logging_duration_s": 0.0,
        "decision_controller_timing_scope": "context_plus_belief_plus_full_memory_controller_selector_no_table_flush",
        "decision_candidate_count": int(candidate_count),
        "decision_viable_count": int(viable_count),
        "preferred_decision_budget_s": float(REAL_TIME_PREFERRED_DECISION_BUDGET_S),
        "hard_decision_budget_s": float(REAL_TIME_HARD_DECISION_BUDGET_S),
        "preferred_20ms_slot_met": bool(preferred_margin_s >= 0.0),
        "hard_100ms_boundary_met": bool(hard_margin_s >= 0.0),
        "preferred_decision_budget_margin_s": float(preferred_margin_s),
        "hard_decision_budget_margin_s": float(hard_margin_s),
        "primitive_step_index_profiled": int(primitive_step_index),
    }


def _mark_prepared_decision_committed(decision: dict[str, object]) -> None:
    fields = dict(decision.get("scheduler_fields", {}))
    fields["scheduler_commit_status"] = "committed_prepared_decision"
    fields["scheduler_prepared_before_primitive_boundary"] = True
    decision["scheduler_fields"] = fields
    for row in list(decision.get("candidate_rows", [])):
        if isinstance(row, dict):
            row.update(fields)


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
    environment_layout_seed = int(scheduled.get("environment_layout_seed", seed))
    environment_active_fan_seed = int(scheduled.get("environment_active_fan_seed", seed))
    environment_parameter_seed = int(scheduled.get("environment_parameter_seed", seed))
    plant_implementation_seed = int(scheduled.get("plant_implementation_seed", seed))
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
    full_w3_randomisation = _uses_full_w3_randomisation_block(
        protocol=protocol,
        environment_block_id=str(scheduled.get("environment_block_id", "")),
    )
    plant_layer = env_layer if full_w3_randomisation else (
        "W2" if protocol.requires_no_glider_latency_variation_audit else env_layer
    )
    implementation_layer = env_layer if full_w3_randomisation else (
        "W2" if protocol.requires_no_glider_latency_variation_audit else env_layer
    )
    implementation_instance = implementation_instance_for_layer(
        implementation_layer,
        plant_implementation_seed,
        latency_case="nominal",
    )
    plant_instance = plant_instance_for_layer(plant_layer, plant_implementation_seed)
    latency_case = str(implementation_instance.latency_case)
    context = build_environment_context(
        state,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=latency_case,
        actuator_case="nominal",
        surrogate_binding=binding,
    )
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
        "fan_position_safety_radius_m": str(scheduled.get("fan_position_safety_radius_m", "")),
        "environment_seed": seed,
        "environment_layout_seed": environment_layout_seed,
        "environment_active_fan_seed": environment_active_fan_seed,
        "environment_parameter_seed": environment_parameter_seed,
        "plant_implementation_seed": plant_implementation_seed,
        "between_episode_environment_variation": bool(scheduled.get("between_episode_environment_variation", False)),
        "between_episode_environment_parameter_variation": bool(
            scheduled.get("between_episode_environment_parameter_variation", False)
        ),
        "between_episode_fan_layout_variation": bool(scheduled.get("between_episode_fan_layout_variation", False)),
        "between_episode_active_fan_count_variation": bool(
            scheduled.get("between_episode_active_fan_count_variation", False)
        ),
        "plant_implementation_variation_scope": str(scheduled.get("plant_implementation_variation_scope", "")),
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
        "plant_W_layer": plant_layer,
        "implementation_W_layer": implementation_layer,
        "full_w3_randomisation_block": bool(full_w3_randomisation),
        "current_x_w_m": float(state[STATE_INDEX["x_w"]]),
        "current_y_w_m": float(state[STATE_INDEX["y_w"]]),
        "current_z_w_m": float(state[STATE_INDEX["z_w"]]),
        "current_speed_m_s": float(local_speed_from_state_vector(state)),
        "current_local_lqr_speed_bin_id": lqr_speed_bin_id(local_speed_from_state_vector(state)),
        "mission_x_min_w_m": float(TRUE_SAFE_BOUNDS.x_w_m[0]),
        "front_wall_target_x_w_m": float(TRUE_SAFE_BOUNDS.x_w_m[1]),
        "mission_terminal_y_min_m": float(TRUE_SAFE_BOUNDS.y_w_m[0]),
        "mission_terminal_y_max_m": float(TRUE_SAFE_BOUNDS.y_w_m[1]),
        "mission_terminal_z_min_m": float(TRUE_SAFE_BOUNDS.z_w_m[0]),
        "mission_terminal_z_max_m": float(TRUE_SAFE_BOUNDS.z_w_m[1]),
        "mission_terminal_specific_energy_reference_m": float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M),
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
        "adaptation_launch_index": _adaptation_launch_index(scheduled),
        "policy_id": str(scheduled.get("policy_id", "")),
    }
    return {
        "context": context,
        "row": row,
        "wind_field": wind_field,
        "implementation_instance": implementation_instance,
        "plant_instance": plant_instance,
    }


def _uses_full_w3_randomisation_block(*, protocol: ValidationProtocol, environment_block_id: str) -> bool:
    return (
        str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        and str(environment_block_id) in R10_FULL_DOMAIN_RANDOMISATION_BLOCK_IDS
    )


def _environment_randomisation_config_for_context(
    *,
    protocol: ValidationProtocol,
    scheduled: dict[str, object],
    scheduled_active_fan_count: int | None,
) -> EnvironmentRandomisationConfig | None:
    block_id = str(scheduled.get("environment_block_id", ""))
    seed_kwargs = _environment_randomisation_seed_kwargs(scheduled)
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id in R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="independent_uniform_xy_bounds",
            fan_position_xy_bounds_m=R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M,
            fan_position_safety_radius_m=R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M,
            **seed_kwargs,
        )
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id in R10_NOMINAL_FAN_PARAMETER_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="fixed_base_positions",
            fan_power_scale_range=(1.0, 1.0),
            amplitude_scale_range=(1.0, 1.0),
            width_scale_range=(1.0, 1.0),
            uncertainty_scale_range=(1.0, 1.0),
            **seed_kwargs,
        )
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id in R10_FIXED_BASE_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="fixed_base_positions",
            **seed_kwargs,
        )
    if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS and block_id in R10_SHIFTED_FAN_POSITION_BLOCK_IDS:
        return EnvironmentRandomisationConfig(
            active_fan_count=scheduled_active_fan_count,
            fan_position_policy="common_shift",
            **seed_kwargs,
        )
    if scheduled_active_fan_count is not None:
        return EnvironmentRandomisationConfig(active_fan_count=scheduled_active_fan_count, **seed_kwargs)
    return None


def _environment_randomisation_seed_kwargs(scheduled: dict[str, object]) -> dict[str, int]:
    seed = int(scheduled.get("environment_seed", 0))
    return {
        "fan_layout_seed": int(scheduled.get("environment_layout_seed", seed)),
        "active_fan_seed": int(scheduled.get("environment_active_fan_seed", seed)),
        "fan_parameter_seed": int(scheduled.get("environment_parameter_seed", seed)),
    }


def _governor_config_for_policy(policy: dict[str, object], *, base_config: GovernorConfig = DEFAULT_GOVERNOR_CONFIG) -> GovernorConfig:
    if bool(policy["safe_explore"]):
        return base_config
    return replace(
        base_config,
        config_id=f"{base_config.config_id}_no_exploration_ablation",
        exploration_bonus_weight=0.0,
    )


def _adaptation_launch_index(scheduled: dict[str, object]) -> int:
    value = scheduled.get("adaptation_launch_index", "")
    if str(value).strip() not in {"", "nan", "None"}:
        return int(_float_value(value, 0.0))
    if str(scheduled.get("launch_role", "")) == "history":
        return int(_float_value(scheduled.get("history_launch_index", 0.0)))
    return int(_float_value(scheduled.get("history_length", 0.0)))


def _schedule_identity_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "library_size_case_id": str(row.get("library_size_case_id", "")),
        "policy_id": str(row.get("policy_id", "")),
        "history_length": int(row.get("history_length", 0)),
        "adaptation_launch_index": _adaptation_launch_index(row),
        "outer_case_index": int(row.get("outer_case_index", 0)),
        "outer_case_id": str(row.get("outer_case_id", "")),
        "outer_case_type": str(row.get("outer_case_type", "")),
        "environment_block_id": str(row.get("environment_block_id", "")),
        "common_final_launch_key": str(row.get("common_final_launch_key", "")),
        "episode_id": str(row.get("episode_id", "")),
    }


def _episode_memory_switch_fields(selector_rows: list[dict[str, object]]) -> dict[str, object]:
    memory_policy_decision_count = 0
    switch_available_count = 0
    accepted_count = 0
    rejected_count = 0
    accepted_steps: list[str] = []
    rejected_steps: list[str] = []
    status_entries: list[str] = []
    for row in selector_rows:
        policy_id = str(row.get("policy_id", ""))
        if MEMORY_POLICY_PREFIX not in policy_id:
            continue
        memory_policy_decision_count += 1
        baseline_variant = str(row.get("selected_memory_shield_baseline_variant_id", ""))
        memory_variant = str(row.get("selected_memory_shield_memory_variant_id", ""))
        status = str(row.get("selected_memory_shield_status", ""))
        if not baseline_variant or not memory_variant or baseline_variant == memory_variant:
            continue
        switch_available_count += 1
        step = int(_float_value(row.get("primitive_step_index", 0)))
        status_entries.append(f"{step}:{status}:{baseline_variant}->{memory_variant}")
        accepted = status.startswith("accepted")
        if accepted:
            accepted_count += 1
            accepted_steps.append(str(step))
        elif status.startswith("rejected"):
            rejected_count += 1
            rejected_steps.append(str(step))
    return {
        "memory_policy_decision_count": int(memory_policy_decision_count),
        "memory_switch_available_count": int(switch_available_count),
        "accepted_memory_switch_count": int(accepted_count),
        "rejected_memory_switch_count": int(rejected_count),
        "memory_changed_selection": bool(accepted_count > 0),
        "memory_changed_selection_source": "selector_memory_shield_accepted_switch",
        "memory_switch_accepted_step_indices": ";".join(accepted_steps),
        "memory_switch_rejected_step_indices": ";".join(rejected_steps),
        "memory_switch_status_sequence": ";".join(status_entries),
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
    physical_hard_failure = any(_rollout_row_is_hard_failure(row) for row in primitive_rows)
    physical_floor_or_ceiling = any(_rollout_row_is_floor_or_ceiling_violation(row) for row in primitive_rows)
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
    initial_launch_speed_m_s = _initial_launch_speed_m_s(primitive_rows)
    expected_low_energy_sink = _expected_low_energy_dry_air_sink(
        scheduled=scheduled,
        primitive_rows=primitive_rows,
        physical_floor_or_ceiling=physical_floor_or_ceiling,
        no_viable=no_viable,
        terminal_useful=terminal_useful,
        lift_capture=lift_capture,
        episode_duration_s=episode_duration_s,
        initial_launch_speed_m_s=initial_launch_speed_m_s,
    )
    hard_failure = bool(physical_hard_failure and not expected_low_energy_sink)
    floor_or_ceiling = bool(physical_floor_or_ceiling and not expected_low_energy_sink)
    safe_success = bool(sequence_compliant and last_continuation_or_terminal and not hard_failure and not floor_or_ceiling and not no_viable)
    comparison_only_policy = bool(policy.get("comparison_only", False))
    memory_switch_fields = _episode_memory_switch_fields(selector_rows)
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_family": str(policy["policy_family"]),
        "safe_explore_active": bool(policy["safe_explore"]),
        "open_loop_comparison_active": bool(policy.get("open_loop", False)),
        "comparison_only_policy": bool(comparison_only_policy),
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
        "physical_hard_failure": bool(physical_hard_failure),
        "physical_floor_or_ceiling_violation": bool(physical_floor_or_ceiling),
        "expected_low_energy_dry_air_sink": bool(expected_low_energy_sink),
        "episode_interpretation_label": _episode_interpretation_label(
            expected_low_energy_dry_air_sink=expected_low_energy_sink,
            hard_failure=hard_failure,
            floor_or_ceiling=floor_or_ceiling,
            no_viable=no_viable,
        ),
        "claim_bearing_episode": bool(not expected_low_energy_sink and not comparison_only_policy),
        "initial_launch_speed_m_s": float(initial_launch_speed_m_s),
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
        **memory_switch_fields,
        "exploration_changed_selection": False,
        "environment_instance_id": str(context_row.get("environment_instance_id", "")),
        "claim_status": "simulation_only_repeated_launch_rollout_episode",
        "belief_update_count_before": int(belief_before.update_count),
        "belief_update_count_after": int(belief_after.update_count),
    }
    row.update(
        _outer_loop_mission_fields_from_primitives(
            primitive_rows,
            hard_failure=hard_failure,
            floor_or_ceiling=floor_or_ceiling,
            no_viable=no_viable,
            expected_low_energy_sink=expected_low_energy_sink,
        )
    )
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
    physical_hard_failure = _rollout_row_is_hard_failure(rollout_row)
    physical_floor_or_ceiling = _rollout_row_is_floor_or_ceiling_violation(rollout_row)
    terminal_useful = bool(rollout_row.get("episode_terminal_useful", False))
    lift_capture = bool(float(rollout_row.get("lift_dwell_time_s", 0.0)) > 0.0)
    episode_duration_s = float(_episode_rollout_duration_s([rollout_row]))
    initial_launch_speed_m_s = _initial_launch_speed_m_s([rollout_row])
    expected_low_energy_sink = _expected_low_energy_dry_air_sink(
        scheduled=scheduled,
        primitive_rows=[rollout_row],
        physical_floor_or_ceiling=physical_floor_or_ceiling,
        no_viable=False,
        terminal_useful=terminal_useful,
        lift_capture=lift_capture,
        episode_duration_s=episode_duration_s,
        initial_launch_speed_m_s=initial_launch_speed_m_s,
    )
    hard_failure = bool(physical_hard_failure and not expected_low_energy_sink)
    floor_or_ceiling = bool(physical_floor_or_ceiling and not expected_low_energy_sink)
    comparison_only_policy = bool(policy.get("comparison_only", False))
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_family": str(policy["policy_family"]),
        "safe_explore_active": bool(policy["safe_explore"]),
        "open_loop_comparison_active": bool(policy.get("open_loop", False)),
        "comparison_only_policy": bool(comparison_only_policy),
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
        "hard_failure": hard_failure,
        "floor_or_ceiling_violation": floor_or_ceiling,
        "physical_hard_failure": bool(physical_hard_failure),
        "physical_floor_or_ceiling_violation": bool(physical_floor_or_ceiling),
        "expected_low_energy_dry_air_sink": bool(expected_low_energy_sink),
        "episode_interpretation_label": _episode_interpretation_label(
            expected_low_energy_dry_air_sink=expected_low_energy_sink,
            hard_failure=hard_failure,
            floor_or_ceiling=floor_or_ceiling,
            no_viable=False,
        ),
        "claim_bearing_episode": bool(not expected_low_energy_sink and not comparison_only_policy),
        "initial_launch_speed_m_s": float(initial_launch_speed_m_s),
        "no_viable_primitive": False,
        "safe_success": bool(
            (_truthy(rollout_row.get("continuation_valid", False)) or _truthy(rollout_row.get("episode_terminal_useful", False)))
            and not hard_failure
            and not floor_or_ceiling
        ),
        "full_safe_success": bool(
            (_truthy(rollout_row.get("continuation_valid", False)) or _truthy(rollout_row.get("episode_terminal_useful", False)))
            and not hard_failure
            and not floor_or_ceiling
            and not _rollout_row_is_terminal_safe_exit_only(rollout_row)
        ),
        "terminal_useful": bool(terminal_useful),
        "terminal_useful_safe_exit_only": bool(_rollout_row_is_terminal_safe_exit_only(rollout_row)),
        "lift_capture": bool(lift_capture),
        "episode_rollout_duration_s": float(episode_duration_s),
        "lift_dwell_time_s": float(rollout_row.get("lift_dwell_time_s", 0.0)),
        "energy_residual_m": float(rollout_row.get("energy_residual_m", 0.0)),
        **_episode_specific_energy_summary([rollout_row]),
        "min_wall_margin_m": float(rollout_row.get("minimum_wall_margin_m", 0.0)),
        "governor_rejection_count": int(selector_row.get("candidate_count", 0)) - int(selector_row.get("viable_count", 0)),
        "belief_observation_count": int(belief_after.update_count),
        "belief_uncertainty": float(max(0.0, 1.0 / math.sqrt(max(1, int(belief_after.update_count))))),
        **_episode_memory_switch_fields([selector_row]),
        "exploration_changed_selection": False,
        "environment_instance_id": str(context_row.get("environment_instance_id", "")),
        "claim_status": "simulation_only_repeated_launch_rollout_episode",
        "belief_update_count_before": int(belief_before.update_count),
        "belief_update_count_after": int(belief_after.update_count),
    }
    row.update(
        _outer_loop_mission_fields_from_primitives(
            [rollout_row],
            hard_failure=hard_failure,
            floor_or_ceiling=floor_or_ceiling,
            no_viable=False,
            expected_low_energy_sink=expected_low_energy_sink,
        )
    )
    row.update(_launch_score_fields_for_role(row))
    return row


def _episode_row_from_blocked(
    scheduled: dict[str, object],
    policy: dict[str, object],
    context_row: dict[str, object],
    *,
    reason: str = "no_viable_primitive",
) -> dict[str, object]:
    comparison_only_policy = bool(policy.get("comparison_only", False))
    row = {
        **_schedule_identity_row(scheduled),
        "launch_role": str(scheduled["launch_role"]),
        "policy_family": str(policy["policy_family"]),
        "safe_explore_active": bool(policy["safe_explore"]),
        "open_loop_comparison_active": bool(policy.get("open_loop", False)),
        "comparison_only_policy": bool(comparison_only_policy),
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
        "physical_hard_failure": False,
        "physical_floor_or_ceiling_violation": False,
        "expected_low_energy_dry_air_sink": False,
        "episode_interpretation_label": "no_viable_primitive",
        "claim_bearing_episode": bool(not comparison_only_policy),
        "initial_launch_speed_m_s": float("nan"),
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
        **_episode_memory_switch_fields([]),
        "exploration_changed_selection": False,
        "environment_instance_id": str(context_row.get("environment_instance_id", "")),
        "claim_status": "simulation_only_repeated_launch_rollout_episode",
        "belief_update_count_before": 0,
        "belief_update_count_after": 0,
    }
    row.update(
        _outer_loop_mission_fields_from_primitives(
            [],
            hard_failure=False,
            floor_or_ceiling=False,
            no_viable=True,
            expected_low_energy_sink=False,
        )
    )
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


def _initial_launch_speed_m_s(primitive_rows: list[dict[str, object]]) -> float:
    if not primitive_rows:
        return float("nan")
    first = primitive_rows[0]
    u = _float_value(first.get("initial_u", 0.0))
    v = _float_value(first.get("initial_v", 0.0))
    w = _float_value(first.get("initial_w", 0.0))
    return float(math.sqrt(u * u + v * v + w * w))


def _expected_low_energy_dry_air_sink(
    *,
    scheduled: dict[str, object],
    primitive_rows: list[dict[str, object]],
    physical_floor_or_ceiling: bool,
    no_viable: bool,
    terminal_useful: bool,
    lift_capture: bool,
    episode_duration_s: float,
    initial_launch_speed_m_s: float,
) -> bool:
    if not primitive_rows:
        return False
    del scheduled, lift_capture
    if not physical_floor_or_ceiling or no_viable or terminal_useful:
        return False
    last = primitive_rows[-1]
    return bool(
        str(last.get("termination_cause", "")) == "floor_margin_stop"
        and str(last.get("failure_label", "")) == "floor_violation"
        and float(episode_duration_s) >= DRY_AIR_ENERGY_DEPLETION_MIN_FLIGHT_TIME_S
        and math.isfinite(float(initial_launch_speed_m_s))
        and float(initial_launch_speed_m_s) < LOW_LAUNCH_SPEED_DRY_AIR_THRESHOLD_M_S
    )


def _episode_interpretation_label(
    *,
    expected_low_energy_dry_air_sink: bool,
    hard_failure: bool,
    floor_or_ceiling: bool,
    no_viable: bool,
) -> str:
    if expected_low_energy_dry_air_sink:
        return "expected_low_energy_dry_air_sink_not_governor_failure"
    if no_viable:
        return "no_viable_primitive"
    if hard_failure:
        return "claim_bearing_hard_failure"
    if floor_or_ceiling:
        return "claim_bearing_floor_or_ceiling_violation"
    return "claim_bearing_rollout"


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


def _specific_energy_residual_for_memory_update(
    *,
    rollout_row: dict[str, object],
    outcome: dict[str, object],
) -> float:
    """Compare observed primitive total specific-energy change with the frozen prediction."""

    observed = _float_value(rollout_row.get("energy_residual_m", 0.0))
    expected = _float_value(outcome.get("expected_energy_residual_m", 0.0))
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


def _outer_loop_mission_fields_from_primitives(
    primitive_rows: list[dict[str, object]],
    *,
    hard_failure: bool,
    floor_or_ceiling: bool,
    no_viable: bool,
    expected_low_energy_sink: bool,
) -> dict[str, object]:
    last_row = primitive_rows[-1] if primitive_rows else {}
    exit_position = _exit_position_from_rollout_row(last_row)
    terminal_energy, terminal_energy_source = _terminal_specific_energy_from_rollout_row(last_row)
    terminal_energy_reserve = (
        float(terminal_energy) - float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M)
        if math.isfinite(float(terminal_energy))
        else float("nan")
    )
    wall_exit = _rollout_row_is_wall_boundary_exit(last_row)
    wall_face = _wall_exit_face(exit_position, wall_exit=wall_exit)
    front_wall_terminal = bool(
        wall_exit
        and wall_face == "front_wall_x_max"
        and _truthy(last_row.get("episode_terminal_useful", False))
        and not hard_failure
        and not floor_or_ceiling
        and not no_viable
        and not expected_low_energy_sink
        and _exit_position_inside_terminal_yz(exit_position)
    )
    wrong_wall_exit = bool(wall_exit and not front_wall_terminal)
    mission_success = bool(front_wall_terminal)
    return {
        "mission_score_version": LAUNCH_SCORE_VERSION,
        "mission_success": mission_success,
        "front_wall_terminal_success": front_wall_terminal,
        "wrong_wall_exit": wrong_wall_exit,
        "terminal_wall_face": wall_face,
        "front_wall_target_x_w_m": float(TRUE_SAFE_BOUNDS.x_w_m[1]),
        "mission_terminal_y_min_m": float(TRUE_SAFE_BOUNDS.y_w_m[0]),
        "mission_terminal_y_max_m": float(TRUE_SAFE_BOUNDS.y_w_m[1]),
        "mission_terminal_z_min_m": float(TRUE_SAFE_BOUNDS.z_w_m[0]),
        "mission_terminal_z_max_m": float(TRUE_SAFE_BOUNDS.z_w_m[1]),
        "final_exit_x_w_m": float(exit_position[0]) if exit_position is not None else float("nan"),
        "final_exit_y_w_m": float(exit_position[1]) if exit_position is not None else float("nan"),
        "final_exit_z_w_m": float(exit_position[2]) if exit_position is not None else float("nan"),
        "terminal_specific_energy_m": float(terminal_energy),
        "terminal_specific_energy_reference_m": float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M),
        "terminal_specific_energy_reserve_m": float(terminal_energy_reserve),
        "terminal_specific_energy_source": terminal_energy_source,
        "mission_outcome_label": _mission_outcome_label(
            mission_success=mission_success,
            wrong_wall_exit=wrong_wall_exit,
            hard_failure=hard_failure,
            floor_or_ceiling=floor_or_ceiling,
            no_viable=no_viable,
            expected_low_energy_sink=expected_low_energy_sink,
            lift_capture=any(_float_value(row.get("lift_dwell_time_s", 0.0)) > 0.0 for row in primitive_rows),
        ),
    }


def _exit_position_from_rollout_row(row: dict[str, object]) -> np.ndarray | None:
    state = _state_vector_from_json(row.get("exit_state_vector", ""))
    if state is not None:
        return np.asarray(state[:3], dtype=float)
    keys = ("exit_x_w", "exit_y_w", "exit_z_w")
    if all(key in row for key in keys):
        try:
            return np.asarray([float(row[key]) for key in keys], dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def _terminal_specific_energy_from_rollout_row(row: dict[str, object]) -> tuple[float, str]:
    if not row:
        return float("nan"), "unavailable"
    state = _state_vector_from_json(row.get("exit_state_vector", ""))
    if state is not None:
        return float(_specific_energy_m(state)), "exit_state_specific_energy_height_plus_speed"
    position = _exit_position_from_rollout_row(row)
    if position is not None:
        return float(position[2]), "exit_position_height_only_fallback"
    return float("nan"), "unavailable"


def _rollout_row_is_wall_boundary_exit(row: dict[str, object]) -> bool:
    if not row:
        return False
    cause = str(row.get("termination_cause", ""))
    label = str(row.get("failure_label", ""))
    boundary_class = str(row.get("boundary_use_class", ""))
    return bool(
        "wall" in cause
        or "xy_boundary" in label
        or label == "uncontrolled_xy_boundary_exit"
        or boundary_class == "episode_terminal_useful"
        and _float_value(row.get("minimum_wall_margin_m", 0.0)) < 0.0
    )


def _wall_exit_face(exit_position: np.ndarray | None, *, wall_exit: bool) -> str:
    if not wall_exit:
        return "none"
    if exit_position is None:
        return "unknown_xy_wall"
    x_w, y_w, _z_w = [float(value) for value in exit_position[:3]]
    x_min, x_max = TRUE_SAFE_BOUNDS.x_w_m
    y_min, y_max = TRUE_SAFE_BOUNDS.y_w_m
    tol = MISSION_FRONT_WALL_X_TOL_M
    if x_w >= float(x_max) - tol:
        return "front_wall_x_max"
    if x_w <= float(x_min) + tol:
        return "rear_wall_x_min"
    if y_w <= float(y_min) + tol:
        return "side_wall_y_min"
    if y_w >= float(y_max) - tol:
        return "side_wall_y_max"
    return "unknown_xy_wall"


def _exit_position_inside_terminal_yz(exit_position: np.ndarray | None) -> bool:
    if exit_position is None:
        return False
    _x_w, y_w, z_w = [float(value) for value in exit_position[:3]]
    y_min, y_max = TRUE_SAFE_BOUNDS.y_w_m
    z_min, z_max = TRUE_SAFE_BOUNDS.z_w_m
    tol = MISSION_FRONT_WALL_X_TOL_M
    return bool(
        float(y_min) - tol <= y_w <= float(y_max) + tol
        and float(z_min) - tol <= z_w <= float(z_max) + tol
    )


def _mission_outcome_label(
    *,
    mission_success: bool,
    wrong_wall_exit: bool,
    hard_failure: bool,
    floor_or_ceiling: bool,
    no_viable: bool,
    expected_low_energy_sink: bool,
    lift_capture: bool,
) -> str:
    if mission_success:
        return "front_wall_terminal_success"
    if expected_low_energy_sink:
        return "expected_low_energy_dry_air_sink_not_scored"
    if floor_or_ceiling:
        return "floor_or_ceiling_violation"
    if hard_failure:
        return "hard_failure"
    if no_viable:
        return "no_viable_primitive"
    if wrong_wall_exit:
        return "wrong_wall_exit"
    if lift_capture:
        return "lift_capture_without_front_wall_terminal"
    return "no_front_wall_terminal"


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
    front_wall_terminal = _truthy(row.get("front_wall_terminal_success", row.get("mission_success", False)))
    mission_success = _truthy(row.get("mission_success", front_wall_terminal))
    wrong_wall_exit = _truthy(row.get("wrong_wall_exit", False))
    expected_low_energy_sink = _truthy(row.get("expected_low_energy_dry_air_sink", False))
    base_penalty, penalty_reason = _base_failure_penalty(
        hard_failure=hard_failure,
        floor_or_ceiling=floor_or_ceiling,
        no_viable_at_launch=no_viable_at_launch,
        no_viable_after_launch=no_viable_after_launch,
        wrong_wall_exit=wrong_wall_exit,
        expected_low_energy_sink=expected_low_energy_sink,
    )
    lift_capture = _truthy(row.get("lift_capture", False))
    mission_component = _mission_completion_component(
        mission_success=mission_success,
        lift_capture=lift_capture,
        safe_success=_truthy(row.get("safe_success", False)),
        penalty_reason=penalty_reason,
    )
    updraft_gain_bonus = _clip(
        UPDRAFT_GAIN_SCORE_PER_M * max(_float_value(row.get("updraft_specific_energy_gain_proxy_m", 0.0)), 0.0),
        0.0,
        UPDRAFT_GAIN_SCORE_CAP,
    )
    lift_dwell_bonus = _clip(
        LIFT_DWELL_SCORE_PER_S * max(_float_value(row.get("lift_dwell_time_s", 0.0)), 0.0),
        0.0,
        LIFT_DWELL_SCORE_CAP,
    )
    terminal_specific_energy = _float_value(row.get("terminal_specific_energy_m", float("nan")), default=float("nan"))
    terminal_specific_energy_reserve = (
        terminal_specific_energy - float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M)
        if math.isfinite(float(terminal_specific_energy))
        else float("nan")
    )
    terminal_specific_energy_bonus = _terminal_specific_energy_bonus(
        mission_success=mission_success,
        terminal_specific_energy_m=terminal_specific_energy,
    )
    additive_component = (
        0.0
        if penalty_reason != "none"
        else float(mission_component + updraft_gain_bonus + lift_dwell_bonus + terminal_specific_energy_bonus)
    )
    return {
        "launch_score_version": LAUNCH_SCORE_VERSION,
        "mission_score_version": LAUNCH_SCORE_VERSION,
        "mission_success": bool(mission_success),
        "front_wall_terminal_success": bool(front_wall_terminal),
        "wrong_wall_exit": bool(wrong_wall_exit),
        "terminal_wall_face": str(row.get("terminal_wall_face", "")),
        "mission_outcome_label": str(row.get("mission_outcome_label", "")),
        "episode_flight_time_s": float(episode_time_s),
        "airborne_time_reward_status": "audit_only_not_rewarded",
        "base_failure_penalty": float(base_penalty),
        "base_failure_penalty_reason": penalty_reason,
        "mission_completion_component": float(mission_component),
        "updraft_gain_bonus": float(updraft_gain_bonus),
        "lift_dwell_bonus": float(lift_dwell_bonus),
        "terminal_specific_energy_m": float(terminal_specific_energy),
        "terminal_specific_energy_reference_m": float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M),
        "terminal_specific_energy_reserve_m": float(terminal_specific_energy_reserve),
        "terminal_specific_energy_bonus": float(terminal_specific_energy_bonus),
        "terminal_specific_energy_bonus_status": (
            "front_wall_terminal_only"
            if mission_success
            else "not_applied_without_front_wall_terminal"
        ),
        "launch_score_additive_component": float(additive_component),
        "launch_score": float(base_penalty + additive_component),
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
        "mission_score_version": LAUNCH_SCORE_VERSION,
        "mission_success": False,
        "front_wall_terminal_success": False,
        "wrong_wall_exit": False,
        "terminal_wall_face": "not_scored_history_launch",
        "mission_outcome_label": "not_scored_history_launch",
        "episode_flight_time_s": float(episode_time_s),
        "airborne_time_reward_status": "audit_only_not_rewarded",
        "base_failure_penalty": float("nan"),
        "base_failure_penalty_reason": "not_scored_history_launch",
        "mission_completion_component": float("nan"),
        "updraft_gain_bonus": float("nan"),
        "lift_dwell_bonus": float("nan"),
        "terminal_specific_energy_m": float("nan"),
        "terminal_specific_energy_reference_m": float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M),
        "terminal_specific_energy_reserve_m": float("nan"),
        "terminal_specific_energy_bonus": float("nan"),
        "terminal_specific_energy_bonus_status": "not_scored_history_launch",
        "launch_score_additive_component": float("nan"),
        "launch_score": float("nan"),
        "launch_score_scope": "history_launch_memory_update_not_outer_loop_score",
    }


def _base_failure_penalty(
    *,
    hard_failure: bool,
    floor_or_ceiling: bool,
    no_viable_at_launch: bool,
    no_viable_after_launch: bool,
    wrong_wall_exit: bool,
    expected_low_energy_sink: bool,
) -> tuple[float, str]:
    if expected_low_energy_sink:
        return 0.0, "expected_low_energy_dry_air_sink_not_scored"
    if hard_failure:
        return -100.0, "hard_failure"
    if floor_or_ceiling:
        return -100.0, "floor_or_ceiling_violation"
    if no_viable_at_launch:
        return -70.0, "no_viable_primitive_at_launch"
    if no_viable_after_launch:
        return -40.0, "no_viable_primitive_after_launch"
    if wrong_wall_exit:
        return WRONG_WALL_EXIT_PENALTY, "wrong_wall_exit"
    return 0.0, "none"


def _mission_completion_component(
    *,
    mission_success: bool,
    lift_capture: bool,
    safe_success: bool,
    penalty_reason: str,
) -> float:
    if penalty_reason != "none":
        return 0.0
    if mission_success:
        return MISSION_COMPLETION_SCORE
    if lift_capture:
        return MISSION_LIFT_CAPTURE_BASE_SCORE
    if safe_success:
        return MISSION_SAFE_ROLLOUT_BASE_SCORE
    return 0.0


def _terminal_specific_energy_bonus(*, mission_success: bool, terminal_specific_energy_m: float) -> float:
    if not mission_success or not math.isfinite(float(terminal_specific_energy_m)):
        return 0.0
    reserve_m = float(terminal_specific_energy_m) - float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M)
    return _clip(
        TERMINAL_SPECIFIC_ENERGY_SCORE_PER_M * max(reserve_m, 0.0),
        0.0,
        TERMINAL_SPECIFIC_ENERGY_SCORE_CAP,
    )


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
        "belief_effective_observation_count": float(features.get("belief_effective_observation_count", 0.0) or 0.0),
        "belief_recency_weight": float(features.get("belief_recency_weight", 0.0) or 0.0),
        "belief_observation_age": int(features.get("belief_observation_age", 0) or 0),
        "belief_launch_recency_weight": float(features.get("belief_launch_recency_weight", 0.0) or 0.0),
        "belief_history_launch_age": int(features.get("belief_history_launch_age", 0) or 0),
        "belief_last_history_launch_index": int(features.get("belief_last_history_launch_index", -1) or -1),
        "belief_current_history_launch_index": int(features.get("belief_current_history_launch_index", -1) or -1),
        "belief_launch_recency_half_life": float(features.get("belief_launch_recency_half_life", 0.0) or 0.0),
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
                features.get("belief_local_specific_energy_residual_m", features.get("belief_local_updraft_gain_residual_m", 0.0)),
            )
            or 0.0
        ),
        "belief_local_specific_energy_residual_m": float(
            features.get(
                "belief_local_specific_energy_residual_m",
                features.get("belief_local_energy_residual_m", 0.0),
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


def _mean_float(rows: list[dict[str, object]], column: str) -> float:
    if not rows:
        return 0.0
    return float(sum(_float_value(row.get(column, 0.0)) for row in rows) / max(1, len(rows)))


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
    buffers["history_plot_trace"].extend(result["history_plot_rows"])
    buffers["history_memory_trace"].extend(result["history_memory_rows"])
    buffers["history_selector_summary"].extend(result["history_selector_rows"])
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
    _write_metric_table_with_large_file_guard(
        run_root=run_root,
        table_name=table_name,
        frame=pd.DataFrame(rows),
        run_id=run_id,
        storage_format=storage_format,
        compression_level=compression_level,
        inline_row_limit=SCHEDULE_INLINE_ROW_LIMIT,
        partition_row_count=SCHEDULE_PARTITION_ROW_COUNT,
    )


def _write_metric_table_with_large_file_guard(
    *,
    run_root: Path,
    table_name: str,
    frame: pd.DataFrame,
    run_id: int,
    storage_format: str,
    compression_level: int,
    inline_row_limit: int,
    partition_row_count: int,
) -> None:
    metrics_path = run_root / "metrics" / f"{table_name}.csv"
    if len(frame) <= int(inline_row_limit):
        _write_csv(metrics_path, frame)
        return

    partitions: list[TablePartition] = []
    extension = table_extension(storage_format)
    for partition_index, start in enumerate(range(0, len(frame), int(partition_row_count))):
        chunk = frame.iloc[start : start + int(partition_row_count)].copy()
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
                    "partition_row_count": int(partition_row_count),
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


def _write_real_time_scheduler_audit_from_partitions(
    run_root: Path,
    partitions: Iterable[TablePartition],
    storage_format: str,
) -> None:
    rows: list[dict[str, object]] = []
    for partition in partitions:
        if partition.table_name != "selector_decision_log":
            continue
        frame = read_table_partition(
            run_root / "tables" / partition.relative_path,
            storage_format=storage_format,
        )
        if not frame.empty:
            rows.extend(frame.to_dict(orient="records"))
    if not rows:
        _write_csv(
            run_root / "metrics" / "real_time_scheduler_audit.csv",
            pd.DataFrame(
                [
                    {
                        "audit_scope": "overall",
                        "selector_decision_count": 0,
                        "audit_status": "no_selector_decision_rows",
                        "real_time_claim_status": "controller_compute_profile_excludes_table_flush_and_posthoc_diagnostics",
                    }
                ]
            ),
        )
        return

    frame = pd.DataFrame(rows)
    duration_source_column = (
        "decision_controller_compute_duration_s"
        if "decision_controller_compute_duration_s" in frame.columns
        else "decision_total_duration_s"
    )
    duration = pd.to_numeric(frame.get(duration_source_column, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    preferred = frame.get("preferred_20ms_slot_met", pd.Series(dtype=bool)).map(_truthy) if "preferred_20ms_slot_met" in frame else pd.Series(False, index=frame.index)
    hard = frame.get("hard_100ms_boundary_met", pd.Series(dtype=bool)).map(_truthy) if "hard_100ms_boundary_met" in frame else pd.Series(False, index=frame.index)
    prepared = (
        frame.get("scheduler_prepared_before_primitive_boundary", pd.Series(dtype=bool)).map(_truthy)
        if "scheduler_prepared_before_primitive_boundary" in frame
        else pd.Series(False, index=frame.index)
    )
    frame["_duration"] = duration
    frame["_duration_source_column"] = str(duration_source_column)
    frame["_preferred"] = preferred.astype(bool)
    frame["_hard"] = hard.astype(bool)
    frame["_prepared"] = prepared.astype(bool)

    audit_rows = [_real_time_scheduler_summary_row("overall", frame)]
    if "library_size_case_id" in frame.columns:
        library_case = frame["library_size_case_id"].astype(str)
        real_flight_required = frame[library_case.isin(REAL_FLIGHT_REQUIRED_LIBRARY_CASE_IDS)]
        optional_extended = frame[library_case.isin(REAL_FLIGHT_OPTIONAL_LIBRARY_CASE_IDS)]
        offline_no_cluster = frame[library_case.isin(OFFLINE_UNRESTRICTED_LIBRARY_CASE_IDS)]
        if not real_flight_required.empty:
            audit_rows.append(_real_time_scheduler_summary_row("real_flight_required:heavy_balanced", real_flight_required))
        if not optional_extended.empty:
            audit_rows.append(_real_time_scheduler_summary_row("real_flight_optional_extended:light_super_light", optional_extended))
        if not offline_no_cluster.empty:
            audit_rows.append(
                _real_time_scheduler_summary_row(
                    "offline_comparison:unrestricted:not_real_flight_candidate:no_cluster_no_merge",
                    offline_no_cluster,
                )
            )
        if "scheduler_decision_source" in frame.columns:
            source = frame["scheduler_decision_source"].astype(str)
            required_inflight = frame[
                library_case.isin(REAL_FLIGHT_REQUIRED_LIBRARY_CASE_IDS)
                & source.ne("initial_launch_precomputed_before_release")
            ]
            optional_inflight = frame[
                library_case.isin(REAL_FLIGHT_OPTIONAL_LIBRARY_CASE_IDS)
                & source.ne("initial_launch_precomputed_before_release")
            ]
            pre_release = frame[source.eq("initial_launch_precomputed_before_release")]
            if not required_inflight.empty:
                audit_rows.append(
                    _real_time_scheduler_summary_row(
                        "real_flight_required:heavy_balanced:inflight_boundary_decisions",
                        required_inflight,
                    )
                )
            if not optional_inflight.empty:
                audit_rows.append(
                    _real_time_scheduler_summary_row(
                        "real_flight_optional_extended:light_super_light:inflight_boundary_decisions",
                        optional_inflight,
                    )
                )
            if not pre_release.empty:
                audit_rows.append(
                    _real_time_scheduler_summary_row(
                        "pre_release_initial_decisions:not_inflight_boundary_compute",
                        pre_release,
                    )
                )
        for case_id, group in frame.groupby("library_size_case_id", dropna=False):
            audit_rows.append(_real_time_scheduler_summary_row(f"library_size_case_id:{case_id}", group))
    if "launch_role" in frame.columns:
        for launch_role, group in frame.groupby("launch_role", dropna=False):
            audit_rows.append(_real_time_scheduler_summary_row(f"launch_role:{launch_role}", group))
    if "policy_id" in frame.columns:
        for policy_id, group in frame.groupby("policy_id", dropna=False):
            audit_rows.append(_real_time_scheduler_summary_row(f"policy_id:{policy_id}", group))
    _write_csv(run_root / "metrics" / "real_time_scheduler_audit.csv", pd.DataFrame(audit_rows))


def _write_memory_opportunity_audit_from_partitions(
    run_root: Path,
    partitions: Iterable[TablePartition],
    storage_format: str,
    *,
    run_id: int = 1,
    compression_level: int = 1,
) -> None:
    rows: list[dict[str, object]] = []
    for partition in partitions:
        if partition.table_name != "selector_decision_log":
            continue
        frame = read_table_partition(
            run_root / "tables" / partition.relative_path,
            storage_format=storage_format,
        )
        if not frame.empty:
            rows.extend(frame.to_dict(orient="records"))
    if not rows:
        empty = pd.DataFrame(
            [
                {
                    "audit_scope": "overall",
                    "decision_count": 0,
                    "audit_status": "no_selector_decision_rows",
                    "memory_policy_version": OUTER_LOOP_MEMORY_POLICY_VERSION,
                }
            ]
        )
        _write_csv(run_root / "metrics" / "memory_opportunity_summary.csv", empty)
        _write_csv(run_root / "metrics" / "memory_opportunity_decision_log.csv", pd.DataFrame())
        return

    frame = pd.DataFrame(rows)
    frame["_memory_policy"] = frame.get("policy_id", pd.Series("", index=frame.index)).astype(str).str.contains(
        MEMORY_POLICY_PREFIX,
        regex=False,
    )
    frame["_baseline_variant"] = frame.get("selected_memory_shield_baseline_variant_id", pd.Series("", index=frame.index)).astype(str)
    frame["_memory_variant"] = frame.get("selected_memory_shield_memory_variant_id", pd.Series("", index=frame.index)).astype(str)
    frame["_memory_switch_available"] = (
        frame["_memory_policy"]
        & frame["_baseline_variant"].ne("")
        & frame["_memory_variant"].ne("")
        & frame["_baseline_variant"].ne(frame["_memory_variant"])
    )
    frame["_shield_status"] = frame.get("selected_memory_shield_status", pd.Series("", index=frame.index)).astype(str)
    frame["_accepted_switch"] = frame["_memory_switch_available"] & frame["_shield_status"].str.startswith("accepted")
    frame["_rejected_switch"] = frame["_memory_switch_available"] & frame["_shield_status"].str.startswith("rejected")
    for column in (
        "selected_memory_shield_score_margin",
        "selected_memory_shield_memory_correction_delta",
        "selected_memory_near_tie_base_score_margin",
        "selected_memory_near_tie_base_score_gap_to_best",
        "selected_memory_near_tie_factor",
        "selected_raw_memory_score_component",
        "selected_effective_memory_score_component",
        "selected_memory_shield_base_score_gap_to_baseline",
        "selected_memory_shield_memory_opportunity_ratio",
        "selected_memory_shield_candidate_path_confidence",
        "selected_memory_shield_candidate_path_uncertainty",
        "selected_memory_shield_exploration_score_component",
        "selected_memory_shield_information_gain_score_component",
        "selected_memory_shield_information_gain",
        "selected_memory_shield_route_score_component",
        "selected_memory_shield_route_exploitation_m",
        "selected_memory_shield_route_information_gain",
        "selected_memory_shield_route_confidence",
        "selected_memory_shield_path_exit_margin_delta_m",
        "selected_calibrated_regime_alpha_abs_deg",
        "selected_calibrated_regime_transition_start_alpha_deg",
        "selected_calibrated_regime_post_stall_alpha_deg",
        "selected_calibrated_transition_activation",
        "selected_calibrated_post_stall_activation",
        "selected_calibrated_regime_mismatch_risk",
        "selected_calibrated_regime_mismatch_score_component",
        "selected_memory_shield_baseline_calibrated_regime_mismatch_risk",
        "selected_memory_shield_memory_calibrated_regime_mismatch_risk",
        "selected_memory_shield_calibrated_regime_mismatch_risk_delta",
        "selected_flow_map_grid_resolution_m",
        "selected_flow_map_query_radius_m",
        "selected_flow_map_reachable_attraction_m",
        "selected_flow_map_reachable_attraction_confidence",
        "selected_flow_map_reachable_attraction_query_count",
        "selected_flow_map_reachable_attraction_observation_count",
        "selected_flow_map_candidate_path_uncertainty",
        "selected_flow_map_memory_guided_exploration_uncertainty",
        "selected_flow_map_information_gain",
        "selected_flow_map_information_gain_path_uncertainty",
        "selected_flow_map_information_gain_reachable_uncertainty",
        "selected_flow_map_information_gain_query_count",
        "selected_flow_map_information_gain_low_confidence_query_count",
        "selected_memory_route_score_component",
        "selected_memory_route_gate",
        "selected_memory_cost_benefit_known_flow_benefit_m",
        "selected_memory_cost_benefit_information_benefit",
        "selected_memory_cost_benefit_total_benefit",
        "selected_memory_cost_benefit_total_cost",
        "selected_memory_cost_benefit_net_value",
        "selected_flow_map_route_exploitation_m",
        "selected_flow_map_route_information_gain",
        "selected_flow_map_route_confidence",
        "selected_flow_map_route_front_progress",
        "selected_flow_map_route_safe_fraction",
    ):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    decision_columns = [
        column
        for column in (
            "library_size_case_id",
            "policy_id",
            "history_length",
            "launch_role",
            "outer_case_index",
            "outer_case_type",
            "environment_block_id",
            "episode_id",
            "primitive_step_index",
            "start_state_family",
            "route_required_entry_class",
            "selected_primitive_id",
            "selected_primitive_variant_id",
            "selected_memory_shield_status",
            "selected_memory_shield_accepted",
            "selected_memory_shield_baseline_variant_id",
            "selected_memory_shield_memory_variant_id",
            "selected_memory_shield_score_margin",
            "selected_memory_shield_memory_correction_delta",
            "selected_memory_near_tie_base_score_margin",
            "selected_memory_near_tie_base_score_gap_to_best",
            "selected_memory_near_tie_factor",
            "selected_raw_memory_score_component",
            "selected_effective_memory_score_component",
            "selected_memory_shield_base_score_gap_to_baseline",
            "selected_memory_shield_memory_opportunity_ratio",
            "selected_memory_shield_candidate_path_confidence",
            "selected_memory_shield_candidate_path_uncertainty",
            "selected_memory_shield_exploration_score_component",
            "selected_memory_shield_information_gain_score_component",
            "selected_memory_shield_information_gain",
            "selected_memory_shield_route_score_component",
            "selected_memory_shield_route_exploitation_m",
            "selected_memory_shield_route_information_gain",
            "selected_memory_shield_route_confidence",
            "selected_memory_shield_path_exit_margin_delta_m",
            "selected_calibrated_regime_alpha_abs_deg",
            "selected_calibrated_regime_transition_start_alpha_deg",
            "selected_calibrated_regime_post_stall_alpha_deg",
            "selected_calibrated_regime_label",
            "selected_calibrated_regime_source_calibration_id",
            "selected_calibrated_regime_mismatch_risk",
            "selected_calibrated_regime_mismatch_score_component",
            "selected_memory_shield_baseline_calibrated_regime_mismatch_risk",
            "selected_memory_shield_memory_calibrated_regime_mismatch_risk",
            "selected_memory_shield_calibrated_regime_mismatch_risk_delta",
            "selected_flow_map_grid_resolution_m",
            "selected_flow_map_query_radius_m",
            "selected_flow_map_reachable_attraction_m",
            "selected_flow_map_reachable_attraction_confidence",
            "selected_flow_map_reachable_attraction_query_count",
            "selected_flow_map_reachable_attraction_observation_count",
            "selected_flow_map_candidate_path_uncertainty",
            "selected_flow_map_memory_guided_exploration_uncertainty",
            "selected_flow_map_information_gain",
            "selected_flow_map_information_gain_path_uncertainty",
            "selected_flow_map_information_gain_reachable_uncertainty",
            "selected_flow_map_information_gain_query_count",
            "selected_flow_map_information_gain_low_confidence_query_count",
            "selected_memory_route_score_component",
            "selected_memory_route_gate",
            "selected_memory_route_horizon_primitives",
            "selected_memory_cost_benefit_known_flow_benefit_m",
            "selected_memory_cost_benefit_information_benefit",
            "selected_memory_cost_benefit_total_benefit",
            "selected_memory_cost_benefit_total_cost",
            "selected_memory_cost_benefit_net_value",
            "selected_flow_map_route_policy",
            "selected_flow_map_route_exploitation_m",
            "selected_flow_map_route_information_gain",
            "selected_flow_map_route_confidence",
            "selected_flow_map_route_front_progress",
            "selected_flow_map_route_safe_fraction",
        )
        if column in frame.columns
    ]
    decision_log = frame[decision_columns].copy()
    decision_log["memory_opportunity_policy"] = (
        "decision_level_baseline_vs_candidate_path_memory_gap_audit"
    )
    decision_log["memory_policy_version"] = OUTER_LOOP_MEMORY_POLICY_VERSION
    _write_metric_table_with_large_file_guard(
        run_root=run_root,
        table_name="memory_opportunity_decision_log",
        frame=decision_log,
        run_id=run_id,
        storage_format=storage_format,
        compression_level=compression_level,
        inline_row_limit=MEMORY_OPPORTUNITY_DECISION_LOG_INLINE_ROW_LIMIT,
        partition_row_count=MEMORY_OPPORTUNITY_DECISION_LOG_PARTITION_ROW_COUNT,
    )

    group_columns = [
        column
        for column in ("launch_role", "environment_block_id", "policy_id", "library_size_case_id")
        if column in frame.columns
    ]
    summary_rows = [_memory_opportunity_summary_row("overall", frame)]
    if group_columns:
        for values, group in frame.groupby(group_columns, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            label = ",".join(f"{column}:{value}" for column, value in zip(group_columns, values))
            summary_rows.append(_memory_opportunity_summary_row(label, group))
    _write_csv(run_root / "metrics" / "memory_opportunity_summary.csv", pd.DataFrame(summary_rows))


def _memory_opportunity_summary_row(audit_scope: str, frame: pd.DataFrame) -> dict[str, object]:
    count = int(len(frame))
    memory_policy = frame["_memory_policy"].astype(bool) if "_memory_policy" in frame else pd.Series(False, index=frame.index)
    switch_available = (
        frame["_memory_switch_available"].astype(bool)
        if "_memory_switch_available" in frame
        else pd.Series(False, index=frame.index)
    )
    accepted = frame["_accepted_switch"].astype(bool) if "_accepted_switch" in frame else pd.Series(False, index=frame.index)
    rejected = frame["_rejected_switch"].astype(bool) if "_rejected_switch" in frame else pd.Series(False, index=frame.index)
    correction = frame.get("selected_memory_shield_memory_correction_delta", pd.Series(0.0, index=frame.index))
    gap = frame.get("selected_memory_shield_base_score_gap_to_baseline", pd.Series(0.0, index=frame.index))
    ratio = frame.get("selected_memory_shield_memory_opportunity_ratio", pd.Series(0.0, index=frame.index))
    confidence = frame.get("selected_memory_shield_candidate_path_confidence", pd.Series(0.0, index=frame.index))
    uncertainty = frame.get("selected_memory_shield_candidate_path_uncertainty", pd.Series(0.0, index=frame.index))
    score_margin = frame.get("selected_memory_shield_score_margin", pd.Series(0.0, index=frame.index))
    attraction = frame.get("selected_flow_map_reachable_attraction_m", pd.Series(0.0, index=frame.index))
    information_gain = frame.get("selected_flow_map_information_gain", pd.Series(0.0, index=frame.index))
    information_component = frame.get(
        "selected_memory_shield_information_gain_score_component",
        pd.Series(0.0, index=frame.index),
    )
    route_component = frame.get("selected_memory_shield_route_score_component", pd.Series(0.0, index=frame.index))
    route_exploitation = frame.get("selected_flow_map_route_exploitation_m", pd.Series(0.0, index=frame.index))
    route_information_gain = frame.get("selected_flow_map_route_information_gain", pd.Series(0.0, index=frame.index))
    route_confidence = frame.get("selected_flow_map_route_confidence", pd.Series(0.0, index=frame.index))
    selected_regime_risk = frame.get("selected_calibrated_regime_mismatch_risk", pd.Series(0.0, index=frame.index))
    regime_risk_delta = frame.get(
        "selected_memory_shield_calibrated_regime_mismatch_risk_delta",
        pd.Series(0.0, index=frame.index),
    )
    cost_benefit_benefit = frame.get(
        "selected_memory_cost_benefit_total_benefit",
        pd.Series(0.0, index=frame.index),
    )
    cost_benefit_cost = frame.get(
        "selected_memory_cost_benefit_total_cost",
        pd.Series(0.0, index=frame.index),
    )
    cost_benefit_net = frame.get(
        "selected_memory_cost_benefit_net_value",
        pd.Series(0.0, index=frame.index),
    )
    return {
        "audit_scope": str(audit_scope),
        "decision_count": count,
        "memory_policy_decision_count": int(memory_policy.sum()),
        "memory_switch_available_count": int(switch_available.sum()),
        "accepted_memory_switch_count": int(accepted.sum()),
        "rejected_memory_switch_count": int(rejected.sum()),
        "mean_memory_correction_delta": float(correction.mean()) if count else 0.0,
        "max_memory_correction_delta": float(correction.max()) if count else 0.0,
        "mean_base_score_gap_to_baseline": float(gap.mean()) if count else 0.0,
        "mean_memory_opportunity_ratio": float(ratio.mean()) if count else 0.0,
        "opportunity_ratio_ge_1_count": int((ratio >= 1.0).sum()) if count else 0,
        "mean_candidate_path_confidence": float(confidence.mean()) if count else 0.0,
        "mean_candidate_path_uncertainty": float(uncertainty.mean()) if count else 0.0,
        "mean_reachable_flow_attraction_m": float(attraction.mean()) if count else 0.0,
        "reachable_flow_attraction_positive_count": int((attraction > 0.0).sum()) if count else 0,
        "mean_flow_map_information_gain": float(information_gain.mean()) if count else 0.0,
        "flow_map_information_gain_positive_count": int((information_gain > 0.0).sum()) if count else 0,
        "mean_information_gain_score_component": float(information_component.mean()) if count else 0.0,
        "information_gain_score_component_positive_count": int((information_component > 0.0).sum()) if count else 0,
        "mean_route_memory_score_component": float(route_component.mean()) if count else 0.0,
        "route_memory_score_component_positive_count": int((route_component > 0.0).sum()) if count else 0,
        "mean_route_flow_exploitation_m": float(route_exploitation.mean()) if count else 0.0,
        "mean_route_flow_information_gain": float(route_information_gain.mean()) if count else 0.0,
        "mean_route_flow_confidence": float(route_confidence.mean()) if count else 0.0,
        "mean_selected_calibrated_regime_mismatch_risk": float(selected_regime_risk.mean()) if count else 0.0,
        "mean_memory_switch_calibrated_regime_risk_delta": float(regime_risk_delta.mean()) if count else 0.0,
        "memory_switch_regime_risk_regression_count": int((regime_risk_delta > 0.0).sum()) if count else 0,
        "mean_memory_cost_benefit_total_benefit": float(cost_benefit_benefit.mean()) if count else 0.0,
        "mean_memory_cost_benefit_total_cost": float(cost_benefit_cost.mean()) if count else 0.0,
        "mean_memory_cost_benefit_net_value": float(cost_benefit_net.mean()) if count else 0.0,
        "memory_cost_benefit_positive_count": int((cost_benefit_net > 0.0).sum()) if count else 0,
        "mean_adaptive_score_margin": float(score_margin.mean()) if count else 0.0,
        "memory_policy_version": OUTER_LOOP_MEMORY_POLICY_VERSION,
        "audit_status": "profiled",
    }


def _real_time_scheduler_summary_row(audit_scope: str, frame: pd.DataFrame) -> dict[str, object]:
    count = int(len(frame))
    duration = pd.to_numeric(frame["_duration"], errors="coerce").fillna(0.0)
    preferred = frame["_preferred"].astype(bool)
    hard = frame["_hard"].astype(bool)
    prepared = frame["_prepared"].astype(bool)
    duration_source = (
        str(frame["_duration_source_column"].iloc[0])
        if "_duration_source_column" in frame and count
        else "decision_total_duration_s"
    )
    evaluated_candidates = pd.to_numeric(
        frame.get("decision_candidate_count", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    total_candidates = pd.to_numeric(
        frame.get("decision_total_library_candidate_count", evaluated_candidates),
        errors="coerce",
    ).fillna(0.0)
    skipped_candidates = pd.to_numeric(
        frame.get("decision_prefilter_skipped_candidate_count", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    requirement = _real_time_timing_requirement_for_scope(audit_scope)
    hard_rate = float(hard.mean()) if count else 0.0
    return {
        "audit_scope": str(audit_scope),
        **requirement,
        "selector_decision_count": count,
        "prepared_before_boundary_count": int(prepared.sum()),
        "prepared_before_boundary_rate": float(prepared.mean()) if count else 0.0,
        "preferred_20ms_slot_met_count": int(preferred.sum()),
        "preferred_20ms_slot_met_rate": float(preferred.mean()) if count else 0.0,
        "hard_100ms_boundary_met_count": int(hard.sum()),
        "hard_100ms_boundary_met_rate": hard_rate,
        "hard_100ms_requirement_passed": _real_time_requirement_passed(requirement, hard_rate, count),
        "max_decision_total_duration_s": float(duration.max()) if count else 0.0,
        "p99_decision_total_duration_s": float(duration.quantile(0.99)) if count else 0.0,
        "mean_decision_total_duration_s": float(duration.mean()) if count else 0.0,
        "timing_duration_source_column": duration_source,
        "timing_boundary_scope": "controller_compute_excludes_table_flush_and_posthoc_diagnostics",
        "mean_evaluated_candidate_count": float(evaluated_candidates.mean()) if count else 0.0,
        "max_evaluated_candidate_count": int(evaluated_candidates.max()) if count else 0,
        "mean_total_library_candidate_count": float(total_candidates.mean()) if count else 0.0,
        "max_total_library_candidate_count": int(total_candidates.max()) if count else 0,
        "mean_prefilter_skipped_candidate_count": float(skipped_candidates.mean()) if count else 0.0,
        "preferred_decision_budget_s": float(REAL_TIME_PREFERRED_DECISION_BUDGET_S),
        "hard_decision_budget_s": float(REAL_TIME_HARD_DECISION_BUDGET_S),
        "audit_status": "profiled",
        "real_time_claim_status": "controller_compute_profile_excludes_table_flush_and_posthoc_diagnostics",
    }


def _real_time_timing_requirement_for_scope(audit_scope: str) -> dict[str, object]:
    scope = str(audit_scope)
    if scope.startswith("real_flight_required:heavy_balanced:inflight_boundary_decisions"):
        return {
            "timing_requirement_tier": "required_real_flight_candidate",
            "timing_requirement_policy": "all_inflight_decisions_must_meet_100ms_for_heavy_and_balanced",
            "hard_100ms_required_rate": 1.0,
        }
    if scope.startswith("real_flight_required:heavy_balanced"):
        return {
            "timing_requirement_tier": "required_real_flight_candidate_aggregate",
            "timing_requirement_policy": "aggregate_includes_pre_release_report_inflight_scope_for_gate",
            "hard_100ms_required_rate": "",
        }
    if scope.startswith("real_flight_optional_extended:light_super_light"):
        return {
            "timing_requirement_tier": "optional_extended_library_diagnostic",
            "timing_requirement_policy": "limited_100ms_violations_allowed_report_rate_not_real_flight_gate",
            "hard_100ms_required_rate": "",
        }
    if "no_cluster_no_merge" in scope:
        return {
            "timing_requirement_tier": "offline_unrestricted_comparison",
            "timing_requirement_policy": "no_real_flight_timing_restriction",
            "hard_100ms_required_rate": "",
        }
    if scope.startswith("pre_release_initial_decisions"):
        return {
            "timing_requirement_tier": "pre_release_preparation",
            "timing_requirement_policy": "not_an_inflight_boundary_compute_gate",
            "hard_100ms_required_rate": "",
        }
    if scope.startswith("library_size_case_id:heavy_cluster") or scope.startswith("library_size_case_id:balanced_cluster"):
        return {
            "timing_requirement_tier": "required_real_flight_candidate_case_diagnostic",
            "timing_requirement_policy": "case_scope_includes_pre_release_report_required_inflight_scope_for_gate",
            "hard_100ms_required_rate": "",
        }
    if scope.startswith("library_size_case_id:light_cluster") or scope.startswith("library_size_case_id:super_light_cluster"):
        return {
            "timing_requirement_tier": "optional_extended_library_diagnostic",
            "timing_requirement_policy": "limited_100ms_violations_allowed_report_rate_not_real_flight_gate",
            "hard_100ms_required_rate": "",
        }
    return {
        "timing_requirement_tier": "aggregate_diagnostic",
        "timing_requirement_policy": "reported_only_mixed_scope",
        "hard_100ms_required_rate": "",
    }


def _real_time_requirement_passed(requirement: dict[str, object], hard_rate: float, count: int) -> object:
    required = requirement.get("hard_100ms_required_rate", "")
    if required == "":
        return ""
    if int(count) <= 0:
        return False
    return bool(float(hard_rate) >= float(required))


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


def _launch_speed_bin_fields(initial_launch_speed_m_s: object) -> dict[str, object]:
    speed = _float_value(initial_launch_speed_m_s, default=float("nan"))
    if not math.isfinite(speed):
        return {
            "launch_speed_bin_id": "unknown_initial_launch_speed",
            "launch_speed_bin_min_m_s": "",
            "launch_speed_bin_max_m_s": "",
            "launch_speed_bin_label": "unknown initial_launch_speed_m_s",
            "start_energy_group_id": "unknown_start_energy",
            "start_energy_group_label": "unknown initial launch speed",
            "start_energy_feasibility_threshold_m_s": START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S,
            "start_energy_group_basis": "initial_launch_speed_m_s",
        }
    for bin_id, lower, upper, label in LAUNCH_SPEED_BIN_DEFINITIONS:
        lower_passed = lower is None or speed >= float(lower)
        upper_passed = upper is None or speed < float(upper)
        if lower_passed and upper_passed:
            speed_bin_id = bin_id
            speed_bin_min = "" if lower is None else float(lower)
            speed_bin_max = "" if upper is None else float(upper)
            speed_bin_label = label
            break
    else:
        speed_bin_id = "unknown_initial_launch_speed"
        speed_bin_min = ""
        speed_bin_max = ""
        speed_bin_label = "unknown initial_launch_speed_m_s"
    if speed < START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S:
        energy_group_id = LOW_START_ENERGY_GROUP_ID
        energy_group_label = f"initial_launch_speed_m_s < {START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S:.1f}"
    else:
        energy_group_id = HIGH_START_ENERGY_GROUP_ID
        energy_group_label = f"initial_launch_speed_m_s >= {START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S:.1f}"
    return {
        "launch_speed_bin_id": speed_bin_id,
        "launch_speed_bin_min_m_s": speed_bin_min,
        "launch_speed_bin_max_m_s": speed_bin_max,
        "launch_speed_bin_label": speed_bin_label,
        "start_energy_group_id": energy_group_id,
        "start_energy_group_label": energy_group_label,
        "start_energy_feasibility_threshold_m_s": START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S,
        "start_energy_group_basis": "initial_launch_speed_m_s",
    }


def _with_launch_speed_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "initial_launch_speed_m_s" not in out.columns:
        out["initial_launch_speed_m_s"] = float("nan")
    bin_rows = [_launch_speed_bin_fields(value) for value in out["initial_launch_speed_m_s"].tolist()]
    bins = pd.DataFrame(bin_rows, index=out.index)
    for column in bins.columns:
        out[column] = bins[column]
    return out


def _launch_condition_success_summary(final: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    frame = _with_launch_speed_analysis_columns(final)
    for column in group_columns:
        if column not in frame.columns:
            frame[column] = "not_applicable"
    bool_columns = [
        "mission_success",
        "safe_success",
        "full_safe_success",
        "front_wall_terminal_success",
        "wrong_wall_exit",
        "expected_low_energy_dry_air_sink",
        "claim_bearing_episode",
        "no_viable_primitive",
        "hard_failure",
        "floor_or_ceiling_violation",
        "physical_hard_failure",
        "physical_floor_or_ceiling_violation",
        "terminal_useful",
        "lift_capture",
        "memory_changed_selection",
        "baseline_vs_policy_selection_changed",
        "exploration_changed_selection",
    ]
    for column in bool_columns:
        if column in frame.columns:
            frame[column] = frame[column].map(_truthy).astype(float)
    numeric_columns = [
        "initial_launch_speed_m_s",
        "launch_score",
        "episode_flight_time_s",
        "lift_dwell_time_s",
        "terminal_specific_energy_m",
        "terminal_specific_energy_reserve_m",
        "terminal_specific_energy_bonus",
        "positive_specific_energy_gain_m",
        "updraft_specific_energy_gain_proxy_m",
        "net_specific_energy_delta_m",
        "gross_specific_energy_loss_m",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    agg: dict[str, tuple[str, str]] = {"launch_count": ("episode_id", "count")}
    for column in bool_columns:
        if column in frame.columns:
            agg[f"{column}_count"] = (column, "sum")
            agg[f"{column}_rate"] = (column, "mean")
    if "initial_launch_speed_m_s" in frame.columns:
        agg["mean_initial_launch_speed_m_s"] = ("initial_launch_speed_m_s", "mean")
        agg["min_initial_launch_speed_m_s"] = ("initial_launch_speed_m_s", "min")
        agg["max_initial_launch_speed_m_s"] = ("initial_launch_speed_m_s", "max")
    if "launch_score" in frame.columns:
        agg["mean_launch_score"] = ("launch_score", "mean")
        agg["median_launch_score"] = ("launch_score", "median")
    for column in numeric_columns:
        if column in frame.columns and column not in {"initial_launch_speed_m_s", "launch_score"}:
            agg[f"mean_{column}"] = (column, "mean")
    return frame.groupby(group_columns, dropna=False).agg(**agg).reset_index()


def _paired_score_delta_group_summary(delta_rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if delta_rows.empty:
        return pd.DataFrame()
    frame = delta_rows.copy()
    for column in group_columns:
        if column not in frame.columns:
            frame[column] = "not_applicable"
    frame["win"] = pd.to_numeric(frame["paired_delta_launch_score"], errors="coerce").fillna(0.0) > 0.0
    frame["loss"] = pd.to_numeric(frame["paired_delta_launch_score"], errors="coerce").fillna(0.0) < 0.0
    for column in ("memory_changed_selection", "baseline_vs_policy_selection_changed", "exploration_changed_selection", "safety_regression"):
        if column in frame.columns:
            frame[column] = frame[column].map(_truthy).astype(float)
    numeric_columns = [
        "paired_delta_launch_score",
        "mission_success_delta",
        "front_wall_terminal_success_delta",
        "wrong_wall_exit_delta",
        "safe_success_delta",
        "hard_failure_delta",
        "floor_or_ceiling_violation_delta",
        "no_viable_primitive_delta",
        "net_specific_energy_delta_m_delta",
        "positive_specific_energy_gain_m_delta",
        "updraft_specific_energy_gain_proxy_m_delta",
        "terminal_specific_energy_m_delta",
        "terminal_specific_energy_bonus_delta",
        "gross_specific_energy_loss_m_delta",
        "episode_flight_time_s_delta",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.groupby(group_columns, dropna=False)
        .agg(
            paired_launch_count=("paired_delta_launch_score", "count"),
            mean_paired_delta_launch_score=("paired_delta_launch_score", "mean"),
            median_paired_delta_launch_score=("paired_delta_launch_score", "median"),
            win_rate=("win", "mean"),
            loss_rate=("loss", "mean"),
            safety_regression_rate=("safety_regression", "mean"),
            mission_success_delta_mean=("mission_success_delta", "mean"),
            front_wall_terminal_success_delta_mean=("front_wall_terminal_success_delta", "mean"),
            wrong_wall_exit_delta_mean=("wrong_wall_exit_delta", "mean"),
            safe_success_delta_mean=("safe_success_delta", "mean"),
            hard_failure_delta_mean=("hard_failure_delta", "mean"),
            floor_or_ceiling_violation_delta_mean=("floor_or_ceiling_violation_delta", "mean"),
            no_viable_primitive_delta_mean=("no_viable_primitive_delta", "mean"),
            memory_changed_selection_rate=("memory_changed_selection", "mean"),
            baseline_vs_policy_selection_changed_rate=("baseline_vs_policy_selection_changed", "mean"),
            exploration_changed_selection_rate=("exploration_changed_selection", "mean"),
            mean_net_specific_energy_delta_m_delta=("net_specific_energy_delta_m_delta", "mean"),
            mean_positive_specific_energy_gain_m_delta=("positive_specific_energy_gain_m_delta", "mean"),
            mean_updraft_specific_energy_gain_proxy_m_delta=("updraft_specific_energy_gain_proxy_m_delta", "mean"),
            mean_terminal_specific_energy_m_delta=("terminal_specific_energy_m_delta", "mean"),
            mean_terminal_specific_energy_bonus_delta=("terminal_specific_energy_bonus_delta", "mean"),
            mean_gross_specific_energy_loss_m_delta=("gross_specific_energy_loss_m_delta", "mean"),
            mean_episode_flight_time_s_delta=("episode_flight_time_s_delta", "mean"),
        )
        .reset_index()
    )


def _write_compact_metric_tables(run_root: Path, episode_rows: list[dict[str, object]], protocol: ValidationProtocol) -> None:
    frame = pd.DataFrame(episode_rows)
    final = frame[frame["launch_role"].astype(str) == "final_heldout"] if not frame.empty else pd.DataFrame()
    if not final.empty:
        final = _with_launch_score_columns(final)
        final = _with_selection_change_flags(final)
        final = _with_launch_speed_analysis_columns(final)
    _write_csv(run_root / "metrics" / "final_launch_score.csv", final)
    speed_group_columns = [
        "environment_block_id",
        "library_size_case_id",
        "policy_id",
        "history_length",
        "launch_speed_bin_id",
        "launch_speed_bin_min_m_s",
        "launch_speed_bin_max_m_s",
        "launch_speed_bin_label",
    ]
    energy_group_columns = [
        "environment_block_id",
        "library_size_case_id",
        "policy_id",
        "history_length",
        "start_energy_group_id",
        "start_energy_group_label",
        "start_energy_group_basis",
        "start_energy_feasibility_threshold_m_s",
    ]
    _write_csv(
        run_root / "metrics" / "speed_bin_policy_ladder_summary.csv",
        _launch_condition_success_summary(final, speed_group_columns),
    )
    _write_csv(
        run_root / "metrics" / "start_energy_group_policy_ladder_summary.csv",
        _launch_condition_success_summary(final, energy_group_columns),
    )
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
                physical_hard_failure_rate=("physical_hard_failure", "mean"),
                floor_or_ceiling_violation_rate=("floor_or_ceiling_violation", "mean"),
                physical_floor_or_ceiling_violation_rate=("physical_floor_or_ceiling_violation", "mean"),
                expected_low_energy_dry_air_sink_rate=("expected_low_energy_dry_air_sink", "mean"),
                claim_bearing_episode_rate=("claim_bearing_episode", "mean"),
                no_viable_primitive_rate=("no_viable_primitive", "mean"),
                mission_success_rate=("mission_success", "mean"),
                front_wall_terminal_success_rate=("front_wall_terminal_success", "mean"),
                wrong_wall_exit_rate=("wrong_wall_exit", "mean"),
                terminal_useful_rate=("terminal_useful", "mean"),
                lift_capture_rate=("lift_capture", "mean"),
                mean_lift_dwell_time_s=("lift_dwell_time_s", "mean"),
                mean_energy_residual_m=("energy_residual_m", "mean"),
                mean_net_specific_energy_delta_m=("net_specific_energy_delta_m", "mean"),
                mean_positive_specific_energy_gain_m=("positive_specific_energy_gain_m", "mean"),
                mean_updraft_specific_energy_gain_proxy_m=("updraft_specific_energy_gain_proxy_m", "mean"),
                mean_gross_specific_energy_loss_m=("gross_specific_energy_loss_m", "mean"),
                mean_terminal_specific_energy_m=("terminal_specific_energy_m", "mean"),
                mean_terminal_specific_energy_reserve_m=("terminal_specific_energy_reserve_m", "mean"),
                mean_terminal_specific_energy_bonus=("terminal_specific_energy_bonus", "mean"),
                mean_episode_flight_time_s=("episode_flight_time_s", "mean"),
                mean_initial_launch_speed_m_s=("initial_launch_speed_m_s", "mean"),
                mean_launch_score=("launch_score", "mean"),
                median_launch_score=("launch_score", "median"),
                mean_min_wall_margin_m=("min_wall_margin_m", "mean"),
                selected_primitive_family_count=("selected_primitive_id", pd.Series.nunique),
                selected_variant_count=("selected_primitive_variant_id", pd.Series.nunique),
                governor_rejection_count=("governor_rejection_count", "sum"),
                belief_observation_count=("belief_observation_count", "max"),
                belief_uncertainty=("belief_uncertainty", "mean"),
                memory_changed_selection_rate=("memory_changed_selection", "mean"),
                baseline_vs_policy_selection_changed_rate=("baseline_vs_policy_selection_changed", "mean"),
                exploration_changed_selection_rate=("exploration_changed_selection", "mean"),
            )
            .reset_index()
        )
    _write_csv(run_root / "metrics" / "policy_history_comparison.csv", comparison)
    memory_delta = _paired_score_delta_rows(final, baseline_policy_id="no_memory_baseline")
    safe_explore_delta = _paired_safe_explore_delta_rows(final)
    _write_csv(run_root / "metrics" / "paired_memory_score_delta.csv", memory_delta)
    _write_csv(run_root / "metrics" / "paired_safe_explore_score_delta.csv", safe_explore_delta)
    paired_delta = pd.concat([memory_delta, safe_explore_delta], ignore_index=True)
    _write_csv(
        run_root / "metrics" / "paired_score_delta_summary.csv",
        _paired_score_delta_summary(paired_delta),
    )
    _write_csv(
        run_root / "metrics" / "paired_score_delta_by_speed_bin_summary.csv",
        _paired_score_delta_group_summary(
            paired_delta,
            [
                "comparison_type",
                "environment_block_id",
                "library_size_case_id",
                "policy_id",
                "baseline_policy_id",
                "history_length",
                "launch_speed_bin_id",
                "launch_speed_bin_min_m_s",
                "launch_speed_bin_max_m_s",
                "launch_speed_bin_label",
            ],
        ),
    )
    _write_csv(
        run_root / "metrics" / "paired_score_delta_by_start_energy_group_summary.csv",
        _paired_score_delta_group_summary(
            paired_delta,
            [
                "comparison_type",
                "environment_block_id",
                "library_size_case_id",
                "policy_id",
                "baseline_policy_id",
                "history_length",
                "start_energy_group_id",
                "start_energy_group_label",
                "start_energy_group_basis",
                "start_energy_feasibility_threshold_m_s",
            ],
        ),
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
                physical_hard_failure_rate=("physical_hard_failure", "mean"),
                expected_low_energy_dry_air_sink_rate=("expected_low_energy_dry_air_sink", "mean"),
                no_viable_primitive_rate=("no_viable_primitive", "mean"),
                mission_success_rate=("mission_success", "mean"),
                front_wall_terminal_success_rate=("front_wall_terminal_success", "mean"),
                wrong_wall_exit_rate=("wrong_wall_exit", "mean"),
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
                claim_bearing_episode_rate=("claim_bearing_episode", "mean"),
                expected_low_energy_dry_air_sink_rate=("expected_low_energy_dry_air_sink", "mean"),
                mission_success_rate=("mission_success", "mean"),
                front_wall_terminal_success_rate=("front_wall_terminal_success", "mean"),
                wrong_wall_exit_rate=("wrong_wall_exit", "mean"),
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
    interpretation = (
        frame.groupby(["launch_role", "episode_interpretation_label"], dropna=False)
        .size()
        .reset_index(name="row_count")
        if not frame.empty and "episode_interpretation_label" in frame.columns
        else pd.DataFrame()
    )
    _write_csv(run_root / "metrics" / "episode_interpretation_summary.csv", interpretation)
    for required_name in TABLE_NAMES:
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
    fields = _launch_speed_bin_fields(row.get("initial_launch_speed_m_s", baseline_row.get("initial_launch_speed_m_s", "")))
    return {
        "comparison_type": str(comparison_type),
        "library_size_case_id": str(row.get("library_size_case_id", "")),
        "environment_block_id": str(row.get("environment_block_id", "")),
        "outer_case_index": int(_float_value(row.get("outer_case_index", 0))),
        "common_final_launch_key": str(row.get("common_final_launch_key", "")),
        "policy_id": str(row.get("policy_id", "")),
        "baseline_policy_id": str(baseline_policy_id),
        "history_length": int(_float_value(row.get("history_length", 0))),
        "initial_launch_speed_m_s": _float_value(row.get("initial_launch_speed_m_s", baseline_row.get("initial_launch_speed_m_s", 0.0))),
        "baseline_initial_launch_speed_m_s": _float_value(baseline_row.get("initial_launch_speed_m_s", row.get("initial_launch_speed_m_s", 0.0))),
        "launch_speed_bin_id": fields["launch_speed_bin_id"],
        "launch_speed_bin_min_m_s": fields["launch_speed_bin_min_m_s"],
        "launch_speed_bin_max_m_s": fields["launch_speed_bin_max_m_s"],
        "launch_speed_bin_label": fields["launch_speed_bin_label"],
        "start_energy_group_id": fields["start_energy_group_id"],
        "start_energy_group_label": fields["start_energy_group_label"],
        "start_energy_group_basis": fields["start_energy_group_basis"],
        "start_energy_feasibility_threshold_m_s": fields["start_energy_feasibility_threshold_m_s"],
        "launch_score": _float_value(row.get("launch_score", 0.0)),
        "baseline_launch_score": _float_value(baseline_row.get("launch_score", 0.0)),
        "paired_delta_launch_score": _float_value(row.get("launch_score", 0.0)) - _float_value(baseline_row.get("launch_score", 0.0)),
        "mission_success_delta": int(_truthy(row.get("mission_success", False))) - int(_truthy(baseline_row.get("mission_success", False))),
        "front_wall_terminal_success_delta": int(_truthy(row.get("front_wall_terminal_success", False)))
        - int(_truthy(baseline_row.get("front_wall_terminal_success", False))),
        "wrong_wall_exit_delta": int(_truthy(row.get("wrong_wall_exit", False))) - int(_truthy(baseline_row.get("wrong_wall_exit", False))),
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
        "terminal_specific_energy_m_delta": _float_value(row.get("terminal_specific_energy_m", 0.0))
        - _float_value(baseline_row.get("terminal_specific_energy_m", 0.0)),
        "terminal_specific_energy_bonus_delta": _float_value(row.get("terminal_specific_energy_bonus", 0.0))
        - _float_value(baseline_row.get("terminal_specific_energy_bonus", 0.0)),
        "gross_specific_energy_loss_m_delta": _float_value(row.get("gross_specific_energy_loss_m", 0.0))
        - _float_value(baseline_row.get("gross_specific_energy_loss_m", 0.0)),
        "episode_flight_time_s_delta": _float_value(row.get("episode_flight_time_s", 0.0))
        - _float_value(baseline_row.get("episode_flight_time_s", 0.0)),
        "memory_changed_selection": bool(row.get("memory_changed_selection", False)),
        "baseline_vs_policy_selection_changed": bool(row.get("baseline_vs_policy_selection_changed", False)),
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
            mission_success_delta_mean=("mission_success_delta", "mean"),
            front_wall_terminal_success_delta_mean=("front_wall_terminal_success_delta", "mean"),
            wrong_wall_exit_delta_mean=("wrong_wall_exit_delta", "mean"),
            memory_changed_selection_rate=("memory_changed_selection", "mean"),
            baseline_vs_policy_selection_changed_rate=("baseline_vs_policy_selection_changed", "mean"),
            exploration_changed_selection_rate=("exploration_changed_selection", "mean"),
            mean_net_specific_energy_delta_m_delta=("net_specific_energy_delta_m_delta", "mean"),
            mean_positive_specific_energy_gain_m_delta=("positive_specific_energy_gain_m_delta", "mean"),
            mean_updraft_specific_energy_gain_proxy_m_delta=("updraft_specific_energy_gain_proxy_m_delta", "mean"),
            mean_terminal_specific_energy_m_delta=("terminal_specific_energy_m_delta", "mean"),
            mean_terminal_specific_energy_bonus_delta=("terminal_specific_energy_bonus_delta", "mean"),
            mean_gross_specific_energy_loss_m_delta=("gross_specific_energy_loss_m_delta", "mean"),
            mean_episode_flight_time_s_delta=("episode_flight_time_s_delta", "mean"),
        )
        .reset_index()
    )


def _with_selection_change_flags(final: pd.DataFrame) -> pd.DataFrame:
    out = final.copy()
    out["selection_signature"] = out["selected_primitive_variant_id"].fillna("").astype(str)
    if "memory_changed_selection" not in out.columns:
        out["memory_changed_selection"] = False
    out["memory_changed_selection"] = out["memory_changed_selection"].map(_truthy).astype(bool)
    if "memory_changed_selection_source" not in out.columns:
        out["memory_changed_selection_source"] = "missing_selector_memory_shield_fallback_false"
    out["baseline_vs_policy_selection_changed"] = False
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
            out.at[index, "baseline_vs_policy_selection_changed"] = signature != baseline_map.get(key, signature)
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
        _gate_row("policy_history_condition_count", set(row["policy_id"] for row in final_schedule) == set(protocol.policy_history_conditions), len(set(row["policy_id"] for row in final_schedule)), len(protocol.policy_history_conditions)),
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
        claim_rows = [row for row in final_rows if _truthy(row.get("claim_bearing_episode", True))]
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
                    "claim_bearing_final_launch_count_positive",
                    len(claim_rows) > 0,
                    len(claim_rows),
                    ">0_after_expected_low_energy_dry_air_sink_exclusion",
                ),
                _gate_row(
                    "expected_low_energy_dry_air_sink_labeled",
                    True,
                    sum(1 for row in final_rows if _truthy(row.get("expected_low_energy_dry_air_sink", False))),
                    "diagnostic_not_claim_bearing_governor_failure",
                ),
                _gate_row(
                    "hard_failure_rate_within_stage_profile",
                    _mean_bool(claim_rows, "hard_failure") <= float(protocol.max_hard_failure_rate),
                    _mean_bool(claim_rows, "hard_failure"),
                    protocol.max_hard_failure_rate,
                ),
                _gate_row(
                    "floor_or_ceiling_violation_rate_within_stage_profile",
                    _mean_bool(claim_rows, "floor_or_ceiling_violation")
                    <= float(protocol.max_floor_or_ceiling_violation_rate),
                    _mean_bool(claim_rows, "floor_or_ceiling_violation"),
                    protocol.max_floor_or_ceiling_violation_rate,
                ),
                _gate_row(
                    "no_viable_primitive_rate_within_stage_profile",
                    _mean_bool(claim_rows, "no_viable_primitive") <= float(protocol.max_no_viable_rate),
                    _mean_bool(claim_rows, "no_viable_primitive"),
                    protocol.max_no_viable_rate,
                ),
                _gate_row(
                    "safe_success_rate_within_stage_profile",
                    _mean_bool(claim_rows, "safe_success") >= float(protocol.min_safe_success_rate),
                    _mean_bool(claim_rows, "safe_success"),
                    protocol.min_safe_success_rate,
                ),
                _gate_row(
                    "terminal_or_lift_capture_within_stage_profile",
                    max(_mean_bool(claim_rows, "terminal_useful"), _mean_bool(claim_rows, "lift_capture"))
                    >= float(protocol.min_terminal_or_lift_capture_rate),
                    max(_mean_bool(claim_rows, "terminal_useful"), _mean_bool(claim_rows, "lift_capture")),
                    protocol.min_terminal_or_lift_capture_rate,
                ),
                _gate_row(
                    "front_wall_mission_success_rate_diagnostic",
                    True,
                    _mean_bool(claim_rows, "mission_success"),
                    "diagnostic_only_score_target_not_current_pass_gate",
                ),
                _gate_row(
                    "wrong_wall_exit_rate_diagnostic",
                    True,
                    _mean_bool(claim_rows, "wrong_wall_exit"),
                    "diagnostic_only_penalised_by_launch_score",
                ),
                _gate_row(
                    "selected_primitive_family_count_diagnostic",
                    True,
                    len(selected_primitives),
                    "diagnostic_only_not_a_governor_pass_gate",
                ),
                _gate_row(
                    "selected_variant_count_diagnostic",
                    True,
                    len(selected_variants),
                    "diagnostic_only_not_a_governor_pass_gate",
                ),
                _gate_row(
                    "lift_dwell_arc_selected_diagnostic",
                    True,
                    "lift_dwell_arc" in selected_primitives,
                    "diagnostic_only_expected_when_viable_lift_dwell_evidence_wins",
                ),
            ]
        )
        if protocol.stage_id == "R10":
            rows.extend(
                [
                    _gate_row(
                        "r10_learning_stage_uses_final_reject_rate_not_candidate_reject_rate",
                        True,
                        _mean_bool(claim_rows, "no_viable_primitive"),
                        "bounded_final_no_viable_rate_plus_tuning_handoff_diagnostics",
                    ),
                    _gate_row(
                        "r10_memory_improvement_is_tuning_signal_not_final_claim_gate",
                        True,
                        "governor_config_selection.csv_and_memory_opportunity_summary.csv",
                        "R11_or_reality_required_for_memory_improvement_claim",
                    ),
                ]
            )
        if protocol.min_full_safe_success_rate is not None:
            rows.append(
                _gate_row(
                    "full_safe_success_rate_within_stage_profile",
                    _mean_bool(claim_rows, "full_safe_success") >= float(protocol.min_full_safe_success_rate),
                    _mean_bool(claim_rows, "full_safe_success"),
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
                    and (
                        "plant_implementation_seed" not in group.columns
                        or group["plant_implementation_seed"].nunique() == 1
                    )
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
        environment_block_id = str(group["environment_block_id"].iloc[0]) if "environment_block_id" in group.columns and not group.empty else ""
        rows.append(
            {
                "outer_case_index": int(key),
                "environment_block_id": environment_block_id,
                "variation_audit_passed": True,
                "glider_model_fixed": True,
                "latency_model_fixed": True,
                "actuator_model_fixed": True,
                "mass_cg_inertia_surface_calibration_not_varied": True,
                "full_w3_randomisation_exception": False,
                "fan_layout_fixed_within_outer_case": True,
                "active_fan_count_fixed_within_outer_case": True,
                "updraft_parameter_noise_allowed": _block_varies_environment_parameters_between_history_episodes(
                    environment_block_id
                ),
                "variation_policy": "fixed_layout_count_and_plant_per_outer_case_parameter_noise_only",
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
                "policy": "balanced_0_1_2_3_4_for_active_fan_number_variation",
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
            else "balanced_0_1_2_3_4"
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
    selector_rows: list[dict[str, object]] | None = None,
) -> None:
    final = pd.DataFrame([row for row in episode_rows if str(row.get("launch_role", "")) == "final_heldout"])
    final_episode_ids = {
        str(row.get("episode_id", ""))
        for row in final.to_dict(orient="records")
        if str(row.get("episode_id", "")).strip()
    }
    metrics = {
        "final_launch_count": int(len(final)),
        "hard_failure_rate": _mean_bool(final.to_dict(orient="records"), "hard_failure") if not final.empty else 1.0,
        "no_viable_primitive_rate": _mean_bool(final.to_dict(orient="records"), "no_viable_primitive") if not final.empty else 1.0,
        "final_reject_rate": _mean_bool(final.to_dict(orient="records"), "no_viable_primitive") if not final.empty else 1.0,
        "safe_success_rate": _mean_bool(final.to_dict(orient="records"), "safe_success") if not final.empty else 0.0,
        "full_safe_success_rate": _mean_bool(final.to_dict(orient="records"), "full_safe_success") if not final.empty else 0.0,
        "mission_success_rate": _mean_bool(final.to_dict(orient="records"), "mission_success") if not final.empty else 0.0,
        "wrong_wall_exit_rate": _mean_bool(final.to_dict(orient="records"), "wrong_wall_exit") if not final.empty else 0.0,
        "terminal_or_lift_capture_rate": max(
            _mean_bool(final.to_dict(orient="records"), "terminal_useful") if not final.empty else 0.0,
            _mean_bool(final.to_dict(orient="records"), "lift_capture") if not final.empty else 0.0,
        ),
    }
    metrics.update(
        _governor_tuning_memory_metrics(
            selector_rows or [],
            final_episode_ids=final_episode_ids,
        )
    )
    metrics.update(
        {
            "governor_learning_strategy_version": OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION,
            "online_memory_scope": ONLINE_MEMORY_SCOPE,
            "global_calibration_scope": (
                R10_GLOBAL_CALIBRATION_SCOPE
                if protocol.stage_id == "R10"
                else "r9_reduced_preflight_initialisation_only"
            ),
            "cross_case_memory_carryover_allowed": False,
            "governor_calibration_search_policy": GOVERNOR_CALIBRATION_SEARCH_POLICY,
            "r11_handoff_scope": R11_GOVERNOR_HANDOFF_SCOPE if protocol.stage_id == "R10" else "",
            "r11_handoff_config_count": 1 if protocol.stage_id == "R10" else 0,
        }
    )
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
        "governor_learning_strategy_version": OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION,
        "online_memory_scope": ONLINE_MEMORY_SCOPE,
        "global_calibration_scope": metrics["global_calibration_scope"],
        "cross_case_memory_carryover_allowed": False,
        "governor_calibration_search_policy": GOVERNOR_CALIBRATION_SEARCH_POLICY,
        "aggregate_evidence_scope": metrics["global_calibration_scope"],
        "r10_aggregate_evidence_scope": R10_GLOBAL_CALIBRATION_SCOPE if protocol.stage_id == "R10" else "",
        "r11_handoff_scope": R11_GOVERNOR_HANDOFF_SCOPE if protocol.stage_id == "R10" else "",
        "r11_handoff_config_count": metrics["r11_handoff_config_count"],
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


def _governor_tuning_memory_metrics(
    selector_rows: list[dict[str, object]],
    *,
    final_episode_ids: set[str],
) -> dict[str, object]:
    if not selector_rows:
        return {
            "selector_decision_count": 0,
            "memory_switch_opportunity_count": 0,
            "accepted_memory_switch_count": 0,
            "rejected_memory_switch_count": 0,
            "memory_switch_acceptance_rate": 0.0,
            "mean_memory_correction_delta": 0.0,
            "max_memory_correction_delta": 0.0,
            "mean_base_score_gap_to_baseline": 0.0,
            "mean_memory_opportunity_ratio": 0.0,
            "mean_candidate_path_confidence": 0.0,
            "mean_candidate_path_uncertainty": 0.0,
            "mean_adaptive_score_margin": 0.0,
            "mean_exploration_score_component": 0.0,
        }
    frame = pd.DataFrame(selector_rows)
    if final_episode_ids and "episode_id" in frame.columns:
        frame = frame[frame["episode_id"].astype(str).isin(final_episode_ids)].copy()
    if frame.empty:
        return _governor_tuning_memory_metrics([], final_episode_ids=set())
    policy = frame.get("policy_id", pd.Series("", index=frame.index)).astype(str)
    baseline_variant = frame.get("selected_memory_shield_baseline_variant_id", pd.Series("", index=frame.index)).astype(str)
    memory_variant = frame.get("selected_memory_shield_memory_variant_id", pd.Series("", index=frame.index)).astype(str)
    opportunity = (
        policy.str.contains(MEMORY_POLICY_PREFIX, regex=False)
        & baseline_variant.ne("")
        & memory_variant.ne("")
        & baseline_variant.ne(memory_variant)
    )
    status = frame.get("selected_memory_shield_status", pd.Series("", index=frame.index)).astype(str)
    accepted = opportunity & status.str.startswith("accepted")
    rejected = opportunity & status.str.startswith("rejected")
    opportunity_frame = frame[opportunity].copy()

    def mean_column(column: str) -> float:
        if opportunity_frame.empty or column not in opportunity_frame.columns:
            return 0.0
        values = pd.to_numeric(opportunity_frame[column], errors="coerce").fillna(0.0)
        return float(values.mean())

    def max_column(column: str) -> float:
        if opportunity_frame.empty or column not in opportunity_frame.columns:
            return 0.0
        values = pd.to_numeric(opportunity_frame[column], errors="coerce").fillna(0.0)
        return float(values.max())

    opportunity_count = int(opportunity.sum())
    accepted_count = int(accepted.sum())
    return {
        "selector_decision_count": int(len(frame)),
        "memory_switch_opportunity_count": opportunity_count,
        "accepted_memory_switch_count": accepted_count,
        "rejected_memory_switch_count": int(rejected.sum()),
        "memory_switch_acceptance_rate": float(accepted_count / max(1, opportunity_count)),
        "mean_memory_correction_delta": mean_column("selected_memory_shield_memory_correction_delta"),
        "max_memory_correction_delta": max_column("selected_memory_shield_memory_correction_delta"),
        "mean_base_score_gap_to_baseline": mean_column("selected_memory_shield_base_score_gap_to_baseline"),
        "mean_memory_opportunity_ratio": mean_column("selected_memory_shield_memory_opportunity_ratio"),
        "mean_candidate_path_confidence": mean_column("selected_memory_shield_candidate_path_confidence"),
        "mean_candidate_path_uncertainty": mean_column("selected_memory_shield_candidate_path_uncertainty"),
        "mean_adaptive_score_margin": mean_column("selected_memory_shield_score_margin"),
        "mean_exploration_score_component": mean_column("selected_memory_shield_exploration_score_component"),
    }


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
    mission_success_rate = _float_metric(metrics.get("mission_success_rate", 0.0), default=0.0)
    wrong_wall_exit_rate = _float_metric(metrics.get("wrong_wall_exit_rate", 0.0), default=0.0)
    terminal_or_lift_rate = _float_metric(metrics.get("terminal_or_lift_capture_rate", 0.0), default=0.0)
    memory_opportunity_count = int(_float_metric(metrics.get("memory_switch_opportunity_count", 0), default=0.0))
    memory_acceptance_rate = _float_metric(metrics.get("memory_switch_acceptance_rate", 0.0), default=0.0)
    mean_candidate_path_confidence = _float_metric(metrics.get("mean_candidate_path_confidence", 0.0), default=0.0)
    max_memory_correction_delta = _float_metric(metrics.get("max_memory_correction_delta", 0.0), default=0.0)

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
        update(
            "memory_switch_min_confidence",
            min(0.55, float(values["memory_switch_min_confidence"]) + 0.05),
            "hard_failure_rate_above_stage_profile_require_stronger_memory_confidence",
        )
        update(
            "memory_switch_min_score_margin",
            min(0.025, float(values["memory_switch_min_score_margin"]) + 0.0025),
            "hard_failure_rate_above_stage_profile_require_stronger_memory_score_margin",
        )
        update(
            "memory_switch_max_base_score_drop",
            max(0.04, float(values["memory_switch_max_base_score_drop"]) * 0.75),
            "hard_failure_rate_above_stage_profile_tighten_baseline_non_regression",
        )
        update(
            "memory_objective_score_cap",
            max(0.08, float(values["memory_objective_score_cap"]) * 0.75),
            "hard_failure_rate_above_stage_profile_cap_aggressive_memory_objective",
        )
        update(
            "memory_objective_min_confidence",
            min(0.45, float(values["memory_objective_min_confidence"]) + 0.05),
            "hard_failure_rate_above_stage_profile_require_stronger_memory_objective_confidence",
        )
        update(
            "memory_objective_max_base_score_drop",
            max(0.06, float(values["memory_objective_max_base_score_drop"]) * 0.75),
            "hard_failure_rate_above_stage_profile_tighten_memory_objective_baseline_tradeoff",
        )
        update(
            "memory_switch_max_transition_success_drop",
            max(0.0, float(values["memory_switch_max_transition_success_drop"]) * 0.50),
            "hard_failure_rate_above_stage_profile_tighten_transition_non_regression",
        )
        update(
            "exploration_switch_min_uncertainty",
            min(0.80, float(values["exploration_switch_min_uncertainty"]) + 0.05),
            "hard_failure_rate_above_stage_profile_make_exploration_more_selective",
        )
        update(
            "exploration_switch_max_base_score_drop",
            max(0.0, float(values["exploration_switch_max_base_score_drop"]) * 0.50),
            "hard_failure_rate_above_stage_profile_reduce_exploration_base_score_drop",
        )
        update(
            "adaptive_switch_max_path_exit_margin_drop_m",
            max(0.02, float(values["adaptive_switch_max_path_exit_margin_drop_m"]) * 0.75),
            "hard_failure_rate_above_stage_profile_tighten_path_exit_margin_shield",
        )
        update(
            "candidate_path_memory_residual_cap_m",
            max(0.35, float(values["candidate_path_memory_residual_cap_m"]) * 0.85),
            "hard_failure_rate_above_stage_profile_cap_memory_residual_effect",
        )
        update(
            "candidate_path_memory_specific_energy_residual_cap_m",
            max(0.50, float(values["candidate_path_memory_specific_energy_residual_cap_m"]) * 0.85),
            "hard_failure_rate_above_stage_profile_cap_specific_energy_residual_effect",
        )
        update(
            "flow_region_attraction_weight",
            max(0.45, float(values["flow_region_attraction_weight"]) * 0.80),
            "hard_failure_rate_above_stage_profile_reduce_reachable_flow_attraction_weight",
        )
        update(
            "flow_region_attraction_score_cap",
            max(0.06, float(values["flow_region_attraction_score_cap"]) * 0.85),
            "hard_failure_rate_above_stage_profile_cap_reachable_flow_attraction_score",
        )
        update(
            "flow_region_attraction_min_confidence",
            min(0.65, float(values["flow_region_attraction_min_confidence"]) + 0.05),
            "hard_failure_rate_above_stage_profile_require_stronger_reachable_flow_confidence",
        )
        update(
            "flow_region_attraction_max_base_score_drop",
            max(0.06, float(values["flow_region_attraction_max_base_score_drop"]) * 0.75),
            "hard_failure_rate_above_stage_profile_tighten_reachable_flow_base_score_drop",
        )
        update(
            "memory_information_gain_weight",
            max(0.06, float(values["memory_information_gain_weight"]) * 0.75),
            "hard_failure_rate_above_stage_profile_reduce_information_gain_weight",
        )
        update(
            "memory_information_gain_score_cap",
            max(0.04, float(values["memory_information_gain_score_cap"]) * 0.75),
            "hard_failure_rate_above_stage_profile_cap_information_gain_score",
        )
        update(
            "memory_information_gain_max_base_score_drop",
            max(0.04, float(values["memory_information_gain_max_base_score_drop"]) * 0.75),
            "hard_failure_rate_above_stage_profile_tighten_information_gain_base_score_drop",
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
        new_updraft_memory_weight = min(0.40, float(values["candidate_path_memory_utility_updraft_weight"]) + 0.05)
        update(
            "candidate_path_memory_utility_updraft_weight",
            new_updraft_memory_weight,
            "terminal_or_lift_capture_below_stage_profile_make_memory_more_lift_aware",
        )
        update(
            "candidate_path_memory_utility_specific_energy_weight",
            max(0.60, 1.0 - new_updraft_memory_weight),
            "terminal_or_lift_capture_below_stage_profile_keep_memory_specific_energy_dominant",
        )

    if safe_success_rate < float(protocol.min_safe_success_rate) and hard_failure_rate <= float(protocol.max_hard_failure_rate):
        update(
            "mission_front_progress_weight",
            min(0.45, float(values["mission_front_progress_weight"]) + 0.04),
            "safe_success_below_stage_profile_increase_front_wall_progress_preference",
        )
        update(
            "mission_front_terminal_weight",
            min(0.90, float(values["mission_front_terminal_weight"]) + 0.05),
            "safe_success_below_stage_profile_increase_front_wall_terminal_preference",
        )
        update(
            "belief_weight",
            min(0.75, float(values["belief_weight"]) + 0.05),
            "safe_success_below_stage_profile_increase_memory_residual_sensitivity",
        )

    if mission_success_rate < float(protocol.min_safe_success_rate) and wrong_wall_exit_rate > 0.0:
        update(
            "mission_wrong_boundary_penalty_weight",
            min(0.60, float(values["mission_wrong_boundary_penalty_weight"]) + 0.05),
            "mission_success_low_with_wrong_wall_exits_increase_wrong_boundary_penalty",
        )

    if memory_opportunity_count > 0 and hard_failure_rate <= float(protocol.max_hard_failure_rate):
        if memory_acceptance_rate < 0.05:
            update(
                "memory_switch_min_confidence",
                max(0.08, float(values["memory_switch_min_confidence"]) - 0.05),
                "memory_opportunities_seen_but_switch_acceptance_low_relax_confidence_gate",
            )
            update(
                "memory_switch_min_score_margin",
                max(0.0, float(values["memory_switch_min_score_margin"]) * 0.50),
                "memory_opportunities_seen_but_switch_acceptance_low_relax_score_margin",
            )
            update(
                "memory_switch_max_base_score_drop",
                min(0.20, float(values["memory_switch_max_base_score_drop"]) + 0.02),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_small_baseline_tradeoff",
            )
            update(
                "memory_switch_max_transition_success_drop",
                min(0.04, float(values["memory_switch_max_transition_success_drop"]) + 0.005),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_small_transition_tradeoff",
            )
            update(
                "adaptive_switch_max_path_exit_margin_drop_m",
                min(0.08, float(values["adaptive_switch_max_path_exit_margin_drop_m"]) + 0.01),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_small_path_margin_tradeoff",
            )
            update(
                "flow_region_attraction_min_confidence",
                max(0.08, float(values["flow_region_attraction_min_confidence"]) - 0.05),
                "memory_opportunities_seen_but_switch_acceptance_low_relax_reachable_flow_confidence",
            )
            update(
                "flow_region_attraction_max_base_score_drop",
                min(0.25, float(values["flow_region_attraction_max_base_score_drop"]) + 0.025),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_reachable_flow_safe_region_bias",
            )
            update(
                "flow_region_attraction_weight",
                min(1.80, float(values["flow_region_attraction_weight"]) + 0.15),
                "memory_opportunities_seen_but_switch_acceptance_low_increase_reachable_flow_attraction_weight",
            )
            update(
                "memory_objective_min_confidence",
                max(0.08, float(values["memory_objective_min_confidence"]) - 0.05),
                "memory_opportunities_seen_but_switch_acceptance_low_relax_aggressive_memory_objective_confidence",
            )
            update(
                "memory_objective_max_base_score_drop",
                min(0.25, float(values["memory_objective_max_base_score_drop"]) + 0.025),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_aggressive_memory_safe_region_tradeoff",
            )
            update(
                "memory_information_gain_weight",
                min(0.35, float(values["memory_information_gain_weight"]) + 0.04),
                "memory_opportunities_seen_but_switch_acceptance_low_increase_information_gain_weight",
            )
            update(
                "memory_information_gain_score_cap",
                min(0.20, float(values["memory_information_gain_score_cap"]) + 0.025),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_information_gain_bias",
            )
            update(
                "memory_information_gain_max_base_score_drop",
                min(0.22, float(values["memory_information_gain_max_base_score_drop"]) + 0.02),
                "memory_opportunities_seen_but_switch_acceptance_low_allow_information_gain_safe_region_tradeoff",
            )
        if mean_candidate_path_confidence < float(values["memory_switch_min_confidence"]):
            update(
                "candidate_path_memory_full_confidence_observations",
                max(1.5, float(values["candidate_path_memory_full_confidence_observations"]) - 0.5),
                "memory_opportunities_low_confidence_make_case_local_evidence_reach_confidence_sooner",
            )
        if max_memory_correction_delta < 0.01 and safe_success_rate < float(protocol.min_safe_success_rate):
            update(
                "belief_weight",
                min(0.90, float(values["belief_weight"]) + 0.08),
                "memory_opportunities_small_correction_and_low_success_increase_residual_sensitivity",
            )
            update(
                "memory_objective_score_cap",
                min(0.30, float(values["memory_objective_score_cap"]) + 0.04),
                "memory_opportunities_small_correction_allow_larger_bounded_memory_objective",
            )
            update(
                "candidate_path_memory_residual_cap_m",
                min(1.00, float(values["candidate_path_memory_residual_cap_m"]) + 0.10),
                "memory_opportunities_small_correction_allow_larger_updraft_residual_when_safe",
            )
            update(
                "candidate_path_memory_specific_energy_residual_cap_m",
                min(1.40, float(values["candidate_path_memory_specific_energy_residual_cap_m"]) + 0.10),
                "memory_opportunities_small_correction_allow_larger_specific_energy_residual_when_safe",
            )
            update(
                "flow_region_attraction_score_cap",
                min(0.25, float(values["flow_region_attraction_score_cap"]) + 0.03),
                "memory_opportunities_small_correction_allow_larger_reachable_flow_attraction_when_safe",
            )
            update(
                "memory_information_gain_weight",
                min(0.40, float(values["memory_information_gain_weight"]) + 0.05),
                "memory_opportunities_small_correction_allow_larger_information_gain_when_safe",
            )
            update(
                "memory_information_gain_score_cap",
                min(0.24, float(values["memory_information_gain_score_cap"]) + 0.03),
                "memory_opportunities_small_correction_raise_information_gain_cap_when_safe",
            )
    elif memory_opportunity_count == 0 and hard_failure_rate <= float(protocol.max_hard_failure_rate) and safe_success_rate < float(protocol.min_safe_success_rate):
        update(
            "exploration_bonus_weight",
            min(0.04, float(values["exploration_bonus_weight"]) + 0.005),
            "no_memory_switch_opportunities_and_low_success_increase_same_family_uncertainty_exploration",
        )
        update(
            "exploration_switch_min_uncertainty",
            max(0.45, float(values["exploration_switch_min_uncertainty"]) - 0.05),
            "no_memory_switch_opportunities_and_low_success_relax_exploration_uncertainty_gate",
        )

    values["minimum_wall_margin_m"] = float(base_config.minimum_wall_margin_m)
    values["config_id"] = f"v53_{protocol.stage_id.lower()}_tuned_mission_governor"
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


def _frozen_controller_source_manifest_fields(source_record_load: FrozenControllerRecordLoadResult | None) -> dict[str, object]:
    if source_record_load is None:
        return {
            "frozen_controller_records_source_root": "",
            "frozen_controller_records_source_reason": "",
            "frozen_controller_records_loaded_count": 0,
            "frozen_controller_bundle_variant_count": 0,
            "frozen_controller_bundle_ready_count": 0,
            "frozen_controller_source_candidate_roots": [],
            "frozen_controller_source_candidate_reasons": [],
            "stale_source_path_relocation_policy": (
                "legacy lqr_contextual_v1_0 result roots are resolved to renamed 05_Results stage roots at read time; "
                "old result manifests are not rewritten"
            ),
            "stale_source_path_relocation_used": False,
        }
    return {
        "frozen_controller_records_source_root": ""
        if source_record_load.resolved_root is None
        else Path(source_record_load.resolved_root).as_posix(),
        "frozen_controller_records_source_reason": str(source_record_load.resolved_reason),
        "frozen_controller_records_loaded_count": int(source_record_load.record_count),
        "frozen_controller_bundle_variant_count": int(source_record_load.bundle_variant_count),
        "frozen_controller_bundle_ready_count": int(source_record_load.bundle_ready_count),
        "frozen_controller_source_candidate_roots": [Path(root).as_posix() for root in source_record_load.attempted_roots],
        "frozen_controller_source_candidate_reasons": list(source_record_load.attempted_reasons),
        "stale_source_path_relocation_policy": (
            "legacy lqr_contextual_v1_0 result roots are resolved to renamed 05_Results stage roots at read time; "
            "old result manifests are not rewritten"
        ),
        "stale_source_path_relocation_used": "relocated_result_root" in str(source_record_load.resolved_reason),
    }


def _write_manifest(
    *,
    run_root: Path,
    config: ValidationRunConfig,
    protocol: ValidationProtocol,
    status: str,
    pass_summary: list[dict[str, object]],
    final_schedule: list[dict[str, object]],
    history_schedule: list[dict[str, object]],
    source_record_load: FrozenControllerRecordLoadResult | None = None,
    duration_s: float = 0.0,
) -> None:
    governor_config = config.governor_config or DEFAULT_GOVERNOR_CONFIG
    payload = {
        "manifest_version": protocol.manifest_version,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_label": str(config.run_label).strip(),
        "run_root": run_root.as_posix(),
        "stage_id": protocol.stage_id,
        "library_root": Path(config.library_root).as_posix(),
        "outcome_root": Path(config.outcome_root).as_posix(),
        "source_w2_root": "" if config.source_w2_root is None else Path(config.source_w2_root).as_posix(),
        **_frozen_controller_source_manifest_fields(source_record_load),
        "history_lengths": sorted(
            {
                int(_policy_condition(policy_id)["history_length"])
                for policy_id in protocol.policy_history_conditions
                if int(_policy_condition(policy_id)["history_length"]) > 0
            }
        ),
        "policy_history_conditions": list(protocol.policy_history_conditions),
        "policy_history_condition_count": len(protocol.policy_history_conditions),
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
        "real_time_outer_loop_scheduler_version": REAL_TIME_OUTER_LOOP_SCHEDULER_VERSION,
        "real_time_outer_loop_scheduler_policy": "prepare_next_decision_before_boundary_with_full_memory_query_controller_row_no_table_flush",
        "real_time_preferred_decision_budget_s": float(REAL_TIME_PREFERRED_DECISION_BUDGET_S),
        "real_time_hard_decision_budget_s": float(REAL_TIME_HARD_DECISION_BUDGET_S),
        "real_time_scheduler_audit": "metrics/real_time_scheduler_audit.csv",
        "real_time_claim_status": "controller_compute_profile_excludes_table_flush_and_posthoc_diagnostics",
        "outer_loop_memory_policy_version": OUTER_LOOP_MEMORY_POLICY_VERSION,
        "outer_loop_memory_policy": (
            "case-local 0.1 m 3D spatial updraft-utility belief map with dense executed-primitive updates "
            "at 0.1 m spacing and launch-index recency decay; the online controller and full diagnostics query "
            "the accumulated map using the same 0.2 m neighbourhood over seven current-to-exit probes before "
            "the 0.100 s boundary through a compact controller-row selector fast path, while table flushing, "
            "full candidate-row expansion, and post-hoc diagnostics remain outside that boundary. "
            "Both use a bounded 0.8 m / "
            "35 deg azimuth / 20 deg elevation sparse 3D reachable-flow attraction cone capped at 0.25 m plus a bounded sparse "
            "short-horizon route-flow probe set from the candidate exit; those queries are collapsed into one "
            "cost-benefit memory value: remembered flow benefit plus small information value minus frozen "
            "mission-score, front-progress, risk, and path-margin costs; accepted only through unchanged "
            "viability filters and the baseline shield with no final-launch special case"
        ),
        "governor_learning_strategy_version": OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION,
        "online_memory_scope": ONLINE_MEMORY_SCOPE,
        "cross_case_memory_carryover_allowed": False,
        "global_governor_calibration_scope": (
            R10_GLOBAL_CALIBRATION_SCOPE
            if protocol.stage_id == "R10"
            else "r9_reduced_preflight_initialisation_only"
            if protocol.stage_id == "R9"
            else "not_applicable_validation_uses_frozen_governor"
        ),
        "governor_calibration_search_policy": GOVERNOR_CALIBRATION_SEARCH_POLICY,
        "r11_governor_handoff_scope": R11_GOVERNOR_HANDOFF_SCOPE if protocol.stage_id == "R10" else "",
        "candidate_path_memory_lookahead_s": float(CANDIDATE_PATH_MEMORY_LOOKAHEAD_S),
        "candidate_path_memory_residual_cap_m": float(governor_config.candidate_path_memory_residual_cap_m),
        "candidate_path_memory_specific_energy_residual_cap_m": float(
            governor_config.candidate_path_memory_specific_energy_residual_cap_m
        ),
        "candidate_path_memory_utility_specific_energy_weight": float(
            governor_config.candidate_path_memory_utility_specific_energy_weight
        ),
        "candidate_path_memory_utility_updraft_weight": float(
            governor_config.candidate_path_memory_utility_updraft_weight
        ),
        "candidate_path_memory_probe_count": int(len(CANDIDATE_PATH_MEMORY_PROBES)),
        "real_time_candidate_path_memory_probe_count": int(len(CANDIDATE_PATH_MEMORY_PROBES)),
        "candidate_path_memory_full_confidence_observations": float(
            governor_config.candidate_path_memory_full_confidence_observations
        ),
        "flow_belief_grid_resolution_m": float(FLOW_BELIEF_GRID_RESOLUTION_M),
        "flow_belief_query_radius_m": float(FLOW_BELIEF_QUERY_RADIUS_M),
        "flow_belief_real_time_query_radius_m": float(FLOW_BELIEF_QUERY_RADIUS_M),
        "flow_belief_history_update_spacing_m": float(FLOW_BELIEF_HISTORY_UPDATE_SPACING_M),
        "flow_belief_history_update_max_samples_per_primitive": int(
            FLOW_BELIEF_HISTORY_UPDATE_MAX_SAMPLES_PER_PRIMITIVE
        ),
        "flow_belief_history_update_policy": FLOW_BELIEF_HISTORY_UPDATE_POLICY,
        "flow_belief_history_update_execution_policy": "dense_observations_applied_in_one_batch_per_executed_primitive",
        "flow_belief_reachable_attraction_lookahead_m": float(FLOW_BELIEF_REACHABLE_ATTRACTION_LOOKAHEAD_M),
        "flow_belief_reachable_attraction_half_angle_rad": float(FLOW_BELIEF_REACHABLE_ATTRACTION_HALF_ANGLE_RAD),
        "flow_belief_reachable_attraction_azimuth_half_angle_rad": float(
            FLOW_BELIEF_REACHABLE_ATTRACTION_AZIMUTH_HALF_ANGLE_RAD
        ),
        "flow_belief_reachable_attraction_elevation_half_angle_rad": float(
            FLOW_BELIEF_REACHABLE_ATTRACTION_ELEVATION_HALF_ANGLE_RAD
        ),
        "flow_belief_reachable_attraction_geometry": "sparse_3d_cone_2_range_3_azimuth_3_elevation_stencil",
        "flow_belief_reachable_attraction_probe_count": int(len(FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES)),
        "flow_belief_real_time_reachable_attraction_probe_count": int(
            len(FLOW_BELIEF_REACHABLE_ATTRACTION_PROBES)
        ),
        "flow_belief_reachable_attraction_cap_m": float(FLOW_BELIEF_REACHABLE_ATTRACTION_CAP_M),
        "flow_belief_route_horizon_primitives": int(round(float(governor_config.memory_route_horizon_primitives))),
        "flow_belief_route_probe_count": int(len(FLOW_BELIEF_ROUTE_PROBE_FRACTIONS)),
        "flow_belief_real_time_route_probe_count": int(len(FLOW_BELIEF_ROUTE_PROBE_FRACTIONS)),
        "flow_belief_route_policy": "bounded_sparse_short_horizon_route_flow_probes_from_candidate_exit",
        "memory_route_planning_weight": float(governor_config.memory_route_planning_weight),
        "memory_route_information_gain_weight": float(governor_config.memory_route_information_gain_weight),
        "memory_route_score_cap": float(governor_config.memory_route_score_cap),
        "memory_route_min_confidence": float(governor_config.memory_route_min_confidence),
        "memory_route_max_base_score_drop": float(governor_config.memory_route_max_base_score_drop),
        "memory_route_min_front_progress_ratio": float(governor_config.memory_route_min_front_progress_ratio),
        "memory_route_discount": float(governor_config.memory_route_discount),
        "memory_cost_benefit_weight": float(governor_config.memory_cost_benefit_weight),
        "memory_cost_benefit_score_cap": float(governor_config.memory_cost_benefit_score_cap),
        "memory_cost_benefit_information_gain_weight": float(
            governor_config.memory_cost_benefit_information_gain_weight
        ),
        "memory_cost_benefit_progress_cost_weight": float(
            governor_config.memory_cost_benefit_progress_cost_weight
        ),
        "memory_cost_benefit_risk_cost_weight": float(governor_config.memory_cost_benefit_risk_cost_weight),
        "memory_cost_benefit_margin_cost_weight": float(governor_config.memory_cost_benefit_margin_cost_weight),
        "memory_objective_score_cap": float(governor_config.memory_objective_score_cap),
        "memory_objective_min_confidence": float(governor_config.memory_objective_min_confidence),
        "memory_objective_max_base_score_drop": float(governor_config.memory_objective_max_base_score_drop),
        "flow_region_attraction_weight": float(governor_config.flow_region_attraction_weight),
        "flow_region_attraction_score_cap": float(governor_config.flow_region_attraction_score_cap),
        "flow_region_attraction_min_confidence": float(governor_config.flow_region_attraction_min_confidence),
        "flow_region_attraction_max_base_score_drop": float(
            governor_config.flow_region_attraction_max_base_score_drop
        ),
        "flow_region_attraction_min_front_progress_ratio": float(
            governor_config.flow_region_attraction_min_front_progress_ratio
        ),
        "memory_information_gain_weight": float(governor_config.memory_information_gain_weight),
        "memory_information_gain_score_cap": float(governor_config.memory_information_gain_score_cap),
        "memory_information_gain_min_uncertainty": float(
            governor_config.memory_information_gain_min_uncertainty
        ),
        "memory_information_gain_max_base_score_drop": float(
            governor_config.memory_information_gain_max_base_score_drop
        ),
        "memory_information_gain_min_front_progress_ratio": float(
            governor_config.memory_information_gain_min_front_progress_ratio
        ),
        "memory_information_gain_allow_cross_family": bool(
            governor_config.memory_information_gain_allow_cross_family
        ),
        "flow_belief_update_policy": (
            FLOW_BELIEF_HISTORY_UPDATE_POLICY
            + "_plus_candidate_path_query_reachable_flow_route_value_cost_benefit_memory"
        ),
        "residual_memory_launch_recency_half_life": float(governor_config.residual_memory_launch_recency_half_life),
        "memory_opportunity_audit": "metrics/memory_opportunity_summary.csv",
        "safe_exploration_policy": "always_available_for_memory_policies_after_viability_filter_and_baseline_shield_no_final_run_branch",
        "thesis_facing_workflow": THESIS_FACING_WORKFLOW,
        "governor_config_override_active": config.governor_config is not None,
        "governor_config": governor_config_to_row(governor_config),
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
        "history_log_mode_requested": str(config.history_log_mode),
        "history_log_mode_resolved": _resolved_history_log_mode(config, protocol),
        "history_debug_sample_stride": int(config.history_debug_sample_stride),
        "history_debug_retention_policy": (
            "R9_reduced_smoke_full_debug_else_R10_R11_history_plot_summary_unless_explicitly_overridden"
            if str(config.history_log_mode) == "auto"
            else "explicit_user_selected_history_log_mode"
        ),
        "history_launch_plot_evidence_retained": True,
        "history_memory_dense_sample_log_policy": (
            "plot_summary_keeps_compact_weighted_per_primitive_memory_summaries_"
            "full_dense_sample_rows_retained_for_final_launches_r9_reduced_smoke_or_sampled_debug"
        ),
        "history_launch_retained_tables": [
            "episode_summary",
            "primitive_execution_log",
            "history_plot_trace",
            "history_memory_trace",
            "history_selector_summary",
        ],
        "history_launch_verbose_debug_tables": [
            "candidate_score_log",
            "selector_decision_log",
            "memory_residual_update_log",
            "belief_snapshot_log",
        ],
        "history_launch_verbose_debug_default": _resolved_history_log_mode(config, protocol) == "full_debug",
        "history_launch_verbose_debug_sampled": _resolved_history_log_mode(config, protocol) == "sampled_debug",
        "final_launch_verbose_debug_tables_retained": True,
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
        "active_governor_path": "mission_aligned_transition_viability_governor_v1",
        "active_governor_mission_terms": (
            "candidate-path front-wall progress, front-wall terminal proxy, progress-gated terminal total "
            "specific-energy proxy, wrong-boundary penalty, updraft/lift utility, residual memory, and unchanged safety filters"
        ),
        "boundary_near_status": "route_state_not_automatic_failure",
        "changed_case_active_fan_count_policy": "balanced_0_1_2_3_4_for_active_fan_number_variation"
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else "not_applicable",
        "changed_case_active_fan_count_sequence": list(R10_ACTIVE_FAN_COUNT_SEQUENCE)
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else [],
        "changed_case_arena_wide_fan_position_block_ids": list(R10_ARENA_WIDE_FAN_POSITION_BLOCK_IDS)
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else [],
        "changed_case_arena_wide_fan_position_xy_bounds_m": R10_ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else (),
        "changed_case_arena_wide_fan_position_safety_radius_m": R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M
        if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
        else "",
        "changed_case_plant_implementation_variation_scope": (
            "full-domain blocks sample plant/implementation per outer case; history launches keep fan layout, "
            "active count, and plant fixed while launch state varies and only mild updraft parameters vary in "
            "R10 L7 plus R11 L3/L6/L7"
            if str(protocol.stage_id) in CHANGED_CASE_VALIDATION_STAGE_IDS
            else "not_applicable"
        ),
        "legacy_recovery_threshold_alias_status": "superseded_by_transition_labels_contract",
        "launch_score_version": LAUNCH_SCORE_VERSION,
        "launch_score_policy": "additive_front_wall_mission_updraft_lift_terminal_specific_energy_score_no_airborne_time_reward",
        "launch_score_airborne_time_reward_status": "episode_flight_time_s_retained_for_audit_only",
        "launch_score_front_wall_target_x_w_m": float(TRUE_SAFE_BOUNDS.x_w_m[1]),
        "launch_score_terminal_y_w_bounds_m": list(TRUE_SAFE_BOUNDS.y_w_m),
        "launch_score_terminal_z_w_bounds_m": list(TRUE_SAFE_BOUNDS.z_w_m),
        "launch_score_mission_completion_score": float(MISSION_COMPLETION_SCORE),
        "launch_score_updraft_gain_bonus_cap": float(UPDRAFT_GAIN_SCORE_CAP),
        "launch_score_lift_dwell_bonus_cap": float(LIFT_DWELL_SCORE_CAP),
        "launch_score_terminal_specific_energy_reference_m": float(TERMINAL_SPECIFIC_ENERGY_REFERENCE_M),
        "launch_score_terminal_specific_energy_bonus_cap": float(TERMINAL_SPECIFIC_ENERGY_SCORE_CAP),
        "launch_score_terminal_specific_energy_bonus_status": "applied_only_after_front_wall_terminal_success",
        "launch_score_wrong_wall_exit_penalty": float(WRONG_WALL_EXIT_PENALTY),
        "launch_score_gravity_m_s2": float(SPECIFIC_ENERGY_GRAVITY_M_S2),
        "low_launch_speed_dry_air_sink_policy": (
            "raw primitive floor_violation is retained, but floor stop after a launch below the claim-bearing speed envelope "
            "is labelled expected_low_energy_dry_air_sink and excluded from governor/memory claim-bearing gates"
        ),
        "low_launch_speed_dry_air_threshold_m_s": float(LOW_LAUNCH_SPEED_DRY_AIR_THRESHOLD_M_S),
        "dry_air_energy_depletion_min_flight_time_s": float(DRY_AIR_ENERGY_DEPLETION_MIN_FLIGHT_TIME_S),
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
    source_record_load: FrozenControllerRecordLoadResult | None = None,
) -> None:
    manifest = {
        "manifest_version": protocol.manifest_version,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "stage_id": protocol.stage_id,
        "run_id": int(config.run_id),
        "run_label": str(config.run_label).strip(),
        "run_root": run_root.as_posix(),
        "blocked_reason": blocked_reason,
        "pass_gate": False,
        **_frozen_controller_source_manifest_fields(source_record_load),
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
        "- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, front-wall progress, front-wall terminal proxy, progress-gated terminal total specific-energy proxy, updraft gain, lift dwell, and residual memory.",
        f"- Memory policy: `{OUTER_LOOP_MEMORY_POLICY_VERSION}` maintains a case-local 0.1 m 3D updraft-utility belief map; each flown primitive writes dense executed-segment residual samples at 0.1 m spacing with launch-index recency decay. The in-flight controller and full diagnostics query the accumulated map through the same 0.2 m neighbourhood over seven probes. The timed in-flight boundary uses a compact controller-row selector fast path before the 0.100 s boundary, while table flushing, full candidate-row expansion, and post-hoc candidate/memory diagnostics stay outside that boundary. Both use bounded current-to-exit, reachable-cone, and short-horizon route-flow probes from the candidate exit. The selector collapses those map queries into one cost-benefit memory value: remembered flow benefit plus small information value minus frozen mission-score, front-progress, risk, and path-margin costs. The value acts only among already-viable candidates and is accepted only through the baseline shield after viability filtering.",
        f"- Governor learning strategy: `{OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION}` keeps online memory `{ONLINE_MEMORY_SCOPE}`; R10 calibration scope is `{R10_GLOBAL_CALIBRATION_SCOPE}` and R11 uses `{R11_GOVERNOR_HANDOFF_SCOPE}`.",
        f"- Calibration search policy: `{GOVERNOR_CALIBRATION_SEARCH_POLICY}`.",
        "- Memory opportunity audit: `memory_opportunity_summary.csv` and `memory_opportunity_decision_log.csv` report baseline-vs-memory candidate gaps, correction deltas, shield status, and accepted/rejected switch reasons; large decision logs are partitioned under `tables/memory_opportunity_decision_log/` with the metrics CSV kept as a small index.",
        "- The adaptive selector uses one baseline shield at every launch; there is no branch that treats a held-out final launch as a known final mission.",
        "- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.",
        f"- Launch score: `{LAUNCH_SCORE_VERSION}`; rewards front-wall terminal mission completion plus capped updraft/lift evidence and terminal total specific energy reserve. Airborne time and generic net/gross energy drift remain audit-only.",
        "- Dry-air or scheduled-zero-fan low-launch-speed floor stops keep the raw primitive `floor_violation` audit label, but are interpreted as expected energy depletion rather than governor or memory failure.",
        f"- Start-energy audit: `speed_bin_policy_ladder_summary.csv` reports mission/safety rates by environment ladder, library tier, policy, repeated-launch history length, and initial-speed bin. `start_energy_group_policy_ladder_summary.csv` separates low-start-energy launches from high-start-energy launches using `{START_ENERGY_FEASIBILITY_SPEED_THRESHOLD_M_S:.1f} m/s` as a fixed post-hoc reporting threshold; paired score-delta summaries are split the same way.",
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


def _run_folder_name(run_id: int, run_label: str = "") -> str:
    label = str(run_label).strip()
    return label if label else f"{int(run_id):03d}"


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
    parser.add_argument("--run-label", default="")
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
    parser.add_argument("--history-log-mode", choices=HISTORY_LOG_MODES, default="auto")
    parser.add_argument("--history-debug-sample-stride", type=int, default=DEFAULT_HISTORY_DEBUG_SAMPLE_STRIDE)
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
            run_label=args.run_label,
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
            history_log_mode=args.history_log_mode,
            history_debug_sample_stride=args.history_debug_sample_stride,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "blocked", "dry_run_schedule", "smoke_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
