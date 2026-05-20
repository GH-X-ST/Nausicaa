from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wing_wind_descriptors import WING_WIND_DESCRIPTOR_COLUMNS


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Dense Archive Constants
# 2) Output Schemas
# 3) Data Containers and Count Helpers
# 4) Manifest and Planning Tables
# =============================================================================


# =============================================================================
# 1) Dense Archive Constants
# =============================================================================
DENSE_TURNING_FAMILIES = (
    "mild_bank",
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
BASELINE_FAMILIES = ("glide", "recovery")
TARGET_LADDER_DEG = (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
DIRECTION_SIGNS = (-1, 1)
START_CLASSES = ("favourable", "mid_arena", "lift_sector", "random_stress")
START_CLASS_FRACTIONS = {
    "favourable": 0.25,
    "mid_arena": 0.20,
    "lift_sector": 0.35,
    "random_stress": 0.20,
}
FAN_LAYOUTS = ("single_fan", "four_fan")
LAYOUT_BRANCH_IDS = ("single_fan_branch", "four_fan_branch")
W0_SINGLE = "W0_single_fan_branch"
W1_SINGLE = "W1_single_fan"
W0_FOUR = "W0_four_fan_branch"
W1_FOUR = "W1_four_fan"
TEST_ENVIRONMENT_MODES = (W0_SINGLE, W1_SINGLE, W0_FOUR, W1_FOUR)
FAN_BRANCH_METADATA = {
    "single_fan": {
        "layout_branch_id": "single_fan_branch",
        "w0_environment_mode": W0_SINGLE,
        "w1_environment_mode": W1_SINGLE,
        "w0_fan_config_id": "single_fan_branch_dry_air",
        "w1_fan_config_id": "single_fan_nominal_updraft",
        "w0_updraft_model_id": "no_updraft_dry_air",
        "w1_updraft_model_id": "single_gaussian_var",
    },
    "four_fan": {
        "layout_branch_id": "four_fan_branch",
        "w0_environment_mode": W0_FOUR,
        "w1_environment_mode": W1_FOUR,
        "w0_fan_config_id": "four_fan_branch_dry_air",
        "w1_fan_config_id": "four_fan_nominal_updraft",
        "w0_updraft_model_id": "no_updraft_dry_air",
        "w1_updraft_model_id": "four_gaussian_var",
    },
}
ENVIRONMENT_ROLE_BY_FAMILY = {
    "glide": "dry_air_capable",
    "recovery": "dry_air_capable",
    "mild_bank": "dry_air_capable",
    "canyon_steep_bank": "updraft_assisted",
    "wingover_lite": "updraft_assisted",
    "bank_yaw_energy_retaining": "updraft_assisted",
}
CAMPAIGN = "10_dense_archive_planning"
PASS_NAME = "phase_b_task1_1_equal_fan_branch_paired_w0_w1_planning_scaffold"
PHASE_B_TASK = "task1_1_equal_fan_branch_paired_w0_w1_planning_scaffold"
SOURCE_STAGE0_MANIFEST = (
    "03_Control/05_Results/09_primitive_library/000_frozen_baseline/"
    "manifests/frozen_baseline_manifest_s000.json"
)
SOURCE_RUN007_MANIFEST = (
    "03_Control/05_Results/10_dense_archive_planning/007/"
    "manifests/archive_count_manifest_s007.json"
)
PROTECTED_PATHS = (
    "03_Control/05_Results/09_primitive_library/002",
    "03_Control/05_Results/09_primitive_library/003",
    "03_Control/05_Results/09_primitive_library/004",
    "03_Control/05_Results/09_primitive_library/005",
    "03_Control/05_Results/09_primitive_library/006",
    "03_Control/05_Results/09_primitive_library/000_frozen_baseline",
    "03_Control/05_Results/10_dense_archive_planning/007",
)
FORBIDDEN_CLAIMS = (
    "W0 archive executed",
    "W1 archive executed",
    "W2/W3/W4/W5 robustness or mission evidence completed",
    "active latency implemented",
    "envelope maps completed",
    "clustering completed",
    "governor generated from dense archive",
    "objective one completed",
    "objective two completed",
    "sim-to-real transfer demonstrated",
)
RECOMMENDED_NEXT_STEP = (
    "Run a small paired W0/W1 pilot sweep for descriptor storage and planning-row "
    "validation before any full archive."
)
LIFT_SECTOR_EDGE_FRACTION_REQUIREMENT = 0.50
LATENCY_CASE_PLANNED = "none"
LATENCY_ACCEPTANCE_ROLE = "nominal_latency_required_later_for_accepted_w1_envelopes"
LATENCY_MODEL_STATUS = "metadata_only_active_latency_deferred"
BRANCH_DECISION_SCOPE = "branch_local_only_no_cross_layout_decision_transfer"
DENSE_TRIAL_DESCRIPTOR_SCHEMA_IMPLEMENTED = True
DENSE_TRIAL_DESCRIPTOR_EXECUTION_PERFORMED = False
SUN24_DAILY_SCHEDULE_STEP = "sun24_dense_archive_pilot_envelope_clustering_scaffold"
SUN24_CONTROLLING_SOURCES = (
    "latest_user_instruction",
    "nausicaa_codex_sun24_envelope_pilot_guidance.md",
    "docs/Glider_Control_Project_Plan.md",
    "docs/Skills.md",
)
SUN24_ENVELOPE_MAP_SCAFFOLD_IMPLEMENTED = True
SUN24_CLUSTERING_SCAFFOLD_IMPLEMENTED = True
SUN24_PILOT_SWEEP_RUNNER_IMPLEMENTED = True
SUN24_PILOT_SWEEP_PERFORMED = False
SUN24_PRODUCTION_DENSE_ARCHIVE_PERFORMED = False
SUN24_HARDWARE_OR_MISSION_CLAIM = False
SUN24_BRANCH_LOCAL_DECISIONS_ONLY = True
NO_CROSS_BRANCH_FLAGS = {
    "no_cross_branch_promotion": True,
    "no_cross_branch_rejection": True,
    "no_cross_branch_cluster_merge": True,
    "no_cross_branch_safety_justification": True,
}

SAMPLING_RANGES = {
    "favourable": {
        "x_range_m": (2.0, 4.4),
        "y_range_m": (1.4, 3.0),
        "z_range_m": (1.2, 2.2),
        "speed_range_m_s": (6.0, 7.0),
        "phi_range_deg": (-5.0, 5.0),
        "theta_range_deg": (-4.0, 4.0),
        "psi_range_deg": (-25.0, 25.0),
        "p_range_rad_s": (-0.15, 0.15),
        "q_range_rad_s": (-0.15, 0.15),
        "r_range_rad_s": (-0.15, 0.15),
        "updraft_radius_range_m": (0.8, 1.8),
        "special_rule": "clean mission-corridor starts safely away from walls",
    },
    "mid_arena": {
        "x_range_m": (2.4, 5.4),
        "y_range_m": (1.0, 3.4),
        "z_range_m": (1.1, 2.6),
        "speed_range_m_s": (5.6, 7.2),
        "phi_range_deg": (-8.0, 8.0),
        "theta_range_deg": (-6.0, 6.0),
        "psi_range_deg": (-60.0, 60.0),
        "p_range_rad_s": (-0.25, 0.25),
        "q_range_rad_s": (-0.25, 0.25),
        "r_range_rad_s": (-0.25, 0.25),
        "updraft_radius_range_m": (0.5, 1.8),
        "special_rule": "neutral centre-arena starts without immediate boundary pressure",
    },
    "lift_sector": {
        "x_range_m": (1.35, 6.45),
        "y_range_m": (0.10, 4.30),
        "z_range_m": (1.0, 2.7),
        "speed_range_m_s": (5.2, 7.1),
        "phi_range_deg": (-10.0, 10.0),
        "theta_range_deg": (-8.0, 8.0),
        "psi_range_deg": (-180.0, 180.0),
        "p_range_rad_s": (-0.35, 0.35),
        "q_range_rad_s": (-0.35, 0.35),
        "r_range_rad_s": (-0.35, 0.35),
        "updraft_radius_range_m": (0.0, 1.4),
        "special_rule": "branch-local fan centres; at least half labelled edge or ring",
    },
    "random_stress": {
        "x_range_m": (1.30, 6.50),
        "y_range_m": (0.10, 4.30),
        "z_range_m": (0.70, 3.10),
        "speed_range_m_s": (4.8, 7.4),
        "phi_range_deg": (-18.0, 18.0),
        "theta_range_deg": (-12.0, 12.0),
        "psi_range_deg": (-180.0, 180.0),
        "p_range_rad_s": (-0.60, 0.60),
        "q_range_rad_s": (-0.60, 0.60),
        "r_range_rad_s": (-0.60, 0.60),
        "updraft_radius_range_m": (0.0, 2.1),
        "special_rule": "safe boundary-near starts with poor heading, low speed, and rates",
    },
}


# =============================================================================
# 2) Output Schemas
# =============================================================================
TARGET_ENVIRONMENT_PLAN_COLUMNS = (
    "fan_layout",
    "layout_branch_id",
    "fan_config_id",
    "updraft_model_id",
    "test_environment_mode",
    "paired_environment_mode",
    "family",
    "family_role",
    "environment_role",
    "validity_gate_role",
    "first_validity_gate_environment",
    "w0_failure_policy",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "count_basis",
    "planned_floor_start_count",
    "planned_target_start_count",
    "planned_floor_trial_count",
    "planned_target_trial_count",
    "pilot_start_count",
    "included_in_paired_archive",
    "branch_decision_scope",
    "no_cross_branch_promotion",
    "no_cross_branch_rejection",
    "no_cross_branch_cluster_merge",
    "no_cross_branch_safety_justification",
    "latency_case_planned",
    "latency_acceptance_role",
    "latency_model_status",
    "no_rollout_performed",
)

SAMPLING_STRATA_COLUMNS = (
    "fan_layout",
    "layout_branch_id",
    "fan_config_id",
    "updraft_model_id",
    "start_class",
    "fraction_requested",
    "pilot_sample_count",
    "floor_archive_count_reference",
    "target_archive_count_reference",
    "x_range_m",
    "y_range_m",
    "z_range_m",
    "speed_range_m_s",
    "phi_range_deg",
    "theta_range_deg",
    "psi_range_deg",
    "p_range_rad_s",
    "q_range_rad_s",
    "r_range_rad_s",
    "updraft_radius_range_m",
    "special_rule",
    "branch_layout_note",
    "layout_specific_sampling_required",
    "no_cross_branch_merge",
    "no_rollout_performed",
)

START_STATE_MANIFEST_COLUMNS = (
    "sample_id",
    "paired_sample_key",
    "seed",
    "sampling_round",
    "fan_layout",
    "layout_branch_id",
    "fan_config_id",
    "updraft_model_id",
    "branch_seed_family",
    "start_class",
    "family",
    "target_heading_deg",
    "direction_sign",
    "environment_role",
    "paired_environment_modes",
    "first_validity_gate_environment",
    "count_basis",
    "x_w_m",
    "y_w_m",
    "z_w_m",
    "speed_m_s",
    "phi_rad",
    "theta_rad",
    "psi_rad",
    "u_m_s",
    "v_m_s",
    "w_m_s",
    "p_rad_s",
    "q_rad_s",
    "r_rad_s",
    "updraft_center_x_m",
    "updraft_center_y_m",
    "updraft_relative_radius_m",
    "updraft_relative_azimuth_rad",
    "updraft_relative_height_m",
    "updraft_sector_label",
    "left_wing_lift_exposure_preference",
    "right_wing_lift_exposure_preference",
    "wing_exposure_bookkeeping_status",
    *WING_WIND_DESCRIPTOR_COLUMNS,
    "true_safe_start",
    "start_generation_status",
    "layout_specific_sample_generated",
    "no_rollout_performed",
)

DRY_RUN_CANDIDATE_COLUMNS = (
    "candidate_id",
    "sample_id",
    "paired_sample_key",
    "seed",
    "sampling_round",
    "fan_layout",
    "layout_branch_id",
    "fan_config_id",
    "updraft_model_id",
    "test_environment_mode",
    "paired_environment_mode",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "environment_role",
    "validity_gate_role",
    "first_validity_gate_environment",
    "w0_failure_policy",
    "acceptance_interpretation",
    *WING_WIND_DESCRIPTOR_COLUMNS,
    "count_basis",
    "planned_floor_trial_count",
    "planned_target_trial_count",
    "pilot_trial_count",
    "latency_case_planned",
    "latency_acceptance_role",
    "latency_model_status",
    "planned_replay_status",
    "planned_result_path",
    "branch_decision_scope",
    "no_cross_branch_promotion",
    "no_cross_branch_rejection",
    "no_cross_branch_cluster_merge",
    "no_cross_branch_safety_justification",
    "no_rollout_performed",
    "notes",
)


# =============================================================================
# 3) Data Containers and Count Helpers
# =============================================================================
@dataclass(frozen=True)
class DenseArchivePlanConfig:
    run_id: int = 8
    random_seed: int = 20260520
    pilot_start_states_per_family_target_direction: int = 10
    w1_floor_start_states_per_family_target_direction: int = 5000
    w1_target_start_states_per_family_target_direction: int = 7500
    w1_floor_total_trials_per_branch: int = 350000
    w1_target_total_trials_per_branch: int = 500000
    w0_floor_total_trials_per_branch: int = 150000
    w0_target_total_trials_per_branch: int = 300000
    test_environment_modes: tuple[str, ...] = TEST_ENVIRONMENT_MODES


def branch_group_count() -> int:
    return len(DENSE_TURNING_FAMILIES) * len(TARGET_LADDER_DEG) * len(DIRECTION_SIGNS)


def branch_baseline_group_count() -> int:
    return len(BASELINE_FAMILIES) * len(DIRECTION_SIGNS)


def branch_start_group_count() -> int:
    return branch_group_count() + branch_baseline_group_count()


def w1_floor_turning_trials_per_branch(config: DenseArchivePlanConfig) -> int:
    return branch_group_count() * int(config.w1_floor_start_states_per_family_target_direction)


def w1_target_turning_trials_per_branch(config: DenseArchivePlanConfig) -> int:
    return branch_group_count() * int(config.w1_target_start_states_per_family_target_direction)


def w1_floor_baseline_or_filler_trials_per_branch(config: DenseArchivePlanConfig) -> int:
    return int(config.w1_floor_total_trials_per_branch) - w1_floor_turning_trials_per_branch(config)


def w1_target_baseline_or_filler_trials_per_branch(config: DenseArchivePlanConfig) -> int:
    return int(config.w1_target_total_trials_per_branch) - w1_target_turning_trials_per_branch(config)


def pilot_start_state_rows_per_branch(config: DenseArchivePlanConfig) -> int:
    return branch_start_group_count() * int(config.pilot_start_states_per_family_target_direction)


def pilot_start_state_rows_all_branches(config: DenseArchivePlanConfig) -> int:
    return len(FAN_LAYOUTS) * pilot_start_state_rows_per_branch(config)


def pilot_candidate_rows_all_branches(config: DenseArchivePlanConfig) -> int:
    return 2 * pilot_start_state_rows_all_branches(config)


def _distribute_total(total: int, count: int) -> tuple[int, ...]:
    base = int(total) // int(count)
    remainder = int(total) % int(count)
    return tuple(base + (1 if index < remainder else 0) for index in range(count))


def _baseline_count_basis() -> str:
    return (
        "blank target because glide/recovery baseline rows are not heading-change targets; "
        "both direction signs are branch-local bookkeeping rows"
    )


def _paired_modes(fan_layout: str) -> tuple[str, str]:
    meta = FAN_BRANCH_METADATA[fan_layout]
    return str(meta["w0_environment_mode"]), str(meta["w1_environment_mode"])


def _environment_values(fan_layout: str, environment_mode: str) -> dict[str, str]:
    meta = FAN_BRANCH_METADATA[fan_layout]
    if environment_mode == meta["w0_environment_mode"]:
        return {
            "fan_config_id": str(meta["w0_fan_config_id"]),
            "updraft_model_id": str(meta["w0_updraft_model_id"]),
            "paired_environment_mode": str(meta["w1_environment_mode"]),
        }
    return {
        "fan_config_id": str(meta["w1_fan_config_id"]),
        "updraft_model_id": str(meta["w1_updraft_model_id"]),
        "paired_environment_mode": str(meta["w0_environment_mode"]),
    }


def _validity_fields(
    fan_layout: str,
    environment_mode: str,
    environment_role: str,
) -> tuple[str, str, str, str]:
    w0_mode, w1_mode = _paired_modes(fan_layout)
    if environment_role == "updraft_assisted":
        if environment_mode == w0_mode:
            return (
                "ablation_only",
                w1_mode,
                "log_ablation_do_not_reject_if_w1_valid",
                "ablation_only_not_rejection",
            )
        return (
            "first_validity_gate",
            w1_mode,
            "not_applicable_w1_validity_gate",
            "first_validity_gate",
        )
    if environment_mode == w0_mode:
        return (
            "baseline_gate",
            w0_mode,
            "w0_failure_can_reject_branch_local_dry_air_capable_baseline",
            "baseline_gate",
        )
    return (
        "updraft_interaction_check",
        w0_mode,
        "w1_interaction_check_not_primary_rejection_gate",
        "interaction_check",
    )


def _row_counts_for_environment(
    config: DenseArchivePlanConfig,
    environment_mode: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    group_count = branch_start_group_count()
    if environment_mode.startswith("W0_"):
        floor = _distribute_total(config.w0_floor_total_trials_per_branch, group_count)
        target = _distribute_total(config.w0_target_total_trials_per_branch, group_count)
        return floor, target, floor, target

    turning_floor = (int(config.w1_floor_start_states_per_family_target_direction),) * branch_group_count()
    turning_target = (int(config.w1_target_start_states_per_family_target_direction),) * branch_group_count()
    filler_floor = _distribute_total(
        w1_floor_baseline_or_filler_trials_per_branch(config),
        branch_baseline_group_count(),
    )
    filler_target = _distribute_total(
        w1_target_baseline_or_filler_trials_per_branch(config),
        branch_baseline_group_count(),
    )
    floor = turning_floor + filler_floor
    target = turning_target + filler_target
    return floor, target, floor, target


# =============================================================================
# 4) Manifest and Planning Tables
# =============================================================================
def build_archive_count_manifest(config: DenseArchivePlanConfig) -> dict[str, object]:
    """Return the equal-branch W0/W1 archive count contract without trials."""

    floor_filler = w1_floor_baseline_or_filler_trials_per_branch(config)
    target_filler = w1_target_baseline_or_filler_trials_per_branch(config)
    if floor_filler < 0 or target_filler < 0:
        raise ValueError("W1 baseline/filler allocation cannot be negative.")

    branch_contract = {
        fan_layout: {
            "layout_branch_id": FAN_BRANCH_METADATA[fan_layout]["layout_branch_id"],
            "w0_floor_total_trials": int(config.w0_floor_total_trials_per_branch),
            "w0_target_total_trials": int(config.w0_target_total_trials_per_branch),
            "w1_floor_total_trials": int(config.w1_floor_total_trials_per_branch),
            "w1_target_total_trials": int(config.w1_target_total_trials_per_branch),
            "branch_decision_scope": BRANCH_DECISION_SCOPE,
        }
        for fan_layout in FAN_LAYOUTS
    }

    return {
        "run_id": int(config.run_id),
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "source_stage0_manifest": SOURCE_STAGE0_MANIFEST,
        "source_run007_manifest": SOURCE_RUN007_MANIFEST,
        "stage0_gate_status_seen": "pending_runner_check",
        "run007_preserved": "pending_runner_check",
        "phase_b_task": PHASE_B_TASK,
        "dense_archive_execution_performed": False,
        "paired_w0_w1_execution_performed": False,
        "full_w0_archive_performed": False,
        "full_w1_archive_performed": False,
        "fan_layouts": list(FAN_LAYOUTS),
        "layout_branch_ids": list(LAYOUT_BRANCH_IDS),
        "test_environment_modes": list(config.test_environment_modes),
        "fan_branch_metadata": FAN_BRANCH_METADATA,
        "target_ladder_deg": list(TARGET_LADDER_DEG),
        "direction_signs": list(DIRECTION_SIGNS),
        "start_classes": list(START_CLASSES),
        "start_class_fractions": dict(START_CLASS_FRACTIONS),
        "dense_turning_families": list(DENSE_TURNING_FAMILIES),
        "baseline_families": list(BASELINE_FAMILIES),
        "environment_role_by_family": dict(ENVIRONMENT_ROLE_BY_FAMILY),
        "branch_count_contract": branch_contract,
        "w1_floor_start_states_per_family_target_direction": int(
            config.w1_floor_start_states_per_family_target_direction
        ),
        "w1_target_start_states_per_family_target_direction": int(
            config.w1_target_start_states_per_family_target_direction
        ),
        "w1_floor_turning_trials_per_branch": w1_floor_turning_trials_per_branch(config),
        "w1_target_turning_trials_per_branch": w1_target_turning_trials_per_branch(config),
        "w1_floor_baseline_or_filler_trials_per_branch": floor_filler,
        "w1_target_baseline_or_filler_trials_per_branch": target_filler,
        "w1_floor_total_trials_per_branch": int(config.w1_floor_total_trials_per_branch),
        "w1_target_total_trials_per_branch": int(config.w1_target_total_trials_per_branch),
        "w1_floor_total_trials_all_branches": len(FAN_LAYOUTS)
        * int(config.w1_floor_total_trials_per_branch),
        "w1_target_total_trials_all_branches": len(FAN_LAYOUTS)
        * int(config.w1_target_total_trials_per_branch),
        "w0_floor_total_trials_per_branch": int(config.w0_floor_total_trials_per_branch),
        "w0_target_total_trials_per_branch": int(config.w0_target_total_trials_per_branch),
        "w0_floor_total_trials_all_branches": len(FAN_LAYOUTS)
        * int(config.w0_floor_total_trials_per_branch),
        "w0_target_total_trials_all_branches": len(FAN_LAYOUTS)
        * int(config.w0_target_total_trials_per_branch),
        "combined_floor_total_trials_all_branches": len(FAN_LAYOUTS)
        * (int(config.w0_floor_total_trials_per_branch) + int(config.w1_floor_total_trials_per_branch)),
        "combined_target_total_trials_all_branches": len(FAN_LAYOUTS)
        * (int(config.w0_target_total_trials_per_branch) + int(config.w1_target_total_trials_per_branch)),
        "pilot_start_states_per_family_target_direction": int(
            config.pilot_start_states_per_family_target_direction
        ),
        "pilot_start_state_rows_per_branch": pilot_start_state_rows_per_branch(config),
        "pilot_start_state_rows_all_branches": pilot_start_state_rows_all_branches(config),
        "pilot_candidate_rows_all_branches": pilot_candidate_rows_all_branches(config),
        "paired_sample_key_scope": "branch_local_pairs_only",
        "no_cross_branch_decision_rule": BRANCH_DECISION_SCOPE,
        "latency_metadata_only": True,
        "active_latency_implementation_deferred": True,
        "wing_wind_descriptor_logging_implemented": True,
        "wing_wind_descriptor_scope": "planning_start_and_candidate_rows_only",
        "wing_wind_descriptor_no_rollout": True,
        "dense_trial_descriptor_schema_implemented": DENSE_TRIAL_DESCRIPTOR_SCHEMA_IMPLEMENTED,
        "dense_trial_descriptor_execution_performed": DENSE_TRIAL_DESCRIPTOR_EXECUTION_PERFORMED,
        "sun24_daily_schedule_step": SUN24_DAILY_SCHEDULE_STEP,
        "sun24_controlling_sources": list(SUN24_CONTROLLING_SOURCES),
        "sun24_envelope_map_scaffold_implemented": SUN24_ENVELOPE_MAP_SCAFFOLD_IMPLEMENTED,
        "sun24_clustering_scaffold_implemented": SUN24_CLUSTERING_SCAFFOLD_IMPLEMENTED,
        "sun24_pilot_sweep_runner_implemented": SUN24_PILOT_SWEEP_RUNNER_IMPLEMENTED,
        "sun24_pilot_sweep_performed": SUN24_PILOT_SWEEP_PERFORMED,
        "sun24_production_dense_archive_performed": SUN24_PRODUCTION_DENSE_ARCHIVE_PERFORMED,
        "sun24_hardware_or_mission_claim": SUN24_HARDWARE_OR_MISSION_CLAIM,
        "sun24_branch_local_decisions_only": SUN24_BRANCH_LOCAL_DECISIONS_ONLY,
        "sun24_no_overclaiming_boundary": (
            "scaffold_only_not_production_archive_not_mission_not_hardware_not_sim_real"
        ),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "recommended_next_step": RECOMMENDED_NEXT_STEP,
        "protected_paths_checked": list(PROTECTED_PATHS),
        "protected_hash_check_status": "pending_runner_check",
        "output_files": {},
    }


def build_target_environment_plan(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Return branch-separated paired W0/W1 planning rows."""

    rows: list[dict[str, object]] = []
    for fan_layout in FAN_LAYOUTS:
        meta = FAN_BRANCH_METADATA[fan_layout]
        for environment_mode in (meta["w0_environment_mode"], meta["w1_environment_mode"]):
            floor_starts, target_starts, floor_trials, target_trials = _row_counts_for_environment(
                config,
                str(environment_mode),
            )
            groups = _branch_environment_groups()
            for index, group in enumerate(groups):
                environment_role = ENVIRONMENT_ROLE_BY_FAMILY[str(group["family"])]
                validity, first_gate, w0_policy, _ = _validity_fields(
                    fan_layout,
                    str(environment_mode),
                    environment_role,
                )
                env_values = _environment_values(fan_layout, str(environment_mode))
                rows.append(
                    {
                        "fan_layout": fan_layout,
                        "layout_branch_id": meta["layout_branch_id"],
                        "fan_config_id": env_values["fan_config_id"],
                        "updraft_model_id": env_values["updraft_model_id"],
                        "test_environment_mode": environment_mode,
                        "paired_environment_mode": env_values["paired_environment_mode"],
                        "family": group["family"],
                        "family_role": group["family_role"],
                        "environment_role": environment_role,
                        "validity_gate_role": validity,
                        "first_validity_gate_environment": first_gate,
                        "w0_failure_policy": w0_policy,
                        "target_heading_deg": group["target_heading_deg"],
                        "direction_sign": group["direction_sign"],
                        "start_class": "all_start_classes_planned",
                        "count_basis": group["count_basis"],
                        "planned_floor_start_count": floor_starts[index],
                        "planned_target_start_count": target_starts[index],
                        "planned_floor_trial_count": floor_trials[index],
                        "planned_target_trial_count": target_trials[index],
                        "pilot_start_count": int(config.pilot_start_states_per_family_target_direction),
                        "included_in_paired_archive": True,
                        "branch_decision_scope": BRANCH_DECISION_SCOPE,
                        **NO_CROSS_BRANCH_FLAGS,
                        "latency_case_planned": LATENCY_CASE_PLANNED,
                        "latency_acceptance_role": LATENCY_ACCEPTANCE_ROLE,
                        "latency_model_status": LATENCY_MODEL_STATUS,
                        "no_rollout_performed": True,
                    }
                )
    return pd.DataFrame(rows, columns=TARGET_ENVIRONMENT_PLAN_COLUMNS)


def build_target_direction_plan(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Compatibility wrapper for the run-008 target-environment table."""

    return build_target_environment_plan(config)


def _branch_environment_groups() -> tuple[dict[str, object], ...]:
    groups: list[dict[str, object]] = []
    for family in DENSE_TURNING_FAMILIES:
        for target in TARGET_LADDER_DEG:
            for direction in DIRECTION_SIGNS:
                groups.append(
                    {
                        "family": family,
                        "family_role": "turning",
                        "target_heading_deg": float(target),
                        "direction_sign": int(direction),
                        "count_basis": "per_branch_turning_family_target_direction_all_start_classes",
                    }
                )
    for family in BASELINE_FAMILIES:
        for direction in DIRECTION_SIGNS:
            groups.append(
                {
                    "family": family,
                    "family_role": "baseline",
                    "target_heading_deg": "",
                    "direction_sign": int(direction),
                    "count_basis": _baseline_count_basis(),
                }
            )
    return tuple(groups)
