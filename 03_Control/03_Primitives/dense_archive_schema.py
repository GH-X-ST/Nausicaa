from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Dense Archive Constants
# 2) Data Containers and Count Helpers
# 3) Manifest and Planning Tables
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
TEST_ENVIRONMENT_MODE = "W0_dry_air"
ENVIRONMENT_ROLE_BY_FAMILY = {
    "glide": "dry_air_capable",
    "recovery": "dry_air_capable",
    "mild_bank": "dry_air_capable",
    "canyon_steep_bank": "updraft_assisted",
    "wingover_lite": "updraft_assisted",
    "bank_yaw_energy_retaining": "updraft_assisted",
}
CAMPAIGN = "10_dense_archive_planning"
PASS_NAME = "phase_b_task1_dense_archive_planning_scaffold"
PHASE_B_TASK = "task1_dense_archive_planning_scaffold"
SOURCE_STAGE0_MANIFEST = (
    "03_Control/05_Results/09_primitive_library/000_frozen_baseline/"
    "manifests/frozen_baseline_manifest_s000.json"
)
PROTECTED_STAGE0_PATHS = (
    "03_Control/05_Results/09_primitive_library/002",
    "03_Control/05_Results/09_primitive_library/003",
    "03_Control/05_Results/09_primitive_library/004",
    "03_Control/05_Results/09_primitive_library/005",
    "03_Control/05_Results/09_primitive_library/006",
    "03_Control/05_Results/09_primitive_library/000_frozen_baseline",
)
FORBIDDEN_CLAIMS = (
    "W0 dense archive executed",
    "W1/W2/W3 robustness completed",
    "envelope maps completed",
    "clustering completed",
    "objective one completed",
    "objective two completed",
    "sim-to-real transfer demonstrated",
)
RECOMMENDED_NEXT_STEP = "Run a 5k to 20k pilot sweep for descriptor logging and storage validation."
LIFT_SECTOR_EDGE_FRACTION_REQUIREMENT = 0.50

TARGET_DIRECTION_PLAN_COLUMNS = (
    "family",
    "family_role",
    "environment_role",
    "test_environment_mode",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "count_basis",
    "planned_min_start_count",
    "planned_target_start_count",
    "planned_min_trial_count",
    "planned_target_trial_count",
    "pilot_trial_count",
    "included_in_dense_w0",
    "no_rollout_performed",
)

SAMPLING_STRATA_COLUMNS = (
    "start_class",
    "fraction_requested",
    "pilot_sample_count",
    "minimum_archive_count_reference",
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
    "test_environment_mode",
    "no_rollout_performed",
)

START_STATE_MANIFEST_COLUMNS = (
    "sample_id",
    "seed",
    "sampling_round",
    "start_class",
    "family",
    "target_heading_deg",
    "direction_sign",
    "environment_role",
    "test_environment_mode",
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
    "updraft_config",
    "updraft_center_x_m",
    "updraft_center_y_m",
    "updraft_relative_radius_m",
    "updraft_relative_azimuth_rad",
    "updraft_relative_height_m",
    "updraft_sector_label",
    "left_wing_lift_exposure_preference",
    "right_wing_lift_exposure_preference",
    "wing_exposure_bookkeeping_status",
    "true_safe_start",
    "start_generation_status",
    "no_rollout_performed",
)

DRY_RUN_CANDIDATE_COLUMNS = (
    "candidate_id",
    "sample_id",
    "seed",
    "sampling_round",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "environment_role",
    "test_environment_mode",
    "count_basis",
    "planned_min_trial_count",
    "planned_target_trial_count",
    "pilot_trial_count",
    "planned_replay_status",
    "planned_result_path",
    "no_rollout_performed",
    "notes",
)

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
        "special_rule": "use measured/fitted fan centres; at least half labelled edge or ring",
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
# 2) Data Containers and Count Helpers
# =============================================================================
@dataclass(frozen=True)
class DenseArchivePlanConfig:
    run_id: int = 7
    random_seed: int = 20260520
    pilot_start_states_per_family_target_direction: int = 10
    minimum_start_states_per_family_target_direction: int = 2000
    target_start_states_per_family_target_direction: int = 5000
    minimum_baseline_w0_trials: int = 20000
    target_baseline_w0_trials: int = 30000
    target_w0_total_trial_range: tuple[int, int] = (350000, 500000)
    test_environment_mode: str = TEST_ENVIRONMENT_MODE


def minimum_w0_turning_trials(config: DenseArchivePlanConfig) -> int:
    return (
        len(DENSE_TURNING_FAMILIES)
        * len(TARGET_LADDER_DEG)
        * len(DIRECTION_SIGNS)
        * int(config.minimum_start_states_per_family_target_direction)
    )


def target_w0_turning_trials(config: DenseArchivePlanConfig) -> int:
    return (
        len(DENSE_TURNING_FAMILIES)
        * len(TARGET_LADDER_DEG)
        * len(DIRECTION_SIGNS)
        * int(config.target_start_states_per_family_target_direction)
    )


def pilot_turning_trial_count(config: DenseArchivePlanConfig) -> int:
    return (
        len(DENSE_TURNING_FAMILIES)
        * len(TARGET_LADDER_DEG)
        * len(DIRECTION_SIGNS)
        * int(config.pilot_start_states_per_family_target_direction)
    )


def pilot_baseline_trial_count(config: DenseArchivePlanConfig) -> int:
    return (
        len(BASELINE_FAMILIES)
        * len(DIRECTION_SIGNS)
        * int(config.pilot_start_states_per_family_target_direction)
    )


def minimum_w0_total_trials(config: DenseArchivePlanConfig) -> int:
    return minimum_w0_turning_trials(config) + int(config.minimum_baseline_w0_trials)


def target_w0_total_trials(config: DenseArchivePlanConfig) -> int:
    return target_w0_turning_trials(config) + int(config.target_baseline_w0_trials)


def baseline_minimum_trials_per_direction(config: DenseArchivePlanConfig) -> int:
    return int(config.minimum_baseline_w0_trials) // (len(BASELINE_FAMILIES) * len(DIRECTION_SIGNS))


def baseline_target_trials_per_direction(config: DenseArchivePlanConfig) -> int:
    return int(config.target_baseline_w0_trials) // (len(BASELINE_FAMILIES) * len(DIRECTION_SIGNS))


# =============================================================================
# 3) Manifest and Planning Tables
# =============================================================================
def build_archive_count_manifest(config: DenseArchivePlanConfig) -> dict[str, object]:
    """Return the Phase B W0 archive count contract without running trials."""

    target_total = target_w0_total_trials(config)
    target_low, target_high = config.target_w0_total_trial_range
    if not int(target_low) <= target_total <= int(target_high):
        raise ValueError("target W0 total trials must lie inside the project-plan target range.")

    return {
        "run_id": int(config.run_id),
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "source_stage0_manifest": SOURCE_STAGE0_MANIFEST,
        "stage0_gate_status_seen": "pending_runner_check",
        "phase_b_task": PHASE_B_TASK,
        "dense_archive_execution_performed": False,
        "full_w0_archive_performed": False,
        "test_environment_mode": config.test_environment_mode,
        "target_ladder_deg": list(TARGET_LADDER_DEG),
        "direction_signs": list(DIRECTION_SIGNS),
        "start_classes": list(START_CLASSES),
        "start_class_fractions": dict(START_CLASS_FRACTIONS),
        "dense_turning_families": list(DENSE_TURNING_FAMILIES),
        "baseline_families": list(BASELINE_FAMILIES),
        "environment_role_by_family": dict(ENVIRONMENT_ROLE_BY_FAMILY),
        "minimum_start_states_per_family_target_direction": int(
            config.minimum_start_states_per_family_target_direction
        ),
        "target_start_states_per_family_target_direction": int(
            config.target_start_states_per_family_target_direction
        ),
        "minimum_w0_turning_trials": minimum_w0_turning_trials(config),
        "target_w0_turning_trials": target_w0_turning_trials(config),
        "minimum_w0_baseline_trials": int(config.minimum_baseline_w0_trials),
        "target_w0_baseline_trials": int(config.target_baseline_w0_trials),
        "minimum_w0_total_trials": minimum_w0_total_trials(config),
        "target_w0_total_trials": target_total,
        "target_w0_total_trial_range": list(config.target_w0_total_trial_range),
        "pilot_start_states_per_family_target_direction": int(
            config.pilot_start_states_per_family_target_direction
        ),
        "pilot_turning_trial_count": pilot_turning_trial_count(config),
        "pilot_baseline_trial_count": pilot_baseline_trial_count(config),
        "pilot_total_candidate_count": pilot_turning_trial_count(config) + pilot_baseline_trial_count(config),
        "sampling_ranges": SAMPLING_RANGES,
        "lift_sector_edge_fraction_requirement": LIFT_SECTOR_EDGE_FRACTION_REQUIREMENT,
        "random_seed": int(config.random_seed),
        "output_files": {},
        "protected_stage0_paths_checked": list(PROTECTED_STAGE0_PATHS),
        "protected_hash_check_status": "pending_runner_check",
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "recommended_next_step": RECOMMENDED_NEXT_STEP,
    }


def build_target_direction_plan(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Return the planned dense W0 family, target, direction, and count table."""

    rows: list[dict[str, object]] = []
    for family in DENSE_TURNING_FAMILIES:
        for target in TARGET_LADDER_DEG:
            for direction in DIRECTION_SIGNS:
                rows.append(
                    {
                        "family": family,
                        "family_role": "turning",
                        "environment_role": ENVIRONMENT_ROLE_BY_FAMILY[family],
                        "test_environment_mode": config.test_environment_mode,
                        "target_heading_deg": float(target),
                        "direction_sign": int(direction),
                        "start_class": "all_start_classes_planned",
                        "count_basis": "per_turning_family_target_direction_all_start_classes",
                        "planned_min_start_count": int(
                            config.minimum_start_states_per_family_target_direction
                        ),
                        "planned_target_start_count": int(
                            config.target_start_states_per_family_target_direction
                        ),
                        "planned_min_trial_count": int(
                            config.minimum_start_states_per_family_target_direction
                        ),
                        "planned_target_trial_count": int(
                            config.target_start_states_per_family_target_direction
                        ),
                        "pilot_trial_count": int(config.pilot_start_states_per_family_target_direction),
                        "included_in_dense_w0": True,
                        "no_rollout_performed": True,
                    }
                )

    count_basis = (
        "blank target because glide/recovery baseline rows are not heading-change targets; "
        "both direction signs are bookkeeping rows that split each baseline family allocation"
    )
    for family in BASELINE_FAMILIES:
        for direction in DIRECTION_SIGNS:
            rows.append(
                {
                    "family": family,
                    "family_role": "baseline",
                    "environment_role": ENVIRONMENT_ROLE_BY_FAMILY[family],
                    "test_environment_mode": config.test_environment_mode,
                    "target_heading_deg": "",
                    "direction_sign": int(direction),
                    "start_class": "all_start_classes_planned",
                    "count_basis": count_basis,
                    "planned_min_start_count": baseline_minimum_trials_per_direction(config),
                    "planned_target_start_count": baseline_target_trials_per_direction(config),
                    "planned_min_trial_count": baseline_minimum_trials_per_direction(config),
                    "planned_target_trial_count": baseline_target_trials_per_direction(config),
                    "pilot_trial_count": int(config.pilot_start_states_per_family_target_direction),
                    "included_in_dense_w0": True,
                    "no_rollout_performed": True,
                }
            )
    return pd.DataFrame(rows, columns=TARGET_DIRECTION_PLAN_COLUMNS)
