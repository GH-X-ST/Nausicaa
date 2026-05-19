from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = REPO_ROOT / "03_Control"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_schema import DenseArchivePlanConfig  # noqa: E402
from dense_start_state_sampling import (  # noqa: E402
    build_dry_run_candidate_inventory,
    build_start_state_manifest,
)
from run_dense_archive_planning import run_dense_archive_planning  # noqa: E402


TARGET_DIRECTION_COLUMNS = [
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
]
SAMPLING_STRATA_COLUMNS = [
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
]
START_STATE_COLUMNS = [
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
]
DRY_RUN_COLUMNS = [
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
]
MANIFEST_FIELDS = {
    "run_id",
    "campaign",
    "pass_name",
    "source_stage0_manifest",
    "stage0_gate_status_seen",
    "phase_b_task",
    "dense_archive_execution_performed",
    "full_w0_archive_performed",
    "test_environment_mode",
    "target_ladder_deg",
    "direction_signs",
    "start_classes",
    "start_class_fractions",
    "dense_turning_families",
    "baseline_families",
    "environment_role_by_family",
    "minimum_start_states_per_family_target_direction",
    "target_start_states_per_family_target_direction",
    "minimum_w0_turning_trials",
    "target_w0_turning_trials",
    "minimum_w0_baseline_trials",
    "target_w0_baseline_trials",
    "minimum_w0_total_trials",
    "target_w0_total_trials",
    "target_w0_total_trial_range",
    "pilot_start_states_per_family_target_direction",
    "pilot_turning_trial_count",
    "pilot_baseline_trial_count",
    "pilot_total_candidate_count",
    "sampling_ranges",
    "lift_sector_edge_fraction_requirement",
    "random_seed",
    "output_files",
    "protected_stage0_paths_checked",
    "protected_hash_check_status",
    "forbidden_claims",
    "recommended_next_step",
}
PROTECTED_PATHS = [
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "002",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "003",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "004",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "005",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "006",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "000_frozen_baseline",
]


@pytest.fixture(scope="module")
def generated_outputs() -> tuple[dict[str, Path], dict[str, str], dict[str, str]]:
    before = _protected_hashes()
    paths = run_dense_archive_planning(
        run_id=7,
        overwrite=True,
        pilot_start_states_per_family_target_direction=10,
    )
    after = _protected_hashes()
    return paths, before, after


def test_runner_writes_expected_outputs_and_manifest_counts(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    for key, path in paths.items():
        assert path.exists(), key
        if key != "root":
            assert "10_dense_archive_planning" in path.parts
            assert "007" in path.parts

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert MANIFEST_FIELDS.issubset(manifest)
    assert manifest["stage0_gate_status_seen"] == "passed"
    assert manifest["test_environment_mode"] == "W0_dry_air"
    assert manifest["dense_archive_execution_performed"] is False
    assert manifest["full_w0_archive_performed"] is False
    assert manifest["minimum_w0_turning_trials"] == 128000
    assert manifest["target_w0_turning_trials"] == 320000
    assert manifest["minimum_w0_baseline_trials"] == 20000
    assert manifest["target_w0_baseline_trials"] == 30000
    assert manifest["minimum_w0_total_trials"] == 148000
    assert manifest["target_w0_total_trials"] == 350000
    assert manifest["target_w0_total_trial_range"] == [350000, 500000]
    assert manifest["pilot_turning_trial_count"] == 640
    assert manifest["pilot_baseline_trial_count"] == 40
    assert manifest["pilot_total_candidate_count"] == 680
    assert manifest["protected_hash_check_status"] == "unchanged"
    assert "W0 dense archive executed" in manifest["forbidden_claims"]


def test_target_direction_plan_schema_counts_and_bookkeeping(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    plan = pd.read_csv(paths["target_direction_plan_csv"])

    assert list(plan.columns) == TARGET_DIRECTION_COLUMNS
    assert len(plan) == 68
    assert set(plan["test_environment_mode"]) == {"W0_dry_air"}
    assert set(plan["no_rollout_performed"]) == {True}
    assert int(plan["planned_min_trial_count"].sum()) == 148000
    assert int(plan["planned_target_trial_count"].sum()) == 350000

    turning = plan[plan["family_role"] == "turning"]
    assert len(turning) == 64
    assert set(turning["family"]) == {
        "mild_bank",
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    }
    assert set(turning["target_heading_deg"].astype(float)) == {
        15.0,
        30.0,
        45.0,
        60.0,
        90.0,
        120.0,
        150.0,
        180.0,
    }
    assert set(turning["direction_sign"].astype(int)) == {-1, 1}
    assert set(turning["planned_min_start_count"]) == {2000}
    assert set(turning["planned_target_start_count"]) == {5000}
    assert set(turning["pilot_trial_count"]) == {10}

    baseline = plan[plan["family_role"] == "baseline"]
    assert len(baseline) == 4
    assert set(baseline["family"]) == {"glide", "recovery"}
    assert baseline["target_heading_deg"].isna().all()
    assert set(baseline["direction_sign"].astype(int)) == {-1, 1}
    assert set(baseline["start_class"]) == {"all_start_classes_planned"}
    assert baseline["count_basis"].str.contains("blank target").all()
    assert set(baseline["planned_min_trial_count"]) == {5000}
    assert set(baseline["planned_target_trial_count"]) == {7500}


def test_sampling_summary_and_start_states_are_schema_safe_and_true_safe(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    summary = pd.read_csv(paths["sampling_strata_summary_csv"])
    starts = pd.read_csv(paths["start_state_manifest_csv"])

    assert list(summary.columns) == SAMPLING_STRATA_COLUMNS
    assert set(summary["start_class"]) == {"favourable", "mid_arena", "lift_sector", "random_stress"}
    assert set(summary["test_environment_mode"]) == {"W0_dry_air"}
    assert int(summary["pilot_sample_count"].sum()) == 680
    assert list(starts.columns) == START_STATE_COLUMNS
    assert len(starts) == 680
    assert starts["sample_id"].is_unique
    assert set(starts["test_environment_mode"]) == {"W0_dry_air"}
    assert set(starts["no_rollout_performed"]) == {True}
    assert set(starts["true_safe_start"]) == {True}
    assert set(starts["start_class"]) == {"favourable", "mid_arena", "lift_sector", "random_stress"}
    assert starts["x_w_m"].between(1.2, 6.6).all()
    assert starts["y_w_m"].between(0.0, 4.4).all()
    assert starts["z_w_m"].between(0.4, 3.5).all()
    assert set(starts["wing_exposure_bookkeeping_status"]) == {"geometry_only_w0_no_wind_query"}

    actual_fractions = starts["start_class"].value_counts(normalize=True).to_dict()
    requested = dict(zip(summary["start_class"], summary["fraction_requested"]))
    for start_class, requested_fraction in requested.items():
        assert abs(actual_fractions[start_class] - float(requested_fraction)) <= 0.051

    lift = starts[starts["start_class"] == "lift_sector"]
    edge_or_ring = lift["updraft_sector_label"].str.contains("edge|ring", regex=True)
    assert float(edge_or_ring.mean()) >= 0.50


def test_dry_run_inventory_schema_and_no_replay_status(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    starts = pd.read_csv(paths["start_state_manifest_csv"])
    inventory = pd.read_csv(paths["dry_run_candidate_inventory_csv"])

    assert list(inventory.columns) == DRY_RUN_COLUMNS
    assert len(inventory) == len(starts)
    assert inventory["candidate_id"].is_unique
    assert set(inventory["planned_replay_status"]) == {"not_replayed_in_this_task"}
    assert set(inventory["no_rollout_performed"]) == {True}
    assert set(inventory["test_environment_mode"]) == {"W0_dry_air"}
    assert inventory["notes"].str.contains("no primitive replay or rollout").all()


def test_generators_are_reproducible_byte_contract() -> None:
    config = DenseArchivePlanConfig()
    starts_a = build_start_state_manifest(config)
    starts_b = build_start_state_manifest(config)
    pd.testing.assert_frame_equal(starts_a, starts_b)

    inventory_a = build_dry_run_candidate_inventory(config, starts_a)
    inventory_b = build_dry_run_candidate_inventory(config, starts_b)
    pd.testing.assert_frame_equal(inventory_a, inventory_b)


def test_protected_stage0_hashes_are_unchanged(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, before, after = generated_outputs
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))

    assert before == after
    assert manifest["protected_hash_check_status"] == "unchanged"
    assert manifest["protected_hash_count_before"] == len(before)
    assert manifest["protected_hash_count_after"] == len(after)


def _protected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in PROTECTED_PATHS:
        assert root.exists(), root
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path.resolve().relative_to(REPO_ROOT.resolve()))] = sha256(
                    path.read_bytes()
                ).hexdigest()
    return hashes
