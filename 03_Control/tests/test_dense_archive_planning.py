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


TARGET_ENVIRONMENT_COLUMNS = [
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
]
SAMPLING_STRATA_COLUMNS = [
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
]
START_STATE_COLUMNS = [
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
    "true_safe_start",
    "start_generation_status",
    "layout_specific_sample_generated",
    "no_rollout_performed",
]
DRY_RUN_COLUMNS = [
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
]
MANIFEST_FIELDS = {
    "run_id",
    "campaign",
    "pass_name",
    "source_stage0_manifest",
    "source_run007_manifest",
    "stage0_gate_status_seen",
    "run007_preserved",
    "phase_b_task",
    "dense_archive_execution_performed",
    "paired_w0_w1_execution_performed",
    "full_w0_archive_performed",
    "full_w1_archive_performed",
    "fan_layouts",
    "layout_branch_ids",
    "test_environment_modes",
    "fan_branch_metadata",
    "target_ladder_deg",
    "direction_signs",
    "start_classes",
    "start_class_fractions",
    "dense_turning_families",
    "baseline_families",
    "environment_role_by_family",
    "branch_count_contract",
    "w1_floor_start_states_per_family_target_direction",
    "w1_target_start_states_per_family_target_direction",
    "w1_floor_turning_trials_per_branch",
    "w1_target_turning_trials_per_branch",
    "w1_floor_baseline_or_filler_trials_per_branch",
    "w1_target_baseline_or_filler_trials_per_branch",
    "w1_floor_total_trials_per_branch",
    "w1_target_total_trials_per_branch",
    "w1_floor_total_trials_all_branches",
    "w1_target_total_trials_all_branches",
    "w0_floor_total_trials_per_branch",
    "w0_target_total_trials_per_branch",
    "w0_floor_total_trials_all_branches",
    "w0_target_total_trials_all_branches",
    "combined_floor_total_trials_all_branches",
    "combined_target_total_trials_all_branches",
    "pilot_start_states_per_family_target_direction",
    "pilot_start_state_rows_per_branch",
    "pilot_start_state_rows_all_branches",
    "pilot_candidate_rows_all_branches",
    "paired_sample_key_scope",
    "no_cross_branch_decision_rule",
    "latency_metadata_only",
    "active_latency_implementation_deferred",
    "forbidden_claims",
    "recommended_next_step",
    "protected_paths_checked",
    "protected_hash_check_status",
    "output_files",
}
EXPECTED_MODES = {
    "W0_single_fan_branch",
    "W1_single_fan",
    "W0_four_fan_branch",
    "W1_four_fan",
}
PROTECTED_PATHS = [
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "002",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "003",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "004",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "005",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "006",
    REPO_ROOT / "03_Control" / "05_Results" / "09_primitive_library" / "000_frozen_baseline",
    REPO_ROOT / "03_Control" / "05_Results" / "10_dense_archive_planning" / "007",
]


@pytest.fixture(scope="module")
def generated_outputs() -> tuple[dict[str, Path], dict[str, str], dict[str, str]]:
    before = _protected_hashes()
    paths = run_dense_archive_planning(
        run_id=8,
        overwrite=True,
        pilot_start_states_per_family_target_direction=10,
    )
    after = _protected_hashes()
    return paths, before, after


def test_manifest_counts_stage0_and_run007_preservation(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    for key, path in paths.items():
        assert path.exists(), key
        if key != "root":
            assert "10_dense_archive_planning" in path.parts
            assert "008" in path.parts

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert MANIFEST_FIELDS.issubset(manifest)
    assert manifest["run_id"] == 8
    assert manifest["pass_name"] == "phase_b_task1_1_equal_fan_branch_paired_w0_w1_planning_scaffold"
    assert manifest["stage0_gate_status_seen"] == "passed"
    assert manifest["run007_preserved"] is True
    assert manifest["protected_hash_check_status"] == "unchanged"
    assert manifest["fan_layouts"] == ["single_fan", "four_fan"]
    assert set(manifest["test_environment_modes"]) == EXPECTED_MODES
    assert manifest["w1_floor_turning_trials_per_branch"] == 320000
    assert manifest["w1_target_turning_trials_per_branch"] == 480000
    assert manifest["w1_floor_baseline_or_filler_trials_per_branch"] == 30000
    assert manifest["w1_target_baseline_or_filler_trials_per_branch"] == 20000
    assert manifest["w1_floor_total_trials_per_branch"] == 350000
    assert manifest["w1_target_total_trials_per_branch"] == 500000
    assert manifest["w1_floor_total_trials_all_branches"] == 700000
    assert manifest["w1_target_total_trials_all_branches"] == 1000000
    assert manifest["w0_floor_total_trials_per_branch"] == 150000
    assert manifest["w0_target_total_trials_per_branch"] == 300000
    assert manifest["w0_floor_total_trials_all_branches"] == 300000
    assert manifest["w0_target_total_trials_all_branches"] == 600000
    assert manifest["combined_floor_total_trials_all_branches"] == 1000000
    assert manifest["combined_target_total_trials_all_branches"] == 1600000
    assert manifest["pilot_start_state_rows_per_branch"] == 680
    assert manifest["pilot_start_state_rows_all_branches"] == 1360
    assert manifest["pilot_candidate_rows_all_branches"] == 2720
    assert manifest["latency_metadata_only"] is True
    assert manifest["active_latency_implementation_deferred"] is True


def test_target_environment_plan_schema_counts_and_branch_rules(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    plan = pd.read_csv(paths["target_environment_plan_csv"])

    assert list(plan.columns) == TARGET_ENVIRONMENT_COLUMNS
    assert len(plan) == 272
    assert set(plan["fan_layout"]) == {"single_fan", "four_fan"}
    assert set(plan["test_environment_mode"]) == EXPECTED_MODES
    assert "W0_dry_air" not in set(plan["test_environment_mode"])
    assert "W1_nominal_updraft" not in set(plan["test_environment_mode"])
    assert set(plan["latency_case_planned"]) == {"none"}
    assert set(plan["latency_model_status"]) == {"metadata_only_active_latency_deferred"}
    assert set(plan["no_rollout_performed"]) == {True}
    for column in (
        "no_cross_branch_promotion",
        "no_cross_branch_rejection",
        "no_cross_branch_cluster_merge",
        "no_cross_branch_safety_justification",
    ):
        assert set(plan[column]) == {True}

    for fan_layout in ("single_fan", "four_fan"):
        branch = plan[plan["fan_layout"] == fan_layout]
        assert len(branch) == 136
        w0 = branch[branch["test_environment_mode"].str.startswith("W0_")]
        w1 = branch[branch["test_environment_mode"].str.startswith("W1_")]
        assert len(w0) == 68
        assert len(w1) == 68
        assert int(w0["planned_floor_trial_count"].sum()) == 150000
        assert int(w0["planned_target_trial_count"].sum()) == 300000
        assert int(w1["planned_floor_trial_count"].sum()) == 350000
        assert int(w1["planned_target_trial_count"].sum()) == 500000

    assisted_w0 = plan[
        plan["test_environment_mode"].str.startswith("W0_")
        & (plan["environment_role"] == "updraft_assisted")
    ]
    assisted_w1 = plan[
        plan["test_environment_mode"].str.startswith("W1_")
        & (plan["environment_role"] == "updraft_assisted")
    ]
    assert set(assisted_w0["validity_gate_role"]) == {"ablation_only"}
    assert set(assisted_w0["w0_failure_policy"]) == {"log_ablation_do_not_reject_if_w1_valid"}
    assert set(assisted_w1["validity_gate_role"]) == {"first_validity_gate"}

    baseline = plan[plan["family_role"] == "baseline"]
    assert len(baseline) == 16
    assert baseline["target_heading_deg"].isna().all()
    assert baseline["count_basis"].str.contains("blank target").all()
    assert set(baseline["direction_sign"].astype(int)) == {-1, 1}


def test_sampling_summary_and_start_states_are_branch_local_and_true_safe(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    summary = pd.read_csv(paths["sampling_strata_summary_csv"])
    starts = pd.read_csv(paths["start_state_manifest_csv"])

    assert list(summary.columns) == SAMPLING_STRATA_COLUMNS
    assert len(summary) == 8
    assert set(summary["fan_layout"]) == {"single_fan", "four_fan"}
    assert set(summary["start_class"]) == {"favourable", "mid_arena", "lift_sector", "random_stress"}
    assert set(summary["layout_specific_sampling_required"]) == {True}
    assert set(summary["no_cross_branch_merge"]) == {True}
    assert summary[summary["start_class"] == "lift_sector"]["special_rule"].str.contains("edge/ring").all()

    assert list(starts.columns) == START_STATE_COLUMNS
    assert len(starts) == 1360
    assert starts["sample_id"].is_unique
    assert starts["paired_sample_key"].is_unique
    assert set(starts["fan_layout"]) == {"single_fan", "four_fan"}
    assert set(starts["layout_specific_sample_generated"]) == {True}
    assert set(starts["no_rollout_performed"]) == {True}
    assert set(starts["true_safe_start"]) == {True}
    assert starts["x_w_m"].between(1.2, 6.6).all()
    assert starts["y_w_m"].between(0.0, 4.4).all()
    assert starts["z_w_m"].between(0.4, 3.5).all()
    assert set(starts["wing_exposure_bookkeeping_status"]) == {"branch_layout_geometry_only_no_wind_query"}

    for fan_layout in ("single_fan", "four_fan"):
        branch = starts[starts["fan_layout"] == fan_layout]
        assert len(branch) == 680
        actual_fractions = branch["start_class"].value_counts(normalize=True).to_dict()
        for start_class, expected in {"favourable": 0.30, "mid_arena": 0.20, "lift_sector": 0.30, "random_stress": 0.20}.items():
            assert abs(actual_fractions[start_class] - expected) <= 1e-12
        lift = branch[branch["start_class"] == "lift_sector"]
        edge_or_ring = lift["updraft_sector_label"].str.contains("edge|ring", regex=True)
        assert float(edge_or_ring.mean()) >= 0.50


def test_dry_run_inventory_pairing_and_latency_metadata(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, _, _ = generated_outputs
    starts = pd.read_csv(paths["start_state_manifest_csv"])
    inventory = pd.read_csv(paths["dry_run_candidate_inventory_csv"])

    assert list(inventory.columns) == DRY_RUN_COLUMNS
    assert len(inventory) == 2720
    assert inventory["candidate_id"].is_unique
    assert set(inventory["planned_replay_status"]) == {"not_replayed_in_this_task"}
    assert set(inventory["latency_case_planned"]) == {"none"}
    assert set(inventory["latency_model_status"]) == {"metadata_only_active_latency_deferred"}
    assert set(inventory["no_rollout_performed"]) == {True}
    assert set(inventory["test_environment_mode"]) == EXPECTED_MODES
    assert set(inventory["paired_sample_key"]) == set(starts["paired_sample_key"])

    grouped = inventory.groupby("paired_sample_key")
    assert grouped.size().eq(2).all()
    for paired_key, group in grouped:
        fan_layouts = set(group["fan_layout"])
        modes = set(group["test_environment_mode"])
        assert len(fan_layouts) == 1, paired_key
        fan_layout = next(iter(fan_layouts))
        expected_modes = (
            {"W0_single_fan_branch", "W1_single_fan"}
            if fan_layout == "single_fan"
            else {"W0_four_fan_branch", "W1_four_fan"}
        )
        assert modes == expected_modes

    assisted_w0 = inventory[
        inventory["test_environment_mode"].str.startswith("W0_")
        & (inventory["environment_role"] == "updraft_assisted")
    ]
    assisted_w1 = inventory[
        inventory["test_environment_mode"].str.startswith("W1_")
        & (inventory["environment_role"] == "updraft_assisted")
    ]
    assert set(assisted_w0["acceptance_interpretation"]) == {"ablation_only_not_rejection"}
    assert set(assisted_w1["acceptance_interpretation"]) == {"first_validity_gate"}


def test_generators_are_reproducible_byte_contract() -> None:
    config = DenseArchivePlanConfig()
    starts_a = build_start_state_manifest(config)
    starts_b = build_start_state_manifest(config)
    pd.testing.assert_frame_equal(starts_a, starts_b)

    inventory_a = build_dry_run_candidate_inventory(config, starts_a)
    inventory_b = build_dry_run_candidate_inventory(config, starts_b)
    pd.testing.assert_frame_equal(inventory_a, inventory_b)


def test_protected_hashes_include_run007_and_remain_unchanged(
    generated_outputs: tuple[dict[str, Path], dict[str, str], dict[str, str]],
) -> None:
    paths, before, after = generated_outputs
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))

    assert before == after
    assert manifest["protected_hash_check_status"] == "unchanged"
    assert manifest["protected_hash_count_before"] == len(before)
    assert manifest["protected_hash_count_after"] == len(after)
    assert any("10_dense_archive_planning/007" in key.replace("\\", "/") for key in before)


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
