from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from run_changed_case_validation import (
    R10_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R10_EXPECTED_HISTORY_LAUNCHES,
    R10_PROTOCOL,
    R11_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R11_EXPECTED_HISTORY_LAUNCHES,
    R11_PROTOCOL,
)
from prim_cat import ACTIVE_PRIMITIVE_IDS, LAUNCH_CAPTURE_PRIMITIVE_IDS
from primitive_variant_registry import ENTRY_ROLE_BY_PRIMITIVE_ID, start_family_is_compatible
from run_lqr_w01_dense_chunked import (
    L6_RICH_SIDE_CANDIDATE_COUNT,
    L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    L6_RICH_SIDE_ROW_COUNT,
    OFFICIAL_W01_ENVIRONMENT_CASES,
    R5_ACTIVE_FAN_COUNT_SEQUENCE,
    R5_EVIDENCE_BLOCKS,
    R5_EVIDENCE_BLOCK_IDS,
    W01DenseRunConfig,
    _row_schedule_for_index as _r5_row_schedule_for_index,
    rich_side_dense_row_count,
)
from run_post_w3_library_size_study import _coverage_medoid_selection, _profile_subset, _representative_score
from run_r5_r10_pipeline import ARCHIVED_STAGES, STAGE_ORDER
from run_v53_algorithm_contract_audit import AlgorithmContractAuditConfig, run_v53_algorithm_contract_audit
from run_repeated_launch_learning_curve import (
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    EMPTY_FROZEN_PRIOR_BASELINE_ID,
    GOVERNOR_CALIBRATION_SEARCH_POLICY,
    NO_UPDRAFT_CHANGED_CASE_BLOCK_ID,
    LIBRARY_SIZE_CASE_IDS,
    ONLINE_MEMORY_SCOPE,
    OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION,
    OPEN_LOOP_COMPARISON_POLICY_ID,
    POLICY_HISTORY_CONDITIONS,
    R10_DEPLOYABLE_CLAIM_LIBRARY_CASE_IDS,
    R11_POLICY_HISTORY_CONDITIONS,
    R10_GLOBAL_CALIBRATION_SCOPE,
    R9_POLICY_HISTORY_CONDITIONS,
    R9_BLOCKS,
    R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R9_EXPECTED_HISTORY_LAUNCHES,
    R9_OUTER_CASES_PER_CONDITION,
    R9_PROTOCOL,
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    _fan_position_policy_for_outer_case,
    _context_payload,
    _episode_row_from_sequence,
    _history_row_for_final,
    _launch_score_fields,
    _launch_score_fields_for_role,
    _outer_case_schedule,
    _pairing_audit_rows,
    _pass_fail_summary,
    _posthoc_executed_score_table,
    _posthoc_final_score_table,
    _posthoc_score_delta_table,
    _policy_condition,
    _real_time_scheduler_decision_fields,
    _scheduled_active_fan_count_for_outer_case,
    _selected_set,
    _tuned_governor_config_from_metrics,
    _uses_full_w3_randomisation_block,
    _with_selection_change_flags,
    REAL_TIME_HARD_DECISION_BUDGET_S,
    REAL_TIME_OUTER_LOOP_SCHEDULER_VERSION,
    REAL_TIME_PREFERRED_DECISION_BUDGET_S,
    R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M,
    R11_L0_DRY_AIR_FIXED_BLOCK_ID,
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    R11_GOVERNOR_HANDOFF_SCOPE,
    validation_route_for_primitive_step,
)
from state_contract import STATE_INDEX
from run_w3_survival import R5_INPUT_KIND, R7_EVIDENCE_BLOCK_IDS, W3_ACTIVE_FAN_COUNT_SEQUENCE, W3_ENVIRONMENT_CASES
from episode_selector import select_compact_representative
from episode_selector import selector_decision_row
from implementation_instance import IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
from plant_instance import (
    AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
)
from viability_governor import (
    CALIBRATED_REGIME_POST_STALL_ALPHA_DEG,
    CALIBRATED_REGIME_SOURCE_CALIBRATION_ID,
    CALIBRATED_REGIME_TRANSITION_START_ALPHA_DEG,
    DEFAULT_GOVERNOR_CONFIG,
    REJECTION_REASONS,
    governor_candidate_row,
    governor_config_from_row,
    governor_score,
)


def _v53_timing_payload() -> dict[str, object]:
    return {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }


def _v53_reference_state_vector(alpha_deg: float, *, u_m_s: float = 5.0) -> list[float]:
    state = [0.0] * (max(STATE_INDEX.values()) + 1)
    state[STATE_INDEX["u"]] = float(u_m_s)
    state[STATE_INDEX["w"]] = float(u_m_s) * float(np.tan(np.deg2rad(alpha_deg)))
    return state


class _V53BeliefStub:
    update_count = 0


def _v53_successful_primitive_row(variant_id: str = "memory_variant") -> dict[str, object]:
    return {
        "primitive_variant_id": variant_id,
        "primitive_id": "glide",
        "controller_id": f"ctrl_{variant_id}",
        "selected_entry_role": "transition_object",
        "transition_entry_class": "launch_gate",
        "start_state_family": "launch_gate",
        "route_required_entry_role": "transition_object",
        "route_required_entry_class": "launch_gate",
        "route_reason": "state_class_transition_entry",
        "continuation_valid": True,
        "episode_terminal_useful": False,
        "failure_label": "success",
        "boundary_use_class": "continuation",
        "lift_dwell_time_s": 0.0,
        "energy_residual_m": 0.0,
        "rollout_duration_s": 0.10,
        "minimum_wall_margin_m": 0.5,
    }


def test_v53_stage_contract_is_r5_r7_r8_r10_r11_with_r6_archived_and_r9_internal() -> None:
    assert STAGE_ORDER == ("R5", "R7", "R8", "R10", "R11")
    assert ARCHIVED_STAGES == ("R6",)

    assert R9_PROTOCOL.validation_evidence_level == "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence"
    assert R9_PROTOCOL.gate_profile == "internal_reduced_fixed_case_preflight_for_r10_initialisation"
    assert R10_PROTOCOL.validation_evidence_level == "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate"
    assert R10_PROTOCOL.gate_profile == "relaxed_changed_case_viability_governor_learning_not_final_validation"
    assert R10_PROTOCOL.max_hard_failure_rate == pytest.approx(0.20)
    assert R10_PROTOCOL.max_floor_or_ceiling_violation_rate == pytest.approx(0.20)
    assert R10_DEPLOYABLE_CLAIM_LIBRARY_CASE_IDS == ("heavy_cluster", "balanced_cluster", "light_cluster")
    assert R11_PROTOCOL.validation_evidence_level == "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation"
    assert R11_PROTOCOL.gate_profile == "strict_final_heldout_validation"
    assert R11_PROTOCOL.max_floor_or_ceiling_violation_rate == pytest.approx(0.0)
    assert R11_PROTOCOL.min_full_safe_success_rate == pytest.approx(0.99)


def test_v53_r9_is_reduced_internal_preflight_and_can_seed_r10_governor() -> None:
    assert tuple(block.case_count for block in R9_BLOCKS) == (1, 1, 1)
    assert R9_OUTER_CASES_PER_CONDITION == 3
    assert R9_POLICY_HISTORY_CONDITIONS == (
        "no_memory_baseline",
        "spatial_flow_belief_memory_h3",
        "spatial_flow_belief_memory_h10",
        "spatial_flow_belief_memory_h30",
    )
    assert R9_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(R9_POLICY_HISTORY_CONDITIONS) * 3 == 60
    assert R9_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * 3 * (3 + 10 + 30) == 645

    tuned, decisions = _tuned_governor_config_from_metrics(
        base_config=DEFAULT_GOVERNOR_CONFIG,
        metrics={
            "hard_failure_rate": 0.50,
            "no_viable_primitive_rate": 0.0,
            "safe_success_rate": 0.0,
            "terminal_or_lift_capture_rate": 0.0,
        },
        protocol=R9_PROTOCOL,
    )

    assert tuned.config_id == "v53_r9_tuned_mission_governor"
    assert tuned.minimum_wall_margin_m == DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m
    assert tuned.maximum_hard_failure_risk < DEFAULT_GOVERNOR_CONFIG.maximum_hard_failure_risk
    assert tuned.exploration_bonus_weight < DEFAULT_GOVERNOR_CONFIG.exploration_bonus_weight
    assert tuned.memory_switch_min_confidence > DEFAULT_GOVERNOR_CONFIG.memory_switch_min_confidence
    assert tuned.candidate_path_memory_residual_cap_m < DEFAULT_GOVERNOR_CONFIG.candidate_path_memory_residual_cap_m
    assert {row["parameter"] for row in decisions} >= {
        "maximum_hard_failure_risk",
        "exploration_bonus_weight",
        "memory_switch_min_confidence",
        "candidate_path_memory_residual_cap_m",
    }


def test_v53_r10_tuning_can_relax_memory_shield_from_selector_opportunity_evidence() -> None:
    tuned, decisions = _tuned_governor_config_from_metrics(
        base_config=DEFAULT_GOVERNOR_CONFIG,
        metrics={
            "hard_failure_rate": 0.0,
            "no_viable_primitive_rate": 0.0,
            "safe_success_rate": 0.0,
            "mission_success_rate": 0.0,
            "wrong_wall_exit_rate": 0.2,
            "terminal_or_lift_capture_rate": 0.0,
            "memory_switch_opportunity_count": 12,
            "memory_switch_acceptance_rate": 0.0,
            "mean_candidate_path_confidence": 0.20,
            "max_memory_correction_delta": 0.0,
        },
        protocol=R10_PROTOCOL,
    )

    assert tuned.config_id == "v53_r10_tuned_mission_governor"
    assert tuned.memory_switch_min_confidence < DEFAULT_GOVERNOR_CONFIG.memory_switch_min_confidence
    assert tuned.memory_switch_min_score_margin < DEFAULT_GOVERNOR_CONFIG.memory_switch_min_score_margin
    assert tuned.memory_switch_max_base_score_drop > DEFAULT_GOVERNOR_CONFIG.memory_switch_max_base_score_drop
    assert tuned.belief_weight > DEFAULT_GOVERNOR_CONFIG.belief_weight
    assert tuned.memory_objective_min_confidence < DEFAULT_GOVERNOR_CONFIG.memory_objective_min_confidence
    assert tuned.memory_objective_max_base_score_drop > DEFAULT_GOVERNOR_CONFIG.memory_objective_max_base_score_drop
    assert tuned.memory_objective_score_cap > DEFAULT_GOVERNOR_CONFIG.memory_objective_score_cap
    assert tuned.mission_wrong_boundary_penalty_weight > DEFAULT_GOVERNOR_CONFIG.mission_wrong_boundary_penalty_weight
    assert {row["parameter"] for row in decisions} >= {
        "memory_switch_min_confidence",
        "memory_switch_min_score_margin",
        "memory_switch_max_base_score_drop",
        "memory_objective_min_confidence",
        "memory_objective_max_base_score_drop",
        "memory_objective_score_cap",
        "mission_wrong_boundary_penalty_weight",
    }


def test_v53_repeated_launch_outer_loop_has_realtime_scheduler_profile_contract() -> None:
    assert REAL_TIME_OUTER_LOOP_SCHEDULER_VERSION == "predictive_next_primitive_scheduler_profile_v1"
    assert REAL_TIME_PREFERRED_DECISION_BUDGET_S == pytest.approx(0.020)
    assert REAL_TIME_HARD_DECISION_BUDGET_S == pytest.approx(0.100)

    fast = _real_time_scheduler_decision_fields(
        primitive_step_index=1,
        scheduler_decision_source="prepared_during_previous_primitive_window",
        context_build_duration_s=0.002,
        belief_query_duration_s=0.001,
        selection_duration_s=0.003,
        candidate_count=24,
        viable_count=6,
    )
    assert fast["scheduler_prepared_before_primitive_boundary"] is True
    assert fast["preferred_20ms_slot_met"] is True
    assert fast["hard_100ms_boundary_met"] is True
    assert fast["decision_candidate_count"] == 24
    assert fast["decision_viable_count"] == 6
    assert fast["decision_controller_compute_duration_s"] == pytest.approx(0.006)
    assert fast["decision_diagnostic_logging_duration_s"] == pytest.approx(0.0)
    assert (
        fast["decision_controller_timing_scope"]
        == "context_plus_belief_plus_full_memory_controller_selector_no_table_flush"
    )
    assert fast["real_time_claim_status"] == "controller_compute_profile_excludes_table_flush_and_posthoc_diagnostics"

    slow = _real_time_scheduler_decision_fields(
        primitive_step_index=2,
        scheduler_decision_source="boundary_compute_no_prepared_decision",
        context_build_duration_s=0.040,
        belief_query_duration_s=0.020,
        selection_duration_s=0.050,
        candidate_count=24,
        viable_count=6,
    )
    assert slow["scheduler_prepared_before_primitive_boundary"] is False
    assert slow["preferred_20ms_slot_met"] is False
    assert slow["hard_100ms_boundary_met"] is False


def test_v53_r5_dense_schedule_is_transition_entry_separated_and_uses_current_randomisation_cases() -> None:
    assert len(ACTIVE_PRIMITIVE_IDS) == 8
    assert len(LAUNCH_CAPTURE_PRIMITIVE_IDS) == 6
    assert not set(LAUNCH_CAPTURE_PRIMITIVE_IDS).intersection(set(ACTIVE_PRIMITIVE_IDS))
    assert rich_side_dense_row_count() == 102400
    assert L6_RICH_SIDE_ROW_COUNT == 102400
    assert L6_RICH_SIDE_CANDIDATE_COUNT == 32
    assert L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE == 50
    assert len(R5_EVIDENCE_BLOCKS) == 8
    assert len(R5_EVIDENCE_BLOCK_IDS) == 8
    assert OFFICIAL_W01_ENVIRONMENT_CASES == (
        ("W0", "dry_air"),
        ("W1", "w1_annular_gp_randomised_single"),
        ("W1", "w1_annular_gp_randomised_four"),
    )
    assert R5_ACTIVE_FAN_COUNT_SEQUENCE == (0, 1, 2, 3, 4)

    config = W01DenseRunConfig(
        run_id=0,
        rows=L6_RICH_SIDE_ROW_COUNT,
        candidate_count=L6_RICH_SIDE_CANDIDATE_COUNT,
        paired_tests_per_candidate=L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    )
    family_counts: dict[str, int] = {}
    environment_counts: dict[tuple[str, str], int] = {}
    evidence_block_counts: dict[str, int] = {}
    for row_index in range(L6_RICH_SIDE_ROW_COUNT):
        schedule = _r5_row_schedule_for_index(row_index, config)
        role = ENTRY_ROLE_BY_PRIMITIVE_ID[schedule.primitive_id]
        assert start_family_is_compatible(entry_role=role, start_state_family=schedule.start_state_family)
        family_counts[schedule.start_state_family] = family_counts.get(schedule.start_state_family, 0) + 1
        key = (schedule.W_layer, schedule.environment_mode)
        environment_counts[key] = environment_counts.get(key, 0) + 1
        evidence_block_counts[schedule.evidence_block_id] = evidence_block_counts.get(schedule.evidence_block_id, 0) + 1

    assert family_counts == {
        "launch_gate": 40960,
        "inflight_nominal": 25600,
        "inflight_lift_region": 15360,
        "inflight_boundary_near": 10240,
        "inflight_recovery_edge": 10240,
    }
    assert environment_counts == {
        ("W0", "dry_air"): 12800,
        ("W1", "w1_annular_gp_randomised_single"): 25600,
        ("W1", "w1_annular_gp_randomised_four"): 64000,
    }
    assert set(evidence_block_counts) == set(R5_EVIDENCE_BLOCK_IDS)
    assert set(evidence_block_counts.values()) == {12800}


def test_v53_r7_and_r8_contracts_are_direct_holdout_and_updraft_scored() -> None:
    assert R5_INPUT_KIND == "r5_frozen_bundle_direct"
    assert W3_ENVIRONMENT_CASES == ("dry_air", "w3_randomised_single", "w3_randomised_four")
    assert W3_ACTIVE_FAN_COUNT_SEQUENCE == (0, 1, 2, 3, 4)
    assert len(R7_EVIDENCE_BLOCK_IDS) == 8

    frame = pd.DataFrame(
        [
            {
                "continuation_valid_rate": 0.5,
                "episode_terminal_useful_rate": 0.0,
                "hard_failure_rate": 0.0,
                "energy_residual_mean_m": 100.0,
                "updraft_gain_proxy_mean_m": 0.0,
                "positive_specific_energy_gain_mean_m": 0.0,
                "lift_dwell_mean_s": 0.0,
            },
            {
                "continuation_valid_rate": 0.5,
                "episode_terminal_useful_rate": 0.0,
                "hard_failure_rate": 0.0,
                "energy_residual_mean_m": -100.0,
                "updraft_gain_proxy_mean_m": 1.0,
                "positive_specific_energy_gain_mean_m": 1.0,
                "lift_dwell_mean_s": 0.0,
            },
        ]
    )
    low_updraft_high_net_energy, high_updraft_low_net_energy = list(_representative_score(frame))
    assert high_updraft_low_net_energy > low_updraft_high_net_energy

    coverage_frame = pd.DataFrame(
        [
            {
                "r7_evidence_block_id": "r7_random_arena_wide",
                "r7_uncertainty_tier": "arena_wide_full_randomisation",
                "r7_active_fan_count_policy": "balanced_0_1_2_3_4_with_arena_wide_fan_position_generalisation",
                "r7_fan_position_policy": "independent_uniform_xy_bounds",
                "transition_pair": "launch_gate_to_inflight_stable",
                "transition_exit_class": "continuation_valid",
            },
            {
                "r7_evidence_block_id": "r7_anchor_dry_air",
                "r7_uncertainty_tier": "anchor_dry_air",
                "r7_active_fan_count_policy": "fixed_zero_active_fans",
                "r7_fan_position_policy": "no_fan_positions",
                "transition_pair": "launch_gate_to_hard_failure",
                "transition_exit_class": "hard_failure",
            },
        ]
    )
    assert len(_profile_subset(coverage_frame, "r7_block:r7_random_arena_wide")) == 1
    assert len(_profile_subset(coverage_frame, "tier:arena_wide_full_randomisation")) == 1
    assert len(_profile_subset(coverage_frame, "transition:launch_gate_to_inflight_stable")) == 1
    assert len(_profile_subset(coverage_frame, "exit:hard_failure")) == 1


def test_v53_r8_coverage_medoid_prefers_worst_case_coverage_over_average_rank() -> None:
    frame = pd.DataFrame(
        [
            {
                "primitive_variant_id": "average_strong_but_rare_case_gap",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 0.95,
                "episode_terminal_useful_rate": 0.3,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:single","env:four","active_fan_count:1","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[1.0,1.0,1.0,0.0]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "broad_case_existing_medoid",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 0.75,
                "episode_terminal_useful_rate": 0.1,
                "hard_failure_rate": 0.02,
                "robustness_coverage_labels_json": '["env:single","env:four","active_fan_count:1","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[0.70,0.70,0.70,0.70]",
                "Q_weight_json": '{"q":1.1}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "unsafe_high_coverage_not_allowed_to_dominate",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 1.0,
                "episode_terminal_useful_rate": 0.5,
                "hard_failure_rate": 0.90,
                "robustness_coverage_labels_json": '["env:single","env:four","active_fan_count:1","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[1.0,1.0,1.0,1.0]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
        ]
    )

    selected = _coverage_medoid_selection(frame, max_representatives=1, case_id="heavy_cluster")

    assert list(selected["primitive_variant_id"]) == ["broad_case_existing_medoid"]
    assert selected["_medoid_selection_reason"].iloc[0] == "best_worst_case_coverage_medoid"


def test_v53_memory_scope_is_per_final_case_and_final_launches_are_paired() -> None:
    outer_cases = _outer_case_schedule(protocol=R9_PROTOCOL, seed=90, smoke_outer_cases_per_block=1)
    final_schedule = []
    for case_id in LIBRARY_SIZE_CASE_IDS:
        for policy_id in R9_POLICY_HISTORY_CONDITIONS:
            for outer in outer_cases:
                final_schedule.append(
                    {
                        **outer,
                            "launch_role": "final_heldout",
                            "episode_id": f"{case_id}_{policy_id}_{outer['outer_case_index']}",
                            "library_size_case_id": case_id,
                        "policy_id": policy_id,
                        "history_length": int(_policy_condition(policy_id)["history_length"]),
                    }
                )

    pairing_rows = _pairing_audit_rows(final_schedule)
    assert pairing_rows
    assert all(row["pairing_passed"] for row in pairing_rows)
    assert all(row["library_case_count"] == len(LIBRARY_SIZE_CASE_IDS) for row in pairing_rows)
    assert all(row["policy_count"] == len(R9_POLICY_HISTORY_CONDITIONS) for row in pairing_rows)

    memory_final = next(row for row in final_schedule if row["policy_id"] == "spatial_flow_belief_memory_h30")
    history = _history_row_for_final(memory_final, 0)
    assert history["library_size_case_id"] == memory_final["library_size_case_id"]
    assert history["policy_id"] == memory_final["policy_id"]
    assert history["history_length"] == memory_final["history_length"]
    assert history["common_final_launch_key"] == memory_final["common_final_launch_key"]
    assert history["launch_state_seed"] != memory_final["launch_state_seed"]
    assert history["environment_seed"] != memory_final["environment_seed"]

    assert _policy_condition("no_memory_baseline")["uses_memory"] is False
    assert _policy_condition(OPEN_LOOP_COMPARISON_POLICY_ID)["open_loop"] is True
    assert _policy_condition(OPEN_LOOP_COMPARISON_POLICY_ID)["comparison_only"] is True
    assert _policy_condition(EMPTY_FROZEN_PRIOR_BASELINE_ID)["uses_memory"] is True
    assert _policy_condition(EMPTY_FROZEN_PRIOR_BASELINE_ID)["updates_memory"] is False
    assert OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION == (
        "case_local_online_memory_plus_r10_global_deterministic_calibration_v1"
    )
    assert ONLINE_MEMORY_SCOPE == "case_local_reset_per_final_schedule_row"
    assert R10_GLOBAL_CALIBRATION_SCOPE == "aggregate_all_r10_final_heldout_rows_and_selector_opportunity_diagnostics"
    assert R11_GOVERNOR_HANDOFF_SCOPE == "single_frozen_r10_governor_config_used_for_r11_validation"
    assert GOVERNOR_CALIBRATION_SEARCH_POLICY == "deterministic_bounded_rule_update_no_profile_ladder_no_black_box_search"


def test_v53_governor_has_no_speed_boundary_and_wall_guard_is_0p10cm() -> None:
    assert DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m == pytest.approx(0.001)
    assert not any("speed" in reason for reason in REJECTION_REASONS)
    assert not any("speed" in key for key in asdict(DEFAULT_GOVERNOR_CONFIG))
    parsed = governor_config_from_row(
        {
            "config_id": "r10_handoff",
            "memory_switch_min_confidence": "0.35",
            "exploration_switch_allow_cross_family": "false",
            "calibrated_regime_mismatch_risk_weight": "0.14",
            "memory_switch_max_calibrated_regime_risk_increase": "0.01",
        }
    )
    assert parsed.memory_switch_min_confidence == pytest.approx(0.35)
    assert parsed.exploration_switch_allow_cross_family is False
    assert parsed.calibrated_regime_mismatch_risk_weight == pytest.approx(0.14)
    assert parsed.memory_switch_max_calibrated_regime_risk_increase == pytest.approx(0.01)

    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    row = governor_candidate_row(
        representative={
            "compact_library_id": "launch",
            "primitive_variant_id": "launch",
            "primitive_id": "glide",
            "entry_role": "transition_object",
            "transition_entry_class": "launch_gate",
            "controller_id": "ctrl_launch",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            **timing_payload,
        },
        outcome={"continuation_probability": 0.8, "transition_success_probability": 0.8, "transition_exit_classes_seen": "post_launch_degraded", "terminal_useful_probability": 0.1, "hard_failure_risk": 0.1},
        context={
            "context_id": "very_low_speed_context",
            "start_state_family": "launch_gate",
            "governor_wall_margin_m": 0.001,
            "wall_margin_m": 0.001,
            "floor_margin_m": 1.0,
            "ceiling_margin_m": 1.0,
            "latency_case": "nominal",
            "speed_margin_m_s": -100.0,
            "minimum_speed_m_s": 0.0,
        },
        governor_mode="continuation_mode",
    )
    assert row["viable"] is True
    assert row["rejection_reason"] == ""


def test_v53_calibrated_regime_risk_score_is_backward_compatible_and_bounded() -> None:
    base_kwargs = {
        "governor_mode": "continuation_mode",
        "continuation_probability": 0.7,
        "terminal_useful_probability": 0.1,
        "hard_failure_risk": 0.05,
        "expected_updraft_gain_proxy_m": 0.0,
        "expected_lift_dwell_time_s": 0.0,
        "wall_margin_m": 0.5,
    }
    nominal = governor_score(**base_kwargs)
    zero_risk = governor_score(**base_kwargs, calibrated_regime_mismatch_risk=0.0)
    full_risk = governor_score(**base_kwargs, calibrated_regime_mismatch_risk=1.0)
    assert zero_risk == pytest.approx(nominal)
    assert full_risk == pytest.approx(nominal - DEFAULT_GOVERNOR_CONFIG.calibrated_regime_mismatch_risk_weight)

    capped_cfg = governor_config_from_row(
        {
            "config_id": "capped_regime_risk_test",
            "calibrated_regime_mismatch_risk_weight": 0.50,
            "calibrated_regime_mismatch_score_cap": 0.18,
        }
    )
    capped_nominal = governor_score(**base_kwargs, calibrated_regime_mismatch_risk=0.0, governor_config=capped_cfg)
    capped_full = governor_score(**base_kwargs, calibrated_regime_mismatch_risk=1.0, governor_config=capped_cfg)
    assert capped_full == pytest.approx(capped_nominal - 0.18)


def test_v53_candidate_rows_classify_calibrated_regime_from_reference_alpha() -> None:
    assert CALIBRATED_REGIME_TRANSITION_START_ALPHA_DEG == pytest.approx(14.0)
    assert CALIBRATED_REGIME_POST_STALL_ALPHA_DEG == pytest.approx(18.0)
    assert "n30_joint_pareto_040_local_s5_yaw0p75_clr0p60" in CALIBRATED_REGIME_SOURCE_CALIBRATION_ID
    timing_payload = _v53_timing_payload()
    base_representative = {
        "compact_library_id": "launch",
        "primitive_variant_id": "candidate",
        "primitive_id": "glide",
        "entry_role": "transition_object",
        "transition_entry_class": "launch_gate",
        "controller_id": "ctrl_candidate",
        "K_gain_checksum": "k",
        "augmented_A_checksum": "a",
        "augmented_B_checksum": "b",
        "augmented_gain_checksum": "g",
        **timing_payload,
    }
    outcome = {
        "continuation_probability": 0.8,
        "transition_success_probability": 0.8,
        "transition_exit_classes_seen": "post_launch_degraded",
        "terminal_useful_probability": 0.1,
        "hard_failure_risk": 0.1,
    }
    context = {
        "context_id": "calibrated_regime_classification",
        "start_state_family": "launch_gate",
        "governor_wall_margin_m": 0.5,
        "wall_margin_m": 0.5,
        "floor_margin_m": 1.0,
        "ceiling_margin_m": 1.0,
        "latency_case": "nominal",
    }

    def row(alpha_deg: float) -> dict[str, object]:
        return governor_candidate_row(
            representative={**base_representative, "reference_state_vector": _v53_reference_state_vector(alpha_deg)},
            outcome=outcome,
            context=context,
            governor_mode="continuation_mode",
        )

    normal = row(13.5)
    transition = row(16.0)
    post_stall = row(18.0)
    assert normal["calibrated_regime_label"] == "normal"
    assert normal["calibrated_regime_mismatch_risk"] == pytest.approx(0.0)
    assert normal["calibrated_regime_transition_start_alpha_deg"] == pytest.approx(14.0)
    assert normal["calibrated_regime_post_stall_alpha_deg"] == pytest.approx(18.0)
    assert normal["calibrated_regime_source_calibration_id"] == CALIBRATED_REGIME_SOURCE_CALIBRATION_ID
    assert transition["calibrated_regime_label"] == "transition"
    assert 0.0 < float(transition["calibrated_regime_mismatch_risk"]) < 1.0
    assert post_stall["calibrated_regime_label"] == "post_stall"
    assert post_stall["calibrated_regime_mismatch_risk"] == pytest.approx(1.0)
    assert post_stall["calibrated_regime_mismatch_score_component"] == pytest.approx(-0.12)


def test_v53_governor_soft_score_tracks_front_wall_mission_utility() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    representative = {
        "compact_library_id": "launch",
        "primitive_variant_id": "launch",
        "primitive_id": "glide",
        "entry_role": "transition_object",
        "transition_entry_class": "launch_gate",
        "controller_id": "ctrl_launch",
        "K_gain_checksum": "k",
        "augmented_A_checksum": "a",
        "augmented_B_checksum": "b",
        "augmented_gain_checksum": "g",
        **timing_payload,
    }
    outcome = {
        "continuation_probability": 0.5,
        "transition_success_probability": 0.5,
        "transition_exit_classes_seen": "post_launch_degraded",
        "terminal_useful_probability": 0.0,
        "hard_failure_risk": 0.1,
    }
    context = {
        "context_id": "mission_soft_score",
        "start_state_family": "launch_gate",
        "current_x_w_m": 1.3,
        "current_y_w_m": 2.2,
        "current_z_w_m": 1.1,
        "mission_x_min_w_m": 1.2,
        "front_wall_target_x_w_m": 6.6,
        "mission_terminal_y_min_m": 0.0,
        "mission_terminal_y_max_m": 4.4,
        "mission_terminal_z_min_m": 0.4,
        "mission_terminal_z_max_m": 3.5,
        "mission_terminal_specific_energy_reference_m": 0.4,
        "governor_wall_margin_m": 0.5,
        "wall_margin_m": 0.5,
        "floor_margin_m": 0.7,
        "ceiling_margin_m": 2.4,
        "latency_case": "nominal",
    }

    def row(exit_x: float, *, exit_y: float = 2.2, exit_z: float = 1.4, speed: float = 5.0, residual: float = 0.0) -> dict[str, object]:
        return governor_candidate_row(
            representative=representative,
            outcome=outcome,
            context=context,
            governor_mode="continuation_mode",
            belief_features={
                "belief_local_updraft_gain_residual_m": residual,
                "belief_candidate_path_exit_x_w_m": exit_x,
                "belief_candidate_path_exit_y_w_m": exit_y,
                "belief_candidate_path_exit_z_w_m": exit_z,
                "belief_candidate_path_speed_m_s": speed,
            },
        )

    short = row(3.0)
    forward = row(5.0)
    energetic = row(5.0, exit_z=2.0, speed=6.5)
    wrong_wall = row(3.0, exit_y=0.0)
    memory = row(5.0, residual=0.2)

    assert forward["mission_front_wall_progress_fraction"] > short["mission_front_wall_progress_fraction"]
    assert forward["base_score_without_memory"] > short["base_score_without_memory"]
    assert energetic["mission_terminal_energy_progress_proxy_m"] > forward["mission_terminal_energy_progress_proxy_m"]
    assert energetic["base_score_without_memory"] > forward["base_score_without_memory"]
    assert wrong_wall["mission_wrong_boundary_proxy"] == pytest.approx(1.0)
    assert wrong_wall["base_score_without_memory"] < short["base_score_without_memory"]
    assert memory["memory_score_component"] == pytest.approx(DEFAULT_GOVERNOR_CONFIG.belief_weight * 0.2)


def test_v53_no_memory_baseline_still_uses_candidate_path_mission_geometry() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    representatives = []
    outcomes = {}
    for variant_id in ("short", "far"):
        representatives.append(
            {
                "compact_library_id": variant_id,
                "primitive_variant_id": variant_id,
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "controller_id": f"ctrl_{variant_id}",
                "K_gain_checksum": "k",
                "augmented_A_checksum": "a",
                "augmented_B_checksum": "b",
                "augmented_gain_checksum": "g",
                **timing_payload,
            }
        )
        outcomes[variant_id] = {
            "continuation_probability": 0.5,
            "transition_success_probability": 0.5,
            "transition_exit_classes_seen": "post_launch_degraded",
            "terminal_useful_probability": 0.0,
            "hard_failure_risk": 0.1,
        }

    def candidate_path_features(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
        variant_id = str(representative["primitive_variant_id"])
        return {
            "belief_candidate_path_residual_memory_active": False,
            "belief_candidate_path_exit_x_w_m": 5.0 if variant_id == "far" else 3.0,
            "belief_candidate_path_exit_y_w_m": 2.2,
            "belief_candidate_path_exit_z_w_m": 1.4,
            "belief_candidate_path_speed_m_s": 5.0,
        }

    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcomes,
        context={
            "context_id": "no_memory_mission_geometry",
            "start_state_family": "launch_gate",
            "current_x_w_m": 1.3,
            "current_y_w_m": 2.2,
            "current_z_w_m": 1.1,
            "mission_x_min_w_m": 1.2,
            "front_wall_target_x_w_m": 6.6,
            "mission_terminal_y_min_m": 0.0,
            "mission_terminal_y_max_m": 4.4,
            "mission_terminal_z_min_m": 0.4,
            "mission_terminal_z_max_m": 3.5,
            "governor_wall_margin_m": 0.5,
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.7,
            "ceiling_margin_m": 2.4,
            "latency_case": "nominal",
        },
        governor_mode="continuation_mode",
        candidate_belief_features=candidate_path_features,
        adaptive_memory_active=False,
    )

    by_variant = {str(row["primitive_variant_id"]): row for row in rows}
    assert selected is not None
    assert selected["primitive_variant_id"] == "far"
    assert by_variant["far"]["mission_front_wall_progress_fraction"] > by_variant["short"]["mission_front_wall_progress_fraction"]
    assert by_variant["far"]["memory_score_component"] == pytest.approx(0.0)
    assert by_variant["far"]["memory_shield_status"] == "not_active_no_candidate_path_memory"


def test_v53_memory_is_bounded_objective_after_viability_not_near_tie_only() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    representatives = []
    outcomes = {}
    for variant_id in ("baseline", "memory_objective", "far_gap"):
        representatives.append(
            {
                "compact_library_id": "launch",
                "primitive_variant_id": variant_id,
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "controller_id": f"ctrl_{variant_id}",
                "K_gain_checksum": "k",
                "augmented_A_checksum": "a",
                "augmented_B_checksum": "b",
                "augmented_gain_checksum": "g",
                **timing_payload,
            }
        )
        outcomes[variant_id] = {
            "continuation_probability": 0.5,
            "transition_success_probability": 0.5,
            "transition_exit_classes_seen": "post_launch_degraded",
            "terminal_useful_probability": 0.0,
            "hard_failure_risk": 0.1,
        }

    def candidate_path_features(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
        del outcome
        variant_id = str(representative["primitive_variant_id"])
        exit_x = {"baseline": 5.0, "memory_objective": 4.8, "far_gap": 3.0}[variant_id]
        residual = {"baseline": 0.0, "memory_objective": 0.4, "far_gap": 5.0}[variant_id]
        return {
            "belief_candidate_path_residual_memory_active": True,
            "belief_candidate_path_exit_x_w_m": exit_x,
            "belief_candidate_path_exit_y_w_m": 2.2,
            "belief_candidate_path_exit_z_w_m": 1.4,
            "belief_candidate_path_speed_m_s": 5.0,
            "belief_candidate_path_memory_utility_m": residual,
            "belief_local_specific_energy_residual_m": residual,
            "belief_candidate_path_confidence": 1.0,
            "belief_uncertainty": 0.0,
        }

    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcomes,
        context={
            "context_id": "bounded_memory_objective_contract",
            "start_state_family": "launch_gate",
            "current_x_w_m": 1.3,
            "current_y_w_m": 2.2,
            "current_z_w_m": 1.1,
            "mission_x_min_w_m": 1.2,
            "front_wall_target_x_w_m": 6.6,
            "mission_terminal_y_min_m": 0.0,
            "mission_terminal_y_max_m": 4.4,
            "mission_terminal_z_min_m": 0.4,
            "mission_terminal_z_max_m": 3.5,
            "governor_wall_margin_m": 0.5,
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.7,
            "ceiling_margin_m": 2.4,
            "latency_case": "nominal",
        },
        governor_mode="continuation_mode",
        candidate_belief_features=candidate_path_features,
        adaptive_memory_active=True,
        governor_config=governor_config_from_row(
            {
                "config_id": "bounded_memory_objective_test",
                "belief_weight": 1.0,
                "memory_near_tie_base_score_margin": 0.03,
                "memory_objective_score_cap": 0.20,
                "memory_objective_min_confidence": 0.1,
                "memory_objective_max_base_score_drop": 0.18,
                "memory_switch_min_confidence": 0.1,
                "memory_switch_min_score_margin": 0.0,
                "memory_switch_max_base_score_drop": 0.05,
                "memory_cost_benefit_score_cap": 0.20,
                "memory_cost_benefit_progress_cost_weight": 0.25,
            }
        ),
    )

    by_variant = {str(row["primitive_variant_id"]): row for row in rows}
    assert selected is not None
    assert selected["primitive_variant_id"] == "memory_objective"
    assert by_variant["memory_objective"]["memory_objective_residual_confidence_gate"] > 0.0
    assert by_variant["memory_objective"]["memory_score_component"] > 0.0
    assert by_variant["far_gap"]["raw_memory_score_component"] > by_variant["memory_objective"]["raw_memory_score_component"]
    assert by_variant["far_gap"]["memory_objective_residual_confidence_gate"] > 0.0
    assert by_variant["far_gap"]["memory_score_component"] <= 0.20 + 1e-9
    assert by_variant["far_gap"]["total_score_with_memory_and_exploration"] < (
        by_variant["memory_objective"]["total_score_with_memory_and_exploration"]
    )
    assert by_variant["memory_objective"]["memory_cost_benefit_total_benefit"] > (
        by_variant["memory_objective"]["memory_cost_benefit_total_cost"]
    )
    assert by_variant["memory_objective"]["memory_shield_status"] == (
        "accepted_cost_benefit_spatial_flow_memory_switch"
    )


def test_v53_memory_shield_rejects_only_calibrated_regime_risk_regression() -> None:
    timing_payload = _v53_timing_payload()
    representatives = []
    outcomes = {}
    for variant_id in ("baseline", "memory"):
        representatives.append(
            {
                "compact_library_id": "launch",
                "primitive_variant_id": variant_id,
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "controller_id": f"ctrl_{variant_id}",
                "K_gain_checksum": "k",
                "augmented_A_checksum": "a",
                "augmented_B_checksum": "b",
                "augmented_gain_checksum": "g",
                **timing_payload,
            }
        )
        outcomes[variant_id] = {
            "continuation_probability": 0.5,
            "transition_success_probability": 0.5,
            "transition_exit_classes_seen": "post_launch_degraded",
            "terminal_useful_probability": 0.0,
            "hard_failure_risk": 0.1,
        }

    context = {
        "context_id": "regime_risk_memory_contract",
        "start_state_family": "launch_gate",
        "current_x_w_m": 1.3,
        "current_y_w_m": 2.2,
        "current_z_w_m": 1.1,
        "mission_x_min_w_m": 1.2,
        "front_wall_target_x_w_m": 6.6,
        "mission_terminal_y_min_m": 0.0,
        "mission_terminal_y_max_m": 4.4,
        "mission_terminal_z_min_m": 0.4,
        "mission_terminal_z_max_m": 3.5,
        "governor_wall_margin_m": 0.5,
        "wall_margin_m": 0.5,
        "floor_margin_m": 0.7,
        "ceiling_margin_m": 2.4,
        "latency_case": "nominal",
    }
    governor_config = governor_config_from_row(
        {
            "config_id": "regime_risk_memory_shield_test",
            "belief_weight": 0.0,
            "memory_switch_min_score_margin": 0.0,
            "memory_switch_max_calibrated_regime_risk_increase": 0.0,
            "memory_cost_benefit_score_cap": 0.35,
            "memory_cost_benefit_progress_cost_weight": 0.0,
        }
    )

    def run(memory_alpha_deg: float) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
        def candidate_path_features(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
            del outcome
            variant_id = str(representative["primitive_variant_id"])
            return {
                "belief_candidate_path_residual_memory_active": True,
                "belief_candidate_path_exit_x_w_m": 5.0 if variant_id == "baseline" else 4.85,
                "belief_candidate_path_exit_y_w_m": 2.2,
                "belief_candidate_path_exit_z_w_m": 1.4,
                "belief_candidate_path_speed_m_s": 5.0,
                "belief_candidate_path_vertical_speed_m_s": 0.0,
                "belief_candidate_path_alpha_proxy_deg": 0.0 if variant_id == "baseline" else memory_alpha_deg,
                "belief_candidate_path_memory_utility_m": 0.0 if variant_id == "baseline" else 0.7,
                "belief_candidate_path_memory_utility_without_attraction_m": 0.0 if variant_id == "baseline" else 0.7,
                "belief_local_specific_energy_residual_m": 0.0,
                "belief_candidate_path_confidence": 1.0,
                "belief_candidate_path_exit_wall_margin_m": 0.5,
                "belief_candidate_path_exit_min_margin_m": 0.5,
                "belief_uncertainty": 0.0,
            }

        selected, rows = select_compact_representative(
            representatives=representatives,
            outcome_rows_by_variant_id=outcomes,
            context=context,
            governor_mode="continuation_mode",
            candidate_belief_features=candidate_path_features,
            adaptive_memory_active=True,
            governor_config=governor_config,
        )
        return selected, {str(row["primitive_variant_id"]): row for row in rows}

    risky_selected, risky_rows = run(25.0)
    assert risky_selected is not None
    assert risky_selected["primitive_variant_id"] == "baseline"
    assert risky_rows["memory"]["calibrated_regime_mismatch_risk"] == pytest.approx(1.0)
    assert risky_selected["memory_shield_status"] == "rejected_calibrated_regime_mismatch_risk_regression"
    assert risky_selected["memory_shield_calibrated_regime_mismatch_risk_delta"] > 0.0

    safe_selected, safe_rows = run(0.0)
    assert safe_selected is not None
    assert safe_selected["primitive_variant_id"] == "memory"
    assert safe_rows["memory"]["calibrated_regime_mismatch_risk"] == pytest.approx(0.0)
    assert safe_selected["memory_shield_status"] == "accepted_cost_benefit_spatial_flow_memory_switch"


def test_v53_episode_summary_memory_changed_selection_uses_selector_shield_acceptance() -> None:
    belief = _V53BeliefStub()
    row = _episode_row_from_sequence(
        scheduled={
            "launch_role": "final_heldout",
            "library_size_case_id": "balanced_cluster",
            "policy_id": "spatial_flow_belief_memory_h3",
            "history_length": 3,
            "outer_case_index": 1,
            "environment_block_id": "block",
            "common_final_launch_key": "launch_001",
            "episode_id": "episode_001",
        },
        policy={"policy_family": "memory", "safe_explore": False},
        primitive_rows=[_v53_successful_primitive_row()],
        selector_rows=[
            {
                "policy_id": "spatial_flow_belief_memory_h3",
                "primitive_step_index": 0,
                "candidate_count": 2,
                "viable_count": 2,
                "selected_memory_shield_status": "accepted_cost_benefit_spatial_flow_memory_switch",
                "selected_memory_shield_baseline_variant_id": "baseline_variant",
                "selected_memory_shield_memory_variant_id": "memory_variant",
            }
        ],
        context_row={"environment_instance_id": "env"},
        belief_before=belief,
        belief_after=belief,
    )

    assert row["memory_policy_decision_count"] == 1
    assert row["memory_switch_available_count"] == 1
    assert row["accepted_memory_switch_count"] == 1
    assert row["rejected_memory_switch_count"] == 0
    assert row["memory_changed_selection"] is True
    assert row["memory_changed_selection_source"] == "selector_memory_shield_accepted_switch"
    assert row["memory_switch_accepted_step_indices"] == "0"
    assert row["memory_switch_status_sequence"] == (
        "0:accepted_cost_benefit_spatial_flow_memory_switch:baseline_variant->memory_variant"
    )


def test_v53_selection_change_flags_preserve_memory_switch_audit_column() -> None:
    final = pd.DataFrame(
        [
            {
                "library_size_case_id": "balanced_cluster",
                "outer_case_index": 1,
                "history_length": 0,
                "policy_id": "no_memory_baseline",
                "selected_primitive_variant_id": "baseline_variant",
                "memory_changed_selection": False,
            },
            {
                "library_size_case_id": "balanced_cluster",
                "outer_case_index": 1,
                "history_length": 3,
                "policy_id": "spatial_flow_belief_memory_h3",
                "selected_primitive_variant_id": "memory_variant",
                "memory_changed_selection": True,
                "memory_changed_selection_source": "selector_memory_shield_accepted_switch",
            },
        ]
    )

    flagged = _with_selection_change_flags(final)
    memory_row = flagged[flagged["policy_id"] == "spatial_flow_belief_memory_h3"].iloc[0]
    assert bool(memory_row["memory_changed_selection"]) is True
    assert str(memory_row["memory_changed_selection_source"]) == "selector_memory_shield_accepted_switch"
    assert bool(memory_row["baseline_vs_policy_selection_changed"]) is True


def test_v53_flow_region_attraction_can_shape_safe_mission_compatible_candidate_band() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    representatives = []
    outcomes = {}
    for variant_id in ("baseline", "flow_region", "poor_progress"):
        representatives.append(
            {
                "compact_library_id": "launch",
                "primitive_variant_id": variant_id,
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "controller_id": f"ctrl_{variant_id}",
                "K_gain_checksum": "k",
                "augmented_A_checksum": "a",
                "augmented_B_checksum": "b",
                "augmented_gain_checksum": "g",
                **timing_payload,
            }
        )
        outcomes[variant_id] = {
            "continuation_probability": 0.5,
            "transition_success_probability": 0.5,
            "transition_exit_classes_seen": "post_launch_degraded",
            "terminal_useful_probability": 0.0,
            "hard_failure_risk": 0.1,
        }

    def candidate_path_features(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
        del outcome
        variant_id = str(representative["primitive_variant_id"])
        exit_x = {"baseline": 5.0, "flow_region": 4.3, "poor_progress": 3.0}[variant_id]
        attraction = {"baseline": 0.0, "flow_region": 0.25, "poor_progress": 0.25}[variant_id]
        return {
            "belief_candidate_path_residual_memory_active": True,
            "belief_candidate_path_exit_x_w_m": exit_x,
            "belief_candidate_path_exit_y_w_m": 2.2,
            "belief_candidate_path_exit_z_w_m": 1.4,
            "belief_candidate_path_speed_m_s": 5.0,
            "belief_candidate_path_memory_utility_without_attraction_m": 0.0,
            "belief_candidate_path_memory_utility_m": attraction,
            "belief_local_specific_energy_residual_m": 0.0,
            "belief_candidate_path_confidence": 0.0,
            "belief_flow_map_reachable_attraction_m": attraction,
            "belief_flow_map_reachable_attraction_confidence": 1.0,
            "belief_uncertainty": 0.0,
        }

    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcomes,
        context={
            "context_id": "flow_region_memory_contract",
            "start_state_family": "launch_gate",
            "current_x_w_m": 1.3,
            "current_y_w_m": 2.2,
            "current_z_w_m": 1.1,
            "mission_x_min_w_m": 1.2,
            "front_wall_target_x_w_m": 6.6,
            "mission_terminal_y_min_m": 0.0,
            "mission_terminal_y_max_m": 4.4,
            "mission_terminal_z_min_m": 0.4,
            "mission_terminal_z_max_m": 3.5,
            "governor_wall_margin_m": 0.5,
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.7,
            "ceiling_margin_m": 2.4,
            "latency_case": "nominal",
        },
        governor_mode="continuation_mode",
        candidate_belief_features=candidate_path_features,
        adaptive_memory_active=True,
        governor_config=governor_config_from_row(
            {
                "config_id": "flow_region_attraction_test",
                "belief_weight": 0.0,
                "memory_near_tie_base_score_margin": 0.01,
                "memory_switch_min_confidence": 0.1,
                "memory_switch_min_score_margin": 0.0,
                "memory_switch_max_base_score_drop": 0.01,
                "flow_region_attraction_weight": 0.45,
                "flow_region_attraction_score_cap": 0.08,
                "flow_region_attraction_min_confidence": 0.1,
                "flow_region_attraction_max_base_score_drop": 0.09,
                "flow_region_attraction_min_front_progress_ratio": 0.50,
                "memory_cost_benefit_score_cap": 0.20,
                "memory_cost_benefit_progress_cost_weight": 0.25,
            }
        ),
    )

    by_variant = {str(row["primitive_variant_id"]): row for row in rows}
    assert selected is not None
    assert selected["primitive_variant_id"] == "flow_region"
    assert by_variant["flow_region"]["memory_cost_benefit_known_flow_benefit_m"] > 0.0
    assert by_variant["flow_region"]["memory_score_component"] > 0.0
    assert by_variant["poor_progress"]["total_score_with_memory_and_exploration"] < (
        by_variant["flow_region"]["total_score_with_memory_and_exploration"]
    )
    assert by_variant["flow_region"]["memory_shield_status"] == (
        "accepted_cost_benefit_spatial_flow_memory_switch"
    )


def test_v53_information_gain_can_select_safe_under_observed_front_progress_candidate() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    representatives = []
    outcomes = {}
    for variant_id, primitive_id in (
        ("baseline", "glide"),
        ("information_gain", "mild_turn_left"),
        ("poor_progress", "mild_turn_right"),
    ):
        representatives.append(
            {
                "compact_library_id": "launch",
                "primitive_variant_id": variant_id,
                "primitive_id": primitive_id,
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "controller_id": f"ctrl_{variant_id}",
                "K_gain_checksum": "k",
                "augmented_A_checksum": "a",
                "augmented_B_checksum": "b",
                "augmented_gain_checksum": "g",
                **timing_payload,
            }
        )
        outcomes[variant_id] = {
            "continuation_probability": 0.5,
            "transition_success_probability": 0.5,
            "transition_exit_classes_seen": "post_launch_degraded",
            "terminal_useful_probability": 0.0,
            "hard_failure_risk": 0.1,
        }

    def candidate_path_features(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
        del outcome
        variant_id = str(representative["primitive_variant_id"])
        exit_x = {"baseline": 5.0, "information_gain": 4.7, "poor_progress": 2.4}[variant_id]
        information_gain = {"baseline": 0.0, "information_gain": 0.9, "poor_progress": 0.9}[variant_id]
        return {
            "belief_candidate_path_residual_memory_active": True,
            "belief_candidate_path_exit_x_w_m": exit_x,
            "belief_candidate_path_exit_y_w_m": 2.2,
            "belief_candidate_path_exit_z_w_m": 1.4,
            "belief_candidate_path_speed_m_s": 5.0,
            "belief_candidate_path_memory_utility_without_attraction_m": 0.0,
            "belief_candidate_path_memory_utility_m": 0.0,
            "belief_local_specific_energy_residual_m": 0.0,
            "belief_candidate_path_confidence": 0.0,
            "belief_flow_map_reachable_attraction_m": 0.0,
            "belief_flow_map_reachable_attraction_confidence": 0.0,
            "belief_uncertainty": information_gain,
            "belief_flow_map_information_gain": information_gain,
            "belief_flow_map_information_gain_path_uncertainty": information_gain,
            "belief_flow_map_information_gain_reachable_uncertainty": information_gain,
        }

    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcomes,
        context={
            "context_id": "information_gain_memory_contract",
            "start_state_family": "launch_gate",
            "current_x_w_m": 1.3,
            "current_y_w_m": 2.2,
            "current_z_w_m": 1.1,
            "mission_x_min_w_m": 1.2,
            "front_wall_target_x_w_m": 6.6,
            "mission_terminal_y_min_m": 0.0,
            "mission_terminal_y_max_m": 4.4,
            "mission_terminal_z_min_m": 0.4,
            "mission_terminal_z_max_m": 3.5,
            "governor_wall_margin_m": 0.5,
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.7,
            "ceiling_margin_m": 2.4,
            "latency_case": "nominal",
        },
        governor_mode="continuation_mode",
        candidate_belief_features=candidate_path_features,
        adaptive_memory_active=True,
        governor_config=governor_config_from_row(
            {
                "config_id": "information_gain_memory_test",
                "belief_weight": 0.0,
                "exploration_bonus_weight": 0.0,
                "memory_switch_min_score_margin": 0.0,
                "memory_information_gain_weight": 0.20,
                "memory_information_gain_score_cap": 0.12,
                "memory_information_gain_min_uncertainty": 0.30,
                "memory_information_gain_max_base_score_drop": 0.14,
                "memory_information_gain_min_front_progress_ratio": 0.50,
                "memory_information_gain_allow_cross_family": True,
                "memory_cost_benefit_information_gain_weight": 0.20,
                "memory_cost_benefit_score_cap": 0.12,
                "memory_cost_benefit_progress_cost_weight": 0.25,
            }
        ),
    )

    by_variant = {str(row["primitive_variant_id"]): row for row in rows}
    assert selected is not None
    assert selected["primitive_variant_id"] == "information_gain"
    assert by_variant["information_gain"]["memory_information_gain_score_component"] > 0.0
    assert by_variant["poor_progress"]["total_score_with_memory_and_exploration"] < (
        by_variant["information_gain"]["total_score_with_memory_and_exploration"]
    )
    assert by_variant["information_gain"]["memory_shield_status"] == (
        "accepted_cost_benefit_spatial_flow_memory_switch"
    )


def test_v53_route_flow_belief_can_select_safe_short_horizon_flow_route() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    representatives = []
    outcomes = {}
    for variant_id, primitive_id in (
        ("baseline", "glide"),
        ("route_flow", "mild_turn_left"),
        ("poor_route", "mild_turn_right"),
    ):
        representatives.append(
            {
                "compact_library_id": "launch",
                "primitive_variant_id": variant_id,
                "primitive_id": primitive_id,
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "controller_id": f"ctrl_{variant_id}",
                "K_gain_checksum": "k",
                "augmented_A_checksum": "a",
                "augmented_B_checksum": "b",
                "augmented_gain_checksum": "g",
                **timing_payload,
            }
        )
        outcomes[variant_id] = {
            "continuation_probability": 0.5,
            "transition_success_probability": 0.5,
            "transition_exit_classes_seen": "post_launch_degraded",
            "terminal_useful_probability": 0.0,
            "hard_failure_risk": 0.1,
        }

    def candidate_path_features(representative: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
        del outcome
        variant_id = str(representative["primitive_variant_id"])
        exit_x = {"baseline": 5.0, "route_flow": 4.55, "poor_route": 2.4}[variant_id]
        route_exploitation = {"baseline": 0.0, "route_flow": 0.65, "poor_route": 0.65}[variant_id]
        route_progress = {"baseline": 0.65, "route_flow": 0.74, "poor_route": 0.10}[variant_id]
        return {
            "belief_candidate_path_residual_memory_active": True,
            "belief_candidate_path_exit_x_w_m": exit_x,
            "belief_candidate_path_exit_y_w_m": 2.2,
            "belief_candidate_path_exit_z_w_m": 1.4,
            "belief_candidate_path_speed_m_s": 5.0,
            "belief_candidate_path_memory_utility_without_attraction_m": 0.0,
            "belief_candidate_path_memory_utility_m": 0.0,
            "belief_local_specific_energy_residual_m": 0.0,
            "belief_candidate_path_confidence": 0.0,
            "belief_flow_map_reachable_attraction_m": 0.0,
            "belief_flow_map_reachable_attraction_confidence": 0.0,
            "belief_uncertainty": 0.0,
            "belief_flow_map_route_policy": "unit_test_route_map",
            "belief_flow_map_route_horizon_primitives": 4,
            "belief_flow_map_route_probe_count": 7,
            "belief_flow_map_route_exploitation_m": route_exploitation,
            "belief_flow_map_route_information_gain": 0.30,
            "belief_flow_map_route_confidence": 1.0,
            "belief_flow_map_route_uncertainty": 0.0,
            "belief_flow_map_route_front_progress": route_progress,
            "belief_flow_map_route_safe_fraction": 1.0,
        }

    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcomes,
        context={
            "context_id": "route_flow_memory_contract",
            "start_state_family": "launch_gate",
            "current_x_w_m": 1.3,
            "current_y_w_m": 2.2,
            "current_z_w_m": 1.1,
            "mission_x_min_w_m": 1.2,
            "front_wall_target_x_w_m": 6.6,
            "mission_terminal_y_min_m": 0.0,
            "mission_terminal_y_max_m": 4.4,
            "mission_terminal_z_min_m": 0.4,
            "mission_terminal_z_max_m": 3.5,
            "governor_wall_margin_m": 0.5,
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.7,
            "ceiling_margin_m": 2.4,
            "latency_case": "nominal",
        },
        governor_mode="continuation_mode",
        candidate_belief_features=candidate_path_features,
        adaptive_memory_active=True,
        governor_config=governor_config_from_row(
            {
                "config_id": "route_flow_memory_test",
                "belief_weight": 0.0,
                "exploration_bonus_weight": 0.0,
                "memory_switch_min_score_margin": 0.0,
                "memory_route_planning_weight": 0.75,
                "memory_route_information_gain_weight": 0.25,
                "memory_route_score_cap": 0.26,
                "memory_route_min_confidence": 0.10,
                "memory_route_max_base_score_drop": 0.22,
                "memory_route_min_front_progress_ratio": 0.40,
                "memory_cost_benefit_score_cap": 0.35,
                "memory_cost_benefit_information_gain_weight": 0.08,
                "memory_cost_benefit_progress_cost_weight": 0.25,
            }
        ),
    )

    by_variant = {str(row["primitive_variant_id"]): row for row in rows}
    assert selected is not None
    assert selected["primitive_variant_id"] == "route_flow"
    assert by_variant["route_flow"]["memory_route_score_component"] > 0.0
    assert by_variant["poor_route"]["total_score_with_memory_and_exploration"] < (
        by_variant["route_flow"]["total_score_with_memory_and_exploration"]
    )
    assert by_variant["route_flow"]["memory_shield_status"] == (
        "accepted_cost_benefit_spatial_flow_memory_switch"
    )


def test_v53_score_rewards_front_wall_mission_and_updraft_without_time_or_energy_loss_penalty() -> None:
    base = {
        "safe_success": True,
        "mission_success": False,
        "front_wall_terminal_success": False,
        "wrong_wall_exit": False,
        "terminal_useful": False,
        "lift_capture": True,
        "hard_failure": False,
        "floor_or_ceiling_violation": False,
        "no_viable_primitive": False,
        "selected_primitive_step_count": 10,
        "episode_rollout_duration_s": 1.0,
        "updraft_specific_energy_gain_proxy_m": 0.1,
        "gross_specific_energy_loss_m": 0.0,
        "net_specific_energy_delta_m": -2.0,
        "min_wall_margin_m": 0.001,
        "speed_margin_m_s": -100.0,
    }
    score = _launch_score_fields(base)
    assert "speed_factor" not in score
    assert "speed_margin_factor" not in score
    assert "energy_loss_factor" not in score
    assert "net_energy_factor" not in score

    more_updraft = _launch_score_fields({**base, "updraft_specific_energy_gain_proxy_m": 0.8})
    longer = _launch_score_fields({**base, "episode_rollout_duration_s": 1.5})
    more_loss = _launch_score_fields({**base, "gross_specific_energy_loss_m": 99.0, "net_specific_energy_delta_m": -99.0})
    front_wall = _launch_score_fields(
        {
            **base,
            "mission_success": True,
            "front_wall_terminal_success": True,
            "terminal_useful": True,
            "terminal_specific_energy_m": 1.0,
        }
    )
    front_wall_more_terminal_energy = _launch_score_fields(
        {
            **base,
            "mission_success": True,
            "front_wall_terminal_success": True,
            "terminal_useful": True,
            "terminal_specific_energy_m": 3.0,
        }
    )
    wrong_wall = _launch_score_fields({**base, "wrong_wall_exit": True, "terminal_wall_face": "side_wall_y_max"})
    assert more_updraft["launch_score"] > score["launch_score"]
    assert longer["launch_score"] == pytest.approx(score["launch_score"])
    assert more_loss["launch_score"] == pytest.approx(score["launch_score"])
    assert front_wall["launch_score"] > score["launch_score"]
    assert front_wall_more_terminal_energy["launch_score"] > front_wall["launch_score"]
    assert front_wall_more_terminal_energy["terminal_specific_energy_bonus"] > front_wall["terminal_specific_energy_bonus"]
    assert wrong_wall["base_failure_penalty_reason"] == "wrong_wall_exit"
    assert wrong_wall["launch_score"] == pytest.approx(-50.0)
    assert score["airborne_time_reward_status"] == "audit_only_not_rewarded"

    final_score = _launch_score_fields_for_role({**base, "launch_role": "final_heldout"})
    history_score = _launch_score_fields_for_role({**base, "launch_role": "history"})
    assert final_score["launch_score_scope"] == "final_heldout_outer_loop_score"
    assert history_score["launch_score_scope"] == "history_launch_memory_update_not_outer_loop_score"
    assert history_score["base_failure_penalty_reason"] == "not_scored_history_launch"


def test_v53_selector_decision_logs_score_components_for_posthoc_audit() -> None:
    selected = {
        "compact_library_id": "compact_001",
        "primitive_variant_id": "variant_001",
        "primitive_id": "glide",
        "entry_role": "transition_object",
        "transition_entry_class": "launch_gate",
        "controller_id": "ctrl_001",
        "score": 1.23,
        "base_library_score_component": 0.70,
        "mission_score_component": 0.20,
        "exploration_score_component": 0.03,
        "memory_score_component": 0.10,
        "calibrated_regime_mismatch_score_component": -0.04,
    }
    row = selector_decision_row(
        episode_id="episode_001",
        primitive_step_index=0,
        policy_id="spatial_flow_belief_memory_h3",
        governor_mode="terminal_episode",
        context={"W_layer": "W3", "environment_mode": "w3_randomised_four"},
        selected=selected,
        candidate_count=4,
        viable_count=3,
    )

    assert row["selected_score"] == pytest.approx(1.23)
    assert row["selected_base_library_score_component"] == pytest.approx(0.70)
    assert row["selected_mission_score_component"] == pytest.approx(0.20)
    assert row["selected_exploration_score_component"] == pytest.approx(0.03)
    assert row["selected_memory_score_component"] == pytest.approx(0.10)
    assert row["selected_calibrated_regime_mismatch_score_component"] == pytest.approx(-0.04)


def test_v53_posthoc_tables_report_final_and_executed_scores_with_paired_deltas() -> None:
    final = pd.DataFrame(
        [
            {
                "episode_id": "open",
                "launch_role": "final_heldout",
                "library_size_case_id": "balanced_cluster",
                "environment_block_id": "r11_l1",
                "outer_case_index": 0,
                "common_final_launch_key": "paired_000",
                "policy_id": OPEN_LOOP_COMPARISON_POLICY_ID,
                "history_length": 0,
                "mission_success": False,
                "safe_success": True,
                "hard_failure": False,
                "no_viable_primitive": False,
                "launch_score": 0.0,
                "launch_score_version": "test",
            },
            {
                "episode_id": "base",
                "launch_role": "final_heldout",
                "library_size_case_id": "balanced_cluster",
                "environment_block_id": "r11_l1",
                "outer_case_index": 0,
                "common_final_launch_key": "paired_000",
                "policy_id": "no_memory_baseline",
                "history_length": 0,
                "mission_success": True,
                "safe_success": True,
                "hard_failure": False,
                "no_viable_primitive": False,
                "launch_score": 10.0,
                "launch_score_version": "test",
            },
            {
                "episode_id": "mem",
                "launch_role": "final_heldout",
                "library_size_case_id": "balanced_cluster",
                "environment_block_id": "r11_l1",
                "outer_case_index": 0,
                "common_final_launch_key": "paired_000",
                "policy_id": "spatial_flow_belief_memory_h3",
                "history_length": 3,
                "mission_success": True,
                "safe_success": True,
                "hard_failure": False,
                "no_viable_primitive": False,
                "launch_score": 12.0,
                "launch_score_version": "test",
            },
        ]
    )
    selector_rows = [
        {
            "episode_id": "base",
            "decision_status": "selected_compact_representative",
            "selected_primitive_variant_id": "base_v0",
            "selected_score": 1.0,
            "selected_base_library_score_component": 0.8,
            "selected_mission_score_component": 0.2,
            "selected_exploration_score_component": 0.0,
            "selected_memory_score_component": 0.0,
            "selected_calibrated_regime_mismatch_score_component": 0.0,
        },
        {
            "episode_id": "mem",
            "decision_status": "selected_compact_representative",
            "selected_primitive_variant_id": "mem_v0",
            "selected_score": 1.2,
            "selected_base_library_score_component": 0.8,
            "selected_mission_score_component": 0.2,
            "selected_exploration_score_component": 0.1,
            "selected_memory_score_component": 0.1,
            "selected_calibrated_regime_mismatch_score_component": 0.0,
        },
        {
            "episode_id": "mem",
            "decision_status": "selected_compact_representative",
            "selected_primitive_variant_id": "mem_v1",
            "selected_score": 0.4,
            "selected_base_library_score_component": 0.3,
            "selected_mission_score_component": 0.1,
            "selected_exploration_score_component": 0.0,
            "selected_memory_score_component": 0.0,
            "selected_calibrated_regime_mismatch_score_component": 0.0,
        },
        {
            "episode_id": "mem",
            "decision_status": "blocked_no_viable_representative",
            "selected_primitive_variant_id": "",
            "selected_score": 99.0,
        },
    ]

    posthoc_final = _posthoc_final_score_table(final)
    posthoc_exec = _posthoc_executed_score_table(final, selector_rows)
    posthoc_delta = _posthoc_score_delta_table(posthoc_exec)

    assert set(posthoc_final["policy_id"]) == {
        OPEN_LOOP_COMPARISON_POLICY_ID,
        "no_memory_baseline",
        "spatial_flow_belief_memory_h3",
    }
    mem = posthoc_exec[posthoc_exec["episode_id"] == "mem"].iloc[0]
    assert mem["selector_decision_count"] == 3
    assert mem["executed_selected_decision_count"] == 2
    assert mem["blocked_selector_decision_count"] == 1
    assert mem["accumulated_selected_score"] == pytest.approx(1.6)
    assert mem["accumulated_memory_score_component"] == pytest.approx(0.1)

    memory_vs_base = posthoc_delta[
        (posthoc_delta["policy_id"] == "spatial_flow_belief_memory_h3")
        & (posthoc_delta["baseline_policy_id"] == "no_memory_baseline")
    ].iloc[0]
    assert memory_vs_base["delta_launch_score"] == pytest.approx(2.0)
    assert memory_vs_base["delta_accumulated_selected_score"] == pytest.approx(0.6)


def test_v53_r10_and_r11_changed_case_randomisation_semantics_match() -> None:
    assert tuple(block.block_id for block in R10_PROTOCOL.blocks) == (
        R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    )
    assert R10_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * 50
    assert R10_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * 50 * (3 + 10 + 30)

    assert tuple(block.block_id for block in R11_PROTOCOL.blocks) == (
        R11_L0_DRY_AIR_FIXED_BLOCK_ID,
        R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
        R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
        R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
        R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
        R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
        R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
        R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    )
    assert R11_PROTOCOL.outer_cases_per_condition == 8 * 20
    assert all(int(block.case_count) == 20 for block in R11_PROTOCOL.blocks)
    assert R11_PROTOCOL.policy_history_conditions == R11_POLICY_HISTORY_CONDITIONS
    assert R11_POLICY_HISTORY_CONDITIONS == (
        OPEN_LOOP_COMPARISON_POLICY_ID,
        "no_memory_baseline",
        "spatial_flow_belief_memory_h3",
        "spatial_flow_belief_memory_h10",
        "spatial_flow_belief_memory_h30",
    )
    assert R11_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(R11_POLICY_HISTORY_CONDITIONS) * 160
    assert R11_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * 160 * (3 + 10 + 30)

    l0_cases = [
        row
        for row in _outer_case_schedule(protocol=R11_PROTOCOL, seed=110)
        if row["environment_block_id"] == R11_L0_DRY_AIR_FIXED_BLOCK_ID
    ]
    assert [int(row["paired_start_condition_index"]) for row in l0_cases] == [
        0,
        3,
        5,
        8,
        10,
        13,
        15,
        18,
        21,
        23,
        26,
        28,
        31,
        34,
        36,
        39,
        41,
        44,
        46,
        49,
    ]

    assert _scheduled_active_fan_count_for_outer_case(
        protocol=R11_PROTOCOL,
        environment_block_id=R11_L0_DRY_AIR_FIXED_BLOCK_ID,
        environment_block_local_index=0,
    ) == 0
    assert (
        _fan_position_policy_for_outer_case(
            protocol=R11_PROTOCOL,
            environment_block_id=R11_L0_DRY_AIR_FIXED_BLOCK_ID,
        )
        == "no_fan_positions"
    )
    assert _scheduled_active_fan_count_for_outer_case(
        protocol=R11_PROTOCOL,
        environment_block_id=R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
        environment_block_local_index=0,
    ) == 0
    assert _scheduled_active_fan_count_for_outer_case(
        protocol=R11_PROTOCOL,
        environment_block_id=R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
        environment_block_local_index=4,
    ) == 4
    assert _scheduled_active_fan_count_for_outer_case(
        protocol=R11_PROTOCOL,
        environment_block_id=R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
        environment_block_local_index=0,
    ) == 4
    assert (
        _fan_position_policy_for_outer_case(
            protocol=R11_PROTOCOL,
            environment_block_id=R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
        )
        == "fixed_base_positions"
    )
    for protocol, block_id in (
        (R10_PROTOCOL, R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID),
        (R11_PROTOCOL, R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID),
    ):
        assert (
            _fan_position_policy_for_outer_case(
                protocol=protocol,
                environment_block_id=block_id,
            )
            == "independent_uniform_xy_bounds"
        )
        broad = next(
            row
            for row in _outer_case_schedule(protocol=protocol, seed=110)
            if row["environment_block_id"] == block_id
        )
        history = _history_row_for_final({**broad, "episode_id": "contract", "launch_role": "final_heldout"}, 0)
        assert broad["fan_position_safety_radius_m"] == pytest.approx(0.5)
        assert R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M == pytest.approx(0.5)
        assert _uses_full_w3_randomisation_block(protocol=protocol, environment_block_id=block_id)
        assert history["environment_seed"] != broad["environment_seed"]
        assert history["environment_parameter_seed"] != broad["environment_parameter_seed"]
        assert history["environment_layout_seed"] == broad["environment_layout_seed"]
        assert history["environment_active_fan_seed"] == broad["environment_active_fan_seed"]
        assert history["plant_implementation_seed"] == broad["plant_implementation_seed"]
        assert history["scheduled_active_fan_count"] == broad["scheduled_active_fan_count"]

    for fixed_block_id in (
        R11_L0_DRY_AIR_FIXED_BLOCK_ID,
        R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
        R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    ):
        outer = next(
            row
            for row in _outer_case_schedule(protocol=R11_PROTOCOL, seed=110)
            if row["environment_block_id"] == fixed_block_id
        )
        history = _history_row_for_final({**outer, "episode_id": "fixed_contract", "launch_role": "final_heldout"}, 0)
        assert history["environment_seed"] == outer["environment_seed"]
        assert history["environment_parameter_seed"] == outer["environment_parameter_seed"]
        assert history["environment_layout_seed"] == outer["environment_layout_seed"]
        assert history["environment_active_fan_seed"] == outer["environment_active_fan_seed"]
        assert history["plant_implementation_seed"] == outer["plant_implementation_seed"]
        assert history["scheduled_active_fan_count"] == outer["scheduled_active_fan_count"]

    for varying_parameter_block_id in (
        R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
        R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
        R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    ):
        outer = next(
            row
            for row in _outer_case_schedule(protocol=R11_PROTOCOL, seed=110)
            if row["environment_block_id"] == varying_parameter_block_id
        )
        history = _history_row_for_final({**outer, "episode_id": "varying_contract", "launch_role": "final_heldout"}, 0)
        assert history["environment_seed"] != outer["environment_seed"]
        assert history["environment_parameter_seed"] != outer["environment_parameter_seed"]
        assert history["environment_layout_seed"] == outer["environment_layout_seed"]
        assert history["environment_active_fan_seed"] == outer["environment_active_fan_seed"]
        assert history["plant_implementation_seed"] == outer["plant_implementation_seed"]
        assert history["scheduled_active_fan_count"] == outer["scheduled_active_fan_count"]

    for fixed_episode_block_id in (
        R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
        R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
    ):
        outer = next(
            row
            for row in _outer_case_schedule(protocol=R11_PROTOCOL, seed=110)
            if row["environment_block_id"] == fixed_episode_block_id
        )
        history = _history_row_for_final({**outer, "episode_id": "fixed_episode_contract", "launch_role": "final_heldout"}, 0)
        assert history["environment_seed"] == outer["environment_seed"]
        assert history["environment_parameter_seed"] == outer["environment_parameter_seed"]
        assert history["environment_layout_seed"] == outer["environment_layout_seed"]
        assert history["environment_active_fan_seed"] == outer["environment_active_fan_seed"]
        assert history["plant_implementation_seed"] == outer["plant_implementation_seed"]
        assert history["scheduled_active_fan_count"] == outer["scheduled_active_fan_count"]

    r11_schedule = _outer_case_schedule(protocol=R11_PROTOCOL, seed=110)
    by_local_index: dict[int, list[dict[str, object]]] = {}
    for row in r11_schedule:
        by_local_index.setdefault(int(row["environment_block_local_index"]), []).append(row)
    assert set(by_local_index) == set(range(20))
    for local_index, rows in by_local_index.items():
        paired_start_index = int(rows[0]["paired_start_condition_index"])
        assert len(rows) == 8
        assert {row["environment_block_id"] for row in rows} == {
            R11_L0_DRY_AIR_FIXED_BLOCK_ID,
            R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
            R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
            R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
            R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
            R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
            R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
            R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
        }
        assert {row["launch_state_seed"] for row in rows} == {
            110 * 100000 + paired_start_index * 37 + 11
        }
        assert {row["paired_start_condition_index"] for row in rows} == {paired_start_index}
        assert {row["paired_start_condition_key"] for row in rows} == {
            f"r11_heldout_paired_start_{paired_start_index:04d}"
        }
        history_rows = [
            _history_row_for_final({**row, "episode_id": "paired_contract", "launch_role": "final_heldout"}, 7)
            for row in rows
        ]
        assert {row["launch_state_seed"] for row in history_rows} == {
            110 * 100000 + paired_start_index * 37 + 11 + 1000000 + 7 * 101
        }


def test_v53_r10_r11_case7_uses_w3_plant_and_implementation_randomisation() -> None:
    state = np.zeros(15, dtype=float)
    state[STATE_INDEX["x_w"]] = 1.3
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.7
    state[STATE_INDEX["u"]] = 5.5

    for protocol, block_id in (
        (R10_PROTOCOL, R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID),
        (R11_PROTOCOL, R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID),
    ):
        outer = next(
            row
            for row in _outer_case_schedule(protocol=protocol, seed=110)
            if row["environment_block_id"] == block_id
        )
        payload = _context_payload(
            state=state,
            scheduled=outer,
            episode_id="case7_contract",
            protocol=protocol,
            start_state_family="launch_gate",
            primitive_step_index=0,
        )
        assert payload["row"]["full_w3_randomisation_block"] is True
        assert payload["plant_instance"].W_layer == "W3"
        assert payload["implementation_instance"].W_layer == "W3"
        assert payload["implementation_instance"].implementation_adjustment_status == "randomised_applied"
        assert payload["implementation_instance"].aileron_effectiveness_scale == pytest.approx(
            IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
        )
        assert payload["implementation_instance"].elevator_effectiveness_scale == pytest.approx(
            IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
        )
        assert payload["implementation_instance"].rudder_effectiveness_scale == pytest.approx(
            IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
        )
        assert payload["plant_instance"].plant_adjustment_status == "randomised_applied"
        assert (
            payload["plant_instance"].control_effectiveness_perturbation_policy
            == "global_plus_axis_scheduled_surface_authority_multiplier_v4"
        )
        assert (
            GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
            <= payload["plant_instance"].global_control_effectiveness_multiplier
            <= GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
        )
        assert (
            AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
            <= payload["plant_instance"].aileron_control_effectiveness_multiplier
            <= AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
        )
        assert (
            ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
            <= payload["plant_instance"].elevator_control_effectiveness_multiplier
            <= ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
        )
        assert (
            RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
            <= payload["plant_instance"].rudder_control_effectiveness_multiplier
            <= RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
        )


def _r10_claim_gate_schedule_and_rows(passing_case_id: str | None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    final_schedule = []
    final_rows = []
    case_ids = list(LIBRARY_SIZE_CASE_IDS)
    policy_ids = list(R10_PROTOCOL.policy_history_conditions)
    for index in range(R10_EXPECTED_FINAL_HELDOUT_LAUNCHES):
        case_id = case_ids[index % len(case_ids)]
        case_passes = case_id == passing_case_id
        final_schedule.append(
            {
                "library_size_case_id": case_id,
                "policy_id": policy_ids[(index // len(case_ids)) % len(policy_ids)],
                "outer_case_index": index,
                "common_final_launch_key": f"r10_case_{index}",
                "launch_state_seed": index,
                "environment_seed": index,
            }
        )
        final_rows.append(
            {
                "launch_role": "final_heldout",
                "library_size_case_id": case_id,
                "safe_success": case_passes,
                "full_safe_success": case_passes,
                "terminal_useful": case_passes,
                "lift_capture": False,
                "hard_failure": not case_passes,
                "floor_or_ceiling_violation": False,
                "no_viable_primitive": False,
                "claim_bearing_episode": True,
                "launch_inflight_recovery_sequence_compliant": True,
                "selected_primitive_id": "glide",
                "selected_primitive_variant_id": "v0",
            }
        )
    return final_schedule, final_rows


def test_v53_r10_claim_gate_passes_when_any_deployable_cluster_profile_passes() -> None:
    final_schedule, final_rows = _r10_claim_gate_schedule_and_rows("balanced_cluster")

    gates = _pass_fail_summary(
        protocol=R10_PROTOCOL,
        max_primitives_per_launch=0,
        max_episode_time_s=20.0,
        final_schedule=final_schedule,
        history_schedule=[{}] * R10_EXPECTED_HISTORY_LAUNCHES,
        episode_rows=final_rows,
        pairing_rows=[{"pairing_passed": True}],
        no_variation_rows=[{"variation_audit_passed": True}],
    )

    aggregate_hard_failure = next(row for row in gates if row["gate_id"] == "hard_failure_rate_within_stage_profile")
    any_cluster_gate = next(
        row for row in gates if row["gate_id"] == "r10_any_deployable_cluster_claim_profile_within_stage_profile"
    )
    assert aggregate_hard_failure["observed"] > R10_PROTOCOL.max_hard_failure_rate
    assert aggregate_hard_failure["passed"] is True
    assert any_cluster_gate["passed"] is True
    assert any_cluster_gate["observed"] == "balanced_cluster"
    assert all(bool(row["passed"]) for row in gates)


def test_v53_r10_claim_gate_ignores_non_deployable_cluster_only_pass() -> None:
    final_schedule, final_rows = _r10_claim_gate_schedule_and_rows("super_light_cluster")

    gates = _pass_fail_summary(
        protocol=R10_PROTOCOL,
        max_primitives_per_launch=0,
        max_episode_time_s=20.0,
        final_schedule=final_schedule,
        history_schedule=[{}] * R10_EXPECTED_HISTORY_LAUNCHES,
        episode_rows=final_rows,
        pairing_rows=[{"pairing_passed": True}],
        no_variation_rows=[{"variation_audit_passed": True}],
    )

    any_cluster_gate = next(
        row for row in gates if row["gate_id"] == "r10_any_deployable_cluster_claim_profile_within_stage_profile"
    )
    assert any_cluster_gate["passed"] is False
    assert any_cluster_gate["observed"] == "none"


def test_v53_r11_full_safe_success_gate_catches_safe_exit_only_passes() -> None:
    final_schedule = []
    for case_id in LIBRARY_SIZE_CASE_IDS:
        for policy_id in POLICY_HISTORY_CONDITIONS:
            final_schedule.append(
                {
                    "library_size_case_id": case_id,
                    "policy_id": policy_id,
                    "outer_case_index": len(final_schedule),
                    "common_final_launch_key": f"case_{len(final_schedule)}",
                    "launch_state_seed": len(final_schedule),
                    "environment_seed": len(final_schedule),
                }
            )
    final_rows = [
        {
            "launch_role": "final_heldout",
            "safe_success": True,
            "full_safe_success": False,
            "terminal_useful": True,
            "lift_capture": False,
            "hard_failure": False,
            "floor_or_ceiling_violation": False,
            "no_viable_primitive": False,
            "launch_inflight_recovery_sequence_compliant": True,
            "selected_primitive_id": "a;b;c;d;e",
            "selected_primitive_variant_id": "v0;v1;v2;v3;v4;v5;v6;v7;v8;v9",
        }
        for _ in final_schedule
    ]
    gates = _pass_fail_summary(
        protocol=R11_PROTOCOL,
        max_primitives_per_launch=0,
        max_episode_time_s=20.0,
        final_schedule=final_schedule,
        history_schedule=[{}] * R11_PROTOCOL.expected_history_launches,
        episode_rows=final_rows,
        pairing_rows=[{"pairing_passed": True}],
        no_variation_rows=[{"variation_audit_passed": True}],
    )
    full_safe_gate = next(row for row in gates if row["gate_id"] == "full_safe_success_rate_within_stage_profile")
    assert full_safe_gate["passed"] is False


def test_v53_launch_then_inflight_then_recovery_routing_contract() -> None:
    assert validation_route_for_primitive_step(0)["route_required_entry_class"] == "launch_gate"
    assert validation_route_for_primitive_step(1)["route_required_entry_class"] == "inflight_stable"


def test_v53_hard_algorithm_contract_audit_writes_ready_report(tmp_path) -> None:
    result = run_v53_algorithm_contract_audit(
        AlgorithmContractAuditConfig(
            output_root=tmp_path / "algorithm_contract_audit",
            run_id=1,
        )
    )

    assert result["status"] == "ready"
    run_root = tmp_path / "algorithm_contract_audit" / "001"
    assert (run_root / "metrics" / "active_code_contract_audit.csv").is_file()
    assert (run_root / "metrics" / "active_source_audit.csv").is_file()
    assert (run_root / "metrics" / "docs_code_consistency_audit.csv").is_file()
    assert (run_root / "metrics" / "legacy_alias_audit.csv").is_file()
    report = (run_root / "reports" / "algorithm_contract_audit_report.md").read_text(encoding="ascii")
    assert "Intentional Legacy Aliases" in report
    assert "expected_energy_residual_m" in report
    assert "R6 / W2 replay" in report
