from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

from run_changed_case_validation import R10_PROTOCOL, R11_PROTOCOL
from prim_cat import ACTIVE_PRIMITIVE_IDS, LAUNCH_CAPTURE_PRIMITIVE_IDS
from primitive_variant_registry import ENTRY_ROLE_BY_PRIMITIVE_ID, start_family_is_compatible
from run_lqr_w01_dense_chunked import (
    L6_RICH_SIDE_CANDIDATE_COUNT,
    L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    L6_RICH_SIDE_ROW_COUNT,
    OFFICIAL_W01_ENVIRONMENT_CASES,
    R5_ACTIVE_FAN_COUNT_SEQUENCE,
    W01DenseRunConfig,
    _row_schedule_for_index as _r5_row_schedule_for_index,
    rich_side_dense_row_count,
)
from run_post_w3_library_size_study import _coverage_medoid_selection, _representative_score
from run_r5_r10_pipeline import ARCHIVED_STAGES, STAGE_ORDER
from run_v53_algorithm_contract_audit import AlgorithmContractAuditConfig, run_v53_algorithm_contract_audit
from run_repeated_launch_learning_curve import (
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    EMPTY_FROZEN_PRIOR_BASELINE_ID,
    LIBRARY_SIZE_CASE_IDS,
    POLICY_HISTORY_CONDITIONS,
    R9_POLICY_HISTORY_CONDITIONS,
    R9_BLOCKS,
    R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R9_EXPECTED_HISTORY_LAUNCHES,
    R9_OUTER_CASES_PER_CONDITION,
    R9_PROTOCOL,
    _fan_position_policy_for_outer_case,
    _history_row_for_final,
    _launch_score_fields,
    _launch_score_fields_for_role,
    _outer_case_schedule,
    _pairing_audit_rows,
    _pass_fail_summary,
    _policy_condition,
    _scheduled_active_fan_count_for_outer_case,
    _selected_set,
    _tuned_governor_config_from_metrics,
    validation_route_for_primitive_step,
)
from run_w3_survival import R5_INPUT_KIND, W3_ACTIVE_FAN_COUNT_SEQUENCE, W3_ENVIRONMENT_CASES
from viability_governor import DEFAULT_GOVERNOR_CONFIG, REJECTION_REASONS, governor_candidate_row


def test_v53_stage_contract_is_r5_r7_r8_r10_r11_with_r6_archived_and_r9_internal() -> None:
    assert STAGE_ORDER == ("R5", "R7", "R8", "R10", "R11")
    assert ARCHIVED_STAGES == ("R6",)

    assert R9_PROTOCOL.validation_evidence_level == "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence"
    assert R9_PROTOCOL.gate_profile == "internal_reduced_fixed_case_preflight_for_r10_initialisation"
    assert R10_PROTOCOL.validation_evidence_level == "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate"
    assert R10_PROTOCOL.gate_profile == "relaxed_changed_case_viability_governor_learning_not_final_validation"
    assert R11_PROTOCOL.validation_evidence_level == "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation"
    assert R11_PROTOCOL.gate_profile == "strict_final_heldout_validation"
    assert R11_PROTOCOL.min_full_safe_success_rate == pytest.approx(0.99)


def test_v53_r9_is_reduced_internal_preflight_and_can_seed_r10_governor() -> None:
    assert tuple(block.case_count for block in R9_BLOCKS) == (1, 1, 1)
    assert R9_OUTER_CASES_PER_CONDITION == 3
    assert R9_POLICY_HISTORY_CONDITIONS == ("no_memory_baseline", "directional_3d_residual_memory_h20")
    assert R9_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(R9_POLICY_HISTORY_CONDITIONS) * 3 == 30
    assert R9_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * 3 * 20 == 300

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

    assert tuned.config_id == "v53_r9_tuned_viability_governor"
    assert tuned.minimum_wall_margin_m == DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m
    assert tuned.maximum_hard_failure_risk < DEFAULT_GOVERNOR_CONFIG.maximum_hard_failure_risk
    assert tuned.exploration_bonus_weight < DEFAULT_GOVERNOR_CONFIG.exploration_bonus_weight
    assert {row["parameter"] for row in decisions} >= {"maximum_hard_failure_risk", "exploration_bonus_weight"}


def test_v53_r5_dense_schedule_is_transition_entry_separated_and_uses_current_randomisation_cases() -> None:
    assert len(ACTIVE_PRIMITIVE_IDS) == 8
    assert len(LAUNCH_CAPTURE_PRIMITIVE_IDS) == 6
    assert not set(LAUNCH_CAPTURE_PRIMITIVE_IDS).intersection(set(ACTIVE_PRIMITIVE_IDS))
    assert rich_side_dense_row_count() == 76800
    assert L6_RICH_SIDE_ROW_COUNT == 76800
    assert L6_RICH_SIDE_CANDIDATE_COUNT == 32
    assert L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE == 100
    assert OFFICIAL_W01_ENVIRONMENT_CASES == (
        ("W0", "dry_air"),
        ("W1", "w1_annular_gp_randomised_single"),
        ("W1", "w1_annular_gp_randomised_four"),
    )
    assert R5_ACTIVE_FAN_COUNT_SEQUENCE == (1, 2, 3, 4)

    config = W01DenseRunConfig(
        run_id=0,
        rows=L6_RICH_SIDE_ROW_COUNT,
        candidate_count=L6_RICH_SIDE_CANDIDATE_COUNT,
        paired_tests_per_candidate=L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    )
    family_counts: dict[str, int] = {}
    environment_counts: dict[tuple[str, str], int] = {}
    for row_index in range(L6_RICH_SIDE_ROW_COUNT):
        schedule = _r5_row_schedule_for_index(row_index, config)
        role = ENTRY_ROLE_BY_PRIMITIVE_ID[schedule.primitive_id]
        assert start_family_is_compatible(entry_role=role, start_state_family=schedule.start_state_family)
        family_counts[schedule.start_state_family] = family_counts.get(schedule.start_state_family, 0) + 1
        key = (schedule.W_layer, schedule.environment_mode)
        environment_counts[key] = environment_counts.get(key, 0) + 1

    assert family_counts == {
        "launch_gate": 30720,
        "inflight_nominal": 19200,
        "inflight_lift_region": 11520,
        "inflight_boundary_near": 7680,
        "inflight_recovery_edge": 7680,
    }
    assert environment_counts == {
        ("W0", "dry_air"): 25600,
        ("W1", "w1_annular_gp_randomised_single"): 25600,
        ("W1", "w1_annular_gp_randomised_four"): 25600,
    }


def test_v53_r7_and_r8_contracts_are_direct_holdout_and_updraft_scored() -> None:
    assert R5_INPUT_KIND == "r5_frozen_bundle_direct"
    assert W3_ENVIRONMENT_CASES == ("w3_randomised_single", "w3_randomised_four")
    assert W3_ACTIVE_FAN_COUNT_SEQUENCE == (1, 2, 3, 4)

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

    memory_final = next(row for row in final_schedule if row["policy_id"] == "directional_3d_residual_memory_h20")
    history = _history_row_for_final(memory_final, 0)
    assert history["library_size_case_id"] == memory_final["library_size_case_id"]
    assert history["policy_id"] == memory_final["policy_id"]
    assert history["history_length"] == memory_final["history_length"]
    assert history["common_final_launch_key"] == memory_final["common_final_launch_key"]
    assert history["launch_state_seed"] != memory_final["launch_state_seed"]
    assert history["environment_seed"] != memory_final["environment_seed"]

    assert _policy_condition("no_memory_baseline")["uses_memory"] is False
    assert _policy_condition(EMPTY_FROZEN_PRIOR_BASELINE_ID)["uses_memory"] is True
    assert _policy_condition(EMPTY_FROZEN_PRIOR_BASELINE_ID)["updates_memory"] is False


def test_v53_governor_has_no_speed_boundary_and_wall_guard_is_0p10cm() -> None:
    assert DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m == pytest.approx(0.001)
    assert not any("speed" in reason for reason in REJECTION_REASONS)
    assert not any("speed" in key for key in asdict(DEFAULT_GOVERNOR_CONFIG))

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


def test_v53_score_rewards_final_path_updraft_and_time_without_speed_or_energy_loss_penalty() -> None:
    base = {
        "safe_success": True,
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
    assert more_updraft["launch_score"] > score["launch_score"]
    assert longer["launch_score"] > score["launch_score"]
    assert more_loss["launch_score"] == pytest.approx(score["launch_score"])

    final_score = _launch_score_fields_for_role({**base, "launch_role": "final_heldout"})
    history_score = _launch_score_fields_for_role({**base, "launch_role": "history"})
    assert final_score["launch_score_scope"] == "final_heldout_outer_loop_score"
    assert history_score["launch_score_scope"] == "history_launch_memory_update_not_outer_loop_score"
    assert history_score["base_failure_penalty_reason"] == "not_scored_history_launch"


def test_v53_r10_and_r11_changed_case_randomisation_semantics_match() -> None:
    for protocol in (R10_PROTOCOL, R11_PROTOCOL):
        assert _scheduled_active_fan_count_for_outer_case(
            protocol=protocol,
            environment_block_id=ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
            environment_block_local_index=0,
        ) == 1
        assert _scheduled_active_fan_count_for_outer_case(
            protocol=protocol,
            environment_block_id="nominal_four_fan_perturbations",
            environment_block_local_index=0,
        ) == 4
        assert (
            _fan_position_policy_for_outer_case(
                protocol=protocol,
                environment_block_id="nominal_single_fan_perturbations",
            )
            == "fixed_base_positions"
        )
        assert (
            _fan_position_policy_for_outer_case(
                protocol=protocol,
                environment_block_id=BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
            )
            == "independent_uniform_xy_bounds"
        )


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
