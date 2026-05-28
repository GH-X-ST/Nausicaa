from __future__ import annotations

import pandas as pd

from run_post_w3_library_size_study import _survived_frame_blocked_reason
import numpy as np

from transition_labels import classify_state, classify_transition, transition_is_chain_compatible, turn_intent_row_fields
from viability_governor import governor_rejection_reason


def test_launch_boundary_exit_is_not_chain_compatible() -> None:
    transition = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "launch_gate",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "model_boundary_only",
            "minimum_wall_margin_m": 0.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )

    assert transition["exit_class"] == "boundary_near"
    assert transition["transition_chain_compatible"] is False


def test_launch_post_launch_handoff_is_chain_compatible() -> None:
    transition = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "launch_gate",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )

    assert transition["exit_class"] == "post_launch_degraded"
    assert transition["transition_chain_compatible"] is True


def test_local_weak_inflight_rollout_is_not_sufficient() -> None:
    transition = classify_transition(
        {
            "entry_role": "inflight_only",
            "start_state_family": "inflight_nominal",
            "outcome_class": "weak",
            "continuation_valid": False,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )

    assert transition["exit_class"] == "recoverable_degraded"
    assert transition["transition_chain_compatible"] is False


def test_recovery_to_inflight_is_chain_compatible() -> None:
    assert transition_is_chain_compatible(
        entry_role="transition_object",
        entry_class="recoverable_degraded",
        exit_class="inflight_stable",
    )


def test_recovery_self_transition_requires_measurable_progress() -> None:
    initial = _recovery_state(phi=0.80, theta=0.45, p=0.90)
    exit_state = _recovery_state(phi=0.74, theta=0.42, p=0.80)

    transition = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.6,
            "ceiling_margin_m": 1.0,
            "initial_state_vector": initial.tolist(),
            "exit_state_vector": exit_state.tolist(),
        }
    )

    assert transition["exit_class"] == "recoverable_degraded"
    assert transition["transition_chain_compatible"] is True
    assert transition["recovery_progress_valid"] is True
    assert transition["recovery_progress_risk_delta"] > 0.0


def test_recovery_self_transition_without_progress_is_rejected() -> None:
    initial = _recovery_state(phi=0.80, theta=0.45, p=0.90)
    exit_state = _recovery_state(phi=0.82, theta=0.46, p=0.92)

    transition = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.6,
            "ceiling_margin_m": 1.0,
            "initial_state_vector": initial.tolist(),
            "exit_state_vector": exit_state.tolist(),
        }
    )

    assert transition["exit_class"] == "recoverable_degraded"
    assert transition["transition_chain_compatible"] is False
    assert transition["recovery_progress_valid"] is False
    assert "without_attitude_rate_progress" in str(transition["transition_failure_reason"])


def test_recovery_to_boundary_near_is_route_not_full_pass() -> None:
    initial = _recovery_state(phi=0.80, theta=0.45, p=0.90)
    exit_state = _recovery_state(phi=0.74, theta=0.42, p=0.80)
    exit_state[0] = 5.30

    transition = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.6,
            "ceiling_margin_m": 1.0,
            "initial_state_vector": initial.tolist(),
            "exit_state_vector": exit_state.tolist(),
        }
    )

    assert transition["exit_class"] == "boundary_near"
    assert transition["transition_chain_compatible"] is False


def test_boundary_near_uses_front_time_margin_not_static_distance_only() -> None:
    state = np.zeros(15, dtype=float)
    state[0] = 5.05
    state[1] = 2.20
    state[2] = 1.20
    state[5] = 0.0
    state[6] = 6.2

    assert classify_state(state) == "boundary_near"


def test_boundary_near_time_margin_ignores_rear_wall_for_forward_flight() -> None:
    state = np.zeros(15, dtype=float)
    state[0] = 1.52
    state[1] = 2.20
    state[2] = 1.20
    state[5] = 0.0
    state[6] = 4.0

    assert classify_state(state) == "inflight_stable"


def test_turn_intent_metrics_are_signed_by_turn_family() -> None:
    initial = np.zeros(15, dtype=float)
    initial[2] = 1.0
    exit_left = initial.copy()
    exit_left[3] = -0.06
    exit_left[9] = -0.45
    exit_left[1] = -0.04
    exit_right = initial.copy()
    exit_right[3] = 0.06
    exit_right[9] = 0.45
    exit_right[1] = 0.04

    left = turn_intent_row_fields(
        {
            "primitive_id": "mild_turn_left",
            "initial_state_vector": initial.tolist(),
            "exit_state_vector": exit_left.tolist(),
        }
    )
    right = turn_intent_row_fields(
        {
            "primitive_id": "mild_turn_right",
            "initial_state_vector": initial.tolist(),
            "exit_state_vector": exit_right.tolist(),
        }
    )

    assert left["turn_intent_correct_sign"] is True
    assert right["turn_intent_correct_sign"] is True
    assert left["turn_signed_bank_delta_rad"] > 0.0
    assert right["turn_signed_bank_delta_rad"] > 0.0
    assert left["turn_signed_exit_roll_rate_rad_s"] > 0.0
    assert right["turn_signed_exit_roll_rate_rad_s"] > 0.0
    assert left["turn_signed_lateral_displacement_m"] > 0.0
    assert right["turn_signed_lateral_displacement_m"] > 0.0
    assert left["turn_intent_roll_rate_score"] > 0.0
    assert right["turn_intent_roll_rate_score"] > 0.0
    assert left["turn_intent_score"] > 0.0
    assert right["turn_intent_score"] > 0.0


def test_r8_blocks_old_w3_summary_without_transition_gate() -> None:
    survived = pd.DataFrame(
        [
            {
                "primitive_variant_id": "v0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "launch_gate",
                "finite_horizon_s": 0.1,
                "controller_input_slots_per_primitive": 5,
                "controller_input_update_period_s": 0.02,
                "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
            }
        ]
    )

    assert _survived_frame_blocked_reason(survived) == "w3_survivor_summary_missing_transition_chain_compatible_rate"


def test_governor_rejects_zero_transition_success() -> None:
    representative = {
        "compact_library_id": "lib_v0",
        "primitive_variant_id": "v0",
        "entry_role": "transition_object",
        "transition_entry_class": "launch_gate",
        "controller_id": "c0",
        "K_gain_checksum": "k",
        "augmented_A_checksum": "a",
        "augmented_B_checksum": "b",
        "augmented_gain_checksum": "g",
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    outcome = {
        "transition_success_probability": 0.0,
        "transition_exit_classes_seen": "boundary_near",
        "hard_failure_risk": 0.0,
        "sample_count": 10,
    }
    context = {
        "start_state_family": "launch_gate",
        "current_state_class": "launch_gate",
        "floor_margin_m": 0.5,
        "ceiling_margin_m": 1.0,
        "latency_case": "nominal",
    }

    assert (
        governor_rejection_reason(
            representative=representative,
            outcome=outcome,
            context=context,
            governor_mode="continuation_mode",
        )
        == "transition_success_probability_zero"
    )


def test_governor_treats_lqr_speed_bin_mismatch_as_audit_metadata_not_rejection() -> None:
    representative = {
        "compact_library_id": "lib_v0",
        "primitive_variant_id": "v0",
        "entry_role": "transition_object",
        "transition_entry_class": "launch_gate",
        "local_lqr_speed_bin_id": "speed_bin_5p0_m_s",
        "controller_id": "c0",
        "K_gain_checksum": "k",
        "augmented_A_checksum": "a",
        "augmented_B_checksum": "b",
        "augmented_gain_checksum": "g",
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
    }
    outcome = {
        "transition_success_probability": 1.0,
        "transition_exit_classes_seen": "post_launch_degraded",
        "hard_failure_risk": 0.0,
        "sample_count": 10,
    }
    context = {
        "start_state_family": "launch_gate",
        "current_state_class": "launch_gate",
        "current_speed_m_s": 3.0,
        "floor_margin_m": 0.5,
        "ceiling_margin_m": 1.0,
        "latency_case": "nominal",
    }

    assert (
        governor_rejection_reason(
            representative=representative,
            outcome=outcome,
            context=context,
            governor_mode="continuation_mode",
        )
        == ""
    )


def _recovery_state(*, phi: float, theta: float, p: float) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = 3.0
    state[1] = 2.0
    state[2] = 1.0
    state[3] = phi
    state[4] = theta
    state[5] = 0.0
    state[6] = 4.0
    state[9] = p
    return state
