from __future__ import annotations

import pandas as pd

from run_post_w3_library_size_study import _survived_frame_blocked_reason
from transition_labels import classify_transition, transition_is_chain_compatible
from viability_governor import governor_rejection_reason


def test_launch_boundary_exit_is_not_chain_compatible() -> None:
    transition = classify_transition(
        {
            "entry_role": "launch_capable",
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
            "entry_role": "launch_capable",
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
        entry_role="terminal_or_recovery",
        entry_class="recoverable_degraded",
        exit_class="inflight_stable",
    )


def test_r8_blocks_old_w3_summary_without_transition_gate() -> None:
    survived = pd.DataFrame(
        [
            {
                "primitive_variant_id": "v0",
                "primitive_id": "launch_capture_glide_stabilise",
                "entry_role": "launch_capable",
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
        "entry_role": "launch_capable",
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
