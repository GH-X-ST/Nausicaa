from __future__ import annotations

import json

import numpy as np

from env_ctx import EnvironmentMetadata, build_environment_context, context_feature_vector
from prim_cat import primitive_by_id
from prim_model import fit_primitive_outcome_model
from prim_select import select_primitive
from state_contract import STATE_INDEX, STATE_SIZE


def _state(*, x_w_m: float = 2.0, u_m_s: float = 5.8) -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = x_w_m
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = u_m_s
    return state


def _context(state: np.ndarray):
    return build_environment_context(
        state,
        wind_field=None,
        metadata=EnvironmentMetadata(environment_id="W0_select", fan_count=0),
        latency_case="none",
    )


def _model_for(context):
    features = json.dumps(list(context_feature_vector(context)))
    return fit_primitive_outcome_model(
        [
            {
                "primitive_id": "glide",
                "evidence_role": "lqr_rollout_candidate",
                "context_feature_vector": features,
                "outcome_class": "accepted",
                "continuation_status": "continuation_success",
                "episode_terminal_status": "not_terminal",
                "episode_utility_label": "continuation_useful",
                "terminal_use_trainable": False,
                "energy_residual_m": 0.05,
                "lift_dwell_time_s": 0.0,
                "minimum_wall_margin_m": 0.9,
                "termination_cause": "controlled_finish",
            },
            {
                "primitive_id": "recovery",
                "evidence_role": "lqr_rollout_candidate",
                "context_feature_vector": features,
                "outcome_class": "weak",
                "continuation_status": "continuation_weak",
                "episode_terminal_status": "not_terminal",
                "episode_utility_label": "continuation_useful",
                "terminal_use_trainable": False,
                "energy_residual_m": 0.0,
                "lift_dwell_time_s": 0.0,
                "minimum_wall_margin_m": 0.7,
                "termination_cause": "weak_energy_result",
            },
        ],
        k_neighbours=1,
    )


def test_selector_returns_viable_best_primitive() -> None:
    context = _context(_state())
    result = select_primitive(
        context=context,
        model=_model_for(context),
        catalogue=(primitive_by_id("glide"), primitive_by_id("recovery")),
    )

    assert result.decision_status == "selected_viable_primitive"
    assert result.governor_mode == "continuation"
    assert result.selected_primitive_id == "glide"
    assert result.viable_count >= 1


def test_selector_rejects_high_wall_risk_without_dropping_candidate_log() -> None:
    context = _context(_state(x_w_m=1.21))
    result = select_primitive(
        context=context,
        model=_model_for(context),
        catalogue=(primitive_by_id("glide"), primitive_by_id("recovery")),
        min_wall_margin_m=0.25,
    )

    assert result.decision_status == "recovery_handoff_no_viable_primitive"
    assert result.selected_primitive_id == "safe_exit_or_recovery_handoff"
    assert result.candidate_count == 2
    assert all(decision.rejection_reason for decision in result.decisions)


def test_selector_modes_treat_boundary_terminal_differently() -> None:
    context = _context(_state())
    features = json.dumps(list(context_feature_vector(context)))
    model = fit_primitive_outcome_model(
        [
            {
                "primitive_id": "lift_dwell_arc",
                "evidence_role": "lqr_rollout_candidate",
                "context_feature_vector": features,
                "outcome_class": "boundary_terminal",
                "continuation_status": "not_continuation_valid",
                "episode_terminal_status": "boundary_terminal",
                "episode_utility_label": "terminal_useful",
                "terminal_use_trainable": True,
                "energy_residual_m": 0.1,
                "lift_dwell_time_s": 0.5,
                "minimum_wall_margin_m": -0.2,
                "termination_cause": "wall_boundary_exit_retained",
            }
        ],
        k_neighbours=1,
    )

    continuation = select_primitive(
        context=context,
        model=model,
        catalogue=(primitive_by_id("lift_dwell_arc"),),
        governor_mode="continuation",
    )
    terminal = select_primitive(
        context=context,
        model=model,
        catalogue=(primitive_by_id("lift_dwell_arc"),),
        governor_mode="terminal_episode",
    )

    assert continuation.decision_status == "recovery_handoff_no_viable_primitive"
    assert continuation.decisions[0].rejection_reason == (
        "predicted_boundary_terminal_not_continuation_valid"
    )
    assert terminal.decision_status == "selected_viable_primitive"
    assert terminal.selected_primitive_id == "lift_dwell_arc"
