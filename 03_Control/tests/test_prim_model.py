from __future__ import annotations

import json

import numpy as np

from env_ctx import EnvironmentMetadata, build_environment_context, context_feature_vector
from prim_cat import primitive_by_id
from prim_model import fit_primitive_outcome_model, predict_primitive_outcome
from state_contract import STATE_INDEX, STATE_SIZE


def _state() -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = 5.8
    return state


def _context():
    return build_environment_context(
        _state(),
        wind_field=None,
        metadata=EnvironmentMetadata(environment_id="W0_model", fan_count=0),
        latency_case="none",
    )


def test_table_model_predicts_nearest_primitive_outcome() -> None:
    context = _context()
    features = json.dumps(list(context_feature_vector(context)))
    model = fit_primitive_outcome_model(
        [
            {
                "primitive_id": "glide",
                "evidence_role": "feedback_rollout_candidate",
                "context_feature_vector": features,
                "outcome_class": "accepted",
                "continuation_status": "continuation_success",
                "episode_terminal_status": "not_terminal",
                "episode_utility_label": "continuation_useful",
                "terminal_use_trainable": False,
                "energy_residual_m": 0.10,
                "lift_dwell_time_s": 0.2,
                "minimum_wall_margin_m": 1.0,
                "termination_cause": "controlled_finish",
            },
            {
                "primitive_id": "glide",
                "evidence_role": "feedback_rollout_candidate",
                "context_feature_vector": features,
                "outcome_class": "weak",
                "continuation_status": "continuation_weak",
                "episode_terminal_status": "not_terminal",
                "episode_utility_label": "continuation_useful",
                "terminal_use_trainable": False,
                "energy_residual_m": 0.0,
                "lift_dwell_time_s": 0.1,
                "minimum_wall_margin_m": 0.8,
                "termination_cause": "weak_energy_result",
            },
        ],
        k_neighbours=2,
    )

    prediction = predict_primitive_outcome(model, context, primitive_by_id("glide"))

    assert model.fitted_row_count == 2
    assert prediction.model_backend == "auditable_knn_table"
    assert prediction.probability_accepted == 0.5
    assert prediction.probability_weak == 0.5
    assert prediction.probability_continuation_success == 1.0
    assert prediction.predicted_energy_residual_m == 0.05
    assert prediction.neighbour_distance == 0.0


def test_unfitted_model_returns_uncertain_prior() -> None:
    model = fit_primitive_outcome_model([])

    prediction = predict_primitive_outcome(model, _context(), primitive_by_id("glide"))

    assert prediction.probability_blocked == 0.25
    assert prediction.uncertainty == float("inf")


def test_default_training_excludes_command_template_diagnostics() -> None:
    context = _context()
    features = json.dumps(list(context_feature_vector(context)))
    rows = [
        {
            "primitive_id": "glide",
            "evidence_role": "diagnostic_model_rollout",
            "context_feature_vector": features,
            "outcome_class": "accepted",
            "continuation_status": "continuation_success",
            "energy_residual_m": 1.0,
            "lift_dwell_time_s": 1.0,
            "minimum_wall_margin_m": 1.0,
            "termination_cause": "diagnostic_template",
        },
        {
            "primitive_id": "glide",
            "evidence_role": "feedback_rollout_candidate",
            "context_feature_vector": features,
            "outcome_class": "boundary_terminal",
            "continuation_status": "not_continuation_valid",
            "episode_terminal_status": "boundary_terminal",
            "episode_utility_label": "terminal_useful",
            "terminal_use_trainable": True,
            "energy_residual_m": 0.2,
            "lift_dwell_time_s": 0.4,
            "minimum_wall_margin_m": -0.1,
            "termination_cause": "wall_boundary_exit_retained",
        },
    ]

    model = fit_primitive_outcome_model(rows, k_neighbours=2)
    prediction = predict_primitive_outcome(model, context, primitive_by_id("glide"))

    assert model.fitted_row_count == 1
    assert prediction.probability_boundary_terminal == 1.0
    assert prediction.probability_continuation_success == 0.0
    assert prediction.probability_terminal_useful == 1.0
    assert prediction.training_evidence_roles == "feedback_rollout_candidate"
