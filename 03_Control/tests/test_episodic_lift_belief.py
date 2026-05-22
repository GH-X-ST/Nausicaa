from __future__ import annotations

import numpy as np

from episodic_lift_belief import (
    BELIEF_LAMBDA_VALUES,
    LiftObservation,
    belief_snapshot_row,
    initial_belief,
    query_belief_features,
    update_belief,
)
from state_contract import STATE_INDEX, STATE_SIZE


def _state() -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.5
    state[STATE_INDEX["u"]] = 5.5
    return state


def test_belief_updates_for_retained_lambda_values() -> None:
    for lambda_value in BELIEF_LAMBDA_VALUES:
        belief = initial_belief(lambda_value=lambda_value)
        updated = update_belief(
            belief,
            LiftObservation(x_w_m=2.0, y_w_m=2.0, lift_evidence_m_s=0.6),
        )
        features = query_belief_features(_state(), updated)

        assert updated.update_count == 1
        assert features["belief_local_lift_m_s"] >= 0.0
        assert belief_snapshot_row(updated)["lambda_value"] == lambda_value
