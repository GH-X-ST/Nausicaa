from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from env_ctx import EnvironmentMetadata, build_environment_context
from prim_cat import primitive_by_id
from prim_roll import (
    OUTCOME_CLASSES,
    RolloutConfig,
    rollout_evidence_row,
    simulate_primitive_rollout,
)
from state_contract import STATE_INDEX, STATE_SIZE


@dataclass(frozen=True)
class ConstantWind:
    value_m_s: float
    name: str = "rollout_constant_wind"
    source: str = "unit_test"

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        return np.column_stack(
            [
                np.zeros(points.shape[0]),
                np.zeros(points.shape[0]),
                np.full(points.shape[0], float(self.value_m_s)),
            ]
        )


def _state(*, x_w_m: float = 2.0, u_m_s: float = 5.8) -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = x_w_m
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = u_m_s
    return state


def _context(state: np.ndarray, wind_value_m_s: float = 0.6):
    return build_environment_context(
        state,
        wind_field=ConstantWind(wind_value_m_s),
        metadata=EnvironmentMetadata(
            environment_id="W1_rollout",
            fan_count=1,
            fan_positions_m=((4.2, 2.4),),
            fan_power_scales=(1.0,),
            updraft_model_id="rollout_constant_wind",
        ),
        latency_case="nominal",
    )


def test_rollout_evidence_row_contains_required_fields_and_claim_boundary() -> None:
    state = _state()
    evidence = simulate_primitive_rollout(
        rollout_id="rollout_000",
        initial_state=state,
        context=_context(state),
        primitive=primitive_by_id("lift_dwell_arc"),
        config=RolloutConfig(W_layer="W1"),
    )
    row = rollout_evidence_row(evidence)

    assert set(row) >= {
        "rollout_id",
        "environment_id",
        "W_layer",
        "initial_state_vector",
        "context_feature_vector",
        "primitive_id",
        "evidence_role",
        "continuation_valid",
        "episode_terminal_useful",
        "continuation_status",
        "episode_terminal_status",
        "outcome_class",
        "claim_status",
    }
    assert row["outcome_class"] in OUTCOME_CLASSES
    assert row["claim_status"] == "simulation_only"
    assert row["evidence_role"] == "interface_smoke"
    assert row["accepted"] is True
    assert "boundary_terminal" not in OUTCOME_CLASSES
    assert isinstance(row["continuation_valid"], bool)
    assert isinstance(row["episode_terminal_useful"], bool)


def test_low_speed_state_is_blocked_as_evidence_not_erased() -> None:
    state = _state(u_m_s=2.0)
    evidence = simulate_primitive_rollout(
        rollout_id="rollout_low_speed",
        initial_state=state,
        context=_context(state),
        primitive=primitive_by_id("glide"),
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.accepted is False
    assert evidence.failure_label == "speed_low"


def test_wall_margin_violation_is_retained_as_boundary_terminal() -> None:
    state = _state(x_w_m=1.21)
    evidence = simulate_primitive_rollout(
        rollout_id="rollout_wall",
        initial_state=state,
        context=_context(state),
        primitive=primitive_by_id("mild_turn_left"),
        config=RolloutConfig(W_layer="W1"),
    )

    assert evidence.outcome_class in {"weak", "failed"}
    assert evidence.outcome_class != "boundary_terminal"
    assert evidence.episode_terminal_status in {"not_terminal", "episode_terminal_useful"}
    assert evidence.continuation_status == "not_continuation_valid"
    assert evidence.continuation_valid is False
    assert evidence.termination_cause == "wall_boundary_exit_retained"
    assert evidence.failure_label == "xy_boundary_terminal"


def test_unknown_claim_status_is_rejected() -> None:
    primitive = primitive_by_id("glide")
    unsafe = primitive.__class__(**{**primitive.__dict__, "claim_status": "hardware_ready"})

    with pytest.raises(ValueError, match="simulation_only"):
        simulate_primitive_rollout(
            rollout_id="bad_claim",
            initial_state=_state(),
            context=_context(_state()),
            primitive=unsafe,
        )
