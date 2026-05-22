from __future__ import annotations

import numpy as np

from env_ctx import EnvironmentMetadata, build_environment_context
from prim_cat import primitive_by_id
from prim_features import (
    PRIMITIVE_FEATURE_SCHEMA_VERSION,
    primitive_feature_record,
    primitive_feature_row,
)
from state_contract import STATE_INDEX, STATE_SIZE


def _state(*, u_m_s: float = 5.8) -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = u_m_s
    return state


def _context(state: np.ndarray):
    return build_environment_context(
        state,
        wind_field=None,
        metadata=EnvironmentMetadata(environment_id="W0_features", fan_count=0),
        latency_case="none",
    )


def test_feature_schema_uses_state_context_primitive_and_mode() -> None:
    state_a = _state(u_m_s=5.0)
    state_b = _state(u_m_s=6.5)
    primitive_a = primitive_by_id("glide")
    primitive_b = primitive_by_id("lift_dwell_arc")

    record_a = primitive_feature_record(
        state=state_a,
        context=_context(state_a),
        primitive=primitive_a,
        governor_mode="continuation",
    )
    record_b = primitive_feature_record(
        state=state_b,
        context=_context(state_b),
        primitive=primitive_b,
        governor_mode="terminal_episode",
    )

    assert record_a.feature_schema_version == PRIMITIVE_FEATURE_SCHEMA_VERSION
    assert record_a.feature_vector != record_b.feature_vector
    assert len(record_a.feature_names) == len(record_a.feature_vector)
    assert primitive_feature_row(record_a)["feature_schema_version"] == PRIMITIVE_FEATURE_SCHEMA_VERSION
