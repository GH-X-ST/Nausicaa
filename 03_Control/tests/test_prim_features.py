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
    state[STATE_INDEX["p"]] = 0.1
    state[STATE_INDEX["delta_a"]] = 0.05
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
        start_state_family="inflight_recovery_edge",
        previous_primitive_status="recovery_edge",
        synthetic_time_since_launch_s=1.2,
    )

    assert record_a.feature_schema_version == PRIMITIVE_FEATURE_SCHEMA_VERSION
    assert record_a.feature_vector != record_b.feature_vector
    assert len(record_a.feature_names) == len(record_a.feature_vector)
    assert primitive_feature_row(record_a)["feature_schema_version"] == PRIMITIVE_FEATURE_SCHEMA_VERSION
    assert "x_w_norm" not in record_a.feature_names
    assert "delta_a_norm" in record_a.feature_names


def test_feature_schema_is_sensitive_to_start_provenance_rates_and_surfaces() -> None:
    state_a = _state()
    state_b = _state()
    state_b[STATE_INDEX["p"]] = 0.4
    state_b[STATE_INDEX["delta_a"]] = -0.2
    primitive = primitive_by_id("glide")

    record_a = primitive_feature_record(
        state=state_a,
        context=_context(state_a),
        primitive=primitive,
        start_state_family="launch_gate",
        previous_primitive_status="episode_start",
    )
    record_b = primitive_feature_record(
        state=state_b,
        context=_context(state_b),
        primitive=primitive,
        start_state_family="inflight_nominal",
        previous_primitive_status="clean_exit",
        synthetic_time_since_launch_s=2.0,
    )

    assert record_a.feature_vector != record_b.feature_vector
