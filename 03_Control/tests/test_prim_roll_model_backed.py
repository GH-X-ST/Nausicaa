from __future__ import annotations

import numpy as np

from env_ctx import EnvironmentMetadata, build_environment_context
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding
from prim_cat import primitive_by_id
from prim_roll import RolloutConfig, simulate_primitive_rollout
from state_contract import STATE_INDEX, STATE_SIZE


def _state(*, x_w_m: float = 2.0, z_w_m: float = 1.6, u_m_s: float = 5.8) -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = x_w_m
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = z_w_m
    state[STATE_INDEX["u"]] = u_m_s
    return state


def _context_and_wind(state: np.ndarray):
    metadata = EnvironmentMetadata(
        environment_id="W1_model_backed",
        fan_count=1,
        fan_positions_m=((4.2, 2.4),),
        fan_power_scales=(1.0,),
        updraft_model_id="single_gaussian_var",
    )
    binding = resolve_surrogate_binding("W1", metadata)
    wind = wind_field_for_binding(binding)
    context = build_environment_context(
        state,
        wind_field=wind,
        metadata=metadata,
        latency_case="nominal",
        surrogate_binding=binding,
    )
    return context, wind


def test_model_backed_rollout_is_distinct_from_smoke_and_finite() -> None:
    state = _state()
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_000",
        episode_id="episode_000",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed"),
        wind_field=wind,
    )

    assert evidence.rollout_backend == "model_backed"
    assert evidence.surrogate_binding_status == "ready"
    assert evidence.trajectory_integrity_status == "finite_model_backed"
    assert np.isfinite(evidence.energy_residual_m)
    assert np.isfinite(evidence.floor_margin_m)
    assert np.isfinite(evidence.ceiling_margin_m)


def test_model_backed_low_speed_initial_state_is_blocked() -> None:
    state = _state(u_m_s=2.0)
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_low_speed",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.failure_label == "speed_low"


def test_model_backed_wall_exit_is_retained_as_rejected_row() -> None:
    state = _state(x_w_m=1.0)
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_wall",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("mild_turn_left"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "rejected"
    assert evidence.failure_label == "wall_violation"
    assert evidence.minimum_wall_margin_m < 0.0
