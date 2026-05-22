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
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_feedback"),
        wind_field=wind,
    )

    assert evidence.rollout_backend == "model_backed_feedback"
    assert evidence.evidence_role == "feedback_rollout_candidate"
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
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_feedback"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.continuation_status == "blocked"
    assert evidence.failure_label == "speed_low"


def test_model_backed_nonfinite_initial_state_is_blocked() -> None:
    context, wind = _context_and_wind(_state())
    state = _state()
    state[STATE_INDEX["u"]] = np.nan

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_nonfinite",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_feedback"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.entry_check_status == "nonfinite_initial_state"
    assert evidence.failure_label == "nonfinite_initial_state"


def test_model_backed_floor_initial_state_is_blocked() -> None:
    state = _state(z_w_m=0.2)
    context, wind = _context_and_wind(_state())

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_floor",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_feedback"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.failure_label == "initial_floor_violation"


def test_model_backed_wall_exit_is_retained_as_boundary_terminal_row() -> None:
    state = _state(x_w_m=6.55)
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_wall",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("mild_turn_left"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_feedback"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "boundary_terminal"
    assert evidence.failure_label == "xy_boundary_terminal"
    assert evidence.episode_terminal_status == "boundary_terminal"
    assert evidence.continuation_status == "not_continuation_valid"
    assert evidence.terminal_use_trainable is True
    assert evidence.minimum_wall_margin_m < 0.0


def test_command_template_rows_are_diagnostic_not_feedback_evidence() -> None:
    state = _state()
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="template_rollout_000",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(
            W_layer="W1",
            rollout_backend="model_backed_command_template",
        ),
        wind_field=wind,
    )

    assert evidence.rollout_backend == "model_backed_command_template"
    assert evidence.evidence_role == "diagnostic_model_rollout"
    assert evidence.feedback_mode == "command_template_diagnostic"
