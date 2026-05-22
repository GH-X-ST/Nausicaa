from __future__ import annotations

import numpy as np

from command_contract import NORMALISED_COMMAND_MAX, NORMALISED_COMMAND_MIN
from env_ctx import EnvironmentMetadata, build_environment_context
from prim_cat import active_primitive_catalogue, primitive_by_id
from prim_ctrl import PrimitiveControlContext, feedback_mode_for_primitive, primitive_command_norm
from state_contract import STATE_INDEX, STATE_SIZE


def _state() -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = 5.8
    return state


def _context(state: np.ndarray):
    return build_environment_context(
        state,
        wind_field=None,
        metadata=EnvironmentMetadata(environment_id="W0_feedback", fan_count=0),
        latency_case="none",
    )


def test_feedback_commands_are_bounded_for_all_active_primitives() -> None:
    state = _state()
    context = _context(state)

    for primitive in active_primitive_catalogue():
        command = primitive_command_norm(
            primitive,
            PrimitiveControlContext(
                state_vector=state,
                environment_context=context,
                time_in_primitive_s=0.0,
            ),
        )

        assert command.primitive_id == primitive.primitive_id
        assert command.feedback_mode == feedback_mode_for_primitive(primitive)
        command_norm = np.asarray(command.command_norm, dtype=float)
        command_rad = np.asarray(command.command_rad, dtype=float)
        assert command_norm.shape == (3,)
        assert np.all(command_norm >= NORMALISED_COMMAND_MIN)
        assert np.all(command_norm <= NORMALISED_COMMAND_MAX)
        assert command_rad.shape == (3,)


def test_recovery_uses_pitch_up_bias_when_speed_margin_is_low() -> None:
    state = _state()
    state[STATE_INDEX["u"]] = 3.1
    command = primitive_command_norm(
        primitive_by_id("recovery"),
        PrimitiveControlContext(
            state_vector=state,
            environment_context=_context(state),
            time_in_primitive_s=0.0,
        ),
    )

    assert command.command_norm[1] > 0.0
