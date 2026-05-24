from __future__ import annotations

import numpy as np

from env_ctx import EnvironmentMetadata, build_environment_context
from lqr_controller import LQR_SYNTHESIS_SOLVED, lqr_controller_for_primitive_id
from prim_cat import active_primitive_catalogue, primitive_by_id
from prim_ctrl import PrimitiveControlContext, lqr_mode_for_primitive, primitive_lqr_command
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
        metadata=EnvironmentMetadata(environment_id="W0_lqr", fan_count=0),
        latency_case="none",
    )


def test_lqr_commands_are_bounded_for_all_active_primitives() -> None:
    state = _state()
    context = _context(state)

    for primitive in active_primitive_catalogue():
        controller = lqr_controller_for_primitive_id(primitive.primitive_id)
        command = primitive_lqr_command(
            primitive,
            PrimitiveControlContext(
                state_vector=state,
                environment_context=context,
                time_in_primitive_s=0.0,
            ),
            controller,
        )

        assert primitive.controller_family == "lqr"
        assert lqr_mode_for_primitive(primitive) == "predictor_compensated_augmented_discrete_lqr"
        assert command.primitive_id == primitive.primitive_id
        assert command.controller_id == controller.controller_id
        assert np.asarray(command.command_norm).shape == (3,)
        assert np.all(np.asarray(command.command_norm) >= -1.0)
        assert np.all(np.asarray(command.command_norm) <= 1.0)
        assert np.asarray(command.command_rad).shape == (3,)
        assert controller.lqr_synthesis_status in {LQR_SYNTHESIS_SOLVED, "blocked_lqr_synthesis"}


def test_recovery_lqr_has_reduced_order_audit_metadata() -> None:
    controller = lqr_controller_for_primitive_id("recovery")

    assert controller.primitive_id == primitive_by_id("recovery").primitive_id
    assert controller.reduced_order_lqr is True
    assert controller.zero_position_gain_expansion_status == "zero_position_gains_verified"
    assert controller.reduced_state_size == 12
    assert controller.full_state_size == 15
    assert controller.lqr_gain_checksum
