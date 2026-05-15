from __future__ import annotations

import numpy as np

from closed_loop_trajectory import ClosedLoopTrajectoryPrimitive
from primitive import build_primitive_context


def test_closed_loop_trajectory_validates_shapes_and_metadata() -> None:
    times = np.array([0.0, 0.1, 0.2])
    x_ref = np.zeros((3, 15), dtype=float)
    x_ref[:, 6] = 5.0
    u_ff = np.zeros((3, 3), dtype=float)
    gains = np.zeros((3, 3, 15), dtype=float)
    primitive = ClosedLoopTrajectoryPrimitive(
        name="shape_test",
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        k_feedback=gains,
        phase_labels=("entry", "pitch_brake", "exit_check"),
    )

    assert primitive.duration_s == 0.2
    assert primitive.metadata["primitive_family"] == "aggressive_high_incidence_reversal"
    assert primitive.metadata["is_real_flight_claim"] is False


def test_closed_loop_trajectory_command_is_finite() -> None:
    times = np.array([0.0, 0.1])
    x_ref = np.zeros((2, 15), dtype=float)
    x_ref[:, 6] = 5.0
    u_ff = np.zeros((2, 3), dtype=float)
    gains = np.zeros((2, 3, 15), dtype=float)
    primitive = ClosedLoopTrajectoryPrimitive(
        name="command_test",
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        k_feedback=gains,
        phase_labels=("entry", "exit_check"),
    )
    context = build_primitive_context(x_trim=x_ref[0], u_trim=np.zeros(3))

    command = primitive.command(0.05, x_ref[0], context)

    assert command.shape == (3,)
    assert np.all(np.isfinite(command))

