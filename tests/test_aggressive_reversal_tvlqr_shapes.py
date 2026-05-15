from __future__ import annotations

import numpy as np

from aggressive_reversal_tvlqr import solve_aggressive_discrete_tvlqr


def test_aggressive_reversal_tvlqr_shapes_are_node_aligned() -> None:
    a_d = np.repeat(np.eye(15, dtype=float).reshape(1, 15, 15), 3, axis=0)
    b_d = np.zeros((3, 15, 3), dtype=float)
    b_d[:, 12:15, :] = np.eye(3)

    gains, s_mats = solve_aggressive_discrete_tvlqr(
        a_d=a_d,
        b_d=b_d,
        q_diag=(1.0,) * 15,
        r_diag=(2.0, 2.0, 2.0),
        phase_labels=("entry", "pitch_brake", "yaw_roll_redirect", "exit_check"),
    )

    assert gains.shape == (4, 3, 15)
    assert s_mats.shape == (4, 15, 15)
    assert np.all(np.isfinite(gains))
    assert np.all(np.isfinite(s_mats))

