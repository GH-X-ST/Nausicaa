from __future__ import annotations

import numpy as np

from latency import (
    AGGREGATE_LIMITS,
    angle_to_command_norm,
    command_norm_to_angle,
)


def test_python_command_policy_uses_full_normalised_range() -> None:
    for name, limit in AGGREGATE_LIMITS.items():
        assert np.isclose(angle_to_command_norm(command_norm_to_angle(1.0, limit), limit), 1.0)
        assert np.isclose(angle_to_command_norm(command_norm_to_angle(-1.0, limit), limit), -1.0)
        assert np.isclose(command_norm_to_angle(1.4, limit), command_norm_to_angle(1.0, limit))
        assert np.isclose(command_norm_to_angle(-1.4, limit), command_norm_to_angle(-1.0, limit))
        assert name in {"delta_a", "delta_e", "delta_r"}
