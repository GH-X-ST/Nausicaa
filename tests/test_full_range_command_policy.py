from __future__ import annotations

import numpy as np

from latency import (
    AGGREGATE_LIMITS,
    CommandToSurfaceConfig,
    CommandToSurfaceLayer,
    LatencyEnvelope,
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


def test_command_layer_does_not_apply_seventy_percent_cap() -> None:
    layer = CommandToSurfaceLayer(
        CommandToSurfaceConfig(mode="nominal", quantise=False, use_onset_delay=False),
        LatencyEnvelope(),
    )
    layer.reset(np.zeros(3))
    desired = np.array(
        [
            command_norm_to_angle(1.0, AGGREGATE_LIMITS["delta_a"]),
            command_norm_to_angle(-1.0, AGGREGATE_LIMITS["delta_e"]),
            command_norm_to_angle(1.0, AGGREGATE_LIMITS["delta_r"]),
        ],
        dtype=float,
    )
    target = layer.apply(desired)
    np.testing.assert_allclose(target, desired)
    fields = layer.log_fields()
    assert np.isclose(fields["command_norm_a"], 1.0)
    assert np.isclose(fields["command_norm_e"], -1.0)
    assert np.isclose(fields["command_norm_r"], 1.0)
