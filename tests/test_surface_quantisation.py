from __future__ import annotations

import numpy as np

from latency import (
    COMMAND_LEVELS,
    AGGREGATE_LIMITS,
    CommandToSurfaceConfig,
    CommandToSurfaceLayer,
    LatencyEnvelope,
    angle_to_command_norm,
    command_norm_to_angle,
    quantise_command_norm,
)


def test_command_levels_are_canonical() -> None:
    assert np.allclose(
        COMMAND_LEVELS,
        np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )


def test_asymmetric_aileron_quantisation() -> None:
    norm_pos = angle_to_command_norm(np.deg2rad(10.0), AGGREGATE_LIMITS["delta_a"])
    norm_neg = angle_to_command_norm(np.deg2rad(-10.0), AGGREGATE_LIMITS["delta_a"])
    assert quantise_command_norm(norm_pos) == 0.4
    assert quantise_command_norm(norm_neg) == -0.4
    assert np.isclose(
        np.rad2deg(command_norm_to_angle(0.4, AGGREGATE_LIMITS["delta_a"])),
        8.8,
    )
    assert np.isclose(
        np.rad2deg(command_norm_to_angle(-0.4, AGGREGATE_LIMITS["delta_a"])),
        -10.4,
    )


def test_onset_delay_holds_initial_target() -> None:
    layer = CommandToSurfaceLayer(
        CommandToSurfaceConfig(mode="nominal", quantise=True, use_onset_delay=True),
        LatencyEnvelope(onset_latency_s=0.075, command_dt_s=0.02),
    )
    layer.reset(np.zeros(3))
    outputs = [layer.apply(np.deg2rad([10.0, 0.0, 0.0]))[0] for _ in range(4)]
    assert np.allclose(outputs, 0.0)
    assert np.isclose(
        np.rad2deg(layer.apply(np.deg2rad([10.0, 0.0, 0.0]))[0]),
        8.8,
    )
