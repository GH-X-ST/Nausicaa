from __future__ import annotations

import numpy as np
import pytest

from feedback import limit_aggregate_command
from latency import (
    AGGREGATE_LIMITS,
    SURFACE_LIMITS,
    aggregate_targets_to_surface_degrees,
    angle_to_command_norm,
    command_norm_to_angle,
)


def test_calibrated_surface_endpoint_signs_are_preserved() -> None:
    expected = {
        "Aileron_L": ((+1.0, +22.0), (-1.0, -26.0)),
        "Aileron_R": ((+1.0, -22.0), (-1.0, +26.0)),
        "Rudder": ((+1.0, +28.0), (-1.0, -35.0)),
        "Elevator": ((+1.0, +22.0), (-1.0, -30.0)),
    }
    for surface_name, cases in expected.items():
        for command_norm, angle_deg in cases:
            actual_deg = np.rad2deg(
                command_norm_to_angle(command_norm, SURFACE_LIMITS[surface_name])
            )
            assert actual_deg == pytest.approx(angle_deg)


def test_inverse_surface_mapping_handles_right_aileron_signs() -> None:
    assert angle_to_command_norm(np.deg2rad(+26.0), SURFACE_LIMITS["Aileron_R"]) == -1.0
    assert angle_to_command_norm(np.deg2rad(-22.0), SURFACE_LIMITS["Aileron_R"]) == +1.0


def test_positive_aggregate_aileron_expands_to_left_down_right_up() -> None:
    target = np.array([np.deg2rad(12.0), 0.0, 0.0], dtype=float)
    surfaces = aggregate_targets_to_surface_degrees(target)

    assert surfaces["aileron_l_deg"] > 0.0
    assert surfaces["aileron_r_deg"] < 0.0


def test_aggregate_command_path_uses_full_calibrated_range() -> None:
    lower = np.deg2rad(
        [
            AGGREGATE_LIMITS["delta_a"].negative_deg,
            AGGREGATE_LIMITS["delta_e"].negative_deg,
            AGGREGATE_LIMITS["delta_r"].negative_deg,
        ]
    )
    upper = np.deg2rad(
        [
            AGGREGATE_LIMITS["delta_a"].positive_deg,
            AGGREGATE_LIMITS["delta_e"].positive_deg,
            AGGREGATE_LIMITS["delta_r"].positive_deg,
        ]
    )

    np.testing.assert_allclose(limit_aggregate_command(10.0 * np.ones(3)), upper)
    np.testing.assert_allclose(limit_aggregate_command(-10.0 * np.ones(3)), lower)
