from __future__ import annotations

import numpy as np
import pytest

from command_contract import (
    as_surface_command_rad,
    clip_normalised_command,
    normalised_command_to_surface_rad,
    surface_rad_to_normalised_command,
)


def test_normalised_command_clip_contract() -> None:
    command = np.asarray([1.2, -1.4, 0.5], dtype=float)

    np.testing.assert_allclose(clip_normalised_command(command), [1.0, -1.0, 0.5])


def test_normalised_surface_conversion_round_trip_stays_inside_limits() -> None:
    command = np.asarray([1.0, -0.5, 0.25], dtype=float)

    surface_rad = normalised_command_to_surface_rad(command)
    recovered = surface_rad_to_normalised_command(surface_rad)

    np.testing.assert_allclose(recovered, command)


def test_surface_command_rejects_out_of_limit_radian_values() -> None:
    with pytest.raises(ValueError, match="outside calibrated aggregate limits"):
        as_surface_command_rad(np.asarray([np.deg2rad(40.0), 0.0, 0.0], dtype=float))
