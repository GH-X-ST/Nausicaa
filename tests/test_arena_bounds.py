from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "03_Control/02_Inner_Loop",
    "03_Control/04_Scenarios",
):
    path = REPO_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena import (  # noqa: E402
    ArenaConfig,
    safe_bounds,
    safety_margins,
    tracker_bounds,
)
from linearisation import STATE_INDEX  # noqa: E402


def _state_at(x_w: float, y_w: float, z_w: float) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[STATE_INDEX["x_w"]] = x_w
    state[STATE_INDEX["y_w"]] = y_w
    state[STATE_INDEX["z_w"]] = z_w
    return state


def test_true_safety_bounds_are_authoritative() -> None:
    assert safe_bounds(ArenaConfig()) == {
        "x_w": (1.2, 6.6),
        "y_w": (0.0, 4.4),
        "z_w": (0.0, 3.0),
    }


def test_tracker_limit_bounds_use_direct_lab_axes() -> None:
    assert tracker_bounds(ArenaConfig()) == {
        "x_w": (0.0, 8.0),
        "y_w": (0.0, 4.8),
        "z_w": (0.0, 3.5),
    }


def test_non_safety_axes_use_tracker_limit_box() -> None:
    assert safe_bounds(ArenaConfig(use_safe_volume=False)) == tracker_bounds(ArenaConfig())


def test_hand_launch_position_is_on_true_safety_x_boundary() -> None:
    margins = safety_margins(_state_at(x_w=1.2, y_w=0.4, z_w=1.5), ArenaConfig())

    assert margins["inside_tracker_limit"] is True
    assert margins["inside_safe_volume"] is True
    assert margins["x_margin_m"] == pytest.approx(0.0)
    assert margins["y_margin_m"] == pytest.approx(0.4)
    assert margins["tracker_x_margin_m"] == pytest.approx(1.2)
    assert margins["tracker_y_margin_m"] == pytest.approx(0.4)


def test_inside_tracker_but_outside_true_safety_is_not_safe() -> None:
    margins = safety_margins(_state_at(x_w=0.5, y_w=0.4, z_w=1.5), ArenaConfig())

    assert margins["inside_tracker_limit"] is True
    assert margins["inside_safe_volume"] is False


def test_outside_direct_tracker_limit_is_not_inside_tracker() -> None:
    margins = safety_margins(_state_at(x_w=8.1, y_w=0.4, z_w=1.5), ArenaConfig())

    assert margins["inside_tracker_limit"] is False
    assert margins["inside_safe_volume"] is False


def test_true_safety_margin_signs() -> None:
    margins = safety_margins(_state_at(x_w=1.2, y_w=2.2, z_w=1.5), ArenaConfig())

    assert margins["inside_safe_volume"] is True
    assert margins["min_wall_distance_m"] == pytest.approx(0.0)
    assert margins["floor_margin_m"] == pytest.approx(1.5)
    assert margins["ceiling_margin_m"] == pytest.approx(1.5)


def test_obsolete_centred_tracker_bounds_are_not_returned() -> None:
    obsolete_centred_bounds = {
        "x_w": (3.9 - 4.0, 3.9 + 4.0),
        "y_w": (2.2 - 2.4, 2.2 + 2.4),
        "z_w": (1.5 - 1.75, 1.5 + 1.75),
    }

    assert tracker_bounds(ArenaConfig()) != obsolete_centred_bounds
