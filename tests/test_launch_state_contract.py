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

from linearisation import STATE_INDEX  # noqa: E402
from scenarios import nominal_hand_launch_state  # noqa: E402


def _trim_like_state() -> np.ndarray:
    x_trim = np.zeros(15, dtype=float)
    x_trim[STATE_INDEX["theta"]] = np.deg2rad(-2.5)
    x_trim[STATE_INDEX["u"]] = 6.45
    x_trim[STATE_INDEX["v"]] = 0.02
    x_trim[STATE_INDEX["w"]] = 0.35
    return x_trim


def test_nominal_hand_launch_state_uses_contract_position() -> None:
    launch = nominal_hand_launch_state(_trim_like_state())

    assert launch.shape == (15,)
    assert launch[STATE_INDEX["x_w"]] == pytest.approx(1.2)
    assert launch[STATE_INDEX["y_w"]] == pytest.approx(0.4)
    assert launch[STATE_INDEX["z_w"]] == pytest.approx(1.5)


def test_nominal_hand_launch_state_preserves_trim_motion_state() -> None:
    x_trim = _trim_like_state()
    launch = nominal_hand_launch_state(x_trim)

    assert np.isfinite(launch).all()
    assert launch[STATE_INDEX["theta"]] == pytest.approx(x_trim[STATE_INDEX["theta"]])
    trim_speed = np.linalg.norm(x_trim[6:9])
    launch_speed = np.linalg.norm(launch[6:9])
    assert launch_speed == pytest.approx(trim_speed)


def test_nominal_hand_launch_state_does_not_mutate_trim_state() -> None:
    x_trim = _trim_like_state()
    before = x_trim.copy()

    nominal_hand_launch_state(x_trim)

    np.testing.assert_allclose(x_trim, before)
