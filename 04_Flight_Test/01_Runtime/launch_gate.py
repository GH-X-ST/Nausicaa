from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flight_config import CONTROLLER_ROOT

import sys

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from state_contract import STATE_INDEX, as_state_vector  # noqa: E402


LAUNCH_GATE_ROLL_LIMIT_DEG = 20.0
LAUNCH_GATE_PITCH_MIN_DEG = -10.0
LAUNCH_GATE_PITCH_MAX_DEG = 20.0
LAUNCH_GATE_YAW_LIMIT_DEG = 20.0
LAUNCH_GATE_SPEED_MIN_M_S = 3.0
LAUNCH_GATE_SPEED_MAX_M_S = 8.0
LAUNCH_GATE_X_W_M = (1.2, 1.4)
LAUNCH_GATE_Y_W_M = (1.8, 2.2)
LAUNCH_GATE_Z_W_M = (1.5, 1.9)


@dataclass(frozen=True)
class LaunchGateStatus:
    approved: bool
    reason: str
    x_w_m: float
    y_w_m: float
    z_w_m: float
    phi_deg: float
    theta_deg: float
    psi_deg: float
    speed_m_s: float


def evaluate_launch_gate(state: np.ndarray) -> LaunchGateStatus:
    """Return whether the measured state matches the R5 launch-gate envelope."""

    x = as_state_vector(state)
    x_w = float(x[STATE_INDEX["x_w"]])
    y_w = float(x[STATE_INDEX["y_w"]])
    z_w = float(x[STATE_INDEX["z_w"]])
    phi = float(x[STATE_INDEX["phi"]])
    theta = float(x[STATE_INDEX["theta"]])
    psi = float(x[STATE_INDEX["psi"]])
    speed = float(np.linalg.norm(x[STATE_INDEX["u"] : STATE_INDEX["w"] + 1]))

    checks = (
        (LAUNCH_GATE_X_W_M[0] <= x_w <= LAUNCH_GATE_X_W_M[1], "x_w_outside_launch_gate"),
        (LAUNCH_GATE_Y_W_M[0] <= y_w <= LAUNCH_GATE_Y_W_M[1], "y_w_outside_launch_gate"),
        (LAUNCH_GATE_Z_W_M[0] <= z_w <= LAUNCH_GATE_Z_W_M[1], "z_w_outside_launch_gate"),
        (
            np.deg2rad(-LAUNCH_GATE_ROLL_LIMIT_DEG) <= phi <= np.deg2rad(LAUNCH_GATE_ROLL_LIMIT_DEG),
            "roll_outside_launch_gate",
        ),
        (
            np.deg2rad(LAUNCH_GATE_PITCH_MIN_DEG) <= theta <= np.deg2rad(LAUNCH_GATE_PITCH_MAX_DEG),
            "pitch_outside_launch_gate",
        ),
        (
            np.deg2rad(-LAUNCH_GATE_YAW_LIMIT_DEG) <= psi <= np.deg2rad(LAUNCH_GATE_YAW_LIMIT_DEG),
            "yaw_outside_launch_gate",
        ),
        (LAUNCH_GATE_SPEED_MIN_M_S <= speed <= LAUNCH_GATE_SPEED_MAX_M_S, "speed_outside_launch_gate"),
    )
    for passed, reason in checks:
        if not bool(passed):
            return _status(False, reason, x_w, y_w, z_w, phi, theta, psi, speed)
    return _status(True, "approved_launch_gate", x_w, y_w, z_w, phi, theta, psi, speed)


def launch_gate_bounds_manifest() -> dict[str, object]:
    return {
        "source": "03_Control/04_Scenarios/state_sampling.py::state_is_launch_gate_compliant",
        "x_w_m": list(LAUNCH_GATE_X_W_M),
        "y_w_m": list(LAUNCH_GATE_Y_W_M),
        "z_w_m": list(LAUNCH_GATE_Z_W_M),
        "roll_deg": [-LAUNCH_GATE_ROLL_LIMIT_DEG, LAUNCH_GATE_ROLL_LIMIT_DEG],
        "pitch_deg": [LAUNCH_GATE_PITCH_MIN_DEG, LAUNCH_GATE_PITCH_MAX_DEG],
        "yaw_deg": [-LAUNCH_GATE_YAW_LIMIT_DEG, LAUNCH_GATE_YAW_LIMIT_DEG],
        "speed_m_s": [LAUNCH_GATE_SPEED_MIN_M_S, LAUNCH_GATE_SPEED_MAX_M_S],
    }


def _status(
    approved: bool,
    reason: str,
    x_w: float,
    y_w: float,
    z_w: float,
    phi: float,
    theta: float,
    psi: float,
    speed: float,
) -> LaunchGateStatus:
    return LaunchGateStatus(
        approved=bool(approved),
        reason=str(reason),
        x_w_m=float(x_w),
        y_w_m=float(y_w),
        z_w_m=float(z_w),
        phi_deg=float(np.rad2deg(phi)),
        theta_deg=float(np.rad2deg(theta)),
        psi_deg=float(np.rad2deg(psi)),
        speed_m_s=float(speed),
    )
