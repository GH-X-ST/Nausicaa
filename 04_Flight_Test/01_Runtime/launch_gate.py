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
LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S = 1.5
LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S = 0.7
LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S = 1.2
LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S = 1.2
LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S = 1.8
LAUNCH_GATE_X_W_M = (1.2, 1.4)
LAUNCH_TRIGGER_X_W_M = 0.5 * (LAUNCH_GATE_X_W_M[0] + LAUNCH_GATE_X_W_M[1])
LAUNCH_GATE_Y_W_M = (1.8, 2.2)
LAUNCH_GATE_Z_W_M = (1.3, 1.8)
DEFAULT_LAUNCH_GATE_BODY_RATE_LIMITS_RAD_S = (
    LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S,
    LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S,
    LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S,
)


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
    v_m_s: float
    w_m_s: float
    p_rad_s: float
    q_rad_s: float
    r_rad_s: float


def evaluate_launch_gate(
    state: np.ndarray,
    *,
    body_rate_limits_rad_s: tuple[float, float, float] = DEFAULT_LAUNCH_GATE_BODY_RATE_LIMITS_RAD_S,
) -> LaunchGateStatus:
    """Return whether the measured state matches the R5 launch-gate envelope."""

    x = as_state_vector(state)
    p_limit, q_limit, r_limit = _body_rate_limits(body_rate_limits_rad_s)
    x_w = float(x[STATE_INDEX["x_w"]])
    y_w = float(x[STATE_INDEX["y_w"]])
    z_w = float(x[STATE_INDEX["z_w"]])
    phi = float(x[STATE_INDEX["phi"]])
    theta = float(x[STATE_INDEX["theta"]])
    psi = float(x[STATE_INDEX["psi"]])
    speed = float(np.linalg.norm(x[STATE_INDEX["u"] : STATE_INDEX["w"] + 1]))
    v_body = float(x[STATE_INDEX["v"]])
    w_body = float(x[STATE_INDEX["w"]])
    p = float(x[STATE_INDEX["p"]])
    q = float(x[STATE_INDEX["q"]])
    r = float(x[STATE_INDEX["r"]])

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
        (
            -LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S <= v_body <= LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S,
            "side_velocity_outside_launch_gate",
        ),
        (
            -LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S <= w_body <= LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
            "vertical_body_velocity_outside_launch_gate",
        ),
        (-p_limit <= p <= p_limit, "roll_rate_outside_launch_gate"),
        (-q_limit <= q <= q_limit, "pitch_rate_outside_launch_gate"),
        (-r_limit <= r <= r_limit, "yaw_rate_outside_launch_gate"),
    )
    for passed, reason in checks:
        if not bool(passed):
            return _status(False, reason, x_w, y_w, z_w, phi, theta, psi, speed, v_body, w_body, p, q, r)
    return _status(True, "approved_launch_gate", x_w, y_w, z_w, phi, theta, psi, speed, v_body, w_body, p, q, r)


def interpolate_launch_plane_state(
    previous_state: np.ndarray | None,
    current_state: np.ndarray,
    *,
    plane_x_w_m: float = LAUNCH_TRIGGER_X_W_M,
) -> np.ndarray | None:
    """Estimate the state where the glider crosses the real-flight launch plane."""

    if previous_state is None:
        return None
    previous = as_state_vector(previous_state)
    current = as_state_vector(current_state)
    previous_x = float(previous[STATE_INDEX["x_w"]])
    current_x = float(current[STATE_INDEX["x_w"]])
    dx = current_x - previous_x
    if dx <= 1e-9:
        return None
    if not (previous_x <= float(plane_x_w_m) <= current_x):
        return None

    alpha = float(np.clip((float(plane_x_w_m) - previous_x) / dx, 0.0, 1.0))
    interpolated = previous + alpha * (current - previous)
    for angle_name in ("phi", "theta", "psi"):
        index = STATE_INDEX[angle_name]
        delta = _wrap_to_pi_scalar(float(current[index] - previous[index]))
        interpolated[index] = previous[index] + alpha * delta
    interpolated[STATE_INDEX["x_w"]] = float(plane_x_w_m)
    return interpolated


def evaluate_launch_plane_gate(
    state: np.ndarray,
    *,
    body_rate_limits_rad_s: tuple[float, float, float] = DEFAULT_LAUNCH_GATE_BODY_RATE_LIMITS_RAD_S,
) -> LaunchGateStatus:
    """Check the interpolated launch-plane state, excluding the old finite x box."""

    x = as_state_vector(state)
    p_limit, q_limit, r_limit = _body_rate_limits(body_rate_limits_rad_s)
    x_w = float(x[STATE_INDEX["x_w"]])
    y_w = float(x[STATE_INDEX["y_w"]])
    z_w = float(x[STATE_INDEX["z_w"]])
    phi = float(x[STATE_INDEX["phi"]])
    theta = float(x[STATE_INDEX["theta"]])
    psi = float(x[STATE_INDEX["psi"]])
    speed = float(np.linalg.norm(x[STATE_INDEX["u"] : STATE_INDEX["w"] + 1]))
    v_body = float(x[STATE_INDEX["v"]])
    w_body = float(x[STATE_INDEX["w"]])
    p = float(x[STATE_INDEX["p"]])
    q = float(x[STATE_INDEX["q"]])
    r = float(x[STATE_INDEX["r"]])

    checks = (
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
        (
            -LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S <= v_body <= LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S,
            "side_velocity_outside_launch_gate",
        ),
        (
            -LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S <= w_body <= LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
            "vertical_body_velocity_outside_launch_gate",
        ),
        (-p_limit <= p <= p_limit, "roll_rate_outside_launch_gate"),
        (-q_limit <= q <= q_limit, "pitch_rate_outside_launch_gate"),
        (-r_limit <= r <= r_limit, "yaw_rate_outside_launch_gate"),
    )
    for passed, reason in checks:
        if not bool(passed):
            return _status(False, reason, x_w, y_w, z_w, phi, theta, psi, speed, v_body, w_body, p, q, r)
    return _status(True, "approved_launch_plane_gate", x_w, y_w, z_w, phi, theta, psi, speed, v_body, w_body, p, q, r)


def launch_gate_bounds_manifest(
    *,
    body_rate_limits_rad_s: tuple[float, float, float] = DEFAULT_LAUNCH_GATE_BODY_RATE_LIMITS_RAD_S,
) -> dict[str, object]:
    p_limit, q_limit, r_limit = _body_rate_limits(body_rate_limits_rad_s)
    return {
        "source": "03_Control/04_Scenarios/state_sampling.py::state_is_launch_gate_compliant",
        "real_flight_trigger_policy": "first_valid_r5_launch_window_with_interpolated_launch_plane_fallback",
        "launch_trigger_x_w_m": LAUNCH_TRIGGER_X_W_M,
        "diagnostic_x_window_m": list(LAUNCH_GATE_X_W_M),
        "x_w_m": list(LAUNCH_GATE_X_W_M),
        "y_w_m": list(LAUNCH_GATE_Y_W_M),
        "z_w_m": list(LAUNCH_GATE_Z_W_M),
        "roll_deg": [-LAUNCH_GATE_ROLL_LIMIT_DEG, LAUNCH_GATE_ROLL_LIMIT_DEG],
        "pitch_deg": [LAUNCH_GATE_PITCH_MIN_DEG, LAUNCH_GATE_PITCH_MAX_DEG],
        "yaw_deg": [-LAUNCH_GATE_YAW_LIMIT_DEG, LAUNCH_GATE_YAW_LIMIT_DEG],
        "speed_m_s": [LAUNCH_GATE_SPEED_MIN_M_S, LAUNCH_GATE_SPEED_MAX_M_S],
        "v_m_s": [-LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S, LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S],
        "w_m_s": [
            -LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
            LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
        ],
        "p_rad_s": [-p_limit, p_limit],
        "q_rad_s": [-q_limit, q_limit],
        "r_rad_s": [-r_limit, r_limit],
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
    v_body: float,
    w_body: float,
    p: float,
    q: float,
    r: float,
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
        v_m_s=float(v_body),
        w_m_s=float(w_body),
        p_rad_s=float(p),
        q_rad_s=float(q),
        r_rad_s=float(r),
    )


def _wrap_to_pi_scalar(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _body_rate_limits(body_rate_limits_rad_s: tuple[float, float, float]) -> tuple[float, float, float]:
    limits = tuple(float(value) for value in body_rate_limits_rad_s)
    if len(limits) != 3:
        raise ValueError("body_rate_limits_rad_s must contain p/q/r limits.")
    if any(not np.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("body_rate_limits_rad_s must contain finite positive p/q/r limits.")
    return limits
