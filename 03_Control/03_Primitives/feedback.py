from __future__ import annotations

import numpy as np

from latency import AGGREGATE_LIMITS
from linearisation import STATE_INDEX
from primitive import PrimitiveContext


def speed_alpha_beta(x: np.ndarray) -> tuple[float, float, float]:
    u, v, w = np.asarray(x[6:9], dtype=float)
    speed = float(np.linalg.norm([u, v, w]))
    alpha = float(np.arctan2(w, max(u, 1e-12)))
    beta = float(np.arcsin(np.clip(v / max(speed, 1e-12), -1.0, 1.0)))
    return speed, alpha, beta


def limit_aggregate_command(command_rad: np.ndarray) -> np.ndarray:
    command = np.asarray(command_rad, dtype=float).reshape(3)
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
    return np.clip(command, lower, upper)


def attitude_hold_command(
    x: np.ndarray,
    context: PrimitiveContext,
    phi_ref_rad: float,
    theta_ref_rad: float,
    gains: tuple[float, float, float, float, float, float] = (
        1.6,
        0.18,
        1.2,
        0.10,
        0.45,
        0.08,
    ),
) -> np.ndarray:
    phi = float(x[STATE_INDEX["phi"]])
    theta = float(x[STATE_INDEX["theta"]])
    p = float(x[STATE_INDEX["p"]])
    q = float(x[STATE_INDEX["q"]])
    r = float(x[STATE_INDEX["r"]])
    speed, _alpha, beta = speed_alpha_beta(x)
    kp_phi, kd_p, kp_theta, kd_q, k_beta, kd_r = gains

    delta_a = kp_phi * (phi_ref_rad - phi) - kd_p * p
    delta_e = (
        context.u_trim[1]
        + kp_theta * (theta_ref_rad - theta)
        - kd_q * q
        + 0.04 * (speed - context.speed_trim_m_s)
    )
    delta_r = -k_beta * beta - kd_r * r
    return limit_aggregate_command(np.array([delta_a, delta_e, delta_r], dtype=float))
