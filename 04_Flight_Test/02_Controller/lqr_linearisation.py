from __future__ import annotations

import numpy as np

from state_contract import STATE_INDEX


LQR_LOCAL_OPERATING_SPEED_GRID_M_S = (
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
)
LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S = 5.0
LQR_STATE_MASK = (
    "phi",
    "theta",
    "psi",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "delta_a",
    "delta_e",
    "delta_r",
)


def lqr_speed_bin_id(speed_m_s: float) -> str:
    """Return the nearest local LQR operating-speed bin label."""

    speed = float(speed_m_s)
    grid = np.asarray(LQR_LOCAL_OPERATING_SPEED_GRID_M_S, dtype=float)
    nearest = float(grid[int(np.argmin(np.abs(grid - speed)))])
    return f"speed_bin_{nearest:.1f}".replace(".", "p") + "_m_s"


def reduced_state_indices() -> tuple[int, ...]:
    """Return canonical-state indices used by the frozen reduced LQR gains."""

    return tuple(STATE_INDEX[name] for name in LQR_STATE_MASK)
