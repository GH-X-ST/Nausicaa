from __future__ import annotations

import numpy as np

from latency import AGGREGATE_LIMITS, COMMAND_LEVELS


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Canonical command constants
# 2) Command vector validation
# 3) Command audit helpers
# =============================================================================


# =============================================================================
# 1) Canonical Command Constants
# =============================================================================
# Commands are normalised aggregate aileron, elevator, and rudder requests.
# Physical surface throws remain sourced from latency.py.
COMMAND_NAMES = ("delta_a_cmd", "delta_e_cmd", "delta_r_cmd")
SURFACE_STATE_NAMES = ("delta_a", "delta_e", "delta_r")
COMMAND_SIZE = 3
COMMAND_INDEX = {name: index for index, name in enumerate(COMMAND_NAMES)}
NORMALISED_COMMAND_MIN = -1.0
NORMALISED_COMMAND_MAX = 1.0
CONTROL_SIGN_CONVENTION = {
    "positive_aileron": "positive roll moment, right wing down",
    "positive_elevator": "positive pitch moment, nose up",
    "positive_rudder": "positive yaw moment, nose right",
}


# =============================================================================
# 2) Command Vector Validation
# =============================================================================
def as_command_vector(u_cmd: np.ndarray) -> np.ndarray:
    """Return `u_cmd` as a finite 3-vector in canonical command order."""

    command = np.asarray(u_cmd, dtype=float)
    if command.size != COMMAND_SIZE:
        raise ValueError(f"command vector must contain {COMMAND_SIZE} values.")
    command = command.reshape(COMMAND_SIZE).copy()
    if not np.all(np.isfinite(command)):
        raise ValueError("command vector must contain only finite values.")
    return command


def clip_normalised_command(u_cmd: np.ndarray) -> np.ndarray:
    """Clip normalised commands to [-1, +1]."""

    command = as_command_vector(u_cmd)
    return np.clip(command, NORMALISED_COMMAND_MIN, NORMALISED_COMMAND_MAX)


# =============================================================================
# 3) Command Audit Helpers
# =============================================================================
def command_dataframe_row(u_cmd: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Return a CSV-ready command row."""

    command = as_command_vector(u_cmd)
    return {
        f"{prefix}{name}": float(command[index])
        for name, index in COMMAND_INDEX.items()
    }


def command_contract_row() -> dict[str, object]:
    """Return command-order and sign-convention metadata for audit output."""

    limits = {
        name: {
            "positive_deg": float(limit.positive_deg),
            "negative_deg": float(limit.negative_deg),
        }
        for name, limit in AGGREGATE_LIMITS.items()
    }
    return {
        "command_names": ",".join(COMMAND_NAMES),
        "surface_state_names": ",".join(SURFACE_STATE_NAMES),
        "normalised_command_min": NORMALISED_COMMAND_MIN,
        "normalised_command_max": NORMALISED_COMMAND_MAX,
        "command_levels": ",".join(f"{value:g}" for value in COMMAND_LEVELS),
        "aggregate_limits_deg": str(limits),
        **CONTROL_SIGN_CONVENTION,
    }
