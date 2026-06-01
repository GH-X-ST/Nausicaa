from __future__ import annotations

import numpy as np

from latency import (
    AGGREGATE_LIMITS,
    COMMAND_LEVELS,
    angle_to_command_norm,
    command_norm_to_angle,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Canonical command constants
# 2) Normalised command validation and conversion
# 3) Command audit helpers
# =============================================================================


# =============================================================================
# 1) Canonical Command Constants
# =============================================================================
# COMMAND_NAMES is the model-facing command order. These values are physical
# aggregate surface targets in radians and are passed to state_derivative.
COMMAND_NAMES = ("delta_a_cmd", "delta_e_cmd", "delta_r_cmd")
NORMALISED_COMMAND_NAMES = ("delta_a_norm", "delta_e_norm", "delta_r_norm")
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
EXECUTABLE_COMMAND_QUANTISATION = "fixed_20_percent_lattice"


# =============================================================================
# 2) Normalised Command Validation and Conversion
# =============================================================================
def _as_finite_vector(values: np.ndarray, label: str) -> np.ndarray:
    command = np.asarray(values, dtype=float)
    if command.size != COMMAND_SIZE:
        raise ValueError(f"{label} must contain {COMMAND_SIZE} values.")
    command = command.reshape(COMMAND_SIZE).copy()
    if not np.all(np.isfinite(command)):
        raise ValueError(f"{label} must contain only finite values.")
    return command


def as_normalised_command_vector(u_norm: np.ndarray) -> np.ndarray:
    """Return a finite normalised aggregate command vector."""

    return _as_finite_vector(u_norm, "normalised command vector")


def clip_normalised_command(u_norm: np.ndarray) -> np.ndarray:
    """Clip normalised commands to [-1, +1]."""

    command = as_normalised_command_vector(u_norm)
    return np.clip(command, NORMALISED_COMMAND_MIN, NORMALISED_COMMAND_MAX)


def quantise_normalised_command_vector(u_norm: np.ndarray) -> np.ndarray:
    """Return the executable 20 percent lattice command vector."""

    clipped = clip_normalised_command(u_norm)
    levels = np.asarray(COMMAND_LEVELS, dtype=float).reshape(1, -1)
    nearest = np.argmin(np.abs(clipped.reshape(-1, 1) - levels), axis=1)
    return COMMAND_LEVELS[nearest].astype(float, copy=True)


def as_surface_command_rad(delta_cmd_rad: np.ndarray) -> np.ndarray:
    """Return finite physical aggregate surface targets in radians."""

    command = _as_finite_vector(delta_cmd_rad, "surface command vector")
    for value, name in zip(command, SURFACE_STATE_NAMES, strict=True):
        limit = AGGREGATE_LIMITS[name]
        lower = min(
            command_norm_to_angle(-1.0, limit),
            command_norm_to_angle(1.0, limit),
        )
        upper = max(
            command_norm_to_angle(-1.0, limit),
            command_norm_to_angle(1.0, limit),
        )
        if value < lower - 1e-12 or value > upper + 1e-12:
            raise ValueError(f"{name} command is outside calibrated aggregate limits.")
    return command


def normalised_command_to_surface_rad(u_norm: np.ndarray) -> np.ndarray:
    """Convert normalised aggregate commands to physical radian targets."""

    clipped = clip_normalised_command(u_norm)
    return np.array(
        [
            command_norm_to_angle(value, AGGREGATE_LIMITS[name])
            for value, name in zip(clipped, SURFACE_STATE_NAMES, strict=True)
        ],
        dtype=float,
    )


def surface_rad_to_normalised_command(delta_cmd_rad: np.ndarray) -> np.ndarray:
    """Convert physical radian targets to normalised aggregate commands."""

    command = as_surface_command_rad(delta_cmd_rad)
    return np.array(
        [
            angle_to_command_norm(value, AGGREGATE_LIMITS[name])
            for value, name in zip(command, SURFACE_STATE_NAMES, strict=True)
        ],
        dtype=float,
    )


def as_command_vector(u_cmd: np.ndarray) -> np.ndarray:
    """Return a model-facing physical command vector in radians."""

    return as_surface_command_rad(u_cmd)


# =============================================================================
# 3) Command Audit Helpers
# =============================================================================
def command_dataframe_row(delta_cmd_rad: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Return a CSV-ready command row."""

    command = as_surface_command_rad(delta_cmd_rad)
    return {
        f"{prefix}{name}": float(command[index])
        for name, index in COMMAND_INDEX.items()
    }


def normalised_command_dataframe_row(
    u_norm: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Return a CSV-ready normalised command row."""

    command = as_normalised_command_vector(u_norm)
    return {
        f"{prefix}{name}": float(command[index])
        for index, name in enumerate(NORMALISED_COMMAND_NAMES)
    }


def command_contract_row() -> dict[str, object]:
    """Return command-order and sign-convention metadata for audit output."""

    limits = {
        name: {
            "positive_rad": float(command_norm_to_angle(1.0, limit)),
            "negative_rad": float(command_norm_to_angle(-1.0, limit)),
            "positive_deg": float(limit.positive_deg),
            "negative_deg": float(limit.negative_deg),
        }
        for name, limit in AGGREGATE_LIMITS.items()
    }
    return {
        "command_names": ",".join(COMMAND_NAMES),
        "command_units": "rad",
        "command_interface_to_state_derivative": "delta_cmd_rad",
        "normalised_command_names": ",".join(NORMALISED_COMMAND_NAMES),
        "surface_state_names": ",".join(SURFACE_STATE_NAMES),
        "normalised_command_min": NORMALISED_COMMAND_MIN,
        "normalised_command_max": NORMALISED_COMMAND_MAX,
        "executable_command_quantisation": EXECUTABLE_COMMAND_QUANTISATION,
        "normalised_to_radian_bridge": "normalised_command_to_surface_rad",
        "radian_to_normalised_bridge": "surface_rad_to_normalised_command",
        "raw_normalised_commands_enter_state_derivative": False,
        "continuous_lqr_commands_enter_state_derivative": False,
        "command_levels": ",".join(f"{value:g}" for value in COMMAND_LEVELS),
        "aggregate_limits": str(limits),
        **CONTROL_SIGN_CONVENTION,
    }
