from __future__ import annotations

from collections.abc import Mapping

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Canonical state constants
# 2) State vector validation
# 3) State packing and audit rows
# =============================================================================


# =============================================================================
# 1) Canonical State Constants
# =============================================================================
# State order is the shared public contract for dynamics, trim, logging, and
# future primitive/controller layers. Angles remain radians internally.
STATE_NAMES = (
    "x_w",
    "y_w",
    "z_w",
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
STATE_SIZE = 15
STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}


# =============================================================================
# 2) State Vector Validation
# =============================================================================
def as_state_vector(x: np.ndarray) -> np.ndarray:
    """Return `x` as a finite 15-vector in canonical state order."""

    state = np.asarray(x, dtype=float)
    if state.size != STATE_SIZE:
        raise ValueError(f"state vector must contain {STATE_SIZE} values.")
    state = state.reshape(STATE_SIZE).copy()
    if not np.all(np.isfinite(state)):
        raise ValueError("state vector must contain only finite values.")
    return state


# =============================================================================
# 3) State Packing and Audit Rows
# =============================================================================
def unpack_state(x: np.ndarray) -> dict[str, float]:
    """Return named state components without changing order or units."""

    state = as_state_vector(x)
    return {name: float(state[index]) for name, index in STATE_INDEX.items()}


def pack_state(values: Mapping[str, float]) -> np.ndarray:
    """Pack named values into canonical state order."""

    supplied = set(values.keys())
    required = set(STATE_NAMES)
    missing = required - supplied
    extra = supplied - required
    if missing:
        raise ValueError(f"state values missing required keys: {sorted(missing)}.")
    if extra:
        raise ValueError(f"state values contain unknown keys: {sorted(extra)}.")
    return as_state_vector(np.array([values[name] for name in STATE_NAMES], dtype=float))


def state_dataframe_row(x: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Return a CSV-ready row with canonical state names."""

    state = as_state_vector(x)
    return {f"{prefix}{name}": float(state[index]) for name, index in STATE_INDEX.items()}
