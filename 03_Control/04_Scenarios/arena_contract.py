from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Arena bound constants
# 2) Position validation and margins
# 3) Classification and audit rows
# =============================================================================


# =============================================================================
# 1) Arena Bound Constants
# =============================================================================
# Bounds use the public world frame: x/y horizontal and z upward from the floor.
@dataclass(frozen=True)
class BoxBounds:
    name: str
    x_w_m: tuple[float, float]
    y_w_m: tuple[float, float]
    z_w_m: tuple[float, float]


TRACKER_LIMIT_BOUNDS = BoxBounds(
    name="tracker_limit",
    x_w_m=(0.0, 8.0),
    y_w_m=(0.0, 4.8),
    z_w_m=(0.0, 3.5),
)
TRUE_SAFE_BOUNDS = BoxBounds(
    name="true_safe",
    x_w_m=(1.2, 6.6),
    y_w_m=(0.0, 4.4),
    z_w_m=(0.0, 3.0),
)


# =============================================================================
# 2) Position Validation and Margins
# =============================================================================
def as_position_vector(position_w: np.ndarray) -> np.ndarray:
    """Return a finite [x_w, y_w, z_w] vector."""

    position = np.asarray(position_w, dtype=float)
    if position.size != 3:
        raise ValueError("position vector must contain 3 values.")
    position = position.reshape(3).copy()
    if not np.all(np.isfinite(position)):
        raise ValueError("position vector must contain only finite values.")
    return position


def _axis_margins(value: float, bounds: tuple[float, float]) -> tuple[float, float]:
    return float(value - bounds[0]), float(bounds[1] - value)


def position_margin_m(position_w: np.ndarray, bounds: BoxBounds) -> dict[str, float]:
    """Return signed margins to each face of the box."""

    x_w, y_w, z_w = as_position_vector(position_w)
    x_min, x_max = _axis_margins(float(x_w), bounds.x_w_m)
    y_min, y_max = _axis_margins(float(y_w), bounds.y_w_m)
    z_min, z_max = _axis_margins(float(z_w), bounds.z_w_m)
    return {
        "x_min_margin_m": x_min,
        "x_max_margin_m": x_max,
        "y_min_margin_m": y_min,
        "y_max_margin_m": y_max,
        "floor_margin_m": z_min,
        "ceiling_margin_m": z_max,
        "min_wall_margin_m": float(min(x_min, x_max, y_min, y_max)),
        "min_margin_m": float(min(x_min, x_max, y_min, y_max, z_min, z_max)),
    }


def inside_bounds(position_w: np.ndarray, bounds: BoxBounds) -> bool:
    """Return True if position is inside the selected bounds."""

    margins = position_margin_m(position_w, bounds)
    return bool(margins["min_margin_m"] >= 0.0)


# =============================================================================
# 3) Classification and Audit Rows
# =============================================================================
def classify_position(position_w: np.ndarray) -> str:
    """Return inside_true_safe, outside_true_safe_inside_tracker, or outside_tracker."""

    if inside_bounds(position_w, TRUE_SAFE_BOUNDS):
        return "inside_true_safe"
    if inside_bounds(position_w, TRACKER_LIMIT_BOUNDS):
        return "outside_true_safe_inside_tracker"
    return "outside_tracker"


def arena_contract_row() -> dict[str, object]:
    """Return tracker and true-safety bounds for audit output."""

    return {
        "tracker_limit_name": TRACKER_LIMIT_BOUNDS.name,
        "tracker_limit_x_w_m": str(TRACKER_LIMIT_BOUNDS.x_w_m),
        "tracker_limit_y_w_m": str(TRACKER_LIMIT_BOUNDS.y_w_m),
        "tracker_limit_z_w_m": str(TRACKER_LIMIT_BOUNDS.z_w_m),
        "true_safe_name": TRUE_SAFE_BOUNDS.name,
        "true_safe_x_w_m": str(TRUE_SAFE_BOUNDS.x_w_m),
        "true_safe_y_w_m": str(TRUE_SAFE_BOUNDS.y_w_m),
        "true_safe_z_w_m": str(TRUE_SAFE_BOUNDS.z_w_m),
    }
