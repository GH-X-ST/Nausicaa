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
    z_w_m=(0.4, 3.5),
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


def heading_aligned_wall_margins_m(
    position_w: np.ndarray,
    heading_rad: float,
    bounds: BoxBounds,
) -> dict[str, float]:
    """Return front/side/rear wall clearances relative to the heading.

    The all-wall minimum remains the hard safety audit value. The governor
    clearance deliberately excludes the rear wall for forward-moving flight so
    a launch near the wall behind the aircraft is not rejected before rollout.
    """

    margins = position_margin_m(position_w, bounds)
    all_wall_margin = float(margins["min_wall_margin_m"])
    if all_wall_margin < 0.0:
        return {
            "front_wall_margin_m": all_wall_margin,
            "left_wall_margin_m": all_wall_margin,
            "right_wall_margin_m": all_wall_margin,
            "rear_wall_margin_m": all_wall_margin,
            "all_wall_margin_m": all_wall_margin,
            "governor_wall_margin_m": all_wall_margin,
        }

    heading = float(heading_rad)
    forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    right = -left
    x_min, x_max = margins["x_min_margin_m"], margins["x_max_margin_m"]
    y_min, y_max = margins["y_min_margin_m"], margins["y_max_margin_m"]
    faces = (
        ("x_min", float(x_min), np.array([-1.0, 0.0], dtype=float)),
        ("x_max", float(x_max), np.array([1.0, 0.0], dtype=float)),
        ("y_min", float(y_min), np.array([0.0, -1.0], dtype=float)),
        ("y_max", float(y_max), np.array([0.0, 1.0], dtype=float)),
    )
    rear_exclusion_cos = -0.5
    side_or_front = [margin for _, margin, normal in faces if float(np.dot(normal, forward)) >= rear_exclusion_cos]
    front = [margin for _, margin, normal in faces if float(np.dot(normal, forward)) > 0.5]
    left_side = [
        margin
        for _, margin, normal in faces
        if float(np.dot(normal, left)) > 0.0 and float(np.dot(normal, forward)) >= rear_exclusion_cos
    ]
    right_side = [
        margin
        for _, margin, normal in faces
        if float(np.dot(normal, right)) > 0.0 and float(np.dot(normal, forward)) >= rear_exclusion_cos
    ]
    rear = [margin for _, margin, normal in faces if float(np.dot(normal, forward)) < rear_exclusion_cos]
    front_margin = float(min(front or side_or_front or [all_wall_margin]))
    left_margin = float(min(left_side or side_or_front or [all_wall_margin]))
    right_margin = float(min(right_side or side_or_front or [all_wall_margin]))
    rear_margin = float(min(rear or [all_wall_margin]))
    governor_margin = float(min(side_or_front or [all_wall_margin]))
    return {
        "front_wall_margin_m": front_margin,
        "left_wall_margin_m": left_margin,
        "right_wall_margin_m": right_margin,
        "rear_wall_margin_m": rear_margin,
        "all_wall_margin_m": all_wall_margin,
        "governor_wall_margin_m": governor_margin,
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
