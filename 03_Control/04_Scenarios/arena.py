from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linearisation import STATE_INDEX


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Arena configuration
# 2) Shared-axis volume bounds
# 3) Safety margin evaluation
# =============================================================================

# =============================================================================
# 1) Arena Configuration
# =============================================================================
# Volumes use the public lab frame: x/y horizontal and z upward from the floor.
@dataclass(frozen=True)
class ArenaConfig:
    physical_volume_m: tuple[float, float, float] = (10.0, 6.2, 5.5)
    tracker_limit_size_m: tuple[float, float, float] = (8.0, 4.8, 3.5)
    true_safe_bounds_m: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = (
        (1.2, 6.6),
        (0.0, 4.4),
        (0.0, 3.0),
    )
    use_safe_volume: bool = True


# =============================================================================
# 2) Shared-Axis Volume Bounds
# =============================================================================
def _as_bound_dict(
    bounds_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {
        "x_w": (float(bounds_m[0][0]), float(bounds_m[0][1])),
        "y_w": (float(bounds_m[1][0]), float(bounds_m[1][1])),
        "z_w": (float(bounds_m[2][0]), float(bounds_m[2][1])),
    }


def _true_safe_centre(config: ArenaConfig) -> tuple[float, float, float]:
    bounds = _as_bound_dict(config.true_safe_bounds_m)
    return (
        0.5 * (bounds["x_w"][0] + bounds["x_w"][1]),
        0.5 * (bounds["y_w"][0] + bounds["y_w"][1]),
        0.5 * (bounds["z_w"][0] + bounds["z_w"][1]),
    )


def _bounds_from_centre(
    centre_m: tuple[float, float, float],
    size_m: tuple[float, float, float],
) -> dict[str, tuple[float, float]]:
    half_size = tuple(0.5 * float(value) for value in size_m)
    return {
        "x_w": (
            round(float(centre_m[0]) - half_size[0], 10),
            round(float(centre_m[0]) + half_size[0], 10),
        ),
        "y_w": (
            round(float(centre_m[1]) - half_size[1], 10),
            round(float(centre_m[1]) + half_size[1], 10),
        ),
        "z_w": (
            round(float(centre_m[2]) - half_size[2], 10),
            round(float(centre_m[2]) + half_size[2], 10),
        ),
    }


def physical_bounds(config: ArenaConfig) -> dict[str, tuple[float, float]]:
    # The nominal room and tracker boxes share the true-safety centre for plotting alignment.
    return _bounds_from_centre(_true_safe_centre(config), config.physical_volume_m)


def safe_bounds(config: ArenaConfig) -> dict[str, tuple[float, float]]:
    if not config.use_safe_volume:
        # Non-safety visual axes use the tracker box, not the larger facility context.
        return tracker_bounds(config)
    # The active safety volume is explicit and must not be recomputed from legacy margins.
    return _as_bound_dict(config.true_safe_bounds_m)


def tracker_bounds(config: ArenaConfig) -> dict[str, tuple[float, float]]:
    return _bounds_from_centre(_true_safe_centre(config), config.tracker_limit_size_m)


# =============================================================================
# 3) Safety Margin Evaluation
# =============================================================================
def safety_margins(
    x: np.ndarray,
    config: ArenaConfig,
) -> dict[str, float | bool]:
    state = np.asarray(x, dtype=float).reshape(15)
    bounds = safe_bounds(config)
    tracker = tracker_bounds(config)
    # Horizontal wall clearance is reported separately from floor and ceiling margins.
    x_w = float(state[STATE_INDEX["x_w"]])
    y_w = float(state[STATE_INDEX["y_w"]])
    z_w = float(state[STATE_INDEX["z_w"]])
    x_margin = min(x_w - bounds["x_w"][0], bounds["x_w"][1] - x_w)
    y_margin = min(y_w - bounds["y_w"][0], bounds["y_w"][1] - y_w)
    floor_margin = z_w - bounds["z_w"][0]
    ceiling_margin = bounds["z_w"][1] - z_w
    inside = (
        x_margin >= 0.0
        and y_margin >= 0.0
        and floor_margin >= 0.0
        and ceiling_margin >= 0.0
    )
    tracker_x_margin = min(x_w - tracker["x_w"][0], tracker["x_w"][1] - x_w)
    tracker_y_margin = min(y_w - tracker["y_w"][0], tracker["y_w"][1] - y_w)
    tracker_floor_margin = z_w - tracker["z_w"][0]
    tracker_ceiling_margin = tracker["z_w"][1] - z_w
    inside_tracker = (
        tracker_x_margin >= 0.0
        and tracker_y_margin >= 0.0
        and tracker_floor_margin >= 0.0
        and tracker_ceiling_margin >= 0.0
    )
    return {
        "inside_safe_volume": bool(inside),
        "min_wall_distance_m": float(min(x_margin, y_margin)),
        "floor_margin_m": float(floor_margin),
        "ceiling_margin_m": float(ceiling_margin),
        "inside_tracker_limit": bool(inside_tracker),
        "tracker_min_wall_distance_m": float(min(tracker_x_margin, tracker_y_margin)),
        "tracker_floor_margin_m": float(tracker_floor_margin),
        "tracker_ceiling_margin_m": float(tracker_ceiling_margin),
        "x_margin_m": float(x_margin),
        "y_margin_m": float(y_margin),
        "tracker_x_margin_m": float(tracker_x_margin),
        "tracker_y_margin_m": float(tracker_y_margin),
    }
