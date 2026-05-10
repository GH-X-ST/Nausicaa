from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linearisation import STATE_INDEX


@dataclass(frozen=True)
class ArenaConfig:
    physical_volume_m: tuple[float, float, float] = (8.4, 4.8, 3.5)
    use_safe_volume: bool = True
    wall_margin_m: float = 0.45
    floor_margin_m: float = 0.25
    ceiling_margin_m: float = 0.25
    wing_span_margin_m: float = 0.40
    vicon_margin_m: float = 0.10
    launch_dispersion_margin_m: float = 0.30


def _total_wall_margin(config: ArenaConfig) -> float:
    return (
        float(config.wall_margin_m)
        + float(config.wing_span_margin_m)
        + float(config.vicon_margin_m)
        + float(config.launch_dispersion_margin_m)
    )


def safe_bounds(config: ArenaConfig) -> dict[str, tuple[float, float]]:
    width_x, width_y, height_z = config.physical_volume_m
    if not config.use_safe_volume:
        return {
            "x_w": (0.0, float(width_x)),
            "y_w": (0.0, float(width_y)),
            "z_w": (0.0, float(height_z)),
        }
    wall = _total_wall_margin(config)
    return {
        "x_w": (wall, float(width_x) - wall),
        "y_w": (wall, float(width_y) - wall),
        "z_w": (float(config.floor_margin_m), float(height_z) - float(config.ceiling_margin_m)),
    }


def safety_margins(
    x: np.ndarray,
    config: ArenaConfig,
) -> dict[str, float | bool]:
    state = np.asarray(x, dtype=float).reshape(15)
    bounds = safe_bounds(config)
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
    return {
        "inside_safe_volume": bool(inside),
        "min_wall_distance_m": float(min(x_margin, y_margin)),
        "floor_margin_m": float(floor_margin),
        "ceiling_margin_m": float(ceiling_margin),
    }
