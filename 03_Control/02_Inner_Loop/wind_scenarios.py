from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


WindModel = Callable[[np.ndarray], np.ndarray] | np.ndarray | None


@dataclass(frozen=True)
class WindScenario:
    name: str
    label: str
    wind_model: WindModel
    wind_mode: str = "panel"
    residual_source: str = "none"


def constant_wind(vector_w_up_m_s: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(vector_w_up_m_s, dtype=float).reshape(3)


def analytic_updraft_residual_proxy(
    strength_m_s: float,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
    sigma_xy_m: float = 1.2,
    z_center_m: float = 4.0,
    sigma_z_m: float = 3.0,
    residual_scale_m_s: float = 0.18,
) -> Callable[[np.ndarray], np.ndarray]:
    # Deterministic proxy used when measured residual-GP data are unavailable.
    def wind(points_w_up_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        dx = pts[:, 0] - float(center_xy_m[0])
        dy = pts[:, 1] - float(center_xy_m[1])
        dz = pts[:, 2] - float(z_center_m)
        radial = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_xy_m * sigma_xy_m))
        vertical = np.exp(-(dz * dz) / (2.0 * sigma_z_m * sigma_z_m))
        residual = residual_scale_m_s * np.sin(1.7 * dx + 0.4) * np.cos(1.3 * dy - 0.2)
        w_up = float(strength_m_s) * radial * vertical + residual * radial
        u_x = 0.05 * w_up * np.tanh(dy / max(sigma_xy_m, 1e-12))
        u_y = -0.05 * w_up * np.tanh(dx / max(sigma_xy_m, 1e-12))
        return np.column_stack([u_x, u_y, w_up])

    return wind


def fixed_wind_scenarios() -> dict[str, WindScenario]:
    return {
        "none": WindScenario(
            name="none",
            label="zero wind",
            wind_model=None,
            wind_mode="panel",
        ),
        "crosswind": WindScenario(
            name="crosswind",
            label="constant crosswind",
            wind_model=constant_wind((0.20, -0.10, 0.0)),
            wind_mode="panel",
        ),
        "mild_updraft_proxy": WindScenario(
            name="mild_updraft_proxy",
            label="analytic updraft residual proxy, mild",
            wind_model=analytic_updraft_residual_proxy(
                strength_m_s=0.22,
                residual_scale_m_s=0.07,
            ),
            wind_mode="panel",
            residual_source="deterministic analytic residual proxy; measured residual-GP data not used",
        ),
        "strong_updraft_proxy": WindScenario(
            name="strong_updraft_proxy",
            label="analytic updraft residual proxy, stress",
            wind_model=analytic_updraft_residual_proxy(
                strength_m_s=0.40,
                center_xy_m=(0.6, -0.4),
                sigma_xy_m=0.85,
                residual_scale_m_s=0.12,
            ),
            wind_mode="panel",
            residual_source="deterministic analytic residual proxy; measured residual-GP data not used",
        ),
    }
