from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Descriptor Constants and Protocols
# 2) Wing Panel Geometry
# 3) Descriptor Evaluation
# =============================================================================


# =============================================================================
# 1) Descriptor Constants and Protocols
# =============================================================================
WING_WIND_DESCRIPTOR_COLUMNS = (
    "wind_descriptor_status",
    "wind_descriptor_environment_mode",
    "wind_descriptor_model_id",
    "wind_descriptor_model_source",
    "w_cg_m_s",
    "w_wing_mean_m_s",
    "w_left_m_s",
    "w_right_m_s",
    "delta_w_lr_m_s",
    "w_panel_max_m_s",
    "w_panel_min_m_s",
    "spanwise_w_gradient_m_s_per_m",
    "local_updraft_uncertainty_m_s",
    "local_updraft_uncertainty_status",
    "wing_panel_sample_count",
)


@dataclass(frozen=True)
class WingWindDescriptorConfig:
    half_span_m: float
    panel_count_per_side: int = 5
    body_x_offset_m: float = 0.0
    body_z_offset_m: float = 0.0
    local_uncertainty_default_m_s: float = float("nan")
    signed_span_positive: str = "left_port"
    world_frame: str = "z_up"


class WindFieldLike(Protocol):
    name: str
    source: str

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        """Return public world-frame wind, shape (N, 3), in m/s."""


def default_wing_wind_descriptor_config() -> WingWindDescriptorConfig:
    """Return descriptor geometry derived from the current Nausicaa wing span."""

    from glider import build_nausicaa_glider

    glider = build_nausicaa_glider()
    wing = next(surface for surface in glider.surfaces if surface.name == "wing")
    return WingWindDescriptorConfig(half_span_m=0.5 * float(wing.span_m))


# =============================================================================
# 2) Wing Panel Geometry
# =============================================================================
def wing_panel_points_w(
    x_w_m: float,
    y_w_m: float,
    z_w_m: float,
    phi_rad: float,
    theta_rad: float,
    psi_rad: float,
    config: WingWindDescriptorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return public z-up wing sample points and signed span coordinates in metres."""

    cfg = _validated_config(config)
    sample_count = 2 * int(cfg.panel_count_per_side) + 1
    signed_span_m = np.linspace(
        float(cfg.half_span_m),
        -float(cfg.half_span_m),
        sample_count,
    )
    # Positive signed span is left/port. Body +y is starboard, so left panels
    # are negative body-y offsets before the 3-2-1 body-to-world rotation.
    offsets_b = np.column_stack(
        [
            np.full(sample_count, float(cfg.body_x_offset_m)),
            -signed_span_m,
            np.full(sample_count, float(cfg.body_z_offset_m)),
        ]
    )
    c_wb = _c_wb_z_down(float(phi_rad), float(theta_rad), float(psi_rad))
    offsets_w_down = offsets_b @ c_wb.T
    offsets_w_up = offsets_w_down.copy()
    offsets_w_up[:, 2] *= -1.0
    origin = np.array([x_w_m, y_w_m, z_w_m], dtype=float)
    return origin + offsets_w_up, signed_span_m


def _validated_config(
    config: WingWindDescriptorConfig | None,
) -> WingWindDescriptorConfig:
    cfg = config or default_wing_wind_descriptor_config()
    if cfg.signed_span_positive != "left_port":
        raise ValueError("signed_span_positive must be 'left_port'.")
    if cfg.world_frame != "z_up":
        raise ValueError("world_frame must be 'z_up'.")
    if not np.isfinite(float(cfg.half_span_m)) or float(cfg.half_span_m) <= 0.0:
        raise ValueError("half_span_m must be finite and positive.")
    if int(cfg.panel_count_per_side) <= 0:
        raise ValueError("panel_count_per_side must be positive.")
    return cfg


def _c_wb_z_down(phi_rad: float, theta_rad: float, psi_rad: float) -> np.ndarray:
    c_phi = np.cos(phi_rad)
    s_phi = np.sin(phi_rad)
    c_theta = np.cos(theta_rad)
    s_theta = np.sin(theta_rad)
    c_psi = np.cos(psi_rad)
    s_psi = np.sin(psi_rad)
    return np.array(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [-s_theta, s_phi * c_theta, c_phi * c_theta],
        ],
        dtype=float,
    )


# =============================================================================
# 3) Descriptor Evaluation
# =============================================================================
def wing_wind_descriptor_row(
    *,
    wind_field: WindFieldLike | None,
    x_w_m: float,
    y_w_m: float,
    z_w_m: float,
    phi_rad: float,
    theta_rad: float,
    psi_rad: float,
    fan_layout: str,
    fan_config_id: str,
    environment_mode: str,
    model_id: str,
    model_source: str,
    dry_air: bool = False,
    config: WingWindDescriptorConfig | None = None,
) -> dict[str, object]:
    """Return one CSV-ready wing-scale vertical-wind descriptor row."""

    del fan_layout, fan_config_id
    cfg = _validated_config(config)
    sample_count = 2 * int(cfg.panel_count_per_side) + 1
    if dry_air:
        return _dry_air_row(
            environment_mode=environment_mode,
            model_id=model_id,
            model_source=model_source,
            sample_count=sample_count,
            uncertainty=float(cfg.local_uncertainty_default_m_s),
        )
    if wind_field is None:
        raise ValueError("wind_field is required unless dry_air=True.")
    if (
        str(model_id) == "analytic_debug_proxy"
        or getattr(wind_field, "name", "") == "analytic_debug_proxy"
    ):
        raise ValueError("analytic_debug_proxy is not valid for planning descriptors.")

    points_w_up_m, signed_span_m = wing_panel_points_w(
        x_w_m=x_w_m,
        y_w_m=y_w_m,
        z_w_m=z_w_m,
        phi_rad=phi_rad,
        theta_rad=theta_rad,
        psi_rad=psi_rad,
        config=cfg,
    )
    cg_w_up_m = np.array([[x_w_m, y_w_m, z_w_m]], dtype=float)
    wind_w = np.asarray(wind_field(np.vstack([cg_w_up_m, points_w_up_m])), dtype=float)
    if wind_w.shape != (sample_count + 1, 3) or not np.all(np.isfinite(wind_w)):
        raise ValueError("wind field must return finite public world wind with shape (N, 3).")

    w_cg = float(wind_w[0, 2])
    w_panel = wind_w[1:, 2]
    left = signed_span_m > 0.0
    right = signed_span_m < 0.0
    w_left = float(np.mean(w_panel[left]))
    w_right = float(np.mean(w_panel[right]))
    uncertainty, uncertainty_status = _local_uncertainty(wind_field, cfg)
    return {
        "wind_descriptor_status": "wind_model_evaluated",
        "wind_descriptor_environment_mode": str(environment_mode),
        "wind_descriptor_model_id": str(model_id),
        "wind_descriptor_model_source": str(model_source),
        "w_cg_m_s": w_cg,
        "w_wing_mean_m_s": float(np.mean(w_panel)),
        "w_left_m_s": w_left,
        "w_right_m_s": w_right,
        "delta_w_lr_m_s": float(w_left - w_right),
        "w_panel_max_m_s": float(np.max(w_panel)),
        "w_panel_min_m_s": float(np.min(w_panel)),
        "spanwise_w_gradient_m_s_per_m": _spanwise_gradient(signed_span_m, w_panel),
        "local_updraft_uncertainty_m_s": uncertainty,
        "local_updraft_uncertainty_status": uncertainty_status,
        "wing_panel_sample_count": sample_count,
    }


def _dry_air_row(
    *,
    environment_mode: str,
    model_id: str,
    model_source: str,
    sample_count: int,
    uncertainty: float,
) -> dict[str, object]:
    return {
        "wind_descriptor_status": "dry_air_zero_wind",
        "wind_descriptor_environment_mode": str(environment_mode),
        "wind_descriptor_model_id": str(model_id),
        "wind_descriptor_model_source": str(model_source),
        "w_cg_m_s": 0.0,
        "w_wing_mean_m_s": 0.0,
        "w_left_m_s": 0.0,
        "w_right_m_s": 0.0,
        "delta_w_lr_m_s": 0.0,
        "w_panel_max_m_s": 0.0,
        "w_panel_min_m_s": 0.0,
        "spanwise_w_gradient_m_s_per_m": 0.0,
        "local_updraft_uncertainty_m_s": uncertainty,
        "local_updraft_uncertainty_status": "not_available_dry_air",
        "wing_panel_sample_count": int(sample_count),
    }


def _local_uncertainty(
    wind_field: WindFieldLike,
    config: WingWindDescriptorConfig,
) -> tuple[float, str]:
    value = getattr(wind_field, "local_updraft_uncertainty_m_s", None)
    if value is None:
        return float(config.local_uncertainty_default_m_s), "not_available_in_model"
    uncertainty = float(value)
    if not np.isfinite(uncertainty):
        return float(config.local_uncertainty_default_m_s), "not_available_in_model"
    return uncertainty, "available_from_model"


def _spanwise_gradient(signed_span_m: np.ndarray, w_panel_m_s: np.ndarray) -> float:
    span = np.asarray(signed_span_m, dtype=float)
    wind = np.asarray(w_panel_m_s, dtype=float)
    span_centered = span - float(np.mean(span))
    denom = float(np.sum(span_centered**2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(span_centered * (wind - float(np.mean(wind)))) / denom)
