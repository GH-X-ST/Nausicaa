from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from wing_wind_descriptors import (
    WING_WIND_DESCRIPTOR_COLUMNS,
    WingWindDescriptorConfig,
    default_wing_wind_descriptor_config,
    wing_panel_points_w,
    wing_wind_descriptor_row,
)


@dataclass(frozen=True)
class ConstantWind:
    value_m_s: float
    name: str = "constant_test_wind"
    source: str = "unit_test"

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        return np.column_stack(
            [
                np.zeros(points.shape[0]),
                np.zeros(points.shape[0]),
                np.full(points.shape[0], float(self.value_m_s)),
            ]
        )


@dataclass(frozen=True)
class LinearSignedSpanWind:
    intercept_m_s: float
    slope_m_s_per_m: float
    name: str = "linear_signed_span_test_wind"
    source: str = "unit_test"

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        signed_span_m = -points[:, 1]
        w_up = float(self.intercept_m_s) + float(self.slope_m_s_per_m) * signed_span_m
        return np.column_stack([np.zeros_like(w_up), np.zeros_like(w_up), w_up])


def _config() -> WingWindDescriptorConfig:
    return WingWindDescriptorConfig(half_span_m=1.0, panel_count_per_side=2)


def test_default_config_uses_current_wing_half_span() -> None:
    config = default_wing_wind_descriptor_config()

    assert config.half_span_m == pytest.approx(0.382)
    assert config.panel_count_per_side == 5
    assert config.signed_span_positive == "left_port"
    assert config.world_frame == "z_up"


def test_panel_points_are_deterministic_left_to_right_at_zero_attitude() -> None:
    points, signed_span = wing_panel_points_w(
        x_w_m=1.0,
        y_w_m=2.0,
        z_w_m=3.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
        config=_config(),
    )

    expected_span = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    assert points.shape == (5, 3)
    assert np.all(np.isfinite(points))
    assert np.allclose(signed_span, expected_span)
    assert np.allclose(points[2], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(points[:, 0], 1.0)
    assert np.allclose(points[:, 2], 3.0)
    assert np.allclose(points[:, 1], 2.0 - expected_span)


def test_yaw_rotation_preserves_left_positive_convention() -> None:
    points, signed_span = wing_panel_points_w(
        x_w_m=0.0,
        y_w_m=0.0,
        z_w_m=0.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=np.pi / 2.0,
        config=_config(),
    )

    assert signed_span[0] == pytest.approx(1.0)
    assert np.allclose(points[0], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(points[2], np.zeros(3), atol=1e-12)
    assert np.allclose(points[-1], np.array([-1.0, 0.0, 0.0]), atol=1e-12)


def test_dry_air_descriptor_row_has_zero_wind_and_explicit_status() -> None:
    row = wing_wind_descriptor_row(
        wind_field=None,
        x_w_m=1.0,
        y_w_m=2.0,
        z_w_m=3.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
        fan_layout="single_fan",
        fan_config_id="single_fan_dry_air",
        environment_mode="dry_air",
        model_id="no_updraft_dry_air",
        model_source="dry_air_zero_wind",
        dry_air=True,
        config=_config(),
    )

    assert tuple(row) == WING_WIND_DESCRIPTOR_COLUMNS
    assert row["wind_descriptor_status"] == "dry_air_zero_wind"
    assert row["local_updraft_uncertainty_status"] == "not_available_dry_air"
    assert np.isnan(float(row["local_updraft_uncertainty_m_s"]))
    for key in (
        "w_cg_m_s",
        "w_wing_mean_m_s",
        "w_left_m_s",
        "w_right_m_s",
        "delta_w_lr_m_s",
        "w_panel_max_m_s",
        "w_panel_min_m_s",
        "spanwise_w_gradient_m_s_per_m",
    ):
        assert row[key] == 0.0


def test_constant_wind_descriptor_row_is_internally_consistent() -> None:
    row = wing_wind_descriptor_row(
        wind_field=ConstantWind(0.4),
        x_w_m=0.0,
        y_w_m=0.0,
        z_w_m=1.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
        fan_layout="single_fan",
        fan_config_id="single_fan_nominal_updraft",
        environment_mode="W1_single_fan",
        model_id="constant_test_wind",
        model_source="unit_test",
        config=_config(),
    )

    assert row["wind_descriptor_status"] == "wind_model_evaluated"
    assert row["wing_panel_sample_count"] == 5
    assert row["w_cg_m_s"] == pytest.approx(0.4)
    assert row["w_wing_mean_m_s"] == pytest.approx(0.4)
    assert row["w_left_m_s"] == pytest.approx(0.4)
    assert row["w_right_m_s"] == pytest.approx(0.4)
    assert row["delta_w_lr_m_s"] == pytest.approx(0.0)
    assert row["spanwise_w_gradient_m_s_per_m"] == pytest.approx(0.0)


def test_linear_wind_descriptor_uses_left_minus_right_and_span_gradient() -> None:
    row = wing_wind_descriptor_row(
        wind_field=LinearSignedSpanWind(intercept_m_s=0.5, slope_m_s_per_m=2.0),
        x_w_m=0.0,
        y_w_m=0.0,
        z_w_m=1.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
        fan_layout="single_fan",
        fan_config_id="single_fan_nominal_updraft",
        environment_mode="W1_single_fan",
        model_id="linear_signed_span_test_wind",
        model_source="unit_test",
        config=_config(),
    )

    assert row["w_left_m_s"] == pytest.approx(2.0)
    assert row["w_right_m_s"] == pytest.approx(-1.0)
    assert row["delta_w_lr_m_s"] == pytest.approx(3.0)
    assert row["spanwise_w_gradient_m_s_per_m"] == pytest.approx(2.0)


def test_analytic_debug_proxy_is_rejected_for_non_dry_descriptors() -> None:
    with pytest.raises(ValueError, match="analytic_debug_proxy"):
        wing_wind_descriptor_row(
            wind_field=ConstantWind(0.4, name="analytic_debug_proxy"),
            x_w_m=0.0,
            y_w_m=0.0,
            z_w_m=1.0,
            phi_rad=0.0,
            theta_rad=0.0,
            psi_rad=0.0,
            fan_layout="single_fan",
            fan_config_id="single_fan_nominal_updraft",
            environment_mode="W1_single_fan",
            model_id="analytic_debug_proxy",
            model_source="unit_test",
            config=_config(),
        )
