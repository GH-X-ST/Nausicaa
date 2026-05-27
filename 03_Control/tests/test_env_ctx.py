from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from env_ctx import (
    ENV_CONTEXT_COLUMNS,
    EnvironmentMetadata,
    build_environment_context,
    context_feature_vector,
    environment_context_row,
)
from env_surrogate import resolve_surrogate_binding
from state_contract import STATE_INDEX, STATE_SIZE
from updraft_models import (
    SINGLE_FAN_CENTER_XY,
    build_randomised_wind_field,
    load_updraft_model,
)


@dataclass(frozen=True)
class ConstantWind:
    value_m_s: float
    name: str = "constant_context_wind"
    source: str = "unit_test"
    local_updraft_uncertainty_m_s: float = 0.05

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
class SpanWind:
    name: str = "span_context_wind"
    source: str = "unit_test"

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        signed_span_m = 2.0 - points[:, 1]
        w_up = 0.4 + 0.2 * signed_span_m
        return np.column_stack([np.zeros_like(w_up), np.zeros_like(w_up), w_up])


def _state() -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = 5.8
    return state


def test_w0_dry_air_context_has_zero_wind_and_safety_margins() -> None:
    metadata = EnvironmentMetadata(environment_id="W0_dry", fan_count=0)
    binding = resolve_surrogate_binding("W0", metadata)
    context = build_environment_context(
        _state(),
        wind_field=None,
        metadata=metadata,
        latency_case="none",
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    row = environment_context_row(context)

    assert tuple(row) == ENV_CONTEXT_COLUMNS
    assert context.w_cg_m_s == 0.0
    assert context.w_wing_mean_m_s == 0.0
    assert context.fan_count == 0
    assert context.W_layer == "W0"
    assert context.wind_mode == "none"
    assert context.surrogate_binding_status == "ready"
    assert context.wall_margin_m > 0.0
    assert context.all_wall_margin_m == context.wall_margin_m
    assert context.governor_wall_margin_m >= context.wall_margin_m
    assert context.front_wall_margin_m > 0.0
    assert context.left_wall_margin_m > 0.0
    assert context.right_wall_margin_m > 0.0
    assert context.rear_wall_margin_m > 0.0
    assert context.floor_margin_m > 0.0
    assert context.ceiling_margin_m > 0.0


def test_heading_aware_governor_margin_does_not_reject_rear_wall_only() -> None:
    state = _state()
    state[STATE_INDEX["x_w"]] = 1.22
    state[STATE_INDEX["y_w"]] = 2.05
    state[STATE_INDEX["psi"]] = 0.44
    metadata = EnvironmentMetadata(environment_id="W0_dry", fan_count=0)
    binding = resolve_surrogate_binding("W0", metadata)

    context = build_environment_context(
        state,
        wind_field=None,
        metadata=metadata,
        latency_case="none",
        actuator_case="nominal",
        surrogate_binding=binding,
    )

    assert context.wall_margin_m < 0.05
    assert context.rear_wall_margin_m < 0.05
    assert context.front_wall_margin_m > 0.05
    assert context.left_wall_margin_m > 0.05
    assert context.right_wall_margin_m > 0.05
    assert context.governor_wall_margin_m > 0.05


def test_w1_measured_updraft_context_uses_same_interface() -> None:
    metadata = EnvironmentMetadata(
        environment_id="W1_measured",
        fan_count=1,
        fan_positions_m=(SINGLE_FAN_CENTER_XY,),
        fan_power_scales=(1.0,),
        updraft_model_id="single_gaussian_var",
    )
    binding = resolve_surrogate_binding("W1", metadata)
    wind = load_updraft_model(binding.updraft_model_id)
    context = build_environment_context(
        _state(),
        wind_field=wind,
        metadata=metadata,
        latency_case="nominal",
        actuator_case="nominal",
        surrogate_binding=binding,
    )

    assert context.environment_id == "W1_measured"
    assert context.fan_positions_m
    assert context.updraft_model_id == "single_gaussian_var"
    assert context.updraft_model_source.endswith("single_var_params.xlsx")
    assert context.surrogate_family == "gaussian_plume"
    assert len(context_feature_vector(context)) == 12
    assert np.all(np.isfinite(context_feature_vector(context)))


def test_w2_context_keeps_annular_gp_as_metadata_not_online_branch() -> None:
    metadata = EnvironmentMetadata(
        environment_id="W2_single",
        fan_count=1,
        fan_positions_m=(SINGLE_FAN_CENTER_XY,),
        fan_power_scales=(1.0,),
        updraft_model_id="single_annular_gp_grid",
    )
    binding = resolve_surrogate_binding("W2", metadata)
    wind = load_updraft_model(binding.updraft_model_id)

    context = build_environment_context(
        _state(),
        wind_field=wind,
        metadata=metadata,
        latency_case="nominal",
        actuator_case="nominal",
        surrogate_binding=binding,
    )

    assert context.W_layer == "W2"
    assert context.wind_mode == "panel"
    assert context.surrogate_family == "gp_corrected_annular_gaussian"
    assert len(context_feature_vector(context)) == 12


def test_shifted_and_power_scaled_environment_metadata_is_audit_only() -> None:
    base = ConstantWind(0.4)
    wind, _ = build_randomised_wind_field(base, seed=7, enabled=True)
    context = build_environment_context(
        _state(),
        wind_field=wind,
        metadata=EnvironmentMetadata(
            environment_id="W3_style_context_only",
            fan_count=1,
            fan_positions_m=(SINGLE_FAN_CENTER_XY,),
            fan_power_scales=(0.8,),
            updraft_model_id=wind.name,
            updraft_amplitude_scale=0.8,
            updraft_width_scale=1.1,
            updraft_centre_shift_m=(0.1, -0.1),
            residual_field_id="residual_context_smoke",
            randomisation_seed=7,
            model_source=wind.source,
        ),
        latency_case="conservative",
        actuator_case="conservative",
    )

    assert context.fan_power_scales == "0.800000"
    assert context.updraft_centre_shift_m == "0.100000;-0.100000"
    assert context.randomisation_seed == 7
    assert context.latency_case == "conservative"


def test_wing_descriptor_sign_enters_context_direction() -> None:
    context = build_environment_context(
        _state(),
        wind_field=SpanWind(),
        metadata=EnvironmentMetadata(
            environment_id="span_sign",
            fan_count=1,
            fan_positions_m=((4.2, 2.4),),
            fan_power_scales=(1.0,),
            updraft_model_id="span_context_wind",
        ),
        latency_case="nominal",
    )

    assert context.delta_w_lr_m_s > 0.0
    assert context.spanwise_w_gradient_m_s_per_m > 0.0
    assert context.lift_direction_x == 0.0
    assert context.lift_direction_y == pytest.approx(1.0)
