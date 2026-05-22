from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, BoxBounds, position_margin_m
from env_surrogate import READY_STATUS, SurrogateBinding
from latency import LATENCY_CASES
from state_contract import STATE_INDEX, as_state_vector
from wing_wind_descriptors import (
    WingWindDescriptorConfig,
    WindFieldLike,
    wing_wind_descriptor_row,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Context schema and dataclasses
# 2) Context construction
# 3) Serialisation helpers
# =============================================================================


# =============================================================================
# 1) Context Schema and Dataclasses
# =============================================================================
ENV_CONTEXT_COLUMNS = (
    "w_cg_m_s",
    "w_wing_mean_m_s",
    "delta_w_lr_m_s",
    "spanwise_w_gradient_m_s_per_m",
    "w_local_uncertainty_m_s",
    "w_local_uncertainty_status",
    "lift_score",
    "lift_direction_x",
    "lift_direction_y",
    "wall_margin_m",
    "floor_margin_m",
    "ceiling_margin_m",
    "speed_margin_m_s",
    "attitude_margin_rad",
    "latency_case",
    "actuator_case",
    "W_layer",
    "wind_mode",
    "environment_id",
    "environment_instance_id",
    "environment_mode",
    "fan_count",
    "fan_positions_m",
    "fan_power_scales",
    "active_fan_mask",
    "updraft_model_id",
    "updraft_model_source",
    "surrogate_family",
    "surrogate_role",
    "surrogate_binding_status",
    "surrogate_blocked_reason",
    "updraft_amplitude_scale",
    "updraft_width_scale",
    "updraft_centre_shift_m",
    "residual_field_id",
    "local_uncertainty_scale",
    "randomisation_label",
    "randomisation_seed",
)

NUMERIC_CONTEXT_FEATURES = (
    "w_cg_m_s",
    "w_wing_mean_m_s",
    "delta_w_lr_m_s",
    "spanwise_w_gradient_m_s_per_m",
    "w_local_uncertainty_m_s",
    "lift_score",
    "lift_direction_x",
    "lift_direction_y",
    "wall_margin_m",
    "floor_margin_m",
    "ceiling_margin_m",
    "speed_margin_m_s",
    "attitude_margin_rad",
)


@dataclass(frozen=True)
class EnvironmentMetadata:
    environment_id: str
    fan_count: int
    fan_positions_m: tuple[tuple[float, float], ...] = ()
    fan_power_scales: tuple[float, ...] = ()
    active_fan_mask: tuple[bool, ...] = ()
    updraft_model_id: str = "dry_air_zero_wind"
    updraft_amplitude_scale: float = 1.0
    updraft_width_scale: float = 1.0
    updraft_centre_shift_m: tuple[float, float] = (0.0, 0.0)
    residual_field_id: str = "none"
    local_uncertainty_scale: float = 1.0
    randomisation_seed: int | None = None
    model_source: str = "not_applicable"
    W_layer: str = "W0"
    wind_mode: str = "none"
    environment_mode: str = ""
    environment_instance_id: str = ""
    claim_status: str = "simulation_only_environment_metadata"


@dataclass(frozen=True)
class EnvironmentContext:
    w_cg_m_s: float
    w_wing_mean_m_s: float
    delta_w_lr_m_s: float
    spanwise_w_gradient_m_s_per_m: float
    w_local_uncertainty_m_s: float
    w_local_uncertainty_status: str
    lift_score: float
    lift_direction_x: float
    lift_direction_y: float
    wall_margin_m: float
    floor_margin_m: float
    ceiling_margin_m: float
    speed_margin_m_s: float
    attitude_margin_rad: float
    latency_case: str
    actuator_case: str
    W_layer: str
    wind_mode: str
    environment_id: str
    environment_instance_id: str
    environment_mode: str
    fan_count: int
    fan_positions_m: str
    fan_power_scales: str
    active_fan_mask: str
    updraft_model_id: str
    updraft_model_source: str
    surrogate_family: str
    surrogate_role: str
    surrogate_binding_status: str
    surrogate_blocked_reason: str
    updraft_amplitude_scale: float
    updraft_width_scale: float
    updraft_centre_shift_m: str
    residual_field_id: str
    local_uncertainty_scale: float
    randomisation_label: str
    randomisation_seed: int | str


# =============================================================================
# 2) Context Construction
# =============================================================================
def build_environment_context(
    state: np.ndarray,
    *,
    wind_field: WindFieldLike | None,
    metadata: EnvironmentMetadata,
    latency_case: str,
    actuator_case: str = "nominal",
    surrogate_binding: SurrogateBinding | None = None,
    bounds: BoxBounds = TRUE_SAFE_BOUNDS,
    minimum_speed_m_s: float = 3.0,
    attitude_limit_rad: float = np.deg2rad(45.0),
    wing_config: WingWindDescriptorConfig | None = None,
) -> EnvironmentContext:
    """Build a CSV-ready local context row from state and environment metadata."""

    x = as_state_vector(state)
    if latency_case not in LATENCY_CASES:
        raise ValueError("latency_case must use the retained latency contract.")
    if actuator_case not in {"nominal", "conservative"}:
        raise ValueError("actuator_case must be 'nominal' or 'conservative'.")
    if not metadata.environment_id:
        raise ValueError("environment_id must be nonempty.")
    if int(metadata.fan_count) < 0:
        raise ValueError("fan_count must be nonnegative.")
    if metadata.fan_power_scales and len(metadata.fan_power_scales) != int(metadata.fan_count):
        raise ValueError("fan_power_scales must match fan_count when supplied.")
    if metadata.active_fan_mask and len(metadata.active_fan_mask) != int(metadata.fan_count):
        raise ValueError("active_fan_mask must match fan_count when supplied.")
    if surrogate_binding is not None and surrogate_binding.surrogate_binding_status not in {
        READY_STATUS,
        "blocked",
    }:
        raise ValueError("surrogate binding status must be ready or blocked.")

    dry_air = wind_field is None
    binding_values = _binding_values(surrogate_binding, metadata, dry_air=dry_air)
    descriptor = wing_wind_descriptor_row(
        wind_field=wind_field,
        x_w_m=float(x[STATE_INDEX["x_w"]]),
        y_w_m=float(x[STATE_INDEX["y_w"]]),
        z_w_m=float(x[STATE_INDEX["z_w"]]),
        phi_rad=float(x[STATE_INDEX["phi"]]),
        theta_rad=float(x[STATE_INDEX["theta"]]),
        psi_rad=float(x[STATE_INDEX["psi"]]),
        fan_layout=str(binding_values["fan_count"]),
        fan_config_id=metadata.environment_id,
        environment_mode=metadata.environment_id,
        model_id=str(binding_values["updraft_model_id"]),
        model_source=str(binding_values["updraft_model_source"]),
        dry_air=dry_air,
        config=wing_config,
    )
    margins = position_margin_m(x[:3], bounds)
    speed_m_s = float(np.linalg.norm(x[6:9]))
    max_attitude_rad = float(max(abs(x[STATE_INDEX["phi"]]), abs(x[STATE_INDEX["theta"]])))
    w_local_uncertainty, w_local_uncertainty_status = _local_uncertainty_for_context(
        descriptor=descriptor,
        metadata=metadata,
        dry_air=dry_air,
    )
    lift_score = _lift_score(
        w_wing_mean_m_s=float(descriptor["w_wing_mean_m_s"]),
        uncertainty_m_s=w_local_uncertainty,
    )
    lift_direction_x, lift_direction_y = _lift_direction(
        delta_w_lr_m_s=float(descriptor["delta_w_lr_m_s"]),
        w_wing_mean_m_s=float(descriptor["w_wing_mean_m_s"]),
    )
    return EnvironmentContext(
        w_cg_m_s=float(descriptor["w_cg_m_s"]),
        w_wing_mean_m_s=float(descriptor["w_wing_mean_m_s"]),
        delta_w_lr_m_s=float(descriptor["delta_w_lr_m_s"]),
        spanwise_w_gradient_m_s_per_m=float(
            descriptor["spanwise_w_gradient_m_s_per_m"]
        ),
        w_local_uncertainty_m_s=w_local_uncertainty,
        w_local_uncertainty_status=w_local_uncertainty_status,
        lift_score=lift_score,
        lift_direction_x=lift_direction_x,
        lift_direction_y=lift_direction_y,
        wall_margin_m=float(margins["min_wall_margin_m"]),
        floor_margin_m=float(margins["floor_margin_m"]),
        ceiling_margin_m=float(margins["ceiling_margin_m"]),
        speed_margin_m_s=float(speed_m_s - float(minimum_speed_m_s)),
        attitude_margin_rad=float(float(attitude_limit_rad) - max_attitude_rad),
        latency_case=str(latency_case),
        actuator_case=str(actuator_case),
        W_layer=str(binding_values["W_layer"]),
        wind_mode=str(binding_values["wind_mode"]),
        environment_id=str(metadata.environment_id),
        environment_instance_id=str(metadata.environment_instance_id or metadata.environment_id),
        environment_mode=str(metadata.environment_mode or metadata.environment_id),
        fan_count=int(binding_values["fan_count"]),
        fan_positions_m=_xy_pairs_text(binding_values["fan_positions_m"]),
        fan_power_scales=_float_tuple_text(binding_values["fan_power_scales"]),
        active_fan_mask=_bool_tuple_text(tuple(metadata.active_fan_mask)),
        updraft_model_id=str(binding_values["updraft_model_id"]),
        updraft_model_source=str(binding_values["updraft_model_source"]),
        surrogate_family=str(binding_values["surrogate_family"]),
        surrogate_role=str(binding_values["surrogate_role"]),
        surrogate_binding_status=str(binding_values["surrogate_binding_status"]),
        surrogate_blocked_reason=str(binding_values["surrogate_blocked_reason"]),
        updraft_amplitude_scale=float(metadata.updraft_amplitude_scale),
        updraft_width_scale=float(metadata.updraft_width_scale),
        updraft_centre_shift_m=_float_tuple_text(metadata.updraft_centre_shift_m),
        residual_field_id=str(metadata.residual_field_id),
        local_uncertainty_scale=float(metadata.local_uncertainty_scale),
        randomisation_label=str(binding_values["randomisation_label"]),
        randomisation_seed=(
            ""
            if binding_values["randomisation_seed"] is None
            else int(binding_values["randomisation_seed"])
        ),
    )


def _lift_score(*, w_wing_mean_m_s: float, uncertainty_m_s: float) -> float:
    useful_lift = max(float(w_wing_mean_m_s), 0.0)
    penalty = max(float(uncertainty_m_s), 0.0)
    return float(np.clip(useful_lift / 1.0 - 0.25 * penalty, 0.0, 1.0))


def _lift_direction(
    *,
    delta_w_lr_m_s: float,
    w_wing_mean_m_s: float,
) -> tuple[float, float]:
    if abs(float(delta_w_lr_m_s)) <= 1e-12 or float(w_wing_mean_m_s) <= 0.0:
        return 0.0, 0.0
    # Positive signed span is left/port, so positive delta points left in local context.
    return 0.0, float(np.sign(delta_w_lr_m_s))


# =============================================================================
# 3) Serialisation Helpers
# =============================================================================
def context_feature_vector(context: EnvironmentContext) -> tuple[float, ...]:
    """Return the numeric context features used by primitive selection code."""

    return tuple(float(getattr(context, name)) for name in NUMERIC_CONTEXT_FEATURES)


def environment_context_row(context: EnvironmentContext) -> dict[str, object]:
    """Return a compact row whose columns are stable for partitioned tables."""

    row = asdict(context)
    return {name: row[name] for name in ENV_CONTEXT_COLUMNS}


def context_feature_vector_json(context: EnvironmentContext) -> str:
    """Return a compact JSON vector for rollout evidence rows."""

    return json.dumps(
        [float(value) for value in context_feature_vector(context)],
        separators=(",", ":"),
    )


def _finite_or_zero(value: object) -> float:
    result = float(value)
    return result if np.isfinite(result) else 0.0


def _local_uncertainty_for_context(
    *,
    descriptor: dict[str, object],
    metadata: EnvironmentMetadata,
    dry_air: bool,
) -> tuple[float, str]:
    raw_value = float(descriptor["local_updraft_uncertainty_m_s"])
    raw_status = str(descriptor.get("local_updraft_uncertainty_status", "unknown"))
    if dry_air:
        return 0.0, "dry_air_zero_uncertainty"
    scale = max(float(metadata.local_uncertainty_scale), 0.0)
    if np.isfinite(raw_value) and raw_value > 0.0:
        return float(raw_value * scale), raw_status
    return float(0.25 * max(scale, 1.0)), "conservative_fallback_nonzero"


def _binding_values(
    binding: SurrogateBinding | None,
    metadata: EnvironmentMetadata,
    *,
    dry_air: bool,
) -> dict[str, object]:
    if binding is None:
        return {
            "W_layer": str(metadata.W_layer),
            "wind_mode": "none" if dry_air else str(metadata.wind_mode or "panel"),
            "fan_count": int(metadata.fan_count),
            "fan_positions_m": tuple(metadata.fan_positions_m),
            "fan_power_scales": tuple(metadata.fan_power_scales),
            "updraft_model_id": str(metadata.updraft_model_id),
            "updraft_model_source": str(metadata.model_source),
            "surrogate_family": "unbound_context",
            "surrogate_role": "context_only",
            "surrogate_binding_status": "not_bound",
            "surrogate_blocked_reason": "",
            "randomisation_label": "none",
            "randomisation_seed": metadata.randomisation_seed,
        }
    return {
        "W_layer": str(binding.W_layer),
        "wind_mode": str(binding.wind_mode),
        "fan_count": int(binding.fan_count),
        "fan_positions_m": tuple(binding.fan_positions_m),
        "fan_power_scales": tuple(binding.fan_power_scales),
        "updraft_model_id": str(binding.updraft_model_id),
        "updraft_model_source": str(binding.updraft_model_source),
        "surrogate_family": str(binding.surrogate_family),
        "surrogate_role": str(binding.surrogate_role),
        "surrogate_binding_status": str(binding.surrogate_binding_status),
        "surrogate_blocked_reason": str(binding.blocked_reason),
        "randomisation_label": str(binding.randomisation_label),
        "randomisation_seed": binding.randomisation_seed,
    }


def _xy_pairs_text(values: tuple[tuple[float, float], ...]) -> str:
    return ";".join(f"{float(x):.6f}:{float(y):.6f}" for x, y in values)


def _float_tuple_text(values: tuple[float, ...]) -> str:
    return ";".join(f"{float(value):.6f}" for value in values)


def _bool_tuple_text(values: tuple[bool, ...]) -> str:
    return ";".join("1" if bool(value) else "0" for value in values)
