from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from env_ctx import EnvironmentMetadata
from updraft_models import FOUR_FAN_CENTERS_XY, SINGLE_FAN_CENTER_XY


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Environment instance dataclasses
# 2) Instance construction and randomisation
# 3) Metadata and row helpers
# =============================================================================


# =============================================================================
# 1) Environment Instance Dataclasses
# =============================================================================
@dataclass(frozen=True)
class EnvironmentRandomisationConfig:
    fan_position_shift_range_m: tuple[float, float] = (-0.20, 0.20)
    fan_power_scale_range: tuple[float, float] = (0.85, 1.15)
    active_fan_subset_policy: str = "at_least_one_active"
    active_fan_count: int | None = None
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15)
    width_scale_range: tuple[float, float] = (0.85, 1.15)
    centre_shift_range_m: tuple[float, float] = (-0.20, 0.20)
    uncertainty_scale_range: tuple[float, float] = (1.0, 1.5)
    launch_perturbation_policy: str = "launch_distribution_sampler"
    latency_model_uncertainty_policy: str = "nominal_or_conservative_case"


@dataclass(frozen=True)
class EnvironmentInstance:
    environment_id: str
    W_layer: str
    environment_mode: str
    fan_count: int
    fan_positions_m: tuple[tuple[float, float], ...]
    fan_power_scales: tuple[float, ...]
    active_fan_mask: tuple[bool, ...]
    updraft_model_id: str
    updraft_amplitude_scale: float
    updraft_width_scale: float
    updraft_centre_shift_m: tuple[float, float]
    residual_field_id: str
    local_uncertainty_scale: float
    randomisation_seed: int
    claim_status: str = "simulation_only_environment_instance"
    instance_status: str = "ready"
    blocked_reason: str = ""


# =============================================================================
# 2) Instance Construction and Randomisation
# =============================================================================
def environment_instance_for_mode(
    W_layer: str,
    environment_mode: str,
    seed: int,
    *,
    randomisation_config: EnvironmentRandomisationConfig | None = None,
) -> EnvironmentInstance:
    """Return one deterministic environment instance for archive rows."""

    layer = str(W_layer).upper()
    mode = str(environment_mode)
    rng = np.random.default_rng(int(seed))
    cfg = randomisation_config or EnvironmentRandomisationConfig()

    if layer == "W0":
        if mode != "dry_air":
            return _blocked_instance(
                W_layer=layer,
                environment_mode=mode,
                seed=seed,
                reason="W0_requires_dry_air_environment",
            )
        return EnvironmentInstance(
            environment_id=f"{layer}_{mode}_s{int(seed):06d}",
            W_layer=layer,
            environment_mode=mode,
            fan_count=0,
            fan_positions_m=(),
            fan_power_scales=(),
            active_fan_mask=(),
            updraft_model_id="dry_air_zero_wind",
            updraft_amplitude_scale=1.0,
            updraft_width_scale=1.0,
            updraft_centre_shift_m=(0.0, 0.0),
            residual_field_id="none",
            local_uncertainty_scale=0.0,
            randomisation_seed=int(seed),
        )

    fan_count = 4 if mode in {"gaussian_four", "annular_gp_four", "w3_randomised_four"} else 1
    positions = _base_fan_positions(fan_count)
    power_scales = tuple(1.0 for _ in range(fan_count))
    active_mask = tuple(True for _ in range(fan_count))
    amplitude_scale = 1.0
    width_scale = 1.0
    centre_shift = (0.0, 0.0)
    uncertainty_scale = 1.0
    residual_field_id = "none"

    if mode == "dry_air":
        return _blocked_instance(
            W_layer=layer,
            environment_mode=mode,
            seed=seed,
            reason=f"{layer}_nonzero_layer_requires_updraft_environment",
        )
    if mode not in {
        "gaussian_single",
        "gaussian_four",
        "annular_gp_single",
        "annular_gp_four",
        "w3_randomised",
        "w3_randomised_single",
        "w3_randomised_four",
    }:
        return _blocked_instance(
            W_layer=layer,
            environment_mode=mode,
            seed=seed,
            reason=f"unknown_environment_mode_{mode}",
        )

    if layer == "W3" or mode in {"w3_randomised", "w3_randomised_single", "w3_randomised_four"}:
        fan_shift = _uniform_pair(rng, cfg.fan_position_shift_range_m)
        positions = tuple((float(x + fan_shift[0]), float(y + fan_shift[1])) for x, y in positions)
        power_scales = tuple(
            float(value)
            for value in rng.uniform(
                cfg.fan_power_scale_range[0],
                cfg.fan_power_scale_range[1],
                size=fan_count,
            )
        )
        if fan_count > 1:
            active_mask = _active_fan_mask(
                rng,
                fan_count,
                active_fan_count=cfg.active_fan_count,
            )
        amplitude_scale = float(
            rng.uniform(cfg.amplitude_scale_range[0], cfg.amplitude_scale_range[1])
        )
        width_scale = float(rng.uniform(cfg.width_scale_range[0], cfg.width_scale_range[1]))
        centre_shift = _uniform_pair(rng, cfg.centre_shift_range_m)
        uncertainty_scale = float(
            rng.uniform(
                cfg.uncertainty_scale_range[0],
                cfg.uncertainty_scale_range[1],
            )
        )
        residual_field_id = "randomised_residual_not_modelled"

    return EnvironmentInstance(
        environment_id=f"{layer}_{mode}_s{int(seed):06d}",
        W_layer=layer,
        environment_mode=mode,
        fan_count=fan_count,
        fan_positions_m=positions,
        fan_power_scales=power_scales,
        active_fan_mask=active_mask,
        updraft_model_id=_model_id_for_instance(layer, fan_count),
        updraft_amplitude_scale=amplitude_scale,
        updraft_width_scale=width_scale,
        updraft_centre_shift_m=centre_shift,
        residual_field_id=residual_field_id,
        local_uncertainty_scale=uncertainty_scale,
        randomisation_seed=int(seed),
    )


def sample_environment_randomisation(
    base_instance: EnvironmentInstance,
    seed: int,
    *,
    randomisation_config: EnvironmentRandomisationConfig | None = None,
) -> EnvironmentInstance:
    """Return a W3-style randomised instance based on an existing mode."""

    cfg = randomisation_config or EnvironmentRandomisationConfig()
    if base_instance.fan_count >= 4:
        mode = "w3_randomised_four"
    else:
        mode = "w3_randomised_single"
    return environment_instance_for_mode(
        "W3",
        mode,
        seed,
        randomisation_config=cfg,
    )


# =============================================================================
# 3) Metadata and Row Helpers
# =============================================================================
def environment_metadata_from_instance(instance: EnvironmentInstance) -> EnvironmentMetadata:
    """Convert an environment instance to the retained context metadata shape."""

    return EnvironmentMetadata(
        environment_id=instance.environment_id,
        fan_count=int(instance.fan_count),
        fan_positions_m=instance.fan_positions_m,
        fan_power_scales=instance.fan_power_scales,
        active_fan_mask=instance.active_fan_mask,
        updraft_model_id=instance.updraft_model_id,
        updraft_amplitude_scale=float(instance.updraft_amplitude_scale),
        updraft_width_scale=float(instance.updraft_width_scale),
        updraft_centre_shift_m=instance.updraft_centre_shift_m,
        residual_field_id=instance.residual_field_id,
        randomisation_seed=int(instance.randomisation_seed),
        model_source="resolved_by_env_instance",
        W_layer=instance.W_layer,
        wind_mode="none" if instance.W_layer == "W0" else "panel",
        local_uncertainty_scale=float(instance.local_uncertainty_scale),
        environment_mode=instance.environment_mode,
        environment_instance_id=instance.environment_id,
        claim_status=instance.claim_status,
    )


def environment_instance_row(instance: EnvironmentInstance) -> dict[str, object]:
    """Return one CSV-ready environment-instance row."""

    row = asdict(instance)
    row["fan_positions_m"] = _xy_pairs_text(instance.fan_positions_m)
    row["fan_power_scales"] = _float_tuple_text(instance.fan_power_scales)
    row["active_fan_mask"] = ";".join("1" if item else "0" for item in instance.active_fan_mask)
    row["updraft_centre_shift_m"] = _float_tuple_text(instance.updraft_centre_shift_m)
    return row


def _base_fan_positions(fan_count: int) -> tuple[tuple[float, float], ...]:
    if int(fan_count) <= 0:
        return ()
    if int(fan_count) >= 4:
        return tuple((float(x), float(y)) for x, y in FOUR_FAN_CENTERS_XY)
    return (tuple(float(value) for value in SINGLE_FAN_CENTER_XY),)


def _model_id_for_instance(W_layer: str, fan_count: int) -> str:
    layer = str(W_layer).upper()
    if layer == "W0":
        return "dry_air_zero_wind"
    if layer == "W1":
        return "four_gaussian_var" if int(fan_count) >= 4 else "single_gaussian_var"
    if layer == "W2":
        return "four_annular_gp_grid" if int(fan_count) >= 4 else "single_annular_gp_grid"
    return "four_annular_gp_grid" if int(fan_count) >= 4 else "single_annular_gp_grid"


def _active_fan_mask(
    rng: np.random.Generator,
    fan_count: int,
    *,
    active_fan_count: int | None = None,
) -> tuple[bool, ...]:
    if active_fan_count is not None:
        count = int(active_fan_count)
        if count < 1 or count > int(fan_count):
            raise ValueError("active_fan_count must be between 1 and fan_count.")
        active_indices = set(int(index) for index in rng.choice(int(fan_count), size=count, replace=False))
        return tuple(index in active_indices for index in range(int(fan_count)))
    mask = tuple(bool(value) for value in rng.integers(0, 2, size=int(fan_count)))
    if any(mask):
        return mask
    return tuple(index == 0 for index in range(int(fan_count)))


def _uniform_pair(
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    return (
        float(rng.uniform(float(bounds[0]), float(bounds[1]))),
        float(rng.uniform(float(bounds[0]), float(bounds[1]))),
    )


def _blocked_instance(
    *,
    W_layer: str,
    environment_mode: str,
    seed: int,
    reason: str,
) -> EnvironmentInstance:
    return EnvironmentInstance(
        environment_id=f"{W_layer}_{environment_mode}_blocked_s{int(seed):06d}",
        W_layer=str(W_layer),
        environment_mode=str(environment_mode),
        fan_count=0,
        fan_positions_m=(),
        fan_power_scales=(),
        active_fan_mask=(),
        updraft_model_id="blocked_unavailable",
        updraft_amplitude_scale=1.0,
        updraft_width_scale=1.0,
        updraft_centre_shift_m=(0.0, 0.0),
        residual_field_id="none",
        local_uncertainty_scale=1.0,
        randomisation_seed=int(seed),
        instance_status="blocked",
        blocked_reason=str(reason),
    )


def _xy_pairs_text(values: tuple[tuple[float, float], ...]) -> str:
    return ";".join(f"{float(x):.6f}:{float(y):.6f}" for x, y in values)


def _float_tuple_text(values: tuple[float, ...]) -> str:
    return ";".join(f"{float(value):.6f}" for value in values)
