from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from updraft_models import (
    FOUR_FAN_CENTERS_XY,
    SINGLE_FAN_CENTER_XY,
    WindField,
    build_environment_adjusted_wind_field,
    build_randomised_wind_field,
    load_updraft_model,
    sample_updraft_randomisation,
    updraft_randomisation_label,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Surrogate ladder constants and dataclass
# 2) Public resolver and validation helpers
# 3) Wind-field loader helpers
# =============================================================================


# =============================================================================
# 1) Surrogate Ladder Constants and Dataclass
# =============================================================================
W_LAYERS = ("W0", "W1", "W2", "W3")
GAUSSIAN_PLUME_IDS = ("single_gaussian_var", "four_gaussian_var")
ANNULAR_GP_IDS = ("single_annular_gp_grid", "four_annular_gp_grid")
DRY_AIR_MODEL_ID = "dry_air_zero_wind"
READY_STATUS = "ready"
R5_ANNULAR_GP_TRAINING_MODES = (
    "w1_annular_gp_randomised_single",
    "w1_annular_gp_randomised_four",
)


@dataclass(frozen=True)
class SurrogateBinding:
    W_layer: str
    wind_mode: str
    updraft_model_id: str
    updraft_model_source: str
    surrogate_family: str
    surrogate_role: str
    surrogate_binding_status: str
    randomisation_label: str
    fan_count: int
    fan_positions_m: tuple[tuple[float, float], ...]
    fan_power_scales: tuple[float, ...]
    active_fan_mask: tuple[bool, ...] = ()
    updraft_amplitude_scale: float = 1.0
    updraft_width_scale: float = 1.0
    updraft_centre_shift_m: tuple[float, float] = (0.0, 0.0)
    local_uncertainty_scale: float = 1.0
    environment_mode: str = ""
    blocked_reason: str = ""
    randomisation_seed: int | None = None


# =============================================================================
# 2) Public Resolver and Validation Helpers
# =============================================================================
def resolve_surrogate_binding(
    W_layer: str,
    environment_metadata,
    *,
    repo_root: Path | None = None,
    randomisation_seed: int | None = None,
) -> SurrogateBinding:
    """Resolve the strict W-layer surrogate without falling back across families."""

    layer = str(W_layer).upper()
    if layer not in W_LAYERS:
        return _blocked_binding(
            W_layer=layer,
            requested_model_id=str(getattr(environment_metadata, "updraft_model_id", "")),
            reason=f"unknown_W_layer_{layer}",
            metadata=environment_metadata,
            randomisation_seed=randomisation_seed,
        )

    requested_model_id = str(
        getattr(environment_metadata, "updraft_model_id", DRY_AIR_MODEL_ID)
        or DRY_AIR_MODEL_ID
    )
    requested_was_default = requested_model_id == DRY_AIR_MODEL_ID

    if layer == "W0":
        if not requested_was_default:
            return _blocked_binding(
                W_layer=layer,
                requested_model_id=requested_model_id,
                reason="W0_requires_dry_air_zero_wind",
                metadata=environment_metadata,
                randomisation_seed=randomisation_seed,
            )
        return SurrogateBinding(
            W_layer=layer,
            wind_mode="none",
            updraft_model_id=DRY_AIR_MODEL_ID,
            updraft_model_source="not_applicable",
            surrogate_family="dry_air_zero_wind",
            surrogate_role="baseline_no_wind",
            surrogate_binding_status=READY_STATUS,
            randomisation_label="none",
            fan_count=0,
            fan_positions_m=(),
            fan_power_scales=(),
            active_fan_mask=(),
            updraft_amplitude_scale=1.0,
            updraft_width_scale=1.0,
            updraft_centre_shift_m=(0.0, 0.0),
            local_uncertainty_scale=0.0,
            environment_mode=str(getattr(environment_metadata, "environment_mode", "dry_air")),
            blocked_reason="",
            randomisation_seed=randomisation_seed,
        )

    model_id = _model_id_for_layer(
        layer=layer,
        requested_model_id=requested_model_id,
        requested_was_default=requested_was_default,
        fan_count=int(getattr(environment_metadata, "fan_count", 1)),
    )
    environment_mode = str(getattr(environment_metadata, "environment_mode", ""))
    invalid_reason = _invalid_ladder_reason(layer, model_id, environment_mode=environment_mode)
    if invalid_reason:
        return _blocked_binding(
            W_layer=layer,
            requested_model_id=model_id,
            reason=invalid_reason,
            metadata=environment_metadata,
            randomisation_seed=randomisation_seed,
        )

    try:
        wind = _cached_base_wind(model_id, _repo_root_key(repo_root))
    except (FileNotFoundError, ValueError, OSError) as exc:
        return _blocked_binding(
            W_layer=layer,
            requested_model_id=model_id,
            reason=f"missing_or_unreadable_surrogate:{type(exc).__name__}:{exc}",
            metadata=environment_metadata,
            randomisation_seed=randomisation_seed,
        )

    fan_positions = _fan_positions_for_model(model_id, environment_metadata)
    fan_count = len(fan_positions)
    power_scales = _fan_power_scales(fan_count, environment_metadata)
    active_mask = _active_fan_mask(fan_count, environment_metadata)
    seed = (
        randomisation_seed
        if randomisation_seed is not None
        else getattr(environment_metadata, "randomisation_seed", None)
    )
    if layer == "W3" or _is_r5_annular_gp_training_mode(layer, environment_mode):
        randomisation = sample_updraft_randomisation(seed=0 if seed is None else int(seed))
        randomisation_label = updraft_randomisation_label(randomisation)
    else:
        randomisation_label = "none"

    return SurrogateBinding(
        W_layer=layer,
        wind_mode="panel",
        updraft_model_id=model_id,
        updraft_model_source=str(wind.source),
        surrogate_family=_surrogate_family(layer, environment_mode=environment_mode),
        surrogate_role=_surrogate_role(layer, environment_mode=environment_mode),
        surrogate_binding_status=READY_STATUS,
        randomisation_label=randomisation_label,
        fan_count=fan_count,
        fan_positions_m=fan_positions,
        fan_power_scales=power_scales,
        active_fan_mask=active_mask,
        updraft_amplitude_scale=float(getattr(environment_metadata, "updraft_amplitude_scale", 1.0)),
        updraft_width_scale=float(getattr(environment_metadata, "updraft_width_scale", 1.0)),
        updraft_centre_shift_m=tuple(
            float(value)
            for value in tuple(getattr(environment_metadata, "updraft_centre_shift_m", (0.0, 0.0)))
        ),
        local_uncertainty_scale=float(getattr(environment_metadata, "local_uncertainty_scale", 1.0)),
        environment_mode=environment_mode,
        blocked_reason="",
        randomisation_seed=None if seed is None else int(seed),
    )


def validate_surrogate_ladder(binding: SurrogateBinding) -> None:
    """Raise if a binding violates the strict surrogate ladder."""

    if binding.surrogate_binding_status != READY_STATUS:
        raise ValueError(f"surrogate binding is blocked: {binding.blocked_reason}")
    reason = _invalid_ladder_reason(
        binding.W_layer,
        binding.updraft_model_id,
        environment_mode=binding.environment_mode,
    )
    if reason:
        raise ValueError(reason)
    if binding.W_layer == "W0" and (
        binding.wind_mode != "none" or binding.updraft_model_id != DRY_AIR_MODEL_ID
    ):
        raise ValueError("W0 binding must be dry-air zero-wind.")
    if binding.W_layer in {"W1", "W2", "W3"} and binding.wind_mode != "panel":
        raise ValueError("non-W0 bindings must use panel wind sampling.")


def surrogate_binding_row(binding: SurrogateBinding) -> dict[str, object]:
    """Return a CSV-ready surrogate binding row."""

    row = asdict(binding)
    row["fan_positions_m"] = _xy_pairs_text(binding.fan_positions_m)
    row["fan_power_scales"] = _float_tuple_text(binding.fan_power_scales)
    row["active_fan_mask"] = ";".join("1" if value else "0" for value in binding.active_fan_mask)
    row["updraft_centre_shift_m"] = _float_tuple_text(binding.updraft_centre_shift_m)
    row["randomisation_seed"] = (
        "" if binding.randomisation_seed is None else int(binding.randomisation_seed)
    )
    return row


# =============================================================================
# 3) Wind-Field Loader Helpers
# =============================================================================
def wind_field_for_binding(
    binding: SurrogateBinding,
    *,
    repo_root: Path | None = None,
) -> WindField | None:
    """Return the wind field for a ready binding, or None for dry air/blocked rows."""

    if binding.surrogate_binding_status != READY_STATUS:
        return None
    if binding.W_layer == "W0":
        return None
    base = _cached_base_wind(binding.updraft_model_id, _repo_root_key(repo_root))
    adjusted = build_environment_adjusted_wind_field(
        base,
        amplitude_scale=binding.updraft_amplitude_scale,
        width_scale=binding.updraft_width_scale,
        centre_shift_m=binding.updraft_centre_shift_m,
        fan_positions_m=binding.fan_positions_m,
        fan_power_scales=binding.fan_power_scales,
        active_fan_mask=binding.active_fan_mask,
        local_uncertainty_scale=binding.local_uncertainty_scale,
        transform_label=str(binding.environment_mode or "environment_instance"),
    )
    if binding.W_layer != "W3" and not _is_r5_annular_gp_training_mode(
        binding.W_layer,
        binding.environment_mode,
    ):
        return adjusted
    wind, _ = build_randomised_wind_field(
        adjusted,
        seed=0 if binding.randomisation_seed is None else int(binding.randomisation_seed),
        enabled=True,
    )
    return wind


@lru_cache(maxsize=16)
def _cached_base_wind(model_id: str, repo_root_key: str) -> WindField:
    root = None if repo_root_key == "" else Path(repo_root_key)
    return load_updraft_model(str(model_id), repo_root=root)


def _repo_root_key(repo_root: Path | None) -> str:
    return "" if repo_root is None else str(Path(repo_root).resolve())


def _model_id_for_layer(
    *,
    layer: str,
    requested_model_id: str,
    requested_was_default: bool,
    fan_count: int,
) -> str:
    if not requested_was_default:
        return requested_model_id
    if layer == "W1":
        return "four_gaussian_var" if int(fan_count) >= 4 else "single_gaussian_var"
    return "four_annular_gp_grid" if int(fan_count) >= 4 else "single_annular_gp_grid"


def _invalid_ladder_reason(layer: str, model_id: str, *, environment_mode: str = "") -> str:
    if layer == "W1" and _is_r5_annular_gp_training_mode(layer, environment_mode):
        if model_id not in ANNULAR_GP_IDS:
            return "W1_annular_gp_training_requires_annular_gp_surrogate"
        return ""
    if layer == "W1" and model_id not in GAUSSIAN_PLUME_IDS:
        return "W1_requires_gaussian_plume_surrogate"
    if layer == "W2" and model_id not in ANNULAR_GP_IDS:
        return "W2_requires_gp_corrected_annular_gaussian_surrogate"
    if layer == "W3" and model_id not in ANNULAR_GP_IDS:
        return "W3_requires_randomised_gp_corrected_annular_gaussian_surrogate"
    return ""


def _surrogate_family(layer: str, *, environment_mode: str = "") -> str:
    if _is_r5_annular_gp_training_mode(layer, environment_mode):
        return "randomised_gp_corrected_annular_gaussian_training"
    if layer == "W1":
        return "gaussian_plume"
    if layer == "W2":
        return "gp_corrected_annular_gaussian"
    if layer == "W3":
        return "randomised_gp_corrected_annular_gaussian"
    return "dry_air_zero_wind"


def _surrogate_role(layer: str, *, environment_mode: str = "") -> str:
    if _is_r5_annular_gp_training_mode(layer, environment_mode):
        return "r5_annular_gp_randomised_training_surrogate"
    return {
        "W1": "measured_gaussian_plume_preflight",
        "W2": "gp_corrected_annular_surrogate",
        "W3": "randomised_gp_corrected_annular_surrogate",
    }.get(layer, "baseline_no_wind")


def _is_r5_annular_gp_training_mode(layer: str, environment_mode: str) -> bool:
    return str(layer).upper() == "W1" and str(environment_mode) in R5_ANNULAR_GP_TRAINING_MODES


def _fan_positions_for_model(
    model_id: str,
    metadata,
) -> tuple[tuple[float, float], ...]:
    supplied = tuple(
        (float(x), float(y))
        for x, y in tuple(getattr(metadata, "fan_positions_m", ()) or ())
    )
    if supplied:
        return supplied
    if model_id.startswith("four_"):
        return tuple((float(x), float(y)) for x, y in FOUR_FAN_CENTERS_XY)
    return (tuple(float(value) for value in SINGLE_FAN_CENTER_XY),)


def _fan_power_scales(fan_count: int, metadata) -> tuple[float, ...]:
    supplied = tuple(float(value) for value in tuple(getattr(metadata, "fan_power_scales", ()) or ()))
    if supplied:
        return supplied
    return tuple(1.0 for _ in range(int(fan_count)))


def _active_fan_mask(fan_count: int, metadata) -> tuple[bool, ...]:
    supplied = tuple(bool(value) for value in tuple(getattr(metadata, "active_fan_mask", ()) or ()))
    if supplied:
        return supplied
    return tuple(True for _ in range(int(fan_count)))


def _blocked_binding(
    *,
    W_layer: str,
    requested_model_id: str,
    reason: str,
    metadata,
    randomisation_seed: int | None,
) -> SurrogateBinding:
    fan_positions = tuple(
        (float(x), float(y))
        for x, y in tuple(getattr(metadata, "fan_positions_m", ()) or ())
    )
    fan_count = int(getattr(metadata, "fan_count", len(fan_positions)))
    if not fan_positions and fan_count > 0:
        fan_positions = tuple((float(x), float(y)) for x, y in FOUR_FAN_CENTERS_XY[:fan_count])
    return SurrogateBinding(
        W_layer=str(W_layer),
        wind_mode="blocked",
        updraft_model_id=str(requested_model_id),
        updraft_model_source="unavailable",
        surrogate_family="blocked",
        surrogate_role="blocked_no_fallback",
        surrogate_binding_status="blocked",
        randomisation_label="none",
        fan_count=fan_count,
        fan_positions_m=fan_positions,
        fan_power_scales=_fan_power_scales(fan_count, metadata),
        active_fan_mask=_active_fan_mask(fan_count, metadata),
        updraft_amplitude_scale=float(getattr(metadata, "updraft_amplitude_scale", 1.0)),
        updraft_width_scale=float(getattr(metadata, "updraft_width_scale", 1.0)),
        updraft_centre_shift_m=tuple(
            float(value)
            for value in tuple(getattr(metadata, "updraft_centre_shift_m", (0.0, 0.0)))
        ),
        local_uncertainty_scale=float(getattr(metadata, "local_uncertainty_scale", 1.0)),
        environment_mode=str(getattr(metadata, "environment_mode", "")),
        blocked_reason=str(reason),
        randomisation_seed=randomisation_seed,
    )


def _xy_pairs_text(values: tuple[tuple[float, float], ...]) -> str:
    return ";".join(f"{float(x):.6f}:{float(y):.6f}" for x, y in values)


def _float_tuple_text(values: tuple[float, ...]) -> str:
    return ";".join(f"{float(value):.6f}" for value in values)
