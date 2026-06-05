from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from command_contract import AGGREGATE_LIMITS, SURFACE_STATE_NAMES, command_norm_to_angle

SURFACE_IMPLEMENTATION_EFFECTIVENESS_SCALE_RANGE = (0.50, 1.00)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Implementation instance dataclass
# 2) Instance construction
# 3) Surface and actuator adjustment helpers
# =============================================================================


# =============================================================================
# 1) Implementation Instance Dataclass
# =============================================================================
@dataclass(frozen=True)
class ImplementationInstance:
    implementation_instance_id: str
    W_layer: str
    actuator_randomisation_seed: int
    latency_case: str
    state_feedback_delay_scale: float
    command_onset_delay_scale: float
    command_transport_delay_scale: float
    latency_jitter_s: float
    actuator_tau_scale: float
    aileron_effectiveness_scale: float
    elevator_effectiveness_scale: float
    rudder_effectiveness_scale: float
    surface_neutral_bias_rad: tuple[float, float, float]
    surface_limit_scale: float
    left_right_aileron_asymmetry_scale: float
    command_quantisation_mode: str
    implementation_adjustment_status: str
    implementation_adjustment_limitations: str
    claim_status: str = "simulation_only_implementation_instance"


# =============================================================================
# 2) Instance Construction
# =============================================================================
def implementation_instance_for_layer(
    W_layer: str,
    seed: int,
    *,
    latency_case: str = "nominal",
) -> ImplementationInstance:
    """Return deterministic implementation/actuator metadata for a W layer."""

    layer = str(W_layer).upper()
    rng = np.random.default_rng(int(seed))
    if layer in {"W0", "W1"}:
        return _implementation_instance(
            layer=layer,
            seed=seed,
            latency_case="none" if layer == "W0" else latency_case,
        )
    if layer == "W2":
        return _implementation_instance(
            layer=layer,
            seed=seed,
            latency_case=latency_case if latency_case in {"nominal", "conservative"} else "nominal",
            state_feedback_delay_scale=1.0,
            command_onset_delay_scale=1.0,
            command_transport_delay_scale=1.0,
            actuator_tau_scale=1.05,
            implementation_adjustment_status="deterministic_hardware_aware",
        )
    if layer == "W3":
        return _implementation_instance(
            layer=layer,
            seed=seed,
            latency_case="conservative" if rng.random() > 0.5 else "nominal",
            state_feedback_delay_scale=float(rng.uniform(0.90, 1.20)),
            command_onset_delay_scale=float(rng.uniform(0.90, 1.25)),
            command_transport_delay_scale=float(rng.uniform(0.90, 1.25)),
            latency_jitter_s=float(rng.uniform(0.0, 0.015)),
            actuator_tau_scale=float(rng.uniform(0.90, 1.25)),
            aileron_effectiveness_scale=float(rng.uniform(*SURFACE_IMPLEMENTATION_EFFECTIVENESS_SCALE_RANGE)),
            elevator_effectiveness_scale=float(rng.uniform(*SURFACE_IMPLEMENTATION_EFFECTIVENESS_SCALE_RANGE)),
            rudder_effectiveness_scale=float(rng.uniform(*SURFACE_IMPLEMENTATION_EFFECTIVENESS_SCALE_RANGE)),
            surface_neutral_bias_rad=tuple(float(value) for value in rng.uniform(-0.015, 0.015, size=3)),
            surface_limit_scale=float(rng.uniform(0.90, 1.00)),
            left_right_aileron_asymmetry_scale=float(rng.uniform(0.95, 1.05)),
            command_quantisation_mode="fixed_20_percent_lattice",
            implementation_adjustment_status="randomised_applied",
            implementation_adjustment_limitations="left-right aileron asymmetry applied to per-strip control mix",
        )
    return _implementation_instance(
        layer=layer,
        seed=seed,
        latency_case=latency_case,
        implementation_adjustment_status="blocked_unknown_W_layer",
        implementation_adjustment_limitations=f"unknown W layer {layer}",
    )


def _implementation_instance(
    *,
    layer: str,
    seed: int,
    latency_case: str,
    state_feedback_delay_scale: float = 1.0,
    command_onset_delay_scale: float = 1.0,
    command_transport_delay_scale: float = 1.0,
    latency_jitter_s: float = 0.0,
    actuator_tau_scale: float = 1.0,
    aileron_effectiveness_scale: float = 1.0,
    elevator_effectiveness_scale: float = 1.0,
    rudder_effectiveness_scale: float = 1.0,
    surface_neutral_bias_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    surface_limit_scale: float = 1.0,
    left_right_aileron_asymmetry_scale: float = 1.0,
    command_quantisation_mode: str = "none",
    implementation_adjustment_status: str = "nominal_no_perturbation",
    implementation_adjustment_limitations: str = "",
) -> ImplementationInstance:
    return ImplementationInstance(
        implementation_instance_id=f"{layer}_impl_s{int(seed):06d}",
        W_layer=str(layer),
        actuator_randomisation_seed=int(seed),
        latency_case=str(latency_case),
        state_feedback_delay_scale=float(state_feedback_delay_scale),
        command_onset_delay_scale=float(command_onset_delay_scale),
        command_transport_delay_scale=float(command_transport_delay_scale),
        latency_jitter_s=float(latency_jitter_s),
        actuator_tau_scale=float(actuator_tau_scale),
        aileron_effectiveness_scale=float(aileron_effectiveness_scale),
        elevator_effectiveness_scale=float(elevator_effectiveness_scale),
        rudder_effectiveness_scale=float(rudder_effectiveness_scale),
        surface_neutral_bias_rad=tuple(float(value) for value in surface_neutral_bias_rad),
        surface_limit_scale=float(surface_limit_scale),
        left_right_aileron_asymmetry_scale=float(left_right_aileron_asymmetry_scale),
        command_quantisation_mode=str(command_quantisation_mode),
        implementation_adjustment_status=str(implementation_adjustment_status),
        implementation_adjustment_limitations=str(implementation_adjustment_limitations),
    )


def implementation_instance_row(instance: ImplementationInstance) -> dict[str, object]:
    """Return CSV-ready implementation metadata."""

    row = asdict(instance)
    row["surface_neutral_bias_rad"] = ";".join(
        f"{float(value):.9g}" for value in instance.surface_neutral_bias_rad
    )
    return row


# =============================================================================
# 3) Surface and Actuator Adjustment Helpers
# =============================================================================
def adjusted_actuator_tau_s(
    base_tau_s: tuple[float, float, float],
    instance: ImplementationInstance,
) -> tuple[float, float, float]:
    """Return actuator lag constants after implementation perturbation."""

    scale = max(float(instance.actuator_tau_scale), 1e-6)
    return tuple(float(value) * scale for value in base_tau_s)


def apply_surface_implementation(
    command_rad: np.ndarray,
    instance: ImplementationInstance,
) -> np.ndarray:
    """Apply deterministic surface effectiveness, bias, and limit perturbations."""

    command = np.asarray(command_rad, dtype=float).reshape(3).copy()
    effectiveness = np.asarray(
        [
            float(instance.aileron_effectiveness_scale),
            float(instance.elevator_effectiveness_scale),
            float(instance.rudder_effectiveness_scale),
        ],
        dtype=float,
    )
    bias = np.asarray(instance.surface_neutral_bias_rad, dtype=float).reshape(3)
    adjusted = command * effectiveness + bias
    return _clip_surface_rad(adjusted, surface_limit_scale=instance.surface_limit_scale)


def apply_aileron_asymmetry_to_aircraft(aircraft, instance: ImplementationInstance):
    """Return an aircraft model with side-specific aileron effectiveness applied."""

    scale = float(instance.left_right_aileron_asymmetry_scale)
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-12):
        return aircraft
    control_mix = np.asarray(aircraft.control_mix, dtype=float).copy()
    r_strip_b = np.asarray(aircraft.r_strip_b, dtype=float)
    aileron_active = np.abs(control_mix[:, 0]) > 0.0
    if not np.any(aileron_active):
        return aircraft
    starboard = aileron_active & (r_strip_b[:, 1] >= 0.0)
    port = aileron_active & (r_strip_b[:, 1] < 0.0)
    control_mix[starboard, 0] *= scale
    control_mix[port, 0] /= max(scale, 1e-12)
    return replace(aircraft, control_mix=control_mix)


def _clip_surface_rad(command_rad: np.ndarray, *, surface_limit_scale: float) -> np.ndarray:
    scale = float(np.clip(float(surface_limit_scale), 1e-6, 1.0))
    clipped = np.asarray(command_rad, dtype=float).reshape(3).copy()
    for index, name in enumerate(SURFACE_STATE_NAMES):
        limit = AGGREGATE_LIMITS[name]
        low = command_norm_to_angle(-1.0, limit) * scale
        high = command_norm_to_angle(1.0, limit) * scale
        clipped[index] = np.clip(clipped[index], min(low, high), max(low, high))
    return clipped
