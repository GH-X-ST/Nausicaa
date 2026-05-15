from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Command lattice and latency dataclasses
# 2) Surface limits
# 3) Latency timing helpers
# 4) Command conversion helpers
# =============================================================================

# =============================================================================
# 1) Command Lattice and Latency Dataclasses
# =============================================================================
# Normalised transmitter levels from the measured command interface
COMMAND_LEVELS = np.array(
    [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    dtype=float,
)


@dataclass(frozen=True)
class SurfaceLimit:
    name: str
    positive_deg: float
    negative_deg: float


@dataclass(frozen=True)
class LatencyEnvelope:
    # Command-response values use the latest accepted 20 Hz one-pole filtered runs.
    onset_latency_s: float = 0.073
    half_response_low_s: float = 0.098
    half_response_nominal_s: float = 0.108
    half_response_high_s: float = 0.118
    half_response_upper_s: float = 0.151
    actuator_t90_s: float = 0.130
    conservative_actuator_bound_s: float = 0.151
    # Vicon latency and filter delay are state-feedback terms, not actuator lag.
    vicon_latency_nominal_s: float = 0.0149
    vicon_latency_p95_s: float = 0.0169
    vicon_filter_delay_s: float = 0.0080
    vicon_filter_cutoff_hz: float = 20.0
    vicon_filter_model: str = "one_pole"
    command_dt_s: float = 0.02


@dataclass(frozen=True)
class CommandToSurfaceConfig:
    mode: str = "nominal"
    use_state_feedback_delay: bool = True


# =============================================================================
# 2) Surface Limits
# =============================================================================
# Direction-specific surface limits are stored in degrees for audit tables
SURFACE_LIMITS = {
    "Aileron_L": SurfaceLimit("Aileron_L", 22.0, -26.0),
    "Aileron_R": SurfaceLimit("Aileron_R", -22.0, 26.0),
    "Rudder": SurfaceLimit("Rudder", 28.0, -35.0),
    "Elevator": SurfaceLimit("Elevator", 22.0, -30.0),
}
AGGREGATE_LIMITS = {
    "delta_a": SurfaceLimit("delta_a_eff", 22.0, -26.0),
    "delta_e": SurfaceLimit("delta_e_eff", 22.0, -30.0),
    "delta_r": SurfaceLimit("delta_r_eff", 28.0, -35.0),
}


# =============================================================================
# 3) Latency Timing Helpers
# =============================================================================
def actuator_tau_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> float:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    half = half_response_s(config, envelope)
    # First-order lag reaches half response after tau * log(2)
    return float((half - envelope.onset_latency_s) / np.log(2.0))


def feedback_delay_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> float:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    if not config.use_state_feedback_delay:
        return 0.0
    # The simulated controller receives Vicon-filtered state, separate from actuator response.
    return float(envelope.vicon_latency_nominal_s + envelope.vicon_filter_delay_s)


def half_response_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> float:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    if config.mode == "low":
        return float(envelope.half_response_low_s)
    if config.mode == "nominal":
        return float(envelope.half_response_nominal_s)
    if config.mode == "high":
        return float(envelope.half_response_high_s)
    if config.mode == "robust_upper":
        return float(envelope.half_response_upper_s)
    raise ValueError("latency mode must be 'low', 'nominal', 'high', or 'robust_upper'.")


def latency_range_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> tuple[float, float]:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    if config.mode == "robust_upper":
        # Robust-upper mode spans nominal to measured upper timing
        return (
            float(envelope.half_response_nominal_s),
            float(envelope.half_response_upper_s),
        )
    return (
        float(envelope.half_response_low_s),
        float(envelope.half_response_high_s),
    )


def latency_range_label(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> str:
    low, high = latency_range_s(config, envelope)
    return f"{low:.6f}:{high:.6f}"


def latency_audit_fields(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> dict[str, float | str]:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    # These fields make the measured command path explicit in every scenario log.
    return {
        "state_feedback_delay_s": feedback_delay_s(config, envelope),
        "actuator_t10_s": float(envelope.onset_latency_s),
        "actuator_t50_nominal_s": float(envelope.half_response_nominal_s),
        "actuator_t90_s": float(envelope.actuator_t90_s),
        "conservative_actuator_bound_s": float(envelope.conservative_actuator_bound_s),
        "vicon_filter_cutoff_hz": float(envelope.vicon_filter_cutoff_hz),
        "vicon_filter_model": str(envelope.vicon_filter_model),
    }


# =============================================================================
# 4) Command Conversion Helpers
# =============================================================================
def angle_to_command_norm(angle_rad: float, limit: SurfaceLimit) -> float:
    # Endpoint signs are physical surface signs; the right aileron intentionally
    # maps negative command to positive physical deflection.
    angle_deg = float(np.rad2deg(angle_rad))
    if abs(angle_deg) <= 1e-12:
        return 0.0

    candidates: list[float] = []
    if abs(limit.positive_deg) > 1e-12:
        positive_ratio = angle_deg / float(limit.positive_deg)
        if positive_ratio >= 0.0:
            candidates.append(float(positive_ratio))
    if abs(limit.negative_deg) > 1e-12:
        negative_ratio = angle_deg / float(limit.negative_deg)
        if negative_ratio >= 0.0:
            candidates.append(float(-negative_ratio))
    if not candidates:
        endpoint = (
            limit.positive_deg
            if abs(angle_deg - limit.positive_deg) <= abs(angle_deg - limit.negative_deg)
            else limit.negative_deg
        )
        norm = 1.0 if endpoint == limit.positive_deg else -1.0
        return float(norm)
    norm = min(candidates, key=lambda value: abs(abs(value) - 1.0) if abs(value) > 1.0 else abs(value))
    return float(np.clip(norm, -1.0, 1.0))


def command_norm_to_angle(norm: float, limit: SurfaceLimit) -> float:
    value = float(np.clip(norm, -1.0, 1.0))
    if value >= 0.0:
        return float(np.deg2rad(value * limit.positive_deg))
    return float(np.deg2rad((-value) * limit.negative_deg))


def quantise_command_norm(norm: float) -> float:
    idx = int(np.argmin(np.abs(COMMAND_LEVELS - float(norm))))
    return float(COMMAND_LEVELS[idx])


def aggregate_targets_to_surface_degrees(target_rad: np.ndarray) -> dict[str, float]:
    # Aggregate commands are expanded to physical surface directions
    delta_a, delta_e, delta_r = np.asarray(target_rad, dtype=float).reshape(3)
    a_norm = angle_to_command_norm(delta_a, AGGREGATE_LIMITS["delta_a"])
    e_norm = angle_to_command_norm(delta_e, AGGREGATE_LIMITS["delta_e"])
    r_norm = angle_to_command_norm(delta_r, AGGREGATE_LIMITS["delta_r"])
    return {
        "aileron_l_deg": float(np.rad2deg(command_norm_to_angle(a_norm, SURFACE_LIMITS["Aileron_L"]))),
        "aileron_r_deg": float(np.rad2deg(command_norm_to_angle(a_norm, SURFACE_LIMITS["Aileron_R"]))),
        "elevator_deg": float(np.rad2deg(command_norm_to_angle(e_norm, SURFACE_LIMITS["Elevator"]))),
        "rudder_deg": float(np.rad2deg(command_norm_to_angle(r_norm, SURFACE_LIMITS["Rudder"]))),
    }
