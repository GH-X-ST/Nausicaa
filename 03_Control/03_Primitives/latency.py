from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Command lattice and latency dataclasses
# 2) Surface limits
# 3) Latency timing helpers
# 4) Command conversion helpers
# 5) Command-to-surface layer
# =============================================================================

# =============================================================================
# 1) Command Lattice and Latency Dataclasses
# =============================================================================
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
    onset_latency_s: float = 0.075
    half_response_low_s: float = 0.101
    half_response_nominal_s: float = 0.111
    half_response_high_s: float = 0.121
    half_response_upper_s: float = 0.133
    vicon_filter_cutoff_hz: float = 20.0
    command_dt_s: float = 0.02


@dataclass(frozen=True)
class CommandToSurfaceConfig:
    mode: str = "nominal"
    quantise: bool = True
    use_onset_delay: bool = True


# =============================================================================
# 2) Surface Limits
# =============================================================================
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
    return float((half - envelope.onset_latency_s) / np.log(2.0))


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


# =============================================================================
# 4) Command Conversion Helpers
# =============================================================================
def angle_to_command_norm(angle_rad: float, limit: SurfaceLimit) -> float:
    angle_deg = float(np.rad2deg(angle_rad))
    if angle_deg >= 0.0:
        norm = angle_deg / max(abs(limit.positive_deg), 1e-12)
    else:
        norm = angle_deg / max(abs(limit.negative_deg), 1e-12)
    return float(np.clip(norm, -1.0, 1.0))


def command_norm_to_angle(norm: float, limit: SurfaceLimit) -> float:
    value = float(np.clip(norm, -1.0, 1.0))
    if value >= 0.0:
        return float(np.deg2rad(value * limit.positive_deg))
    return float(np.deg2rad(value * abs(limit.negative_deg)))


def quantise_command_norm(norm: float) -> float:
    idx = int(np.argmin(np.abs(COMMAND_LEVELS - float(norm))))
    return float(COMMAND_LEVELS[idx])


def aggregate_targets_to_surface_degrees(target_rad: np.ndarray) -> dict[str, float]:
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


# =============================================================================
# 5) Command-to-Surface Layer
# =============================================================================
class CommandToSurfaceLayer:
    def __init__(
        self,
        config: CommandToSurfaceConfig | None = None,
        envelope: LatencyEnvelope | None = None,
    ):
        self.config = config or CommandToSurfaceConfig()
        self.envelope = envelope or LatencyEnvelope()
        self._buffers = [deque(), deque(), deque()]
        self._last_target = np.zeros(3)
        self._last_norm = np.zeros(3)

    @property
    def actuator_tau_vector_s(self) -> tuple[float, float, float]:
        tau = actuator_tau_s(self.config, self.envelope)
        return (tau, tau, tau)

    def reset(self, initial_target_rad: np.ndarray | list[float] | tuple[float, ...]) -> None:
        initial = np.asarray(initial_target_rad, dtype=float).reshape(3)
        self._buffers = []
        delay_steps = self._delay_steps()
        for value in initial:
            self._buffers.append(deque([float(value)] * delay_steps))
        self._last_target = initial.copy()
        self._last_norm = self._angle_vector_to_norm(initial)

    def apply(self, desired_rad: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        desired = np.asarray(desired_rad, dtype=float).reshape(3)
        norm = self._angle_vector_to_norm(desired)
        if self.config.quantise:
            norm = np.asarray([quantise_command_norm(value) for value in norm], dtype=float)
        target = np.asarray(
            [
                command_norm_to_angle(norm[0], AGGREGATE_LIMITS["delta_a"]),
                command_norm_to_angle(norm[1], AGGREGATE_LIMITS["delta_e"]),
                command_norm_to_angle(norm[2], AGGREGATE_LIMITS["delta_r"]),
            ],
            dtype=float,
        )
        delayed = target.copy()
        if self.config.use_onset_delay:
            for idx, buffer in enumerate(self._buffers):
                buffer.append(float(target[idx]))
                delayed[idx] = buffer.popleft()
        self._last_norm = norm.copy()
        self._last_target = delayed.copy()
        return delayed

    def log_fields(self) -> dict[str, float | str]:
        surface = aggregate_targets_to_surface_degrees(self._last_target)
        tau = actuator_tau_s(self.config, self.envelope)
        return {
            "command_norm_a": float(self._last_norm[0]),
            "command_norm_e": float(self._last_norm[1]),
            "command_norm_r": float(self._last_norm[2]),
            "delta_a_target_deg": float(np.rad2deg(self._last_target[0])),
            "delta_e_target_deg": float(np.rad2deg(self._last_target[1])),
            "delta_r_target_deg": float(np.rad2deg(self._last_target[2])),
            **surface,
            "latency_mode": self.config.mode,
            "latency_s": half_response_s(self.config, self.envelope),
            "latency_range_s": latency_range_label(self.config, self.envelope),
            "onset_latency_s": float(self.envelope.onset_latency_s),
            "half_response_s": half_response_s(self.config, self.envelope),
            "actuator_tau_s": float(tau),
            "vicon_filter_cutoff_hz": float(self.envelope.vicon_filter_cutoff_hz),
        }

    def _delay_steps(self) -> int:
        if not self.config.use_onset_delay:
            return 0
        dt = max(float(self.envelope.command_dt_s), 1e-12)
        return int(np.ceil(float(self.envelope.onset_latency_s) / dt))

    @staticmethod
    def _angle_vector_to_norm(angle_rad: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                angle_to_command_norm(angle_rad[0], AGGREGATE_LIMITS["delta_a"]),
                angle_to_command_norm(angle_rad[1], AGGREGATE_LIMITS["delta_e"]),
                angle_to_command_norm(angle_rad[2], AGGREGATE_LIMITS["delta_r"]),
            ],
            dtype=float,
        )
