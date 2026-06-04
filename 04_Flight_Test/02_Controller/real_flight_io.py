from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from command_contract import (
    normalised_command_to_surface_rad,
    quantise_normalised_command_vector,
)
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Hardware command adapter
# 2) Vicon rigid-body state adapter
# 3) Math helpers
# =============================================================================


# =============================================================================
# 1) Hardware Command Adapter
# =============================================================================
PHYSICAL_SURFACE_ORDER = ("Aileron_L", "Aileron_R", "Rudder", "Elevator")
RECEIVER_CHANNEL_SURFACE_ORDER = ("Aileron_R", "Aileron_L", "Rudder", "Elevator")
SERVO_SIGNS = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=float)
PHYSICAL_FROM_AGGREGATE_MATRIX = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
ARDUINO_PACKET_HEADER = ord("V")
ARDUINO_SURFACE_COUNT = 4
ARDUINO_PACKET_LENGTH = 15


@dataclass(frozen=True)
class HardwareCommandPacket:
    aggregate_command_norm: tuple[float, float, float]
    physical_surface_norm: tuple[float, float, float, float]
    packet_surface_norm: tuple[float, float, float, float]
    packet_codes_by_surface: tuple[int, int, int, int]
    receiver_channel_codes: tuple[int, int, int, int]
    packet_bytes: bytes


def aggregate_to_physical_surface_norm(command_norm: np.ndarray) -> np.ndarray:
    """Expand aggregate aileron/elevator/rudder command to physical surfaces."""

    aggregate = quantise_normalised_command_vector(command_norm)
    return np.clip(PHYSICAL_FROM_AGGREGATE_MATRIX @ aggregate, -1.0, 1.0)


def encode_arduino_command_packet(
    command_norm: np.ndarray,
    *,
    sequence: int = 0,
    active_surface_mask: int = 0,
) -> HardwareCommandPacket:
    """Encode the real-flight 15-byte Nano transmitter packet."""

    aggregate = quantise_normalised_command_vector(command_norm)
    physical_norm = aggregate_to_physical_surface_norm(aggregate)
    packet_surface_norm = np.clip(SERVO_SIGNS * physical_norm, -1.0, 1.0)
    packet_codes_by_surface = tuple(
        int(np.rint((float(value) + 1.0) * 0.5 * 65535.0))
        for value in packet_surface_norm
    )
    receiver_index = tuple(
        PHYSICAL_SURFACE_ORDER.index(surface_name)
        for surface_name in RECEIVER_CHANNEL_SURFACE_ORDER
    )
    receiver_codes = tuple(packet_codes_by_surface[index] for index in receiver_index)

    packet = bytearray(ARDUINO_PACKET_LENGTH)
    packet[0] = ARDUINO_PACKET_HEADER
    packet[1] = ARDUINO_SURFACE_COUNT
    packet[2] = int(active_surface_mask) & 0xFF
    packet[3:7] = int(max(0, sequence)).to_bytes(4, byteorder="little", signed=False)
    write_index = 7
    for code in receiver_codes:
        packet[write_index : write_index + 2] = int(code).to_bytes(
            2,
            byteorder="little",
            signed=False,
        )
        write_index += 2

    return HardwareCommandPacket(
        aggregate_command_norm=tuple(float(value) for value in aggregate),
        physical_surface_norm=tuple(float(value) for value in physical_norm),
        packet_surface_norm=tuple(float(value) for value in packet_surface_norm),
        packet_codes_by_surface=tuple(int(value) for value in packet_codes_by_surface),
        receiver_channel_codes=tuple(int(value) for value in receiver_codes),
        packet_bytes=bytes(packet),
    )


# =============================================================================
# 2) Vicon Rigid-Body State Adapter
# =============================================================================
@dataclass(frozen=True)
class NausicaaViconSample:
    timestamp_s: float
    position_m: tuple[float, float, float]
    euler_rad: tuple[float, float, float] | None = None
    quaternion_xyzw: tuple[float, float, float, float] | None = None
    vicon_latency_s: float = 0.0
    frame_number: int | None = None
    frame_rate_hz: float | None = None


@dataclass(frozen=True)
class ViconArenaFrameTransform:
    """Map Vicon rigid-body pose into the controller world frame."""

    position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_alignment_rad: float = 0.0
    attitude_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    attitude_offset_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def position_to_world(self, raw_position_m: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_position_m, dtype=float).reshape(3)
        offset = np.asarray(self.position_offset_m, dtype=float).reshape(3)
        return _yaw_rotation_matrix(float(self.yaw_alignment_rad)) @ raw + offset

    def euler_to_world(self, raw_euler_rad: np.ndarray) -> np.ndarray:
        euler = np.asarray(raw_euler_rad, dtype=float).reshape(3).copy()
        signs = np.asarray(self.attitude_signs, dtype=float).reshape(3)
        if not np.all(np.isin(signs, [-1.0, 1.0])):
            raise ValueError("attitude_signs must contain only +1 or -1.")
        offset = np.asarray(self.attitude_offset_rad, dtype=float).reshape(3)
        euler = signs * euler
        euler[2] = float(euler[2]) + float(self.yaw_alignment_rad)
        euler = euler + offset
        euler = np.asarray([_wrap_to_pi_scalar(value) for value in euler], dtype=float)
        return euler


class NausicaaViconStateAdapter:
    """Pack one rigid-body Vicon stream into the canonical flight state."""

    def __init__(
        self,
        *,
        derivative_cutoff_hz: float = 8.0,
        body_rate_limit_rad_s: float = 6.0,
        body_rate_observer_window_frames: int = 7,
        actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
        arena_transform: ViconArenaFrameTransform | None = None,
    ) -> None:
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.body_rate_limit_rad_s = float(body_rate_limit_rad_s)
        if not np.isfinite(self.body_rate_limit_rad_s) or self.body_rate_limit_rad_s <= 0.0:
            raise ValueError("body_rate_limit_rad_s must be finite and positive.")
        self.actuator_tau_s = np.asarray(actuator_tau_s, dtype=float).reshape(3)
        if not np.all(np.isfinite(self.actuator_tau_s)) or np.any(self.actuator_tau_s <= 0.0):
            raise ValueError("actuator_tau_s must contain finite positive values.")
        self.arena_transform = arena_transform or ViconArenaFrameTransform()
        self._body_rate_observer = SO3AngularRateObserver(
            window_frames=body_rate_observer_window_frames,
            cutoff_hz=self.derivative_cutoff_hz,
            rate_limit_rad_s=self.body_rate_limit_rad_s,
        )
        self._previous_timestamp_s: float | None = None
        self._previous_frame_number: int | None = None
        self._previous_position_m: np.ndarray | None = None
        self._previous_euler_rad: np.ndarray | None = None
        self._filtered_world_velocity_m_s = np.zeros(3)
        self._surface_state_rad = np.zeros(3)
        self._last_estimator_status: dict[str, object] = {
            "dt_s": 0.0,
            "dt_source": "not_initialised",
            "frame_delta": 0,
            "frame_rate_hz": 0.0,
            "body_rate_limited": False,
            "rate_confidence": 0.0,
            "rate_window_frames": 0,
            "spike_rejected": False,
            "observer_mode": "not_initialised",
            "attitude_transform_applied": True,
        }

    def reset_angular_rate_filter(self) -> None:
        """Reset the SO(3) angular-rate observer."""

        self._body_rate_observer.reset()

    def estimator_status(self) -> dict[str, object]:
        return dict(self._last_estimator_status)

    def update(
        self,
        sample: NausicaaViconSample,
        *,
        command_norm: np.ndarray | None = None,
    ) -> np.ndarray:
        timestamp_s = float(sample.timestamp_s)
        raw_position_m = np.asarray(sample.position_m, dtype=float).reshape(3)
        raw_euler_rad = _resolve_euler_rad(sample)
        position_m = self.arena_transform.position_to_world(raw_position_m)
        euler_rad = self.arena_transform.euler_to_world(raw_euler_rad)
        if not np.isfinite(timestamp_s) or not np.all(np.isfinite(position_m)) or not np.all(np.isfinite(euler_rad)):
            raise ValueError("Vicon sample must contain finite timestamp, position, and attitude.")
        c_wb = _c_wb_numpy(*euler_rad)

        dt_s, dt_source, frame_delta, frame_rate_hz = self._resolve_sample_dt_s(
            sample,
            timestamp_s,
        )
        if dt_s > 0.0 and self._previous_position_m is not None:
            raw_world_velocity = (position_m - self._previous_position_m) / dt_s
            self._filtered_world_velocity_m_s = _low_pass_update(
                self._filtered_world_velocity_m_s,
                raw_world_velocity,
                dt_s,
                self.derivative_cutoff_hz,
            )
            if command_norm is not None:
                target_surface_rad = normalised_command_to_surface_rad(
                    quantise_normalised_command_vector(command_norm)
                )
                alpha = 1.0 - np.exp(-dt_s / self.actuator_tau_s)
                self._surface_state_rad = self._surface_state_rad + alpha * (
                    target_surface_rad - self._surface_state_rad
                )
            body_rates, observer_status = self._body_rate_observer.update(c_wb, dt_s=dt_s)
        elif command_norm is not None:
            body_rates, observer_status = self._body_rate_observer.update(c_wb, dt_s=0.0)
            self._surface_state_rad = normalised_command_to_surface_rad(
                quantise_normalised_command_vector(command_norm)
            )
        else:
            body_rates, observer_status = self._body_rate_observer.update(c_wb, dt_s=0.0)

        body_velocity = _body_velocity_from_world_up(
            self._filtered_world_velocity_m_s,
            euler_rad,
        )
        state = np.zeros(STATE_SIZE, dtype=float)
        state[STATE_INDEX["x_w"] : STATE_INDEX["z_w"] + 1] = position_m
        state[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1] = euler_rad
        state[STATE_INDEX["u"] : STATE_INDEX["w"] + 1] = body_velocity
        state[STATE_INDEX["p"] : STATE_INDEX["r"] + 1] = body_rates
        state[STATE_INDEX["delta_a"] : STATE_INDEX["delta_r"] + 1] = self._surface_state_rad

        if dt_s > 0.0 or self._previous_position_m is None:
            self._previous_timestamp_s = timestamp_s
            self._previous_frame_number = _sample_frame_number(sample)
            self._previous_position_m = position_m
            self._previous_euler_rad = euler_rad
        self._last_estimator_status = {
            "dt_s": float(dt_s),
            "dt_source": str(dt_source),
            "frame_delta": int(frame_delta),
            "frame_rate_hz": float(frame_rate_hz),
            **observer_status,
            "attitude_transform_applied": True,
        }
        return as_state_vector(state)

    def _resolve_sample_dt_s(
        self,
        sample: NausicaaViconSample,
        timestamp_s: float,
    ) -> tuple[float, str, int, float]:
        current_frame = _sample_frame_number(sample)
        frame_rate_hz = _sample_frame_rate_hz(sample)
        if (
            current_frame is not None
            and self._previous_frame_number is not None
            and frame_rate_hz > 0.0
        ):
            frame_delta = int(current_frame) - int(self._previous_frame_number)
            if frame_delta > 0:
                return float(frame_delta) / float(frame_rate_hz), "vicon_frame_time", frame_delta, frame_rate_hz
            return 0.0, "duplicate_or_reordered_vicon_frame", frame_delta, frame_rate_hz
        if self._previous_timestamp_s is not None:
            dt_s = float(timestamp_s) - float(self._previous_timestamp_s)
            if dt_s > 0.0 and np.isfinite(dt_s):
                return dt_s, "host_time_fallback", 0, frame_rate_hz
        return 0.0, "not_initialised", 0, frame_rate_hz


class SO3AngularRateObserver:
    """Confidence-aware body-rate observer over corrected SO(3) rotations."""

    def __init__(
        self,
        *,
        window_frames: int = 7,
        cutoff_hz: float = 8.0,
        rate_limit_rad_s: float = 6.0,
    ) -> None:
        self.window_frames = max(3, int(window_frames))
        self.cutoff_hz = float(cutoff_hz)
        self.rate_limit_rad_s = float(rate_limit_rad_s)
        if not np.isfinite(self.rate_limit_rad_s) or self.rate_limit_rad_s <= 0.0:
            raise ValueError("rate_limit_rad_s must be finite and positive.")
        self._previous_c_wb: np.ndarray | None = None
        self._raw_rate_window: deque[np.ndarray] = deque(maxlen=self.window_frames)
        self._filtered_rate_rad_s = np.zeros(3)
        self._pose_frame_count = 0
        self._last_status = self._status("not_initialised")

    def reset(self) -> None:
        self._previous_c_wb = None
        self._raw_rate_window.clear()
        self._filtered_rate_rad_s = np.zeros(3)
        self._pose_frame_count = 0
        self._last_status = self._status("reset")

    def update(self, c_wb: np.ndarray, *, dt_s: float) -> tuple[np.ndarray, dict[str, object]]:
        current = np.asarray(c_wb, dtype=float).reshape(3, 3)
        if self._previous_c_wb is None:
            self._previous_c_wb = current
            self._pose_frame_count = 1
            self._last_status = self._status("warming_first_frame")
            return self._filtered_rate_rad_s.copy(), dict(self._last_status)
        if not np.isfinite(dt_s) or float(dt_s) <= 0.0:
            self._last_status = self._status("duplicate_or_zero_dt")
            return self._filtered_rate_rad_s.copy(), dict(self._last_status)

        raw_rate = _body_rates_from_rotation_delta(self._previous_c_wb, current, float(dt_s))
        self._previous_c_wb = current
        self._pose_frame_count += 1
        raw_rate = np.asarray(raw_rate, dtype=float).reshape(3)
        if not np.all(np.isfinite(raw_rate)):
            self._last_status = self._status("nonfinite_rate_rejected")
            return self._filtered_rate_rad_s.copy(), dict(self._last_status)

        clipped_raw_rate = np.clip(raw_rate, -self.rate_limit_rad_s, self.rate_limit_rad_s)
        body_rate_limited = bool(np.any(np.abs(clipped_raw_rate - raw_rate) > 1e-9))
        self._raw_rate_window.append(clipped_raw_rate)
        robust_rate, spike_rejected, residual_norm = self._robust_window_rate(clipped_raw_rate)
        if spike_rejected:
            candidate_rate = robust_rate
        elif len(self._raw_rate_window) < 5:
            candidate_rate = clipped_raw_rate
        else:
            candidate_rate = 0.75 * clipped_raw_rate + 0.25 * self._filtered_rate_rad_s

        self._filtered_rate_rad_s = _low_pass_update(
            self._filtered_rate_rad_s,
            candidate_rate,
            float(dt_s),
            self.cutoff_hz,
        )
        self._filtered_rate_rad_s = np.clip(
            self._filtered_rate_rad_s,
            -self.rate_limit_rad_s,
            self.rate_limit_rad_s,
        )
        mode = "so3_window_spike_downweighted" if spike_rejected else "so3_window"
        self._last_status = self._status(
            mode,
            body_rate_limited=body_rate_limited,
            spike_rejected=spike_rejected,
            raw_body_rate_norm=float(np.linalg.norm(raw_rate)),
            residual_norm_rad_s=float(residual_norm),
        )
        return self._filtered_rate_rad_s.copy(), dict(self._last_status)

    def _robust_window_rate(self, current_rate: np.ndarray) -> tuple[np.ndarray, bool, float]:
        rates = np.asarray(tuple(self._raw_rate_window), dtype=float)
        if rates.shape[0] < 5:
            return np.asarray(current_rate, dtype=float), False, 0.0
        robust_center = np.median(rates, axis=0)
        deviations = np.linalg.norm(rates - robust_center.reshape(1, 3), axis=1)
        robust_scale = float(np.median(deviations)) + 1e-6
        residual_norm = float(np.linalg.norm(np.asarray(current_rate, dtype=float) - robust_center))
        spike_rejected = bool(residual_norm > max(1.5, 6.0 * robust_scale))
        return robust_center, spike_rejected, residual_norm

    def _status(
        self,
        mode: str,
        *,
        body_rate_limited: bool = False,
        spike_rejected: bool = False,
        raw_body_rate_norm: float = 0.0,
        residual_norm_rad_s: float = 0.0,
    ) -> dict[str, object]:
        return {
            "body_rate_limited": bool(body_rate_limited),
            "rate_confidence": float(self._confidence(spike_rejected=spike_rejected)),
            "rate_window_frames": int(min(self._pose_frame_count, self.window_frames)),
            "spike_rejected": bool(spike_rejected),
            "observer_mode": str(mode),
            "raw_body_rate_norm_rad_s": float(raw_body_rate_norm),
            "body_rate_residual_norm_rad_s": float(residual_norm_rad_s),
        }

    def _confidence(self, *, spike_rejected: bool) -> float:
        frame_count = int(self._pose_frame_count)
        if frame_count <= 1:
            confidence = 0.0
        elif frame_count == 2:
            confidence = 0.25
        elif frame_count == 3:
            confidence = 0.45
        elif frame_count == 4:
            confidence = 0.60
        elif frame_count == 5:
            confidence = 0.80
        elif frame_count < self.window_frames:
            confidence = 0.90
        else:
            confidence = 0.95
        if spike_rejected:
            confidence = min(confidence, 0.50)
        return float(confidence)


# =============================================================================
# 3) Math Helpers
# =============================================================================
def _resolve_euler_rad(sample: NausicaaViconSample) -> np.ndarray:
    if sample.euler_rad is not None:
        return np.asarray(sample.euler_rad, dtype=float).reshape(3)
    if sample.quaternion_xyzw is not None:
        return _quaternion_xyzw_to_euler_rad(sample.quaternion_xyzw)
    raise ValueError("Vicon sample requires euler_rad or quaternion_xyzw.")


def _sample_frame_number(sample: NausicaaViconSample) -> int | None:
    if sample.frame_number is None:
        return None
    try:
        frame_number = int(sample.frame_number)
    except Exception:
        return None
    return frame_number if frame_number >= 0 else None


def _sample_frame_rate_hz(sample: NausicaaViconSample) -> float:
    if sample.frame_rate_hz is None:
        return 0.0
    try:
        frame_rate_hz = float(sample.frame_rate_hz)
    except Exception:
        return 0.0
    if not np.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        return 0.0
    return frame_rate_hz


def _quaternion_xyzw_to_euler_rad(quaternion_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion_xyzw must be finite and nonzero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=float)


def _low_pass_update(previous: np.ndarray, raw: np.ndarray, dt_s: float, cutoff_hz: float) -> np.ndarray:
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        return np.asarray(raw, dtype=float).copy()
    tau_s = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = float(np.clip(1.0 - np.exp(-float(dt_s) / tau_s), 0.0, 1.0))
    return np.asarray(previous, dtype=float) + alpha * (
        np.asarray(raw, dtype=float) - np.asarray(previous, dtype=float)
    )


def _yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    c_yaw, s_yaw = np.cos(float(yaw_rad)), np.sin(float(yaw_rad))
    return np.asarray(
        [
            [c_yaw, -s_yaw, 0.0],
            [s_yaw, c_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _c_wb_numpy(phi: float, theta: float, psi: float) -> np.ndarray:
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    return np.asarray(
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


def _body_velocity_from_world_up(world_velocity_m_s: np.ndarray, euler_rad: np.ndarray) -> np.ndarray:
    velocity_internal = np.asarray(world_velocity_m_s, dtype=float).reshape(3).copy()
    velocity_internal[2] *= -1.0
    c_wb = _c_wb_numpy(*np.asarray(euler_rad, dtype=float).reshape(3))
    return c_wb.T @ velocity_internal


def _body_rates_from_rotation_delta(previous_c_wb: np.ndarray, current_c_wb: np.ndarray, dt_s: float) -> np.ndarray:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return np.zeros(3)
    previous = np.asarray(previous_c_wb, dtype=float).reshape(3, 3)
    current = np.asarray(current_c_wb, dtype=float).reshape(3, 3)
    relative_body_rotation = previous.T @ current
    trace_term = float(np.clip((np.trace(relative_body_rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(trace_term))
    vee = np.asarray(
        [
            relative_body_rotation[2, 1] - relative_body_rotation[1, 2],
            relative_body_rotation[0, 2] - relative_body_rotation[2, 0],
            relative_body_rotation[1, 0] - relative_body_rotation[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-6:
        rotation_vector = 0.5 * vee
    else:
        rotation_vector = (angle / (2.0 * np.sin(angle))) * vee
    return rotation_vector / float(dt_s)


def _wrap_to_pi_scalar(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (np.asarray(angle_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
