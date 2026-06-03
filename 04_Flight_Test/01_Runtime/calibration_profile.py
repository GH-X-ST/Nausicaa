from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FlightCalibrationProfile:
    profile_id: str
    profile_version: str
    vicon_position_offset_m: tuple[float, float, float]
    vicon_yaw_alignment_deg: float
    vicon_attitude_signs: tuple[float, float, float]
    requested_vicon_tracking_rate_hz: float
    derivative_cutoff_hz: float
    body_rate_limit_rad_s: float
    body_rate_observer_window_frames: int
    launch_gate_required_consecutive_frames: int
    launch_gate_rate_confidence_min: float
    launch_gate_body_rate_limits_rad_s: tuple[float, float, float]
    rejected_launch_attempt_min_speed_m_s: float

    def to_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile_hash"] = self.profile_hash()
        return payload

    def profile_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


ACTIVE_CALIBRATION_PROFILE = FlightCalibrationProfile(
    profile_id="nausicaa_real_flight_vicon_calibration_20260602",
    profile_version="1.0",
    vicon_position_offset_m=(4.136158795250567, 2.4114272057075916, 0.03414746062731508),
    vicon_yaw_alignment_deg=0.0,
    vicon_attitude_signs=(1.0, -1.0, -1.0),
    requested_vicon_tracking_rate_hz=200.0,
    derivative_cutoff_hz=8.0,
    body_rate_limit_rad_s=6.0,
    body_rate_observer_window_frames=7,
    launch_gate_required_consecutive_frames=2,
    launch_gate_rate_confidence_min=0.65,
    launch_gate_body_rate_limits_rad_s=(1.2, 1.2, 1.8),
    rejected_launch_attempt_min_speed_m_s=2.0,
)


def calibration_profile_for_runtime_values(
    *,
    profile_id: str,
    profile_version: str = "manual",
    vicon_position_offset_m: tuple[float, float, float],
    vicon_yaw_alignment_deg: float,
    vicon_attitude_signs: tuple[float, float, float],
    requested_vicon_tracking_rate_hz: float,
    derivative_cutoff_hz: float = ACTIVE_CALIBRATION_PROFILE.derivative_cutoff_hz,
    body_rate_limit_rad_s: float = ACTIVE_CALIBRATION_PROFILE.body_rate_limit_rad_s,
    body_rate_observer_window_frames: int = ACTIVE_CALIBRATION_PROFILE.body_rate_observer_window_frames,
    launch_gate_required_consecutive_frames: int = ACTIVE_CALIBRATION_PROFILE.launch_gate_required_consecutive_frames,
    launch_gate_rate_confidence_min: float = ACTIVE_CALIBRATION_PROFILE.launch_gate_rate_confidence_min,
    launch_gate_body_rate_limits_rad_s: tuple[float, float, float] = ACTIVE_CALIBRATION_PROFILE.launch_gate_body_rate_limits_rad_s,
    rejected_launch_attempt_min_speed_m_s: float = ACTIVE_CALIBRATION_PROFILE.rejected_launch_attempt_min_speed_m_s,
) -> FlightCalibrationProfile:
    return FlightCalibrationProfile(
        profile_id=str(profile_id),
        profile_version=str(profile_version),
        vicon_position_offset_m=tuple(float(value) for value in vicon_position_offset_m),
        vicon_yaw_alignment_deg=float(vicon_yaw_alignment_deg),
        vicon_attitude_signs=tuple(float(value) for value in vicon_attitude_signs),
        requested_vicon_tracking_rate_hz=float(requested_vicon_tracking_rate_hz),
        derivative_cutoff_hz=float(derivative_cutoff_hz),
        body_rate_limit_rad_s=float(body_rate_limit_rad_s),
        body_rate_observer_window_frames=int(body_rate_observer_window_frames),
        launch_gate_required_consecutive_frames=int(launch_gate_required_consecutive_frames),
        launch_gate_rate_confidence_min=float(launch_gate_rate_confidence_min),
        launch_gate_body_rate_limits_rad_s=tuple(float(value) for value in launch_gate_body_rate_limits_rad_s),
        rejected_launch_attempt_min_speed_m_s=float(rejected_launch_attempt_min_speed_m_s),
    )
