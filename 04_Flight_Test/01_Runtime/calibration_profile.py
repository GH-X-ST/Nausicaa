from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
CALIBRATION_DATA_DIR = RUNTIME_ROOT / "calibration_data"
ACTIVE_VICON_POSITION_CALIBRATION_PATH = CALIBRATION_DATA_DIR / "active_vicon_position_calibration.json"
ACTIVE_VICON_ATTITUDE_CALIBRATION_PATH = CALIBRATION_DATA_DIR / "active_vicon_attitude_calibration.json"

_DEFAULT_VICON_POSITION_CALIBRATION = {
    "profile_id": "nausicaa_vicon_position_20260603_232209",
    "profile_version": "1.0",
    "vicon_position_offset_m": (3.9085570805120877, 2.4963799346478033, 0.03709411857233755),
    "requested_vicon_tracking_rate_hz": 200.0,
}
_DEFAULT_VICON_ATTITUDE_CALIBRATION = {
    "profile_id": "nausicaa_vicon_attitude_initial",
    "profile_version": "1.0",
    "vicon_yaw_alignment_deg": 0.0,
    "vicon_attitude_signs": (1.0, -1.0, -1.0),
    "vicon_attitude_offset_rad": (0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class FlightCalibrationProfile:
    profile_id: str
    profile_version: str
    vicon_position_offset_m: tuple[float, float, float]
    vicon_yaw_alignment_deg: float
    vicon_attitude_signs: tuple[float, float, float]
    vicon_attitude_offset_rad: tuple[float, float, float]
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


def _read_json_payload(path: Path, fallback: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return dict(fallback)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {path}")
    merged = dict(fallback)
    merged.update(payload)
    return merged


def _tuple3(payload: dict[str, object], key: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    values = payload.get(key, fallback)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{key} must contain exactly three values.")
    return tuple(float(value) for value in values)


def _build_active_calibration_profile() -> FlightCalibrationProfile:
    position_payload = _read_json_payload(
        ACTIVE_VICON_POSITION_CALIBRATION_PATH,
        _DEFAULT_VICON_POSITION_CALIBRATION,
    )
    attitude_payload = _read_json_payload(
        ACTIVE_VICON_ATTITUDE_CALIBRATION_PATH,
        _DEFAULT_VICON_ATTITUDE_CALIBRATION,
    )
    position_id = str(position_payload.get("profile_id", _DEFAULT_VICON_POSITION_CALIBRATION["profile_id"]))
    attitude_id = str(attitude_payload.get("profile_id", _DEFAULT_VICON_ATTITUDE_CALIBRATION["profile_id"]))
    position_version = str(
        position_payload.get("profile_version", _DEFAULT_VICON_POSITION_CALIBRATION["profile_version"])
    )
    attitude_version = str(
        attitude_payload.get("profile_version", _DEFAULT_VICON_ATTITUDE_CALIBRATION["profile_version"])
    )
    return FlightCalibrationProfile(
        profile_id=f"{position_id}+{attitude_id}",
        profile_version=f"position:{position_version};attitude:{attitude_version}",
        vicon_position_offset_m=_tuple3(
            position_payload,
            "vicon_position_offset_m",
            _DEFAULT_VICON_POSITION_CALIBRATION["vicon_position_offset_m"],
        ),
        vicon_yaw_alignment_deg=float(
            attitude_payload.get(
                "vicon_yaw_alignment_deg",
                _DEFAULT_VICON_ATTITUDE_CALIBRATION["vicon_yaw_alignment_deg"],
            )
        ),
        vicon_attitude_signs=_tuple3(
            attitude_payload,
            "vicon_attitude_signs",
            _DEFAULT_VICON_ATTITUDE_CALIBRATION["vicon_attitude_signs"],
        ),
        vicon_attitude_offset_rad=_tuple3(
            attitude_payload,
            "vicon_attitude_offset_rad",
            _DEFAULT_VICON_ATTITUDE_CALIBRATION["vicon_attitude_offset_rad"],
        ),
        requested_vicon_tracking_rate_hz=float(
            position_payload.get(
                "requested_vicon_tracking_rate_hz",
                _DEFAULT_VICON_POSITION_CALIBRATION["requested_vicon_tracking_rate_hz"],
            )
        ),
        derivative_cutoff_hz=8.0,
        body_rate_limit_rad_s=6.0,
        body_rate_observer_window_frames=7,
        launch_gate_required_consecutive_frames=2,
        launch_gate_rate_confidence_min=0.65,
        launch_gate_body_rate_limits_rad_s=(1.2, 1.2, 1.8),
        rejected_launch_attempt_min_speed_m_s=2.0,
    )


ACTIVE_CALIBRATION_PROFILE = _build_active_calibration_profile()


def calibration_profile_for_runtime_values(
    *,
    profile_id: str,
    profile_version: str = "manual",
    vicon_position_offset_m: tuple[float, float, float],
    vicon_yaw_alignment_deg: float,
    vicon_attitude_signs: tuple[float, float, float],
    requested_vicon_tracking_rate_hz: float,
    vicon_attitude_offset_rad: tuple[float, float, float] = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad,
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
        vicon_attitude_offset_rad=tuple(float(value) for value in vicon_attitude_offset_rad),
        requested_vicon_tracking_rate_hz=float(requested_vicon_tracking_rate_hz),
        derivative_cutoff_hz=float(derivative_cutoff_hz),
        body_rate_limit_rad_s=float(body_rate_limit_rad_s),
        body_rate_observer_window_frames=int(body_rate_observer_window_frames),
        launch_gate_required_consecutive_frames=int(launch_gate_required_consecutive_frames),
        launch_gate_rate_confidence_min=float(launch_gate_rate_confidence_min),
        launch_gate_body_rate_limits_rad_s=tuple(float(value) for value in launch_gate_body_rate_limits_rad_s),
        rejected_launch_attempt_min_speed_m_s=float(rejected_launch_attempt_min_speed_m_s),
    )
