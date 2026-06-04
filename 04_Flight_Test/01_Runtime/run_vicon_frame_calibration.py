from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration_profile import (
    ACTIVE_CALIBRATION_PROFILE,
    ACTIVE_VICON_ATTITUDE_CALIBRATION_PATH,
    ACTIVE_VICON_POSITION_CALIBRATION_PATH,
    calibration_profile_for_runtime_values,
)
from flight_config import DEFAULT_VICON_POSITION_OFFSET_M, RESULT_ROOT
from flight_logger import FlightLogger
from vicon_rigid_body import LiveNausicaaViconRigidBody

CONTROLLER_ROOT = Path(__file__).resolve().parents[1] / "02_Controller"
if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import ViconArenaFrameTransform  # noqa: E402


# =============================================================================
# EDIT THIS TO THE PHYSICAL POINT WHERE YOU HOLD THE GLIDER
# =============================================================================
# Select one mode before running this script:
#   "position"   updates only x/y/z Vicon origin offset
#   "attitude"   updates only roll/pitch/yaw rigid-body alignment
#   "pose"       updates both position and attitude
#   "diagnostic" measures both and writes neither
CALIBRATION_MODE = "attitude"
CALIBRATION_MODES = ("position", "attitude", "pose", "diagnostic")
KNOWN_ARENA_POINT_M = (1.2, 1.6, 0.1)
KNOWN_ARENA_ATTITUDE_DEG = (0.0, 0.0, 0.0)
CURRENT_POSITION_OFFSET_M = DEFAULT_VICON_POSITION_OFFSET_M
CURRENT_YAW_ALIGNMENT_DEG = ACTIVE_CALIBRATION_PROFILE.vicon_yaw_alignment_deg
CURRENT_ATTITUDE_SIGNS = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_signs
CURRENT_ATTITUDE_OFFSET_RAD = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad
VICON_HOST = "192.168.0.100:801"
VICON_SUBJECT_NAME = "Nausicaa"
CALIBRATION_SAMPLE_COUNT = 150
CALIBRATION_TIMEOUT_S = 15.0
POSITION_CALIBRATION_PATH = ACTIVE_VICON_POSITION_CALIBRATION_PATH
ATTITUDE_CALIBRATION_PATH = ACTIVE_VICON_ATTITUDE_CALIBRATION_PATH
# =============================================================================


def _wrap_angle_vector(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    return (data + np.pi) % (2.0 * np.pi) - np.pi


def _mean_angle_vector(values: list[np.ndarray] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("angle samples must be an Nx3 array.")
    return np.arctan2(np.mean(np.sin(array), axis=0), np.mean(np.cos(array), axis=0))


def _angle_std_vector(values: list[np.ndarray] | np.ndarray, mean_rad: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    deltas = _wrap_angle_vector(array - np.asarray(mean_rad, dtype=float).reshape(3))
    return np.std(deltas, axis=0)


def _write_json_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_flags_for_calibration_mode(calibration_mode: str) -> tuple[bool, bool, bool]:
    mode = str(calibration_mode)
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"calibration_mode must be one of {CALIBRATION_MODES}, got {mode!r}.")
    update_active_profile = mode != "diagnostic"
    update_active_position = mode in {"position", "pose"}
    update_active_attitude = mode in {"attitude", "pose"}
    return update_active_profile, update_active_position, update_active_attitude


def _calibration_mode_from_update_flags(update_position: bool, update_attitude: bool) -> str:
    if update_position and update_attitude:
        return "pose"
    if update_position:
        return "position"
    if update_attitude:
        return "attitude"
    return "diagnostic"


def _update_active_vicon_calibration_files(
    *,
    position_calibration_path: Path,
    attitude_calibration_path: Path,
    recommended_offset_m: tuple[float, float, float],
    recommended_attitude_offset_rad: tuple[float, float, float],
    yaw_alignment_deg: float,
    attitude_signs: tuple[float, float, float],
    update_position: bool,
    update_attitude: bool,
    profile_id: str,
    profile_version: str,
) -> dict[str, object]:
    if not update_position and not update_attitude:
        raise ValueError("At least one of update_position or update_attitude must be true.")
    resolved_position_offset_m = (
        tuple(float(value) for value in recommended_offset_m)
        if update_position
        else ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m
    )
    resolved_attitude_offset_rad = (
        tuple(float(value) for value in recommended_attitude_offset_rad)
        if update_attitude
        else ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad
    )
    position_profile_id = f"{profile_id}_position"
    attitude_profile_id = f"{profile_id}_attitude"
    if update_position:
        _write_json_payload(
            position_calibration_path,
            {
                "calibration_type": "vicon_position",
                "profile_id": position_profile_id,
                "profile_version": str(profile_version),
                "requested_vicon_tracking_rate_hz": float(
                    ACTIVE_CALIBRATION_PROFILE.requested_vicon_tracking_rate_hz
                ),
                "vicon_position_offset_m": list(resolved_position_offset_m),
            },
        )
    if update_attitude:
        _write_json_payload(
            attitude_calibration_path,
            {
                "calibration_type": "vicon_attitude",
                "profile_id": attitude_profile_id,
                "profile_version": str(profile_version),
                "vicon_attitude_offset_rad": list(resolved_attitude_offset_rad),
                "vicon_attitude_signs": [float(value) for value in attitude_signs],
                "vicon_yaw_alignment_deg": float(yaw_alignment_deg),
            },
        )
    profile = calibration_profile_for_runtime_values(
        profile_id=profile_id,
        profile_version=profile_version,
        vicon_position_offset_m=resolved_position_offset_m,
        vicon_yaw_alignment_deg=float(yaw_alignment_deg) if update_attitude else ACTIVE_CALIBRATION_PROFILE.vicon_yaw_alignment_deg,
        vicon_attitude_signs=tuple(float(value) for value in attitude_signs)
        if update_attitude
        else ACTIVE_CALIBRATION_PROFILE.vicon_attitude_signs,
        vicon_attitude_offset_rad=resolved_attitude_offset_rad,
        requested_vicon_tracking_rate_hz=ACTIVE_CALIBRATION_PROFILE.requested_vicon_tracking_rate_hz,
        derivative_cutoff_hz=ACTIVE_CALIBRATION_PROFILE.derivative_cutoff_hz,
        body_rate_limit_rad_s=ACTIVE_CALIBRATION_PROFILE.body_rate_limit_rad_s,
        body_rate_observer_window_frames=ACTIVE_CALIBRATION_PROFILE.body_rate_observer_window_frames,
        launch_gate_required_consecutive_frames=ACTIVE_CALIBRATION_PROFILE.launch_gate_required_consecutive_frames,
        launch_gate_rate_confidence_min=ACTIVE_CALIBRATION_PROFILE.launch_gate_rate_confidence_min,
        launch_gate_body_rate_limits_rad_s=ACTIVE_CALIBRATION_PROFILE.launch_gate_body_rate_limits_rad_s,
        rejected_launch_attempt_min_speed_m_s=ACTIVE_CALIBRATION_PROFILE.rejected_launch_attempt_min_speed_m_s,
    )
    manifest = profile.to_manifest()
    manifest["position_calibration_path"] = position_calibration_path.as_posix()
    manifest["attitude_calibration_path"] = attitude_calibration_path.as_posix()
    manifest["updated_vicon_position_offset_m"] = bool(update_position)
    manifest["updated_vicon_attitude_offset_rad"] = bool(update_attitude)
    return manifest


def run_vicon_frame_calibration(
    *,
    known_arena_point_m: tuple[float, float, float],
    known_arena_attitude_deg: tuple[float, float, float],
    current_position_offset_m: tuple[float, float, float],
    current_yaw_alignment_deg: float,
    current_attitude_signs: tuple[float, float, float],
    current_attitude_offset_rad: tuple[float, float, float],
    vicon_host: str,
    subject_name: str,
    sample_count: int,
    timeout_s: float,
    run_label: str,
    update_active_profile: bool = True,
    update_active_position: bool = True,
    update_active_attitude: bool = False,
    profile_id: str = "",
    profile_version: str = "1.0",
    position_calibration_path: Path = POSITION_CALIBRATION_PATH,
    attitude_calibration_path: Path = ATTITUDE_CALIBRATION_PATH,
) -> dict[str, object]:
    update_active_position = bool(update_active_profile and update_active_position)
    update_active_attitude = bool(update_active_profile and update_active_attitude)
    logger = FlightLogger(RESULT_ROOT / "vicon_frame_calibration" / run_label)
    reader = LiveNausicaaViconRigidBody(host=vicon_host, subject_name=subject_name)
    transform = ViconArenaFrameTransform(
        position_offset_m=current_position_offset_m,
        yaw_alignment_rad=float(np.deg2rad(current_yaw_alignment_deg)),
        attitude_signs=current_attitude_signs,
        attitude_offset_rad=current_attitude_offset_rad,
    )
    known = np.asarray(known_arena_point_m, dtype=float).reshape(3)
    known_attitude = np.asarray([np.deg2rad(float(value)) for value in known_arena_attitude_deg], dtype=float)
    raw_samples: list[np.ndarray] = []
    transformed_samples: list[np.ndarray] = []
    raw_euler_samples: list[np.ndarray] = []
    transformed_euler_samples: list[np.ndarray] = []
    invalid_reasons: dict[str, int] = {}

    print("[CALIBRATE] Hold the glider still at the known arena point.")
    print(
        f"[CALIBRATE] known x/y/z = "
        f"{known[0]:.3f}, {known[1]:.3f}, {known[2]:.3f} m"
    )
    print(
        f"[CALIBRATE] known roll/pitch/yaw = "
        f"{known_arena_attitude_deg[0]:.2f}, {known_arena_attitude_deg[1]:.2f}, "
        f"{known_arena_attitude_deg[2]:.2f} deg"
    )
    print(f"[CALIBRATE] collecting {sample_count} valid Vicon samples, timeout={timeout_s:.1f}s")
    try:
        reader.open()
        started = time.perf_counter()
        next_print_s = 0.0
        while len(raw_samples) < int(sample_count) and (time.perf_counter() - started) <= float(timeout_s):
            sample, status = reader.read_latest()
            elapsed_s = time.perf_counter() - started
            if sample is None or not status.valid:
                invalid_reasons[status.reason] = invalid_reasons.get(status.reason, 0) + 1
                if elapsed_s + 1e-12 >= next_print_s:
                    print(f"[CALIBRATE] waiting for valid Vicon: {status.reason}")
                    next_print_s = elapsed_s + 1.0
                time.sleep(0.02)
                continue
            raw = np.asarray(sample.position_m, dtype=float).reshape(3)
            raw_euler = np.asarray(sample.euler_rad, dtype=float).reshape(3)
            transformed = transform.position_to_world(raw)
            transformed_euler = transform.euler_to_world(raw_euler)
            raw_samples.append(raw)
            transformed_samples.append(transformed)
            raw_euler_samples.append(raw_euler)
            transformed_euler_samples.append(transformed_euler)
            logger.append_metric_row(
                "vicon_frame_calibration_samples.csv",
                {
                    "sample_index": len(raw_samples),
                    "frame_number": status.frame_number,
                    "vicon_latency_s": status.vicon_latency_s,
                    "raw_x_m": float(raw[0]),
                    "raw_y_m": float(raw[1]),
                    "raw_z_m": float(raw[2]),
                    "current_x_w_m": float(transformed[0]),
                    "current_y_w_m": float(transformed[1]),
                    "current_z_w_m": float(transformed[2]),
                    "raw_phi_rad": float(raw_euler[0]),
                    "raw_theta_rad": float(raw_euler[1]),
                    "raw_psi_rad": float(raw_euler[2]),
                    "current_phi_rad": float(transformed_euler[0]),
                    "current_theta_rad": float(transformed_euler[1]),
                    "current_psi_rad": float(transformed_euler[2]),
                },
            )
            if elapsed_s + 1e-12 >= next_print_s:
                print(
                    f"[CALIBRATE] valid={len(raw_samples)}/{sample_count} "
                    f"raw=({raw[0]:.3f},{raw[1]:.3f},{raw[2]:.3f}) "
                    f"current=({transformed[0]:.3f},{transformed[1]:.3f},{transformed[2]:.3f}) "
                    f"att=({np.rad2deg(transformed_euler[0]):.1f},"
                    f"{np.rad2deg(transformed_euler[1]):.1f},"
                    f"{np.rad2deg(transformed_euler[2]):.1f})deg"
                )
                next_print_s = elapsed_s + 1.0
            time.sleep(0.02)
    finally:
        reader.close()

    if not raw_samples:
        logger.write_manifest(
            "vicon_frame_calibration_manifest.json",
            {
                "status": "failed_no_valid_samples",
                "invalid_reasons": invalid_reasons,
                "known_arena_point_m": tuple(float(value) for value in known),
            },
        )
        logger.close()
        raise RuntimeError(f"No valid Vicon samples collected. Reasons: {invalid_reasons}")

    raw_array = np.vstack(raw_samples)
    transformed_array = np.vstack(transformed_samples)
    raw_mean = raw_array.mean(axis=0)
    transformed_mean = transformed_array.mean(axis=0)
    raw_std = raw_array.std(axis=0)
    transformed_error = transformed_mean - known
    raw_euler_mean = _mean_angle_vector(raw_euler_samples)
    transformed_euler_mean = _mean_angle_vector(transformed_euler_samples)
    raw_euler_std = _angle_std_vector(raw_euler_samples, raw_euler_mean)
    transformed_attitude_error = _wrap_angle_vector(transformed_euler_mean - known_attitude)

    yaw = float(np.deg2rad(current_yaw_alignment_deg))
    rotated_raw_mean = ViconArenaFrameTransform(
        position_offset_m=(0.0, 0.0, 0.0),
        yaw_alignment_rad=yaw,
    ).position_to_world(raw_mean)
    recommended_offset = known - rotated_raw_mean
    recommended_attitude_offset = _wrap_angle_vector(
        np.asarray(current_attitude_offset_rad, dtype=float).reshape(3) - transformed_attitude_error
    )

    result = {
        "status": "ok",
        "calibration_mode": _calibration_mode_from_update_flags(update_active_position, update_active_attitude),
        "valid_sample_count": len(raw_samples),
        "invalid_reasons": invalid_reasons,
        "known_arena_point_m": tuple(float(value) for value in known),
        "known_arena_attitude_deg": tuple(float(value) for value in known_arena_attitude_deg),
        "raw_vicon_mean_m": tuple(float(value) for value in raw_mean),
        "raw_vicon_std_m": tuple(float(value) for value in raw_std),
        "raw_vicon_euler_mean_rad": tuple(float(value) for value in raw_euler_mean),
        "raw_vicon_euler_std_rad": tuple(float(value) for value in raw_euler_std),
        "current_transformed_mean_m": tuple(float(value) for value in transformed_mean),
        "current_transform_error_m": tuple(float(value) for value in transformed_error),
        "current_transformed_attitude_mean_rad": tuple(float(value) for value in transformed_euler_mean),
        "current_transformed_attitude_error_rad": tuple(float(value) for value in transformed_attitude_error),
        "current_position_offset_m": tuple(float(value) for value in current_position_offset_m),
        "recommended_position_offset_m": tuple(float(value) for value in recommended_offset),
        "current_yaw_alignment_deg": float(current_yaw_alignment_deg),
        "current_attitude_signs_phi_theta_psi": tuple(float(value) for value in current_attitude_signs),
        "current_attitude_offset_rad": tuple(float(value) for value in current_attitude_offset_rad),
        "recommended_attitude_offset_rad": tuple(float(value) for value in recommended_attitude_offset),
        "recommended_attitude_offset_deg": tuple(float(np.rad2deg(value)) for value in recommended_attitude_offset),
        "active_profile_position_update_requested": bool(update_active_position),
        "active_profile_attitude_update_requested": bool(update_active_attitude),
        "run_root": (RESULT_ROOT / "vicon_frame_calibration" / run_label).as_posix(),
    }
    if update_active_position or update_active_attitude:
        resolved_profile_id = profile_id or f"nausicaa_real_flight_vicon_calibration_{run_label}"
        result["active_profile_update"] = _update_active_vicon_calibration_files(
            position_calibration_path=position_calibration_path,
            attitude_calibration_path=attitude_calibration_path,
            recommended_offset_m=tuple(float(value) for value in recommended_offset),
            recommended_attitude_offset_rad=tuple(float(value) for value in recommended_attitude_offset),
            yaw_alignment_deg=float(current_yaw_alignment_deg),
            attitude_signs=tuple(float(value) for value in current_attitude_signs),
            update_position=update_active_position,
            update_attitude=update_active_attitude,
            profile_id=resolved_profile_id,
            profile_version=profile_version,
        )
    logger.write_manifest("vicon_frame_calibration_manifest.json", result)
    logger.write_report(
        "vicon_frame_calibration_report.md",
        [
            "# Vicon Frame Calibration Report",
            f"- Calibration mode: `{result['calibration_mode']}`",
            f"- Known arena point (m): `{result['known_arena_point_m']}`",
            f"- Valid samples: `{result['valid_sample_count']}`",
            f"- Raw Vicon mean (m): `{result['raw_vicon_mean_m']}`",
            f"- Raw Vicon std (m): `{result['raw_vicon_std_m']}`",
            f"- Raw Vicon Euler mean (rad): `{result['raw_vicon_euler_mean_rad']}`",
            f"- Raw Vicon Euler std (rad): `{result['raw_vicon_euler_std_rad']}`",
            f"- Current transformed mean (m): `{result['current_transformed_mean_m']}`",
            f"- Current transform error (m): `{result['current_transform_error_m']}`",
            f"- Current attitude mean (rad): `{result['current_transformed_attitude_mean_rad']}`",
            f"- Current attitude error (rad): `{result['current_transformed_attitude_error_rad']}`",
            f"- Recommended position offset (m): `{result['recommended_position_offset_m']}`",
            f"- Recommended attitude offset (rad): `{result['recommended_attitude_offset_rad']}`",
            f"- Recommended attitude offset (deg): `{result['recommended_attitude_offset_deg']}`",
            f"- Active profile position update requested: `{result['active_profile_position_update_requested']}`",
            f"- Active profile attitude update requested: `{result['active_profile_attitude_update_requested']}`",
            f"- Current yaw alignment (deg): `{result['current_yaw_alignment_deg']}`",
        ],
    )
    logger.close()

    print("[CALIBRATE] done")
    print(f"  calibration mode:          {result['calibration_mode']}")
    print(f"  raw Vicon mean m:          {result['raw_vicon_mean_m']}")
    print(f"  current transformed mean: {result['current_transformed_mean_m']}")
    print(f"  current error m:          {result['current_transform_error_m']}")
    print(f"  current attitude error:   {result['current_transformed_attitude_error_rad']} rad")
    print(f"  recommended offset m:     {result['recommended_position_offset_m']}")
    print(f"  recommended attitude off: {result['recommended_attitude_offset_deg']} deg")
    print()
    if update_active_position or update_active_attitude:
        update = result["active_profile_update"]
        print(f"[CALIBRATE] position file: {update['position_calibration_path']}")
        print(f"[CALIBRATE] attitude file: {update['attitude_calibration_path']}")
        print(f"[CALIBRATE] position offset updated: {update['updated_vicon_position_offset_m']}")
        print(f"[CALIBRATE] attitude offset updated: {update['updated_vicon_attitude_offset_rad']}")
        print(f"[CALIBRATE] profile id: {update['profile_id']}")
        print(f"[CALIBRATE] profile hash: {update['profile_hash']}")
        print("[CALIBRATE] start the next runtime script as a new process so it reloads the profile.")
    else:
        print("[CALIBRATE] active profile update skipped")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Vicon raw pose against one known arena pose.")
    parser.add_argument("--known-point-m", nargs=3, type=float, default=KNOWN_ARENA_POINT_M)
    parser.add_argument("--known-attitude-deg", nargs=3, type=float, default=KNOWN_ARENA_ATTITUDE_DEG)
    parser.add_argument("--current-offset-m", nargs=3, type=float, default=CURRENT_POSITION_OFFSET_M)
    parser.add_argument("--yaw-deg", type=float, default=CURRENT_YAW_ALIGNMENT_DEG)
    parser.add_argument("--attitude-signs", nargs=3, type=float, default=CURRENT_ATTITUDE_SIGNS)
    parser.add_argument(
        "--current-attitude-offset-deg",
        nargs=3,
        type=float,
        default=tuple(float(np.rad2deg(value)) for value in CURRENT_ATTITUDE_OFFSET_RAD),
    )
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--subject-name", default=VICON_SUBJECT_NAME)
    parser.add_argument("--sample-count", type=int, default=CALIBRATION_SAMPLE_COUNT)
    parser.add_argument("--timeout-s", type=float, default=CALIBRATION_TIMEOUT_S)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--calibration-mode",
        choices=CALIBRATION_MODES,
        default=CALIBRATION_MODE,
        help=(
            "position writes only x/y/z; attitude writes only roll/pitch/yaw alignment; "
            "pose writes both; diagnostic writes neither."
        ),
    )
    parser.add_argument("--position-calibration-path", type=Path, default=POSITION_CALIBRATION_PATH)
    parser.add_argument("--attitude-calibration-path", type=Path, default=ATTITUDE_CALIBRATION_PATH)
    parser.add_argument("--profile-id", default="")
    parser.add_argument("--profile-version", default="1.0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    update_active_profile, update_active_position, update_active_attitude = _update_flags_for_calibration_mode(
        args.calibration_mode
    )
    print(f"[CALIBRATE] calibration_mode={args.calibration_mode}")
    run_vicon_frame_calibration(
        known_arena_point_m=tuple(args.known_point_m),
        known_arena_attitude_deg=tuple(args.known_attitude_deg),
        current_position_offset_m=tuple(args.current_offset_m),
        current_yaw_alignment_deg=float(args.yaw_deg),
        current_attitude_signs=tuple(float(value) for value in args.attitude_signs),
        current_attitude_offset_rad=tuple(float(np.deg2rad(value)) for value in args.current_attitude_offset_deg),
        vicon_host=args.vicon_host,
        subject_name=args.subject_name,
        sample_count=int(args.sample_count),
        timeout_s=float(args.timeout_s),
        run_label=run_label,
        update_active_profile=update_active_profile,
        update_active_position=update_active_position,
        update_active_attitude=update_active_attitude,
        profile_id=str(args.profile_id),
        profile_version=str(args.profile_version),
        position_calibration_path=Path(args.position_calibration_path),
        attitude_calibration_path=Path(args.attitude_calibration_path),
    )


if __name__ == "__main__":
    main()
