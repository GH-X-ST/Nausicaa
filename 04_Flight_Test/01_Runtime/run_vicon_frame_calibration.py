from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration_profile import ACTIVE_CALIBRATION_PROFILE, calibration_profile_for_runtime_values
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
KNOWN_ARENA_POINT_M = (1.2, 1.6, 0.1)
CURRENT_POSITION_OFFSET_M = DEFAULT_VICON_POSITION_OFFSET_M
CURRENT_YAW_ALIGNMENT_DEG = 0.0
VICON_HOST = "192.168.0.100:801"
VICON_SUBJECT_NAME = "Nausicaa"
CALIBRATION_SAMPLE_COUNT = 150
CALIBRATION_TIMEOUT_S = 15.0
CALIBRATION_PROFILE_PATH = Path(__file__).with_name("calibration_profile.py")
# =============================================================================


def _format_float_tuple(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(repr(float(value)) for value in values) + ")"


def _update_active_calibration_profile_file(
    *,
    profile_path: Path,
    recommended_offset_m: tuple[float, float, float],
    profile_id: str,
    profile_version: str,
) -> dict[str, object]:
    profile = calibration_profile_for_runtime_values(
        profile_id=profile_id,
        profile_version=profile_version,
        vicon_position_offset_m=recommended_offset_m,
        vicon_yaw_alignment_deg=ACTIVE_CALIBRATION_PROFILE.vicon_yaw_alignment_deg,
        vicon_attitude_signs=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_signs,
        requested_vicon_tracking_rate_hz=ACTIVE_CALIBRATION_PROFILE.requested_vicon_tracking_rate_hz,
        derivative_cutoff_hz=ACTIVE_CALIBRATION_PROFILE.derivative_cutoff_hz,
        body_rate_limit_rad_s=ACTIVE_CALIBRATION_PROFILE.body_rate_limit_rad_s,
        body_rate_observer_window_frames=ACTIVE_CALIBRATION_PROFILE.body_rate_observer_window_frames,
        launch_gate_required_consecutive_frames=ACTIVE_CALIBRATION_PROFILE.launch_gate_required_consecutive_frames,
        launch_gate_rate_confidence_min=ACTIVE_CALIBRATION_PROFILE.launch_gate_rate_confidence_min,
        launch_gate_body_rate_limits_rad_s=ACTIVE_CALIBRATION_PROFILE.launch_gate_body_rate_limits_rad_s,
        rejected_launch_attempt_min_speed_m_s=ACTIVE_CALIBRATION_PROFILE.rejected_launch_attempt_min_speed_m_s,
    )
    text = profile_path.read_text(encoding="utf-8")
    replacements = (
        (r'profile_id="[^"]+"', f'profile_id="{profile.profile_id}"'),
        (r'profile_version="[^"]+"', f'profile_version="{profile.profile_version}"'),
        (
            r"vicon_position_offset_m=\([^)]*\),",
            f"vicon_position_offset_m={_format_float_tuple(profile.vicon_position_offset_m)},",
        ),
    )
    for pattern, replacement in replacements:
        text, count = re.subn(pattern, lambda _match: replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not update active calibration profile field matching: {pattern}")

    profile_path.write_text(text, encoding="utf-8")
    manifest = profile.to_manifest()
    manifest["profile_path"] = profile_path.as_posix()
    return manifest


def run_vicon_frame_calibration(
    *,
    known_arena_point_m: tuple[float, float, float],
    current_position_offset_m: tuple[float, float, float],
    current_yaw_alignment_deg: float,
    vicon_host: str,
    subject_name: str,
    sample_count: int,
    timeout_s: float,
    run_label: str,
    update_active_profile: bool = True,
    profile_id: str = "",
    profile_version: str = "1.0",
    profile_path: Path = CALIBRATION_PROFILE_PATH,
) -> dict[str, object]:
    logger = FlightLogger(RESULT_ROOT / "vicon_frame_calibration" / run_label)
    reader = LiveNausicaaViconRigidBody(host=vicon_host, subject_name=subject_name)
    transform = ViconArenaFrameTransform(
        position_offset_m=current_position_offset_m,
        yaw_alignment_rad=float(np.deg2rad(current_yaw_alignment_deg)),
    )
    known = np.asarray(known_arena_point_m, dtype=float).reshape(3)
    raw_samples: list[np.ndarray] = []
    transformed_samples: list[np.ndarray] = []
    invalid_reasons: dict[str, int] = {}

    print("[CALIBRATE] Hold the glider still at the known arena point.")
    print(
        f"[CALIBRATE] known x/y/z = "
        f"{known[0]:.3f}, {known[1]:.3f}, {known[2]:.3f} m"
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
            transformed = transform.position_to_world(raw)
            raw_samples.append(raw)
            transformed_samples.append(transformed)
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
                },
            )
            if elapsed_s + 1e-12 >= next_print_s:
                print(
                    f"[CALIBRATE] valid={len(raw_samples)}/{sample_count} "
                    f"raw=({raw[0]:.3f},{raw[1]:.3f},{raw[2]:.3f}) "
                    f"current=({transformed[0]:.3f},{transformed[1]:.3f},{transformed[2]:.3f})"
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

    yaw = float(np.deg2rad(current_yaw_alignment_deg))
    rotated_raw_mean = ViconArenaFrameTransform(
        position_offset_m=(0.0, 0.0, 0.0),
        yaw_alignment_rad=yaw,
    ).position_to_world(raw_mean)
    recommended_offset = known - rotated_raw_mean

    result = {
        "status": "ok",
        "valid_sample_count": len(raw_samples),
        "invalid_reasons": invalid_reasons,
        "known_arena_point_m": tuple(float(value) for value in known),
        "raw_vicon_mean_m": tuple(float(value) for value in raw_mean),
        "raw_vicon_std_m": tuple(float(value) for value in raw_std),
        "current_transformed_mean_m": tuple(float(value) for value in transformed_mean),
        "current_transform_error_m": tuple(float(value) for value in transformed_error),
        "current_position_offset_m": tuple(float(value) for value in current_position_offset_m),
        "recommended_position_offset_m": tuple(float(value) for value in recommended_offset),
        "current_yaw_alignment_deg": float(current_yaw_alignment_deg),
        "run_root": (RESULT_ROOT / "vicon_frame_calibration" / run_label).as_posix(),
    }
    if update_active_profile:
        resolved_profile_id = profile_id or f"nausicaa_real_flight_vicon_calibration_{run_label}"
        result["active_profile_update"] = _update_active_calibration_profile_file(
            profile_path=profile_path,
            recommended_offset_m=tuple(float(value) for value in recommended_offset),
            profile_id=resolved_profile_id,
            profile_version=profile_version,
        )
    logger.write_manifest("vicon_frame_calibration_manifest.json", result)
    logger.write_report(
        "vicon_frame_calibration_report.md",
        [
            "# Vicon Frame Calibration Report",
            f"- Known arena point (m): `{result['known_arena_point_m']}`",
            f"- Valid samples: `{result['valid_sample_count']}`",
            f"- Raw Vicon mean (m): `{result['raw_vicon_mean_m']}`",
            f"- Raw Vicon std (m): `{result['raw_vicon_std_m']}`",
            f"- Current transformed mean (m): `{result['current_transformed_mean_m']}`",
            f"- Current transform error (m): `{result['current_transform_error_m']}`",
            f"- Recommended position offset (m): `{result['recommended_position_offset_m']}`",
            f"- Current yaw alignment (deg): `{result['current_yaw_alignment_deg']}`",
        ],
    )
    logger.close()

    print("[CALIBRATE] done")
    print(f"  raw Vicon mean m:          {result['raw_vicon_mean_m']}")
    print(f"  current transformed mean: {result['current_transformed_mean_m']}")
    print(f"  current error m:          {result['current_transform_error_m']}")
    print(f"  recommended offset m:     {result['recommended_position_offset_m']}")
    print()
    if update_active_profile:
        update = result["active_profile_update"]
        print(f"[CALIBRATE] active profile updated: {update['profile_path']}")
        print(f"[CALIBRATE] profile id: {update['profile_id']}")
        print(f"[CALIBRATE] profile hash: {update['profile_hash']}")
        print("[CALIBRATE] start the next runtime script as a new process so it reloads the profile.")
    else:
        print("[CALIBRATE] active profile update skipped by --no-update-active-profile")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Vicon raw origin offset against one known arena point.")
    parser.add_argument("--known-point-m", nargs=3, type=float, default=KNOWN_ARENA_POINT_M)
    parser.add_argument("--current-offset-m", nargs=3, type=float, default=CURRENT_POSITION_OFFSET_M)
    parser.add_argument("--yaw-deg", type=float, default=CURRENT_YAW_ALIGNMENT_DEG)
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--subject-name", default=VICON_SUBJECT_NAME)
    parser.add_argument("--sample-count", type=int, default=CALIBRATION_SAMPLE_COUNT)
    parser.add_argument("--timeout-s", type=float, default=CALIBRATION_TIMEOUT_S)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--no-update-active-profile",
        action="store_true",
        help="Diagnostic only: do not write the measured offset into calibration_profile.py.",
    )
    parser.add_argument("--profile-id", default="")
    parser.add_argument("--profile-version", default="1.0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_vicon_frame_calibration(
        known_arena_point_m=tuple(args.known_point_m),
        current_position_offset_m=tuple(args.current_offset_m),
        current_yaw_alignment_deg=float(args.yaw_deg),
        vicon_host=args.vicon_host,
        subject_name=args.subject_name,
        sample_count=int(args.sample_count),
        timeout_s=float(args.timeout_s),
        run_label=run_label,
        update_active_profile=not bool(args.no_update_active_profile),
        profile_id=str(args.profile_id),
        profile_version=str(args.profile_version),
    )


if __name__ == "__main__":
    main()
