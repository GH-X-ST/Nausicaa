from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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
KNOWN_ARENA_POINT_M = (1.2, 1.6, 0.2)
CURRENT_POSITION_OFFSET_M = DEFAULT_VICON_POSITION_OFFSET_M
CURRENT_YAW_ALIGNMENT_DEG = 0.0
VICON_HOST = "192.168.0.100:801"
VICON_SUBJECT_NAME = "Nausicaa"
CALIBRATION_SAMPLE_COUNT = 150
CALIBRATION_TIMEOUT_S = 15.0
# =============================================================================


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
    print("Paste this into run_experiment_sequence.py if the axis directions are correct:")
    print(f"VICON_POSITION_OFFSET_M = {result['recommended_position_offset_m']}")
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
    )


if __name__ == "__main__":
    main()
