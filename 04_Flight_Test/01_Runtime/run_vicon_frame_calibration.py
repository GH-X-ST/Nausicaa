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
#   "single_fan" checks Fan_1 against the single-fan simulation centre and writes nothing
#   "four_fan"   checks Fan_1..Fan_4 against the fixed four-fan simulation centres and writes nothing
CALIBRATION_MODE = "single_fan"
FAN_CHECK_MODES = ("single_fan", "four_fan")
CALIBRATION_MODES = ("position", "attitude", "pose", "diagnostic", *FAN_CHECK_MODES)
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
DEFAULT_FAN_VICON_SUBJECT_NAMES = ("Fan_1", "Fan_2", "Fan_3", "Fan_4")
DEFAULT_FAN_POSITION_TOLERANCE_M = 0.05
DEFAULT_FAN_CHECK_PRINT_INTERVAL_S = 10.0
DEFAULT_FAN_CHECK_TIMEOUT_S = 0.0
DEFAULT_FAN_POSITION_ERROR_AXIS = "xy"
FAN_POSITION_CHECK_AXIS = "xy"
DEFAULT_SIM_FAN_MARKER_HEIGHT_M = 0.75
# Keep these fixed-layout targets aligned with 03_Control/04_Scenarios/updraft_models.py.
SIM_SINGLE_FAN_TARGETS_M = {
    "Fan_1": (4.2, 2.4, DEFAULT_SIM_FAN_MARKER_HEIGHT_M),
}
SIM_FOUR_FAN_TARGETS_M = {
    "Fan_1": (3.0, 3.6, DEFAULT_SIM_FAN_MARKER_HEIGHT_M),
    "Fan_2": (5.4, 3.6, DEFAULT_SIM_FAN_MARKER_HEIGHT_M),
    "Fan_3": (3.0, 1.2, DEFAULT_SIM_FAN_MARKER_HEIGHT_M),
    "Fan_4": (5.4, 1.2, DEFAULT_SIM_FAN_MARKER_HEIGHT_M),
}
RUNTIME_REPLAY_FAN_TARGETS_M = {
    "Fan_1": (2.7, 1.6, 0.75),
    "Fan_2": (2.7, 2.8, 0.75),
    "Fan_3": (4.5, 1.6, 0.75),
    "Fan_4": (4.5, 2.8, 0.75),
}
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


def _format_vector(values: tuple[float, float, float] | np.ndarray, *, unit: str = "") -> str:
    array = np.asarray(values, dtype=float).reshape(3)
    suffix = f" {unit}" if unit else ""
    return f"({array[0]:+.6f}, {array[1]:+.6f}, {array[2]:+.6f}){suffix}"


def _update_flags_for_calibration_mode(calibration_mode: str) -> tuple[bool, bool, bool]:
    mode = str(calibration_mode)
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"calibration_mode must be one of {CALIBRATION_MODES}, got {mode!r}.")
    if mode in FAN_CHECK_MODES:
        return False, False, False
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


def _fan_targets_for_mode(calibration_mode: str) -> dict[str, tuple[float, float, float]]:
    mode = str(calibration_mode)
    if mode == "single_fan":
        return dict(SIM_SINGLE_FAN_TARGETS_M)
    if mode == "four_fan":
        return dict(SIM_FOUR_FAN_TARGETS_M)
    raise ValueError(f"fan targets are only defined for {FAN_CHECK_MODES}, got {mode!r}.")


def _fan_position_error_m(
    position_m: np.ndarray,
    target_m: np.ndarray,
) -> tuple[np.ndarray, float]:
    delta = np.asarray(position_m, dtype=float).reshape(3) - np.asarray(target_m, dtype=float).reshape(3)
    error = float(np.max(np.abs(delta[:2])))
    return delta, error


def run_vicon_fan_position_check(
    *,
    calibration_mode: str,
    current_position_offset_m: tuple[float, float, float],
    current_yaw_alignment_deg: float,
    current_attitude_signs: tuple[float, float, float],
    current_attitude_offset_rad: tuple[float, float, float],
    vicon_host: str,
    subject_name: str,
    run_label: str,
    tolerance_m: float = DEFAULT_FAN_POSITION_TOLERANCE_M,
    print_interval_s: float = DEFAULT_FAN_CHECK_PRINT_INTERVAL_S,
    timeout_s: float = DEFAULT_FAN_CHECK_TIMEOUT_S,
    error_axis: str = DEFAULT_FAN_POSITION_ERROR_AXIS,
) -> dict[str, object]:
    targets = _fan_targets_for_mode(calibration_mode)
    target_subjects = tuple(targets)
    tolerance = float(tolerance_m)
    interval = max(0.1, float(print_interval_s))
    timeout = max(0.0, float(timeout_s))
    requested_axis = str(error_axis)
    axis = FAN_POSITION_CHECK_AXIS
    logger = FlightLogger(RESULT_ROOT / "vicon_fan_position_check" / run_label)
    reader = LiveNausicaaViconRigidBody(host=vicon_host, subject_name=subject_name)
    transform = ViconArenaFrameTransform(
        position_offset_m=current_position_offset_m,
        yaw_alignment_rad=float(np.deg2rad(current_yaw_alignment_deg)),
        attitude_signs=current_attitude_signs,
        attitude_offset_rad=current_attitude_offset_rad,
    )

    manifest = {
        "status": "running",
        "calibration_mode": str(calibration_mode),
        "vicon_host": str(vicon_host),
        "tracked_fan_subject_names": target_subjects,
        "vicon_default_fan_subject_names": DEFAULT_FAN_VICON_SUBJECT_NAMES,
        "simulation_target_positions_m": targets,
        "runtime_replay_fan_positions_m": RUNTIME_REPLAY_FAN_TARGETS_M,
        "tolerance_m": tolerance,
        "error_axis": axis,
        "error_metric": "max_abs_xy_per_axis",
        "tolerance_applies_per_axis": True,
        "requested_error_axis": requested_axis,
        "z_display_only": True,
        "print_interval_s": interval,
        "timeout_s": timeout,
        "loaded_active_profile_id": ACTIVE_CALIBRATION_PROFILE.profile_id,
        "loaded_active_profile_hash": ACTIVE_CALIBRATION_PROFILE.profile_hash(),
        "current_position_offset_m": tuple(float(value) for value in current_position_offset_m),
        "current_yaw_alignment_deg": float(current_yaw_alignment_deg),
        "current_attitude_signs_phi_theta_psi": tuple(float(value) for value in current_attitude_signs),
        "current_attitude_offset_rad": tuple(float(value) for value in current_attitude_offset_rad),
        "notes": (
            "Fan check modes do not update calibration files. Live Vicon fan translations are transformed "
            "through the active arena transform before independent horizontal x/y comparison. Measured z "
            "is displayed and logged for operator awareness only; fan height is not part of the placement tolerance. "
            "The fixed simulation targets mirror "
            "03_Control/04_Scenarios/updraft_models.py; runtime replay fan positions are listed separately "
            "because they are hardware-free placeholders, not the target for physical placement."
        ),
    }
    logger.write_manifest("vicon_fan_position_check_manifest.json", manifest)

    print(f"[FAN_CHECK] calibration_mode={calibration_mode}")
    print(f"[FAN_CHECK] default Vicon fan subjects: {DEFAULT_FAN_VICON_SUBJECT_NAMES}")
    print(f"[FAN_CHECK] active fan subjects for this check: {target_subjects}")
    if requested_axis != axis:
        print(f"[FAN_CHECK] requested axis {requested_axis!r} ignored; fan placement checks x/y only")
    print(
        f"[FAN_CHECK] tolerance={tolerance:.3f} m per x/y axis; "
        f"print interval={interval:.1f}s"
    )
    print("[FAN_CHECK] z is display-only; adjust fan floor x/y position, not height")
    print("[FAN_CHECK] simulation targets:")
    for subject, target in targets.items():
        print(
            f"  {subject}: target_xy=({_as_float_text(target[0])}, {_as_float_text(target[1])}) m; "
            f"sim_z_reference={_as_float_text(target[2])} m"
        )
    print("[FAN_CHECK] runtime replay fan positions, for reference only:")
    for subject, target in RUNTIME_REPLAY_FAN_TARGETS_M.items():
        print(f"  {subject}: replay={_format_vector(target, unit='m')}")

    latest_rows: list[dict[str, object]] = []
    status = "not_started"
    started = time.perf_counter()
    check_index = 0
    try:
        reader.open()
        while True:
            check_index += 1
            elapsed_s = time.perf_counter() - started
            try:
                if reader.client is not None:
                    reader.client.GetFrame()
                fans = reader.read_fans(target_subjects)
            except Exception as exc:
                fans = ()
                latest_rows = [
                    {
                        "check_index": check_index,
                        "elapsed_s": elapsed_s,
                        "fan_subject": "",
                        "visible": False,
                        "reason": f"fan_tracker_failed:{type(exc).__name__}:{exc}",
                        "within_tolerance": False,
                        "all_within_tolerance": False,
                    }
                ]
            rows = []
            all_visible = True
            all_within_tolerance = True
            fans_by_subject = {fan.subject_name: fan for fan in fans}
            for subject in target_subjects:
                fan = fans_by_subject.get(subject)
                target = np.asarray(targets[subject], dtype=float).reshape(3)
                visible = bool(fan is not None and fan.visible and fan.position_m is not None)
                reason = "fan_not_returned" if fan is None else fan.reason
                raw = np.full(3, np.nan, dtype=float)
                world = np.full(3, np.nan, dtype=float)
                delta = np.full(3, np.nan, dtype=float)
                error = float("nan")
                within = False
                frame_number = -1 if fan is None else int(fan.frame_number)
                if visible and fan is not None and fan.position_m is not None:
                    raw = np.asarray(fan.position_m, dtype=float).reshape(3)
                    world = transform.position_to_world(raw)
                    delta, error = _fan_position_error_m(world, target)
                    within = bool(error <= tolerance)
                all_visible = all_visible and visible
                all_within_tolerance = all_within_tolerance and within
                row = {
                    "check_index": check_index,
                    "elapsed_s": elapsed_s,
                    "fan_subject": subject,
                    "visible": visible,
                    "reason": reason,
                    "frame_number": frame_number,
                    "target_x_w_m": float(target[0]),
                    "target_y_w_m": float(target[1]),
                    "target_z_w_m": float(target[2]),
                    "raw_x_m": "" if not np.isfinite(raw[0]) else float(raw[0]),
                    "raw_y_m": "" if not np.isfinite(raw[1]) else float(raw[1]),
                    "raw_z_m": "" if not np.isfinite(raw[2]) else float(raw[2]),
                    "x_w_m": "" if not np.isfinite(world[0]) else float(world[0]),
                    "y_w_m": "" if not np.isfinite(world[1]) else float(world[1]),
                    "z_w_m": "" if not np.isfinite(world[2]) else float(world[2]),
                    "dx_m": "" if not np.isfinite(delta[0]) else float(delta[0]),
                    "dy_m": "" if not np.isfinite(delta[1]) else float(delta[1]),
                    "dz_m": "" if not np.isfinite(delta[2]) else float(delta[2]),
                    "error_axis": axis,
                    "error_m": "" if not np.isfinite(error) else float(error),
                    "within_tolerance": within,
                    "all_visible": False,
                    "all_within_tolerance": False,
                }
                rows.append(row)
            for row in rows:
                row["all_visible"] = all_visible
                row["all_within_tolerance"] = all_within_tolerance
                logger.append_metric_row("vicon_fan_position_check.csv", row)
            latest_rows = rows or latest_rows
            status = "within_tolerance" if all_visible and all_within_tolerance else "waiting"
            print(
                f"[FAN_CHECK] check={check_index} elapsed={elapsed_s:.1f}s "
                f"status={status} visible={sum(1 for row in rows if row['visible'])}/{len(target_subjects)}"
            )
            for row in rows:
                if row["visible"]:
                    print(
                        f"  {row['fan_subject']}: xy="
                        f"({_as_float_text(row['x_w_m'])}, {_as_float_text(row['y_w_m'])}) m "
                        f"target_xy=({_as_float_text(row['target_x_w_m'])}, {_as_float_text(row['target_y_w_m'])}) m "
                        f"dxy=({_as_float_text(row['dx_m'])}, {_as_float_text(row['dy_m'])}) m "
                        f"max_abs_xy={_as_float_text(row['error_m'])} m "
                        f"z_display={_as_float_text(row['z_w_m'])} m "
                        f"sim_z_ref={_as_float_text(row['target_z_w_m'])} m "
                        f"dz_display={_as_float_text(row['dz_m'])} m "
                        f"{'OK' if row['within_tolerance'] else 'MOVE_XY'}"
                    )
                else:
                    print(f"  {row['fan_subject']}: not visible ({row['reason']})")
            if status == "within_tolerance":
                print("[FAN_CHECK] all requested fans are within tolerance")
                break
            if timeout > 0.0 and elapsed_s >= timeout:
                status = "timeout"
                print("[FAN_CHECK] timeout before all requested fans were within tolerance")
                break
            print(f"[FAN_CHECK] adjust fan placement; next update in {interval:.1f}s")
            time.sleep(interval)
    finally:
        reader.close()
        result = dict(manifest)
        result["status"] = status
        result["check_count"] = int(check_index)
        result["elapsed_s"] = float(time.perf_counter() - started)
        result["latest_rows"] = latest_rows
        logger.write_manifest("vicon_fan_position_check_manifest.json", result)
        logger.write_report(
            "vicon_fan_position_check_report.md",
            [
                "# Vicon Fan Position Check Report",
                f"- Status: `{result['status']}`",
                f"- Calibration mode: `{result['calibration_mode']}`",
                f"- Error axis: `{result['error_axis']}`",
                f"- Error metric: `{result['error_metric']}`",
                f"- Tolerance applies per axis: `{result['tolerance_applies_per_axis']}`",
                f"- Requested error axis: `{result['requested_error_axis']}`",
                f"- Z display only: `{result['z_display_only']}`",
                f"- Tolerance (m): `{result['tolerance_m']}`",
                f"- Simulation target positions (m): `{result['simulation_target_positions_m']}`",
                f"- Runtime replay fan positions for reference (m): `{result['runtime_replay_fan_positions_m']}`",
                f"- Latest rows: `{result['latest_rows']}`",
            ],
        )
        logger.close()
    return result


def _as_float_text(value: object) -> str:
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return "n/a"


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
    loaded_active_position_offset = np.asarray(ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m, dtype=float).reshape(3)
    current_position_offset = np.asarray(current_position_offset_m, dtype=float).reshape(3)
    loaded_vs_current_offset_delta = current_position_offset - loaded_active_position_offset

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
    print("[CALIBRATE] loaded active calibration at script start:")
    print(f"  profile id:              {ACTIVE_CALIBRATION_PROFILE.profile_id}")
    print(f"  profile hash:            {ACTIVE_CALIBRATION_PROFILE.profile_hash()}")
    print(f"  position file:           {POSITION_CALIBRATION_PATH}")
    print(f"  attitude file:           {ATTITUDE_CALIBRATION_PATH}")
    print(f"  last saved position off: {_format_vector(loaded_active_position_offset, unit='m')}")
    print(f"  current offset in use:   {_format_vector(current_position_offset, unit='m')}")
    print(f"  current - last saved:    {_format_vector(loaded_vs_current_offset_delta, unit='m')}")
    print(
        "  attitude off in use:     "
        f"{_format_vector(tuple(float(np.rad2deg(value)) for value in current_attitude_offset_rad), unit='deg')}"
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
    recommended_minus_loaded_active_offset = recommended_offset - loaded_active_position_offset

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
        "loaded_active_profile_id": ACTIVE_CALIBRATION_PROFILE.profile_id,
        "loaded_active_profile_version": ACTIVE_CALIBRATION_PROFILE.profile_version,
        "loaded_active_profile_hash": ACTIVE_CALIBRATION_PROFILE.profile_hash(),
        "loaded_active_position_calibration_path": POSITION_CALIBRATION_PATH.as_posix(),
        "loaded_active_attitude_calibration_path": ATTITUDE_CALIBRATION_PATH.as_posix(),
        "loaded_active_position_offset_m": tuple(float(value) for value in loaded_active_position_offset),
        "current_position_offset_m": tuple(float(value) for value in current_position_offset_m),
        "current_minus_loaded_active_position_offset_m": tuple(
            float(value) for value in loaded_vs_current_offset_delta
        ),
        "recommended_position_offset_m": tuple(float(value) for value in recommended_offset),
        "recommended_minus_loaded_active_position_offset_m": tuple(
            float(value) for value in recommended_minus_loaded_active_offset
        ),
        "current_yaw_alignment_deg": float(current_yaw_alignment_deg),
        "current_attitude_signs_phi_theta_psi": tuple(float(value) for value in current_attitude_signs),
        "loaded_active_attitude_offset_rad": tuple(
            float(value) for value in ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad
        ),
        "loaded_active_attitude_offset_deg": tuple(
            float(np.rad2deg(value)) for value in ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad
        ),
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
            f"- Loaded active profile id: `{result['loaded_active_profile_id']}`",
            f"- Loaded active profile hash: `{result['loaded_active_profile_hash']}`",
            f"- Loaded active position calibration path: `{result['loaded_active_position_calibration_path']}`",
            f"- Loaded active attitude calibration path: `{result['loaded_active_attitude_calibration_path']}`",
            f"- Loaded active position offset (m): `{result['loaded_active_position_offset_m']}`",
            f"- Current minus loaded active position offset (m): `{result['current_minus_loaded_active_position_offset_m']}`",
            f"- Recommended position offset (m): `{result['recommended_position_offset_m']}`",
            f"- Recommended minus loaded active position offset (m): `{result['recommended_minus_loaded_active_position_offset_m']}`",
            f"- Loaded active attitude offset (deg): `{result['loaded_active_attitude_offset_deg']}`",
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
    print("[CALIBRATE] position calibration comparison")
    print(f"  loaded profile id:        {result['loaded_active_profile_id']}")
    print(f"  loaded profile hash:      {result['loaded_active_profile_hash']}")
    print(f"  loaded position file:     {result['loaded_active_position_calibration_path']}")
    print(f"  last saved offset m:      {_format_vector(result['loaded_active_position_offset_m'], unit='m')}")
    print(f"  current offset in use m:  {_format_vector(result['current_position_offset_m'], unit='m')}")
    print(
        "  current - last saved m:   "
        f"{_format_vector(result['current_minus_loaded_active_position_offset_m'], unit='m')}"
    )
    print(f"  recommended offset m:     {_format_vector(result['recommended_position_offset_m'], unit='m')}")
    print(
        "  recommended - last saved: "
        f"{_format_vector(result['recommended_minus_loaded_active_position_offset_m'], unit='m')}"
    )
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
            "pose writes both; diagnostic writes neither; single_fan/four_fan check live fan placement only."
        ),
    )
    parser.add_argument("--fan-position-tolerance-m", type=float, default=DEFAULT_FAN_POSITION_TOLERANCE_M)
    parser.add_argument("--fan-check-interval-s", type=float, default=DEFAULT_FAN_CHECK_PRINT_INTERVAL_S)
    parser.add_argument("--fan-check-timeout-s", type=float, default=DEFAULT_FAN_CHECK_TIMEOUT_S)
    parser.add_argument(
        "--fan-position-error-axis",
        choices=("xy", "xyz"),
        default=DEFAULT_FAN_POSITION_ERROR_AXIS,
        help="Compatibility option only; fan placement tolerance is always x/y and z is display-only.",
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
    if args.calibration_mode in FAN_CHECK_MODES:
        run_vicon_fan_position_check(
            calibration_mode=str(args.calibration_mode),
            current_position_offset_m=tuple(args.current_offset_m),
            current_yaw_alignment_deg=float(args.yaw_deg),
            current_attitude_signs=tuple(float(value) for value in args.attitude_signs),
            current_attitude_offset_rad=tuple(float(np.deg2rad(value)) for value in args.current_attitude_offset_deg),
            vicon_host=args.vicon_host,
            subject_name=args.subject_name,
            run_label=run_label,
            tolerance_m=float(args.fan_position_tolerance_m),
            print_interval_s=float(args.fan_check_interval_s),
            timeout_s=float(args.fan_check_timeout_s),
            error_axis=str(args.fan_position_error_axis),
        )
        return
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
