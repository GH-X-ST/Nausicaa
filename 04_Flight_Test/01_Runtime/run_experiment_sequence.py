from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from calibration_profile import ACTIVE_CALIBRATION_PROFILE, calibration_profile_for_runtime_values
from experiment_cases import EXPERIMENT_CASES, experiment_case_manifest, get_experiment_case
from flight_config import (
    DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
    RESULT_ROOT,
    FlightRuntimeConfig,
)
from flight_logger import FlightLogger
from frozen_flight_controller import FrozenFlightController
from nano_serial import FakeNanoSerialTx, NanoSerialTx
from run_real_flight import run_real_flight


# =============================================================================
# CLICK-AND-GO SETTINGS FOR THE NEXT REAL EXPERIMENT BLOCK
# Current setup: E0.1 dry-air shakedown, 5 valid throws, hardware armed on COM11.
# Failed launch-gate attempts are ignored and do not count as throws.
# =============================================================================
CURRENT_EXPERIMENT_CASE = "E0.1"
CURRENT_SESSION_LABEL = ""  # Empty means a new timestamped result folder.
TARGET_VALID_THROWS_OVERRIDE: int | None = 2
PRE_ARM_VICON_INACTIVE_DELAY_S = 3.0
COOLDOWN_AFTER_VALID_THROW_S = 5.0
RETRY_AFTER_INVALID_START_S = 5.0
MAX_INVALID_ATTEMPTS: int | None = None
SERIAL_PORT = "COM11"
VICON_HOST = "192.168.0.100:801"
MODE = "armed"  # Hardware output enabled. Use "dry-run" for hardware-free tests.
MAX_ACTIVE_FLIGHT_DURATION_S = 20.0
LAUNCH_WAIT_TIMEOUT_S = 120.0
POST_EXIT_NEUTRAL_TAIL_S = 0.30
VICON_TRACKING_RATE_HZ = 200.0
VICON_POLL_PERIOD_S = 1.0 / VICON_TRACKING_RATE_HZ
VICON_POSITION_OFFSET_M = ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m
VICON_YAW_ALIGNMENT_DEG = ACTIVE_CALIBRATION_PROFILE.vicon_yaw_alignment_deg
VICON_ATTITUDE_SIGNS = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_signs
VICON_ATTITUDE_OFFSET_RAD = ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad
# =============================================================================


def run_experiment_sequence(
    *,
    case_id: str,
    session_label: str,
    mode: str,
    serial_port: str,
    vicon_host: str,
    target_valid_throws: int | None,
    cooldown_s: float,
    retry_cooldown_s: float,
    max_invalid_attempts: int | None,
    max_duration_s: float,
    launch_wait_timeout_s: float,
    post_exit_neutral_tail_s: float,
    vicon_poll_period_s: float,
    vicon_position_offset_m: tuple[float, float, float],
    vicon_yaw_alignment_deg: float,
    vicon_attitude_signs: tuple[float, float, float],
    vicon_attitude_offset_rad: tuple[float, float, float],
    pre_arm_delay_s: float = 0.0,
) -> dict[str, object]:
    case = get_experiment_case(case_id)
    target = int(target_valid_throws if target_valid_throws is not None else case.target_valid_throws)
    if target <= 0:
        raise ValueError("target valid throws must be positive.")
    calibration_profile = calibration_profile_for_runtime_values(
        profile_id=ACTIVE_CALIBRATION_PROFILE.profile_id,
        profile_version=ACTIVE_CALIBRATION_PROFILE.profile_version,
        vicon_position_offset_m=vicon_position_offset_m,
        vicon_yaw_alignment_deg=vicon_yaw_alignment_deg,
        vicon_attitude_signs=vicon_attitude_signs,
        vicon_attitude_offset_rad=vicon_attitude_offset_rad,
        requested_vicon_tracking_rate_hz=1.0 / float(vicon_poll_period_s),
    )
    calibration_source = (
        "active_calibration_profile"
        if calibration_profile.profile_hash() == ACTIVE_CALIBRATION_PROFILE.profile_hash()
        else "manual_runtime_vicon_transform"
    )
    session = session_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = RESULT_ROOT / case.case_id / session
    session_logger = FlightLogger(session_root)
    base_config = FlightRuntimeConfig(
        run_label="controller_session",
        library_tier=DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
        controller_mode=case.controller_mode,
        experiment_case_id=case.case_id,
        experiment_case_name=case.case_name,
        experiment_memory_enabled=case.memory_enabled,
        experiment_layout_id=case.layout_id,
        serial_port=serial_port,
        vicon_host=vicon_host,
        max_duration_s=max_duration_s,
        launch_wait_timeout_s=launch_wait_timeout_s,
        post_exit_neutral_tail_s=post_exit_neutral_tail_s,
        vicon_poll_period_s=vicon_poll_period_s,
        retry_cooldown_s=retry_cooldown_s,
        vicon_position_offset_m=vicon_position_offset_m,
        vicon_yaw_alignment_deg=vicon_yaw_alignment_deg,
        vicon_attitude_signs=vicon_attitude_signs,
        vicon_attitude_offset_rad=vicon_attitude_offset_rad,
        calibration_profile_id=calibration_profile.profile_id,
        calibration_profile_hash=calibration_profile.profile_hash(),
        vicon_calibration_source=calibration_source,
        output_root=session_root,
    )
    controller = FrozenFlightController(base_config)
    session_logger.write_manifest(
        "experiment_sequence_manifest.json",
        {
            "case": asdict(case),
            "session_label": session,
            "target_valid_throws": target,
            "mode": mode,
            "library_tier": DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
            "controller_mode": case.controller_mode,
            "case_registry": experiment_case_manifest(),
            "memory_policy": controller.memory_summary(),
            "pre_arm_vicon_inactive_delay_s": float(pre_arm_delay_s),
            "cooldown_after_valid_throw_s": float(cooldown_s),
            "retry_after_invalid_start_s": float(retry_cooldown_s),
            "vicon_tracking_rate_hz": float(1.0 / vicon_poll_period_s),
            "vicon_poll_period_s": float(vicon_poll_period_s),
            "calibration_profile": calibration_profile.to_manifest(),
            "vicon_attitude_signs_phi_theta_psi": tuple(float(value) for value in vicon_attitude_signs),
            "vicon_attitude_offset_rad_phi_theta_psi": tuple(float(value) for value in vicon_attitude_offset_rad),
            "launch_gate_body_rate_limits_rad_s": tuple(
                float(value) for value in base_config.launch_gate_body_rate_limits_rad_s
            ),
            "launch_gate_body_rate_limits_policy": "formal_evidence_default",
            "between_throw_policy": (
                "pre_arm_and_cooldown_stream_neutral_only_without_vicon_reads;"
                "next_throw_reopens_lazy_vicon_launch_gate_wait"
            ),
        },
    )
    valid_count = 0
    invalid_count = 0
    attempt_index = 0
    print(f"[START] case={case.case_id} {case.case_name}")
    print(
        f"[START] controller_mode={case.controller_mode} memory_enabled={case.memory_enabled} "
        f"layout={case.layout_id} target_valid_throws={target}"
    )
    print("[START] between throws: Vicon is inactive during cooldown; neutral is held until the next launch-gate wait.")
    try:
        while valid_count < target:
            if max_invalid_attempts is not None and invalid_count >= int(max_invalid_attempts):
                print(f"[STOP] max invalid attempts reached: {invalid_count}")
                break
            attempt_index += 1
            next_valid_throw_index = valid_count + 1
            print(
                f"[ARM] case={case.case_id} valid={valid_count}/{target} "
                f"next_throw={next_valid_throw_index:03d} invalid_attempts={invalid_count}"
            )
            print("[ARM] pre-arm delay: Vicon inactive, neutral held; prepare throw now.")
            _cooldown(
                pre_arm_delay_s,
                label="pre_arm_vicon_inactive_before_launch_gate",
                mode=mode,
                serial_port=serial_port,
                controller=controller,
                serial_period_s=base_config.serial_period_s,
                serial_baud=base_config.serial_baud,
            )
            print("[ARM] Vicon launch-gate tracking active now; throw through the start gate.")
            print("[ARM] NoFrame/missing-subject states are ignored safely until the gate passes.")
            output_root = session_root
            if invalid_count >= 0:
                pass
            preferred_run_root = session_root / f"throw_{next_valid_throw_index:03d}"
            run_root = _available_run_root(preferred_run_root)
            if run_root != preferred_run_root:
                print(
                    f"[WARN] preferred run folder is still present; "
                    f"recording this attempt in {run_root.name}"
                )
            run_label = run_root.name
            config = FlightRuntimeConfig(
                run_label=run_label,
                library_tier=DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
                controller_mode=case.controller_mode,
                experiment_case_id=case.case_id,
                experiment_case_name=case.case_name,
                experiment_memory_enabled=case.memory_enabled,
                experiment_layout_id=case.layout_id,
                throw_index=next_valid_throw_index,
                attempt_index=attempt_index,
                serial_port=serial_port,
                vicon_host=vicon_host,
                max_duration_s=max_duration_s,
                launch_wait_timeout_s=launch_wait_timeout_s,
                post_exit_neutral_tail_s=post_exit_neutral_tail_s,
                vicon_poll_period_s=vicon_poll_period_s,
                retry_cooldown_s=retry_cooldown_s,
                vicon_position_offset_m=vicon_position_offset_m,
                vicon_yaw_alignment_deg=vicon_yaw_alignment_deg,
                vicon_attitude_signs=vicon_attitude_signs,
                vicon_attitude_offset_rad=vicon_attitude_offset_rad,
                calibration_profile_id=calibration_profile.profile_id,
                calibration_profile_hash=calibration_profile.profile_hash(),
                vicon_calibration_source=calibration_source,
                launch_gate_body_rate_limits_rad_s=base_config.launch_gate_body_rate_limits_rad_s,
                output_root=output_root,
            )
            summary = run_real_flight(
                config,
                mode=mode,
                controller=controller,
                expected_visible_fan_range=(case.expected_visible_fan_min, case.expected_visible_fan_max),
            )
            if bool(summary.get("valid_throw", False)):
                valid_count += 1
                session_logger.append_metric_row("experiment_sequence_summary.csv", _session_row(case, summary, valid_count, invalid_count))
                print(
                    f"[DONE] case={case.case_id} throw={valid_count:03d}/{target} "
                    f"speed={float(summary.get('launch_speed_m_s', 0.0)):.2f}m/s "
                    f"term={summary.get('termination_reason', '') or 'duration'} "
                    f"decisions={summary.get('controller_decision_count', 0)} "
                    f"max_dt={float(summary.get('max_decision_time_s', 0.0)):.4f}s "
                    f"memory_cells={summary.get('memory_cell_count', 0)}"
                )
                if valid_count < target:
                    print("[RECOVERY] completed throw; Vicon is now inactive during cooldown, neutral command is held.")
                    _cooldown(
                        cooldown_s,
                        label="cooldown_before_rearm",
                        mode=mode,
                        serial_port=serial_port,
                        controller=controller,
                        serial_period_s=base_config.serial_period_s,
                        serial_baud=base_config.serial_baud,
                    )
                    print("[REARM] cooldown complete; next throw will use the same lazy Vicon launch-gate wait.")
            else:
                invalid_count += 1
                invalid_root = session_root / "invalid_attempts"
                _move_invalid_attempt(run_root, invalid_root / f"attempt_{invalid_count:03d}")
                session_logger.append_metric_row("experiment_sequence_summary.csv", _session_row(case, summary, valid_count, invalid_count))
                print(
                    f"[INVALID] case={case.case_id} attempt={invalid_count:03d} "
                    f"reason={summary.get('cancellation_reason', '')} valid={valid_count}/{target}"
                )
                print("[RETRY] invalid start ignored; Vicon remains inactive during retry cooldown.")
                _cooldown(
                    retry_cooldown_s,
                    label="retry_after_invalid_start",
                    mode=mode,
                    serial_port=serial_port,
                    controller=controller,
                    serial_period_s=base_config.serial_period_s,
                    serial_baud=base_config.serial_baud,
                )
                print("[REARM] retry cooldown complete; waiting again for a valid launch gate.")
    finally:
        session_logger.write_manifest(
            "experiment_sequence_final_summary.json",
            {
                "case": asdict(case),
                "session_label": session,
                "target_valid_throws": target,
                "valid_throw_count": valid_count,
                "invalid_attempt_count": invalid_count,
                "controller_memory": controller.memory_summary(),
                "pre_arm_vicon_inactive_delay_s": float(pre_arm_delay_s),
                "vicon_tracking_rate_hz": float(1.0 / vicon_poll_period_s),
                "vicon_poll_period_s": float(vicon_poll_period_s),
                "calibration_profile": calibration_profile.to_manifest(),
                "vicon_attitude_signs_phi_theta_psi": tuple(float(value) for value in vicon_attitude_signs),
                "vicon_attitude_offset_rad_phi_theta_psi": tuple(float(value) for value in vicon_attitude_offset_rad),
                "launch_gate_body_rate_limits_rad_s": tuple(
                    float(value) for value in base_config.launch_gate_body_rate_limits_rad_s
                ),
                "launch_gate_body_rate_limits_policy": "formal_evidence_default",
                "between_throw_policy": (
                    "pre_arm_and_cooldown_streamed_neutral_only_without_vicon_reads;"
                    "each_throw_started_with_lazy_vicon_launch_gate_wait"
                ),
            },
        )
        session_logger.close()
    print(f"[COMPLETE] case={case.case_id} valid={valid_count}/{target} invalid_attempts={invalid_count}")
    return {
        "case_id": case.case_id,
        "session_root": session_root.as_posix(),
        "valid_throw_count": valid_count,
        "invalid_attempt_count": invalid_count,
        "memory": controller.memory_summary(),
    }


def _session_row(case, summary: dict[str, object], valid_count: int, invalid_count: int) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "case_name": case.case_name,
        "layout_id": case.layout_id,
        "controller_mode": case.controller_mode,
        "memory_enabled": case.memory_enabled,
        "valid_throw_count": int(valid_count),
        "invalid_attempt_count": int(invalid_count),
        "latest_run_root": summary.get("run_root", ""),
        "latest_valid_throw": bool(summary.get("valid_throw", False)),
        "latest_launch_speed_m_s": float(summary.get("launch_speed_m_s", 0.0)),
        "latest_termination_reason": summary.get("termination_reason", ""),
        "latest_controller_decisions": int(summary.get("controller_decision_count", 0)),
        "latest_max_decision_time_s": float(summary.get("max_decision_time_s", 0.0)),
        "latest_visible_fan_count": int(summary.get("fan_visible_count_latest", 0)),
        "latest_fan_expected_count_ok": bool(summary.get("fan_expected_count_ok_latest", False)),
        "latest_memory_update_observation_count": int(summary.get("memory_update_observation_count", 0)),
        "latest_memory_cell_count": int(summary.get("memory_cell_count", 0)),
        "latest_cancellation_reason": summary.get("cancellation_reason", ""),
    }


def _available_run_root(preferred: Path) -> Path:
    if not preferred.exists():
        return preferred
    for retry_index in range(1, 1000):
        candidate = preferred.with_name(f"{preferred.name}_retry_{retry_index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free run folder beside {preferred}")


def _move_invalid_attempt(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[WARN] invalid archive target already exists; leaving source in place: {target}")
        return False
    attempts = 5
    last_error: OSError | None = None
    for attempt_index in range(1, attempts + 1):
        try:
            source.replace(target)
            return True
        except OSError as exc:
            last_error = exc
            if attempt_index < attempts:
                time.sleep(0.25)
    if last_error is not None:
        print(
            f"[WARN] invalid attempt archive failed after {attempts} tries; "
            f"leaving source in place: {source} -> {target} "
            f"({type(last_error).__name__}: {last_error})"
        )
    return False


def _cooldown(
    duration_s: float,
    *,
    label: str,
    mode: str,
    serial_port: str,
    controller: FrozenFlightController,
    serial_period_s: float,
    serial_baud: int,
) -> None:
    duration = max(0.0, float(duration_s))
    if duration <= 0.0:
        return
    tx = NanoSerialTx(serial_port, serial_baud) if mode == "armed" else FakeNanoSerialTx()
    started = time.perf_counter()
    next_status_s = 0.0
    packet_count = 0
    try:
        tx.open()
        tx.write_line("SET_NEUTRAL")
        print(f"[WAIT] {label}: Vicon inactive; streaming neutral only.")
        while True:
            elapsed_s = time.perf_counter() - started
            if elapsed_s >= duration:
                break
            remaining_s = max(0.0, duration - elapsed_s)
            if elapsed_s + 1e-12 >= next_status_s:
                print(f"[WAIT] {label}: {remaining_s:.0f}s, Vicon inactive, holding neutral")
                next_status_s = elapsed_s + 1.0
            tx.write_packet(controller.neutral_packet())
            packet_count += 1
            sleep_s = max(0.0, min(float(serial_period_s), duration - (time.perf_counter() - started)))
            if mode == "armed":
                time.sleep(sleep_s)
            else:
                if sleep_s > 0.0:
                    time.sleep(min(0.02, sleep_s))
                break
    except Exception as exc:
        print(f"[WAIT] {label}: neutral hold failed ({type(exc).__name__}: {exc}); sleeping without command stream.")
        while time.perf_counter() - started < duration:
            remaining_s = max(0.0, duration - (time.perf_counter() - started))
            if (time.perf_counter() - started) + 1e-12 >= next_status_s:
                print(f"[WAIT] {label}: {remaining_s:.0f}s")
                next_status_s = (time.perf_counter() - started) + 1.0
            time.sleep(1.0 if mode == "armed" else min(0.02, remaining_s))
            if mode != "armed":
                break
    finally:
        try:
            tx.write_line("SET_NEUTRAL")
        except Exception:
            pass
        tx.close()
    print(f"[WAIT] {label}: complete, neutral_packets={packet_count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an operator-facing repeated real-flight experiment case.")
    parser.add_argument("--case-id", default=CURRENT_EXPERIMENT_CASE, choices=tuple(sorted(EXPERIMENT_CASES)))
    parser.add_argument("--session-label", default=CURRENT_SESSION_LABEL)
    parser.add_argument("--mode", choices=("dry-run", "vicon-smoke", "armed"), default=MODE)
    parser.add_argument("--serial-port", default=SERIAL_PORT)
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--target-valid-throws", type=int, default=TARGET_VALID_THROWS_OVERRIDE)
    parser.add_argument("--pre-arm-delay-s", type=float, default=PRE_ARM_VICON_INACTIVE_DELAY_S)
    parser.add_argument("--cooldown-s", type=float, default=COOLDOWN_AFTER_VALID_THROW_S)
    parser.add_argument("--retry-cooldown-s", type=float, default=RETRY_AFTER_INVALID_START_S)
    parser.add_argument("--max-invalid-attempts", type=int, default=MAX_INVALID_ATTEMPTS)
    parser.add_argument("--duration-s", type=float, default=MAX_ACTIVE_FLIGHT_DURATION_S)
    parser.add_argument("--launch-wait-timeout-s", type=float, default=LAUNCH_WAIT_TIMEOUT_S)
    parser.add_argument("--post-exit-neutral-tail-s", type=float, default=POST_EXIT_NEUTRAL_TAIL_S)
    parser.add_argument("--vicon-tracking-rate-hz", type=float, default=VICON_TRACKING_RATE_HZ)
    parser.add_argument("--vicon-offset-m", nargs=3, type=float, default=None)
    parser.add_argument("--vicon-yaw-deg", type=float, default=VICON_YAW_ALIGNMENT_DEG)
    parser.add_argument("--vicon-attitude-signs", nargs=3, type=float, default=VICON_ATTITUDE_SIGNS)
    parser.add_argument(
        "--vicon-attitude-offset-deg",
        nargs=3,
        type=float,
        default=tuple(float(math.degrees(value)) for value in VICON_ATTITUDE_OFFSET_RAD),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experiment_sequence(
        case_id=args.case_id,
        session_label=args.session_label,
        mode=args.mode,
        serial_port=args.serial_port,
        vicon_host=args.vicon_host,
        target_valid_throws=args.target_valid_throws,
        pre_arm_delay_s=args.pre_arm_delay_s,
        cooldown_s=args.cooldown_s,
        retry_cooldown_s=args.retry_cooldown_s,
        max_invalid_attempts=args.max_invalid_attempts,
        max_duration_s=args.duration_s,
        launch_wait_timeout_s=args.launch_wait_timeout_s,
        post_exit_neutral_tail_s=args.post_exit_neutral_tail_s,
        vicon_poll_period_s=1.0 / float(args.vicon_tracking_rate_hz),
        vicon_position_offset_m=tuple(args.vicon_offset_m) if args.vicon_offset_m is not None else VICON_POSITION_OFFSET_M,
        vicon_yaw_alignment_deg=float(args.vicon_yaw_deg),
        vicon_attitude_signs=tuple(args.vicon_attitude_signs),
        vicon_attitude_offset_rad=tuple(float(math.radians(value)) for value in args.vicon_attitude_offset_deg),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
