from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from exit_gate import evaluate_exit_gate, exit_gate_bounds_manifest
from flight_config import (
    CONTROLLER_ROOT,
    FlightRuntimeConfig,
    RESULT_ROOT,
)
from flight_logger import FlightLogger
from launch_gate import (
    LAUNCH_TRIGGER_X_W_M,
    evaluate_launch_gate,
    evaluate_launch_plane_gate,
    interpolate_launch_plane_state,
    launch_gate_bounds_manifest,
)
from nano_serial import FakeNanoSerialTx, NanoSerialTx
from vicon_rigid_body import LiveNausicaaViconRigidBody, ReplayNausicaaViconRigidBody

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import (  # noqa: E402
    NausicaaViconStateAdapter,
    ViconArenaFrameTransform,
    encode_arduino_command_packet,
)
from state_contract import state_dataframe_row  # noqa: E402


# =============================================================================
# CLICK-AND-GO CALIBRATION SETTINGS FOR THE NEXT GLIDER MODEL CHECK
# Current setup: dry-air calibration, hardware armed on COM11.
# Failed launch-gate attempts are ignored and do not count as throws.
# =============================================================================
CALIBRATION_BLOCK = "pulse_ladder_30"  # neutral_30 or pulse_ladder_30
CURRENT_SESSION_LABEL = ""  # Empty means timestamped folder.
TARGET_VALID_THROWS_OVERRIDE: int | None = None  # None uses block defaults below.
SERIAL_PORT = "COM11"
VICON_HOST = "192.168.0.100:801"
MODE = "armed"  # Use dry-run for hardware-free tests.
PRE_ARM_DELAY_S = 3.0
COOLDOWN_AFTER_VALID_THROW_S = 5.0
RETRY_AFTER_INVALID_START_S = 5.0
LAUNCH_WAIT_TIMEOUT_S = 120.0
MAX_ACTIVE_FLIGHT_DURATION_S = 20.0
POST_EXIT_NEUTRAL_TAIL_S = 0.30
VICON_TRACKING_RATE_HZ = 200.0
CALIBRATION_LAUNCH_GATE_REQUIRED_CONSECUTIVE_FRAMES = 2
CALIBRATION_REJECTED_LAUNCH_ATTEMPT_MIN_SPEED_M_S = 2.0
# Keep these arena-frame values matched to run_experiment_sequence.py.
# Paste the calibration script's recommended full x/y/z offset here.
VICON_POSITION_OFFSET_M = (4.136158795250567, 2.4114272057075916, 0.03414746062731508)
VICON_YAW_ALIGNMENT_DEG = 0.0
# Recovered from 20260601_205149 orientation check: pitch and yaw were reversed.
VICON_ATTITUDE_SIGNS = (1.0, -1.0, -1.0)

NEUTRAL_VALID_THROWS = 30
PULSE_VALID_THROWS_PER_CASE = 3
PULSE_START_DELAY_S = 0.25
PULSE_DURATION_BY_ABS_COMMAND = {
    0.2: 0.25,
    0.4: 0.20,
    0.6: 0.15,
    0.8: 0.12,
    1.0: 0.10,
}
# =============================================================================


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    case_name: str
    command_axis: str
    command_value: float
    pulse_start_s: float
    pulse_duration_s: float
    target_valid_throws: int

    @property
    def is_neutral(self) -> bool:
        return self.command_axis == "neutral"


def calibration_cases_for_block(block_id: str) -> list[CalibrationCase]:
    block = str(block_id)
    if block == "neutral_30":
        return [
            CalibrationCase(
                case_id="C0_neutral",
                case_name="Dry-air open-loop neutral glide characterisation",
                command_axis="neutral",
                command_value=0.0,
                pulse_start_s=0.0,
                pulse_duration_s=0.0,
                target_valid_throws=NEUTRAL_VALID_THROWS,
            )
        ]
    if block == "pulse_ladder_30":
        cases: list[CalibrationCase] = []
        axis_order = (("elevator", "delta_e"), ("aileron", "delta_a"), ("rudder", "delta_r"))
        command_values = (0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0)
        for axis_label, axis_name in axis_order:
            for value in command_values:
                abs_value = round(abs(float(value)), 1)
                cases.append(
                    CalibrationCase(
                        case_id=f"C1_{axis_label}_{_command_label(value)}",
                        case_name=f"{axis_label} pulse {value:+.1f}",
                        command_axis=axis_name,
                        command_value=float(value),
                        pulse_start_s=PULSE_START_DELAY_S,
                        pulse_duration_s=float(PULSE_DURATION_BY_ABS_COMMAND[abs_value]),
                        target_valid_throws=PULSE_VALID_THROWS_PER_CASE,
                    )
                )
        return cases
    raise ValueError("CALIBRATION_BLOCK must be 'neutral_30' or 'pulse_ladder_30'.")


def run_calibration_sequence(
    *,
    block_id: str,
    session_label: str,
    mode: str,
    serial_port: str,
    vicon_host: str,
    pre_arm_delay_s: float,
    cooldown_s: float,
    retry_cooldown_s: float,
    launch_wait_timeout_s: float,
    max_duration_s: float,
    post_exit_neutral_tail_s: float,
    vicon_tracking_rate_hz: float,
    vicon_position_offset_m: tuple[float, float, float],
    vicon_yaw_alignment_deg: float,
    vicon_attitude_signs: tuple[float, float, float],
    target_valid_throws: int | None = None,
) -> dict[str, object]:
    cases = calibration_cases_for_block(block_id)
    if target_valid_throws is not None:
        override = int(target_valid_throws)
        if override <= 0:
            raise ValueError("target_valid_throws must be positive when provided.")
        cases = [replace(case, target_valid_throws=override) for case in cases]
    session = session_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = RESULT_ROOT / "glider_calibration" / str(block_id) / session
    session_logger = FlightLogger(session_root)
    base_config = _base_config(
        run_label="calibration_session",
        mode=mode,
        serial_port=serial_port,
        vicon_host=vicon_host,
        launch_wait_timeout_s=launch_wait_timeout_s,
        max_duration_s=max_duration_s,
        post_exit_neutral_tail_s=post_exit_neutral_tail_s,
        vicon_tracking_rate_hz=vicon_tracking_rate_hz,
        vicon_position_offset_m=vicon_position_offset_m,
        vicon_yaw_alignment_deg=vicon_yaw_alignment_deg,
        vicon_attitude_signs=vicon_attitude_signs,
        output_root=session_root,
    )
    session_logger.write_manifest(
        "glider_calibration_sequence_manifest.json",
        {
            "block_id": block_id,
            "session_label": session,
            "mode": mode,
            "case_count": len(cases),
            "cases": [asdict(case) for case in cases],
            "target_valid_throws_override": target_valid_throws,
            "launch_gate_bounds": launch_gate_bounds_manifest(
                body_rate_limits_rad_s=base_config.launch_gate_body_rate_limits_rad_s
            ),
            "exit_gate_bounds": exit_gate_bounds_manifest(),
            "pulse_policy": (
                "single_axis_0p2_lattice_pulse_after_launch_then_neutral;"
                "no_closed_loop_governor_no_memory_no_combined_axis_commands"
            ),
            "pre_arm_delay_s": float(pre_arm_delay_s),
            "cooldown_after_valid_throw_s": float(cooldown_s),
            "retry_after_invalid_start_s": float(retry_cooldown_s),
            "launch_gate_required_consecutive_frames": int(base_config.launch_gate_required_consecutive_frames),
            "rejected_launch_attempt_min_speed_m_s": float(CALIBRATION_REJECTED_LAUNCH_ATTEMPT_MIN_SPEED_M_S),
            "vicon_tracking_rate_hz": float(vicon_tracking_rate_hz),
            "vicon_position_offset_m": tuple(float(value) for value in vicon_position_offset_m),
            "vicon_attitude_signs": tuple(float(value) for value in vicon_attitude_signs),
        },
    )

    total_valid = 0
    total_invalid = 0
    try:
        for case in cases:
            valid_count = 0
            invalid_count = 0
            while valid_count < int(case.target_valid_throws):
                next_throw = valid_count + 1
                print(
                    f"[CAL] block={block_id} case={case.case_id} valid={valid_count}/{case.target_valid_throws} "
                    f"next_throw={next_throw:03d} invalid={invalid_count}"
                )
                _neutral_cooldown(
                    pre_arm_delay_s,
                    label=f"{case.case_id}_pre_arm",
                    mode=mode,
                    serial_port=serial_port,
                    serial_baud=base_config.serial_baud,
                    serial_period_s=base_config.serial_period_s,
                )
                preferred_run_root = session_root / case.case_id / f"throw_{next_throw:03d}"
                run_root = _available_run_root(preferred_run_root)
                if run_root != preferred_run_root:
                    print(
                        f"[CAL_WARN] preferred run folder is still present; "
                        f"recording this attempt in {run_root.name}"
                    )
                run_label = f"{case.case_id}/{run_root.name}"
                config = _base_config(
                    run_label=run_label,
                    mode=mode,
                    serial_port=serial_port,
                    vicon_host=vicon_host,
                    launch_wait_timeout_s=launch_wait_timeout_s,
                    max_duration_s=max_duration_s,
                    post_exit_neutral_tail_s=post_exit_neutral_tail_s,
                    vicon_tracking_rate_hz=vicon_tracking_rate_hz,
                    vicon_position_offset_m=vicon_position_offset_m,
                    vicon_yaw_alignment_deg=vicon_yaw_alignment_deg,
                    vicon_attitude_signs=vicon_attitude_signs,
                    output_root=session_root,
                )
                summary = run_single_calibration_throw(
                    config=config,
                    case=case,
                    mode=mode,
                    run_root=run_root,
                )
                if bool(summary.get("valid_throw", False)):
                    valid_count += 1
                    total_valid += 1
                    session_logger.append_metric_row(
                        "glider_calibration_sequence_summary.csv",
                        _sequence_row(block_id, case, summary, valid_count, invalid_count),
                    )
                    print(
                        f"[CAL_DONE] case={case.case_id} valid={valid_count}/{case.target_valid_throws} "
                        f"speed={float(summary.get('launch_speed_m_s', 0.0)):.2f} "
                        f"term={summary.get('termination_reason', '')}"
                    )
                    if valid_count < int(case.target_valid_throws):
                        _neutral_cooldown(
                            cooldown_s,
                            label=f"{case.case_id}_cooldown",
                            mode=mode,
                            serial_port=serial_port,
                            serial_baud=base_config.serial_baud,
                            serial_period_s=base_config.serial_period_s,
                        )
                else:
                    invalid_count += 1
                    total_invalid += 1
                    invalid_root = session_root / case.case_id / "invalid_attempts" / f"attempt_{invalid_count:03d}"
                    _move_if_exists(run_root, invalid_root)
                    session_logger.append_metric_row(
                        "glider_calibration_sequence_summary.csv",
                        _sequence_row(block_id, case, summary, valid_count, invalid_count),
                    )
                    print(
                        f"[CAL_INVALID] case={case.case_id} invalid={invalid_count} "
                        f"reason={summary.get('cancellation_reason', '')}"
                    )
                    _neutral_cooldown(
                        retry_cooldown_s,
                        label=f"{case.case_id}_retry",
                        mode=mode,
                        serial_port=serial_port,
                        serial_baud=base_config.serial_baud,
                        serial_period_s=base_config.serial_period_s,
                    )
    finally:
        session_logger.write_manifest(
            "glider_calibration_sequence_final_summary.json",
            {
                "block_id": block_id,
                "session_label": session,
                "total_valid_throw_count": total_valid,
                "total_invalid_attempt_count": total_invalid,
                "cases": [asdict(case) for case in cases],
            },
        )
        session_logger.close()

    return {
        "block_id": block_id,
        "session_root": session_root.as_posix(),
        "total_valid_throw_count": total_valid,
        "total_invalid_attempt_count": total_invalid,
    }


def run_single_calibration_throw(
    *,
    config: FlightRuntimeConfig,
    case: CalibrationCase,
    mode: str,
    run_root: Path,
) -> dict[str, object]:
    logger = FlightLogger(run_root)
    tx = NanoSerialTx(config.serial_port, config.serial_baud) if mode == "armed" else FakeNanoSerialTx()
    vicon = (
        LiveNausicaaViconRigidBody(host=config.vicon_host, subject_name=config.vicon_subject_name)
        if mode in {"armed", "vicon-smoke"}
        else ReplayNausicaaViconRigidBody(dt_s=config.vicon_poll_period_s)
    )
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=config.derivative_cutoff_hz,
        body_rate_limit_rad_s=config.body_rate_limit_rad_s,
        body_rate_observer_window_frames=config.body_rate_observer_window_frames,
        actuator_tau_s=config.actuator_tau_s,
        arena_transform=ViconArenaFrameTransform(
            position_offset_m=config.vicon_position_offset_m,
            yaw_alignment_rad=float(np.deg2rad(config.vicon_yaw_alignment_deg)),
            attitude_signs=config.vicon_attitude_signs,
        ),
    )
    logger.write_manifest(
        "glider_calibration_throw_manifest.json",
        {
            "mode": mode,
            "config": asdict(config),
            "calibration_case": asdict(case),
            "launch_gate_bounds": launch_gate_bounds_manifest(
                body_rate_limits_rad_s=config.launch_gate_body_rate_limits_rad_s
            ),
            "exit_gate_bounds": exit_gate_bounds_manifest(),
            "control_policy": (
                "neutral_only"
                if case.is_neutral
                else "single_axis_0p2_lattice_pulse_after_launch_then_neutral"
            ),
        },
    )
    summary: dict[str, object] = {
        "run_root": run_root.as_posix(),
        "case_id": case.case_id,
        "case_name": case.case_name,
        "valid_throw": False,
        "flight_cancelled": False,
        "launch_gate_approved": False,
        "cancellation_reason": "",
        "termination_reason": "",
        "exit_gate_triggered": False,
        "state_sample_count": 0,
        "packet_count": 0,
        "serial_write_error_count": 0,
        "launch_speed_m_s": 0.0,
        "completed": False,
    }
    sequence = 0
    try:
        tx.open()
        vicon.open()
        launch_state = _await_launch_gate(
            config=config,
            tx=tx,
            vicon=vicon,
            adapter=adapter,
            logger=logger,
            summary=summary,
            mode=mode,
            sequence=sequence,
        )
        if launch_state is None:
            summary["flight_cancelled"] = True
            return summary
        summary["valid_throw"] = True
        summary["launch_speed_m_s"] = float(np.linalg.norm(launch_state[6:9]))
        started = time.perf_counter()
        next_serial_s = 0.0
        last_command = np.zeros(3, dtype=float)
        while time.perf_counter() - started <= float(config.max_duration_s):
            elapsed_s = time.perf_counter() - started
            sample, status = vicon.read_latest()
            if sample is None or not status.valid:
                command = np.zeros(3, dtype=float)
                state = None
                estimator = {}
            else:
                state = adapter.update(sample, command_norm=last_command)
                estimator = adapter.estimator_status()
                safety = evaluate_exit_gate(state)
                logger.append_metric_row(
                    "state_samples.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        "t_s": elapsed_s,
                        "frame_number": status.frame_number,
                        "vicon_frame_rate_hz": status.frame_rate_hz,
                        "vicon_latency_s": status.vicon_latency_s,
                        **{f"estimator_{key}": value for key, value in estimator.items()},
                        **state_dataframe_row(state),
                        "exit_gate_inside": safety.inside,
                        "exit_gate_reason": safety.reason,
                        "exit_gate_min_margin_m": safety.min_margin_m,
                    },
                )
                summary["state_sample_count"] = int(summary["state_sample_count"]) + 1
                if not safety.inside:
                    summary["exit_gate_triggered"] = True
                    summary["termination_reason"] = safety.reason
                    _append_event(logger, "exit_gate_terminate_calibration_record", **asdict(safety))
                    break
                command = _command_for_case(case, elapsed_s)
            if elapsed_s + 1e-12 >= next_serial_s:
                packet = encode_arduino_command_packet(command, sequence=sequence)
                sequence += 1
                _write_packet(tx, packet.packet_bytes, logger=logger, summary=summary, event="calibration_command")
                last_command = np.asarray(packet.aggregate_command_norm, dtype=float)
                logger.append_metric_row(
                    "command_schedule.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        "t_s": elapsed_s,
                        "case_id": case.case_id,
                        "command_axis": case.command_axis,
                        "scheduled_command_value": case.command_value,
                        "pulse_active": _pulse_active(case, elapsed_s),
                        "delta_a_cmd_norm": last_command[0],
                        "delta_e_cmd_norm": last_command[1],
                        "delta_r_cmd_norm": last_command[2],
                        "packet_bytes": packet.packet_bytes,
                    },
                )
                next_serial_s = elapsed_s + float(config.serial_period_s)
            if mode in {"armed", "vicon-smoke"}:
                time.sleep(float(config.vicon_poll_period_s))
        _send_neutral_tail(
            tx=tx,
            logger=logger,
            summary=summary,
            sequence_start=sequence,
            duration_s=config.post_exit_neutral_tail_s,
            serial_period_s=config.serial_period_s,
            mode=mode,
        )
        summary["completed"] = True
        return summary
    finally:
        logger.write_manifest("glider_calibration_throw_summary.json", summary)
        try:
            tx.write_line("SET_NEUTRAL")
        except Exception:
            pass
        tx.close()
        vicon.close()
        logger.close()


def _await_launch_gate(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    vicon: LiveNausicaaViconRigidBody | ReplayNausicaaViconRigidBody,
    adapter: NausicaaViconStateAdapter,
    logger: FlightLogger,
    summary: dict[str, object],
    mode: str,
    sequence: int,
) -> np.ndarray | None:
    started = time.perf_counter()
    previous_state: np.ndarray | None = None
    consecutive = 0
    next_serial_s = 0.0
    next_status_s = 0.0
    latest_reason = "not_evaluated"
    required = max(1, int(config.launch_gate_required_consecutive_frames))
    rate_conf_min = float(config.launch_gate_rate_confidence_min)
    while time.perf_counter() - started <= float(config.launch_wait_timeout_s):
        elapsed_s = time.perf_counter() - started
        sample, status = vicon.read_latest()
        if sample is None or not status.valid:
            latest_reason = status.reason
            if elapsed_s + 1e-12 >= next_status_s:
                print(f"[CAL_LAUNCH_WAIT] t={elapsed_s:.1f}s Vicon invalid: {status.reason}")
                next_status_s = elapsed_s + 1.0
            next_serial_s, sequence = _neutral_if_due(
                tx=tx,
                logger=logger,
                summary=summary,
                elapsed_s=elapsed_s,
                next_serial_s=next_serial_s,
                serial_period_s=config.serial_period_s,
                sequence=sequence,
            )
            if mode in {"armed", "vicon-smoke"}:
                time.sleep(float(config.vicon_poll_period_s))
            continue
        state = adapter.update(sample, command_norm=np.zeros(3))
        estimator = adapter.estimator_status()
        full_gate = evaluate_launch_gate(state, body_rate_limits_rad_s=config.launch_gate_body_rate_limits_rad_s)
        launch_plane_state = interpolate_launch_plane_state(previous_state, state)
        plane_gate = (
            evaluate_launch_plane_gate(launch_plane_state, body_rate_limits_rad_s=config.launch_gate_body_rate_limits_rad_s)
            if launch_plane_state is not None
            else None
        )
        if full_gate.approved:
            gate = full_gate
            launch_state = state
            trigger_source = "valid_r5_launch_window"
            approved = True
        elif plane_gate is not None:
            gate = plane_gate
            launch_state = launch_plane_state
            trigger_source = "valid_interpolated_launch_plane" if plane_gate.approved else "crossed_plane_rejected"
            approved = bool(plane_gate.approved)
        else:
            gate = full_gate
            launch_state = None
            trigger_source = "tracking"
            approved = False
        rate_conf = float(estimator.get("rate_confidence", 0.0))
        rate_ok = bool(rate_conf >= rate_conf_min)
        if approved and not rate_ok:
            approved = False
            launch_state = None
            latest_reason = "rate_confidence_below_launch_gate"
        else:
            latest_reason = gate.reason
        consecutive = consecutive + 1 if approved else 0
        logger.append_metric_row(
            "prelaunch_state_samples.csv",
            {
                "t_host_s": time.perf_counter(),
                "frame_number": status.frame_number,
                "vicon_frame_rate_hz": status.frame_rate_hz,
                "vicon_latency_s": status.vicon_latency_s,
                **{f"estimator_{key}": value for key, value in estimator.items()},
                "trigger_source": trigger_source,
                "trigger_approved": approved,
                "base_trigger_approved_before_rate_confidence": bool(gate.approved),
                "rate_confidence_ok": rate_ok,
                "launch_gate_consecutive_approved": consecutive,
                "launch_trigger_x_w_m": float(LAUNCH_TRIGGER_X_W_M),
                "crossed_launch_plane": bool(launch_plane_state is not None),
                "full_window_reason": full_gate.reason,
                "interpolated_plane_reason": plane_gate.reason if plane_gate is not None else "",
                **asdict(gate),
                **state_dataframe_row(state),
            },
        )
        if elapsed_s + 1e-12 >= next_status_s:
            print(
                f"[CAL_LAUNCH_WAIT] t={elapsed_s:.1f}s trigger={trigger_source} gate={gate.reason} "
                f"approved={approved} consecutive={consecutive}/{required} rate_conf={rate_conf:.2f}/{rate_conf_min:.2f}"
            )
            next_status_s = elapsed_s + 1.0
        next_serial_s, sequence = _neutral_if_due(
            tx=tx,
            logger=logger,
            summary=summary,
            elapsed_s=elapsed_s,
            next_serial_s=next_serial_s,
            serial_period_s=config.serial_period_s,
            sequence=sequence,
        )
        if approved and consecutive >= required:
            summary["launch_gate_approved"] = True
            _append_event(
                logger,
                "launch_gate_approved_start_calibration_record",
                required_consecutive_frames=required,
                rate_confidence=rate_conf,
                trigger_source=trigger_source,
                **asdict(gate),
            )
            return launch_state.copy() if launch_state is not None else state.copy()
        if launch_plane_state is not None and not approved:
            launch_attempt_speed = float(np.linalg.norm(launch_plane_state[6:9]))
            if launch_attempt_speed >= float(CALIBRATION_REJECTED_LAUNCH_ATTEMPT_MIN_SPEED_M_S):
                summary["cancellation_reason"] = f"rejected_launch_attempt:{latest_reason}"
                gate_details = asdict(gate)
                gate_reason = gate_details.pop("reason", latest_reason)
                _append_event(
                    logger,
                    "calibration_record_rejected_launch_attempt",
                    cancellation_reason=summary["cancellation_reason"],
                    launch_gate_reason=gate_reason,
                    launch_attempt_speed_m_s=launch_attempt_speed,
                    trigger_source=trigger_source,
                    **gate_details,
                )
                return None
        previous_state = state.copy()
        if mode in {"armed", "vicon-smoke"}:
            time.sleep(float(config.vicon_poll_period_s))
    summary["cancellation_reason"] = f"launch_gate_timeout:{latest_reason}"
    _append_event(logger, "calibration_record_cancelled_before_launch", reason=summary["cancellation_reason"])
    return None


def _command_for_case(case: CalibrationCase, elapsed_s: float) -> np.ndarray:
    command = np.zeros(3, dtype=float)
    if case.is_neutral or not _pulse_active(case, elapsed_s):
        return command
    axis_to_index = {"delta_a": 0, "delta_e": 1, "delta_r": 2}
    command[axis_to_index[case.command_axis]] = float(case.command_value)
    return command


def _pulse_active(case: CalibrationCase, elapsed_s: float) -> bool:
    return bool(
        not case.is_neutral
        and float(case.pulse_start_s) <= float(elapsed_s) < float(case.pulse_start_s + case.pulse_duration_s)
    )


def _neutral_if_due(
    *,
    tx: NanoSerialTx | FakeNanoSerialTx,
    logger: FlightLogger,
    summary: dict[str, object],
    elapsed_s: float,
    next_serial_s: float,
    serial_period_s: float,
    sequence: int,
) -> tuple[float, int]:
    if elapsed_s + 1e-12 < next_serial_s:
        return next_serial_s, sequence
    packet = encode_arduino_command_packet(np.zeros(3), sequence=sequence)
    _write_packet(tx, packet.packet_bytes, logger=logger, summary=summary, event="prelaunch_neutral")
    return elapsed_s + float(serial_period_s), sequence + 1


def _send_neutral_tail(
    *,
    tx: NanoSerialTx | FakeNanoSerialTx,
    logger: FlightLogger,
    summary: dict[str, object],
    sequence_start: int,
    duration_s: float,
    serial_period_s: float,
    mode: str,
) -> None:
    started = time.perf_counter()
    sequence = int(sequence_start)
    packet_count = 0
    while time.perf_counter() - started < max(0.0, float(duration_s)):
        packet = encode_arduino_command_packet(np.zeros(3), sequence=sequence)
        sequence += 1
        if _write_packet(tx, packet.packet_bytes, logger=logger, summary=summary, event="post_exit_neutral"):
            packet_count += 1
        if mode == "armed":
            time.sleep(float(serial_period_s))
        else:
            break
    summary["post_exit_neutral_packets"] = packet_count


def _write_packet(
    tx: NanoSerialTx | FakeNanoSerialTx,
    packet: bytes,
    *,
    logger: FlightLogger,
    summary: dict[str, object],
    event: str,
) -> bool:
    try:
        tx.write_packet(packet)
    except Exception as exc:
        summary["serial_write_error_count"] = int(summary.get("serial_write_error_count", 0)) + 1
        logger.append_metric_row(
            "serial_events.csv",
            {"t_host_s": time.perf_counter(), "event": event, "error": f"{type(exc).__name__}:{exc}"},
        )
        return False
    summary["packet_count"] = int(summary.get("packet_count", 0)) + 1
    return True


def _neutral_cooldown(
    duration_s: float,
    *,
    label: str,
    mode: str,
    serial_port: str,
    serial_baud: int,
    serial_period_s: float,
) -> None:
    duration = max(0.0, float(duration_s))
    if duration <= 0.0:
        return
    tx = NanoSerialTx(serial_port, serial_baud) if mode == "armed" else FakeNanoSerialTx()
    started = time.perf_counter()
    sequence = 0
    try:
        tx.open()
        tx.write_line("SET_NEUTRAL")
        print(f"[CAL_WAIT] {label}: Vicon inactive; holding neutral.")
        while time.perf_counter() - started < duration:
            packet = encode_arduino_command_packet(np.zeros(3), sequence=sequence)
            sequence += 1
            try:
                tx.write_packet(packet.packet_bytes)
            except Exception:
                break
            if mode == "armed":
                time.sleep(float(serial_period_s))
            else:
                break
    finally:
        try:
            tx.write_line("SET_NEUTRAL")
        except Exception:
            pass
        tx.close()


def _base_config(
    *,
    run_label: str,
    mode: str,
    serial_port: str,
    vicon_host: str,
    launch_wait_timeout_s: float,
    max_duration_s: float,
    post_exit_neutral_tail_s: float,
    vicon_tracking_rate_hz: float,
    vicon_position_offset_m: tuple[float, float, float],
    vicon_yaw_alignment_deg: float,
    vicon_attitude_signs: tuple[float, float, float],
    output_root: Path,
) -> FlightRuntimeConfig:
    del mode
    return FlightRuntimeConfig(
        run_label=run_label,
        controller_mode="open_loop_neutral",
        experiment_case_id="glider_calibration",
        experiment_case_name="dry_air_glider_model_calibration",
        experiment_memory_enabled=False,
        experiment_layout_id="dry_air_no_fans",
        serial_port=serial_port,
        vicon_host=vicon_host,
        max_duration_s=max_duration_s,
        launch_wait_timeout_s=launch_wait_timeout_s,
        post_exit_neutral_tail_s=post_exit_neutral_tail_s,
        launch_gate_required_consecutive_frames=CALIBRATION_LAUNCH_GATE_REQUIRED_CONSECUTIVE_FRAMES,
        vicon_poll_period_s=1.0 / float(vicon_tracking_rate_hz),
        vicon_position_offset_m=vicon_position_offset_m,
        vicon_yaw_alignment_deg=vicon_yaw_alignment_deg,
        vicon_attitude_signs=vicon_attitude_signs,
        output_root=output_root,
    )


def _sequence_row(
    block_id: str,
    case: CalibrationCase,
    summary: dict[str, object],
    valid_count: int,
    invalid_count: int,
) -> dict[str, object]:
    return {
        "block_id": block_id,
        "case_id": case.case_id,
        "case_name": case.case_name,
        "command_axis": case.command_axis,
        "command_value": case.command_value,
        "pulse_start_s": case.pulse_start_s,
        "pulse_duration_s": case.pulse_duration_s,
        "valid_count_for_case": valid_count,
        "invalid_count_for_case": invalid_count,
        "latest_valid_throw": bool(summary.get("valid_throw", False)),
        "latest_run_root": summary.get("run_root", ""),
        "launch_speed_m_s": summary.get("launch_speed_m_s", 0.0),
        "termination_reason": summary.get("termination_reason", ""),
        "state_sample_count": summary.get("state_sample_count", 0),
        "packet_count": summary.get("packet_count", 0),
        "serial_write_error_count": summary.get("serial_write_error_count", 0),
        "cancellation_reason": summary.get("cancellation_reason", ""),
    }


def _append_event(logger: FlightLogger, event: str, **details: object) -> None:
    logger.append_metric_row(
        "runtime_events.csv",
        {
            "t_host_s": time.perf_counter(),
            "event": event,
            "details_json": json.dumps(details, sort_keys=True, separators=(",", ":")),
        },
    )


def _command_label(value: float) -> str:
    return f"{float(value):+.1f}".replace("+", "pos").replace("-", "neg").replace(".", "p")


def _available_run_root(preferred: Path) -> Path:
    if not preferred.exists():
        return preferred
    for retry_index in range(1, 1000):
        candidate = preferred.with_name(f"{preferred.name}_retry_{retry_index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free run folder beside {preferred}")


def _move_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[CAL_WARN] invalid archive target already exists; leaving source in place: {target}")
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
            f"[CAL_WARN] invalid attempt archive failed after {attempts} tries; "
            f"leaving source in place: {source} -> {target} "
            f"({type(last_error).__name__}: {last_error})"
        )
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dry-air glider calibration throws.")
    parser.add_argument("--block", choices=("neutral_30", "pulse_ladder_30"), default=CALIBRATION_BLOCK)
    parser.add_argument("--session-label", default=CURRENT_SESSION_LABEL)
    parser.add_argument("--mode", choices=("dry-run", "vicon-smoke", "armed"), default=MODE)
    parser.add_argument("--serial-port", default=SERIAL_PORT)
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--pre-arm-delay-s", type=float, default=PRE_ARM_DELAY_S)
    parser.add_argument("--cooldown-s", type=float, default=COOLDOWN_AFTER_VALID_THROW_S)
    parser.add_argument("--retry-cooldown-s", type=float, default=RETRY_AFTER_INVALID_START_S)
    parser.add_argument("--launch-wait-timeout-s", type=float, default=LAUNCH_WAIT_TIMEOUT_S)
    parser.add_argument("--duration-s", type=float, default=MAX_ACTIVE_FLIGHT_DURATION_S)
    parser.add_argument("--post-exit-neutral-tail-s", type=float, default=POST_EXIT_NEUTRAL_TAIL_S)
    parser.add_argument("--vicon-tracking-rate-hz", type=float, default=VICON_TRACKING_RATE_HZ)
    parser.add_argument("--vicon-offset-m", nargs=3, type=float, default=None)
    parser.add_argument("--vicon-yaw-deg", type=float, default=VICON_YAW_ALIGNMENT_DEG)
    parser.add_argument("--vicon-attitude-signs", nargs=3, type=float, default=VICON_ATTITUDE_SIGNS)
    parser.add_argument(
        "--target-valid-throws",
        type=int,
        default=TARGET_VALID_THROWS_OVERRIDE,
        help="Optional one-off override for valid throws per calibration case. Defaults to the top-of-file setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_calibration_sequence(
        block_id=args.block,
        session_label=args.session_label,
        mode=args.mode,
        serial_port=args.serial_port,
        vicon_host=args.vicon_host,
        pre_arm_delay_s=args.pre_arm_delay_s,
        cooldown_s=args.cooldown_s,
        retry_cooldown_s=args.retry_cooldown_s,
        launch_wait_timeout_s=args.launch_wait_timeout_s,
        max_duration_s=args.duration_s,
        post_exit_neutral_tail_s=args.post_exit_neutral_tail_s,
        vicon_tracking_rate_hz=args.vicon_tracking_rate_hz,
        vicon_position_offset_m=tuple(args.vicon_offset_m) if args.vicon_offset_m is not None else VICON_POSITION_OFFSET_M,
        vicon_yaw_alignment_deg=float(args.vicon_yaw_deg),
        vicon_attitude_signs=tuple(float(value) for value in args.vicon_attitude_signs),
        target_valid_throws=args.target_valid_throws,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
