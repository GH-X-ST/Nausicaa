from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from flight_config import (
    CONTROLLER_ROOT,
    DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
    DEFAULT_VICON_POSITION_OFFSET_M,
    DEFAULT_VICON_ATTITUDE_SIGNS,
    REAL_FLIGHT_LIBRARY_TIER_SELECTION_REASON,
    FlightRuntimeConfig,
    default_run_label,
)
from flight_logger import FlightLogger
from frozen_flight_controller import FrozenFlightController
from exit_gate import evaluate_exit_gate, exit_gate_bounds_manifest
from launch_gate import (
    LAUNCH_TRIGGER_X_W_M,
    evaluate_launch_gate,
    evaluate_launch_plane_gate,
    interpolate_launch_plane_state,
    launch_gate_bounds_manifest,
)
from nano_serial import FakeNanoSerialTx, NanoSerialTx
from safety_monitor import evaluate_safety
from vicon_rigid_body import FanViconSample, LiveNausicaaViconRigidBody, ReplayNausicaaViconRigidBody

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import NausicaaViconStateAdapter, ViconArenaFrameTransform  # noqa: E402
from state_contract import STATE_INDEX, state_dataframe_row  # noqa: E402


def run_real_flight(
    config: FlightRuntimeConfig,
    *,
    mode: str,
    controller: FrozenFlightController | None = None,
    run_root: Path | None = None,
    expected_visible_fan_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    logger = FlightLogger(Path(run_root) if run_root is not None else config.run_root)
    controller = controller or FrozenFlightController(config)
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=config.derivative_cutoff_hz,
        body_rate_limit_rad_s=config.body_rate_limit_rad_s,
        actuator_tau_s=config.actuator_tau_s,
        arena_transform=ViconArenaFrameTransform(
            position_offset_m=config.vicon_position_offset_m,
            yaw_alignment_rad=float(np.deg2rad(config.vicon_yaw_alignment_deg)),
            attitude_signs=config.vicon_attitude_signs,
        ),
    )
    tx = NanoSerialTx(config.serial_port, config.serial_baud) if mode in {"armed", "packet-smoke"} else FakeNanoSerialTx()
    vicon = (
        LiveNausicaaViconRigidBody(host=config.vicon_host, subject_name=config.vicon_subject_name)
        if mode in {"armed", "vicon-smoke"}
        else ReplayNausicaaViconRigidBody(dt_s=config.vicon_poll_period_s)
    )
    logger.write_manifest(
        "real_flight_runtime_manifest.json",
        {
            "mode": mode,
            "config": asdict(config),
            "deployment_library_tier_policy": {
                "default_real_flight_library_tier": DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
                "selection_reason": REAL_FLIGHT_LIBRARY_TIER_SELECTION_REASON,
                "balanced_cluster_role": "fallback_if_additional_primitive_diversity_is_needed",
            },
            "control_boundary": "vicon_rigid_body_to_canonical_state_to_frozen_governor_to_quantised_packet",
            "surface_marker_tracking_enabled": False,
            "latency_quantification_enabled": False,
            "servo_command_limit_norm": [-1.0, 1.0],
            "launch_trigger_policy": "wait_for_r5_launch_gate_before_active_record",
            "launch_gate_bounds": launch_gate_bounds_manifest(),
            "launch_wait_timeout_s": float(config.launch_wait_timeout_s),
            "launch_gate_required_consecutive_frames": int(config.launch_gate_required_consecutive_frames),
            "exit_gate_bounds": exit_gate_bounds_manifest(),
            "post_exit_neutral_tail_s": float(config.post_exit_neutral_tail_s),
            "runtime_rates": {
                "vicon_poll_hz": float(1.0 / config.vicon_poll_period_s),
                "serial_command_repeat_hz": float(1.0 / config.serial_period_s),
                "governor_decision_hz": float(1.0 / config.governor_period_s),
                "derivative_cutoff_hz": float(config.derivative_cutoff_hz),
                "body_rate_limit_rad_s": float(config.body_rate_limit_rad_s),
                "rate_policy": (
                    "Vicon is polled at the tracking rate; serial packets are still repeated "
                    "at the firmware-safe command period; governor selection remains 10 Hz; "
                    "active-flight angular-rate history is reset at launch."
                ),
            },
            "vicon_arena_frame_transform": {
                "description": config.vicon_frame_description,
                "position_offset_m": tuple(float(value) for value in config.vicon_position_offset_m),
                "yaw_alignment_deg": float(config.vicon_yaw_alignment_deg),
                "attitude_signs_phi_theta_psi": tuple(float(value) for value in config.vicon_attitude_signs),
                "attitude_sign_reason": "recovered_vicon_orientation_check_20260601_205149_pitch_and_yaw_reversed",
            },
            "experiment_case": {
                "case_id": config.experiment_case_id,
                "case_name": config.experiment_case_name,
                "layout_id": config.experiment_layout_id,
                "memory_enabled": bool(config.experiment_memory_enabled),
                "throw_index": int(config.throw_index),
                "attempt_index": int(config.attempt_index),
            },
            "expected_visible_fan_range": expected_visible_fan_range,
        },
    )
    summary = {
        "mode": mode,
        "run_root": (Path(run_root) if run_root is not None else config.run_root).as_posix(),
        "experiment_case_id": config.experiment_case_id,
        "experiment_case_name": config.experiment_case_name,
        "experiment_layout_id": config.experiment_layout_id,
        "throw_index": int(config.throw_index),
        "attempt_index": int(config.attempt_index),
        "valid_throw": False,
        "state_sample_count": 0,
        "controller_decision_count": 0,
        "packet_count": 0,
        "neutral_failsafe_count": 0,
        "serial_write_error_count": 0,
        "serial_write_timeout_count": 0,
        "max_decision_time_s": 0.0,
        "launch_speed_m_s": 0.0,
        "launch_gate_approved": False,
        "flight_cancelled": False,
        "cancellation_reason": "",
        "exit_gate_triggered": False,
        "termination_reason": "",
        "post_exit_neutral_packets": 0,
        "fan_visible_count_latest": 0,
        "fan_expected_count_ok_latest": False,
        "memory_update_observation_count": 0,
        "memory_cell_count": 0,
        "completed": False,
    }
    primitive_step_index = 0
    latest_decision = None
    latest_state = None
    started = time.perf_counter()
    next_governor_s = 0.0
    next_serial_s = 0.0
    decision_records: list[dict[str, object]] = []
    terminal_record_appended = False

    try:
        tx.open()
        vicon.open()
        if mode == "packet-smoke":
            _run_packet_smoke(config=config, tx=tx, controller=controller, logger=logger)
            summary["completed"] = True
            return summary

        launched_state = _await_launch_gate(
            config=config,
            tx=tx,
            vicon=vicon,
            adapter=adapter,
            controller=controller,
            logger=logger,
            mode=mode,
            summary=summary,
            expected_visible_fan_range=expected_visible_fan_range,
        )
        if launched_state is None:
            if _write_packet_safe(
                tx,
                controller.neutral_packet(),
                logger=logger,
                summary=summary,
                event="launch_gate_cancel_neutral",
            ):
                summary["neutral_failsafe_count"] += 1
            summary["flight_cancelled"] = True
            summary["completed"] = False
            return summary
        latest_state = launched_state
        pending_launch_decision_state = launched_state.copy()
        summary["valid_throw"] = True
        summary["launch_speed_m_s"] = float(np.linalg.norm(latest_state[6:9]))
        started = time.perf_counter()
        next_governor_s = 0.0
        next_serial_s = 0.0

        while (time.perf_counter() - started) <= float(config.max_duration_s):
            loop_elapsed_s = time.perf_counter() - started
            sample, status = vicon.read_latest()
            if sample is None or not status.valid:
                packet = controller.neutral_packet()
                if loop_elapsed_s + 1e-12 >= next_serial_s:
                    if _write_packet_safe(
                        tx,
                        packet,
                        logger=logger,
                        summary=summary,
                        event="vicon_invalid_neutral_command",
                    ):
                        summary["neutral_failsafe_count"] += 1
                    next_serial_s += float(config.serial_period_s)
                _append_runtime_event(
                    logger,
                    "vicon_invalid_neutral_command",
                    reason=status.reason,
                )
                if mode in {"armed", "vicon-smoke"}:
                    _sleep_until_next_runtime_poll(
                        config=config,
                        started=started,
                        next_serial_s=next_serial_s,
                        next_governor_s=next_governor_s,
                    )
                continue

            latest_state = adapter.update(sample, command_norm=controller.last_command_norm())
            estimator = adapter.estimator_status()
            _append_fan_positions(
                logger=logger,
                vicon=vicon,
                adapter=adapter,
                phase="active",
                expected_visible_fan_range=expected_visible_fan_range,
                summary=summary,
            )
            safety = evaluate_safety(latest_state)
            exit_gate = evaluate_exit_gate(latest_state)
            summary["state_sample_count"] += 1
            logger.append_metric_row(
                "state_samples.csv",
                {
                    "t_host_s": time.perf_counter(),
                    "frame_number": status.frame_number,
                    "vicon_frame_rate_hz": status.frame_rate_hz,
                    "vicon_latency_s": status.vicon_latency_s,
                    **{f"estimator_{key}": value for key, value in estimator.items()},
                    **state_dataframe_row(latest_state),
                    **asdict(safety),
                    **{f"exit_gate_{key}": value for key, value in asdict(exit_gate).items()},
                },
            )
            if not exit_gate.inside:
                if latest_decision is not None:
                    decision_records.append(
                        {
                            "t_s": float(loop_elapsed_s),
                            "state": latest_state.copy(),
                            "expected_energy_residual_m": float(latest_decision.expected_energy_residual_m),
                            "primitive_variant_id": latest_decision.primitive_variant_id,
                        }
                    )
                    terminal_record_appended = True
                _send_neutral_tail(
                    config=config,
                    tx=tx,
                    controller=controller,
                    logger=logger,
                    mode=mode,
                    summary=summary,
                    reason=exit_gate.reason,
                )
                summary["exit_gate_triggered"] = True
                summary["termination_reason"] = str(exit_gate.reason)
                _append_runtime_event(
                    logger,
                    "exit_gate_terminate_active_record",
                    **asdict(exit_gate),
                )
                break

            if loop_elapsed_s + 1e-12 >= next_governor_s:
                decision_state = (
                    pending_launch_decision_state
                    if primitive_step_index == 0 and pending_launch_decision_state is not None
                    else latest_state
                )
                if primitive_step_index == 0 and pending_launch_decision_state is not None:
                    _append_runtime_event(
                        logger,
                        "first_decision_uses_approved_launch_state",
                        source="launch_gate_interpolated_or_window_state",
                    )
                    pending_launch_decision_state = None
                latest_decision = controller.decide(decision_state, primitive_step_index=primitive_step_index)
                decision_records.append(
                    {
                        "t_s": float(loop_elapsed_s),
                        "state": decision_state.copy(),
                        "expected_energy_residual_m": float(latest_decision.expected_energy_residual_m),
                        "primitive_variant_id": latest_decision.primitive_variant_id,
                    }
                )
                primitive_step_index += 1
                next_governor_s += float(config.governor_period_s)
                summary["controller_decision_count"] += 1
                summary["max_decision_time_s"] = max(
                    float(summary["max_decision_time_s"]),
                    float(latest_decision.decision_time_s),
                )
                logger.append_metric_row(
                    "controller_decisions.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        **asdict(latest_decision),
                    },
                )

            if loop_elapsed_s + 1e-12 >= next_serial_s:
                packet = controller.packet_for_last_command()
                _write_packet_safe(
                    tx,
                    packet,
                    logger=logger,
                    summary=summary,
                    event="active_command_packet",
                )
                next_serial_s += float(config.serial_period_s)

            if mode in {"armed", "vicon-smoke"}:
                _sleep_until_next_runtime_poll(
                    config=config,
                    started=started,
                    next_serial_s=next_serial_s,
                    next_governor_s=next_governor_s,
                )

        if bool(summary["launch_gate_approved"]):
            if latest_decision is not None and latest_state is not None and not terminal_record_appended:
                decision_records.append(
                    {
                        "t_s": float(time.perf_counter() - started),
                        "state": latest_state.copy(),
                        "expected_energy_residual_m": float(latest_decision.expected_energy_residual_m),
                        "primitive_variant_id": latest_decision.primitive_variant_id,
                    }
                )
            memory_summary = controller.update_memory_from_decision_records(decision_records)
            summary["memory_update_observation_count"] = int(memory_summary.observation_count)
            summary["memory_cell_count"] = int(memory_summary.updated_cell_count)
            logger.append_metric_row("memory_update_summary.csv", asdict(memory_summary))
        summary["completed"] = True
        return summary
    finally:
        if latest_state is not None:
            _write_packet_safe(
                tx,
                controller.neutral_packet(),
                logger=logger,
                summary=summary,
                event="final_neutral_packet",
                quiet=True,
            )
        tx.close()
        vicon.close()
        logger.write_manifest("real_flight_runtime_summary.json", summary)
        logger.write_report(
            "real_flight_runtime_report.md",
            [
                "# Real Flight Runtime Report",
                f"- Mode: `{mode}`",
                f"- Run root: `{summary['run_root']}`",
                f"- Experiment case: `{summary['experiment_case_id']}`",
                f"- Valid throw: `{summary['valid_throw']}`",
                f"- Launch gate approved: `{summary['launch_gate_approved']}`",
                f"- Launch speed (m/s): `{float(summary['launch_speed_m_s']):.3f}`",
                f"- Flight cancelled: `{summary['flight_cancelled']}`",
                f"- Cancellation reason: `{summary['cancellation_reason']}`",
                f"- Exit gate triggered: `{summary['exit_gate_triggered']}`",
                f"- Termination reason: `{summary['termination_reason']}`",
                f"- State samples: `{summary['state_sample_count']}`",
                f"- Controller decisions: `{summary['controller_decision_count']}`",
                f"- Packets sent: `{summary['packet_count']}`",
                f"- Neutral failsafe commands: `{summary['neutral_failsafe_count']}`",
                f"- Serial write errors: `{summary['serial_write_error_count']}`",
                f"- Serial write timeouts: `{summary['serial_write_timeout_count']}`",
                f"- Post-exit neutral packets: `{summary['post_exit_neutral_packets']}`",
                f"- Latest visible fan count: `{summary['fan_visible_count_latest']}`",
                f"- Fan expected count OK: `{summary['fan_expected_count_ok_latest']}`",
                f"- Memory update observations: `{summary['memory_update_observation_count']}`",
                f"- Memory cells: `{summary['memory_cell_count']}`",
                f"- Max decision time (s): `{float(summary['max_decision_time_s']):.6f}`",
            ],
        )
        logger.close()


def _await_launch_gate(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    vicon: LiveNausicaaViconRigidBody | ReplayNausicaaViconRigidBody,
    adapter: NausicaaViconStateAdapter,
    controller: FrozenFlightController,
    logger: FlightLogger,
    mode: str,
    summary: dict[str, object],
    expected_visible_fan_range: tuple[int, int] | None,
) -> np.ndarray | None:
    started = time.perf_counter()
    next_serial_s = 0.0
    consecutive_approved = 0
    required_consecutive = max(1, int(config.launch_gate_required_consecutive_frames))
    latest_gate_reason = "not_evaluated"
    next_invalid_log_s = 0.0
    next_console_status_s = 0.0
    previous_state: np.ndarray | None = None

    while (time.perf_counter() - started) <= float(config.launch_wait_timeout_s):
        elapsed_s = time.perf_counter() - started
        sample, status = vicon.read_latest()
        if sample is None or not status.valid:
            latest_gate_reason = status.reason
            if elapsed_s + 1e-12 >= next_console_status_s:
                print(
                    f"[LAUNCH_WAIT] t={elapsed_s:.1f}s "
                    f"Vicon invalid: {status.reason}; holding neutral"
                )
                next_console_status_s = elapsed_s + 1.0
            next_serial_s = _send_neutral_if_due(
                tx=tx,
                controller=controller,
                logger=logger,
                summary=summary,
                elapsed_s=elapsed_s,
                next_serial_s=next_serial_s,
                serial_period_s=float(config.serial_period_s),
            )
            if elapsed_s + 1e-12 >= next_invalid_log_s:
                logger.append_metric_row(
                    "prelaunch_events.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        "event": "vicon_invalid_waiting_for_launch",
                        "reason": status.reason,
                    },
                )
                next_invalid_log_s = elapsed_s + 0.50
            if mode in {"armed", "vicon-smoke"}:
                time.sleep(float(config.vicon_poll_period_s))
            continue

        state = adapter.update(sample, command_norm=controller.last_command_norm())
        estimator = adapter.estimator_status()
        _append_fan_positions(
            logger=logger,
            vicon=vicon,
            adapter=adapter,
            phase="prelaunch",
            expected_visible_fan_range=expected_visible_fan_range,
            summary=summary,
        )
        full_window_gate = evaluate_launch_gate(state)
        launch_plane_state = interpolate_launch_plane_state(previous_state, state)
        crossed_launch_plane = launch_plane_state is not None
        plane_gate = (
            evaluate_launch_plane_gate(launch_plane_state)
            if crossed_launch_plane and launch_plane_state is not None
            else None
        )
        if full_window_gate.approved:
            gate = full_window_gate
            trigger_approved = True
            trigger_source = "valid_r5_launch_window"
            launch_state = state
        elif plane_gate is not None:
            gate = plane_gate
            trigger_approved = bool(plane_gate.approved)
            trigger_source = "valid_interpolated_launch_plane" if trigger_approved else "crossed_plane_rejected"
            launch_state = launch_plane_state if trigger_approved else None
        else:
            gate = full_window_gate
            trigger_approved = False
            trigger_source = "tracking"
            launch_state = None
        latest_gate_reason = gate.reason
        consecutive_approved = consecutive_approved + 1 if trigger_approved else 0
        if elapsed_s + 1e-12 >= next_console_status_s:
            speed_m_s = float(np.linalg.norm(state[6:9]))
            print(
                f"[LAUNCH_WAIT] t={elapsed_s:.1f}s trigger={trigger_source} gate={gate.reason} "
                f"approved={trigger_approved} consecutive={consecutive_approved}/{required_consecutive} "
                f"x={state[0]:.2f} y={state[1]:.2f} z={state[2]:.2f} speed={speed_m_s:.2f}m/s"
            )
            next_console_status_s = elapsed_s + 1.0
        logger.append_metric_row(
            "prelaunch_state_samples.csv",
            {
                "t_host_s": time.perf_counter(),
                "frame_number": status.frame_number,
                "vicon_frame_rate_hz": status.frame_rate_hz,
                "vicon_latency_s": status.vicon_latency_s,
                **{f"estimator_{key}": value for key, value in estimator.items()},
                "launch_gate_consecutive_approved": consecutive_approved,
                "trigger_policy": "positive_x_launch_plane_crossing",
                "launch_trigger_x_w_m": float(LAUNCH_TRIGGER_X_W_M),
                "crossed_launch_plane": bool(crossed_launch_plane),
                "trigger_approved": bool(trigger_approved),
                "trigger_source": trigger_source,
                "full_window_reason": full_window_gate.reason,
                "full_window_approved": bool(full_window_gate.approved),
                "interpolated_plane_reason": plane_gate.reason if plane_gate is not None else "",
                "interpolated_plane_approved": bool(plane_gate.approved) if plane_gate is not None else False,
                "diagnostic_box_reason": full_window_gate.reason,
                "diagnostic_box_approved": bool(full_window_gate.approved),
                **asdict(gate),
                **state_dataframe_row(state),
            },
        )

        next_serial_s = _send_neutral_if_due(
            tx=tx,
            controller=controller,
            logger=logger,
            summary=summary,
            elapsed_s=elapsed_s,
            next_serial_s=next_serial_s,
            serial_period_s=float(config.serial_period_s),
        )

        if trigger_approved and consecutive_approved >= required_consecutive:
            summary["launch_gate_approved"] = True
            _append_runtime_event(
                logger,
                "launch_gate_approved_start_active_record",
                required_consecutive_frames=required_consecutive,
                launch_trigger_x_w_m=float(LAUNCH_TRIGGER_X_W_M),
                trigger_policy="first_valid_r5_launch_window_with_interpolated_plane_fallback",
                trigger_source=trigger_source,
                **asdict(gate),
            )
            return launch_state.copy() if launch_state is not None else state.copy()

        previous_state = state.copy()

        if mode in {"armed", "vicon-smoke"}:
            time.sleep(float(config.vicon_poll_period_s))

    summary["cancellation_reason"] = f"launch_gate_timeout:{latest_gate_reason}"
    _append_runtime_event(
        logger,
        "flight_record_cancelled_before_launch",
        reason=summary["cancellation_reason"],
        launch_wait_timeout_s=float(config.launch_wait_timeout_s),
    )
    return None


def _append_fan_positions(
    *,
    logger: FlightLogger,
    vicon: LiveNausicaaViconRigidBody | ReplayNausicaaViconRigidBody,
    adapter: NausicaaViconStateAdapter,
    phase: str,
    expected_visible_fan_range: tuple[int, int] | None,
    summary: dict[str, object],
) -> None:
    if not hasattr(vicon, "read_fans"):
        return
    try:
        fans = vicon.read_fans()
    except Exception as exc:
        logger.append_metric_row(
            "fan_positions.csv",
            {
                "t_host_s": time.perf_counter(),
                "phase": phase,
                "fan_subject": "",
                "visible": False,
                "reason": f"fan_tracker_failed:{type(exc).__name__}",
                "x_w": "",
                "y_w": "",
                "z_w": "",
                "visible_count": 0,
                "expected_count_ok": False,
            },
        )
        return
    visible_count = sum(1 for fan in fans if fan.visible)
    if expected_visible_fan_range is None:
        expected_ok = True
    else:
        expected_ok = int(expected_visible_fan_range[0]) <= visible_count <= int(expected_visible_fan_range[1])
    summary["fan_visible_count_latest"] = int(visible_count)
    summary["fan_expected_count_ok_latest"] = bool(expected_ok)
    for fan in fans:
        world = ("", "", "")
        if fan.visible and fan.position_m is not None:
            position = adapter.arena_transform.position_to_world(np.asarray(fan.position_m, dtype=float))
            world = tuple(float(value) for value in position)
        logger.append_metric_row(
            "fan_positions.csv",
            {
                "t_host_s": time.perf_counter(),
                "phase": phase,
                "fan_subject": fan.subject_name,
                "visible": bool(fan.visible),
                "reason": fan.reason,
                "x_w": world[0],
                "y_w": world[1],
                "z_w": world[2],
                "visible_count": int(visible_count),
                "expected_count_ok": bool(expected_ok),
            },
        )


def _send_neutral_if_due(
    *,
    tx: NanoSerialTx | FakeNanoSerialTx,
    controller: FrozenFlightController,
    logger: FlightLogger,
    summary: dict[str, object],
    elapsed_s: float,
    next_serial_s: float,
    serial_period_s: float,
) -> float:
    if elapsed_s + 1e-12 < float(next_serial_s):
        return float(next_serial_s)
    if _write_packet_safe(
        tx,
        controller.neutral_packet(),
        logger=logger,
        summary=summary,
        event="launch_wait_neutral_packet",
    ):
        summary["neutral_failsafe_count"] = int(summary["neutral_failsafe_count"]) + 1
    return float(next_serial_s) + float(serial_period_s)


def _send_neutral_tail(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    controller: FrozenFlightController,
    logger: FlightLogger,
    mode: str,
    summary: dict[str, object],
    reason: str,
) -> None:
    packet_count = max(1, int(np.ceil(float(config.post_exit_neutral_tail_s) / float(config.serial_period_s))))
    for _ in range(packet_count):
        if _write_packet_safe(
            tx,
            controller.neutral_packet(),
            logger=logger,
            summary=summary,
            event="post_exit_neutral_tail_packet",
        ):
            summary["neutral_failsafe_count"] = int(summary["neutral_failsafe_count"]) + 1
            summary["post_exit_neutral_packets"] = int(summary["post_exit_neutral_packets"]) + 1
        if mode in {"armed", "vicon-smoke"}:
            time.sleep(float(config.serial_period_s))
    _append_runtime_event(
        logger,
        "post_exit_neutral_tail_sent",
        reason=str(reason),
        neutral_packet_count=packet_count,
        post_exit_neutral_tail_s=float(config.post_exit_neutral_tail_s),
    )


def _append_runtime_event(logger: FlightLogger, event: str, **details: object) -> None:
    logger.append_metric_row(
        "runtime_events.csv",
        {
            "t_host_s": time.perf_counter(),
            "event": str(event),
            "details_json": json.dumps(details, sort_keys=True, separators=(",", ":")),
        },
    )


def _sleep_until_next_runtime_poll(
    *,
    config: FlightRuntimeConfig,
    started: float,
    next_serial_s: float,
    next_governor_s: float,
) -> None:
    elapsed_s = time.perf_counter() - float(started)
    waits = [float(config.vicon_poll_period_s)]
    for next_event_s in (float(next_serial_s), float(next_governor_s)):
        wait_s = next_event_s - elapsed_s
        if wait_s > 0.0:
            waits.append(wait_s)
    time.sleep(max(0.0, min(waits)))


def _write_packet_safe(
    tx: NanoSerialTx | FakeNanoSerialTx,
    packet: bytes,
    *,
    logger: FlightLogger,
    summary: dict[str, object],
    event: str,
    quiet: bool = False,
) -> bool:
    try:
        tx.write_packet(packet)
    except Exception as exc:
        summary["serial_write_error_count"] = int(summary.get("serial_write_error_count", 0)) + 1
        if "Timeout" in type(exc).__name__:
            summary["serial_write_timeout_count"] = int(summary.get("serial_write_timeout_count", 0)) + 1
        _append_runtime_event(
            logger,
            "serial_write_failed",
            source_event=str(event),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if not quiet:
            print(f"[SERIAL] write failed during {event}: {type(exc).__name__}: {exc}")
        return False
    summary["packet_count"] = int(summary["packet_count"]) + 1
    return True


def _run_packet_smoke(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    controller: FrozenFlightController,
    logger: FlightLogger,
) -> None:
    del config
    commands = (
        np.zeros(3),
        np.asarray([0.2, 0.0, 0.0], dtype=float),
        np.asarray([-0.2, 0.0, 0.0], dtype=float),
        np.zeros(3),
    )
    for index, command in enumerate(commands):
        controller._last_command_norm = command.copy()  # packet smoke deliberately bypasses selection.
        packet = controller.packet_for_last_command()
        tx.write_packet(packet)
        logger.append_metric_row(
            "packet_smoke.csv",
            {
                "packet_index": index,
                "command_norm": tuple(float(value) for value in command),
                "packet_hex": packet.hex(),
            },
        )
        time.sleep(0.20)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-contained Nausicaa real-flight controller runtime.")
    parser.add_argument("--mode", choices=("dry-run", "packet-smoke", "vicon-smoke", "armed"), default="dry-run")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--library-tier", choices=("balanced_cluster", "heavy_cluster"), default=DEFAULT_REAL_FLIGHT_LIBRARY_TIER)
    parser.add_argument("--serial-port", default="COM11")
    parser.add_argument("--vicon-host", default="192.168.0.100:801")
    parser.add_argument("--vicon-offset-m", nargs=3, type=float, default=None)
    parser.add_argument("--vicon-yaw-deg", type=float, default=0.0)
    parser.add_argument("--vicon-attitude-signs", nargs=3, type=float, default=DEFAULT_VICON_ATTITUDE_SIGNS)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--launch-wait-timeout-s", type=float, default=8.0)
    parser.add_argument("--launch-gate-frames", type=int, default=1)
    parser.add_argument("--post-exit-neutral-tail-s", type=float, default=0.30)
    parser.add_argument("--vicon-tracking-rate-hz", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = FlightRuntimeConfig(
        run_label=args.run_label or default_run_label(),
        library_tier=args.library_tier,
        serial_port=args.serial_port,
        vicon_host=args.vicon_host,
        vicon_position_offset_m=tuple(args.vicon_offset_m) if args.vicon_offset_m is not None else DEFAULT_VICON_POSITION_OFFSET_M,
        vicon_yaw_alignment_deg=float(args.vicon_yaw_deg),
        vicon_attitude_signs=tuple(args.vicon_attitude_signs),
        max_duration_s=float(args.duration_s),
        launch_wait_timeout_s=float(args.launch_wait_timeout_s),
        launch_gate_required_consecutive_frames=int(args.launch_gate_frames),
        post_exit_neutral_tail_s=float(args.post_exit_neutral_tail_s),
        vicon_poll_period_s=1.0 / float(args.vicon_tracking_rate_hz),
    )
    summary = run_real_flight(config, mode=str(args.mode))
    print(f"run_root={summary['run_root']}")
    print(f"completed={summary['completed']}")
    print(f"controller_decisions={summary['controller_decision_count']}")
    print(f"packets={summary['packet_count']}")


if __name__ == "__main__":
    main()
