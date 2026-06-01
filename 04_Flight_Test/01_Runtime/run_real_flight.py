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
    OPERATIONAL_REGION_CENTER_M,
    REAL_FLIGHT_LIBRARY_TIER_SELECTION_REASON,
    FlightRuntimeConfig,
    default_run_label,
)
from flight_logger import FlightLogger
from frozen_flight_controller import FrozenFlightController
from exit_gate import evaluate_exit_gate, exit_gate_bounds_manifest
from launch_gate import evaluate_launch_gate, launch_gate_bounds_manifest
from nano_serial import FakeNanoSerialTx, NanoSerialTx
from safety_monitor import evaluate_safety
from vicon_rigid_body import FanViconSample, LiveNausicaaViconRigidBody, ReplayNausicaaViconRigidBody

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import NausicaaViconStateAdapter, ViconArenaFrameTransform  # noqa: E402
from state_contract import state_dataframe_row  # noqa: E402


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
        actuator_tau_s=config.actuator_tau_s,
        arena_transform=ViconArenaFrameTransform(
            position_offset_m=config.vicon_position_offset_m,
            yaw_alignment_rad=float(np.deg2rad(config.vicon_yaw_alignment_deg)),
        ),
    )
    tx = NanoSerialTx(config.serial_port, config.serial_baud) if mode in {"armed", "packet-smoke"} else FakeNanoSerialTx()
    vicon = (
        LiveNausicaaViconRigidBody(host=config.vicon_host, subject_name=config.vicon_subject_name)
        if mode in {"armed", "vicon-smoke"}
        else ReplayNausicaaViconRigidBody(dt_s=config.serial_period_s)
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
            "vicon_arena_frame_transform": {
                "description": config.vicon_frame_description,
                "position_offset_m": tuple(float(value) for value in config.vicon_position_offset_m),
                "yaw_alignment_deg": float(config.vicon_yaw_alignment_deg),
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
            tx.write_packet(controller.neutral_packet())
            summary["packet_count"] += 1
            summary["neutral_failsafe_count"] += 1
            summary["flight_cancelled"] = True
            summary["completed"] = False
            return summary
        latest_state = launched_state
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
                tx.write_packet(packet)
                summary["packet_count"] += 1
                summary["neutral_failsafe_count"] += 1
                _append_runtime_event(
                    logger,
                    "vicon_invalid_neutral_command",
                    reason=status.reason,
                )
                if mode == "armed":
                    time.sleep(config.serial_period_s)
                continue

            latest_state = adapter.update(sample, command_norm=controller.last_command_norm())
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
                    "vicon_latency_s": status.vicon_latency_s,
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
                latest_decision = controller.decide(latest_state, primitive_step_index=primitive_step_index)
                decision_records.append(
                    {
                        "t_s": float(loop_elapsed_s),
                        "state": latest_state.copy(),
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
                tx.write_packet(packet)
                summary["packet_count"] += 1
                next_serial_s += float(config.serial_period_s)

            if mode in {"armed", "vicon-smoke"}:
                time.sleep(max(0.0, min(config.serial_period_s, next_serial_s - (time.perf_counter() - started))))

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
            try:
                tx.write_packet(controller.neutral_packet())
            except Exception:
                pass
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

    while (time.perf_counter() - started) <= float(config.launch_wait_timeout_s):
        elapsed_s = time.perf_counter() - started
        sample, status = vicon.read_latest()
        if sample is None or not status.valid:
            latest_gate_reason = status.reason
            next_serial_s = _send_neutral_if_due(
                tx=tx,
                controller=controller,
                summary=summary,
                elapsed_s=elapsed_s,
                next_serial_s=next_serial_s,
                serial_period_s=float(config.serial_period_s),
            )
            logger.append_metric_row(
                "prelaunch_events.csv",
                {
                    "t_host_s": time.perf_counter(),
                    "event": "vicon_invalid_waiting_for_launch",
                    "reason": status.reason,
                },
            )
            if mode == "armed":
                time.sleep(config.serial_period_s)
            continue

        state = adapter.update(sample, command_norm=controller.last_command_norm())
        _append_fan_positions(
            logger=logger,
            vicon=vicon,
            adapter=adapter,
            phase="prelaunch",
            expected_visible_fan_range=expected_visible_fan_range,
            summary=summary,
        )
        gate = evaluate_launch_gate(state)
        latest_gate_reason = gate.reason
        consecutive_approved = consecutive_approved + 1 if gate.approved else 0
        logger.append_metric_row(
            "prelaunch_state_samples.csv",
            {
                "t_host_s": time.perf_counter(),
                "frame_number": status.frame_number,
                "vicon_latency_s": status.vicon_latency_s,
                "launch_gate_consecutive_approved": consecutive_approved,
                **asdict(gate),
                **state_dataframe_row(state),
            },
        )

        next_serial_s = _send_neutral_if_due(
            tx=tx,
            controller=controller,
            summary=summary,
            elapsed_s=elapsed_s,
            next_serial_s=next_serial_s,
            serial_period_s=float(config.serial_period_s),
        )

        if consecutive_approved >= required_consecutive:
            summary["launch_gate_approved"] = True
            _append_runtime_event(
                logger,
                "launch_gate_approved_start_active_record",
                required_consecutive_frames=required_consecutive,
                **asdict(gate),
            )
            return state

        if mode == "armed":
            time.sleep(config.serial_period_s)

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
    summary: dict[str, object],
    elapsed_s: float,
    next_serial_s: float,
    serial_period_s: float,
) -> float:
    if elapsed_s + 1e-12 < float(next_serial_s):
        return float(next_serial_s)
    tx.write_packet(controller.neutral_packet())
    summary["packet_count"] = int(summary["packet_count"]) + 1
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
        tx.write_packet(controller.neutral_packet())
        summary["packet_count"] = int(summary["packet_count"]) + 1
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
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--launch-wait-timeout-s", type=float, default=8.0)
    parser.add_argument("--launch-gate-frames", type=int, default=1)
    parser.add_argument("--post-exit-neutral-tail-s", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = FlightRuntimeConfig(
        run_label=args.run_label or default_run_label(),
        library_tier=args.library_tier,
        serial_port=args.serial_port,
        vicon_host=args.vicon_host,
        vicon_position_offset_m=tuple(args.vicon_offset_m) if args.vicon_offset_m is not None else OPERATIONAL_REGION_CENTER_M,
        vicon_yaw_alignment_deg=float(args.vicon_yaw_deg),
        max_duration_s=float(args.duration_s),
        launch_wait_timeout_s=float(args.launch_wait_timeout_s),
        launch_gate_required_consecutive_frames=int(args.launch_gate_frames),
        post_exit_neutral_tail_s=float(args.post_exit_neutral_tail_s),
    )
    summary = run_real_flight(config, mode=str(args.mode))
    print(f"run_root={summary['run_root']}")
    print(f"completed={summary['completed']}")
    print(f"controller_decisions={summary['controller_decision_count']}")
    print(f"packets={summary['packet_count']}")


if __name__ == "__main__":
    main()
