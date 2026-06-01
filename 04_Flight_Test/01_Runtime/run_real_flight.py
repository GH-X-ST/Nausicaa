from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from flight_config import CONTROLLER_ROOT, FlightRuntimeConfig, default_run_label
from flight_logger import FlightLogger
from frozen_flight_controller import FrozenFlightController
from nano_serial import FakeNanoSerialTx, NanoSerialTx
from safety_monitor import evaluate_safety
from vicon_rigid_body import LiveNausicaaViconRigidBody, ReplayNausicaaViconRigidBody

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import NausicaaViconStateAdapter  # noqa: E402
from state_contract import state_dataframe_row  # noqa: E402


def run_real_flight(config: FlightRuntimeConfig, *, mode: str) -> dict[str, object]:
    logger = FlightLogger(config.run_root)
    controller = FrozenFlightController(config)
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=config.derivative_cutoff_hz,
        actuator_tau_s=config.actuator_tau_s,
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
            "control_boundary": "vicon_rigid_body_to_canonical_state_to_frozen_governor_to_quantised_packet",
            "surface_marker_tracking_enabled": False,
            "latency_quantification_enabled": False,
            "servo_command_limit_norm": [-1.0, 1.0],
        },
    )
    summary = {
        "mode": mode,
        "run_root": config.run_root.as_posix(),
        "state_sample_count": 0,
        "controller_decision_count": 0,
        "packet_count": 0,
        "neutral_failsafe_count": 0,
        "max_decision_time_s": 0.0,
        "completed": False,
    }
    primitive_step_index = 0
    latest_decision = None
    latest_state = None
    started = time.perf_counter()
    next_governor_s = 0.0
    next_serial_s = 0.0

    try:
        tx.open()
        vicon.open()
        if mode == "packet-smoke":
            _run_packet_smoke(config=config, tx=tx, controller=controller, logger=logger)
            summary["completed"] = True
            return summary

        while (time.perf_counter() - started) <= float(config.max_duration_s):
            loop_elapsed_s = time.perf_counter() - started
            sample, status = vicon.read_latest()
            if sample is None or not status.valid:
                packet = controller.neutral_packet()
                tx.write_packet(packet)
                summary["packet_count"] += 1
                summary["neutral_failsafe_count"] += 1
                logger.append_metric_row(
                    "runtime_events.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        "event": "vicon_invalid_neutral_command",
                        "reason": status.reason,
                    },
                )
                if mode == "armed":
                    time.sleep(config.serial_period_s)
                continue

            latest_state = adapter.update(sample, command_norm=controller.last_command_norm())
            safety = evaluate_safety(latest_state)
            summary["state_sample_count"] += 1
            logger.append_metric_row(
                "state_samples.csv",
                {
                    "t_host_s": time.perf_counter(),
                    "frame_number": status.frame_number,
                    "vicon_latency_s": status.vicon_latency_s,
                    **state_dataframe_row(latest_state),
                    **asdict(safety),
                },
            )
            if not safety.safe:
                packet = controller.neutral_packet()
                tx.write_packet(packet)
                summary["packet_count"] += 1
                summary["neutral_failsafe_count"] += 1
                logger.append_metric_row(
                    "runtime_events.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        "event": "safety_neutral_command",
                        "reason": safety.reason,
                    },
                )
                break

            if loop_elapsed_s + 1e-12 >= next_governor_s:
                latest_decision = controller.decide(latest_state, primitive_step_index=primitive_step_index)
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
                f"- Run root: `{config.run_root.as_posix()}`",
                f"- State samples: `{summary['state_sample_count']}`",
                f"- Controller decisions: `{summary['controller_decision_count']}`",
                f"- Packets sent: `{summary['packet_count']}`",
                f"- Neutral failsafe commands: `{summary['neutral_failsafe_count']}`",
                f"- Max decision time (s): `{float(summary['max_decision_time_s']):.6f}`",
            ],
        )
        logger.close()


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
    parser.add_argument("--library-tier", choices=("balanced_cluster", "heavy_cluster"), default="balanced_cluster")
    parser.add_argument("--serial-port", default="COM11")
    parser.add_argument("--vicon-host", default="192.168.0.100:801")
    parser.add_argument("--duration-s", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = FlightRuntimeConfig(
        run_label=args.run_label or default_run_label(),
        library_tier=args.library_tier,
        serial_port=args.serial_port,
        vicon_host=args.vicon_host,
        max_duration_s=float(args.duration_s),
    )
    summary = run_real_flight(config, mode=str(args.mode))
    print(f"run_root={summary['run_root']}")
    print(f"completed={summary['completed']}")
    print(f"controller_decisions={summary['controller_decision_count']}")
    print(f"packets={summary['packet_count']}")


if __name__ == "__main__":
    main()
