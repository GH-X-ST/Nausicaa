from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from flight_config import CONTROLLER_ROOT, RESULT_ROOT, default_run_label
from flight_logger import FlightLogger
from nano_serial import FakeNanoSerialTx, NanoSerialTx

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import encode_arduino_command_packet  # noqa: E402


SURFACE_CHECK_SEQUENCE = (
    ("neutral", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."),
    ("positive_aileron", (0.4, 0.0, 0.0), "Viewed from behind the glider: left aileron trailing edge UP, right aileron trailing edge DOWN."),
    ("negative_aileron", (-0.4, 0.0, 0.0), "Viewed from behind the glider: left aileron trailing edge DOWN, right aileron trailing edge UP."),
    ("positive_elevator", (0.0, 0.4, 0.0), "Elevator trailing edge UP; this is the nose-up pitch command."),
    ("negative_elevator", (0.0, -0.4, 0.0), "Elevator trailing edge DOWN; this is the nose-down pitch command."),
    ("positive_rudder", (0.0, 0.0, 0.4), "Viewed from behind the glider: rudder trailing edge RIGHT; this is the nose-right yaw command."),
    ("negative_rudder", (0.0, 0.0, -0.4), "Viewed from behind the glider: rudder trailing edge LEFT; this is the nose-left yaw command."),
    ("neutral_end", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."),
)
DEFAULT_MODE = "armed"
DEFAULT_SERIAL_PORT = "COM11"
DEFAULT_DWELL_S = 2.0
DEFAULT_PRINT_TELEMETRY = False
COMMAND_REPEAT_PERIOD_S = 0.020


def run_surface_sign_check(
    *,
    serial_port: str,
    mode: str,
    dwell_s: float,
    run_label: str,
    print_telemetry: bool = DEFAULT_PRINT_TELEMETRY,
) -> dict[str, object]:
    logger = FlightLogger(RESULT_ROOT / "surface_sign_check" / run_label)
    tx = NanoSerialTx(serial_port, 1_000_000) if mode == "armed" else FakeNanoSerialTx()
    packet_count = 0
    telemetry_count = 0
    try:
        if mode == "armed":
            print(f"[ARMED] Hardware output enabled on {serial_port}. Ctrl+C stops the check; neutral is sent on exit.")
        else:
            print("[DRY-RUN] No hardware commands will be sent.")
        print(f"[CONFIG] dwell_s={float(dwell_s):.2f}, repeat_period_s={COMMAND_REPEAT_PERIOD_S:.3f}, telemetry_print={print_telemetry}")
        tx.open()
        for text_command, wait_s in (("HELLO", 0.75), ("SET_NEUTRAL", 0.25)):
            print(f"[SETUP] {text_command}")
            tx.write_line(text_command)
            packet_count += 1
            local_telemetry = _drain_telemetry(tx, logger, wait_s, label=text_command, print_telemetry=print_telemetry)
            telemetry_count += local_telemetry
            print(f"        telemetry_lines={local_telemetry}")
        for sequence, (step_name, command, expected_motion) in enumerate(SURFACE_CHECK_SEQUENCE):
            print(f"[{sequence:02d}] {step_name}: command={command}")
            print(f"     EXPECTED: {expected_motion}")
            started = time.perf_counter()
            local_packets = 0
            local_telemetry = 0
            packet = encode_arduino_command_packet(np.asarray(command, dtype=float), sequence=sequence)
            while time.perf_counter() - started <= max(0.0, float(dwell_s)):
                packet = encode_arduino_command_packet(np.asarray(command, dtype=float), sequence=sequence * 1000 + local_packets)
                tx.write_packet(packet.packet_bytes)
                packet_count += 1
                local_packets += 1
                drained = _drain_telemetry(tx, logger, 0.0, label=step_name, print_telemetry=print_telemetry)
                telemetry_count += drained
                local_telemetry += drained
                sleep_s = COMMAND_REPEAT_PERIOD_S if mode == "armed" else min(COMMAND_REPEAT_PERIOD_S, max(0.0, float(dwell_s)))
                if sleep_s <= 0.0:
                    break
                time.sleep(sleep_s)
            print(f"     SENT: packets={local_packets}, telemetry_lines={local_telemetry}")
            logger.append_metric_row(
                "surface_sign_check.csv",
                {
                    "sequence": sequence,
                    "step_name": step_name,
                    "delta_a_norm": float(command[0]),
                    "delta_e_norm": float(command[1]),
                    "delta_r_norm": float(command[2]),
                    "physical_surface_norm": packet.physical_surface_norm,
                    "packet_surface_norm": packet.packet_surface_norm,
                    "receiver_channel_codes": packet.receiver_channel_codes,
                    "packet_hex": packet.packet_bytes.hex(),
                    "repeat_packet_count": local_packets,
                    "expected_motion": expected_motion,
                },
            )
    finally:
        try:
            neutral = encode_arduino_command_packet(np.zeros(3), sequence=999)
            for index in range(10):
                neutral = encode_arduino_command_packet(np.zeros(3), sequence=9990 + index)
                tx.write_packet(neutral.packet_bytes)
                packet_count += 1
                time.sleep(COMMAND_REPEAT_PERIOD_S if mode == "armed" else 0.0)
            tx.write_line("SET_NEUTRAL")
            packet_count += 1
        except Exception:
            pass
        tx.close()
        logger.write_manifest(
            "surface_sign_check_manifest.json",
            {
                "mode": mode,
                "serial_port": serial_port,
                "run_label": run_label,
                "packet_count": packet_count,
                "telemetry_count": telemetry_count,
                "command_repeat_period_s": COMMAND_REPEAT_PERIOD_S,
                "print_telemetry": print_telemetry,
                "firmware_timeout_awareness": "commands_are_repeated_during_dwell_to_avoid_250ms_firmware_timeout",
                "authority": "full_normalised_minus1_to_plus1_with_20_percent_lattice",
            },
        )
        logger.close()
    print("[DONE] Surface sign check finished; neutral command sent.")
    return {"run_root": (RESULT_ROOT / "surface_sign_check" / run_label).as_posix(), "packet_count": packet_count}


def _drain_telemetry(
    tx: NanoSerialTx | FakeNanoSerialTx,
    logger: FlightLogger,
    wait_s: float,
    *,
    label: str,
    print_telemetry: bool,
) -> int:
    if isinstance(tx, FakeNanoSerialTx):
        return 0
    started = time.perf_counter()
    count = 0
    while time.perf_counter() - started <= max(0.0, float(wait_s)):
        line = tx.read_telemetry_line()
        if line:
            if print_telemetry:
                print(f"     TELEMETRY: {line}")
            logger.append_metric_row(
                "serial_telemetry.csv",
                {
                    "t_host_s": time.perf_counter(),
                    "label": label,
                    "line_text": line,
                },
            )
            count += 1
        if wait_s <= 0.0:
            break
        time.sleep(0.01)
    if wait_s <= 0.0:
        line = tx.read_telemetry_line()
        if line:
            if print_telemetry:
                print(f"     TELEMETRY: {line}")
            logger.append_metric_row(
                "serial_telemetry.csv",
                {
                    "t_host_s": time.perf_counter(),
                    "label": label,
                    "line_text": line,
                },
            )
            count += 1
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive each real-flight surface command to verify servo signs.")
    parser.add_argument("--mode", choices=("dry-run", "armed"), default=DEFAULT_MODE)
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--dwell-s", type=float, default=DEFAULT_DWELL_S)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--print-telemetry", action="store_true", default=DEFAULT_PRINT_TELEMETRY)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_surface_sign_check(
        serial_port=args.serial_port,
        mode=args.mode,
        dwell_s=args.dwell_s,
        run_label=args.run_label or default_run_label("surface_sign_check"),
        print_telemetry=args.print_telemetry,
    )


if __name__ == "__main__":
    main()
