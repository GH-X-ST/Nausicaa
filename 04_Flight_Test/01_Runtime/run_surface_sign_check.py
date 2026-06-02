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


COMMAND_LATTICE_20_PERCENT = tuple(round(-1.0 + 0.2 * index, 1) for index in range(11))
DEFAULT_MODE = "armed"
DEFAULT_SERIAL_PORT = "COM11"
DEFAULT_DWELL_S = 2.0
DEFAULT_PRINT_TELEMETRY = False
COMMAND_REPEAT_PERIOD_S = 0.020
LIVE_PRINT_PERIOD_S = 0.25


def _expected_motion(axis_name: str, value: float) -> str:
    command = float(value)
    if abs(command) < 1e-12:
        return "All relevant surfaces should be neutral for this axis step."
    if axis_name == "aileron":
        if command > 0.0:
            return "Viewed from behind: left aileron trailing edge UP, right aileron trailing edge DOWN."
        return "Viewed from behind: left aileron trailing edge DOWN, right aileron trailing edge UP."
    if axis_name == "elevator":
        if command > 0.0:
            return "Elevator trailing edge UP; this is the nose-up pitch command."
        return "Elevator trailing edge DOWN; this is the nose-down pitch command."
    if axis_name == "rudder":
        if command > 0.0:
            return "Viewed from behind: rudder trailing edge RIGHT; this is the nose-right yaw command."
        return "Viewed from behind: rudder trailing edge LEFT; this is the nose-left yaw command."
    return "Unknown axis; stop and inspect the script."


def _axis_command(axis_name: str, value: float) -> tuple[float, float, float]:
    command = float(value)
    if axis_name == "aileron":
        return (command, 0.0, 0.0)
    if axis_name == "elevator":
        return (0.0, command, 0.0)
    if axis_name == "rudder":
        return (0.0, 0.0, command)
    raise ValueError(f"unknown axis_name: {axis_name}")


def _build_surface_check_sequence() -> tuple[tuple[str, tuple[float, float, float], str], ...]:
    sequence: list[tuple[str, tuple[float, float, float], str]] = [
        ("neutral_start", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."),
    ]
    for axis_name in ("aileron", "elevator", "rudder"):
        sequence.append((f"{axis_name}_neutral_before_sweep", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."))
        for command_value in COMMAND_LATTICE_20_PERCENT:
            step_name = f"{axis_name}_{command_value:+.1f}"
            sequence.append((step_name, _axis_command(axis_name, command_value), _expected_motion(axis_name, command_value)))
        sequence.append((f"{axis_name}_neutral_after_sweep", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."))
    sequence.append(("neutral_end", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."))
    return tuple(sequence)


SURFACE_CHECK_SEQUENCE = _build_surface_check_sequence()


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
        print(
            f"[CONFIG] dwell_s={float(dwell_s):.2f}, repeat_period_s={COMMAND_REPEAT_PERIOD_S:.3f}, "
            f"telemetry_print={print_telemetry}, lattice={COMMAND_LATTICE_20_PERCENT}"
        )
        tx.open()
        for text_command, wait_s in (("HELLO", 0.75), ("SET_NEUTRAL", 0.25)):
            print(f"[SETUP] {text_command}")
            tx.write_line(text_command)
            packet_count += 1
            local_telemetry = _drain_telemetry(tx, logger, wait_s, label=text_command, print_telemetry=print_telemetry)
            telemetry_count += local_telemetry
            print(f"        telemetry_lines={local_telemetry}")
        for sequence, (step_name, command, expected_motion) in enumerate(SURFACE_CHECK_SEQUENCE):
            print(f"[{sequence:02d}/{len(SURFACE_CHECK_SEQUENCE) - 1:02d}] {step_name}: command={command}")
            print(f"     EXPECTED: {expected_motion}")
            started = time.perf_counter()
            local_packets = 0
            local_telemetry = 0
            next_live_print_s = 0.0
            packet = encode_arduino_command_packet(np.asarray(command, dtype=float), sequence=sequence)
            dwell_duration_s = max(0.0, float(dwell_s))
            while True:
                elapsed_s = time.perf_counter() - started
                if local_packets > 0 and elapsed_s >= dwell_duration_s:
                    break
                if elapsed_s + 1e-12 >= next_live_print_s:
                    print(
                        f"     LIVE: t={elapsed_s:.2f}/{float(dwell_s):.2f}s "
                        f"cmd=({command[0]:+.1f},{command[1]:+.1f},{command[2]:+.1f}) "
                        f"packets={local_packets}"
                    )
                    next_live_print_s = elapsed_s + LIVE_PRINT_PERIOD_S
                packet = encode_arduino_command_packet(np.asarray(command, dtype=float), sequence=sequence * 1000 + local_packets)
                tx.write_packet(packet.packet_bytes)
                packet_count += 1
                local_packets += 1
                drained = _drain_telemetry(tx, logger, 0.0, label=step_name, print_telemetry=print_telemetry)
                telemetry_count += drained
                local_telemetry += drained
                sleep_s = COMMAND_REPEAT_PERIOD_S if mode == "armed" else min(COMMAND_REPEAT_PERIOD_S, dwell_duration_s)
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
