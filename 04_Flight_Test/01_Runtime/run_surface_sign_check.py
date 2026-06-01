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
    ("positive_aileron", (0.4, 0.0, 0.0), "Ailerons should move in opposition for positive roll: left aileron up, right aileron down if the airframe convention matches the model."),
    ("negative_aileron", (-0.4, 0.0, 0.0), "Ailerons should move in the opposite direction from the positive-aileron step."),
    ("positive_elevator", (0.0, 0.4, 0.0), "Elevator should deflect for nose-up positive pitch command."),
    ("negative_elevator", (0.0, -0.4, 0.0), "Elevator should deflect opposite to the positive-elevator step."),
    ("positive_rudder", (0.0, 0.0, 0.4), "Rudder should deflect for nose-right positive yaw command."),
    ("negative_rudder", (0.0, 0.0, -0.4), "Rudder should deflect opposite to the positive-rudder step."),
    ("neutral_end", (0.0, 0.0, 0.0), "All control surfaces should return to neutral."),
)


def run_surface_sign_check(
    *,
    serial_port: str,
    mode: str,
    dwell_s: float,
    run_label: str,
) -> dict[str, object]:
    logger = FlightLogger(RESULT_ROOT / "surface_sign_check" / run_label)
    tx = NanoSerialTx(serial_port, 1_000_000) if mode == "armed" else FakeNanoSerialTx()
    packet_count = 0
    try:
        tx.open()
        for sequence, (step_name, command, expected_motion) in enumerate(SURFACE_CHECK_SEQUENCE):
            packet = encode_arduino_command_packet(np.asarray(command, dtype=float), sequence=sequence)
            print(f"[{sequence:02d}] {step_name}: command={command}")
            print(f"     EXPECTED: {expected_motion}")
            tx.write_packet(packet.packet_bytes)
            packet_count += 1
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
                    "expected_motion": expected_motion,
                },
            )
            time.sleep(max(0.0, float(dwell_s)))
    finally:
        try:
            neutral = encode_arduino_command_packet(np.zeros(3), sequence=999)
            tx.write_packet(neutral.packet_bytes)
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
                "authority": "full_normalised_minus1_to_plus1_with_20_percent_lattice",
            },
        )
        logger.close()
    print("[DONE] Surface sign check finished; neutral command sent.")
    return {"run_root": (RESULT_ROOT / "surface_sign_check" / run_label).as_posix(), "packet_count": packet_count}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive each real-flight surface command to verify servo signs.")
    parser.add_argument("--mode", choices=("dry-run", "armed"), default="dry-run")
    parser.add_argument("--serial-port", default="COM11")
    parser.add_argument("--dwell-s", type=float, default=1.5)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_surface_sign_check(
        serial_port=args.serial_port,
        mode=args.mode,
        dwell_s=args.dwell_s,
        run_label=args.run_label or default_run_label("surface_sign_check"),
    )


if __name__ == "__main__":
    main()
