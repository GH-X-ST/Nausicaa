from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class SerialLike(Protocol):
    def write(self, data: bytes) -> int: ...
    def readline(self) -> bytes: ...
    def close(self) -> None: ...


@dataclass
class NanoSerialTx:
    port: str
    baud: int = 1_000_000
    timeout_s: float = 0.020
    _serial: SerialLike | None = field(default=None, init=False, repr=False)

    def open(self) -> "NanoSerialTx":
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for hardware mode. Install with: "
                "C:\\ProgramData\\miniforge3\\python.exe -m pip install pyserial"
            ) from exc
        self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout_s, write_timeout=self.timeout_s)
        time.sleep(2.0)
        return self

    def write_packet(self, packet: bytes) -> int:
        if self._serial is None:
            raise RuntimeError("serial transmitter is not open.")
        return int(self._serial.write(packet))

    def write_line(self, text: str) -> int:
        if self._serial is None:
            raise RuntimeError("serial transmitter is not open.")
        return int(self._serial.write((str(text) + "\n").encode("ascii")))

    def read_telemetry_line(self) -> str:
        if self._serial is None:
            return ""
        try:
            return self._serial.readline().decode("ascii", errors="replace").strip()
        except Exception:
            return ""

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


@dataclass
class FakeNanoSerialTx:
    packets: list[bytes] = field(default_factory=list)
    closed: bool = False

    def open(self) -> "FakeNanoSerialTx":
        return self

    def write_packet(self, packet: bytes) -> int:
        self.packets.append(bytes(packet))
        return len(packet)

    def write_line(self, text: str) -> int:
        payload = (str(text) + "\n").encode("ascii")
        self.packets.append(payload)
        return len(payload)

    def read_telemetry_line(self) -> str:
        return ""

    def close(self) -> None:
        self.closed = True
