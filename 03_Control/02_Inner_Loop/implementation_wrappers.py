from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImplementationWrapperConfig:
    name: str = "deterministic_direct_servo_proxy"
    command_limit_rad: tuple[float, float, float] = (
        np.deg2rad(25.0),
        np.deg2rad(25.0),
        np.deg2rad(25.0),
    )
    deadband_rad: tuple[float, float, float] = (
        np.deg2rad(0.15),
        np.deg2rad(0.15),
        np.deg2rad(0.15),
    )
    quantization_rad: tuple[float, float, float] = (
        np.deg2rad(0.25),
        np.deg2rad(0.25),
        np.deg2rad(0.25),
    )
    extra_delay_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ImplementationCommandWrapper:
    def __init__(self, config: ImplementationWrapperConfig | None = None):
        self.config = config or ImplementationWrapperConfig()
        self._previous = np.zeros(3)
        self._queues: list[deque[float]] = [deque(), deque(), deque()]

    def reset(
        self,
        dt_s: float,
        initial_command_rad: np.ndarray | list[float] | tuple[float, ...],
    ) -> None:
        initial = np.asarray(initial_command_rad, dtype=float).reshape(3)
        self._previous = initial.copy()
        self._queues = []
        for delay_s, value in zip(self.config.extra_delay_s, initial):
            steps = max(0, int(round(float(delay_s) / max(float(dt_s), 1e-12))))
            self._queues.append(deque([float(value)] * steps))

    def apply(self, command_rad: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        command = np.asarray(command_rad, dtype=float).reshape(3)
        limit = np.asarray(self.config.command_limit_rad, dtype=float).reshape(3)
        deadband = np.asarray(self.config.deadband_rad, dtype=float).reshape(3)
        quant = np.asarray(self.config.quantization_rad, dtype=float).reshape(3)

        clipped = np.clip(command, -limit, limit)
        small_move = np.abs(clipped - self._previous) < deadband
        held = clipped.copy()
        held[small_move] = self._previous[small_move]
        quant_safe = np.maximum(quant, 1e-12)
        quantized = np.round(held / quant_safe) * quant_safe

        delayed = np.empty(3, dtype=float)
        for idx, queue in enumerate(self._queues):
            queue.append(float(quantized[idx]))
            delayed[idx] = queue.popleft() if queue else float(quantized[idx])

        self._previous = delayed.copy()
        return delayed

    def summary(self) -> dict[str, object]:
        return {
            "name": self.config.name,
            "command_limit_deg": np.rad2deg(self.config.command_limit_rad).tolist(),
            "deadband_deg": np.rad2deg(self.config.deadband_rad).tolist(),
            "quantization_deg": np.rad2deg(self.config.quantization_rad).tolist(),
            "extra_delay_s": list(self.config.extra_delay_s),
        }
