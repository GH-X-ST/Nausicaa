from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from flight_config import CONTROLLER_ROOT

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from state_contract import STATE_INDEX, as_state_vector  # noqa: E402


@dataclass(frozen=True)
class SafetyStatus:
    safe: bool
    reason: str
    min_margin_m: float
    floor_margin_m: float
    ceiling_margin_m: float


def evaluate_safety(state: np.ndarray) -> SafetyStatus:
    try:
        x = as_state_vector(state)
    except Exception as exc:
        return SafetyStatus(False, f"invalid_state:{type(exc).__name__}", -float("inf"), -float("inf"), -float("inf"))
    margins = position_margin_m(
        x[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]],
        TRUE_SAFE_BOUNDS,
    )
    min_margin = float(margins["min_margin_m"])
    if min_margin < 0.0:
        return SafetyStatus(
            False,
            "outside_true_safe_bounds",
            min_margin,
            float(margins["floor_margin_m"]),
            float(margins["ceiling_margin_m"]),
        )
    return SafetyStatus(
        True,
        "safe",
        min_margin,
        float(margins["floor_margin_m"]),
        float(margins["ceiling_margin_m"]),
    )
