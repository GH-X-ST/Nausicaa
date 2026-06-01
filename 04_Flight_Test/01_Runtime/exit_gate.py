from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flight_config import CONTROLLER_ROOT

import sys

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from state_contract import STATE_INDEX, as_state_vector  # noqa: E402


@dataclass(frozen=True)
class ExitGateStatus:
    inside: bool
    reason: str
    x_w_m: float
    y_w_m: float
    z_w_m: float
    min_margin_m: float
    floor_margin_m: float
    ceiling_margin_m: float


def evaluate_exit_gate(state: np.ndarray) -> ExitGateStatus:
    """Return whether the flight is still inside the validated operational box."""

    try:
        x = as_state_vector(state)
    except Exception as exc:
        return ExitGateStatus(
            False,
            f"invalid_state:{type(exc).__name__}",
            float("nan"),
            float("nan"),
            float("nan"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
        )
    position = x[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]]
    margins = position_margin_m(position, TRUE_SAFE_BOUNDS)
    x_w, y_w, z_w = (float(value) for value in position)
    if float(margins["x_min_margin_m"]) < 0.0:
        reason = "exit_gate_x_min"
    elif float(margins["x_max_margin_m"]) < 0.0:
        reason = "exit_gate_front_wall"
    elif float(margins["y_min_margin_m"]) < 0.0:
        reason = "exit_gate_y_min"
    elif float(margins["y_max_margin_m"]) < 0.0:
        reason = "exit_gate_y_max"
    elif float(margins["floor_margin_m"]) < 0.0:
        reason = "exit_gate_floor"
    elif float(margins["ceiling_margin_m"]) < 0.0:
        reason = "exit_gate_ceiling"
    else:
        reason = "inside_operational_region"
    inside = reason == "inside_operational_region"
    return ExitGateStatus(
        inside=inside,
        reason=reason,
        x_w_m=x_w,
        y_w_m=y_w,
        z_w_m=z_w,
        min_margin_m=float(margins["min_margin_m"]),
        floor_margin_m=float(margins["floor_margin_m"]),
        ceiling_margin_m=float(margins["ceiling_margin_m"]),
    )


def exit_gate_bounds_manifest() -> dict[str, object]:
    return {
        "policy": "terminate_active_flight_at_first_true_safe_box_exit_then_send_neutral",
        "x_w_m": list(TRUE_SAFE_BOUNDS.x_w_m),
        "y_w_m": list(TRUE_SAFE_BOUNDS.y_w_m),
        "z_w_m": list(TRUE_SAFE_BOUNDS.z_w_m),
    }
