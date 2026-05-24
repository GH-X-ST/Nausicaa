from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from env_ctx import EnvironmentContext
from lqr_controller import LQRCommand, LQRController, TimingAwareControllerState, lqr_command_for_state
from prim_cat import PrimitiveDefinition


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Control context dataclass
# 2) LQR command helpers
# =============================================================================


@dataclass(frozen=True)
class PrimitiveControlContext:
    state_vector: np.ndarray
    environment_context: EnvironmentContext
    time_in_primitive_s: float = 0.0
    timing_state: TimingAwareControllerState | None = None


def lqr_mode_for_primitive(primitive: PrimitiveDefinition) -> str:
    """Return the active LQR feedback mode for one primitive."""

    if primitive.controller_family != "lqr":
        return "blocked_non_lqr_primitive"
    return "predictor_compensated_augmented_discrete_lqr"


def primitive_lqr_command(
    primitive: PrimitiveDefinition,
    control_context: PrimitiveControlContext,
    controller: LQRController,
) -> LQRCommand:
    """Return a clipped LQR command for one primitive state."""

    if controller.primitive_id != primitive.primitive_id:
        raise ValueError("controller primitive_id must match the primitive definition.")
    return lqr_command_for_state(
        controller=controller,
        state_vector=control_context.state_vector,
        timing_state=control_context.timing_state,
    )


def primitive_control_command_row(command: LQRCommand) -> dict[str, object]:
    """Return a compact audit row for an LQR command."""

    row = asdict(command)
    row["command_norm"] = ";".join(f"{value:.9f}" for value in command.command_norm)
    row["command_rad"] = ";".join(f"{value:.9f}" for value in command.command_rad)
    row["raw_command_rad"] = ";".join(f"{value:.9f}" for value in command.raw_command_rad)
    row["timing_state_source"] = command.timing_state_source
    return row
