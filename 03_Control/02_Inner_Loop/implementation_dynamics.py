from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Actuator dataclasses
# 2) Vector helper
# 3) Actuator update functions
# =============================================================================

# =============================================================================
# 1) Actuator Dataclasses
# =============================================================================
# Actuator parameters are SI/radian values shared by mismatch wrappers and rollout tests.
@dataclass(frozen=True)
class ActuatorParams:
    tau_s: np.ndarray
    rate_limit_rad_s: np.ndarray


@dataclass(frozen=True)
class ActuatorState:
    delta_rad: np.ndarray


# =============================================================================
# 2) Vector Helper
# =============================================================================
def _vector3(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    # Three-axis actuator vector
    return np.asarray(values, dtype=float).reshape(3)


# =============================================================================
# 3) Actuator Update Functions
# =============================================================================
def rate_limit_step(
    delta_target_rad: np.ndarray | list[float] | tuple[float, ...],
    delta_prev_rad: np.ndarray | list[float] | tuple[float, ...],
    rate_limit_rad_s: np.ndarray | list[float] | tuple[float, ...],
    dt_s: float,
) -> np.ndarray:
    delta_target = _vector3(delta_target_rad)
    delta_prev = _vector3(delta_prev_rad)
    rate_limit = _vector3(rate_limit_rad_s)
    # Symmetric rate limit per time step
    max_step = rate_limit * float(dt_s)
    return delta_prev + np.clip(delta_target - delta_prev, -max_step, max_step)


def actuator_step(
    state: ActuatorState,
    delta_cmd_rad: np.ndarray | list[float] | tuple[float, ...],
    dt_s: float,
    params: ActuatorParams,
) -> ActuatorState:
    delta_cmd = _vector3(delta_cmd_rad)
    tau_s = _vector3(params.tau_s)
    delta_prev = _vector3(state.delta_rad)
    # First-order actuator lag
    delta_target = delta_prev + float(dt_s) * (delta_cmd - delta_prev) / tau_s
    delta_next = rate_limit_step(
        delta_target_rad=delta_target,
        delta_prev_rad=delta_prev,
        rate_limit_rad_s=params.rate_limit_rad_s,
        dt_s=dt_s,
    )
    return ActuatorState(delta_rad=delta_next)
