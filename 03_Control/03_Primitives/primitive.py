from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from linearisation import STATE_INDEX


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Primitive dataclasses and protocol
# 2) Primitive context builder
# 3) Shared entry-condition checks
# =============================================================================

# =============================================================================
# 1) Primitive Dataclasses and Protocol
# =============================================================================
# Primitive contracts keep entry checks, duration, and command generation separable.
@dataclass(frozen=True)
class PrimitiveContext:
    x_trim: np.ndarray
    u_trim: np.ndarray
    speed_trim_m_s: float
    theta_trim_rad: float
    min_entry_altitude_m: float = 0.75


@dataclass(frozen=True)
class EntryConditionResult:
    passed: bool
    reasons: tuple[str, ...]


class FlightPrimitive(Protocol):
    name: str
    duration_s: float

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        ...

    def command(
        self,
        t_s: float,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> np.ndarray:
        ...

    def target_label(self, t_s: float) -> str:
        ...


# =============================================================================
# 2) Primitive Context Builder
# =============================================================================
def build_primitive_context(
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    min_entry_altitude_m: float = 0.75,
) -> PrimitiveContext:
    x_trim = np.asarray(x_trim, dtype=float).reshape(15)
    u_trim = np.asarray(u_trim, dtype=float).reshape(3)
    # Context stores trim quantities once for primitive entry and feedback checks
    return PrimitiveContext(
        x_trim=x_trim,
        u_trim=u_trim,
        speed_trim_m_s=float(np.linalg.norm(x_trim[6:9])),
        theta_trim_rad=float(x_trim[STATE_INDEX["theta"]]),
        min_entry_altitude_m=float(min_entry_altitude_m),
    )


# =============================================================================
# 3) Shared Entry-Condition Checks
# =============================================================================
def base_entry_conditions(
    x0: np.ndarray,
    context: PrimitiveContext,
    max_abs_phi_rad: float = np.deg2rad(65.0),
    max_abs_theta_rad: float = np.deg2rad(40.0),
) -> EntryConditionResult:
    x = np.asarray(x0, dtype=float).reshape(15)
    reasons: list[str] = []
    if not np.all(np.isfinite(x)):
        reasons.append("state contains non-finite values")
    # Entry checks use speed, altitude, and attitude only
    speed = float(np.linalg.norm(x[6:9]))
    if speed < 2.5 or speed > 9.5:
        reasons.append(f"speed out of range: {speed:.3f} m/s")
    altitude = float(x[STATE_INDEX["z_w"]])
    if altitude < context.min_entry_altitude_m:
        reasons.append(f"altitude below entry floor: {altitude:.3f} m")
    if abs(float(x[STATE_INDEX["phi"]])) > max_abs_phi_rad:
        reasons.append("absolute bank angle exceeds entry limit")
    if abs(float(x[STATE_INDEX["theta"]])) > max_abs_theta_rad:
        reasons.append("absolute pitch angle exceeds entry limit")
    return EntryConditionResult(passed=not reasons, reasons=tuple(reasons))
