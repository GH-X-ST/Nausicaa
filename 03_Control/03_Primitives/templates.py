from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback import attitude_hold_command
from primitive import (
    EntryConditionResult,
    PrimitiveContext,
    base_entry_conditions,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Nominal glide primitive
# 2) Bank-reversal primitive
# 3) Recovery primitive
# =============================================================================

# =============================================================================
# 1) Nominal Glide Primitive
# =============================================================================
@dataclass(frozen=True)
class NominalGlidePrimitive:
    name: str = "nominal_glide"
    duration_s: float = 2.0

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        return base_entry_conditions(x0=x0, context=context)

    def command(
        self,
        t_s: float,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> np.ndarray:
        del t_s
        # Nominal glide holds the trimmed pitch and zero bank reference
        return attitude_hold_command(
            x=x,
            context=context,
            phi_ref_rad=0.0,
            theta_ref_rad=context.theta_trim_rad,
            gains=(0.7, 0.10, 0.7, 0.08, 0.30, 0.06),
        )

    def target_label(self, t_s: float) -> str:
        del t_s
        return "trim"


# =============================================================================
# 2) Bank-Reversal Primitive
# =============================================================================
@dataclass(frozen=True)
class BankReversalPrimitive:
    name: str = "bank_reversal"
    duration_s: float = 2.5
    bank_angle_rad: float = np.deg2rad(12.0)

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        return base_entry_conditions(x0=x0, context=context)

    def phi_ref(self, t_s: float) -> float:
        b = float(self.bank_angle_rad)
        # Schedule covers roll-in, hold, reversal, hold, and exit phases
        schedule = (
            (0.0, 0.55, 0.0, b),
            (0.55, 1.10, b, b),
            (1.10, 1.75, b, -b),
            (1.75, 2.20, -b, -b),
            (2.20, self.duration_s, -b, 0.0),
        )
        for t0, t1, y0, y1 in schedule:
            if t_s <= t1:
                span = max(t1 - t0, 1e-12)
                frac = np.clip((t_s - t0) / span, 0.0, 1.0)
                return float(y0 + frac * (y1 - y0))
        return 0.0

    def command(
        self,
        t_s: float,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> np.ndarray:
        return attitude_hold_command(
            x=x,
            context=context,
            phi_ref_rad=self.phi_ref(t_s),
            theta_ref_rad=context.theta_trim_rad,
        )

    def target_label(self, t_s: float) -> str:
        return f"phi_ref_deg={np.rad2deg(self.phi_ref(t_s)):.1f}"


# =============================================================================
# 3) Recovery Primitive
# =============================================================================
@dataclass(frozen=True)
class RecoveryPrimitive:
    name: str = "recovery"
    duration_s: float = 2.0

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        # Recovery allows wider attitude errors than nominal manoeuvre entries
        return base_entry_conditions(
            x0=x0,
            context=context,
            max_abs_phi_rad=np.deg2rad(75.0),
            max_abs_theta_rad=np.deg2rad(50.0),
        )

    def command(
        self,
        t_s: float,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> np.ndarray:
        del t_s
        return attitude_hold_command(
            x=x,
            context=context,
            phi_ref_rad=0.0,
            theta_ref_rad=context.theta_trim_rad,
            gains=(1.1, 0.16, 0.8, 0.08, 0.35, 0.08),
        )

    def target_label(self, t_s: float) -> str:
        del t_s
        return "phi_ref_deg=0.0"
