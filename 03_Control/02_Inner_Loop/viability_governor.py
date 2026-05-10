from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linearisation import STATE_INDEX
from primitives import FlightPrimitive, PrimitiveContext


@dataclass(frozen=True)
class GovernorLimits:
    min_altitude_m: float = 0.75
    speed_bounds_m_s: tuple[float, float] = (2.5, 9.5)
    max_abs_phi_rad: float = np.deg2rad(75.0)
    max_abs_theta_rad: float = np.deg2rad(50.0)
    max_abs_alpha_rad: float = np.deg2rad(25.0)


@dataclass(frozen=True)
class GovernorDecision:
    accepted: bool
    reasons: tuple[str, ...]


class ViabilityGovernor:
    def __init__(self, limits: GovernorLimits | None = None):
        self.limits = limits or GovernorLimits()
        self.rejection_rows: list[dict[str, str | float]] = []

    def evaluate(
        self,
        scenario_name: str,
        primitive: FlightPrimitive,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> GovernorDecision:
        x = np.asarray(x0, dtype=float).reshape(15)
        reasons = list(primitive.entry_conditions(x, context).reasons)
        if not np.all(np.isfinite(x)):
            reasons.append("governor: state contains non-finite values")

        speed = float(np.linalg.norm(x[6:9]))
        low, high = self.limits.speed_bounds_m_s
        if speed < low or speed > high:
            reasons.append(f"governor: speed out of bounds {speed:.3f} m/s")
        altitude = float(x[STATE_INDEX["z_w"]])
        if altitude < self.limits.min_altitude_m:
            reasons.append(f"governor: altitude below floor {altitude:.3f} m")
        if abs(float(x[STATE_INDEX["phi"]])) > self.limits.max_abs_phi_rad:
            reasons.append("governor: bank angle outside viability envelope")
        if abs(float(x[STATE_INDEX["theta"]])) > self.limits.max_abs_theta_rad:
            reasons.append("governor: pitch angle outside viability envelope")

        alpha = float(np.arctan2(x[STATE_INDEX["w"]], max(x[STATE_INDEX["u"]], 1e-12)))
        if abs(alpha) > self.limits.max_abs_alpha_rad:
            reasons.append("governor: angle of attack outside viability envelope")

        unique_reasons = tuple(dict.fromkeys(reasons))
        accepted = not unique_reasons
        if not accepted:
            self.rejection_rows.append(
                {
                    "scenario": scenario_name,
                    "primitive": primitive.name,
                    "speed_m_s": speed,
                    "altitude_m": altitude,
                    "phi_deg": float(np.rad2deg(x[STATE_INDEX["phi"]])),
                    "theta_deg": float(np.rad2deg(x[STATE_INDEX["theta"]])),
                    "alpha_deg": float(np.rad2deg(alpha)),
                    "reason": "; ".join(unique_reasons),
                }
            )
        return GovernorDecision(accepted=accepted, reasons=unique_reasons)

    def write_rejection_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "scenario",
            "primitive",
            "speed_m_s",
            "altitude_m",
            "phi_deg",
            "theta_deg",
            "alpha_deg",
            "reason",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rejection_rows)
