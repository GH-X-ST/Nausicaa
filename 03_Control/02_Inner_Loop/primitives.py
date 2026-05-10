from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from linearisation import STATE_INDEX


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


def build_primitive_context(
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    min_entry_altitude_m: float = 0.75,
) -> PrimitiveContext:
    x_trim = np.asarray(x_trim, dtype=float).reshape(15)
    u_trim = np.asarray(u_trim, dtype=float).reshape(3)
    return PrimitiveContext(
        x_trim=x_trim,
        u_trim=u_trim,
        speed_trim_m_s=float(np.linalg.norm(x_trim[6:9])),
        theta_trim_rad=float(x_trim[STATE_INDEX["theta"]]),
        min_entry_altitude_m=float(min_entry_altitude_m),
    )


def _base_entry_conditions(
    x0: np.ndarray,
    context: PrimitiveContext,
    max_abs_phi_rad: float = np.deg2rad(65.0),
    max_abs_theta_rad: float = np.deg2rad(40.0),
) -> EntryConditionResult:
    x = np.asarray(x0, dtype=float).reshape(15)
    reasons: list[str] = []
    if not np.all(np.isfinite(x)):
        reasons.append("state contains non-finite values")
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


def _surface_limit(command: np.ndarray) -> np.ndarray:
    return np.clip(command, np.deg2rad(-25.0), np.deg2rad(25.0))


def _speed_alpha_beta(x: np.ndarray) -> tuple[float, float, float]:
    u, v, w = np.asarray(x[6:9], dtype=float)
    speed = float(np.linalg.norm([u, v, w]))
    alpha = float(np.arctan2(w, max(u, 1e-12)))
    beta = float(np.arcsin(np.clip(v / max(speed, 1e-12), -1.0, 1.0)))
    return speed, alpha, beta


def _attitude_hold_command(
    x: np.ndarray,
    context: PrimitiveContext,
    phi_ref_rad: float,
    theta_ref_rad: float,
    gains: tuple[float, float, float, float, float, float] = (
        1.6,
        0.18,
        1.2,
        0.10,
        0.45,
        0.08,
    ),
) -> np.ndarray:
    phi = float(x[STATE_INDEX["phi"]])
    theta = float(x[STATE_INDEX["theta"]])
    p = float(x[STATE_INDEX["p"]])
    q = float(x[STATE_INDEX["q"]])
    r = float(x[STATE_INDEX["r"]])
    speed, _alpha, beta = _speed_alpha_beta(x)
    kp_phi, kd_p, kp_theta, kd_q, k_beta, kd_r = gains

    delta_a = kp_phi * (phi_ref_rad - phi) - kd_p * p
    delta_e = (
        context.u_trim[1]
        + kp_theta * (theta_ref_rad - theta)
        - kd_q * q
        + 0.04 * (speed - context.speed_trim_m_s)
    )
    delta_r = -k_beta * beta - kd_r * r
    return _surface_limit(np.array([delta_a, delta_e, delta_r], dtype=float))


@dataclass(frozen=True)
class NominalGlidePrimitive:
    name: str = "nominal_glide"
    duration_s: float = 6.0

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        return _base_entry_conditions(x0=x0, context=context)

    def command(
        self,
        t_s: float,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> np.ndarray:
        del t_s
        return _attitude_hold_command(
            x=x,
            context=context,
            phi_ref_rad=0.0,
            theta_ref_rad=context.theta_trim_rad,
            gains=(0.7, 0.10, 0.7, 0.08, 0.30, 0.06),
        )

    def target_label(self, t_s: float) -> str:
        del t_s
        return "trim"


@dataclass(frozen=True)
class BankReversalPrimitive:
    name: str = "bank_reversal"
    duration_s: float = 8.0
    bank_angle_rad: float = np.deg2rad(18.0)

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        return _base_entry_conditions(x0=x0, context=context)

    def _phi_ref(self, t_s: float) -> float:
        b = float(self.bank_angle_rad)
        schedule = (
            (0.0, 1.3, 0.0, b),
            (1.3, 3.0, b, b),
            (3.0, 4.5, b, -b),
            (4.5, 6.2, -b, -b),
            (6.2, self.duration_s, -b, 0.0),
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
        return _attitude_hold_command(
            x=x,
            context=context,
            phi_ref_rad=self._phi_ref(t_s),
            theta_ref_rad=context.theta_trim_rad,
        )

    def target_label(self, t_s: float) -> str:
        phi_ref = np.rad2deg(self._phi_ref(t_s))
        return f"phi_ref_deg={phi_ref:.1f}"


@dataclass(frozen=True)
class RecoveryPrimitive:
    name: str = "recovery"
    duration_s: float = 6.0

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        return _base_entry_conditions(
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
        return _attitude_hold_command(
            x=x,
            context=context,
            phi_ref_rad=0.0,
            theta_ref_rad=context.theta_trim_rad,
            gains=(1.1, 0.16, 0.8, 0.08, 0.35, 0.08),
        )

    def target_label(self, t_s: float) -> str:
        del t_s
        return "phi_ref_deg=0.0"
