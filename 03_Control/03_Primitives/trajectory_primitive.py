from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback import limit_aggregate_command
from linearisation import STATE_INDEX
from primitive import (
    EntryConditionResult,
    PrimitiveContext,
    base_entry_conditions,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Entry limits
# 2) Trajectory primitive
# 3) Interpolation helpers
# =============================================================================

# =============================================================================
# 1) Entry Limits
# =============================================================================
# Trajectory entry limits add local reference consistency checks to the shared primitive gate.
@dataclass(frozen=True)
class TrajectoryEntryLimits:
    max_position_error_m: float = 0.18
    max_attitude_error_rad: float = np.deg2rad(8.0)
    max_surface_error_rad: float = np.deg2rad(8.0)
    max_abs_phi_rad: float = np.deg2rad(65.0)
    max_abs_theta_rad: float = np.deg2rad(40.0)


# =============================================================================
# 2) Trajectory Primitive
# =============================================================================
@dataclass(frozen=True)
class TrajectoryPrimitive:
    name: str
    times_s: np.ndarray
    x_ref: np.ndarray
    u_ff: np.ndarray
    k_lqr: np.ndarray
    a_mats: np.ndarray | None
    b_mats: np.ndarray | None
    s_mats: np.ndarray | None = None
    entry_limits: object | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        x_ref = np.asarray(self.x_ref, dtype=float)
        u_ff = np.asarray(self.u_ff, dtype=float)
        k_lqr = np.asarray(self.k_lqr, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times_s must be a one-dimensional array with at least two samples.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing.")
        if x_ref.shape != (times.size, 15):
            raise ValueError("x_ref must have shape (N, 15).")
        if u_ff.shape != (times.size, 3):
            raise ValueError("u_ff must have shape (N, 3).")
        if k_lqr.ndim != 3 or k_lqr.shape[0] != times.size or k_lqr.shape[1] != 3:
            raise ValueError("k_lqr must have shape (N, 3, n_error).")
        if k_lqr.shape[2] > 15:
            raise ValueError("k_lqr cannot use more than the canonical 15 state errors.")
        if self.a_mats is not None and np.asarray(self.a_mats).shape != (times.size, 15, 15):
            raise ValueError("a_mats must have shape (N, 15, 15).")
        if self.b_mats is not None and np.asarray(self.b_mats).shape != (times.size, 15, 3):
            raise ValueError("b_mats must have shape (N, 15, 3).")
        if self.s_mats is not None and np.asarray(self.s_mats).shape != (times.size, 15, 15):
            raise ValueError("s_mats must have shape (N, 15, 15).")
        if not (
            np.all(np.isfinite(times))
            and np.all(np.isfinite(x_ref))
            and np.all(np.isfinite(u_ff))
            and np.all(np.isfinite(k_lqr))
        ):
            raise ValueError("trajectory primitive arrays must be finite.")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "x_ref", x_ref)
        object.__setattr__(self, "u_ff", u_ff)
        object.__setattr__(self, "k_lqr", k_lqr)
        object.__setattr__(
            self,
            "a_mats",
            None if self.a_mats is None else np.asarray(self.a_mats, dtype=float),
        )
        object.__setattr__(
            self,
            "b_mats",
            None if self.b_mats is None else np.asarray(self.b_mats, dtype=float),
        )
        object.__setattr__(
            self,
            "s_mats",
            None if self.s_mats is None else np.asarray(self.s_mats, dtype=float),
        )
        object.__setattr__(
            self,
            "metadata",
            {} if self.metadata is None else dict(self.metadata),
        )

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1] - self.times_s[0])

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        limits = _entry_limits(self.entry_limits)
        base = base_entry_conditions(
            x0=x0,
            context=context,
            max_abs_phi_rad=limits.max_abs_phi_rad,
            max_abs_theta_rad=limits.max_abs_theta_rad,
        )
        x = np.asarray(x0, dtype=float).reshape(15)
        x_start = self.x_ref[0]
        reasons = list(base.reasons)
        if not np.all(np.isfinite(x)):
            reasons.append("trajectory entry state contains non-finite values")
        position_error = float(np.linalg.norm(x[:3] - x_start[:3]))
        if position_error > limits.max_position_error_m:
            reasons.append(f"trajectory entry position error {position_error:.3f} m")
        attitude_error = np.abs(_wrap_angle(x[3:6] - x_start[3:6]))
        if float(np.max(attitude_error)) > limits.max_attitude_error_rad:
            reasons.append("trajectory entry attitude error exceeds limit")
        surface_error = float(np.max(np.abs(x[12:15] - x_start[12:15])))
        if surface_error > limits.max_surface_error_rad:
            reasons.append("trajectory entry surface-state error exceeds limit")
        return EntryConditionResult(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))

    def command(
        self,
        t_s: float,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> np.ndarray:
        del context
        x_now = np.asarray(x, dtype=float).reshape(15)
        x_ref = self.reference_state(t_s)
        u_ff = self.feedforward_command(t_s)
        k = self.feedback_gain(t_s)
        error = trajectory_error(x_now, x_ref, k.shape[1])
        command = u_ff - k @ error
        return limit_aggregate_command(command)

    def target_label(self, t_s: float) -> str:
        idx, _frac = _segment(self.times_s, t_s)
        label = str(self.metadata.get("target_label", "trajectory_ref"))
        return f"{label}[{idx}]"

    def reference_state(self, t_s: float) -> np.ndarray:
        return _interp_rows(self.times_s, self.x_ref, t_s, wrap_attitude=True)

    def feedforward_command(self, t_s: float) -> np.ndarray:
        return _interp_rows(self.times_s, self.u_ff, t_s, wrap_attitude=False)

    def feedback_gain(self, t_s: float) -> np.ndarray:
        return _interp_rows(self.times_s, self.k_lqr, t_s, wrap_attitude=False)


def trajectory_error(x: np.ndarray, x_ref: np.ndarray, n_error: int = 15) -> np.ndarray:
    error = np.asarray(x, dtype=float).reshape(15) - np.asarray(x_ref, dtype=float).reshape(15)
    error[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1] = _wrap_angle(
        error[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1]
    )
    return error[: int(n_error)]


# =============================================================================
# 3) Interpolation Helpers
# =============================================================================
def _entry_limits(value: object | None) -> TrajectoryEntryLimits:
    if value is None:
        return TrajectoryEntryLimits()
    if isinstance(value, TrajectoryEntryLimits):
        return value
    if isinstance(value, dict):
        return TrajectoryEntryLimits(**value)
    return TrajectoryEntryLimits(
        max_position_error_m=float(
            getattr(value, "max_position_error_m", TrajectoryEntryLimits.max_position_error_m)
        ),
        max_attitude_error_rad=float(
            getattr(value, "max_attitude_error_rad", TrajectoryEntryLimits.max_attitude_error_rad)
        ),
        max_surface_error_rad=float(
            getattr(value, "max_surface_error_rad", TrajectoryEntryLimits.max_surface_error_rad)
        ),
        max_abs_phi_rad=float(
            getattr(value, "max_abs_phi_rad", TrajectoryEntryLimits.max_abs_phi_rad)
        ),
        max_abs_theta_rad=float(
            getattr(value, "max_abs_theta_rad", TrajectoryEntryLimits.max_abs_theta_rad)
        ),
    )


def _segment(times_s: np.ndarray, t_s: float) -> tuple[int, float]:
    t_clamped = float(np.clip(float(t_s), float(times_s[0]), float(times_s[-1])))
    right = int(np.searchsorted(times_s, t_clamped, side="right"))
    idx = int(np.clip(right - 1, 0, times_s.size - 2))
    span = max(float(times_s[idx + 1] - times_s[idx]), 1e-12)
    frac = (t_clamped - float(times_s[idx])) / span
    return idx, float(np.clip(frac, 0.0, 1.0))


def _interp_rows(
    times_s: np.ndarray,
    values: np.ndarray,
    t_s: float,
    wrap_attitude: bool,
) -> np.ndarray:
    idx, frac = _segment(times_s, t_s)
    v0 = values[idx]
    v1 = values[idx + 1]
    delta = v1 - v0
    if wrap_attitude and values.ndim == 2 and values.shape[1] >= 6:
        delta = np.asarray(delta, dtype=float).copy()
        delta[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1] = _wrap_angle(
            delta[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1]
        )
    return np.asarray(v0 + frac * delta, dtype=float)


def _wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
