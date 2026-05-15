from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback import limit_aggregate_command
from linearisation import STATE_INDEX
from primitive import EntryConditionResult, PrimitiveContext, base_entry_conditions


@dataclass(frozen=True)
class ClosedLoopTrajectoryPrimitive:
    name: str
    times_s: np.ndarray
    x_ref: np.ndarray
    u_ff: np.ndarray
    k_feedback: np.ndarray
    phase_labels: tuple[str, ...]
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        x_ref = np.asarray(self.x_ref, dtype=float)
        u_ff = np.asarray(self.u_ff, dtype=float)
        k_feedback = np.asarray(self.k_feedback, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times_s must have shape (N,) with N >= 2.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing.")
        if x_ref.shape != (times.size, 15):
            raise ValueError("x_ref must have shape (N, 15).")
        if u_ff.shape != (times.size, 3):
            raise ValueError("u_ff must have shape (N, 3).")
        if k_feedback.ndim != 3 or k_feedback.shape[0:2] != (times.size, 3):
            raise ValueError("k_feedback must have shape (N, 3, n_error).")
        if k_feedback.shape[2] > 15:
            raise ValueError("k_feedback cannot use more than 15 state errors.")
        if len(self.phase_labels) != times.size:
            raise ValueError("phase_labels must have one label per time sample.")
        if not (
            np.all(np.isfinite(times))
            and np.all(np.isfinite(x_ref))
            and np.all(np.isfinite(u_ff))
            and np.all(np.isfinite(k_feedback))
        ):
            raise ValueError("trajectory arrays must be finite.")
        metadata = {} if self.metadata is None else dict(self.metadata)
        metadata.setdefault("primitive_family", "aggressive_high_incidence_reversal")
        metadata.setdefault("model_status", "high_incidence_simulation_surrogate")
        metadata.setdefault("is_real_flight_claim", False)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "x_ref", x_ref)
        object.__setattr__(self, "u_ff", u_ff)
        object.__setattr__(self, "k_feedback", k_feedback)
        object.__setattr__(self, "phase_labels", tuple(str(label) for label in self.phase_labels))
        object.__setattr__(self, "metadata", metadata)

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1] - self.times_s[0])

    def entry_conditions(
        self,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> EntryConditionResult:
        base = base_entry_conditions(
            x0=x0,
            context=context,
            max_abs_phi_rad=np.deg2rad(125.0),
            max_abs_theta_rad=np.deg2rad(125.0),
        )
        x = np.asarray(x0, dtype=float).reshape(15)
        reasons = list(base.reasons)
        position_error_m = float(np.linalg.norm(x[:3] - self.x_ref[0, :3]))
        if position_error_m > 0.35:
            reasons.append(f"trajectory entry position error {position_error_m:.3f} m")
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
        # Commands are aggregate surface radians in canonical [aileron, elevator, rudder] order.
        return limit_aggregate_command(command)

    def target_label(self, t_s: float) -> str:
        idx, _frac = _segment(self.times_s, t_s)
        return self.phase_labels[idx]

    def reference_state(self, t_s: float) -> np.ndarray:
        return _interp_rows(self.times_s, self.x_ref, t_s, wrap_attitude=True)

    def feedforward_command(self, t_s: float) -> np.ndarray:
        return _interp_rows(self.times_s, self.u_ff, t_s, wrap_attitude=False)

    def feedback_gain(self, t_s: float) -> np.ndarray:
        return _interp_rows(self.times_s, self.k_feedback, t_s, wrap_attitude=False)


def trajectory_error(x: np.ndarray, x_ref: np.ndarray, n_error: int = 15) -> np.ndarray:
    error = np.asarray(x, dtype=float).reshape(15) - np.asarray(x_ref, dtype=float).reshape(15)
    error[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1] = _wrap_angle(
        error[STATE_INDEX["phi"] : STATE_INDEX["psi"] + 1]
    )
    return error[: int(n_error)]


def _segment(times_s: np.ndarray, t_s: float) -> tuple[int, float]:
    t_clamped = float(np.clip(t_s, float(times_s[0]), float(times_s[-1])))
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

