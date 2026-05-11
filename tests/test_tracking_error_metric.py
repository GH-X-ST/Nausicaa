from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "03_Control/02_Inner_Loop",
    "03_Control/03_Primitives",
):
    path = REPO_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linearisation import STATE_INDEX  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import primitive_tracking_error_rms  # noqa: E402
from templates import BankReversalPrimitive, NominalGlidePrimitive  # noqa: E402


def _context(theta_trim_rad: float = 0.08):
    x_trim = np.zeros(15, dtype=float)
    x_trim[STATE_INDEX["theta"]] = theta_trim_rad
    x_trim[STATE_INDEX["u"]] = 5.0
    return build_primitive_context(x_trim=x_trim, u_trim=np.zeros(3, dtype=float))


def _states(times_s: np.ndarray, phi_rad: np.ndarray, theta_rad: float) -> np.ndarray:
    states = np.zeros((times_s.size, 15), dtype=float)
    states[:, STATE_INDEX["phi"]] = phi_rad
    states[:, STATE_INDEX["theta"]] = theta_rad
    return states


def test_nominal_glide_zero_tracking_error_for_trim_attitude() -> None:
    context = _context()
    primitive = NominalGlidePrimitive()
    times_s = np.linspace(0.0, primitive.duration_s, 6)
    states = _states(
        times_s=times_s,
        phi_rad=np.zeros_like(times_s),
        theta_rad=context.theta_trim_rad,
    )

    error_rms = primitive_tracking_error_rms(primitive, context, times_s, states)

    assert error_rms == 0.0


def test_bank_reversal_zero_tracking_error_for_exact_roll_schedule() -> None:
    context = _context()
    primitive = BankReversalPrimitive()
    times_s = np.linspace(0.0, primitive.duration_s, 9)
    phi_ref = np.asarray([primitive.phi_ref(float(t_s)) for t_s in times_s], dtype=float)
    states = _states(times_s=times_s, phi_rad=phi_ref, theta_rad=context.theta_trim_rad)

    error_rms = primitive_tracking_error_rms(primitive, context, times_s, states)

    assert error_rms == 0.0


def test_fixed_roll_offset_produces_positive_finite_tracking_error() -> None:
    context = _context()
    primitive = BankReversalPrimitive()
    times_s = np.linspace(0.0, primitive.duration_s, 9)
    phi_ref = np.asarray([primitive.phi_ref(float(t_s)) for t_s in times_s], dtype=float)
    states = _states(
        times_s=times_s,
        phi_rad=phi_ref + 0.1,
        theta_rad=context.theta_trim_rad,
    )

    error_rms = primitive_tracking_error_rms(primitive, context, times_s, states)

    assert np.isfinite(error_rms)
    assert error_rms > 0.0

