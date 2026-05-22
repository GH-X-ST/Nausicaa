from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider, state_derivative
from glider import build_nausicaa_glider
from state_contract import STATE_INDEX, STATE_SIZE


def _rk4_step(x: np.ndarray, u_cmd: np.ndarray, aircraft, dt_s: float) -> np.ndarray:
    k1 = state_derivative(x, u_cmd, aircraft, wind_model=None, wind_mode="none")
    k2 = state_derivative(x + 0.5 * dt_s * k1, u_cmd, aircraft, wind_model=None, wind_mode="none")
    k3 = state_derivative(x + 0.5 * dt_s * k2, u_cmd, aircraft, wind_model=None, wind_mode="none")
    k4 = state_derivative(x + dt_s * k3, u_cmd, aircraft, wind_model=None, wind_mode="none")
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def test_glider_construction_and_rk4_smoke_are_finite() -> None:
    glider = build_nausicaa_glider()
    aircraft = adapt_glider(glider)
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["z_w"]] = 1.5
    state[STATE_INDEX["u"]] = 6.0
    command = np.zeros(3)

    derivative = state_derivative(state, command, aircraft, wind_model=None, wind_mode="none")
    next_state = _rk4_step(state, command, aircraft, dt_s=0.02)

    assert glider.mass_kg > 0.0
    assert aircraft.strip_count > 0
    assert derivative.shape == (STATE_SIZE,)
    assert np.all(np.isfinite(derivative))
    assert next_state.shape == (STATE_SIZE,)
    assert np.all(np.isfinite(next_state))
