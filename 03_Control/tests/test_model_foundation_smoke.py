from __future__ import annotations

import numpy as np
import pytest

from flight_dynamics import adapt_glider, build_symbolic_dynamics, evaluate_state, state_derivative
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


def test_control_surface_effectiveness_schedule_tracks_alpha_regime_and_symbolic_path() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    command = np.zeros(3)

    normal = np.zeros(STATE_SIZE)
    normal[STATE_INDEX["z_w"]] = 1.5
    normal[STATE_INDEX["u"]] = 6.0
    normal[STATE_INDEX["w"]] = 0.2

    transition = normal.copy()
    transition[STATE_INDEX["u"]] = 4.0
    transition[STATE_INDEX["w"]] = float(4.0 * np.tan(np.deg2rad(16.0)))

    post_stall = normal.copy()
    post_stall[STATE_INDEX["u"]] = 3.0
    post_stall[STATE_INDEX["w"]] = 2.0

    normal_eval = evaluate_state(normal, command, aircraft, wind_model=None, wind_mode="none")
    transition_eval = evaluate_state(transition, command, aircraft, wind_model=None, wind_mode="none")
    post_stall_eval = evaluate_state(post_stall, command, aircraft, wind_model=None, wind_mode="none")

    normal_scale = np.asarray(normal_eval["control_effectiveness_scale"], dtype=float)
    transition_scale = np.asarray(transition_eval["control_effectiveness_scale"], dtype=float)
    post_stall_scale = np.asarray(post_stall_eval["control_effectiveness_scale"], dtype=float)

    assert normal_eval["control_effectiveness_model"] == "alpha_regime_scheduled_v1"
    assert normal_scale.tolist() == pytest.approx([0.85, 0.75, 0.85])
    assert post_stall_scale.tolist() == pytest.approx([0.45, 0.45, 0.40])
    assert np.all(transition_scale < normal_scale)
    assert np.all(transition_scale > post_stall_scale)

    symbolic = build_symbolic_dynamics(aircraft, wind_mode="none")
    symbolic_derivative = np.asarray(symbolic.function(post_stall, command)).reshape(-1)
    assert symbolic_derivative.shape == (STATE_SIZE,)
    assert np.all(np.isfinite(symbolic_derivative))
