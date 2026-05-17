from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider, state_derivative
from glider import build_nausicaa_glider
from linearisation import STATE_INDEX, linearise_trim


def test_symbolic_linearisation_matches_finite_difference_selected_entries() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    model = linearise_trim(aircraft=aircraft)
    eps = 1e-6
    for state_name in ("u", "w", "q", "delta_e"):
        idx = STATE_INDEX[state_name]
        perturb = np.zeros(15)
        perturb[idx] = eps
        f_plus = state_derivative(model.x_trim + perturb, model.u_trim, aircraft, wind_mode="none")
        f_minus = state_derivative(model.x_trim - perturb, model.u_trim, aircraft, wind_mode="none")
        column = (f_plus - f_minus) / (2.0 * eps)
        assert np.allclose(column[[6, 8, 10]], model.a[[6, 8, 10], idx], atol=1e-5)
