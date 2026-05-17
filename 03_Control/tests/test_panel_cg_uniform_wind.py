from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider, evaluate_state
from glider import build_nausicaa_glider
from linearisation import linearise_trim


def test_uniform_wind_panel_and_cg_modes_match() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    model = linearise_trim(aircraft=aircraft)
    wind = np.array([0.2, -0.1, 0.05], dtype=float)
    panel = evaluate_state(
        model.x_trim,
        model.u_trim,
        aircraft,
        wind_model=wind,
        wind_mode="panel",
    )
    cg = evaluate_state(
        model.x_trim,
        model.u_trim,
        aircraft,
        wind_model=wind,
        wind_mode="cg",
    )
    assert np.allclose(panel["f_aero_b"], cg["f_aero_b"], atol=1e-12)
    assert np.allclose(panel["m_aero_b"], cg["m_aero_b"], atol=1e-12)
