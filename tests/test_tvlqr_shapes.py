from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from linearisation import linearise_trim
from tvlqr import TVLQRConfig, linearise_trajectory_finite_difference, solve_discrete_tvlqr


def test_trajectory_linearisation_and_tvlqr_shapes_are_finite() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    model = linearise_trim(aircraft=aircraft)
    x_ref = np.vstack([model.x_trim, model.x_trim])
    u_ff = np.vstack([model.u_trim, model.u_trim])

    a_mats, b_mats = linearise_trajectory_finite_difference(
        x_ref=x_ref,
        u_ff=u_ff,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        rho_kg_m3=1.225,
        actuator_tau_s=(0.06, 0.06, 0.06),
    )
    k_lqr, s_mats = solve_discrete_tvlqr(
        a_mats=a_mats,
        b_mats=b_mats,
        dt_s=0.02,
        config=TVLQRConfig(
            q_diag=(1.0,) * 15,
            r_diag=(10.0, 10.0, 10.0),
        ),
    )

    assert a_mats.shape == (2, 15, 15)
    assert b_mats.shape == (2, 15, 3)
    assert k_lqr.shape == (2, 3, 15)
    assert s_mats.shape == (2, 15, 15)
    assert np.all(np.isfinite(a_mats))
    assert np.all(np.isfinite(b_mats))
    assert np.all(np.isfinite(k_lqr))
    assert np.all(np.isfinite(s_mats))
