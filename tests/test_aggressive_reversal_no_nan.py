from __future__ import annotations

import numpy as np

from aggressive_reversal_ocp import AggressiveReversalConfig, AggressiveReversalTarget, solve_aggressive_reversal_ocp
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from linearisation import linearise_trim
from scenarios import aggressive_reversal_entry_state


def test_aggressive_reversal_result_arrays_are_finite() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    trim = linearise_trim(aircraft=aircraft)
    result = solve_aggressive_reversal_ocp(
        target=AggressiveReversalTarget(90.0),
        config=AggressiveReversalConfig(n_intervals=4),
        x0=aggressive_reversal_entry_state(trim.x_trim),
        aircraft=aircraft,
        u_trim=trim.u_trim,
    )

    assert np.all(np.isfinite(result.times_s))
    assert np.all(np.isfinite(result.x_ref))
    assert np.all(np.isfinite(result.u_ff))
    assert np.all(np.isfinite(result.nu_ff))


def test_aggressive_reversal_failed_propagation_uses_finite_fallback() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    trim = linearise_trim(aircraft=aircraft)
    result = solve_aggressive_reversal_ocp(
        target=AggressiveReversalTarget(30.0),
        config=AggressiveReversalConfig(
            n_intervals=4,
            integration_speed_abort_m_s=0.1,
        ),
        x0=aggressive_reversal_entry_state(trim.x_trim),
        aircraft=aircraft,
        u_trim=trim.u_trim,
    )

    assert np.all(np.isfinite(result.times_s))
    assert np.all(np.isfinite(result.x_ref))
    assert np.all(np.isfinite(result.u_ff))
    assert np.all(np.isfinite(result.nu_ff))
    assert result.metrics["fallback_used"] is True
    assert result.metrics["propagation_success"] is False
