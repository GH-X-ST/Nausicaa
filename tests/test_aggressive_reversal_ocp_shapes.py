from __future__ import annotations

import numpy as np

from aggressive_reversal_ocp import (
    AggressiveReversalConfig,
    AggressiveReversalTarget,
    deterministic_aggressive_guess_names,
    solve_aggressive_reversal_ocp,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from linearisation import linearise_trim
from scenarios import aggressive_reversal_entry_state


def test_aggressive_reversal_ocp_shapes() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    trim = linearise_trim(aircraft=aircraft)
    config = AggressiveReversalConfig(n_intervals=4)
    result = solve_aggressive_reversal_ocp(
        target=AggressiveReversalTarget(30.0),
        config=config,
        x0=aggressive_reversal_entry_state(trim.x_trim),
        aircraft=aircraft,
        u_trim=trim.u_trim,
        wind_model=None,
        wind_mode="none",
        initial_guess_name=deterministic_aggressive_guess_names(30.0)[0],
    )

    assert result.times_s.shape == (5,)
    assert result.x_ref.shape == (5, 15)
    assert result.u_ff.shape == (5, 3)
    assert result.nu_ff.shape == (5, 3)
    assert len(result.phase_labels) == 5

