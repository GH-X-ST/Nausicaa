from __future__ import annotations

from aggressive_reversal_ocp import (
    AggressiveReversalConfig,
    AggressiveReversalTarget,
    aggressive_reversal_metric_row,
    solve_aggressive_reversal_ocp,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from linearisation import linearise_trim
from scenarios import aggressive_reversal_entry_state


def test_aggressive_reversal_metric_row_contains_required_fields() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    trim = linearise_trim(aircraft=aircraft)
    result = solve_aggressive_reversal_ocp(
        target=AggressiveReversalTarget(30.0),
        config=AggressiveReversalConfig(n_intervals=4),
        x0=aggressive_reversal_entry_state(trim.x_trim),
        aircraft=aircraft,
        u_trim=trim.u_trim,
    )

    row = aggressive_reversal_metric_row(result, seed=1, initial_guess_name="pitch_brake_yaw_seed")

    for key in (
        "target_heading_deg",
        "actual_heading_change_deg",
        "max_alpha_deg",
        "min_wall_distance_m",
        "saturation_fraction",
        "model_status",
        "is_real_flight_claim",
    ):
        assert key in row
    assert row["is_real_flight_claim"] is False

