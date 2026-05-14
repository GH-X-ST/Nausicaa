from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from latency import AGGREGATE_LIMITS, command_norm_to_angle
from linearisation import linearise_trim
from scenarios import arena_feasible_entry_state
from turn_trajectory_optimisation import (
    TurnOptimisationConfig,
    TurnTarget,
    normalised_command_to_radians,
    solve_turn_ocp,
)


def test_normalised_command_mapping_matches_calibrated_latency_limits() -> None:
    command_norm = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.4, -0.6, 0.8],
        ],
        dtype=float,
    )

    mapped = normalised_command_to_radians(command_norm)

    expected = np.array(
        [
            [
                command_norm_to_angle(row[0], AGGREGATE_LIMITS["delta_a"]),
                command_norm_to_angle(row[1], AGGREGATE_LIMITS["delta_e"]),
                command_norm_to_angle(row[2], AGGREGATE_LIMITS["delta_r"]),
            ]
            for row in command_norm
        ],
        dtype=float,
    )
    np.testing.assert_allclose(mapped, expected)


def test_zero_degree_ocp_smoke_returns_finite_diagnostics() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    x0 = arena_feasible_entry_state(linear_model.x_trim, altitude_m=2.7)
    target = TurnTarget(target_heading_deg=0.0, direction="left")
    config = TurnOptimisationConfig(
        n_intervals=2,
        t_min_s=0.04,
        t_max_s=0.08,
        ipopt_max_iter=30,
        max_solver_time_s=5.0,
    )

    result = solve_turn_ocp(
        target=target,
        config=config,
        x0=x0,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        u_trim=linear_model.u_trim,
        initial_guess_name="trim_glide",
    )

    assert result.times_s.shape == (3,)
    assert result.x_ref.shape == (3, 15)
    assert result.u_ff.shape == (2, 3)
    assert result.nu_ff is not None
    assert result.nu_ff.shape == (2, 3)
    assert np.all(np.isfinite(result.times_s))
    assert np.all(np.isfinite(result.x_ref))
    assert np.all(np.isfinite(result.u_ff))
    assert result.feasibility_label != "physically_infeasible_candidate"
    assert "dynamics_defect_max" in result.metrics

