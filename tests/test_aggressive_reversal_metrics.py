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
        "finite_arrays",
        "source_trajectory_success",
        "source_feasibility_label",
        "source_failure_reason",
        "propagation_success",
        "fallback_used",
        "gain_arrays_finite",
        "primitive_constructed",
        "closed_loop_replay_success",
        "manoeuvre_success",
        "first_bad_step",
        "first_bad_time_s",
        "first_bad_reason",
        "first_bad_state_norm",
        "first_bad_speed_m_s",
        "first_bad_alpha_deg",
        "first_bad_beta_deg",
        "first_bad_bank_deg",
        "first_bad_pitch_deg",
        "first_bad_rate_norm_rad_s",
        "first_bad_nu_a",
        "first_bad_nu_e",
        "first_bad_nu_r",
        "first_bad_command_a_rad",
        "first_bad_command_e_rad",
        "first_bad_command_r_rad",
    ):
        assert key in row
    assert row["is_real_flight_claim"] is False


def test_failed_fallback_does_not_report_physical_heading_change() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    trim = linearise_trim(aircraft=aircraft)
    result = solve_aggressive_reversal_ocp(
        target=AggressiveReversalTarget(90.0),
        config=AggressiveReversalConfig(
            n_intervals=4,
            integration_speed_abort_m_s=0.1,
        ),
        x0=aggressive_reversal_entry_state(trim.x_trim),
        aircraft=aircraft,
        u_trim=trim.u_trim,
    )
    row = aggressive_reversal_metric_row(result, seed=1, initial_guess_name="pitch_brake_yaw_seed")

    assert row["feasibility_label"] == "solver_failure"
    assert row["success"] is False
    assert row["fallback_used"] is True
    assert row["actual_heading_change_deg"] == 0.0
    assert row["directed_heading_change_deg"] == -0.0
    assert row["first_bad_reason"] == "speed_abort"
    assert row["first_bad_step"] != ""
    assert row["first_bad_time_s"] != ""
