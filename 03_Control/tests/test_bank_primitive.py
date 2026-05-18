from __future__ import annotations

import numpy as np
import pytest

import bank_primitive
from bank_primitive import (
    BANK_CAMPAIGN,
    GLIDE_PROXY_SOURCE,
    INITIAL_POSITION_W_M,
    PRIMITIVE_FAMILY,
    BankFeedbackGains,
    BankPrimitiveConfig,
    bank_feedback_command_norm,
    bank_reference_bank_rad,
    build_bank_cases,
    build_bank_primitive_spec,
    rollout_bank_case,
    run_bank_batch,
    terminal_glide_entry_proxy,
    terminal_glide_entry_x_margin_m,
    terminal_true_safe_x_margin_m,
)
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
    surface_rad_to_normalised_command,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from primitive_interface import evaluate_entry_set
from trim_solver import TrimTarget, solve_straight_trim


@pytest.fixture(scope="module")
def bank_context():
    config = BankPrimitiveConfig()
    aircraft = adapt_glider(build_nausicaa_glider())
    trim = solve_straight_trim(
        aircraft,
        TrimTarget(
            speed_m_s=config.speed_m_s,
            altitude_m=config.altitude_m,
            wind_model=None,
            wind_mode="none",
            actuator_tau_s=config.actuator_tau_s,
        ),
    )
    cases = build_bank_cases(config, trim)
    batch = run_bank_batch(config=config, aircraft=aircraft)
    return {
        "config": config,
        "aircraft": aircraft,
        "trim": trim,
        "cases": cases,
        "batch": batch,
    }


def _case_result(bank_context, name: str):
    for result in bank_context["batch"].case_results:
        if result.case_spec.name == name:
            return result
    raise AssertionError(f"case not found: {name}")


def test_bank_spec_family_and_entry_geometry_bounds() -> None:
    spec = build_bank_primitive_spec(BankPrimitiveConfig())

    assert BANK_CAMPAIGN == "06_bank_primitive"
    assert spec.family == PRIMITIVE_FAMILY
    assert spec.name == "bank_w0_nominal"
    assert spec.metadata["actual_bank_primitive_implemented"] == "true"
    assert spec.metadata["terminal_glide_entry_proxy_source"] == GLIDE_PROXY_SOURCE
    assert spec.entry_set.lower["x_w"] == pytest.approx(1.25)
    assert spec.entry_set.upper["x_w"] == pytest.approx(1.35)
    for variable in ("x_w", "y_w", "z_w", "speed_m_s", "alpha_rad", "beta_rad"):
        assert variable in spec.entry_set.lower
        assert variable in spec.entry_set.upper


def test_bank_case_inventory_and_default_position(bank_context) -> None:
    cases = bank_context["cases"]

    assert [case.name for case in cases] == [
        "bank_w0_left_mild",
        "bank_w0_right_mild",
        "bank_w0_left_sideslip_entry",
        "bank_w0_right_sideslip_entry",
        "bank_w0_0p80_handoff_boundary",
    ]
    roles = {case.name: case.role for case in cases}
    assert roles["bank_w0_left_mild"] == "required"
    assert roles["bank_w0_right_mild"] == "required"
    assert roles["bank_w0_0p80_handoff_boundary"] == "diagnostic"
    assert cases[-1].t_final_s == pytest.approx(0.80)
    np.testing.assert_allclose(cases[0].x0[0:3], INITIAL_POSITION_W_M)


def test_entry_set_geometry_passes_and_fails(bank_context) -> None:
    spec = build_bank_primitive_spec(bank_context["config"])
    required = bank_context["cases"][0]
    bad_state = required.x0.copy()
    bad_state[0] = 1.40

    pass_checks = evaluate_entry_set(required.x0, spec.entry_set)
    fail_checks = evaluate_entry_set(bad_state, spec.entry_set)

    assert all(check.pass_check for check in pass_checks)
    assert any(
        check.variable == "x_w" and not check.pass_check
        for check in fail_checks
    )


def test_bank_reference_profile(bank_context) -> None:
    config = bank_context["config"]
    left = bank_context["cases"][0]
    right = bank_context["cases"][1]

    assert bank_reference_bank_rad(0.0, right, config) == pytest.approx(0.0)
    assert bank_reference_bank_rad(0.12, right, config) == pytest.approx(
        np.deg2rad(config.target_bank_deg)
    )
    assert bank_reference_bank_rad(0.30, right, config) == pytest.approx(
        np.deg2rad(config.target_bank_deg)
    )
    assert bank_reference_bank_rad(0.60, right, config) == pytest.approx(0.0)
    assert bank_reference_bank_rad(0.30, left, config) == pytest.approx(
        -np.deg2rad(config.target_bank_deg)
    )


def test_feedback_uses_normalised_trim_command_space(bank_context) -> None:
    trim = bank_context["trim"]
    case = bank_context["cases"][1]
    u_trim_norm = surface_rad_to_normalised_command(trim.u_cmd_trim)

    requested = bank_feedback_command_norm(
        trim.x_trim,
        trim.x_trim,
        bank_reference_bank_rad(0.0, case, bank_context["config"]),
        u_trim_norm,
        BankFeedbackGains(),
    )

    np.testing.assert_allclose(requested, u_trim_norm)
    assert not np.allclose(requested, trim.u_cmd_trim)


def test_command_bridge_consistency(bank_context) -> None:
    result = _case_result(bank_context, "bank_w0_right_mild")

    np.testing.assert_allclose(
        result.u_norm_applied,
        np.vstack([clip_normalised_command(row) for row in result.u_norm_requested]),
    )
    np.testing.assert_allclose(
        result.delta_cmd_rad,
        np.vstack(
            [
                normalised_command_to_surface_rad(row)
                for row in result.u_norm_applied
            ]
        ),
    )


def test_required_cases_succeed_with_handoff_margins(bank_context) -> None:
    for name, direction in (
        ("bank_w0_left_mild", -1),
        ("bank_w0_right_mild", 1),
    ):
        result = _case_result(bank_context, name)
        lateral_displacement = result.x[-1, 1] - result.x[0, 1]

        assert result.success is True
        assert result.failure_label == "success"
        assert np.sign(lateral_displacement) == direction
        assert abs(lateral_displacement) > 0.05
        assert terminal_glide_entry_proxy(result.x[-1]) is True
        assert terminal_glide_entry_x_margin_m(result.x[-1]) > 0.0
        assert terminal_true_safe_x_margin_m(result.x[-1]) > 0.0
        assert all(check.pass_check for check in result.entry_checks)
        assert all(check.pass_check for check in result.exit_checks if check.required)
        assert all(check.pass_check for check in result.bank_checks if check.required)


def test_boundary_case_is_x_handoff_limited_only(bank_context) -> None:
    boundary = _case_result(bank_context, "bank_w0_0p80_handoff_boundary")

    assert boundary.success is False
    assert boundary.failure_label == "terminal_recovery_limited"
    assert boundary.notes == "glide_entry_x_bound_limited"
    assert boundary.metrics["failure_label"] == "terminal_recovery_limited"
    assert boundary.metrics["notes"] == "glide_entry_x_bound_limited"
    assert np.all(np.isfinite(boundary.x))
    assert terminal_true_safe_x_margin_m(boundary.x[-1]) > 0.0
    assert terminal_glide_entry_proxy(boundary.x[-1]) is False
    assert terminal_glide_entry_x_margin_m(boundary.x[-1]) < 0.0


def test_bank_checks_include_required_handoff_metrics(bank_context) -> None:
    result = _case_result(bank_context, "bank_w0_right_mild")
    check_names = {check.name for check in result.bank_checks}

    for name in (
        "terminal_x_w_m",
        "terminal_true_safe_x_margin_m",
        "terminal_glide_entry_x_margin_m",
        "terminal_glide_entry_proxy",
        "terminal_glide_entry_proxy_source",
    ):
        assert name in check_names


def test_rk4_receives_physical_radian_command(monkeypatch, bank_context) -> None:
    captured: list[np.ndarray] = []

    def fake_rk4_step(
        x,
        delta_cmd_rad,
        dt_s,
        aircraft,
        wind_model,
        wind_mode,
        actuator_tau_s,
    ):
        captured.append(np.asarray(delta_cmd_rad, dtype=float).copy())
        return np.asarray(x, dtype=float).copy()

    monkeypatch.setattr(bank_primitive, "rk4_step", fake_rk4_step)
    config = BankPrimitiveConfig(t_final_s=0.02)
    case = build_bank_cases(config, bank_context["trim"])[1]
    result = rollout_bank_case(
        case,
        config=config,
        aircraft=bank_context["aircraft"],
        wind_model=None,
    )

    assert captured
    np.testing.assert_allclose(captured[0], result.delta_cmd_rad[0])
    assert not np.allclose(captured[0], result.u_norm_requested[0])
