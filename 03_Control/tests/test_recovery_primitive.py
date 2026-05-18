from __future__ import annotations

import numpy as np
import pytest

import recovery_primitive
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
    surface_rad_to_normalised_command,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from primitive_interface import evaluate_entry_set
from recovery_primitive import (
    GLIDE_PROXY_SOURCE,
    PRIMITIVE_FAMILY,
    RECOVERY_CAMPAIGN,
    RecoveryFeedbackGains,
    RecoveryPrimitiveConfig,
    build_recovery_cases,
    build_recovery_primitive_spec,
    recovery_feedback_command_norm,
    rollout_recovery_case,
    run_recovery_batch,
    terminal_glide_entry_proxy,
    terminal_glide_entry_x_margin_m,
)
from trim_solver import TrimTarget, solve_straight_trim


@pytest.fixture(scope="module")
def recovery_context():
    config = RecoveryPrimitiveConfig()
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
    cases = build_recovery_cases(config, trim)
    batch = run_recovery_batch(config=config, aircraft=aircraft)
    return {
        "config": config,
        "aircraft": aircraft,
        "trim": trim,
        "cases": cases,
        "batch": batch,
    }


def _case_result(recovery_context, name: str):
    for result in recovery_context["batch"].case_results:
        if result.case_spec.name == name:
            return result
    raise AssertionError(f"case not found: {name}")


def test_recovery_spec_family_and_entry_geometry_bounds() -> None:
    spec = build_recovery_primitive_spec(RecoveryPrimitiveConfig())

    assert RECOVERY_CAMPAIGN == "05_recovery_primitive"
    assert spec.family == PRIMITIVE_FAMILY
    assert spec.metadata["actual_recovery_primitive_implemented"] == "true"
    assert spec.metadata["terminal_glide_entry_proxy_source"] == GLIDE_PROXY_SOURCE
    for variable in ("x_w", "y_w", "z_w", "speed_m_s", "alpha_rad", "beta_rad"):
        assert variable in spec.entry_set.lower
        assert variable in spec.entry_set.upper
    assert spec.entry_set.upper["x_w"] == pytest.approx(1.35)


def test_recovery_case_inventory_includes_boundary_case(recovery_context) -> None:
    cases = recovery_context["cases"]

    assert [case.name for case in cases] == [
        "recovery_w0_moderate_attitude_rate",
        "recovery_w0_low_speed_pitch_up",
        "recovery_w0_sideslip_yaw_rate",
        "recovery_w0_high_rate_boundary",
        "recovery_w0_0p80_boundary",
    ]
    roles = {case.name: case.role for case in cases}
    assert roles["recovery_w0_moderate_attitude_rate"] == "required"
    assert roles["recovery_w0_low_speed_pitch_up"] == "optional"
    assert roles["recovery_w0_0p80_boundary"] == "diagnostic"
    assert cases[-1].t_final_s == pytest.approx(0.80)


def test_entry_set_geometry_passes_and_fails(recovery_context) -> None:
    spec = build_recovery_primitive_spec(recovery_context["config"])
    required = recovery_context["cases"][0]
    bad_state = required.x0.copy()
    bad_state[0] = 1.6

    pass_checks = evaluate_entry_set(required.x0, spec.entry_set)
    fail_checks = evaluate_entry_set(bad_state, spec.entry_set)

    assert all(check.pass_check for check in pass_checks)
    assert any(
        check.variable == "x_w" and not check.pass_check
        for check in fail_checks
    )


def test_feedback_uses_normalised_trim_command_space(recovery_context) -> None:
    trim = recovery_context["trim"]
    x_ref = trim.x_trim
    u_trim_norm = surface_rad_to_normalised_command(trim.u_cmd_trim)

    requested = recovery_feedback_command_norm(
        x_ref,
        x_ref,
        u_trim_norm,
        RecoveryFeedbackGains(),
    )

    np.testing.assert_allclose(requested, u_trim_norm)
    assert not np.allclose(requested, trim.u_cmd_trim)


def test_command_bridge_consistency(recovery_context) -> None:
    result = _case_result(recovery_context, "recovery_w0_moderate_attitude_rate")

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


def test_terminal_glide_entry_proxy_and_x_margin(recovery_context) -> None:
    required = _case_result(recovery_context, "recovery_w0_moderate_attitude_rate")
    boundary = _case_result(recovery_context, "recovery_w0_0p80_boundary")

    assert terminal_glide_entry_proxy(required.x[-1]) is True
    assert terminal_glide_entry_x_margin_m(required.x[-1]) > 0.0
    assert terminal_glide_entry_proxy(boundary.x[-1]) is False
    assert terminal_glide_entry_x_margin_m(boundary.x[-1]) < 0.0


def test_required_case_succeeds(recovery_context) -> None:
    required = _case_result(recovery_context, "recovery_w0_moderate_attitude_rate")

    assert required.success is True
    assert required.failure_label == "success"
    assert required.metrics["primitive_success"] is True
    assert required.metrics["success"] is True
    assert all(check.pass_check for check in required.entry_checks)
    assert all(check.pass_check for check in required.exit_checks if check.required)
    assert all(check.pass_check for check in required.recovery_checks if check.required)


def test_0p80_boundary_is_handoff_limited(recovery_context) -> None:
    boundary = _case_result(recovery_context, "recovery_w0_0p80_boundary")

    assert boundary.success is False
    assert boundary.failure_label == "terminal_recovery_limited"
    assert boundary.notes == "glide_entry_x_bound_limited"
    assert boundary.metrics["failure_label"] == "terminal_recovery_limited"
    assert boundary.metrics["notes"] == "glide_entry_x_bound_limited"


def test_rk4_receives_physical_radian_command(monkeypatch, recovery_context) -> None:
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

    monkeypatch.setattr(recovery_primitive, "rk4_step", fake_rk4_step)
    config = RecoveryPrimitiveConfig(t_final_s=0.02)
    case = build_recovery_cases(config, recovery_context["trim"])[0]
    result = rollout_recovery_case(
        case,
        config=config,
        aircraft=recovery_context["aircraft"],
        wind_model=None,
    )

    assert captured
    np.testing.assert_allclose(captured[0], result.delta_cmd_rad[0])
    assert not np.allclose(captured[0], result.u_norm_requested[0])
