from __future__ import annotations

import numpy as np
import pytest

import glide_primitive
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
    surface_rad_to_normalised_command,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from glide_primitive import (
    GLIDE_CAMPAIGN,
    PRIMITIVE_FAMILY,
    GlideFeedbackGains,
    GlidePrimitiveConfig,
    build_glide_primitive_spec,
    glide_feedback_command_norm,
    initial_glide_state,
    rollout_glide_primitive,
    terminal_recoverable_proxy,
)
from trim_solver import TrimTarget, solve_straight_trim


@pytest.fixture(scope="module")
def glide_context():
    config = GlidePrimitiveConfig()
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
    x0 = initial_glide_state(config, trim)
    result = rollout_glide_primitive(
        x0,
        config=config,
        aircraft=aircraft,
        wind_model=None,
    )
    return {
        "config": config,
        "aircraft": aircraft,
        "trim": trim,
        "x0": x0,
        "result": result,
    }


def test_glide_spec_is_first_actual_w0_glide_metadata() -> None:
    spec = build_glide_primitive_spec(GlidePrimitiveConfig())

    assert GLIDE_CAMPAIGN == "04_glide_primitive"
    assert spec.family == PRIMITIVE_FAMILY
    assert spec.metadata["actual_glide_primitive_implemented"] == "true"
    assert spec.metadata["controller_type"] == "trim_hold_pr_feedback"
    assert spec.metadata["wind_mode"] == "none"
    assert {check.name for check in spec.exit_checks} == {
        "finite_state",
        "true_safe_margin",
        "rollout_success",
    }


def test_feedback_adds_correction_in_normalised_command_space(glide_context) -> None:
    trim = glide_context["trim"]
    x_ref = glide_context["x0"]
    u_trim_norm = surface_rad_to_normalised_command(trim.u_cmd_trim)

    requested = glide_feedback_command_norm(
        x_ref,
        x_ref,
        u_trim_norm,
        GlideFeedbackGains(),
    )

    np.testing.assert_allclose(requested, u_trim_norm)
    assert not np.allclose(requested, trim.u_cmd_trim)


def test_default_glide_rollout_succeeds_and_uses_interface_checks(glide_context) -> None:
    result = glide_context["result"]

    assert result.success is True
    assert result.failure_label == "success"
    assert all(check.pass_check for check in result.entry_checks)
    assert all(check.pass_check for check in result.exit_checks if check.required)
    assert all(check.pass_check for check in result.glide_checks if check.required)
    assert result.metrics["finite_state_success"] is True
    assert result.metrics["rollout_success"] is True
    assert result.metrics["primitive_success"] is True
    assert result.metrics["success"] is True


def test_closed_loop_command_bridge_values_are_consistent(glide_context) -> None:
    result = glide_context["result"]

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


def test_terminal_recoverable_proxy_is_not_unconditional(glide_context) -> None:
    result = glide_context["result"]
    terminal = result.x[-1].copy()

    assert terminal_recoverable_proxy(terminal, result.primitive_spec) is True
    terminal[6:9] = [3.0, 0.0, 0.0]
    assert terminal_recoverable_proxy(terminal, result.primitive_spec) is False


def test_rk4_receives_physical_radian_command(monkeypatch, glide_context) -> None:
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

    monkeypatch.setattr(glide_primitive, "rk4_step", fake_rk4_step)
    config = GlidePrimitiveConfig(t_final_s=0.02)
    result = rollout_glide_primitive(
        glide_context["x0"],
        config=config,
        aircraft=glide_context["aircraft"],
        wind_model=None,
    )

    assert captured
    np.testing.assert_allclose(captured[0], result.delta_cmd_rad[0])
    assert not np.allclose(captured[0], result.u_norm_requested[0])
