from __future__ import annotations

import numpy as np
import pytest

import rollout
from command_contract import clip_normalised_command, normalised_command_to_surface_rad
from primitive_contract import PrimitiveExitCheck, PrimitiveSpec
from primitive_interface import (
    SMOKE_NOTES,
    PrimitiveExecutionConfig,
    build_interface_smoke_spec,
    evaluate_entry_set,
    evaluate_exit_checks,
    execute_open_loop_primitive_interface,
    state_entry_variables,
)
from rollout import CommandSchedule, RolloutConfig, RolloutResult, make_constant_command_schedule


def _smoke_state() -> np.ndarray:
    state = np.zeros(15)
    state[0:3] = [2.5, 2.2, 1.5]
    state[6] = 6.5
    return state


def _simple_rollout_result(x_log: np.ndarray) -> RolloutResult:
    sample_count = x_log.shape[0]
    commands = np.zeros((sample_count, 3))
    metrics = {
        "rollout_success": bool(np.all(np.isfinite(x_log))),
    }
    return RolloutResult(
        time_s=np.arange(sample_count, dtype=float) * 0.02,
        x=x_log,
        u_norm_requested=commands,
        u_norm_applied=commands,
        delta_cmd_rad=commands,
        success=False,
        failure_label="not_run",
        metrics=metrics,
        notes="test_rollout",
    )


def test_smoke_spec_is_valid_glide_placeholder() -> None:
    spec = build_interface_smoke_spec()

    assert spec.name == "glide_interface_smoke_contract"
    assert spec.family == "glide"
    assert spec.metadata["actual_glide_primitive_implemented"] == "false"
    assert {check.name for check in spec.exit_checks} == {
        "finite_state",
        "true_safe_margin",
        "rollout_success",
    }


def test_state_entry_variables_include_direct_and_derived_values() -> None:
    variables = state_entry_variables(_smoke_state())

    for name in (
        "x_w",
        "y_w",
        "z_w",
        "phi",
        "theta",
        "psi",
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "delta_a",
        "delta_e",
        "delta_r",
        "speed_m_s",
        "alpha_rad",
        "beta_rad",
        "bank_rad",
        "pitch_rad",
        "yaw_rad",
    ):
        assert name in variables
    assert variables["speed_m_s"] == 6.5
    assert variables["alpha_rad"] == 0.0
    assert variables["beta_rad"] == 0.0


def test_entry_set_passes_for_default_smoke_state() -> None:
    spec = build_interface_smoke_spec()
    checks = evaluate_entry_set(_smoke_state(), spec.entry_set)

    assert checks
    assert all(check.pass_check for check in checks)


def test_entry_set_fails_cleanly_for_speed_or_position_outside_bounds() -> None:
    spec = build_interface_smoke_spec()
    state = _smoke_state()
    state[6] = 3.0
    speed_checks = evaluate_entry_set(state, spec.entry_set)

    assert any(
        check.variable == "speed_m_s" and not check.pass_check
        for check in speed_checks
    )

    state = _smoke_state()
    state[0] = 7.0
    position_checks = evaluate_entry_set(state, spec.entry_set)
    assert any(
        check.variable == "x_w" and not check.pass_check
        for check in position_checks
    )


def test_unknown_entry_set_variable_raises_value_error() -> None:
    spec = build_interface_smoke_spec()
    bad_entry = type(spec.entry_set)(
        name="bad",
        description="bad",
        lower={**spec.entry_set.lower, "unknown": 0.0},
        upper={**spec.entry_set.upper, "unknown": 1.0},
    )

    with pytest.raises(ValueError, match="unknown entry-set variable"):
        evaluate_entry_set(_smoke_state(), bad_entry)


def test_exit_checks_use_full_history_and_rollout_metric() -> None:
    spec = build_interface_smoke_spec()
    x_log = np.vstack([_smoke_state(), _smoke_state()])
    result = _simple_rollout_result(x_log)

    checks = evaluate_exit_checks(spec, result)

    assert all(check.pass_check for check in checks)
    assert {
        "finite_state",
        "true_safe_margin",
        "rollout_success",
    } == {check.name for check in checks}

    unsafe_log = x_log.copy()
    unsafe_log[0, 0] = 0.5
    unsafe_result = _simple_rollout_result(unsafe_log)
    unsafe_checks = evaluate_exit_checks(spec, unsafe_result)
    margin_check = next(check for check in unsafe_checks if check.name == "true_safe_margin")
    assert margin_check.pass_check is False

    nonfinite_log = x_log.copy()
    nonfinite_log[1, 6] = np.nan
    nonfinite_result = _simple_rollout_result(nonfinite_log)
    finite_check = next(
        check for check in evaluate_exit_checks(spec, nonfinite_result)
        if check.name == "finite_state"
    )
    assert finite_check.pass_check is False


def test_unknown_required_exit_check_fails() -> None:
    base = build_interface_smoke_spec()
    spec = PrimitiveSpec(
        name=base.name,
        family=base.family,
        duration_s=base.duration_s,
        entry_set=base.entry_set,
        exit_checks=(
            PrimitiveExitCheck(
                name="unknown_required",
                description="unknown required check",
                required=True,
            ),
        ),
        metadata=base.metadata,
    )
    x_log = np.vstack([_smoke_state(), _smoke_state()])
    checks = evaluate_exit_checks(spec, _simple_rollout_result(x_log))

    assert checks[0].name == "unknown_required"
    assert checks[0].pass_check is False
    assert checks[0].reason == "unknown_exit_check"


def test_execution_metric_success_remains_false_for_smoke() -> None:
    config = PrimitiveExecutionConfig(dt_s=0.02, t_final_s=0.24)
    schedule = make_constant_command_schedule(np.zeros(3), 0.24, 0.02)

    result = execute_open_loop_primitive_interface(
        build_interface_smoke_spec(),
        _smoke_state(),
        schedule,
        config,
        aircraft=None,
        wind_model=None,
    )

    assert result.entry_pass is True
    assert result.exit_pass is True
    assert result.rollout_result is not None
    assert result.metrics["finite_state_success"] is True
    assert result.metrics["rollout_success"] is True
    assert result.metrics["primitive_success"] is False
    assert result.metrics["success"] is False
    assert result.metrics["failure_label"] == "not_run"
    assert result.metrics["notes"] == SMOKE_NOTES


def test_rollout_bridge_passes_radian_commands_to_state_derivative(monkeypatch) -> None:
    requested = np.array([0.5, -0.5, 0.25])
    expected_rad = normalised_command_to_surface_rad(
        clip_normalised_command(requested)
    )
    calls: list[np.ndarray] = []

    def fake_state_derivative(
        x,
        u_cmd,
        aircraft,
        wind_model=None,
        rho=1.225,
        actuator_tau_s=(0.06, 0.06, 0.06),
        wind_mode="none",
    ):
        del aircraft, wind_model, rho, actuator_tau_s, wind_mode
        calls.append(np.asarray(u_cmd, dtype=float).copy())
        return np.zeros_like(np.asarray(x, dtype=float))

    monkeypatch.setattr(rollout, "state_derivative", fake_state_derivative)
    config = PrimitiveExecutionConfig(dt_s=0.02, t_final_s=0.02)
    schedule = CommandSchedule(
        times_s=np.array([0.0, 0.02]),
        u_norm_requested=np.vstack([requested, requested]),
    )

    result = execute_open_loop_primitive_interface(
        build_interface_smoke_spec(),
        _smoke_state(),
        schedule,
        config,
        aircraft=object(),
        wind_model=None,
    )

    assert result.rollout_result is not None
    assert calls
    for command in calls:
        assert np.allclose(command, expected_rad)
        assert not np.allclose(command, requested)
