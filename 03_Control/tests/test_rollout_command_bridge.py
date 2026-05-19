from __future__ import annotations

import numpy as np
import pytest

import rollout
from command_contract import normalised_command_to_surface_rad
from rollout import (
    CommandSchedule,
    RolloutConfig,
    make_constant_command_schedule,
    rk4_step,
    rollout_open_loop_normalised,
    sample_command_schedule,
    validate_rollout_config,
)


def _state() -> np.ndarray:
    state = np.zeros(15)
    state[0:3] = [2.5, 2.2, 1.5]
    state[6] = 6.5
    return state


def test_rollout_config_rejects_non_multiple_timing() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        validate_rollout_config(RolloutConfig(dt_s=0.02, t_final_s=0.25))


def test_constant_schedule_and_zero_order_hold_sampling() -> None:
    schedule = make_constant_command_schedule([0.1, -0.2, 0.3], 0.24, 0.02)

    assert schedule.times_s.shape == (13,)
    assert schedule.u_norm_requested.shape == (13, 3)
    assert np.allclose(sample_command_schedule(schedule, 0.00), [0.1, -0.2, 0.3])
    assert np.allclose(sample_command_schedule(schedule, 0.13), [0.1, -0.2, 0.3])


def test_rollout_clips_requested_command_before_radian_conversion(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    def fake_state_derivative(
        x: np.ndarray,
        u_cmd: np.ndarray,
        aircraft: object,
        wind_model: object = None,
        rho: float = 1.225,
        actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
        wind_mode: str = "panel",
    ) -> np.ndarray:
        del aircraft, wind_model, rho, actuator_tau_s, wind_mode
        calls.append(np.asarray(u_cmd, dtype=float).copy())
        derivative = np.zeros(15)
        derivative[0] = float(x[6])
        return derivative

    monkeypatch.setattr(rollout, "state_derivative", fake_state_derivative)
    config = RolloutConfig(dt_s=0.02, t_final_s=0.04, wind_mode="none")
    schedule = make_constant_command_schedule([2.0, 0.0, 0.0], 0.04, 0.02)
    result = rollout_open_loop_normalised(
        _state(),
        schedule,
        config,
        aircraft=object(),
        wind_model=None,
    )
    expected = normalised_command_to_surface_rad(np.array([1.0, 0.0, 0.0]))

    assert np.allclose(result.u_norm_requested[:, 0], 2.0)
    assert np.allclose(result.u_norm_applied[:, 0], 1.0)
    assert np.allclose(result.delta_cmd_rad, expected)
    assert calls
    assert all(np.allclose(call, expected) for call in calls)
    assert not any(np.allclose(call, [2.0, 0.0, 0.0]) for call in calls)


def test_sampled_schedule_accepts_time_varying_requested_commands() -> None:
    schedule = CommandSchedule(
        times_s=np.array([0.0, 0.1, 0.2]),
        u_norm_requested=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ]
        ),
    )

    assert np.allclose(sample_command_schedule(schedule, 0.05), [0.0, 0.0, 0.0])
    assert np.allclose(sample_command_schedule(schedule, 0.15), [0.5, 0.0, 0.0])


def test_command_delay_does_not_count_as_saturation(monkeypatch) -> None:
    def fake_state_derivative(
        x: np.ndarray,
        u_cmd: np.ndarray,
        aircraft: object,
        wind_model: object = None,
        rho: float = 1.225,
        actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
        wind_mode: str = "panel",
    ) -> np.ndarray:
        del x, u_cmd, aircraft, wind_model, rho, actuator_tau_s, wind_mode
        return np.zeros(15)

    monkeypatch.setattr(rollout, "state_derivative", fake_state_derivative)
    times_s = np.arange(0.0, 0.22, 0.02)
    commands = np.zeros((times_s.size, 3))
    commands[times_s >= 0.10, 0] = 1.0
    result = rollout_open_loop_normalised(
        _state(),
        CommandSchedule(times_s=times_s, u_norm_requested=commands),
        RolloutConfig(
            dt_s=0.02,
            t_final_s=0.20,
            wind_mode="none",
            latency_case="nominal",
        ),
        aircraft=object(),
        wind_model=None,
    )

    assert result.metrics["saturation_fraction"] == 0.0
    assert result.metrics["saturation_time_s"] == 0.0


def test_rk4_step_uses_physical_command_for_actuator_state(monkeypatch) -> None:
    def fake_state_derivative(
        x: np.ndarray,
        u_cmd: np.ndarray,
        aircraft: object,
        wind_model: object = None,
        rho: float = 1.225,
        actuator_tau_s: tuple[float, float, float] = (1.0, 1.0, 1.0),
        wind_mode: str = "panel",
    ) -> np.ndarray:
        del aircraft, wind_model, rho, actuator_tau_s, wind_mode
        derivative = np.zeros(15)
        derivative[12:15] = np.asarray(u_cmd) - np.asarray(x)[12:15]
        return derivative

    monkeypatch.setattr(rollout, "state_derivative", fake_state_derivative)
    command_rad = np.array([0.1, -0.2, 0.3])

    next_state = rk4_step(
        np.zeros(15),
        command_rad,
        0.1,
        aircraft=object(),
        wind_model=None,
        wind_mode="none",
        actuator_tau_s=(1.0, 1.0, 1.0),
    )

    assert np.all(np.sign(next_state[12:15]) == np.sign(command_rad))
    assert np.all(np.abs(next_state[12:15]) < np.abs(command_rad))
