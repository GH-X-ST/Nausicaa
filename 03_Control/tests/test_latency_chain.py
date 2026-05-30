from __future__ import annotations

import numpy as np
import pytest

from latency import (
    actuator_tau_for_case,
    delayed_command_sample,
    delayed_state_sample,
    latency_adjusted_command_sample,
    latency_case_config,
)


def _step_commands() -> tuple[np.ndarray, np.ndarray]:
    times_s = np.arange(0.0, 0.42, 0.02)
    commands = np.zeros((times_s.size, 3))
    commands[times_s >= 0.10, 0] = 1.0
    return times_s, commands


def test_delayed_state_sample_interpolates_history() -> None:
    times_s = np.array([0.0, 0.1, 0.2])
    states = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

    assert np.allclose(delayed_state_sample(times_s, states, 0.05), [0.5, 1.0])
    assert np.allclose(delayed_state_sample(times_s, states, -1.0), [0.0, 0.0])
    assert np.allclose(delayed_state_sample(times_s, states, 1.0), [2.0, 4.0])


def test_delayed_command_sample_uses_zero_order_hold() -> None:
    times_s = np.array([0.0, 0.1, 0.2])
    commands = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert np.allclose(
        delayed_command_sample(times_s, commands, 0.05),
        [0.0, 0.0, 0.0],
    )
    assert np.allclose(
        delayed_command_sample(times_s, commands, 0.15),
        [0.5, 0.0, 0.0],
    )
    assert np.allclose(
        delayed_command_sample(times_s, commands, 0.30),
        [1.0, 0.0, 0.0],
    )


def test_latency_adjusted_command_sample_keeps_command_path_delay_separate() -> None:
    times_s, commands = _step_commands()
    nominal = latency_case_config("nominal")

    assert np.allclose(
        latency_adjusted_command_sample(times_s, commands, 0.16, nominal),
        [0.0, 0.0, 0.0],
    )
    assert np.allclose(
        latency_adjusted_command_sample(times_s, commands, 0.18, nominal),
        [1.0, 0.0, 0.0],
    )


def test_latency_adjusted_command_sample_returns_executable_lattice() -> None:
    times_s = np.array([0.0, 0.1])
    commands = np.array([[0.13, -0.29, 0.91], [0.0, 0.0, 0.0]])

    sample = latency_adjusted_command_sample(
        times_s,
        commands,
        0.16,
        latency_case_config("nominal"),
    )

    assert np.allclose(sample, [0.2, -0.2, 1.0])


def test_actuator_tau_for_case_preserves_tau_semantics() -> None:
    fallback_tau = (0.06, 0.07, 0.08)

    assert actuator_tau_for_case(latency_case_config("none"), fallback_tau) == pytest.approx(
        fallback_tau
    )
    assert actuator_tau_for_case(
        latency_case_config("actuator_lag_only"),
        fallback_tau,
    ) == pytest.approx(fallback_tau)

    nominal_tau = actuator_tau_for_case(latency_case_config("nominal"), fallback_tau)
    conservative_tau = actuator_tau_for_case(
        latency_case_config("conservative"),
        fallback_tau,
    )
    assert all(value > 0.0 for value in nominal_tau)
    assert all(value > nominal_tau[0] for value in conservative_tau)
