from __future__ import annotations

import numpy as np
import pytest

import rollout
from command_contract import normalised_command_to_surface_rad
from latency import (
    delayed_command_sample,
    delayed_state_sample,
    latency_adjusted_command_sample,
    latency_case_config,
)
from primitive_library_generators import generate_command_profile
from primitive_library_schema import PrimitiveCandidateSpec
from rollout import CommandSchedule, RolloutConfig, rollout_open_loop_normalised


def _safe_state() -> np.ndarray:
    state = np.zeros(15)
    state[0:3] = [2.5, 2.2, 1.5]
    state[6] = 6.5
    return state


def _step_schedule() -> CommandSchedule:
    times_s = np.arange(0.0, 0.42, 0.02)
    commands = np.zeros((times_s.size, 3))
    commands[times_s >= 0.10, 0] = 1.0
    return CommandSchedule(times_s=times_s, u_norm_requested=commands)


def _surface_derivative_only(
    x: np.ndarray,
    u_cmd: np.ndarray,
    aircraft: object,
    wind_model: object = None,
    rho: float = 1.225,
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
    wind_mode: str = "panel",
) -> np.ndarray:
    del aircraft, wind_model, rho, wind_mode
    derivative = np.zeros(15)
    derivative[12:15] = (
        np.asarray(u_cmd, dtype=float) - np.asarray(x, dtype=float)[12:15]
    ) / np.asarray(actuator_tau_s, dtype=float)
    return derivative


def _first_surface_crossing(
    time_s: np.ndarray,
    delta_a: np.ndarray,
    threshold: float,
) -> float:
    indices = np.flatnonzero(np.asarray(delta_a) >= float(threshold))
    if indices.size == 0:
        return float("inf")
    return float(time_s[int(indices[0])])


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


def test_latency_adjusted_command_sample_operates_on_normalised_commands() -> None:
    schedule = _step_schedule()
    nominal = latency_case_config("nominal")

    assert np.allclose(
        latency_adjusted_command_sample(
            schedule.times_s,
            schedule.u_norm_requested,
            0.16,
            nominal,
        ),
        [0.0, 0.0, 0.0],
    )
    assert np.allclose(
        latency_adjusted_command_sample(
            schedule.times_s,
            schedule.u_norm_requested,
            0.18,
            nominal,
        ),
        [1.0, 0.0, 0.0],
    )


def test_rollout_surface_response_ordering_for_latency_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout, "state_derivative", _surface_derivative_only)
    schedule = _step_schedule()
    target_delta_a = normalised_command_to_surface_rad(np.array([1.0, 0.0, 0.0]))[0]
    threshold = 0.05 * target_delta_a
    crossings: dict[str, float] = {}

    for latency_case in ("none", "actuator_lag_only", "nominal", "conservative"):
        result = rollout_open_loop_normalised(
            _safe_state(),
            schedule,
            RolloutConfig(
                dt_s=0.02,
                t_final_s=0.40,
                wind_mode="none",
                latency_case=latency_case,
            ),
            aircraft=object(),
            wind_model=None,
        )
        crossings[latency_case] = _first_surface_crossing(
            result.time_s,
            result.x[:, 12],
            threshold,
        )

    assert crossings["none"] == pytest.approx(0.10)
    assert crossings["none"] < crossings["actuator_lag_only"]
    assert crossings["actuator_lag_only"] < crossings["nominal"]
    assert crossings["nominal"] < crossings["conservative"]


def test_latency_helper_can_be_applied_to_primitive_command_profile() -> None:
    spec = PrimitiveCandidateSpec(
        primitive_id="test_mild_bank",
        parent_primitive_id="mild_bank_none",
        variant_id="test_mild_bank",
        family="mild_bank",
        target_heading_deg=None,
        updraft_config="none",
        wind_fidelity="W0",
        start_condition="favourable",
        direction_sign=1,
    )
    time_s = np.arange(0.0, 0.22, 0.02)
    commands, _ = generate_command_profile(spec, time_s)

    delayed = latency_adjusted_command_sample(
        time_s,
        commands,
        0.12,
        latency_case_config("nominal"),
    )

    assert delayed.shape == (3,)
    assert np.all(np.isfinite(delayed))
    assert np.all(delayed <= 1.0)
    assert np.all(delayed >= -1.0)
