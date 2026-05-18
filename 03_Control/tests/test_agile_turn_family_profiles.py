from __future__ import annotations

import numpy as np
import pytest

from agile_turn_family_comparison import (
    DEFAULT_HORIZON_GRID_S,
    FAMILY_NAMES,
    PLANNED_ESCALATION_HORIZON_GRID_S,
    AgileTurnFamilyConfig,
    family_command_profile,
    family_inventory,
    horizon_grid_s,
    metrics_for_family_candidate,
    phase_labels_for_family,
    replay_family_candidate,
)
from state_contract import STATE_INDEX, STATE_SIZE


RETIRED_LABELS = {
    "dive_perch_redirect_30",
    "reduced_perch_redirect_30",
    "early_unload_recovery_30",
    "speed_collapse_pitch_redirect",
}


def _safe_synthetic_state(sample_count: int, yaw_final_rad: float = 0.10) -> np.ndarray:
    state = np.zeros((sample_count, STATE_SIZE), dtype=float)
    state[:, STATE_INDEX["x_w"]] = np.linspace(1.30, 2.00, sample_count)
    state[:, STATE_INDEX["y_w"]] = 2.20
    state[:, STATE_INDEX["z_w"]] = 1.80
    state[:, STATE_INDEX["psi"]] = np.linspace(0.0, yaw_final_rad, sample_count)
    state[:, STATE_INDEX["u"]] = 5.1
    return state


def test_active_family_inventory_excludes_retired_high_alpha_labels() -> None:
    assert family_inventory() == FAMILY_NAMES
    assert set(family_inventory()) == {
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    }
    assert not (set(family_inventory()) & RETIRED_LABELS)


def test_horizon_grids_are_fixed_step_compatible() -> None:
    assert DEFAULT_HORIZON_GRID_S[15.0] == (0.50, 0.66, 0.80)
    assert DEFAULT_HORIZON_GRID_S[30.0] == (0.76, 0.96, 1.16)
    assert PLANNED_ESCALATION_HORIZON_GRID_S[45.0] == (1.00, 1.20, 1.40)
    assert PLANNED_ESCALATION_HORIZON_GRID_S[60.0] == (1.16, 1.40, 1.66)
    for target in (15.0, 30.0, 45.0, 60.0):
        for horizon_s in horizon_grid_s(target):
            assert np.isclose(horizon_s / 0.02, round(horizon_s / 0.02))


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_family_profiles_are_finite_bounded_and_directional(family_name: str) -> None:
    config_left = AgileTurnFamilyConfig(t_final_s=0.50, target_heading_deg=15.0, direction_sign=1)
    config_right = AgileTurnFamilyConfig(t_final_s=0.50, target_heading_deg=15.0, direction_sign=-1)
    time_s = np.arange(26, dtype=float) * config_left.dt_s

    left = family_command_profile(config_left, time_s, family_name)
    right = family_command_profile(config_right, time_s, family_name)

    assert left.shape == (time_s.size, 3)
    assert np.all(np.isfinite(left))
    assert np.max(np.abs(left)) <= 1.0
    np.testing.assert_allclose(left[:, 0], -right[:, 0])
    np.testing.assert_allclose(left[:, 1], right[:, 1])
    np.testing.assert_allclose(left[:, 2], -right[:, 2])


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_family_phase_labels_include_expected_phases(family_name: str) -> None:
    config = AgileTurnFamilyConfig(t_final_s=0.80, target_heading_deg=15.0)
    time_s = np.arange(41, dtype=float) * config.dt_s
    labels = set(phase_labels_for_family(family_name, time_s, config.t_final_s))

    assert "entry" in labels
    if family_name == "canyon_steep_bank":
        assert {"roll_in", "turn_hold", "heading_capture", "unload_exit"} <= labels
    elif family_name == "wingover_lite":
        assert {"shallow_pre_dive", "climb_roll", "crest_redirect", "unload_descend", "exit_glide"} <= labels
    else:
        assert {"bank_yaw_redirect", "heading_capture", "early_unload", "exit_glide"} <= labels


def test_replay_bridge_clips_requested_commands_and_uses_radian_commands(monkeypatch) -> None:
    captured: list[np.ndarray] = []

    def fake_rk4_step(x, command_rad, dt_s, aircraft, wind_model, wind_mode, actuator_tau_s):
        captured.append(np.asarray(command_rad, dtype=float).copy())
        next_state = np.asarray(x, dtype=float).copy()
        next_state[STATE_INDEX["x_w"]] += next_state[STATE_INDEX["u"]] * float(dt_s)
        return next_state

    monkeypatch.setattr("agile_turn_family_comparison.rk4_step", fake_rk4_step)
    config = AgileTurnFamilyConfig(t_final_s=0.04, target_heading_deg=15.0)
    time_s = np.array([0.0, 0.02, 0.04])
    requested = np.array([[2.0, -2.0, 0.5], [2.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    x0 = _safe_synthetic_state(1)[0]

    _, applied, command_rad = replay_family_candidate(x0, requested, time_s, config, aircraft=object())

    assert np.max(np.abs(applied)) <= 1.0
    assert captured
    assert not np.allclose(captured[0], requested[0])
    np.testing.assert_allclose(captured[0], command_rad[0])


def test_synthetic_underturning_recoverable_candidate_is_horizon_limited() -> None:
    config = AgileTurnFamilyConfig(t_final_s=0.50, target_heading_deg=30.0)
    time_s = np.arange(26, dtype=float) * config.dt_s
    state = _safe_synthetic_state(time_s.size, yaw_final_rad=np.deg2rad(8.0))
    requested = np.zeros((time_s.size, 3), dtype=float)
    applied = np.zeros_like(requested)
    phase = tuple("exit_glide" for _ in time_s)

    metrics = metrics_for_family_candidate(
        config,
        "canyon_steep_bank",
        time_s,
        state,
        requested,
        applied,
        phase,
    )

    assert metrics["recoverable"] is True
    assert metrics["useful_recoverable_candidate"] is False
    assert metrics["horizon_limited"] is True
    assert metrics["failure_label"] == "under_turning"
