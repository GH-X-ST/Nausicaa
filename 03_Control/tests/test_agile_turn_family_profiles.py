from __future__ import annotations

import numpy as np
import pytest

from agile_turn_family_comparison import (
    CANDIDATE_CLASSES,
    FAMILY_NAMES,
    TARGET_HORIZON_GRID_S,
    AgileTurnFamilyConfig,
    candidate_ranking_key,
    compare_agile_turn_families,
    family_command_profile,
    family_inventory,
    heading_accuracy_metrics,
    horizon_grid_s,
    metrics_for_family_candidate,
    phase_labels_for_family,
    replay_family_candidate,
    target_ladder_deg,
)
from state_contract import STATE_INDEX, STATE_SIZE


def _safe_state_history(
    sample_count: int,
    terminal_heading_deg: float,
    speed_m_s: float = 6.0,
) -> np.ndarray:
    state = np.zeros((sample_count, STATE_SIZE), dtype=float)
    state[:, STATE_INDEX["x_w"]] = np.linspace(1.50, 2.00, sample_count)
    state[:, STATE_INDEX["y_w"]] = 2.20
    state[:, STATE_INDEX["z_w"]] = 1.80
    state[:, STATE_INDEX["psi"]] = np.linspace(0.0, np.deg2rad(terminal_heading_deg), sample_count)
    state[:, STATE_INDEX["u"]] = speed_m_s
    return state


def _zero_commands(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    requested = np.zeros((sample_count, 3), dtype=float)
    applied = np.zeros_like(requested)
    return requested, applied


def test_active_family_inventory_and_ladder_are_exact() -> None:
    assert family_inventory() == FAMILY_NAMES
    assert set(family_inventory()) == {
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    }
    assert target_ladder_deg() == (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
    assert set(CANDIDATE_CLASSES) == {
        "commandable_target_candidate",
        "accurate_boundary_evidence",
        "safe_partial_turn_evidence",
        "unsafe_or_nonrecoverable_boundary",
    }


def test_horizon_grids_match_precision_ladder_and_fixed_step() -> None:
    assert TARGET_HORIZON_GRID_S[15.0] == (0.36, 0.46, 0.60, 0.76, 0.90)
    assert TARGET_HORIZON_GRID_S[30.0] == (0.46, 0.60, 0.80, 1.00, 1.20)
    assert TARGET_HORIZON_GRID_S[180.0] == (1.60, 2.20, 2.80, 3.40, 4.00)
    for target in target_ladder_deg():
        for horizon_s in horizon_grid_s(target):
            assert np.isclose(horizon_s / 0.02, round(horizon_s / 0.02))


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_family_profiles_are_finite_bounded_and_directional(family_name: str) -> None:
    config_left = AgileTurnFamilyConfig(t_final_s=0.60, target_heading_deg=15.0, direction_sign=1)
    config_right = AgileTurnFamilyConfig(t_final_s=0.60, target_heading_deg=15.0, direction_sign=-1)
    time_s = np.arange(31, dtype=float) * config_left.dt_s

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
    config = AgileTurnFamilyConfig(t_final_s=0.90, target_heading_deg=15.0)
    time_s = np.arange(46, dtype=float) * config.dt_s
    labels = set(phase_labels_for_family(family_name, time_s, config.t_final_s))

    assert "entry" in labels
    if family_name == "canyon_steep_bank":
        assert {"roll_in", "turn_hold", "heading_capture", "unload_exit"} <= labels
    elif family_name == "wingover_lite":
        assert {"shallow_pre_dive", "climb_roll", "crest_redirect", "unload_descend", "exit_glide"} <= labels
    else:
        assert {"bank_yaw_redirect", "heading_capture", "early_unload", "exit_glide"} <= labels


def test_heading_accuracy_uses_terminal_unwrapped_yaw() -> None:
    state = _safe_state_history(5, terminal_heading_deg=30.0)
    state[2, STATE_INDEX["psi"]] = np.deg2rad(40.0)

    metrics = heading_accuracy_metrics(state, direction_sign=1, target_heading_deg=30.0)

    assert metrics["terminal_heading_change_deg"] == pytest.approx(30.0)
    assert metrics["peak_heading_change_deg"] == pytest.approx(40.0)
    assert metrics["heading_band_pass"] is True


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
    x0 = _safe_state_history(1, terminal_heading_deg=0.0)[0]

    _, applied, command_rad = replay_family_candidate(x0, requested, time_s, config, aircraft=object())

    assert np.max(np.abs(applied)) <= 1.0
    assert captured
    assert not np.allclose(captured[0], requested[0])
    np.testing.assert_allclose(captured[0], command_rad[0])


def test_safe_wrong_heading_for_30deg_is_partial_evidence_only() -> None:
    config = AgileTurnFamilyConfig(t_final_s=0.60, target_heading_deg=30.0)
    time_s = np.arange(31, dtype=float) * config.dt_s
    state = _safe_state_history(time_s.size, terminal_heading_deg=18.7)
    requested, applied = _zero_commands(time_s.size)
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

    assert metrics["candidate_class"] == "safe_partial_turn_evidence"
    assert metrics["commandable_target_candidate"] is False
    assert metrics["heading_band_pass"] is False
    assert metrics["active_limiting_mechanism"] == "under_turning_target_miss"


def test_commandable_candidate_requires_heading_speed_exposure_safety_and_recovery() -> None:
    config = AgileTurnFamilyConfig(t_final_s=0.60, target_heading_deg=30.0)
    time_s = np.arange(31, dtype=float) * config.dt_s
    state = _safe_state_history(time_s.size, terminal_heading_deg=30.0)
    requested, applied = _zero_commands(time_s.size)
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

    assert metrics["candidate_class"] == "commandable_target_candidate"
    assert metrics["commandable_target_candidate"] is True
    assert metrics["heading_band_pass"] is True
    assert metrics["true_safe_trajectory"] is True
    assert metrics["recoverable"] is True


def test_shorter_commandable_candidate_ranks_higher_than_longer_one() -> None:
    config_short = AgileTurnFamilyConfig(t_final_s=0.60, target_heading_deg=30.0)
    config_long = AgileTurnFamilyConfig(t_final_s=1.20, target_heading_deg=30.0)
    short_time = np.arange(31, dtype=float) * config_short.dt_s
    long_time = np.arange(61, dtype=float) * config_long.dt_s
    short_state = _safe_state_history(short_time.size, terminal_heading_deg=30.0)
    long_state = _safe_state_history(long_time.size, terminal_heading_deg=30.0)
    short_requested, short_applied = _zero_commands(short_time.size)
    long_requested, long_applied = _zero_commands(long_time.size)

    short_metrics = metrics_for_family_candidate(
        config_short,
        "canyon_steep_bank",
        short_time,
        short_state,
        short_requested,
        short_applied,
        tuple("exit_glide" for _ in short_time),
    )
    long_metrics = metrics_for_family_candidate(
        config_long,
        "canyon_steep_bank",
        long_time,
        long_state,
        long_requested,
        long_applied,
        tuple("exit_glide" for _ in long_time),
    )

    assert candidate_ranking_key(short_metrics) > candidate_ranking_key(long_metrics)


def test_compare_returns_no_selection_for_zero_turn_search(monkeypatch) -> None:
    def fake_rk4_step(x, command_rad, dt_s, aircraft, wind_model, wind_mode, actuator_tau_s):
        next_state = np.asarray(x, dtype=float).copy()
        next_state[STATE_INDEX["x_w"]] += next_state[STATE_INDEX["u"]] * float(dt_s)
        return next_state

    monkeypatch.setattr("agile_turn_family_comparison.rk4_step", fake_rk4_step)
    monkeypatch.setattr("agile_turn_family_comparison.horizon_grid_s", lambda target: (0.04,))
    config = AgileTurnFamilyConfig(t_final_s=0.04, target_heading_deg=30.0)
    x0 = _safe_state_history(1, terminal_heading_deg=0.0)[0]

    result = compare_agile_turn_families(config, x0=x0, aircraft=object())

    assert result.selected_candidate is None
    assert result.selected_family == ""
    assert all(row["candidate_class"] == "safe_partial_turn_evidence" for row in result.ranking_rows)
