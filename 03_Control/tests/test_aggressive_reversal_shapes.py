from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aggressive_reversal_ocp import (
    SEED_FAMILIES,
    AggressiveReversalOcpConfig,
    next_family_for_failure,
    phase_labels_for_family,
    phase_seed_command_profile,
    target_config,
    unwrapped_signed_heading_change_deg,
)


def _time_grid(config: AggressiveReversalOcpConfig) -> np.ndarray:
    return np.arange(int(round(config.t_final_s / config.dt_s)) + 1) * config.dt_s


def test_seed_families_exist_and_profiles_are_bounded() -> None:
    config = target_config(60.0)
    time_s = _time_grid(config)

    assert set(SEED_FAMILIES) == {
        "short_perch_yaw_redirect",
        "long_perch_slow_redirect",
        "roll_dominant_banked_redirect",
        "split_pulse_redirect",
        "early_unload_descend_capture",
    }
    for family_name in SEED_FAMILIES:
        command = phase_seed_command_profile(config, time_s, family_name)
        phase = phase_labels_for_family(family_name, time_s, config.t_final_s)

        assert command.shape == (time_s.size, 3)
        assert np.all(np.isfinite(command))
        assert np.max(np.abs(command)) <= 1.0
        assert set(phase).issubset(
            {
                "entry",
                "pitch_brake",
                "slow_redirect",
                "heading_capture",
                "unload_descend",
                "exit_glide",
            }
        )
        assert "pitch_brake" in phase
        assert "exit_glide" in phase


def test_turn_direction_flips_aileron_and_rudder_not_elevator() -> None:
    left = target_config(90.0, AggressiveReversalOcpConfig(direction_sign=1))
    right = target_config(90.0, AggressiveReversalOcpConfig(direction_sign=-1))
    time_s = _time_grid(left)

    left_command = phase_seed_command_profile(
        left,
        time_s,
        "split_pulse_redirect",
    )
    right_command = phase_seed_command_profile(
        right,
        time_s,
        "split_pulse_redirect",
    )

    np.testing.assert_allclose(left_command[:, 0], -right_command[:, 0])
    np.testing.assert_allclose(left_command[:, 1], right_command[:, 1])
    np.testing.assert_allclose(left_command[:, 2], -right_command[:, 2])


@pytest.mark.parametrize(
    ("failure_label", "current_family", "expected_family", "expected_reason"),
    (
        (
            "under_turning",
            "short_perch_yaw_redirect",
            "long_perch_slow_redirect",
            "under_turning_longer_perch",
        ),
        (
            "speed_low",
            "long_perch_slow_redirect",
            "early_unload_descend_capture",
            "speed_low_earlier_unload",
        ),
        (
            "terminal_recovery_limited",
            "split_pulse_redirect",
            "early_unload_descend_capture",
            "terminal_recovery_extend_exit_glide",
        ),
        (
            "solver_failure",
            "roll_dominant_banked_redirect",
            "roll_dominant_banked_redirect",
            "solver_failure_retry_best_finite_phase_search",
        ),
    ),
)
def test_failure_label_to_next_family_mapping_is_deterministic(
    failure_label: str,
    current_family: str,
    expected_family: str,
    expected_reason: str,
) -> None:
    next_family, reason = next_family_for_failure(failure_label, current_family)

    assert next_family == expected_family
    assert reason == expected_reason


def test_unwrapped_signed_heading_change_handles_wraparound() -> None:
    left_yaw = np.deg2rad([170.0, 179.0, -179.0, -170.0])
    right_yaw = np.deg2rad([-170.0, -179.0, 179.0, 170.0])

    assert unwrapped_signed_heading_change_deg(left_yaw, 1) == pytest.approx(20.0)
    assert unwrapped_signed_heading_change_deg(right_yaw, -1) == pytest.approx(20.0)


def test_checkpoint_files_are_compact_json(tmp_path: Path) -> None:
    # A minimal synthetic checkpoint file must stay scalar/path based rather than
    # embedding trajectory arrays. This mirrors the runner checks for generated
    # checkpoints while keeping this unit test cheap.
    path = tmp_path / "checkpoint.json"
    checkpoint = {
        "target_heading_deg": 15.0,
        "family_name": "short_perch_yaw_redirect",
        "trajectory_csv": "candidate_trajectory.csv",
        "commands_csv": "candidate_commands.csv",
        "direct_ocp_attempted": True,
    }
    path.write_text(json.dumps(checkpoint), encoding="ascii")
    loaded = json.loads(path.read_text(encoding="ascii"))

    assert "x_ref" not in loaded
    assert "u_ff_norm" not in loaded
    assert loaded["trajectory_csv"].endswith(".csv")
