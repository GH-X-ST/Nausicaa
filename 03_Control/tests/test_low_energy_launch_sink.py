from __future__ import annotations

from run_repeated_launch_learning_curve import _expected_low_energy_dry_air_sink


def test_low_speed_floor_stop_is_not_claim_bearing_in_lift_environment() -> None:
    assert _expected_low_energy_dry_air_sink(
        scheduled={"environment_mode": "w3_randomised_four", "scheduled_active_fan_count": 4},
        primitive_rows=[
            {
                "initial_u": 4.2,
                "initial_v": 0.0,
                "initial_w": 0.0,
                "termination_cause": "floor_margin_stop",
                "failure_label": "floor_violation",
            }
        ],
        physical_floor_or_ceiling=True,
        no_viable=False,
        terminal_useful=False,
        lift_capture=True,
        episode_duration_s=0.6,
        initial_launch_speed_m_s=4.2,
    )


def test_high_speed_floor_stop_remains_claim_bearing() -> None:
    assert not _expected_low_energy_dry_air_sink(
        scheduled={"environment_mode": "w3_randomised_four", "scheduled_active_fan_count": 4},
        primitive_rows=[
            {
                "initial_u": 5.4,
                "initial_v": 0.0,
                "initial_w": 0.0,
                "termination_cause": "floor_margin_stop",
                "failure_label": "floor_violation",
            }
        ],
        physical_floor_or_ceiling=True,
        no_viable=False,
        terminal_useful=False,
        lift_capture=True,
        episode_duration_s=0.6,
        initial_launch_speed_m_s=5.4,
    )
