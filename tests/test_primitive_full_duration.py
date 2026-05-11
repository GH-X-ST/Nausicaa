from __future__ import annotations

from pathlib import Path

from run_one import run_scenario


def test_full_duration_scenarios_exist_and_report_metrics(tmp_path: Path) -> None:
    cases = {
        "s4_full_nominal_glide_no_wind": 0.20,
        "s4_full_bank_reversal_left_no_wind": 0.35,
        "s4_full_recovery_no_wind": 0.30,
    }
    required = (
        "duration_s",
        "termination_reason",
        "success",
        "height_change_m",
        "terminal_speed_m_s",
        "min_wall_distance_m",
        "max_alpha_deg",
        "saturation_fraction",
    )
    for scenario_id, smoke_duration_s in cases.items():
        row = run_scenario(scenario_id, seed=3, output_root=tmp_path)
        assert float(row["duration_s"]) > smoke_duration_s
        for key in required:
            assert key in row

