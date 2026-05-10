from __future__ import annotations

from run_one import run_scenario


def test_scenario_reproducibility_for_same_seed() -> None:
    first = run_scenario("s0_no_wind", seed=7)
    second = run_scenario("s0_no_wind", seed=7)
    assert first["success"] == second["success"]
    assert first["height_change_m"] == second["height_change_m"]
    assert first["terminal_speed_m_s"] == second["terminal_speed_m_s"]
    assert first["max_alpha_deg"] == second["max_alpha_deg"]
