from __future__ import annotations

from run_one import run_scenario


def test_scenario_reproducibility_for_same_seed(tmp_path) -> None:
    first = run_scenario("s0_no_wind", seed=7, output_root=tmp_path / "first")
    second = run_scenario("s0_no_wind", seed=7, output_root=tmp_path / "second")
    assert first["success"] == second["success"]
    assert first["height_change_m"] == second["height_change_m"]
    assert first["terminal_speed_m_s"] == second["terminal_speed_m_s"]
    assert first["max_alpha_deg"] == second["max_alpha_deg"]
