from __future__ import annotations

from pathlib import Path

from run_one import run_scenario


def test_governor_rejection_log_contains_reason(tmp_path: Path) -> None:
    row = run_scenario("s11_governor_rejection", seed=3, output_root=tmp_path)
    assert row["success"] is False
    assert "governor" in str(row["governor_rejection_reason"])
    path = tmp_path / "metrics" / "s11_governor_rejection_seed3_governor_rejections.csv"
    assert path.exists()
    assert "reason" in path.read_text(encoding="utf-8")
