from __future__ import annotations

import csv
from pathlib import Path

from run_one import run_scenario


def test_governor_writes_rollout_candidate_table(tmp_path: Path) -> None:
    row = run_scenario("s4_governor_selection", seed=4, output_root=tmp_path)
    assert row["success"] is True
    assert row["selected_primitive"] == "recovery"
    assert int(row["candidate_count"]) > 1
    assert int(row["rejected_count"]) >= 1
    assert "recovery" in str(row["selected_primitive"])

    candidate_path = Path(str(row["candidate_table_path"]))
    assert candidate_path.exists()
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    assert any(candidate["selected"] == "True" for candidate in candidates)
    assert any(candidate["rejection_reason"] for candidate in candidates)
