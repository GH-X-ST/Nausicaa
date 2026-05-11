from __future__ import annotations

import csv
import math
from pathlib import Path

from linearisation import STATE_NAMES
from run_one import run_scenario


def test_full_duration_primitives_log_finite_states(tmp_path: Path) -> None:
    for scenario_id in (
        "s4_full_nominal_glide_no_wind",
        "s4_full_bank_reversal_left_no_wind",
        "s4_full_recovery_no_wind",
    ):
        row = run_scenario(scenario_id, seed=2, output_root=tmp_path)
        assert row["success"] is True
        with Path(row["log_path"]).open(newline="", encoding="utf-8") as handle:
            for log_row in csv.DictReader(handle):
                for state_name in STATE_NAMES:
                    assert math.isfinite(float(log_row[state_name]))

