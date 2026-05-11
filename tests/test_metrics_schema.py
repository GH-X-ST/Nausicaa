from __future__ import annotations

import math
from pathlib import Path

from metrics import METRIC_SCHEMA_KEYS
from run_one import run_scenario


def test_metrics_schema_keys_are_stable(tmp_path: Path) -> None:
    row = run_scenario("s4_full_nominal_glide_no_wind", seed=5, output_root=tmp_path)
    for key in METRIC_SCHEMA_KEYS:
        assert key in row
    assert "failure_class" in row
    for key in ("latency_s", "duration_s", "terminal_speed_m_s"):
        assert row[key] is None or math.isfinite(float(row[key]))

