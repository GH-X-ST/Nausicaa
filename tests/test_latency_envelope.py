from __future__ import annotations

from pathlib import Path

import numpy as np

from latency import CommandToSurfaceConfig, LatencyEnvelope, latency_range_s
from run_one import run_scenario
from scenarios import s4_audit_scenarios


def test_latency_scenarios_cover_low_nominal_high(tmp_path: Path) -> None:
    scenario_ids = set(s4_audit_scenarios())
    assert "s4_latency_low_bank_reversal_left" in scenario_ids
    assert "s4_latency_nominal_bank_reversal_left" in scenario_ids
    assert "s4_latency_high_bank_reversal_left" in scenario_ids

    envelope = LatencyEnvelope()
    low, high = latency_range_s(CommandToSurfaceConfig(mode="nominal"), envelope)
    assert np.isclose(high - low, 0.02, atol=1e-12)

    row = run_scenario(
        "s4_latency_low_bank_reversal_left",
        seed=1,
        output_root=tmp_path,
    )
    assert row["latency_mode"] == "low"
    assert row["latency_s"] is not None
    assert row["latency_range_s"] is not None

