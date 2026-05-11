from __future__ import annotations

from pathlib import Path

import numpy as np

from latency import (
    CommandToSurfaceConfig,
    LatencyEnvelope,
    feedback_delay_s,
    half_response_s,
    latency_audit_fields,
    latency_range_s,
)
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
    assert np.isclose(feedback_delay_s(CommandToSurfaceConfig(), envelope), 0.0229)
    assert np.isclose(envelope.onset_latency_s, 0.073)
    assert np.isclose(envelope.half_response_nominal_s, 0.108)
    assert np.isclose(envelope.actuator_t90_s, 0.130)
    assert np.isclose(envelope.conservative_actuator_bound_s, 0.151)
    assert np.isclose(half_response_s(CommandToSurfaceConfig(mode="robust_upper"), envelope), 0.151)
    assert envelope.vicon_filter_cutoff_hz == 20.0
    assert envelope.vicon_filter_model == "one_pole"

    row = run_scenario(
        "s4_latency_low_bank_reversal_left",
        seed=1,
        output_root=tmp_path,
    )
    assert row["latency_mode"] == "low"
    assert row["latency_s"] is not None
    assert row["latency_range_s"] is not None
    assert np.isclose(float(row["state_feedback_delay_s"]), 0.0229)
    assert np.isclose(float(row["actuator_t10_s"]), 0.073)
    assert np.isclose(float(row["actuator_t90_s"]), 0.130)
    assert np.isclose(float(row["conservative_actuator_bound_s"]), 0.151)
    assert str(row["vicon_filter_model"]) == "one_pole"


def test_latency_audit_fields_match_measured_contract() -> None:
    fields = latency_audit_fields(CommandToSurfaceConfig(mode="nominal"), LatencyEnvelope())

    assert np.isclose(float(fields["state_feedback_delay_s"]), 0.0229)
    assert np.isclose(float(fields["actuator_t10_s"]), 0.073)
    assert np.isclose(float(fields["actuator_t50_nominal_s"]), 0.108)
    assert np.isclose(float(fields["actuator_t90_s"]), 0.130)
    assert np.isclose(float(fields["conservative_actuator_bound_s"]), 0.151)
    assert np.isclose(float(fields["vicon_filter_cutoff_hz"]), 20.0)
    assert fields["vicon_filter_model"] == "one_pole"
