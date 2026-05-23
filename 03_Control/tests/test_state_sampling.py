from __future__ import annotations

import pandas as pd
import numpy as np

from state_contract import STATE_INDEX, STATE_NAMES
from state_sampling import (
    archive_state_sample_for_row,
    archive_state_sample_row,
    measured_log_schema_row,
    measured_log_state_sample_rows,
    state_is_launch_gate_compliant,
)


def test_archive_state_sampler_is_deterministic_and_labels_rows() -> None:
    first = archive_state_sample_for_row(2, seed=10, W_layer="W1", environment_mode="gaussian_single")
    second = archive_state_sample_for_row(2, seed=10, W_layer="W1", environment_mode="gaussian_single")
    row = archive_state_sample_row(first)

    assert first.paired_start_key == second.paired_start_key
    assert first.start_state_family == "launch_gate"
    assert first.state_envelope_label == "approved_launch_gate"
    assert row["state_sample_source"] == "synthetic_launch_gate"
    assert row["previous_primitive_status"] == "launch_start"
    assert row["state_sample_detail"] == "deterministic_mixed_start_launch_gate"
    assert row["launch_gate_compliant"] is True
    for name in STATE_NAMES:
        assert f"initial_{name}" in row
    assert "initial_state_vector_json" in row


def test_mixed_start_sampler_has_required_40_60_schedule_and_bounds() -> None:
    samples = [
        archive_state_sample_for_row(index, seed=21, W_layer="W1", environment_mode="gaussian_single")
        for index in range(20)
    ]
    families = [sample.start_state_family for sample in samples]

    assert families.count("launch_gate") == 8
    assert families.count("inflight_nominal") == 5
    assert families.count("inflight_lift_region") == 3
    assert families.count("inflight_boundary_near") == 2
    assert families.count("inflight_recovery_edge") == 2
    assert all(state_is_launch_gate_compliant(sample.state_vector) for sample in samples[:8])
    assert not any(sample.launch_gate_compliant for sample in samples[8:])
    assert {sample.state_sample_source for sample in samples}.issubset(
        {"synthetic_launch_gate", "synthetic_inflight", "rollout_exit_resampled", "stress_sample"}
    )
    assert {sample.previous_primitive_status for sample in samples}.issubset(
        {"launch_start", "clean_exit", "terminal_useful_exit", "recovery_edge"}
    )
    boundary_samples = [sample for sample in samples if sample.start_state_family == "inflight_boundary_near"]
    assert all("legacy_boundary_terminal" in sample.state_sample_detail for sample in boundary_samples)
    assert {sample.state_envelope_label for sample in samples}.issubset(
        {"approved_launch_gate", "local_primitive_envelope", "lift_region", "boundary_near", "recovery_edge"}
    )


def test_inflight_samples_include_rates_and_surface_states() -> None:
    sample = archive_state_sample_for_row(8, seed=22, W_layer="W1", environment_mode="gaussian_single")
    state = sample.state_vector

    assert sample.start_state_family == "inflight_nominal"
    assert np.linalg.norm(state[[STATE_INDEX["p"], STATE_INDEX["q"], STATE_INDEX["r"]]]) > 0.0
    assert np.linalg.norm(
        state[[STATE_INDEX["delta_a"], STATE_INDEX["delta_e"], STATE_INDEX["delta_r"]]]
    ) > 0.0


def test_measured_log_compatibility_shape_without_real_logs(tmp_path) -> None:
    path = tmp_path / "measured_log.csv"
    data = {name: [0.0] for name in STATE_NAMES}
    data["u"] = [5.5]
    data["paired_start_key"] = ["real_000"]
    pd.DataFrame(data).to_csv(path, index=False)

    rows = measured_log_state_sample_rows(path)
    schema = measured_log_schema_row()

    assert rows[0].state_sample_source == "measured_log"
    assert rows[0].paired_start_key == "real_000"
    assert rows[0].launch_gate_compliant is False
    assert "x_w" in schema["required_state_columns"]
