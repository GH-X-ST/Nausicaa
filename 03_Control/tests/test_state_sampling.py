from __future__ import annotations

import pandas as pd

from state_contract import STATE_NAMES
from state_sampling import (
    archive_state_sample_for_row,
    archive_state_sample_row,
    measured_log_schema_row,
    measured_log_state_sample_rows,
)


def test_archive_state_sampler_is_deterministic_and_labels_rows() -> None:
    first = archive_state_sample_for_row(2, seed=10, W_layer="W1", environment_mode="gaussian_single")
    second = archive_state_sample_for_row(2, seed=10, W_layer="W1", environment_mode="gaussian_single")
    row = archive_state_sample_row(first)

    assert first.paired_start_key == second.paired_start_key
    assert first.state_envelope_label == "boundary_near_x"
    assert row["state_sample_source"] == "deterministic_boundary_near"
    assert "initial_x_w" in row
    assert "initial_state_vector_json" in row


def test_measured_log_compatibility_shape_without_real_logs(tmp_path) -> None:
    path = tmp_path / "measured_log.csv"
    data = {name: [0.0] for name in STATE_NAMES}
    data["u"] = [5.5]
    data["paired_start_key"] = ["real_000"]
    pd.DataFrame(data).to_csv(path, index=False)

    rows = measured_log_state_sample_rows(path)
    schema = measured_log_schema_row()

    assert rows[0].state_sample_source == "measured_log_compatible"
    assert rows[0].paired_start_key == "real_000"
    assert "x_w" in schema["required_state_columns"]
