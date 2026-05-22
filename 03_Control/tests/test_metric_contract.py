from __future__ import annotations

import pytest

from metric_contract import (
    FAILURE_LABELS,
    REQUIRED_METRIC_COLUMNS,
    empty_metric_row,
    metric_schema_dataframe,
    validate_metric_row,
)


def test_metric_row_contains_required_foundation_keys() -> None:
    row = empty_metric_row()

    assert set(REQUIRED_METRIC_COLUMNS) == set(row)
    assert row["success"] is False
    assert row["finite_state_success"] is False
    assert row["rollout_success"] is False
    assert row["primitive_success"] is False
    assert row["closed_loop_replay_success"] is False
    validate_metric_row(row)


def test_metric_validation_rejects_bad_failure_label_and_unknown_keys() -> None:
    row = empty_metric_row()
    row["failure_label"] = "bad_label"
    with pytest.raises(ValueError, match="failure_label"):
        validate_metric_row(row)

    row = empty_metric_row()
    row["unknown"] = 1.0
    with pytest.raises(ValueError, match="unknown"):
        validate_metric_row(row)


def test_metric_validation_rejects_non_boolean_success_flags() -> None:
    row = empty_metric_row()
    row["success"] = 1
    with pytest.raises(ValueError, match="success"):
        validate_metric_row(row)

    row = empty_metric_row()
    row["finite_state_success"] = 1
    with pytest.raises(ValueError, match="finite_state_success"):
        validate_metric_row(row)


def test_metric_schema_dataframe_contains_all_columns() -> None:
    schema = metric_schema_dataframe()

    assert set(REQUIRED_METRIC_COLUMNS) == set(schema["column"])
    assert schema["contextual_foundation"].all()
    assert "success" in FAILURE_LABELS
