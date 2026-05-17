from __future__ import annotations

import pytest

from metric_contract import (
    AGILE_METRIC_COLUMNS,
    FAILURE_LABELS,
    REQUIRED_METRIC_COLUMNS,
    empty_metric_row,
    metric_schema_dataframe,
    validate_metric_row,
)
from primitive_contract import (
    PrimitiveEntrySet,
    PrimitiveExitCheck,
    PrimitiveSpec,
    allowed_primitive_families,
    primitive_spec_row,
    validate_primitive_spec,
)


def _valid_primitive(family: str = "glide") -> PrimitiveSpec:
    return PrimitiveSpec(
        name=f"{family}_audit",
        family=family,  # type: ignore[arg-type]
        duration_s=1.0,
        entry_set=PrimitiveEntrySet(
            name="entry",
            description="audit entry set",
            lower={"speed_m_s": 5.0},
            upper={"speed_m_s": 8.0},
        ),
        exit_checks=(
            PrimitiveExitCheck(
                name="finite_state",
                description="state remains finite",
                required=True,
            ),
        ),
        metadata={"source": "contract_test"},
    )


def test_allowed_primitive_families_are_exact_and_include_agile() -> None:
    assert allowed_primitive_families() == (
        "glide",
        "bank",
        "recovery",
        "agile_reversal",
    )
    assert "agile_reversal" in allowed_primitive_families()


def test_primitive_validation_accepts_valid_specs() -> None:
    spec = _valid_primitive("agile_reversal")

    validate_primitive_spec(spec)
    row = primitive_spec_row(spec)

    assert row["primitive_family"] == "agile_reversal"
    assert row["entry_variables"] == "speed_m_s"


def test_primitive_validation_rejects_invalid_family_or_empty_contracts() -> None:
    with pytest.raises(ValueError, match="unknown"):
        validate_primitive_spec(_valid_primitive("turn"))  # type: ignore[arg-type]

    no_entry = PrimitiveSpec(
        name="bad",
        family="glide",
        duration_s=1.0,
        entry_set=PrimitiveEntrySet("entry", "empty", {}, {}),
        exit_checks=(PrimitiveExitCheck("finite", "finite", True),),
        metadata={},
    )
    with pytest.raises(ValueError, match="entry set"):
        validate_primitive_spec(no_entry)

    no_exit = PrimitiveSpec(
        name="bad",
        family="glide",
        duration_s=1.0,
        entry_set=PrimitiveEntrySet("entry", "bounded", {"u": 1.0}, {"u": 2.0}),
        exit_checks=(),
        metadata={},
    )
    with pytest.raises(ValueError, match="exit"):
        validate_primitive_spec(no_exit)


def test_metric_row_contains_required_keys_and_accepts_agile_fields() -> None:
    row = empty_metric_row(include_agile=True)

    assert set(REQUIRED_METRIC_COLUMNS) <= set(row)
    assert set(AGILE_METRIC_COLUMNS) <= set(row)
    assert row["success"] is False
    assert row["finite_state_success"] is False
    assert row["rollout_success"] is False
    assert row["primitive_success"] is False
    assert row["closed_loop_replay_success"] is False
    assert row["source_trajectory_success"] is False
    assert row["gain_construction_success"] is False
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

    row = empty_metric_row(include_agile=True)
    with pytest.raises(ValueError, match="unknown"):
        validate_metric_row(row, allow_agile=False)


def test_metric_validation_rejects_non_boolean_success_flags() -> None:
    row = empty_metric_row()
    row["success"] = 1
    with pytest.raises(ValueError, match="success"):
        validate_metric_row(row)

    row = empty_metric_row()
    row["finite_state_success"] = 1
    with pytest.raises(ValueError, match="finite_state_success"):
        validate_metric_row(row)

    row = empty_metric_row(include_agile=True)
    row["source_trajectory_success"] = 1
    with pytest.raises(ValueError, match="source_trajectory_success"):
        validate_metric_row(row)


def test_metric_schema_dataframe_contains_all_columns() -> None:
    schema = metric_schema_dataframe()

    assert set(REQUIRED_METRIC_COLUMNS) <= set(schema["column"])
    assert set(AGILE_METRIC_COLUMNS) <= set(schema["column"])
    assert "success" in FAILURE_LABELS
