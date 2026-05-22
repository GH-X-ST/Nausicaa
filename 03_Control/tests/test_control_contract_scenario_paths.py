from __future__ import annotations

import pytest

from scenario_contract import (
    LATENCY_CASES,
    WIND_MODES,
    ScenarioSpec,
    scenario_spec_row,
    validate_scenario_spec,
)


def _scenario(wind_mode: str = "none", latency_case: str = "nominal") -> ScenarioSpec:
    return ScenarioSpec(
        name="audit_scenario",
        wind_mode=wind_mode,  # type: ignore[arg-type]
        latency_case=latency_case,  # type: ignore[arg-type]
        dt_s=0.02,
        t_final_s=1.0,
        seed=1,
        use_true_safe_bounds=True,
        description="contract audit scenario",
    )


def test_scenario_validation_accepts_allowed_modes_and_latency_cases() -> None:
    for wind_mode in WIND_MODES:
        validate_scenario_spec(_scenario(wind_mode=wind_mode))
    for latency_case in LATENCY_CASES:
        validate_scenario_spec(_scenario(latency_case=latency_case))


def test_scenario_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="wind_mode"):
        validate_scenario_spec(_scenario(wind_mode="uniform"))
    with pytest.raises(ValueError, match="latency_case"):
        validate_scenario_spec(_scenario(latency_case="medium"))
    with pytest.raises(ValueError, match="dt_s"):
        validate_scenario_spec(
            ScenarioSpec("bad", "none", "nominal", 0.0, 1.0, 1, True, "bad")
        )
    with pytest.raises(ValueError, match="t_final_s"):
        validate_scenario_spec(
            ScenarioSpec("bad", "none", "nominal", 0.02, 0.0, 1, True, "bad")
        )


def test_scenario_spec_row_is_csv_ready() -> None:
    row = scenario_spec_row(_scenario(wind_mode="panel", latency_case="conservative"))

    assert row["wind_mode"] == "panel"
    assert row["latency_case"] == "conservative"
    assert row["use_true_safe_bounds"] is True
