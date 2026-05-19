from __future__ import annotations

import pytest

from result_paths import make_result_tree
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


def test_make_result_tree_creates_required_subfolders(tmp_path) -> None:
    paths = make_result_tree(tmp_path, "00_contracts", 1)

    assert paths["root"] == tmp_path / "00_contracts" / "001"
    for name in ("metrics", "logs", "figures", "manifests", "reports"):
        assert paths[name].is_dir()


def test_make_result_tree_requires_overwrite_and_safe_campaign(tmp_path) -> None:
    make_result_tree(tmp_path, "contracts", 1)

    with pytest.raises(ValueError, match="already exists"):
        make_result_tree(tmp_path, "contracts", 1)
    make_result_tree(tmp_path, "contracts", 1, overwrite=True)
    with pytest.raises(ValueError, match="campaign"):
        make_result_tree(tmp_path, "../bad", 1)
    with pytest.raises(ValueError, match="campaign"):
        make_result_tree(tmp_path, "BadName", 1)
