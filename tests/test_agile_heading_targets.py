from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from linearisation import linearise_trim
from run_agile_feasibility import run_agile_feasibility
from run_one import run_scenario
from run_sweep import run_entry_sweep
from scenarios import agile_heading_target_scenarios, build_scenario


def _trim_state() -> np.ndarray:
    aircraft = adapt_glider(build_nausicaa_glider())
    return linearise_trim(aircraft=aircraft).x_trim


def test_baseline_and_agile_scenario_ids_construct() -> None:
    x_trim = _trim_state()
    baseline_ids = (
        "s4_full_nominal_glide_no_wind",
        "s4_full_bank_reversal_left_no_wind",
        "s4_full_bank_reversal_right_no_wind",
        "s4_full_recovery_no_wind",
        "s9_agile_reversal_left_no_wind",
        *agile_heading_target_scenarios(),
    )

    for scenario_id in baseline_ids:
        scenario = build_scenario(scenario_id, x_trim=x_trim, repo_root=Path.cwd(), seed=1)
        assert scenario.scenario_id == scenario_id
        assert scenario.x0.shape == (15,)


def test_agile_heading_target_metrics_are_reported(tmp_path: Path) -> None:
    required = {
        "target_heading_deg",
        "heading_change_deg",
        "actual_heading_change_deg",
        "heading_error_deg",
        "forward_travel_m",
        "min_wall_distance_m",
        "saturation_fraction",
        "exit_recoverable",
        "feasibility_label",
    }
    allowed_labels = {
        "fixed_start_feasible",
        "fixed_start_safe_but_under_turning",
        "fixed_start_unsafe",
        "fixed_start_unrecoverable",
    }

    for scenario_id in agile_heading_target_scenarios():
        row = run_scenario(scenario_id, seed=2, output_root=tmp_path)
        assert required.issubset(row)
        assert row["primitive_family"] == "agile_tvlqr_scaffold"
        assert row["is_full_turn_claim"] is False
        assert row["feasibility_label"] in allowed_labels


def test_agile_feasibility_summary_keeps_all_targets_and_gates_sweeps(tmp_path: Path) -> None:
    rows = run_agile_feasibility(
        seed=3,
        output_root=tmp_path,
        targets_deg=(30.0, 60.0),
        run_sweeps=False,
    )

    assert [row["target_heading_deg"] for row in rows] == [30.0, 60.0]
    assert all("largest_fixed_start_feasible_target_deg" in row for row in rows)
    assert all(row["sweep_sample_count"] == 0 for row in rows)
    assert all("random_entry_gate_passed" in row for row in rows)
    assert all("updraft_stress_gate_passed" in row for row in rows)


def test_agile_sweep_default_is_development_sample_count() -> None:
    samples_default = inspect.signature(run_agile_feasibility).parameters["sweep_samples"].default
    assert samples_default >= 50


def test_small_agile_entry_sweep_repeats_run_ids(tmp_path: Path) -> None:
    first = run_entry_sweep(
        "s9_agile_reversal_left_target_030_no_wind",
        primitive=None,
        seed=7,
        sample_count=2,
        output_root=tmp_path / "first",
    )
    second = run_entry_sweep(
        "s9_agile_reversal_left_target_030_no_wind",
        primitive=None,
        seed=7,
        sample_count=2,
        output_root=tmp_path / "second",
    )

    assert [row["sample_index"] for row in first] == [row["sample_index"] for row in second]
    assert [row["run_id"] for row in first] == [row["run_id"] for row in second]
