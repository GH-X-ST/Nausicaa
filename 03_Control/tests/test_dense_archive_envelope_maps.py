from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dense_archive_envelope_maps import (
    ENVELOPE_MAP_COLUMNS,
    EnvelopeMapConfig,
    build_envelope_map,
    envelope_cell_id,
)


EXPECTED_COLUMNS = (
    "envelope_cell_id",
    "layout_branch_id",
    "fan_layout",
    "test_environment_mode",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "latency_case",
    "radius_bin_m",
    "speed_bin_m_s",
    "wing_mean_bin_m_s",
    "margin_bin_m",
    "trial_count",
    "evaluated_trial_count",
    "success_count",
    "success_fraction",
    "nominal_pass_count",
    "conservative_pass_count",
    "dominant_failure_label",
    "dominant_governor_rejection_cause",
    "mean_heading_error_deg",
    "median_heading_error_deg",
    "mean_energy_residual_m",
    "mean_lift_dwell_fraction",
    "min_true_margin_m_min",
    "saturation_fraction_mean",
    "cell_status",
    "branch_decision_scope",
)


def test_empty_input_returns_exact_schema() -> None:
    result = build_envelope_map(pd.DataFrame())

    assert ENVELOPE_MAP_COLUMNS == EXPECTED_COLUMNS
    assert tuple(result.columns) == EXPECTED_COLUMNS
    assert result.empty


def test_cell_id_format_and_nan_bins_are_deterministic() -> None:
    row = _row(
        updraft_relative_radius_m=np.nan,
        speed0_m_s=np.nan,
        w_wing_mean_m_s=np.nan,
        min_true_margin_m=np.nan,
        target_heading_deg="",
    )

    assert envelope_cell_id(row) == (
        "cell|branch=single_fan_branch|fan=single_fan|env=W1_single_fan|"
        "family=mild_bank|target=none|dir=1|start=favourable|latency=nominal|"
        "radius=r_nan|speed=v_nan|wing_mean=ww_nan|margin=m_nan"
    )


def test_status_counts_latency_and_success_fraction_semantics() -> None:
    rows = [
        _row(trial_descriptor_id="not_1", descriptor_status="not_replayed"),
        _row(trial_descriptor_id="not_2", descriptor_status="synthetic_descriptor_only"),
        _row(
            trial_descriptor_id="success_1",
            start_class="mid_arena",
            descriptor_status="replay_evaluated",
            success_flag=True,
            failure_label="success",
            latency_pass_label="nominal_pass",
        ),
        _row(
            trial_descriptor_id="success_2",
            start_class="mid_arena",
            descriptor_status="nonfinite_state",
            success_flag=True,
            failure_label="success",
            latency_pass_label="conservative_pass",
            heading_error_deg=2.0,
            energy_residual_m=0.4,
            lift_dwell_fraction=0.5,
            min_true_margin_m=0.40,
            saturation_fraction=0.2,
        ),
        _row(
            trial_descriptor_id="fail_1",
            start_class="random_stress",
            descriptor_status="entry_invalid",
            success_flag=False,
            failure_label="true_safety_violation",
            governor_rejection_cause="entry_reject",
        ),
        _row(
            trial_descriptor_id="fail_2",
            start_class="random_stress",
            descriptor_status="replay_evaluated",
            success_flag=False,
            failure_label="target_miss",
            governor_rejection_cause="entry_reject",
        ),
        _row(
            trial_descriptor_id="mixed_1",
            start_class="lift_sector",
            descriptor_status="replay_evaluated",
            success_flag=True,
            failure_label="success",
        ),
        _row(
            trial_descriptor_id="mixed_2",
            start_class="lift_sector",
            descriptor_status="not_replayed",
            success_flag=False,
            failure_label="not_run",
        ),
    ]
    result = build_envelope_map(pd.DataFrame(rows))

    assert tuple(result.columns) == EXPECTED_COLUMNS
    by_start = {row["start_class"]: row for row in result.to_dict(orient="records")}
    assert by_start["favourable"]["cell_status"] == "not_replayed"
    assert by_start["favourable"]["evaluated_trial_count"] == 0
    assert by_start["favourable"]["success_fraction"] == 0.0

    assert by_start["mid_arena"]["cell_status"] == "all_success"
    assert by_start["mid_arena"]["latency_case"] == "nominal"
    assert by_start["mid_arena"]["evaluated_trial_count"] == 2
    assert by_start["mid_arena"]["success_count"] == 2
    assert by_start["mid_arena"]["success_fraction"] == 1.0
    assert by_start["mid_arena"]["nominal_pass_count"] == 1
    assert by_start["mid_arena"]["conservative_pass_count"] == 1
    assert by_start["mid_arena"]["mean_heading_error_deg"] == pytest.approx(1.5)
    assert by_start["mid_arena"]["median_heading_error_deg"] == pytest.approx(1.5)
    assert by_start["mid_arena"]["mean_energy_residual_m"] == pytest.approx(0.3)
    assert by_start["mid_arena"]["mean_lift_dwell_fraction"] == pytest.approx(0.45)
    assert by_start["mid_arena"]["min_true_margin_m_min"] == pytest.approx(0.40)
    assert by_start["mid_arena"]["saturation_fraction_mean"] == pytest.approx(0.15)

    assert by_start["random_stress"]["cell_status"] == "all_failure"
    assert by_start["random_stress"]["evaluated_trial_count"] == 2
    assert by_start["random_stress"]["success_fraction"] == 0.0
    assert by_start["random_stress"]["dominant_governor_rejection_cause"] == "entry_reject"
    assert by_start["random_stress"]["dominant_failure_label"] in {
        "target_miss",
        "true_safety_violation",
    }

    assert by_start["lift_sector"]["cell_status"] == "mixed_boundary"
    assert by_start["lift_sector"]["evaluated_trial_count"] == 1
    assert by_start["lift_sector"]["success_fraction"] == 1.0


def test_branch_local_grouping_prevents_cross_branch_merge() -> None:
    rows = [
        _row(trial_descriptor_id="single", fan_layout="single_fan", layout_branch_id="single_fan_branch"),
        _row(trial_descriptor_id="four", fan_layout="four_fan", layout_branch_id="four_fan_branch"),
    ]
    result = build_envelope_map(pd.DataFrame(rows), EnvelopeMapConfig())

    assert len(result) == 2
    assert set(result["fan_layout"]) == {"single_fan", "four_fan"}
    assert all("branch=" in cell for cell in result["envelope_cell_id"])


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "trial_descriptor_id": "trial_001",
        "layout_branch_id": "single_fan_branch",
        "fan_layout": "single_fan",
        "test_environment_mode": "W1_single_fan",
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "start_class": "favourable",
        "latency_case": "nominal",
        "updraft_relative_radius_m": 0.55,
        "speed0_m_s": 6.1,
        "w_wing_mean_m_s": 0.12,
        "min_true_margin_m": 0.45,
        "descriptor_status": "replay_evaluated",
        "success_flag": False,
        "failure_label": "target_miss",
        "governor_rejection_cause": "none",
        "heading_error_deg": 1.0,
        "energy_residual_m": 0.2,
        "lift_dwell_fraction": 0.4,
        "saturation_fraction": 0.1,
        "latency_pass_label": "nominal_pass",
        "branch_decision_scope": "branch_local_only_no_cross_layout_decision_transfer",
    }
    row.update(overrides)
    return row
