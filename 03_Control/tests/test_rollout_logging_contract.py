from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logging_contract import (
    command_dataframe,
    metric_dataframe,
    trajectory_dataframe,
    write_rollout_outputs,
)
from metric_contract import REQUIRED_METRIC_COLUMNS, empty_metric_row
from rollout import RolloutResult
from state_contract import STATE_NAMES


def _metric_row() -> dict[str, object]:
    row = empty_metric_row(include_agile=True)
    row.update(
        {
            "scenario_name": "rollout_smoke",
            "wind_mode": "none",
            "latency_case": "none",
            "failure_label": "not_run",
            "finite_state_success": True,
            "rollout_success": True,
            "notes": "rollout_smoke_no_primitive",
            "saturation_time_s": 0.0,
        }
    )
    return row


def _result() -> RolloutResult:
    time_s = np.array([0.0, 0.02])
    x = np.zeros((2, 15))
    u_norm_requested = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    u_norm_applied = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    delta_cmd_rad = np.zeros((2, 3))
    return RolloutResult(
        time_s=time_s,
        x=x,
        u_norm_requested=u_norm_requested,
        u_norm_applied=u_norm_applied,
        delta_cmd_rad=delta_cmd_rad,
        success=False,
        failure_label="not_run",
        metrics=_metric_row(),
        notes="rollout_smoke_no_primitive",
    )


def test_trajectory_dataframe_uses_canonical_state_columns() -> None:
    table = trajectory_dataframe(np.array([0.0, 0.02]), np.zeros((2, 15)))

    assert list(table.columns) == ["time_s", *STATE_NAMES]
    with pytest.raises(ValueError, match="shape"):
        trajectory_dataframe(np.array([0.0]), np.zeros((1, 14)))


def test_command_dataframe_records_requested_applied_and_radian_commands() -> None:
    result = _result()
    table = command_dataframe(
        result.time_s,
        result.u_norm_requested,
        result.u_norm_applied,
        result.delta_cmd_rad,
    )

    assert "u_norm_requested_delta_a_norm" in table.columns
    assert "u_norm_applied_delta_a_norm" in table.columns
    assert "delta_cmd_rad_delta_a_cmd" in table.columns
    assert table.loc[1, "u_norm_requested_delta_a_norm"] == 2.0
    assert table.loc[1, "u_norm_applied_delta_a_norm"] == 1.0
    assert not any(
        column.startswith("u_norm_effective_target_") for column in table.columns
    )


def test_command_dataframe_optionally_records_effective_targets() -> None:
    result = _result()
    effective = np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.75]])
    table = command_dataframe(
        result.time_s,
        result.u_norm_requested,
        result.u_norm_applied,
        result.delta_cmd_rad,
        u_norm_effective_target=effective,
    )

    assert "u_norm_effective_target_delta_a_norm" in table.columns
    assert "u_norm_effective_target_delta_e_norm" in table.columns
    assert "u_norm_effective_target_delta_r_norm" in table.columns
    assert table.loc[1, "u_norm_effective_target_delta_a_norm"] == 0.5
    assert table.loc[1, "u_norm_effective_target_delta_e_norm"] == -0.25
    assert table.loc[1, "u_norm_effective_target_delta_r_norm"] == 0.75


def test_metric_dataframe_validates_metric_schema() -> None:
    table = metric_dataframe(_metric_row())

    assert set(REQUIRED_METRIC_COLUMNS) <= set(table.columns)
    assert table.loc[0, "notes"] == "rollout_smoke_no_primitive"


def test_write_rollout_outputs_creates_expected_files(tmp_path) -> None:
    outputs = write_rollout_outputs(
        _result(),
        tmp_path,
        "01_rollout_smoke",
        1,
        overwrite=False,
    )

    for key in ("trajectory_csv", "commands_csv", "metrics_csv", "manifest_json", "report_md"):
        assert outputs[key].exists()
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    metrics = pd.read_csv(outputs["metrics_csv"])
    assert manifest["rollout_implemented"] is True
    assert manifest["primitive_implemented"] is False
    assert manifest["command_bridge"] == "normalised_command_to_surface_rad"
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())
    assert metrics.loc[0, "run_id"] == "s001"
