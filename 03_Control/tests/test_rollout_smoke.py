from __future__ import annotations

import json

import numpy as np
import pandas as pd

from logging_contract import write_rollout_outputs
from rollout import (
    RolloutConfig,
    make_constant_command_schedule,
    rollout_open_loop_normalised,
)


def _smoke_state() -> np.ndarray:
    state = np.zeros(15)
    state[0:3] = [2.5, 2.2, 1.5]
    state[6] = 6.5
    return state


def _smoke_result():
    config = RolloutConfig(dt_s=0.02, t_final_s=0.24, wind_mode="none")
    schedule = make_constant_command_schedule(np.zeros(3), 0.24, 0.02)
    return rollout_open_loop_normalised(
        _smoke_state(),
        schedule,
        config,
        aircraft=None,
        wind_model=None,
        seed=1,
        scenario_name="rollout_smoke",
    )


def test_smoke_rollout_shapes_and_contract_success_semantics() -> None:
    result = _smoke_result()

    assert result.x.shape == (13, 15)
    assert result.u_norm_requested.shape == (13, 3)
    assert result.u_norm_applied.shape == (13, 3)
    assert result.delta_cmd_rad.shape == (13, 3)
    assert result.success is False
    assert result.failure_label == "not_run"
    assert result.notes == "rollout_smoke_no_primitive"
    assert result.metrics["finite_state_success"] is True
    assert result.metrics["rollout_success"] is True
    assert result.metrics["primitive_success"] is False
    assert result.metrics["closed_loop_replay_success"] is False
    assert result.metrics["success"] is False
    assert result.metrics["saturation_fraction"] == 0.0
    assert result.metrics["saturation_time_s"] == 0.0


def test_nonfinite_initial_state_returns_clear_rollout_failure() -> None:
    config = RolloutConfig(dt_s=0.02, t_final_s=0.24, wind_mode="none")
    schedule = make_constant_command_schedule(np.zeros(3), 0.24, 0.02)
    state = _smoke_state()
    state[6] = np.nan

    result = rollout_open_loop_normalised(
        state,
        schedule,
        config,
        aircraft=object(),
        wind_model=None,
    )

    assert result.failure_label == "nonfinite_state"
    assert result.metrics["finite_state_success"] is False
    assert result.metrics["rollout_success"] is False


def test_smoke_outputs_manifest_remains_audit_only(tmp_path) -> None:
    result = _smoke_result()
    outputs = write_rollout_outputs(
        result,
        tmp_path,
        "01_rollout_smoke",
        1,
        overwrite=False,
    )

    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    metrics = pd.read_csv(outputs["metrics_csv"])
    assert outputs["trajectory_csv"].exists()
    assert outputs["commands_csv"].exists()
    assert outputs["metrics_csv"].exists()
    assert metrics.loc[0, "run_id"] == "s001"
    assert manifest["rollout_implemented"] is True
    assert manifest["controller_implemented"] is False
    assert manifest["primitive_implemented"] is False
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["raw_normalised_commands_enter_state_derivative"] is False
