from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_d2_agile_boundary_refinement as d2
from dense_archive_table_io import TableManifest, write_table_manifest, write_table_partition


def test_d2_descriptors_use_closed_loop_only_for_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    index, samples = _reference_frames()
    monkeypatch.setattr(d2, "adapt_glider", lambda glider: object())
    monkeypatch.setattr(d2, "build_nausicaa_glider", lambda: object())
    monkeypatch.setattr(d2, "rollout_reference_feedback", _fake_rollout)

    descriptors = d2.build_d2_descriptors(
        index,
        samples,
        row_count=3,
        latency_case="nominal",
        random_seed=1,
    )

    assert len(descriptors) == 3
    assert descriptors["closed_loop_replay_performed"].all()
    assert not descriptors["open_loop_optimizer_success_used_for_acceptance"].any()
    assert not descriptors["command_template_success_used_for_acceptance"].any()
    assert set(descriptors["latency_case"]) == {"nominal"}
    assert set(descriptors["latency_pass_label"]).issubset({"nominal_pass", "nominal_fail"})


def test_d2_summary_preserves_branch_and_target_rows() -> None:
    rows = []
    for branch, fan, env in (
        ("single_fan_branch", "single_fan", "W1_single_fan"),
        ("four_fan_branch", "four_fan", "W1_four_fan"),
    ):
        row = _descriptor_row(branch, fan, env, 15.0)
        rows.append(row)
        rows.append({**row, "target_heading_deg": 45.0, "success_flag": False})
    summary = d2.build_d2_summary(pd.DataFrame(rows))

    assert set(summary["layout_branch_id"]) == {"single_fan_branch", "four_fan_branch"}
    assert set(summary["target_heading_deg"]) == {15.0, 45.0}


def test_runner_blocks_before_archive_when_runtime_budget_is_exhausted(tmp_path: Path) -> None:
    reference_root = _write_reference_fixture(tmp_path, started_epoch=time.time() - 10000.0)
    result_root = tmp_path / "d2"

    outputs = d2.run_d2_agile_boundary_refinement(
        d2.D2AgileBoundaryConfig(
            run_id=18,
            reference_run_id=17,
            result_root=result_root,
            reference_root=reference_root,
            runtime_budget_hours=1.0,
            runtime_buffer_fraction=0.20,
        )
    )

    manifest = json.loads(outputs.manifest_json.read_text(encoding="ascii"))
    assert manifest["status"] == "D2_execution_blocked_by_runtime_budget"
    assert manifest["runtime_budget_blocked_before_archive"] is True
    assert not outputs.descriptors_path.exists()


def test_runner_refuses_non_empty_output_root(tmp_path: Path) -> None:
    reference_root = _write_reference_fixture(tmp_path, started_epoch=time.time())
    result_root = tmp_path / "d2"
    blocked_root = result_root / "018"
    blocked_root.mkdir(parents=True)
    (blocked_root / "existing.txt").write_text("x", encoding="ascii")

    with pytest.raises(RuntimeError, match="non-empty"):
        d2.run_d2_agile_boundary_refinement(
            d2.D2AgileBoundaryConfig(
                run_id=18,
                reference_run_id=17,
                result_root=result_root,
                reference_root=reference_root,
                row_count_override=1,
            )
        )


def test_no_overclaiming_text_keeps_d2_simulation_scope() -> None:
    assert "simulation boundary refinement only" in d2.NO_OVERCLAIMING_TEXT
    assert "No production-floor completion" in d2.NO_OVERCLAIMING_TEXT
    assert "hardware readiness" in d2.NO_OVERCLAIMING_TEXT
    assert "sim-to-real" in d2.NO_OVERCLAIMING_TEXT


def _reference_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    time_s = np.array([0.0, 0.02, 0.04])
    x_ref = np.zeros((3, 15), dtype=float)
    x_ref[:, 0] = [2.0, 2.02, 2.04]
    x_ref[:, 1] = 2.0
    x_ref[:, 2] = 1.4
    x_ref[:, 5] = np.deg2rad([0.0, 7.5, 15.0])
    x_ref[:, 6] = 6.0
    index = pd.DataFrame(
        [
            {
                "trajectory_id": "traj",
                "trajectory_generation_method": "slsqp_direct_shooting_reference",
                "optimizer_status": "success",
                "optimizer_message": "ok",
                "optimizer_success": True,
                "optimizer_iterations": 1,
                "optimizer_wall_time_s": 0.1,
                "objective_cost": 0.0,
                "heading_cost": 0.0,
                "speed_loss_cost": 0.0,
                "height_loss_cost": 0.0,
                "saturation_cost": 0.0,
                "safety_status": "pass",
                "layout_branch_id": "single_fan_branch",
                "fan_layout": "single_fan",
                "test_environment_mode": "W0_single_fan_branch",
                "updraft_model_id": "",
                "sample_id": "s0",
                "family": "canyon_steep_bank",
                "target_heading_deg": 15.0,
                "direction_sign": 1,
                "reference_horizon_s": 0.04,
                "terminal_heading_target_deg": 15.0,
                "achieved_heading_deg": 15.0,
                "terminal_heading_error_deg": 0.0,
                "terminal_speed_m_s": 6.0,
                "height_loss_m": 0.0,
                "minimum_safety_margin_m": 1.0,
                "actuator_saturation_fraction": 0.0,
                "open_loop_success_flag": True,
                "failure_label": "success",
                "controller_config_id": "traj_tvlqr",
                "tvlqr_status": "tvlqr_synthesised",
                "agile_evidence_class": "trajectory_optimised_tvlqr_nominal_latency",
                "closed_loop_success_flag": True,
                "closed_loop_tracking_error_rms": 0.0,
                "latency_case": "nominal",
                "latency_pass_label": "nominal_pass",
            }
        ]
    )
    rows = []
    for sample_index, sample_time in enumerate(time_s):
        row = {
            "trajectory_id": "traj",
            "controller_config_id": "traj_tvlqr",
            "sample_index": sample_index,
            "time_s": float(sample_time),
        }
        for state_index in range(15):
            row[f"x_ref_{state_index:02d}"] = float(x_ref[sample_index, state_index])
        for name in ("delta_a", "delta_e", "delta_r"):
            row[f"u_norm_ref_{name}"] = 0.0
            row[f"delta_cmd_ref_rad_{name}"] = 0.0
        for command_index in range(3):
            for state_index in range(15):
                row[f"k_{command_index}_{state_index:02d}"] = 0.0
        rows.append(row)
    return index, pd.DataFrame(rows)


def _fake_rollout(trajectory, gains, **kwargs):  # noqa: ANN001
    return {
        "time_s": trajectory.time_s,
        "x_closed_loop": trajectory.x_ref,
        "u_norm_requested": trajectory.u_norm_ref,
        "u_norm_effective_target": trajectory.u_norm_ref,
        "u_norm_applied": trajectory.u_norm_ref,
        "delta_cmd_rad": trajectory.delta_cmd_ref_rad,
        "latency_fields": {"latency_case": "nominal"},
    }


def _descriptor_row(branch: str, fan: str, environment: str, target: float) -> dict[str, object]:
    return {
        "layout_branch_id": branch,
        "fan_layout": fan,
        "test_environment_mode": environment,
        "family": "canyon_steep_bank",
        "target_heading_deg": target,
        "success_flag": target == 15.0,
        "minimum_safety_margin_m": 0.5,
        "terminal_heading_error_deg": 1.0,
        "latency_pass_label": "nominal_pass" if target == 15.0 else "nominal_fail",
    }


def _write_reference_fixture(tmp_path: Path, *, started_epoch: float) -> Path:
    root = tmp_path / "refs"
    run = root / "017"
    index, samples = _reference_frames()
    (run / "metrics").mkdir(parents=True)
    (run / "manifests").mkdir(parents=True)
    index.to_csv(run / "metrics" / "agile_reference_index_s017.csv", index=False)
    partition = write_table_partition(
        samples,
        run / "tables" / "agile_reference_samples" / "part-00000.csv.gz",
        storage_format="csv_gz",
    )
    write_table_manifest(
        run / "manifests" / "table_manifest_s017.json",
        TableManifest(
            run_id=17,
            root=str(run),
            storage_format="csv_gz",
            tables=(partition,),
        ),
    )
    (run / "manifests" / "runtime_budget_state_s017.json").write_text(
        json.dumps(
            {
                "runtime_budget_started_epoch_s": started_epoch,
                "runtime_budget_hours": 1.0,
                "runtime_buffer_fraction": 0.20,
            }
        )
        + "\n",
        encoding="ascii",
    )
    (run / "manifests" / "agile_reference_manifest_s017.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "runtime_budget_status": "pass",
                "selected_d2_scale_class": "primary",
            }
        )
        + "\n",
        encoding="ascii",
    )
    return root
