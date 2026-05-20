from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import run_w0_dense_archive_chunk as chunk
from dense_archive_table_io import read_table_partition, write_table_partition
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_chunk_runner_writes_one_partition_manifest_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_planning(result_root)
    calls = 0

    def fake_replay(starts, selected, config):
        del starts, config
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            [_descriptor(row["candidate_id"], row["layout_branch_id"]) for row in selected],
            columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS,
        )

    monkeypatch.setattr(chunk, "_run_pilot_replays", fake_replay)

    paths = chunk.run_w0_dense_archive_chunk(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        layout_branch_id="single_fan_branch",
        chunk_index=0,
        chunk_count=2,
        chunk_size=2,
        storage_format="csv_gz",
        resume=True,
    )
    second = chunk.run_w0_dense_archive_chunk(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        layout_branch_id="single_fan_branch",
        chunk_index=0,
        chunk_count=2,
        chunk_size=2,
        storage_format="csv_gz",
        resume=True,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    frame = read_table_partition(paths["partition_path"])
    assert calls == 1
    assert second == paths
    assert manifest["status"] == "complete"
    assert manifest["row_count"] == 2
    assert len(frame) == 2
    assert chunk.chunk_status(
        chunk.W0ChunkConfig(
            run_id=13,
            planning_run_id=12,
            result_root=result_root,
            layout_branch_id="single_fan_branch",
            chunk_index=0,
            chunk_count=2,
            chunk_size=2,
            storage_format="csv_gz",
        )
    ) == "complete"


def test_corrupt_chunk_is_rejected_by_default(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_planning(result_root)
    cfg = chunk.W0ChunkConfig(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        layout_branch_id="single_fan_branch",
        chunk_index=0,
        chunk_count=2,
        chunk_size=2,
        storage_format="csv_gz",
    )
    paths = chunk.output_paths(cfg)
    paths.manifest_json.parent.mkdir(parents=True)
    paths.manifest_json.write_text("{\"status\":\"incomplete\"}\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="incomplete/corrupt"):
        chunk.run_w0_dense_archive_chunk(
            run_id=13,
            planning_run_id=12,
            result_root=result_root,
            layout_branch_id="single_fan_branch",
            chunk_index=0,
            chunk_count=2,
            chunk_size=2,
            storage_format="csv_gz",
            resume=True,
        )


def _write_planning(result_root: Path) -> None:
    planning = result_root.parent / "10_dense_archive_planning" / "012"
    for branch, fan, mode in (
        ("single_fan_branch", "single_fan", "W0_single_fan_branch"),
        ("four_fan_branch", "four_fan", "W0_four_fan_branch"),
    ):
        starts = []
        candidates = []
        for index in range(4):
            sample_id = f"{branch}_{index}"
            starts.append(_start(sample_id, branch, fan))
            candidates.append(_candidate(sample_id, branch, fan, mode))
        write_table_partition(
            pd.DataFrame(starts),
            planning / "tables" / "start_states" / f"layout_branch_id={branch}" / "part-00000.csv.gz",
            storage_format="csv_gz",
        )
        write_table_partition(
            pd.DataFrame(candidates),
            planning / "tables" / "candidate_index" / f"layout_branch_id={branch}" / "part-00000.csv.gz",
            storage_format="csv_gz",
        )


def _start(sample_id: str, branch: str, fan: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 1,
        "sampling_round": "test",
        "fan_layout": fan,
        "layout_branch_id": branch,
        "fan_config_id": f"{fan}_dry_air",
        "updraft_model_id": "no_updraft_dry_air",
        "start_class": "favourable",
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "x_w_m": 2.0,
        "y_w_m": 1.0,
        "z_w_m": 1.0,
        "speed_m_s": 6.0,
        "phi_rad": 0.0,
        "theta_rad": 0.0,
        "psi_rad": 0.0,
        "u_m_s": 6.0,
        "v_m_s": 0.0,
        "w_m_s": 0.0,
        "p_rad_s": 0.0,
        "q_rad_s": 0.0,
        "r_rad_s": 0.0,
    }


def _candidate(sample_id: str, branch: str, fan: str, mode: str) -> dict[str, object]:
    return {
        "candidate_id": f"candidate_{sample_id}",
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 1,
        "sampling_round": "test",
        "fan_layout": fan,
        "layout_branch_id": branch,
        "fan_config_id": f"{fan}_dry_air",
        "updraft_model_id": "no_updraft_dry_air",
        "test_environment_mode": mode,
        "paired_environment_mode": mode.replace("W0", "W1"),
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "start_class": "favourable",
        "environment_role": "dry_air_capable",
        "validity_gate_role": "baseline_gate",
        "acceptance_interpretation": "baseline_gate",
    }


def _descriptor(candidate_id: str, branch: str) -> dict[str, object]:
    row = {column: "" for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS}
    row.update(
        {
            "trial_descriptor_id": f"trial_{candidate_id}",
            "layout_branch_id": branch,
            "fan_layout": "single_fan" if branch == "single_fan_branch" else "four_fan",
            "fan_config_id": "dry",
            "test_environment_mode": "W0_single_fan_branch"
            if branch == "single_fan_branch"
            else "W0_four_fan_branch",
            "paired_environment_mode": "W1",
            "environment_role": "dry_air_capable",
            "validity_gate_role": "baseline_gate",
            "acceptance_interpretation": "baseline_gate",
            "candidate_id": candidate_id,
            "sample_id": candidate_id,
            "paired_sample_key": candidate_id,
            "seed": 1,
            "replay_seed": 1,
            "sampling_round": "test",
            "updraft_model_id": "no_updraft_dry_air",
            "family": "mild_bank",
            "target_heading_deg": 30.0,
            "direction_sign": 1,
            "start_class": "favourable",
            "updraft_relative_radius_m": 0.5,
            "speed0_m_s": 6.0,
            "w_wing_mean_m_s": 0.0,
            "delta_w_lr_m_s": 0.0,
            "min_true_margin_m": 0.5,
            "latency_case": "nominal",
            "latency_acceptance_scope": "nominal",
            "latency_pass_label": "nominal_pass",
            "state_feedback_delay_s": 0.0,
            "command_onset_delay_s": 0.0,
            "command_transport_delay_s": 0.0,
            "actuator_tau_s": "0.0;0.0;0.0",
            "actuator_t50_s": 0.0,
            "actuator_t90_s": 0.0,
            "latency_jitter_s": 0.0,
            "timing_model_version": "test",
            "state_feedback_delay_applied": False,
            "success_flag": True,
            "failure_label": "success",
            "governor_rejection_cause": "none",
            "robustness_label": "not_evaluated",
            "physics_priority_level": "test",
            "sim_real_match_key": candidate_id,
            "sim_real_match_key_version": "test",
            "sim_real_transfer_result": "not_evaluated",
            "descriptor_status": "replay_evaluated",
            "heading_error_deg": 0.0,
            "energy_residual_m": 0.0,
            "lift_dwell_fraction": 0.0,
            "saturation_fraction": 0.0,
        }
    )
    return row
