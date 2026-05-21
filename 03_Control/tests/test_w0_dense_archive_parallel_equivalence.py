from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pandas as pd

import run_w0_dense_archive_chunk as chunk
import run_w0_dense_archive_chunked as chunked
from dense_archive_table_io import list_table_partitions, read_table_partition, write_table_partition
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_serial_and_parallel_chunked_outputs_are_equivalent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_planning(result_root)
    monkeypatch.setattr(chunk, "_run_pilot_replays", _fake_replay)
    monkeypatch.setattr(chunked, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(chunked.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(chunked, "_memory_total_gb", lambda: 32.0)

    chunked.run_w0_dense_archive_chunked(
        run_id=21,
        planning_run_id=12,
        result_root=result_root,
        target_trials_total=4,
        target_trials_per_branch=2,
        chunk_size=1,
        workers=1,
        max_workers=1,
        storage_format="csv_gz",
        resume=True,
    )
    chunked.run_w0_dense_archive_chunked(
        run_id=22,
        planning_run_id=12,
        result_root=result_root,
        target_trials_total=4,
        target_trials_per_branch=2,
        chunk_size=1,
        workers=2,
        max_workers=2,
        storage_format="csv_gz",
        resume=True,
    )

    serial = _load_run(result_root / "021")
    parallel = _load_run(result_root / "022")
    serial["trial_descriptor_id"] = serial["trial_descriptor_id"].str.replace("s21", "sxx", regex=False)
    parallel["trial_descriptor_id"] = parallel["trial_descriptor_id"].str.replace("s22", "sxx", regex=False)
    pd.testing.assert_frame_equal(
        serial.sort_values("candidate_id").reset_index(drop=True),
        parallel.sort_values("candidate_id").reset_index(drop=True),
    )


class _ImmediateExecutor:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "_ImmediateExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - failure is surfaced by future.
            future.set_exception(exc)
        return future


def _write_planning(result_root: Path) -> None:
    planning = result_root.parent / "10_dense_archive_planning" / "012"
    for branch, fan, mode in (
        ("single_fan_branch", "single_fan", "W0_single_fan_branch"),
        ("four_fan_branch", "four_fan", "W0_four_fan_branch"),
    ):
        starts = []
        candidates = []
        for index in range(2):
            sample_id = f"{branch}_{index}"
            starts.append(_start(sample_id, branch, fan))
            candidates.append(_candidate(sample_id, branch, fan, mode))
        starts_frame = _with_chunk_columns(pd.DataFrame(starts), chunk_size=1)
        candidates_frame = _with_chunk_columns(pd.DataFrame(candidates), chunk_size=1)
        for chunk_index in range(2):
            mask = candidates_frame["archive_chunk_index"].astype(int).eq(chunk_index)
            write_table_partition(
                starts_frame[mask].reset_index(drop=True),
                planning
                / "tables"
                / "start_states"
                / f"layout_branch_id={branch}"
                / f"archive_chunk_index={chunk_index:05d}"
                / "part-00000.csv.gz",
                storage_format="csv_gz",
            )
            write_table_partition(
                candidates_frame[mask].reset_index(drop=True),
                planning
                / "tables"
                / "candidate_index"
                / f"layout_branch_id={branch}"
                / f"archive_chunk_index={chunk_index:05d}"
                / "part-00000.csv.gz",
                storage_format="csv_gz",
            )


def _with_chunk_columns(frame: pd.DataFrame, *, chunk_size: int) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["archive_chunk_index"] = result.index // int(chunk_size)
    result["archive_chunk_count"] = len(result) // int(chunk_size)
    result["chunk_local_index"] = result.index % int(chunk_size)
    result["archive_chunk_size"] = int(chunk_size)
    result["archive_branch_trial_index"] = result.index
    return result


def _load_run(root: Path) -> pd.DataFrame:
    return pd.concat(
        [
            read_table_partition(path)
            for path in list_table_partitions(root, "trial_outcomes")
        ],
        ignore_index=True,
    )


def _fake_replay(starts, selected, config) -> pd.DataFrame:
    del starts
    return pd.DataFrame(
        [_descriptor(row["candidate_id"], row["layout_branch_id"], config.run_id) for row in selected],
        columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS,
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


def _descriptor(candidate_id: str, branch: str, run_id: int) -> dict[str, object]:
    row = {column: "" for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS}
    row.update(
        {
            "trial_descriptor_id": f"s{run_id}_trial_{candidate_id}",
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
            "latency_execution_status": "command_delay_plus_actuator_lag",
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
