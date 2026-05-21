from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import pandas as pd
import pytest

import dense_archive_runtime as runtime
import profile_paired_w0_w1_archive as paired_profile
import run_paired_w0_w1_archive_chunk as paired_chunk
import run_paired_w0_w1_archive_chunked as paired_chunked
import run_paired_w0_w1_partitioned_planning as paired_planning
from dense_archive_chunking import (
    GenericChunkSpec,
    assert_unique_output_paths,
    partition_path,
)
from dense_archive_table_io import (
    list_table_partitions,
    read_table_partition,
    write_table_partition,
)
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS
from run_paired_w0_w1_partitioned_planning import PAIRED_ENVIRONMENT_MODES


def test_paired_planning_requires_chunk_divisibility() -> None:
    config = paired_planning.PairedW0W1PartitionedPlanningConfig(
        run_id=20,
        paired_scale_mode="production",
        w0_target_trials_per_branch=3,
        w1_floor_trials_per_branch=4,
        w1_target_trials_per_branch=4,
        partition_rows=2,
    )

    with pytest.raises(ValueError, match="divisible by partition_rows"):
        paired_planning._validate_config(config)


def test_paired_dry_run_explicit_8_workers_has_unique_paths(tmp_path: Path) -> None:
    result_root = _short_result_root(tmp_path)
    _write_paired_planning(result_root, rows_per_environment=4, chunk_size=2)

    paths = paired_chunked.run_paired_w0_w1_archive_chunked(
        run_id=14,
        planning_run_id=13,
        result_root=result_root,
        workers=8,
        max_workers=8,
        chunk_size=2,
        storage_format="csv_gz",
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    partition_paths = [chunk["partition_path"] for chunk in manifest["chunks"]]
    assert manifest["selected_worker_count"] == 8
    assert manifest["scheduled_chunk_count"] == 8
    assert set(chunk["test_environment_mode"] for chunk in manifest["chunks"]) == set(
        PAIRED_ENVIRONMENT_MODES
    )
    assert len(partition_paths) == len(set(partition_paths))
    assert manifest["runtime_core_version"] == runtime.RUNTIME_CORE_VERSION
    assert "--workers 8 --max-workers 8" in manifest["recommended_command"]


def test_paired_auto_workers_selects_8_on_i9_32gb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = _short_result_root(tmp_path)
    _write_paired_planning(result_root, rows_per_environment=2, chunk_size=1)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(runtime, "_memory_total_gb", lambda: 32.0)

    paths = paired_chunked.run_paired_w0_w1_archive_chunked(
        run_id=14,
        planning_run_id=13,
        result_root=result_root,
        workers="auto",
        max_workers=8,
        chunk_size=1,
        storage_format="csv_gz",
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    assert manifest["selected_worker_count"] == 8
    assert manifest["os_cpu_count"] == 20
    assert manifest["memory_total_gb"] == 32.0
    assert manifest["worker_fallback_reason"] == "none"


def test_paired_auto_workers_fall_back_only_when_memory_guardrail_requires(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = _short_result_root(tmp_path)
    _write_paired_planning(result_root, rows_per_environment=2, chunk_size=1)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(runtime, "_memory_total_gb", lambda: 32.0)
    monkeypatch.setattr(paired_chunked, "DEFAULT_ESTIMATED_WORKER_MEMORY_GB", 4.0)

    paths = paired_chunked.run_paired_w0_w1_archive_chunked(
        run_id=14,
        planning_run_id=13,
        result_root=result_root,
        workers="auto",
        max_workers=8,
        chunk_size=1,
        storage_format="csv_gz",
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    assert manifest["selected_worker_count"] == 6
    assert "memory_guardrail" in manifest["worker_fallback_reason"]


def test_paired_profile_is_isolated_and_records_worker_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = _short_result_root(tmp_path)
    _write_paired_planning(result_root, rows_per_environment=2, chunk_size=1)
    monkeypatch.setattr(paired_profile, "_run_pilot_replays", _fake_replay)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(runtime, "_memory_total_gb", lambda: 32.0)

    paths = paired_profile.profile_paired_w0_w1_archive(
        planning_run_id=13,
        result_root=result_root,
        sample_trials=4,
        storage_format="csv_gz",
        workers="auto",
    )

    payload = json.loads(paths["profile_json"].read_text(encoding="ascii"))
    assert paths["root"] == result_root / "profiles" / "paired_planning_s013"
    assert not (result_root / "014" / "tables").exists()
    assert payload["selected_worker_count"] == 8
    assert payload["os_cpu_count"] == 20
    assert payload["memory_safety_margin_gb"] == 8.0
    assert set(payload["rows_per_second_by_worker_count"]) == {"1", "4", "6", "8"}
    assert "GPU acceleration is deferred" in payload["gpu_acceleration_assessment"]
    assert payload["runtime_core_version"] == runtime.RUNTIME_CORE_VERSION
    assert payload["storage_contract_version"] == runtime.STORAGE_CONTRACT_VERSION


def test_w1_chunks_run_when_paired_w0_chunks_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = _short_result_root(tmp_path)
    _write_paired_planning(
        result_root,
        rows_per_environment=1,
        chunk_size=1,
        break_w0_start_alignment=True,
    )
    monkeypatch.setattr(paired_chunk, "_run_pilot_replays", _fake_replay)

    paths = paired_chunked.run_paired_w0_w1_archive_chunked(
        run_id=14,
        planning_run_id=13,
        result_root=result_root,
        workers=1,
        max_workers=1,
        chunk_size=1,
        storage_format="csv_gz",
        resume=True,
        continue_on_chunk_failure=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    status_by_mode = {
        chunk["test_environment_mode"]: chunk["status"]
        for chunk in manifest["chunks"]
    }
    assert status_by_mode["W0_single_fan_branch"] == "failed"
    assert status_by_mode["W0_four_fan_branch"] == "failed"
    assert status_by_mode["W1_single_fan"] == "complete"
    assert status_by_mode["W1_four_fan"] == "complete"


def test_serial_and_parallel_paired_outputs_are_equivalent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = _short_result_root(tmp_path)
    _write_paired_planning(result_root, rows_per_environment=2, chunk_size=1)
    monkeypatch.setattr(paired_chunk, "_run_pilot_replays", _fake_replay)
    monkeypatch.setattr(paired_chunked, "ProcessPoolExecutor", _ImmediateExecutor)

    paired_chunked.run_paired_w0_w1_archive_chunked(
        run_id=21,
        planning_run_id=13,
        result_root=result_root,
        workers=1,
        max_workers=1,
        chunk_size=1,
        storage_format="csv_gz",
        resume=True,
    )
    paired_chunked.run_paired_w0_w1_archive_chunked(
        run_id=22,
        planning_run_id=13,
        result_root=result_root,
        workers=2,
        max_workers=2,
        chunk_size=1,
        storage_format="csv_gz",
        resume=True,
    )

    serial = _load_run(result_root / "021")
    parallel = _load_run(result_root / "022")
    serial["trial_descriptor_id"] = serial["trial_descriptor_id"].str.replace(
        "s21",
        "sxx",
        regex=False,
    )
    parallel["trial_descriptor_id"] = parallel["trial_descriptor_id"].str.replace(
        "s22",
        "sxx",
        regex=False,
    )
    pd.testing.assert_frame_equal(
        serial.sort_values(["test_environment_mode", "candidate_id"]).reset_index(drop=True),
        parallel.sort_values(["test_environment_mode", "candidate_id"]).reset_index(drop=True),
    )


def test_shared_chunk_core_accepts_fake_w2_like_environment_without_new_storage(
    tmp_path: Path,
) -> None:
    spec = GenericChunkSpec(
        run_id=1,
        planning_run_id=1,
        result_root=tmp_path,
        layout_branch_id="fake_branch",
        test_environment_mode="W2_fake_environment",
        chunk_index=0,
        chunk_count=1,
        chunk_size=1,
        storage_format="csv_gz",
        simulation_stage="fake_w2_storage_smoke",
    )
    path = partition_path(
        tmp_path,
        table_name="candidate_index",
        layout_branch_id=spec.layout_branch_id,
        test_environment_mode=spec.test_environment_mode,
        chunk_index=spec.chunk_index,
        storage_format=spec.storage_format,
    )
    partition = write_table_partition(
        pd.DataFrame([{"candidate_id": "fake"}]),
        path,
        storage_format="csv_gz",
    )

    assert "test_environment_mode=W2_fake_environment" in partition.relative_path
    assert_unique_output_paths([spec], run_root=tmp_path)
    assert read_table_partition(path)["candidate_id"].to_list() == ["fake"]


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
        except BaseException as exc:  # pragma: no cover - surfaced by future.
            future.set_exception(exc)
        return future


def _short_result_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}" / "r"


def _write_paired_planning(
    result_root: Path,
    *,
    rows_per_environment: int,
    chunk_size: int,
    break_w0_start_alignment: bool = False,
) -> None:
    planning = result_root.parent / "10_dense_archive_planning" / "013"
    for mode in PAIRED_ENVIRONMENT_MODES:
        branch, fan = _branch_fan_for_mode(mode)
        starts = []
        candidates = []
        for index in range(int(rows_per_environment)):
            sample_id = f"{mode}_{index}"
            starts.append(_start(sample_id, branch, fan, mode))
            candidates.append(_candidate(sample_id, branch, fan, mode))
        starts_frame = _with_chunk_columns(pd.DataFrame(starts), chunk_size=chunk_size)
        candidates_frame = _with_chunk_columns(pd.DataFrame(candidates), chunk_size=chunk_size)
        if break_w0_start_alignment and mode.startswith("W0_"):
            starts_frame["sample_id"] = "bad_" + starts_frame["sample_id"].astype(str)
        for chunk_index in range(int(rows_per_environment) // int(chunk_size)):
            mask = candidates_frame["archive_chunk_index"].astype(int).eq(chunk_index)
            write_table_partition(
                starts_frame[mask].reset_index(drop=True),
                partition_path(
                    planning,
                    table_name="start_states",
                    layout_branch_id=branch,
                    test_environment_mode=mode,
                    chunk_index=chunk_index,
                    storage_format="csv_gz",
                ),
                storage_format="csv_gz",
            )
            write_table_partition(
                candidates_frame[mask].reset_index(drop=True),
                partition_path(
                    planning,
                    table_name="candidate_index",
                    layout_branch_id=branch,
                    test_environment_mode=mode,
                    chunk_index=chunk_index,
                    storage_format="csv_gz",
                ),
                storage_format="csv_gz",
            )


def _branch_fan_for_mode(mode: str) -> tuple[str, str]:
    if "single" in mode:
        return "single_fan_branch", "single_fan"
    return "four_fan_branch", "four_fan"


def _paired_mode(mode: str) -> str:
    mapping = {
        "W0_single_fan_branch": "W1_single_fan",
        "W1_single_fan": "W0_single_fan_branch",
        "W0_four_fan_branch": "W1_four_fan",
        "W1_four_fan": "W0_four_fan_branch",
    }
    return mapping[mode]


def _with_chunk_columns(frame: pd.DataFrame, *, chunk_size: int) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["archive_chunk_index"] = result.index // int(chunk_size)
    result["archive_chunk_count"] = len(result) // int(chunk_size)
    result["chunk_local_index"] = result.index % int(chunk_size)
    result["archive_chunk_size"] = int(chunk_size)
    result["archive_branch_trial_index"] = result.index
    return result


def _start(sample_id: str, branch: str, fan: str, mode: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 1,
        "sampling_round": "test",
        "fan_layout": fan,
        "layout_branch_id": branch,
        "fan_config_id": f"{fan}_dry_air",
        "updraft_model_id": "no_updraft_dry_air",
        "test_environment_mode": mode,
        "paired_environment_mode": _paired_mode(mode),
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
        "paired_environment_mode": _paired_mode(mode),
        "latency_case_planned": "nominal",
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "start_class": "favourable",
        "environment_role": "updraft_capable" if mode.startswith("W1_") else "dry_air_gate",
        "validity_gate_role": "w1_independent" if mode.startswith("W1_") else "w0_baseline",
        "acceptance_interpretation": "nominal_w1" if mode.startswith("W1_") else "w0_gate",
    }


def _fake_replay(starts, selected, config) -> pd.DataFrame:
    del starts
    run_id = int(getattr(config, "run_id", 0))
    rows = [
        _descriptor(row, run_id=run_id, success=True)
        for row in selected
    ]
    return pd.DataFrame(rows, columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS)


def _load_run(root: Path) -> pd.DataFrame:
    return pd.concat(
        [
            read_table_partition(path)
            for path in list_table_partitions(root, "trial_outcomes")
        ],
        ignore_index=True,
    )


def _descriptor(
    candidate: dict[str, object],
    *,
    run_id: int,
    success: bool,
) -> dict[str, object]:
    mode = str(candidate["test_environment_mode"])
    branch = str(candidate["layout_branch_id"])
    row = {column: "" for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS}
    row.update(
        {
            "trial_descriptor_id": f"s{run_id}_{mode}_{candidate['candidate_id']}",
            "layout_branch_id": branch,
            "fan_layout": candidate["fan_layout"],
            "fan_config_id": candidate["fan_config_id"],
            "test_environment_mode": mode,
            "paired_environment_mode": candidate["paired_environment_mode"],
            "environment_role": candidate["environment_role"],
            "validity_gate_role": candidate["validity_gate_role"],
            "acceptance_interpretation": candidate["acceptance_interpretation"],
            "candidate_id": candidate["candidate_id"],
            "sample_id": candidate["sample_id"],
            "paired_sample_key": candidate["paired_sample_key"],
            "seed": 1,
            "replay_seed": 7,
            "sampling_round": "test",
            "updraft_model_id": candidate["updraft_model_id"],
            "family": candidate["family"],
            "target_heading_deg": candidate["target_heading_deg"],
            "direction_sign": candidate["direction_sign"],
            "start_class": candidate["start_class"],
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
            "success_flag": bool(success),
            "failure_label": "success" if success else "target_miss",
            "governor_rejection_cause": "none" if success else "target_miss",
            "robustness_label": "not_evaluated",
            "physics_priority_level": "test",
            "sim_real_match_key": str(candidate["candidate_id"]),
            "sim_real_match_key_version": "test",
            "sim_real_transfer_result": "not_evaluated",
            "descriptor_status": "replay_evaluated",
            "heading_error_deg": 0.0 if success else 20.0,
            "energy_residual_m": 0.0,
            "lift_dwell_fraction": 0.0,
            "saturation_fraction": 0.0,
        }
    )
    return row
