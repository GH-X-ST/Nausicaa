from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_w0_partitioned_planning as planning
from dense_archive_table_io import list_table_partitions, read_table_partition


def test_partitioned_planning_writes_w0_only_partitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    monkeypatch.setattr(planning, "_build_w0_tables", _fake_w0_tables)

    paths = planning.run_w0_partitioned_planning(
        run_id=12,
        source_planning_run_id=10,
        result_root=result_root,
        target_trials_total=8,
        target_trials_per_branch=4,
        floor_trials_per_branch=2,
        partition_rows=2,
        storage_format="csv_gz",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["actual_candidate_rows_total"] == 8
    assert manifest["actual_candidate_rows_by_branch"] == {
        "four_fan_branch": 4,
        "single_fan_branch": 4,
    }
    candidate_paths = list_table_partitions(paths["root"], "candidate_index")
    assert len(candidate_paths) == 4
    assert all("archive_chunk_index=" in path.as_posix() for path in candidate_paths)
    candidates = pd.concat([read_table_partition(path) for path in candidate_paths])
    required = {
        "archive_chunk_index",
        "archive_chunk_count",
        "chunk_local_index",
        "archive_chunk_size",
        "archive_branch_trial_index",
    }
    assert required.issubset(candidates.columns)
    assert set(candidates["archive_chunk_count"]) == {2}
    assert set(candidates["archive_chunk_size"]) == {2}
    assert set(candidates["test_environment_mode"]) == {
        "W0_single_fan_branch",
        "W0_four_fan_branch",
    }
    assert not any((paths["root"] / "tables").rglob("*W1*"))
    assert paths["single_preview_csv"].exists()
    assert paths["four_preview_csv"].exists()


def test_partitioned_planning_rejects_non_chunk_aligned_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(planning, "_build_w0_tables", _fake_w0_tables)

    try:
        planning.run_w0_partitioned_planning(
            run_id=12,
            source_planning_run_id=10,
            result_root=tmp_path / "10_dense_archive_planning",
            target_trials_total=10,
            target_trials_per_branch=5,
            floor_trials_per_branch=2,
            partition_rows=2,
            storage_format="csv_gz",
        )
    except ValueError as exc:
        assert "divisible by partition_rows" in str(exc)
    else:  # pragma: no cover - assertion branch.
        raise AssertionError("non chunk-aligned W0 planning target was accepted")


def _fake_w0_tables(config: planning.W0PartitionedPlanningConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for branch, fan, mode in (
        ("single_fan_branch", "single_fan", "W0_single_fan_branch"),
        ("four_fan_branch", "four_fan", "W0_four_fan_branch"),
    ):
        for index in range(4):
            sample_id = f"{branch}_{index}"
            starts.append(_start(sample_id, branch, fan))
            candidates.append(_candidate(sample_id, branch, fan, mode))
    return planning._add_archive_chunk_columns(
        pd.DataFrame(starts),
        pd.DataFrame(candidates),
        chunk_size=int(config.partition_rows),
        target_trials_per_branch=int(config.target_trials_per_branch),
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
