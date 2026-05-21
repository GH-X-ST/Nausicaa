from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dense_archive_table_io import filesystem_path, list_table_partitions
import run_paired_w0_w1_partitioned_planning as planning


def test_proof_scale_defaults_write_exact_counts_and_seed_contract(tmp_path: Path) -> None:
    result_root = _short_root(tmp_path) / "10_dense_archive_planning"

    paths = planning.run_paired_w0_w1_partitioned_planning(
        run_id=13,
        result_root=result_root,
        paired_scale_mode="proof",
        proof_target_trials_per_environment=4,
        partition_rows=2,
        storage_format="csv_gz",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    counts = pd.read_csv(paths["branch_environment_counts_csv"])
    assert manifest["paired_scale_mode"] == "proof"
    assert manifest["active_environment_modes"] == list(planning.PAIRED_ENVIRONMENT_MODES)
    assert manifest["proof_target_trials_per_environment"] == 4
    assert manifest["effective_pilot_start_states_per_family_target_direction"] == 1
    assert manifest["paired_identity_seed_field"] == "seed"
    assert manifest["paired_seed_stable_across_w0_w1"] is True
    assert set(counts["candidate_rows"]) == {4}
    assert set(counts["test_environment_mode"]) == set(planning.PAIRED_ENVIRONMENT_MODES)


def test_direct_cli_runs_from_repo_root_without_pytest_path(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = Path("03_Control") / "04_Scenarios" / (
        "run_paired_w0_w1_partitioned_planning.py"
    )
    result_root = _long_root(tmp_path) / "direct_cli_planning"
    run_id = 913
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-id",
            str(run_id),
            "--result-root",
            str(result_root),
            "--storage-format",
            "csv_gz",
            "--proof-target-trials-per-environment",
            "4",
            "--partition-rows",
            "4",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    output_root = result_root / f"{run_id:03d}"
    assert filesystem_path(output_root).exists()
    assert (
        filesystem_path(
            output_root
            / "manifests"
            / f"paired_w0_w1_planning_manifest_s{run_id:03d}.json"
        ).exists()
    )
    assert (
        filesystem_path(
            output_root / "manifests" / f"table_manifest_s{run_id:03d}.json"
        ).exists()
    )
    assert (
        filesystem_path(
            output_root
            / "metrics_summary"
            / f"paired_w0_w1_branch_environment_counts_s{run_id:03d}.csv"
        ).exists()
    )
    assert (
        filesystem_path(
            output_root / "sample_preview" / f"paired_w0_w1_preview_s{run_id:03d}.csv"
        ).exists()
    )
    assert len(list_table_partitions(output_root, "start_states")) == 4
    assert len(list_table_partitions(output_root, "candidate_index")) == 4


def test_proof_mode_does_not_enforce_production_w1_floor() -> None:
    config = planning.PairedW0W1PartitionedPlanningConfig(
        run_id=20,
        paired_scale_mode="proof",
        proof_target_trials_per_environment=4,
        w1_floor_trials_per_branch=350000,
        w1_target_trials_per_branch=4,
        partition_rows=2,
    )

    planning._validate_config(config)


def test_production_mode_enforces_w1_floor_and_active_w1_branches() -> None:
    low_target = planning.PairedW0W1PartitionedPlanningConfig(
        run_id=20,
        paired_scale_mode="production",
        w1_floor_trials_per_branch=6,
        w1_target_trials_per_branch=4,
        partition_rows=2,
    )
    missing_w1_branch = planning.PairedW0W1PartitionedPlanningConfig(
        run_id=20,
        paired_scale_mode="production",
        active_environment_modes=("W1_single_fan",),
        w1_floor_trials_per_branch=4,
        w1_target_trials_per_branch=4,
        partition_rows=2,
    )

    with pytest.raises(ValueError, match="must not exceed target"):
        planning._validate_config(low_target)
    with pytest.raises(ValueError, match="requires both W1 branches active"):
        planning._validate_config(missing_w1_branch)


def test_production_floor_planning_manifest_uses_separate_w0_w1_counts(
    tmp_path: Path,
) -> None:
    result_root = _short_root(tmp_path) / "10_dense_archive_planning"

    paths = planning.run_paired_w0_w1_partitioned_planning(
        run_id=15,
        result_root=result_root,
        paired_scale_mode="production",
        w0_target_trials_per_branch=2,
        w1_floor_trials_per_branch=4,
        w1_target_trials_per_branch=4,
        partition_rows=2,
        storage_format="csv_gz",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    counts = pd.read_csv(paths["branch_environment_counts_csv"])
    count_by_mode = dict(zip(counts["test_environment_mode"], counts["candidate_rows"]))
    assert manifest["paired_scale_mode"] == "production"
    assert manifest["w0_target_trials_per_branch"] == 2
    assert manifest["w1_floor_trials_per_branch"] == 4
    assert manifest["w1_target_trials_per_branch"] == 4
    assert manifest["candidate_rows_total"] == 12
    assert count_by_mode == {
        "W0_four_fan_branch": 2,
        "W0_single_fan_branch": 2,
        "W1_four_fan": 4,
        "W1_single_fan": 4,
    }
    assert "D1a production-floor" in manifest["no_overclaiming_statement"]
    assert "proof only" not in manifest["no_overclaiming_statement"]


def test_default_run_id_guard_checks_013_and_014_before_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    monkeypatch.setattr(planning, "DEFAULT_RESULT_ROOT", planning_root)
    monkeypatch.setattr(planning, "DEFAULT_PAIRED_ARCHIVE_RESULT_ROOT", archive_root)
    (archive_root / "014").mkdir(parents=True)

    with pytest.raises(ValueError, match="013/014"):
        planning.run_paired_w0_w1_partitioned_planning(
            run_id=13,
            paired_scale_mode="proof",
            proof_target_trials_per_environment=2,
            partition_rows=1,
            storage_format="csv_gz",
        )


def _short_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}"


def _long_root(tmp_path: Path) -> Path:
    root = _short_root(tmp_path)
    for index in range(10):
        root = root / f"long_planning_path_segment_{index:02d}"
    return root
