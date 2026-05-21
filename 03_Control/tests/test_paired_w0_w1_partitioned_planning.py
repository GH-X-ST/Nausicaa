from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
