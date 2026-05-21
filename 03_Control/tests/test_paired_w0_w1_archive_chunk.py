from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_paired_w0_w1_archive_chunk as chunk
import run_paired_w0_w1_partitioned_planning as planning
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_chunk_manifest_records_scale_mode_without_physics_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    planning.run_paired_w0_w1_partitioned_planning(
        run_id=13,
        result_root=planning_root,
        paired_scale_mode="proof",
        active_environment_modes=("W1_single_fan",),
        proof_target_trials_per_environment=1,
        partition_rows=1,
        storage_format="csv_gz",
    )
    monkeypatch.setattr(
        chunk,
        "_run_pilot_replays",
        lambda starts, selected, config: pd.DataFrame(
            columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS
        ),
    )

    paths = chunk.run_paired_w0_w1_archive_chunk(
        run_id=14,
        planning_run_id=13,
        result_root=archive_root,
        layout_branch_id="single_fan_branch",
        test_environment_mode="W1_single_fan",
        chunk_index=0,
        chunk_count=1,
        chunk_size=1,
        paired_scale_mode="proof",
        storage_format="csv_gz",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["paired_scale_mode"] == "proof"
    assert manifest["test_environment_mode"] == "W1_single_fan"
    assert manifest["runtime_core_version"] == "dense_archive_runtime_v1"


def test_chunk_manifest_carries_d1a_metadata_from_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    _set_d1a_contract(monkeypatch, "thesis_primary", w0=1, w1=1)
    planning.run_paired_w0_w1_partitioned_planning(
        run_id=15,
        result_root=planning_root,
        paired_scale_mode="production",
        w0_target_trials_per_branch=1,
        w1_floor_trials_per_branch=1,
        w1_target_trials_per_branch=1,
        partition_rows=1,
        storage_format="csv_gz",
        d1a_evidence_class="thesis_primary",
    )
    monkeypatch.setattr(
        chunk,
        "_run_pilot_replays",
        lambda starts, selected, config: pd.DataFrame(
            columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS
        ),
    )

    paths = chunk.run_paired_w0_w1_archive_chunk(
        run_id=16,
        planning_run_id=15,
        result_root=archive_root,
        layout_branch_id="single_fan_branch",
        test_environment_mode="W1_single_fan",
        chunk_index=0,
        chunk_count=1,
        chunk_size=1,
        paired_scale_mode="production",
        storage_format="csv_gz",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["d1a_evidence_class"] == "thesis_primary"
    assert manifest["d1a_target_contract"] == "updated_thesis_scale_v1"
    assert manifest["d1a_w0_trials_per_environment"] == 1
    assert manifest["d1a_w1_trials_per_environment"] == 1
    assert "D1a thesis-scale" in manifest["no_overclaiming_statement"]


def _short_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}"


def _set_d1a_contract(monkeypatch, evidence_class: str, *, w0: int, w1: int) -> None:
    updated = dict(planning.D1A_EVIDENCE_CONTRACTS)
    updated[evidence_class] = {
        "w0_trials_per_environment": int(w0),
        "w1_trials_per_environment": int(w1),
    }
    monkeypatch.setattr(planning, "D1A_EVIDENCE_CONTRACTS", updated)
