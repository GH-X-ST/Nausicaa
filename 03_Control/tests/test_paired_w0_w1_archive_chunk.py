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


def _short_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}"
