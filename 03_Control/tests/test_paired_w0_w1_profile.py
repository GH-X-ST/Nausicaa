from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import profile_paired_w0_w1_archive as profile
import run_paired_w0_w1_partitioned_planning as planning
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_profile_records_scale_and_active_environment_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    active_modes = ("W1_single_fan", "W1_four_fan")
    planning.run_paired_w0_w1_partitioned_planning(
        run_id=13,
        result_root=planning_root,
        paired_scale_mode="proof",
        active_environment_modes=active_modes,
        proof_target_trials_per_environment=1,
        partition_rows=1,
        storage_format="csv_gz",
    )
    monkeypatch.setattr(
        profile,
        "_run_pilot_replays",
        lambda starts, selected, config: pd.DataFrame(
            columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS
        ),
    )

    paths = profile.profile_paired_w0_w1_archive(
        planning_run_id=13,
        result_root=archive_root,
        paired_scale_mode="proof",
        active_environment_modes=active_modes,
        sample_trials=2,
        storage_format="csv_gz",
        workers=1,
    )

    payload = json.loads(paths["profile_json"].read_text(encoding="ascii"))
    assert payload["paired_scale_mode"] == "proof"
    assert payload["active_environment_modes"] == list(active_modes)
    assert payload["sample_trials_requested"] == 2


def _short_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}"
