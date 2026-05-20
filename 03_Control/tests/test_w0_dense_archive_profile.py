from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import profile_w0_dense_archive as profile
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_profile_records_worker_runtime_memory_and_gpu_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(profile, "_load_profile_inputs", _fake_inputs)
    monkeypatch.setattr(profile, "_run_pilot_replays", _fake_replay)
    monkeypatch.setattr(profile, "build_envelope_map", lambda frame: pd.DataFrame({"cell": [len(frame)]}))
    monkeypatch.setattr(profile, "_memory_total_gb", lambda: 32.0, raising=False)

    paths = profile.profile_w0_dense_archive(
        planning_run_id=12,
        result_root=tmp_path / "11_w0_dense_archive",
        profile_root=tmp_path / "profiles",
        sample_trials=4,
        storage_format="csv_gz",
        workers="auto",
    )

    payload = json.loads(paths["profile_json"].read_text(encoding="ascii"))
    assert payload["selected_worker_count"] in {6, 8}
    assert "1" in payload["rows_per_second_by_worker_count"]
    assert "4" in payload["rows_per_second_by_worker_count"]
    assert "6" in payload["rows_per_second_by_worker_count"]
    assert "8" in payload["rows_per_second_by_worker_count"]
    assert payload["memory_safety_margin_gb"] == 8.0
    assert "GPU acceleration is deferred" in payload["gpu_acceleration_assessment"]
    assert paths["profile_csv"].exists()
    assert not (tmp_path / "11_w0_dense_archive" / "013").exists()


def _fake_inputs(config) -> tuple[pd.DataFrame, list[dict[str, object]], float]:
    del config
    selected = [
        {"candidate_id": f"candidate_{index}", "layout_branch_id": "single_fan_branch"}
        for index in range(4)
    ]
    return pd.DataFrame({"sample_id": [f"sample_{index}" for index in range(4)]}), selected, 0.01


def _fake_replay(starts, selected, config) -> pd.DataFrame:
    del starts, config
    return pd.DataFrame(
        [_descriptor(row["candidate_id"], row["layout_branch_id"]) for row in selected],
        columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS,
    )


def _descriptor(candidate_id: str, branch: str) -> dict[str, object]:
    row = {column: "" for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS}
    row.update(
        {
            "trial_descriptor_id": f"trial_{candidate_id}",
            "layout_branch_id": branch,
            "fan_layout": "single_fan",
            "test_environment_mode": "W0_single_fan_branch",
            "family": "mild_bank",
            "target_heading_deg": 30.0,
            "direction_sign": 1,
            "start_class": "favourable",
            "latency_case": "nominal",
            "success_flag": True,
            "failure_label": "success",
            "governor_rejection_cause": "none",
            "robustness_label": "not_evaluated",
            "descriptor_status": "replay_evaluated",
            "updraft_relative_radius_m": 0.5,
            "speed0_m_s": 6.0,
            "w_wing_mean_m_s": 0.0,
            "delta_w_lr_m_s": 0.0,
            "min_true_margin_m": 0.5,
            "heading_error_deg": 0.0,
            "energy_residual_m": 0.0,
            "lift_dwell_fraction": 0.0,
            "saturation_fraction": 0.0,
        }
    )
    return row
