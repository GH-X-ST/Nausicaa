from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = REPO_ROOT / "03_Control"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_w0_dense_archive as w0_runner  # noqa: E402


TRACKED_DENSE_PLANNING_ROOT = CONTROL_DIR / "05_Results" / "10_dense_archive_planning"
TRACKED_W0_ROOT = CONTROL_DIR / "05_Results" / "11_w0_dense_archive"


def test_sibling_planning_root_inference(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    config = w0_runner.W0DenseArchiveConfig(result_root=result_root, planning_run_id=10)

    start_path, candidate_path = w0_runner._planning_paths(config)

    assert start_path == (
        tmp_path
        / "10_dense_archive_planning"
        / "010"
        / "metrics"
        / "equal_branch_start_state_manifest_pilot_s010.csv"
    )
    assert candidate_path == (
        tmp_path
        / "10_dense_archive_planning"
        / "010"
        / "metrics"
        / "equal_branch_dry_run_candidate_inventory_pilot_s010.csv"
    )


def test_missing_planning_csv_raises_before_output_creation(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"

    with pytest.raises(FileNotFoundError, match="missing planning"):
        w0_runner.run_w0_dense_archive(
            run_id=12,
            planning_run_id=10,
            result_root=result_root,
            archive_scale_mode="reduced",
        )

    assert not (result_root / "012").exists()


def test_w0_filtering_and_equal_branch_selection_are_deterministic() -> None:
    rows = [
        _candidate("single_2", "s2", "single_fan_branch", "W0_single_fan_branch", 2),
        _candidate("single_1", "s1", "single_fan_branch", "W0_single_fan_branch", 1),
        _candidate("single_w1", "sw1", "single_fan_branch", "W1_single_fan", 3),
        _candidate("four_2", "f2", "four_fan_branch", "W0_four_fan_branch", 2),
        _candidate("four_1", "f1", "four_fan_branch", "W0_four_fan_branch", 1),
        _candidate("four_w1", "fw1", "four_fan_branch", "W1_four_fan", 3),
    ]
    frame = pd.DataFrame(rows)
    config = w0_runner.W0DenseArchiveConfig(max_trials=4)

    w0_only = w0_runner._filter_w0_candidates(frame)
    selected_a = w0_runner._select_w0_candidates(w0_only, config)
    selected_b = w0_runner._select_w0_candidates(w0_only.sample(frac=1.0), config)

    assert set(w0_only["test_environment_mode"]) == {
        "W0_single_fan_branch",
        "W0_four_fan_branch",
    }
    assert [row["candidate_id"] for row in selected_a] == [
        "four_1",
        "four_2",
        "single_1",
        "single_2",
    ]
    assert [row["candidate_id"] for row in selected_a] == [
        row["candidate_id"] for row in selected_b
    ]


def test_strict_rejects_below_500k_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_planning_tables(result_root, single_count=4, four_count=4)
    replay_called = False

    def fail_if_replayed(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        nonlocal replay_called
        replay_called = True
        raise AssertionError("strict scale gate should reject before replay")

    monkeypatch.setattr(w0_runner, "_run_pilot_replays", fail_if_replayed)

    with pytest.raises(RuntimeError, match="strict W0 archive scale gate"):
        w0_runner.run_w0_dense_archive(
            run_id=12,
            planning_run_id=10,
            result_root=result_root,
            archive_scale_mode="strict",
        )

    assert replay_called is False
    assert not (result_root / "012").exists()


def test_strict_scale_status_passes_at_500k_target() -> None:
    config = w0_runner.W0DenseArchiveConfig()

    status = w0_runner._w0_scale_status(
        selected_by_branch={
            "single_fan_branch": 250000,
            "four_fan_branch": 250000,
        },
        config=config,
    )

    assert status == "meets_user_500k_target"
    w0_runner._enforce_scale_gate(config, status)


def test_reduced_mode_writes_below_floor_outputs_and_no_overclaims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_planning_tables(result_root, single_count=2, four_count=2)
    monkeypatch.setattr(w0_runner, "_run_pilot_replays", _fake_replay)

    paths = w0_runner.run_w0_dense_archive(
        run_id=12,
        planning_run_id=10,
        result_root=result_root,
        archive_scale_mode="reduced",
        max_trials=4,
        floor_trials_per_branch=10,
        target_trials_per_branch=20,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    report = paths["report_md"].read_text(encoding="ascii")
    assert manifest["archive_scale_mode"] == "reduced"
    assert manifest["w0_scale_status"] == "reduced_below_w0_floor"
    assert manifest["w0_dense_archive_performed"] is True
    assert manifest["w0_full_archive_performed"] is False
    assert manifest["reduced_w0_archive_performed"] is True
    assert manifest["production_w1_archive_performed"] is False
    assert manifest["w2_w3_w4_w5_performed"] is False
    assert manifest["hardware_or_mission_claim"] is False
    assert manifest["sim_to_real_transfer_claim"] is False
    assert "Reduced W0 archive performed: `true`" in report
    for key in (
        "trial_descriptors_csv",
        "envelope_map_csv",
        "cluster_representatives_csv",
        "cluster_diagnostics_csv",
    ):
        assert pd.read_csv(paths[key]).empty is False


def test_reduced_mode_records_floor_met_below_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_planning_tables(result_root, single_count=3, four_count=3)
    monkeypatch.setattr(w0_runner, "_run_pilot_replays", _fake_replay)

    paths = w0_runner.run_w0_dense_archive(
        run_id=12,
        planning_run_id=10,
        result_root=result_root,
        archive_scale_mode="reduced",
        max_trials=6,
        floor_trials_per_branch=2,
        target_trials_per_branch=4,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    report = paths["report_md"].read_text(encoding="ascii")
    assert manifest["w0_scale_status"] == "meets_w0_floor_below_user_target"
    assert manifest["floor_trials_per_branch"] == 2
    assert manifest["floor_trials_total"] == 4
    assert manifest["target_trials_per_branch"] == 4
    assert manifest["target_trials_total"] == 8
    assert manifest["reduced_w0_archive_performed"] is True
    assert "Floor trials per branch: `2`" in report
    assert "Target trials per branch: `4`" in report


def test_branch_balance_enforced_by_scale_status() -> None:
    config = w0_runner.W0DenseArchiveConfig(
        floor_trials_per_branch=2,
        target_trials_per_branch=3,
    )

    status = w0_runner._w0_scale_status(
        selected_by_branch={
            "single_fan_branch": 3,
            "four_fan_branch": 1,
        },
        config=config,
    )

    assert status == "below_w0_floor"


def test_no_writes_to_existing_runs_007_to_011(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    for run_id in range(7, 12):
        marker = result_root / f"{run_id:03d}" / "marker.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("preserve\n", encoding="ascii")
    _write_planning_tables(result_root, single_count=1, four_count=1)

    with pytest.raises(RuntimeError):
        w0_runner.run_w0_dense_archive(
            run_id=12,
            planning_run_id=10,
            result_root=result_root,
            archive_scale_mode="strict",
        )

    for run_id in range(7, 12):
        assert (result_root / f"{run_id:03d}" / "marker.txt").read_text(
            encoding="ascii"
        ) == "preserve\n"


def _write_planning_tables(
    w0_result_root: Path,
    *,
    single_count: int,
    four_count: int,
) -> None:
    planning_metrics = (
        w0_result_root.parent / "10_dense_archive_planning" / "010" / "metrics"
    )
    planning_metrics.mkdir(parents=True)
    candidates = [
        _candidate(f"single_{index:04d}", f"single_sample_{index:04d}", "single_fan_branch", "W0_single_fan_branch", index)
        for index in range(single_count)
    ]
    candidates.extend(
        _candidate(f"four_{index:04d}", f"four_sample_{index:04d}", "four_fan_branch", "W0_four_fan_branch", index)
        for index in range(four_count)
    )
    candidates.extend(
        [
            _candidate("single_w1", "single_w1_sample", "single_fan_branch", "W1_single_fan", 999),
            _candidate("four_w1", "four_w1_sample", "four_fan_branch", "W1_four_fan", 999),
        ]
    )
    starts = [
        _start(row["sample_id"], row["layout_branch_id"], row["fan_layout"])
        for row in candidates
    ]
    pd.DataFrame(starts).to_csv(
        planning_metrics / "equal_branch_start_state_manifest_pilot_s010.csv",
        index=False,
    )
    pd.DataFrame(candidates).to_csv(
        planning_metrics / "equal_branch_dry_run_candidate_inventory_pilot_s010.csv",
        index=False,
    )


def _candidate(
    candidate_id: str,
    sample_id: str,
    branch: str,
    mode: str,
    index: int,
) -> dict[str, object]:
    fan = "single_fan" if branch == "single_fan_branch" else "four_fan"
    paired = "W1_single_fan" if mode == "W0_single_fan_branch" else "W0_single_fan_branch"
    if fan == "four_fan":
        paired = "W1_four_fan" if mode == "W0_four_fan_branch" else "W0_four_fan_branch"
    return {
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 20260525 + int(index),
        "sampling_round": "pilot_round_0",
        "fan_layout": fan,
        "layout_branch_id": branch,
        "fan_config_id": f"{fan}_dry_air",
        "updraft_model_id": "no_updraft_dry_air" if mode.startswith("W0_") else f"{fan}_model",
        "test_environment_mode": mode,
        "paired_environment_mode": paired,
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "start_class": "favourable",
        "environment_role": "dry_air_capable",
        "validity_gate_role": "baseline_gate",
        "first_validity_gate_environment": mode,
        "w0_failure_policy": "baseline_gate",
        "acceptance_interpretation": "baseline_gate",
        "count_basis": "test",
        "planned_floor_trial_count": 1,
        "planned_target_trial_count": 1,
        "pilot_trial_count": 1,
        "latency_case_planned": "nominal",
        "latency_acceptance_role": "nominal",
        "latency_model_status": "active",
        "planned_replay_status": "not_replayed_in_this_task",
        "planned_result_path": "",
        "branch_decision_scope": "branch_local_only_no_cross_layout_decision_transfer",
        "no_cross_branch_promotion": True,
        "no_cross_branch_rejection": True,
        "no_cross_branch_cluster_merge": True,
        "no_cross_branch_safety_justification": True,
        "no_rollout_performed": True,
        "notes": "",
        "w_wing_mean_m_s": 0.0,
        "delta_w_lr_m_s": 0.0,
    }


def _start(sample_id: str, branch: str, fan: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 20260525,
        "sampling_round": "pilot_round_0",
        "fan_layout": fan,
        "layout_branch_id": branch,
        "fan_config_id": f"{fan}_dry_air",
        "updraft_model_id": "no_updraft_dry_air",
        "start_class": "favourable",
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "environment_role": "dry_air_capable",
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
        "updraft_relative_radius_m": 0.5,
        "w_wing_mean_m_s": 0.0,
        "delta_w_lr_m_s": 0.0,
    }


def _fake_replay(
    start_states: pd.DataFrame,
    selected: list[dict[str, object]],
    config: w0_runner.W0DenseArchiveConfig,
) -> pd.DataFrame:
    del start_states
    rows: list[dict[str, object]] = []
    for index, row in enumerate(selected):
        rows.append(
            {
                "trial_descriptor_id": f"trial_{index:04d}",
                "sim_real_match_key": f"match_{index:04d}",
                "layout_branch_id": row["layout_branch_id"],
                "fan_layout": row["fan_layout"],
                "fan_config_id": row["fan_config_id"],
                "updraft_model_id": row["updraft_model_id"],
                "test_environment_mode": row["test_environment_mode"],
                "paired_environment_mode": row["paired_environment_mode"],
                "family": row["family"],
                "target_heading_deg": row["target_heading_deg"],
                "direction_sign": row["direction_sign"],
                "start_class": row["start_class"],
                "latency_case": config.latency_case,
                "updraft_relative_radius_m": 0.5 + 0.01 * index,
                "speed0_m_s": 6.0,
                "w_wing_mean_m_s": 0.0,
                "delta_w_lr_m_s": 0.0,
                "min_true_margin_m": 0.5,
                "descriptor_status": "replay_evaluated",
                "success_flag": index % 2 == 0,
                "failure_label": "success" if index % 2 == 0 else "target_miss",
                "governor_rejection_cause": "none",
                "robustness_label": "not_evaluated",
                "heading_error_deg": 1.0 if index % 2 == 0 else 12.0,
                "energy_residual_m": 0.0,
                "lift_dwell_fraction": 0.0,
                "saturation_fraction": 0.0,
                "latency_pass_label": "nominal_pass" if index % 2 == 0 else "nominal_fail",
                "physics_priority_level": "test",
                "branch_decision_scope": "branch_local_only_no_cross_layout_decision_transfer",
            }
        )
    return pd.DataFrame(rows)
