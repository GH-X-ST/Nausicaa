from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import aggregate_paired_w0_w1_archive as aggregate
from dense_archive_chunking import partition_path
from dense_archive_table_io import write_table_partition
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_paired_aggregation_writes_upload_and_governor_packages(
    tmp_path: Path,
) -> None:
    result_root = _short_result_root(tmp_path)
    _write_fake_paired_chunks(result_root, run_id=14)
    _write_progress_and_profile(result_root, run_id=14, planning_run_id=13)

    paths = aggregate.aggregate_paired_w0_w1_archive(
        run_id=14,
        planning_run_id=13,
        result_root=result_root,
        storage_format="csv_gz",
        expected_trials_per_environment=2,
        build_upload_package=True,
        build_governor_package=True,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    table_manifest = json.loads(paths["table_manifest_json"].read_text(encoding="ascii"))
    failed_valid = pd.read_csv(paths["w0_failed_w1_valid_summary_csv"])
    envelope = pd.read_csv(paths["w1_nominal_latency_envelope_summary_csv"])
    package = paths["upload_package_dir"]
    governor = paths["governor_package_dir"]

    assert manifest["selected_worker_count"] == 8
    assert manifest["runtime_core_version"] == "dense_archive_runtime_v1"
    assert manifest["storage_contract_version"] == "dense_archive_storage_contract_v1"
    assert manifest["w1_scheduled_independent_of_w0_success"] is True
    assert manifest["governor_artifacts_scan_raw_tables"] is False
    assert manifest["governor_package_contains_w0_candidates"] is False
    assert manifest["governor_package_branch_local_only"] is True
    assert manifest["paired_identity_seed_field"] == "seed"
    assert manifest["paired_seed_stable_across_w0_w1"] is True
    assert "candidate_id" not in manifest["paired_join_keys"]
    assert "replay_seed" not in manifest["paired_join_keys"]
    assert set(manifest["paired_audit_only_fields"]) == {
        "candidate_id",
        "sample_id",
        "test_environment_mode",
        "replay_seed",
    }
    assert all(
        "latency_execution_status" in partition["columns"]
        for partition in table_manifest["tables"]
    )
    assert "W0_failed_W1_valid_single_fan" in manifest["paired_summary_labels"]
    assert "W0_failed_W1_valid_four_fan" in manifest["paired_summary_labels"]
    assert {
        "W0_failed_W1_valid_single_fan": 1,
        "W0_failed_W1_valid_four_fan": 1,
    } == dict(zip(failed_valid["paired_summary_label"], failed_valid["trial_count"]))
    assert set(envelope["test_environment_mode"]) == {"W1_single_fan", "W1_four_fan"}
    assert set(envelope["latency_case"]) == {"nominal"}
    assert (package / "final_manifest.json").exists()
    assert (package / "paired_comparison_summary.csv").exists()
    assert (package / "w1_nominal_latency_envelope_summary.csv").exists()
    package_preview = pd.read_csv(package / "W1_single_fan_preview.csv")
    assert "latency_execution_status" in package_preview.columns
    assert (governor / "governor_package_manifest.json").exists()
    assert not any("tables" in path.parts for path in package.rglob("*") if path.is_file())
    assert not any("tables" in path.parts for path in governor.rglob("*") if path.is_file())
    candidates = pd.concat(
        pd.read_csv(path)
        for path in governor.rglob("*_candidates.csv.gz")
    )
    assert set(candidates["test_environment_mode"]) == {"W1_single_fan", "W1_four_fan"}


def test_paired_aggregation_rejects_corrupt_chunk_checksum(tmp_path: Path) -> None:
    result_root = _short_result_root(tmp_path)
    _write_fake_paired_chunks(result_root, run_id=14)
    manifest_path = (
        result_root
        / "014"
        / "chunk_manifests"
        / "layout_branch_id=single_fan_branch"
        / "test_environment_mode=W1_single_fan"
        / "chunk-00000.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["checksum_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        aggregate.aggregate_paired_w0_w1_archive(
            run_id=14,
            planning_run_id=13,
            result_root=result_root,
            storage_format="csv_gz",
        )


def test_default_paired_aggregation_requires_all_four_environments(tmp_path: Path) -> None:
    result_root = _short_result_root(tmp_path)
    _write_fake_paired_chunks(result_root, run_id=14, environments=(
        ("single_fan_branch", "single_fan", "W1_single_fan"),
        ("four_fan_branch", "four_fan", "W1_four_fan"),
    ))
    _write_progress_and_profile(result_root, run_id=14, planning_run_id=13)

    with pytest.raises(RuntimeError, match="missing chunk manifests for environment"):
        aggregate.aggregate_paired_w0_w1_archive(
            run_id=14,
            planning_run_id=13,
            result_root=result_root,
            storage_format="csv_gz",
        )


def test_w1_only_active_aggregation_does_not_require_w0_chunks(tmp_path: Path) -> None:
    result_root = _short_result_root(tmp_path)
    _write_fake_paired_chunks(result_root, run_id=14, environments=(
        ("single_fan_branch", "single_fan", "W1_single_fan"),
        ("four_fan_branch", "four_fan", "W1_four_fan"),
    ))
    _write_progress_and_profile(result_root, run_id=14, planning_run_id=13)

    paths = aggregate.aggregate_paired_w0_w1_archive(
        run_id=14,
        planning_run_id=13,
        result_root=result_root,
        storage_format="csv_gz",
        active_environment_modes=("W1_single_fan", "W1_four_fan"),
        expected_trials_per_environment=2,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["active_environment_modes"] == ["W1_single_fan", "W1_four_fan"]
    assert manifest["trial_count_by_environment"] == {
        "W1_four_fan": 2,
        "W1_single_fan": 2,
    }


def test_production_aggregation_accepts_separate_w0_w1_expected_counts(
    tmp_path: Path,
) -> None:
    result_root = _short_result_root(tmp_path)
    row_counts = {
        "W0_single_fan_branch": 1,
        "W0_four_fan_branch": 1,
        "W1_single_fan": 3,
        "W1_four_fan": 3,
    }
    _write_fake_paired_chunks(
        result_root,
        run_id=16,
        planning_run_id=15,
        row_counts_by_environment=row_counts,
    )
    _write_progress_and_profile(result_root, run_id=16, planning_run_id=15)

    paths = aggregate.aggregate_paired_w0_w1_archive(
        run_id=16,
        planning_run_id=15,
        result_root=result_root,
        storage_format="csv_gz",
        paired_scale_mode="production",
        expected_w0_trials_per_environment=1,
        expected_w1_trials_per_environment=3,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["expected_trials_per_environment"] is None
    assert manifest["expected_w0_trials_per_environment"] == 1
    assert manifest["expected_w1_trials_per_environment"] == 3
    assert manifest["expected_trials_by_environment"] == row_counts
    assert manifest["trial_count_by_environment"] == row_counts
    assert "D1a production-floor" in manifest["no_overclaiming_statement"]
    assert "proof only" not in manifest["no_overclaiming_statement"]


def test_aggregation_rejects_uniform_and_role_expected_count_conflict(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        aggregate.aggregate_paired_w0_w1_archive(
            run_id=14,
            planning_run_id=13,
            result_root=_short_result_root(tmp_path),
            storage_format="csv_gz",
            expected_trials_per_environment=2,
            expected_w0_trials_per_environment=1,
        )


def test_seed_instability_is_rejected_before_pairing(tmp_path: Path) -> None:
    result_root = _short_result_root(tmp_path)
    _write_fake_paired_chunks(result_root, run_id=14, unstable_seed=True)
    _write_progress_and_profile(result_root, run_id=14, planning_run_id=13)

    with pytest.raises(RuntimeError, match="paired identity seed is not stable"):
        aggregate.aggregate_paired_w0_w1_archive(
            run_id=14,
            planning_run_id=13,
            result_root=result_root,
            storage_format="csv_gz",
        )


def test_paired_diagnostic_slice_filters_environment(tmp_path: Path) -> None:
    result_root = _short_result_root(tmp_path)
    _write_fake_paired_chunks(result_root, run_id=14)

    path = aggregate.export_diagnostic_slice(
        run_id=14,
        result_root=result_root,
        layout_branch_id="single_fan_branch",
        test_environment_mode="W0_single_fan_branch",
        failure_label="target_miss",
        max_rows=10,
    )

    frame = pd.read_csv(path)
    assert set(frame["layout_branch_id"]) == {"single_fan_branch"}
    assert set(frame["test_environment_mode"]) == {"W0_single_fan_branch"}
    assert set(frame["failure_label"]) == {"target_miss"}


def _write_fake_paired_chunks(
    result_root: Path,
    *,
    run_id: int,
    planning_run_id: int = 13,
    environments: tuple[tuple[str, str, str], ...] | None = None,
    row_counts_by_environment: dict[str, int] | None = None,
    unstable_seed: bool = False,
) -> None:
    root = result_root / f"{run_id:03d}"
    active_environments = environments or (
        ("single_fan_branch", "single_fan", "W0_single_fan_branch"),
        ("single_fan_branch", "single_fan", "W1_single_fan"),
        ("four_fan_branch", "four_fan", "W0_four_fan_branch"),
        ("four_fan_branch", "four_fan", "W1_four_fan"),
    )
    for branch, fan, mode in active_environments:
        row_count = (
            int(row_counts_by_environment[mode])
            if row_counts_by_environment is not None
            else 2
        )
        rows = []
        for pair_index in range(row_count):
            latency_case = "nominal"
            latency_pass_label = "nominal_pass"
            if pair_index == 0:
                success = mode.startswith("W1_")
            else:
                success = True
            if pair_index == 1 and mode.startswith("W1_"):
                latency_case = "ideal"
                latency_pass_label = "ideal_only_pass"
            rows.append(
                _descriptor(
                    branch=branch,
                    fan=fan,
                    mode=mode,
                    pair_index=pair_index,
                    success=success,
                    latency_case=latency_case,
                    latency_pass_label=latency_pass_label,
                    unstable_seed=unstable_seed,
                )
            )
        frame = pd.DataFrame(rows, columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS)
        path = partition_path(
            root,
            table_name="trial_outcomes",
            layout_branch_id=branch,
            test_environment_mode=mode,
            chunk_index=0,
            storage_format="csv_gz",
        )
        partition = write_table_partition(frame, path, storage_format="csv_gz")
        manifest_path = (
            root
            / "chunk_manifests"
            / f"layout_branch_id={branch}"
            / f"test_environment_mode={mode}"
            / "chunk-00000.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "run_id": run_id,
                    "planning_run_id": planning_run_id,
                    "layout_branch_id": branch,
                    "test_environment_mode": mode,
                    "paired_environment_mode": _paired_mode(mode),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "chunk_size": row_count,
                    "storage_format": "csv_gz",
                    "compression_level": 1,
                    "latency_case": "nominal",
                    "dt_s": 0.02,
                    "horizon_s": 0.60,
                    "row_count": row_count,
                    "partition_path": path.resolve().as_posix(),
                    "checksum_sha256": partition.checksum_sha256,
                    "planning_read_s": 0.1,
                    "selection_s": 0.0,
                    "simulation_s": 0.1,
                    "descriptor_build_s": 0.0,
                    "write_s": 0.1,
                    "total_s": 0.3,
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )


def _short_result_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}" / "r"


def _write_progress_and_profile(
    result_root: Path,
    *,
    run_id: int,
    planning_run_id: int,
) -> None:
    progress = {
        "selected_worker_count": 8,
        "os_cpu_count": 20,
        "memory_total_gb": 32.0,
        "memory_safety_margin_gb": 8.0,
        "estimated_worker_memory_gb": 2.0,
        "worker_fallback_reason": "none",
        "rows_per_second_by_worker_count": {"1": 10.0, "4": 40.0, "6": 60.0, "8": 80.0},
    }
    progress_path = (
        result_root
        / f"{run_id:03d}"
        / "manifests"
        / f"paired_w0_w1_progress_s{run_id:03d}.json"
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="ascii")
    profile_path = (
        result_root
        / "profiles"
        / f"paired_planning_s{planning_run_id:03d}"
        / f"paired_w0_w1_profile_s{planning_run_id:03d}.json"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                **progress,
                "estimated_runtime_s_by_workers": {"8": 0.1},
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def _paired_mode(mode: str) -> str:
    mapping = {
        "W0_single_fan_branch": "W1_single_fan",
        "W1_single_fan": "W0_single_fan_branch",
        "W0_four_fan_branch": "W1_four_fan",
        "W1_four_fan": "W0_four_fan_branch",
    }
    return mapping[mode]


def _descriptor(
    *,
    branch: str,
    fan: str,
    mode: str,
    pair_index: int,
    success: bool,
    latency_case: str,
    latency_pass_label: str,
    unstable_seed: bool = False,
) -> dict[str, object]:
    audit_prefix = "w1" if mode.startswith("W1_") else "w0"
    row = {column: "" for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS}
    row.update(
        {
            "trial_descriptor_id": f"trial_{branch}_{mode}_{pair_index}",
            "layout_branch_id": branch,
            "fan_layout": fan,
            "fan_config_id": f"{fan}_dry_air",
            "test_environment_mode": mode,
            "paired_environment_mode": _paired_mode(mode),
            "environment_role": "updraft_capable" if mode.startswith("W1_") else "dry_air_gate",
            "validity_gate_role": "w1_independent" if mode.startswith("W1_") else "w0_gate",
            "acceptance_interpretation": "nominal_w1" if mode.startswith("W1_") else "w0_gate",
            "candidate_id": f"candidate_{audit_prefix}_{branch}_{pair_index}",
            "sample_id": f"sample_{audit_prefix}_{branch}_{pair_index}",
            "paired_sample_key": f"pair_{branch}_{pair_index}",
            "seed": 12 if unstable_seed and mode.startswith("W1_") else 11,
            "replay_seed": 17 if mode.startswith("W0_") else 99,
            "sampling_round": "test",
            "updraft_model_id": "fixture_updraft",
            "family": "mild_bank",
            "target_heading_deg": 30.0,
            "direction_sign": 1,
            "start_class": "favourable",
            "updraft_relative_radius_m": 0.5,
            "speed0_m_s": 6.0,
            "w_wing_mean_m_s": 0.0,
            "delta_w_lr_m_s": 0.0,
            "min_true_margin_m": 0.5,
            "latency_case": latency_case,
            "latency_acceptance_scope": "nominal",
            "latency_pass_label": latency_pass_label,
            "latency_execution_status": "command_delay_plus_actuator_lag"
            if latency_case == "nominal"
            else "ideal_timing",
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
            "sim_real_match_key": f"candidate_{audit_prefix}_{branch}_{pair_index}",
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
