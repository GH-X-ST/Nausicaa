from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import aggregate_w0_dense_archive as aggregate
from dense_archive_table_io import write_table_partition
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS


def test_strict_aggregation_writes_compact_upload_package(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)

    paths = aggregate.aggregate_w0_dense_archive(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        expected_trials_total=8,
        expected_trials_per_branch=4,
        storage_format="csv_gz",
        archive_scale_mode="strict",
        build_upload_package=True,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    package = paths["upload_package_dir"]
    assert manifest["trial_count_total"] == 8
    assert manifest["w0_full_archive_performed"] is True
    assert "GPU acceleration is deferred" in manifest["gpu_acceleration_assessment"]
    for name in (
        "final_manifest.json",
        "final_report.md",
        "table_manifest_summary.json",
        "chunk_manifest_summary.csv",
        "branch_counts.csv",
        "failure_summary.csv",
        "envelope_summary.csv",
        "cluster_diagnostics_summary.csv",
        "top_mixed_boundary_examples.csv",
        "top_failure_examples.csv",
        "top_success_examples.csv",
        "profiling_summary.json",
        "profiling_summary.csv",
        "schema_summary.csv",
        "command_history.md",
        "single_fan_branch_preview.csv",
        "four_fan_branch_preview.csv",
    ):
        assert (package / name).exists()
    assert not any("tables" in path.parts for path in package.rglob("*") if path.is_file())
    assert all(path.stat().st_size < aggregate.UPLOAD_PACKAGE_MAX_BYTES for path in package.rglob("*") if path.is_file())


def test_strict_aggregation_rejects_missing_count(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)

    with pytest.raises(RuntimeError, match="branch chunk schedule"):
        aggregate.aggregate_w0_dense_archive(
            run_id=13,
            planning_run_id=12,
            result_root=result_root,
            expected_trials_total=10,
            expected_trials_per_branch=5,
            storage_format="csv_gz",
            archive_scale_mode="strict",
        )


def test_aggregation_rejects_corrupt_chunk_checksum(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)
    manifest_path = (
        result_root
        / "013"
        / "chunk_manifests"
        / "layout_branch_id=single_fan_branch"
        / "chunk-00000.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["checksum_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        aggregate.aggregate_w0_dense_archive(
            run_id=13,
            planning_run_id=12,
            result_root=result_root,
            expected_trials_total=8,
            expected_trials_per_branch=4,
            storage_format="csv_gz",
            archive_scale_mode="strict",
        )


def test_aggregation_rejects_wrong_latency_case(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)
    manifest_path = (
        result_root
        / "013"
        / "chunk_manifests"
        / "layout_branch_id=single_fan_branch"
        / "chunk-00000.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["latency_case"] = "none"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="latency_case mismatch"):
        aggregate.aggregate_w0_dense_archive(
            run_id=13,
            planning_run_id=12,
            result_root=result_root,
            expected_trials_total=8,
            expected_trials_per_branch=4,
            storage_format="csv_gz",
            latency_case="nominal",
            archive_scale_mode="strict",
        )


def test_aggregation_reads_profile_from_custom_result_root(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)
    profile_root = result_root / "profiles" / "planning_s012"
    profile_root.mkdir(parents=True)
    (profile_root / "w0_profile_s012.json").write_text(
        json.dumps({"rows_per_second_by_worker_count": {"8": 80.0}}) + "\n",
        encoding="ascii",
    )

    paths = aggregate.aggregate_w0_dense_archive(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        expected_trials_total=8,
        expected_trials_per_branch=4,
        storage_format="csv_gz",
        archive_scale_mode="strict",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["profile_rows_per_second_by_worker_count"] == {"8": 80.0}


def test_targeted_diagnostic_slice_filters_without_full_upload(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)

    path = aggregate.export_diagnostic_slice(
        run_id=13,
        result_root=result_root,
        layout_branch_id="single_fan_branch",
        failure_label="target_miss",
        max_rows=10,
    )

    frame = pd.read_csv(path)
    assert set(frame["layout_branch_id"]) == {"single_fan_branch"}
    assert set(frame["failure_label"]) == {"target_miss"}


def _write_fake_chunks(result_root: Path, *, run_id: int) -> None:
    root = result_root / f"{run_id:03d}"
    for branch, fan, mode in (
        ("single_fan_branch", "single_fan", "W0_single_fan_branch"),
        ("four_fan_branch", "four_fan", "W0_four_fan_branch"),
    ):
        for chunk_index in range(2):
            rows = [
                _descriptor(
                    f"{branch}_{chunk_index}_{row_index}",
                    branch,
                    fan,
                    mode,
                    success=(row_index == 0),
                )
                for row_index in range(2)
            ]
            frame = pd.DataFrame(rows, columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS)
            partition_path = (
                root
                / "tables"
                / "trial_outcomes"
                / f"layout_branch_id={branch}"
                / f"chunk-{chunk_index:05d}.csv.gz"
            )
            partition = write_table_partition(frame, partition_path, storage_format="csv_gz")
            manifest_path = (
                root
                / "chunk_manifests"
                / f"layout_branch_id={branch}"
                / f"chunk-{chunk_index:05d}.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "run_id": run_id,
                        "planning_run_id": 12,
                        "layout_branch_id": branch,
                        "chunk_index": chunk_index,
                        "chunk_count": 2,
                        "chunk_size": 2,
                        "storage_format": "csv_gz",
                        "latency_case": "nominal",
                        "dt_s": 0.02,
                        "horizon_s": 0.60,
                        "row_count": 2,
                        "partition_path": partition.relative_path
                        if partition.relative_path.startswith("03_Control")
                        else partition_path.resolve().as_posix(),
                        "checksum_sha256": partition.checksum_sha256,
                        "planning_read_s": 0.1,
                        "selection_s": 0.1,
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


def _descriptor(
    trial_id: str,
    branch: str,
    fan: str,
    mode: str,
    *,
    success: bool,
) -> dict[str, object]:
    row = {column: "" for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS}
    row.update(
        {
            "trial_descriptor_id": f"trial_{trial_id}",
            "layout_branch_id": branch,
            "fan_layout": fan,
            "fan_config_id": f"{fan}_dry_air",
            "test_environment_mode": mode,
            "paired_environment_mode": mode.replace("W0", "W1"),
            "environment_role": "dry_air_capable",
            "validity_gate_role": "baseline_gate",
            "acceptance_interpretation": "baseline_gate",
            "candidate_id": f"candidate_{trial_id}",
            "sample_id": f"sample_{trial_id}",
            "paired_sample_key": f"pair_{trial_id}",
            "seed": 1,
            "replay_seed": 1,
            "sampling_round": "test",
            "updraft_model_id": "no_updraft_dry_air",
            "family": "mild_bank",
            "target_heading_deg": 30.0,
            "direction_sign": 1,
            "start_class": "favourable",
            "updraft_relative_radius_m": 0.5,
            "speed0_m_s": 6.0,
            "w_wing_mean_m_s": 0.0,
            "delta_w_lr_m_s": 0.0,
            "min_true_margin_m": 0.5,
            "duration_s": 0.6,
            "latency_case": "nominal",
            "latency_acceptance_scope": "nominal",
            "latency_pass_label": "nominal_pass" if success else "nominal_fail",
            "state_feedback_delay_s": 0.0,
            "command_onset_delay_s": 0.0,
            "command_transport_delay_s": 0.0,
            "actuator_tau_s": "0.0;0.0;0.0",
            "actuator_t50_s": 0.0,
            "actuator_t90_s": 0.0,
            "latency_jitter_s": 0.0,
            "timing_model_version": "test",
            "state_feedback_delay_applied": False,
            "success_flag": success,
            "failure_label": "success" if success else "target_miss",
            "governor_rejection_cause": "none",
            "robustness_label": "not_evaluated",
            "physics_priority_level": "test",
            "sim_real_match_key": trial_id,
            "sim_real_match_key_version": "test",
            "sim_real_transfer_result": "not_evaluated",
            "descriptor_status": "replay_evaluated",
            "heading_error_deg": 0.0 if success else 10.0,
            "energy_residual_m": 0.0,
            "lift_dwell_fraction": 0.0,
            "saturation_fraction": 0.0,
        }
    )
    return row
