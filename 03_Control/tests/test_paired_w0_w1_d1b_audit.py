from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import run_paired_w0_w1_d1b_audit as audit
from dense_archive_table_io import TableManifest, write_table_manifest, write_table_partition


def test_d1b_audit_writes_required_fixture_outputs(tmp_path: Path) -> None:
    result_root = _write_fixture_archive(tmp_path)
    config = audit.D1bAuditConfig(
        archive_run_id=16,
        planning_run_id=15,
        result_root=result_root,
        audit_root=result_root / "016" / "d1b_audit",
        expected_w0_trials_per_environment=2,
        expected_w1_trials_per_environment=3,
        observed_wall_time_min=1.0,
    )

    outputs = audit.run_d1b_audit(config)

    manifest = json.loads(outputs.manifest_json.read_text(encoding="ascii"))
    branch_summary = pd.read_csv(outputs.branch_environment_summary_csv)
    ladder = pd.read_csv(outputs.w1_target_ladder_summary_csv)
    latency = pd.read_csv(outputs.latency_acceptance_summary_csv)

    assert manifest["final_d1b_classification"] == "ready_for_D2_boundary_refinement"
    assert manifest["row_counts_by_environment"] == {
        "W0_single_fan_branch": 2,
        "W0_four_fan_branch": 2,
        "W1_single_fan": 3,
        "W1_four_fan": 3,
    }
    assert {
        "row_count",
        "success_count",
        "failure_count",
        "nominal_pass_count",
        "latency_label_counts_json",
    }.issubset(branch_summary.columns)
    assert {
        "success_rate",
        "nominal_pass_rate",
        "failure_label_distribution_json",
        "heading_error_deg_median",
    }.issubset(ladder.columns)
    assert set(latency["interpretation"]) == {
        "nominal open-loop evidence is not hardware-ready "
        "delayed-state-feedback evidence"
    }
    assert outputs.recommendation_md.exists()


def test_agile_ladder_preserves_full_targets_and_branches() -> None:
    rows = []
    for branch, fan, env in (
        ("single_fan_branch", "single_fan", "W1_single_fan"),
        ("four_fan_branch", "four_fan", "W1_four_fan"),
    ):
        rows.append(_trial_row(branch, fan, env, "canyon_steep_bank", 15.0, -1, True))
    frame = pd.DataFrame(rows)

    summary = audit.build_agile_family_ladder_summary(frame)

    assert len(summary) == 2 * 3 * 8 * 2
    assert set(summary["target_heading_deg"]) == set(audit.TARGET_LADDER_DEG)
    assert set(summary["layout_branch_id"]) == {"single_fan_branch", "four_fan_branch"}
    assert set(summary["direction_sign"]) == {-1, 1}


def test_reproducibility_selector_preserves_branch_and_w0_w1_ratio() -> None:
    rows = []
    for branch, fan, w0, w1 in (
        ("single_fan_branch", "single_fan", "W0_single_fan_branch", "W1_single_fan"),
        ("four_fan_branch", "four_fan", "W0_four_fan_branch", "W1_four_fan"),
    ):
        for index in range(10):
            rows.append(_trial_row(branch, fan, w0, "mild_bank", 15.0, 1, index % 2 == 0))
        for index in range(40):
            rows.append(_trial_row(branch, fan, w1, "wingover_lite", 30.0, -1, index % 3 == 0))
    frame = pd.DataFrame(rows)

    selected = audit.select_reproducibility_rows(frame, total_rows=20)

    assert len(selected) == 20
    assert selected.groupby("layout_branch_id").size().to_dict() == {
        "four_fan_branch": 10,
        "single_fan_branch": 10,
    }
    assert selected.groupby("test_environment_mode").size().to_dict() == {
        "W0_four_fan_branch": 2,
        "W0_single_fan_branch": 2,
        "W1_four_fan": 8,
        "W1_single_fan": 8,
    }


def test_no_overclaiming_validation_rejects_missing_denials() -> None:
    errors = audit._no_overclaiming_errors("D1b simulation evidence only.")

    assert errors
    assert any("production-floor" in item for item in errors)


def test_audit_refuses_non_empty_output_root(tmp_path: Path) -> None:
    root = tmp_path / "d1b_audit"
    root.mkdir()
    (root / "existing.txt").write_text("stop", encoding="ascii")

    with pytest.raises(RuntimeError, match="non-empty"):
        audit._prepare_output_root(root)


def _write_fixture_archive(tmp_path: Path) -> Path:
    result_root = tmp_path / "12_paired_w0_w1_archive"
    archive_root = result_root / "016"
    planning_root = tmp_path / "10_dense_archive_planning" / "015"
    rows = [
        _trial_row("single_fan_branch", "single_fan", "W0_single_fan_branch", "mild_bank", 15.0, 1, False),
        _trial_row("single_fan_branch", "single_fan", "W0_single_fan_branch", "mild_bank", 30.0, -1, False),
        _trial_row("single_fan_branch", "single_fan", "W1_single_fan", "mild_bank", 15.0, 1, True),
        _trial_row("single_fan_branch", "single_fan", "W1_single_fan", "wingover_lite", 30.0, -1, False),
        _trial_row("single_fan_branch", "single_fan", "W1_single_fan", "canyon_steep_bank", 45.0, 1, False),
        _trial_row("four_fan_branch", "four_fan", "W0_four_fan_branch", "mild_bank", 15.0, 1, False),
        _trial_row("four_fan_branch", "four_fan", "W0_four_fan_branch", "mild_bank", 30.0, -1, False),
        _trial_row("four_fan_branch", "four_fan", "W1_four_fan", "mild_bank", 15.0, 1, True),
        _trial_row("four_fan_branch", "four_fan", "W1_four_fan", "wingover_lite", 30.0, -1, False),
        _trial_row("four_fan_branch", "four_fan", "W1_four_fan", "canyon_steep_bank", 45.0, 1, False),
    ]
    partition = write_table_partition(
        pd.DataFrame(rows),
        archive_root / "tables" / "trial_outcomes" / "part-00000.csv.gz",
        storage_format="csv_gz",
    )
    write_table_manifest(
        archive_root / "manifests" / "table_manifest_s016.json",
        TableManifest(
            run_id=16,
            root=archive_root.as_posix(),
            storage_format="csv_gz",
            tables=(partition,),
        ),
    )
    (planning_root / "manifests").mkdir(parents=True)
    _write_json(
        planning_root / "manifests" / "paired_w0_w1_planning_manifest_s015.json",
        {
            "run_id": 15,
            "d1a_evidence_class": "thesis_primary",
            "d1a_target_contract": "updated_thesis_scale_v1",
        },
    )
    _write_json(
        archive_root / "manifests" / "paired_w0_w1_progress_s016.json",
        {
            "status": "complete",
            "scheduled_chunk_count": 100,
            "completed_chunk_count": 100,
            "pending_chunk_count": 0,
            "failed_chunk_count": 0,
        },
    )
    _write_json(
        archive_root / "manifests" / "paired_w0_w1_manifest_s016.json",
        {
            "run_id": 16,
            "planning_run_id": 15,
            "d1a_evidence_class": "thesis_primary",
            "d1a_target_contract": "updated_thesis_scale_v1",
            "chunk_manifest_count": 100,
            "trial_count_total": 10,
            "trial_count_by_environment": {
                "W0_single_fan_branch": 2,
                "W0_four_fan_branch": 2,
                "W1_single_fan": 3,
                "W1_four_fan": 3,
            },
            "trial_count_by_branch": {
                "single_fan_branch": 5,
                "four_fan_branch": 5,
            },
            "w1_scheduled_independent_of_w0_success": True,
            "single_fan_and_four_fan_never_merged": True,
            "branch_local_decisions_only": True,
            "governor_package_contains_w0_candidates": False,
            "governor_artifacts_scan_raw_tables": False,
            "governor_package_branch_local_only": True,
            "w2_w3_w4_w5_performed": False,
            "hardware_or_mission_claim": False,
            "sim_to_real_transfer_claim": False,
            "storage_format": "csv_gz",
            "selected_worker_count": 8,
            "worker_fallback_reason": "none",
            "no_overclaiming_statement": (
                "D1a thesis-scale paired W0/W1 aggregation simulation evidence only; "
                "no production-floor completion, W2/W3/W4/W5 completion, mission "
                "success, hardware readiness, or sim-to-real completion claim is made."
            ),
        },
    )
    upload = archive_root / "upload_package"
    governor = archive_root / "compressed_governor_package"
    upload.mkdir(parents=True)
    governor.mkdir(parents=True)
    _write_json(upload / "final_manifest.json", {"run_id": 16})
    _write_json(governor / "governor_package_manifest.json", {"raw_tables_included": False})
    metrics = archive_root / "metrics_summary"
    metrics.mkdir()
    pd.DataFrame(
        [
            {
                "paired_summary_label": "W0_failed_W1_valid_single_fan",
                "layout_branch_id": "single_fan_branch",
                "fan_layout": "single_fan",
                "trial_count": 1,
            },
            {
                "paired_summary_label": "W0_failed_W1_valid_four_fan",
                "layout_branch_id": "four_fan_branch",
                "fan_layout": "four_fan",
                "trial_count": 1,
            },
        ]
    ).to_csv(metrics / "w0_failed_w1_valid_summary_s016.csv", index=False)
    return result_root


def _trial_row(
    branch: str,
    fan: str,
    environment: str,
    family: str,
    target: float,
    direction: int,
    success: bool,
) -> dict[str, object]:
    index = abs(hash((branch, environment, family, target, direction, success))) % 100000
    return {
        "trial_descriptor_id": f"trial_{branch}_{environment}_{family}_{target}_{direction}_{index}",
        "layout_branch_id": branch,
        "fan_layout": fan,
        "fan_config_id": f"{branch}_config",
        "test_environment_mode": environment,
        "paired_environment_mode": environment.replace("W0_", "W1_"),
        "environment_role": "dry_air_capable",
        "candidate_id": f"candidate_{index}",
        "sample_id": f"sample_{index}",
        "paired_sample_key": f"paired_{index}",
        "seed": index,
        "replay_seed": index + 10,
        "sampling_round": 0,
        "updraft_model_id": f"{fan}_model",
        "family": family,
        "target_heading_deg": target,
        "direction_sign": direction,
        "start_class": "favourable",
        "x0_w_m": float(index % 3),
        "y0_w_m": float(index % 5),
        "z0_w_m": 1.0,
        "speed0_m_s": 6.0,
        "phi0_rad": 0.0,
        "theta0_rad": 0.0,
        "psi0_rad": 0.0,
        "u0_m_s": 6.0,
        "v0_m_s": 0.0,
        "w0_m_s": 0.0,
        "p0_rad_s": 0.0,
        "q0_rad_s": 0.0,
        "r0_rad_s": 0.0,
        "updraft_relative_radius_m": 1.0,
        "updraft_relative_azimuth_rad": 0.0,
        "w_wing_mean_m_s": 0.5,
        "w_left_m_s": 0.4,
        "w_right_m_s": 0.6,
        "delta_w_lr_m_s": -0.2,
        "terminal_speed_m_s": 6.1,
        "heading_error_deg": 1.0 if success else 20.0,
        "min_true_margin_m": 0.5 if success else 0.1,
        "energy_residual_m": 0.2,
        "lift_dwell_fraction": 0.6,
        "latency_case": "nominal",
        "latency_pass_label": "nominal_pass" if success else "nominal_fail",
        "latency_execution_status": "active_nominal_open_loop",
        "success_flag": success,
        "failure_label": "success" if success else "target_miss",
        "governor_rejection_cause": "none",
        "descriptor_status": "replay_evaluated",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
