from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dense_archive_table_io import load_table_manifest, read_table_partition
from prim_cat import ACTIVE_PRIMITIVE_IDS
from run_lqr_w01_dense_chunked import W01DenseRunConfig, run_lqr_w01_dense_chunked


def test_w01_tiny_smoke_covers_primitives_start_families_and_layers(tmp_path: Path) -> None:
    result = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(
            run_id=2,
            output_root=tmp_path,
            rows=2400,
            seed=2,
            candidate_chunk_size=400,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            compression_level=1,
            candidate_count=1,
        )
    )
    run_root = Path(result["run_root"])
    manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    frame = pd.concat(
        [
            read_table_partition(run_root / "tables" / partition.relative_path, storage_format=partition.storage_format)
            for partition in manifest.tables
        ],
        ignore_index=True,
    )

    assert set(frame["primitive_id"]) == set(ACTIVE_PRIMITIVE_IDS)
    assert set(frame["start_state_family"]) == {
        "launch_gate",
        "inflight_nominal",
        "inflight_lift_region",
        "inflight_boundary_near",
        "inflight_recovery_edge",
    }
    assert {"W0", "W1"}.issubset(set(frame["W_layer"]))
    assert {"dry_air", "w1_annular_gp_randomised_single", "w1_annular_gp_randomised_four"}.issubset(
        set(frame["environment_mode"])
    )
    assert set(frame["small_library_selection_allowed"]) == {False}
    assert set(frame["pd_pid_fallback_allowed"]) == {False}
    assert set(frame["schedule_mode"]) == {"balanced_paired"}
    assert "context_W_layer" in frame.columns
    assert "surrogate_surrogate_binding_status" in frame.columns
    assert "environment_environment_id" in frame.columns
    assert "implementation_instance_status" in frame.columns
    assert "plant_instance_status" in frame.columns
    assert "timing_aware_synthesis_level" in frame.columns
    assert set(frame["controller_design_role"]) == {"active_timing_aware_w01"}
    assert set(frame["timing_augmentation_type"]) == {
        "actuator_surface_state_command_fifo_predictor_compensated"
    }
    assert frame["controller_id"].astype(str).str.startswith("lqrta_").all()
    assert frame["active_timing_aware_controller_used"].astype(bool).all()
    assert not frame["baseline_controller_active"].astype(bool).any()
    model_rows = frame[frame["rollout_backend"].astype(str).eq("model_backed_lqr")]
    assert not model_rows.empty
    assert set(model_rows["timing_state_source"]) == {"history_backed_fifo"}
    dry_model_rows = model_rows[model_rows["environment_mode"].astype(str).eq("dry_air")]
    assert not dry_model_rows.empty
    assert set(dry_model_rows["implementation_W_layer"]) == {"W3"}
    assert set(dry_model_rows["plant_W_layer"]) == {"W3"}
    assert set(dry_model_rows["environment_W_layer"]) == {"W0"}
    assert set(dry_model_rows["environment_updraft_model_id"]) == {"dry_air_zero_wind"}
    assert "augmented_gain_checksum" in frame.columns
    assert not frame["candidate_weight_label"].astype(str).eq("W0_W1").any()

    run_manifest = json.loads((run_root / "manifests" / "run_manifest.json").read_text(encoding="ascii"))
    assert run_manifest["schedule_mode"] == "balanced_paired"
    assert run_manifest["project_title_version"] == "LQR-Stabilised Contextual Primitive v5.3"
    assert run_manifest["method_evidence_level"] == "w01_smoke_or_preflight_only"
    assert run_manifest["w01_dense_evidence_complete"] is False
    assert run_manifest["w01_dense_required_for_w3"] is True
    assert run_manifest["w01_dense_required_for_w2"] is False
    assert run_manifest["primitive_timing_contract"]["primitive_timing_contract_version"] == "v411_0p10s_5slot_20ms"
    assert set(frame["finite_horizon_s"]) == {0.1}
    assert set(frame["controller_input_slots_per_primitive"]) == {5}
    assert set(frame["controller_input_update_period_s"]) == {0.02}
    assert run_manifest["active_controller_design_role"] == "active_timing_aware_w01"
    assert run_manifest["per_start_family_row_counts"]["launch_gate"] > 0
    assert run_manifest["r5_launch_aware_decision"] == "R5_TRANSITION_AWARE_DENSE_INCOMPLETE_RESUME_REQUIRED"
    assert run_manifest["active_primitive_count"] == 8
    assert (run_root / "metrics" / "r5_launch_gate_entry_diagnosis.csv").is_file()
    diagnosis = pd.read_csv(run_root / "metrics" / "r5_launch_gate_entry_diagnosis.csv")
    assert {
        "start_state_family",
        "primitive_id",
        "primitive_family",
        "entry_role",
        "regime_label",
        "accepted_count",
        "weak_count",
        "failed_count",
        "rejected_count",
        "blocked_count",
        "continuation_valid_count",
        "terminal_useful_count",
        "hard_failure_count",
        "entry_role_rejection_count",
        "total_rows",
        "rejection_rate",
    }.issubset(diagnosis.columns)
    assert "launch_entry_evidence_for_8_families" in set(diagnosis["regime_label"])
    assert (run_root / "reports" / "r5_launch_capture_diagnosis.md").is_file()
    assert (run_root / "reports" / "r5_launch_gate_entry_diagnosis.md").is_file()
    l6_report = (run_root / "reports" / "l6_move_on_check.md").read_text(encoding="ascii")
    assert "predictor_compensated_augmented_discrete_lqr_v1" in l6_report
    assert "below_19200_fallback_scale_threshold" in l6_report


def test_w01_transition_entry_schedule_avoids_launch_gate_entry_rejections(tmp_path: Path) -> None:
    result = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(
            run_id=3,
            output_root=tmp_path,
            rows=2400,
            seed=3,
            candidate_chunk_size=400,
            workers=1,
            max_workers=1,
            candidate_count=1,
        )
    )
    run_root = Path(result["run_root"])
    manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    frame = read_table_partition(
        run_root / "tables" / manifest.tables[0].relative_path,
        storage_format=manifest.tables[0].storage_format,
    )
    launch_rows = frame[frame["start_state_family"].eq("launch_gate")]
    non_launch_rows = frame[frame["start_state_family"].ne("launch_gate")]

    assert not launch_rows.empty
    assert set(launch_rows["entry_role"]) == {"transition_object"}
    assert set(non_launch_rows["entry_role"]) == {"transition_object"}
    assert not frame["entry_check_status"].astype(str).eq("entry_role_incompatible_start").any()
    assert not frame["failure_label"].astype(str).eq("entry_role_incompatible_start_family").any()
