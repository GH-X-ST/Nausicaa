from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fixed_gate_primitive_rollout as rollout_backend
from episode_schema import validate_primitive_rollout_evidence_frame
from fixed_gate_primitive_rollout import (
    FixedGatePrimitiveRolloutConfig,
    run_fixed_gate_primitive_rollouts,
)
from primitive_envelope_clustering import build_primitive_envelope_clusters
from run_fixed_gate_repeated_launch_policy_eval import run_fixed_gate_repeated_launch_policy_eval
from run_fixed_gate_w0_w1_archive import run_fixed_gate_w0_w1_archive
from run_matched_real_replay import build_matched_replay_table
from run_reachable_state_extraction import run_reachable_state_extraction
from run_real_flight_episode_ingest import real_flight_start_rejection_label
from run_w2_focused_replay import run_w2_focused_replay
from run_w3_domain_randomised_replay import run_w3_domain_randomised_replay


def test_open_loop_rollout_rows_are_diagnostic_and_not_accepted() -> None:
    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([_candidate_row("sample_a", "single_fan_branch", "W1")]),
        FixedGatePrimitiveRolloutConfig(dt_s=0.02, horizon_s=0.04, latency_case="none"),
    )

    validate_primitive_rollout_evidence_frame(rows)
    assert set(rows["controller_mode"]) == {"open_loop_rollout"}
    assert set(rows["feedback_mode"]) == {"open_loop"}
    assert set(rows["evidence_role"]) == {"ablation_diagnostic"}
    assert set(rows["claim_status"]) == {"simulation_only"}
    assert set(rows["wind_mode"]) == {"panel"}
    assert set(rows["updraft_model_id"]) == {"single_gaussian_var"}
    assert set(rows["wind_binding_status"]) == {"measured_updraft_bound"}
    assert not rows["wind_descriptor_model_source"].astype(str).str.contains("dry_air").any()
    assert not rows["accepted"].astype(bool).any()


def test_feedback_requested_without_delayed_path_writes_blocked_partial() -> None:
    candidate = _candidate_row("sample_b", "single_fan_branch", "W1")
    candidate["controller_mode"] = "feedback_stabilised_primitive"
    candidate["feedback_mode"] = "delayed_state_feedback"

    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([candidate]),
        FixedGatePrimitiveRolloutConfig(dt_s=0.02, horizon_s=0.04, latency_case="nominal"),
    )

    assert rows.loc[0, "evidence_role"] == "blocked_partial"
    assert rows.loc[0, "feedback_mode"] == "unavailable"
    assert rows.loc[0, "accepted"] is False or not bool(rows.loc[0, "accepted"])
    assert "blocked_true_delayed_state_feedback_unavailable" in rows.loc[0, "failure_label"]


def test_instant_feedback_uses_partial_feedback_not_mission_candidate() -> None:
    candidate = _candidate_row("sample_feedback", "single_fan_branch", "W0")
    candidate["primitive_family"] = "recovery"

    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([candidate]),
        FixedGatePrimitiveRolloutConfig(
            dt_s=0.02,
            horizon_s=0.04,
            latency_case="none",
            controller_mode="feedback_stabilised_primitive",
            feedback_mode="instant_state_feedback",
        ),
    )

    validate_primitive_rollout_evidence_frame(rows)
    assert rows.loc[0, "controller_mode"] == "feedback_stabilised_primitive"
    assert rows.loc[0, "feedback_mode"] == "instant_state_feedback"
    assert rows.loc[0, "evidence_role"] == "partial_feedback"
    assert rows.loc[0, "claim_status"] == "simulation_only"
    assert "primitive_entry" in rows.loc[0, "entry_check_status"]
    assert rows.loc[0, "state_feedback_delay_applied"] is False or not bool(rows.loc[0, "state_feedback_delay_applied"])


def test_w1_instant_feedback_uses_measured_wind_and_partial_feedback() -> None:
    candidate = _candidate_row("sample_w1_feedback", "four_fan_branch", "W1")
    candidate["primitive_family"] = "recovery"

    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([candidate]),
        FixedGatePrimitiveRolloutConfig(
            dt_s=0.02,
            horizon_s=0.04,
            latency_case="none",
            controller_mode="feedback_stabilised_primitive",
            feedback_mode="instant_state_feedback",
        ),
    )

    validate_primitive_rollout_evidence_frame(rows)
    assert rows.loc[0, "controller_mode"] == "feedback_stabilised_primitive"
    assert rows.loc[0, "feedback_mode"] == "instant_state_feedback"
    assert rows.loc[0, "evidence_role"] == "partial_feedback"
    assert rows.loc[0, "wind_mode"] == "panel"
    assert rows.loc[0, "updraft_model_id"] == "four_gaussian_var"
    assert rows.loc[0, "wind_binding_status"] == "measured_updraft_bound"
    assert rows.loc[0, "wind_descriptor_status"] == "wind_model_evaluated"
    assert "dry_air" not in str(rows.loc[0, "wind_descriptor_model_source"])
    assert rows.loc[0, "state_feedback_delay_applied"] is False or not bool(rows.loc[0, "state_feedback_delay_applied"])


def test_missing_w1_updraft_model_blocks_without_dry_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(_: str) -> object:
        raise FileNotFoundError("missing measured workbook")

    monkeypatch.setattr(rollout_backend, "load_updraft_model", fail_load)
    candidate = _candidate_row("sample_missing_wind", "single_fan_branch", "W1")
    candidate["primitive_family"] = "recovery"

    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([candidate]),
        FixedGatePrimitiveRolloutConfig(
            dt_s=0.02,
            horizon_s=0.04,
            latency_case="none",
            controller_mode="feedback_stabilised_primitive",
            feedback_mode="instant_state_feedback",
        ),
    )

    assert rows.loc[0, "evidence_role"] == "blocked_partial"
    assert rows.loc[0, "wind_mode"] == "none"
    assert str(rows.loc[0, "wind_binding_status"]).startswith("blocked_updraft_model_unavailable")
    assert str(rows.loc[0, "wind_descriptor_status"]).startswith("blocked_updraft_model_unavailable")
    assert "dry_air" not in str(rows.loc[0, "wind_descriptor_model_source"])


def test_schema_rejects_open_loop_partial_feedback_relabel() -> None:
    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([_candidate_row("sample_bad", "single_fan_branch", "W1")]),
        FixedGatePrimitiveRolloutConfig(dt_s=0.02, horizon_s=0.04, latency_case="none"),
    )
    bad = rows.copy()
    bad.loc[0, "evidence_role"] = "partial_feedback"

    with pytest.raises(ValueError, match="open-loop"):
        validate_primitive_rollout_evidence_frame(bad)


def test_w0_w1_archive_writes_split_evidence_tables_and_move_on_statuses(tmp_path: Path) -> None:
    paths = run_fixed_gate_w0_w1_archive(
        run_id=301,
        rows_per_branch=2,
        latency_case="none",
        result_root=tmp_path,
        storage_format="csv",
        overwrite=True,
    )

    rows = pd.read_csv(paths["rollout_rows_csv"])
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))

    assert {"single_fan_branch", "four_fan_branch"} == set(rows["fan_branch"])
    assert {"W0", "W1"} == set(rows["W_layer"])
    assert {"ablation_diagnostic", "partial_feedback", "blocked_partial"}.issubset(set(rows["evidence_role"]))
    assert manifest["open_loop_rows_promoted_to_mission_candidate"] is False
    assert manifest["open_loop_rows_promoted_to_partial_feedback"] is False
    assert manifest["code_ready_status"] == "ready"
    assert manifest["archive_prepared_status"] == "ready"
    assert manifest["w1_measured_updraft_row_count"] > 0
    assert set(manifest["w1_measured_updraft_row_count_by_branch"]) == {"single_fan_branch", "four_fan_branch"}
    assert "accepted_w0_partial_feedback_row_count" in manifest
    assert "accepted_w1_partial_feedback_row_count" in manifest
    assert manifest["mission_evidence_ready_status"] == "blocked_no_mission_or_partial_feedback_rows_for_both_branches"
    assert manifest["W1_independent_of_W0_success"] is True
    assert paths["diagnostic_rows_csv"].exists()
    assert paths["partial_feedback_rows_csv"].exists()
    assert paths["mission_candidate_rows_csv"].exists()
    assert paths["branch_coverage_summary_csv"].exists()
    assert paths["branch_coverage_report_md"].exists()
    coverage = pd.read_csv(paths["branch_coverage_summary_csv"])
    assert {
        "row_count_by_branch_layer",
        "non_dry_w1_measured_updraft_by_branch",
        "rejection_failure_counts_by_branch_primitive",
        "weak_partial_feedback_margin_energy_summary",
        "readiness_status",
    }.issubset(set(coverage["summary_section"]))
    assert set(coverage["archive_prepared_status"]) == {"ready"}
    with pytest.raises(RuntimeError, match="result tree already exists"):
        run_fixed_gate_w0_w1_archive(
            run_id=301,
            rows_per_branch=2,
            latency_case="none",
            result_root=tmp_path,
            storage_format="csv",
            overwrite=False,
        )


def test_clustering_keeps_diagnostic_medoids_out_of_governor_package() -> None:
    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([_candidate_row("sample_c", "single_fan_branch", "W1")]),
        FixedGatePrimitiveRolloutConfig(dt_s=0.02, horizon_s=0.04, latency_case="none"),
    )

    outputs = build_primitive_envelope_clusters(rows)

    assert len(outputs["cluster_medoids"]) == 1
    assert outputs["cluster_medoids"].iloc[0]["evidence_role"] == "ablation_diagnostic"
    assert outputs["governor_candidate_package"].empty


def test_reachable_extraction_accepts_only_mission_candidate_gate_rows(tmp_path: Path) -> None:
    source = _accepted_mission_rollout_row()
    rollout_csv = tmp_path / "rollouts.csv"
    pd.DataFrame([source]).to_csv(rollout_csv, index=False)

    paths = run_reachable_state_extraction(
        rollout_csv=rollout_csv,
        run_id=302,
        max_rows=4,
        result_root=tmp_path,
        overwrite=True,
    )

    reachable = pd.read_csv(paths["reachable_states_csv"])
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))

    assert len(reachable) == 1
    assert set(reachable["entry_source"]) == {"reachable_downstream"}
    assert manifest["readiness_status"] == "ready"


def test_w2_w3_runners_consume_selected_w1_rows_not_broad_samples(tmp_path: Path) -> None:
    source_csv = tmp_path / "source_rows.csv"
    rows = run_fixed_gate_primitive_rollouts(
        pd.DataFrame([_candidate_row("sample_d", "single_fan_branch", "W1")]),
        FixedGatePrimitiveRolloutConfig(dt_s=0.02, horizon_s=0.04, latency_case="none"),
    )
    rows.to_csv(source_csv, index=False)

    w2_paths = run_w2_focused_replay(
        source_csv=source_csv,
        run_id=303,
        max_cases=2,
        latency_case="none",
        result_root=tmp_path,
        overwrite=True,
    )
    w3_paths = run_w3_domain_randomised_replay(
        source_csv=w2_paths["replay_rows_csv"],
        run_id=304,
        max_cases=2,
        latency_case="none",
        result_root=tmp_path,
        overwrite=True,
    )

    w2_manifest = json.loads(w2_paths["manifest_json"].read_text(encoding="ascii"))
    w3_manifest = json.loads(w3_paths["manifest_json"].read_text(encoding="ascii"))
    assert w2_manifest["source_policy"] == "selected_W1_or_medoid_cases_only"
    assert w3_manifest["source_policy"] == "selected_W1_or_W2_medoid_cases_only"
    assert w2_manifest["dense_all_state_sweep"] is False
    assert w3_manifest["dense_all_state_sweep"] is False


def test_policy_eval_uses_clustering_package_not_default_toy_candidates(tmp_path: Path) -> None:
    package_csv = tmp_path / "governor_candidate_package.csv"
    pd.DataFrame(columns=["fan_branch", "primitive_id", "primitive_family"]).to_csv(package_csv, index=False)

    paths = run_fixed_gate_repeated_launch_policy_eval(
        governor_candidate_package_csv=package_csv,
        run_id=305,
        episodes_per_policy=1,
        result_root=tmp_path,
        overwrite=True,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["candidate_package_source"] == "fixed_gate_cluster_selection"
    assert manifest["default_toy_candidates_used"] is False
    assert manifest["policy_readiness_status"] == "blocked_no_clustering_candidate_package"


def test_real_ingest_labels_readiness_and_matched_replay_blocks_limited_transfer() -> None:
    low_speed = pd.DataFrame([_real_log_row(speed=1.0)])
    assert real_flight_start_rejection_label(low_speed) == "low_speed_stop"

    real = pd.DataFrame(
        [
            {
                "episode_id": "real_a",
                "initial_state_vector": "x",
                "termination_cause": "vicon_lost",
                "lift_capture_success": False,
                "energy_residual_m": 0.0,
                "lift_dwell_time_s": 0.0,
                "claim_status": "instrumentation_limited",
            }
        ]
    )
    sim = pd.DataFrame(
        [
            {
                "episode_id": "sim_a",
                "matched_replay_id": "real_a",
                "initial_state_vector": "x",
                "termination_cause": "vicon_lost",
                "lift_capture_success": False,
                "energy_residual_m": 0.0,
                "lift_dwell_time_s": 0.0,
            }
        ]
    )

    matched = build_matched_replay_table(real, sim)
    assert matched.loc[0, "transfer_label"] == "instrumentation_limited"


def _candidate_row(sample_id: str, fan_branch: str, W_layer: str) -> dict[str, object]:
    state = _gate_state()
    return {
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "fan_branch": fan_branch,
        "W_layer": W_layer,
        "test_environment_mode": f"{W_layer}_{fan_branch}",
        "entry_source": "launch_gate_main",
        "launch_gate_id": "fixed_gate_main_v1",
        "initial_state_vector": ";".join(f"{value:.12g}" for value in state),
        "initial_state_admission_status": "admitted_main_gate",
        "primitive_id": f"primitive_{sample_id}",
        "primitive_family": "glide",
    }


def _accepted_mission_rollout_row() -> dict[str, object]:
    row = _candidate_row("accepted_source", "single_fan_branch", "W1")
    row.update(
        {
            "trial_descriptor_id": "accepted_source",
            "accepted": True,
            "evidence_role": "mission_candidate",
            "x_terminal_w_m": 2.0,
            "y_terminal_w_m": 2.1,
            "z_terminal_w_m": 1.8,
            "phi_terminal_rad": 0.0,
            "theta_terminal_rad": 0.0,
            "psi_terminal_rad": 0.0,
            "u_terminal_m_s": 5.2,
            "v_terminal_m_s": 0.0,
            "w_terminal_m_s": 0.0,
            "p_terminal_rad_s": 0.0,
            "q_terminal_rad_s": 0.0,
            "r_terminal_rad_s": 0.0,
        }
    )
    return row


def _gate_state() -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = 1.3
    state[1] = 2.0
    state[2] = 1.7
    state[6] = 5.5
    return state


def _real_log_row(speed: float) -> dict[str, object]:
    state = _gate_state()
    return {
        "time_s": 0.0,
        "vicon_valid": True,
        "controller_ready": True,
        "x_w_m": state[0],
        "y_w_m": state[1],
        "z_w_m": state[2],
        "phi_rad": state[3],
        "theta_rad": state[4],
        "psi_rad": state[5],
        "u_m_s": float(speed),
        "v_m_s": 0.0,
        "w_m_s": 0.0,
    }
