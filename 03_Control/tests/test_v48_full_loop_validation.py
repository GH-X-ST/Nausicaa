from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from episodic_lift_belief import LiftObservation, initial_belief, update_belief
from run_full_loop_validation import FullLoopValidationConfig, run_full_loop_validation
from run_outcome_model_build import build_outcome_model_rows
from run_v48_source_audit import V48SourceAuditConfig, run_v48_source_audit
from viability_governor import governor_candidate_row


def test_v48_source_audit_accepts_default_chain(tmp_path: Path) -> None:
    result = run_v48_source_audit(V48SourceAuditConfig(output_root=tmp_path / "full_loop_validation" / "001"))

    assert result["status"] == "source_audit_pass"
    assert result["blockers"] == []


def test_v48_source_audit_rejects_missing_method_roots(tmp_path: Path) -> None:
    result = run_v48_source_audit(
        V48SourceAuditConfig(
            output_root=tmp_path / "full_loop_validation" / "bad",
            w01_root=tmp_path / "missing_w01",
            w2_root=tmp_path / "missing_w2",
            w3_root=tmp_path / "missing_w3",
            post_w3_root=tmp_path / "missing_post_w3",
            outcome_smoke_root=tmp_path / "missing_outcome",
            governor_smoke_root=tmp_path / "missing_governor",
        )
    )

    assert result["status"] == "blocked_source_audit_failed"
    assert "W01_table_row_count_not_76800" in result["blockers"]


def test_v48_outcome_model_keeps_terminal_and_continuation_separate() -> None:
    rows = build_outcome_model_rows(
        [
            {
                "compact_library_id": "lib_1",
                "primitive_variant_id": "primvar_1",
                "primitive_id": "glide",
                "entry_role": "launch_capable",
                "controller_id": "ctrl_1",
                "continuation_valid_rate": 0.25,
                "episode_terminal_useful_rate": 0.75,
                "hard_failure_rate": 0.10,
                "expected_energy_residual_m": 0.2,
                "expected_lift_dwell_time_s": 0.4,
                "minimum_wall_margin_min_m": 0.3,
                "floor_margin_min_m": 0.4,
                "ceiling_margin_min_m": 0.5,
            }
        ]
    )

    assert rows[0]["continuation_probability"] == 0.25
    assert rows[0]["terminal_useful_probability"] == 0.75
    assert rows[0]["prediction_source"] == "W3_summary_interpretable"


def test_v48_governor_rejects_incompatible_without_controller_failure() -> None:
    representative = {
        "compact_library_id": "lib_1",
        "primitive_variant_id": "primvar_lift",
        "primitive_id": "lift_entry",
        "entry_role": "inflight_only",
        "controller_id": "ctrl_lift",
        "K_gain_checksum": "k",
        "augmented_A_checksum": "a",
        "augmented_B_checksum": "b",
        "augmented_gain_checksum": "g",
    }
    outcome = {
        "continuation_probability": 1.0,
        "terminal_useful_probability": 0.0,
        "hard_failure_risk": 0.0,
    }
    context = {
        "context_id": "ctx",
        "W_layer": "W0",
        "environment_mode": "dry_air",
        "start_state_family": "launch_gate",
        "latency_case": "none",
        "wall_margin_m": 1.0,
        "floor_margin_m": 1.0,
        "ceiling_margin_m": 1.0,
        "speed_margin_m_s": 1.0,
    }

    row = governor_candidate_row(
        representative=representative,
        outcome=outcome,
        context=context,
        governor_mode="continuation_mode",
    )

    assert row["viable"] is False
    assert row["rejection_reason"] == "entry_role_incompatible_start_family"
    assert row["claim_status"] == "simulation_only_viability_governor_candidate"


def test_v48_belief_lambda_update_exact() -> None:
    observation = LiftObservation(x_w_m=2.0, y_w_m=1.0, lift_evidence_m_s=2.0)
    for lambda_value, expected in ((0.0, 2.0), (0.5, 1.0), (0.8, 0.4), (0.95, 0.1)):
        belief = initial_belief(lambda_value=lambda_value)
        updated = update_belief(belief, observation)
        values = [value for line in updated.values for value in line]
        assert max(values) == pytest.approx(expected)


def test_v48_full_loop_dry_run_covers_policies_and_environments(tmp_path: Path) -> None:
    result = run_full_loop_validation(
        FullLoopValidationConfig(
            run_id=1,
            output_root=tmp_path / "full_loop_validation",
            outcome_model_root=tmp_path / "outcome_model",
            episodes_per_policy=7,
            max_primitives_per_episode=1,
            dry_run_schedule=True,
        )
    )
    schedule = pd.read_csv(tmp_path / "full_loop_validation" / "001" / "metrics" / "episode_schedule.csv")

    assert result["status"] == "dry_run_schedule"
    assert set(schedule["policy_id"]) == {
        "no_memory_baseline",
        "static_map_baseline",
        "context_only_without_memory",
        "context_plus_memory_lambda_0_5",
        "context_plus_memory_lambda_0_8",
        "context_plus_memory_lambda_0_95",
    }
    assert set(schedule["W_layer"]) == {"W0", "W1", "W2", "W3"}
    assert {"dry_air", "gaussian_single", "gaussian_four", "annular_gp_single", "annular_gp_four", "w3_randomised_single", "w3_randomised_four"}.issubset(
        set(schedule["environment_mode"])
    )


def test_v48_full_loop_smoke_preserves_frozen_identity(tmp_path: Path) -> None:
    result = run_full_loop_validation(
        FullLoopValidationConfig(
            run_id=2,
            output_root=tmp_path / "full_loop_validation",
            outcome_model_root=tmp_path / "outcome_model",
            episodes_per_policy=1,
            max_primitives_per_episode=1,
            seed=48,
        )
    )
    primitive_log = pd.read_csv(tmp_path / "full_loop_validation" / "002" / "metrics" / "primitive_execution_log.csv")

    assert result["status"] == "complete"
    assert not primitive_log.empty
    assert (primitive_log["source_representative_controller_id"] == primitive_log["controller_id"]).all()
    assert (primitive_log["source_representative_K_gain_checksum"] == primitive_log["lqr_gain_checksum"]).all()
    assert set(primitive_log["timing_state_source"]).issubset({"history_backed_fifo", "not_evaluated_blocked_before_rollout"})
