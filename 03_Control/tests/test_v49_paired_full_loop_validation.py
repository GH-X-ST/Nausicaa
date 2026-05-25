from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from run_full_loop_validation import FullLoopValidationConfig, run_full_loop_validation
from run_v49_source_audit import V49SourceAuditConfig, run_v49_source_audit
from run_v49_validation_figures import V49ValidationFigureConfig, run_v49_validation_figures


@pytest.fixture(scope="module")
def v49_smoke_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("v49_full_loop")
    result = run_full_loop_validation(
        FullLoopValidationConfig(
            run_id=4,
            output_root=base / "full_loop_validation",
            outcome_model_root=base / "outcome_model",
            episodes_per_policy=3,
            max_primitives_per_episode=1,
            seed=49,
            resume=True,
            repair_incomplete=True,
        )
    )
    assert result["status"] == "complete"
    return base / "full_loop_validation" / "004"


def test_v49_source_audit_accepts_frozen_chain_and_marks_v48_full_loop_infrastructure(tmp_path: Path) -> None:
    result = run_v49_source_audit(V49SourceAuditConfig(output_root=tmp_path / "full_loop_validation" / "003"))
    summary = pd.read_csv(tmp_path / "full_loop_validation" / "003" / "metrics" / "source_audit_summary.csv")

    assert result["status"] == "source_audit_pass"
    infra = summary[summary["stage"].astype(str).str.startswith("full_loop_")]
    assert not infra.empty
    assert set(infra["source_role"]) == {"infrastructure_evidence_not_final_memory_comparison"}


def test_v49_paired_schedule_uses_identical_physical_episode_keys(v49_smoke_root: Path) -> None:
    schedule = pd.read_csv(v49_smoke_root / "metrics" / "episode_schedule.csv")
    expected_policies = {
        "no_memory_baseline",
        "static_map_baseline",
        "context_only_without_memory",
        "context_plus_memory_lambda_0_5",
        "context_plus_memory_lambda_0_8",
        "context_plus_memory_lambda_0_95",
    }

    for _, group in schedule.groupby("paired_episode_index"):
        assert set(group["policy_id"]) == expected_policies
        assert group["common_random_key"].nunique() == 1
        assert group["launch_state_seed"].nunique() == 1
        assert group["environment_seed"].nunique() == 1
        assert group["W_layer"].nunique() == 1
        assert group["environment_mode"].nunique() == 1


def test_v49_belief_persists_for_memory_policies_only(v49_smoke_root: Path) -> None:
    summary = pd.read_csv(v49_smoke_root / "metrics" / "belief_evolution_summary.csv")
    by_policy = summary.set_index("policy_id")

    assert int(by_policy.loc["context_plus_memory_lambda_0_5", "final_belief_update_count"]) >= 3
    assert int(by_policy.loc["context_plus_memory_lambda_0_8", "final_belief_update_count"]) >= 3
    assert int(by_policy.loc["context_plus_memory_lambda_0_95", "final_belief_update_count"]) >= 3
    assert int(by_policy.loc["no_memory_baseline", "final_belief_update_count"]) == 0
    assert int(by_policy.loc["context_only_without_memory", "final_belief_update_count"]) == 0
    assert int(by_policy.loc["static_map_baseline", "final_belief_update_count"]) == 0
    assert by_policy.loc["context_plus_memory_lambda_0_8", "belief_persistence_status"] == "persistent_updates_observed"


def test_v49_static_map_prior_is_nonzero_or_explicitly_blocked(v49_smoke_root: Path) -> None:
    prior_path = v49_smoke_root / "manifests" / "static_map_prior.json"
    blocked_path = v49_smoke_root / "manifests" / "blocked_static_map_note.json"

    assert prior_path.is_file() or blocked_path.is_file()
    if prior_path.is_file():
        payload = json.loads(prior_path.read_text(encoding="ascii"))
        values = [abs(float(value)) for row in payload["values"] for value in row]
        assert payload["status"] == "ready"
        assert max(values) > 0.0
        assert payload["nonzero"] is True
    else:
        payload = json.loads(blocked_path.read_text(encoding="ascii"))
        assert payload["status"] == "blocked_static_map_unavailable_no_prior"
        assert payload["blocked_reason"] == "static_map_prior_nonfinite_or_zero"


def test_v49_paired_comparison_has_matched_keys_and_finite_differences(v49_smoke_root: Path) -> None:
    comparison = pd.read_csv(v49_smoke_root / "metrics" / "paired_policy_comparison.csv")

    assert not comparison.empty
    assert {
        "context_plus_memory_lambda_0_5_vs_context_only_without_memory",
        "context_plus_memory_lambda_0_8_vs_context_only_without_memory",
        "context_plus_memory_lambda_0_95_vs_context_only_without_memory",
        "context_plus_memory_lambda_0_8_vs_no_memory_baseline",
    }.issubset(set(comparison["comparison_id"]))
    diff_columns = [column for column in comparison.columns if column.endswith("_paired_difference")]
    assert diff_columns
    assert pd.to_numeric(comparison[diff_columns].stack(), errors="coerce").notna().all()


def test_v49_figures_and_chapter_tables_are_generated(v49_smoke_root: Path, tmp_path: Path) -> None:
    result = run_v49_validation_figures(
        V49ValidationFigureConfig(
            input_root=v49_smoke_root,
            output_root=tmp_path / "figures" / "v49_validation",
            run_id=1,
        )
    )
    figure_root = tmp_path / "figures" / "v49_validation" / "001"
    expected_figures = {
        "policy_terminal_hard_failure_bar.png",
        "memory_lambda_comparison.png",
        "belief_evolution_example.png",
        "prediction_alignment_summary.png",
        "governor_rejection_summary.png",
        "termination_summary.png",
    }
    expected_tables = {
        "chapter7_w01_w2_w3_evidence_table.csv",
        "chapter7_post_w3_compact_library_table.csv",
        "chapter7_full_loop_policy_table.csv",
        "chapter7_claim_boundary_table.csv",
    }
    audit = pd.read_csv(figure_root / "metrics" / "file_size_audit.csv")

    assert result["status"] == "complete"
    assert expected_figures == {path.name for path in (figure_root / "figures").glob("*.png")}
    assert expected_tables == {path.name for path in (figure_root / "tables").glob("*.csv")}
    assert not audit["above_100mb"].astype(bool).any()
