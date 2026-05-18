from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggressive_reversal_ocp import (
    SEED_FAMILIES,
    AggressiveReversalOcpResult,
    seed_family_inventory_for_target,
)
from run_aggressive_reversal_search import DEFAULT_TARGETS_DEG, run_search


@pytest.fixture(scope="module")
def aggressive_full_ladder_outputs(tmp_path_factory):
    import run_aggressive_reversal_search as runner

    output_root = tmp_path_factory.mktemp("aggressive_full_ladder")
    original_root = runner.DEFAULT_RESULTS_ROOT
    runner.DEFAULT_RESULTS_ROOT = output_root
    try:
        outputs = run_search(
            run_id=1,
            overwrite=False,
            targets_deg=DEFAULT_TARGETS_DEG,
            max_ipopt_iter=1,
            ocp_max_cpu_time_s=0.05,
            ocp_node_count=1,
        )
    finally:
        runner.DEFAULT_RESULTS_ROOT = original_root
    return outputs


def _fake_result(target_heading_deg: float, success: bool = False) -> AggressiveReversalOcpResult:
    time_s = np.array([0.0, 0.02])
    x_ref = np.zeros((2, 15), dtype=float)
    x_ref[:, 6] = 6.5
    u = np.zeros((2, 3), dtype=float)
    failure_label = "success" if success else "under_turning"
    metrics = {
        "target_heading_deg": float(target_heading_deg),
        "actual_heading_change_deg": 0.5 * float(target_heading_deg),
        "heading_error_deg": 0.5 * float(target_heading_deg),
        "forward_travel_m": 0.1,
        "turn_volume_proxy_m2": 0.0,
        "height_change_m": 0.0,
        "max_altitude_gain_m": 0.0,
        "speed_min_m_s": 6.5,
        "terminal_speed_m_s": 6.5,
        "max_alpha_deg": 0.0,
        "max_beta_deg": 0.0,
        "max_bank_deg": 0.0,
        "max_pitch_deg": 0.0,
        "max_rate_rad_s": 0.0,
        "min_true_wall_margin_m": 1.0,
        "min_floor_margin_m": 1.0,
        "min_ceiling_margin_m": 1.0,
        "saturation_fraction": 0.0,
        "saturation_time_s": 0.0,
        "finite_state_success": True,
        "rollout_success": True,
        "source_trajectory_success": True,
        "terminal_recoverable_proxy": False,
        "recoverable": False,
        "success": success,
        "primitive_success": success,
        "failure_label": failure_label,
        "notes": "fake_ladder_result",
        "energy_exploitation": False,
        "families_attempted": ";".join(SEED_FAMILIES),
        "selected_family": "short_perch_yaw_redirect",
        "selected_method": "phase_search",
        "next_family_reason": "under_turning_longer_perch",
        "limiting_mechanism": "insufficient_manoeuvre_seed",
        "best_finite_candidate": "short_perch_yaw_redirect",
        "best_recoverable_candidate": "",
        "best_successful_candidate": "short_perch_yaw_redirect" if success else "",
        "nlp_constructed": True,
        "ipopt_called": True,
        "solver_status": "fake_not_run",
        "solver_iter_count": 0,
        "solver_objective": np.nan,
        "constraint_residual_max": np.nan,
        "replay_defect_max": 0.0,
        "direct_ocp_attempted": True,
        "direct_ocp_converged": False,
        "phase_search_attempted": True,
        "replay_finite": True,
    }
    return AggressiveReversalOcpResult(
        target_heading_deg=float(target_heading_deg),
        direction_sign=1,
        success=success,
        failure_label=failure_label,
        time_s=time_s,
        x_ref=x_ref,
        u_ff_norm=u,
        u_norm_applied=u,
        delta_cmd_rad=u,
        phase=("entry", "exit_glide"),
        metrics=metrics,
        notes="fake_ladder_result",
    )


def test_full_ladder_records_all_targets_and_seed_families(
    aggressive_full_ladder_outputs,
) -> None:
    summary = pd.read_csv(aggressive_full_ladder_outputs["summary_csv"])

    assert list(summary["target_heading_deg"]) == list(DEFAULT_TARGETS_DEG)
    for _, row in summary.iterrows():
        expected_families = seed_family_inventory_for_target(float(row["target_heading_deg"]))
        assert set(row["families_attempted"].split(";")) == set(expected_families)
        assert row["selected_family"] in SEED_FAMILIES
        assert bool(row["phase_search_attempted"]) is True
        assert bool(row["direct_ocp_attempted"]) is True
        assert bool(row["nlp_constructed"]) is True
        assert bool(row["ipopt_called"]) is True
        assert row["limiting_mechanism"] in {
            "none",
            "optimiser_convergence",
            "physical_safety_boundary",
            "model_boundary_behaviour",
            "insufficient_manoeuvre_seed",
            "physical_boundary",
            "energy_budget_limited",
            "high_alpha_drag_limited",
            "recovery_handoff_limited",
            "turn_authority_limited",
            "safety_volume_limited",
            "solver_formulation_limited",
        }


def test_full_ladder_manifest_summarises_largest_targets(
    aggressive_full_ladder_outputs,
) -> None:
    manifest = json.loads(
        aggressive_full_ladder_outputs["manifest_json"].read_text(encoding="ascii")
    )

    assert manifest["targets_completed_deg"] == list(DEFAULT_TARGETS_DEG)
    assert "largest_finite_target_deg" in manifest
    assert "largest_recoverable_target_deg" in manifest
    assert "largest_successful_target_deg" in manifest
    for target in DEFAULT_TARGETS_DEG:
        outcome = manifest["target_outcomes"][str(int(target))]
        assert "nlp_constructed" in outcome
        assert "ipopt_called" in outcome
        assert "solver_status" in outcome
        assert "constraint_residual_max" in outcome
        assert "replay_defect_max" in outcome
        assert "direct_ocp_attempted" in outcome
        assert "direct_ocp_converged" in outcome
        assert "replay_finite" in outcome
        assert "recoverable" in outcome
        assert "primitive_success" in outcome


def test_resume_skips_completed_targets_and_overwrite_regenerates(
    monkeypatch,
    tmp_path,
) -> None:
    import run_aggressive_reversal_search as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "solve_aggressive_reversal_ocp",
        lambda config, aircraft=None: _fake_result(config.target_heading_deg),
    )
    first = run_search(run_id=1, overwrite=True, targets_deg=(15.0, 30.0))
    marker = first["root"] / "logs" / "marker.txt"
    marker.write_text("preserve-until-overwrite", encoding="ascii")

    second = run_search(run_id=1, overwrite=False, targets_deg=(15.0, 30.0, 60.0))
    second_manifest = json.loads(second["manifest_json"].read_text(encoding="ascii"))
    assert second_manifest["targets_skipped_by_resume_deg"] == [15.0, 30.0]
    assert marker.exists()
    assert set(second_manifest["target_outcomes"]) == {"15", "30", "60"}

    third = run_search(run_id=1, overwrite=True, targets_deg=(15.0,))
    third_manifest = json.loads(third["manifest_json"].read_text(encoding="ascii"))
    assert not marker.exists()
    assert set(third_manifest["target_outcomes"]) == {"15"}


def test_late_target_failure_preserves_earlier_evidence(monkeypatch, tmp_path) -> None:
    import run_aggressive_reversal_search as runner

    def fake_solver(config, aircraft=None):
        if config.target_heading_deg == 60.0:
            raise RuntimeError("forced late-target failure")
        return _fake_result(config.target_heading_deg, success=config.target_heading_deg == 15.0)

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(runner, "solve_aggressive_reversal_ocp", fake_solver)
    outputs = run_search(run_id=1, overwrite=True, targets_deg=(15.0, 30.0, 60.0))
    summary = pd.read_csv(outputs["summary_csv"])
    rows = {float(row["target_heading_deg"]): row for _, row in summary.iterrows()}

    assert bool(rows[15.0]["success"]) is True
    assert rows[30.0]["failure_label"] == "under_turning"
    assert rows[60.0]["failure_label"] == "solver_failure"
    assert (
        outputs["root"]
        / "metrics"
        / "aggressive_reversal_target_015_trajectory_s001.csv"
    ).exists()
