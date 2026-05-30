from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_lqr_w01_dense_chunked import (
    PROJECT_TITLE_VERSION,
    _expected_launch_gate_rows_per_active_primitive,
    _expected_r5_rows_for_transition_entry,
    _r5_transition_training_tables,
    _wilson_lower_bound,
    _wilson_upper_bound,
)
from run_r5_r10_pipeline import _r5_launch_gate_audit_passed
from run_w3_survival import W3SurvivalConfig, run_w3_survival
from prim_cat import ACTIVE_PRIMITIVE_IDS


def test_wilson_bounds_are_conservative() -> None:
    assert _wilson_lower_bound(10, 10) < 1.0
    assert _wilson_lower_bound(0, 10) == 0.0
    assert _wilson_upper_bound(0, 10) > 0.0
    assert _wilson_upper_bound(10, 10) == 1.0


def test_r5_expected_transition_rows_follow_dense_percentage_mix() -> None:
    assert _expected_r5_rows_for_transition_entry("launch_gate") == 160
    assert _expected_r5_rows_for_transition_entry("inflight_stable") == 160
    assert _expected_r5_rows_for_transition_entry("boundary_near") == 40
    assert _expected_r5_rows_for_transition_entry("recoverable_degraded") == 40
    assert _expected_launch_gate_rows_per_active_primitive() == 5120


def test_r5_pipeline_launch_gate_audit_uses_dense_percentage_mix(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True)
    expected = _expected_launch_gate_rows_per_active_primitive()
    pd.DataFrame(
        [
            {
                "start_state_family": "launch_gate",
                "transition_entry_class": "launch_gate",
                "primitive_id": primitive_id,
                "entry_role": "transition_object",
                "total_rows": expected,
                "entry_role_rejection_count": 0,
                "accepted_count": 1,
                "weak_count": 0,
                "continuation_valid_count": 1,
                "terminal_useful_count": 0,
                "hard_failure_count": 0,
                "blocked_count": 0,
                "rejected_count": 0,
                "r5_launch_entry_gate_passed": True,
            }
            for primitive_id in ACTIVE_PRIMITIVE_IDS
        ]
    ).to_csv(metrics / "r5_launch_gate_entry_diagnosis.csv", index=False)

    assert _r5_launch_gate_audit_passed(tmp_path)


def test_r5_transition_training_selects_entry_classes_independently() -> None:
    frame = pd.DataFrame(
        [
            _training_row(
                entry_class="launch_gate",
                exit_class="hard_failure",
                compatible=False,
                environment_mode=environment_mode,
            )
            for environment_mode in ("dry_air", "w1_annular_gp_randomised_single", "w1_annular_gp_randomised_four")
            for _ in range(60)
        ]
        + [
            _training_row(
                entry_class="inflight_stable",
                exit_class="inflight_stable",
                compatible=True,
                environment_mode=environment_mode,
            )
            for environment_mode in ("dry_air", "w1_annular_gp_randomised_single", "w1_annular_gp_randomised_four")
            for _ in range(60)
        ]
    )

    summary, selected, _, blockers = _r5_transition_training_tables(frame)

    launch = summary[summary["transition_entry_class"].astype(str).eq("launch_gate")].iloc[0]
    inflight = summary[summary["transition_entry_class"].astype(str).eq("inflight_stable")].iloc[0]

    assert bool(launch["r5_training_eligible"]) is False
    assert bool(inflight["r5_training_eligible"]) is True
    assert set(selected["transition_entry_class"]) == {"inflight_stable"}
    assert "missing_selected_launch_gate_transition_object_for_glide" in blockers


def test_r7_refuses_r5_root_without_selected_transition_manifest(tmp_path: Path) -> None:
    r5_root = tmp_path / "r5" / "001"
    (r5_root / "manifests").mkdir(parents=True)
    (r5_root / "metrics").mkdir(parents=True)
    (r5_root / "manifests" / "run_manifest.json").write_text(
        json.dumps(
            {
                "project_title_version": PROJECT_TITLE_VERSION,
                "w01_dense_evidence_complete": True,
                "method_evidence_level": "w01_dense_evidence_complete",
                "W2_required_for_move_on": False,
                "R7_W3_direct_source": "r5_transition_selected_for_r7_frozen_controller_bundle",
                "official_W_layers": {"W1": ["w1_annular_gp_randomised_single", "w1_annular_gp_randomised_four"]},
                "primitive_timing_contract": {"primitive_timing_contract_version": "v411_0p10s_5slot_20ms"},
            }
        )
        + "\n",
        encoding="ascii",
    )
    (r5_root / "manifests" / "frozen_w01_controller_bundle.json").write_text('{"records":[]}\n', encoding="ascii")

    result = run_w3_survival(
        W3SurvivalConfig(
            run_id=1,
            input_root=r5_root,
            output_root=tmp_path / "w3",
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
        )
    )
    manifest = json.loads((Path(result["run_root"]) / "manifests" / "w3_survival_manifest.json").read_text(encoding="ascii"))

    assert result["status"] == "blocked"
    assert manifest["blocked_reason"] == "missing_r5_transition_training_manifest"


def _training_row(*, entry_class: str, exit_class: str, compatible: bool, environment_mode: str) -> dict[str, object]:
    return {
        "primitive_id": "glide",
        "primitive_family": "glide",
        "entry_role": "transition_object",
        "primitive_variant_id": "primvar_glide_transition_object_test",
        "controller_id": "lqrta_glide_test",
        "candidate_index": 0,
        "candidate_weight_label": "glide_robust_anchor_nominal_000",
        "transition_entry_class": entry_class,
        "transition_exit_class": exit_class,
        "transition_chain_compatible": compatible,
        "environment_mode": environment_mode,
        "outcome_class": "accepted" if compatible else "failed",
        "boundary_use_class": "continuation_valid" if compatible else "hard_failure",
        "trajectory_integrated_updraft_gain_m": 0.2,
        "lift_dwell_time_s": 0.04,
        "rollout_duration_s": 0.1,
        "saturation_fraction": 0.1,
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
    }
