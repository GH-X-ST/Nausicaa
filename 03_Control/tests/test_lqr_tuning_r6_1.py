from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lqr_tuning import (
    R61_CANDIDATES_PER_PRIMITIVE,
    R61SelectionThresholds,
    r6_1_candidate_family_counts,
    r6_1_candidate_summary,
    r6_1_candidate_weight_specs,
    r6_1_stage_e_decision,
)
from prim_cat import active_primitive_catalogue
from run_lqr_tuning_sweep import LQRTuningSweepConfig, _r6_1_r7_status, _write_r6_1_report, run_lqr_tuning_sweep
from state_sampling import (
    archive_state_sample_for_r6_1_pair,
    r6_1_stage_c_start_state_family_for_pair,
)


def test_r6_1_candidate_generation_has_required_family_split() -> None:
    primitive_id = active_primitive_catalogue()[0].primitive_id
    candidates = r6_1_candidate_weight_specs(primitive_id=primitive_id)

    assert len(candidates) == R61_CANDIDATES_PER_PRIMITIVE
    assert r6_1_candidate_family_counts(candidates) == {
        "nominal": 1,
        "local": 7,
        "conservative_high_r": 8,
        "aggressive_low_r": 8,
        "diagnostic_exploit": 8,
    }
    assert len({candidate.weight_label for candidate in candidates}) == R61_CANDIDATES_PER_PRIMITIVE


def test_r6_1_stage_c_strata_are_exact_and_pair_keys_are_common() -> None:
    families = [r6_1_stage_c_start_state_family_for_pair(index) for index in range(12)]
    assert families == (
        ["launch_gate"] * 4
        + ["inflight_nominal"] * 3
        + ["inflight_lift_region"] * 2
        + ["inflight_boundary_near"] * 2
        + ["inflight_recovery_edge"]
    )
    sample_a = archive_state_sample_for_r6_1_pair(
        7,
        seed=111,
        stage_label="stage_c",
        start_state_family=families[7],
    )
    sample_b = archive_state_sample_for_r6_1_pair(
        7,
        seed=111,
        stage_label="stage_c",
        start_state_family=families[7],
    )
    assert sample_a.paired_start_key == sample_b.paired_start_key
    assert sample_a.state_sampling_version == "r6_1_stratified_common_random_v1"


def _synthetic_summary_rows(*, primitive_id: str, candidate_index: int, hard_failure_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(20):
        family = "launch_gate" if index < 10 else "inflight_nominal"
        failed = index < int(hard_failure_count)
        rows.append(
            {
                "primitive_id": primitive_id,
                "candidate_index": candidate_index,
                "candidate_weight_label": f"{primitive_id}_r61_nominal_{candidate_index:03d}",
                "start_state_family": family,
                "outcome_class": "failed" if failed else "accepted",
                "boundary_use_class": "hard_failure" if failed else "continuation_valid",
                "continuation_valid": not failed,
                "episode_terminal_useful": False,
                "hard_gate_status": "blocked" if failed else "passed",
                "hard_gate_reason": "z_boundary_exit" if failed else "",
                "termination_cause": "z_boundary_exit" if failed else "",
                "lqr_synthesis_status": "solved",
                "sampled_data_check": "sampled_stable",
                "controller_executable": True,
                "lqr_Q_weights_json": json.dumps({"q_attitude": 4.0, "q_velocity": 2.0, "q_rates": 1.6, "q_surfaces": 0.15}),
                "lqr_R_weights_json": json.dumps({"r_aileron": 1.0, "r_elevator": 0.9, "r_rudder": 1.1}),
                "lqr_gain_checksum": "abc123",
                "energy_residual_m": 1.0,
                "lift_dwell_time_s": 1.0,
                "minimum_wall_margin_m": 2.0,
                "saturation_fraction": 0.05,
                "uncertainty_m_s": 0.1,
            }
        )
    return rows


def test_r6_1_selection_allows_bounded_stress_failures_but_blocks_systemic() -> None:
    primitive_id = "glide"
    rows = _synthetic_summary_rows(primitive_id=primitive_id, candidate_index=0, hard_failure_count=4)
    rows.extend(_synthetic_summary_rows(primitive_id=primitive_id, candidate_index=1, hard_failure_count=0))
    for row in rows[-20:]:
        row["lqr_synthesis_status"] = "blocked_unstable_linearisation"
    summary = r6_1_candidate_summary(pd.DataFrame(rows), thresholds=R61SelectionThresholds())

    accepted = summary.loc[summary["candidate_index"] == 0].iloc[0]
    systemic = summary.loc[summary["candidate_index"] == 1].iloc[0]
    assert accepted["eligibility_status"] == "eligible"
    assert systemic["eligibility_status"] == "blocked"
    assert systemic["selection_reason"] == "unsolved_lqr"


def test_r6_1_stage_e_is_conditional_and_reports_no_help_reason() -> None:
    finalist_summary = pd.DataFrame(
        [
            {
                "primitive_id": "glide",
                "candidate_index": 0,
                "robust_score": 1.00,
                "eligibility_status": "eligible",
                "selection_reason": "eligible_preferred_thresholds",
            },
            {
                "primitive_id": "glide",
                "candidate_index": 1,
                "robust_score": 0.99,
                "eligibility_status": "eligible",
                "selection_reason": "eligible_preferred_thresholds",
            },
            {
                "primitive_id": "recovery",
                "candidate_index": 0,
                "robust_score": -1.0,
                "eligibility_status": "blocked",
                "selection_reason": "unsolved_lqr",
            },
        ]
    )
    decisions = r6_1_stage_e_decision(finalist_summary, thresholds=R61SelectionThresholds(score_tie_margin=0.05))

    assert decisions["glide"]["stage_e_required"] is True
    assert decisions["glide"]["candidate_indices"] == [0, 1]
    assert decisions["recovery"]["stage_e_required"] is False
    assert decisions["recovery"]["stage_e_reason"] == "stage_e_not_applicable_all_systemic_blocked"


def test_r6_1_r7_status_blocks_when_any_primitive_lacks_controller() -> None:
    registry_rows = [
        {"primitive_id": primitive.primitive_id, "selected_controller_status": "selected"}
        for primitive in active_primitive_catalogue()[:-1]
    ]
    status = _r6_1_r7_status(
        registry_rows,
        registry_status="complete",
        registry_reason="eligible_registry_backed_complete",
    )
    assert status.startswith("R7 blocked: missing_selected_or_accepted_fallback_controller:")


def test_r6_1_report_ends_with_blocked_reason_when_controller_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "tune_111_r6_1"
    (run_root / "reports").mkdir(parents=True)
    r7_status = "R7 blocked: missing_selected_or_accepted_fallback_controller:glide"
    _write_r6_1_report(
        run_root=run_root,
        config=LQRTuningSweepConfig(run_id=111, output_root=tmp_path, strategy="r6_1_staged"),
        manifest={"strategy": "r6_1_staged", "registry_status": "blocked", "registry_claim_status": "simulation_only_blocked"},
        candidate_rows=[],
        registry_rows=[],
        stage_e_decisions={
            "glide": {
                "primitive_id": "glide",
                "stage_e_required": False,
                "stage_e_reason": "stage_e_not_applicable_no_candidate_above_minimum_gate",
            }
        },
        r7_status=r7_status,
    )

    text = (run_root / "reports" / "r6_1_smart_lqr_tuning_report.md").read_text(encoding="ascii")
    assert text.rstrip().endswith(r7_status)


def test_r6_1_dry_run_writes_compact_schedule_outputs(tmp_path: Path) -> None:
    result = run_lqr_tuning_sweep(
        LQRTuningSweepConfig(
            run_id=111,
            output_root=tmp_path,
            strategy="r6_1_staged",
            dry_run_schedule=True,
            candidate_chunk_size=1024,
            workers=1,
            max_workers=1,
        )
    )
    run_root = Path(result["run_root"])

    assert (run_root / "manifests" / "run_manifest.json").is_file()
    assert (run_root / "metrics" / "runtime_summary.csv").is_file()
    assert (run_root / "metrics" / "chunk_summary.csv").is_file()
    assert (run_root / "metrics" / "r6_selection_thresholds.csv").is_file()
    assert list((run_root / "tables").glob("*")) == []
    manifest = json.loads((run_root / "manifests" / "run_manifest.json").read_text(encoding="ascii"))
    assert manifest["strategy"] == "r6_1_staged"
    assert manifest["stage_c_planned_rows"] == 6144
