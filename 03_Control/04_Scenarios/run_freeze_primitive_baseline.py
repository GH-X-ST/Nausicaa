from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
CAMPAIGN = "09_primitive_library"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN
FREEZE_DIR_NAME = "000_frozen_baseline"
HASHED_SUFFIXES = (".csv", ".json", ".md")
REQUIRED_BASELINE_RUNS = (2, 3, 4, 5, 6)
FUTURE_TARGETS_DEG = (45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
FUTURE_DIRECTIONS = (-1, 1)
FUTURE_STARTS = ("lift_sector", "random_stress")
AGGRESSIVE_FAMILIES = (
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
FUTURE_UPDRAFT_CONFIGS = ("U1_single_fan", "U4_four_fan")
FUTURE_WIND_FIDELITIES = ("W1", "W2")
REQUIRED_EXTERNAL_TESTS = (
    "03_Control/tests/test_primitive_library_runner.py",
    "03_Control/tests/test_primitive_library_selection.py",
    "03_Control/tests/test_primitive_library_w3_stress.py",
    "03_Control/tests/test_primitive_library_governor.py",
    "03_Control/tests/test_primitive_library_outer_loop.py",
    "03_Control/tests/test_primitive_baseline_freeze.py",
)
NON_BLOCKING_SHORTLIST_STATUSES = (
    "selected_for_w3_stress",
    "selected_for_governor_seed",
)
BLOCKING_SHORTLIST_STATUSES = (
    "needs_seed_refinement",
    "governor_reject_entry_envelope",
    "boundary_only",
    "deferred_not_relevant",
)
PHASE_B_BLOCKED_REASON = (
    "Stage 0.1 closes validation and blocker-table correctness only; "
    "dense archive work must start in a separate task."
)
FORBIDDEN_SCOPE_THIS_PASS = (
    "target expansion",
    "both directions",
    "new start-state strata",
    "dense archive-count manifest",
    "W0 dense sweep",
    "W1/W2 replay",
    "envelope maps",
    "clustering",
    "mission objectives",
    "real-flight transfer",
)
BLOCKER_COLUMNS = (
    "blocker_id",
    "source_run_id",
    "source_file",
    "blocker_scope",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_condition",
    "updraft_config",
    "wind_fidelity",
    "objective",
    "evidence_status",
    "failure_label",
    "active_limiting_mechanism",
    "claim_implication",
    "recommended_next_stage",
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data Containers and Path Helpers
# 2) Source Hashing
# 3) Blocker and Plot-Ready Table Builders
# 4) External Validation
# 5) Report and Manifest Writers
# 6) Freeze Workflow
# 7) CLI Entry Point
# =============================================================================


# =============================================================================
# 1) Data Containers and Path Helpers
# =============================================================================
@dataclass(frozen=True)
class FreezeOutputs:
    root: Path
    manifest_json: Path
    blocker_csv: Path
    plot_ready_summary_csv: Path
    plot_ready_blocker_counts_csv: Path
    claim_boundary_md: Path
    baseline_summary_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "blocker_csv": self.blocker_csv,
            "plot_ready_summary_csv": self.plot_ready_summary_csv,
            "plot_ready_blocker_counts_csv": self.plot_ready_blocker_counts_csv,
            "claim_boundary_md": self.claim_boundary_md,
            "baseline_summary_md": self.baseline_summary_md,
        }


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _prepare_output_tree(result_root: Path, run_id: int, overwrite: bool) -> FreezeOutputs:
    if int(run_id) != 0:
        raise ValueError("Stage 0 freeze outputs must use run_id=0.")

    root = Path(result_root) / FREEZE_DIR_NAME
    if root.exists() and not overwrite:
        raise ValueError(f"freeze output tree already exists: {root}")
    if root.exists() and overwrite:
        _clear_generated_files(root)

    manifest_dir = root / "manifests"
    metrics_dir = root / "metrics"
    reports_dir = root / "reports"
    for path in (root, manifest_dir, metrics_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    suffix = f"s{int(run_id):03d}"
    return FreezeOutputs(
        root=root,
        manifest_json=manifest_dir / f"frozen_baseline_manifest_{suffix}.json",
        blocker_csv=metrics_dir / f"baseline_blockers_{suffix}.csv",
        plot_ready_summary_csv=metrics_dir / f"baseline_plot_ready_summary_{suffix}.csv",
        plot_ready_blocker_counts_csv=metrics_dir / f"baseline_plot_ready_blocker_counts_{suffix}.csv",
        claim_boundary_md=reports_dir / f"claim_boundary_{suffix}.md",
        baseline_summary_md=reports_dir / f"baseline_summary_{suffix}.md",
    )


def _clear_generated_files(root: Path) -> None:
    # The freeze pass may only clear its own generated output tree.
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _string_or_none(value: object) -> str:
    if pd.isna(value):
        return "none"
    text = str(value)
    return text if text else "none"


def _float_or_blank(value: object) -> object:
    if pd.isna(value) or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# =============================================================================
# 2) Source Hashing
# =============================================================================
def _hash_baseline_sources(
    result_root: Path,
    baseline_runs: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    hash_rows: list[dict[str, object]] = []

    for run_id in baseline_runs:
        run_dir = Path(result_root) / f"{int(run_id):03d}"
        exists = run_dir.exists()
        files = []
        if exists:
            files = [
                path
                for path in sorted(run_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in HASHED_SUFFIXES
            ]
        run_rows.append(
            {
                "run_id": f"{int(run_id):03d}",
                "exists": bool(exists),
                "hashable_file_count": int(len(files)),
                "path": _repo_relative(run_dir),
            }
        )
        for path in files:
            # Source hashes are deliberately limited to required baseline runs.
            hash_rows.append(
                {
                    "source_run_id": f"{int(run_id):03d}",
                    "path": _repo_relative(path),
                    "suffix": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
    return run_rows, hash_rows


def _hash_map(hash_rows: list[dict[str, object]]) -> dict[str, str]:
    return {str(row["path"]): str(row["sha256"]) for row in hash_rows}


# =============================================================================
# 3) Blocker and Plot-Ready Table Builders
# =============================================================================
def build_baseline_blocker_table(result_root: Path) -> pd.DataFrame:
    """Return baseline blockers and explicit not-evaluated future coverage gaps."""

    result_root = Path(result_root)
    rows: list[dict[str, object]] = []
    rows.extend(_evidence_blockers_from_run_002(result_root))
    rows.extend(_shortlist_blockers_from_run_003(result_root))
    rows.extend(_w3_blockers_from_run_004(result_root))
    rows.extend(_governor_blockers_from_run_005(result_root))
    rows.extend(_outer_loop_blockers_from_run_006(result_root))
    rows.extend(_future_not_evaluated_blockers())

    for index, row in enumerate(rows, start=1):
        row.setdefault("blocker_id", f"blocker_{index:05d}")
        for column in BLOCKER_COLUMNS:
            row.setdefault(column, "none")
    return pd.DataFrame(rows, columns=BLOCKER_COLUMNS)


def _evidence_blockers_from_run_002(result_root: Path) -> list[dict[str, object]]:
    source = result_root / "002" / "metrics" / "primitive_evidence_library_s002.csv"
    evidence = _read_csv_if_present(source)
    if evidence.empty:
        return []

    failure_mask = (
        (evidence.get("failure_label", "success") != "success")
        | (~evidence.get("heading_band_pass", True).astype(bool))
        | (~evidence.get("true_safe_trajectory", True).astype(bool))
        | evidence.get("candidate_class", "").isin(("boundary_evidence", "not_evaluated"))
    )
    blockers = evidence[failure_mask].copy()
    rows = []
    for index, row in blockers.iterrows():
        rows.append(
            {
                "blocker_id": f"evidence_002_{int(index):05d}",
                "source_run_id": "002",
                "source_file": _repo_relative(source),
                "blocker_scope": "primitive_evidence",
                "family": _string_or_none(row.get("family")),
                "target_heading_deg": _float_or_blank(row.get("target_heading_deg")),
                "direction_sign": int(row.get("direction_sign", 1)),
                "start_condition": _string_or_none(row.get("start_condition")),
                "updraft_config": _string_or_none(row.get("updraft_config")),
                "wind_fidelity": _string_or_none(row.get("wind_fidelity")),
                "objective": "primitive_archive_baseline",
                "evidence_status": _string_or_none(row.get("evaluation_status")),
                "failure_label": _string_or_none(row.get("failure_label")),
                "active_limiting_mechanism": _string_or_none(row.get("active_limiting_mechanism")),
                "claim_implication": "limits baseline primitive envelope claim",
                "recommended_next_stage": "stage2_expand_archive",
            }
        )
    return rows


def _shortlist_blockers_from_run_003(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = result_root / "003" / "metrics" / "higher_target_growth_request_s003.csv"
    requests = _read_csv_if_present(source)
    if not requests.empty:
        for index, row in requests.iterrows():
            rows.append(
                {
                    "blocker_id": f"higher_target_003_{int(index):05d}",
                    "source_run_id": "003",
                    "source_file": _repo_relative(source),
                    "blocker_scope": "higher_target_request",
                    "family": _string_or_none(row.get("recommended_families")),
                    "target_heading_deg": _float_or_blank(row.get("requested_target_deg")),
                    "direction_sign": "not_evaluated",
                    "start_condition": _string_or_none(row.get("recommended_start_conditions")),
                    "updraft_config": _string_or_none(row.get("recommended_updraft_configs")),
                    "wind_fidelity": _string_or_none(row.get("recommended_wind_fidelities")),
                    "objective": "widening_versus_growth_question",
                    "evidence_status": _string_or_none(row.get("request_status")),
                    "failure_label": _string_or_none(row.get("current_failure_mode")),
                    "active_limiting_mechanism": _string_or_none(row.get("reason")),
                    "claim_implication": "baseline cannot support full target ladder or growth decision",
                    "recommended_next_stage": "stage2_expand_archive",
                }
            )

    source = result_root / "003" / "metrics" / "candidate_shortlist_s003.csv"
    shortlist = _read_csv_if_present(source)
    if not shortlist.empty:
        # Run-003 selected rows are baseline evidence, not blockers. Only rows
        # held back from W3/governor progression are emitted here.
        status = shortlist.get("selection_status", "")
        mask = status.isin(BLOCKING_SHORTLIST_STATUSES)
        for index, row in shortlist[mask].iterrows():
            rows.append(
                {
                    "blocker_id": f"shortlist_003_{int(index):05d}",
                    "source_run_id": "003",
                    "source_file": _repo_relative(source),
                    "blocker_scope": "shortlist_selection",
                    "family": _string_or_none(row.get("family")),
                    "target_heading_deg": _float_or_blank(row.get("target_heading_deg")),
                    "direction_sign": int(row.get("direction_sign", 1)),
                    "start_condition": _string_or_none(row.get("start_condition")),
                    "updraft_config": _string_or_none(row.get("updraft_config")),
                    "wind_fidelity": _string_or_none(row.get("wind_fidelity")),
                    "objective": "selected_w3_planning",
                    "evidence_status": _string_or_none(row.get("selection_status")),
                    "failure_label": _string_or_none(row.get("failure_label")),
                    "active_limiting_mechanism": _string_or_none(row.get("active_limiting_mechanism")),
                    "claim_implication": "baseline shortlist is narrow and planning-only for excluded rows",
                    "recommended_next_stage": "stage3_archive_selection",
                }
            )
    return rows


def _w3_blockers_from_run_004(result_root: Path) -> list[dict[str, object]]:
    source = result_root / "004" / "metrics" / "w3_stress_candidate_summary_s004.csv"
    summary = _read_csv_if_present(source)
    if summary.empty:
        return []

    mask = summary.get("candidate_w3_status", "") != "w3_supported"
    rows = []
    for index, row in summary[mask].iterrows():
        rows.append(
            {
                "blocker_id": f"w3_004_{int(index):05d}",
                "source_run_id": "004",
                "source_file": _repo_relative(source),
                "blocker_scope": "selected_w3_stress",
                "family": _string_or_none(row.get("family")),
                "target_heading_deg": _float_or_blank(row.get("target_heading_deg")),
                "direction_sign": "not_logged",
                "start_condition": _string_or_none(row.get("start_condition")),
                "updraft_config": _string_or_none(row.get("updraft_config")),
                "wind_fidelity": _string_or_none(row.get("wind_fidelity")),
                "objective": "robustness_baseline",
                "evidence_status": _string_or_none(row.get("candidate_w3_status")),
                "failure_label": _string_or_none(row.get("dominant_failure_label")),
                "active_limiting_mechanism": _string_or_none(row.get("dominant_limiting_mechanism")),
                "claim_implication": "selected W3 evidence is not a full robustness funnel",
                "recommended_next_stage": "stage5_robustness_funnel",
            }
        )
    return rows


def _governor_blockers_from_run_005(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = result_root / "005" / "metrics" / "governor_accept_reject_decisions_s005.csv"
    decisions = _read_csv_if_present(source)
    if not decisions.empty:
        status_column = "governor_decision_status"
        if status_column in decisions.columns:
            mask = decisions[status_column] != "accepted_governor_seed"
        else:
            mask = pd.Series([True] * len(decisions), index=decisions.index)
        for index, row in decisions[mask].iterrows():
            rows.append(
                {
                    "blocker_id": f"governor_005_{int(index):05d}",
                    "source_run_id": "005",
                    "source_file": _repo_relative(source),
                    "blocker_scope": "governor_seed",
                    "family": _string_or_none(row.get("family")),
                    "target_heading_deg": _float_or_blank(row.get("target_heading_deg")),
                    "direction_sign": _string_or_none(row.get("direction_sign")),
                    "start_condition": _string_or_none(row.get("start_condition")),
                    "updraft_config": _string_or_none(row.get("updraft_config")),
                    "wind_fidelity": _string_or_none(row.get("wind_fidelity")),
                    "objective": "governor_seed_baseline",
                    "evidence_status": _string_or_none(row.get(status_column)),
                    "failure_label": _string_or_none(row.get("governor_rejection_reason")),
                    "active_limiting_mechanism": _string_or_none(row.get("active_limiting_mechanism")),
                    "claim_implication": "baseline governor is offline seed evidence only",
                    "recommended_next_stage": "stage6_generalise_governor",
                }
            )

    source = result_root / "005" / "metrics" / "governor_rejection_summary_s005.csv"
    rejection_summary = _read_csv_if_present(source)
    if not rejection_summary.empty:
        for index, row in rejection_summary.iterrows():
            if int(row.get("rejected_count", 0)) <= 0:
                continue
            rows.append(
                {
                    "blocker_id": f"governor_summary_005_{int(index):05d}",
                    "source_run_id": "005",
                    "source_file": _repo_relative(source),
                    "blocker_scope": "governor_rejection_summary",
                    "family": "mixed",
                    "target_heading_deg": "",
                    "direction_sign": "mixed",
                    "start_condition": "mixed",
                    "updraft_config": "mixed",
                    "wind_fidelity": "mixed",
                    "objective": "governor_seed_baseline",
                    "evidence_status": _string_or_none(row.get("governor_decision_status")),
                    "failure_label": _string_or_none(row.get("governor_decision_status")),
                    "active_limiting_mechanism": _string_or_none(row.get("governor_decision_status")),
                    "claim_implication": "baseline governor has explicit rejection cases",
                    "recommended_next_stage": "stage6_generalise_governor",
                }
            )
    return rows


def _outer_loop_blockers_from_run_006(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = result_root / "006" / "metrics" / "outer_loop_mission_summary_s006.csv"
    summary = _read_csv_if_present(source)
    if not summary.empty:
        for index, row in summary.iterrows():
            if bool(row.get("sustained_outer_loop_mission_success", False)):
                continue
            rows.append(
                {
                    "blocker_id": f"outer_loop_006_{int(index):05d}",
                    "source_run_id": "006",
                    "source_file": _repo_relative(source),
                    "blocker_scope": "outer_loop_mission",
                    "family": "governor_selected_seed",
                    "target_heading_deg": "",
                    "direction_sign": "not_logged",
                    "start_condition": _string_or_none(row.get("scenario_id")),
                    "updraft_config": _outer_loop_updraft_from_scenario(row.get("scenario_id")),
                    "wind_fidelity": "governor_seed",
                    "objective": "objective_one_sustained_operation",
                    "evidence_status": _string_or_none(row.get("mission_success_label")),
                    "failure_label": _string_or_none(row.get("mission_stop_reason")),
                    "active_limiting_mechanism": _outer_loop_mechanism(row),
                    "claim_implication": "run_006 is short governed transit and rejection evidence only",
                    "recommended_next_stage": "stage7_objective_one_campaign",
                }
            )

    source = result_root / "006" / "metrics" / "outer_loop_coverage_gap_summary_s006.csv"
    gaps = _read_csv_if_present(source)
    if not gaps.empty:
        for index, row in gaps.iterrows():
            rows.append(
                {
                    "blocker_id": f"coverage_gap_006_{int(index):05d}",
                    "source_run_id": "006",
                    "source_file": _repo_relative(source),
                    "blocker_scope": "outer_loop_coverage_gap",
                    "family": "governor_selected_seed",
                    "target_heading_deg": "",
                    "direction_sign": "not_logged",
                    "start_condition": _string_or_none(row.get("scenario_id")),
                    "updraft_config": _outer_loop_updraft_from_scenario(row.get("scenario_id")),
                    "wind_fidelity": "governor_seed",
                    "objective": "objective_one_sustained_operation",
                    "evidence_status": _string_or_none(row.get("coverage_gap_type")),
                    "failure_label": _string_or_none(row.get("gap_reason")),
                    "active_limiting_mechanism": _gap_mechanism(row),
                    "claim_implication": "baseline cannot claim sustained operation or volume coverage",
                    "recommended_next_stage": "stage7_objective_one_campaign",
                }
            )
    return rows


def _future_not_evaluated_blockers() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in AGGRESSIVE_FAMILIES:
        for target in FUTURE_TARGETS_DEG:
            for direction_sign in FUTURE_DIRECTIONS:
                for start_condition in FUTURE_STARTS:
                    for updraft_config in FUTURE_UPDRAFT_CONFIGS:
                        for wind_fidelity in FUTURE_WIND_FIDELITIES:
                            rows.append(
                                {
                                    "blocker_id": (
                                        "future_target_"
                                        f"{family}_{int(target):03d}_d{direction_sign:+d}_"
                                        f"{start_condition}_{updraft_config}_{wind_fidelity}"
                                    ).replace("+", "p").replace("-", "m"),
                                    "source_run_id": "not_evaluated",
                                    "source_file": "project_plan_required_future_gap",
                                    "blocker_scope": "future_archive_gap",
                                    "family": family,
                                    "target_heading_deg": float(target),
                                    "direction_sign": int(direction_sign),
                                    "start_condition": start_condition,
                                    "updraft_config": updraft_config,
                                    "wind_fidelity": wind_fidelity,
                                    "objective": "widening_versus_growth_question",
                                    "evidence_status": "not_evaluated",
                                    "failure_label": "not_evaluated_full_target_ladder_gap",
                                    "active_limiting_mechanism": "missing_archive_evidence",
                                    "claim_implication": (
                                        "frozen baseline cannot answer widening-versus-growth "
                                        "for higher aggressive target coverage"
                                    ),
                                    "recommended_next_stage": "stage2_expand_archive",
                                }
                            )

    rows.extend(
        [
            {
                "blocker_id": "future_objective_one_sustained_operation",
                "source_run_id": "not_evaluated",
                "source_file": "project_plan_required_future_gap",
                "blocker_scope": "future_mission_gap",
                "family": "mission_policy",
                "target_heading_deg": "",
                "direction_sign": "mixed",
                "start_condition": "held_out_randomised_starts",
                "updraft_config": "measured_or_fitted_updraft",
                "wind_fidelity": "W1_W2_W3_W4",
                "objective": "objective_one_sustained_operation",
                "evidence_status": "not_evaluated",
                "failure_label": "not_evaluated_sustained_operation",
                "active_limiting_mechanism": "mission_campaign_absent",
                "claim_implication": "frozen baseline cannot claim prolonged updraft exploitation",
                "recommended_next_stage": "stage7_objective_one_campaign",
            },
            {
                "blocker_id": "future_objective_two_volume_coverage",
                "source_run_id": "not_evaluated",
                "source_file": "project_plan_required_future_gap",
                "blocker_scope": "future_mission_gap",
                "family": "mission_policy",
                "target_heading_deg": "",
                "direction_sign": "mixed",
                "start_condition": "held_out_randomised_starts",
                "updraft_config": "measured_or_fitted_updraft",
                "wind_fidelity": "W1_W2_W3_W4",
                "objective": "objective_two_volume_coverage",
                "evidence_status": "not_evaluated",
                "failure_label": "not_evaluated_volume_coverage",
                "active_limiting_mechanism": "mission_campaign_absent",
                "claim_implication": "frozen baseline cannot claim 3D volume coverage",
                "recommended_next_stage": "stage8_objective_two_campaign",
            },
            {
                "blocker_id": "future_real_flight_transfer",
                "source_run_id": "not_evaluated",
                "source_file": "project_plan_required_future_gap",
                "blocker_scope": "future_transfer_gap",
                "family": "sim_real_gap_analysis",
                "target_heading_deg": "",
                "direction_sign": "mixed",
                "start_condition": "real_flight_logs",
                "updraft_config": "real_arena_updraft",
                "wind_fidelity": "W5",
                "objective": "real_flight_transfer",
                "evidence_status": "not_evaluated",
                "failure_label": "not_evaluated_real_flight_transfer",
                "active_limiting_mechanism": "no_paired_real_logs_in_frozen_baseline",
                "claim_implication": "frozen baseline cannot claim sim-to-real transfer",
                "recommended_next_stage": "stage9_sim_real_gap_analysis",
            },
        ]
    )
    return rows


def _outer_loop_updraft_from_scenario(scenario_id: object) -> str:
    text = str(scenario_id)
    if text.startswith("U1_"):
        return "U1_single_fan"
    if text.startswith("U4_"):
        return "U4_four_fan"
    return "not_applicable"


def _outer_loop_mechanism(row: pd.Series) -> str:
    if bool(row.get("clearance_limited_after_first_step", False)):
        return "clearance_limited_after_first_step"
    if not bool(row.get("energy_gain_demonstrated", False)):
        return "energy_gain_not_demonstrated"
    if int(row.get("steps_accepted", 0)) == 0:
        return "no_go_rejection"
    return "sustained_operation_not_demonstrated"


def _gap_mechanism(row: pd.Series) -> str:
    flags = []
    for column in (
        "blocked_by_target_steering_unavailable",
        "blocked_by_clearance",
        "blocked_by_lift_confidence",
        "blocked_by_wind_fidelity",
        "blocked_by_recovery",
    ):
        if bool(row.get(column, False)):
            flags.append(column)
    return ",".join(flags) if flags else _string_or_none(row.get("coverage_gap_type"))


def build_plot_ready_summary(
    result_root: Path,
    run_rows: list[dict[str, object]],
    hash_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Return compact count tables that can be plotted without re-reading sources."""

    rows: list[dict[str, object]] = []
    for run in run_rows:
        rows.append(
            {
                "plot_group": "baseline_file_inventory",
                "source_run_id": run["run_id"],
                "metric_name": "hashable_file_count",
                "metric_label": "Hashable source files",
                "metric_value": int(run["hashable_file_count"]),
                "source_file": run["path"],
            }
        )

    hashes_by_run = pd.DataFrame(hash_rows)
    if not hashes_by_run.empty:
        for source_run_id, group in hashes_by_run.groupby("source_run_id"):
            rows.append(
                {
                    "plot_group": "baseline_file_inventory",
                    "source_run_id": source_run_id,
                    "metric_name": "hashed_bytes",
                    "metric_label": "Hashed source bytes",
                    "metric_value": int(group["size_bytes"].sum()),
                    "source_file": "source_hash_manifest",
                }
            )

    _append_value_counts(
        rows,
        result_root / "002" / "metrics" / "primitive_evidence_library_s002.csv",
        "002",
        "candidate_class",
        "primitive_candidate_class_count",
        "Candidate class count",
    )
    _append_value_counts(
        rows,
        result_root / "004" / "metrics" / "w3_stress_candidate_summary_s004.csv",
        "004",
        "candidate_w3_status",
        "w3_candidate_status_count",
        "W3 candidate status count",
    )
    _append_value_counts(
        rows,
        result_root / "005" / "metrics" / "governor_rejection_summary_s005.csv",
        "005",
        "governor_decision_status",
        "governor_decision_status_count",
        "Governor decision status count",
    )
    _append_value_counts(
        rows,
        result_root / "006" / "metrics" / "outer_loop_mission_summary_s006.csv",
        "006",
        "mission_success_label",
        "outer_loop_mission_label_count",
        "Outer-loop mission label count",
    )
    _append_shortlist_blocking_counts(
        rows,
        result_root / "003" / "metrics" / "candidate_shortlist_s003.csv",
    )
    return pd.DataFrame(
        rows,
        columns=(
            "plot_group",
            "source_run_id",
            "metric_name",
            "metric_label",
            "metric_value",
            "source_file",
        ),
    )


def _append_value_counts(
    rows: list[dict[str, object]],
    source: Path,
    run_id: str,
    column: str,
    plot_group: str,
    metric_label: str,
) -> None:
    frame = _read_csv_if_present(source)
    if frame.empty or column not in frame.columns:
        return
    for value, count in frame[column].value_counts(dropna=False).sort_index().items():
        rows.append(
            {
                "plot_group": plot_group,
                "source_run_id": run_id,
                "metric_name": _string_or_none(value),
                "metric_label": metric_label,
                "metric_value": int(count),
                "source_file": _repo_relative(source),
            }
        )


def build_plot_ready_blocker_counts(blockers: pd.DataFrame) -> pd.DataFrame:
    """Return grouped blocker counts for later baseline plotting."""

    if blockers.empty:
        return pd.DataFrame(
            columns=(
                "blocker_scope",
                "objective",
                "evidence_status",
                "failure_label",
                "active_limiting_mechanism",
                "blocker_count",
            )
        )
    return (
        blockers.groupby(
            [
                "blocker_scope",
                "objective",
                "evidence_status",
                "failure_label",
                "active_limiting_mechanism",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="blocker_count")
        .sort_values(
            [
                "blocker_scope",
                "objective",
                "evidence_status",
                "failure_label",
                "active_limiting_mechanism",
            ]
        )
    )


def _append_shortlist_blocking_counts(rows: list[dict[str, object]], source: Path) -> None:
    shortlist = _read_csv_if_present(source)
    if shortlist.empty or "selection_status" not in shortlist.columns:
        return
    status = shortlist["selection_status"]
    blocking_count = int(status.isin(BLOCKING_SHORTLIST_STATUSES).sum())
    non_blocking_count = int(status.isin(NON_BLOCKING_SHORTLIST_STATUSES).sum())
    for metric_name, value in (
        ("blocking_shortlist_rows", blocking_count),
        ("non_blocking_shortlist_rows", non_blocking_count),
    ):
        rows.append(
            {
                "plot_group": "run003_shortlist_blocking_status",
                "source_run_id": "003",
                "metric_name": metric_name,
                "metric_label": "Run-003 shortlist blocking classification count",
                "metric_value": value,
                "source_file": _repo_relative(source),
            }
        )


# =============================================================================
# 4) External Validation
# =============================================================================
def run_stage0_external_validation(
    *,
    repo_root: Path = REPO_ROOT,
    test_paths: tuple[str, ...] = REQUIRED_EXTERNAL_TESTS,
) -> dict[str, object]:
    """Run the Stage 0 external regression suite and return auditable status fields."""

    repo_root = Path(repo_root)
    test_paths = tuple(str(path) for path in test_paths)
    missing = [path for path in test_paths if not (repo_root / path).exists()]
    command = [sys.executable, "-m", "pytest", "-q", *test_paths]
    if missing:
        return {
            "external_validation_status": "failed_missing_test_files",
            "external_validation_exit_code": None,
            "external_validation_command": command,
            "external_validation_test_paths": list(test_paths),
            "external_validation_missing_tests": missing,
            "external_validation_stdout_tail": "",
            "external_validation_stderr_tail": "",
        }

    snapshot = _snapshot_baseline_tree(repo_root)
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        _restore_baseline_tree(repo_root, snapshot)

    status = "passed" if result.returncode == 0 else "failed"
    return {
        "external_validation_status": status,
        "external_validation_exit_code": int(result.returncode),
        "external_validation_command": command,
        "external_validation_test_paths": list(test_paths),
        "external_validation_missing_tests": [],
        "external_validation_stdout_tail": _tail_text(result.stdout),
        "external_validation_stderr_tail": _tail_text(result.stderr),
    }


def update_stage0_validation_status(
    *,
    outputs: FreezeOutputs,
    validation: dict[str, object],
) -> None:
    """Update the Stage 0 manifest and summary after external validation."""

    manifest = json.loads(outputs.manifest_json.read_text(encoding="ascii"))
    manifest.update(validation)
    manifest["external_validation_failures"] = _external_validation_failures(validation)
    manifest["overall_stage0_gate_status"] = _overall_stage0_status(
        str(manifest.get("freeze_gate_status", "failed")),
        str(manifest.get("external_validation_status", "pending")),
    )
    manifest["phase_a_stage0_complete"] = manifest["overall_stage0_gate_status"] == "passed"
    manifest["phase_b_implementation_allowed"] = False
    manifest["phase_b_blocked_reason"] = PHASE_B_BLOCKED_REASON
    outputs.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    _write_baseline_summary_from_manifest(outputs.baseline_summary_md, manifest)


def _snapshot_baseline_tree(repo_root: Path) -> dict[str, bytes]:
    result_root = repo_root / "03_Control" / "05_Results" / CAMPAIGN
    snapshot: dict[str, bytes] = {}
    for run_id in REQUIRED_BASELINE_RUNS:
        run_dir = result_root / f"{run_id:03d}"
        if not run_dir.exists():
            continue
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                snapshot[_repo_relative(path)] = path.read_bytes()
    return snapshot


def _restore_baseline_tree(repo_root: Path, snapshot: dict[str, bytes]) -> None:
    result_root = repo_root / "03_Control" / "05_Results" / CAMPAIGN
    baseline_roots = [result_root / f"{run_id:03d}" for run_id in REQUIRED_BASELINE_RUNS]
    expected = {repo_root / relative for relative in snapshot}

    for run_dir in baseline_roots:
        if not run_dir.exists():
            continue
        for path in sorted(run_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() and path not in expected:
                path.unlink()
    for relative, content in snapshot.items():
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)


def _tail_text(text: str, max_lines: int = 30, max_chars: int = 4000) -> str:
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def _external_validation_failures(validation: dict[str, object]) -> list[str]:
    status = str(validation.get("external_validation_status", "pending"))
    if status == "passed" or status == "pending":
        return []
    if status == "failed_missing_test_files":
        missing = validation.get("external_validation_missing_tests", [])
        return [f"missing required external validation test: {path}" for path in missing]
    exit_code = validation.get("external_validation_exit_code")
    return [f"external validation command failed with exit code {exit_code}"]


def _overall_stage0_status(freeze_gate_status: str, external_validation_status: str) -> str:
    if freeze_gate_status != "passed":
        return "failed"
    if external_validation_status == "passed":
        return "passed"
    if external_validation_status == "pending":
        return "pending_external_validation"
    return "failed"


# =============================================================================
# 5) Report and Manifest Writers
# =============================================================================
def _write_claim_boundary(
    path: Path,
    *,
    freeze_gate_status: str,
    missing_runs: list[str],
) -> None:
    lines = [
        "# Frozen Baseline Claim Boundary",
        "",
        f"Freeze gate status: `{freeze_gate_status}`.",
        "",
        "## Evidence Scope",
        "",
        "Runs `002` to `006` support only narrow baseline claims:",
        "",
        "- deterministic primitive evidence from run `002`;",
        "- shortlist and W3 planning evidence from run `003`;",
        "- selected W3 stress evidence from run `004`;",
        "- offline governor seed evidence from run `005`;",
        "- short governed-transit and rejection evidence from run `006`.",
        "",
        "The frozen baseline does not answer the final widening-versus-growth research question.",
        "It also does not demonstrate either final project objective.",
        "",
        "## Forbidden Claims",
        "",
        "- sustained updraft exploitation;",
        "- prolonged confined arena operation;",
        "- objective-one sustained operation;",
        "- objective-two volume coverage;",
        "- volume coverage mission completion;",
        "- successful real flight transfer;",
        "- full target ladder evidence;",
        "- high angle reversal transfer evidence;",
        "- paired sim-to-real validation;",
        "- final widening-versus-growth conclusion.",
        "",
        "## Missing Baseline Runs",
        "",
    ]
    if missing_runs:
        for run_id in missing_runs:
            lines.append(f"- Missing required run `{run_id}`.")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Plot Status",
            "",
            "Rendered baseline plots are deferred in this freeze pass.",
            "The plot-ready CSVs are sufficient for Stage 0 freeze auditing only, not for the final Phase A writing package.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_baseline_summary(
    path: Path,
    *,
    run_rows: list[dict[str, object]],
    hash_rows: list[dict[str, object]],
    blockers: pd.DataFrame,
    freeze_gate_status: str,
    external_validation_status: str,
    overall_stage0_gate_status: str,
) -> None:
    lines = [
        "# Frozen Baseline Summary",
        "",
        f"- Freeze gate status: `{freeze_gate_status}`",
        "- External validation required: `true`",
        f"- External validation status: `{external_validation_status}`",
        f"- Overall Stage 0 gate status: `{overall_stage0_gate_status}`",
        f"- Source files hashed: `{len(hash_rows)}`",
        f"- Blocker rows written: `{len(blockers)}`",
        "- Rendered plots: `deferred`",
        "- Plot-ready CSVs: `written_for_freeze_audit_only`",
        "- Phase B implementation allowed: `false`",
        "",
        "## Baseline Runs",
        "",
    ]
    for row in run_rows:
        lines.append(
            "- "
            f"`{row['run_id']}`: exists `{str(row['exists']).lower()}`, "
            f"hashable files `{row['hashable_file_count']}`"
        )
    lines.extend(
        [
            "",
            "## Allowed Claim",
            "",
            "The repository contains a frozen narrow primitive-library baseline with deterministic evidence, selected W3 stress, offline governor seed evidence, and short outer-loop transit/rejection evidence.",
            "",
            "## Forbidden Claims",
            "",
            "No sustained updraft exploitation, prolonged operation, objective-one success, objective-two volume coverage, real-flight transfer, full target ladder, high-angle reversal transfer, paired sim-real validation, or final widening-versus-growth claim is allowed from this baseline.",
            "",
            "## Phase B Boundary",
            "",
            PHASE_B_BLOCKED_REASON,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_baseline_summary_from_manifest(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Frozen Baseline Summary",
        "",
        f"- Freeze gate status: `{manifest.get('freeze_gate_status')}`",
        f"- External validation required: `{str(manifest.get('external_validation_required')).lower()}`",
        f"- External validation status: `{manifest.get('external_validation_status')}`",
        f"- Overall Stage 0 gate status: `{manifest.get('overall_stage0_gate_status')}`",
        f"- Source files hashed: `{manifest.get('source_hash_count')}`",
        f"- Blocker rows written: `{manifest.get('blocker_row_count')}`",
        f"- Rendered plots: `{manifest.get('rendered_plots_status')}`",
        "- Plot-ready CSVs: `written_for_freeze_audit_only`",
        f"- Phase B implementation allowed: `{str(manifest.get('phase_b_implementation_allowed')).lower()}`",
        "",
        "## Baseline Runs",
        "",
    ]
    for row in manifest.get("baseline_run_inventory", []):
        lines.append(
            "- "
            f"`{row['run_id']}`: exists `{str(row['exists']).lower()}`, "
            f"hashable files `{row['hashable_file_count']}`"
        )
    lines.extend(
        [
            "",
            "## Allowed Claim",
            "",
            str(manifest.get("allowed_stage0_claim")),
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in manifest.get("forbidden_claims", []):
        lines.append(f"- {claim}")
    lines.extend(
        [
            "",
            "## Phase B Boundary",
            "",
            str(manifest.get("phase_b_blocked_reason", PHASE_B_BLOCKED_REASON)),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _build_manifest(
    *,
    run_id: int,
    baseline_runs: tuple[int, ...],
    outputs: FreezeOutputs,
    run_rows: list[dict[str, object]],
    before_hash_rows: list[dict[str, object]],
    after_hash_rows: list[dict[str, object]],
    blockers: pd.DataFrame,
    plot_ready_summary: pd.DataFrame,
    plot_ready_blocker_counts: pd.DataFrame,
    freeze_gate_status: str,
    freeze_gate_failures: list[str],
    external_validation_status: str,
    external_validation_failures: list[str],
    overall_stage0_gate_status: str,
) -> dict[str, object]:
    result_root = outputs.root.parent
    shortlist_counts = _shortlist_status_counts(result_root)
    external_tests = [
        {
            "path": path,
            "exists": bool((REPO_ROOT / path).exists()),
        }
        for path in REQUIRED_EXTERNAL_TESTS
    ]
    external_missing_tests = [
        item["path"] for item in external_tests if not bool(item["exists"])
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": f"s{int(run_id):03d}",
        "campaign": CAMPAIGN,
        "pass_name": "stage0_strict_baseline_freeze_gate",
        "baseline_runs_required": [f"{int(item):03d}" for item in baseline_runs],
        "freeze_gate_status": freeze_gate_status,
        "freeze_gate_failures": freeze_gate_failures,
        "external_validation_required": True,
        "external_validation_status": external_validation_status,
        "external_validation_exit_code": None,
        "external_validation_command": [sys.executable, "-m", "pytest", "-q", *REQUIRED_EXTERNAL_TESTS],
        "external_validation_test_paths": list(REQUIRED_EXTERNAL_TESTS),
        "external_validation_missing_tests": external_missing_tests,
        "external_validation_stdout_tail": "",
        "external_validation_stderr_tail": "",
        "external_validation_failures": external_validation_failures,
        "required_external_validation_tests": external_tests,
        "overall_stage0_gate_status": overall_stage0_gate_status,
        "source_hash_scope": "only required baseline runs 002 to 006; generated 000_frozen_baseline outputs excluded",
        "hashed_suffixes": list(HASHED_SUFFIXES),
        "baseline_run_inventory": run_rows,
        "source_hashes_before": before_hash_rows,
        "source_hashes_after": after_hash_rows,
        "source_hashes_unchanged_after_writing": _hash_map(before_hash_rows) == _hash_map(after_hash_rows),
        "source_hash_count": int(len(before_hash_rows)),
        "blocker_row_count": int(len(blockers)),
        "not_evaluated_blocker_row_count": int((blockers["evidence_status"] == "not_evaluated").sum())
        if "evidence_status" in blockers
        else 0,
        "plot_ready_summary_row_count": int(len(plot_ready_summary)),
        "plot_ready_blocker_count_rows": int(len(plot_ready_blocker_counts)),
        "run003_shortlist_blocking_row_count": shortlist_counts["blocking_shortlist_rows"],
        "run003_shortlist_non_blocking_row_count": shortlist_counts["non_blocking_shortlist_rows"],
        "run003_shortlist_selection_status_counts": shortlist_counts["selection_status_counts"],
        "rendered_plots_status": "deferred",
        "plot_ready_csvs_sufficient_for_freeze_audit_only": True,
        "plot_ready_csvs_sufficient_for_final_phase_a_writing_package": False,
        "phase_a_stage0_complete": overall_stage0_gate_status == "passed",
        "phase_b_implementation_allowed": False,
        "phase_b_blocked_reason": PHASE_B_BLOCKED_REASON,
        "recommended_next_branch_after_stage0": "rewrite/phase-b-dense-archive-planning",
        "forbidden_scope_this_pass": list(FORBIDDEN_SCOPE_THIS_PASS),
        "allowed_stage0_claim": (
            "narrow frozen baseline: deterministic primitive evidence, shortlist/W3 planning, "
            "selected W3 stress, offline governor seed evidence, and short governed-transit/rejection evidence"
        ),
        "forbidden_claims": [
            "sustained updraft exploitation",
            "prolonged confined arena operation",
            "objective-one sustained operation",
            "objective-two volume coverage",
            "volume coverage mission completion",
            "successful real flight transfer",
            "full target ladder evidence",
            "high angle reversal transfer evidence",
            "paired sim-to-real validation",
            "final widening-versus-growth conclusion",
        ],
        "output_files": {key: _repo_relative(path) for key, path in outputs.as_dict().items()},
    }


def _shortlist_status_counts(result_root: Path) -> dict[str, object]:
    shortlist = _read_csv_if_present(result_root / "003" / "metrics" / "candidate_shortlist_s003.csv")
    if shortlist.empty or "selection_status" not in shortlist.columns:
        return {
            "blocking_shortlist_rows": 0,
            "non_blocking_shortlist_rows": 0,
            "selection_status_counts": {},
        }
    status = shortlist["selection_status"]
    return {
        "blocking_shortlist_rows": int(status.isin(BLOCKING_SHORTLIST_STATUSES).sum()),
        "non_blocking_shortlist_rows": int(status.isin(NON_BLOCKING_SHORTLIST_STATUSES).sum()),
        "selection_status_counts": {
            str(key): int(value)
            for key, value in status.value_counts(dropna=False).sort_index().items()
        },
    }


def _status_fields(
    run_rows: list[dict[str, object]],
    before_hash_rows: list[dict[str, object]],
    after_hash_rows: list[dict[str, object]],
    *,
    allow_missing_runs_for_tests: bool,
) -> tuple[str, list[str], str, list[str], str]:
    failures: list[str] = []
    missing = [row["run_id"] for row in run_rows if not row["exists"]]
    empty = [row["run_id"] for row in run_rows if row["exists"] and int(row["hashable_file_count"]) == 0]
    if missing:
        failures.append(f"missing required baseline runs: {', '.join(missing)}")
    if empty:
        failures.append(f"baseline runs have no hashable source files: {', '.join(empty)}")
    if _hash_map(before_hash_rows) != _hash_map(after_hash_rows):
        failures.append("baseline source hashes changed while writing freeze outputs")
    if not before_hash_rows and not allow_missing_runs_for_tests:
        failures.append("no baseline source hashes were written")

    freeze_gate_status = "failed" if failures else "passed"

    missing_tests = [path for path in REQUIRED_EXTERNAL_TESTS if not (REPO_ROOT / path).exists()]
    external_failures = [f"missing required external validation test: {path}" for path in missing_tests]
    external_validation_status = "failed_missing_test_files" if external_failures else "pending"

    if freeze_gate_status == "failed" or external_validation_status.startswith("failed"):
        overall = "failed"
    else:
        overall = "pending_external_validation"
    return freeze_gate_status, failures, external_validation_status, external_failures, overall


# =============================================================================
# 5) Freeze Workflow
# =============================================================================
def freeze_primitive_baseline(
    *,
    result_root: Path,
    baseline_runs: tuple[int, ...] = REQUIRED_BASELINE_RUNS,
    run_id: int = 0,
    overwrite: bool = False,
    allow_missing_runs_for_tests: bool = False,
) -> dict[str, Path]:
    """Hash baseline evidence and write baseline claim boundary outputs."""

    result_root = Path(result_root)
    outputs = _prepare_output_tree(result_root, run_id, overwrite)
    baseline_runs = tuple(int(item) for item in baseline_runs)

    run_rows, before_hash_rows = _hash_baseline_sources(result_root, baseline_runs)
    blockers = build_baseline_blocker_table(result_root)
    plot_ready_summary = build_plot_ready_summary(result_root, run_rows, before_hash_rows)
    plot_ready_blocker_counts = build_plot_ready_blocker_counts(blockers)

    blockers.to_csv(outputs.blocker_csv, index=False)
    plot_ready_summary.to_csv(outputs.plot_ready_summary_csv, index=False)
    plot_ready_blocker_counts.to_csv(outputs.plot_ready_blocker_counts_csv, index=False)

    _, after_hash_rows = _hash_baseline_sources(result_root, baseline_runs)
    (
        freeze_gate_status,
        freeze_gate_failures,
        external_validation_status,
        external_validation_failures,
        overall_stage0_gate_status,
    ) = _status_fields(
        run_rows,
        before_hash_rows,
        after_hash_rows,
        allow_missing_runs_for_tests=allow_missing_runs_for_tests,
    )
    missing_runs = [row["run_id"] for row in run_rows if not row["exists"]]

    _write_claim_boundary(
        outputs.claim_boundary_md,
        freeze_gate_status=freeze_gate_status,
        missing_runs=missing_runs,
    )
    _write_baseline_summary(
        outputs.baseline_summary_md,
        run_rows=run_rows,
        hash_rows=before_hash_rows,
        blockers=blockers,
        freeze_gate_status=freeze_gate_status,
        external_validation_status=external_validation_status,
        overall_stage0_gate_status=overall_stage0_gate_status,
    )
    manifest = _build_manifest(
        run_id=run_id,
        baseline_runs=baseline_runs,
        outputs=outputs,
        run_rows=run_rows,
        before_hash_rows=before_hash_rows,
        after_hash_rows=after_hash_rows,
        blockers=blockers,
        plot_ready_summary=plot_ready_summary,
        plot_ready_blocker_counts=plot_ready_blocker_counts,
        freeze_gate_status=freeze_gate_status,
        freeze_gate_failures=freeze_gate_failures,
        external_validation_status=external_validation_status,
        external_validation_failures=external_validation_failures,
        overall_stage0_gate_status=overall_stage0_gate_status,
    )
    outputs.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return outputs.as_dict()


# =============================================================================
# 6) CLI Entry Point
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--baseline-runs", nargs="+", type=int, default=list(REQUIRED_BASELINE_RUNS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = freeze_primitive_baseline(
        result_root=RESULT_ROOT,
        baseline_runs=tuple(int(item) for item in args.baseline_runs),
        run_id=int(args.run_id),
        overwrite=bool(args.overwrite),
    )
    if bool(args.run_validation):
        validation = run_stage0_external_validation(repo_root=REPO_ROOT)
        update_stage0_validation_status(
            outputs=FreezeOutputs(
                root=paths["root"],
                manifest_json=paths["manifest_json"],
                blocker_csv=paths["blocker_csv"],
                plot_ready_summary_csv=paths["plot_ready_summary_csv"],
                plot_ready_blocker_counts_csv=paths["plot_ready_blocker_counts_csv"],
                claim_boundary_md=paths["claim_boundary_md"],
                baseline_summary_md=paths["baseline_summary_md"],
            ),
            validation=validation,
        )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    for key, path in paths.items():
        print(f"{key}={path}")
    if manifest["overall_stage0_gate_status"] == "failed":
        print("overall_stage0_gate_status=failed", file=sys.stderr)
        return 1
    if bool(args.run_validation) and manifest["overall_stage0_gate_status"] != "passed":
        print("overall_stage0_gate_status=pending_or_failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
