from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and data containers
# 2) Source loading and shortlist construction
# 3) Coverage decisions and W3 planning
# 4) Higher-target request logic
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
SELECTION_STATUSES = (
    "selected_for_w3_stress",
    "selected_for_governor_seed",
    "needs_seed_refinement",
    "governor_reject_entry_envelope",
    "boundary_only",
    "deferred_not_relevant",
)

COVERAGE_DECISIONS = (
    "covered_keep",
    "covered_send_to_w3",
    "entry_envelope_reject",
    "generator_refinement_needed",
    "boundary_keep_for_discussion",
    "higher_target_screen_recommended",
    "not_evaluated",
)

HIGHER_TARGET_REQUEST_STATUSES = (
    "recommended_next",
    "possible_later",
    "defer_boundary_only",
    "refine_30_seed",
    "not_requested_boundary_only",
    "not_requested_insufficient_evidence",
)

FUTURE_TARGETS_DEG = (45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
ACTIVE_EVIDENCE_TARGETS_DEG = (15.0, 30.0)


@dataclass(frozen=True)
class SourceEvidence:
    manifest: dict[str, object]
    evidence: pd.DataFrame
    library_summary: pd.DataFrame
    envelope_group_summary: pd.DataFrame
    coverage_region_summary: pd.DataFrame


# =============================================================================
# 2) Source Loading and Shortlist Construction
# =============================================================================
def load_source_evidence(source_root: Path, source_run_id: int = 2) -> SourceEvidence:
    """Read the run evidence required for the planning-only shortlist pass."""

    suffix = f"s{source_run_id:03d}"
    paths = {
        "manifest": source_root / "manifests" / f"primitive_library_manifest_{suffix}.json",
        "evidence": source_root / "metrics" / f"primitive_evidence_library_{suffix}.csv",
        "library_summary": source_root / "metrics" / f"primitive_library_summary_{suffix}.csv",
        "group_summary": source_root / "metrics" / f"primitive_envelope_group_summary_{suffix}.csv",
        "coverage_summary": source_root / "metrics" / f"primitive_coverage_region_summary_{suffix}.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run-{source_run_id:03d} source evidence: {missing}")

    return SourceEvidence(
        manifest=json.loads(paths["manifest"].read_text(encoding="ascii")),
        evidence=pd.read_csv(paths["evidence"]),
        library_summary=pd.read_csv(paths["library_summary"]),
        envelope_group_summary=pd.read_csv(paths["group_summary"]),
        coverage_region_summary=pd.read_csv(paths["coverage_summary"]),
    )


def build_candidate_shortlist(evidence: pd.DataFrame) -> pd.DataFrame:
    """Return one planning row per source evidence row with selection status."""

    rows = []
    for _, row in evidence.iterrows():
        record = row.to_dict()
        status = classify_selection_status(record)
        record.update(
            {
                "selection_status": status,
                "selection_reason": _selection_reason(record, status),
                "source_run_id": "s002",
                "shortlist_pass": "s003",
                "planning_only": True,
            }
        )
        rows.append(record)

    shortlist = pd.DataFrame(rows)
    return shortlist.sort_values(by=["coverage_region_id", "selection_status", "primitive_id"]).reset_index(drop=True)


def classify_selection_status(row: dict[str, object]) -> str:
    """Classify one run-002 evidence row for shortlist planning."""

    candidate_class = str(row.get("candidate_class", ""))
    coverage_status = str(row.get("coverage_status", ""))
    entry_status = str(row.get("entry_envelope_status", ""))
    envelope_status = str(row.get("envelope_status", ""))
    evaluation_status = str(row.get("evaluation_status", "evaluated"))

    if evaluation_status != "evaluated" or candidate_class == "not_evaluated":
        return "deferred_not_relevant"
    if candidate_class == "updraft_assisted_commandable":
        return "selected_for_w3_stress"
    if candidate_class in ("w0_standalone_commandable", "w0_updraft_pending_target_candidate"):
        return "selected_for_governor_seed"
    if entry_status == "outside_entry_envelope_governor_reject" or coverage_status == "uncovered_governor_reject":
        return "governor_reject_entry_envelope"
    if envelope_status == "candidate_family_needs_refinement" or coverage_status == "uncovered_needs_refinement":
        return "needs_seed_refinement"
    if candidate_class == "boundary_evidence" or coverage_status == "uncovered_boundary":
        return "boundary_only"
    return "deferred_not_relevant"


def _selection_reason(row: dict[str, object], selection_status: str) -> str:
    reasons = {
        "selected_for_w3_stress": "updraft_assisted_w1_w2_candidate_pending_w3_stress",
        "selected_for_governor_seed": "commandable_seed_for_later_governor_examples",
        "needs_seed_refinement": "current_seed_misses_target_but_is_not_a_hard_boundary",
        "governor_reject_entry_envelope": "entry_clearance_or_footprint_exceeds_available_envelope",
        "boundary_only": "finite_planning_evidence_kept_for_discussion_not_commandable",
        "deferred_not_relevant": "not_needed_for_run_003_shortlist",
    }
    if selection_status == "boundary_only" and str(row.get("failure_label", "")):
        return f"{reasons[selection_status]}:{row.get('failure_label')}"
    return reasons[selection_status]


# =============================================================================
# 3) Coverage Decisions and W3 Planning
# =============================================================================
def build_coverage_decision_summary(
    coverage: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    """Return one coverage decision row per run-002 coverage region."""

    by_primitive = {
        str(row["primitive_id"]): row
        for _, row in shortlist.iterrows()
    }
    rows = []
    for _, row in coverage.iterrows():
        best_id = str(row["best_primitive_id"])
        best = by_primitive.get(best_id)
        best_status = "deferred_not_relevant" if best is None else str(best["selection_status"])
        decision, reason = _coverage_decision(row.to_dict(), best_status)
        rows.append(
            {
                "coverage_region_id": row["coverage_region_id"],
                "row_count": int(row["row_count"]),
                "best_primitive_id": best_id,
                "best_family": row["best_family"],
                "best_candidate_class": row["best_candidate_class"],
                "best_selection_status": best_status,
                "coverage_status_s002": row["coverage_status"],
                "coverage_decision_s003": decision,
                "library_growth_trigger_s003": bool(decision == "higher_target_screen_recommended"),
                "reason": reason,
                "needs_refinement": bool(decision == "generator_refinement_needed"),
                "needs_governor_reject": bool(decision == "entry_envelope_reject"),
                "needs_w3": bool(decision == "covered_send_to_w3"),
                "needs_higher_target_screen": bool(decision == "higher_target_screen_recommended"),
                "best_heading_error_deg": float(row["best_heading_error_deg"]),
                "best_path_length_xy_m": float(row["best_path_length_xy_m"]),
                "best_footprint_m2": float(row["best_footprint_m2"]),
                "best_terminal_speed_m_s": float(row["best_terminal_speed_m_s"]),
                "best_energy_residual_m": float(row["best_energy_residual_m"]),
                "best_min_true_margin_m": float(row["best_min_true_margin_m"]),
                "target_heading_deg": _target_from_region_id(str(row["coverage_region_id"])),
                "source_run_id": "s002",
            }
        )
    return pd.DataFrame(rows).sort_values(by=["coverage_region_id"]).reset_index(drop=True)


def build_w3_stress_plan(shortlist: pd.DataFrame, max_w3_candidates: int = 5) -> pd.DataFrame:
    """Return a planning-only W3 stress table with no execution side effects."""

    candidates = shortlist[shortlist["selection_status"] == "selected_for_w3_stress"].copy()
    if candidates.empty:
        return pd.DataFrame(columns=_w3_columns())

    candidates["_target_priority"] = candidates["target_heading_deg"].isna().astype(int)
    candidates = candidates.sort_values(
        by=[
            "_target_priority",
            "target_heading_deg",
            "terminal_heading_error_deg",
            "turn_footprint_proxy_m2",
            "path_length_xy_m",
            "terminal_speed_m_s",
            "min_true_margin_m",
        ],
        ascending=[True, True, True, True, True, False, False],
        na_position="first",
    ).head(max_w3_candidates)

    rows = []
    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        rows.append(
            {
                "w3_plan_id": f"w3_s003_{index:02d}",
                "source_primitive_id": row["primitive_id"],
                "family": row["family"],
                "target_heading_deg": row["target_heading_deg"],
                "updraft_config": row["updraft_config"],
                "wind_fidelity": row["wind_fidelity"],
                "start_condition": row["start_condition"],
                "selection_reason": row["selection_reason"],
                "base_candidate_class": row["candidate_class"],
                "base_coverage_status": row["coverage_status"],
                "stress_seed_count": 25,
                "start_position_perturbation_m": 0.10,
                "speed_perturbation_m_s": 0.25,
                "attitude_perturbation_deg": 3.0,
                "updraft_strength_scale_range": "0.85:1.15",
                "updraft_center_shift_m": 0.10,
                "updraft_width_scale_range": "0.90:1.10",
                "latency_perturbation_s": 0.02,
                "acceptance_metrics": "finite,true_safe,heading_band,terminal_speed,recovery_proxy,command_saturation",
                "not_implemented_in_this_pass": True,
            }
        )
    return pd.DataFrame(rows, columns=_w3_columns())


def _coverage_decision(row: dict[str, object], best_selection_status: str) -> tuple[str, str]:
    coverage_status = str(row.get("coverage_status", ""))
    if coverage_status == "covered_by_existing_envelope":
        if best_selection_status == "selected_for_w3_stress":
            return "covered_send_to_w3", "covered_in_w1_w2_but_requires_selected_w3_stress_before_claim"
        return "covered_keep", "covered_by_existing_envelope"
    if coverage_status == "updraft_pending_coverage":
        return "covered_send_to_w3", "updraft_pending_candidate_requires_selected_w3_stress"
    if coverage_status == "uncovered_governor_reject":
        return "entry_envelope_reject", "entry_clearance_or_footprint_limited_not_library_growth"
    if coverage_status == "uncovered_needs_refinement":
        return "generator_refinement_needed", "refine_existing_seed_before_higher_target_request"
    if coverage_status == "not_evaluated_model_unavailable":
        return "not_evaluated", "source_wind_model_unavailable"
    if coverage_status == "requires_library_growth":
        return "higher_target_screen_recommended", "explicit_run002_library_growth_trigger"
    return "boundary_keep_for_discussion", "boundary_or_true_safety_limited_not_commandable"


def _w3_columns() -> list[str]:
    return [
        "w3_plan_id",
        "source_primitive_id",
        "family",
        "target_heading_deg",
        "updraft_config",
        "wind_fidelity",
        "start_condition",
        "selection_reason",
        "base_candidate_class",
        "base_coverage_status",
        "stress_seed_count",
        "start_position_perturbation_m",
        "speed_perturbation_m_s",
        "attitude_perturbation_deg",
        "updraft_strength_scale_range",
        "updraft_center_shift_m",
        "updraft_width_scale_range",
        "latency_perturbation_s",
        "acceptance_metrics",
        "not_implemented_in_this_pass",
    ]


# =============================================================================
# 4) Higher-Target Request Logic
# =============================================================================
def build_higher_target_growth_request(
    coverage_decision: pd.DataFrame,
    mission_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return coverage-driven future-target requests without automatic escalation."""

    mission = pd.DataFrame() if mission_coverage is None else mission_coverage.copy()
    rows = []
    for target in FUTURE_TARGETS_DEG:
        row = _higher_target_row(float(target), coverage_decision, mission)
        rows.append(row)
    return pd.DataFrame(rows)


def _higher_target_row(target: float, coverage_decision: pd.DataFrame, mission: pd.DataFrame) -> dict[str, object]:
    mission_rows = _mission_rows_for_target(target, mission)
    mission_critical = _any_truthy(mission_rows, "mission_critical_region_present")
    safe_alternative = _safe_15_30_alternative_present(coverage_decision, mission_rows)
    plausible_short_footprint = _any_truthy(mission_rows, "plausible_shorter_footprint_entry_envelope")
    mission_failure_mode = _first_nonempty(mission_rows, "current_failure_mode", default="")
    target_30_statuses = set(_coverage_statuses_for_target(coverage_decision, 30.0))
    target_30_reasons = set(_reasons_for_target(coverage_decision, 30.0))
    target_30_decisions = set(_decisions_for_target(coverage_decision, 30.0))

    if target in (45.0, 60.0):
        if _mission_supports_library_growth(mission_critical, safe_alternative, plausible_short_footprint, mission_failure_mode):
            status = "recommended_next"
            reason = "mission_critical_region_has_no_safe_baseline_15_30_alternative_and_indicates_library_growth"
        elif "generator_refinement_needed" in target_30_decisions:
            status = "refine_30_seed"
            reason = "existing_30_deg_evidence_needs_seed_refinement_before_higher_target_request"
        elif target_30_statuses & {"uncovered_governor_reject", "uncovered_boundary"} or target_30_reasons:
            status = "defer_boundary_only"
            reason = "30_deg_uncovered_region_is_boundary_or_entry_envelope_limited_not_library_growth"
        else:
            status = "not_requested_insufficient_evidence"
            reason = "no_mission_critical_coverage_gap_supports_higher_target_request"
    else:
        status = "not_requested_boundary_only"
        reason = "targets_above_60_deg_are_not_next_step_without_specific_coverage_evidence"

    return {
        "requested_target_deg": target,
        "request_status": status,
        "reason": reason,
        "source_evidence": _source_evidence_summary(coverage_decision, mission_rows),
        "recommended_families": _recommended_families(mission_rows, status),
        "recommended_start_conditions": _recommended_start_conditions(mission_rows, status),
        "recommended_wind_fidelities": _recommended_wind_fidelities(mission_rows, status),
        "recommended_updraft_configs": _recommended_updraft_configs(mission_rows, status),
        "hard_stop_rule": _hard_stop_rule(status),
        "current_coverage_status": ",".join(sorted(target_30_statuses)) if target in (45.0, 60.0) else "not_evaluated_in_run_002",
        "current_failure_mode": mission_failure_mode or ",".join(sorted(target_30_reasons)) or "none",
        "mission_critical_region_present": bool(mission_critical),
        "safe_15_or_30_alternative_present": bool(safe_alternative),
        "plausible_shorter_footprint_entry_envelope": bool(plausible_short_footprint),
        "coverage_decision_source": "mission_coverage_row" if not mission_rows.empty else "run_002_coverage_summary",
    }


def _mission_supports_library_growth(
    mission_critical: bool,
    safe_alternative: bool,
    plausible_short_footprint: bool,
    failure_mode: str,
) -> bool:
    library_growth_modes = {
        "requires_library_growth",
        "library_growth_gap",
        "coverage_gap_after_envelope_widening",
        "outer_loop_task_gap",
    }
    return (
        bool(mission_critical)
        and not bool(safe_alternative)
        and bool(plausible_short_footprint)
        and failure_mode in library_growth_modes
    )


def _mission_rows_for_target(target: float, mission: pd.DataFrame) -> pd.DataFrame:
    if mission.empty:
        return mission
    if "requested_target_deg" in mission.columns:
        return mission[np.isclose(mission["requested_target_deg"].astype(float), target)]
    if "required_target_deg" in mission.columns:
        return mission[np.isclose(mission["required_target_deg"].astype(float), target)]
    return mission.iloc[0:0]


def _safe_15_30_alternative_present(coverage_decision: pd.DataFrame, mission_rows: pd.DataFrame) -> bool:
    if not mission_rows.empty and "safe_15_or_30_alternative_present" in mission_rows.columns:
        return bool(mission_rows["safe_15_or_30_alternative_present"].astype(bool).any())
    covered = coverage_decision[
        coverage_decision["target_heading_deg"].isin([np.nan, 15.0, 30.0])
        & coverage_decision["coverage_decision_s003"].isin(["covered_keep", "covered_send_to_w3"])
    ]
    return not covered.empty


def _coverage_statuses_for_target(coverage_decision: pd.DataFrame, target: float) -> Iterable[str]:
    rows = coverage_decision[np.isclose(coverage_decision["target_heading_deg"].fillna(-1).astype(float), target)]
    return [str(value) for value in rows["coverage_status_s002"].dropna().unique()]


def _reasons_for_target(coverage_decision: pd.DataFrame, target: float) -> Iterable[str]:
    rows = coverage_decision[np.isclose(coverage_decision["target_heading_deg"].fillna(-1).astype(float), target)]
    return [str(value) for value in rows["reason"].dropna().unique()]


def _decisions_for_target(coverage_decision: pd.DataFrame, target: float) -> Iterable[str]:
    rows = coverage_decision[np.isclose(coverage_decision["target_heading_deg"].fillna(-1).astype(float), target)]
    return [str(value) for value in rows["coverage_decision_s003"].dropna().unique()]


def _source_evidence_summary(coverage_decision: pd.DataFrame, mission_rows: pd.DataFrame) -> str:
    if not mission_rows.empty:
        region = _first_nonempty(mission_rows, "coverage_region_id", default="mission_coverage")
        return f"mission_coverage:{region}"
    counts = coverage_decision["coverage_decision_s003"].value_counts(dropna=False).to_dict()
    return f"run_002_coverage_decisions:{counts}"


def _recommended_families(mission_rows: pd.DataFrame, status: str) -> str:
    if status != "recommended_next":
        return "none"
    return _first_nonempty(mission_rows, "recommended_families", default="canyon_steep_bank,wingover_lite,bank_yaw_energy_retaining")


def _recommended_start_conditions(mission_rows: pd.DataFrame, status: str) -> str:
    if status != "recommended_next":
        return "none"
    return _first_nonempty(mission_rows, "recommended_start_conditions", default="favourable_start,mid_arena_start")


def _recommended_wind_fidelities(mission_rows: pd.DataFrame, status: str) -> str:
    if status != "recommended_next":
        return "none"
    return _first_nonempty(mission_rows, "recommended_wind_fidelities", default="W0,W1,W2")


def _recommended_updraft_configs(mission_rows: pd.DataFrame, status: str) -> str:
    if status != "recommended_next":
        return "none"
    return _first_nonempty(mission_rows, "recommended_updraft_configs", default="none,U1_single_fan,U4_four_fan")


def _hard_stop_rule(status: str) -> str:
    if status == "recommended_next":
        return "future_pass_must_still_pass_true_safety_entry_clearance_recovery_and_command_bridge"
    if status == "refine_30_seed":
        return "do_not_request_45_60_until_30_deg_refinement_is_resolved"
    if status == "defer_boundary_only":
        return "do_not_escalate_boundary_or_governor_reject_without_mission_critical_growth_row"
    return "not_part_of_run_003_execution"


def _any_truthy(rows: pd.DataFrame, column: str) -> bool:
    if rows.empty or column not in rows.columns:
        return False
    return bool(rows[column].astype(bool).any())


def _first_nonempty(rows: pd.DataFrame, column: str, default: str) -> str:
    if rows.empty or column not in rows.columns:
        return default
    values = [str(value) for value in rows[column].dropna().tolist() if str(value)]
    return values[0] if values else default


def _target_from_region_id(region_id: str) -> float:
    first = region_id.split("|", maxsplit=1)[0]
    if first == "target_none":
        return np.nan
    if first.startswith("target_"):
        return float(first.replace("target_", ""))
    return np.nan
