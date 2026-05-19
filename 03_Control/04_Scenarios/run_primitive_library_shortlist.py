from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from primitive_library_selection import (
    COVERAGE_DECISIONS,
    FUTURE_TARGETS_DEG,
    HIGHER_TARGET_REQUEST_STATUSES,
    SELECTION_STATUSES,
    W3_REQUIRED_ROLES,
    build_candidate_shortlist,
    build_coverage_decision_summary,
    build_higher_target_growth_request,
    build_w3_stress_plan,
    load_source_evidence,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and output helpers
# 2) Shortlist pass
# 3) CLI entry point
# =============================================================================


# =============================================================================
# 1) Paths and Output Helpers
# =============================================================================
CAMPAIGN = "09_primitive_library"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _prepare_result_tree(run_id: int, overwrite: bool) -> dict[str, Path]:
    root = RESULT_ROOT / f"{run_id:03d}"
    if root.exists() and overwrite:
        _clear_result_files(root)
    if root.exists() and not overwrite:
        raise ValueError(f"result tree already exists: {root}")
    paths = {
        "root": root,
        "manifests": root / "manifests",
        "metrics": root / "metrics",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _clear_result_files(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _write_manifest(
    path: Path,
    *,
    run_id: int,
    source_run_id: int,
    output_files: dict[str, Path],
    source_manifest: dict[str, object],
    shortlist: pd.DataFrame,
    coverage: pd.DataFrame,
    w3_plan: pd.DataFrame,
    higher_targets: pd.DataFrame,
    max_w3_candidates: int,
) -> dict[str, object]:
    suffix = f"s{run_id:03d}"
    source_suffix = f"s{source_run_id:03d}"
    manifest = {
        "run_id": suffix,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign": CAMPAIGN,
        "pass_name": "primitive_library_shortlist_and_w3_planning",
        "source_run_id": source_suffix,
        "source_evidence_root": f"03_Control/05_Results/{CAMPAIGN}/{source_run_id:03d}",
        "run_002_only_source": source_run_id == 2,
        "planning_only": True,
        "no_replay_performed": True,
        "no_w3_execution_performed": True,
        "coverage_driven_higher_target_logic": True,
        "thirty_deg_uncovered_does_not_auto_recommend_45_60": True,
        "max_w3_candidates": int(max_w3_candidates),
        "selection_statuses": SELECTION_STATUSES,
        "coverage_decisions": COVERAGE_DECISIONS,
        "higher_target_request_statuses": HIGHER_TARGET_REQUEST_STATUSES,
        "future_target_ladder_deg": FUTURE_TARGETS_DEG,
        "candidate_shortlist_rows": int(len(shortlist)),
        "coverage_decision_rows": int(len(coverage)),
        "w3_planned_candidate_count": int(len(w3_plan)),
        **_w3_diversity_manifest_fields(w3_plan),
        "higher_target_recommended_next_count": int((higher_targets["request_status"] == "recommended_next").sum()),
        "request_status_counts": higher_targets["request_status"].value_counts(dropna=False).to_dict(),
        "selection_status_counts": shortlist["selection_status"].value_counts(dropna=False).to_dict(),
        "coverage_decision_counts": coverage["coverage_decision_s003"].value_counts(dropna=False).to_dict(),
        "source_manifest_summary": {
            "central_research_question": source_manifest.get("central_research_question"),
            "w1_complete": source_manifest.get("w1_complete"),
            "w2_complete": source_manifest.get("w2_complete"),
            "archived_boundary_reference_preserved": source_manifest.get("archived_boundary_reference_preserved"),
        },
        "output_files": {key: _repo_relative(value) for key, value in output_files.items()},
        "w3_stress_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "real_flight_validation_claim": False,
        "tvlqr_implemented": False,
        "ocp_implemented": False,
        "high_incidence_validation_claim": False,
        "old_perch_like_branch_active": False,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest


def _write_report(
    path: Path,
    *,
    source_run_id: int,
    shortlist: pd.DataFrame,
    coverage: pd.DataFrame,
    w3_plan: pd.DataFrame,
    higher_targets: pd.DataFrame,
) -> None:
    diversity = _w3_diversity_manifest_fields(w3_plan)
    lines = [
        "# Primitive Library Shortlist and W3 Planning Report",
        "",
        f"This planning-only pass reads run `s{source_run_id:03d}` evidence and writes a derived run `s003` shortlist.",
        "It does not replay candidates, execute W3, implement a governor, implement OCP/TVLQR, or make a real-flight claim.",
        "",
        "## Shortlist Summary",
        "",
    ]
    for status, count in shortlist["selection_status"].value_counts(dropna=False).sort_index().items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Coverage Decisions", ""])
    for decision, count in coverage["coverage_decision_s003"].value_counts(dropna=False).sort_index().items():
        lines.append(f"- `{decision}`: `{count}`")
    lines.extend(
        [
            "",
            f"- W3 stress candidates planned: `{len(w3_plan)}`",
            "- W3 rows are planning rows only; `not_implemented_in_this_pass=True`.",
            "",
            "## W3 Diversity Check",
            "",
            f"- Selected W3 roles: `{diversity['w3_roles_selected']}`",
            f"- Missing W3 roles: `{diversity['w3_missing_roles']}`",
            f"- Distinct families selected: `{diversity['w3_distinct_family_count']}`",
            f"- Target-labelled steering candidate included: `{diversity['w3_target_candidate_present']}`",
            f"- Baseline glide/recovery/mild-bank represented: `{diversity['w3_baseline_candidate_present']}`",
            f"- Glide count: `{diversity['w3_glide_count']}`",
            f"- Recovery count: `{diversity['w3_recovery_count']}`",
            f"- Mild-bank count: `{diversity['w3_mild_bank_count']}`",
            f"- Target-steering count: `{diversity['w3_target_steering_count']}`",
            "- W3 is still not executed in this pass.",
            "",
            "## Higher-Target Requests",
            "",
        ]
    )
    for _, row in higher_targets.iterrows():
        lines.append(
            f"- `{row['requested_target_deg']:.0f} deg`: `{row['request_status']}` - {row['reason']}"
        )
    lines.extend(
        [
            "",
            "The higher-target table is coverage-driven. A failed or uncovered 30 deg row is",
            "not enough by itself to request 45/60 deg. Boundary or entry-envelope",
            "limitations defer higher-target work unless a separate mission-critical",
            "coverage row proves the larger heading is required and plausible.",
            "",
            "## Unimplemented Scope",
            "",
            "- W3 selected stress: `False`",
            "- Governor: `False`",
            "- Outer loop: `False`",
            "- OCP/TVLQR: `False`",
            "- Real-flight validation: `False`",
            "- High-incidence validation claim: `False`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _w3_diversity_manifest_fields(w3_plan: pd.DataFrame) -> dict[str, object]:
    if w3_plan.empty:
        roles_selected: list[str] = []
        families = pd.Series(dtype=object)
    else:
        roles_selected = [str(value) for value in w3_plan["w3_role"].dropna().tolist()]
        families = w3_plan["family"].dropna()
    missing_roles = [role for role in W3_REQUIRED_ROLES if role not in roles_selected]
    family_counts = families.value_counts(dropna=False).to_dict()
    return {
        "w3_diversity_selection_fixed": True,
        "w3_required_roles": list(W3_REQUIRED_ROLES),
        "w3_roles_selected": roles_selected,
        "w3_missing_roles": missing_roles,
        "w3_distinct_family_count": int(families.nunique()),
        "w3_target_candidate_present": bool((w3_plan["target_heading_deg"].notna()).any()) if not w3_plan.empty else False,
        "w3_baseline_candidate_present": bool(families.isin(["glide", "recovery", "mild_bank"]).any()) if not w3_plan.empty else False,
        "w3_glide_count": int(family_counts.get("glide", 0)),
        "w3_recovery_count": int(family_counts.get("recovery", 0)),
        "w3_mild_bank_count": int(family_counts.get("mild_bank", 0)),
        "w3_target_steering_count": int(roles_selected.count("target_steering")),
    }


# =============================================================================
# 2) Shortlist Pass
# =============================================================================
def run_primitive_library_shortlist(
    *,
    source_run_id: int = 2,
    run_id: int = 3,
    overwrite: bool = False,
    max_w3_candidates: int = 5,
) -> dict[str, Path]:
    """Read run-002 evidence and write the run-003 planning shortlist."""

    paths = _prepare_result_tree(run_id, overwrite)
    suffix = f"s{run_id:03d}"
    source_root = RESULT_ROOT / f"{source_run_id:03d}"
    source = load_source_evidence(source_root, source_run_id=source_run_id)

    shortlist = build_candidate_shortlist(source.evidence)
    coverage = build_coverage_decision_summary(source.coverage_region_summary, shortlist)
    w3_plan = build_w3_stress_plan(shortlist, max_w3_candidates=max_w3_candidates)
    higher_targets = build_higher_target_growth_request(coverage)

    shortlist_csv = paths["metrics"] / f"candidate_shortlist_{suffix}.csv"
    coverage_csv = paths["metrics"] / f"coverage_decision_summary_{suffix}.csv"
    w3_csv = paths["metrics"] / f"w3_stress_plan_{suffix}.csv"
    higher_csv = paths["metrics"] / f"higher_target_growth_request_{suffix}.csv"
    report_md = paths["reports"] / f"primitive_library_shortlist_report_{suffix}.md"
    manifest_json = paths["manifests"] / f"primitive_library_shortlist_manifest_{suffix}.json"

    shortlist.to_csv(shortlist_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    w3_plan.to_csv(w3_csv, index=False)
    higher_targets.to_csv(higher_csv, index=False)
    _write_report(
        report_md,
        source_run_id=source_run_id,
        shortlist=shortlist,
        coverage=coverage,
        w3_plan=w3_plan,
        higher_targets=higher_targets,
    )
    output_files = {
        "candidate_shortlist_csv": shortlist_csv,
        "coverage_decision_summary_csv": coverage_csv,
        "w3_stress_plan_csv": w3_csv,
        "higher_target_growth_request_csv": higher_csv,
        "report_md": report_md,
        "manifest_json": manifest_json,
    }
    _write_manifest(
        manifest_json,
        run_id=run_id,
        source_run_id=source_run_id,
        output_files=output_files,
        source_manifest=source.manifest,
        shortlist=shortlist,
        coverage=coverage,
        w3_plan=w3_plan,
        higher_targets=higher_targets,
        max_w3_candidates=max_w3_candidates,
    )
    return {
        "root": paths["root"],
        "manifest": manifest_json,
        "candidate_shortlist_csv": shortlist_csv,
        "coverage_decision_summary_csv": coverage_csv,
        "w3_stress_plan_csv": w3_csv,
        "higher_target_growth_request_csv": higher_csv,
        "report": report_md,
    }


# =============================================================================
# 3) CLI Entry Point
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", type=int, default=2)
    parser.add_argument("--run-id", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-w3-candidates", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_primitive_library_shortlist(
        source_run_id=int(args.source_run_id),
        run_id=int(args.run_id),
        overwrite=bool(args.overwrite),
        max_w3_candidates=int(args.max_w3_candidates),
    )
    for key in (
        "root",
        "manifest",
        "candidate_shortlist_csv",
        "coverage_decision_summary_csv",
        "w3_stress_plan_csv",
        "higher_target_growth_request_csv",
        "report",
    ):
        print(f"{key}={paths[key]}")


if __name__ == "__main__":
    main()
