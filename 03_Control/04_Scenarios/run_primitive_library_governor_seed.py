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

from primitive_library_governor import (
    GOVERNOR_DECISION_STATUSES,
    build_governor_coverage_update,
    build_governor_decision_cases,
    build_governor_rejection_summary,
    build_governor_seed_table,
    evaluate_governor_cases,
    load_governor_sources,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and output helpers
# 2) Run-005 governor seed workflow
# 3) CLI entry point
# =============================================================================


# =============================================================================
# 1) Paths and Output Helpers
# =============================================================================
CAMPAIGN = "09_primitive_library"
PASS_NAME = "primitive_library_governor_seed"
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
        "metrics": root / "metrics",
        "manifests": root / "manifests",
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
    source_w3_run_id: int,
    source_plan_run_id: int,
    source_evidence_run_id: int,
    output_files: dict[str, Path],
    seed_table: pd.DataFrame,
    cases: pd.DataFrame,
    decisions: pd.DataFrame,
    rejection_summary: pd.DataFrame,
    coverage_update: pd.DataFrame,
) -> dict[str, object]:
    suffix = f"s{run_id:03d}"
    accepted_seed_count = int(seed_table["governor_seed_candidate"].astype(bool).sum())
    excluded_target_count = int((seed_table["seed_table_status"] == "excluded_marginal_target_steering").sum())
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": suffix,
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "source_w3_run": f"s{source_w3_run_id:03d}",
        "source_w3_plan_run": f"s{source_plan_run_id:03d}",
        "source_evidence_run": f"s{source_evidence_run_id:03d}",
        "run_004_only_w3_stress_source": True,
        "run_003_w3_plan_source": True,
        "run_002_only_physical_evidence_source": True,
        "no_dynamics_replay_performed": True,
        "governor_seed_implemented": True,
        "governor_query_implemented": True,
        "governor_implemented": True,
        "numerical_clearance_contract_implemented": True,
        "clearance_fields_from_run_002": True,
        "clearance_check_case_label_independent": True,
        "governor_online_flight_ready": False,
        "outer_loop_implemented": False,
        "real_flight_validation_claim": False,
        "hardware_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "high_incidence_validation_claim": False,
        "target_steering_governor_allowed": False,
        "target_steering_next_action": "refine_seed_before_governor",
        "accepted_seed_candidate_count": accepted_seed_count,
        "excluded_target_steering_count": excluded_target_count,
        "seed_table_rows": int(len(seed_table)),
        "decision_case_count": int(len(cases)),
        "accepted_decision_count": int(decisions["accepted"].astype(bool).sum()),
        "rejected_decision_count": int((~decisions["accepted"].astype(bool)).sum()),
        "decision_status_counts": decisions["governor_decision_status"].value_counts(dropna=False).to_dict(),
        "rejection_summary_rows": int(len(rejection_summary)),
        "coverage_status_counts": coverage_update["governor_coverage_status_s005"].value_counts(dropna=False).to_dict(),
        "governor_decision_statuses": list(GOVERNOR_DECISION_STATUSES),
        "expected_interpretation": (
            "four W3-supported baseline/updraft-support primitives seed the offline governor; "
            "the 15 deg target-steering candidate remains excluded until refined"
        ),
        "output_files": {key: _repo_relative(value) for key, value in output_files.items()},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest


def _write_report(
    path: Path,
    *,
    seed_table: pd.DataFrame,
    cases: pd.DataFrame,
    decisions: pd.DataFrame,
    rejection_summary: pd.DataFrame,
    coverage_update: pd.DataFrame,
) -> None:
    accepted_seeds = seed_table[seed_table["governor_seed_candidate"].astype(bool)]
    target = seed_table[seed_table["seed_table_status"] == "excluded_marginal_target_steering"]
    lines = [
        "# Primitive Library Governor Seed Report",
        "",
        "This run converts selected W3 stress evidence into an offline governor accept/reject seed layer.",
        "It does not replay dynamics, implement an outer loop, implement OCP/TVLQR, touch hardware, or claim real-flight readiness.",
        "Clearance is checked numerically as available margin minus the primitive-specific requirement copied from run-002 evidence; case labels do not force clearance rejection.",
        "",
        "## Accepted Governor Seeds",
        "",
    ]
    for _, row in accepted_seeds.iterrows():
        lines.append(
            "- "
            f"`{row['source_primitive_id']}`: role `{row['w3_role']}`, "
            f"status `{row['candidate_w3_status']}`, updraft `{row['updraft_config']}`, "
            f"wind `{row['wind_fidelity']}`"
        )
    lines.extend(
        [
            "",
            f"Accepted seed candidate count: `{len(accepted_seeds)}`",
            "",
            "## Excluded Target Steering",
            "",
        ]
    )
    if not target.empty:
        row = target.iloc[0]
        lines.append(
            f"- `{row['source_primitive_id']}` remains `{row['seed_table_status']}` "
            f"with W3 status `{row['candidate_w3_status']}` and next action "
            f"`{row['candidate_w3_recommendation']}`."
        )
    lines.extend(
        [
            "",
            "The target-steering candidate is not governor-allowed in run-005.",
            "",
            "## Decision Cases",
            "",
            f"- Decision cases: `{len(cases)}`",
            f"- Accepted decisions: `{int(decisions['accepted'].astype(bool).sum())}`",
            f"- Rejected decisions: `{int((~decisions['accepted'].astype(bool)).sum())}`",
            "",
            "| decision_status | case_count | accepted_count | rejected_count |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in rejection_summary.iterrows():
        if int(row["case_count"]) > 0:
            lines.append(
                "| "
                f"{row['governor_decision_status']} | {int(row['case_count'])} | "
                f"{int(row['accepted_count'])} | {int(row['rejected_count'])} |"
            )
    lines.extend(["", "## Coverage Update", ""])
    for _, row in coverage_update.iterrows():
        lines.append(
            "- "
            f"`{row['source_primitive_id']}` -> `{row['governor_coverage_status_s005']}` "
            f"({row['recommended_next_step_s005']})"
        )
    lines.extend(
        [
            "",
            "## No-Overclaiming Statement",
            "",
            "- Offline governor seed/query layer implemented: `true`",
            "- Online flight-ready governor: `false`",
            "- Outer-loop mission simulation: `false`",
            "- OCP/TVLQR: `false`",
            "- Hardware or real-flight validation claim: `false`",
            "- High-incidence validation claim: `false`",
            "",
            "## Next Step",
            "",
            "Use the accepted seed table as the input contract for a later outer-loop governor simulation.",
            "Target steering must be refined before it can be considered by that governor.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 2) Run-005 Governor Seed Workflow
# =============================================================================
def run_primitive_library_governor_seed(
    *,
    source_w3_run_id: int = 4,
    source_plan_run_id: int = 3,
    source_evidence_run_id: int = 2,
    run_id: int = 5,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Build the run-005 offline governor seed evidence."""

    paths = _prepare_result_tree(run_id, overwrite)
    suffix = f"s{run_id:03d}"
    sources = load_governor_sources(
        RESULT_ROOT,
        source_w3_run_id=source_w3_run_id,
        source_plan_run_id=source_plan_run_id,
        source_evidence_run_id=source_evidence_run_id,
    )
    seed_table = build_governor_seed_table(
        sources["candidate_summary"],
        sources["coverage_update"],
        sources["manifest"],
        sources["source_evidence"],
    )
    cases = build_governor_decision_cases(seed_table, source_evidence=sources["source_evidence"])
    decisions = evaluate_governor_cases(cases, seed_table)
    rejection_summary = build_governor_rejection_summary(decisions)
    coverage_update = build_governor_coverage_update(seed_table, decisions)

    seed_csv = paths["metrics"] / f"governor_seed_candidate_table_{suffix}.csv"
    cases_csv = paths["metrics"] / f"governor_decision_cases_{suffix}.csv"
    decisions_csv = paths["metrics"] / f"governor_accept_reject_decisions_{suffix}.csv"
    rejection_csv = paths["metrics"] / f"governor_rejection_summary_{suffix}.csv"
    coverage_csv = paths["metrics"] / f"governor_coverage_update_{suffix}.csv"
    manifest_json = paths["manifests"] / f"governor_seed_manifest_{suffix}.json"
    report_md = paths["reports"] / f"governor_seed_report_{suffix}.md"

    seed_table.to_csv(seed_csv, index=False)
    cases.to_csv(cases_csv, index=False)
    decisions.to_csv(decisions_csv, index=False)
    rejection_summary.to_csv(rejection_csv, index=False)
    coverage_update.to_csv(coverage_csv, index=False)

    output_files = {
        "governor_seed_candidate_table_csv": seed_csv,
        "governor_decision_cases_csv": cases_csv,
        "governor_accept_reject_decisions_csv": decisions_csv,
        "governor_rejection_summary_csv": rejection_csv,
        "governor_coverage_update_csv": coverage_csv,
        "manifest_json": manifest_json,
        "report_md": report_md,
    }
    _write_report(
        report_md,
        seed_table=seed_table,
        cases=cases,
        decisions=decisions,
        rejection_summary=rejection_summary,
        coverage_update=coverage_update,
    )
    _write_manifest(
        manifest_json,
        run_id=run_id,
        source_w3_run_id=source_w3_run_id,
        source_plan_run_id=source_plan_run_id,
        source_evidence_run_id=source_evidence_run_id,
        output_files=output_files,
        seed_table=seed_table,
        cases=cases,
        decisions=decisions,
        rejection_summary=rejection_summary,
        coverage_update=coverage_update,
    )
    return {
        "root": paths["root"],
        "seed_table_csv": seed_csv,
        "decision_cases_csv": cases_csv,
        "decisions_csv": decisions_csv,
        "rejection_summary_csv": rejection_csv,
        "coverage_update_csv": coverage_csv,
        "manifest": manifest_json,
        "report": report_md,
    }


# =============================================================================
# 3) CLI Entry Point
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-w3-run-id", type=int, default=4)
    parser.add_argument("--source-plan-run-id", type=int, default=3)
    parser.add_argument("--source-evidence-run-id", type=int, default=2)
    parser.add_argument("--run-id", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_primitive_library_governor_seed(
        source_w3_run_id=int(args.source_w3_run_id),
        source_plan_run_id=int(args.source_plan_run_id),
        source_evidence_run_id=int(args.source_evidence_run_id),
        run_id=int(args.run_id),
        overwrite=bool(args.overwrite),
    )
    for key in (
        "root",
        "manifest",
        "seed_table_csv",
        "decision_cases_csv",
        "decisions_csv",
        "rejection_summary_csv",
        "coverage_update_csv",
        "report",
    ):
        print(f"{key}={paths[key]}")


if __name__ == "__main__":
    main()
