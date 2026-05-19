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
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from primitive_library_schema import PrimitiveLibraryConfig
from primitive_library_w3_stress import (
    ALLOWED_CANDIDATE_W3_STATUSES,
    DEFAULT_RANDOM_SEED,
    W3TrialLog,
    build_w3_trial_table,
    load_source_evidence,
    load_w3_plan,
    run_w3_stress_trial_table,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and output helpers
# 2) Run-004 workflow
# 3) CLI entry point
# =============================================================================


# =============================================================================
# 1) Paths and Output Helpers
# =============================================================================
CAMPAIGN = "09_primitive_library"
PASS_NAME = "selected_w3_stress"
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
        "logs": root / "logs" / "candidates",
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
    evidence_run_id: int,
    random_seed: int,
    output_files: dict[str, Path],
    w3_plan: pd.DataFrame,
    trial_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    coverage_update: pd.DataFrame,
) -> dict[str, object]:
    suffix = f"s{run_id:03d}"
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": suffix,
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "source_w3_plan_run": f"s{source_run_id:03d}",
        "evidence_source_run": f"s{evidence_run_id:03d}",
        "run_003_w3_plan_only_shortlist_source": True,
        "run_002_only_physical_evidence_source": True,
        "exact_selected_candidate_count": int(len(w3_plan)),
        "random_seed": int(random_seed),
        "candidate_count": int(len(candidate_summary)),
        "trial_count": int(len(trial_summary)),
        "trial_success_count": int(trial_summary["trial_success"].astype(bool).sum()),
        "candidate_status_counts": candidate_summary["candidate_w3_status"].value_counts(dropna=False).to_dict(),
        "coverage_status_counts": coverage_update["coverage_status_s004"].value_counts(dropna=False).to_dict(),
        "selected_w3_roles": [str(value) for value in w3_plan["w3_role"].tolist()],
        "selected_source_primitive_ids": [str(value) for value in w3_plan["source_primitive_id"].tolist()],
        "allowed_candidate_w3_statuses": list(ALLOWED_CANDIDATE_W3_STATUSES),
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad -> rk4_step/state_derivative",
        "w3_stress_implemented": True,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "real_flight_validation_claim": False,
        "tvlqr_implemented": False,
        "ocp_implemented": False,
        "high_incidence_validation_claim": False,
        "hardware_implemented": False,
        "old_perch_like_branch_active": False,
        "output_files": {key: _repo_relative(value) for key, value in output_files.items()},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest


def _write_report(
    path: Path,
    *,
    w3_plan: pd.DataFrame,
    trial_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    coverage_update: pd.DataFrame,
) -> None:
    lines = [
        "# Selected W3 Stress Report",
        "",
        "This run executes selected W3 simulation stress for the five run-003 primitive-library candidates.",
        "It is not a governor, outer-loop, OCP, TVLQR, hardware, real-flight, or high-incidence validation pass.",
        "",
        "## Source W3 Plan",
        "",
        f"- Selected candidates: `{len(w3_plan)}`",
        f"- Trial rows: `{len(trial_summary)}`",
        f"- Trial successes: `{int(trial_summary['trial_success'].astype(bool).sum())}`",
        "",
        "## Candidate-Level W3 Results",
        "",
        "| source_primitive_id | role | status | success_fraction | recommendation |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in candidate_summary.iterrows():
        lines.append(
            "| "
            f"{row['source_primitive_id']} | {row['w3_role']} | {row['candidate_w3_status']} | "
            f"{float(row['trial_success_fraction']):.3f} | {row['candidate_w3_recommendation']} |"
        )
    lines.extend(["", "## Coverage Update", ""])
    for _, row in coverage_update.iterrows():
        lines.append(
            "- "
            f"`{row['source_primitive_id']}`: `{row['coverage_status_s004']}` -> "
            f"`{row['recommended_next_step']}`"
        )
    lines.extend(["", "## Dominant Failure Mechanisms", ""])
    for mechanism, count in trial_summary["active_limiting_mechanism"].value_counts(dropna=False).sort_index().items():
        lines.append(f"- `{mechanism}`: `{count}`")
    lines.extend(
        [
            "",
            "## Recommended Next Step Toward Governor",
            "",
            f"- `{_next_recommended_step(candidate_summary)}`",
            "",
            "## No-Overclaiming Statement",
            "",
            "- W3 stress simulation evidence: `true`",
            "- Governor implemented: `false`",
            "- Outer-loop implemented: `false`",
            "- OCP/TVLQR implemented: `false`",
            "- Hardware or real-flight validation claim: `false`",
            "- High-incidence validation claim: `false`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_selected_logs(
    log_dir: Path,
    suffix: str,
    logs: list[W3TrialLog],
    mode: str,
) -> dict[str, Path]:
    if mode == "none":
        return {}
    selected = logs if mode == "all" else _selected_logs(logs)
    output_files: dict[str, Path] = {}
    for log in selected:
        source_id = str(log.summary["source_primitive_id"])
        seed = int(log.summary["stress_seed_index"])
        stem = f"{source_id}_seed{seed:03d}_{suffix}"
        trajectory_csv = log_dir / f"{stem}_trajectory.csv"
        command_csv = log_dir / f"{stem}_commands.csv"
        log.trajectory_dataframe().to_csv(trajectory_csv, index=False)
        log.command_dataframe().to_csv(command_csv, index=False)
        output_files[f"{stem}_trajectory_csv"] = trajectory_csv
        output_files[f"{stem}_commands_csv"] = command_csv
    return output_files


def _selected_logs(logs: list[W3TrialLog]) -> list[W3TrialLog]:
    selected: dict[tuple[str, int], W3TrialLog] = {}
    by_source: dict[str, list[W3TrialLog]] = {}
    for log in logs:
        by_source.setdefault(str(log.summary["source_primitive_id"]), []).append(log)
    for source_id, group in by_source.items():
        nominal = [log for log in group if int(log.summary["stress_seed_index"]) == 0][0]
        selected[(source_id, int(nominal.summary["stress_seed_index"]))] = nominal
        successes = [log for log in group if bool(log.summary["trial_success"])]
        if successes:
            best = sorted(
                successes,
                key=lambda item: (
                    -float(item.summary["min_true_margin_m"]),
                    int(item.summary["stress_seed_index"]),
                ),
            )[0]
            selected[(source_id, int(best.summary["stress_seed_index"]))] = best
        finite_failures = [
            log
            for log in group
            if bool(log.summary["finite_replay"]) and not bool(log.summary["trial_success"])
        ]
        if finite_failures:
            worst = sorted(
                finite_failures,
                key=lambda item: (
                    float(item.summary["min_true_margin_m"]),
                    int(item.summary["stress_seed_index"]),
                ),
            )[0]
            selected[(source_id, int(worst.summary["stress_seed_index"]))] = worst
    return [selected[key] for key in sorted(selected)]


def _next_recommended_step(candidate_summary: pd.DataFrame) -> str:
    statuses = set(candidate_summary["candidate_w3_status"].astype(str))
    if "w3_supported" in statuses:
        return "proceed_to_governor_seed"
    if "w3_marginal" in statuses:
        return "refine_seed_before_governor"
    if not candidate_summary.empty and candidate_summary["trial_count"].max() < 25:
        return "repeat_selected_w3_with_larger_seed_count"
    return "stop_and_keep_boundary_evidence"


# =============================================================================
# 2) Run-004 Workflow
# =============================================================================
def run_primitive_library_w3_stress(
    *,
    source_run_id: int = 3,
    evidence_run_id: int = 2,
    run_id: int = 4,
    overwrite: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_seeds_per_candidate: int | None = None,
    write_trial_logs: str = "selected",
) -> dict[str, Path]:
    """Run selected W3 stress trials and write run-004 evidence."""

    paths = _prepare_result_tree(run_id, overwrite)
    suffix = f"s{run_id:03d}"
    w3_plan = load_w3_plan(RESULT_ROOT, source_run_id=source_run_id)
    _, source_evidence = load_source_evidence(RESULT_ROOT, evidence_run_id=evidence_run_id)
    trial_table = build_w3_trial_table(
        w3_plan,
        seeds_per_candidate=max_seeds_per_candidate,
        random_seed=random_seed,
    )
    config = PrimitiveLibraryConfig(run_id=run_id)
    trial_summary, candidate_summary, coverage_update, logs = run_w3_stress_trial_table(
        trial_table,
        source_evidence,
        config,
    )

    trial_csv = paths["metrics"] / f"w3_stress_trial_summary_{suffix}.csv"
    candidate_csv = paths["metrics"] / f"w3_stress_candidate_summary_{suffix}.csv"
    coverage_csv = paths["metrics"] / f"w3_stress_coverage_update_{suffix}.csv"
    report_md = paths["reports"] / f"w3_stress_report_{suffix}.md"
    manifest_json = paths["manifests"] / f"w3_stress_manifest_{suffix}.json"
    trial_summary.to_csv(trial_csv, index=False)
    candidate_summary.to_csv(candidate_csv, index=False)
    coverage_update.to_csv(coverage_csv, index=False)

    log_files = _write_selected_logs(paths["logs"], suffix, logs, mode=write_trial_logs)
    output_files = {
        "trial_summary_csv": trial_csv,
        "candidate_summary_csv": candidate_csv,
        "coverage_update_csv": coverage_csv,
        "report_md": report_md,
        "manifest_json": manifest_json,
        **log_files,
    }
    _write_report(
        report_md,
        w3_plan=w3_plan,
        trial_summary=trial_summary,
        candidate_summary=candidate_summary,
        coverage_update=coverage_update,
    )
    _write_manifest(
        manifest_json,
        run_id=run_id,
        source_run_id=source_run_id,
        evidence_run_id=evidence_run_id,
        random_seed=random_seed,
        output_files=output_files,
        w3_plan=w3_plan,
        trial_summary=trial_summary,
        candidate_summary=candidate_summary,
        coverage_update=coverage_update,
    )
    return {
        "root": paths["root"],
        "trial_summary_csv": trial_csv,
        "candidate_summary_csv": candidate_csv,
        "coverage_update_csv": coverage_csv,
        "manifest": manifest_json,
        "report": report_md,
    }


# =============================================================================
# 3) CLI Entry Point
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", type=int, default=3)
    parser.add_argument("--evidence-run-id", type=int, default=2)
    parser.add_argument("--run-id", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-seeds-per-candidate", type=int, default=None)
    parser.add_argument("--write-trial-logs", choices=("all", "selected", "none"), default="selected")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_primitive_library_w3_stress(
        source_run_id=int(args.source_run_id),
        evidence_run_id=int(args.evidence_run_id),
        run_id=int(args.run_id),
        overwrite=bool(args.overwrite),
        random_seed=int(args.random_seed),
        max_seeds_per_candidate=args.max_seeds_per_candidate,
        write_trial_logs=str(args.write_trial_logs),
    )
    for key in ("root", "manifest", "trial_summary_csv", "candidate_summary_csv", "coverage_update_csv", "report"):
        print(f"{key}={paths[key]}")


if __name__ == "__main__":
    main()
