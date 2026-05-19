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

from primitive_library_outer_loop import (
    CAMPAIGN,
    OUTER_LOOP_SCENARIOS,
    OuterLoopConfig,
    build_outer_loop_scenarios,
    load_outer_loop_sources,
    run_outer_loop_missions,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Output Helpers
# 2) Run-006 Workflow
# 3) CLI Entry Point
# =============================================================================


# =============================================================================
# 1) Paths and Output Helpers
# =============================================================================
PASS_NAME = "outer_loop_mission_simulation"
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
        "logs": root / "logs",
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
    source_governor_run_id: int,
    sources: dict[str, object],
    outputs: dict[str, Path],
    results: dict[str, pd.DataFrame],
    scenarios: tuple[object, ...],
) -> dict[str, object]:
    suffix = f"s{run_id:03d}"
    source_manifest = sources["manifest"]
    summary = results["mission_summary"]
    candidate_log = results["candidate_log"]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": suffix,
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "source_governor_run": f"s{source_governor_run_id:03d}",
        "source_w3_run": source_manifest.get("source_w3_run", "s004"),
        "source_plan_run": source_manifest.get("source_w3_plan_run", "s003"),
        "source_evidence_run": source_manifest.get("source_evidence_run", "s002"),
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "outer_loop_scenarios": list(OUTER_LOOP_SCENARIOS),
        "outer_loop_implemented": True,
        "outer_loop_mission_simulation_implemented": True,
        "governor_implemented": True,
        "governor_online_flight_ready": False,
        "real_flight_validation_claim": False,
        "hardware_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "high_incidence_validation_claim": False,
        "target_steering_used": False,
        "target_steering_governor_allowed": False,
        "higher_target_primitives_added": False,
        "new_primitive_generated": False,
        "accepted_seed_candidate_count": 4,
        "active_seed_candidate_count": int(len(sources["active_seed_table"])),
        "excluded_target_steering_count": int(len(sources["target_steering_table"])),
        "mission_summary_rows": int(len(summary)),
        "candidate_query_count": int(len(candidate_log)),
        "candidate_accept_count": int(candidate_log["accepted"].astype(bool).sum()) if not candidate_log.empty else 0,
        "candidate_reject_count": int((~candidate_log["accepted"].astype(bool)).sum()) if not candidate_log.empty else 0,
        "mission_success_counts": summary["mission_success_label"].value_counts(dropna=False).to_dict()
        if not summary.empty
        else {},
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad -> rk4_step/state_derivative",
        "output_files": {key: _repo_relative(value) for key, value in outputs.items()},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest


def _write_report(path: Path, results: dict[str, pd.DataFrame], sources: dict[str, object]) -> None:
    seeds = sources["active_seed_table"]
    target = sources["target_steering_table"]
    summary = results["mission_summary"]
    gaps = results["coverage_gap_summary"]
    lift = results["lift_dwell_summary"]
    lines = [
        "# Outer-Loop Mission Simulation Report",
        "",
        "Run-006 uses the run-005 offline governor seed layer to select or reject existing primitive seeds.",
        "It does not add primitives, implement OCP/TVLQR, touch hardware, or claim real-flight readiness.",
        "",
        "## Governor Seeds",
        "",
    ]
    for _, row in seeds.iterrows():
        lines.append(
            "- "
            f"`{row['source_primitive_id']}`: role `{row['w3_role']}`, "
            f"updraft `{row['updraft_config']}`, wind `{row['wind_fidelity']}`"
        )
    lines.extend(["", "## Target Steering", ""])
    if not target.empty:
        row = target.iloc[0]
        lines.append(
            f"`{row['source_primitive_id']}` remains excluded with status "
            f"`{row['seed_table_status']}` and is not selectable in run-006."
        )
    lines.extend(["", "## Mission Outcomes", ""])
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"`{row['scenario_id']}`: `{row['mission_success_label']}`, "
            f"stop `{row['mission_stop_reason']}`, accepted `{int(row['steps_accepted'])}` steps, "
            f"energy delta `{float(row['energy_delta_m']):.3f} m`"
        )
    lines.extend(["", "## Lift Dwell", ""])
    for _, row in lift.iterrows():
        lines.append(
            "- "
            f"`{row['scenario_id']}`: dwell fraction `{float(row['lift_region_fraction']):.3f}`, "
            f"mean energy residual `{float(row['mean_energy_residual_m']):.3f} m`"
        )
    lines.extend(["", "## Coverage Gaps", ""])
    for _, row in gaps.iterrows():
        lines.append(
            "- "
            f"`{row['scenario_id']}`: `{row['coverage_gap_type']}`, "
            f"next `{row['recommended_next_action']}`, higher target `{row['higher_target_request_status']}`"
        )
    lines.extend(
        [
            "",
            "## No-Overclaiming",
            "",
            "- Outer-loop mission simulation implemented: `true`",
            "- Online flight-ready governor: `false`",
            "- Target steering used: `false`",
            "- Higher-target primitives added: `false`",
            "- Hardware or real-flight validation claim: `false`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 2) Run-006 Workflow
# =============================================================================
def run_primitive_library_outer_loop(
    *,
    source_governor_run_id: int = 5,
    run_id: int = 6,
    overwrite: bool = False,
    max_steps: int = 8,
    dt_s: float = 0.02,
    scenario: str = "all",
    write_step_logs: str = "selected",
) -> dict[str, Path]:
    paths = _prepare_result_tree(run_id, overwrite)
    suffix = f"s{run_id:03d}"
    sources = load_outer_loop_sources(RESULT_ROOT, governor_run_id=source_governor_run_id)
    scenarios = _filter_scenarios(build_outer_loop_scenarios(max_steps=max_steps), scenario)
    config = OuterLoopConfig(dt_s=float(dt_s), max_steps=int(max_steps), write_step_logs=write_step_logs)
    results = run_outer_loop_missions(scenarios, sources, config)

    outputs = {
        "mission_summary_csv": paths["metrics"] / f"outer_loop_mission_summary_{suffix}.csv",
        "step_log_csv": paths["metrics"] / f"outer_loop_step_log_{suffix}.csv",
        "candidate_decision_log_csv": paths["metrics"] / f"outer_loop_candidate_decision_log_{suffix}.csv",
        "governor_rejection_log_csv": paths["metrics"] / f"outer_loop_governor_rejection_log_{suffix}.csv",
        "energy_trace_csv": paths["metrics"] / f"outer_loop_energy_trace_{suffix}.csv",
        "lift_dwell_summary_csv": paths["metrics"] / f"outer_loop_lift_dwell_summary_{suffix}.csv",
        "coverage_gap_summary_csv": paths["metrics"] / f"outer_loop_coverage_gap_summary_{suffix}.csv",
        "manifest_json": paths["manifests"] / f"outer_loop_mission_manifest_{suffix}.json",
        "report_md": paths["reports"] / f"outer_loop_mission_report_{suffix}.md",
    }
    if write_step_logs == "selected":
        outputs["selected_step_log_csv"] = paths["logs"] / f"outer_loop_selected_step_logs_{suffix}.csv"

    results["mission_summary"].to_csv(outputs["mission_summary_csv"], index=False)
    results["step_log"].to_csv(outputs["step_log_csv"], index=False)
    results["candidate_log"].to_csv(outputs["candidate_decision_log_csv"], index=False)
    results["governor_rejection_log"].to_csv(outputs["governor_rejection_log_csv"], index=False)
    results["energy_trace"].to_csv(outputs["energy_trace_csv"], index=False)
    results["lift_dwell_summary"].to_csv(outputs["lift_dwell_summary_csv"], index=False)
    results["coverage_gap_summary"].to_csv(outputs["coverage_gap_summary_csv"], index=False)
    if "selected_step_log_csv" in outputs:
        results["selected_logs"].to_csv(outputs["selected_step_log_csv"], index=False)

    _write_report(outputs["report_md"], results, sources)
    _write_manifest(
        outputs["manifest_json"],
        run_id=run_id,
        source_governor_run_id=source_governor_run_id,
        sources=sources,
        outputs=outputs,
        results=results,
        scenarios=scenarios,
    )
    return {"root": paths["root"], **outputs}


def _filter_scenarios(scenarios: tuple[object, ...], scenario: str) -> tuple[object, ...]:
    if scenario == "all":
        return scenarios
    if scenario == "U1":
        return tuple(item for item in scenarios if item.scenario_id.startswith("U1_"))
    if scenario == "U4":
        return tuple(item for item in scenarios if item.scenario_id.startswith("U4_"))
    matches = tuple(item for item in scenarios if item.scenario_id == scenario)
    if not matches:
        raise ValueError(f"unknown outer-loop scenario selector: {scenario}")
    return matches


# =============================================================================
# 3) CLI Entry Point
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-governor-run-id", type=int, default=5)
    parser.add_argument("--run-id", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--scenario", choices=("all", "U1", "U4"), default="all")
    parser.add_argument("--write-step-logs", choices=("selected", "none"), default="selected")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_primitive_library_outer_loop(
        source_governor_run_id=int(args.source_governor_run_id),
        run_id=int(args.run_id),
        overwrite=bool(args.overwrite),
        max_steps=int(args.max_steps),
        dt_s=float(args.dt_s),
        scenario=str(args.scenario),
        write_step_logs=str(args.write_step_logs),
    )
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
