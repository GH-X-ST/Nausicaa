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
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agile_turn_family_comparison import (
    AGILE_TURN_CAMPAIGN,
    DEFAULT_HORIZON_GRID_S,
    FAMILY_NAMES,
    PLANNED_ESCALATION_HORIZON_GRID_S,
    AgileTurnFamilyConfig,
    AgileTurnFamilyComparisonResult,
    AgileTurnCandidateResult,
    acceptance_thresholds,
    candidate_ranking_key,
    compare_agile_turn_families,
    family_inventory,
    horizon_grid_s,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from logging_contract import command_dataframe, trajectory_dataframe
from result_paths import make_result_tree


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Output helpers
# 2) Comparison runner
# 3) CLI
# =============================================================================


# =============================================================================
# 1) Output Helpers
# =============================================================================
DEFAULT_RESULTS_ROOT = CONTROL_DIR / "05_Results"
BOUNDARY_REFERENCE_NOTE = (
    "Existing run-002 high-alpha pitch-brake/perch-like evidence is preserved as "
    "archived boundary evidence only; retired speed-collapse branches are not "
    "active reusable candidates in this comparison."
)
NO_OVERCLAIM_FLAGS = {
    "actual_agile_turn_family_comparison_implemented": True,
    "actual_agile_reversal_primitive_implemented": False,
    "updraft_validation_claim": False,
    "w1_w2_w3_updraft_validation_claim": False,
    "real_flight_validation_claim": False,
    "ocp_implemented": False,
    "tvlqr_implemented": False,
    "governor_implemented": False,
    "outer_loop_implemented": False,
    "vicon_implemented": False,
    "hardware_implemented": False,
    "high_incidence_validation_claim": False,
    "raw_normalised_commands_enter_state_derivative": False,
}


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _json_safe(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return _repo_relative(value)
    return str(value)


def _run_suffix(run_id: int) -> str:
    return f"s{run_id:03d}"


def _target_token(target_heading_deg: float) -> str:
    return f"{int(round(float(target_heading_deg))):03d}"


def _clear_generated_files(run_root: Path) -> None:
    """Clear generated files while preserving sync-managed directories."""

    expected_parent = (DEFAULT_RESULTS_ROOT / AGILE_TURN_CAMPAIGN).resolve()
    resolved_root = run_root.resolve()
    if expected_parent not in resolved_root.parents:
        raise ValueError("refusing to clear files outside the agile-turn result tree.")
    for path in sorted(run_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()


def _best_for_family(
    result: AgileTurnFamilyComparisonResult,
    family_name: str,
) -> AgileTurnCandidateResult:
    family_candidates = [
        candidate for candidate in result.family_results
        if candidate.family_name == family_name
    ]
    if not family_candidates:
        raise ValueError(f"no candidates found for family {family_name}.")
    return max(family_candidates, key=lambda candidate: candidate_ranking_key(candidate.metrics))


def _write_best_candidate_logs(
    paths: dict[str, Path],
    run_id: int,
    result: AgileTurnFamilyComparisonResult,
    output_paths: dict[str, Path],
) -> None:
    suffix = _run_suffix(run_id)
    for family_name in family_inventory():
        candidate = _best_for_family(result, family_name)
        target = _target_token(result.target_heading_deg)
        base = f"agile_turn_{family_name}_target_{target}_{suffix}"
        trajectory_csv = paths["metrics"] / f"{base}_trajectory.csv"
        commands_csv = paths["metrics"] / f"{base}_commands.csv"
        trajectory_dataframe(candidate.time_s, candidate.x_ref).assign(
            phase=list(candidate.phase)
        ).to_csv(trajectory_csv, index=False)
        command_dataframe(
            candidate.time_s,
            candidate.u_norm_requested,
            candidate.u_norm_applied,
            candidate.delta_cmd_rad,
        ).assign(phase=list(candidate.phase)).to_csv(commands_csv, index=False)
        output_paths[f"{base}_trajectory_csv"] = trajectory_csv
        output_paths[f"{base}_commands_csv"] = commands_csv


def _target_summary_row(result: AgileTurnFamilyComparisonResult) -> dict[str, object]:
    selected = result.selected_candidate
    candidates = result.family_results
    best_heading = max(candidates, key=lambda candidate: candidate.metrics["actual_heading_change_deg"])
    best_terminal_speed = max(candidates, key=lambda candidate: candidate.metrics["terminal_speed_m_s"])
    recoverable = [candidate for candidate in candidates if bool(candidate.metrics["recoverable"])]
    best_recoverable_heading = max(
        recoverable if recoverable else candidates,
        key=lambda candidate: candidate.metrics["actual_heading_change_deg"],
    )
    best_energy = min(candidates, key=lambda candidate: candidate.metrics["energy_lost_per_deg_m_per_deg"])
    return {
        "target_heading_deg": float(result.target_heading_deg),
        "selected_family": selected.family_name,
        "selected_horizon_s": float(selected.metrics["horizon_s"]),
        "selected_candidate_id": selected.metrics["candidate_id"],
        "selected_candidate_class": selected.metrics["candidate_class"],
        "selected_actual_heading_change_deg": selected.metrics["actual_heading_change_deg"],
        "selected_terminal_speed_m_s": selected.metrics["terminal_speed_m_s"],
        "selected_recoverable": selected.metrics["recoverable"],
        "selected_strict_family_success": selected.metrics["strict_family_success"],
        "selected_useful_recoverable_candidate": selected.metrics["useful_recoverable_candidate"],
        "best_by_heading": best_heading.metrics["candidate_id"],
        "best_by_terminal_speed": best_terminal_speed.metrics["candidate_id"],
        "best_by_recoverable_heading": best_recoverable_heading.metrics["candidate_id"],
        "best_by_energy_lost_per_deg": best_energy.metrics["candidate_id"],
        "selection_reason": result.notes,
    }


def _family_summary_rows(results: tuple[AgileTurnFamilyComparisonResult, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_name in family_inventory():
        candidates = [
            candidate
            for result in results
            for candidate in result.family_results
            if candidate.family_name == family_name
        ]
        best = max(candidates, key=lambda candidate: candidate_ranking_key(candidate.metrics))
        selected_targets = [
            result.target_heading_deg
            for result in results
            if result.selected_family == family_name
        ]
        useful_at_30 = any(
            bool(candidate.metrics["useful_recoverable_candidate"])
            and float(candidate.metrics["target_heading_deg"]) == 30.0
            for candidate in candidates
        )
        useful_at_15 = any(
            bool(candidate.metrics["useful_recoverable_candidate"])
            and float(candidate.metrics["target_heading_deg"]) == 15.0
            for candidate in candidates
        )
        if useful_at_30:
            status = "selected_for_next_stage"
            cause = "recoverable_at_30_under_current_gates"
        elif useful_at_15:
            status = "retained_as_thesis_discussion_evidence"
            cause = "useful_at_15_only_not_ready_for_45_60_escalation"
        elif any(bool(candidate.metrics["horizon_limited"]) for candidate in candidates):
            status = "retained_as_thesis_discussion_evidence"
            cause = "horizon_limited"
        elif any(bool(candidate.metrics["energy_limited"]) for candidate in candidates):
            status = "retained_as_thesis_discussion_evidence"
            cause = "physics_energy_limited"
        elif any(bool(candidate.metrics["safety_limited"]) for candidate in candidates):
            status = "retained_as_thesis_discussion_evidence"
            cause = "physics_safety_limited"
        elif any(bool(candidate.metrics["exposure_limited"]) for candidate in candidates):
            status = "rejected_for_active_primitive"
            cause = "conservative_exposure_gate_limited"
        else:
            status = "retained_as_thesis_discussion_evidence"
            cause = "turn_authority_limited"
        rows.append(
            {
                "family_name": family_name,
                "family_status": status,
                "failure_cause_summary": cause,
                "selected_targets_deg": ",".join(str(int(round(target))) for target in selected_targets),
                "best_candidate_id": best.metrics["candidate_id"],
                "best_candidate_class": best.metrics["candidate_class"],
                "best_actual_heading_change_deg": best.metrics["actual_heading_change_deg"],
                "best_terminal_speed_m_s": best.metrics["terminal_speed_m_s"],
                "best_recoverable": best.metrics["recoverable"],
            }
        )
    return rows


def _write_report(
    path: Path,
    manifest: dict[str, object],
    target_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Agile Turn Family Comparison Report",
        "",
        "This W0/no-wind evidence pass compares reusable speed-retaining turn families.",
        BOUNDARY_REFERENCE_NOTE,
        "",
        "No OCP, TVLQR, governor, outer loop, updraft validation, real-flight, hardware,",
        "or high-incidence validation claim is made from this pass.",
        "",
        "## Command Path",
        "",
        "- Requested command: `u_norm_requested`.",
        "- Applied command: `u_norm_applied`, clipped to the normalised contract.",
        "- Plant command: `delta_cmd_rad`.",
        "- `rk4_step` and `state_derivative` receive physical radian commands only.",
        "",
        "## Acceptance Gates",
        "",
        f"- Strict success heading gate: `0.8 * target_heading_deg`.",
        f"- Useful recoverable heading gate: `0.6 * target_heading_deg`, or `15 deg` for the `30 deg` target.",
        f"- Strict terminal/min speed: `{manifest['acceptance_thresholds']['strict_terminal_speed_m_s']}` / `{manifest['acceptance_thresholds']['strict_min_speed_m_s']}` m/s.",
        f"- Useful terminal/min speed: `{manifest['acceptance_thresholds']['useful_terminal_speed_m_s']}` / `{manifest['acceptance_thresholds']['useful_min_speed_m_s']}` m/s.",
        "",
        "## Target Summary",
        "",
        "| target_deg | selected_family | horizon_s | class | heading_deg | terminal_speed_m_s | reason |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in target_rows:
        lines.append(
            "| {target_heading_deg:.0f} | {selected_family} | {selected_horizon_s:.2f} | "
            "{selected_candidate_class} | {selected_actual_heading_change_deg:.3f} | "
            "{selected_terminal_speed_m_s:.3f} | {selection_reason} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Family Status",
            "",
            "| family | status | failure or retention cause | best_heading_deg | best_terminal_speed_m_s |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in family_rows:
        lines.append(
            "| {family_name} | {family_status} | {failure_cause_summary} | "
            "{best_actual_heading_change_deg:.3f} | {best_terminal_speed_m_s:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Escalation",
            "",
            f"- Escalation allowed from 30 deg evidence: `{manifest['escalation_allowed']}`.",
            f"- Escalation targets run: `{manifest['escalation_targets_run_deg']}`.",
            f"- Escalation reason: `{manifest['escalation_reason']}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 2) Comparison Runner
# =============================================================================
def run_comparison(
    run_id: int = 1,
    targets_deg: tuple[float, ...] = (15.0, 30.0),
    overwrite: bool = False,
    escalate: bool = False,
) -> dict[str, Path]:
    """Run the W0 agile-turn family comparison and write raw evidence."""

    run_root = DEFAULT_RESULTS_ROOT / AGILE_TURN_CAMPAIGN / f"{run_id:03d}"
    if overwrite and run_root.exists():
        _clear_generated_files(run_root)
    paths = make_result_tree(DEFAULT_RESULTS_ROOT, AGILE_TURN_CAMPAIGN, run_id, overwrite=overwrite)
    suffix = _run_suffix(run_id)
    aircraft = adapt_glider(build_nausicaa_glider())

    requested_targets = tuple(float(target) for target in targets_deg)
    active_targets = [target for target in requested_targets if target in (15.0, 30.0)]
    if any(target not in (15.0, 30.0, 45.0, 60.0) for target in requested_targets):
        raise ValueError("agile turn comparison supports 15/30 deg plus planned 45/60 deg escalation.")
    if any(target in (45.0, 60.0) for target in requested_targets) and not escalate:
        raise ValueError("45/60 deg targets require --escalate and useful 30 deg evidence.")

    results: list[AgileTurnFamilyComparisonResult] = []
    candidate_rows: list[dict[str, object]] = []
    output_paths: dict[str, Path] = {}
    for target in active_targets:
        config = AgileTurnFamilyConfig(
            t_final_s=horizon_grid_s(target)[0],
            target_heading_deg=target,
        )
        result = compare_agile_turn_families(config, families=FAMILY_NAMES, aircraft=aircraft)
        results.append(result)
        candidate_rows.extend(
            {"run_id": suffix, **row}
            for row in result.ranking_rows
        )
        _write_best_candidate_logs(paths, run_id, result, output_paths)

    useful_30 = any(
        bool(candidate.metrics["useful_recoverable_candidate"])
        for result in results
        if result.target_heading_deg == 30.0
        for candidate in result.family_results
    )
    escalation_targets_run: list[float] = []
    if escalate and useful_30:
        for target in requested_targets:
            if target not in (45.0, 60.0):
                continue
            config = AgileTurnFamilyConfig(
                t_final_s=horizon_grid_s(target)[0],
                target_heading_deg=target,
            )
            result = compare_agile_turn_families(config, families=FAMILY_NAMES, aircraft=aircraft)
            results.append(result)
            escalation_targets_run.append(target)
            candidate_rows.extend(
                {"run_id": suffix, **row}
                for row in result.ranking_rows
            )
            _write_best_candidate_logs(paths, run_id, result, output_paths)

    if escalate and not useful_30:
        escalation_reason = "blocked_no_useful_recoverable_30deg_candidate"
    elif escalation_targets_run:
        escalation_reason = "escalation_allowed_by_useful_recoverable_30deg_candidate"
    else:
        escalation_reason = "not_requested_default_15_30_only"

    target_rows = [_target_summary_row(result) for result in results]
    family_rows = _family_summary_rows(tuple(results))
    candidate_summary_csv = paths["metrics"] / f"agile_turn_candidate_summary_{suffix}.csv"
    target_summary_csv = paths["metrics"] / f"agile_turn_target_summary_{suffix}.csv"
    family_summary_csv = paths["metrics"] / f"agile_turn_family_summary_{suffix}.csv"
    pd.DataFrame(candidate_rows).to_csv(candidate_summary_csv, index=False)
    pd.DataFrame(target_rows).to_csv(target_summary_csv, index=False)
    pd.DataFrame(family_rows).to_csv(family_summary_csv, index=False)
    output_paths.update(
        {
            "candidate_summary_csv": candidate_summary_csv,
            "target_summary_csv": target_summary_csv,
            "family_summary_csv": family_summary_csv,
        }
    )

    manifest_json = paths["manifests"] / f"agile_turn_family_comparison_manifest_{suffix}.json"
    report_md = paths["reports"] / f"agile_turn_family_comparison_report_{suffix}.md"
    output_paths["manifest_json"] = manifest_json
    output_paths["report_md"] = report_md
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": suffix,
        "campaign": AGILE_TURN_CAMPAIGN,
        "comparison_scope": "w0_no_wind_reusable_speed_retaining_family_comparison",
        "targets_requested_deg": list(requested_targets),
        "targets_run_deg": [result.target_heading_deg for result in results],
        "active_family_inventory": list(family_inventory()),
        "retired_high_alpha_branch_active": False,
        "boundary_reference_note": BOUNDARY_REFERENCE_NOTE,
        "default_horizon_grid_s": {str(key): list(values) for key, values in DEFAULT_HORIZON_GRID_S.items()},
        "planned_escalation_horizon_grid_s": {
            str(key): list(values) for key, values in PLANNED_ESCALATION_HORIZON_GRID_S.items()
        },
        "acceptance_thresholds": acceptance_thresholds(),
        "escalation_allowed": bool(useful_30),
        "escalation_requested": bool(escalate),
        "escalation_targets_run_deg": escalation_targets_run,
        "escalation_reason": escalation_reason,
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad -> rk4_step/state_derivative",
        "state_derivative_command_input": "delta_cmd_rad",
        "output_files": {key: _repo_relative(path) for key, path in output_paths.items()},
        **NO_OVERCLAIM_FLAGS,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=_json_safe), encoding="ascii")
    _write_report(report_md, manifest, target_rows, family_rows)
    output_paths["root"] = paths["root"]
    return output_paths


# =============================================================================
# 3) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the W0 agile-turn family comparison.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--targets", nargs="+", type=float, default=[15.0, 30.0])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--escalate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_comparison(
        run_id=args.run_id,
        targets_deg=tuple(args.targets),
        overwrite=args.overwrite,
        escalate=args.escalate,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['manifest_json']}")
    print(f"report={outputs['report_md']}")
    print(f"targets_run_deg={manifest['targets_run_deg']}")
    print(f"escalation_allowed={manifest['escalation_allowed']}")
    print(f"escalation_reason={manifest['escalation_reason']}")


if __name__ == "__main__":
    main()
