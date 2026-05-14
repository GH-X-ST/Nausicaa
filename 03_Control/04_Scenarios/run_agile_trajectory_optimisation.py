from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup
# 2) Output helpers
# 3) Phase 1 solve sequence
# 4) Phase 2 replay sequence
# 5) Reports and CLI
# =============================================================================

# =============================================================================
# 1) Import Path Setup
# =============================================================================
def _add_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "03_Control/02_Inner_Loop",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
    ):
        path = repo_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _add_paths()

from arena import ArenaConfig  # noqa: E402
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from latency import CommandToSurfaceConfig, CommandToSurfaceLayer, LatencyEnvelope  # noqa: E402
from linearisation import linearise_trim  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, simulate_primitive, write_log  # noqa: E402
from scenarios import arena_feasible_entry_state  # noqa: E402
from templates import RecoveryPrimitive  # noqa: E402
from turn_trajectory_optimisation import (  # noqa: E402
    ACCEPTED_LABELS,
    OptimisedTurnResult,
    TurnOptimisationConfig,
    TurnTarget,
    build_turn_trajectory_primitive,
    deterministic_initial_guess_names,
    primitive_open_loop_copy,
    save_turn_result,
    solve_turn_ocp,
)  # noqa: E402


# =============================================================================
# 2) Output Helpers
# =============================================================================
@dataclass(frozen=True)
class TurnOutputPaths:
    root: Path
    metrics_dir: Path
    logs_dir: Path
    manifests_dir: Path
    trajectories_dir: Path


def _output_paths(output_root: str | Path | None, run_tvlqr_replay: bool = False) -> TurnOutputPaths:
    default_case = "09_tight_turn_ocp_phase2" if run_tvlqr_replay else "08_tight_turn_ocp_phase1"
    root = (
        Path(output_root)
        if output_root is not None
        else REPO_ROOT
        / "03_Control"
        / "05_Results"
        / "03_primitives"
        / default_case
        / "001"
    )
    paths = TurnOutputPaths(
        root=root,
        metrics_dir=root / "metrics",
        logs_dir=root / "logs",
        manifests_dir=root / "manifests",
        trajectories_dir=root / "trajectories",
    )
    for path in (paths.metrics_dir, paths.logs_dir, paths.manifests_dir, paths.trajectories_dir):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _result_row(
    result: OptimisedTurnResult,
    *,
    seed: int,
    solve_kind: str,
    target_requested_deg: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": int(seed),
        "solve_kind": solve_kind,
        "target_requested_deg": float(target_requested_deg),
        "success": bool(result.success),
        "feasibility_label": result.feasibility_label,
        "failure_reason": result.failure_reason,
        "score": _result_score(result),
    }
    row.update(result.metrics)
    row.update(
        {
            "solver_return_status": result.solver_stats.get("return_status", ""),
            "initial_guess_name": result.solver_stats.get("initial_guess_name", ""),
            "solver_name": result.solver_stats.get("solver_name", ""),
        }
    )
    return row


def _result_score(result: OptimisedTurnResult) -> float:
    metrics = result.metrics
    target = abs(float(metrics.get("target_heading_deg", 0.0)))
    directed = float(metrics.get("directed_heading_change_deg", 0.0))
    heading_shortfall = max(0.0, target - directed)
    slack = float(metrics.get("slack_max", 0.0))
    wall = float(metrics.get("min_wall_distance_m", 0.0))
    defect = float(metrics.get("dynamics_defect_max", 0.0))
    return float(directed - 3.0 * heading_shortfall - 80.0 * slack + 4.0 * wall - 500.0 * defect)


def _best_result(results: list[OptimisedTurnResult]) -> OptimisedTurnResult:
    accepted = [result for result in results if result.success and not result.target.allow_safety_slack]
    pool = accepted or results
    return max(pool, key=_result_score)


# =============================================================================
# 3) Phase 1 Solve Sequence
# =============================================================================
def run_phase_1_2(
    *,
    targets_deg: tuple[float, ...],
    direction: str,
    seed: int,
    output_root: str | Path | None,
    run_tvlqr_replay: bool,
    allow_high_alpha: bool = False,
    n_intervals: int = 18,
    max_solver_time_s: float = 30.0,
    ipopt_max_iter: int = 220,
) -> dict[str, object]:
    unsupported = sorted(set(float(target) for target in targets_deg) - {0.0, 30.0})
    if unsupported:
        raise ValueError(
            "Phase 1/2 runner only supports targets 0 and 30 deg; "
            f"unsupported targets: {unsupported}"
        )

    np.random.seed(int(seed))
    paths = _output_paths(output_root, run_tvlqr_replay=run_tvlqr_replay)
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    x0 = arena_feasible_entry_state(linear_model.x_trim, altitude_m=2.7)
    context = build_primitive_context(
        linear_model.x_trim,
        linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    base_config = TurnOptimisationConfig(
        n_intervals=int(n_intervals),
        max_solver_time_s=float(max_solver_time_s),
        ipopt_max_iter=int(ipopt_max_iter),
    )

    all_results: list[OptimisedTurnResult] = []
    candidate_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    best_by_target: dict[float, OptimisedTurnResult] = {}

    for target_deg in targets_deg:
        target_results: list[OptimisedTurnResult] = []
        hard_target = TurnTarget(
            target_heading_deg=float(target_deg),
            direction=direction,
            wind_case="w0",
            allow_high_alpha=bool(allow_high_alpha),
            allow_safety_slack=False,
        )
        for guess_name in deterministic_initial_guess_names(float(target_deg)):
            result = solve_turn_ocp(
                target=hard_target,
                config=base_config,
                x0=x0,
                aircraft=aircraft,
                wind_model=None,
                wind_mode="none",
                u_trim=linear_model.u_trim,
                initial_guess_name=guess_name,
            )
            target_results.append(result)
            all_results.append(result)
            candidate_rows.append(
                _result_row(
                    result,
                    seed=seed,
                    solve_kind="hard",
                    target_requested_deg=float(target_deg),
                )
            )
            save_turn_result(
                result,
                paths.root,
                stem=_result_stem(result, seed=seed, solve_kind="hard"),
            )
            if result.success:
                break

        hard_success = any(result.success for result in target_results)
        if float(target_deg) == 30.0 and not hard_success:
            soft_target = replace(hard_target, allow_safety_slack=True)
            soft_config = replace(base_config, max_solver_time_s=max_solver_time_s)
            for guess_name in deterministic_initial_guess_names(float(target_deg)):
                result = solve_turn_ocp(
                    target=soft_target,
                    config=soft_config,
                    x0=x0,
                    aircraft=aircraft,
                    wind_model=None,
                    wind_mode="none",
                    u_trim=linear_model.u_trim,
                    initial_guess_name=guess_name,
                )
                target_results.append(result)
                all_results.append(result)
                candidate_rows.append(
                    _result_row(
                        result,
                        seed=seed,
                        solve_kind="soft_boundary",
                        target_requested_deg=float(target_deg),
                    )
                )
                save_turn_result(
                    result,
                    paths.root,
                    stem=_result_stem(result, seed=seed, solve_kind="soft_boundary"),
                )

        best = _best_result(target_results)
        best_by_target[float(target_deg)] = best
        best_rows.append(
            _result_row(
                best,
                seed=seed,
                solve_kind="best",
                target_requested_deg=float(target_deg),
            )
        )

    _write_rows(paths.metrics_dir / f"turn_ocp_candidates_seed{int(seed)}.csv", candidate_rows)
    _write_rows(paths.metrics_dir / f"turn_ocp_best_by_target_seed{int(seed)}.csv", best_rows)

    replay_rows: list[dict[str, object]] = []
    if run_tvlqr_replay and 30.0 in best_by_target and best_by_target[30.0].success:
        replay_rows = _run_phase_2_replay(
            result=best_by_target[30.0],
            seed=seed,
            paths=paths,
            context=context,
            aircraft=aircraft,
        )
        _write_rows(paths.metrics_dir / f"turn_tvlqr_replay_seed{int(seed)}.csv", replay_rows)

    _write_manifest(
        paths=paths,
        seed=seed,
        targets_deg=targets_deg,
        candidate_rows=candidate_rows,
        best_rows=best_rows,
        replay_rows=replay_rows,
    )
    _write_reports(
        paths=paths,
        seed=seed,
        candidate_rows=candidate_rows,
        best_rows=best_rows,
        replay_rows=replay_rows,
    )
    return {
        "candidate_rows": candidate_rows,
        "best_rows": best_rows,
        "replay_rows": replay_rows,
        "output_root": str(paths.root),
    }


def _result_stem(result: OptimisedTurnResult, *, seed: int, solve_kind: str) -> str:
    target_tag = f"{int(round(abs(float(result.target.target_heading_deg)))):03d}"
    guess = str(result.solver_stats.get("initial_guess_name", "guess"))
    slack = "soft" if result.target.allow_safety_slack else "hard"
    return f"turn_ocp_target_{target_tag}_{solve_kind}_{slack}_{guess}_seed{int(seed)}"


# =============================================================================
# 4) Phase 2 Replay Sequence
# =============================================================================
def _run_phase_2_replay(
    *,
    result: OptimisedTurnResult,
    seed: int,
    paths: TurnOutputPaths,
    context,
    aircraft,
) -> list[dict[str, object]]:
    primitive = build_turn_trajectory_primitive(
        result=result,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        command_layer=CommandToSurfaceLayer(),
    )
    save_turn_result(
        result,
        paths.root,
        stem=f"turn_ocp_target_030_selected_tvlqr_seed{int(seed)}",
        primitive=primitive,
    )
    replay_rows: list[dict[str, object]] = []
    x0 = np.asarray(result.x_ref[0], dtype=float)
    no_latency = CommandToSurfaceConfig(
        mode="nominal",
        quantise=False,
        use_onset_delay=False,
        use_state_feedback_delay=False,
    )
    runs = (
        ("open_loop_no_latency", primitive_open_loop_copy(primitive), no_latency),
        ("closed_loop_no_latency", primitive, no_latency),
        ("closed_loop_nominal_latency", primitive, CommandToSurfaceConfig(mode="nominal")),
    )
    closed_loop_safe = False
    for label, run_primitive, latency_config in runs:
        if label == "closed_loop_nominal_latency" and not closed_loop_safe:
            replay_rows.append(
                {
                    "replay_kind": label,
                    "success": False,
                    "termination_reason": "skipped because closed_loop_no_latency failed",
                }
            )
            continue
        log_path = paths.logs_dir / f"turn_tvlqr_{label}_seed{int(seed)}.csv"
        rollout = simulate_primitive(
            scenario_id=f"turn_ocp_{label}",
            seed=seed,
            primitive=run_primitive,
            x0=x0,
            context=context,
            aircraft=aircraft,
            wind_model=None,
            wind_model_name="none",
            wind_mode="none",
            command_layer=CommandToSurfaceLayer(
                config=latency_config,
                envelope=LatencyEnvelope(),
            ),
            log_path=log_path,
            repo_root=REPO_ROOT,
            rollout_config=RolloutConfig(),
            arena_config=ArenaConfig(),
            wind_param_label="none",
        )
        write_log(rollout, log_path)
        row = {"replay_kind": label, **rollout.metrics}
        replay_rows.append(row)
        if label == "closed_loop_no_latency":
            closed_loop_safe = bool(rollout.metrics.get("success", False)) and bool(
                rollout.metrics.get("exit_recoverable", False)
            )

    replay_rows.extend(
        _terminal_altitude_sensitivity_rows(
            result=result,
            seed=seed,
            paths=paths,
            context=context,
            aircraft=aircraft,
        )
    )
    return replay_rows


def _terminal_altitude_sensitivity_rows(
    *,
    result: OptimisedTurnResult,
    seed: int,
    paths: TurnOutputPaths,
    context,
    aircraft,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    terminal_state = np.asarray(result.x_ref[-1], dtype=float)
    for altitude_m in (0.75, 1.0, 1.2):
        sensitivity_context = replace(context, min_entry_altitude_m=float(altitude_m))
        log_path = paths.logs_dir / (
            f"turn_recovery_terminal_altitude_{int(100 * altitude_m):03d}_seed{int(seed)}.csv"
        )
        recovery = RecoveryPrimitive(duration_s=0.76)
        entry = recovery.entry_conditions(terminal_state, sensitivity_context)
        if not entry.passed:
            rows.append(
                {
                    "replay_kind": "terminal_altitude_sensitivity",
                    "terminal_altitude_min_m": float(altitude_m),
                    "success": False,
                    "termination_reason": "; ".join(entry.reasons),
                }
            )
            continue
        rollout = simulate_primitive(
            scenario_id=f"turn_recovery_terminal_altitude_{altitude_m:.2f}",
            seed=seed,
            primitive=recovery,
            x0=terminal_state,
            context=sensitivity_context,
            aircraft=aircraft,
            wind_model=None,
            wind_model_name="none",
            wind_mode="none",
            command_layer=CommandToSurfaceLayer(),
            log_path=log_path,
            repo_root=REPO_ROOT,
            rollout_config=RolloutConfig(),
            arena_config=ArenaConfig(),
            wind_param_label="none",
        )
        write_log(rollout, log_path)
        rows.append(
            {
                "replay_kind": "terminal_altitude_sensitivity",
                "terminal_altitude_min_m": float(altitude_m),
                **rollout.metrics,
            }
        )
    return rows


# =============================================================================
# 5) Reports and CLI
# =============================================================================
def _write_manifest(
    *,
    paths: TurnOutputPaths,
    seed: int,
    targets_deg: tuple[float, ...],
    candidate_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    replay_rows: list[dict[str, object]],
) -> None:
    manifest = {
        "status": "phase_1_2_complete",
        "seed": int(seed),
        "targets_deg": [float(target) for target in targets_deg],
        "candidate_count": len(candidate_rows),
        "best_by_target_count": len(best_rows),
        "replay_count": len(replay_rows),
        "scope": "Phase 1/2 only; no entry sweep, W0-W3 stress, outer-loop, or hardware code",
        "frozen_invariants": {
            "state_order": "[x_w,y_w,z_w,phi,theta,psi,u,v,w,p,q,r,delta_a,delta_e,delta_r]",
            "command_order": "[delta_a_cmd,delta_e_cmd,delta_r_cmd]",
            "command_range": "full calibrated normalised [-1,+1]",
            "safety_volume": "true safety volume",
        },
    }
    path = paths.manifests_dir / f"turn_ocp_manifest_seed{int(seed)}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_reports(
    *,
    paths: TurnOutputPaths,
    seed: int,
    candidate_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    replay_rows: list[dict[str, object]],
) -> None:
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    prior_paths = (
        docs_dir / "control" / "agile_problem_1_2_7_report.md",
        docs_dir / "control" / "agile_feasibility_boundary.md",
    )
    prior_notes = [
        f"- `{path.as_posix()}`: {'found' if path.exists() else 'not found in this checkout'}"
        for path in prior_paths
    ]
    best_30 = next(
        (row for row in best_rows if float(row.get("target_requested_deg", -1.0)) == 30.0),
        None,
    )
    report = [
        "# Tight-Turn Trajectory Optimisation Phase 1/2 Report",
        "",
        f"Seed: `{int(seed)}`",
        f"Output root: `{paths.root.as_posix()}`",
        "",
        "## Scope",
        "",
        "Implemented Phase 1/2 only: OCP smoke, W0 30 deg hard solve, W0 30 deg soft-boundary diagnostic when required, and TVLQR replay only for an accepted hard 30 deg candidate.",
        "",
        "## Prior Agile Boundary Evidence",
        "",
        *prior_notes,
        "",
        "## Best 30 Deg Result",
        "",
    ]
    if best_30 is None:
        report.append("No 30 deg target row was produced.")
    else:
        report.extend(
            [
                f"- label: `{best_30.get('feasibility_label')}`",
                f"- success: `{best_30.get('success')}`",
                f"- directed heading change deg: `{best_30.get('directed_heading_change_deg')}`",
                f"- actual wrapped heading change deg: `{best_30.get('actual_heading_change_deg')}`",
                f"- dynamics defect max: `{best_30.get('dynamics_defect_max')}`",
                f"- slack max: `{best_30.get('slack_max')}`",
                f"- failure reason: `{best_30.get('failure_reason')}`",
            ]
        )
    phase2_summary = _phase2_gate_summary(replay_rows)
    report.extend(
        [
            "",
            "## Replay",
            "",
            f"Replay rows produced: `{len(replay_rows)}`",
            f"- closed-loop no-latency gate: `{phase2_summary['closed_loop_no_latency']}`",
            f"- nominal-latency gate: `{phase2_summary['closed_loop_nominal_latency']}`",
            f"- terminal-altitude recovery sensitivity gate: `{phase2_summary['terminal_altitude_sensitivity']}`",
            f"- promoted beyond Phase 2: `{phase2_summary['promoted_beyond_phase2']}`",
            f"- replay limitation: `{phase2_summary['limitation']}`",
            "",
            "## Metrics Paths",
            "",
            f"- `{(paths.metrics_dir / f'turn_ocp_candidates_seed{int(seed)}.csv').as_posix()}`",
            f"- `{(paths.metrics_dir / f'turn_ocp_best_by_target_seed{int(seed)}.csv').as_posix()}`",
            f"- `{(paths.metrics_dir / f'turn_tvlqr_replay_seed{int(seed)}.csv').as_posix()}` if replay ran",
            "",
            "## Limitation",
            "",
            "This report is simulation-only and does not include Phase 3/4 continuation, entry sweeps, W0-W3 stress, outer-loop simulation, or hardware/Vicon execution.",
            "",
        ]
    )
    report_dir = docs_dir / "control"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "turn_trajectory_optimisation_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    if best_30 is not None and str(best_30.get("feasibility_label")) not in ACCEPTED_LABELS:
        boundary = [
            "# Tight-Turn 30 Deg Feasibility Boundary",
            "",
            "The hard 30 deg Phase 1/2 gate did not produce an accepted primitive.",
            "",
            f"- best label: `{best_30.get('feasibility_label')}`",
            f"- failure reason: `{best_30.get('failure_reason')}`",
            f"- directed heading change deg: `{best_30.get('directed_heading_change_deg')}`",
            f"- heading threshold deg: `{best_30.get('heading_threshold_deg')}`",
            f"- min wall distance m: `{best_30.get('min_wall_distance_m')}`",
            f"- min floor margin m: `{best_30.get('min_floor_margin_m')}`",
            f"- terminal altitude m: `{best_30.get('terminal_z_w_m')}`",
            f"- max alpha deg: `{best_30.get('max_alpha_deg')}`",
            f"- saturation fraction: `{best_30.get('saturation_fraction')}`",
            f"- slack max: `{best_30.get('slack_max')}`",
            "",
            "Solver failure is not treated as physical infeasibility. A physical infeasibility label requires the smoke case, at least two deterministic hard guesses, soft-boundary diagnostic, and active constraints to be recorded.",
            "",
        ]
        (report_dir / "turn_feasibility_boundary.md").write_text(
            "\n".join(boundary),
            encoding="utf-8",
        )


def _phase2_gate_summary(replay_rows: list[dict[str, object]]) -> dict[str, object]:
    def passed(kind: str) -> bool | str:
        rows = [row for row in replay_rows if str(row.get("replay_kind")) == kind]
        if not rows:
            return "not_run"
        return all(str(row.get("success")) == "True" or row.get("success") is True for row in rows)

    closed = passed("closed_loop_no_latency")
    nominal = passed("closed_loop_nominal_latency")
    sensitivity = passed("terminal_altitude_sensitivity")
    promoted = closed is True and nominal is True and sensitivity is True
    failed = [
        str(row.get("termination_reason"))
        for row in replay_rows
        if not (str(row.get("success")) == "True" or row.get("success") is True)
        and str(row.get("termination_reason"))
    ]
    limitation = "; ".join(dict.fromkeys(failed)) if failed else ""
    return {
        "closed_loop_no_latency": closed,
        "closed_loop_nominal_latency": nominal,
        "terminal_altitude_sensitivity": sensitivity,
        "promoted_beyond_phase2": promoted,
        "limitation": limitation,
    }


def _parse_targets(values: list[str]) -> tuple[float, ...]:
    targets = tuple(float(value) for value in values)
    if not targets:
        raise ValueError("at least one target is required")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--direction", choices=("left", "right"), default="left")
    parser.add_argument("--wind-case", choices=("w0",), default="w0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tvlqr-replay", action="store_true")
    parser.add_argument("--allow-high-alpha", action="store_true")
    parser.add_argument("--n-intervals", type=int, default=18)
    parser.add_argument("--max-solver-time-s", type=float, default=30.0)
    parser.add_argument("--ipopt-max-iter", type=int, default=220)
    args = parser.parse_args()
    if args.wind_case != "w0":
        raise SystemExit("Phase 1/2 runner supports W0 only.")
    result = run_phase_1_2(
        targets_deg=_parse_targets(args.targets),
        direction=args.direction,
        seed=args.seed,
        output_root=args.output_root,
        run_tvlqr_replay=args.run_tvlqr_replay,
        allow_high_alpha=args.allow_high_alpha,
        n_intervals=args.n_intervals,
        max_solver_time_s=args.max_solver_time_s,
        ipopt_max_iter=args.ipopt_max_iter,
    )
    print("turn trajectory optimisation phase 1/2 complete")
    print(f"output_root: {result['output_root']}")
    print(f"candidate_rows: {len(result['candidate_rows'])}")
    print(f"best_rows: {len(result['best_rows'])}")
    print(f"replay_rows: {len(result['replay_rows'])}")


if __name__ == "__main__":
    main()
