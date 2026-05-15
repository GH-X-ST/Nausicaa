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
    default_case = (
        "11_tight_turn_phase2_latency_recovery"
        if run_tvlqr_replay
        else "08_tight_turn_ocp_phase1"
    )
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


def _seed_tag(seed: int) -> str:
    return f"s{int(seed):03d}"


def _metrics_name(kind: str, seed: int) -> str:
    return f"{kind}_{_seed_tag(seed)}.csv"


def _result_row(
    result: OptimisedTurnResult,
    *,
    seed: int,
    solve_kind: str,
    target_requested_deg: float,
) -> dict[str, object]:
    variant = str(result.solver_stats.get("candidate_variant", "baseline"))
    row: dict[str, object] = {
        "seed": int(seed),
        "solve_kind": solve_kind,
        "candidate_variant": variant,
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


def _with_candidate_variant(
    result: OptimisedTurnResult,
    variant_name: str,
) -> OptimisedTurnResult:
    return replace(
        result,
        solver_stats={
            **result.solver_stats,
            "candidate_variant": str(variant_name),
        },
    )


def _phase2_candidate_config_variants(
    base_config: TurnOptimisationConfig,
) -> tuple[tuple[str, TurnOptimisationConfig], ...]:
    return (
        ("baseline", base_config),
        (
            "latency075",
            replace(
                base_config,
                t_min_s=0.72,
                t_max_s=0.90,
                terminal_speed_bounds_m_s=(5.2, 7.6),
                terminal_bank_deg=35.0,
                terminal_pitch_deg=25.0,
                terminal_alpha_deg=10.0,
                terminal_beta_deg=22.0,
                terminal_rate_max_rad_s=1.20,
                terminal_speed_target_m_s=6.2,
                recovery_weight=20.0,
                smoothness_weight=0.18,
                saturation_weight=0.08,
                terminal_speed_weight=3.0,
                terminal_alpha_weight=14.0,
                terminal_beta_weight=7.0,
                terminal_rate_weight=2.5,
            ),
        ),
        (
            "latency090",
            replace(
                base_config,
                t_min_s=0.88,
                t_max_s=1.10,
                terminal_speed_bounds_m_s=(5.6, 7.2),
                terminal_bank_deg=25.0,
                terminal_pitch_deg=18.0,
                terminal_alpha_deg=8.0,
                terminal_beta_deg=18.0,
                terminal_rate_max_rad_s=0.90,
                terminal_speed_target_m_s=6.3,
                recovery_weight=28.0,
                smoothness_weight=0.24,
                saturation_weight=0.10,
                terminal_speed_weight=5.0,
                terminal_alpha_weight=20.0,
                terminal_beta_weight=10.0,
                terminal_rate_weight=4.0,
            ),
        ),
        (
            "latency105",
            replace(
                base_config,
                t_min_s=1.03,
                t_max_s=1.25,
                terminal_speed_bounds_m_s=(5.8, 7.0),
                terminal_bank_deg=20.0,
                terminal_pitch_deg=15.0,
                terminal_alpha_deg=7.0,
                terminal_beta_deg=15.0,
                terminal_rate_max_rad_s=0.70,
                terminal_speed_target_m_s=6.4,
                recovery_weight=34.0,
                smoothness_weight=0.30,
                saturation_weight=0.12,
                terminal_speed_weight=7.0,
                terminal_alpha_weight=28.0,
                terminal_beta_weight=14.0,
                terminal_rate_weight=6.0,
            ),
        ),
    )


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
    results_by_target: dict[float, list[OptimisedTurnResult]] = {}

    for target_deg in targets_deg:
        target_results: list[OptimisedTurnResult] = []
        hard_target = TurnTarget(
            target_heading_deg=float(target_deg),
            direction=direction,
            wind_case="w0",
            allow_high_alpha=bool(allow_high_alpha),
            allow_safety_slack=False,
        )
        config_variants = (
            _phase2_candidate_config_variants(base_config)
            if run_tvlqr_replay and float(target_deg) == 30.0
            else (("baseline", base_config),)
        )
        for variant_name, variant_config in config_variants:
            solve_kind = "hard" if variant_name == "baseline" else f"hard_{variant_name}"
            for guess_name in deterministic_initial_guess_names(float(target_deg)):
                result = solve_turn_ocp(
                    target=hard_target,
                    config=variant_config,
                    x0=x0,
                    aircraft=aircraft,
                    wind_model=None,
                    wind_mode="none",
                    u_trim=linear_model.u_trim,
                    initial_guess_name=guess_name,
                )
                result = _with_candidate_variant(result, variant_name)
                target_results.append(result)
                all_results.append(result)
                candidate_rows.append(
                    _result_row(
                        result,
                        seed=seed,
                        solve_kind=solve_kind,
                        target_requested_deg=float(target_deg),
                    )
                )
                save_turn_result(
                    result,
                    paths.root,
                    stem=_result_stem(result, seed=seed, solve_kind=solve_kind),
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
                result = _with_candidate_variant(result, "soft_boundary")
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

        results_by_target[float(target_deg)] = target_results
        if not (run_tvlqr_replay and float(target_deg) == 30.0):
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

    replay_rows: list[dict[str, object]] = []
    if run_tvlqr_replay:
        phase2_results = [
            result
            for result in results_by_target.get(30.0, [])
            if result.success and not result.target.allow_safety_slack
        ]
        if phase2_results:
            selected, replay_rows = _select_phase2_result(
                results=phase2_results,
                seed=seed,
                paths=paths,
                context=context,
                aircraft=aircraft,
            )
            best_by_target[30.0] = selected
            selected_row = _result_row(
                selected,
                seed=seed,
                solve_kind="phase2_selected",
                target_requested_deg=30.0,
            )
            selected_summary = _phase2_gate_summary(selected_row, replay_rows)
            selected_row.update(selected_summary)
            best_rows.append(selected_row)
        else:
            replay_rows = [
                {
                    "replay_kind": "phase2_not_run",
                    "candidate_variant": "",
                    "success": False,
                    "termination_reason": "hard 30 deg OCP was not accepted",
                    "failure_class": "ocp_regression",
                }
            ]
            if 30.0 in results_by_target:
                best = _best_result(results_by_target[30.0])
                best_by_target[30.0] = best
                best_rows.append(
                    _result_row(
                        best,
                        seed=seed,
                        solve_kind="best",
                        target_requested_deg=30.0,
                    )
                )
        _write_rows(paths.metrics_dir / _metrics_name("turn_tvlqr_replay", seed), replay_rows)

    _write_rows(paths.metrics_dir / _metrics_name("turn_ocp_candidates", seed), candidate_rows)
    _write_rows(paths.metrics_dir / _metrics_name("turn_ocp_best_by_target", seed), best_rows)

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
    guess = str(result.solver_stats.get("initial_guess_name", "guess")).replace("_seed", "")
    slack = "soft" if result.target.allow_safety_slack else "hard"
    return f"ocp{target_tag}_{solve_kind}_{slack}_{guess}_{_seed_tag(seed)}"


def _candidate_tag(result: OptimisedTurnResult) -> str:
    variant = str(result.solver_stats.get("candidate_variant", "baseline"))
    guess = str(result.solver_stats.get("initial_guess_name", "guess")).replace("_seed", "")
    return f"{variant}_{guess}".replace("-", "_")


# =============================================================================
# 4) Phase 2 Replay Sequence
# =============================================================================
def _select_phase2_result(
    *,
    results: list[OptimisedTurnResult],
    seed: int,
    paths: TurnOutputPaths,
    context,
    aircraft,
) -> tuple[OptimisedTurnResult, list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    ranked: list[tuple[tuple[float, ...], OptimisedTurnResult]] = []
    for result in results:
        tag = _candidate_tag(result)
        rows = _run_phase_2_replay(
            result=result,
            seed=seed,
            paths=paths,
            context=context,
            aircraft=aircraft,
            artifact_tag=tag,
            save_selected_alias=False,
        )
        all_rows.extend(rows)
        ranked.append((_phase2_selection_score(result, rows), result))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[0][1]
    selected_tag = _candidate_tag(selected)
    _save_selected_phase2_artifact(
        result=selected,
        seed=seed,
        paths=paths,
        context=context,
        aircraft=aircraft,
    )
    for row in all_rows:
        row["selected_candidate"] = str(row.get("candidate_tag")) == selected_tag
    return selected, all_rows


def _phase2_selection_score(
    result: OptimisedTurnResult,
    replay_rows: list[dict[str, object]],
) -> tuple[float, ...]:
    best_row = _result_row(
        result,
        seed=0,
        solve_kind="phase2_candidate",
        target_requested_deg=30.0,
    )
    summary = _phase2_gate_summary(best_row, replay_rows)
    gate_count = sum(
        1.0
        for key in (
            "ocp_hard_30",
            "open_loop_no_latency",
            "closed_loop_no_latency",
            "closed_loop_nominal_latency",
            "terminal_altitude_sensitivity",
        )
        if summary[key] is True
    )
    promoted = 1.0 if summary["phase2_status"] == "promoted_phase2" else 0.0
    latency_alpha = _replay_metric(
        replay_rows,
        "closed_loop_nominal_latency",
        "max_alpha_deg",
        default=1.0e6,
    )
    terminal_speed_error = abs(float(result.metrics.get("terminal_speed_m_s", 0.0)) - 6.3)
    directed_heading = float(result.metrics.get("directed_heading_change_deg", 0.0))
    saturation = float(result.metrics.get("saturation_fraction", 1.0))
    return (
        promoted,
        gate_count,
        -latency_alpha,
        -terminal_speed_error,
        directed_heading,
        -saturation,
        _result_score(result),
    )


def _replay_metric(
    rows: list[dict[str, object]],
    replay_kind: str,
    field: str,
    *,
    default: float,
) -> float:
    matches = [row for row in rows if str(row.get("replay_kind")) == replay_kind]
    if not matches:
        return float(default)
    try:
        return float(matches[0].get(field, default))
    except (TypeError, ValueError):
        return float(default)


def _save_selected_phase2_artifact(
    *,
    result: OptimisedTurnResult,
    seed: int,
    paths: TurnOutputPaths,
    context,
    aircraft,
) -> None:
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
        stem=f"ocp030_selected_tvlqr_{_seed_tag(seed)}",
        primitive=primitive,
    )


def _run_phase_2_replay(
    *,
    result: OptimisedTurnResult,
    seed: int,
    paths: TurnOutputPaths,
    context,
    aircraft,
    artifact_tag: str | None = None,
    save_selected_alias: bool = True,
) -> list[dict[str, object]]:
    tag = artifact_tag or _candidate_tag(result)
    variant = str(result.solver_stats.get("candidate_variant", "baseline"))
    primitive = build_turn_trajectory_primitive(
        result=result,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        command_layer=CommandToSurfaceLayer(),
    )
    stem = f"ocp030_tvlqr_{tag}_{_seed_tag(seed)}"
    save_turn_result(result, paths.root, stem=stem, primitive=primitive)
    if save_selected_alias:
        save_turn_result(
            result,
            paths.root,
            stem=f"ocp030_selected_tvlqr_{_seed_tag(seed)}",
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
                    "candidate_variant": variant,
                    "candidate_tag": tag,
                    "success": False,
                    "termination_reason": "skipped because closed_loop_no_latency failed",
                }
            )
            continue
        log_path = paths.logs_dir / _replay_log_name(label, seed, candidate_tag=tag)
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
        row = {
            "replay_kind": label,
            "candidate_variant": variant,
            "candidate_tag": tag,
            **rollout.metrics,
        }
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
            candidate_variant=variant,
            candidate_tag=tag,
        )
    )
    return replay_rows


def _replay_log_name(label: str, seed: int, *, candidate_tag: str | None = None) -> str:
    names = {
        "open_loop_no_latency": "tvlqr_open",
        "closed_loop_no_latency": "tvlqr_closed",
        "closed_loop_nominal_latency": "tvlqr_latency",
    }
    suffix = "" if candidate_tag is None else f"_{candidate_tag}"
    return f"{names.get(label, label)}{suffix}_{_seed_tag(seed)}.csv"


def _terminal_altitude_sensitivity_rows(
    *,
    result: OptimisedTurnResult,
    seed: int,
    paths: TurnOutputPaths,
    context,
    aircraft,
    candidate_variant: str = "baseline",
    candidate_tag: str = "baseline",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    terminal_state = np.asarray(result.x_ref[-1], dtype=float)
    for altitude_m in (0.75, 1.0, 1.2):
        sensitivity_context = replace(context, min_entry_altitude_m=float(altitude_m))
        log_path = (
            paths.logs_dir
            / f"recovery{int(100 * altitude_m):03d}_{candidate_tag}_{_seed_tag(seed)}.csv"
        )
        recovery = RecoveryPrimitive(duration_s=0.76)
        entry = recovery.entry_conditions(terminal_state, sensitivity_context)
        if not entry.passed:
            rows.append(
                {
                    "replay_kind": "terminal_altitude_sensitivity",
                    "candidate_variant": candidate_variant,
                    "candidate_tag": candidate_tag,
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
                "candidate_variant": candidate_variant,
                "candidate_tag": candidate_tag,
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
        "phase2_summary": _phase2_gate_summary(_best_30_row(best_rows), replay_rows),
        "scope": "Phase 1/2 only; no entry sweep, W0-W3 stress, outer-loop, or hardware code",
        "frozen_invariants": {
            "state_order": "[x_w,y_w,z_w,phi,theta,psi,u,v,w,p,q,r,delta_a,delta_e,delta_r]",
            "command_order": "[delta_a_cmd,delta_e_cmd,delta_r_cmd]",
            "command_range": "full calibrated normalised [-1,+1]",
            "safety_volume": "true safety volume",
        },
    }
    path = paths.manifests_dir / f"manifest_{_seed_tag(seed)}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _previous_phase2_debug_summary(seed: int) -> dict[str, object]:
    root = (
        REPO_ROOT
        / "03_Control"
        / "05_Results"
        / "03_primitives"
        / "10_tight_turn_phase2_tvlqr_debug"
        / "001"
    )
    best_rows = _read_csv_rows(root / "metrics" / _metrics_name("turn_ocp_best_by_target", seed))
    replay_rows = _read_csv_rows(root / "metrics" / _metrics_name("turn_tvlqr_replay", seed))
    best_30 = _best_30_row(best_rows)
    return {
        "output_root": root.as_posix(),
        "found": bool(best_rows or replay_rows),
        "summary": _phase2_gate_summary(best_30, replay_rows) if best_rows or replay_rows else {},
        "best_30": best_30,
    }


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
    previous = _previous_phase2_debug_summary(seed)
    prior_paths = (
        docs_dir / "control" / "agile_problem_1_2_7_report.md",
        docs_dir / "control" / "agile_feasibility_boundary.md",
        docs_dir / "control" / "turn_trajectory_optimisation_report.md",
        docs_dir / "control" / "phase2_tvlqr_ocp30_boundary.md",
    )
    prior_notes = [
        f"- `{path.as_posix()}`: {'found' if path.exists() else 'not found in this checkout'}"
        for path in prior_paths
    ]
    best_30 = _best_30_row(best_rows)
    phase2_summary = _phase2_gate_summary(best_30, replay_rows)
    previous_summary = previous.get("summary", {})
    previous_best = previous.get("best_30")
    report = [
        "# Phase 2 TVLQR OCP30 Latency And Recovery Robustness Report",
        "",
        f"Seed: `{int(seed)}`",
        f"Output root: `{paths.root.as_posix()}`",
        "",
        "## Scope",
        "",
        "Phase 2 robustness only: W0 30 deg OCP reproduction, deterministic latency/recovery candidate variants, TrajectoryPrimitive conversion, open-loop replay, closed-loop TVLQR replay, nominal-latency replay, and terminal recovery sensitivity.",
        "",
        "## Prior Boundary Evidence",
        "",
        *prior_notes,
        "",
        "## Previous Phase 2 Debug Evidence",
        "",
        f"- previous output root: `{previous['output_root']}`",
        f"- previous metrics found: `{previous['found']}`",
        f"- previous phase 2 status: `{previous_summary.get('phase2_status', '')}`",
        f"- previous active failure class: `{previous_summary.get('active_failure_class', '')}`",
        f"- previous nominal-latency gate: `{previous_summary.get('closed_loop_nominal_latency', '')}`",
        f"- previous terminal recovery gate: `{previous_summary.get('terminal_altitude_sensitivity', '')}`",
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
                f"- candidate variant: `{best_30.get('candidate_variant')}`",
            ]
        )
    if isinstance(previous_best, dict):
        report.extend(
            [
                "",
                "## Before/After Phase 2 Status",
                "",
                f"- previous selected variant: `{previous_best.get('candidate_variant', 'baseline')}`",
                f"- previous limitation: `{previous_summary.get('limitation', '')}`",
                f"- current selected variant: `{best_30.get('candidate_variant') if best_30 else ''}`",
                f"- current active failure class: `{phase2_summary['active_failure_class']}`",
                f"- current phase 2 status: `{phase2_summary['phase2_status']}`",
            ]
        )
    report.extend(
        [
            "",
            "## Phase 2 Gate Summary",
            "",
            f"Replay rows produced: `{len(replay_rows)}`",
            f"- hard 30 deg OCP reproduced: `{phase2_summary['ocp_hard_30']}`",
            f"- open-loop no-latency gate: `{phase2_summary['open_loop_no_latency']}`",
            f"- closed-loop no-latency gate: `{phase2_summary['closed_loop_no_latency']}`",
            f"- nominal-latency gate: `{phase2_summary['closed_loop_nominal_latency']}`",
            f"- terminal-altitude recovery sensitivity gate: `{phase2_summary['terminal_altitude_sensitivity']}`",
            f"- phase 2 status: `{phase2_summary['phase2_status']}`",
            f"- active failure class: `{phase2_summary['active_failure_class']}`",
            f"- all failure classes: `{phase2_summary['all_failure_classes']}`",
            f"- limitation: `{phase2_summary['limitation']}`",
            "",
            "## Metrics Paths",
            "",
            f"- `{(paths.metrics_dir / _metrics_name('turn_ocp_candidates', seed)).as_posix()}`",
            f"- `{(paths.metrics_dir / _metrics_name('turn_ocp_best_by_target', seed)).as_posix()}`",
            f"- `{(paths.metrics_dir / _metrics_name('turn_tvlqr_replay', seed)).as_posix()}`",
            "",
            "## Limitation",
            "",
            "This report is simulation-only and does not include Phase 3/4 continuation, entry sweeps, W0-W3 stress, outer-loop simulation, or hardware/Vicon execution. A promoted Phase 2 result is still a simulation primitive candidate, not a hardware or Vicon claim.",
            "",
        ]
    )
    report_dir = docs_dir / "control"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase2_tvlqr_ocp30_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    boundary_intro = (
        "The W0 30 deg OCP candidate was promoted by the Phase 2 latency/recovery gates; "
        "the prior boundary is superseded by the current evidence."
        if phase2_summary["phase2_status"] == "promoted_phase2"
        else "The W0 30 deg OCP candidate was not promoted beyond Phase 2."
    )
    boundary = [
        "# Phase 2 TVLQR OCP30 Boundary Report",
        "",
        boundary_intro,
        "",
        f"- phase 2 status: `{phase2_summary['phase2_status']}`",
        f"- active failure class: `{phase2_summary['active_failure_class']}`",
        f"- all failure classes: `{phase2_summary['all_failure_classes']}`",
        f"- limitation: `{phase2_summary['limitation']}`",
        f"- hard 30 deg OCP reproduced: `{phase2_summary['ocp_hard_30']}`",
        f"- open-loop no-latency gate: `{phase2_summary['open_loop_no_latency']}`",
        f"- closed-loop no-latency gate: `{phase2_summary['closed_loop_no_latency']}`",
        f"- nominal-latency gate: `{phase2_summary['closed_loop_nominal_latency']}`",
        f"- terminal recovery sensitivity gate: `{phase2_summary['terminal_altitude_sensitivity']}`",
        "",
        "No physical sign, command-order, state-order, arena-bound, or command-authority changes were made to force promotion.",
        "",
    ]
    if best_30 is not None:
        boundary.extend(
            [
                "## Best 30 Deg OCP Row",
                "",
                f"- candidate variant: `{best_30.get('candidate_variant')}`",
                f"- label: `{best_30.get('feasibility_label')}`",
                f"- failure reason: `{best_30.get('failure_reason')}`",
                f"- directed heading change deg: `{best_30.get('directed_heading_change_deg')}`",
                f"- heading threshold deg: `{best_30.get('heading_threshold_deg')}`",
                f"- min wall distance m: `{best_30.get('min_wall_distance_m')}`",
                f"- terminal altitude m: `{best_30.get('terminal_z_w_m')}`",
                f"- terminal alpha deg: `{best_30.get('terminal_abs_alpha_deg')}`",
                f"- terminal beta deg: `{best_30.get('terminal_abs_beta_deg')}`",
                f"- terminal rate norm rad/s: `{best_30.get('terminal_rate_norm_rad_s')}`",
                f"- max alpha deg: `{best_30.get('max_alpha_deg')}`",
                f"- saturation fraction: `{best_30.get('saturation_fraction')}`",
                f"- slack max: `{best_30.get('slack_max')}`",
                "",
            ]
        )
    (report_dir / "phase2_tvlqr_ocp30_boundary.md").write_text(
        "\n".join(boundary),
        encoding="utf-8",
    )


def _best_30_row(best_rows: list[dict[str, object]]) -> dict[str, object] | None:
    return next(
        (row for row in best_rows if float(row.get("target_requested_deg", -1.0)) == 30.0),
        None,
    )


def _phase2_gate_summary(
    best_30: dict[str, object] | None,
    replay_rows: list[dict[str, object]],
) -> dict[str, object]:
    replay_rows = _rows_for_best_candidate(best_30, replay_rows)
    ocp_ok = _ocp_30_gate(best_30)
    open_loop = _single_replay_gate(replay_rows, "open_loop_no_latency")
    closed = _single_replay_gate(replay_rows, "closed_loop_no_latency")
    nominal = _single_replay_gate(replay_rows, "closed_loop_nominal_latency")
    sensitivity = _terminal_sensitivity_gate(replay_rows)
    failure_classes = _phase2_failure_classes(
        ocp_ok=ocp_ok,
        open_loop=open_loop,
        closed=closed,
        nominal=nominal,
        sensitivity=sensitivity,
        replay_rows=replay_rows,
    )
    promoted = (
        ocp_ok is True
        and open_loop is True
        and closed is True
        and nominal is True
        and sensitivity is True
    )
    failed = [
        str(row.get("termination_reason"))
        for row in replay_rows
        if not _row_success(row) and str(row.get("termination_reason"))
    ]
    return {
        "ocp_hard_30": ocp_ok,
        "open_loop_no_latency": open_loop,
        "closed_loop_no_latency": closed,
        "closed_loop_nominal_latency": nominal,
        "terminal_altitude_sensitivity": sensitivity,
        "phase2_status": "promoted_phase2" if promoted else "boundary_only",
        "active_failure_class": "" if promoted else failure_classes[0],
        "all_failure_classes": ";".join(failure_classes),
        "limitation": "; ".join(dict.fromkeys(failed)) if failed else "",
    }


def _ocp_30_gate(best_30: dict[str, object] | None) -> bool:
    if best_30 is None:
        return False
    label = str(best_30.get("feasibility_label", ""))
    return (
        label in ACCEPTED_LABELS
        and _as_bool(best_30.get("success", False))
        and float(best_30.get("slack_max", 1.0)) <= 1e-8
        and _as_bool(best_30.get("heading_gate_passed", False))
        and _as_bool(best_30.get("inside_true_safety_volume", False))
        and _as_bool(best_30.get("exit_recoverable_gate", False))
    )


def _single_replay_gate(rows: list[dict[str, object]], kind: str) -> bool | str:
    matches = [row for row in rows if str(row.get("replay_kind")) == kind]
    if not matches:
        return "not_run"
    return all(_row_success(row) for row in matches)


def _terminal_sensitivity_gate(rows: list[dict[str, object]]) -> bool | str:
    matches = [row for row in rows if str(row.get("replay_kind")) == "terminal_altitude_sensitivity"]
    if not matches:
        return "not_run"
    altitudes = {
        round(float(row.get("terminal_altitude_min_m", -1.0)), 2)
        for row in matches
    }
    required = {0.75, 1.0, 1.2}
    return required.issubset(altitudes) and all(_row_success(row) for row in matches)


def _row_success(row: dict[str, object]) -> bool:
    return _as_bool(row.get("success", False))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _rows_for_best_candidate(
    best_30: dict[str, object] | None,
    replay_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if best_30 is None:
        return replay_rows
    variant = str(best_30.get("candidate_variant", "")).strip()
    tag = str(best_30.get("candidate_tag", "")).strip()
    if tag:
        filtered = [row for row in replay_rows if str(row.get("candidate_tag", "")).strip() == tag]
        return filtered or replay_rows
    if variant:
        filtered = [
            row
            for row in replay_rows
            if str(row.get("candidate_variant", "")).strip() == variant
        ]
        return filtered or replay_rows
    return replay_rows


def _phase2_failure_classes(
    *,
    ocp_ok: bool,
    open_loop: bool | str,
    closed: bool | str,
    nominal: bool | str,
    sensitivity: bool | str,
    replay_rows: list[dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    if not ocp_ok:
        failures.append("ocp_regression")
    if open_loop is not True:
        failures.append("open_loop_replay_mismatch")
    if open_loop is True and closed is not True:
        failures.append("tvlqr_feedback_limited")
    if open_loop is True and closed is True and nominal is not True:
        failures.append(_latency_failure_class(replay_rows))
    if sensitivity is not True:
        failures.append("terminal_recovery_limited")
    return list(dict.fromkeys(failures or ["unknown_phase2_failure"]))


def _latency_failure_class(replay_rows: list[dict[str, object]]) -> str:
    rows = [
        row
        for row in replay_rows
        if str(row.get("replay_kind")) == "closed_loop_nominal_latency"
    ]
    text = " ".join(
        f"{row.get('termination_reason', '')} {row.get('failure_class', '')} "
        f"{row.get('feasibility_label', '')}"
        for row in rows
    ).lower()
    if "alpha" in text or "model_limited_high_alpha" in text:
        return "latency_limited_high_alpha"
    return "latency_limited_timing"


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
