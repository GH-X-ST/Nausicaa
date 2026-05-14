from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup and search config
# 2) Template family generation
# 3) Candidate evaluation and scoring
# 4) Search runner and output writers
# 5) CLI entry point
# =============================================================================

# =============================================================================
# 1) Import Path Setup and Search Config
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
from optimise_template import (  # noqa: E402
    AgileTurnTemplate,
    agile_phase_metadata,
    agile_template_to_dict,
    build_agile_reversal_candidate,
)
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, simulate_primitive, write_log  # noqa: E402
from scenarios import arena_feasible_entry_state  # noqa: E402


TARGETS_DEG = (30.0, 60.0, 90.0, 120.0, 180.0)
FAMILIES = (
    "high_bank_roll_recovery",
    "brake_roll_yaw_recovery",
    "pitch_up_redirect_recovery",
)


@dataclass(frozen=True)
class SearchOutputPaths:
    root: Path
    metrics_dir: Path
    logs_dir: Path
    manifest_dir: Path


# =============================================================================
# 2) Template Family Generation
# =============================================================================
def agile_search_templates(
    targets_deg: tuple[float, ...] = TARGETS_DEG,
) -> tuple[AgileTurnTemplate, ...]:
    templates: list[AgileTurnTemplate] = []
    for target in targets_deg:
        templates.extend(_target_templates(float(target)))
    return tuple(templates)


def _target_templates(target_heading_deg: float) -> tuple[AgileTurnTemplate, ...]:
    scale = min(max(float(target_heading_deg) / 30.0, 1.0), 3.0)
    roll_base = min(0.34 + 0.08 * (scale - 1.0), 0.58)
    yaw_base = min(0.36 + 0.09 * (scale - 1.0), 0.64)
    hold_base = min(0.06 + 0.04 * (scale - 1.0), 0.16)
    recover_base = min(0.18 + 0.04 * (scale - 1.0), 0.30)
    target_tag = f"{int(round(target_heading_deg)):03d}"
    return (
        AgileTurnTemplate(
            elevator_brake_norm=0.0,
            aileron_roll_norm=1.0,
            rudder_yaw_norm=0.55,
            t_e_start_s=0.08,
            t_a_start_s=0.08,
            t_r_start_s=0.10,
            t_e_duration_s=0.0,
            t_a_duration_s=roll_base,
            t_r_duration_s=yaw_base,
            hold_duration_s=hold_base,
            recover_duration_s=recover_base,
            target_heading_deg=target_heading_deg,
            family="high_bank_roll_recovery",
            candidate_id=f"{target_tag}_a",
        ),
        AgileTurnTemplate(
            elevator_brake_norm=-1.0,
            aileron_roll_norm=1.0,
            rudder_yaw_norm=1.0,
            t_e_start_s=0.04,
            t_a_start_s=0.10,
            t_r_start_s=0.10,
            t_e_duration_s=min(0.08 + 0.02 * (scale - 1.0), 0.14),
            t_a_duration_s=roll_base + 0.08,
            t_r_duration_s=yaw_base + 0.08,
            hold_duration_s=hold_base,
            recover_duration_s=recover_base,
            target_heading_deg=target_heading_deg,
            family="brake_roll_yaw_recovery",
            candidate_id=f"{target_tag}_a",
        ),
        AgileTurnTemplate(
            elevator_brake_norm=1.0,
            aileron_roll_norm=1.0,
            rudder_yaw_norm=0.85,
            t_e_start_s=0.04,
            t_a_start_s=0.12,
            t_r_start_s=0.12,
            t_e_duration_s=min(0.10 + 0.03 * (scale - 1.0), 0.18),
            t_a_duration_s=roll_base + 0.10,
            t_r_duration_s=yaw_base + 0.10,
            hold_duration_s=hold_base + 0.02,
            recover_duration_s=recover_base + 0.04,
            target_heading_deg=target_heading_deg,
            family="pitch_up_redirect_recovery",
            candidate_id=f"{target_tag}_a",
        ),
    )


# =============================================================================
# 3) Candidate Evaluation and Scoring
# =============================================================================
def _evaluate_template(
    template: AgileTurnTemplate,
    seed: int,
    paths: SearchOutputPaths,
) -> dict[str, object]:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(
        x_trim=linear_model.x_trim,
        u_trim=linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    x0 = arena_feasible_entry_state(linear_model.x_trim)
    x0[0] = 1.45
    command_config = CommandToSurfaceConfig(mode="nominal")
    command_layer = CommandToSurfaceLayer(config=command_config, envelope=LatencyEnvelope())
    primitive = build_agile_reversal_candidate(
        template=template,
        x0=x0,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="panel",
        command_layer=command_layer,
    )
    target_tag = f"{int(round(float(template.target_heading_deg or 0.0))):03d}"
    log_name = f"{template.candidate_id}_seed{int(seed)}.csv"
    log_path = paths.logs_dir / f"target_{target_tag}" / template.family / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = simulate_primitive(
        scenario_id=f"s9_search_target_{target_tag}_{template.family}",
        seed=seed,
        primitive=primitive,
        x0=x0,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_model_name="none",
        wind_mode="panel",
        command_layer=CommandToSurfaceLayer(
            config=command_config,
            envelope=LatencyEnvelope(),
        ),
        log_path=log_path,
        repo_root=REPO_ROOT,
        rollout_config=RolloutConfig(),
        arena_config=ArenaConfig(),
        wind_param_label="none",
    )
    write_log(result, log_path)
    row = _candidate_row(
        template=template,
        primitive_name=primitive.name,
        metrics=dict(result.metrics),
        log_path=log_path,
    )
    row["finite_trajectory_arrays"] = bool(
        np.all(np.isfinite(primitive.times_s))
        and np.all(np.isfinite(primitive.x_ref))
        and np.all(np.isfinite(primitive.u_ff))
        and np.all(np.isfinite(primitive.a_mats))
        and np.all(np.isfinite(primitive.b_mats))
        and np.all(np.isfinite(primitive.k_lqr))
        and np.all(np.isfinite(primitive.s_mats))
    )
    return row


def _candidate_row(
    template: AgileTurnTemplate,
    primitive_name: str,
    metrics: dict[str, object],
    log_path: Path,
) -> dict[str, object]:
    target = float(template.target_heading_deg or 0.0)
    heading = abs(float(metrics.get("actual_heading_change_deg") or 0.0))
    heading_error = abs(target - heading)
    min_wall = _float_or(metrics.get("min_wall_distance_m"), -1.0)
    height_change = _float_or(metrics.get("height_change_m"), 0.0)
    terminal_speed = _float_or(metrics.get("terminal_speed_m_s"), 99.0)
    max_alpha = _float_or(metrics.get("max_alpha_deg"), 99.0)
    saturation_fraction = _float_or(metrics.get("saturation_fraction"), 1.0)
    exit_recoverable = bool(metrics.get("exit_recoverable", False))
    success = bool(metrics.get("success", False))
    feasibility_label = str(metrics.get("feasibility_label") or "not_tested")
    score = _score_candidate(
        target_heading_deg=target,
        actual_heading_deg=heading,
        min_wall_distance_m=min_wall,
        height_change_m=height_change,
        terminal_speed_m_s=terminal_speed,
        max_alpha_deg=max_alpha,
        saturation_fraction=saturation_fraction,
        exit_recoverable=exit_recoverable,
        success=success,
    )
    failure_reason = _failure_reason(metrics, feasibility_label)
    row = {
        "target_heading_deg": target,
        "family": template.family,
        "candidate_id": template.candidate_id,
        "primitive_name": primitive_name,
        "score": score,
        "success": success,
        "feasibility_label": feasibility_label,
        "failure_reason": failure_reason,
        "actual_heading_change_deg": float(metrics.get("actual_heading_change_deg") or 0.0),
        "heading_error_deg": heading_error,
        "min_wall_distance_m": min_wall,
        "height_change_m": height_change,
        "terminal_speed_m_s": terminal_speed,
        "max_alpha_deg": max_alpha,
        "saturation_fraction": saturation_fraction,
        "exit_recoverable": exit_recoverable,
        "termination_reason": str(metrics.get("termination_reason") or ""),
        "phase_metadata_json": json.dumps(agile_phase_metadata(template), sort_keys=True),
        "template_json": json.dumps(agile_template_to_dict(template), sort_keys=True),
        "log_path": _relative_repo_path(log_path),
    }
    row.update(
        {
            f"template_{key}": value
            for key, value in agile_template_to_dict(template).items()
        }
    )
    return row


def _score_candidate(
    target_heading_deg: float,
    actual_heading_deg: float,
    min_wall_distance_m: float,
    height_change_m: float,
    terminal_speed_m_s: float,
    max_alpha_deg: float,
    saturation_fraction: float,
    exit_recoverable: bool,
    success: bool,
) -> float:
    heading_reward = 2.0 * actual_heading_deg
    heading_error_penalty = 3.0 * abs(target_heading_deg - actual_heading_deg)
    wall_risk_penalty = 140.0 * max(0.05 - min_wall_distance_m, 0.0)
    height_loss_penalty = 12.0 * max(-height_change_m, 0.0)
    terminal_speed_penalty = 4.0 * abs(terminal_speed_m_s - 6.5)
    high_alpha_penalty = 2.0 * max(max_alpha_deg - 20.0, 0.0)
    saturation_penalty = 8.0 * saturation_fraction
    unrecoverable_penalty = 50.0 if not exit_recoverable else 0.0
    unsafe_penalty = 100.0 if not success else 0.0
    return float(
        heading_reward
        - heading_error_penalty
        - wall_risk_penalty
        - height_loss_penalty
        - terminal_speed_penalty
        - high_alpha_penalty
        - saturation_penalty
        - unrecoverable_penalty
        - unsafe_penalty
    )


def _failure_reason(metrics: dict[str, object], feasibility_label: str) -> str:
    if feasibility_label == "fixed_start_feasible":
        return ""
    reason = str(metrics.get("termination_reason") or "")
    if reason:
        return reason
    if not bool(metrics.get("exit_recoverable", False)):
        return "terminal state outside recoverable exit bounds"
    if feasibility_label == "fixed_start_safe_but_under_turning":
        return "heading target not achieved"
    return feasibility_label


def _float_or(value: object, default: float) -> float:
    if value in {None, ""}:
        return float(default)
    value_float = float(value)
    return value_float if np.isfinite(value_float) else float(default)


def _relative_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# =============================================================================
# 4) Search Runner and Output Writers
# =============================================================================
def run_agile_template_search(
    seed: int,
    output_root: str | Path | None = None,
    targets_deg: tuple[float, ...] = TARGETS_DEG,
) -> list[dict[str, object]]:
    paths = _output_paths(output_root)
    candidates = [
        _evaluate_template(template, seed=seed, paths=paths)
        for template in agile_search_templates(targets_deg=targets_deg)
    ]
    best_rows = _best_by_target(candidates)
    selected_templates = {
        f"{int(round(float(row['target_heading_deg']))):03d}": json.loads(
            str(row["template_json"])
        )
        for row in best_rows
    }
    _write_rows(
        paths.metrics_dir / f"agile_template_search_candidates_seed{int(seed)}.csv",
        candidates,
    )
    _write_rows(
        paths.metrics_dir / f"agile_template_search_best_by_target_seed{int(seed)}.csv",
        best_rows,
    )
    summary = _summary_rows(seed=seed, candidates=candidates, best_rows=best_rows)
    _write_rows(
        paths.metrics_dir / f"agile_template_search_summary_seed{int(seed)}.csv",
        summary,
    )
    _write_manifest(
        paths=paths,
        seed=seed,
        candidates=candidates,
        best_rows=best_rows,
        selected_templates=selected_templates,
    )
    return candidates


def _output_paths(output_root: str | Path | None) -> SearchOutputPaths:
    root = (
        Path(output_root)
        if output_root is not None
        else REPO_ROOT
        / "03_Control"
        / "05_Results"
        / "03_primitives"
        / "06_agile_template_search"
        / "001"
    )
    return SearchOutputPaths(
        root=root,
        metrics_dir=root / "metrics",
        logs_dir=root / "logs",
        manifest_dir=root / "manifests",
    )


def _best_by_target(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    targets = sorted({float(row["target_heading_deg"]) for row in rows})
    for target in targets:
        target_rows = [row for row in rows if float(row["target_heading_deg"]) == target]
        feasible = [
            row
            for row in target_rows
            if row.get("feasibility_label") == "fixed_start_feasible"
        ]
        pool = feasible or target_rows
        selected = max(pool, key=lambda row: float(row["score"]))
        best = dict(selected)
        best["selected_for_target"] = True
        best_rows.append(best)
    return best_rows


def _summary_rows(
    seed: int,
    candidates: list[dict[str, object]],
    best_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    passed_30 = any(
        float(row["target_heading_deg"]) == 30.0
        and row["feasibility_label"] == "fixed_start_feasible"
        and abs(float(row["actual_heading_change_deg"])) >= 24.0
        and bool(row["success"])
        and bool(row["exit_recoverable"])
        and float(row["min_wall_distance_m"]) > 0.0
        for row in best_rows
    )
    return [
        {
            "seed": int(seed),
            "candidate_count": len(candidates),
            "target_count": len(best_rows),
            "families": ";".join(FAMILIES),
            "target_headings_deg": ";".join(
                f"{target:.0f}" for target in TARGETS_DEG
            ),
            "gate_30_deg_passed": bool(passed_30),
        }
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    paths: SearchOutputPaths,
    seed: int,
    candidates: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    selected_templates: dict[str, dict[str, object]],
) -> None:
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": int(seed),
        "candidate_count": len(candidates),
        "families": list(FAMILIES),
        "targets_deg": list(TARGETS_DEG),
        "selected_templates": selected_templates,
        "best_by_target": best_rows,
    }
    path = paths.manifest_dir / f"agile_template_search_seed{int(seed)}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# =============================================================================
# 5) CLI Entry Point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    rows = run_agile_template_search(seed=args.seed, output_root=args.output_root)
    print("agile template search complete")
    print(f"candidate_count: {len(rows)}")
    print(f"output_root: {_output_paths(args.output_root).root}")


if __name__ == "__main__":
    main()
