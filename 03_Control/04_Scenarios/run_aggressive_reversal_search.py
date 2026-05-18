from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggressive_reversal_ocp import (
    AGGRESSIVE_CAMPAIGN,
    REPLAY_DEFECT_TOL,
    SEED_FAMILIES,
    AggressiveReversalOcpConfig,
    AggressiveReversalOcpResult,
    limiting_mechanism,
    solve_aggressive_reversal_ocp,
    target_config,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from logging_contract import command_dataframe, trajectory_dataframe
from metric_contract import empty_metric_row
from result_paths import make_result_tree


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and helpers
# 2) Output writing
# 3) Search runner
# 4) CLI
# =============================================================================


# =============================================================================
# 1) Constants and Helpers
# =============================================================================
DEFAULT_RESULTS_ROOT = CONTROL_DIR / "05_Results"
DEFAULT_TARGETS_DEG = (15.0, 30.0, 60.0, 90.0, 120.0, 180.0)
VALIDATION_COMMANDS = (
    "python -m py_compile "
    "03_Control/03_Primitives/aggressive_reversal_ocp.py "
    "03_Control/03_Primitives/aggressive_reversal_primitive.py "
    "03_Control/04_Scenarios/run_aggressive_reversal_search.py",
    "python 03_Control/04_Scenarios/run_aggressive_reversal_search.py "
    "--run-id 1 --overwrite",
    "python -m pytest -q "
    "03_Control/tests/test_aggressive_reversal_shapes.py "
    "03_Control/tests/test_aggressive_reversal_smoke.py "
    "03_Control/tests/test_aggressive_reversal_target_ladder.py",
    "python -m pytest -q 03_Control/tests",
)


@dataclass(frozen=True)
class AggressiveSearchOutputs:
    root: Path
    summary_csv: Path
    manifest_json: Path
    report_md: Path


def _target_token(target_heading_deg: float) -> str:
    return f"{int(round(float(target_heading_deg))):03d}"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _suffix(run_id: int) -> str:
    return f"s{int(run_id):03d}"


def _json_float(value: object) -> float | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _target_metric_row(
    result: AggressiveReversalOcpResult,
    run_id: int,
) -> dict[str, object]:
    metrics = result.metrics
    row = empty_metric_row(include_agile=True)
    row.update(
        {
            "run_id": _suffix(run_id),
            "seed": 1,
            "primitive_name": f"aggressive_reversal_{_target_token(result.target_heading_deg)}",
            "primitive_family": "agile_reversal",
            "scenario_name": "aggressive_reversal_full_ladder_w0",
            "wind_mode": "none",
            "latency_case": "none",
            "success": bool(metrics.get("success", False)),
            "finite_state_success": bool(metrics.get("finite_state_success", False)),
            "rollout_success": bool(metrics.get("rollout_success", False)),
            "primitive_success": bool(metrics.get("primitive_success", False)),
            "closed_loop_replay_success": bool(metrics.get("replay_finite", False)),
            "failure_label": str(metrics.get("failure_label", result.failure_label)),
            "duration_s": float(result.time_s[-1] - result.time_s[0]),
            "initial_speed_m_s": float(np.linalg.norm(result.x_ref[0, 6:9])),
            "terminal_speed_m_s": float(metrics.get("terminal_speed_m_s", np.nan)),
            "height_change_m": float(metrics.get("height_change_m", np.nan)),
            "min_true_wall_margin_m": float(metrics.get("min_true_wall_margin_m", np.nan)),
            "min_floor_margin_m": float(metrics.get("min_floor_margin_m", np.nan)),
            "min_ceiling_margin_m": float(metrics.get("min_ceiling_margin_m", np.nan)),
            "max_alpha_deg": float(metrics.get("max_alpha_deg", np.nan)),
            "max_beta_deg": float(metrics.get("max_beta_deg", np.nan)),
            "max_bank_deg": float(metrics.get("max_bank_deg", np.nan)),
            "max_pitch_deg": float(metrics.get("max_pitch_deg", np.nan)),
            "max_rate_rad_s": float(metrics.get("max_rate_rad_s", np.nan)),
            "saturation_fraction": float(metrics.get("saturation_fraction", np.nan)),
            "notes": str(metrics.get("notes", result.notes)),
            "target_heading_deg": float(result.target_heading_deg),
            "actual_heading_change_deg": float(metrics.get("actual_heading_change_deg", np.nan)),
            "forward_travel_m": float(metrics.get("forward_travel_m", np.nan)),
            "turn_volume_proxy_m2": float(metrics.get("turn_volume_proxy_m2", np.nan)),
            "exit_recoverable": bool(metrics.get("recoverable", False)),
            "source_trajectory_success": bool(metrics.get("source_trajectory_success", False)),
            "gain_construction_success": bool(metrics.get("phase_search_attempted", False)),
            "saturation_time_s": float(metrics.get("saturation_time_s", np.nan)),
        }
    )
    return row


def _summary_row(
    result: AggressiveReversalOcpResult,
    run_id: int,
    trajectory_path: Path,
    command_path: Path,
    metric_path: Path,
) -> dict[str, object]:
    metrics = result.metrics
    return {
        "run_id": _suffix(run_id),
        "target_heading_deg": float(result.target_heading_deg),
        "direction_sign": int(result.direction_sign),
        "success": bool(metrics.get("success", False)),
        "primitive_success": bool(metrics.get("primitive_success", False)),
        "failure_label": str(metrics.get("failure_label", result.failure_label)),
        "notes": str(metrics.get("notes", result.notes)),
        "actual_heading_change_deg": float(metrics.get("actual_heading_change_deg", np.nan)),
        "heading_error_deg": float(metrics.get("heading_error_deg", np.nan)),
        "terminal_speed_m_s": float(metrics.get("terminal_speed_m_s", np.nan)),
        "min_true_wall_margin_m": float(metrics.get("min_true_wall_margin_m", np.nan)),
        "min_floor_margin_m": float(metrics.get("min_floor_margin_m", np.nan)),
        "min_ceiling_margin_m": float(metrics.get("min_ceiling_margin_m", np.nan)),
        "saturation_fraction": float(metrics.get("saturation_fraction", np.nan)),
        "families_attempted": str(metrics.get("families_attempted", "")),
        "selected_family": str(metrics.get("selected_family", "")),
        "selected_method": str(metrics.get("selected_method", "")),
        "next_family_reason": str(metrics.get("next_family_reason", "")),
        "limiting_mechanism": str(metrics.get("limiting_mechanism", "")),
        "best_finite_candidate": str(metrics.get("best_finite_candidate", "")),
        "best_recoverable_candidate": str(metrics.get("best_recoverable_candidate", "")),
        "best_successful_candidate": str(metrics.get("best_successful_candidate", "")),
        "nlp_constructed": bool(metrics.get("nlp_constructed", False)),
        "ipopt_called": bool(metrics.get("ipopt_called", False)),
        "solver_status": str(metrics.get("solver_status", "")),
        "solver_iter_count": int(metrics.get("solver_iter_count", 0)),
        "solver_objective": float(metrics.get("solver_objective", np.nan)),
        "constraint_residual_max": float(metrics.get("constraint_residual_max", np.nan)),
        "replay_defect_max": float(metrics.get("replay_defect_max", np.nan)),
        "direct_ocp_attempted": bool(metrics.get("direct_ocp_attempted", False)),
        "direct_ocp_converged": bool(metrics.get("direct_ocp_converged", False)),
        "phase_search_attempted": bool(metrics.get("phase_search_attempted", False)),
        "replay_finite": bool(metrics.get("replay_finite", False)),
        "recoverable": bool(metrics.get("recoverable", False)),
        "energy_exploitation": bool(metrics.get("energy_exploitation", False)),
        "trajectory_csv": _repo_relative(trajectory_path),
        "commands_csv": _repo_relative(command_path),
        "metrics_csv": _repo_relative(metric_path),
    }


def _failure_result_from_exception(
    target_heading_deg: float,
    exc: Exception,
    base_config: AggressiveReversalOcpConfig,
) -> AggressiveReversalOcpResult:
    time_s = np.array([0.0, base_config.dt_s], dtype=float)
    x_ref = np.full((time_s.size, 15), np.nan, dtype=float)
    u_norm = np.zeros((time_s.size, 3), dtype=float)
    metrics = {
        "target_heading_deg": float(target_heading_deg),
        "actual_heading_change_deg": 0.0,
        "heading_error_deg": float(target_heading_deg),
        "forward_travel_m": np.nan,
        "turn_volume_proxy_m2": np.nan,
        "height_change_m": np.nan,
        "max_altitude_gain_m": np.nan,
        "speed_min_m_s": np.nan,
        "terminal_speed_m_s": np.nan,
        "max_alpha_deg": np.nan,
        "max_beta_deg": np.nan,
        "max_bank_deg": np.nan,
        "max_pitch_deg": np.nan,
        "max_rate_rad_s": np.nan,
        "min_true_wall_margin_m": np.nan,
        "min_floor_margin_m": np.nan,
        "min_ceiling_margin_m": np.nan,
        "saturation_fraction": 0.0,
        "saturation_time_s": 0.0,
        "finite_state_success": False,
        "rollout_success": False,
        "source_trajectory_success": False,
        "terminal_recoverable_proxy": False,
        "recoverable": False,
        "success": False,
        "primitive_success": False,
        "failure_label": "solver_failure",
        "notes": f"runner_exception:{type(exc).__name__}",
        "families_attempted": ";".join(SEED_FAMILIES),
        "selected_family": "",
        "selected_method": "runner_failure",
        "next_family_reason": "solver_failure_retry_best_finite_phase_search",
        "limiting_mechanism": limiting_mechanism("solver_failure"),
        "best_finite_candidate": "",
        "best_recoverable_candidate": "",
        "best_successful_candidate": "",
        "nlp_constructed": False,
        "ipopt_called": False,
        "solver_status": f"runner_exception:{type(exc).__name__}",
        "solver_iter_count": 0,
        "solver_objective": np.nan,
        "constraint_residual_max": np.nan,
        "replay_defect_max": np.nan,
        "direct_ocp_attempted": False,
        "direct_ocp_converged": False,
        "phase_search_attempted": False,
        "replay_finite": False,
        "energy_exploitation": False,
    }
    return AggressiveReversalOcpResult(
        target_heading_deg=float(target_heading_deg),
        direction_sign=base_config.direction_sign,
        success=False,
        failure_label="solver_failure",
        time_s=time_s,
        x_ref=x_ref,
        u_ff_norm=u_norm,
        u_norm_applied=u_norm,
        delta_cmd_rad=u_norm,
        phase=("entry", "entry"),
        metrics=metrics,
        notes=str(metrics["notes"]),
    )


def _largest_true(summary: list[dict[str, object]], key: str) -> float:
    values = [float(row["target_heading_deg"]) for row in summary if bool(row[key])]
    return max(values) if values else 0.0


# =============================================================================
# 2) Output Writing
# =============================================================================
def _write_target_outputs(
    result: AggressiveReversalOcpResult,
    paths: dict[str, Path],
    run_id: int,
) -> dict[str, Path]:
    token = _target_token(result.target_heading_deg)
    suffix = _suffix(run_id)
    trajectory_path = (
        paths["metrics"] / f"aggressive_reversal_target_{token}_trajectory_{suffix}.csv"
    )
    command_path = (
        paths["metrics"] / f"aggressive_reversal_target_{token}_commands_{suffix}.csv"
    )
    metric_path = (
        paths["metrics"] / f"aggressive_reversal_target_{token}_metrics_{suffix}.csv"
    )
    trajectory_dataframe(result.time_s, result.x_ref).to_csv(trajectory_path, index=False)
    command_dataframe(
        result.time_s,
        result.u_ff_norm,
        result.u_norm_applied,
        result.delta_cmd_rad,
    ).to_csv(command_path, index=False)
    pd.DataFrame([_target_metric_row(result, run_id)]).to_csv(metric_path, index=False)
    return {
        "trajectory_csv": trajectory_path,
        "commands_csv": command_path,
        "metrics_csv": metric_path,
    }


def _write_summary(
    rows: list[dict[str, object]],
    paths: dict[str, Path],
    run_id: int,
) -> Path:
    path = paths["metrics"] / f"aggressive_reversal_target_summary_{_suffix(run_id)}.csv"
    pd.DataFrame(rows).sort_values("target_heading_deg").to_csv(path, index=False)
    return path


def _manifest(
    rows: list[dict[str, object]],
    output_files: dict[str, Path],
    run_id: int,
    requested_targets: tuple[float, ...],
    skipped_targets: tuple[float, ...],
) -> dict[str, Any]:
    target_outcomes = {
        str(int(round(float(row["target_heading_deg"])))): {
            "target_heading_deg": float(row["target_heading_deg"]),
            "success": bool(row["success"]),
            "primitive_success": bool(row["primitive_success"]),
            "failure_label": str(row["failure_label"]),
            "families_attempted": str(row["families_attempted"]),
            "selected_family": str(row["selected_family"]),
            "selected_method": str(row["selected_method"]),
            "limiting_mechanism": str(row["limiting_mechanism"]),
            "nlp_constructed": bool(row["nlp_constructed"]),
            "ipopt_called": bool(row["ipopt_called"]),
            "solver_status": str(row["solver_status"]),
            "constraint_residual_max": _json_float(row["constraint_residual_max"]),
            "replay_defect_max": _json_float(row["replay_defect_max"]),
            "direct_ocp_attempted": bool(row["direct_ocp_attempted"]),
            "direct_ocp_converged": bool(row["direct_ocp_converged"]),
            "replay_finite": bool(row["replay_finite"]),
            "recoverable": bool(row["recoverable"]),
        }
        for row in rows
    }
    any_attempt = any(bool(row["direct_ocp_attempted"]) for row in rows)
    any_success = any(bool(row["success"]) for row in rows)
    any_recoverable = any(bool(row["recoverable"]) for row in rows)
    any_finite = any(bool(row["replay_finite"]) for row in rows)
    overall_status = "pass" if any_success else "boundary_evidence"
    if not any_finite:
        overall_status = "needs_review"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": _suffix(run_id),
        "campaign": AGGRESSIVE_CAMPAIGN,
        "overall_status": overall_status,
        "targets_requested_deg": list(requested_targets),
        "targets_completed_deg": sorted(float(row["target_heading_deg"]) for row in rows),
        "targets_skipped_by_resume_deg": list(skipped_targets),
        "target_outcomes": target_outcomes,
        "largest_finite_target_deg": _largest_true(rows, "replay_finite"),
        "largest_recoverable_target_deg": _largest_true(rows, "recoverable"),
        "largest_successful_target_deg": _largest_true(rows, "success"),
        "seed_families": list(SEED_FAMILIES),
        "replay_defect_tolerance": REPLAY_DEFECT_TOL,
        "actual_agile_reversal_primitive_implemented": any_finite,
        "primitive_implemented": any_finite,
        "local_feedback_controller_implemented": True,
        "ocp_implemented": any_attempt,
        "direct_ocp_attempts_recorded": any_attempt,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "vicon_implemented": False,
        "hardware_implemented": False,
        "updraft_validation_claim": False,
        "high_incidence_validation_claim": False,
        "high_incidence_real_validation_claim": False,
        "full_ladder_requested": True,
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad",
        "state_derivative_command_input": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "output_files": {name: _repo_relative(path) for name, path in output_files.items()},
        "validation_commands": list(VALIDATION_COMMANDS),
        "notes": (
            "Guided W0 aggressive-reversal OCP evidence. Boundary outcomes do not "
            "claim high-incidence or real-flight validation."
        ),
        "any_recoverable": any_recoverable,
    }


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Guided Aggressive Reversal OCP Report",
        "",
        "This is W0/no-wind aggressive-reversal evidence only. It preserves the",
        "existing plant, state order, command bridge, safety volume, and surface",
        "limits. It is not high-incidence validation, not real-flight validation,",
        "not TVLQR, not governor, and not outer-loop evidence.",
        "",
        "## Summary",
        "",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Largest finite target: `{manifest['largest_finite_target_deg']}` deg",
        f"- Largest recoverable target: `{manifest['largest_recoverable_target_deg']}` deg",
        f"- Largest successful target: `{manifest['largest_successful_target_deg']}` deg",
        f"- Replay defect tolerance: `{manifest['replay_defect_tolerance']}`",
        "",
        "## Target Outcomes",
        "",
        "| Target deg | Method | Family | OCP attempted | OCP converged | Replay finite | Recoverable | Success | Failure label | Limiter |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda item: float(item["target_heading_deg"])):
        lines.append(
            "| "
            f"{row['target_heading_deg']} | "
            f"{row['selected_method']} | "
            f"{row['selected_family']} | "
            f"{row['direct_ocp_attempted']} | "
            f"{row['direct_ocp_converged']} | "
            f"{row['replay_finite']} | "
            f"{row['recoverable']} | "
            f"{row['success']} | "
            f"{row['failure_label']} | "
            f"{row['limiting_mechanism']} |"
        )
    lines.extend(
        [
            "",
            "## OCP Diagnostics",
            "",
            "| Target deg | nlp_constructed | ipopt_called | direct_ocp_attempted | direct_ocp_converged | solver_status | constraint_residual_max | replay_defect_max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: float(item["target_heading_deg"])):
        lines.append(
            "| "
            f"{row['target_heading_deg']} | "
            f"{row['nlp_constructed']} | "
            f"{row['ipopt_called']} | "
            f"{row['direct_ocp_attempted']} | "
            f"{row['direct_ocp_converged']} | "
            f"{row['solver_status']} | "
            f"{row['constraint_residual_max']} | "
            f"{row['replay_defect_max']} |"
        )
    lines.extend(
        [
            "",
            "## Manoeuvre-Family Guidance",
            "",
            "Every target is seeded from the fixed family inventory: "
            f"`{', '.join(SEED_FAMILIES)}`.",
            "Failure labels map deterministically to the next family or limiter; no",
            "unconstrained smooth-turn-only path is used.",
            "",
            "## No-Overclaiming Flags",
            "",
            f"- OCP implemented: `{manifest['ocp_implemented']}`",
            f"- TVLQR implemented: `{manifest['tvlqr_implemented']}`",
            f"- Governor implemented: `{manifest['governor_implemented']}`",
            f"- Outer loop implemented: `{manifest['outer_loop_implemented']}`",
            f"- High-incidence validation claim: `{manifest['high_incidence_validation_claim']}`",
            f"- Raw normalised commands enter state derivative: `{manifest['raw_normalised_commands_enter_state_derivative']}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 3) Search Runner
# =============================================================================
def _prepare_tree(run_id: int, overwrite: bool) -> dict[str, Path]:
    root = DEFAULT_RESULTS_ROOT / AGGRESSIVE_CAMPAIGN / f"{run_id:03d}"
    if overwrite and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    if not root.exists():
        return make_result_tree(
            DEFAULT_RESULTS_ROOT,
            AGGRESSIVE_CAMPAIGN,
            run_id,
            overwrite=False,
        )
    paths = {"root": root}
    for name in ("metrics", "logs", "figures", "manifests", "reports"):
        paths[name] = root / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def _load_existing_summary(summary_path: Path) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []
    return pd.read_csv(summary_path).to_dict(orient="records")


def run_search(
    run_id: int = 1,
    overwrite: bool = False,
    targets_deg: tuple[float, ...] = DEFAULT_TARGETS_DEG,
    max_ipopt_iter: int = 80,
    ocp_max_cpu_time_s: float = 0.75,
    ocp_node_count: int = 2,
) -> dict[str, Path]:
    """Run the guided full-target aggressive-reversal evidence ladder."""

    paths = _prepare_tree(run_id, overwrite)
    suffix = _suffix(run_id)
    summary_path = paths["metrics"] / f"aggressive_reversal_target_summary_{suffix}.csv"
    manifest_path = (
        paths["manifests"] / f"aggressive_reversal_manifest_{suffix}.json"
    )
    report_path = paths["reports"] / f"aggressive_reversal_report_{suffix}.md"
    summary_rows = _load_existing_summary(summary_path) if not overwrite else []
    completed = {
        float(row["target_heading_deg"])
        for row in summary_rows
        if "target_heading_deg" in row
    }
    skipped: list[float] = []
    aircraft = adapt_glider(build_nausicaa_glider())
    base_config = AggressiveReversalOcpConfig(
        max_ipopt_iter=max_ipopt_iter,
        ocp_max_cpu_time_s=ocp_max_cpu_time_s,
        ocp_node_count=ocp_node_count,
        run_id=suffix,
        checkpoint_dir=str(paths["logs"] / "checkpoints"),
        candidate_log_dir=str(paths["logs"] / "candidates"),
    )
    output_files: dict[str, Path] = {
        "summary_csv": summary_path,
        "manifest_json": manifest_path,
        "report_md": report_path,
    }
    for target in targets_deg:
        target_value = float(target)
        if target_value in completed and not overwrite:
            skipped.append(target_value)
            continue
        config = target_config(target_value, base_config)
        try:
            result = solve_aggressive_reversal_ocp(config, aircraft=aircraft)
        except Exception as exc:
            result = _failure_result_from_exception(target_value, exc, config)
        target_outputs = _write_target_outputs(result, paths, run_id)
        row = _summary_row(
            result,
            run_id,
            target_outputs["trajectory_csv"],
            target_outputs["commands_csv"],
            target_outputs["metrics_csv"],
        )
        summary_rows = [
            existing
            for existing in summary_rows
            if float(existing["target_heading_deg"]) != target_value
        ]
        summary_rows.append(row)
        output_files[f"target_{_target_token(target_value)}_trajectory_csv"] = (
            target_outputs["trajectory_csv"]
        )
        output_files[f"target_{_target_token(target_value)}_commands_csv"] = (
            target_outputs["commands_csv"]
        )
        output_files[f"target_{_target_token(target_value)}_metrics_csv"] = (
            target_outputs["metrics_csv"]
        )
        _write_summary(summary_rows, paths, run_id)
        manifest = _manifest(
            summary_rows,
            output_files,
            run_id,
            tuple(float(value) for value in targets_deg),
            tuple(skipped),
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False),
            encoding="ascii",
        )
        _write_report(report_path, manifest, summary_rows)
    if not targets_deg:
        _write_summary(summary_rows, paths, run_id)
    manifest = _manifest(
        summary_rows,
        output_files,
        run_id,
        tuple(float(value) for value in targets_deg),
        tuple(skipped),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="ascii",
    )
    _write_report(report_path, manifest, summary_rows)
    return {
        "root": paths["root"],
        "summary_csv": summary_path,
        "manifest_json": manifest_path,
        "report_md": report_path,
    }


# =============================================================================
# 4) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guided W0 aggressive-reversal OCP evidence ladder."
    )
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--targets", nargs="*", type=float, default=list(DEFAULT_TARGETS_DEG))
    parser.add_argument("--max-ipopt-iter", type=int, default=80)
    parser.add_argument("--ocp-max-cpu-time-s", type=float, default=0.75)
    parser.add_argument("--ocp-node-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_search(
        run_id=args.run_id,
        overwrite=args.overwrite,
        targets_deg=tuple(float(value) for value in args.targets),
        max_ipopt_iter=args.max_ipopt_iter,
        ocp_max_cpu_time_s=args.ocp_max_cpu_time_s,
        ocp_node_count=args.ocp_node_count,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['manifest_json']}")
    print(f"report={outputs['report_md']}")
    print(f"overall_status={manifest['overall_status']}")
    print(f"largest_finite_target_deg={manifest['largest_finite_target_deg']}")
    print(f"largest_recoverable_target_deg={manifest['largest_recoverable_target_deg']}")
    print(f"largest_successful_target_deg={manifest['largest_successful_target_deg']}")


if __name__ == "__main__":
    main()
