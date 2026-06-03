"""Fit neutral dry-air grey-box parameters from real open-loop throws.

The fit is deliberately narrow: it uses only completed neutral_30 open-loop
throws and an aligned first-motion replay window. Pulse/control-effectiveness
throws are not used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_RUNTIME_ROOT = REPO_ROOT / "04_Flight_Test" / "01_Runtime"
PRIMITIVE_ROOT = REPO_ROOT / "03_Control" / "03_Primitives"
SCENARIO_ROOT = REPO_ROOT / "03_Control" / "04_Scenarios"
for path in (FLIGHT_RUNTIME_ROOT, PRIMITIVE_ROOT, SCENARIO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_real_glider_calibration_prep as prep  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402
from command_contract import normalised_command_to_surface_rad  # noqa: E402
from flight_dynamics import adapt_glider, state_derivative  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402


DEFAULT_SESSION_SEARCH_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_RUN_LABEL = "N09_neutral_valid_only_aligned_0p20_fit"
DEFAULT_ALIGNMENT_WINDOW_S = 0.20
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_HELDOUT_COUNT = 5
DEFAULT_HELDOUT_SEED = 606
DEFAULT_WORKERS = 8

FIT_PARAMETER_FIELDS = [
    "iteration",
    "parameter",
    "cd0_strip_scale",
    "drag_area_fuse_scale",
    "efficiency_strip_scale",
    "delta_a_trim_rad",
    "delta_e_trim_rad",
    "delta_r_trim_rad",
    "train_count",
    "train_objective",
    "train_dx_mae_m",
    "train_dy_mae_m",
    "train_altitude_loss_mae_m",
    "train_sink_mae_m_s",
    "train_dx_mean_m",
    "train_dy_mean_m",
    "train_altitude_loss_mean_m",
    "train_sink_mean_m_s",
]

REPLAY_RESIDUAL_FIELDS = [
    "split",
    "session_label",
    "case_id",
    "throw_id",
    "replay_status",
    "alignment_window_s",
    "duration_s",
    "x0_m",
    "y0_m",
    "z0_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "actual_dx_m",
    "sim_dx_m",
    "dx_residual_actual_minus_sim_m",
    "actual_dy_m",
    "sim_dy_m",
    "dy_residual_actual_minus_sim_m",
    "actual_altitude_loss_m",
    "sim_altitude_loss_m",
    "altitude_loss_residual_actual_minus_sim_m",
    "actual_sink_rate_m_s",
    "sim_sink_rate_m_s",
    "sink_rate_residual_actual_minus_sim_m_s",
    "actual_final_phi_deg",
    "sim_final_phi_deg",
    "actual_final_theta_deg",
    "sim_final_theta_deg",
    "actual_final_psi_deg",
    "sim_final_psi_deg",
]

PARAMETER_GRID = {
    "cd0_strip_scale": [3.0, 4.0, 5.0, 6.0, 7.5, 9.0],
    "drag_area_fuse_scale": [5.0, 7.5, 10.0, 12.5, 15.0],
    "efficiency_strip_scale": [0.55, 0.70, 0.85, 1.00],
    "delta_e_trim_rad": [-0.06, -0.03, 0.0, 0.03, 0.06],
    "delta_a_trim_rad": [-0.08, -0.04, 0.0, 0.04, 0.08],
    "delta_r_trim_rad": [-0.08, -0.04, 0.0, 0.04, 0.08],
}

REFINE_GRID = {
    "cd0_strip_scale": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "drag_area_fuse_scale": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "efficiency_strip_scale": [-0.12, -0.06, 0.0, 0.06, 0.12],
    "delta_e_trim_rad": [-0.03, -0.015, 0.0, 0.015, 0.03],
    "delta_a_trim_rad": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "delta_r_trim_rad": [-0.04, -0.02, 0.0, 0.02, 0.04],
}


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = run_fit(
        session_root=args.session_root,
        output_root=args.output_root,
        run_label=args.run_label,
        heldout_count=args.heldout_count,
        heldout_seed=args.heldout_seed,
        replay_dt_s=args.replay_dt_s,
        alignment_window_s=args.alignment_window_s,
        workers=args.workers,
        coordinate_passes=args.coordinate_passes,
        fit_neutral_trim=args.fit_neutral_trim,
    )
    print(f"[DONE] neutral dry-air fit written to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit neutral dry-air calibration using open-loop real throws.")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_SEARCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--heldout-count", type=int, default=DEFAULT_HELDOUT_COUNT)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S)
    parser.add_argument("--alignment-window-s", type=float, default=DEFAULT_ALIGNMENT_WINDOW_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--coordinate-passes", type=int, default=2)
    parser.add_argument(
        "--fit-neutral-trim",
        action="store_true",
        help="Also fit static aileron/elevator/rudder neutral offsets. Off by default to avoid absorbing launch scatter.",
    )
    return parser


def run_fit(
    *,
    session_root: Path,
    output_root: Path,
    run_label: str,
    heldout_count: int,
    heldout_seed: int,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    coordinate_passes: int,
    fit_neutral_trim: bool,
) -> Path:
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = load_neutral_open_loop_rows(session_root)
    heldout_indices = prep.stratified_heldout_indices(
        valid_rows,
        heldout_count=heldout_count,
        heldout_seed=heldout_seed,
        group_key="session_label",
    )
    train_rows = [row for index, row in enumerate(valid_rows) if index not in heldout_indices]
    heldout_rows = [row for index, row in enumerate(valid_rows) if index in heldout_indices]

    current = {
        "cd0_strip_scale": float(active_calibration.CD0_STRIP_SCALE),
        "drag_area_fuse_scale": float(active_calibration.DRAG_AREA_FUSE_SCALE),
        "efficiency_strip_scale": float(active_calibration.EFFICIENCY_STRIP_SCALE),
        "delta_a_trim_rad": float(getattr(active_calibration, "DELTA_A_TRIM_RAD", 0.0)),
        "delta_e_trim_rad": float(getattr(active_calibration, "DELTA_E_TRIM_RAD", 0.0)),
        "delta_r_trim_rad": float(getattr(active_calibration, "DELTA_R_TRIM_RAD", 0.0)),
    }

    history: list[dict[str, Any]] = []
    iteration = 0
    parameter_order = ["cd0_strip_scale", "drag_area_fuse_scale", "efficiency_strip_scale"]
    if fit_neutral_trim:
        parameter_order.extend(["delta_e_trim_rad", "delta_a_trim_rad", "delta_r_trim_rad"])

    baseline = evaluate_candidates(
        [dict(current)],
        train_rows,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    history.append(history_row(iteration, "baseline", baseline))
    iteration += 1

    for parameter in parameter_order:
        values = PARAMETER_GRID[parameter]
        candidates = []
        for value in values:
            candidate = dict(current)
            candidate[parameter] = float(value)
            candidates.append(candidate)
        best = evaluate_candidates(
            candidates,
            train_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        current = best["parameters"]
        history.append(history_row(iteration, parameter, best))
        iteration += 1

    for _ in range(max(0, int(coordinate_passes))):
        for parameter in parameter_order:
            offsets = REFINE_GRID[parameter]
            candidates = []
            for offset in offsets:
                candidate = dict(current)
                candidate[parameter] = bounded_parameter_value(parameter, candidate[parameter] + float(offset))
                candidates.append(candidate)
            best = evaluate_candidates(
                candidates,
                train_rows,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
            )
            current = best["parameters"]
            history.append(history_row(iteration, parameter, best))
            iteration += 1

    train_replay = simulate_rows(
        train_rows,
        current,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    heldout_replay = simulate_rows(
        heldout_rows,
        current,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    for row in train_replay:
        row["split"] = "train"
    for row in heldout_replay:
        row["split"] = "heldout"
    all_replay = train_replay + heldout_replay

    write_csv(output_dir / "metrics" / "neutral_fit_candidate_history.csv", history, FIT_PARAMETER_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_fit_aligned_replay_residuals.csv", all_replay, REPLAY_RESIDUAL_FIELDS)
    write_manifest(
        output_dir,
        run_label=run_label,
        valid_rows=valid_rows,
        heldout_indices=heldout_indices,
        best_parameters=current,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
        fit_neutral_trim=fit_neutral_trim,
    )
    write_report(output_dir, train_replay, heldout_replay, current)
    return output_dir


def load_neutral_open_loop_rows(session_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in prep.resolve_session_roots(session_root):
        summary = prep._load_json(root / "manifests" / "glider_calibration_sequence_final_summary.json")
        if summary.get("block_id") != "neutral_30":
            continue
        session_label = root.name
        for throw_dir in prep._valid_throw_dirs(root):
            row = prep.summarize_valid_throw(session_label, root, throw_dir)
            if str(row.get("command_axis", "")) in {"", "neutral"}:
                rows.append(row)
    if len(rows) < 10:
        raise RuntimeError(f"Need at least 10 neutral open-loop throws, found {len(rows)}.")
    return rows


def bounded_parameter_value(parameter: str, value: float) -> float:
    if parameter in {"cd0_strip_scale", "drag_area_fuse_scale"}:
        return float(np.clip(value, 0.2, 30.0))
    if parameter == "efficiency_strip_scale":
        return float(np.clip(value, 0.2, 2.0))
    return float(np.clip(value, -0.18, 0.18))


def evaluate_candidates(
    candidates: list[dict[str, float]],
    rows: list[dict[str, Any]],
    *,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> dict[str, Any]:
    payloads = [(candidate, rows, replay_dt_s, alignment_window_s) for candidate in candidates]
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(evaluate_candidate_payload, payloads))
    else:
        results = [evaluate_candidate_payload(payload) for payload in payloads]
    return min(results, key=lambda item: item["summary"]["objective"])


def evaluate_candidate_payload(payload: tuple[dict[str, float], list[dict[str, Any]], float, float]) -> dict[str, Any]:
    candidate, rows, replay_dt_s, alignment_window_s = payload
    replay_rows = simulate_rows(
        rows,
        candidate,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=1,
    )
    return {"parameters": dict(candidate), "summary": objective_summary(replay_rows)}


def simulate_rows(
    rows: list[dict[str, Any]],
    parameters: dict[str, float],
    *,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    if int(workers) > 1 and len(rows) > 1:
        payloads = [(row, parameters, replay_dt_s, alignment_window_s) for row in rows]
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            return list(executor.map(simulate_row_payload, payloads))
    return [simulate_row_payload((row, parameters, replay_dt_s, alignment_window_s)) for row in rows]


def simulate_row_payload(payload: tuple[dict[str, Any], dict[str, float], float, float]) -> dict[str, Any]:
    row, parameters, replay_dt_s, alignment_window_s = payload
    throw_dir = Path(str(row.get("_throw_dir", "")))
    if not throw_dir.exists():
        return blocked_row(row, "missing_throw_dir", alignment_window_s)
    sample_rows = prep._read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return blocked_row(row, "missing_state_samples", alignment_window_s)
    aligned = prep._aligned_state_from_sample_rows(sample_rows, alignment_window_s)
    if aligned["status"] != "ok":
        return blocked_row(row, str(aligned["status"]), alignment_window_s)
    x0 = np.asarray(aligned["state"], dtype=float)
    trim = np.asarray(
        [
            parameters["delta_a_trim_rad"],
            parameters["delta_e_trim_rad"],
            parameters["delta_r_trim_rad"],
        ],
        dtype=float,
    )
    x0[12:15] = trim

    t_first = prep._float(sample_rows[0], "t_s", 0.0)
    t_last = prep._float(sample_rows[-1], "t_s", t_first)
    duration_s = max(0.0, t_last - t_first - float(aligned["alignment_elapsed_s"]))
    if duration_s <= 0.0:
        return blocked_row(row, "invalid_duration", alignment_window_s)

    aircraft = calibrated_aircraft(parameters)
    actuator_tau_s = prep._actuator_tau_from_manifest(prep._throw_manifest(throw_dir))
    command_schedule = [(0.0, trim)]
    x = x0.copy()
    t_s = 0.0
    max_abs_phi = abs(float(x[3]))
    max_abs_theta = abs(float(x[4]))
    max_abs_p = abs(float(x[9]))
    max_abs_q = abs(float(x[10]))
    max_abs_r = abs(float(x[11]))
    while t_s < duration_s - 1e-12:
        dt_s = min(float(replay_dt_s), duration_s - t_s)
        command = command_schedule[0][1]
        try:
            x = rk4_step(x, command, aircraft, actuator_tau_s, dt_s)
        except Exception:
            return blocked_row(row, "state_derivative_failed", alignment_window_s)
        t_s += dt_s
        if not np.all(np.isfinite(x)):
            return blocked_row(row, "nonfinite_replay_state", alignment_window_s)
        max_abs_phi = max(max_abs_phi, abs(float(x[3])))
        max_abs_theta = max(max_abs_theta, abs(float(x[4])))
        max_abs_p = max(max_abs_p, abs(float(x[9])))
        max_abs_q = max(max_abs_q, abs(float(x[10])))
        max_abs_r = max(max_abs_r, abs(float(x[11])))

    actual_final = prep._state_vector_from_sample_row(sample_rows[-1])
    actual_dx = float(actual_final[0] - x0[0])
    actual_dy = float(actual_final[1] - x0[1])
    actual_altitude_loss = float(x0[2] - actual_final[2])
    actual_sink_rate = prep._ratio(actual_altitude_loss, duration_s)
    sim_dx = float(x[0] - x0[0])
    sim_dy = float(x[1] - x0[1])
    sim_altitude_loss = float(x0[2] - x[2])
    sim_sink_rate = prep._ratio(sim_altitude_loss, duration_s)

    return {
        "session_label": row.get("session_label", ""),
        "case_id": row.get("case_id", ""),
        "throw_id": row.get("throw_id", ""),
        "replay_status": "ok",
        "alignment_window_s": float(alignment_window_s),
        "duration_s": duration_s,
        "x0_m": float(x0[0]),
        "y0_m": float(x0[1]),
        "z0_m": float(x0[2]),
        "u0_m_s": float(x0[6]),
        "v0_m_s": float(x0[7]),
        "w0_m_s": float(x0[8]),
        "p0_rad_s": float(x0[9]),
        "q0_rad_s": float(x0[10]),
        "r0_rad_s": float(x0[11]),
        "actual_dx_m": actual_dx,
        "sim_dx_m": sim_dx,
        "dx_residual_actual_minus_sim_m": actual_dx - sim_dx,
        "actual_dy_m": actual_dy,
        "sim_dy_m": sim_dy,
        "dy_residual_actual_minus_sim_m": actual_dy - sim_dy,
        "actual_altitude_loss_m": actual_altitude_loss,
        "sim_altitude_loss_m": sim_altitude_loss,
        "altitude_loss_residual_actual_minus_sim_m": actual_altitude_loss - sim_altitude_loss,
        "actual_sink_rate_m_s": actual_sink_rate,
        "sim_sink_rate_m_s": sim_sink_rate,
        "sink_rate_residual_actual_minus_sim_m_s": actual_sink_rate - sim_sink_rate,
        "actual_final_phi_deg": math.degrees(float(actual_final[3])),
        "sim_final_phi_deg": math.degrees(float(x[3])),
        "actual_final_theta_deg": math.degrees(float(actual_final[4])),
        "sim_final_theta_deg": math.degrees(float(x[4])),
        "actual_final_psi_deg": math.degrees(float(actual_final[5])),
        "sim_final_psi_deg": math.degrees(float(x[5])),
    }


def calibrated_aircraft(parameters: dict[str, float]) -> Any:
    base = adapt_glider(build_nausicaa_glider())
    cd0_ratio = parameters["cd0_strip_scale"] / float(active_calibration.CD0_STRIP_SCALE)
    drag_ratio = parameters["drag_area_fuse_scale"] / float(active_calibration.DRAG_AREA_FUSE_SCALE)
    efficiency_ratio = parameters["efficiency_strip_scale"] / float(active_calibration.EFFICIENCY_STRIP_SCALE)
    return replace(
        base,
        cd0_strip=np.asarray(base.cd0_strip, dtype=float) * cd0_ratio,
        drag_area_fuse_m2=float(base.drag_area_fuse_m2) * drag_ratio,
        efficiency_strip=np.asarray(base.efficiency_strip, dtype=float) * efficiency_ratio,
    )


def rk4_step(x: np.ndarray, command: np.ndarray, aircraft: Any, actuator_tau_s: tuple[float, float, float], dt_s: float) -> np.ndarray:
    k1 = state_derivative(x, command, aircraft, wind_model=None, actuator_tau_s=actuator_tau_s, wind_mode="panel")
    k2 = state_derivative(x + 0.5 * dt_s * k1, command, aircraft, wind_model=None, actuator_tau_s=actuator_tau_s, wind_mode="panel")
    k3 = state_derivative(x + 0.5 * dt_s * k2, command, aircraft, wind_model=None, actuator_tau_s=actuator_tau_s, wind_mode="panel")
    k4 = state_derivative(x + dt_s * k3, command, aircraft, wind_model=None, actuator_tau_s=actuator_tau_s, wind_mode="panel")
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def blocked_row(row: dict[str, Any], status: str, alignment_window_s: float) -> dict[str, Any]:
    return {
        "session_label": row.get("session_label", ""),
        "case_id": row.get("case_id", ""),
        "throw_id": row.get("throw_id", ""),
        "replay_status": status,
        "alignment_window_s": float(alignment_window_s),
    }


def objective_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    dx = residual_values(rows, "dx_residual_actual_minus_sim_m")
    dy = residual_values(rows, "dy_residual_actual_minus_sim_m")
    alt = residual_values(rows, "altitude_loss_residual_actual_minus_sim_m")
    sink = residual_values(rows, "sink_rate_residual_actual_minus_sim_m_s")
    objective = (
        mae(dx) / 0.30
        + mae(dy) / 0.45
        + mae(alt) / 0.12
        + mae(sink) / 0.10
        + abs(mean_value(dx)) / 0.25
        + abs(mean_value(dy)) / 0.35
        + abs(mean_value(sink)) / 0.08
    )
    return {
        "objective": float(objective),
        "count": float(len(dx)),
        "dx_mae_m": mae(dx),
        "dy_mae_m": mae(dy),
        "altitude_loss_mae_m": mae(alt),
        "sink_mae_m_s": mae(sink),
        "dx_mean_m": mean_value(dx),
        "dy_mean_m": mean_value(dy),
        "altitude_loss_mean_m": mean_value(alt),
        "sink_mean_m_s": mean_value(sink),
    }


def residual_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("replay_status") != "ok":
            continue
        value = prep._to_float(row.get(key))
        if math.isfinite(value):
            values.append(value)
    return values


def mae(values: list[float]) -> float:
    return float(sum(abs(value) for value in values) / len(values)) if values else float("inf")


def mean_value(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("inf")


def history_row(iteration: int, parameter: str, result: dict[str, Any]) -> dict[str, Any]:
    params = result["parameters"]
    summary = result["summary"]
    return {
        "iteration": iteration,
        "parameter": parameter,
        "cd0_strip_scale": params["cd0_strip_scale"],
        "drag_area_fuse_scale": params["drag_area_fuse_scale"],
        "efficiency_strip_scale": params["efficiency_strip_scale"],
        "delta_a_trim_rad": params["delta_a_trim_rad"],
        "delta_e_trim_rad": params["delta_e_trim_rad"],
        "delta_r_trim_rad": params["delta_r_trim_rad"],
        "train_count": summary["count"],
        "train_objective": summary["objective"],
        "train_dx_mae_m": summary["dx_mae_m"],
        "train_dy_mae_m": summary["dy_mae_m"],
        "train_altitude_loss_mae_m": summary["altitude_loss_mae_m"],
        "train_sink_mae_m_s": summary["sink_mae_m_s"],
        "train_dx_mean_m": summary["dx_mean_m"],
        "train_dy_mean_m": summary["dy_mean_m"],
        "train_altitude_loss_mean_m": summary["altitude_loss_mean_m"],
        "train_sink_mean_m_s": summary["sink_mean_m_s"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}" if math.isfinite(value) else ""
    return value


def write_manifest(
    output_dir: Path,
    *,
    run_label: str,
    valid_rows: list[dict[str, Any]],
    heldout_indices: set[int],
    best_parameters: dict[str, float],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    fit_neutral_trim: bool,
) -> None:
    manifest = {
        "fit_id": str(run_label),
        "fit_scope": "neutral_30_open_loop_only",
        "valid_throw_count": len(valid_rows),
        "heldout_policy": "randomised_stratified_by_session_label",
        "heldout_indices": sorted(int(index) for index in heldout_indices),
        "heldout_throw_keys": [
            {
                "index": int(index),
                "session_label": str(valid_rows[index].get("session_label", "")),
                "throw_id": str(valid_rows[index].get("throw_id", "")),
            }
            for index in sorted(heldout_indices)
        ],
        "alignment_window_s": float(alignment_window_s),
        "replay_dt_s": float(replay_dt_s),
        "workers": int(workers),
        "fit_neutral_trim": bool(fit_neutral_trim),
        "best_parameters": dict(best_parameters),
        "active_calibration_before_fit": {
            "cd0_strip_scale": float(active_calibration.CD0_STRIP_SCALE),
            "drag_area_fuse_scale": float(active_calibration.DRAG_AREA_FUSE_SCALE),
            "efficiency_strip_scale": float(active_calibration.EFFICIENCY_STRIP_SCALE),
            "delta_a_trim_rad": float(getattr(active_calibration, "DELTA_A_TRIM_RAD", 0.0)),
            "delta_e_trim_rad": float(getattr(active_calibration, "DELTA_E_TRIM_RAD", 0.0)),
            "delta_r_trim_rad": float(getattr(active_calibration, "DELTA_R_TRIM_RAD", 0.0)),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = output_dir / "manifests" / "neutral_dry_air_fit_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    train_replay: list[dict[str, Any]],
    heldout_replay: list[dict[str, Any]],
    best_parameters: dict[str, float],
) -> None:
    train = objective_summary(train_replay)
    heldout = objective_summary(heldout_replay)
    lines = [
        "# Neutral Dry-Air Aligned-Replay Fit",
        "",
        "This fit uses only neutral open-loop real throws and a 0.20 s first-motion alignment window.",
        "Pulse/control-effectiveness throws are intentionally excluded.",
        "",
        "## Best Parameters",
        "",
        f"- cd0 strip scale: `{best_parameters['cd0_strip_scale']:.6g}`",
        f"- fuselage drag-area scale: `{best_parameters['drag_area_fuse_scale']:.6g}`",
        f"- strip efficiency scale: `{best_parameters['efficiency_strip_scale']:.6g}`",
        f"- aileron neutral trim: `{best_parameters['delta_a_trim_rad']:.6g}` rad",
        f"- elevator neutral trim: `{best_parameters['delta_e_trim_rad']:.6g}` rad",
        f"- rudder neutral trim: `{best_parameters['delta_r_trim_rad']:.6g}` rad",
        "",
        "## Replay Fit Quality",
        "",
        f"- train count: `{int(train['count'])}`",
        f"- train dx MAE: `{train['dx_mae_m']:.4f}` m",
        f"- train dy MAE: `{train['dy_mae_m']:.4f}` m",
        f"- train altitude-loss MAE: `{train['altitude_loss_mae_m']:.4f}` m",
        f"- train sink-rate MAE: `{train['sink_mae_m_s']:.4f}` m/s",
        f"- held-out count: `{int(heldout['count'])}`",
        f"- held-out dx MAE: `{heldout['dx_mae_m']:.4f}` m",
        f"- held-out dy MAE: `{heldout['dy_mae_m']:.4f}` m",
        f"- held-out altitude-loss MAE: `{heldout['altitude_loss_mae_m']:.4f}` m",
        f"- held-out sink-rate MAE: `{heldout['sink_mae_m_s']:.4f}` m/s",
        "",
        "## Interpretation",
        "",
        "If held-out dy remains large after neutral trim fitting, lateral error should not be forced into pulse effectiveness. Inspect physical trim/asymmetry and then fit control derivatives separately.",
    ]
    path = output_dir / "reports" / "neutral_dry_air_fit_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
