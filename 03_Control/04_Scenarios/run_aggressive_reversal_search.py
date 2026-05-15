from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


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

from aggressive_reversal_ocp import (  # noqa: E402
    AggressiveReversalConfig,
    AggressiveReversalTarget,
    aggressive_reversal_metric_row,
    deterministic_aggressive_guess_names,
    save_aggressive_reversal_result,
    solve_aggressive_reversal_ocp,
)
from aggressive_reversal_primitive import build_aggressive_reversal_primitive  # noqa: E402
from aggressive_reversal_tvlqr import (  # noqa: E402
    discrete_linearise_rollout_map,
    solve_aggressive_discrete_tvlqr,
)
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from linearisation import INPUT_NAMES, STATE_NAMES, linearise_trim  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from scenarios import aggressive_reversal_entry_state  # noqa: E402


AGGRESSIVE_METRIC_FIELDS = (
    "seed",
    "target_heading_deg",
    "direction",
    "initial_guess_name",
    "success",
    "feasibility_label",
    "failure_reason",
    "actual_heading_change_deg",
    "directed_heading_change_deg",
    "heading_error_deg",
    "forward_travel_m",
    "turn_volume_proxy_m2",
    "height_change_m",
    "duration_s",
    "terminal_speed_m_s",
    "terminal_z_w_m",
    "max_alpha_deg",
    "max_beta_deg",
    "max_bank_deg",
    "max_pitch_deg",
    "max_rate_rad_s",
    "min_wall_distance_m",
    "min_floor_margin_m",
    "min_ceiling_margin_m",
    "inside_true_safety_volume",
    "saturation_fraction",
    "saturation_time_s",
    "exit_recoverable",
    "latency_case",
    "feedback_mode",
    "model_status",
    "is_real_flight_claim",
    "trajectory_npz",
    "log_path",
)


def run_aggressive_reversal_search(
    *,
    targets_deg: tuple[float, ...],
    direction: str,
    seed: int,
    output_root: str | Path | None,
    wind_case: str = "w0",
    quick: bool = False,
    use_tvlqr: bool = False,
) -> dict[str, object]:
    """Run target-ladder aggressive-reversal OCP and optional new TVLQR replay."""
    root = Path(output_root) if output_root is not None else _default_output_root(seed)
    metrics_dir = root / "metrics"
    manifests_dir = root / "manifests"
    logs_dir = root / "logs"
    trajectories_dir = root / "trajectories"
    for path in (metrics_dir, manifests_dir, logs_dir, trajectories_dir):
        path.mkdir(parents=True, exist_ok=True)

    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(linear_model.x_trim, linear_model.u_trim)
    x0 = aggressive_reversal_entry_state(linear_model.x_trim)
    config = AggressiveReversalConfig(n_intervals=12 if quick else 36)
    wind_model = None
    wind_mode = "none"
    if wind_case.lower() != "w0":
        raise ValueError("Only wind_case='w0' is supported in this cleanup/exploration task.")

    candidate_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    tvlqr_rows: list[dict[str, object]] = []
    best_by_target: dict[float, object] = {}
    for target_deg in targets_deg:
        target = AggressiveReversalTarget(
            target_heading_deg=float(target_deg),
            direction=direction,
            wind_case=wind_case,
        )
        results = []
        for guess_name in deterministic_aggressive_guess_names(float(target_deg)):
            result = solve_aggressive_reversal_ocp(
                target=target,
                config=config,
                x0=x0,
                aircraft=aircraft,
                u_trim=linear_model.u_trim,
                wind_model=wind_model,
                wind_mode=wind_mode,
                initial_guess_name=guess_name,
            )
            stem = f"{int(round(float(target_deg))):03d}_{_safe_name(guess_name)}_s{seed:03d}"
            saved = save_aggressive_reversal_result(result, trajectories_dir, stem)
            log_path = logs_dir / f"{stem}.csv"
            _write_result_log(result, log_path)
            row = aggressive_reversal_metric_row(
                result,
                seed=seed,
                initial_guess_name=guess_name,
                trajectory_npz=_rel(saved["trajectory_npz"]),
                log_path=_rel(log_path),
            )
            candidate_rows.append(row)
            results.append((result.objective_value, result, row))
        best = min(results, key=lambda item: (not np.isfinite(item[0]), item[0]))
        best_result = best[1]
        best_by_target[float(target_deg)] = best_result
        replay_rows.append(
            aggressive_reversal_metric_row(
                best_result,
                seed=seed,
                initial_guess_name=str(best_result.solver_stats.get("initial_guess_name", "")),
                trajectory_npz=best[2]["trajectory_npz"],
                log_path=best[2]["log_path"],
                latency_case="none",
                feedback_mode="open_loop",
            )
        )

    if use_tvlqr:
        for target_deg in targets_deg:
            if float(target_deg) not in best_by_target:
                continue
            result = best_by_target[float(target_deg)]
            try:
                a_d, b_d = discrete_linearise_rollout_map(
                    x_ref=result.x_ref,
                    u_ff=result.u_ff,
                    times_s=result.times_s,
                    aircraft=aircraft,
                    wind_model=wind_model,
                    wind_mode=wind_mode,
                    rho_kg_m3=result.config.rho_kg_m3,
                    actuator_tau_s=(0.06, 0.06, 0.06),
                )
                k_feedback, s_mats = solve_aggressive_discrete_tvlqr(
                    a_d=a_d,
                    b_d=b_d,
                    q_diag=(0.1, 0.2, 0.1, 1.5, 1.2, 1.0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05),
                    r_diag=(35.0, 35.0, 35.0),
                    qf_diag=(0.2, 0.4, 0.2, 2.0, 1.6, 1.4, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.08, 0.08, 0.08),
                    phase_labels=result.phase_labels,
                )
                primitive = build_aggressive_reversal_primitive(
                    result=replace_with_feedback(result, k_feedback),
                    context=context,
                    aircraft=aircraft,
                    wind_model=wind_model,
                    wind_mode=wind_mode,
                )
                stem = f"{int(round(float(target_deg))):03d}_tvlqr_s{seed:03d}"
                tvlqr_npz = trajectories_dir / f"{stem}.npz"
                np.savez_compressed(
                    tvlqr_npz,
                    times_s=result.times_s,
                    x_ref=result.x_ref,
                    u_ff=result.u_ff,
                    k_feedback=k_feedback,
                    s_mats=s_mats,
                    phase_labels=np.asarray(result.phase_labels, dtype=object),
                )
                row = aggressive_reversal_metric_row(
                    result,
                    seed=seed,
                    initial_guess_name=str(result.solver_stats.get("initial_guess_name", "")),
                    trajectory_npz=_rel(tvlqr_npz),
                    log_path="",
                    latency_case="none",
                    feedback_mode="single_aggressive_tvlqr",
                )
                row["success"] = bool(np.all(np.isfinite(k_feedback)) and primitive.duration_s > 0.0)
                row["failure_reason"] = "" if row["success"] else "nonfinite_trajectory"
            except Exception as exc:
                row = aggressive_reversal_metric_row(
                    result,
                    seed=seed,
                    initial_guess_name=str(result.solver_stats.get("initial_guess_name", "")),
                    latency_case="none",
                    feedback_mode="single_aggressive_tvlqr",
                )
                row["success"] = False
                row["failure_reason"] = f"solver_failure: {exc}"
                row["feasibility_label"] = "solver_failure"
            tvlqr_rows.append(row)

    candidate_path = metrics_dir / f"aggressive_reversal_candidates_s{seed:03d}.csv"
    replay_path = metrics_dir / f"aggressive_reversal_replay_s{seed:03d}.csv"
    tvlqr_path = metrics_dir / f"aggressive_reversal_tvlqr_s{seed:03d}.csv"
    _write_rows(candidate_path, candidate_rows)
    _write_rows(replay_path, replay_rows)
    if use_tvlqr:
        _write_rows(tvlqr_path, tvlqr_rows)
    manifest_path = manifests_dir / f"aggressive_reversal_manifest_s{seed:03d}.json"
    manifest = {
        "ect_layer_sequence": "Cleanup -> Exploration -> Candidate",
        "seed": int(seed),
        "targets_deg": [float(value) for value in targets_deg],
        "direction": direction,
        "wind_case": wind_case,
        "quick": bool(quick),
        "use_tvlqr": bool(use_tvlqr),
        "metrics": {
            "candidates": _rel(candidate_path),
            "replay": _rel(replay_path),
            "tvlqr": _rel(tvlqr_path) if use_tvlqr else "",
        },
        "trajectory_count": len(list(trajectories_dir.glob("*.npz"))),
        "model_status": "high_incidence_simulation_surrogate",
        "is_real_flight_claim": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_boundary_report(replay_rows, root)
    return {
        "output_root": root,
        "candidate_metrics": candidate_path,
        "replay_metrics": replay_path,
        "tvlqr_metrics": tvlqr_path if use_tvlqr else None,
        "manifest": manifest_path,
        "replay_rows": replay_rows,
        "candidate_rows": candidate_rows,
        "tvlqr_rows": tvlqr_rows,
    }


def replace_with_feedback(result: object, k_feedback: np.ndarray) -> object:
    class _ResultWithFeedback:
        pass

    wrapped = _ResultWithFeedback()
    for name in (
        "success",
        "failure_reason",
        "feasibility_label",
        "target",
        "config",
        "times_s",
        "x_ref",
        "u_ff",
        "nu_ff",
        "phase_labels",
        "objective_value",
        "metrics",
        "solver_stats",
    ):
        setattr(wrapped, name, getattr(result, name))
    wrapped.k_feedback = np.asarray(k_feedback, dtype=float)
    return wrapped


def _default_output_root(seed: int) -> Path:
    del seed
    return (
        REPO_ROOT
        / "03_Control"
        / "05_Results"
        / "03_primitives"
        / "20_aggressive_reversal_rewrite"
        / "001"
    )


def _write_result_log(result: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t_s", "phase"] + list(STATE_NAMES)
    for name in INPUT_NAMES:
        fields.append(f"{name}_rad")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, t_s in enumerate(result.times_s):
            row = {"t_s": float(t_s), "phase": result.phase_labels[idx]}
            row.update({name: float(result.x_ref[idx, j]) for j, name in enumerate(STATE_NAMES)})
            row.update({f"{name}_rad": float(result.u_ff[idx, j]) for j, name in enumerate(INPUT_NAMES)})
            writer.writerow(row)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AGGRESSIVE_METRIC_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in AGGRESSIVE_METRIC_FIELDS})


def _write_boundary_report(rows: list[dict[str, object]], output_root: Path) -> None:
    docs_path = REPO_ROOT / "docs" / "control" / "aggressive_reversal_boundary_report.md"
    lines = [
        "# Aggressive Reversal Boundary Report",
        "",
        "ECT layer sequence: Cleanup -> Exploration -> Candidate",
        "",
        f"Output root: `{_rel(output_root)}`",
        "",
        "| Target deg | Heading change deg | Max alpha deg | Max beta deg | Failure label |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{float(row.get('target_heading_deg', 0.0)):.0f} | "
            f"{float(row.get('actual_heading_change_deg', 0.0)):.3f} | "
            f"{float(row.get('max_alpha_deg', 0.0)):.3f} | "
            f"{float(row.get('max_beta_deg', 0.0)):.3f} | "
            f"{row.get('feasibility_label', '')} |"
        )
    lines.extend(
        [
            "",
            "The aggressive high-incidence reversal results are simulation-surrogate/boundary evidence only.",
            "They are not real-flight claims until separate Transfer-layer gates pass.",
        ]
    )
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    return "_".join(part for part in text.split("_") if part) or "candidate"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=float, nargs="+", required=True)
    parser.add_argument("--direction", default="left")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wind-case", default="w0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--use-tvlqr", action="store_true")
    args = parser.parse_args()
    result = run_aggressive_reversal_search(
        targets_deg=tuple(float(value) for value in args.targets),
        direction=args.direction,
        seed=args.seed,
        output_root=args.output_root,
        wind_case=args.wind_case,
        quick=args.quick,
        use_tvlqr=args.use_tvlqr,
    )
    print("aggressive reversal search complete")
    print(f"output_root: {_rel(Path(result['output_root']))}")
    print(f"manifest: {_rel(Path(result['manifest']))}")


if __name__ == "__main__":
    main()

