"""Plot real-flight launches against theory and active calibrated replays."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = ROOT / "03_Control"
INNER_LOOP = CONTROL_ROOT / "02_Inner_Loop"
PRIMITIVES = CONTROL_ROOT / "03_Primitives"
for path in (INNER_LOOP, PRIMITIVES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import glider as glider_module  # noqa: E402
import run_control_surface_effectiveness_study as surface_study  # noqa: E402
import run_real_glider_calibration_prep as prep  # noqa: E402


FIGURE_RUN_VERSION = "real_flight_replay_comparison_v1"
DEFAULT_SURFACE_RESULT_ROOT = (
    CONTROL_ROOT / "05_Results" / "control_surface_effectiveness" / "control_surface_effectiveness_v3_0_final_cnbeta"
)
DEFAULT_NEUTRAL_ROOT = ROOT / "04_Flight_Test" / "05_Results" / "cal" / "n30"
DEFAULT_OUTPUT_ROOT = CONTROL_ROOT / "A_figures" / "real_flight_replay_comparison"
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_WORKERS = 8
MAX_REPRESENTATIVE_ALPHA_DEG = 35.0
DEFAULT_EXTRA_NEUTRAL_SAMPLE_SEED = 607
DEFAULT_GOOD_NEUTRAL_MIN_CONFIDENCE = 0.85
DEFAULT_GOOD_NEUTRAL_MAX_ALPHA_DEG = 25.0

MODEL_LABELS = {
    "real": "real flight",
    "theory": "uncalibrated theory replay",
    "calibrated": "active calibrated replay",
}
MODEL_COLORS = {
    "real": "#111111",
    "theory": "#8a8a8a",
    "calibrated": "#1f77b4",
}
MODEL_STYLES = {
    "real": "-",
    "theory": "--",
    "calibrated": "-",
}


@dataclass(frozen=True)
class SelectedLaunch:
    figure_id: str
    surface_axis: str
    command_value: float
    row: dict[str, Any]
    selection_note: str


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    selected = select_representative_launches(
        surface_result_root=Path(args.surface_result_root),
        neutral_root=Path(args.neutral_root),
        extra_neutral_samples=int(args.extra_neutral_samples),
        neutral_sample_seed=int(args.neutral_sample_seed),
        good_neutral_min_confidence=float(args.good_neutral_min_confidence),
        good_neutral_max_alpha_deg=float(args.good_neutral_max_alpha_deg),
    )
    traces = run_replays(selected, replay_dt_s=float(args.replay_dt_s), workers=DEFAULT_WORKERS)
    figure_rows = write_figures(selected, traces, output_root=output_root)
    write_summary(output_root / "metrics" / "real_flight_replay_comparison_summary.csv", figure_rows)
    manifest = {
        "figure_run_version": FIGURE_RUN_VERSION,
        "status": "complete",
        "surface_result_root": Path(args.surface_result_root).as_posix(),
        "neutral_root": Path(args.neutral_root).as_posix(),
        "output_root": output_root.as_posix(),
        "replay_dt_s": float(args.replay_dt_s),
        "workers": int(DEFAULT_WORKERS),
        "model_comparison": {
            "uncalibrated": "comparison-only pure theory/geometry baseline; calibration disabled in memory",
            "calibrated": (
                "active neutral dry-air residual-calibrated replay model plus active elevator and rudder effectiveness"
            ),
        },
        "representative_selection": (
            "neutral plus highest-confidence max positive/negative surface throws; "
            f"prefers max_abs_alpha_deg <= {MAX_REPRESENTATIVE_ALPHA_DEG:g} when available; "
            f"extra neutral samples use seed {int(args.neutral_sample_seed)} from confidence >= "
            f"{float(args.good_neutral_min_confidence):g}, max_abs_alpha_deg <= "
            f"{float(args.good_neutral_max_alpha_deg):g}"
        ),
        "figures": [row["figure_path"] for row in figure_rows],
    }
    (output_root / "manifests" / "real_flight_replay_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    write_report(output_root / "reports" / "real_flight_replay_comparison_report.md", figure_rows, manifest)
    print(output_root.as_posix())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate representative real-flight vs uncalibrated/calibrated replay figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--surface-result-root", default=DEFAULT_SURFACE_RESULT_ROOT.as_posix())
    parser.add_argument("--neutral-root", default=DEFAULT_NEUTRAL_ROOT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S)
    parser.add_argument("--extra-neutral-samples", type=int, default=0)
    parser.add_argument("--neutral-sample-seed", type=int, default=DEFAULT_EXTRA_NEUTRAL_SAMPLE_SEED)
    parser.add_argument("--good-neutral-min-confidence", type=float, default=DEFAULT_GOOD_NEUTRAL_MIN_CONFIDENCE)
    parser.add_argument("--good-neutral-max-alpha-deg", type=float, default=DEFAULT_GOOD_NEUTRAL_MAX_ALPHA_DEG)
    return parser


def select_representative_launches(
    *,
    surface_result_root: Path,
    neutral_root: Path,
    extra_neutral_samples: int = 0,
    neutral_sample_seed: int = DEFAULT_EXTRA_NEUTRAL_SAMPLE_SEED,
    good_neutral_min_confidence: float = DEFAULT_GOOD_NEUTRAL_MIN_CONFIDENCE,
    good_neutral_max_alpha_deg: float = DEFAULT_GOOD_NEUTRAL_MAX_ALPHA_DEG,
) -> list[SelectedLaunch]:
    rows = _read_csv(surface_result_root / "control_surface_inventory.csv")
    selected: list[SelectedLaunch] = []
    neutral_candidates = select_neutral_launch_candidates(neutral_root)
    neutral = select_neutral_launch(neutral_root, candidates=neutral_candidates)
    selected.append(
        SelectedLaunch(
            figure_id="neutral",
            surface_axis="neutral",
            command_value=0.0,
            row=neutral,
            selection_note="best valid neutral launch from n30 logs",
        )
    )
    for index, row in enumerate(
        select_extra_neutral_launches(
            neutral_candidates,
            excluded_throw_dirs={str(neutral.get("_throw_dir", neutral.get("throw_dir", "")))},
            count=max(0, int(extra_neutral_samples)),
            seed=int(neutral_sample_seed),
            min_confidence=float(good_neutral_min_confidence),
            max_alpha_deg=float(good_neutral_max_alpha_deg),
        ),
        start=1,
    ):
        selected.append(
            SelectedLaunch(
                figure_id=f"neutral_random_good_{index:02d}",
                surface_axis="neutral",
                command_value=0.0,
                row=row,
                selection_note=(
                    "random sampled good neutral launch; "
                    f"seed {int(neutral_sample_seed)}, confidence >= {float(good_neutral_min_confidence):g}, "
                    f"max_abs_alpha_deg <= {float(good_neutral_max_alpha_deg):g}"
                ),
            )
        )
    for surface in ("elevator", "rudder", "aileron"):
        for sign_label, sign in (("neg", -1.0), ("pos", 1.0)):
            row = select_max_surface_launch(rows, surface=surface, sign=sign)
            selected.append(
                SelectedLaunch(
                    figure_id=f"max_{surface}_{sign_label}",
                    surface_axis=surface,
                    command_value=surface_study.to_float(row.get("command_value"), sign),
                    row=row,
                    selection_note=(
                        "max command throw; highest launch confidence among low/moderate alpha cases when possible"
                    ),
                )
            )
    return selected


def select_max_surface_launch(rows: list[dict[str, Any]], *, surface: str, sign: float) -> dict[str, Any]:
    candidates = [
        _normalise_inventory_row(row)
        for row in rows
        if str(row.get("filter_status", "")) == "kept"
        and str(row.get("surface_axis", "")) == surface
        and math.isclose(abs(surface_study.to_float(row.get("command_abs"))), 1.0, rel_tol=0.0, abs_tol=1e-9)
        and surface_study.to_float(row.get("command_value")) * float(sign) > 0.0
    ]
    if not candidates:
        raise RuntimeError(f"No kept max-command launch found for {surface} sign {sign:+g}")
    moderate_alpha = [
        row
        for row in candidates
        if surface_study.to_float(row.get("max_abs_alpha_deg"), float("inf")) <= MAX_REPRESENTATIVE_ALPHA_DEG
    ]
    pool = moderate_alpha if moderate_alpha else candidates
    return sorted(
        pool,
        key=lambda row: (
            surface_study.to_float(row.get("launch_confidence_score"), -1.0),
            surface_study.to_float(row.get("effective_flight_duration_s"), -1.0),
        ),
        reverse=True,
    )[0]


def select_neutral_launch_candidates(neutral_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for manifest_path in sorted(neutral_root.rglob("manifests/glider_calibration_throw_manifest.json")):
        throw_dir = manifest_path.parents[1]
        if "c0_neu" not in throw_dir.as_posix():
            continue
        row = neutral_inventory_row(throw_dir)
        if row is not None:
            candidates.append(row)
    if not candidates:
        raise RuntimeError(f"No valid neutral launch found under {neutral_root}")
    return candidates


def select_neutral_launch(
    neutral_root: Path,
    *,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if candidates is None:
        candidates = select_neutral_launch_candidates(neutral_root)
    moderate_alpha = [
        row
        for row in candidates
        if surface_study.to_float(row.get("max_abs_alpha_deg"), float("inf")) <= MAX_REPRESENTATIVE_ALPHA_DEG
    ]
    pool = moderate_alpha if moderate_alpha else candidates
    return sorted(
        pool,
        key=lambda row: (
            surface_study.to_float(row.get("launch_confidence_score"), -1.0),
            surface_study.to_float(row.get("effective_flight_duration_s"), -1.0),
        ),
        reverse=True,
    )[0]


def select_extra_neutral_launches(
    candidates: list[dict[str, Any]],
    *,
    excluded_throw_dirs: set[str],
    count: int,
    seed: int,
    min_confidence: float,
    max_alpha_deg: float,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    pool = [
        row
        for row in candidates
        if str(row.get("_throw_dir", row.get("throw_dir", ""))) not in excluded_throw_dirs
        and surface_study.to_float(row.get("launch_confidence_score"), -1.0) >= float(min_confidence)
        and surface_study.to_float(row.get("max_abs_alpha_deg"), float("inf")) <= float(max_alpha_deg)
    ]
    if len(pool) < count:
        pool = [
            row
            for row in candidates
            if str(row.get("_throw_dir", row.get("throw_dir", ""))) not in excluded_throw_dirs
            and surface_study.to_float(row.get("max_abs_alpha_deg"), float("inf")) <= MAX_REPRESENTATIVE_ALPHA_DEG
        ]
    if len(pool) < count:
        pool = [
            row
            for row in candidates
            if str(row.get("_throw_dir", row.get("throw_dir", ""))) not in excluded_throw_dirs
        ]
    order = np.random.default_rng(int(seed)).permutation(len(pool))
    return [pool[int(index)] for index in order[: min(int(count), len(pool))]]


def neutral_inventory_row(throw_dir: Path) -> dict[str, Any] | None:
    sample_rows = surface_study.read_csv(throw_dir / "metrics" / "state_samples.csv")
    if len(sample_rows) < 3:
        return None
    summary = surface_study.load_json(throw_dir / "manifests" / "glider_calibration_throw_summary.json")
    if summary and summary.get("valid_throw") is False:
        return None
    first = sample_rows[0]
    last = sample_rows[-1]
    t0 = surface_study.to_float(first.get("t_s"), 0.0)
    t1 = surface_study.to_float(last.get("t_s"), t0)
    duration_s = max(0.0, t1 - t0)
    if duration_s < 0.65:
        return None
    u0 = surface_study.to_float(first.get("u"))
    v0 = surface_study.to_float(first.get("v"))
    w0 = surface_study.to_float(first.get("w"))
    if (
        not surface_study.all_finite(u0, v0, w0)
        or u0 < surface_study.RELAXED_U_MIN_M_S
        or u0 > surface_study.RELAXED_U_MAX_M_S
        or abs(v0) > surface_study.RELAXED_ABS_V_MAX_M_S
        or abs(w0) > surface_study.RELAXED_ABS_W_MAX_M_S
    ):
        return None
    row: dict[str, Any] = {
        "_throw_dir": throw_dir.as_posix(),
        "throw_dir": throw_dir.as_posix(),
        "dataset_root": throw_dir.parents[2].as_posix(),
        "session_label": throw_dir.parents[1].name,
        "trial_id": "/".join(throw_dir.parts[-3:]),
        "throw_id": throw_dir.name,
        "case_id": "C0_neutral",
        "surface_axis": "neutral",
        "command_axis": "neutral",
        "command_value": 0.0,
        "command_abs": 0.0,
        "effective_flight_duration_s": duration_s,
        "usable_window_start_s": 0.15,
        "usable_window_end_s": min(duration_s, 0.80),
        "valid_throw": True,
        "state_sample_count": len(sample_rows),
        "x0_m": surface_study.to_float(first.get("x_w")),
        "y0_m": surface_study.to_float(first.get("y_w")),
        "z0_m": surface_study.to_float(first.get("z_w")),
        "u0_m_s": u0,
        "v0_m_s": v0,
        "w0_m_s": w0,
        "speed0_m_s": math.sqrt(u0 * u0 + v0 * v0 + w0 * w0),
        "phi0_deg": math.degrees(surface_study.to_float(first.get("phi"))),
        "theta0_deg": math.degrees(surface_study.to_float(first.get("theta"))),
        "psi0_deg": math.degrees(surface_study.to_float(first.get("psi"))),
        "p0_rad_s": surface_study.to_float(first.get("p")),
        "q0_rad_s": surface_study.to_float(first.get("q")),
        "r0_rad_s": surface_study.to_float(first.get("r")),
        "max_abs_alpha_deg": surface_study.safe_max_abs([surface_study.alpha_deg_from_row(row) for row in sample_rows]),
        "split": "neutral_reference",
        "filter_status": "kept",
    }
    row.update(surface_study.launch_confidence_from_inventory_row(row))
    return row


def _normalise_inventory_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["_throw_dir"] = str(out.get("throw_dir", ""))
    return out


def run_replays(
    selected: list[SelectedLaunch],
    *,
    replay_dt_s: float,
    workers: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: list[tuple[str, dict[str, Any], str, float]] = []
    for launch in selected:
        tasks.append((launch.figure_id, launch.row, "theory", float(replay_dt_s)))
        tasks.append((launch.figure_id, launch.row, "calibrated", float(replay_dt_s)))
    max_workers = min(int(workers), max(1, len(tasks)))
    traces: dict[str, dict[str, list[dict[str, Any]]]] = {
        launch.figure_id: {"real": measured_trace(launch.row)} for launch in selected
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for figure_id, model_kind, status, trace in executor.map(_replay_worker, tasks, chunksize=1):
            if status != "ok":
                raise RuntimeError(f"Replay failed for {figure_id} {model_kind}: {status}")
            traces[figure_id][model_kind] = trace
    return traces


def _replay_worker(payload: tuple[str, dict[str, Any], str, float]) -> tuple[str, str, str, list[dict[str, Any]]]:
    figure_id, row, model_kind, replay_dt_s = payload
    aircraft = build_aircraft(model_kind)
    trace, status = simulate_replay_trace(row, aircraft=aircraft, replay_dt_s=float(replay_dt_s))
    return figure_id, model_kind, status, trace


def build_aircraft(model_kind: str) -> Any:
    if model_kind == "calibrated":
        return surface_study.adapt_glider(surface_study.build_nausicaa_glider())
    if model_kind != "theory":
        raise ValueError(f"Unknown model kind: {model_kind}")
    old_value = glider_module.NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
    try:
        glider_module.NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE = False
        return surface_study.adapt_glider(glider_module.build_nausicaa_glider())
    finally:
        glider_module.NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE = old_value


def measured_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    throw_dir = Path(str(row.get("_throw_dir", row.get("throw_dir", ""))))
    sample_rows = surface_study.read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return []
    t0 = surface_study.to_float(sample_rows[0].get("t_s"), 0.0)
    trace: list[dict[str, Any]] = []
    for sample in sample_rows:
        out = {key: surface_study.to_float(sample.get(key)) for key in ("x_w", "y_w", "z_w", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "delta_a", "delta_e", "delta_r")}
        out["t_s"] = surface_study.to_float(sample.get("t_s"), t0) - t0
        trace.append(out)
    return trace


def simulate_replay_trace(
    row: dict[str, Any],
    *,
    aircraft: Any,
    replay_dt_s: float,
) -> tuple[list[dict[str, Any]], str]:
    throw_dir = Path(str(row.get("_throw_dir", row.get("throw_dir", ""))))
    sample_rows = surface_study.read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return [], "missing_state_samples"
    try:
        x0 = prep._state_vector_from_sample_row(sample_rows[0])
    except Exception:
        return [], "nonfinite_initial_state"
    if not np.all(np.isfinite(x0)):
        return [], "nonfinite_initial_state"
    t0 = surface_study.to_float(sample_rows[0].get("t_s"), 0.0)
    t1 = surface_study.to_float(sample_rows[-1].get("t_s"), t0)
    duration_s = max(0.0, t1 - t0)
    if duration_s <= 0.0:
        return [], "invalid_duration"
    manifest = surface_study.load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")
    actuator_tau_s = prep._actuator_tau_from_manifest(manifest)
    command_schedule, _ = prep._load_replay_command_schedule(
        throw_dir,
        row,
        command_onset_delay_s=float(prep.DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S),
    )
    return surface_study.simulate_trace(
        x0,
        command_schedule,
        aircraft=aircraft,
        actuator_tau_s=actuator_tau_s,
        duration_s=duration_s,
        replay_dt_s=float(replay_dt_s),
    )


def write_figures(
    selected: list[SelectedLaunch],
    traces: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for launch in selected:
        figure_path = output_root / "figures" / f"real_vs_replay_{launch.figure_id}.png"
        plot_launch_comparison(launch, traces[launch.figure_id], output_path=figure_path)
        real_final = final_state_summary(traces[launch.figure_id]["real"])
        theory_final = final_state_summary(traces[launch.figure_id]["theory"])
        calibrated_final = final_state_summary(traces[launch.figure_id]["calibrated"])
        rows.append(
            {
                "figure_run_version": FIGURE_RUN_VERSION,
                "figure_id": launch.figure_id,
                "surface_axis": launch.surface_axis,
                "command_value": launch.command_value,
                "throw_dir": str(launch.row.get("_throw_dir", launch.row.get("throw_dir", ""))),
                "launch_confidence_score": surface_study.to_float(launch.row.get("launch_confidence_score")),
                "max_abs_alpha_deg": surface_study.to_float(launch.row.get("max_abs_alpha_deg")),
                "selection_note": launch.selection_note,
                "figure_path": figure_path.as_posix(),
                "theory_final_dx_error_m": real_final["dx_m"] - theory_final["dx_m"],
                "calibrated_final_dx_error_m": real_final["dx_m"] - calibrated_final["dx_m"],
                "theory_final_dy_error_m": real_final["dy_m"] - theory_final["dy_m"],
                "calibrated_final_dy_error_m": real_final["dy_m"] - calibrated_final["dy_m"],
                "theory_altitude_loss_error_m": real_final["altitude_loss_m"] - theory_final["altitude_loss_m"],
                "calibrated_altitude_loss_error_m": real_final["altitude_loss_m"] - calibrated_final["altitude_loss_m"],
            }
        )
    return rows


def plot_launch_comparison(
    launch: SelectedLaunch,
    traces: dict[str, list[dict[str, Any]]],
    *,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=(1.35, 1.0, 1.0))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[2, 2]),
    ]

    for model_key in ("real", "theory", "calibrated"):
        trace = traces.get(model_key, [])
        if not trace:
            continue
        x = values(trace, "x_w")
        y = values(trace, "y_w")
        z = values(trace, "z_w")
        ax3d.plot(
            x,
            y,
            z,
            color=MODEL_COLORS[model_key],
            linestyle=MODEL_STYLES[model_key],
            linewidth=2.2 if model_key != "theory" else 1.8,
            label=MODEL_LABELS[model_key],
        )
        ax3d.scatter(x[0], y[0], z[0], color=MODEL_COLORS[model_key], s=18)
        ax3d.scatter(x[-1], y[-1], z[-1], color=MODEL_COLORS[model_key], marker="x", s=32)

    time_series_specs = (
        ("x_w", "x (m)", 1.0),
        ("y_w", "y (m)", 1.0),
        ("z_w", "z (m)", 1.0),
        ("phi", "roll phi (deg)", 180.0 / math.pi),
        ("theta", "pitch theta (deg)", 180.0 / math.pi),
        ("psi", "yaw psi (deg)", 180.0 / math.pi),
    )
    for axis, (field, ylabel, scale) in zip(axes, time_series_specs, strict=True):
        for model_key in ("real", "theory", "calibrated"):
            trace = traces.get(model_key, [])
            if not trace:
                continue
            axis.plot(
                values(trace, "t_s"),
                values(trace, field) * float(scale),
                color=MODEL_COLORS[model_key],
                linestyle=MODEL_STYLES[model_key],
                linewidth=2.0 if model_key != "theory" else 1.6,
                label=MODEL_LABELS[model_key],
            )
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)

    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax3d.view_init(elev=22.0, azim=-58.0)
    ax3d.legend(loc="upper left", frameon=False)

    command_text = (
        "neutral"
        if launch.surface_axis == "neutral"
        else f"{launch.surface_axis} command {launch.command_value:+.1f}"
    )
    fig.suptitle(
        f"Real launch vs theory/calibrated replay: {command_text}\n"
        f"{short_throw_label(Path(str(launch.row.get('_throw_dir', launch.row.get('throw_dir', '')))))}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def short_throw_label(throw_dir: Path) -> str:
    try:
        relative = throw_dir.resolve().relative_to(ROOT.resolve())
        parts = relative.parts
        if len(parts) >= 6:
            return "/".join(parts[-5:])
        return relative.as_posix()
    except ValueError:
        parts = throw_dir.parts
        return "/".join(parts[-5:]) if len(parts) >= 5 else throw_dir.as_posix()


def values(trace: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([surface_study.to_float(row.get(field)) for row in trace], dtype=float)


def final_state_summary(trace: list[dict[str, Any]]) -> dict[str, float]:
    if not trace:
        return {"dx_m": float("nan"), "dy_m": float("nan"), "altitude_loss_m": float("nan")}
    first = trace[0]
    last = trace[-1]
    return {
        "dx_m": surface_study.to_float(last.get("x_w")) - surface_study.to_float(first.get("x_w")),
        "dy_m": surface_study.to_float(last.get("y_w")) - surface_study.to_float(first.get("y_w")),
        "altitude_loss_m": surface_study.to_float(first.get("z_w")) - surface_study.to_float(last.get("z_w")),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "figure_run_version",
        "figure_id",
        "surface_axis",
        "command_value",
        "throw_dir",
        "launch_confidence_score",
        "max_abs_alpha_deg",
        "selection_note",
        "figure_path",
        "theory_final_dx_error_m",
        "calibrated_final_dx_error_m",
        "theory_final_dy_error_m",
        "calibrated_final_dy_error_m",
        "theory_altitude_loss_error_m",
        "calibrated_altitude_loss_error_m",
    ]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Real-Flight Replay Comparison Figures",
        "",
        "These figures compare measured real-flight launches against two dry-air replay models:",
        "",
        "- uncalibrated theory replay: comparison-only pure theory/geometry baseline",
        "- active calibrated replay: current neutral residual-calibrated model with active elevator and rudder effectiveness",
        "",
        f"- replay dt: `{manifest['replay_dt_s']}` s",
        f"- workers: `{manifest['workers']}`",
        "",
        "## Figures",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['figure_id']}`: `{row['figure_path']}` "
            f"(command `{row['command_value']}`, launch confidence `{row['launch_confidence_score']}`)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
