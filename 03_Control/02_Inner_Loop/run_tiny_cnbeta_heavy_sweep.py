"""Heavy but narrow Cn_beta sweep for neutral replay lateral correction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fit_neutral_aero_residual_calibration as fit  # noqa: E402
import run_fit_neutral_dry_air_calibration as replay_fit  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402


DEFAULT_REFERENCE_RESULT_ROOT = (
    REPO_ROOT
    / "03_Control"
    / "05_Results"
    / "glider_model_calibration_prep"
    / "n30_tiny_cnbeta_diagnostic_v1"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_RUN_LABEL = "n30_tiny_cnbeta_heavy_sweep_v1"
DEFAULT_WORKERS = 8

ATTACHED_CNBETA_GRID = (0.0, -0.0125, -0.025, -0.0375, -0.0487479808)
TRANSITION_CNBETA_GRID = (0.0, -0.007337912235, -0.01467582447, -0.022013736705, -0.02935164894, -0.04402747341)
POST_STALL_CNBETA_GRID = (0.0, -0.01870860145, -0.0374172029, -0.05612580435, -0.0748344058, -0.1122516087)

LONGITUDINAL_TOLERANCES = {
    "dx_mae_m": 0.02,
    "altitude_loss_mae_m": 0.02,
    "sink_mae_m_s": 0.02,
    "pitch_mae_deg": 0.50,
}
TRAIN_LATERAL_TOLERANCES = {
    "dy_mae_m": 0.03,
    "roll_mae_deg": 1.0,
    "yaw_mae_deg": 1.0,
}

CANDIDATE_FIELDS = [
    "candidate_id",
    "split",
    "throw_count",
    "yaw_moment_beta_coeff",
    "transition_yaw_moment_beta_coeff",
    "post_stall_yaw_moment_beta_rbf_20_coeff",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_mae_m_s",
    "roll_mae_deg",
    "pitch_mae_deg",
    "yaw_mae_deg",
    "objective",
    "heldout_strict_gate",
    "train_guard_gate",
    "selection_score",
    "selection_reason",
]
SELECTED_FIELDS = [
    "selection_class",
    "candidate_id",
    "selection_reason",
    "selection_score",
    "yaw_moment_beta_coeff",
    "transition_yaw_moment_beta_coeff",
    "post_stall_yaw_moment_beta_rbf_20_coeff",
    "heldout_dx_mae_m",
    "heldout_dy_mae_m",
    "heldout_altitude_loss_mae_m",
    "heldout_sink_mae_m_s",
    "heldout_roll_mae_deg",
    "heldout_pitch_mae_deg",
    "heldout_yaw_mae_deg",
    "delta_dx_mae_m",
    "delta_dy_mae_m",
    "delta_altitude_loss_mae_m",
    "delta_sink_mae_m_s",
    "delta_roll_mae_deg",
    "delta_pitch_mae_deg",
    "delta_yaw_mae_deg",
]


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_root) / str(args.run_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)

    reference_manifest = json.loads(
        (Path(args.reference_result_root) / "manifests" / "neutral_aero_residual_fit_manifest.json").read_text()
    )
    train_rows, heldout_rows = split_rows_from_manifest(reference_manifest)
    base_parameters = fit.active_parameter_dict()
    candidates = build_candidate_grid(base_parameters)
    rows = evaluate_candidate_grid(
        candidates,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        replay_dt_s=float(reference_manifest["replay_dt_s"]),
        alignment_window_s=float(reference_manifest["alignment_window_s"]),
        workers=DEFAULT_WORKERS,
    )
    selected = selected_rows(rows)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_cnbeta_heavy_sweep_candidates.csv", rows, CANDIDATE_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_cnbeta_heavy_sweep_selected.csv", selected, SELECTED_FIELDS)
    write_manifest(
        output_dir,
        run_label=str(args.run_label),
        reference_manifest=reference_manifest,
        candidate_count=len(candidates),
        train_count=len(train_rows),
        heldout_count=len(heldout_rows),
        selected=selected,
    )
    print(output_dir.as_posix())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a heavier narrow Cn_beta grid sweep on the frozen neutral split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-result-root", type=Path, default=DEFAULT_REFERENCE_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    return parser


def split_rows_from_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loaded = fit.load_neutral_rows(Path(str(manifest["session_root"])))
    launch_filter = manifest["aligned_launch_filter"]
    valid_rows, _ = fit.filter_aligned_launch_rows(
        loaded,
        alignment_window_s=float(manifest["alignment_window_s"]),
        enabled=bool(launch_filter["enabled"]),
        u_min_m_s=float(launch_filter["u_min_m_s"]),
        u_max_m_s=float(launch_filter["u_max_m_s"]),
        v_abs_max_m_s=float(launch_filter["v_abs_max_m_s"]),
        w_abs_max_m_s=float(launch_filter["w_abs_max_m_s"]),
    )
    heldout_indices = {int(index) for index in manifest["heldout_indices"]}
    train_rows = [row for index, row in enumerate(valid_rows) if index not in heldout_indices]
    heldout_rows = [row for index, row in enumerate(valid_rows) if index in heldout_indices]
    return train_rows, heldout_rows


def build_candidate_grid(base_parameters: dict[str, float]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    index = 0
    for attached in ATTACHED_CNBETA_GRID:
        for transition in TRANSITION_CNBETA_GRID:
            for post_stall in POST_STALL_CNBETA_GRID:
                parameters = dict(base_parameters)
                parameters["yaw_moment_beta_coeff"] = float(attached)
                parameters["transition_yaw_moment_beta_coeff"] = float(transition)
                parameters["post_stall_yaw_moment_beta_rbf_20_coeff"] = float(post_stall)
                candidates.append(
                    {
                        "candidate_id": f"H{index:04d}_Cn_beta",
                        "parameters": parameters,
                    }
                )
                index += 1
    return candidates


def evaluate_candidate_grid(
    candidates: list[dict[str, Any]],
    *,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = [
        (candidate, train_rows, heldout_rows, float(replay_dt_s), float(alignment_window_s))
        for candidate in candidates
    ]
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            nested = list(executor.map(evaluate_candidate_payload, payloads, chunksize=1))
    else:
        nested = [evaluate_candidate_payload(payload) for payload in payloads]
    rows = [row for candidate_rows in nested for row in candidate_rows]
    decorate_rows_with_gates(rows)
    return rows


def evaluate_candidate_payload(
    payload: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], float, float],
) -> list[dict[str, Any]]:
    candidate, train_rows, heldout_rows, replay_dt_s, alignment_window_s = payload
    parameters = dict(candidate["parameters"])
    output = []
    for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            replay_rows = replay_fit.simulate_rows(
                rows,
                parameters,
                replay_dt_s=float(replay_dt_s),
                alignment_window_s=float(alignment_window_s),
                workers=1,
            )
        summary = replay_fit.objective_summary(replay_rows)
        output.append(summary_row(str(candidate["candidate_id"]), split, rows, parameters, summary))
    return output


def summary_row(
    candidate_id: str,
    split: str,
    rows: list[dict[str, Any]],
    parameters: dict[str, float],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "split": split,
        "throw_count": len(rows),
        "yaw_moment_beta_coeff": float(parameters.get("yaw_moment_beta_coeff", 0.0)),
        "transition_yaw_moment_beta_coeff": float(parameters.get("transition_yaw_moment_beta_coeff", 0.0)),
        "post_stall_yaw_moment_beta_rbf_20_coeff": float(
            parameters.get("post_stall_yaw_moment_beta_rbf_20_coeff", 0.0)
        ),
        "dx_mae_m": finite(summary.get("dx_mae_m")),
        "dy_mae_m": finite(summary.get("dy_mae_m")),
        "altitude_loss_mae_m": finite(summary.get("altitude_loss_mae_m")),
        "sink_mae_m_s": finite(summary.get("sink_mae_m_s")),
        "roll_mae_deg": finite(summary.get("final_phi_mae_deg")),
        "pitch_mae_deg": finite(summary.get("final_theta_mae_deg")),
        "yaw_mae_deg": finite(summary.get("final_psi_mae_deg")),
        "objective": finite(summary.get("objective")),
        "heldout_strict_gate": "",
        "train_guard_gate": "",
        "selection_score": "",
        "selection_reason": "",
    }


def decorate_rows_with_gates(rows: list[dict[str, Any]]) -> None:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_id"]), {})[str(row["split"])] = row
    base = current_active_rows(by_candidate)
    base_train = base["train"]
    base_heldout = base["heldout"]
    for candidate_id, splits in by_candidate.items():
        train = splits.get("train")
        heldout = splits.get("heldout")
        if train is None or heldout is None:
            continue
        heldout_gate, heldout_reason = heldout_strict_gate(heldout, base_heldout)
        train_gate, train_reason = train_guard_gate(train, base_train)
        score = selection_score(heldout, base_heldout)
        for split_row in (train, heldout):
            split_row["heldout_strict_gate"] = bool(heldout_gate)
            split_row["train_guard_gate"] = bool(train_gate)
            split_row["selection_score"] = float(score)
            split_row["selection_reason"] = (
                "passes_strict_heldout_and_train_guard"
                if heldout_gate and train_gate
                else f"blocked:{heldout_reason};{train_reason}"
            )


def current_active_rows(by_candidate: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    attached = float(getattr(active_calibration, "YAW_MOMENT_BETA_COEFF", 0.0))
    transition = float(getattr(active_calibration, "TRANSITION_YAW_MOMENT_BETA_COEFF", 0.0))
    post_stall = float(active_calibration.POST_STALL_YAW_MOMENT_RBF_COEFFS[1][0])
    for splits in by_candidate.values():
        heldout = splits.get("heldout")
        if heldout is None:
            continue
        if (
            math.isclose(float(heldout["yaw_moment_beta_coeff"]), attached, rel_tol=0.0, abs_tol=1.0e-10)
            and math.isclose(
                float(heldout["transition_yaw_moment_beta_coeff"]),
                transition,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            and math.isclose(
                float(heldout["post_stall_yaw_moment_beta_rbf_20_coeff"]),
                post_stall,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        ):
            return splits
    raise RuntimeError("Current active Cn_beta candidate is missing from sweep grid.")


def heldout_strict_gate(row: dict[str, Any], base: dict[str, Any]) -> tuple[bool, str]:
    lateral = (
        float(row["dy_mae_m"]) < float(base["dy_mae_m"]) - 1.0e-9
        and float(row["roll_mae_deg"]) <= float(base["roll_mae_deg"]) + 1.0e-9
        and float(row["yaw_mae_deg"]) <= float(base["yaw_mae_deg"]) + 1.0e-9
    )
    if not lateral:
        return False, "heldout_lateral_not_all_nonworse_with_dy_improved"
    for key, tolerance in LONGITUDINAL_TOLERANCES.items():
        if float(row[key]) > float(base[key]) + float(tolerance):
            return False, f"heldout_{key}_degraded"
    return True, "heldout_pass"


def train_guard_gate(row: dict[str, Any], base: dict[str, Any]) -> tuple[bool, str]:
    for key, tolerance in TRAIN_LATERAL_TOLERANCES.items():
        if float(row[key]) > float(base[key]) + float(tolerance):
            return False, f"train_{key}_degraded"
    for key, tolerance in LONGITUDINAL_TOLERANCES.items():
        if float(row[key]) > float(base[key]) + float(tolerance):
            return False, f"train_{key}_degraded"
    return True, "train_pass"


def selection_score(row: dict[str, Any], base: dict[str, Any]) -> float:
    dy_gain = max(0.0, float(base["dy_mae_m"]) - float(row["dy_mae_m"]))
    roll_gain = max(0.0, float(base["roll_mae_deg"]) - float(row["roll_mae_deg"]))
    yaw_gain = max(0.0, float(base["yaw_mae_deg"]) - float(row["yaw_mae_deg"]))
    longitudinal_penalty = 0.0
    for key in ("dx_mae_m", "altitude_loss_mae_m", "sink_mae_m_s"):
        longitudinal_penalty += max(0.0, float(row[key]) - float(base[key]))
    longitudinal_penalty += 0.02 * max(0.0, float(row["pitch_mae_deg"]) - float(base["pitch_mae_deg"]))
    return float(dy_gain + 0.01 * roll_gain + 0.02 * yaw_gain - 3.0 * longitudinal_penalty)


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_id"]), {})[str(row["split"])] = row
    base = current_active_rows(by_candidate)
    base_heldout = base["heldout"]
    accepted = [
        splits
        for splits in by_candidate.values()
        if splits.get("heldout", {}).get("heldout_strict_gate") is True
        and splits.get("heldout", {}).get("train_guard_gate") is True
    ]
    if accepted:
        best = max(accepted, key=lambda splits: float(splits["heldout"]["selection_score"]))
        reason = "best_strict_heavy_sweep_candidate"
    else:
        best = base
        reason = "no_strict_candidate_beats_current_active"
    heldout = best["heldout"]
    return [
        {
            "selection_class": "strict_best",
            "candidate_id": heldout["candidate_id"],
            "selection_reason": reason,
            "selection_score": heldout["selection_score"],
            "yaw_moment_beta_coeff": heldout["yaw_moment_beta_coeff"],
            "transition_yaw_moment_beta_coeff": heldout["transition_yaw_moment_beta_coeff"],
            "post_stall_yaw_moment_beta_rbf_20_coeff": heldout["post_stall_yaw_moment_beta_rbf_20_coeff"],
            "heldout_dx_mae_m": heldout["dx_mae_m"],
            "heldout_dy_mae_m": heldout["dy_mae_m"],
            "heldout_altitude_loss_mae_m": heldout["altitude_loss_mae_m"],
            "heldout_sink_mae_m_s": heldout["sink_mae_m_s"],
            "heldout_roll_mae_deg": heldout["roll_mae_deg"],
            "heldout_pitch_mae_deg": heldout["pitch_mae_deg"],
            "heldout_yaw_mae_deg": heldout["yaw_mae_deg"],
            "delta_dx_mae_m": float(heldout["dx_mae_m"]) - float(base_heldout["dx_mae_m"]),
            "delta_dy_mae_m": float(heldout["dy_mae_m"]) - float(base_heldout["dy_mae_m"]),
            "delta_altitude_loss_mae_m": float(heldout["altitude_loss_mae_m"]) - float(
                base_heldout["altitude_loss_mae_m"]
            ),
            "delta_sink_mae_m_s": float(heldout["sink_mae_m_s"]) - float(base_heldout["sink_mae_m_s"]),
            "delta_roll_mae_deg": float(heldout["roll_mae_deg"]) - float(base_heldout["roll_mae_deg"]),
            "delta_pitch_mae_deg": float(heldout["pitch_mae_deg"]) - float(base_heldout["pitch_mae_deg"]),
            "delta_yaw_mae_deg": float(heldout["yaw_mae_deg"]) - float(base_heldout["yaw_mae_deg"]),
        }
    ]


def write_manifest(
    output_dir: Path,
    *,
    run_label: str,
    reference_manifest: dict[str, Any],
    candidate_count: int,
    train_count: int,
    heldout_count: int,
    selected: list[dict[str, Any]],
) -> None:
    payload = {
        "run_label": run_label,
        "status": "complete",
        "reference_result_root": DEFAULT_REFERENCE_RESULT_ROOT.as_posix(),
        "active_calibration_id": str(active_calibration.CALIBRATION_ID),
        "loaded_throw_count": int(reference_manifest["loaded_throw_count"]),
        "valid_throw_count": int(reference_manifest["valid_throw_count"]),
        "filtered_throw_count": int(reference_manifest["filtered_throw_count"]),
        "train_throw_count": int(train_count),
        "heldout_throw_count": int(heldout_count),
        "heldout_policy": str(reference_manifest["heldout_policy"]),
        "heldout_seed": int(reference_manifest["heldout_seed"]),
        "workers": int(DEFAULT_WORKERS),
        "candidate_count": int(candidate_count),
        "attached_cnbeta_grid": list(ATTACHED_CNBETA_GRID),
        "transition_cnbeta_grid": list(TRANSITION_CNBETA_GRID),
        "post_stall_cnbeta_grid": list(POST_STALL_CNBETA_GRID),
        "selection": selected,
        "claim_boundary": "diagnostic_heavy_sweep_for_tiny_cnbeta_replay_correction_only",
    }
    (output_dir / "manifests" / "neutral_aero_residual_cnbeta_heavy_sweep_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
