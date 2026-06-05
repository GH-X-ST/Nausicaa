"""Evaluate tiny Cn_beta lateral candidates against a frozen neutral baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fit_neutral_aero_residual_calibration as fit  # noqa: E402
import run_fit_neutral_dry_air_calibration as replay_fit  # noqa: E402


DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "03_Control"
    / "05_Results"
    / "glider_model_calibration_prep"
    / "n30_tiny_cnbeta_diagnostic_v1"
)
DEFAULT_WORKERS = 8
SUMMARY_FIELDS = [
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
]


def main() -> None:
    args = build_arg_parser().parse_args()
    result_root = Path(args.result_root)
    summary_path = result_root / "metrics" / "neutral_aero_residual_cnbeta_candidate_summary.csv"
    rows = evaluate_candidates(result_root=result_root, workers=DEFAULT_WORKERS)
    write_csv(summary_path, rows, SUMMARY_FIELDS)
    print(summary_path.as_posix())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate attached/transition/post-stall Cn_beta candidate combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser


def evaluate_candidates(*, result_root: Path, workers: int) -> list[dict[str, Any]]:
    manifest = json.loads((result_root / "manifests" / "neutral_aero_residual_fit_manifest.json").read_text())
    train_rows, heldout_rows = split_rows_from_manifest(manifest)
    coefficients = cnbeta_coefficients(result_root)
    base = fit.active_parameter_dict()
    candidates = {
        "L0_active": dict(base),
        "L1_attached_Cn_beta": {
            **base,
            "yaw_moment_beta_coeff": coefficients["attached"],
        },
        "L2_transition_Cn_beta": {
            **base,
            "transition_yaw_moment_beta_coeff": coefficients["transition"],
        },
        "L3_post_stall_Cn_beta": {
            **base,
            "post_stall_yaw_moment_beta_rbf_20_coeff": coefficients["post_stall"],
        },
        "L4_transition_post_stall_Cn_beta": {
            **base,
            "transition_yaw_moment_beta_coeff": coefficients["transition"],
            "post_stall_yaw_moment_beta_rbf_20_coeff": coefficients["post_stall"],
        },
        "L5_all_Cn_beta": {
            **base,
            "yaw_moment_beta_coeff": coefficients["attached"],
            "transition_yaw_moment_beta_coeff": coefficients["transition"],
            "post_stall_yaw_moment_beta_rbf_20_coeff": coefficients["post_stall"],
        },
    }
    output: list[dict[str, Any]] = []
    for candidate_id, parameters in candidates.items():
        for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                replay_rows = replay_fit.simulate_rows(
                    rows,
                    parameters,
                    replay_dt_s=float(manifest["replay_dt_s"]),
                    alignment_window_s=float(manifest["alignment_window_s"]),
                    workers=int(workers),
                )
            summary = replay_fit.objective_summary(replay_rows)
            output.append(summary_row(candidate_id, split, rows, parameters, summary))
    return output


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


def cnbeta_coefficients(result_root: Path) -> dict[str, float]:
    rows = read_csv(result_root / "metrics" / "neutral_aero_residual_lateral_ablation.csv")
    out: dict[str, float] = {}
    for regime in ("attached", "transition", "post_stall"):
        matches = [
            row
            for row in rows
            if row.get("term") == "Cn_beta"
            and row.get("regime_family") == regime
            and row.get("split") == "heldout"
        ]
        if not matches:
            raise RuntimeError(f"Missing heldout Cn_beta row for {regime}")
        out[regime] = float(matches[0]["fit_coefficient"])
    return out


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
        "dx_mae_m": summary.get("dx_mae_m"),
        "dy_mae_m": summary.get("dy_mae_m"),
        "altitude_loss_mae_m": summary.get("altitude_loss_mae_m"),
        "sink_mae_m_s": summary.get("sink_mae_m_s"),
        "roll_mae_deg": summary.get("final_phi_mae_deg"),
        "pitch_mae_deg": summary.get("final_theta_mae_deg"),
        "yaw_mae_deg": summary.get("final_psi_mae_deg"),
        "objective": summary.get("objective"),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
