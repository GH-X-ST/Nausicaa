"""Targeted neutral replay ablation for post-stall Cmq and blend timing.

This is intentionally narrower than the compact joint sweep: it keeps lateral
and coupling terms off, applies the current longitudinal seed, and compares a
small hand-specified Cmq/blend grid by replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INNER_LOOP = Path(__file__).resolve().parent
if str(INNER_LOOP) not in sys.path:
    sys.path.insert(0, str(INNER_LOOP))

import run_fit_neutral_aero_residual_calibration as sysid  # noqa: E402


DEFAULT_RUN_LABEL = "n30_cmq_blend_targeted_ablation_v1"
DEFAULT_CMQ_VALUES = (4.0, 6.0, 8.0)
DEFAULT_BLEND_PAIRS_DEG = ((12.0, 22.0), (12.0, 24.0), (14.0, 22.0), (14.0, 24.0))
LONGITUDINAL_SEED = {
    "attached_pitch_moment_bias_coeff": 0.11309832420327923,
    "transition_pitch_moment_bias_coeff": 0.05711558897899738,
    "post_stall_pitch_moment_coeff": 0.07585874586245771,
}


SUMMARY_FIELDS = [
    "candidate_id",
    "split",
    "cmq",
    "blend_start_deg",
    "blend_full_deg",
    "count",
    "longitudinal_objective",
    "combined_objective",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_mae_m_s",
    "roll_mae_deg",
    "pitch_mae_deg",
    "yaw_mae_deg",
    "dx_mean_m",
    "dy_mean_m",
    "altitude_loss_mean_m",
    "sink_mean_m_s",
    "roll_mean_deg",
    "pitch_mean_deg",
    "yaw_mean_deg",
]


def parse_blend_pair(value: str) -> tuple[float, float]:
    parts = value.replace(",", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("blend pair must be START:FULL degrees")
    start_deg = float(parts[0])
    full_deg = float(parts[1])
    if full_deg <= start_deg:
        raise argparse.ArgumentTypeError("blend full alpha must be greater than blend start alpha")
    return start_deg, full_deg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=sysid.DEFAULT_SESSION_ROOT)
    parser.add_argument("--output-root", type=Path, default=sysid.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--heldout-count", type=int, default=0)
    parser.add_argument("--heldout-fraction", type=float, default=sysid.DEFAULT_HELDOUT_FRACTION)
    parser.add_argument("--heldout-seed", type=int, default=sysid.DEFAULT_HELDOUT_SEED)
    parser.add_argument("--alignment-window-s", type=float, default=sysid.DEFAULT_ALIGNMENT_WINDOW_S)
    parser.add_argument("--replay-dt-s", type=float, default=sysid.DEFAULT_REPLAY_DT_S)
    parser.add_argument("--workers", type=int, default=sysid.DEFAULT_WORKERS)
    parser.add_argument(
        "--cmq-values",
        type=float,
        nargs="+",
        default=None,
        help="Diagnostic post-stall pitch damping coefficient values to test.",
    )
    parser.add_argument(
        "--blend-pairs",
        type=parse_blend_pair,
        nargs="+",
        default=None,
        metavar="START:FULL",
        help="Diagnostic residual blend start/full alpha pairs in degrees.",
    )
    return parser


def candidate_parameters(
    base_parameters: dict[str, float],
    *,
    cmq: float,
    blend_start_deg: float,
    blend_full_deg: float,
) -> dict[str, float]:
    params = dict(base_parameters)
    params.update(LONGITUDINAL_SEED)
    params["post_stall_pitch_damping_coeff"] = sysid.replay_fit.bounded_parameter_value(
        "post_stall_pitch_damping_coeff",
        float(cmq),
    )
    params["post_stall_residual_blend_start_alpha_deg"] = sysid.replay_fit.bounded_parameter_value(
        "post_stall_residual_blend_start_alpha_deg",
        float(blend_start_deg),
    )
    params["post_stall_residual_blend_full_alpha_deg"] = sysid.replay_fit.bounded_parameter_value(
        "post_stall_residual_blend_full_alpha_deg",
        float(blend_full_deg),
    )
    for key in [
        "side_force_beta_coeff",
        "transition_side_force_r_hat_coeff",
        "transition_yaw_moment_p_hat_coeff",
        sysid.lateral_surface_parameter_name(
            "post_stall_yaw_moment",
            "p_hat",
            sysid.SURFACE_RBF_ALPHA_CENTERS_DEG[0],
        ),
    ]:
        params[key] = 0.0
    return params


def summary_row(
    *,
    candidate_id: str,
    split: str,
    cmq: float,
    blend_start_deg: float,
    blend_full_deg: float,
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    long_summary = sysid.replay_fit.objective_summary(replay_rows, objective_mode="longitudinal")
    combined_summary = sysid.replay_fit.objective_summary(replay_rows, objective_mode="combined")
    return {
        "candidate_id": candidate_id,
        "split": split,
        "cmq": float(cmq),
        "blend_start_deg": float(blend_start_deg),
        "blend_full_deg": float(blend_full_deg),
        "count": int(long_summary["count"]),
        "longitudinal_objective": long_summary["objective"],
        "combined_objective": combined_summary["objective"],
        "dx_mae_m": long_summary["dx_mae_m"],
        "dy_mae_m": long_summary["dy_mae_m"],
        "altitude_loss_mae_m": long_summary["altitude_loss_mae_m"],
        "sink_mae_m_s": long_summary["sink_mae_m_s"],
        "roll_mae_deg": long_summary["final_phi_mae_deg"],
        "pitch_mae_deg": long_summary["final_theta_mae_deg"],
        "yaw_mae_deg": long_summary["final_psi_mae_deg"],
        "dx_mean_m": long_summary["dx_mean_m"],
        "dy_mean_m": long_summary["dy_mean_m"],
        "altitude_loss_mean_m": long_summary["altitude_loss_mean_m"],
        "sink_mean_m_s": long_summary["sink_mean_m_s"],
        "roll_mean_deg": long_summary["final_phi_mean_deg"],
        "pitch_mean_deg": long_summary["final_theta_mean_deg"],
        "yaw_mean_deg": long_summary["final_psi_mean_deg"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_root / args.run_label
    metrics_dir = output_dir / "metrics"
    manifest_dir = output_dir / "manifests"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    loaded_rows = sysid.load_neutral_rows(args.session_root)
    valid_rows, filtered_rows = sysid.filter_aligned_launch_rows(
        loaded_rows,
        alignment_window_s=args.alignment_window_s,
        enabled=True,
        u_min_m_s=sysid.DEFAULT_ALIGNED_U_MIN_M_S,
        u_max_m_s=sysid.DEFAULT_ALIGNED_U_MAX_M_S,
        v_abs_max_m_s=sysid.DEFAULT_ALIGNED_V_ABS_MAX_M_S,
        w_abs_max_m_s=sysid.DEFAULT_ALIGNED_W_ABS_MAX_M_S,
    )
    heldout_count = sysid.resolved_heldout_count(
        filtered_valid_count=len(valid_rows),
        heldout_count=args.heldout_count,
        heldout_fraction=args.heldout_fraction,
    )
    heldout_indices = sysid.prep.stratified_heldout_indices(
        valid_rows,
        heldout_count=heldout_count,
        heldout_seed=args.heldout_seed,
        group_key="session_label",
    )
    train_rows = [row for index, row in enumerate(valid_rows) if index not in heldout_indices]
    heldout_rows = [row for index, row in enumerate(valid_rows) if index in heldout_indices]

    cmq_values = tuple(args.cmq_values) if args.cmq_values is not None else DEFAULT_CMQ_VALUES
    blend_pairs_deg = tuple(args.blend_pairs) if args.blend_pairs is not None else DEFAULT_BLEND_PAIRS_DEG
    base_parameters = sysid.active_parameter_dict()
    summary_rows: list[dict[str, Any]] = []
    replay_rows_out: list[dict[str, Any]] = []
    for blend_start_deg, blend_full_deg in blend_pairs_deg:
        for cmq in cmq_values:
            candidate_id = f"cmq{cmq:g}_blend{blend_start_deg:g}_{blend_full_deg:g}"
            params = candidate_parameters(
                base_parameters,
                cmq=cmq,
                blend_start_deg=blend_start_deg,
                blend_full_deg=blend_full_deg,
            )
            for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
                replay_rows = sysid.replay_fit.simulate_rows(
                    rows,
                    params,
                    replay_dt_s=args.replay_dt_s,
                    alignment_window_s=args.alignment_window_s,
                    workers=args.workers,
                )
                summary_rows.append(
                    summary_row(
                        candidate_id=candidate_id,
                        split=split,
                        cmq=cmq,
                        blend_start_deg=blend_start_deg,
                        blend_full_deg=blend_full_deg,
                        replay_rows=replay_rows,
                    )
                )
                for replay_row in replay_rows:
                    tagged = dict(replay_row)
                    tagged.update(
                        {
                            "candidate_id": candidate_id,
                            "split": split,
                            "cmq": float(cmq),
                            "blend_start_deg": float(blend_start_deg),
                            "blend_full_deg": float(blend_full_deg),
                        }
                    )
                    replay_rows_out.append(tagged)

    replay_fields = [
        "candidate_id",
        "cmq",
        "blend_start_deg",
        "blend_full_deg",
        *sysid.replay_fit.REPLAY_RESIDUAL_FIELDS,
    ]
    write_csv(metrics_dir / "cmq_blend_targeted_ablation.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(metrics_dir / "cmq_blend_targeted_replay_rows.csv", replay_rows_out, replay_fields)

    heldout_rows_summary = [row for row in summary_rows if row["split"] == "heldout"]
    best_longitudinal = min(heldout_rows_summary, key=lambda row: row["longitudinal_objective"])
    best_combined = min(heldout_rows_summary, key=lambda row: row["combined_objective"])
    manifest = {
        "fit_id": args.run_label,
        "policy": "targeted longitudinal diagnostic only; lateral and coupling terms held at zero",
        "loaded_throw_count": len(loaded_rows),
        "valid_throw_count": len(valid_rows),
        "filtered_throw_count": int(sum(not bool(row.get("kept", False)) for row in filtered_rows)),
        "heldout_count": len(heldout_rows),
        "heldout_seed": int(args.heldout_seed),
        "alignment_window_s": float(args.alignment_window_s),
        "replay_dt_s": float(args.replay_dt_s),
        "workers": int(args.workers),
        "tested_cmq": list(cmq_values),
        "tested_blend_pairs_deg": [list(pair) for pair in blend_pairs_deg],
        "longitudinal_seed": LONGITUDINAL_SEED,
        "best_heldout_longitudinal": best_longitudinal,
        "best_heldout_combined": best_combined,
    }
    (manifest_dir / "cmq_blend_targeted_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return output_dir


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = run(args)
    print(f"[DONE] targeted Cmq/blend ablation written to {output_dir}")


if __name__ == "__main__":
    main()
