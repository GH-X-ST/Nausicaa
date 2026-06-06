"""Audit-only local Pareto replay around the heavy 40 ms SysID front.

This script consumes an existing heavy SysID result and evaluates only a
narrow local candidate grid on the same held-out split. It does not run the
full residual extraction or calibration workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import run_fit_neutral_aero_residual_calibration as sysid


DEFAULT_SOURCE_RUN_DIR = Path("03_Control/05_Results/glider_model_calibration_prep/n30_joint_pareto_040_heavy_v1")
DEFAULT_RUN_LABEL = "n30_joint_pareto_040_local_promising_v1"
LONGITUDINAL_SOURCE_IDS = (
    "proposal_stage_5_stage5_transition_blend",
    "proposal_stage_9_stage9_post_blend_post_stall_lift_dr",
    "active_baseline",
)
YAW_BETA_KEY = "yaw_moment_beta_coeff"
POST_STALL_CLR_KEY = "post_stall_roll_moment_r_hat_rbf_20_coeff"
YAW_BETA_SCALE_GRID = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
POST_STALL_CLR_SCALE_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
STRICT_LONGITUDINAL_TOLERANCES = {
    "dx_mae_m": 0.025,
    "altitude_loss_mae_m": 0.010,
    "sink_mae_m_s": 0.010,
    "final_theta_mae_deg": 0.5,
}
YAW_SOURCE_PATTERN = re.compile(r"ablation_attached_Cn_beta__yaw_moment_beta_coeff__s(?P<scale>[0-9pm]+)")
CLR_SOURCE_PATTERN = re.compile(
    r"ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s(?P<scale>[0-9pm]+)"
)


def main() -> None:
    args = build_arg_parser().parse_args()
    run_local_audit(
        source_run_dir=Path(args.source_run_dir),
        output_root=Path(args.output_root),
        run_label=str(args.run_label),
        workers=int(args.workers),
        selected_limit=int(args.selected_limit),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=Path("03_Control/05_Results/glider_model_calibration_prep"))
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--selected-limit", type=int, default=8)
    return parser


def run_local_audit(
    *,
    source_run_dir: Path,
    output_root: Path,
    run_label: str,
    workers: int,
    selected_limit: int,
) -> Path:
    source_run_dir = source_run_dir.resolve()
    manifest = read_json(source_run_dir / "manifests" / "neutral_aero_residual_fit_manifest.json")
    heavy_candidates = read_csv(source_run_dir / "metrics" / "neutral_aero_residual_joint_pareto_heavy_candidates.csv")
    base_parameters = {str(key): float(value) for key, value in dict(manifest["base_parameters"]).items()}
    alignment_window_s = float(manifest["joint_pareto_audit"]["alignment_window_s"])
    replay_dt_s = float(manifest["replay_dt_s"])
    heldout_indices = {int(value) for value in manifest["heldout_indices"]}
    workers = max(1, int(workers))

    heldout_rows = heldout_rows_from_manifest(manifest, heldout_indices=heldout_indices)
    longitudinal_sources = local_longitudinal_sources(
        heavy_candidates,
        base_parameters=base_parameters,
    )
    source_values = {
        YAW_BETA_KEY: infer_scaled_source_value(
            heavy_candidates,
            base_parameters=base_parameters,
            key=YAW_BETA_KEY,
            source_pattern=YAW_SOURCE_PATTERN,
        ),
        POST_STALL_CLR_KEY: infer_scaled_source_value(
            heavy_candidates,
            base_parameters=base_parameters,
            key=POST_STALL_CLR_KEY,
            source_pattern=CLR_SOURCE_PATTERN,
        ),
    }
    candidate_states = build_local_candidate_states(
        longitudinal_sources=longitudinal_sources,
        base_parameters=base_parameters,
        source_values=source_values,
        alignment_window_s=alignment_window_s,
    )
    expected_count = len(LONGITUDINAL_SOURCE_IDS) * (
        1 + len(YAW_BETA_SCALE_GRID) + len(POST_STALL_CLR_SCALE_GRID) + len(YAW_BETA_SCALE_GRID) * len(POST_STALL_CLR_SCALE_GRID)
    )
    if len(candidate_states) != expected_count:
        raise RuntimeError(f"Expected {expected_count} local candidates, generated {len(candidate_states)}.")

    evaluated = sysid.compact_joint_sweep_evaluate_states(
        candidate_states,
        rows=heldout_rows,
        split="heldout",
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    reference_candidates = [
        state
        for state in evaluated
        if str(state.get("lateral_source_id", "")) == "no_lateral_update"
    ]
    reference_by_longitudinal = {
        str(state.get("longitudinal_source_id", "")): state
        for state in reference_candidates
    }
    global_reference_state = min(
        reference_candidates or evaluated,
        key=lambda state: sysid.finite_value(state.get("longitudinal_score")),
    )
    candidate_rows = sysid.joint_pareto_audit_candidate_rows(
        evaluated,
        base_parameters=base_parameters,
        reference_state=global_reference_state,
        reference_by_longitudinal=reference_by_longitudinal,
        global_reference_state=global_reference_state,
        alignment_window_s=alignment_window_s,
        profile="local_audit_only",
        longitudinal_tolerances=STRICT_LONGITUDINAL_TOLERANCES,
    )
    sysid.joint_pareto_audit_mark_pareto(candidate_rows)
    selected_rows = sysid.joint_pareto_audit_selected_rows(candidate_rows, selected_limit=selected_limit)
    stage_replay_rows = sysid.joint_pareto_heavy_stage_replay_summary_rows(
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        base_parameters=base_parameters,
        heldout_rows=heldout_rows,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )

    output_dir = output_root / run_label
    write_outputs(
        output_dir=output_dir,
        run_label=run_label,
        source_run_dir=source_run_dir,
        workers=workers,
        alignment_window_s=alignment_window_s,
        replay_dt_s=replay_dt_s,
        heldout_indices=heldout_indices,
        source_values=source_values,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        stage_replay_rows=stage_replay_rows,
    )
    return output_dir


def heldout_rows_from_manifest(manifest: dict[str, Any], *, heldout_indices: set[int]) -> list[dict[str, Any]]:
    launch_filter = dict(manifest.get("aligned_launch_filter", {}))
    loaded_rows = sysid.load_neutral_rows(Path(str(manifest["session_root"])))
    valid_rows, _ = sysid.filter_aligned_launch_rows(
        loaded_rows,
        alignment_window_s=float(manifest["alignment_window_s"]),
        enabled=bool(launch_filter.get("enabled", True)),
        u_min_m_s=float(launch_filter.get("u_min_m_s", sysid.DEFAULT_ALIGNED_U_MIN_M_S)),
        u_max_m_s=float(launch_filter.get("u_max_m_s", sysid.DEFAULT_ALIGNED_U_MAX_M_S)),
        v_abs_max_m_s=float(launch_filter.get("v_abs_max_m_s", sysid.DEFAULT_ALIGNED_V_ABS_MAX_M_S)),
        w_abs_max_m_s=float(launch_filter.get("w_abs_max_m_s", sysid.DEFAULT_ALIGNED_W_ABS_MAX_M_S)),
    )
    return [row for index, row in enumerate(valid_rows) if index in heldout_indices]


def local_longitudinal_sources(
    heavy_candidates: list[dict[str, str]],
    *,
    base_parameters: dict[str, float],
) -> list[dict[str, Any]]:
    no_lateral_rows = [
        row for row in heavy_candidates if str(row.get("lateral_source_id", "")) == "no_lateral_update"
    ]
    by_source = {str(row.get("longitudinal_source_id", "")): row for row in no_lateral_rows}
    missing = [source_id for source_id in LONGITUDINAL_SOURCE_IDS if source_id not in by_source]
    if missing:
        raise RuntimeError("Missing heavy no-lateral rows for local sources: " + ",".join(missing))
    out: list[dict[str, Any]] = []
    for source_id in LONGITUDINAL_SOURCE_IDS:
        row = by_source[source_id]
        params = dict(base_parameters)
        params.update(parse_parameter_updates(row.get("parameter_updates_json", "{}")))
        out.append(
            {
                "candidate_id": source_id,
                "parameters": params,
                "updates": sysid.parameter_updates(base_parameters, params),
            }
        )
    return out


def infer_scaled_source_value(
    heavy_candidates: list[dict[str, str]],
    *,
    base_parameters: dict[str, float],
    key: str,
    source_pattern: re.Pattern[str],
) -> float:
    base_value = float(base_parameters.get(key, 0.0))
    inferred: list[float] = []
    for row in heavy_candidates:
        updates = parse_parameter_updates(row.get("parameter_updates_json", "{}"))
        if key not in updates:
            continue
        match = source_pattern.search(str(row.get("lateral_source_id", "")))
        if not match:
            continue
        scale = scale_label_to_float(match.group("scale"))
        if abs(scale) <= 1.0e-12:
            continue
        inferred.append(base_value + (float(updates[key]) - base_value) / scale)
    if not inferred:
        raise RuntimeError(f"Could not infer local source value for {key}.")
    median = float(sorted(inferred)[len(inferred) // 2])
    if max(abs(value - median) for value in inferred) > 1.0e-8:
        raise RuntimeError(f"Inconsistent inferred source values for {key}: {inferred[:6]}")
    return median


def build_local_candidate_states(
    *,
    longitudinal_sources: list[dict[str, Any]],
    base_parameters: dict[str, float],
    source_values: dict[str, float],
    alignment_window_s: float,
) -> list[dict[str, Any]]:
    prefix = f"jp{int(round(float(alignment_window_s) * 1000.0)):03d}local"
    states: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_state(
        *,
        long_index: int,
        longitudinal: dict[str, Any],
        lateral_source_id: str,
        bundle_order: int,
        yaw_scale: float | None = None,
        clr_scale: float | None = None,
    ) -> None:
        lateral_parameters = dict(base_parameters)
        if yaw_scale is not None:
            lateral_parameters[YAW_BETA_KEY] = scaled_parameter_value(
                base_parameters=base_parameters,
                key=YAW_BETA_KEY,
                source_value=source_values[YAW_BETA_KEY],
                scale=float(yaw_scale),
            )
        if clr_scale is not None:
            lateral_parameters[POST_STALL_CLR_KEY] = scaled_parameter_value(
                base_parameters=base_parameters,
                key=POST_STALL_CLR_KEY,
                source_value=source_values[POST_STALL_CLR_KEY],
                scale=float(clr_scale),
            )
        params = sysid.joint_pareto_combined_parameters(
            longitudinal["parameters"],
            base_parameters=base_parameters,
            lateral_parameters=lateral_parameters,
        )
        signature = sysid.compact_joint_sweep_signature(params, base_parameters)
        if signature in seen:
            return
        seen.add(signature)
        states.append(
            {
                "candidate_id": (
                    f"{prefix}_L{long_index:02d}_{sysid.short_source_id(str(longitudinal['candidate_id']), limit=28)}"
                    f"_{sysid.short_source_id(lateral_source_id, limit=52)}"
                ),
                "parameters": params,
                "updates": sysid.parameter_updates(base_parameters, params),
                "sweep_stage": "joint_pareto_local_audit_only",
                "summary": None,
                "longitudinal_source_id": str(longitudinal["candidate_id"]),
                "lateral_source_id": lateral_source_id,
                "bundle_order": int(bundle_order),
            }
        )

    for long_index, longitudinal in enumerate(longitudinal_sources):
        add_state(
            long_index=long_index,
            longitudinal=longitudinal,
            lateral_source_id="no_lateral_update",
            bundle_order=0,
        )
        for yaw_scale in YAW_BETA_SCALE_GRID:
            add_state(
                long_index=long_index,
                longitudinal=longitudinal,
                lateral_source_id=f"local_yaw_beta_s{scale_id(yaw_scale)}",
                bundle_order=1,
                yaw_scale=yaw_scale,
            )
        for clr_scale in POST_STALL_CLR_SCALE_GRID:
            add_state(
                long_index=long_index,
                longitudinal=longitudinal,
                lateral_source_id=f"local_post_stall_Cl_r_s{scale_id(clr_scale)}",
                bundle_order=1,
                clr_scale=clr_scale,
            )
        for yaw_scale in YAW_BETA_SCALE_GRID:
            for clr_scale in POST_STALL_CLR_SCALE_GRID:
                add_state(
                    long_index=long_index,
                    longitudinal=longitudinal,
                    lateral_source_id=f"local_yaw_beta_s{scale_id(yaw_scale)}+local_post_stall_Cl_r_s{scale_id(clr_scale)}",
                    bundle_order=2,
                    yaw_scale=yaw_scale,
                    clr_scale=clr_scale,
                )
    return states


def scaled_parameter_value(
    *,
    base_parameters: dict[str, float],
    key: str,
    source_value: float,
    scale: float,
) -> float:
    base_value = float(base_parameters.get(key, 0.0))
    value = base_value + float(scale) * (float(source_value) - base_value)
    return sysid.replay_fit.bounded_parameter_value(key, value)


def write_outputs(
    *,
    output_dir: Path,
    run_label: str,
    source_run_dir: Path,
    workers: int,
    alignment_window_s: float,
    replay_dt_s: float,
    heldout_indices: set[int],
    source_values: dict[str, float],
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    stage_replay_rows: list[dict[str, Any]],
) -> None:
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    sysid.write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_pareto_local_candidates.csv",
        candidate_rows,
        sysid.JOINT_PARETO_AUDIT_FIELDS,
    )
    sysid.write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_pareto_local_selected.csv",
        selected_rows,
        sysid.JOINT_PARETO_AUDIT_FIELDS,
    )
    sysid.write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_pareto_local_stage_replay.csv",
        stage_replay_rows,
        sysid.JOINT_PARETO_HEAVY_STAGE_REPLAY_FIELDS,
    )
    manifest = {
        "fit_id": str(run_label),
        "artifact": "neutral_aero_residual_joint_pareto_local",
        "audit_only": True,
        "source_run": source_run_dir.name,
        "source_run_dir": str(source_run_dir),
        "rerun_full_calibration": False,
        "alignment_window_s": float(alignment_window_s),
        "replay_dt_s": float(replay_dt_s),
        "workers": int(workers),
        "heldout_indices": sorted(int(value) for value in heldout_indices),
        "longitudinal_source_ids": list(LONGITUDINAL_SOURCE_IDS),
        "yaw_beta_scale_grid": [float(value) for value in YAW_BETA_SCALE_GRID],
        "post_stall_clr_scale_grid": [float(value) for value in POST_STALL_CLR_SCALE_GRID],
        "source_values": {key: float(value) for key, value in source_values.items()},
        "longitudinal_tolerances": dict(STRICT_LONGITUDINAL_TOLERANCES),
        "candidate_count": len(candidate_rows),
        "accepted_count": sum(bool(row.get("accepted", False)) for row in candidate_rows),
        "selected_count": len(selected_rows),
        "stage_replay_row_count": len(stage_replay_rows),
        "candidate_csv": "metrics/neutral_aero_residual_joint_pareto_local_candidates.csv",
        "selected_csv": "metrics/neutral_aero_residual_joint_pareto_local_selected.csv",
        "stage_replay_csv": "metrics/neutral_aero_residual_joint_pareto_local_stage_replay.csv",
        "command": sys.argv,
        "command_line": sysid.replay_fit.powershell_command_line(sys.argv),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "manifests" / "neutral_aero_residual_joint_pareto_local_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Narrow 40 ms Local Pareto SysID Sweep",
        "",
        f"- fit id: `{run_label}`",
        "- audit only: `true`",
        f"- source run: `{source_run_dir.name}`",
        "- rerun full calibration: `false`",
        f"- alignment window: `{alignment_window_s:.3f}` s",
        f"- replay dt: `{replay_dt_s:.4f}` s",
        f"- workers: `{workers}`",
        f"- candidate count: `{len(candidate_rows)}`",
        f"- accepted count: `{manifest['accepted_count']}`",
        f"- selected count: `{len(selected_rows)}`",
        "- varied lateral terms: `yaw_moment_beta_coeff`, `post_stall_roll_moment_r_hat_rbf_20_coeff`",
        "- regime replay note: rows are grouped by measured sample-level alpha regime; `throw_count` is the number of unique throws containing at least one sample in that regime and is not additive across regimes.",
        "",
        "## Selected Candidates",
        "",
        sysid.joint_pareto_audit_report_lines(
            candidate_rows=candidate_rows,
            selected_rows=selected_rows,
            enabled=True,
            alignment_window_s=alignment_window_s,
        ),
        "",
        "## Held-Out Sample-Regime Replay",
        "",
        sysid.joint_pareto_heavy_stage_replay_report_lines(stage_replay_rows),
    ]
    (output_dir / "reports" / "neutral_aero_residual_joint_pareto_local_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def parse_parameter_updates(value: Any) -> dict[str, float]:
    parsed = json.loads(str(value or "{}"))
    return {str(key): float(item) for key, item in dict(parsed).items()}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def scale_label_to_float(label: str) -> float:
    return float(str(label).replace("m", "-").replace("p", "."))


def scale_id(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
