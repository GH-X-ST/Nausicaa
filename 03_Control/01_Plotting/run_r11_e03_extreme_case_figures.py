"""Plot representative extreme-case R11 E03 trajectories in the E03 figure style."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from run_r11_balanced_ladder_case_figures import (
    DEFAULT_POLICY_IDS,
    DEFAULT_UPDRAFT_NX,
    DEFAULT_UPDRAFT_NY,
    DEFAULT_UPDRAFT_NZ,
    R11BalancedLadderFigureConfig,
    _plot_ladder_case,
    _read_neutral_rollout_rows,
    _read_selected_primitive_rows,
    _run_label_slug_from_root,
    _safe_file_component,
    _short_ladder_id,
)


FIGURE_RUN_VERSION = "r11_e03_extreme_case_paths_v1"
DEFAULT_R11_ROOT = Path("03_Control/05_Results/R11_validation/E03.1")
DEFAULT_OUTPUT_ROOT = Path("03_Control/A_figures/R11_E03_extreme_case_paths")
DEFAULT_NEUTRAL_ROLLOUT_PATH = Path(
    "03_Control/A_figures/R11_E03_balanced_neutral_baseline/metrics/neutral_rollout_by_case.csv"
)
DEFAULT_LIBRARY_SIZE_CASE = "balanced_cluster"
NO_MEMORY_POLICY_ID = "no_memory_baseline"
MEMORY_POLICY_IDS = (
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
)
CLOSED_LOOP_POLICY_IDS = (NO_MEMORY_POLICY_ID, *MEMORY_POLICY_IDS)


@dataclass(frozen=True)
class R11ExtremeCaseFigureConfig:
    r11_root: Path = DEFAULT_R11_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    neutral_rollout_path: Path = DEFAULT_NEUTRAL_ROLLOUT_PATH
    default_library_size_case_id: str = DEFAULT_LIBRARY_SIZE_CASE
    memory_scope: str = "all"
    policy_ids: tuple[str, ...] = DEFAULT_POLICY_IDS
    updraft_nx: int = DEFAULT_UPDRAFT_NX
    updraft_ny: int = DEFAULT_UPDRAFT_NY
    updraft_nz: int = DEFAULT_UPDRAFT_NZ


def run_r11_e03_extreme_case_figures(config: R11ExtremeCaseFigureConfig) -> dict[str, object]:
    r11_root = Path(config.r11_root)
    output_root = Path(config.output_root)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    outer_schedule = pd.read_csv(r11_root / "metrics" / "outer_case_schedule.csv")
    final_score = pd.read_csv(r11_root / "metrics" / "final_launch_score.csv")
    neutral = pd.read_csv(config.neutral_rollout_path)
    neutral = neutral[neutral.get("neutral_status", pd.Series(["complete"])).astype(str).eq("complete")].copy()

    selected_rows = _select_extreme_cases(
        final_score=final_score,
        neutral=neutral,
        default_library_size_case_id=config.default_library_size_case_id,
        memory_scope=config.memory_scope,
    )

    run_label_slug = _run_label_slug_from_root(r11_root)
    figure_rows: list[dict[str, object]] = []
    for selected in selected_rows:
        tier = str(selected["library_size_case_id"])
        case_index = int(selected["outer_case_index"])
        case_matches = outer_schedule[outer_schedule["outer_case_index"].astype(int) == case_index]
        if case_matches.empty:
            raise ValueError(f"missing_outer_case_schedule:{case_index}")
        case_row = case_matches.iloc[0].to_dict()
        plot_config = R11BalancedLadderFigureConfig(
            r11_root=r11_root,
            output_root=output_root,
            neutral_rollout_path=config.neutral_rollout_path,
            library_size_case_id=tier,
            policy_ids=config.policy_ids,
            updraft_nx=config.updraft_nx,
            updraft_ny=config.updraft_ny,
            updraft_nz=config.updraft_nz,
        )
        primitive_log = _read_selected_primitive_rows(
            r11_root=r11_root,
            outer_case_indices=(case_index,),
            config=plot_config,
        )
        neutral_rows = _read_neutral_rollout_rows(
            neutral_rollout_path=config.neutral_rollout_path,
            outer_case_indices=(case_index,),
        )
        summary = final_score[
            (final_score["library_size_case_id"].astype(str) == tier)
            & (final_score["outer_case_index"].astype(int) == case_index)
            & (final_score["policy_id"].astype(str).isin(config.policy_ids))
        ].copy()
        if summary.empty:
            raise ValueError(f"missing_final_score_summary:{tier}:{case_index}")

        category = _safe_file_component(selected["selection_category"])
        tier_slug = _safe_file_component(tier)
        ladder = _short_ladder_id(case_row["environment_block_id"])
        output_path = (
            output_root
            / "figures"
            / f"r11_{run_label_slug}_{category}_{tier_slug}_{ladder}_case{case_index:04d}.png"
        )
        plot_row = _plot_ladder_case(
            primitive_log=primitive_log,
            neutral_frame=neutral_rows,
            summary=summary,
            case_row=case_row,
            output_path=output_path,
            config=plot_config,
        )
        figure_rows.append(
            {
                **selected,
                "figure_path": output_path.as_posix(),
                "environment_block_label": case_row.get("environment_block_label", ""),
                "paired_start_condition_index": int(case_row.get("paired_start_condition_index", -1)),
                **plot_row,
            }
        )

    selection_frame = pd.DataFrame(figure_rows)
    selection_path = output_root / "metrics" / "r11_e03_extreme_case_figure_selection.csv"
    selection_frame.to_csv(selection_path, index=False)

    manifest = {
        "figure_run_version": FIGURE_RUN_VERSION,
        "r11_root": r11_root.as_posix(),
        "neutral_rollout_path": Path(config.neutral_rollout_path).as_posix(),
        "output_root": output_root.as_posix(),
        "default_library_size_case_id": str(config.default_library_size_case_id),
        "memory_scope": str(config.memory_scope),
        "policy_ids": list(config.policy_ids),
        "figure_count": int(len(figure_rows)),
        "selection_metrics_path": selection_path.as_posix(),
        "figures": [str(row["figure_path"]) for row in figure_rows],
        "claim_status": "diagnostic_extreme_case_visualisation_not_new_evidence_run",
    }
    manifest_path = output_root / "manifests" / "r11_e03_extreme_case_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    _write_report(output_root=output_root, figure_rows=figure_rows, manifest=manifest)
    return {**manifest, "status": "complete"}


def _select_extreme_cases(
    *,
    final_score: pd.DataFrame,
    neutral: pd.DataFrame,
    default_library_size_case_id: str,
    memory_scope: str,
) -> list[dict[str, object]]:
    final = final_score.copy()
    if "launch_role" in final.columns:
        final = final[final["launch_role"].astype(str).eq("final_heldout")].copy()
    closed = final[final["policy_id"].astype(str).isin(CLOSED_LOOP_POLICY_IDS)].copy()
    if closed.empty:
        raise ValueError("no_closed_loop_final_score_rows_found")

    default_closed = closed[
        closed["library_size_case_id"].astype(str).eq(str(default_library_size_case_id))
    ].copy()
    if default_closed.empty:
        raise ValueError(f"no_default_library_rows:{default_library_size_case_id}")

    open_closed = _best_closed_loop_by_case(default_closed).merge(
        _neutral_score_columns(neutral),
        on="outer_case_index",
        how="inner",
    )
    if open_closed.empty:
        raise ValueError("no_open_closed_join_rows")
    open_closed["open_closed_delta_launch_score"] = (
        open_closed["launch_score"] - open_closed["neutral_diagnostic_score"]
    )
    open_closed["abs_open_closed_delta_launch_score"] = open_closed[
        "open_closed_delta_launch_score"
    ].abs()
    largest_gap = open_closed.sort_values(
        ["abs_open_closed_delta_launch_score", "launch_score"], ascending=[False, False]
    ).iloc[0]
    fail_success = open_closed[
        (~open_closed["neutral_mission_success"].map(_truthy))
        & (open_closed["mission_success"].map(_truthy))
    ].copy()
    if fail_success.empty:
        raise ValueError("no_open_loop_fail_closed_loop_success_case_found")
    open_fail_closed_success = fail_success.sort_values(
        ["open_closed_delta_launch_score", "launch_score"], ascending=[False, False]
    ).iloc[0]

    memory_source = closed
    if str(memory_scope).strip().lower() == "default":
        memory_source = default_closed
    memory_cases = _memory_delta_by_case(memory_source)
    positive_memory = memory_cases[
        (memory_cases["memory_delta_launch_score"] > 0.0)
        & (memory_cases["selection_changed"].map(_truthy))
    ].copy()
    if not positive_memory.empty:
        memory_case = positive_memory.sort_values(
            ["memory_delta_launch_score", "history_length"], ascending=[False, True]
        ).iloc[0]
        memory_reason = "largest_positive_memory_delta_with_selection_change"
    else:
        memory_case = memory_cases.sort_values(
            ["abs_memory_delta_launch_score", "history_length"], ascending=[False, True]
        ).iloc[0]
        memory_reason = "largest_absolute_memory_delta"

    return [
        _open_closed_selection_row(
            row=largest_gap,
            category="open_closed_largest_score_gap",
            reason="largest_abs_true_neutral_open_loop_vs_best_closed_loop_score_gap",
        ),
        _memory_selection_row(row=memory_case, category="memory_largest_score_impact", reason=memory_reason),
        _open_closed_selection_row(
            row=open_fail_closed_success,
            category="open_fail_closed_success",
            reason="largest_true_neutral_open_loop_failure_to_closed_loop_success_score_gain",
        ),
    ]


def _best_closed_loop_by_case(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_mission_success_order"] = ranked["mission_success"].map(_truthy).astype(int)
    ranked["_safe_success_order"] = ranked["safe_success"].map(_truthy).astype(int)
    ranked = ranked.sort_values(
        [
            "library_size_case_id",
            "outer_case_index",
            "_mission_success_order",
            "_safe_success_order",
            "launch_score",
        ],
        ascending=[True, True, False, False, False],
    )
    return ranked.groupby(["library_size_case_id", "outer_case_index"], as_index=False).first()


def _neutral_score_columns(neutral: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "outer_case_index",
        "neutral_diagnostic_score",
        "mission_success",
        "front_wall_terminal_success",
        "termination_cause",
        "final_exit_x_w_m",
        "final_exit_y_w_m",
        "final_exit_z_w_m",
    ]
    available = [column for column in columns if column in neutral.columns]
    out = neutral[available].copy()
    rename = {
        "mission_success": "neutral_mission_success",
        "front_wall_terminal_success": "neutral_front_wall_terminal_success",
        "termination_cause": "neutral_termination_cause",
        "final_exit_x_w_m": "neutral_final_exit_x_w_m",
        "final_exit_y_w_m": "neutral_final_exit_y_w_m",
        "final_exit_z_w_m": "neutral_final_exit_z_w_m",
    }
    return out.rename(columns=rename)


def _memory_delta_by_case(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["policy_id"].astype(str).eq(NO_MEMORY_POLICY_ID)].copy()
    memory = frame[frame["policy_id"].astype(str).isin(MEMORY_POLICY_IDS)].copy()
    if baseline.empty or memory.empty:
        raise ValueError("missing_memory_or_no_memory_rows")
    baseline_cols = [
        "library_size_case_id",
        "outer_case_index",
        "environment_block_id",
        "launch_score",
        "mission_success",
        "safe_success",
        "mission_outcome_label",
        "termination_cause",
        "selection_signature",
    ]
    baseline = baseline[baseline_cols].rename(
        columns={
            "launch_score": "baseline_launch_score",
            "mission_success": "baseline_mission_success",
            "safe_success": "baseline_safe_success",
            "mission_outcome_label": "baseline_mission_outcome_label",
            "termination_cause": "baseline_termination_cause",
            "selection_signature": "baseline_selection_signature",
        }
    )
    merged = memory.merge(
        baseline,
        on=["library_size_case_id", "outer_case_index", "environment_block_id"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("no_memory_delta_join_rows")
    merged["memory_delta_launch_score"] = merged["launch_score"] - merged["baseline_launch_score"]
    merged["abs_memory_delta_launch_score"] = merged["memory_delta_launch_score"].abs()
    merged["selection_changed"] = merged["selection_signature"].astype(str).ne(
        merged["baseline_selection_signature"].astype(str)
    )
    return merged


def _open_closed_selection_row(*, row: pd.Series, category: str, reason: str) -> dict[str, object]:
    return {
        "selection_category": category,
        "selection_reason": reason,
        "library_size_case_id": str(row["library_size_case_id"]),
        "outer_case_index": int(row["outer_case_index"]),
        "environment_block_id": str(row["environment_block_id"]),
        "selected_policy_id": str(row["policy_id"]),
        "selected_history_length": int(float(row.get("history_length", 0))),
        "selected_launch_score": float(row["launch_score"]),
        "selected_mission_success": bool(_truthy(row.get("mission_success", False))),
        "neutral_diagnostic_score": float(row["neutral_diagnostic_score"]),
        "neutral_mission_success": bool(_truthy(row.get("neutral_mission_success", False))),
        "open_closed_delta_launch_score": float(row["open_closed_delta_launch_score"]),
        "abs_open_closed_delta_launch_score": float(row["abs_open_closed_delta_launch_score"]),
        "mission_outcome_label": str(row.get("mission_outcome_label", "")),
        "termination_cause": str(row.get("termination_cause", "")),
        "neutral_termination_cause": str(row.get("neutral_termination_cause", "")),
    }


def _memory_selection_row(*, row: pd.Series, category: str, reason: str) -> dict[str, object]:
    return {
        "selection_category": category,
        "selection_reason": reason,
        "library_size_case_id": str(row["library_size_case_id"]),
        "outer_case_index": int(row["outer_case_index"]),
        "environment_block_id": str(row["environment_block_id"]),
        "selected_policy_id": str(row["policy_id"]),
        "selected_history_length": int(float(row.get("history_length", 0))),
        "selected_launch_score": float(row["launch_score"]),
        "baseline_launch_score": float(row["baseline_launch_score"]),
        "memory_delta_launch_score": float(row["memory_delta_launch_score"]),
        "abs_memory_delta_launch_score": float(row["abs_memory_delta_launch_score"]),
        "selected_mission_success": bool(_truthy(row.get("mission_success", False))),
        "baseline_mission_success": bool(_truthy(row.get("baseline_mission_success", False))),
        "selected_safe_success": bool(_truthy(row.get("safe_success", False))),
        "baseline_safe_success": bool(_truthy(row.get("baseline_safe_success", False))),
        "selection_changed": bool(_truthy(row.get("selection_changed", False))),
        "memory_changed_selection": bool(_truthy(row.get("memory_changed_selection", False))),
        "mission_outcome_label": str(row.get("mission_outcome_label", "")),
        "baseline_mission_outcome_label": str(row.get("baseline_mission_outcome_label", "")),
        "termination_cause": str(row.get("termination_cause", "")),
        "baseline_termination_cause": str(row.get("baseline_termination_cause", "")),
    }


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _write_report(
    *,
    output_root: Path,
    figure_rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    lines = [
        "# R11 E03 Extreme-Case Trajectory Figures",
        "",
        f"- Figure run version: `{FIGURE_RUN_VERSION}`",
        f"- R11 root: `{manifest['r11_root']}`",
        f"- Neutral open-loop source: `{manifest['neutral_rollout_path']}`",
        f"- Default open/closed library tier: `{manifest['default_library_size_case_id']}`",
        f"- Memory selection scope: `{manifest['memory_scope']}`",
        f"- Figure count: `{manifest['figure_count']}`",
        "",
        "Each figure uses the same R11 E03 trajectory style as the balanced ladder figures: seeded fan/updraft context, true neutral open-loop path when available, and final held-out closed-loop policy paths.",
        "",
        "| Category | Reason | Library tier | Ladder | Outer case | Selected policy | Score delta | Figure |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in figure_rows:
        delta = row.get("open_closed_delta_launch_score", row.get("memory_delta_launch_score", 0.0))
        lines.append(
            "| {category} | {reason} | {tier} | {ladder} | {case} | {policy} | {delta:.3f} | `{figure}` |".format(
                category=row.get("selection_category", ""),
                reason=row.get("selection_reason", ""),
                tier=row.get("library_size_case_id", ""),
                ladder=row.get("environment_block_id", ""),
                case=int(row.get("outer_case_index", -1)),
                policy=row.get("selected_policy_id", ""),
                delta=float(delta),
                figure=row.get("figure_path", ""),
            )
        )
    (output_root / "reports" / "r11_e03_extreme_case_figures_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r11-root", type=Path, default=DEFAULT_R11_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--neutral-rollout", type=Path, default=DEFAULT_NEUTRAL_ROLLOUT_PATH)
    parser.add_argument("--default-library-size-case-id", default=DEFAULT_LIBRARY_SIZE_CASE)
    parser.add_argument("--memory-scope", choices=("all", "default"), default="all")
    parser.add_argument("--updraft-nx", type=int, default=DEFAULT_UPDRAFT_NX)
    parser.add_argument("--updraft-ny", type=int, default=DEFAULT_UPDRAFT_NY)
    parser.add_argument("--updraft-nz", type=int, default=DEFAULT_UPDRAFT_NZ)
    args = parser.parse_args(argv)
    result = run_r11_e03_extreme_case_figures(
        R11ExtremeCaseFigureConfig(
            r11_root=args.r11_root,
            output_root=args.output_root,
            neutral_rollout_path=args.neutral_rollout,
            default_library_size_case_id=args.default_library_size_case_id,
            memory_scope=args.memory_scope,
            updraft_nx=args.updraft_nx,
            updraft_ny=args.updraft_ny,
            updraft_nz=args.updraft_nz,
        )
    )
    print(result)
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
