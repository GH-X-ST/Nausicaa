from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("03_Control/05_Results/R11_validation/E01/metrics/speed_bin_policy_ladder_summary.csv")
DEFAULT_OUTPUT = Path("03_Control/A_figures/R11_E01_appendix_tables/r11_e01_speed_cluster_policy_appendix_tables.tex")
DEFAULT_COMBINED_OUTPUT = Path(
    "03_Control/A_figures/R11_E01_appendix_tables/r11_e01_speed_cluster_policy_appendix_tables.tex"
)

SPEED_ORDER = (
    "v0_lt_4_0_m_s",
    "v0_4_0_to_5_0_m_s",
    "v0_5_0_to_6_0_m_s",
    "v0_6_0_to_7_0_m_s",
    "v0_ge_7_0_m_s",
)
SPEED_LABELS = {
    "v0_lt_4_0_m_s": r"$<4.0$",
    "v0_4_0_to_5_0_m_s": r"$4.0$--$5.0$",
    "v0_5_0_to_6_0_m_s": r"$5.0$--$6.0$",
    "v0_6_0_to_7_0_m_s": r"$6.0$--$7.0$",
    "v0_ge_7_0_m_s": r"$\geq 7.0$",
}

LIBRARY_ORDER = (
    "heavy_cluster",
    "balanced_cluster",
    "light_cluster",
    "super_light_cluster",
    "no_cluster_no_merge",
)
LIBRARY_LABELS = {
    "heavy_cluster": "Heavy",
    "balanced_cluster": "Balanced",
    "light_cluster": "Light",
    "super_light_cluster": "Super-light",
    "no_cluster_no_merge": "No-cluster",
}

POLICY_ORDER = (
    "true_neutral_open_loop",
    "no_memory_baseline",
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
)
POLICY_LABELS = {
    "true_neutral_open_loop": "Open",
    "no_memory_baseline": "$h=0$",
    "spatial_flow_belief_memory_h3": "$h=3$",
    "spatial_flow_belief_memory_h10": "$h=10$",
    "spatial_flow_belief_memory_h30": "$h=30$",
}

LADDER_ORDER = (
    "r11_l0_dry_air_fixed",
    "r11_l1_single_fan_fixed_nominal",
    "r11_l2_four_fan_fixed_nominal",
    "r11_l3_fan_parameter_uncertainty",
    "r11_l4_local_fan_position_uncertainty",
    "r11_l5_active_fan_count_uncertainty",
    "r11_l6_environment_only_full_uncertainty",
    "r11_l7_full_domain_randomisation_arena_wide",
)
LADDER_LABELS = {
    "r11_l0_dry_air_fixed": "L0 dry air fixed",
    "r11_l1_single_fan_fixed_nominal": "L1 single-fan fixed nominal",
    "r11_l2_four_fan_fixed_nominal": "L2 four-fan fixed nominal",
    "r11_l3_fan_parameter_uncertainty": "L3 fan-parameter uncertainty",
    "r11_l4_local_fan_position_uncertainty": "L4 local fan-position uncertainty",
    "r11_l5_active_fan_count_uncertainty": "L5 active fan-count uncertainty",
    "r11_l6_environment_only_full_uncertainty": "L6 environment-only full uncertainty",
    "r11_l7_full_domain_randomisation_arena_wide": "L7 full-domain arena-wide randomisation",
}


@dataclass(frozen=True)
class MetricColumn:
    source: str
    header: str
    formatter: str
    higher_is_better: bool
    count_source: str | None = None


METRIC_COLUMNS = (
    MetricColumn("mission_success_rate", r"Target", "rate", True, "mission_success_count"),
    MetricColumn("safe_success_rate", r"Safe", "rate", True, "safe_success_count"),
    MetricColumn("front_wall_terminal_success_rate", r"Front", "rate", True, "front_wall_terminal_success_count"),
    MetricColumn("wrong_wall_exit_rate", r"Side", "rate", False, "wrong_wall_exit_count"),
    MetricColumn("expected_low_energy_dry_air_sink_rate", r"Sink", "rate", False, "expected_low_energy_dry_air_sink_count"),
    MetricColumn("no_viable_primitive_rate", r"NoPr", "rate", False, "no_viable_primitive_count"),
    MetricColumn("hard_failure_rate", r"Hard", "rate", False, "hard_failure_count"),
    MetricColumn("mean_launch_score", r"$\bar{J}$", "score", True),
    MetricColumn("median_launch_score", r"$\tilde{J}$", "score", True),
    MetricColumn("mean_terminal_specific_energy_m", r"$E_T$", "energy", True),
    MetricColumn("mean_terminal_specific_energy_reserve_m", r"$E_R$", "energy", True),
    MetricColumn("mean_positive_specific_energy_gain_m", r"$E_+$", "energy", True),
    MetricColumn("mean_updraft_specific_energy_gain_proxy_m", r"$E_u$", "energy", True),
    MetricColumn("mean_lift_dwell_time_s", r"$t_L$", "time", True),
)


COLSPEC_WEIGHTS = (
    1.30,
    1.45,
    0.50,
    0.45,
    0.85,
    0.80,
    0.80,
    0.75,
    0.80,
    0.75,
    0.75,
    1.05,
    1.05,
    1.00,
    1.00,
    1.00,
    1.00,
    0.90,
)


def main() -> None:
    args = _parse_args()
    frame, source_label = _load_input_frame(args)
    tex = build_appendix_tables(frame, source_label=source_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="ascii")
    _write_optional_summary_outputs(frame, args)
    print(args.output)


def _write_optional_summary_outputs(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    summary_output = args.summary_output
    comparison_output = args.open_loop_comparison_output
    if args.neutral_rollout:
        output_stem = args.output.with_suffix("")
        if summary_output is None:
            summary_output = output_stem.with_name(output_stem.name + "_summary.csv")
        if comparison_output is None:
            comparison_output = output_stem.with_name(output_stem.name + "_open_loop_comparison.csv")
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(summary_output, index=False)
    if comparison_output is not None:
        comparison = _open_loop_comparison_from_summary(frame)
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_output, index=False)


def _open_loop_comparison_from_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "true_neutral_open_loop" not in set(frame["policy_id"].astype(str)):
        return pd.DataFrame()
    open_loop = frame[frame["policy_id"].astype(str).eq("true_neutral_open_loop")].copy()
    controlled = frame[~frame["policy_id"].astype(str).eq("true_neutral_open_loop")].copy()
    join_columns = ["environment_block_id", "library_size_case_id", "launch_speed_bin_id"]
    baseline_columns = join_columns + [
        "launch_count",
        "mission_success_rate",
        "safe_success_rate",
        "front_wall_terminal_success_rate",
        "wrong_wall_exit_rate",
        "expected_low_energy_dry_air_sink_rate",
        "hard_failure_rate",
        "mean_launch_score",
        "median_launch_score",
        "mean_terminal_specific_energy_m",
        "mean_terminal_specific_energy_reserve_m",
        "mean_positive_specific_energy_gain_m",
    ]
    available_baseline = [column for column in baseline_columns if column in open_loop.columns]
    baseline = open_loop[available_baseline].rename(
        columns={
            "launch_count": "open_loop_launch_count",
            "mission_success_rate": "open_loop_mission_success_rate",
            "safe_success_rate": "open_loop_safe_success_rate",
            "front_wall_terminal_success_rate": "open_loop_front_wall_terminal_success_rate",
            "wrong_wall_exit_rate": "open_loop_wrong_wall_exit_rate",
            "expected_low_energy_dry_air_sink_rate": "open_loop_expected_low_energy_dry_air_sink_rate",
            "hard_failure_rate": "open_loop_hard_failure_rate",
            "mean_launch_score": "open_loop_mean_launch_score",
            "median_launch_score": "open_loop_median_launch_score",
            "mean_terminal_specific_energy_m": "open_loop_mean_terminal_specific_energy_m",
            "mean_terminal_specific_energy_reserve_m": "open_loop_mean_terminal_specific_energy_reserve_m",
            "mean_positive_specific_energy_gain_m": "open_loop_mean_positive_specific_energy_gain_m",
        }
    )
    merged = controlled.merge(baseline, on=join_columns, how="left")
    if merged.empty:
        return merged
    for metric in (
        "mission_success_rate",
        "safe_success_rate",
        "front_wall_terminal_success_rate",
        "wrong_wall_exit_rate",
        "expected_low_energy_dry_air_sink_rate",
        "hard_failure_rate",
        "mean_launch_score",
        "median_launch_score",
        "mean_terminal_specific_energy_m",
        "mean_terminal_specific_energy_reserve_m",
        "mean_positive_specific_energy_gain_m",
    ):
        baseline_name = f"open_loop_{metric}"
        if metric in merged.columns and baseline_name in merged.columns:
            merged[f"delta_control_minus_open_loop_{metric}"] = (
                pd.to_numeric(merged[metric], errors="coerce")
                - pd.to_numeric(merged[baseline_name], errors="coerce")
            )
    output_columns = [
        "environment_block_id",
        "launch_speed_bin_id",
        "library_size_case_id",
        "policy_id",
        "history_length",
        "launch_count",
        "open_loop_launch_count",
        "mission_success_rate",
        "open_loop_mission_success_rate",
        "delta_control_minus_open_loop_mission_success_rate",
        "safe_success_rate",
        "open_loop_safe_success_rate",
        "delta_control_minus_open_loop_safe_success_rate",
        "wrong_wall_exit_rate",
        "open_loop_wrong_wall_exit_rate",
        "delta_control_minus_open_loop_wrong_wall_exit_rate",
        "hard_failure_rate",
        "open_loop_hard_failure_rate",
        "delta_control_minus_open_loop_hard_failure_rate",
        "mean_launch_score",
        "open_loop_mean_launch_score",
        "delta_control_minus_open_loop_mean_launch_score",
        "median_launch_score",
        "open_loop_median_launch_score",
        "delta_control_minus_open_loop_median_launch_score",
        "mean_terminal_specific_energy_m",
        "open_loop_mean_terminal_specific_energy_m",
        "delta_control_minus_open_loop_mean_terminal_specific_energy_m",
    ]
    return merged[[column for column in output_columns if column in merged.columns]]


def build_appendix_tables(frame: pd.DataFrame, *, source_label: str) -> str:
    ordered = frame.copy()
    ordered["environment_block_id"] = pd.Categorical(ordered["environment_block_id"], LADDER_ORDER, ordered=True)
    ordered["launch_speed_bin_id"] = pd.Categorical(ordered["launch_speed_bin_id"], SPEED_ORDER, ordered=True)
    ordered["library_size_case_id"] = pd.Categorical(ordered["library_size_case_id"], LIBRARY_ORDER, ordered=True)
    ordered["policy_id"] = pd.Categorical(ordered["policy_id"], POLICY_ORDER, ordered=True)
    ordered = ordered.sort_values(["environment_block_id", "launch_speed_bin_id", "library_size_case_id", "policy_id"])

    lines = [
        r"% Auto-generated by 03_Control/01_Plotting/run_r11_appendix_speed_tables.py.",
        rf"% Source: {latex_escape(source_label)}",
        r"% Each table uses width=16.2cm. Column X weights sum to 16.2.",
        r"\subsection{Held-Out Repeated-Launch Validation Tables}",
        "",
        r"\noindent\footnotesize",
        r"These appendix tables report the held-out repeated-launch validation used to test whether a fixed primitive library and an outer-loop governor can reach the front-wall target region while exploiting uncertain indoor updrafts.",
        r"Each row is one experimental condition: initial launch-speed bin $v_0$, primitive-library compression tier, and repeated-launch history length $h$.",
        r"When multiple validation repeats are supplied, the tables aggregate independent randomized held-out repeats that use the same frozen R10 governor; the raw runs are not modified.",
        r"The same paired launch starts are reused across environment ladders, library tiers, and memory policies within each repeat, so rows can be compared directly within the same speed bin.",
        r"Speed-bin rows are conditional diagnostics under randomized launch sampling, not equal-count stratified tests.",
        r"The open-loop row applies the same launch state and environment but sends neutral control throughout the flight. It is a shared baseline repeated under each library tier only to make within-table comparison direct; it does not use a primitive library.",
        r"The no-memory baseline is $h=0$; $h=3$, $h=10$, and $h=30$ use the previous 3, 10, or 30 launches to update the online spatial flow-belief map.",
        r"$n$ is the number of held-out launches in the row. Count-backed rate cells are reported as count/$n$ (percentage). Target is successful arrival at the front-wall target region; Safe is retained safe termination without hard boundary failure; Front is any front-wall terminal exit; Side is wrong-wall exit; Sink is a physically low-energy floor stop; NoPr is no viable continuation primitive; Hard is a claim-bearing hard failure.",
        r"$\bar{J}$ and $\tilde{J}$ are the mean and median mission scores. $E_T$ is terminal specific energy, $E_R$ is terminal energy reserve, $E_+$ is positive specific-energy gain, $E_u$ is updraft energy-gain proxy, and $t_L$ is lift dwell time.",
        r"Blue marks the best value and red marks the worst value within each environment-ladder table for each metric column; lower is better for Side, Sink, NoPr, and Hard, while higher is better for the remaining metrics. NoPr is not applicable to the open-loop row and is reported as zero because no primitive selection is attempted.",
        "",
    ]
    for ladder_id in LADDER_ORDER:
        ladder_frame = ordered[ordered["environment_block_id"].astype(str) == ladder_id]
        if ladder_frame.empty:
            continue
        lines.extend(_ladder_table(ladder_id, ladder_frame))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_input_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    run_roots = [Path(path) for path in (args.run_root or [])]
    final_scores = [Path(path) for path in (args.final_score or [])]
    if run_roots or final_scores:
        score_paths = final_scores + [root / "metrics" / "final_launch_score.csv" for root in run_roots]
        frame = _summary_from_final_launch_scores(score_paths, neutral_rollout_paths=args.neutral_rollout)
        output = args.output
        if output == DEFAULT_OUTPUT and len(score_paths) > 1:
            args.output = DEFAULT_COMBINED_OUTPUT
        source_paths = score_paths + list(args.neutral_rollout or [])
        source_label = "; ".join(str(path).replace("\\", "/") for path in source_paths)
        return frame, source_label
    if args.neutral_rollout:
        raise ValueError("--neutral-rollout requires --run-root or --final-score so open-loop rows can be paired to launch metadata")
    return pd.read_csv(args.input), str(args.input).replace("\\", "/")


def _summary_from_final_launch_scores(
    score_paths: list[Path],
    *,
    neutral_rollout_paths: list[Path] | None = None,
) -> pd.DataFrame:
    if not score_paths:
        raise ValueError("no_final_launch_score_inputs")
    frames: list[pd.DataFrame] = []
    for path in score_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        run_label = path.parents[1].name if path.parent.name == "metrics" else path.stem
        frame = frame.copy()
        frame["validation_run_label"] = run_label
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    if neutral_rollout_paths:
        raw = _append_open_loop_neutral_rows(raw, neutral_rollout_paths)
    group_columns = [
        "environment_block_id",
        "library_size_case_id",
        "policy_id",
        "history_length",
        "launch_speed_bin_id",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in raw.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys))
        launch_count = int(len(group))
        row["launch_count"] = launch_count
        for source in (
            "launch_speed_bin_min_m_s",
            "launch_speed_bin_max_m_s",
            "launch_speed_bin_label",
            "start_energy_feasibility_threshold_m_s",
            "start_energy_group_basis",
        ):
            if source in group.columns:
                row[source] = _first_non_null(group[source])
        _add_boolean_count_rate(row, group, "mission_success", "mission_success")
        _add_boolean_count_rate(row, group, "safe_success", "safe_success")
        _add_boolean_count_rate(row, group, "full_safe_success", "full_safe_success")
        _add_boolean_count_rate(row, group, "front_wall_terminal_success", "front_wall_terminal_success")
        _add_boolean_count_rate(row, group, "wrong_wall_exit", "wrong_wall_exit")
        _add_boolean_count_rate(row, group, "expected_low_energy_dry_air_sink", "expected_low_energy_dry_air_sink")
        _add_boolean_count_rate(row, group, "claim_bearing_episode", "claim_bearing_episode")
        _add_boolean_count_rate(row, group, "no_viable_primitive", "no_viable_primitive")
        _add_boolean_count_rate(row, group, "hard_failure", "hard_failure")
        _add_boolean_count_rate(row, group, "floor_or_ceiling_violation", "floor_or_ceiling_violation")
        _add_boolean_count_rate(row, group, "physical_hard_failure", "physical_hard_failure")
        _add_boolean_count_rate(row, group, "physical_floor_or_ceiling_violation", "physical_floor_or_ceiling_violation")
        _add_boolean_count_rate(row, group, "terminal_useful", "terminal_useful")
        _add_boolean_count_rate(row, group, "lift_capture", "lift_capture")
        _add_boolean_count_rate(row, group, "memory_changed_selection", "memory_changed_selection")
        _add_boolean_count_rate(row, group, "exploration_changed_selection", "exploration_changed_selection")
        _add_mean(row, group, "initial_launch_speed_m_s", "mean_initial_launch_speed_m_s")
        _add_min_max(row, group, "initial_launch_speed_m_s", "min_initial_launch_speed_m_s", "max_initial_launch_speed_m_s")
        _add_mean(row, group, "launch_score", "mean_launch_score")
        _add_median(row, group, "launch_score", "median_launch_score")
        _add_mean(row, group, "episode_flight_time_s", "mean_episode_flight_time_s")
        _add_mean(row, group, "lift_dwell_time_s", "mean_lift_dwell_time_s")
        _add_mean(row, group, "terminal_specific_energy_m", "mean_terminal_specific_energy_m")
        _add_mean(row, group, "terminal_specific_energy_reserve_m", "mean_terminal_specific_energy_reserve_m")
        _add_mean(row, group, "terminal_specific_energy_bonus", "mean_terminal_specific_energy_bonus")
        _add_mean(row, group, "positive_specific_energy_gain_m", "mean_positive_specific_energy_gain_m")
        _add_mean(row, group, "updraft_specific_energy_gain_proxy_m", "mean_updraft_specific_energy_gain_proxy_m")
        _add_mean(row, group, "net_specific_energy_delta_m", "mean_net_specific_energy_delta_m")
        _add_mean(row, group, "gross_specific_energy_loss_m", "mean_gross_specific_energy_loss_m")
        rows.append(row)
    return pd.DataFrame(rows)


def _append_open_loop_neutral_rows(raw: pd.DataFrame, neutral_rollout_paths: list[Path]) -> pd.DataFrame:
    neutral_frames: list[pd.DataFrame] = []
    raw_run_labels = sorted(str(value) for value in raw["validation_run_label"].dropna().unique())
    for path in neutral_rollout_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        run_label = _infer_neutral_run_label(path)
        if run_label is None:
            if len(raw_run_labels) != 1:
                raise ValueError(
                    f"cannot infer validation run label for {path}; provide one neutral file per labelled run path"
                )
            run_label = raw_run_labels[0]
        frame = frame.copy()
        frame["validation_run_label"] = run_label
        neutral_frames.append(frame)
    if not neutral_frames:
        return raw

    neutral = pd.concat(neutral_frames, ignore_index=True)
    neutral_status = neutral["neutral_status"] if "neutral_status" in neutral.columns else pd.Series("complete", index=neutral.index)
    neutral = neutral[neutral_status.astype(str).eq("complete")].copy()
    if neutral.empty:
        return raw

    join_columns = ["validation_run_label", "environment_block_id", "outer_case_index"]
    for column in join_columns:
        if column not in neutral.columns:
            raise ValueError(f"neutral rollout input is missing {column}")
        if column not in raw.columns:
            raise ValueError(f"final launch score input is missing {column}")

    metadata_columns = [
        "validation_run_label",
        "environment_block_id",
        "outer_case_index",
        "outer_case_id",
        "outer_case_type",
        "common_final_launch_key",
        "initial_launch_speed_m_s",
        "launch_speed_bin_id",
        "launch_speed_bin_min_m_s",
        "launch_speed_bin_max_m_s",
        "launch_speed_bin_label",
        "start_energy_group_id",
        "start_energy_group_label",
        "start_energy_feasibility_threshold_m_s",
        "start_energy_group_basis",
        "terminal_specific_energy_reference_m",
    ]
    available_metadata = [column for column in metadata_columns if column in raw.columns]
    representative = (
        raw.sort_values(["validation_run_label", "environment_block_id", "outer_case_index"])
        .groupby(join_columns, dropna=False, sort=False)[available_metadata]
        .first()
        .reset_index(drop=True)
    )

    neutral_with_meta = neutral.merge(
        representative,
        on=join_columns,
        how="left",
        suffixes=("", "_controller"),
    )
    missing = neutral_with_meta["launch_speed_bin_id"].isna() if "launch_speed_bin_id" in neutral_with_meta.columns else pd.Series(True)
    if bool(missing.any()):
        missing_count = int(missing.sum())
        raise ValueError(f"{missing_count} neutral rows could not be paired to final launch metadata")

    libraries = [str(value) for value in raw["library_size_case_id"].dropna().unique()]
    rows: list[pd.DataFrame] = []
    for library_id in libraries:
        synthetic = pd.DataFrame(index=neutral_with_meta.index)
        synthetic["validation_run_label"] = neutral_with_meta["validation_run_label"]
        synthetic["library_size_case_id"] = library_id
        synthetic["policy_id"] = "true_neutral_open_loop"
        synthetic["history_length"] = -1
        synthetic["adaptation_launch_index"] = -1
        synthetic["outer_case_index"] = neutral_with_meta["outer_case_index"]
        for column in (
            "outer_case_id",
            "outer_case_type",
            "environment_block_id",
            "common_final_launch_key",
            "launch_speed_bin_id",
            "launch_speed_bin_min_m_s",
            "launch_speed_bin_max_m_s",
            "launch_speed_bin_label",
            "start_energy_group_id",
            "start_energy_group_label",
            "start_energy_feasibility_threshold_m_s",
            "start_energy_group_basis",
        ):
            if column in neutral_with_meta.columns:
                synthetic[column] = neutral_with_meta[column]

        synthetic["episode_id"] = [
            f"open_loop_neutral_{run}_{block}_{int(idx):04d}_{library_id}"
            for run, block, idx in zip(
                neutral_with_meta["validation_run_label"],
                neutral_with_meta["environment_block_id"],
                neutral_with_meta["outer_case_index"],
            )
        ]
        synthetic["launch_role"] = "final_heldout_open_loop_neutral"
        synthetic["policy_family"] = "open_loop_neutral"
        synthetic["safe_explore_active"] = False
        synthetic["selected_primitive_variant_id"] = "none_open_loop_neutral"
        synthetic["selected_primitive_id"] = "none_open_loop_neutral"
        synthetic["selected_controller_id"] = "neutral_command"
        synthetic["selected_primitive_step_count"] = 0
        synthetic["launch_sequence_policy"] = "open_loop_neutral_no_primitive_selection"
        synthetic["termination_cause"] = neutral_with_meta["termination_cause"]
        synthetic["trajectory_status"] = neutral_with_meta.get("trajectory_status", "finite_neutral_rollout")
        synthetic["initial_launch_speed_m_s"] = pd.to_numeric(neutral_with_meta["initial_launch_speed_m_s"], errors="coerce")
        synthetic["hard_failure"] = _bool_series(neutral_with_meta["hard_failure"])
        synthetic["floor_or_ceiling_violation"] = _bool_series(neutral_with_meta["floor_or_ceiling_violation"])
        synthetic["physical_hard_failure"] = synthetic["hard_failure"]
        synthetic["physical_floor_or_ceiling_violation"] = synthetic["floor_or_ceiling_violation"]
        synthetic["expected_low_energy_dry_air_sink"] = neutral_with_meta["termination_cause"].astype(str).eq("floor_margin_stop")
        synthetic["claim_bearing_episode"] = True
        synthetic["no_viable_primitive"] = False
        synthetic["safe_success"] = _bool_series(neutral_with_meta.get("safe_geometric", pd.Series(True, index=neutral_with_meta.index)))
        synthetic["full_safe_success"] = synthetic["safe_success"]
        synthetic["terminal_useful"] = _bool_series(neutral_with_meta["mission_success"])
        synthetic["terminal_useful_safe_exit_only"] = synthetic["terminal_useful"]
        synthetic["lift_capture"] = False
        synthetic["episode_rollout_duration_s"] = pd.to_numeric(neutral_with_meta["episode_flight_time_s"], errors="coerce")
        synthetic["episode_flight_time_s"] = synthetic["episode_rollout_duration_s"]
        synthetic["lift_dwell_time_s"] = math.nan
        synthetic["energy_residual_m"] = pd.to_numeric(neutral_with_meta["net_specific_energy_delta_m"], errors="coerce")
        synthetic["episode_specific_energy_start_m"] = pd.to_numeric(neutral_with_meta["start_specific_energy_m"], errors="coerce")
        synthetic["episode_specific_energy_end_m"] = pd.to_numeric(neutral_with_meta["terminal_specific_energy_m"], errors="coerce")
        synthetic["net_specific_energy_delta_m"] = pd.to_numeric(neutral_with_meta["net_specific_energy_delta_m"], errors="coerce")
        synthetic["gross_specific_energy_gain_m"] = synthetic["net_specific_energy_delta_m"].clip(lower=0.0)
        synthetic["gross_specific_energy_loss_m"] = (-synthetic["net_specific_energy_delta_m"]).clip(lower=0.0)
        synthetic["positive_specific_energy_gain_m"] = synthetic["net_specific_energy_delta_m"].clip(lower=0.0)
        synthetic["updraft_specific_energy_gain_proxy_m"] = math.nan
        synthetic["updraft_gain_proxy_source"] = "not_available_for_open_loop_neutral_postprocess"
        synthetic["min_wall_margin_m"] = pd.to_numeric(neutral_with_meta["min_wall_margin_m"], errors="coerce")
        synthetic["governor_rejection_count"] = 0
        synthetic["belief_observation_count"] = 0
        synthetic["belief_uncertainty"] = math.nan
        synthetic["memory_changed_selection"] = False
        synthetic["exploration_changed_selection"] = False
        synthetic["environment_instance_id"] = neutral_with_meta.get("environment_instance_id", "")
        synthetic["claim_status"] = "open_loop_neutral_postprocess_baseline"
        synthetic["belief_update_count_before"] = 0
        synthetic["belief_update_count_after"] = 0
        synthetic["mission_success"] = _bool_series(neutral_with_meta["mission_success"])
        synthetic["front_wall_terminal_success"] = _bool_series(neutral_with_meta["front_wall_terminal_success"])
        synthetic["wrong_wall_exit"] = _bool_series(neutral_with_meta["wrong_wall_exit"])
        synthetic["terminal_wall_face"] = neutral_with_meta.get("terminal_wall_face", "")
        synthetic["final_exit_x_w_m"] = pd.to_numeric(neutral_with_meta["final_exit_x_w_m"], errors="coerce")
        synthetic["final_exit_y_w_m"] = pd.to_numeric(neutral_with_meta["final_exit_y_w_m"], errors="coerce")
        synthetic["final_exit_z_w_m"] = pd.to_numeric(neutral_with_meta["final_exit_z_w_m"], errors="coerce")
        synthetic["terminal_specific_energy_m"] = pd.to_numeric(neutral_with_meta["terminal_specific_energy_m"], errors="coerce")
        reference = pd.to_numeric(
            neutral_with_meta.get("terminal_specific_energy_reference_m", pd.Series(0.4, index=neutral_with_meta.index)),
            errors="coerce",
        ).fillna(0.4)
        synthetic["terminal_specific_energy_reference_m"] = reference
        synthetic["terminal_specific_energy_reserve_m"] = synthetic["terminal_specific_energy_m"] - reference
        synthetic["terminal_specific_energy_source"] = "neutral_rollout_exit_state_specific_energy"
        synthetic["mission_outcome_label"] = neutral_with_meta.get("trajectory_status", "finite_neutral_rollout")
        synthetic["episode_interpretation_label"] = neutral_with_meta.get("trajectory_status", "finite_neutral_rollout")
        synthetic["launch_score"] = pd.to_numeric(neutral_with_meta["neutral_diagnostic_score"], errors="coerce")
        synthetic["launch_score_scope"] = "open_loop_neutral_postprocess_score"
        synthetic["terminal_specific_energy_bonus"] = math.nan
        rows.append(synthetic)

    synthetic_raw = pd.concat(rows, ignore_index=True)
    aligned_columns = list(dict.fromkeys(list(raw.columns) + list(synthetic_raw.columns)))
    return pd.concat(
        [raw.reindex(columns=aligned_columns), synthetic_raw.reindex(columns=aligned_columns)],
        ignore_index=True,
    )


def _infer_neutral_run_label(path: Path) -> str | None:
    for part in path.parts:
        match = re.search(r"R11_([A-Z]\d{2})", part)
        if match:
            return match.group(1)
    return None


def _add_boolean_count_rate(row: dict[str, object], group: pd.DataFrame, source: str, prefix: str) -> None:
    if source not in group.columns:
        row[f"{prefix}_count"] = 0
        row[f"{prefix}_rate"] = math.nan
        return
    values = _bool_series(group[source])
    count = int(values.sum())
    row[f"{prefix}_count"] = count
    row[f"{prefix}_rate"] = count / max(1, len(group))


def _add_mean(row: dict[str, object], group: pd.DataFrame, source: str, output: str) -> None:
    row[output] = float(pd.to_numeric(group.get(source, pd.Series(dtype=float)), errors="coerce").mean())


def _add_median(row: dict[str, object], group: pd.DataFrame, source: str, output: str) -> None:
    row[output] = float(pd.to_numeric(group.get(source, pd.Series(dtype=float)), errors="coerce").median())


def _add_min_max(row: dict[str, object], group: pd.DataFrame, source: str, min_output: str, max_output: str) -> None:
    values = pd.to_numeric(group.get(source, pd.Series(dtype=float)), errors="coerce")
    row[min_output] = float(values.min())
    row[max_output] = float(values.max())


def _first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if not values.empty else math.nan


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _ladder_table(ladder_id: str, frame: pd.DataFrame) -> list[str]:
    table_id = ladder_id.replace("r11_", "").replace("_", "-")
    policy_order = tuple(policy_id for policy_id in POLICY_ORDER if policy_id in set(frame["policy_id"].astype(str)))
    caption = (
        f"Held-out repeated-launch validation under {LADDER_LABELS[ladder_id]}. "
        "Rows condition performance on launch speed, primitive-library compression, and the number of previous launches used by the memory-assisted governor."
    )
    label = f"tab:r11-heldout-{table_id}-speed-cluster-policy"
    metric_extrema = _metric_extrema(frame)
    lines = [
        r"\begin{longtblr}[",
        rf"  caption = {{{caption}}},",
        rf"  label = {{{label}}},",
        r"]{",
        r"  rows={font=\tiny},",
        r"  width=16.2cm,",
        r"  rowhead=1,",
        r"  rowsep=1pt,",
        r"  colsep=1pt,",
        r"  colspec = {",
        _colspec(),
        r"  },",
        r"}",
        r"\toprule",
        _header_row(),
        r"\midrule",
    ]
    row_counter = 0
    for speed_id in SPEED_ORDER:
        speed_frame = frame[frame["launch_speed_bin_id"].astype(str) == speed_id]
        if speed_frame.empty:
            continue
        if row_counter:
            lines.append(r"\midrule")
        speed_started = False
        for library_id in LIBRARY_ORDER:
            library_frame = speed_frame[speed_frame["library_size_case_id"].astype(str) == library_id]
            if library_frame.empty:
                continue
            library_started = False
            for policy_id in policy_order:
                row = library_frame[library_frame["policy_id"].astype(str) == policy_id]
                if row.empty:
                    continue
                lines.append(
                    _table_row(
                        row.iloc[0],
                        speed_id,
                        library_id,
                        policy_id,
                        metric_extrema,
                        speed_started,
                        library_started,
                        len(policy_order),
                    )
                )
                speed_started = True
                library_started = True
                row_counter += 1
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtblr}",
        ]
    )
    return lines


def _colspec() -> str:
    alignments = ["l", "l", "c", "c"] + ["c"] * len(METRIC_COLUMNS)
    specs = [f"    X[{weight:.2f},{align},m]" for weight, align in zip(COLSPEC_WEIGHTS, alignments)]
    return "\n".join(specs)


def _header_row() -> str:
    headers = [
        r"\SetCell{m} \tiny\textbf{$v_0$}",
        r"\SetCell{m} \tiny\textbf{Cluster}",
        r"\SetCell{m} \tiny\textbf{$h$}",
        r"\SetCell{m} \tiny\textbf{$n$}",
    ]
    headers.extend(rf"\SetCell{{m}} \tiny\textbf{{{column.header}}}" for column in METRIC_COLUMNS)
    return "\n  " + "\n  & ".join(headers) + r" \\"


def _table_row(
    row: pd.Series,
    speed_id: str,
    library_id: str,
    policy_id: str,
    metric_extrema: dict[str, tuple[float, float]],
    speed_started: bool,
    library_started: bool,
    policy_count: int,
) -> str:
    cells = []
    if not speed_started:
        cells.append(rf"\SetCell[r={len(LIBRARY_ORDER) * policy_count}]{{l,m}} {SPEED_LABELS[speed_id]}")
    else:
        cells.append("")
    if not library_started:
        cells.append(rf"\SetCell[r={policy_count}]{{l,m}} {LIBRARY_LABELS[library_id]}")
    else:
        cells.append("")
    cells.append(POLICY_LABELS[policy_id])
    cells.append(_format_count(row.get("launch_count", "")))
    for metric in METRIC_COLUMNS:
        cells.append(_format_metric_value(row, metric, metric_extrema))
    return "\n  " + "\n  & ".join(cells) + r" \\"


def _metric_extrema(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    extrema: dict[str, tuple[float, float]] = {}
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(frame.get(metric.source, pd.Series(dtype=float)), errors="coerce").dropna()
        if values.empty:
            continue
        extrema[metric.source] = (float(values.min()), float(values.max()))
    return extrema


def _format_metric_value(row: pd.Series, metric: MetricColumn, metric_extrema: dict[str, tuple[float, float]]) -> str:
    value = row.get(metric.source, math.nan)
    number = _finite_float(value)
    if number is None:
        return "--"
    if metric.formatter == "rate":
        text = _format_count_rate(row, metric, number)
    elif metric.formatter == "score":
        text = f"{number:.1f}"
    elif metric.formatter in {"energy", "time"}:
        text = f"{number:.2f}"
    else:
        text = f"{number:.2f}"
    extrema = metric_extrema.get(metric.source)
    if extrema is None or math.isclose(extrema[0], extrema[1], rel_tol=0.0, abs_tol=1e-12):
        return text
    best = extrema[1] if metric.higher_is_better else extrema[0]
    worst = extrema[0] if metric.higher_is_better else extrema[1]
    if math.isclose(number, best, rel_tol=0.0, abs_tol=1e-12):
        return rf"\textcolor{{ICLBlue}}{{{text}}}"
    if math.isclose(number, worst, rel_tol=0.0, abs_tol=1e-12):
        return rf"\textcolor{{ICLRed}}{{{text}}}"
    return text


def _format_count_rate(row: pd.Series, metric: MetricColumn, rate: float) -> str:
    if metric.count_source is None:
        return f"{100.0 * rate:.1f}"
    count = _finite_float(row.get(metric.count_source, math.nan))
    total = _finite_float(row.get("launch_count", math.nan))
    if count is None or total is None:
        return f"{100.0 * rate:.1f}"
    return f"{int(round(count))}/{int(round(total))} ({100.0 * rate:.1f})"


def _format_count(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "--"
    return str(int(round(number)))


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate appendix R11 speed/cluster/history tables in thesis tblr style.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="R11 validation run root containing metrics/final_launch_score.csv. May be supplied multiple times.",
    )
    parser.add_argument(
        "--final-score",
        type=Path,
        action="append",
        help="Direct final_launch_score.csv input. May be supplied multiple times.",
    )
    parser.add_argument(
        "--neutral-rollout",
        type=Path,
        action="append",
        help=(
            "neutral_rollout_by_case.csv from the true open-loop diagnostic. "
            "When supplied with final scores, open-loop rows are paired to the same outer cases and repeated under each library tier."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional CSV output for the grouped table data. Defaults beside --output when --neutral-rollout is supplied.",
    )
    parser.add_argument(
        "--open-loop-comparison-output",
        type=Path,
        help="Optional CSV output for controller-minus-open-loop grouped deltas.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
