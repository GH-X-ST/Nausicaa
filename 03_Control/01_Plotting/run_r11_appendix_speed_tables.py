from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("03_Control/05_Results/R11_validation/D01/metrics/speed_bin_policy_ladder_summary.csv")
DEFAULT_OUTPUT = Path("03_Control/A_figures/R11_D01_appendix_tables/r11_d01_speed_cluster_policy_appendix_tables.tex")
DEFAULT_COMBINED_OUTPUT = Path(
    "03_Control/A_figures/R11_D01_D02_appendix_tables/r11_d01_d02_speed_cluster_policy_appendix_tables.tex"
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
    "no_memory_baseline",
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
)
POLICY_LABELS = {
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
    print(args.output)


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
        r"The no-memory baseline is $h=0$; $h=3$, $h=10$, and $h=30$ use the previous 3, 10, or 30 launches to update the online spatial flow-belief map.",
        r"$n$ is the number of held-out launches in the row. Count-backed rate cells are reported as count/$n$ (percentage). Target is successful arrival at the front-wall target region; Safe is retained safe termination without hard boundary failure; Front is any front-wall terminal exit; Side is wrong-wall exit; Sink is a physically low-energy floor stop; NoPr is no viable continuation primitive; Hard is a claim-bearing hard failure.",
        r"$\bar{J}$ and $\tilde{J}$ are the mean and median mission scores. $E_T$ is terminal specific energy, $E_R$ is terminal energy reserve, $E_+$ is positive specific-energy gain, $E_u$ is updraft energy-gain proxy, and $t_L$ is lift dwell time.",
        r"Blue marks the best value and red marks the worst value within each environment-ladder table for each metric column; lower is better for Side, Sink, NoPr, and Hard, while higher is better for the remaining metrics.",
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
        frame = _summary_from_final_launch_scores(score_paths)
        output = args.output
        if output == DEFAULT_OUTPUT and len(score_paths) > 1:
            args.output = DEFAULT_COMBINED_OUTPUT
        source_label = "; ".join(str(path).replace("\\", "/") for path in score_paths)
        return frame, source_label
    return pd.read_csv(args.input), str(args.input).replace("\\", "/")


def _summary_from_final_launch_scores(score_paths: list[Path]) -> pd.DataFrame:
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
    caption = (
        f"Held-out repeated-launch validation under {LADDER_LABELS[ladder_id]}. "
        "Rows condition performance on launch speed, primitive-library compression, and the number of previous launches used by the memory-assisted governor."
    )
    label = f"tab:r11-d01-{table_id}-speed-cluster-policy"
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
            for policy_id in POLICY_ORDER:
                row = library_frame[library_frame["policy_id"].astype(str) == policy_id]
                if row.empty:
                    continue
                lines.append(_table_row(row.iloc[0], speed_id, library_id, policy_id, metric_extrema, speed_started, library_started))
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
) -> str:
    cells = []
    if not speed_started:
        cells.append(rf"\SetCell[r=20]{{l,m}} {SPEED_LABELS[speed_id]}")
    else:
        cells.append("")
    if not library_started:
        cells.append(rf"\SetCell[r=4]{{l,m}} {LIBRARY_LABELS[library_id]}")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
