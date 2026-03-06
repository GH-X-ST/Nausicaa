"""
Plot worst-case traceability from WorstCaseReport.

Input workbook:
    C_results/nausicaa_workflow_iter3.xlsx
Sheet:
    WorstCaseReport

Output figure:
    B_figures/13_nausicaa_workflow_worstcase_traceability.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### User settings
INPUT_XLSX = Path("C_results/nausicaa_workflow_iter3.xlsx")
SHEET_NAME = "WorstCaseReport"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "13_nausicaa_workflow_worstcase_traceability.png"

REQUIRED_COLUMNS = [
    "candidate_id",
    "metric",
    "mode",
    "value",
    "scenario_id",
    "mass_scale",
    "cg_x_shift_mac",
    "incidence_bias_deg",
    "drag_factor",
    "eff_a",
    "eff_e",
    "eff_r",
    "bias_a_deg",
    "bias_e_deg",
    "bias_r_deg",
]

TARGET_METRICS = [
    {
        "metric": "Nom sink rate",
        "mode": "max",
        "label": "Nom sink rate [m/s]",
        "color": "#4c78a8",
    },
    {
        "metric": "Nom alpha margin",
        "mode": "min",
        "label": "Nom alpha margin [deg]",
        "color": "#54a24b",
    },
    {
        "metric": "Nom CL margin to cap",
        "mode": "min",
        "label": "Nom CL margin to cap [-]",
        "color": "#f58518",
    },
    {
        "metric": "Max roll tau",
        "mode": "max",
        "label": "Max roll tau [s]",
        "color": "#cc4a74",
    },
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4


### Helpers
def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{SHEET_NAME}': {missing}")


def load_worstcase_df(xlsx_path: Path) -> pd.DataFrame:
    """Load WorstCaseReport and enforce required types."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if SHEET_NAME not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_NAME}' not found in {xlsx_path}.")

    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    validate_columns(df)

    df = df.copy()
    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="raise").astype(int)
    df["metric"] = df["metric"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["scenario_id"] = pd.to_numeric(df["scenario_id"], errors="coerce")

    for col in [
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "drag_factor",
        "eff_a",
        "eff_e",
        "eff_r",
        "bias_a_deg",
        "bias_e_deg",
        "bias_r_deg",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter WorstCaseReport to the configured target metrics and ensure one row
    per (candidate_id, metric, mode).
    """
    selected_parts = []

    for spec in TARGET_METRICS:
        d = df[(df["metric"] == spec["metric"]) & (df["mode"] == spec["mode"])].copy()
        d = d[d["value"].notna()]
        if d.empty:
            continue

        # Defensive deduplication: keep worst row if duplicates exist.
        group_cols = ["candidate_id", "metric", "mode"]
        rows = []
        for _, g in d.groupby(group_cols, as_index=False):
            if spec["mode"] == "max":
                idx = g["value"].idxmax()
            else:
                idx = g["value"].idxmin()
            rows.append(d.loc[idx])
        d_unique = pd.DataFrame(rows)
        selected_parts.append(d_unique)

    if not selected_parts:
        raise ValueError("No rows found for target metrics in WorstCaseReport.")

    out = pd.concat(selected_parts, ignore_index=True)
    out = out.sort_values(by=["metric", "candidate_id"]).reset_index(drop=True)
    return out


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def build_trace_table_text(trace_df: pd.DataFrame) -> str:
    """
    Build compact fixed-width scenario fingerprint text table.
    """
    cols = [
        "candidate_id",
        "metric",
        "mode",
        "value",
        "scenario_id",
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "drag_factor",
        "eff_a",
        "eff_e",
        "eff_r",
        "bias_a_deg",
        "bias_e_deg",
        "bias_r_deg",
    ]
    d = trace_df[cols].copy()

    rename_map = {
        "candidate_id": "cid",
        "metric": "metric",
        "mode": "mode",
        "value": "value",
        "scenario_id": "sid",
        "mass_scale": "mass",
        "cg_x_shift_mac": "cg",
        "incidence_bias_deg": "inc",
        "drag_factor": "drag",
        "eff_a": "ea",
        "eff_e": "ee",
        "eff_r": "er",
        "bias_a_deg": "ba",
        "bias_e_deg": "be",
        "bias_r_deg": "br",
    }
    d = d.rename(columns=rename_map)

    d["sid"] = d["sid"].round().astype("Int64")

    for c in ["value", "mass", "cg", "inc", "drag", "ea", "ee", "er", "ba", "be", "br"]:
        d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "nan")

    metric_short = {
        "Nom sink rate": "sink",
        "Nom alpha margin": "alpha_margin",
        "Nom CL margin to cap": "cl_margin",
        "Max roll tau": "roll_tau",
    }
    d["metric"] = d["metric"].map(lambda m: metric_short.get(m, m))

    d = d.sort_values(by=["cid", "metric"]).reset_index(drop=True)
    return d.to_string(index=False)


def plot_worstcase_traceability(trace_df: pd.DataFrame, out_path: Path) -> None:
    """
    Two-part figure:
      Part A: worst-case value bars by candidate for each target metric.
      Part B: scenario fingerprint table for traceability.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
        }
    )

    candidate_ids = sorted(trace_df["candidate_id"].dropna().astype(int).unique().tolist())
    if not candidate_ids:
        raise ValueError("No candidate IDs found in selected worst-case data.")

    fig = plt.figure(figsize=(12.8, 8.0), dpi=600, constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.30], hspace=0.36, wspace=0.25)

    metric_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    # Part A: one panel per metric.
    for ax, spec in zip(metric_axes, TARGET_METRICS):
        style_axes(ax)

        d = trace_df[(trace_df["metric"] == spec["metric"]) & (trace_df["mode"] == spec["mode"])].copy()
        d = d.sort_values("candidate_id")

        x_pos = np.arange(len(candidate_ids), dtype=float)
        values = []
        scenario_ids = []
        for cid in candidate_ids:
            row = d[d["candidate_id"] == cid]
            if row.empty:
                values.append(np.nan)
                scenario_ids.append(np.nan)
            else:
                values.append(float(row.iloc[0]["value"]))
                scenario_ids.append(float(row.iloc[0]["scenario_id"]))

        values_arr = np.asarray(values, dtype=float)
        sid_arr = np.asarray(scenario_ids, dtype=float)

        bars = ax.bar(
            x_pos,
            np.nan_to_num(values_arr, nan=0.0),
            width=0.65,
            color=spec["color"],
            alpha=0.88,
            edgecolor="none",
            zorder=3,
        )

        # Hide bars for missing values.
        for b, v in zip(bars, values_arr):
            if not np.isfinite(v):
                b.set_alpha(0.0)

        finite_vals = values_arr[np.isfinite(values_arr)]
        if finite_vals.size > 0:
            y_min = float(np.min(finite_vals))
            y_max = float(np.max(finite_vals))
            y_span = max(1e-9, y_max - y_min)
            y_off = 0.05 * y_span if y_span > 1e-9 else 0.06 * max(1.0, abs(y_max))

            for i, (v, sid) in enumerate(zip(values_arr, sid_arr)):
                if not np.isfinite(v):
                    continue
                sid_txt = f"S{int(round(sid))}" if np.isfinite(sid) else "S?"
                ax.text(
                    x_pos[i],
                    v + y_off,
                    sid_txt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=(0.08, 0.08, 0.08, 0.95),
                    zorder=10,
                )

            # Margin boundary helper lines for min-margins.
            if spec["metric"] in {"Nom alpha margin", "Nom CL margin to cap"}:
                ax.axhline(
                    y=0.0,
                    color="#e45756",
                    linewidth=1.0,
                    linestyle=(0, (3, 2)),
                    zorder=2,
                )

            low = y_min - 0.12 * y_span if y_span > 1e-9 else y_min - abs(y_min) * 0.08
            high = y_max + 0.25 * y_span if y_span > 1e-9 else y_max + abs(y_max) * 0.20 + 0.02
            if spec["metric"] in {"Nom alpha margin", "Nom CL margin to cap"}:
                low = min(low, -0.05)
            ax.set_ylim(low, high)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(cid) for cid in candidate_ids])
        ax.set_xlabel("Candidate ID")
        ax.set_ylabel(spec["label"])
        ax.set_title(f"{spec['metric']} ({spec['mode']})", pad=5)

    # Part B: traceability table.
    table_ax = fig.add_subplot(gs[2, :])
    table_ax.axis("off")

    table_title = (
        "Scenario fingerprint for plotted worst-case values "
        "(cid, sid, mass, cg, inc, drag, eff_*, bias_*)"
    )
    table_ax.text(
        0.0,
        1.03,
        table_title,
        ha="left",
        va="bottom",
        fontsize=9,
        transform=table_ax.transAxes,
    )

    table_text = build_trace_table_text(trace_df)
    table_ax.text(
        0.0,
        1.0,
        table_text,
        ha="left",
        va="top",
        fontsize=7.6,
        family="monospace",
        transform=table_ax.transAxes,
    )

    fig.suptitle("Worst-case metric traceability by candidate", y=0.995, fontsize=11)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_worstcase_df(INPUT_XLSX)
    trace_df = select_target_rows(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_worstcase_traceability(trace_df=trace_df, out_path=OUT_PATH)

    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows in WorstCaseReport: {len(df)}")
    print(f"Rows plotted (target metrics): {len(trace_df)}")
    print(
        "Target metrics: "
        + ", ".join([f"{m['metric']} ({m['mode']})" for m in TARGET_METRICS])
    )
    print(f"Candidates: {sorted(trace_df['candidate_id'].unique().tolist())}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

