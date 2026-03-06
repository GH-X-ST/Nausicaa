"""
Plot multistart selection-stage counts per rerun rank.

Input workbook:
    C_results/weight_sweep_top_rerun_iter3.xlsx
Sheets:
    TopRerunAllStarts (required)
    RunInfo (optional, to infer KEEP_TOP_K)

Output figure:
    B_figures/07_weight_sweep_top_rerun_stage_counts.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_top_rerun_iter3.xlsx")
STARTS_SHEET = "TopRerunAllStarts"
RUNINFO_SHEET = "RunInfo"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "07_weight_sweep_top_rerun_stage_counts.png"

DEFAULT_KEEP_TOP_K = 3

REQUIRED_COLUMNS = [
    "rerun_rank",
    "success",
    "kept_after_dedup",
    "kept_rank",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

COLOR_FAILURE = "#e45756"
COLOR_SUCCESS_DROPPED = "#f58518"
COLOR_KEPT_NOT_TOPK = "#4c78a8"
COLOR_KEPT_TOPK = "#54a24b"


### Helpers
def coerce_bool_like(series: pd.Series) -> pd.Series:
    """Convert bool-like values to strict boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).astype(int).astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    mapped = lowered.map(mapping)
    if mapped.isna().any():
        bad_values = series[mapped.isna()].dropna().unique().tolist()
        raise ValueError(f"Unable to coerce bool-like column. Bad values: {bad_values}")
    return mapped.astype(bool)


def infer_keep_top_k(xlsx_path: Path, default_value: int = DEFAULT_KEEP_TOP_K) -> int:
    """
    Infer KEEP_TOP_K from RunInfo with fallback default.
    Preference order: rerun_keep_top_k -> top_k_requested -> base_keep_top_k.
    """
    xls = pd.ExcelFile(xlsx_path)
    if RUNINFO_SHEET not in xls.sheet_names:
        return int(default_value)

    runinfo = pd.read_excel(xlsx_path, sheet_name=RUNINFO_SHEET)
    if runinfo.empty or "Key" not in runinfo.columns or "Value" not in runinfo.columns:
        return int(default_value)

    kv = dict(zip(runinfo["Key"].astype(str), runinfo["Value"]))
    for key in ["rerun_keep_top_k", "top_k_requested", "base_keep_top_k"]:
        if key in kv:
            try:
                return int(float(kv[key]))
            except Exception:
                continue
    return int(default_value)


def load_starts_df(xlsx_path: Path) -> pd.DataFrame:
    """Load TopRerunAllStarts and enforce required fields."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if STARTS_SHEET not in xls.sheet_names:
        raise KeyError(f"Sheet '{STARTS_SHEET}' not found in {xlsx_path}.")

    df = pd.read_excel(xlsx_path, sheet_name=STARTS_SHEET)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{STARTS_SHEET}': {missing}")

    df = df.copy()
    df["rerun_rank"] = pd.to_numeric(df["rerun_rank"], errors="raise").astype(int)
    df["success"] = coerce_bool_like(df["success"])
    df["kept_after_dedup"] = coerce_bool_like(df["kept_after_dedup"])
    df["kept_rank"] = pd.to_numeric(df["kept_rank"], errors="coerce")

    return df


def build_stage_summary(df: pd.DataFrame, keep_top_k: int) -> pd.DataFrame:
    """
    Build one row per rerun_rank with stage counts.
    """
    rows = []
    for rerun_rank, d in df.groupby("rerun_rank", sort=True, observed=True):
        n_total = int(len(d))
        n_success = int(d["success"].sum())
        n_kept_dedup = int(d["kept_after_dedup"].sum())

        topk_mask = d["kept_after_dedup"] & d["kept_rank"].notna() & (d["kept_rank"] <= keep_top_k)
        n_kept_topk = int(topk_mask.sum())

        n_failures = n_total - n_success
        n_success_dropped_dedup = n_success - n_kept_dedup
        n_kept_dedup_not_topk = n_kept_dedup - n_kept_topk

        counts = [n_failures, n_success_dropped_dedup, n_kept_dedup_not_topk, n_kept_topk]
        if any(c < 0 for c in counts):
            raise ValueError(
                "Inconsistent stage counts detected for rerun_rank="
                f"{rerun_rank}. Counts={counts}."
            )

        rows.append(
            {
                "rerun_rank": int(rerun_rank),
                "n_total": n_total,
                "n_success": n_success,
                "n_kept_dedup": n_kept_dedup,
                "n_kept_topk": n_kept_topk,
                "n_failures": n_failures,
                "n_success_dropped_dedup": n_success_dropped_dedup,
                "n_kept_dedup_not_topk": n_kept_dedup_not_topk,
            }
        )

    summary = pd.DataFrame(rows).sort_values("rerun_rank").reset_index(drop=True)
    return summary


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def annotate_segments(ax: plt.Axes, x: np.ndarray, bottoms: np.ndarray, heights: np.ndarray) -> None:
    """Annotate segment counts when segment height is non-zero."""
    for xi, b, h in zip(x, bottoms, heights):
        if h <= 0:
            continue
        ax.text(
            xi,
            b + 0.5 * h,
            f"{int(h)}",
            ha="center",
            va="center",
            fontsize=8,
            color=(0.05, 0.05, 0.05, 0.95),
            zorder=10,
        )


def plot_stage_counts(summary: pd.DataFrame, keep_top_k: int, out_path: Path) -> None:
    """
    Stacked bars per rerun_rank:
      failures,
      success but dropped in dedup,
      kept after dedup but not top-k,
      kept top-k.
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

    x_labels = summary["rerun_rank"].to_numpy(dtype=int)
    x = np.arange(len(x_labels), dtype=float)

    failures = summary["n_failures"].to_numpy(dtype=float)
    succ_drop = summary["n_success_dropped_dedup"].to_numpy(dtype=float)
    dedup_not_topk = summary["n_kept_dedup_not_topk"].to_numpy(dtype=float)
    kept_topk = summary["n_kept_topk"].to_numpy(dtype=float)
    n_total = summary["n_total"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.9, 4.0), dpi=600)
    fig.patch.set_facecolor("white")
    style_axes(ax)

    width = 0.70

    b0 = np.zeros_like(x)
    ax.bar(
        x,
        failures,
        width=width,
        bottom=b0,
        color=COLOR_FAILURE,
        edgecolor="none",
        alpha=0.90,
        label="Failures",
        zorder=3,
    )

    b1 = b0 + failures
    ax.bar(
        x,
        succ_drop,
        width=width,
        bottom=b1,
        color=COLOR_SUCCESS_DROPPED,
        edgecolor="none",
        alpha=0.90,
        label="Success but dropped in dedup",
        zorder=3,
    )

    b2 = b1 + succ_drop
    ax.bar(
        x,
        dedup_not_topk,
        width=width,
        bottom=b2,
        color=COLOR_KEPT_NOT_TOPK,
        edgecolor="none",
        alpha=0.90,
        label=f"Kept after dedup but not top-{keep_top_k}",
        zorder=3,
    )

    b3 = b2 + dedup_not_topk
    ax.bar(
        x,
        kept_topk,
        width=width,
        bottom=b3,
        color=COLOR_KEPT_TOPK,
        edgecolor="none",
        alpha=0.90,
        label=f"Kept top-{keep_top_k}",
        zorder=3,
    )

    annotate_segments(ax=ax, x=x, bottoms=b0, heights=failures)
    annotate_segments(ax=ax, x=x, bottoms=b1, heights=succ_drop)
    annotate_segments(ax=ax, x=x, bottoms=b2, heights=dedup_not_topk)
    annotate_segments(ax=ax, x=x, bottoms=b3, heights=kept_topk)

    for xi, nt in zip(x, n_total):
        ax.text(
            xi,
            nt + 0.15,
            f"n={int(nt)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=(0.05, 0.05, 0.05, 0.95),
            zorder=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_xlabel("Rerun rank")
    ax.set_ylabel("Count of starts")
    ax.set_title(f"Multistart Selection Stages (KEEP_TOP_K = {keep_top_k})", pad=6)

    y_max = float(np.max(n_total))
    ax.set_ylim(0.0, max(1.0, y_max + 1.0))

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=8.3,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_starts_df(INPUT_XLSX)
    keep_top_k = infer_keep_top_k(INPUT_XLSX, default_value=DEFAULT_KEEP_TOP_K)

    summary = build_stage_summary(df=df, keep_top_k=keep_top_k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_stage_counts(summary=summary, keep_top_k=keep_top_k, out_path=OUT_PATH)

    print(f"Source sheet used: {STARTS_SHEET}")
    print(f"Rows plotted: {len(df)}")
    print(f"KEEP_TOP_K: {keep_top_k}")
    print(summary.to_string(index=False))
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

