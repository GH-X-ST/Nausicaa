"""
Plot workflow correlation heatmap (focused input-vs-output block by default).

Input workbook:
    C_results/nausicaa_workflow_iter5.xlsx
Preferred sheet:
    CorrelationMatrix
Fallback sheet:
    CorrelationData (compute correlation)

Output figure:
    B_figures/14_nausicaa_workflow_correlation_heatmap.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cmocean
except ImportError:
    cmocean = None


### User settings
INPUT_XLSX = Path("C_results/nausicaa_workflow_iter5.xlsx")
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "14_nausicaa_workflow_correlation_heatmap.png"

SHEET_CORR_MATRIX = "CorrelationMatrix"
SHEET_CORR_DATA = "CorrelationData"

IDENTIFIER_VARS = ["candidate_id", "scenario_id"]

FOCUS_INPUT_VARS = [
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
    "w_gust_nom",
    "w_gust_turn",
    "ixx_scale",
    "iyy_scale",
    "izz_scale",
    "wing_E_scale",
    "htail_E_scale",
    "wing_thickness_scale",
    "tail_thickness_scale",
]

FOCUS_OUTPUT_VARS = [
    "nom_sink_rate_mps",
    "nom_alpha_margin_deg",
    "nom_cl_margin_to_cap",
    "nom_util_e",
    "nom_roll_tau_s",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80


### Helpers
def load_from_correlation_matrix(xlsx_path: Path) -> pd.DataFrame | None:
    """
    Load precomputed correlation matrix from CorrelationMatrix sheet.
    Returns None if sheet not present.
    """
    xls = pd.ExcelFile(xlsx_path)
    if SHEET_CORR_MATRIX not in xls.sheet_names:
        return None

    cm = pd.read_excel(xlsx_path, sheet_name=SHEET_CORR_MATRIX)
    if cm.empty:
        return None

    # Expected format has row labels in first column (often "Unnamed: 0").
    first_col = cm.columns[0]
    cm = cm.set_index(first_col)

    cm.index = cm.index.astype(str).str.strip()
    cm.columns = [str(c).strip() for c in cm.columns]

    cm = cm.apply(pd.to_numeric, errors="coerce")

    # Keep only overlapping index/column names and align order.
    common = [v for v in cm.index.tolist() if v in cm.columns.tolist()]
    cm = cm.loc[common, common]
    return cm


def load_from_correlation_data(xlsx_path: Path) -> pd.DataFrame | None:
    """
    Fallback: compute correlation from raw CorrelationData numeric columns.
    Returns None if sheet missing or unusable.
    """
    xls = pd.ExcelFile(xlsx_path)
    if SHEET_CORR_DATA not in xls.sheet_names:
        return None

    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_CORR_DATA)
    if raw.empty:
        return None

    numeric = raw.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        return None

    cm = numeric.corr(numeric_only=True)
    cm.index = cm.index.astype(str)
    cm.columns = cm.columns.astype(str)
    return cm


def load_correlation_matrix(xlsx_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Prefer CorrelationMatrix, fallback to CorrelationData-derived correlation.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    cm = load_from_correlation_matrix(xlsx_path)
    source = SHEET_CORR_MATRIX

    if cm is None or cm.empty:
        cm = load_from_correlation_data(xlsx_path)
        source = SHEET_CORR_DATA

    if cm is None or cm.empty:
        raise ValueError(
            f"Could not load usable correlation data from '{SHEET_CORR_MATRIX}' "
            f"or '{SHEET_CORR_DATA}'."
        )

    return cm, source


def drop_identifier_vars(cm: pd.DataFrame) -> pd.DataFrame:
    """Drop ID variables from rows/cols when present."""
    keep_rows = [v for v in cm.index.tolist() if v not in IDENTIFIER_VARS]
    keep_cols = [v for v in cm.columns.tolist() if v not in IDENTIFIER_VARS]
    return cm.loc[keep_rows, keep_cols]


def build_focus_block(cm: pd.DataFrame) -> Tuple[pd.DataFrame, bool, List[str], List[str]]:
    """
    Build focused input-vs-output block if possible; fallback to full matrix.
    Returns:
      block, is_focused, used_input_vars, used_output_vars
    """
    used_inputs = [v for v in FOCUS_INPUT_VARS if v in cm.index and v in cm.columns]
    used_outputs = [v for v in FOCUS_OUTPUT_VARS if v in cm.columns and v in cm.index]

    if used_inputs and used_outputs:
        block = cm.loc[used_inputs, used_outputs]
        return block, True, used_inputs, used_outputs

    # Fallback to full matrix (square) with shared variables only.
    common = [v for v in cm.index.tolist() if v in cm.columns.tolist()]
    block = cm.loc[common, common]
    return block, False, common, common


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def annotate_cells(ax: plt.Axes, arr: np.ndarray) -> None:
    """Annotate each heatmap cell with correlation value."""
    n_rows, n_cols = arr.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if not np.isfinite(v):
                txt = "nan"
                txt_color = "black"
            else:
                txt = f"{v:+.2f}"
                txt_color = "white" if abs(v) >= 0.55 else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
                zorder=6,
            )


def plot_correlation_heatmap(
    block: pd.DataFrame,
    source: str,
    is_focused: bool,
    out_path: Path,
) -> None:
    """Plot focused or full correlation heatmap with readable labels."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
        }
    )

    arr = block.to_numpy(dtype=float)
    n_rows, n_cols = arr.shape

    fig_w = max(5.8, 0.55 * n_cols + 2.6)
    fig_h = max(4.0, 0.45 * n_rows + 2.1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)
    fig.patch.set_facecolor("white")

    cmap = cmocean.cm.balance if cmocean is not None else plt.get_cmap("coolwarm")
    im = ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")

    style_axes(ax)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(block.columns.tolist(), rotation=55, ha="right")
    ax.set_yticklabels(block.index.tolist())

    # Thin cell grid for readability.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=(1.0, 1.0, 1.0, 0.45), linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate = (n_rows * n_cols) <= 140
    if annotate:
        annotate_cells(ax, arr)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Correlation [-]")
    cbar.ax.tick_params(width=0.6, length=2)
    cbar.outline.set_linewidth(AXIS_EDGE_LW)

    if is_focused:
        ax.set_xlabel("Outputs")
        ax.set_ylabel("Uncertainty inputs")
        title = "Correlation Heatmap: Inputs vs Key Outputs"
    else:
        ax.set_xlabel("Variables")
        ax.set_ylabel("Variables")
        title = "Correlation Heatmap: Full Matrix"

    ax.set_title(title, pad=8)

    fig.text(
        0.01,
        0.995,
        (
            f"Source: {source}. "
            "Identifier vars candidate_id/scenario_id removed when present."
        ),
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": (1.0, 1.0, 1.0, 0.8),
            "edgecolor": "none",
        },
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    cm, source = load_correlation_matrix(INPUT_XLSX)
    cm = drop_identifier_vars(cm)

    block, is_focused, used_inputs, used_outputs = build_focus_block(cm)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_correlation_heatmap(
        block=block,
        source=source,
        is_focused=is_focused,
        out_path=OUT_PATH,
    )

    print(f"Correlation source: {source}")
    print(f"Matrix shape after ID drop: {cm.shape}")
    print(f"Plotted block shape: {block.shape}")
    if is_focused:
        print(f"Focused inputs used ({len(used_inputs)}): {used_inputs}")
        print(f"Focused outputs used ({len(used_outputs)}): {used_outputs}")
    else:
        print("Focused block unavailable; plotted full matrix.")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

