###### Initialization

### Imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# Legend styling
LEGEND_FONT_SIZE = 10.5
AXIS_EDGE_LW = 0.80
LEGEND_FRAME_LW = AXIS_EDGE_LW
LEGEND_LOC = "upper right"
LEGEND_BBOX_TO_ANCHOR = (1.15, 1.05)
LEGEND_HANDLE_LENGTH = 1.5
LEGEND_BORDERPAD = 0.5
LEGEND_LABEL_SPACING = 0.4
# Manual y-axis limits per plot
WRMSE_Y_LIMITS = (0.0, 0.60)
SAE_Y_LIMITS = (0.0, 50.0)


MODEL_SPECS = [
    {
        "key": "single_var",
        "legend_label": "Axisymmetric Gaussian plume",
        "excel": Path("B_results") / "single_var_analysis.xlsx",
        "color": "#EE3B2A",
        "linestyle": "--",
        "marker": "^",
        "line_alpha": 0.80,
        "line_width": 1.3,
    },
    {
        "key": "single_annular_var",
        "legend_label": "Axisymmetric annular-Gaussian",
        "excel": Path("B_results") / "single_annular_var_analysis.xlsx",
        "color": "#6BADD7",
        "linestyle": "-",
        "marker": "s",
        "line_alpha": 0.60,
        "line_width": 1.3,
    },
    {
        "key": "single_annular_bemt",
        "legend_label": "Harmonic annular-Gaussian",
        "excel": Path("B_results") / "single_annular_bemt_analysis.xlsx",
        "color": "#206FB6",
        "linestyle": "-.",
        "marker": "o",
        "line_alpha": 1.00,
        "line_width": 1.3,
    },
    {
        "key": "single_annular_gp",
        "legend_label": "Residual annular Gaussian process",
        "excel": Path("B_results") / "single_annular_gp_analysis.xlsx",
        "color": "#073068",
        "linestyle": ":",
        "marker": "D",
        "line_alpha": 0.80,
        "line_width": 1.7,
    },
]

# Visual stacking requested by user.
# Top -> bottom: GP, HAG, AG, G
STACK_ORDER_TOP_TO_BOTTOM = [
    "single_annular_gp",
    "single_annular_bemt",
    "single_annular_var",
    "single_var",
]
STACK_ORDER_BOTTOM_TO_TOP = list(reversed(STACK_ORDER_TOP_TO_BOTTOM))


def load_per_height_metrics(excel_path: Path):
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing analysis file: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    df = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0])

    required = {"sheet", "z_m", "n_samples", "accumulate_SAE_mps", "weighted_RMSE_mps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{excel_path} is missing required columns: {sorted(missing)}")

    d = df.copy()
    d["sheet"] = d["sheet"].astype(str)

    total_rows = d[d["sheet"].str.upper() == "TOTAL"].copy()
    per_height = d[d["sheet"].str.upper() != "TOTAL"].copy()

    per_height["z_m"] = pd.to_numeric(per_height["z_m"], errors="coerce")
    per_height["n_samples"] = pd.to_numeric(per_height["n_samples"], errors="coerce")
    per_height["accumulate_SAE_mps"] = pd.to_numeric(per_height["accumulate_SAE_mps"], errors="coerce")
    per_height["weighted_RMSE_mps"] = pd.to_numeric(per_height["weighted_RMSE_mps"], errors="coerce")
    per_height = per_height.dropna(subset=["z_m", "n_samples", "accumulate_SAE_mps", "weighted_RMSE_mps"])

    per_height["weighted_rmse_per_height_mps"] = per_height["weighted_RMSE_mps"]
    per_height["sae_per_height_mps"] = per_height["accumulate_SAE_mps"]
    per_height = per_height.sort_values("z_m")

    if total_rows.empty:
        total_sae = float(per_height["sae_per_height_mps"].sum())
        total_n = float(per_height["n_samples"].sum())
        total_weighted_rmse = np.nan
    else:
        total_row = total_rows.iloc[0]
        total_sae = float(total_row["accumulate_SAE_mps"])
        total_n = float(total_row["n_samples"])
        total_weighted_rmse = float(total_row["weighted_RMSE_mps"])

    total_metrics = {
        "total_n_samples": int(total_n),
        "total_sae_mps": total_sae,
        "total_weighted_rmse_mps": total_weighted_rmse,
    }

    cols = ["sheet", "z_m", "n_samples", "sae_per_height_mps", "weighted_rmse_per_height_mps"]
    return per_height[cols], total_metrics


def plot_metric_2d(
    model_frames,
    metric_key: str,
    y_label: str,
    out_path: Path,
    y_limits: tuple[float, float],
):
    fig, ax = plt.subplots(figsize=(6.8, 2.75), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=(0.85, 0.85, 0.85, 1.0), linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    marker_size = 15
    axis_label_size = 12
    tick_label_size = 11

    all_x = set()
    spec_by_key = {spec["key"]: spec for spec in MODEL_SPECS}
    draw_specs = [
        spec_by_key[k]
        for k in STACK_ORDER_BOTTOM_TO_TOP
        if k in spec_by_key and k in model_frames
    ]
    if not draw_specs:
        draw_specs = [spec for spec in MODEL_SPECS if spec["key"] in model_frames]

    for draw_idx, spec in enumerate(draw_specs, start=1):
        key = spec["key"]

        frame = model_frames[key].sort_values("z_m")
        x = frame["z_m"].to_numpy(dtype=float)
        y = frame[metric_key].to_numpy(dtype=float)

        all_x.update(x.tolist())
        ax.plot(
            x,
            y,
            color=spec["color"],
            linewidth=float(spec.get("line_width", 1.3)),
            alpha=spec["line_alpha"],
            linestyle=spec["linestyle"],
            zorder=float(draw_idx),
            label=spec["legend_label"],
        )
        ax.scatter(
            x,
            y,
            s=marker_size,
            marker=spec["marker"],
            color=spec["color"],
            alpha=1.0,
            edgecolors="none",
            zorder=float(draw_idx) + 10.0,
        )

    if all_x:
        ax.set_xticks(sorted(all_x))
    ax.tick_params(axis="x", labelrotation=30)
    ax.tick_params(axis="both", labelsize=tick_label_size)

    y_min, y_max = y_limits
    if y_max <= y_min:
        raise ValueError(f"Invalid y_limits: {y_limits}. Expected y_max > y_min.")
    ax.set_ylim(float(y_min), float(y_max))

    ax.set_xlabel(r"Measurement height above fan outlet plane, $z_{{\mathrm{{fan}}}}$ (m)", fontsize=axis_label_size)
    ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    legend_kwargs = {
        "loc": LEGEND_LOC,
        "frameon": True,
        "framealpha": 1.0,
        "edgecolor": "black",
        "fontsize": LEGEND_FONT_SIZE,
        "handlelength": LEGEND_HANDLE_LENGTH,
        "borderpad": LEGEND_BORDERPAD,
        "labelspacing": LEGEND_LABEL_SPACING,
    }
    if LEGEND_BBOX_TO_ANCHOR is not None:
        legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
    legend_specs = [
        spec_by_key[k]
        for k in STACK_ORDER_TOP_TO_BOTTOM
        if k in spec_by_key and k in model_frames
    ]
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_labels_ordered = [
        spec["legend_label"] for spec in legend_specs if spec["legend_label"] in handle_by_label
    ]
    legend_handles_ordered = [handle_by_label[label] for label in legend_labels_ordered]

    if legend_handles_ordered:
        leg = ax.legend(legend_handles_ordered, legend_labels_ordered, **legend_kwargs)
    else:
        leg = ax.legend(**legend_kwargs)
    if leg is not None:
        leg.get_frame().set_linewidth(LEGEND_FRAME_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def main():
    out_dir = Path("A_figures") / "Fitting_Analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_frames = {}
    total_rows = []

    for spec in MODEL_SPECS:
        per_height, total_metrics = load_per_height_metrics(spec["excel"])
        model_frames[spec["key"]] = per_height
        total_rows.append(
            {
                "model": spec["key"],
                "total_n_samples": total_metrics["total_n_samples"],
                "total_sae_mps": total_metrics["total_sae_mps"],
                "total_weighted_rmse_mps": total_metrics["total_weighted_rmse_mps"],
            }
        )

    plot_metric_2d(
        model_frames=model_frames,
        metric_key="weighted_rmse_per_height_mps",
        y_label=r"WRMSE per-height (m $\!$s$^{-1}$)",
        out_path=out_dir / "single_total_weighted_rmse_per_height.png",
        y_limits=WRMSE_Y_LIMITS,
    )
    plot_metric_2d(
        model_frames=model_frames,
        metric_key="sae_per_height_mps",
        y_label=r"SAE per-height (m $\!$s$^{-1}$)",
        out_path=out_dir / "single_total_sae_per_height.png",
        y_limits=SAE_Y_LIMITS,
    )

    totals_df = pd.DataFrame(total_rows)
    print(totals_df.to_string(index=False))
    print(f"Done. Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

