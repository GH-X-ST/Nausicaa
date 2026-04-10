"""Plot finite-difference step-size trade-offs from the sensitivity study.

Usage:
    python F_analysis/step_size.py
    python F_analysis/step_size.py --group geometry
    python F_analysis/step_size.py --parameter wing_span_m
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import F_analysis.sensitivity as sensitivity

DEFAULT_FIGURE = sensitivity.FIGURES_DIR / "step_size_tradeoff.png"
GEOMETRY_STEP_QUANTITIES = [
    "sink_rate_mps",
    "mass_total_kg",
    "roll_tau_s",
    "static_margin",
    "static_margin_min_margin",
]
REQUIREMENT_STEP_QUANTITIES = [
    "objective",
    "sink_rate_mps",
    "roll_tau_s",
    "static_margin_min_margin",
    "nom_cl_margin_to_cap",
    "roll_tau_limit_margin",
    "bank_entry_margin_deg",
    "turn_footprint_margin_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=sensitivity.OUTPUT_STEP_SIZE_CSV,
        help="Path to the step-size table CSV.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Path for the output figure.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "geometry", "requirement"),
        default="all",
        help="Optional parameter group filter.",
    )
    parser.add_argument(
        "--parameter",
        default="",
        help="Optional exact parameter_name filter.",
    )
    return parser.parse_args()


def quantity_subset_for_group(group: str) -> list[str]:
    if group == "geometry":
        return GEOMETRY_STEP_QUANTITIES
    if group == "requirement":
        return REQUIREMENT_STEP_QUANTITIES
    return sorted(set(GEOMETRY_STEP_QUANTITIES + REQUIREMENT_STEP_QUANTITIES))


def parameter_order_for_group(group: str) -> list[str]:
    if group == "geometry":
        return sensitivity.GEOMETRY_PARAM_ORDER
    if group == "requirement":
        return sensitivity.REQUIREMENT_PARAM_ORDER
    return sensitivity.GEOMETRY_PARAM_ORDER + sensitivity.REQUIREMENT_PARAM_ORDER


def build_filtered_table(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    if args.group != "all":
        df = df.loc[df["group"] == args.group].copy()
    if args.parameter:
        df = df.loc[df["parameter_name"] == args.parameter].copy()
    allowed_quantities = quantity_subset_for_group(args.group)
    if args.parameter:
        geometry_set = set(sensitivity.GEOMETRY_PARAM_ORDER)
        if args.parameter in geometry_set:
            allowed_quantities = GEOMETRY_STEP_QUANTITIES
        else:
            allowed_quantities = REQUIREMENT_STEP_QUANTITIES
    df = df.loc[df["quantity_name"].isin(allowed_quantities)].copy()
    return df


def make_tradeoff_plot(df: pd.DataFrame, figure_path: Path, group: str) -> None:
    if df.empty:
        raise ValueError("No rows remain after filtering the step-size table.")

    parameter_names = [
        name
        for name in parameter_order_for_group(group)
        if name in set(df["parameter_name"])
    ]
    if not parameter_names:
        parameter_names = sorted(df["parameter_name"].unique().tolist())

    n_panels = len(parameter_names)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 4.2 * nrows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, parameter_name in zip(axes_array, parameter_names, strict=False):
        parameter_df = df.loc[df["parameter_name"] == parameter_name].copy()
        quantity_names = sorted(
            parameter_df["quantity_name"].unique().tolist(),
            key=lambda name: (
                name not in sensitivity.GEOMETRY_HEATMAP_METRICS
                and name not in sensitivity.REQUIREMENT_HEATMAP_METRICS,
                sensitivity.QUANTITY_LABELS.get(name, name),
            ),
        )
        for quantity_name in quantity_names:
            quantity_df = parameter_df.loc[
                parameter_df["quantity_name"] == quantity_name
            ].sort_values("step_size", kind="mergesort")
            x_values = quantity_df["step_size"].to_numpy(dtype=float)
            y_values = np.maximum(
                quantity_df["absolute_error_estimate"].to_numpy(dtype=float),
                1e-18,
            )
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.5,
                markersize=5.0,
                label=sensitivity.QUANTITY_LABELS.get(quantity_name, quantity_name),
            )
            selected_df = quantity_df.loc[quantity_df["selected_for_final"].astype(bool)]
            if not selected_df.empty:
                axis.scatter(
                    selected_df["step_size"],
                    np.maximum(selected_df["absolute_error_estimate"], 1e-18),
                    marker="*",
                    s=80,
                    zorder=5,
                    color="black",
                )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(sensitivity.PARAMETER_LABELS.get(parameter_name, parameter_name))
        axis.set_xlabel("Step size")
        axis.set_ylabel("Absolute error estimate")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, loc="best")

    for axis in axes_array[n_panels:]:
        axis.axis("off")

    fig.suptitle("Finite-difference step-size trade-off", fontsize=14)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_df = build_filtered_table(args)
    make_tradeoff_plot(table_df, args.figure, args.group)
    print(f"Saved step-size trade-off figure: {args.figure}")


if __name__ == "__main__":
    main()
