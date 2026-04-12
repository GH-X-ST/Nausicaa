"""Plot finite-difference step-size trade-offs from a saved step-size table.

Usage:
    python F_analysis/plot_step_size.py
    python F_analysis/plot_step_size.py --group geometry
    python F_analysis/plot_step_size.py --parameter wing_span_m

Prerequisite:
    Run `python F_analysis/solve_step_size.py` once to create the saved table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import F_analysis.sensitivity_core as sensitivity_core

DEFAULT_FIGURE = sensitivity_core.OUTPUT_FIGURE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=sensitivity_core.OUTPUT_STEP_SIZE_CSV,
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
    parser.add_argument(
        "--max-quantities",
        type=int,
        default=3,
        help="Maximum number of quantities shown per parameter panel.",
    )
    return parser.parse_args()


def quantity_subset_for_group(group: str) -> list[str]:
    if group == "geometry":
        return sensitivity_core.GEOMETRY_STEP_PLOT_QUANTITIES
    if group == "requirement":
        return sensitivity_core.REQUIREMENT_STEP_PLOT_QUANTITIES
    return sorted(
        set(
            sensitivity_core.GEOMETRY_STEP_PLOT_QUANTITIES
            + sensitivity_core.REQUIREMENT_STEP_PLOT_QUANTITIES
        )
    )


def build_filtered_table(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    if args.group != "all":
        df = df.loc[df["group"] == args.group].copy()
    if args.parameter:
        df = df.loc[df["parameter_name"] == args.parameter].copy()
    allowed_quantities = quantity_subset_for_group(args.group)
    if args.parameter:
        geometry_set = set(sensitivity_core.GEOMETRY_PARAM_ORDER)
        if args.parameter in geometry_set:
            allowed_quantities = sensitivity_core.GEOMETRY_STEP_PLOT_QUANTITIES
        else:
            allowed_quantities = sensitivity_core.REQUIREMENT_STEP_PLOT_QUANTITIES
    df = df.loc[df["quantity_name"].isin(allowed_quantities)].copy()
    return df


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(
            f"Step-size table not found at {args.csv}. "
            "Run `python F_analysis/solve_step_size.py` first."
        )
    table_df = build_filtered_table(args)
    sensitivity_core.make_step_size_figure(
        table_df,
        figure_path=args.figure,
        max_quantities=args.max_quantities,
    )
    print(f"Saved step-size trade-off figure: {args.figure}")


if __name__ == "__main__":
    main()
