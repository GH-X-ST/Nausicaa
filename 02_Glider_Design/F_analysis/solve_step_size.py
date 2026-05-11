"""Run the finite-difference step-size ladder for the selected design.

Outputs:
- `C_results/step_size_table.csv`
- `C_results/step_size_analysis.xlsx`

The saved ladder is the provenance source for `F_analysis/sensitivity.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allows direct execution as `python F_analysis/solve_step_size.py`.
    sys.path.insert(0, str(PROJECT_ROOT))

import F_analysis.sensitivity_core as sensitivity_core

# =============================================================================
# SECTION MAP
# =============================================================================
# 1) CLI parsing
# 2) Step-size study export
# =============================================================================

# =============================================================================
# 1) CLI Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=sensitivity_core.OUTPUT_STEP_SIZE_CSV,
        help="Path for the saved step-size table CSV.",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=sensitivity_core.OUTPUT_STEP_SIZE_XLSX,
        help="Path for the saved step-size workbook.",
    )
    return parser.parse_args()


# =============================================================================
# 2) Step-Size Study Export
# =============================================================================

def main() -> None:
    args = parse_args()
    context = sensitivity_core.load_selected_baseline()
    # This is the expensive pass: each parameter runs a finite-difference ladder.
    baseline_result, _sensitivity_df, step_size_df = (
        sensitivity_core.compute_study_tables(context)
    )
    # Rebuild the compact table from the saved ladder so both exports use the
    # same step-selection logic as `F_analysis/sensitivity.py`.
    sensitivity_df = sensitivity_core.build_sensitivity_table_from_step_size_table(
        step_size_df
    )
    baseline_df = sensitivity_core.build_baseline_table(context, baseline_result)
    metadata_df = sensitivity_core.build_metadata_table(
        context,
        baseline_result,
        sensitivity_df,
    )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.xlsx.parent.mkdir(parents=True, exist_ok=True)
    step_size_df.to_csv(args.csv, index=False)
    sensitivity_core.write_step_size_excel(
        metadata_df=metadata_df,
        baseline_df=baseline_df,
        step_size_df=step_size_df,
        output_path=args.xlsx,
    )
    print(f"Saved step-size table CSV: {args.csv}")
    print(f"Saved step-size workbook: {args.xlsx}")


if __name__ == "__main__":
    main()
