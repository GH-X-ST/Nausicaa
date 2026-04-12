"""Export the local first-order sensitivity study for the selected design.

Usage:
    python F_analysis/sensitivity.py

Prerequisite:
    Run `python F_analysis/solve_step_size.py` first. That script performs the
    expensive finite-difference ladder study and stores the saved step-size
    table consumed here.

Outputs:
    - C_results/sensitivity_analysis.xlsx
    - C_results/sensitivity_table.csv
    - C_results/sensitivity_thesis_table.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import F_analysis.sensitivity_core as sensitivity_core


def main() -> None:
    sensitivity_core.export_saved_step_size_analysis()


if __name__ == "__main__":
    main()
