from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup
# 2) Batch scenario CLI
# =============================================================================

# =============================================================================
# 1) Import Path Setup
# =============================================================================
def _add_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "03_Control/02_Inner_Loop",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
    ):
        path = repo_root / rel
        if str(path) not in sys.path:
            # Batch runs are launched as scripts, so paths are resolved from repo root.
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _add_paths()

from run_one import run_scenario  # noqa: E402
from scenarios import batch_scenarios  # noqa: E402


# =============================================================================
# 2) Batch Scenario CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    # Scenario ordering is centralised in scenarios.py for reproducible comparisons.
    rows = [run_scenario(scenario_id, args.seed) for scenario_id in batch_scenarios()]
    out_path = (
        REPO_ROOT
        / "03_Control"
        / "05_Results"
        / "metrics"
        / f"batch_seed{args.seed}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        # One CSV row per scenario keeps batch metrics comparable across seeds.
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("batch complete")
    print(f"metrics: 03_Control/05_Results/metrics/batch_seed{args.seed}.csv")


if __name__ == "__main__":
    main()
