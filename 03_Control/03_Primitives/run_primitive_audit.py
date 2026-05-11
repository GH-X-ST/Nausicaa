from __future__ import annotations

import argparse
import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup
# 2) Scenario audit CLI
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
            # CLI execution keeps the repository layout importable without package installation.
            sys.path.insert(0, str(path))
    return repo_root


_add_paths()

from run_one import run_scenario  # noqa: E402
from scenarios import s4_audit_scenarios  # noqa: E402


# =============================================================================
# 2) Scenario Audit CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    # Audit scenarios cover nominal, manoeuvre, recovery, measured-wind, and rejection paths.
    rows = [
        run_scenario(scenario_id, seed=args.seed, output_root=args.output_root)
        for scenario_id in s4_audit_scenarios()
    ]
    print(
        f"{'scenario_id':40s} {'ok':>5s} {'primitive':18s} "
        f"{'duration':>8s} {'speed':>8s} {'alpha':>8s} {'wall':>8s}"
    )
    for row in rows:
        print(
            f"{str(row['scenario_id']):40s} {str(row['success']):>5s} "
            f"{str(row['selected_primitive']):18s} "
            f"{float(row['duration_s']):8.3f} "
            f"{float(row['terminal_speed_m_s']):8.3f} "
            f"{float(row['max_alpha_deg']):8.3f} "
            f"{float(row['min_wall_distance_m']):8.3f}"
        )
    print("primitive audit complete")


if __name__ == "__main__":
    main()
