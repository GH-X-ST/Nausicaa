from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plotting import REPO_ROOT, friendly_scenario_name, generate_scenario_figure


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge and CLI parsing
# 2) Batch plotting entry point
# =============================================================================

# =============================================================================
# 1) Import Path Bridge and CLI Parsing
# =============================================================================
_SCENARIOS = REPO_ROOT / "03_Control" / "04_Scenarios"
if str(_SCENARIOS) not in sys.path:
    # Batch plotting shares the scenario catalog without package installation.
    sys.path.insert(0, str(_SCENARIOS))

from scenarios import batch_scenarios, controller_audit_scenarios  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scenario-set", choices=("audit", "all"), default="audit")
    parser.add_argument("--save-pdf", action="store_true")
    return parser


# =============================================================================
# 2) Batch Plotting Entry Point
# =============================================================================
def main() -> None:
    args = _parser().parse_args()
    scenario_ids = batch_scenarios() if args.scenario_set == "all" else controller_audit_scenarios()
    for scenario_id in scenario_ids:
        try:
            paths = generate_scenario_figure(
                scenario_id=scenario_id,
                seed=args.seed,
                output_root=None if args.output_root is None else Path(args.output_root),
                save_png=True,
                save_pdf=args.save_pdf,
            )
        except (FileNotFoundError, ValueError) as exc:
            # Rejected scenarios have no executed trajectory, so the batch reports and continues.
            print(f"{friendly_scenario_name(scenario_id)}: skipped ({exc})")
            continue
        print(f"{friendly_scenario_name(scenario_id)}: {', '.join(str(path) for path in paths.values())}")


if __name__ == "__main__":
    main()
