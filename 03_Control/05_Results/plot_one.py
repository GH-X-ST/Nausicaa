from __future__ import annotations

import argparse
from pathlib import Path

from plotting import generate_scenario_figure


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) CLI argument parsing
# 2) Single-scenario plotting entry point
# =============================================================================

# =============================================================================
# 1) CLI Argument Parsing
# =============================================================================
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-control-reference", action="store_true")
    parser.add_argument("--no-environment-reference", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--save-pdf", action="store_true")
    return parser


# =============================================================================
# 2) Single-Scenario Plotting Entry Point
# =============================================================================
def main() -> None:
    args = _parser().parse_args()
    try:
        paths = generate_scenario_figure(
            scenario_id=args.scenario,
            seed=args.seed,
            output_root=None if args.output_root is None else Path(args.output_root),
            include_control_reference=not args.no_control_reference,
            include_environment_reference=not args.no_environment_reference,
            save_png=not args.no_png,
            save_pdf=args.save_pdf,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"plot skipped: {exc}")
        raise SystemExit(1) from exc
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
