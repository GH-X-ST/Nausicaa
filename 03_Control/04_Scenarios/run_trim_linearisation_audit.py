from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trim_linearisation_audit import run_trim_linearisation_audit


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) CLI
# =============================================================================


# =============================================================================
# 1) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trim and linearisation audit.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_trim_linearisation_audit(
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['manifest_json']}")
    print(f"report={outputs['report_md']}")
    print(f"overall_status={manifest['overall_status']}")


if __name__ == "__main__":
    main()
