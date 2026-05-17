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

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from glide_primitive import (
    GLIDE_CAMPAIGN,
    GlidePrimitiveConfig,
    initial_glide_state,
    rollout_glide_primitive,
    write_glide_outputs,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) W0 Glide Runner
# 2) CLI
# =============================================================================


# =============================================================================
# 1) W0 Glide Runner
# =============================================================================
DEFAULT_RESULTS_ROOT = CONTROL_DIR / "05_Results"


def run_smoke(run_id: int = 1, overwrite: bool = False) -> dict[str, Path]:
    """Run the first actual W0 glide primitive and write evidence outputs."""

    config = GlidePrimitiveConfig()
    aircraft = adapt_glider(build_nausicaa_glider())
    x0 = initial_glide_state(config)
    result = rollout_glide_primitive(
        x0,
        config=config,
        aircraft=aircraft,
        wind_model=None,
    )
    return write_glide_outputs(
        result,
        DEFAULT_RESULTS_ROOT,
        GLIDE_CAMPAIGN,
        run_id,
        overwrite=overwrite,
    )


# =============================================================================
# 2) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the W0 glide primitive.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_smoke(run_id=args.run_id, overwrite=args.overwrite)
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['manifest_json']}")
    print(f"report={outputs['report_md']}")
    print(f"overall_status={manifest['overall_status']}")
    print(f"primitive_success={manifest['primitive_success']}")
    print(f"failure_label={manifest['failure_label']}")


if __name__ == "__main__":
    main()
