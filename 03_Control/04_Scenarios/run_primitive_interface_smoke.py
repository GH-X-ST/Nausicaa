from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from primitive_interface import (
    PRIMITIVE_INTERFACE_CAMPAIGN,
    PrimitiveExecutionConfig,
    build_interface_smoke_spec,
    execute_open_loop_primitive_interface,
    write_primitive_interface_outputs,
)
from rollout import make_constant_command_schedule


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Smoke Scenario Construction
# 2) CLI
# =============================================================================


# =============================================================================
# 1) Smoke Scenario Construction
# =============================================================================
DEFAULT_RESULTS_ROOT = CONTROL_DIR / "05_Results"


def _initial_smoke_state() -> np.ndarray:
    # Public world position uses z up; body velocity uses x-forward body axes.
    return np.array(
        [
            2.5,
            2.2,
            1.5,
            0.0,
            0.0,
            0.0,
            6.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def run_smoke(run_id: int = 1, overwrite: bool = False) -> dict[str, Path]:
    """Run the deterministic primitive-interface smoke case and write outputs."""

    config = PrimitiveExecutionConfig(dt_s=0.02, t_final_s=0.24)
    schedule = make_constant_command_schedule(
        np.zeros(3),
        t_final_s=config.t_final_s,
        dt_s=config.dt_s,
    )
    aircraft = adapt_glider(build_nausicaa_glider())
    result = execute_open_loop_primitive_interface(
        build_interface_smoke_spec(),
        _initial_smoke_state(),
        schedule,
        config,
        aircraft=aircraft,
        wind_model=None,
    )
    return write_primitive_interface_outputs(
        result,
        DEFAULT_RESULTS_ROOT,
        PRIMITIVE_INTERFACE_CAMPAIGN,
        run_id,
        overwrite=overwrite,
    )


# =============================================================================
# 2) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the primitive-interface smoke audit.")
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


if __name__ == "__main__":
    main()
