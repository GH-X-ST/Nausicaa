from __future__ import annotations

import argparse
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
from logging_contract import write_rollout_outputs
from rollout import (
    RolloutConfig,
    make_constant_command_schedule,
    rollout_open_loop_normalised,
)


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
ROLLOUT_SMOKE_CAMPAIGN = "01_rollout_smoke"


def _initial_smoke_state() -> np.ndarray:
    # Public world frame uses z up; body velocity uses the canonical body axis.
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
    """Run the deterministic no-wind rollout smoke case and write outputs."""

    config = RolloutConfig(dt_s=0.02, t_final_s=0.24, wind_mode="none")
    schedule = make_constant_command_schedule(
        np.zeros(3),
        t_final_s=config.t_final_s,
        dt_s=config.dt_s,
    )
    aircraft = adapt_glider(build_nausicaa_glider())
    result = rollout_open_loop_normalised(
        _initial_smoke_state(),
        schedule,
        config,
        aircraft=aircraft,
        wind_model=None,
        seed=1,
        scenario_name="rollout_smoke",
    )
    return write_rollout_outputs(
        result,
        DEFAULT_RESULTS_ROOT,
        ROLLOUT_SMOKE_CAMPAIGN,
        run_id,
        overwrite=overwrite,
    )


# =============================================================================
# 2) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the rollout smoke audit.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_smoke(run_id=args.run_id, overwrite=args.overwrite)
    import json

    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['manifest_json']}")
    print(f"report={outputs['report_md']}")
    print(f"success={manifest['success']}")
    print(f"failure_label={manifest['failure_label']}")


if __name__ == "__main__":
    main()
