from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from episode_logging import write_episode_log
from fixed_gate_contract import FIXED_LAUNCH_GATE
from repeated_launch_episode import RepeatedLaunchEpisodeConfig, run_repeated_launch_episode


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_repeated_launch_episodes_smoke(
    *,
    run_id: int,
    candidate_package_csv: Path | None = None,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "episode_smoke"
    candidates = _default_candidates() if candidate_package_csv is None else pd.read_csv(candidate_package_csv)
    initial_state = _gate_centre_state()
    results = run_repeated_launch_episode(
        initial_state,
        candidates,
        RepeatedLaunchEpisodeConfig(episode_id=f"fg_episode_s{int(run_id):03d}_000"),
    )
    outputs = write_episode_log(
        root,
        episode_summary=results["episode_summary"],
        primitive_steps=results["primitive_steps"],
        governor_queries=results["governor_queries"],
        belief_updates=results["belief_updates"],
        manifest_extra={
            "campaign": CAMPAIGN,
            "pass_name": "repeated_launch_episode_smoke",
            "candidate_package_csv": "" if candidate_package_csv is None else str(candidate_package_csv),
        },
        overwrite=overwrite,
    )
    return outputs.as_dict()


def _gate_centre_state() -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = sum(FIXED_LAUNCH_GATE.x_w_m) / 2.0
    state[1] = sum(FIXED_LAUNCH_GATE.y_w_m) / 2.0
    state[2] = sum(FIXED_LAUNCH_GATE.z_w_m) / 2.0
    state[6] = 5.5
    return state


def _default_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "fg_smoke_glide",
                "primitive_id": "fg_smoke_glide",
                "primitive_family": "glide",
                "entry_source": "launch_gate_main",
                "recommended_use": "thesis",
                "energy_residual_m": -0.05,
                "minimum_margin_m": 0.4,
                "dwell_time_s": 0.3,
                "duration_s": 0.5,
                "forward_displacement_m": 0.3,
                "lift_confidence": 0.8,
            },
            {
                "candidate_id": "fg_smoke_lift_dwell",
                "primitive_id": "fg_smoke_lift_dwell",
                "primitive_family": "lift_dwell_arc",
                "entry_source": "reachable_downstream",
                "recommended_use": "thesis",
                "energy_residual_m": 0.03,
                "minimum_margin_m": 0.3,
                "dwell_time_s": 0.8,
                "duration_s": 0.8,
                "forward_displacement_m": 0.2,
                "lift_confidence": 0.9,
            },
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--candidate-package-csv", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_repeated_launch_episodes_smoke(
        run_id=args.run_id,
        candidate_package_csv=args.candidate_package_csv,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
