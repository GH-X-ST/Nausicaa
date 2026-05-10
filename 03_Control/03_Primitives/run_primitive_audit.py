from __future__ import annotations

import sys
from pathlib import Path


def _add_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "03_Control/02_Inner_Loop",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
    ):
        path = repo_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_add_paths()

from run_one import run_scenario  # noqa: E402


def main() -> None:
    for scenario_id in (
        "s0_no_wind",
        "s1_latency_nominal_no_wind",
        "s1_latency_robust_upper_no_wind",
        "s11_governor_rejection",
    ):
        run_scenario(scenario_id, seed=1)
    print("primitive audit complete")


if __name__ == "__main__":
    main()
