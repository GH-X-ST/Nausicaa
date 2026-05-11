from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "03_Control/02_Inner_Loop",
    "03_Control/03_Primitives",
    "03_Control/04_Scenarios",
):
    path = REPO_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena import ArenaConfig, safety_margins  # noqa: E402
from linearisation import STATE_INDEX  # noqa: E402
from scenarios import build_scenario, s4_audit_scenarios  # noqa: E402


def _trim_like_state() -> np.ndarray:
    x_trim = np.zeros(15, dtype=float)
    x_trim[STATE_INDEX["u"]] = 6.5
    x_trim[STATE_INDEX["z_w"]] = 2.7
    return x_trim


def test_s4_audit_entry_states_start_inside_true_safety_volume() -> None:
    config = ArenaConfig()
    x_trim = _trim_like_state()
    outside: list[str] = []

    for scenario_id in s4_audit_scenarios():
        scenario = build_scenario(scenario_id, x_trim=x_trim, repo_root=REPO_ROOT, seed=1)
        margins = safety_margins(scenario.x0, config)
        if not bool(margins["inside_safe_volume"]):
            outside.append(scenario_id)

    assert outside == []

