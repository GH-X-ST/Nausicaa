from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from governor import ViabilityGovernor
from primitive import build_primitive_context
from trajectory_primitive import TrajectoryEntryLimits, TrajectoryPrimitive


def _primitive() -> TrajectoryPrimitive:
    times = np.array([0.0, 0.1, 0.2], dtype=float)
    x_ref = np.zeros((3, 15), dtype=float)
    u_ff = np.deg2rad(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, -3.0, 2.0],
                [8.0, -6.0, 4.0],
            ],
            dtype=float,
        )
    )
    return TrajectoryPrimitive(
        name="unit_agile_trajectory",
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        k_lqr=np.zeros((3, 3, 15), dtype=float),
        a_mats=np.zeros((3, 15, 15), dtype=float),
        b_mats=np.zeros((3, 15, 3), dtype=float),
        s_mats=np.zeros((3, 15, 15), dtype=float),
        entry_limits=TrajectoryEntryLimits(max_position_error_m=0.5),
    )


def test_trajectory_primitive_command_interpolates_finite_command_order() -> None:
    primitive = _primitive()
    context = build_primitive_context(np.zeros(15, dtype=float), np.zeros(3, dtype=float))
    command = primitive.command(0.05, np.zeros(15, dtype=float), context)

    assert command.shape == (3,)
    assert np.all(np.isfinite(command))
    np.testing.assert_allclose(command, np.deg2rad([2.0, -1.5, 1.0]))


def test_rejected_trajectory_candidate_writes_reason(tmp_path: Path) -> None:
    primitive = _primitive()
    context = build_primitive_context(np.zeros(15, dtype=float), np.zeros(3, dtype=float))
    x0 = np.zeros(15, dtype=float)
    x0[2] = 0.1
    governor = ViabilityGovernor()
    decision = governor.select_primitive(
        scenario_id="unit_trajectory_rejection",
        primitives=(primitive,),
        x0=x0,
        context=context,
        rollout_callable=lambda candidate: {"success": True},
    )
    path = tmp_path / "rejections.csv"
    governor.write_rejection_log(path)

    assert decision.accepted is False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "reason" in rows[0]
    assert rows[0]["reason"]
