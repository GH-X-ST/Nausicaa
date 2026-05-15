from __future__ import annotations

import csv

import run_aggressive_reversal_search as runner
from aggressive_reversal_ocp import AggressiveReversalConfig


def test_tvlqr_is_not_success_for_failed_or_fallback_source(tmp_path, monkeypatch) -> None:
    def forced_abort_config(*, n_intervals: int) -> AggressiveReversalConfig:
        return AggressiveReversalConfig(
            n_intervals=n_intervals,
            integration_speed_abort_m_s=0.1,
        )

    monkeypatch.setattr(runner, "AggressiveReversalConfig", forced_abort_config)

    result = runner.run_aggressive_reversal_search(
        targets_deg=(30.0,),
        direction="left",
        seed=1,
        output_root=tmp_path,
        wind_case="w0",
        quick=True,
        use_tvlqr=True,
    )

    with open(result["tvlqr_metrics"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["success"] == "False"
    assert not (row["success"] == "True" and row["feasibility_label"] == "solver_failure")
    assert row["source_feasibility_label"] == "solver_failure"
    assert row["fallback_used"] == "True"
    assert row["failure_reason"] == "source_trajectory_not_promoted"
    assert row["gain_arrays_finite"] == "False"
    assert row["primitive_constructed"] == "False"
    assert row["closed_loop_replay_success"] == "False"


def test_gain_only_diagnostic_is_not_manoeuvre_success(tmp_path) -> None:
    result = runner.run_aggressive_reversal_search(
        targets_deg=(30.0,),
        direction="left",
        seed=1,
        output_root=tmp_path,
        wind_case="w0",
        quick=True,
        use_tvlqr=True,
    )

    with open(result["tvlqr_metrics"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert not (row["success"] == "True" and row["feasibility_label"] == "solver_failure")
    if row["gain_arrays_finite"] == "True":
        assert row["closed_loop_replay_success"] == "False"
        assert row["success"] == "False"
