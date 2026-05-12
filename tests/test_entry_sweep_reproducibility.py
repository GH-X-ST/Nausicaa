from __future__ import annotations

from pathlib import Path

import numpy as np

from run_sweep import run_entry_sweep, sample_entry_states


def test_entry_samples_repeat_for_same_seed() -> None:
    x_nominal = np.zeros(15, dtype=float)
    x_nominal[:3] = [1.45, 2.4, 2.7]
    first = sample_entry_states(x_nominal, seed=11, sample_count=4)
    second = sample_entry_states(x_nominal, seed=11, sample_count=4)
    np.testing.assert_allclose(first, second)


def test_entry_sweep_metric_order_repeats_for_same_seed(tmp_path: Path) -> None:
    first = run_entry_sweep(
        "s9_agile_reversal_left_no_wind",
        primitive=None,
        seed=5,
        sample_count=2,
        output_root=tmp_path / "first",
    )
    second = run_entry_sweep(
        "s9_agile_reversal_left_no_wind",
        primitive=None,
        seed=5,
        sample_count=2,
        output_root=tmp_path / "second",
    )
    assert [row["sample_index"] for row in first] == [row["sample_index"] for row in second]
    assert [row["run_id"] for row in first] == [row["run_id"] for row in second]
