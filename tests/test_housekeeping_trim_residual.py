from __future__ import annotations

import csv

from run_housekeeping_reproduction import run_housekeeping_reproduction


def test_housekeeping_trim_residual_excludes_position_derivative(tmp_path) -> None:
    result = run_housekeeping_reproduction(
        seed=1,
        output_root=tmp_path,
        quick=True,
    )
    trim_path = result["metrics"]["trim_linearisation_audit"]

    with open(trim_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    trim_residual = float(row["trim_residual_norm"])
    full_derivative = float(row["full_state_derivative_norm"])
    position_derivative = float(row["position_derivative_norm"])
    fd_error = float(row["linearisation_finite_difference_error"])

    assert row["success"] == "True"
    assert trim_residual < 1e-6
    assert full_derivative > 1.0
    assert position_derivative > 1.0
    assert fd_error < 1e-3
