from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from linearisation import linearise_trim
from trim_linearisation_audit import (
    DYNAMIC_RESIDUAL_NORM_TOL,
    JACOBIAN_MAX_ABS_TOL,
    build_trim_linearisation_cases,
    compare_jacobians,
    dynamic_residual_from_xdot,
    finite_difference_linearisation,
    reduced_model_audit_rows,
    run_trim_linearisation_audit,
    uniform_wind_consistency_check,
)


def test_case_inventory_marks_required_optional_and_diagnostic_speeds() -> None:
    cases = {case.name: case for case in build_trim_linearisation_cases()}

    assert cases["natural_glide_6p5_none"].required is True
    assert cases["natural_glide_6p5_none"].speed_m_s == 6.5
    assert cases["natural_glide_5p5_none"].required is False
    assert cases["natural_glide_5p5_none"].speed_m_s == 5.5
    assert cases["natural_glide_7p5_none"].required is False
    assert cases["natural_glide_7p5_none"].speed_m_s == 7.5
    assert cases["natural_glide_4p5_none"].required is False
    assert cases["natural_glide_4p5_none"].speed_m_s == 4.5
    assert cases["uniform_cg_updraft_6p5"].wind_mode == "cg"
    assert cases["uniform_cg_updraft_6p5"].wind_w_m_s == (0.0, 0.0, 0.5)


def test_dynamic_residual_excludes_position_rates() -> None:
    x_dot = np.zeros(15)
    x_dot[0:3] = [1.0, -2.0, 3.0]

    residual = dynamic_residual_from_xdot(x_dot)

    assert residual["position_rate_norm"] > 0.0
    assert residual["dynamic_residual_norm"] == 0.0
    assert residual["dynamic_residual_norm"] <= DYNAMIC_RESIDUAL_NORM_TOL


def test_compare_jacobians_identical_matrices_pass() -> None:
    a_matrix = np.eye(15)
    b_matrix = np.ones((15, 3))

    comparison = compare_jacobians(a_matrix, b_matrix, a_matrix.copy(), b_matrix.copy())

    assert comparison["pass"] is True
    assert comparison["a_max_abs_error"] == 0.0
    assert comparison["b_max_abs_error"] == 0.0
    assert JACOBIAN_MAX_ABS_TOL == 1e-4


def test_finite_difference_linearisation_shapes_are_finite() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    model = linearise_trim(aircraft=aircraft)

    a_fd, b_fd = finite_difference_linearisation(
        model.x_trim,
        model.u_trim,
        aircraft,
        rho_kg_m3=1.225,
        actuator_tau_s=(0.06, 0.06, 0.06),
    )

    assert a_fd.shape == (15, 15)
    assert b_fd.shape == (15, 3)
    assert np.all(np.isfinite(a_fd))
    assert np.all(np.isfinite(b_fd))


def test_reduced_model_audit_rows_have_expected_shapes() -> None:
    model = linearise_trim()

    rows = reduced_model_audit_rows(model)

    assert set(rows["model_name"]) == {"longitudinal", "lateral"}
    longitudinal = rows[rows["model_name"] == "longitudinal"].iloc[0]
    lateral = rows[rows["model_name"] == "lateral"].iloc[0]
    assert longitudinal["a_rows"] == 5
    assert longitudinal["a_cols"] == 5
    assert longitudinal["b_rows"] == 5
    assert longitudinal["b_cols"] == 1
    assert lateral["a_rows"] == 6
    assert lateral["a_cols"] == 6
    assert lateral["b_rows"] == 6
    assert lateral["b_cols"] == 2
    assert rows["hard_gate_pass"].astype(bool).all()


def test_uniform_wind_consistency_check_passes_for_constant_wind() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    model = linearise_trim(aircraft=aircraft)

    row = uniform_wind_consistency_check(model.x_trim, model.u_trim, aircraft)

    assert row["finite"] is True
    assert row["pass"] is True
    assert row["max_abs_xdot_diff"] <= 1e-9


def test_audit_writes_outputs_manifest_and_relative_paths(tmp_path: Path) -> None:
    outputs = run_trim_linearisation_audit(output_root=tmp_path, run_id=1)
    root = tmp_path / "02_trim_linearisation" / "001"
    expected = (
        root / "metrics" / "trim_cases_s001.csv",
        root / "metrics" / "dynamic_residuals_s001.csv",
        root / "metrics" / "linearisation_summary_s001.csv",
        root / "metrics" / "finite_difference_check_s001.csv",
        root / "metrics" / "key_derivatives_s001.csv",
        root / "metrics" / "reduced_model_audit_s001.csv",
        root / "metrics" / "actuator_dynamics_s001.csv",
        root / "metrics" / "uniform_wind_consistency_s001.csv",
        root / "logs" / "linear_model_s001.npz",
        root / "manifests" / "trim_linearisation_manifest_s001.json",
        root / "reports" / "trim_linearisation_report_s001.md",
    )

    assert outputs["root"] == root
    for path in expected:
        assert path.exists()

    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    assert manifest["trim_audit_implemented"] is True
    assert manifest["linearisation_audit_implemented"] is True
    assert manifest["controller_implemented"] is False
    assert manifest["primitive_implemented"] is False
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["dynamic_residual_norm_tol"] == DYNAMIC_RESIDUAL_NORM_TOL
    assert manifest["jacobian_max_abs_tol"] == JACOBIAN_MAX_ABS_TOL
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())

    trim_cases = pd.read_csv(outputs["trim_cases_csv"])
    case_names = set(trim_cases["case_name"])
    assert "natural_glide_6p5_none" in case_names
    assert "natural_glide_5p5_none" in case_names
    assert "natural_glide_7p5_none" in case_names
    assert "natural_glide_4p5_none" in case_names
    assert "agile_reversal_knots" in case_names
    required = trim_cases[trim_cases["case_name"] == "natural_glide_6p5_none"].iloc[0]
    assert bool(required["converged"]) is True
    assert required["linearisation_status"] == "linearised"

    report = outputs["report_md"].read_text(encoding="ascii")
    assert "trajectory-knot linearisation" in report
    assert "entry and exit speed" in report
