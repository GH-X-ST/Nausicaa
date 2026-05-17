from __future__ import annotations

import math

import numpy as np
import pandas as pd

import run_aerosandbox_envelope_check as checker


def _small_grid() -> pd.DataFrame:
    return checker.build_verification_grid(
        alpha_deg=(-4.0, 0.0, 4.0),
        beta_deg=(0.0,),
        speed_m_s=(6.5,),
        elevator_deg=(-10.0, 0.0, 10.0),
        aileron_deg=(-10.0, 0.0, 10.0),
        rudder_deg=(-10.0, 0.0, 10.0),
    )


def test_build_verification_grid_contains_envelope_and_control_cases() -> None:
    grid = checker.build_verification_grid()

    assert grid["alpha_deg"].nunique() > 1
    assert grid["beta_deg"].nunique() > 1
    assert grid["speed_m_s"].nunique() > 1
    assert {"clean", "elevator", "aileron", "rudder"} <= set(grid["case_type"])


def test_local_model_coefficients_are_finite_for_small_grid() -> None:
    coeffs = checker.local_model_coefficients(_small_grid())

    required = ["cx_body", "cz_body", "cl_roll", "cm_pitch", "cn_yaw"]
    assert coeffs["valid_local"].all()
    assert np.isfinite(coeffs[required].to_numpy(dtype=float)).all()


def test_local_clean_alpha_sweep_has_positive_lift_slope() -> None:
    coeffs = checker.local_model_coefficients(_small_grid())
    working = coeffs.copy()
    working["alpha_rad"] = np.deg2rad(working["alpha_deg"])
    clean = (working["case_type"] == "clean") & (working["beta_deg"] == 0.0)

    slope = checker.estimate_slope(working, "alpha_rad", "cl_lift", clean)

    assert slope is not None
    assert slope > 0.0


def test_local_drag_is_positive_when_available() -> None:
    coeffs = checker.local_model_coefficients(_small_grid())
    finite_drag = coeffs["cd_drag"].dropna()

    assert not finite_drag.empty
    assert (finite_drag > 0.0).all()


def test_estimate_slope_returns_linear_dataset_slope() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 3.0, 5.0, 7.0]})

    assert math.isclose(checker.estimate_slope(data, "x", "y"), 2.0)


def test_status_labels_mark_pitch_moment_review_separately() -> None:
    labels = checker._status_labels(
        {
            "local_cl_alpha_per_rad": 4.9,
            "aerosandbox_cl_alpha_per_rad": 4.8,
            "cl_alpha_percent_difference": -2.0,
            "local_cm_alpha_per_rad": 2.4,
            "aerosandbox_cm_alpha_per_rad": 0.17,
            "cm_alpha_percent_difference": -93.0,
            "local_elevator_sign_ok": True,
            "local_aileron_sign_ok": True,
            "local_rudder_sign_ok": True,
        }
    )

    assert labels["cl_alpha_status"] == "pass"
    assert labels["control_sign_status"] == "pass"
    assert labels["cm_alpha_status"] == "needs_review"
    assert labels["overall_low_alpha_status"] == "pass_with_pitch_moment_review"


def test_workflow_writes_required_outputs_without_aerosandbox(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        checker,
        "aerosandbox_coefficients",
        lambda grid: (None, "unavailable"),
    )

    result = checker.run_aerosandbox_envelope_check(
        output_root=tmp_path,
        alpha_min_deg=-2.0,
        alpha_max_deg=2.0,
        alpha_step_deg=2.0,
        beta_values_deg=(0.0,),
        speed_values_m_s=(6.5,),
    )

    required = [
        "local_envelope_coefficients.csv",
        "pointwise_comparison.csv",
        "comparison_summary.csv",
        "manifest.json",
        "report.md",
        "figures/cl_vs_alpha.png",
        "figures/cd_vs_alpha.png",
        "figures/cm_vs_alpha.png",
        "figures/drag_polar.png",
        "figures/cl_alpha_by_speed.png",
        "figures/local_control_derivatives.png",
    ]
    for rel_path in required:
        assert (tmp_path / rel_path).exists()
    assert result["aerosandbox_available"] is False
    summary = pd.read_csv(tmp_path / "comparison_summary.csv")
    for column in (
        "cl_alpha_status",
        "control_sign_status",
        "cm_alpha_status",
        "overall_low_alpha_status",
    ):
        assert column in summary.columns
    assert "not proof of high-incidence" in (tmp_path / "report.md").read_text(
        encoding="ascii"
    )
