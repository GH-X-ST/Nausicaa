from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_longitudinal_moment_audit as audit


def _small_grid() -> pd.DataFrame:
    return audit.build_longitudinal_moment_grid(
        alpha_deg=(-2.0, 0.0, 2.0),
        speed_m_s=(6.5, 7.5),
        elevator_deg=(0.0,),
    )


def test_build_longitudinal_moment_grid_shape_is_deterministic() -> None:
    grid = audit.build_longitudinal_moment_grid(
        alpha_deg=(-2.0, 0.0, 2.0),
        speed_m_s=(6.5, 7.5),
        elevator_deg=(-10.0, 0.0, 10.0),
    )

    assert len(grid) == 18
    assert grid["case_id"].is_unique
    assert set(grid["case_type"]) == {"clean", "elevator"}


def test_surface_breakdown_has_required_surface_labels_and_finite_totals() -> None:
    breakdown = audit.surface_force_moment_breakdown(_small_grid())
    total = breakdown[breakdown["surface_name"] == "total"]

    assert {"total", "wing", "horizontal_tail", "vertical_tail"} <= set(
        breakdown["surface_name"]
    )
    assert total["valid_local"].all()
    assert np.isfinite(total[["cl_lift", "cd_drag", "cm_pitch"]].to_numpy()).all()


def test_surface_slope_summary_has_finite_total_cm_alpha() -> None:
    breakdown = audit.surface_force_moment_breakdown(_small_grid())
    summary = audit.surface_slope_summary(breakdown)
    total = summary[summary["surface_name"] == "total"].iloc[0]

    assert np.isfinite(float(total["cm_alpha_per_rad"]))
    assert np.isfinite(float(total["cl_alpha_per_rad"]))


def test_geometry_and_static_margin_diagnostics_are_finite() -> None:
    geometry = audit.geometry_reference_table()
    aerosandbox_geometry = audit.aerosandbox_geometry_reference_table()
    static_margin = audit.static_margin_proxy_table()

    assert {"wing", "horizontal_tail", "vertical_tail"} <= set(geometry["surface_name"])
    assert np.isfinite(geometry.filter(like="_build_m").to_numpy(dtype=float)).all()
    assert np.isfinite(
        aerosandbox_geometry.filter(like="_asb_m").to_numpy(dtype=float)
    ).all()
    assert "x_cg_over_mac" in set(static_margin["quantity"])
    assert np.isfinite(static_margin["value"].to_numpy(dtype=float)).all()


def test_workflow_writes_required_outputs(tmp_path) -> None:
    result = audit.run_longitudinal_moment_audit(
        output_root=tmp_path,
        alpha_min_deg=-2.0,
        alpha_max_deg=2.0,
        alpha_step_deg=2.0,
        speed_values_m_s=(6.5,),
    )

    required = [
        "surface_force_moment_breakdown.csv",
        "surface_slope_summary.csv",
        "geometry_reference.csv",
        "aerosandbox_geometry_reference.csv",
        "static_margin_proxy.csv",
        "manifest.json",
        "report.md",
        "figures/cm_alpha_surface_breakdown.png",
        "figures/cl_alpha_surface_breakdown.png",
        "figures/geometry_side_view.png",
    ]
    for rel_path in required:
        assert (tmp_path / rel_path).exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="ascii"))
    assert manifest["aerosandbox_imported"] is False
    assert manifest["high_incidence_validation_claim"] == "false"
    assert result["interpretation_status"] in {
        "pass",
        "pass_with_pitch_moment_review",
        "needs_review",
    }
