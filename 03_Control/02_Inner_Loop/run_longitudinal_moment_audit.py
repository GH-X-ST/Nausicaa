from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from A_model_parameters.mass_properties_estimate import MASS_KG, R_CG_BUILD_M
from flight_dynamics import adapt_glider, evaluate_state
from glider import (
    HORIZONTAL_TAIL,
    VERTICAL_TAIL,
    WING,
    Glider,
    LiftingSurface,
    build_nausicaa_glider,
)
from run_aerosandbox_envelope_check import (
    COMPARISON_SCOPE,
    HIGH_INCIDENCE_VALIDATION_CLAIM,
    RHO_KG_M3,
    _alpha_values_from_range,
    _save_audit_figure,
    _wind_axis_coefficients,
    estimate_slope,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "05_Results"
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Shared constants and grid construction
# 2) Local strip-load decomposition
# 3) Geometry and static-margin diagnostics
# 4) Output writing and plotting
# 5) Public workflow and CLI
# =============================================================================


# =============================================================================
# 1) Shared Constants and Grid Construction
# =============================================================================
DEFAULT_OUTPUT_ROOT = Path(
    "03_Control/05_Results/00_model_audit/longitudinal_moment/001"
)

SURFACE_LABELS = {
    WING: "wing",
    HORIZONTAL_TAIL: "horizontal_tail",
    VERTICAL_TAIL: "vertical_tail",
}

BREAKDOWN_COLUMNS = (
    "case_id",
    "alpha_deg",
    "speed_m_s",
    "delta_e_deg",
    "surface_name",
    "surface_code",
    "fx_b_n",
    "fy_b_n",
    "fz_b_n",
    "mx_b_nm",
    "my_b_nm",
    "mz_b_nm",
    "cx_body",
    "cy_body",
    "cz_body",
    "cl_roll",
    "cm_pitch",
    "cn_yaw",
    "cl_lift",
    "cd_drag",
    "valid_local",
    "local_status",
)


def _default_tuple(
    values: tuple[float, ...] | None,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(float(value) for value in (default if values is None else values))


def build_longitudinal_moment_grid(
    alpha_deg: tuple[float, ...] | None = None,
    speed_m_s: tuple[float, ...] | None = None,
    elevator_deg: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Return the deterministic alpha/speed/elevator grid used by this audit."""

    alpha_values = _default_tuple(
        alpha_deg,
        (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    )
    speed_values = _default_tuple(speed_m_s, (4.5, 5.5, 6.5, 7.5, 8.5))
    elevator_values = _default_tuple(elevator_deg, (-10.0, 0.0, 10.0))

    rows: list[dict[str, float | str]] = []
    case_index = 0
    for delta_e_deg in elevator_values:
        case_type = "clean" if math.isclose(delta_e_deg, 0.0, abs_tol=1e-12) else "elevator"
        for speed_value_m_s in speed_values:
            for alpha_value_deg in alpha_values:
                rows.append(
                    {
                        "case_id": f"longitudinal_{case_index:06d}",
                        "case_type": case_type,
                        "alpha_deg": float(alpha_value_deg),
                        "beta_deg": 0.0,
                        "speed_m_s": float(speed_value_m_s),
                        "delta_a_deg": 0.0,
                        "delta_e_deg": float(delta_e_deg),
                        "delta_r_deg": 0.0,
                    }
                )
                case_index += 1
    return pd.DataFrame(rows)


def _state_from_grid_row(row: pd.Series) -> np.ndarray:
    alpha_rad = np.deg2rad(float(row["alpha_deg"]))
    speed_m_s = float(row["speed_m_s"])
    state = np.zeros(15, dtype=float)
    state[6] = speed_m_s * np.cos(alpha_rad)
    state[7] = 0.0
    state[8] = speed_m_s * np.sin(alpha_rad)
    state[12] = 0.0
    state[13] = np.deg2rad(float(row["delta_e_deg"]))
    state[14] = 0.0
    return state


# =============================================================================
# 2) Local Strip-Load Decomposition
# =============================================================================
def _coefficient_row(
    row: pd.Series,
    glider: Glider,
    force_b: np.ndarray,
    moment_b: np.ndarray,
    surface_name: str,
    surface_code: int,
    status: str = "ok",
) -> dict[str, Any]:
    q_dyn_pa = 0.5 * RHO_KG_M3 * float(row["speed_m_s"]) ** 2
    cl_lift, cd_drag = _wind_axis_coefficients(
        force_b=force_b,
        q_dyn_pa=q_dyn_pa,
        s_ref_m2=glider.s_ref_m2,
        alpha_deg=float(row["alpha_deg"]),
        beta_deg=0.0,
    )
    values = (
        force_b[0],
        force_b[1],
        force_b[2],
        moment_b[0],
        moment_b[1],
        moment_b[2],
        cl_lift,
        cd_drag,
    )
    valid = bool(np.all(np.isfinite(values)))
    if not valid:
        status = "nonfinite_local_surface_coefficients"
    return {
        "case_id": row["case_id"],
        "alpha_deg": float(row["alpha_deg"]),
        "speed_m_s": float(row["speed_m_s"]),
        "delta_e_deg": float(row["delta_e_deg"]),
        "surface_name": surface_name,
        "surface_code": int(surface_code),
        "fx_b_n": float(force_b[0]),
        "fy_b_n": float(force_b[1]),
        "fz_b_n": float(force_b[2]),
        "mx_b_nm": float(moment_b[0]),
        "my_b_nm": float(moment_b[1]),
        "mz_b_nm": float(moment_b[2]),
        "cx_body": float(force_b[0] / (q_dyn_pa * glider.s_ref_m2)),
        "cy_body": float(force_b[1] / (q_dyn_pa * glider.s_ref_m2)),
        "cz_body": float(force_b[2] / (q_dyn_pa * glider.s_ref_m2)),
        "cl_roll": float(moment_b[0] / (q_dyn_pa * glider.s_ref_m2 * glider.b_ref_m)),
        "cm_pitch": float(moment_b[1] / (q_dyn_pa * glider.s_ref_m2 * glider.c_ref_m)),
        "cn_yaw": float(moment_b[2] / (q_dyn_pa * glider.s_ref_m2 * glider.b_ref_m)),
        "cl_lift": float(cl_lift),
        "cd_drag": float(cd_drag),
        "valid_local": valid,
        "local_status": status,
    }


def _breakdown_for_grid_row(
    row: pd.Series,
    glider: Glider,
    aircraft: object,
) -> list[dict[str, Any]]:
    state = _state_from_grid_row(row)
    loads = evaluate_state(
        state,
        np.zeros(3, dtype=float),
        aircraft,
        wind_model=None,
        rho=RHO_KG_M3,
        wind_mode="none",
    )
    f_strip_b = np.asarray(loads["strips"]["f_strip_b"], dtype=float)
    m_strip_b = np.asarray(loads["strips"]["m_strip_b"], dtype=float)

    rows: list[dict[str, Any]] = []
    rows.append(
        _coefficient_row(
            row=row,
            glider=glider,
            force_b=np.sum(f_strip_b, axis=0),
            moment_b=np.sum(m_strip_b, axis=0),
            surface_name="total",
            surface_code=-1,
        )
    )
    for surface_code, surface_name in SURFACE_LABELS.items():
        mask = glider.surface_code == surface_code
        rows.append(
            _coefficient_row(
                row=row,
                glider=glider,
                force_b=np.sum(f_strip_b[mask], axis=0),
                moment_b=np.sum(m_strip_b[mask], axis=0),
                surface_name=surface_name,
                surface_code=surface_code,
            )
        )
    return rows


def surface_force_moment_breakdown(grid: pd.DataFrame) -> pd.DataFrame:
    """Evaluate per-surface strip force and moment totals over the audit grid."""

    glider = build_nausicaa_glider()
    aircraft = adapt_glider(glider)
    rows: list[dict[str, Any]] = []
    for _, row in grid.iterrows():
        rows.extend(_breakdown_for_grid_row(row, glider, aircraft))
    return pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS)


def surface_slope_summary(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Summarise neutral-elevator alpha slopes by lifting surface."""

    neutral = breakdown[np.isclose(breakdown["delta_e_deg"], 0.0)].copy()
    neutral["alpha_rad"] = np.deg2rad(neutral["alpha_deg"])
    rows: list[dict[str, Any]] = []
    for surface_name, group in neutral.groupby("surface_name", sort=False):
        surface_code = int(group["surface_code"].iloc[0])
        rows.append(
            {
                "surface_name": surface_name,
                "surface_code": surface_code,
                "sample_count": int(len(group)),
                "cl_alpha_per_rad": estimate_slope(group, "alpha_rad", "cl_lift"),
                "cd_alpha_per_rad": estimate_slope(group, "alpha_rad", "cd_drag"),
                "cm_alpha_per_rad": estimate_slope(group, "alpha_rad", "cm_pitch"),
                "valid_slope": bool(group["valid_local"].all()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 3) Geometry and Static-Margin Diagnostics
# =============================================================================
def _body_to_build(position_b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            R_CG_BUILD_M[0] - position_b[0],
            position_b[1],
            R_CG_BUILD_M[2] + position_b[2],
        ],
        dtype=float,
    )


def _quarter_chord_b(surface: LiftingSurface) -> np.ndarray:
    return np.asarray(surface.root_le_b, dtype=float) + np.array(
        [-0.25 * surface.chord_m, 0.0, 0.0],
        dtype=float,
    )


def _trailing_edge_b(surface: LiftingSurface) -> np.ndarray:
    return np.asarray(surface.root_le_b, dtype=float) + np.array(
        [-surface.chord_m, 0.0, 0.0],
        dtype=float,
    )


def _surface_area(surface: LiftingSurface) -> float:
    return float(surface.span_m * surface.chord_m)


def geometry_reference_table() -> pd.DataFrame:
    """Return model geometry diagnostics in body and build/audit frames."""

    glider = build_nausicaa_glider()
    rows: list[dict[str, Any]] = []
    for surface in glider.surfaces:
        root_b = np.asarray(surface.root_le_b, dtype=float)
        quarter_b = _quarter_chord_b(surface)
        trailing_b = _trailing_edge_b(surface)
        root_build = _body_to_build(root_b)
        quarter_build = _body_to_build(quarter_b)
        trailing_build = _body_to_build(trailing_b)
        rows.append(
            {
                "surface_name": surface.name,
                "surface_code": surface.code,
                "span_m": float(surface.span_m),
                "chord_m": float(surface.chord_m),
                "area_m2": _surface_area(surface),
                "dihedral_deg": float(surface.dihedral_deg),
                "vertical_surface": bool(surface.vertical),
                "root_le_x_b_m": float(root_b[0]),
                "root_le_y_b_m": float(root_b[1]),
                "root_le_z_b_m": float(root_b[2]),
                "quarter_chord_x_b_m": float(quarter_b[0]),
                "quarter_chord_y_b_m": float(quarter_b[1]),
                "quarter_chord_z_b_m": float(quarter_b[2]),
                "trailing_edge_x_b_m": float(trailing_b[0]),
                "trailing_edge_z_b_m": float(trailing_b[2]),
                "root_le_x_build_m": float(root_build[0]),
                "root_le_z_build_m": float(root_build[2]),
                "quarter_chord_x_build_m": float(quarter_build[0]),
                "quarter_chord_z_build_m": float(quarter_build[2]),
                "trailing_edge_x_build_m": float(trailing_build[0]),
                "trailing_edge_z_build_m": float(trailing_build[2]),
            }
        )
    return pd.DataFrame(rows)


def aerosandbox_geometry_reference_table() -> pd.DataFrame:
    """Return audit-only AeroSandbox coordinate conversion diagnostics."""

    geometry = geometry_reference_table()
    rows: list[dict[str, Any]] = []
    for _, row in geometry.iterrows():
        rows.append(
            {
                "surface_name": row["surface_name"],
                "surface_code": int(row["surface_code"]),
                "root_le_x_asb_m": -float(row["root_le_x_b_m"]),
                "root_le_y_asb_m": float(row["root_le_y_b_m"]),
                "root_le_z_asb_m": -float(row["root_le_z_b_m"]),
                "quarter_chord_x_asb_m": -float(row["quarter_chord_x_b_m"]),
                "quarter_chord_y_asb_m": float(row["quarter_chord_y_b_m"]),
                "quarter_chord_z_asb_m": -float(row["quarter_chord_z_b_m"]),
                "conversion": "x_asb=-x_b, y_asb=y_b, z_asb=-z_b",
            }
        )
    return pd.DataFrame(rows)


def static_margin_proxy_table() -> pd.DataFrame:
    """Return simple CG, AC, and tail-volume diagnostics for audit context."""

    glider = build_nausicaa_glider()
    geometry = geometry_reference_table().set_index("surface_name")
    wing = geometry.loc["wing"]
    horizontal_tail = geometry.loc["horizontal_tail"]
    wing_ac_x_m = float(wing["quarter_chord_x_build_m"])
    htail_ac_x_m = float(horizontal_tail["quarter_chord_x_build_m"])
    cg_x_m = float(R_CG_BUILD_M[0])
    htail_arm_m = htail_ac_x_m - cg_x_m
    tail_volume = (
        float(horizontal_tail["area_m2"]) * htail_arm_m / (glider.s_ref_m2 * glider.c_ref_m)
    )
    rows = [
        {
            "quantity": "mass",
            "value": float(MASS_KG),
            "unit": "kg",
            "status": "current_mass_property_input",
        },
        {
            "quantity": "x_cg_from_wing_le",
            "value": cg_x_m,
            "unit": "m",
            "status": "current_mass_property_input",
        },
        {
            "quantity": "x_cg_over_mac",
            "value": cg_x_m / glider.c_ref_m,
            "unit": "-",
            "status": "current_mass_property_input",
        },
        {
            "quantity": "wing_quarter_chord_from_wing_le",
            "value": wing_ac_x_m,
            "unit": "m",
            "status": "current_geometry_input",
        },
        {
            "quantity": "cg_minus_wing_quarter_chord",
            "value": cg_x_m - wing_ac_x_m,
            "unit": "m",
            "status": "positive_aft_of_wing_quarter_chord",
        },
        {
            "quantity": "cg_minus_wing_quarter_chord_over_mac",
            "value": (cg_x_m - wing_ac_x_m) / glider.c_ref_m,
            "unit": "-",
            "status": "positive_aft_of_wing_quarter_chord",
        },
        {
            "quantity": "horizontal_tail_quarter_chord_from_wing_le",
            "value": htail_ac_x_m,
            "unit": "m",
            "status": "current_geometry_input",
        },
        {
            "quantity": "horizontal_tail_arm_from_cg",
            "value": htail_arm_m,
            "unit": "m",
            "status": "positive_aft_of_cg",
        },
        {
            "quantity": "horizontal_tail_volume_proxy",
            "value": tail_volume,
            "unit": "-",
            "status": "geometry_proxy_only",
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# 4) Output Writing and Plotting
# =============================================================================
def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _summary_value(summary: pd.DataFrame, surface_name: str, column: str) -> float | None:
    rows = summary.loc[summary["surface_name"] == surface_name, column]
    if rows.empty:
        return None
    return _finite_float(rows.iloc[0])


def _interpretation_label(summary: pd.DataFrame) -> str:
    total_cm_alpha = _summary_value(summary, "total", "cm_alpha_per_rad")
    total_cl_alpha = _summary_value(summary, "total", "cl_alpha_per_rad")
    if total_cm_alpha is None or total_cl_alpha is None:
        return "needs_review"
    if total_cl_alpha <= 0.0:
        return "needs_review"
    if abs(total_cm_alpha) > 0.5:
        return "pass_with_pitch_moment_review"
    return "pass"


def _plot_slope_bars(
    summary: pd.DataFrame,
    value_col: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    data = summary[summary["surface_name"] != "total"].copy()
    ax.bar(data["surface_name"], data[value_col], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.4, color="0.85")
    ax.tick_params(axis="x", rotation=15)
    _save_audit_figure(fig, path)
    plt.close(fig)


def _plot_geometry_side_view(geometry: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for _, row in geometry.iterrows():
        surface_name = str(row["surface_name"])
        ax.plot(
            [row["root_le_x_build_m"], row["trailing_edge_x_build_m"]],
            [row["root_le_z_build_m"], row["trailing_edge_z_build_m"]],
            marker="o",
            label=surface_name,
        )
        ax.plot(
            row["quarter_chord_x_build_m"],
            row["quarter_chord_z_build_m"],
            marker="x",
            color=ax.lines[-1].get_color(),
        )
    ax.plot(
        [R_CG_BUILD_M[0]],
        [R_CG_BUILD_M[2]],
        marker="s",
        color="black",
        label="CG",
    )
    ax.set_xlabel("x from wing LE (m, positive aft)")
    ax.set_ylabel("z from fuselage rod (m, positive down)")
    ax.grid(True, linewidth=0.4, color="0.85")
    ax.legend(fontsize=8)
    _save_audit_figure(fig, path)
    plt.close(fig)


def _write_figures(
    summary: pd.DataFrame,
    geometry: pd.DataFrame,
    output_root: Path,
) -> dict[str, Path]:
    figure_dir = output_root / "figures"
    figure_paths = {
        "cm_alpha_surface_breakdown": figure_dir / "cm_alpha_surface_breakdown.png",
        "cl_alpha_surface_breakdown": figure_dir / "cl_alpha_surface_breakdown.png",
        "geometry_side_view": figure_dir / "geometry_side_view.png",
    }
    _plot_slope_bars(
        summary,
        "cm_alpha_per_rad",
        r"$C_m/\alpha$ per rad",
        figure_paths["cm_alpha_surface_breakdown"],
    )
    _plot_slope_bars(
        summary,
        "cl_alpha_per_rad",
        r"$C_L/\alpha$ per rad",
        figure_paths["cl_alpha_surface_breakdown"],
    )
    _plot_geometry_side_view(geometry, figure_paths["geometry_side_view"])
    return figure_paths


def _write_manifest(
    output_root: Path,
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    output_files: dict[str, str],
) -> Path:
    total_cm_alpha = _summary_value(summary, "total", "cm_alpha_per_rad")
    wing_cm_alpha = _summary_value(summary, "wing", "cm_alpha_per_rad")
    htail_cm_alpha = _summary_value(summary, "horizontal_tail", "cm_alpha_per_rad")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_scope": COMPARISON_SCOPE,
        "high_incidence_validation_claim": HIGH_INCIDENCE_VALIDATION_CLAIM,
        "audit_scope": "low_alpha_longitudinal_moment_strip_breakdown_only",
        "aerosandbox_imported": False,
        "grid_rows": int(len(grid)),
        "alpha_deg": sorted(grid["alpha_deg"].unique().tolist()),
        "speed_m_s": sorted(grid["speed_m_s"].unique().tolist()),
        "elevator_deg": sorted(grid["delta_e_deg"].unique().tolist()),
        "interpretation_status": _interpretation_label(summary),
        "total_cm_alpha_per_rad": total_cm_alpha,
        "wing_cm_alpha_per_rad": wing_cm_alpha,
        "horizontal_tail_cm_alpha_per_rad": htail_cm_alpha,
        "output_files": output_files,
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return path


def _write_report(
    output_root: Path,
    summary: pd.DataFrame,
    static_margin: pd.DataFrame,
) -> Path:
    total_cm_alpha = _summary_value(summary, "total", "cm_alpha_per_rad")
    wing_cm_alpha = _summary_value(summary, "wing", "cm_alpha_per_rad")
    htail_cm_alpha = _summary_value(summary, "horizontal_tail", "cm_alpha_per_rad")
    total_cl_alpha = _summary_value(summary, "total", "cl_alpha_per_rad")
    x_cg_over_mac = static_margin.loc[
        static_margin["quantity"] == "x_cg_over_mac",
        "value",
    ].iloc[0]
    interpretation = _interpretation_label(summary)
    if interpretation == "pass":
        finding_lines = [
            "The corrected strip force point leaves a small positive total Cm-alpha.",
            "The wing positive contribution and horizontal-tail negative contribution",
            "mostly offset each other, so the audit no longer shows the earlier",
            "erroneous wing-dominated pitching moment.",
        ]
        next_action_lines = [
            "Keep this result as low-alpha attached-flow sanity evidence only.",
            "High-incidence controller claims still need the separate validation path.",
        ]
    else:
        finding_lines = [
            "The current low-alpha Cm-alpha still needs pitch-moment review.",
            "Use the surface breakdown to isolate whether wing, tail, or frame",
            "convention terms are driving the residual before controller claims resume.",
        ]
        next_action_lines = [
            "Review the wing pitching-moment reference, strip force application point,",
            "and body-axis sign convention before using this model for longitudinal",
            "controller claims.",
        ]
    lines = [
        "# Longitudinal Moment Audit Report",
        "",
        "This audit decomposes low-alpha local strip forces and moments by lifting surface.",
        "It does not tune the model and it does not validate high-incidence agile reversal.",
        "",
        "## Scope",
        "",
        f"- Comparison scope: `{COMPARISON_SCOPE}`",
        f"- High-incidence validation claim: `{HIGH_INCIDENCE_VALIDATION_CLAIM}`",
        "- AeroSandbox is not imported by this audit.",
        "- Forces and moments are aerodynamic strip loads only, grouped by surface code.",
        "",
        "## Status",
        "",
        f"- Interpretation status: `{interpretation}`",
        f"- Total CL-alpha per rad: `{total_cl_alpha:.6g}`",
        f"- Total Cm-alpha per rad: `{total_cm_alpha:.6g}`",
        f"- Wing Cm-alpha per rad: `{wing_cm_alpha:.6g}`",
        f"- Horizontal-tail Cm-alpha per rad: `{htail_cm_alpha:.6g}`",
        f"- x_CG/MAC from current mass properties: `{x_cg_over_mac:.6g}`",
        "",
        "## Finding",
        "",
        *finding_lines,
        "",
        "## Next Action",
        "",
        *next_action_lines,
    ]
    path = output_root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# =============================================================================
# 5) Public Workflow and CLI
# =============================================================================
def run_longitudinal_moment_audit(
    output_root: str | Path | None = None,
    alpha_min_deg: float = -8.0,
    alpha_max_deg: float = 12.0,
    alpha_step_deg: float = 2.0,
    speed_values_m_s: tuple[float, ...] = (4.5, 5.5, 6.5, 7.5, 8.5),
) -> dict[str, object]:
    """Run the longitudinal low-alpha strip-moment audit and write outputs."""

    output_path = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_path.mkdir(parents=True, exist_ok=True)
    alpha_values = _alpha_values_from_range(alpha_min_deg, alpha_max_deg, alpha_step_deg)
    grid = build_longitudinal_moment_grid(
        alpha_deg=alpha_values,
        speed_m_s=tuple(float(value) for value in speed_values_m_s),
    )
    breakdown = surface_force_moment_breakdown(grid)
    summary = surface_slope_summary(breakdown)
    geometry = geometry_reference_table()
    aerosandbox_geometry = aerosandbox_geometry_reference_table()
    static_margin = static_margin_proxy_table()

    breakdown_path = output_path / "surface_force_moment_breakdown.csv"
    summary_path = output_path / "surface_slope_summary.csv"
    geometry_path = output_path / "geometry_reference.csv"
    aerosandbox_geometry_path = output_path / "aerosandbox_geometry_reference.csv"
    static_margin_path = output_path / "static_margin_proxy.csv"
    breakdown.to_csv(breakdown_path, index=False)
    summary.to_csv(summary_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    aerosandbox_geometry.to_csv(aerosandbox_geometry_path, index=False)
    static_margin.to_csv(static_margin_path, index=False)
    figure_paths = _write_figures(summary, geometry, output_path)

    output_files = {
        "surface_force_moment_breakdown": str(breakdown_path),
        "surface_slope_summary": str(summary_path),
        "geometry_reference": str(geometry_path),
        "aerosandbox_geometry_reference": str(aerosandbox_geometry_path),
        "static_margin_proxy": str(static_margin_path),
        **{name: str(path) for name, path in figure_paths.items()},
    }
    manifest_path = _write_manifest(output_path, grid, summary, output_files)
    report_path = _write_report(output_path, summary, static_margin)

    return {
        "output_root": output_path,
        "surface_force_moment_breakdown_csv": breakdown_path,
        "surface_slope_summary_csv": summary_path,
        "geometry_reference_csv": geometry_path,
        "aerosandbox_geometry_reference_csv": aerosandbox_geometry_path,
        "static_margin_proxy_csv": static_margin_path,
        "manifest_json": manifest_path,
        "report_md": report_path,
        "interpretation_status": _interpretation_label(summary),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Nausicaa longitudinal low-alpha strip-moment audit."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha-min-deg", type=float, default=-8.0)
    parser.add_argument("--alpha-max-deg", type=float, default=12.0)
    parser.add_argument("--alpha-step-deg", type=float, default=2.0)
    parser.add_argument(
        "--speed-values-m-s",
        type=float,
        nargs="+",
        default=[4.5, 5.5, 6.5, 7.5, 8.5],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_longitudinal_moment_audit(
        output_root=args.output_root,
        alpha_min_deg=args.alpha_min_deg,
        alpha_max_deg=args.alpha_max_deg,
        alpha_step_deg=args.alpha_step_deg,
        speed_values_m_s=tuple(args.speed_values_m_s),
    )
    print(f"output_root={result['output_root']}")
    print(f"interpretation_status={result['interpretation_status']}")
    print(f"report_md={result['report_md']}")


if __name__ == "__main__":
    main()
