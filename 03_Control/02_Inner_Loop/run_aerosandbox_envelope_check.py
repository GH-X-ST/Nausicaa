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

from flight_dynamics import adapt_glider, evaluate_state
from glider import Glider, LiftingSurface, build_nausicaa_glider

RESULTS_DIR = Path(__file__).resolve().parents[1] / "05_Results"
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))

try:
    from plot_style import PlotStyle, save_figure
except ImportError:  # pragma: no cover - fallback only when plot_style is removed.
    PlotStyle = None
    save_figure = None


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Shared constants and grid construction
# 2) Local model coefficient evaluation
# 3) AeroSandbox model construction and evaluation
# 4) Envelope comparison and summaries
# 5) Output writing and plotting
# 6) Public workflow and CLI
# =============================================================================


# =============================================================================
# 1) Shared Constants and Grid Construction
# =============================================================================
RHO_KG_M3 = 1.225
DEFAULT_OUTPUT_ROOT = Path(
    "03_Control/05_Results/00_model_audit/aerosandbox_envelope/001"
)
COMPARISON_SCOPE = "low_alpha_attached_flow_sanity_only"
HIGH_INCIDENCE_VALIDATION_CLAIM = "false"

LOCAL_OUTPUT_COLUMNS = (
    "case_id",
    "case_type",
    "alpha_deg",
    "beta_deg",
    "speed_m_s",
    "delta_a_deg",
    "delta_e_deg",
    "delta_r_deg",
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

AEROSANDBOX_OUTPUT_COLUMNS = (
    "case_id",
    "case_type",
    "alpha_deg",
    "beta_deg",
    "speed_m_s",
    "delta_a_deg",
    "delta_e_deg",
    "delta_r_deg",
    "cl_lift_asb",
    "cd_drag_asb",
    "cm_pitch_asb",
    "cy_asb",
    "cl_roll_asb",
    "cn_yaw_asb",
    "valid_aerosandbox",
    "aerosandbox_status",
)


def _default_tuple(values: tuple[float, ...] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in (default if values is None else values))


def _alpha_values_from_range(
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_step_deg: float,
) -> tuple[float, ...]:
    if alpha_step_deg <= 0.0:
        raise ValueError("alpha_step_deg must be positive.")
    count = int(round((alpha_max_deg - alpha_min_deg) / alpha_step_deg))
    values = [alpha_min_deg + idx * alpha_step_deg for idx in range(count + 1)]
    return tuple(float(round(value, 10)) for value in values if value <= alpha_max_deg + 1e-9)


def _control_cases(
    elevator_deg: tuple[float, ...],
    aileron_deg: tuple[float, ...],
    rudder_deg: tuple[float, ...],
) -> list[tuple[str, float, float, float]]:
    cases: list[tuple[str, float, float, float]] = [("clean", 0.0, 0.0, 0.0)]
    for value in aileron_deg:
        if not math.isclose(value, 0.0, abs_tol=1e-12):
            cases.append(("aileron", value, 0.0, 0.0))
    for value in elevator_deg:
        if not math.isclose(value, 0.0, abs_tol=1e-12):
            cases.append(("elevator", 0.0, value, 0.0))
    for value in rudder_deg:
        if not math.isclose(value, 0.0, abs_tol=1e-12):
            cases.append(("rudder", 0.0, 0.0, value))
    return cases


def build_verification_grid(
    alpha_deg: tuple[float, ...] | None = None,
    beta_deg: tuple[float, ...] | None = None,
    speed_m_s: tuple[float, ...] | None = None,
    elevator_deg: tuple[float, ...] | None = None,
    aileron_deg: tuple[float, ...] | None = None,
    rudder_deg: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Return a deterministic low-alpha envelope grid for model sanity checks."""

    alpha_values = _default_tuple(
        alpha_deg,
        (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    )
    beta_values = _default_tuple(beta_deg, (-6.0, 0.0, 6.0))
    speed_values = _default_tuple(speed_m_s, (4.5, 5.5, 6.5, 7.5, 8.5))
    control_cases = _control_cases(
        elevator_deg=_default_tuple(elevator_deg, (-10.0, 0.0, 10.0)),
        aileron_deg=_default_tuple(aileron_deg, (-10.0, 0.0, 10.0)),
        rudder_deg=_default_tuple(rudder_deg, (-10.0, 0.0, 10.0)),
    )

    rows: list[dict[str, float | str]] = []
    case_index = 0
    for case_type, delta_a_deg, delta_e_deg, delta_r_deg in control_cases:
        for speed_value_m_s in speed_values:
            for beta_value_deg in beta_values:
                for alpha_value_deg in alpha_values:
                    rows.append(
                        {
                            "case_id": f"case_{case_index:06d}",
                            "case_type": case_type,
                            "alpha_deg": float(alpha_value_deg),
                            "beta_deg": float(beta_value_deg),
                            "speed_m_s": float(speed_value_m_s),
                            "delta_a_deg": float(delta_a_deg),
                            "delta_e_deg": float(delta_e_deg),
                            "delta_r_deg": float(delta_r_deg),
                        }
                    )
                    case_index += 1
    return pd.DataFrame(rows)


# =============================================================================
# 2) Local Model Coefficient Evaluation
# =============================================================================
def _state_from_grid_row(row: pd.Series) -> np.ndarray:
    alpha_rad = np.deg2rad(float(row["alpha_deg"]))
    beta_rad = np.deg2rad(float(row["beta_deg"]))
    speed_m_s = float(row["speed_m_s"])

    # This velocity convention is locked to flight_dynamics.py:
    # alpha = atan2(w, u), beta = asin(v / V), body z positive down.
    state = np.zeros(15, dtype=float)
    state[6] = speed_m_s * np.cos(alpha_rad) * np.cos(beta_rad)
    state[7] = speed_m_s * np.sin(beta_rad)
    state[8] = speed_m_s * np.sin(alpha_rad) * np.cos(beta_rad)
    state[12] = np.deg2rad(float(row["delta_a_deg"]))
    state[13] = np.deg2rad(float(row["delta_e_deg"]))
    state[14] = np.deg2rad(float(row["delta_r_deg"]))
    return state


def _wind_axis_coefficients(
    force_b: np.ndarray,
    q_dyn_pa: float,
    s_ref_m2: float,
    alpha_deg: float,
    beta_deg: float,
) -> tuple[float, float]:
    coefficients_b = force_b / (q_dyn_pa * s_ref_m2)
    alpha_rad = np.deg2rad(float(alpha_deg))
    beta_rad = np.deg2rad(float(beta_deg))
    v_hat_b = np.array(
        [
            np.cos(alpha_rad) * np.cos(beta_rad),
            np.sin(beta_rad),
            np.sin(alpha_rad) * np.cos(beta_rad),
        ],
        dtype=float,
    )
    drag_dir_b = -v_hat_b

    # Lift is the body-up direction projected normal to the relative wind.
    # This preserves positive lift as upward force for beta=0 without inventing
    # a sideslip-specific stability-axis convention.
    body_up_b = np.array([0.0, 0.0, -1.0], dtype=float)
    lift_dir_b = body_up_b - np.dot(body_up_b, v_hat_b) * v_hat_b
    lift_norm = np.linalg.norm(lift_dir_b)
    if lift_norm <= 1e-12:
        return float("nan"), float(np.dot(coefficients_b, drag_dir_b))
    lift_dir_b = lift_dir_b / lift_norm
    return float(np.dot(coefficients_b, lift_dir_b)), float(np.dot(coefficients_b, drag_dir_b))


def _local_row_coefficients(
    row: pd.Series,
    glider: Glider,
    aircraft: object,
) -> dict[str, Any]:
    state = _state_from_grid_row(row)
    q_dyn_pa = 0.5 * RHO_KG_M3 * float(row["speed_m_s"]) ** 2
    try:
        loads = evaluate_state(
            state,
            np.zeros(3, dtype=float),
            aircraft,
            wind_model=None,
            rho=RHO_KG_M3,
            wind_mode="none",
        )
        force_b = np.asarray(loads["f_aero_b"], dtype=float).reshape(3)
        moment_b = np.asarray(loads["m_aero_b"], dtype=float).reshape(3)
        cx_body = force_b[0] / (q_dyn_pa * glider.s_ref_m2)
        cy_body = force_b[1] / (q_dyn_pa * glider.s_ref_m2)
        cz_body = force_b[2] / (q_dyn_pa * glider.s_ref_m2)
        cl_roll = moment_b[0] / (q_dyn_pa * glider.s_ref_m2 * glider.b_ref_m)
        cm_pitch = moment_b[1] / (q_dyn_pa * glider.s_ref_m2 * glider.c_ref_m)
        cn_yaw = moment_b[2] / (q_dyn_pa * glider.s_ref_m2 * glider.b_ref_m)
        cl_lift, cd_drag = _wind_axis_coefficients(
            force_b=force_b,
            q_dyn_pa=q_dyn_pa,
            s_ref_m2=glider.s_ref_m2,
            alpha_deg=float(row["alpha_deg"]),
            beta_deg=float(row["beta_deg"]),
        )
        values = [cx_body, cy_body, cz_body, cl_roll, cm_pitch, cn_yaw, cl_lift, cd_drag]
        valid = bool(np.all(np.isfinite(values)))
        status = "ok" if valid else "nonfinite_local_coefficients"
    except Exception as exc:  # pragma: no cover - exercised only on model failure.
        cx_body = cy_body = cz_body = np.nan
        cl_roll = cm_pitch = cn_yaw = np.nan
        cl_lift = cd_drag = np.nan
        valid = False
        status = f"local_evaluation_failed: {exc}"

    return {
        "case_id": row["case_id"],
        "case_type": row["case_type"],
        "alpha_deg": float(row["alpha_deg"]),
        "beta_deg": float(row["beta_deg"]),
        "speed_m_s": float(row["speed_m_s"]),
        "delta_a_deg": float(row["delta_a_deg"]),
        "delta_e_deg": float(row["delta_e_deg"]),
        "delta_r_deg": float(row["delta_r_deg"]),
        "cx_body": float(cx_body),
        "cy_body": float(cy_body),
        "cz_body": float(cz_body),
        "cl_roll": float(cl_roll),
        "cm_pitch": float(cm_pitch),
        "cn_yaw": float(cn_yaw),
        "cl_lift": float(cl_lift),
        "cd_drag": float(cd_drag),
        "valid_local": valid,
        "local_status": status,
    }


def local_model_coefficients(grid: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the current Nausicaa strip/panel model over the verification grid."""

    glider = build_nausicaa_glider()
    aircraft = adapt_glider(glider)
    rows = [_local_row_coefficients(row, glider, aircraft) for _, row in grid.iterrows()]
    return pd.DataFrame(rows, columns=LOCAL_OUTPUT_COLUMNS)


# =============================================================================
# 3) AeroSandbox Model Construction and Evaluation
# =============================================================================
def _aerosandbox_xyz_le_from_body(surface: LiftingSurface) -> list[float]:
    return [
        -float(surface.root_le_b[0]),
        float(surface.root_le_b[1]),
        -float(surface.root_le_b[2]),
    ]


def _control_surface_for_asb(asb: Any, surface: LiftingSurface) -> Any | None:
    control = surface.control_surface
    if control is None:
        return None
    return asb.ControlSurface(
        name=control.name,
        symmetric=control.name != "aileron",
        deflection=0.0,
        hinge_point=1.0 - float(control.chord_fraction),
        trailing_edge=True,
    )


def _horizontal_asb_wing(asb: Any, surface: LiftingSurface, airfoil: Any) -> Any:
    root = _aerosandbox_xyz_le_from_body(surface)
    half_span_m = 0.5 * surface.span_m
    dihedral_rad = np.deg2rad(surface.dihedral_deg)
    eta_values = [0.0, 1.0]
    control = surface.control_surface
    if control is not None:
        eta_values.extend([float(control.eta_start), float(control.eta_end)])
    eta_values = sorted(set(round(value, 8) for value in eta_values))
    asb_control = _control_surface_for_asb(asb, surface)
    xsecs = []
    for eta in eta_values:
        controls = []
        if (
            asb_control is not None
            and control is not None
            and control.eta_start <= eta <= control.eta_end
        ):
            controls = [asb_control]
        xsecs.append(
            asb.WingXSec(
                xyz_le=[
                    root[0],
                    root[1] + eta * half_span_m * np.cos(dihedral_rad),
                    root[2] + eta * half_span_m * np.sin(dihedral_rad),
                ],
                chord=float(surface.chord_m),
                twist=0.0,
                airfoil=airfoil,
                control_surfaces=controls,
            )
        )
    return asb.Wing(name=surface.name, symmetric=surface.symmetric, xsecs=xsecs)


def _vertical_asb_wing(asb: Any, surface: LiftingSurface, airfoil: Any) -> Any:
    root = _aerosandbox_xyz_le_from_body(surface)
    asb_control = _control_surface_for_asb(asb, surface)
    controls = [] if asb_control is None else [asb_control]
    xsecs = [
        asb.WingXSec(
            xyz_le=root,
            chord=float(surface.chord_m),
            twist=0.0,
            airfoil=airfoil,
            control_surfaces=controls,
        ),
        asb.WingXSec(
            xyz_le=[root[0], root[1], root[2] + float(surface.span_m)],
            chord=float(surface.chord_m),
            twist=0.0,
            airfoil=airfoil,
            control_surfaces=controls,
        ),
    ]
    return asb.Wing(name=surface.name, symmetric=False, xsecs=xsecs)


def _build_aerosandbox_airplane(asb: Any) -> tuple[Any, str]:
    glider = build_nausicaa_glider()
    airfoil = asb.Airfoil("naca0002")
    wings = []
    for surface in glider.surfaces:
        if surface.vertical:
            wings.append(_vertical_asb_wing(asb, surface, airfoil))
        else:
            wings.append(_horizontal_asb_wing(asb, surface, airfoil))
    airplane = asb.Airplane(
        name="Nausicaa 03_Control low-alpha audit",
        xyz_ref=[0.0, 0.0, 0.0],
        wings=wings,
        s_ref=float(glider.s_ref_m2),
        b_ref=float(glider.b_ref_m),
        c_ref=float(glider.c_ref_m),
    )
    return airplane, "AeroSandbox AeroBuildup.run"


def _asb_deflection_map(
    delta_a_deg: float,
    delta_e_deg: float,
    delta_r_deg: float,
) -> dict[str, float]:
    # AeroSandbox positive flap deflection increases local incidence. Its
    # mirrored aileron and elevator signs are opposite to the aggregate
    # Nausicaa command signs used by the 03_Control model.
    return {
        "aileron": -float(delta_a_deg),
        "elevator": -float(delta_e_deg),
        "rudder": float(delta_r_deg),
    }


def _result_array(result: dict[str, Any], key: str, count: int) -> np.ndarray:
    value = np.asarray(result[key], dtype=float).reshape(-1)
    if value.size == 1 and count > 1:
        value = np.full(count, float(value[0]))
    return value


def _aerosandbox_group_rows(
    group: pd.DataFrame,
    airplane: Any,
    asb: Any,
    status_prefix: str,
) -> list[dict[str, Any]]:
    count = len(group)
    op_point = asb.OperatingPoint(
        velocity=group["speed_m_s"].to_numpy(dtype=float),
        alpha=group["alpha_deg"].to_numpy(dtype=float),
        beta=group["beta_deg"].to_numpy(dtype=float),
        p=0.0,
        q=0.0,
        r=0.0,
    )
    result = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=[0.0, 0.0, 0.0],
    ).run()
    arrays = {
        "cl_lift_asb": _result_array(result, "CL", count),
        "cd_drag_asb": _result_array(result, "CD", count),
        "cm_pitch_asb": _result_array(result, "Cm", count),
        "cy_asb": _result_array(result, "CY", count),
        "cl_roll_asb": _result_array(result, "Cl", count),
        "cn_yaw_asb": _result_array(result, "Cn", count),
    }

    rows: list[dict[str, Any]] = []
    for idx, (_, source) in enumerate(group.iterrows()):
        values = [float(arrays[key][idx]) for key in arrays]
        valid = bool(np.all(np.isfinite(values)))
        row = {
            "case_id": source["case_id"],
            "case_type": source["case_type"],
            "alpha_deg": float(source["alpha_deg"]),
            "beta_deg": float(source["beta_deg"]),
            "speed_m_s": float(source["speed_m_s"]),
            "delta_a_deg": float(source["delta_a_deg"]),
            "delta_e_deg": float(source["delta_e_deg"]),
            "delta_r_deg": float(source["delta_r_deg"]),
            "valid_aerosandbox": valid,
            "aerosandbox_status": status_prefix if valid else "nonfinite_aerosandbox_coefficients",
        }
        row.update({key: float(arrays[key][idx]) for key in arrays})
        rows.append(row)
    return rows


def aerosandbox_coefficients(grid: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Evaluate an AeroSandbox low-alpha model over the verification grid when AeroSandbox is installed."""

    try:
        import aerosandbox as asb
    except ImportError:
        return None, "unavailable"

    try:
        airplane, analysis_method = _build_aerosandbox_airplane(asb)
    except Exception as exc:
        return None, f"installed_but_analysis_unavailable: {exc}"

    can_deflect_controls = hasattr(airplane, "with_control_deflections")
    rows: list[dict[str, Any]] = []
    for controls, group in grid.groupby(
        ["delta_a_deg", "delta_e_deg", "delta_r_deg"],
        sort=True,
    ):
        delta_a_deg, delta_e_deg, delta_r_deg = (float(value) for value in controls)
        if can_deflect_controls:
            airplane_eval = airplane.with_control_deflections(
                _asb_deflection_map(delta_a_deg, delta_e_deg, delta_r_deg)
            )
            status = f"ok: {analysis_method}"
        elif any(abs(value) > 1e-12 for value in controls):
            continue
        else:
            airplane_eval = airplane
            status = "ok_clean_only_control_deflection_api_unavailable"
        try:
            rows.extend(_aerosandbox_group_rows(group, airplane_eval, asb, status))
        except Exception as exc:
            for _, source in group.iterrows():
                rows.append(
                    {
                        "case_id": source["case_id"],
                        "case_type": source["case_type"],
                        "alpha_deg": float(source["alpha_deg"]),
                        "beta_deg": float(source["beta_deg"]),
                        "speed_m_s": float(source["speed_m_s"]),
                        "delta_a_deg": float(source["delta_a_deg"]),
                        "delta_e_deg": float(source["delta_e_deg"]),
                        "delta_r_deg": float(source["delta_r_deg"]),
                        "cl_lift_asb": np.nan,
                        "cd_drag_asb": np.nan,
                        "cm_pitch_asb": np.nan,
                        "cy_asb": np.nan,
                        "cl_roll_asb": np.nan,
                        "cn_yaw_asb": np.nan,
                        "valid_aerosandbox": False,
                        "aerosandbox_status": f"aerosandbox_case_failed: {exc}",
                    }
                )

    if not rows:
        return None, "installed_but_analysis_unavailable: no AeroSandbox rows evaluated"
    status = "available"
    if not can_deflect_controls:
        status = "available_clean_only_control_deflection_api_unavailable"
    return pd.DataFrame(rows, columns=AEROSANDBOX_OUTPUT_COLUMNS), status


# =============================================================================
# 4) Envelope Comparison and Summaries
# =============================================================================
def _with_radian_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("alpha", "beta", "delta_a", "delta_e", "delta_r"):
        out[f"{column}_rad"] = np.deg2rad(out[f"{column}_deg"])
    return out


def _near_zero(series: pd.Series) -> pd.Series:
    return series.abs() <= 1e-12


def estimate_slope(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    selector: pd.Series | None = None,
) -> float | None:
    """Return a least-squares slope or None if insufficient finite samples exist."""

    data = df if selector is None else df.loc[selector]
    x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data[y_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2 or np.unique(x[finite]).size < 2:
        return None
    a = np.column_stack([x[finite], np.ones(np.count_nonzero(finite))])
    slope, _intercept = np.linalg.lstsq(a, y[finite], rcond=None)[0]
    return float(slope)


def _clean_selector(df: pd.DataFrame) -> pd.Series:
    return (
        (df["case_type"] == "clean")
        & _near_zero(df["delta_a_deg"])
        & _near_zero(df["delta_e_deg"])
        & _near_zero(df["delta_r_deg"])
    )


def _central_selector(df: pd.DataFrame) -> pd.Series:
    return _near_zero(df["alpha_deg"]) & _near_zero(df["beta_deg"])


def _minimum_value_and_alpha(
    df: pd.DataFrame,
    value_col: str,
) -> tuple[float | None, float | None]:
    finite = df[np.isfinite(pd.to_numeric(df[value_col], errors="coerce"))]
    if finite.empty:
        return None, None
    idx = finite[value_col].astype(float).idxmin()
    return float(finite.loc[idx, value_col]), float(finite.loc[idx, "alpha_deg"])


def _percent_difference(local_value: float | None, asb_value: float | None) -> float | None:
    if local_value is None or asb_value is None or abs(local_value) <= 1e-12:
        return None
    return float(100.0 * (asb_value - local_value) / abs(local_value))


def _summary_for_dataframe(
    df: pd.DataFrame,
    prefix: str,
    cl_col: str,
    cd_col: str,
    cm_col: str,
    cy_col: str,
    cl_roll_col: str,
    cn_col: str,
) -> dict[str, float | None]:
    working = _with_radian_columns(df)
    clean = _clean_selector(working)
    alpha_sweep = clean & _near_zero(working["beta_deg"])
    beta_sweep = clean & _near_zero(working["alpha_deg"])
    central = _central_selector(working)
    elevator = central & _near_zero(working["delta_a_deg"]) & _near_zero(working["delta_r_deg"])
    aileron = central & _near_zero(working["delta_e_deg"]) & _near_zero(working["delta_r_deg"])
    rudder = central & _near_zero(working["delta_a_deg"]) & _near_zero(working["delta_e_deg"])
    cd_min, alpha_at_cd_min_deg = _minimum_value_and_alpha(working.loc[clean], cd_col)
    return {
        f"{prefix}_cl_alpha_per_rad": estimate_slope(working, "alpha_rad", cl_col, alpha_sweep),
        f"{prefix}_cm_alpha_per_rad": estimate_slope(working, "alpha_rad", cm_col, alpha_sweep),
        f"{prefix}_cd_min": cd_min,
        f"{prefix}_alpha_at_cd_min_deg": alpha_at_cd_min_deg,
        f"{prefix}_cy_beta_per_rad": estimate_slope(working, "beta_rad", cy_col, beta_sweep),
        f"{prefix}_cl_roll_beta_per_rad": estimate_slope(working, "beta_rad", cl_roll_col, beta_sweep),
        f"{prefix}_cn_beta_per_rad": estimate_slope(working, "beta_rad", cn_col, beta_sweep),
        f"{prefix}_cm_delta_e_per_rad": estimate_slope(working, "delta_e_rad", cm_col, elevator),
        f"{prefix}_cl_delta_a_per_rad": estimate_slope(working, "delta_a_rad", cl_roll_col, aileron),
        f"{prefix}_cn_delta_r_per_rad": estimate_slope(working, "delta_r_rad", cn_col, rudder),
    }


def compare_envelope(
    local: pd.DataFrame,
    aerosandbox: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return pointwise comparison rows and a derivative/trend summary."""

    if aerosandbox is None:
        comparison = local.copy()
        for column in AEROSANDBOX_OUTPUT_COLUMNS:
            if column not in comparison.columns:
                comparison[column] = np.nan
        comparison["comparison_status"] = "local_only_aerosandbox_unavailable"
        aerosandbox_status = "unavailable"
        sample_count_aerosandbox = 0
    else:
        comparison = local.merge(
            aerosandbox,
            on=[
                "case_id",
                "case_type",
                "alpha_deg",
                "beta_deg",
                "speed_m_s",
                "delta_a_deg",
                "delta_e_deg",
                "delta_r_deg",
            ],
            how="left",
        )
        comparison["comparison_status"] = np.where(
            comparison["valid_aerosandbox"].fillna(False),
            "matched",
            "local_only_missing_aerosandbox_case",
        )
        aerosandbox_status = "available"
        sample_count_aerosandbox = int(aerosandbox["valid_aerosandbox"].sum())

    difference_pairs = {
        "cl_lift": "cl_lift_asb",
        "cd_drag": "cd_drag_asb",
        "cm_pitch": "cm_pitch_asb",
        "cy_body": "cy_asb",
        "cl_roll": "cl_roll_asb",
        "cn_yaw": "cn_yaw_asb",
    }
    for local_col, asb_col in difference_pairs.items():
        comparison[f"{local_col}_minus_aerosandbox"] = comparison[local_col] - comparison[asb_col]

    local_summary = _summary_for_dataframe(
        local,
        prefix="local",
        cl_col="cl_lift",
        cd_col="cd_drag",
        cm_col="cm_pitch",
        cy_col="cy_body",
        cl_roll_col="cl_roll",
        cn_col="cn_yaw",
    )
    if aerosandbox is None:
        asb_summary: dict[str, float | None] = {
            "aerosandbox_cl_alpha_per_rad": None,
            "aerosandbox_cm_alpha_per_rad": None,
            "aerosandbox_cd_min": None,
            "aerosandbox_alpha_at_cd_min_deg": None,
            "aerosandbox_cy_beta_per_rad": None,
            "aerosandbox_cl_roll_beta_per_rad": None,
            "aerosandbox_cn_beta_per_rad": None,
            "aerosandbox_cm_delta_e_per_rad": None,
            "aerosandbox_cl_delta_a_per_rad": None,
            "aerosandbox_cn_delta_r_per_rad": None,
        }
    else:
        asb_summary = _summary_for_dataframe(
            aerosandbox,
            prefix="aerosandbox",
            cl_col="cl_lift_asb",
            cd_col="cd_drag_asb",
            cm_col="cm_pitch_asb",
            cy_col="cy_asb",
            cl_roll_col="cl_roll_asb",
            cn_col="cn_yaw_asb",
        )

    finite_cd = pd.to_numeric(local["cd_drag"], errors="coerce").dropna()
    summary: dict[str, object] = {
        "aerosandbox_status": aerosandbox_status,
        "sample_count_local": int(local["valid_local"].sum()),
        "sample_count_aerosandbox": sample_count_aerosandbox,
        **local_summary,
        **asb_summary,
        "cl_alpha_percent_difference": _percent_difference(
            local_summary["local_cl_alpha_per_rad"],
            asb_summary["aerosandbox_cl_alpha_per_rad"],
        ),
        "cm_alpha_percent_difference": _percent_difference(
            local_summary["local_cm_alpha_per_rad"],
            asb_summary["aerosandbox_cm_alpha_per_rad"],
        ),
        "local_cl_alpha_sign_ok": bool(
            local_summary["local_cl_alpha_per_rad"] is not None
            and local_summary["local_cl_alpha_per_rad"] > 0.0
        ),
        "local_cd_positive_ok": bool((finite_cd > 0.0).all()) if not finite_cd.empty else False,
        "local_elevator_sign_ok": bool(
            local_summary["local_cm_delta_e_per_rad"] is not None
            and local_summary["local_cm_delta_e_per_rad"] > 0.0
        ),
        "local_aileron_sign_ok": bool(
            local_summary["local_cl_delta_a_per_rad"] is not None
            and local_summary["local_cl_delta_a_per_rad"] > 0.0
        ),
        "local_rudder_sign_ok": bool(
            local_summary["local_cn_delta_r_per_rad"] is not None
            and local_summary["local_cn_delta_r_per_rad"] > 0.0
        ),
        "comparison_scope": COMPARISON_SCOPE,
        "high_incidence_validation_claim": HIGH_INCIDENCE_VALIDATION_CLAIM,
    }
    return comparison, summary


# =============================================================================
# 5) Output Writing and Plotting
# =============================================================================
def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _summary_value(summary: dict[str, object], key: str) -> str:
    value = summary.get(key)
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"


def _clean_beta_zero(df: pd.DataFrame) -> pd.DataFrame:
    data = df[_clean_selector(df) & _near_zero(df["beta_deg"])].copy()
    return data.sort_values(["speed_m_s", "alpha_deg"])


def _mean_by_alpha(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    clean = _clean_beta_zero(df)
    return clean.groupby("alpha_deg", as_index=False)[value_col].mean()


def _save_audit_figure(fig: Any, path: Path) -> None:
    if PlotStyle is not None and save_figure is not None:
        save_figure(fig, path, PlotStyle())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")


def _plot_xy(
    local: pd.DataFrame,
    aerosandbox: pd.DataFrame | None,
    y_local: str,
    y_asb: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    local_mean = _mean_by_alpha(local, y_local)
    ax.plot(local_mean["alpha_deg"], local_mean[y_local], marker="o", label="Local")
    if aerosandbox is not None:
        asb_mean = _mean_by_alpha(aerosandbox, y_asb)
        ax.plot(asb_mean["alpha_deg"], asb_mean[y_asb], marker="s", label="AeroSandbox")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, color="0.85")
    ax.legend()
    _save_audit_figure(fig, path)
    plt.close(fig)


def _plot_drag_polar(
    local: pd.DataFrame,
    aerosandbox: pd.DataFrame | None,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    local_clean = _clean_beta_zero(local)
    ax.plot(local_clean["cd_drag"], local_clean["cl_lift"], ".", label="Local")
    if aerosandbox is not None:
        asb_clean = _clean_beta_zero(aerosandbox)
        ax.plot(asb_clean["cd_drag_asb"], asb_clean["cl_lift_asb"], ".", label="AeroSandbox")
    ax.set_xlabel(r"$C_D$")
    ax.set_ylabel(r"$C_L$")
    ax.grid(True, linewidth=0.4, color="0.85")
    ax.legend()
    _save_audit_figure(fig, path)
    plt.close(fig)


def _plot_cl_alpha_by_speed(local: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    clean = _clean_beta_zero(local)
    for speed_m_s, group in clean.groupby("speed_m_s"):
        ax.plot(group["alpha_deg"], group["cl_lift"], marker="o", label=f"{speed_m_s:g} m/s")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_L$")
    ax.grid(True, linewidth=0.4, color="0.85")
    ax.legend(fontsize=8)
    _save_audit_figure(fig, path)
    plt.close(fig)


def _plot_control_derivatives(summary: dict[str, object], path: Path) -> None:
    labels = [r"$C_m/\delta_e$", r"$C_l/\delta_a$", r"$C_n/\delta_r$"]
    keys = [
        "local_cm_delta_e_per_rad",
        "local_cl_delta_a_per_rad",
        "local_cn_delta_r_per_rad",
    ]
    values = [_finite_float(summary.get(key)) or 0.0 for key in keys]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Local derivative per rad")
    ax.grid(True, axis="y", linewidth=0.4, color="0.85")
    _save_audit_figure(fig, path)
    plt.close(fig)


def _write_figures(
    local: pd.DataFrame,
    aerosandbox: pd.DataFrame | None,
    summary: dict[str, object],
    output_root: Path,
) -> dict[str, Path]:
    figure_dir = output_root / "figures"
    figure_paths = {
        "cl_vs_alpha": figure_dir / "cl_vs_alpha.png",
        "cd_vs_alpha": figure_dir / "cd_vs_alpha.png",
        "cm_vs_alpha": figure_dir / "cm_vs_alpha.png",
        "drag_polar": figure_dir / "drag_polar.png",
        "cl_alpha_by_speed": figure_dir / "cl_alpha_by_speed.png",
        "local_control_derivatives": figure_dir / "local_control_derivatives.png",
    }
    _plot_xy(local, aerosandbox, "cl_lift", "cl_lift_asb", r"$C_L$", figure_paths["cl_vs_alpha"])
    _plot_xy(local, aerosandbox, "cd_drag", "cd_drag_asb", r"$C_D$", figure_paths["cd_vs_alpha"])
    _plot_xy(local, aerosandbox, "cm_pitch", "cm_pitch_asb", r"$C_m$", figure_paths["cm_vs_alpha"])
    _plot_drag_polar(local, aerosandbox, figure_paths["drag_polar"])
    _plot_cl_alpha_by_speed(local, figure_paths["cl_alpha_by_speed"])
    _plot_control_derivatives(summary, figure_paths["local_control_derivatives"])
    return figure_paths


def _write_manifest(
    output_root: Path,
    grid: pd.DataFrame,
    aerosandbox: pd.DataFrame | None,
    aerosandbox_status: str,
    summary: dict[str, object],
    output_files: dict[str, str],
) -> Path:
    try:
        import aerosandbox as asb

        aerosandbox_version = getattr(asb, "__version__", "unknown")
    except ImportError:
        aerosandbox_version = None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_scope": COMPARISON_SCOPE,
        "high_incidence_validation_claim": HIGH_INCIDENCE_VALIDATION_CLAIM,
        "aerosandbox_status": aerosandbox_status,
        "aerosandbox_version": aerosandbox_version,
        "aerosandbox_analysis_method": "AeroBuildup.run" if aerosandbox is not None else None,
        "grid_rows": int(len(grid)),
        "alpha_deg": sorted(grid["alpha_deg"].unique().tolist()),
        "beta_deg": sorted(grid["beta_deg"].unique().tolist()),
        "speed_m_s": sorted(grid["speed_m_s"].unique().tolist()),
        "case_types": sorted(grid["case_type"].unique().tolist()),
        "aerosandbox_control_cases_evaluated": bool(
            aerosandbox is not None and (aerosandbox["case_type"] != "clean").any()
        ),
        "summary": summary,
        "output_files": output_files,
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return path


def _write_report(
    output_root: Path,
    aerosandbox_status: str,
    summary: dict[str, object],
) -> Path:
    lines = [
        "# AeroSandbox Envelope Check Report",
        "",
        "This is a low-alpha attached-flow sanity comparison only.",
        "It is not proof of high-incidence agile-reversal fidelity.",
        "Any mismatch is logged, not hidden.",
        "High-incidence validation must later use OCP replay and real Vicon flight logs.",
        "",
        "## Scope",
        "",
        f"- Comparison scope: `{COMPARISON_SCOPE}`",
        f"- High-incidence validation claim: `{HIGH_INCIDENCE_VALIDATION_CLAIM}`",
        "- The sweep covers alpha, beta, speed, and one-axis local-model control perturbations.",
        "- Wind is zero for every case.",
        "",
        "## Status",
        "",
        f"- AeroSandbox status: `{aerosandbox_status}`",
        f"- Local samples: `{summary['sample_count_local']}`",
        f"- AeroSandbox samples: `{summary['sample_count_aerosandbox']}`",
        "",
        "## Main Local Trends",
        "",
        f"- Local CL-alpha per rad: `{_summary_value(summary, 'local_cl_alpha_per_rad')}`",
        f"- Local Cm-alpha per rad: `{_summary_value(summary, 'local_cm_alpha_per_rad')}`",
        f"- Local CD min: `{_summary_value(summary, 'local_cd_min')}`",
        f"- Local alpha at CD min: `{_summary_value(summary, 'local_alpha_at_cd_min_deg')}` deg",
        f"- Elevator sign OK: `{summary['local_elevator_sign_ok']}`",
        f"- Aileron sign OK: `{summary['local_aileron_sign_ok']}`",
        f"- Rudder sign OK: `{summary['local_rudder_sign_ok']}`",
        "",
        "## AeroSandbox Trends",
        "",
        f"- AeroSandbox CL-alpha per rad: `{_summary_value(summary, 'aerosandbox_cl_alpha_per_rad')}`",
        f"- AeroSandbox Cm-alpha per rad: `{_summary_value(summary, 'aerosandbox_cm_alpha_per_rad')}`",
        f"- CL-alpha percent difference: `{_summary_value(summary, 'cl_alpha_percent_difference')}`",
        f"- Cm-alpha percent difference: `{_summary_value(summary, 'cm_alpha_percent_difference')}`",
    ]
    path = output_root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# =============================================================================
# 6) Public Workflow and CLI
# =============================================================================
def run_aerosandbox_envelope_check(
    output_root: str | Path | None = None,
    alpha_min_deg: float = -8.0,
    alpha_max_deg: float = 12.0,
    alpha_step_deg: float = 2.0,
    beta_values_deg: tuple[float, ...] = (-6.0, 0.0, 6.0),
    speed_values_m_s: tuple[float, ...] = (4.5, 5.5, 6.5, 7.5, 8.5),
) -> dict[str, Path | str | bool]:
    """Run the low-alpha envelope sanity-check workflow and write CSV/PNG/Markdown outputs."""

    output_path = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_path.mkdir(parents=True, exist_ok=True)
    alpha_values = _alpha_values_from_range(alpha_min_deg, alpha_max_deg, alpha_step_deg)
    grid = build_verification_grid(
        alpha_deg=alpha_values,
        beta_deg=tuple(float(value) for value in beta_values_deg),
        speed_m_s=tuple(float(value) for value in speed_values_m_s),
    )
    local = local_model_coefficients(grid)
    aerosandbox, aerosandbox_status = aerosandbox_coefficients(grid)
    comparison, summary = compare_envelope(local, aerosandbox)
    summary["aerosandbox_status"] = aerosandbox_status

    local_path = output_path / "local_envelope_coefficients.csv"
    asb_path = output_path / "aerosandbox_envelope_coefficients.csv"
    comparison_path = output_path / "pointwise_comparison.csv"
    summary_path = output_path / "comparison_summary.csv"
    local.to_csv(local_path, index=False)
    if aerosandbox is not None:
        aerosandbox.to_csv(asb_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    figure_paths = _write_figures(local, aerosandbox, summary, output_path)

    output_files = {
        "local_envelope_coefficients": str(local_path),
        "pointwise_comparison": str(comparison_path),
        "comparison_summary": str(summary_path),
        **{name: str(path) for name, path in figure_paths.items()},
    }
    if aerosandbox is not None:
        output_files["aerosandbox_envelope_coefficients"] = str(asb_path)
    manifest_path = _write_manifest(
        output_root=output_path,
        grid=grid,
        aerosandbox=aerosandbox,
        aerosandbox_status=aerosandbox_status,
        summary=summary,
        output_files=output_files,
    )
    report_path = _write_report(output_path, aerosandbox_status, summary)

    return {
        "output_root": output_path,
        "local_csv": local_path,
        "aerosandbox_csv": asb_path if aerosandbox is not None else "",
        "pointwise_comparison_csv": comparison_path,
        "comparison_summary_csv": summary_path,
        "manifest_json": manifest_path,
        "report_md": report_path,
        "aerosandbox_available": aerosandbox is not None,
        "aerosandbox_status": aerosandbox_status,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Nausicaa low-alpha AeroSandbox envelope sanity check."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha-min-deg", type=float, default=-8.0)
    parser.add_argument("--alpha-max-deg", type=float, default=12.0)
    parser.add_argument("--alpha-step-deg", type=float, default=2.0)
    parser.add_argument("--speeds", type=float, nargs="+", default=[4.5, 5.5, 6.5, 7.5, 8.5])
    parser.add_argument("--betas", type=float, nargs="+", default=[-6.0, 0.0, 6.0])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_aerosandbox_envelope_check(
        output_root=args.output_root,
        alpha_min_deg=args.alpha_min_deg,
        alpha_max_deg=args.alpha_max_deg,
        alpha_step_deg=args.alpha_step_deg,
        beta_values_deg=tuple(args.betas),
        speed_values_m_s=tuple(args.speeds),
    )
    print(f"output_root={result['output_root']}")
    print(f"aerosandbox_status={result['aerosandbox_status']}")
    print(f"report_md={result['report_md']}")


if __name__ == "__main__":
    main()
