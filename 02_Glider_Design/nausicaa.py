from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any, Literal

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import pandas as pd

# =============================================================================
# User settings
# =============================================================================

PROJECT_ROOT = Path(".")
CACHE_DIR = Path("A_cache")
FIGURES_DIR = Path("B_figures")
RESULTS_DIR = Path("C_results")

GENERATE_POLARS = True
N_ALPHA = 25
MAKE_PLOTS = True
PLOT_DPI = 300

PRIMARY_AIRFOIL_NAME = "naca0002"
TRY_FLAT_PLATE_AIRFOIL = False

G = 9.81
RHO = 1.225

DESIGN_SPEED_MPS = 6.5
ALPHA_MIN_DEG = -4.0
ALPHA_MAX_DEG = 12.0
STALL_ALPHA_LIMIT_DEG = 14.0
MAX_CL_AT_DESIGN_POINT = 1.20

DELTA_A_MIN_DEG = -30.0
DELTA_A_MAX_DEG = 30.0
DELTA_E_MIN_DEG = -30.0
DELTA_E_MAX_DEG = 30.0
DELTA_R_MIN_DEG = -30.0
DELTA_R_MAX_DEG = 30.0

DIHEDRAL_DEG = 10.0

WING_SPAN_MIN_M = 0.65
WING_SPAN_MAX_M = 1.40
WING_CHORD_MIN_M = 0.08
WING_CHORD_MAX_M = 0.22

TAIL_ARM_MIN_M = 0.40
TAIL_ARM_MAX_M = 0.85
HT_SPAN_MIN_M = 0.18
HT_SPAN_MAX_M = 0.45
VT_HEIGHT_MIN_M = 0.08
VT_HEIGHT_MAX_M = 0.26

HT_AR = 4.0
VT_AR = 2.0

N_WING_XSECS = 11
N_TAIL_XSECS = 7

AILERON_ETA_INBOARD = 0.25
AILERON_ETA_OUTBOARD = 0.45
AILERON_CHORD_FRACTION = 0.28
ELEVATOR_CHORD_FRACTION = 0.30
RUDDER_CHORD_FRACTION = 0.35

WING_DENSITY_KG_M3 = 33.0
WING_THICKNESS_M = 0.004
TAIL_THICKNESS_M = 0.003
NOSE_X_M = -0.11
FUSE_RADIUS_M = 0.002
BOOM_LINEAR_DENSITY_KG_M = 0.009
GLUE_FRACTION = 0.08
BALLAST_MAX_KG = 0.025

STATIC_MARGIN_MIN = 0.05
STATIC_MARGIN_MAX = 0.10
VH_MIN = 0.50
VH_MAX = 0.70
VV_MIN = 0.03
VV_MAX = 0.05

MIN_L_OVER_D = 8.0
MIN_RE_WING = 20_000.0
MIN_WING_LOADING_N_M2 = 2.0
MAX_WING_LOADING_N_M2 = 20.0

CNB_MIN = 0.0
CLB_MAX = 0.0
CMQ_MAX = -0.01

MIN_ROLL_RATE_RAD_S = 0.6
MIN_ROLL_ACCEL_RAD_S2 = 2.0
MAX_ROLL_TAU_S = 0.45

SERVO_TORQUE_LIMIT_NM = 0.12
SERVO_SAFETY_FACTOR = 1.5
HINGE_MOMENT_COEFF = 4.0

MASS_WEIGHT_IN_OBJECTIVE = 0.20
BALLAST_WEIGHT_IN_OBJECTIVE = 0.40
CONTROL_TRIM_WEIGHT = 2e-4

SpanAxis = Literal["y", "z"]


# =============================================================================
# Utility helpers
# =============================================================================

def get_git_version() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "describe", "--always", "--dirty", "--tags"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def ensure_output_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def to_scalar(value: Any) -> Any:
    try:
        array = onp.asarray(value)
        if array.shape == ():
            return float(array)
        return float(array.flatten()[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


# =============================================================================
# Airfoil setup
# =============================================================================

def build_flat_plate_airfoil() -> asb.Airfoil:
    eps = 1e-4
    coordinates = onp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, eps],
            [0.0, eps],
            [0.0, 0.0],
        ]
    )
    airfoil = asb.Airfoil(name="flat_plate", coordinates=coordinates)

    def cl_function(alpha_deg: Any, re: Any = None, mach: Any = None) -> Any:
        _ = re
        _ = mach
        alpha_rad = np.radians(alpha_deg)
        return 2.0 * np.sin(alpha_rad) * np.cos(alpha_rad)

    def cd_function(alpha_deg: Any, re: Any = None, mach: Any = None) -> Any:
        _ = re
        _ = mach
        alpha_rad = np.radians(alpha_deg)
        return 2.0 * np.sin(alpha_rad) ** 2

    def cm_function(alpha_deg: Any, re: Any = None, mach: Any = None) -> Any:
        _ = alpha_deg
        _ = re
        _ = mach
        return 0.0

    airfoil.CL_function = cl_function
    airfoil.CD_function = cd_function
    airfoil.CM_function = cm_function
    return airfoil


def build_reference_airfoil() -> tuple[asb.Airfoil, str]:
    if TRY_FLAT_PLATE_AIRFOIL:
        try:
            flat_plate = build_flat_plate_airfoil()
            _ = flat_plate.CL_function(0.0)
            print("Using flat-plate airfoil model.", flush=True)
            return flat_plate, "flat_plate"
        except Exception as exc:
            print(
                f"[WARN] Flat-plate setup failed ({exc}); falling back to NACA0002.",
                flush=True,
            )

    airfoil = asb.Airfoil(name=PRIMARY_AIRFOIL_NAME)
    if GENERATE_POLARS:
        try:
            airfoil.generate_polars(
                cache_filename=str(CACHE_DIR / f"{PRIMARY_AIRFOIL_NAME}.json"),
                alphas=np.linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, N_ALPHA),
            )
        except Exception as exc:
            print(
                f"[WARN] Polar generation failed ({exc}); using default aerodynamic model.",
                flush=True,
            )

    return airfoil, PRIMARY_AIRFOIL_NAME

# =============================================================================
# Geometry builders
# =============================================================================

def build_main_wing(airfoil: asb.Airfoil, span_m: Any, chord_m: Any) -> asb.Wing:
    aileron_surface = asb.ControlSurface(
        name="aileron",
        symmetric=False,
        hinge_point=1.0 - AILERON_CHORD_FRACTION,
        trailing_edge=True,
    )

    xsecs = []
    for eta in onp.linspace(0.0, 1.0, N_WING_XSECS):
        y_le = eta * span_m / 2.0
        z_le = y_le * np.tan(np.radians(DIHEDRAL_DEG))
        controls = []
        if AILERON_ETA_INBOARD <= eta <= AILERON_ETA_OUTBOARD:
            controls = [aileron_surface]

        xsecs.append(
            asb.WingXSec(
                xyz_le=[0.0, y_le, z_le],
                chord=chord_m,
                twist=0.0,
                airfoil=airfoil,
                control_surfaces=controls,
            )
        )

    return asb.Wing(name="Main Wing", symmetric=True, xsecs=xsecs)


def build_horizontal_tail(
    airfoil: asb.Airfoil,
    tail_arm_m: Any,
    span_m: Any,
) -> tuple[asb.Wing, Any]:
    chord_m = span_m / HT_AR
    elevator_surface = asb.ControlSurface(
        name="elevator",
        symmetric=True,
        hinge_point=1.0 - ELEVATOR_CHORD_FRACTION,
        trailing_edge=True,
    )

    xsecs = []
    for eta in onp.linspace(0.0, 1.0, N_TAIL_XSECS):
        y_le = eta * span_m / 2.0
        xsecs.append(
            asb.WingXSec(
                xyz_le=[tail_arm_m, y_le, 0.0],
                chord=chord_m,
                twist=0.0,
                airfoil=airfoil,
                control_surfaces=[elevator_surface],
            )
        )

    htail = asb.Wing(name="Horizontal Tail", symmetric=True, xsecs=xsecs)
    return htail, chord_m


def build_vertical_tail(
    airfoil: asb.Airfoil,
    tail_arm_m: Any,
    height_m: Any,
) -> tuple[asb.Wing, Any]:
    chord_m = height_m / VT_AR
    rudder_surface = asb.ControlSurface(
        name="rudder",
        symmetric=True,
        hinge_point=1.0 - RUDDER_CHORD_FRACTION,
        trailing_edge=True,
    )

    xsecs = []
    for eta in onp.linspace(0.0, 1.0, N_TAIL_XSECS):
        z_le = eta * height_m
        xsecs.append(
            asb.WingXSec(
                xyz_le=[tail_arm_m, 0.0, z_le],
                chord=chord_m,
                twist=0.0,
                airfoil=airfoil,
                control_surfaces=[rudder_surface],
            )
        )

    vtail = asb.Wing(name="Vertical Tail", symmetric=False, xsecs=xsecs)
    return vtail, chord_m


def build_fuselage(tail_arm_m: Any, htail_chord_m: Any) -> asb.Fuselage:
    tail_x_m = tail_arm_m + 1.05 * htail_chord_m
    return asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[NOSE_X_M, 0.0, 0.0], radius=FUSE_RADIUS_M),
            asb.FuselageXSec(xyz_c=[tail_x_m, 0.0, 0.0], radius=FUSE_RADIUS_M),
        ],
    )


# =============================================================================
# Mass model and dynamics helpers
# =============================================================================

def surface_span(surface: asb.Wing, span_axis: SpanAxis) -> Any:
    coords = []
    for xsec in surface.xsecs:
        coord = xsec.xyz_le[1] if span_axis == "y" else xsec.xyz_le[2]
        coords.append(coord)

    coords = np.array(coords)
    half_span = np.max(np.abs(coords))

    if surface.symmetric and span_axis == "y":
        return 2.0 * half_span

    return np.max(coords) - np.min(coords)


def surface_mid_chord_xyz(surface: asb.Wing, span_axis: SpanAxis) -> tuple[Any, Any, Any]:
    root = surface.xsecs[0]
    tip = surface.xsecs[-1]

    x_root = root.xyz_le[0] + 0.5 * root.chord
    x_tip = tip.xyz_le[0] + 0.5 * tip.chord
    x_cg = 0.5 * (x_root + x_tip)

    if span_axis == "y" and surface.symmetric:
        y_cg = 0.0
    else:
        y_cg = 0.5 * (root.xyz_le[1] + tip.xyz_le[1])

    z_cg = 0.5 * (root.xyz_le[2] + tip.xyz_le[2])
    return x_cg, y_cg, z_cg


def flat_plate_mass_properties(
    surface: asb.Wing,
    density_kg_m3: float,
    thickness_m: float,
    span_axis: SpanAxis,
) -> asb.MassProperties:
    span_m = surface_span(surface, span_axis=span_axis)
    area_m2 = surface.area()
    chord_m = area_m2 / np.maximum(span_m, 1e-8)

    mass_kg = density_kg_m3 * area_m2 * thickness_m
    x_cg, y_cg, z_cg = surface_mid_chord_xyz(surface, span_axis=span_axis)

    b_dim = span_m
    c_dim = chord_m
    t_dim = thickness_m

    if span_axis == "y":
        i_xx = (mass_kg / 12.0) * (b_dim ** 2 + t_dim ** 2)
        i_yy = (mass_kg / 12.0) * (c_dim ** 2 + t_dim ** 2)
        i_zz = (mass_kg / 12.0) * (b_dim ** 2 + c_dim ** 2)
    else:
        i_xx = (mass_kg / 12.0) * (b_dim ** 2 + t_dim ** 2)
        i_yy = (mass_kg / 12.0) * (b_dim ** 2 + c_dim ** 2)
        i_zz = (mass_kg / 12.0) * (c_dim ** 2 + t_dim ** 2)

    return asb.MassProperties(
        mass=mass_kg,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg,
        Ixx=i_xx,
        Iyy=i_yy,
        Izz=i_zz,
        Ixy=0.0,
        Ixz=0.0,
        Iyz=0.0,
    )


def build_mass_model(
    opti: asb.Opti,
    wing: asb.Wing,
    htail: asb.Wing,
    vtail: asb.Wing,
    wing_chord_m: Any,
    tail_arm_m: Any,
) -> tuple[dict[str, asb.MassProperties], asb.MassProperties, Any]:
    mass_props: dict[str, asb.MassProperties] = {}

    mass_props["wing"] = flat_plate_mass_properties(
        surface=wing,
        density_kg_m3=WING_DENSITY_KG_M3,
        thickness_m=WING_THICKNESS_M,
        span_axis="y",
    )
    mass_props["htail_surfaces"] = flat_plate_mass_properties(
        surface=htail,
        density_kg_m3=WING_DENSITY_KG_M3,
        thickness_m=TAIL_THICKNESS_M,
        span_axis="y",
    )
    mass_props["vtail_surfaces"] = flat_plate_mass_properties(
        surface=vtail,
        density_kg_m3=WING_DENSITY_KG_M3,
        thickness_m=TAIL_THICKNESS_M,
        span_axis="z",
    )

    mass_props["linkages"] = asb.MassProperties(mass=0.001, x_cg=0.5 * tail_arm_m)
    mass_props["receiver"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.005,
        x_cg=NOSE_X_M + 0.010,
    )
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.013,
        x_cg=0.25 * wing_chord_m,
    )
    mass_props["servos"] = asb.mass_properties_from_radius_of_gyration(
        mass=4.0 * 0.0022,
        x_cg=0.30 * wing_chord_m,
    )

    boom_length_m = np.maximum(tail_arm_m - NOSE_X_M, 0.05)
    mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
        mass=BOOM_LINEAR_DENSITY_KG_M * boom_length_m,
        x_cg=0.5 * (NOSE_X_M + tail_arm_m),
    )
    mass_props["pod"] = asb.MassProperties(mass=0.007, x_cg=NOSE_X_M + 0.015)

    ballast_mass_kg = opti.variable(
        init_guess=0.0,
        lower_bound=0.0,
        upper_bound=BALLAST_MAX_KG,
    )
    mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
        mass=ballast_mass_kg,
        x_cg=0.30 * wing_chord_m,
    )

    subtotal = asb.MassProperties(mass=0.0)
    for component in mass_props.values():
        subtotal = subtotal + component

    mass_props["glue"] = subtotal * GLUE_FRACTION
    total_mass = subtotal + mass_props["glue"]

    return mass_props, total_mass, ballast_mass_kg


def aileron_effectiveness_proxy(
    aero: dict[str, Any],
    eta_inboard: float,
    eta_outboard: float,
    chord_fraction: float,
) -> Any:
    c_l_alpha = np.maximum(np.abs(aero["CLa"]), 1e-3)
    span_factor = np.maximum(eta_outboard ** 2 - eta_inboard ** 2, 1e-4)
    tau_aileron = 0.9 * chord_fraction
    cl_delta_a = c_l_alpha * tau_aileron * span_factor
    return np.clip(cl_delta_a, 1e-3, 2.0)


def estimate_servo_hinge_moment(
    q_dyn: Any,
    control_area_m2: Any,
    mean_chord_m: Any,
    deflection_deg: Any,
) -> Any:
    delta_rad = np.radians(np.abs(deflection_deg))
    moment_arm_m = 0.25 * mean_chord_m
    return q_dyn * control_area_m2 * HINGE_MOMENT_COEFF * delta_rad * moment_arm_m

# =============================================================================
# Reporting and export
# =============================================================================

def constraint_record(
    name: str,
    value: Any,
    lower: float | None = None,
    upper: float | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    value_f = to_scalar(value)
    passed = True

    if lower is not None and value_f < lower - tol:
        passed = False
    if upper is not None and value_f > upper + tol:
        passed = False

    return {
        "Constraint": name,
        "Value": value_f,
        "Lower": lower,
        "Upper": upper,
        "Tolerance": tol,
        "Pass": passed,
    }


def build_mass_rows(
    mass_props: dict[str, asb.MassProperties],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name, mp in mass_props.items():
        rows.append(
            {
                "Component": name,
                "Mass_kg": to_scalar(mp.mass),
                "Mass_g": to_scalar(mp.mass) * 1e3,
                "x_cg_m": to_scalar(mp.xyz_cg[0]),
                "y_cg_m": to_scalar(mp.xyz_cg[1]),
                "z_cg_m": to_scalar(mp.xyz_cg[2]),
                "Ixx_kgm2": to_scalar(mp.inertia_tensor[0, 0]),
                "Iyy_kgm2": to_scalar(mp.inertia_tensor[1, 1]),
                "Izz_kgm2": to_scalar(mp.inertia_tensor[2, 2]),
            }
        )

    rows.sort(key=lambda row: row["Mass_kg"], reverse=True)
    return rows


def build_aero_rows(aero: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(aero.keys()):
        rows.append({"Coefficient": key, "Value": to_scalar(aero[key])})
    return rows


def save_results(
    summary_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    mass_rows: list[dict[str, Any]],
    aero_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_df = pd.DataFrame(summary_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    mass_df = pd.DataFrame(mass_rows)
    aero_df = pd.DataFrame(aero_rows)
    constraints_df = pd.DataFrame(constraint_rows)

    csv_path = RESULTS_DIR / "nausicaa_results.csv"
    xlsx_path = RESULTS_DIR / "nausicaa_results.xlsx"

    summary_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        geometry_df.to_excel(writer, sheet_name="Geometry", index=False)
        mass_df.to_excel(writer, sheet_name="MassBreakdown", index=False)
        aero_df.to_excel(writer, sheet_name="Aerodynamics", index=False)
        constraints_df.to_excel(writer, sheet_name="Constraints", index=False)

    return {
        "results_csv": csv_path,
        "results_xlsx": xlsx_path,
    }


def make_plots(
    airplane: asb.Airplane,
    mass_props: dict[str, asb.MassProperties],
    total_mass: asb.MassProperties,
) -> dict[str, Path]:
    figure_outputs: dict[str, Path] = {}

    if not MAKE_PLOTS:
        return figure_outputs

    import aerosandbox.tools.pretty_plots as pretty
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.ioff()

    airplane.draw_three_view(show=False)
    three_view_path = FIGURES_DIR / "three_view.png"
    pretty.show_plot(tight_layout=False, show=False, savefig=str(three_view_path))
    figure_outputs["three_view"] = three_view_path

    fig, _ax = plt.subplots(
        figsize=(12, 5),
        subplot_kw={"aspect": "equal"},
        dpi=PLOT_DPI,
    )
    _ = fig

    mass_copy = copy.deepcopy(mass_props)
    if "ballast" in mass_copy and to_scalar(mass_copy["ballast"].mass) < 1e-6:
        mass_copy.pop("ballast")

    labels = [name.replace("_", " ").title() for name in mass_copy.keys()]
    values = [component.mass for component in mass_copy.values()]

    pretty.pie(
        values=values,
        names=labels,
        center_text=f"$\\bf{{Mass\\ Budget}}$\nTOGW: {to_scalar(total_mass.mass) * 1e3:.2f} g",
        label_format=(
            lambda name, value, percentage: f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%"
        ),
        startangle=110,
        arm_length=30,
        arm_radius=20,
        y_max_labels=1.1,
    )

    mass_plot_path = FIGURES_DIR / "mass_budget.png"
    pretty.show_plot(show=False, savefig=str(mass_plot_path))
    figure_outputs["mass_budget"] = mass_plot_path

    plt.close("all")
    return figure_outputs


def print_console_report(
    summary_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
    figure_paths: dict[str, Path],
) -> None:
    summary = {row["Metric"]: row["Value"] for row in summary_rows}
    geometry = {row["Parameter"]: row["Value"] for row in geometry_rows}

    def fmt(key: str, digits: int = 3) -> str:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return f"{value:.{digits}f}"
        return str(value)

    passed = sum(1 for row in constraint_rows if bool(row["Pass"]))
    total = len(constraint_rows)

    print("\n=== Nausicaa Baseline Design Report ===", flush=True)
    print("Design assumptions:", flush=True)
    print("  - Steady-flow reference condition (no updraft / no roll-in trajectory)", flush=True)
    print("  - Rectangular wing, fixed dihedral, shared tail arm", flush=True)
    print("  - Control limits set to +/-30 deg", flush=True)

    print("\nTrimmed flight point:", flush=True)
    print(
        f"  V = {DESIGN_SPEED_MPS:.2f} m/s | alpha = {fmt('alpha_trim_deg', 3)} deg",
        flush=True,
    )
    print(
        f"  delta_a = {fmt('delta_a_trim_deg', 3)} deg | "
        f"delta_e = {fmt('delta_e_trim_deg', 3)} deg | "
        f"delta_r = {fmt('delta_r_trim_deg', 3)} deg",
        flush=True,
    )

    print("\nPerformance:", flush=True)
    print(
        f"  Sink rate = {fmt('sink_rate_mps', 4)} m/s | "
        f"L/D = {fmt('L_over_D', 3)}",
        flush=True,
    )
    print(
        f"  Mass = {fmt('mass_total_kg', 4)} kg "
        f"({to_scalar(summary.get('mass_total_kg', 0.0)) * 1e3:.1f} g)",
        flush=True,
    )

    print("\nStability and control:", flush=True)
    print(
        f"  Static margin = {fmt('static_margin', 4)} | "
        f"Vh = {fmt('tail_volume_horizontal', 4)} | "
        f"Vv = {fmt('tail_volume_vertical', 4)}",
        flush=True,
    )
    print(
        f"  Roll rate ss = {fmt('roll_rate_ss_radps', 3)} rad/s | "
        f"Roll tau = {fmt('roll_tau_s', 3)} s",
        flush=True,
    )
    print(
        f"  Max servo utilization = {fmt('max_servo_utilization', 3)}",
        flush=True,
    )

    print("\nGeometry:", flush=True)
    print(
        f"  Wing span = {to_scalar(geometry.get('wing_span_m', 0.0)):.3f} m | "
        f"Wing chord = {to_scalar(geometry.get('wing_chord_m', 0.0)):.3f} m",
        flush=True,
    )
    print(
        f"  Tail arm = {to_scalar(geometry.get('tail_arm_m', 0.0)):.3f} m | "
        f"H-tail span = {to_scalar(geometry.get('htail_span_m', 0.0)):.3f} m | "
        f"V-tail height = {to_scalar(geometry.get('vtail_height_m', 0.0)):.3f} m",
        flush=True,
    )

    print(f"\nConstraint checks: {passed}/{total} passed", flush=True)

    print("\nSaved files:", flush=True)
    for key, path in output_paths.items():
        print(f"  {key}: {path}", flush=True)

    for key, path in figure_paths.items():
        print(f"  figure_{key}: {path}", flush=True)

# =============================================================================
# Main optimization workflow
# =============================================================================

def main() -> None:
    version = get_git_version()
    print(f"CODE_VERSION: {version}", flush=True)

    ensure_output_dirs()
    airfoil, airfoil_label = build_reference_airfoil()

    opti = asb.Opti()

    alpha_deg = opti.variable(
        init_guess=4.0,
        lower_bound=ALPHA_MIN_DEG,
        upper_bound=ALPHA_MAX_DEG,
    )

    delta_a_deg = opti.variable(
        init_guess=0.0,
        lower_bound=DELTA_A_MIN_DEG,
        upper_bound=DELTA_A_MAX_DEG,
    )
    delta_e_deg = opti.variable(
        init_guess=0.0,
        lower_bound=DELTA_E_MIN_DEG,
        upper_bound=DELTA_E_MAX_DEG,
    )
    delta_r_deg = opti.variable(
        init_guess=0.0,
        lower_bound=DELTA_R_MIN_DEG,
        upper_bound=DELTA_R_MAX_DEG,
    )

    wing_span_m = opti.variable(
        init_guess=1.00,
        lower_bound=WING_SPAN_MIN_M,
        upper_bound=WING_SPAN_MAX_M,
    )
    wing_chord_m = opti.variable(
        init_guess=0.14,
        lower_bound=WING_CHORD_MIN_M,
        upper_bound=WING_CHORD_MAX_M,
    )
    tail_arm_m = opti.variable(
        init_guess=0.62,
        lower_bound=TAIL_ARM_MIN_M,
        upper_bound=TAIL_ARM_MAX_M,
    )
    htail_span_m = opti.variable(
        init_guess=0.30,
        lower_bound=HT_SPAN_MIN_M,
        upper_bound=HT_SPAN_MAX_M,
    )
    vtail_height_m = opti.variable(
        init_guess=0.16,
        lower_bound=VT_HEIGHT_MIN_M,
        upper_bound=VT_HEIGHT_MAX_M,
    )

    wing = build_main_wing(airfoil=airfoil, span_m=wing_span_m, chord_m=wing_chord_m)
    htail, htail_chord_m = build_horizontal_tail(
        airfoil=airfoil,
        tail_arm_m=tail_arm_m,
        span_m=htail_span_m,
    )
    vtail, vtail_chord_m = build_vertical_tail(
        airfoil=airfoil,
        tail_arm_m=tail_arm_m,
        height_m=vtail_height_m,
    )
    fuselage = build_fuselage(tail_arm_m=tail_arm_m, htail_chord_m=htail_chord_m)

    airplane = asb.Airplane(
        name="Nausicaa",
        wings=[wing, htail, vtail],
        fuselages=[fuselage],
    ).with_control_deflections(
        {
            "aileron": delta_a_deg,
            "elevator": delta_e_deg,
            "rudder": delta_r_deg,
        }
    )

    op_point = asb.OperatingPoint(
        velocity=DESIGN_SPEED_MPS,
        alpha=alpha_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )

    mass_props, total_mass, ballast_mass_kg = build_mass_model(
        opti=opti,
        wing=wing,
        htail=htail,
        vtail=vtail,
        wing_chord_m=wing_chord_m,
        tail_arm_m=tail_arm_m,
    )

    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=total_mass.xyz_cg,
    ).run_with_stability_derivatives(
        alpha=True,
        beta=True,
        p=True,
        q=True,
        r=True,
    )

    wing_area_m2 = wing.area()
    wing_mac_m = wing.mean_aerodynamic_chord()
    htail_area_m2 = htail.area()
    vtail_area_m2 = vtail.area()

    wing_loading_n_m2 = total_mass.mass * G / np.maximum(wing_area_m2, 1e-8)
    reynolds_wing = op_point.reynolds(wing_mac_m)
    static_margin = (aero["x_np"] - total_mass.x_cg) / np.maximum(wing_mac_m, 1e-8)

    tail_volume_horizontal = htail_area_m2 * tail_arm_m / np.maximum(
        wing_area_m2 * wing_mac_m,
        1e-8,
    )
    tail_volume_vertical = vtail_area_m2 * tail_arm_m / np.maximum(
        wing_area_m2 * wing_span_m,
        1e-8,
    )

    l_over_d = aero["L"] / np.maximum(aero["D"], 1e-8)
    sink_rate_mps = aero["D"] * DESIGN_SPEED_MPS / np.maximum(total_mass.mass * G, 1e-8)

    cl_delta_a = aileron_effectiveness_proxy(
        aero=aero,
        eta_inboard=AILERON_ETA_INBOARD,
        eta_outboard=AILERON_ETA_OUTBOARD,
        chord_fraction=AILERON_CHORD_FRACTION,
    )

    q_dyn = 0.5 * RHO * DESIGN_SPEED_MPS ** 2
    i_xx = np.maximum(total_mass.inertia_tensor[0, 0], 1e-8)
    clp_mag = np.maximum(np.abs(aero["Clp"]), 1e-5)
    delta_a_max_rad = np.radians(DELTA_A_MAX_DEG)

    roll_accel0_rad_s2 = (
        q_dyn
        * wing_area_m2
        * wing_span_m
        * np.abs(cl_delta_a)
        * delta_a_max_rad
        / i_xx
    )

    roll_tau_s = (2.0 * i_xx * DESIGN_SPEED_MPS) / np.maximum(
        q_dyn * wing_area_m2 * wing_span_m ** 2 * clp_mag,
        1e-8,
    )

    roll_rate_ss_radps = (
        2.0
        * DESIGN_SPEED_MPS
        / np.maximum(wing_span_m, 1e-8)
        * np.abs(cl_delta_a)
        * delta_a_max_rad
        / clp_mag
    )

    aileron_area_m2 = (
        wing_area_m2
        * (AILERON_ETA_OUTBOARD - AILERON_ETA_INBOARD)
        * AILERON_CHORD_FRACTION
    )
    elevator_area_m2 = htail_area_m2 * ELEVATOR_CHORD_FRACTION
    rudder_area_m2 = vtail_area_m2 * RUDDER_CHORD_FRACTION

    aileron_chord_m = wing_chord_m * AILERON_CHORD_FRACTION
    elevator_chord_m = htail_chord_m * ELEVATOR_CHORD_FRACTION
    rudder_chord_m = vtail_chord_m * RUDDER_CHORD_FRACTION

    hinge_moment_aileron_nm = estimate_servo_hinge_moment(
        q_dyn=q_dyn,
        control_area_m2=0.5 * aileron_area_m2,
        mean_chord_m=aileron_chord_m,
        deflection_deg=DELTA_A_MAX_DEG,
    )
    hinge_moment_elevator_nm = estimate_servo_hinge_moment(
        q_dyn=q_dyn,
        control_area_m2=elevator_area_m2,
        mean_chord_m=elevator_chord_m,
        deflection_deg=np.abs(delta_e_deg),
    )
    hinge_moment_rudder_nm = estimate_servo_hinge_moment(
        q_dyn=q_dyn,
        control_area_m2=rudder_area_m2,
        mean_chord_m=rudder_chord_m,
        deflection_deg=np.abs(delta_r_deg),
    )

    servo_torque_available_nm = SERVO_TORQUE_LIMIT_NM / SERVO_SAFETY_FACTOR

    trim_effort = delta_e_deg ** 2 + 0.3 * delta_r_deg ** 2 + 0.15 * delta_a_deg ** 2

    objective = (
        sink_rate_mps
        + MASS_WEIGHT_IN_OBJECTIVE * total_mass.mass
        + BALLAST_WEIGHT_IN_OBJECTIVE * ballast_mass_kg
        + CONTROL_TRIM_WEIGHT * trim_effort
    )
    opti.minimize(objective)

    opti.subject_to(
        [
            aero["L"] >= total_mass.mass * G,
            aero["D"] >= 1e-3,
            aero["Cm"] == 0.0,
            aero["Cl"] == 0.0,
            aero["Cn"] == 0.0,
            aero["CL"] <= MAX_CL_AT_DESIGN_POINT,
            alpha_deg <= STALL_ALPHA_LIMIT_DEG,
            l_over_d >= MIN_L_OVER_D,
            opti.bounded(
                MIN_WING_LOADING_N_M2,
                wing_loading_n_m2,
                MAX_WING_LOADING_N_M2,
            ),
            reynolds_wing >= MIN_RE_WING,
            opti.bounded(STATIC_MARGIN_MIN, static_margin, STATIC_MARGIN_MAX),
            opti.bounded(VH_MIN, tail_volume_horizontal, VH_MAX),
            opti.bounded(VV_MIN, tail_volume_vertical, VV_MAX),
            aero["Clb"] <= CLB_MAX,
            aero["Cnb"] >= CNB_MIN,
            aero["Cmq"] <= CMQ_MAX,
            roll_rate_ss_radps >= MIN_ROLL_RATE_RAD_S,
            roll_accel0_rad_s2 >= MIN_ROLL_ACCEL_RAD_S2,
            roll_tau_s <= MAX_ROLL_TAU_S,
            hinge_moment_aileron_nm <= servo_torque_available_nm,
            hinge_moment_elevator_nm <= servo_torque_available_nm,
            hinge_moment_rudder_nm <= servo_torque_available_nm,
        ]
    )

    plugin_options = {"print_time": False, "verbose": False}
    ipopt_options = {
        "max_iter": 3000,
        "check_derivatives_for_naninf": "yes",
        "hessian_approximation": "limited-memory",
    }
    opti.solver("ipopt", plugin_options, ipopt_options)

    print("Starting optimization...", flush=True)
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        print(f"\n[SOLVE FAILED] {exc}", flush=True)
        print("No feasible design was found with the current settings.", flush=True)
        return

    airplane_num = solution(airplane)
    wing_num = copy.deepcopy(airplane_num.wings[0])
    htail_num = copy.deepcopy(airplane_num.wings[1])
    vtail_num = copy.deepcopy(airplane_num.wings[2])

    mass_props_num = solution(mass_props)
    total_mass_num = solution(total_mass)
    aero_num = solution(aero)

    objective_num = to_scalar(solution(objective))
    alpha_num = to_scalar(solution(alpha_deg))
    delta_a_num = to_scalar(solution(delta_a_deg))
    delta_e_num = to_scalar(solution(delta_e_deg))
    delta_r_num = to_scalar(solution(delta_r_deg))

    sink_rate_num = to_scalar(solution(sink_rate_mps))
    l_over_d_num = to_scalar(solution(l_over_d))
    mass_total_num = to_scalar(total_mass_num.mass)
    ballast_mass_num = to_scalar(solution(ballast_mass_kg))

    static_margin_num = to_scalar(solution(static_margin))
    tail_volume_h_num = to_scalar(solution(tail_volume_horizontal))
    tail_volume_v_num = to_scalar(solution(tail_volume_vertical))

    wing_loading_num = to_scalar(solution(wing_loading_n_m2))
    reynolds_num = to_scalar(solution(reynolds_wing))

    roll_rate_num = to_scalar(solution(roll_rate_ss_radps))
    roll_accel_num = to_scalar(solution(roll_accel0_rad_s2))
    roll_tau_num = to_scalar(solution(roll_tau_s))

    hinge_aileron_num = to_scalar(solution(hinge_moment_aileron_nm))
    hinge_elevator_num = to_scalar(solution(hinge_moment_elevator_nm))
    hinge_rudder_num = to_scalar(solution(hinge_moment_rudder_nm))

    max_servo_utilization = max(
        hinge_aileron_num,
        hinge_elevator_num,
        hinge_rudder_num,
    ) / servo_torque_available_nm

    wing_span_num = to_scalar(wing_num.span())
    wing_area_num = to_scalar(wing_num.area())
    wing_chord_num = wing_area_num / max(wing_span_num, 1e-8)

    htail_span_num = to_scalar(htail_num.span())
    htail_area_num = to_scalar(htail_num.area())
    htail_chord_num = htail_area_num / max(htail_span_num, 1e-8)

    vtail_height_num = to_scalar(surface_span(vtail_num, "z"))
    vtail_area_num = to_scalar(vtail_num.area())
    vtail_chord_num = vtail_area_num / max(vtail_height_num, 1e-8)

    tail_arm_num = to_scalar(solution(tail_arm_m))

    summary_rows = [
        {"Metric": "code_version", "Value": version, "Unit": "-"},
        {"Metric": "airfoil_model", "Value": airfoil_label, "Unit": "-"},
        {"Metric": "objective", "Value": objective_num, "Unit": "-"},
        {"Metric": "design_speed_mps", "Value": DESIGN_SPEED_MPS, "Unit": "m/s"},
        {"Metric": "alpha_trim_deg", "Value": alpha_num, "Unit": "deg"},
        {"Metric": "delta_a_trim_deg", "Value": delta_a_num, "Unit": "deg"},
        {"Metric": "delta_e_trim_deg", "Value": delta_e_num, "Unit": "deg"},
        {"Metric": "delta_r_trim_deg", "Value": delta_r_num, "Unit": "deg"},
        {"Metric": "mass_total_kg", "Value": mass_total_num, "Unit": "kg"},
        {"Metric": "mass_total_g", "Value": mass_total_num * 1e3, "Unit": "g"},
        {
            "Metric": "mass_total_lbm",
            "Value": mass_total_num / 0.45359237,
            "Unit": "lbm",
        },
        {"Metric": "ballast_mass_kg", "Value": ballast_mass_num, "Unit": "kg"},
        {"Metric": "sink_rate_mps", "Value": sink_rate_num, "Unit": "m/s"},
        {"Metric": "L_over_D", "Value": l_over_d_num, "Unit": "-"},
        {
            "Metric": "wing_loading_n_m2",
            "Value": wing_loading_num,
            "Unit": "N/m^2",
        },
        {"Metric": "reynolds_wing", "Value": reynolds_num, "Unit": "-"},
        {
            "Metric": "static_margin",
            "Value": static_margin_num,
            "Unit": "MAC fraction",
        },
        {
            "Metric": "tail_volume_horizontal",
            "Value": tail_volume_h_num,
            "Unit": "-",
        },
        {
            "Metric": "tail_volume_vertical",
            "Value": tail_volume_v_num,
            "Unit": "-",
        },
        {
            "Metric": "roll_rate_ss_radps",
            "Value": roll_rate_num,
            "Unit": "rad/s",
        },
        {
            "Metric": "roll_accel0_rad_s2",
            "Value": roll_accel_num,
            "Unit": "rad/s^2",
        },
        {"Metric": "roll_tau_s", "Value": roll_tau_num, "Unit": "s"},
        {
            "Metric": "max_servo_utilization",
            "Value": max_servo_utilization,
            "Unit": "fraction",
        },
    ]

    geometry_rows = [
        {"Parameter": "wing_span_m", "Value": wing_span_num, "Unit": "m"},
        {"Parameter": "wing_chord_m", "Value": wing_chord_num, "Unit": "m"},
        {"Parameter": "wing_area_m2", "Value": wing_area_num, "Unit": "m^2"},
        {
            "Parameter": "wing_aspect_ratio",
            "Value": wing_span_num ** 2 / max(wing_area_num, 1e-8),
            "Unit": "-",
        },
        {"Parameter": "tail_arm_m", "Value": tail_arm_num, "Unit": "m"},
        {"Parameter": "htail_span_m", "Value": htail_span_num, "Unit": "m"},
        {"Parameter": "htail_chord_m", "Value": htail_chord_num, "Unit": "m"},
        {"Parameter": "htail_area_m2", "Value": htail_area_num, "Unit": "m^2"},
        {"Parameter": "vtail_height_m", "Value": vtail_height_num, "Unit": "m"},
        {"Parameter": "vtail_chord_m", "Value": vtail_chord_num, "Unit": "m"},
        {"Parameter": "vtail_area_m2", "Value": vtail_area_num, "Unit": "m^2"},
        {"Parameter": "dihedral_deg", "Value": DIHEDRAL_DEG, "Unit": "deg"},
        {
            "Parameter": "aileron_semispan_start",
            "Value": AILERON_ETA_INBOARD,
            "Unit": "eta",
        },
        {
            "Parameter": "aileron_semispan_end",
            "Value": AILERON_ETA_OUTBOARD,
            "Unit": "eta",
        },
        {
            "Parameter": "aileron_chord_fraction",
            "Value": AILERON_CHORD_FRACTION,
            "Unit": "-",
        },
    ]

    mass_rows = build_mass_rows(mass_props_num)
    aero_rows = build_aero_rows(aero_num)

    constraint_rows = [
        constraint_record("Lift >= Weight", aero_num["L"], lower=mass_total_num * G),
        constraint_record("Drag >= 0", aero_num["D"], lower=1e-3),
        constraint_record("Trim Cm", aero_num["Cm"], lower=0.0, upper=0.0, tol=1e-3),
        constraint_record("Trim Cl", aero_num["Cl"], lower=0.0, upper=0.0, tol=1e-3),
        constraint_record("Trim Cn", aero_num["Cn"], lower=0.0, upper=0.0, tol=1e-3),
        constraint_record(
            "CL <= CLmax",
            aero_num["CL"],
            upper=MAX_CL_AT_DESIGN_POINT,
        ),
        constraint_record("Alpha <= stall margin", alpha_num, upper=STALL_ALPHA_LIMIT_DEG),
        constraint_record("L/D minimum", l_over_d_num, lower=MIN_L_OVER_D),
        constraint_record(
            "Wing loading minimum",
            wing_loading_num,
            lower=MIN_WING_LOADING_N_M2,
        ),
        constraint_record(
            "Wing loading maximum",
            wing_loading_num,
            upper=MAX_WING_LOADING_N_M2,
        ),
        constraint_record("Wing Reynolds", reynolds_num, lower=MIN_RE_WING),
        constraint_record("Static margin minimum", static_margin_num, lower=STATIC_MARGIN_MIN),
        constraint_record("Static margin maximum", static_margin_num, upper=STATIC_MARGIN_MAX),
        constraint_record("Vh minimum", tail_volume_h_num, lower=VH_MIN),
        constraint_record("Vh maximum", tail_volume_h_num, upper=VH_MAX),
        constraint_record("Vv minimum", tail_volume_v_num, lower=VV_MIN),
        constraint_record("Vv maximum", tail_volume_v_num, upper=VV_MAX),
        constraint_record("Clb <= 0", aero_num["Clb"], upper=CLB_MAX),
        constraint_record("Cnb >= 0", aero_num["Cnb"], lower=CNB_MIN),
        constraint_record("Cmq <= -0.01", aero_num["Cmq"], upper=CMQ_MAX),
        constraint_record("Roll rate minimum", roll_rate_num, lower=MIN_ROLL_RATE_RAD_S),
        constraint_record("Roll accel minimum", roll_accel_num, lower=MIN_ROLL_ACCEL_RAD_S2),
        constraint_record("Roll time constant", roll_tau_num, upper=MAX_ROLL_TAU_S),
        constraint_record(
            "Aileron servo torque",
            hinge_aileron_num,
            upper=servo_torque_available_nm,
        ),
        constraint_record(
            "Elevator servo torque",
            hinge_elevator_num,
            upper=servo_torque_available_nm,
        ),
        constraint_record(
            "Rudder servo torque",
            hinge_rudder_num,
            upper=servo_torque_available_nm,
        ),
    ]

    output_paths = save_results(
        summary_rows=summary_rows,
        geometry_rows=geometry_rows,
        mass_rows=mass_rows,
        aero_rows=aero_rows,
        constraint_rows=constraint_rows,
    )

    figure_paths: dict[str, Path] = {}
    try:
        figure_paths = make_plots(
            airplane=airplane_num,
            mass_props=mass_props_num,
            total_mass=total_mass_num,
        )
    except Exception as exc:
        print(f"[WARN] Plot generation failed: {exc}", flush=True)

    print_console_report(
        summary_rows=summary_rows,
        geometry_rows=geometry_rows,
        constraint_rows=constraint_rows,
        output_paths=output_paths,
        figure_paths=figure_paths,
    )


if __name__ == "__main__":
    main()
