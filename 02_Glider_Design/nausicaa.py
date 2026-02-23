from __future__ import annotations

import argparse
import copy
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

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
PLOT_DPI = 1000
RUN_WORKFLOW = False

PRIMARY_AIRFOIL_NAME = "naca0002"

# Physical constants
G = 9.81
RHO = 1.225

# Arena geometry (meters)
ARENA_LENGTH_M = 8.4
ARENA_WIDTH_M = 4.8
ARENA_HEIGHT_M = 3.5

# Two-speed design points
V_TURN_MPS = 3.0
V_NOM_MPS = 3.5

# Manoeuvre definition (coordinated, level turn)
TURN_BANK_DEG = 50.0
WALL_CLEARANCE_M = 0.50
TURN_DEFLECTION_UTIL_MAX = 0.80

# Stall / margin settings for manoeuvre case
TURN_ALPHA_MARGIN_DEG = 2.0
TURN_CL_CAP = 0.90

# Trim operating-point envelope
DESIGN_SPEED_MPS = V_NOM_MPS
ALPHA_MIN_DEG = -4.0
ALPHA_MAX_DEG = 12.0
STALL_ALPHA_LIMIT_DEG = 14.0
MAX_CL_AT_DESIGN_POINT = 1.20

# Control-surface deflection limits
DELTA_A_MIN_DEG = -30.0
DELTA_A_MAX_DEG = 30.0
DELTA_E_MIN_DEG = -30.0
DELTA_E_MAX_DEG = 30.0
DELTA_R_MIN_DEG = -30.0
DELTA_R_MAX_DEG = 30.0

# Fixed geometry assumptions
DIHEDRAL_DEG = 10.0

# Wing design bounds
WING_SPAN_MIN_M = 0.30
WING_SPAN_MAX_M = 2.00
WING_CHORD_MIN_M = 0.08
WING_CHORD_MAX_M = 0.22

# Tail design bounds
TAIL_ARM_MIN_M = 0.40
TAIL_ARM_MAX_M = 0.85
HT_SPAN_MIN_M = 0.20
HT_SPAN_MAX_M = 0.45
VT_HEIGHT_MIN_M = 0.10
VT_HEIGHT_MAX_M = 0.30

# Tail aspect-ratio assumptions
HT_AR = 4.0
VT_AR = 2.0

# Mesh density used for AeroSandbox lifting surfaces
N_WING_XSECS = 11
N_TAIL_XSECS = 7

# Control-surface geometry
AILERON_ETA_INBOARD = 0.30
AILERON_ETA_OUTBOARD = 0.70
AILERON_CHORD_FRACTION = 0.30
ELEVATOR_CHORD_FRACTION = 0.30
RUDDER_CHORD_FRACTION = 0.35

# Mass model assumptions
WING_DENSITY_KG_M3 = 33.0
WING_THICKNESS_M = 0.006
TAIL_THICKNESS_M = 0.003
NOSE_X_M = -0.11
FUSE_RADIUS_M = 0.002
BOOM_LINEAR_DENSITY_KG_M = 0.009
GLUE_FRACTION = 0.08
BALLAST_MAX_KG = 0.025

# Avionics / hardware masses (kg)
BATTERY_MASS_KG = 0.0090
RECEIVER_MASS_KG = 0.0050
FLIGHT_CONTROLLER_MASS_KG = 0.0050
REGULATOR_MASS_KG = 0.0004
SERVO_MASS_KG = 0.0022

# Servo layout (4 total): 2 aileron + elevator + rudder
AILERON_SERVO_SPAN_ETA = 0.45
TAIL_SERVO_X_FRAC = 0.30

# Battery sliding range
BATTERY_X_MAX_FRAC = 0.60
BATTERY_X_MIN_M = NOSE_X_M + 0.015

# Optional carbon spar as line-mass
WING_SPAR_LINEAR_DENSITY_KG_M = 0.001

# Static-stability design window
STATIC_MARGIN_MIN = 0.05
STATIC_MARGIN_MAX = 0.10
VH_MIN = 0.50
VH_MAX = 0.70
VV_MIN = 0.03
VV_MAX = 0.05

# Aerodynamic and loading constraints
MIN_L_OVER_D = 8.0
MIN_RE_WING = 20_000.0
MIN_WING_LOADING_N_M2 = 2.0
MAX_WING_LOADING_N_M2 = 20.0

# Lateral-directional stability derivative limits
CNB_MIN = 0.0
CLB_MAX = 0.0
CMQ_MAX = -0.01

# Roll-performance targets
MIN_ROLL_RATE_RAD_S = 0.6
MIN_ROLL_ACCEL_RAD_S2 = 2.0
MAX_ROLL_TAU_S = 0.45

# Servo sizing assumptions
# Update from servo datasheet (N*m)
SERVO_TORQUE_LIMIT_NM = 0.18
# Derate for shocks, backlash, and installation losses
SERVO_SAFETY_FACTOR = 2.0
# Rough hinge-moment proxy that should be calibrated
HINGE_MOMENT_COEFF = 0.02
LINKAGE_EFFICIENCY = 0.80
SERVO_ARM_M = 0.004
CONTROL_HORN_ARM_M = 0.008

# Objective-function weights
MASS_WEIGHT_IN_OBJECTIVE = 0.20
BALLAST_WEIGHT_IN_OBJECTIVE = 0.40
CONTROL_TRIM_WEIGHT = 2e-4
BOUNDARY_HIT_REL_TOL = 1e-3
BOUNDARY_HIT_ABS_TOL = 1e-6

SpanAxis = Literal["y", "z"]

# Float during post-processing, symbolic expression during optimization
Scalar: TypeAlias = Any
AeroMap: TypeAlias = dict[str, Scalar]
MassPropertiesMap: TypeAlias = dict[str, asb.MassProperties]
ReportValue: TypeAlias = str | float | int | bool | None
ReportRow: TypeAlias = dict[str, ReportValue]
ReportRows: TypeAlias = list[ReportRow]
PathMap: TypeAlias = dict[str, Path]


@dataclass(frozen=True)
class WorkflowConfig:
    n_starts: int = 20
    keep_top_k: int = 5
    random_seed: int = 1
    n_scenarios: int = 50
    scenario_seed: int = 2
    dedup_span_m: float = 0.01
    dedup_chord_m: float = 0.005
    dedup_tail_arm_m: float = 0.01
    mass_scale_min: float = 0.90
    mass_scale_max: float = 1.10
    cg_x_shift_mac_min: float = -0.06
    cg_x_shift_mac_max: float = 0.06
    incidence_bias_deg_min: float = -2.0
    incidence_bias_deg_max: float = 2.0
    control_eff_min: float = 0.70
    control_eff_max: float = 1.00
    drag_factor_min: float = 1.00
    drag_factor_max: float = 1.25
    stall_alpha_margin_deg: float = 2.0
    cl_margin: float = 0.15
    max_trim_util_fraction: float = 0.80


@dataclass
class Candidate:
    candidate_id: int
    objective: float
    wing_span_m: float
    wing_chord_m: float
    tail_arm_m: float
    htail_span_m: float
    vtail_height_m: float
    alpha_deg: float
    delta_a_deg: float
    delta_e_deg: float
    delta_r_deg: float
    sink_rate_mps: float
    l_over_d: float
    mass_total_kg: float
    ballast_mass_kg: float
    static_margin: float
    vh: float
    vv: float
    roll_tau_s: float
    roll_rate_ss_radps: float
    roll_accel0_rad_s2: float
    max_servo_util: float
    airplane: asb.Airplane | None = None
    total_mass: asb.MassProperties | None = None
    mass_props: MassPropertiesMap | None = None
    aero: dict[str, float] | None = None
    summary_rows: ReportRows | None = None
    geometry_rows: ReportRows | None = None
    mass_rows: ReportRows | None = None
    aero_rows: ReportRows | None = None
    constraint_rows: ReportRows | None = None
    boundary_rows: ReportRows | None = None
    design_points_rows: ReportRows | None = None
    wing_area_m2: float = float("nan")
    wing_mac_m: float = float("nan")
    airfoil_label: str = ""


def default_initial_guess() -> dict[str, float]:
    return {
        "wing_span_m": 1.00,
        "wing_chord_m": 0.14,
        "tail_arm_m": 0.62,
        "htail_span_m": 0.30,
        "vtail_height_m": 0.16,
        "alpha_deg": 4.0,
        "delta_a_deg": 0.0,
        "delta_e_deg": 0.0,
        "delta_r_deg": 0.0,
    }


_AIRFOIL_CACHE: tuple[asb.Airfoil, str] | None = None


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


def to_scalar(value: Scalar) -> float | Scalar:
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


def to_float_if_possible(value: Scalar) -> float | None:
    scalar = to_scalar(value)
    if isinstance(scalar, (int, float, onp.integer, onp.floating)):
        return float(scalar)
    try:
        array = onp.asarray(value, dtype=float).flatten()
        if array.size > 0:
            return float(array[0])
    except Exception:
        return None
    return None


# =============================================================================
# Airfoil setup
# =============================================================================

def build_reference_airfoil() -> tuple[asb.Airfoil, str]:
    airfoil = asb.Airfoil(name=PRIMARY_AIRFOIL_NAME)
    if GENERATE_POLARS:
        # Precompute/cache XFoil-like polars for faster repeat runs
        try:
            airfoil.generate_polars(
                cache_filename=str(CACHE_DIR / f"{PRIMARY_AIRFOIL_NAME}.json"),
                alphas=np.linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, N_ALPHA),
            )
        except Exception as exc:
            print(
                (
                    f"[WARN] Polar generation failed ({exc}); "
                    "using default aerodynamic model."
                ),
                flush=True,
            )

    return airfoil, PRIMARY_AIRFOIL_NAME


def get_reference_airfoil_cached() -> tuple[asb.Airfoil, str]:
    global _AIRFOIL_CACHE
    if _AIRFOIL_CACHE is None:
        _AIRFOIL_CACHE = build_reference_airfoil()
    return _AIRFOIL_CACHE


# =============================================================================
# Geometry builders
# =============================================================================

def build_main_wing(airfoil: asb.Airfoil, span_m: Scalar, chord_m: Scalar) -> asb.Wing:
    # Ailerons live on the outboard quarter of the semispan
    aileron_surface = asb.ControlSurface(
        name="aileron",
        symmetric=False,
        hinge_point=1.0 - AILERON_CHORD_FRACTION,
        trailing_edge=True,
    )

    xsecs = []
    for eta in onp.linspace(0.0, 1.0, N_WING_XSECS):
        # Straight rectangular planform with fixed dihedral
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
    tail_arm_m: Scalar,
    span_m: Scalar,
) -> tuple[asb.Wing, Scalar]:
    # Rectangular horizontal tail from span and fixed aspect ratio
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
    tail_arm_m: Scalar,
    height_m: Scalar,
) -> tuple[asb.Wing, Scalar]:
    # Rectangular vertical tail from height and fixed aspect ratio
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


def build_fuselage(tail_arm_m: Scalar, htail_chord_m: Scalar) -> asb.Fuselage:
    # Lightweight boom/pod representation with nose and tail stations
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

def surface_span(surface: asb.Wing, span_axis: SpanAxis) -> Scalar:
    # Span along Y for horizontal surfaces, Z for the vertical tail
    coords = []
    for xsec in surface.xsecs:
        coord = xsec.xyz_le[1] if span_axis == "y" else xsec.xyz_le[2]
        coords.append(coord)

    coords = np.array(coords)
    half_span = np.max(np.abs(coords))

    if surface.symmetric and span_axis == "y":
        return 2.0 * half_span

    return np.max(coords) - np.min(coords)


def surface_mid_chord_xyz(
    surface: asb.Wing,
    span_axis: SpanAxis,
) -> tuple[Scalar, Scalar, Scalar]:
    # Mid-chord centroid approximation between root and tip sections
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


def point_mass(
    mass_kg: Scalar,
    x_m: Scalar,
    y_m: Scalar = 0.0,
    z_m: Scalar = 0.0,
) -> asb.MassProperties:
    # Avionics are modeled as point masses; offsets capture inertia via parallel-axis terms.
    return asb.MassProperties(
        mass=mass_kg,
        x_cg=x_m,
        y_cg=y_m,
        z_cg=z_m,
        Ixx=0.0,
        Iyy=0.0,
        Izz=0.0,
        Ixy=0.0,
        Ixz=0.0,
        Iyz=0.0,
    )


def flat_plate_mass_properties(
    surface: asb.Wing,
    density_kg_m3: float,
    thickness_m: float,
    span_axis: SpanAxis,
) -> asb.MassProperties:
    # Thin-plate mass and inertia approximation for foam lifting surfaces
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
    wing_chord_m: Scalar,
    tail_arm_m: Scalar,
) -> tuple[MassPropertiesMap, asb.MassProperties, Scalar, Scalar]:
    mass_props: MassPropertiesMap = {}

    # Structural lifting-surface masses
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

    # Fixed onboard components
    mass_props["linkages"] = point_mass(0.001, x_m=0.5 * tail_arm_m)

    # Battery as a dedicated CG-trim slider
    battery_eta = opti.variable(
        init_guess=0.60,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    x_batt_max = BATTERY_X_MAX_FRAC * wing_chord_m
    x_batt = BATTERY_X_MIN_M + battery_eta * (x_batt_max - BATTERY_X_MIN_M)
    mass_props["battery"] = point_mass(BATTERY_MASS_KG, x_m=x_batt)

    # Central avionics cluster near battery/pod region
    x_pod = TAIL_SERVO_X_FRAC * wing_chord_m
    mass_props["flight_controller"] = point_mass(
        FLIGHT_CONTROLLER_MASS_KG,
        x_m=x_batt + 0.015,
    )
    mass_props["regulator"] = point_mass(
        REGULATOR_MASS_KG,
        x_m=x_batt + 0.010,
    )
    mass_props["receiver"] = point_mass(
        RECEIVER_MASS_KG,
        x_m=x_batt + 0.020,
    )

    # Servo layout: elevator/rudder central, ailerons spanwise in wing
    mass_props["servo_elevator"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_pod,
        y_m=0.0,
        z_m=0.0,
    )
    mass_props["servo_rudder"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_pod,
        y_m=0.0,
        z_m=0.0,
    )

    half_span = 0.5 * surface_span(wing, span_axis="y")
    servo_eta = np.clip(AILERON_SERVO_SPAN_ETA, AILERON_ETA_INBOARD, AILERON_ETA_OUTBOARD)
    y_servo = servo_eta * half_span
    z_servo = np.abs(y_servo) * np.tan(np.radians(DIHEDRAL_DEG))
    mass_props["servo_aileron_R"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_pod,
        y_m=y_servo,
        z_m=z_servo,
    )
    mass_props["servo_aileron_L"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_pod,
        y_m=-y_servo,
        z_m=z_servo,
    )

    # Optional wing spar represented as line-mass collapsed at quarter chord
    mass_props["wing_spar"] = point_mass(
        WING_SPAR_LINEAR_DENSITY_KG_M * surface_span(wing, span_axis="y"),
        x_m=0.25 * wing_chord_m,
        y_m=0.0,
        z_m=0.0,
    )

    boom_length_m = np.maximum(tail_arm_m - NOSE_X_M, 0.05)
    mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
        mass=BOOM_LINEAR_DENSITY_KG_M * boom_length_m,
        x_cg=0.5 * (NOSE_X_M + tail_arm_m),
    )
    mass_props["pod"] = point_mass(0.004, x_m=x_pod)

    # Ballast is optimized to close CG/stability constraints
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

    # Lump glue/assembly overhead as a fraction of subtotal
    mass_props["glue"] = subtotal * GLUE_FRACTION
    total_mass = subtotal + mass_props["glue"]

    return mass_props, total_mass, ballast_mass_kg, battery_eta


def aileron_effectiveness_proxy(
    aero: AeroMap,
    eta_inboard: float,
    eta_outboard: float,
    chord_fraction: float,
) -> Scalar:
    # Quick control-power proxy: Cl_delta_a ~ CLa * tau * outboard leverage
    c_l_alpha = np.maximum(np.abs(aero["CLa"]), 1e-3)
    span_factor = np.maximum(eta_outboard ** 2 - eta_inboard ** 2, 1e-4)
    tau_aileron = 0.9 * chord_fraction
    cl_delta_a = c_l_alpha * tau_aileron * span_factor
    return np.clip(cl_delta_a, 1e-3, 2.0)


def estimate_servo_hinge_moment(
    q_dyn: Scalar,
    control_area_m2: Scalar,
    mean_chord_m: Scalar,
    deflection_deg: Scalar,
) -> Scalar:
    # Quasi-steady hinge moment estimate for servo sizing checks
    delta_rad = np.radians(np.abs(deflection_deg))
    moment_arm_m = 0.25 * mean_chord_m
    return q_dyn * control_area_m2 * HINGE_MOMENT_COEFF * delta_rad * moment_arm_m

# =============================================================================
# Reporting and export
# =============================================================================

def constraint_record(
    name: str,
    value: Scalar,
    lower: float | None = None,
    upper: float | None = None,
    tol: float = 1e-6,
) -> ReportRow:
    # Store each constraint with pass/fail status for CSV/XLSX reporting
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


def design_variable_boundary_record(
    name: str,
    value: Scalar,
    lower: float,
    upper: float,
    unit: str,
    rel_tol: float = BOUNDARY_HIT_REL_TOL,
    abs_tol: float = BOUNDARY_HIT_ABS_TOL,
) -> ReportRow:
    # Track variables that are effectively pinned to explicit box bounds.
    value_f = float(to_scalar(value))
    scale = max(abs(lower), abs(upper), abs(upper - lower), 1e-9)
    tol = max(abs_tol, rel_tol * scale)
    at_lower = value_f <= lower + tol
    at_upper = value_f >= upper - tol

    if at_lower and at_upper:
        bound_hit = "both"
        warning = (
            f"[WARN] {name} is pinned by coincident bounds "
            f"(value={value_f:.6g}, lower={lower:.6g}, upper={upper:.6g})."
        )
    elif at_lower:
        bound_hit = "lower"
        warning = (
            f"[WARN] {name} hit lower bound "
            f"(value={value_f:.6g}, lower={lower:.6g}, tol={tol:.2g})."
        )
    elif at_upper:
        bound_hit = "upper"
        warning = (
            f"[WARN] {name} hit upper bound "
            f"(value={value_f:.6g}, upper={upper:.6g}, tol={tol:.2g})."
        )
    else:
        bound_hit = "none"
        warning = ""

    return {
        "Variable": name,
        "Value": value_f,
        "Lower": lower,
        "Upper": upper,
        "Unit": unit,
        "Tolerance": tol,
        "BoundHit": bound_hit,
        "IsAtBoundary": bound_hit != "none",
        "Warning": warning,
    }


def build_mass_rows(
    mass_props: MassPropertiesMap,
) -> ReportRows:
    rows: ReportRows = []

    # Component-by-component mass and principal inertia table
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


def build_aero_rows(aero: AeroMap) -> ReportRows:
    # Flat coefficient table for easier post-processing
    rows: ReportRows = []
    for key in sorted(aero.keys()):
        rows.append({"Coefficient": key, "Value": to_scalar(aero[key])})
    return rows


def save_results(
    summary_rows: ReportRows,
    geometry_rows: ReportRows,
    mass_rows: ReportRows,
    aero_rows: ReportRows,
    constraint_rows: ReportRows,
    design_points_rows: ReportRows | None = None,
    boundary_rows: ReportRows | None = None,
) -> PathMap:
    # Persist a compact CSV plus a multi-sheet workbook
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
        if boundary_rows is not None:
            pd.DataFrame(boundary_rows).to_excel(
                writer,
                sheet_name="DesignVarBounds",
                index=False,
            )
        if design_points_rows is not None:
            pd.DataFrame(design_points_rows).to_excel(
                writer,
                sheet_name="DesignPoints",
                index=False,
            )

    return {
        "results_csv": csv_path,
        "results_xlsx": xlsx_path,
    }


def make_plots(
    airplane: asb.Airplane,
    mass_props: MassPropertiesMap,
    total_mass: asb.MassProperties,
) -> PathMap:
    figure_outputs: PathMap = {}

    if not MAKE_PLOTS:
        return figure_outputs

    # Local imports keep startup light when plotting is disabled
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
        # Hide zero-ballast slice to avoid clutter
        mass_copy.pop("ballast")

    labels = [name.replace("_", " ").title() for name in mass_copy.keys()]
    values = [component.mass for component in mass_copy.values()]

    pretty.pie(
        values=values,
        names=labels,
        center_text=(
            f"$\\bf{{Mass\\ Budget}}$\n"
            f"TOGW: {to_scalar(total_mass.mass) * 1e3:.2f} g"
        ),
        label_format=(
            lambda name, value, percentage: (
                f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%"
            )
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
    summary_rows: ReportRows,
    geometry_rows: ReportRows,
    constraint_rows: ReportRows,
    boundary_rows: ReportRows | None,
    output_paths: PathMap,
    figure_paths: PathMap,
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
    print(
        "  - Steady-flow reference condition "
        "(no updraft / no roll-in trajectory)",
        flush=True,
    )
    print("  - Rectangular wing, fixed dihedral, shared tail arm", flush=True)
    print("  - Control limits set to +/-30 deg", flush=True)

    print("\nTrimmed flight point:", flush=True)
    print(
        f"  V_nom = {V_NOM_MPS:.2f} m/s | alpha = {fmt('alpha_trim_deg', 3)} deg",
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

    print("\nDesign-variable boundary warnings:", flush=True)
    boundary_hits = [
        row for row in (boundary_rows or []) if bool(row.get("IsAtBoundary", False))
    ]
    if not boundary_hits:
        print("  None.", flush=True)
    else:
        for row in boundary_hits:
            print(f"  {row['Warning']}", flush=True)

    print("\nSaved files:", flush=True)
    for key, path in output_paths.items():
        print(f"  {key}: {path}", flush=True)

    for key, path in figure_paths.items():
        print(f"  figure_{key}: {path}", flush=True)

# =============================================================================
# Main optimization workflow
# =============================================================================

def legacy_single_run_main(
    init_override: dict[str, float] | None = None,
    ipopt_options: dict[str, Any] | None = None,
    export_outputs: bool = True,
) -> Candidate | None:
    version = get_git_version()
    print(f"CODE_VERSION: {version}", flush=True)

    if init_override is None:
        init_override = {}

    def init_value(name: str, default: float) -> float:
        return float(init_override.get(name, default))

    # Initialization
    ensure_output_dirs()
    airfoil, airfoil_label = get_reference_airfoil_cached()

    opti = asb.Opti()

    # Trim-state design variables
    alpha_deg = opti.variable(
        init_guess=init_value("alpha_deg", 4.0),
        lower_bound=ALPHA_MIN_DEG,
        upper_bound=ALPHA_MAX_DEG,
    )

    # Control-surface trim variables
    delta_a_deg = opti.variable(
        init_guess=init_value("delta_a_deg", 0.0),
        lower_bound=DELTA_A_MIN_DEG,
        upper_bound=DELTA_A_MAX_DEG,
    )
    delta_e_deg = opti.variable(
        init_guess=init_value("delta_e_deg", 0.0),
        lower_bound=DELTA_E_MIN_DEG,
        upper_bound=DELTA_E_MAX_DEG,
    )
    delta_r_deg = opti.variable(
        init_guess=init_value("delta_r_deg", 0.0),
        lower_bound=DELTA_R_MIN_DEG,
        upper_bound=DELTA_R_MAX_DEG,
    )

    # Primary geometry design variables
    wing_span_m = opti.variable(
        init_guess=init_value("wing_span_m", 1.00),
        lower_bound=WING_SPAN_MIN_M,
        upper_bound=WING_SPAN_MAX_M,
    )
    wing_chord_m = opti.variable(
        init_guess=init_value("wing_chord_m", 0.14),
        lower_bound=WING_CHORD_MIN_M,
        upper_bound=WING_CHORD_MAX_M,
    )
    tail_arm_m = opti.variable(
        init_guess=init_value("tail_arm_m", 0.62),
        lower_bound=TAIL_ARM_MIN_M,
        upper_bound=TAIL_ARM_MAX_M,
    )
    htail_span_m = opti.variable(
        init_guess=init_value("htail_span_m", 0.30),
        lower_bound=HT_SPAN_MIN_M,
        upper_bound=HT_SPAN_MAX_M,
    )
    vtail_height_m = opti.variable(
        init_guess=init_value("vtail_height_m", 0.16),
        lower_bound=VT_HEIGHT_MIN_M,
        upper_bound=VT_HEIGHT_MAX_M,
    )

    variable_map = {
        "alpha_deg": alpha_deg,
        "delta_a_deg": delta_a_deg,
        "delta_e_deg": delta_e_deg,
        "delta_r_deg": delta_r_deg,
        "wing_span_m": wing_span_m,
        "wing_chord_m": wing_chord_m,
        "tail_arm_m": tail_arm_m,
        "htail_span_m": htail_span_m,
        "vtail_height_m": vtail_height_m,
    }
    for name, variable in variable_map.items():
        if name in init_override:
            opti.set_initial(variable, float(init_override[name]))

    # Airframe assembly
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

    atmos = asb.Atmosphere(altitude=0.0)

    # Nominal operating condition for efficiency objective
    op_point_nom = asb.OperatingPoint(
        atmosphere=atmos,
        velocity=V_NOM_MPS,
        alpha=alpha_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )

    # Manoeuvre operating condition for arena-feasibility constraints
    op_point_turn = asb.OperatingPoint(
        atmosphere=atmos,
        velocity=V_TURN_MPS,
        alpha=alpha_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )

    mass_props, total_mass, ballast_mass_kg, battery_eta = build_mass_model(
        opti=opti,
        wing=wing,
        htail=htail,
        vtail=vtail,
        wing_chord_m=wing_chord_m,
        tail_arm_m=tail_arm_m,
    )

    # Nominal aerodynamics about current CG with stability derivatives
    aero_nom = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point_nom,
        xyz_ref=total_mass.xyz_cg,
    ).run_with_stability_derivatives(
        alpha=True,
        beta=True,
        p=True,
        q=True,
        r=True,
    )

    # Manoeuvre-point aerodynamics for two-speed constraints
    aero_turn = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point_turn,
        xyz_ref=total_mass.xyz_cg,
    ).run()

    # Derived performance and stability metrics
    wing_area_m2 = wing.area()
    wing_mac_m = wing.mean_aerodynamic_chord()
    htail_area_m2 = htail.area()
    vtail_area_m2 = vtail.area()

    wing_loading_n_m2 = total_mass.mass * G / np.maximum(wing_area_m2, 1e-8)
    reynolds_wing = op_point_nom.reynolds(wing_mac_m)
    static_margin = (aero_nom["x_np"] - total_mass.x_cg) / np.maximum(wing_mac_m, 1e-8)

    tail_volume_horizontal = htail_area_m2 * tail_arm_m / np.maximum(
        wing_area_m2 * wing_mac_m,
        1e-8,
    )
    tail_volume_vertical = vtail_area_m2 * tail_arm_m / np.maximum(
        wing_area_m2 * wing_span_m,
        1e-8,
    )

    l_over_d = aero_nom["L"] / np.maximum(aero_nom["D"], 1e-8)
    sink_rate_nom_mps = (
        aero_nom["D"] * V_NOM_MPS / np.maximum(total_mass.mass * G, 1e-8)
    )

    cl_delta_a = aileron_effectiveness_proxy(
        aero=aero_nom,
        eta_inboard=AILERON_ETA_INBOARD,
        eta_outboard=AILERON_ETA_OUTBOARD,
        chord_fraction=AILERON_CHORD_FRACTION,
    )

    # Linearized roll response estimates
    q_dyn = 0.5 * RHO * V_NOM_MPS ** 2
    i_xx = np.maximum(total_mass.inertia_tensor[0, 0], 1e-8)
    clp_mag = np.maximum(np.abs(aero_nom["Clp"]), 1e-5)
    delta_a_max_rad = np.radians(DELTA_A_MAX_DEG)

    roll_accel0_rad_s2 = (
        q_dyn
        * wing_area_m2
        * wing_span_m
        * np.abs(cl_delta_a)
        * delta_a_max_rad
        / i_xx
    )

    roll_tau_s = (2.0 * i_xx * V_NOM_MPS) / np.maximum(
        q_dyn * wing_area_m2 * wing_span_m ** 2 * clp_mag,
        1e-8,
    )

    roll_rate_ss_radps = (
        2.0
        * V_NOM_MPS
        / np.maximum(wing_span_m, 1e-8)
        * np.abs(cl_delta_a)
        * delta_a_max_rad
        / clp_mag
    )

    # Control-surface areas/chords for hinge-moment checks
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

    # Apply servo safety factor to rated torque
    servo_torque_available_nm = (
        SERVO_TORQUE_LIMIT_NM / SERVO_SAFETY_FACTOR
        * (CONTROL_HORN_ARM_M / SERVO_ARM_M)
        * LINKAGE_EFFICIENCY
    )

    # Penalize unnecessary trim deflections
    trim_effort = delta_e_deg ** 2 + 0.3 * delta_r_deg ** 2 + 0.15 * delta_a_deg ** 2

    # Arena footprint for coordinated level turn
    phi_turn_rad = np.radians(TURN_BANK_DEG)
    turn_radius_m = V_TURN_MPS ** 2 / (G * np.tan(phi_turn_rad))

    # Objective: nominal-speed sink with light penalties on mass, ballast, and trim effort
    objective = (
        sink_rate_nom_mps
        + MASS_WEIGHT_IN_OBJECTIVE * total_mass.mass
        + BALLAST_WEIGHT_IN_OBJECTIVE * ballast_mass_kg
        + CONTROL_TRIM_WEIGHT * trim_effort
    )
    opti.minimize(objective)

    # Feasibility constraints
    opti.subject_to(
        [
            # Nominal-speed trim/performance constraints
            aero_nom["L"] >= total_mass.mass * G,
            aero_nom["D"] >= 1e-3,
            aero_nom["Cm"] == 0.0,
            aero_nom["Cl"] == 0.0,
            aero_nom["Cn"] == 0.0,
            aero_nom["CL"] <= MAX_CL_AT_DESIGN_POINT,
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
            aero_nom["Clb"] <= CLB_MAX,
            aero_nom["Cnb"] >= CNB_MIN,
            aero_nom["Cmq"] <= CMQ_MAX,
            roll_rate_ss_radps >= MIN_ROLL_RATE_RAD_S,
            roll_accel0_rad_s2 >= MIN_ROLL_ACCEL_RAD_S2,
            roll_tau_s <= MAX_ROLL_TAU_S,
            hinge_moment_aileron_nm <= servo_torque_available_nm,
            hinge_moment_elevator_nm <= servo_torque_available_nm,
            hinge_moment_rudder_nm <= servo_torque_available_nm,
            # Manoeuvre-speed constraints
            turn_radius_m + 0.5 * wing_span_m + WALL_CLEARANCE_M <= 0.5 * ARENA_WIDTH_M,
            aero_turn["CL"] <= TURN_CL_CAP,
            alpha_deg <= (STALL_ALPHA_LIMIT_DEG - TURN_ALPHA_MARGIN_DEG),
            opti.bounded(
                -TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG,
                delta_a_deg,
                TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG,
            ),
            opti.bounded(
                -TURN_DEFLECTION_UTIL_MAX * DELTA_E_MAX_DEG,
                delta_e_deg,
                TURN_DEFLECTION_UTIL_MAX * DELTA_E_MAX_DEG,
            ),
            opti.bounded(
                -TURN_DEFLECTION_UTIL_MAX * DELTA_R_MAX_DEG,
                delta_r_deg,
                TURN_DEFLECTION_UTIL_MAX * DELTA_R_MAX_DEG,
            ),
        ]
    )

    # IPOPT setup
    plugin_options = {"print_time": False, "verbose": False}
    solver_options = {
        "max_iter": 3000,
        "check_derivatives_for_naninf": "yes",
        "hessian_approximation": "limited-memory",
        "print_level": 0,
        "sb": "yes",
    }
    if ipopt_options is not None:
        solver_options.update(ipopt_options)
    opti.solver("ipopt", plugin_options, solver_options)

    print("Starting optimization...", flush=True)
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        print(f"\n[SOLVE FAILED] {exc}", flush=True)
        print("No feasible design was found with the current settings", flush=True)
        return None

    # Numeric post-processing for reports and exports
    airplane_num = solution(airplane)
    wing_num = copy.deepcopy(airplane_num.wings[0])
    htail_num = copy.deepcopy(airplane_num.wings[1])
    vtail_num = copy.deepcopy(airplane_num.wings[2])

    mass_props_num = solution(mass_props)
    total_mass_num = solution(total_mass)
    aero_nom_num = solution(aero_nom)
    aero_turn_num = solution(aero_turn)

    objective_num = to_scalar(solution(objective))
    alpha_num = to_scalar(solution(alpha_deg))
    delta_a_num = to_scalar(solution(delta_a_deg))
    delta_e_num = to_scalar(solution(delta_e_deg))
    delta_r_num = to_scalar(solution(delta_r_deg))
    wing_span_design_num = to_scalar(solution(wing_span_m))
    wing_chord_design_num = to_scalar(solution(wing_chord_m))
    tail_arm_design_num = to_scalar(solution(tail_arm_m))
    htail_span_design_num = to_scalar(solution(htail_span_m))
    vtail_height_design_num = to_scalar(solution(vtail_height_m))
    battery_eta_num = to_scalar(solution(battery_eta))

    sink_rate_num = to_scalar(solution(sink_rate_nom_mps))
    l_over_d_num = to_scalar(solution(l_over_d))
    mass_total_num = to_scalar(total_mass_num.mass)
    ballast_mass_num = to_scalar(solution(ballast_mass_kg))
    battery_x_num = to_scalar(
        BATTERY_X_MIN_M
        + battery_eta_num
        * (BATTERY_X_MAX_FRAC * wing_chord_design_num - BATTERY_X_MIN_M)
    )

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

    turn_radius_num = float(V_TURN_MPS ** 2 / (G * onp.tan(onp.radians(TURN_BANK_DEG))))
    turn_footprint_lhs_num = (
        turn_radius_num
        + 0.5 * float(wing_span_design_num)
        + WALL_CLEARANCE_M
    )

    wing_span_num = to_scalar(wing_num.span())
    wing_area_num = to_scalar(wing_num.area())
    wing_chord_num = wing_area_num / max(wing_span_num, 1e-8)

    htail_span_num = to_scalar(htail_num.span())
    htail_area_num = to_scalar(htail_num.area())
    htail_chord_num = htail_area_num / max(htail_span_num, 1e-8)

    vtail_height_num = to_scalar(surface_span(vtail_num, "z"))
    vtail_area_num = to_scalar(vtail_num.area())
    vtail_chord_num = vtail_area_num / max(vtail_height_num, 1e-8)

    tail_arm_num = tail_arm_design_num

    boundary_rows = [
        design_variable_boundary_record(
            name="alpha_deg",
            value=alpha_num,
            lower=ALPHA_MIN_DEG,
            upper=ALPHA_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_a_deg",
            value=delta_a_num,
            lower=DELTA_A_MIN_DEG,
            upper=DELTA_A_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_e_deg",
            value=delta_e_num,
            lower=DELTA_E_MIN_DEG,
            upper=DELTA_E_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_r_deg",
            value=delta_r_num,
            lower=DELTA_R_MIN_DEG,
            upper=DELTA_R_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="wing_span_m",
            value=wing_span_design_num,
            lower=WING_SPAN_MIN_M,
            upper=WING_SPAN_MAX_M,
            unit="m",
        ),
        design_variable_boundary_record(
            name="wing_chord_m",
            value=wing_chord_design_num,
            lower=WING_CHORD_MIN_M,
            upper=WING_CHORD_MAX_M,
            unit="m",
        ),
        design_variable_boundary_record(
            name="tail_arm_m",
            value=tail_arm_design_num,
            lower=TAIL_ARM_MIN_M,
            upper=TAIL_ARM_MAX_M,
            unit="m",
        ),
        design_variable_boundary_record(
            name="htail_span_m",
            value=htail_span_design_num,
            lower=HT_SPAN_MIN_M,
            upper=HT_SPAN_MAX_M,
            unit="m",
        ),
        design_variable_boundary_record(
            name="vtail_height_m",
            value=vtail_height_design_num,
            lower=VT_HEIGHT_MIN_M,
            upper=VT_HEIGHT_MAX_M,
            unit="m",
        ),
        design_variable_boundary_record(
            name="ballast_mass_kg",
            value=ballast_mass_num,
            lower=0.0,
            upper=BALLAST_MAX_KG,
            unit="kg",
        ),
        design_variable_boundary_record(
            name="battery_slider_eta",
            value=battery_eta_num,
            lower=0.0,
            upper=1.0,
            unit="-",
        ),
    ]

    # Report tables
    summary_rows = [
        {"Metric": "code_version", "Value": version, "Unit": "-"},
        {"Metric": "airfoil_model", "Value": airfoil_label, "Unit": "-"},
        {"Metric": "objective", "Value": objective_num, "Unit": "-"},
        {"Metric": "v_nom_mps", "Value": V_NOM_MPS, "Unit": "m/s"},
        {"Metric": "v_turn_mps", "Value": V_TURN_MPS, "Unit": "m/s"},
        {"Metric": "turn_bank_deg", "Value": TURN_BANK_DEG, "Unit": "deg"},
        {"Metric": "arena_width_m", "Value": ARENA_WIDTH_M, "Unit": "m"},
        {"Metric": "wall_clearance_m", "Value": WALL_CLEARANCE_M, "Unit": "m"},
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
        {"Metric": "battery_slider_eta", "Value": battery_eta_num, "Unit": "-"},
        {"Metric": "battery_x_m", "Value": battery_x_num, "Unit": "m"},
        {"Metric": "sink_rate_mps", "Value": sink_rate_num, "Unit": "m/s"},
        {"Metric": "L_over_D", "Value": l_over_d_num, "Unit": "-"},
        {"Metric": "turn_radius_m", "Value": turn_radius_num, "Unit": "m"},
        {"Metric": "turn_footprint_lhs_m", "Value": turn_footprint_lhs_num, "Unit": "m"},
        {"Metric": "turn_CL", "Value": to_scalar(aero_turn_num["CL"]), "Unit": "-"},
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
    boundary_hits = [row for row in boundary_rows if bool(row["IsAtBoundary"])]
    summary_rows.append(
        {
            "Metric": "design_var_bound_hits_count",
            "Value": len(boundary_hits),
            "Unit": "-",
        }
    )
    if boundary_hits:
        summary_rows.append(
            {
                "Metric": "design_var_bound_hits",
                "Value": "; ".join(
                    f"{row['Variable']}:{row['BoundHit']}" for row in boundary_hits
                ),
                "Unit": "-",
            }
        )

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

    # Expanded per-component outputs
    mass_rows = build_mass_rows(mass_props_num)
    aero_rows = []
    aero_rows.extend(
        {"Coefficient": f"nominal_{row['Coefficient']}", "Value": row["Value"]}
        for row in build_aero_rows(aero_nom_num)
    )
    aero_rows.extend(
        {"Coefficient": f"turn_{row['Coefficient']}", "Value": row["Value"]}
        for row in build_aero_rows(aero_turn_num)
    )

    # Constraint audit table (mirrors optimization constraints)
    constraint_rows = [
        constraint_record("Lift >= Weight", aero_nom_num["L"], lower=mass_total_num * G),
        constraint_record("Drag >= 0", aero_nom_num["D"], lower=1e-3),
        constraint_record(
            "Nominal Trim Cm",
            aero_nom_num["Cm"],
            lower=0.0,
            upper=0.0,
            tol=1e-3,
        ),
        constraint_record(
            "Nominal Trim Cl",
            aero_nom_num["Cl"],
            lower=0.0,
            upper=0.0,
            tol=1e-3,
        ),
        constraint_record(
            "Nominal Trim Cn",
            aero_nom_num["Cn"],
            lower=0.0,
            upper=0.0,
            tol=1e-3,
        ),
        constraint_record(
            "CL <= CLmax",
            aero_nom_num["CL"],
            upper=MAX_CL_AT_DESIGN_POINT,
        ),
        constraint_record(
            "Alpha <= turn stall margin",
            alpha_num,
            upper=STALL_ALPHA_LIMIT_DEG - TURN_ALPHA_MARGIN_DEG,
        ),
        constraint_record("Turn CL cap", aero_turn_num["CL"], upper=TURN_CL_CAP),
        constraint_record(
            "Turn footprint in width",
            turn_footprint_lhs_num,
            upper=0.5 * ARENA_WIDTH_M,
        ),
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
        constraint_record(
            "Static margin minimum",
            static_margin_num,
            lower=STATIC_MARGIN_MIN,
        ),
        constraint_record(
            "Static margin maximum",
            static_margin_num,
            upper=STATIC_MARGIN_MAX,
        ),
        constraint_record("Vh minimum", tail_volume_h_num, lower=VH_MIN),
        constraint_record("Vh maximum", tail_volume_h_num, upper=VH_MAX),
        constraint_record("Vv minimum", tail_volume_v_num, lower=VV_MIN),
        constraint_record("Vv maximum", tail_volume_v_num, upper=VV_MAX),
        constraint_record("Clb <= 0", aero_nom_num["Clb"], upper=CLB_MAX),
        constraint_record("Cnb >= 0", aero_nom_num["Cnb"], lower=CNB_MIN),
        constraint_record("Cmq <= -0.01", aero_nom_num["Cmq"], upper=CMQ_MAX),
        constraint_record(
            "Roll rate minimum",
            roll_rate_num,
            lower=MIN_ROLL_RATE_RAD_S,
        ),
        constraint_record(
            "Roll accel minimum",
            roll_accel_num,
            lower=MIN_ROLL_ACCEL_RAD_S2,
        ),
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
        constraint_record(
            "Turn aileron trim utilization",
            abs(delta_a_num),
            upper=TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG,
        ),
        constraint_record(
            "Turn elevator trim utilization",
            abs(delta_e_num),
            upper=TURN_DEFLECTION_UTIL_MAX * DELTA_E_MAX_DEG,
        ),
        constraint_record(
            "Turn rudder trim utilization",
            abs(delta_r_num),
            upper=TURN_DEFLECTION_UTIL_MAX * DELTA_R_MAX_DEG,
        ),
    ]

    design_points_rows: ReportRows = [
        {"DesignPoint": "Settings", "Metric": "V_NOM_MPS", "Value": V_NOM_MPS, "Unit": "m/s"},
        {"DesignPoint": "Settings", "Metric": "V_TURN_MPS", "Value": V_TURN_MPS, "Unit": "m/s"},
        {
            "DesignPoint": "Settings",
            "Metric": "TURN_BANK_DEG",
            "Value": TURN_BANK_DEG,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Settings",
            "Metric": "WALL_CLEARANCE_M",
            "Value": WALL_CLEARANCE_M,
            "Unit": "m",
        },
        {"DesignPoint": "Arena", "Metric": "ARENA_LENGTH_M", "Value": ARENA_LENGTH_M, "Unit": "m"},
        {"DesignPoint": "Arena", "Metric": "ARENA_WIDTH_M", "Value": ARENA_WIDTH_M, "Unit": "m"},
        {"DesignPoint": "Arena", "Metric": "ARENA_HEIGHT_M", "Value": ARENA_HEIGHT_M, "Unit": "m"},
        {
            "DesignPoint": "Nominal",
            "Metric": "alpha_nom_deg",
            "Value": alpha_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "delta_a_nom_deg",
            "Value": delta_a_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "delta_e_nom_deg",
            "Value": delta_e_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "delta_r_nom_deg",
            "Value": delta_r_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "sink_nom_mps",
            "Value": sink_rate_num,
            "Unit": "m/s",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "CL_nom",
            "Value": to_scalar(aero_nom_num["CL"]),
            "Unit": "-",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "alpha_turn_deg",
            "Value": alpha_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_a_turn_deg",
            "Value": delta_a_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_e_turn_deg",
            "Value": delta_e_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_r_turn_deg",
            "Value": delta_r_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "CL_turn",
            "Value": to_scalar(aero_turn_num["CL"]),
            "Unit": "-",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "turn_radius_m",
            "Value": turn_radius_num,
            "Unit": "m",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "turn_footprint_lhs_m",
            "Value": turn_footprint_lhs_num,
            "Unit": "m",
        },
    ]

    aero_scalar_map: dict[str, float] = {}
    for key, value in aero_nom_num.items():
        maybe_scalar = to_float_if_possible(value)
        if maybe_scalar is not None:
            aero_scalar_map[f"nominal_{key}"] = maybe_scalar
    for key, value in aero_turn_num.items():
        maybe_scalar = to_float_if_possible(value)
        if maybe_scalar is not None:
            aero_scalar_map[f"turn_{key}"] = maybe_scalar

    candidate = Candidate(
        candidate_id=-1,
        objective=float(to_scalar(objective_num)),
        wing_span_m=float(to_scalar(wing_span_num)),
        wing_chord_m=float(to_scalar(wing_chord_num)),
        tail_arm_m=float(to_scalar(tail_arm_num)),
        htail_span_m=float(to_scalar(htail_span_num)),
        vtail_height_m=float(to_scalar(vtail_height_num)),
        alpha_deg=float(to_scalar(alpha_num)),
        delta_a_deg=float(to_scalar(delta_a_num)),
        delta_e_deg=float(to_scalar(delta_e_num)),
        delta_r_deg=float(to_scalar(delta_r_num)),
        sink_rate_mps=float(to_scalar(sink_rate_num)),
        l_over_d=float(to_scalar(l_over_d_num)),
        mass_total_kg=float(to_scalar(mass_total_num)),
        ballast_mass_kg=float(to_scalar(ballast_mass_num)),
        static_margin=float(to_scalar(static_margin_num)),
        vh=float(to_scalar(tail_volume_h_num)),
        vv=float(to_scalar(tail_volume_v_num)),
        roll_tau_s=float(to_scalar(roll_tau_num)),
        roll_rate_ss_radps=float(to_scalar(roll_rate_num)),
        roll_accel0_rad_s2=float(to_scalar(roll_accel_num)),
        max_servo_util=float(to_scalar(max_servo_utilization)),
        airplane=airplane_num,
        total_mass=total_mass_num,
        mass_props=mass_props_num,
        aero=aero_scalar_map,
        summary_rows=summary_rows,
        geometry_rows=geometry_rows,
        mass_rows=mass_rows,
        aero_rows=aero_rows,
        constraint_rows=constraint_rows,
        boundary_rows=boundary_rows,
        design_points_rows=design_points_rows,
        wing_area_m2=float(to_scalar(wing_area_num)),
        wing_mac_m=float(to_scalar(wing_num.mean_aerodynamic_chord())),
        airfoil_label=airfoil_label,
    )

    if not export_outputs:
        return candidate

    # Persist outputs and figures
    output_paths = save_results(
        summary_rows=summary_rows,
        geometry_rows=geometry_rows,
        mass_rows=mass_rows,
        aero_rows=aero_rows,
        constraint_rows=constraint_rows,
        design_points_rows=design_points_rows,
        boundary_rows=boundary_rows,
    )

    figure_paths: PathMap = {}
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
        boundary_rows=boundary_rows,
        output_paths=output_paths,
        figure_paths=figure_paths,
    )
    return candidate


def build_and_solve_once(
    init: dict[str, float],
    ipopt_options: dict[str, Any] | None = None,
) -> Candidate | None:
    return legacy_single_run_main(
        init_override=init,
        ipopt_options=ipopt_options,
        export_outputs=False,
    )


def sample_initial_guess(rng: onp.random.Generator) -> dict[str, float]:
    return {
        "wing_span_m": float(rng.uniform(WING_SPAN_MIN_M, WING_SPAN_MAX_M)),
        "wing_chord_m": float(rng.uniform(WING_CHORD_MIN_M, WING_CHORD_MAX_M)),
        "tail_arm_m": float(rng.uniform(TAIL_ARM_MIN_M, TAIL_ARM_MAX_M)),
        "htail_span_m": float(rng.uniform(HT_SPAN_MIN_M, HT_SPAN_MAX_M)),
        "vtail_height_m": float(rng.uniform(VT_HEIGHT_MIN_M, VT_HEIGHT_MAX_M)),
        "alpha_deg": float(rng.uniform(ALPHA_MIN_DEG, ALPHA_MAX_DEG)),
        "delta_a_deg": float(rng.uniform(DELTA_A_MIN_DEG, DELTA_A_MAX_DEG)),
        "delta_e_deg": float(rng.uniform(DELTA_E_MIN_DEG, DELTA_E_MAX_DEG)),
        "delta_r_deg": float(rng.uniform(DELTA_R_MIN_DEG, DELTA_R_MAX_DEG)),
    }


def candidates_are_duplicates(
    candidate_a: Candidate,
    candidate_b: Candidate,
    config: WorkflowConfig,
) -> bool:
    return (
        abs(candidate_a.wing_span_m - candidate_b.wing_span_m) < config.dedup_span_m
        and abs(candidate_a.wing_chord_m - candidate_b.wing_chord_m)
        < config.dedup_chord_m
        and abs(candidate_a.tail_arm_m - candidate_b.tail_arm_m) < config.dedup_tail_arm_m
    )


def run_multistart(config: WorkflowConfig) -> list[Candidate]:
    rng = onp.random.default_rng(config.random_seed)
    feasible_candidates: list[Candidate] = []

    for start_index in range(config.n_starts):
        init = sample_initial_guess(rng)
        candidate = build_and_solve_once(init=init, ipopt_options=None)
        if candidate is None:
            print(f"[multistart] start {start_index + 1}/{config.n_starts} failed", flush=True)
            continue
        candidate.candidate_id = start_index + 1
        feasible_candidates.append(candidate)
        print(
            (
                f"[multistart] start {start_index + 1}/{config.n_starts} feasible "
                f"(objective={candidate.objective:.5f})"
            ),
            flush=True,
        )

    if not feasible_candidates:
        return []

    deduped: list[Candidate] = []
    for candidate in sorted(feasible_candidates, key=lambda item: item.objective):
        if any(candidates_are_duplicates(candidate, kept, config) for kept in deduped):
            continue
        deduped.append(candidate)

    deduped = sorted(deduped, key=lambda item: item.objective)[: config.keep_top_k]
    for idx, candidate in enumerate(deduped, start=1):
        candidate.candidate_id = idx
    return deduped


def sample_scenarios(config: WorkflowConfig) -> pd.DataFrame:
    rng = onp.random.default_rng(config.scenario_seed)
    scenario_count = max(config.n_scenarios, 0)
    return pd.DataFrame(
        {
            "scenario_id": onp.arange(scenario_count, dtype=int),
            "mass_scale": rng.uniform(
                config.mass_scale_min,
                config.mass_scale_max,
                scenario_count,
            ),
            "cg_x_shift_mac": rng.uniform(
                config.cg_x_shift_mac_min,
                config.cg_x_shift_mac_max,
                scenario_count,
            ),
            "incidence_bias_deg": rng.uniform(
                config.incidence_bias_deg_min,
                config.incidence_bias_deg_max,
                scenario_count,
            ),
            "control_eff": rng.uniform(
                config.control_eff_min,
                config.control_eff_max,
                scenario_count,
            ),
            "drag_factor": rng.uniform(
                config.drag_factor_min,
                config.drag_factor_max,
                scenario_count,
            ),
        }
    )


def trim_candidate_under_scenario(
    candidate: Candidate,
    scenario_row: dict[str, Any],
    config: WorkflowConfig,
) -> dict[str, Any]:
    scenario_id = int(scenario_row["scenario_id"])
    mass_scale = float(scenario_row["mass_scale"])
    cg_x_shift_mac = float(scenario_row["cg_x_shift_mac"])
    incidence_bias_deg = float(scenario_row["incidence_bias_deg"])
    control_eff = float(scenario_row["control_eff"])
    drag_factor = float(scenario_row["drag_factor"])

    trim_limit_a = config.max_trim_util_fraction * max(
        abs(DELTA_A_MIN_DEG),
        abs(DELTA_A_MAX_DEG),
    )
    trim_limit_e = config.max_trim_util_fraction * max(
        abs(DELTA_E_MIN_DEG),
        abs(DELTA_E_MAX_DEG),
    )
    trim_limit_r = config.max_trim_util_fraction * max(
        abs(DELTA_R_MIN_DEG),
        abs(DELTA_R_MAX_DEG),
    )

    opti = asb.Opti()
    alpha_deg = opti.variable(
        init_guess=candidate.alpha_deg,
        lower_bound=ALPHA_MIN_DEG,
        upper_bound=ALPHA_MAX_DEG,
    )
    delta_a_deg = opti.variable(
        init_guess=candidate.delta_a_deg,
        lower_bound=-trim_limit_a,
        upper_bound=trim_limit_a,
    )
    delta_e_deg = opti.variable(
        init_guess=candidate.delta_e_deg,
        lower_bound=-trim_limit_e,
        upper_bound=trim_limit_e,
    )
    delta_r_deg = opti.variable(
        init_guess=candidate.delta_r_deg,
        lower_bound=-trim_limit_r,
        upper_bound=trim_limit_r,
    )

    airfoil, _ = get_reference_airfoil_cached()
    wing = build_main_wing(
        airfoil=airfoil,
        span_m=candidate.wing_span_m,
        chord_m=candidate.wing_chord_m,
    )
    htail, htail_chord_m = build_horizontal_tail(
        airfoil=airfoil,
        tail_arm_m=candidate.tail_arm_m,
        span_m=candidate.htail_span_m,
    )
    vtail, _vtail_chord_m = build_vertical_tail(
        airfoil=airfoil,
        tail_arm_m=candidate.tail_arm_m,
        height_m=candidate.vtail_height_m,
    )
    fuselage = build_fuselage(
        tail_arm_m=candidate.tail_arm_m,
        htail_chord_m=htail_chord_m,
    )

    airplane = asb.Airplane(
        name=f"Nausicaa candidate {candidate.candidate_id}",
        wings=[wing, htail, vtail],
        fuselages=[fuselage],
    ).with_control_deflections(
        {
            "aileron": control_eff * delta_a_deg,
            "elevator": control_eff * delta_e_deg,
            "rudder": control_eff * delta_r_deg,
        }
    )

    base_x_cg = 0.25 * candidate.wing_chord_m
    base_y_cg = 0.0
    base_z_cg = 0.0
    if candidate.total_mass is not None:
        base_x_cg = float(to_scalar(candidate.total_mass.x_cg))
        base_y_cg = float(to_scalar(candidate.total_mass.y_cg))
        base_z_cg = float(to_scalar(candidate.total_mass.z_cg))

    cg_shift_m = cg_x_shift_mac * max(candidate.wing_mac_m, 1e-8)
    xyz_ref = [base_x_cg + cg_shift_m, base_y_cg, base_z_cg]

    op_point = asb.OperatingPoint(
        velocity=V_NOM_MPS,
        alpha=alpha_deg + incidence_bias_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=xyz_ref,
    ).run()

    weight_n = mass_scale * candidate.mass_total_kg * G
    drag_with_factor = aero["D"] * drag_factor
    sink_rate_mps = drag_with_factor * V_NOM_MPS / np.maximum(weight_n, 1e-8)
    max_alpha_cap = STALL_ALPHA_LIMIT_DEG - config.stall_alpha_margin_deg
    max_cl_cap = MAX_CL_AT_DESIGN_POINT - config.cl_margin
    trim_penalty = delta_e_deg ** 2 + 0.3 * delta_r_deg ** 2 + 0.15 * delta_a_deg ** 2

    opti.minimize(sink_rate_mps + CONTROL_TRIM_WEIGHT * trim_penalty)
    opti.subject_to(
        [
            aero["L"] >= weight_n,
            aero["Cm"] == 0.0,
            aero["CL"] <= max_cl_cap,
            alpha_deg <= max_alpha_cap,
            opti.bounded(-trim_limit_e, delta_e_deg, trim_limit_e),
            opti.bounded(-trim_limit_r, delta_r_deg, trim_limit_r),
            opti.bounded(-trim_limit_a, delta_a_deg, trim_limit_a),
        ]
    )
    opti.solver(
        "ipopt",
        {"print_time": False, "verbose": False},
        {
            "max_iter": 800,
            "hessian_approximation": "limited-memory",
            "print_level": 0,
            "sb": "yes",
        },
    )

    try:
        solution = opti.solve()
    except RuntimeError:
        return {
            "candidate_id": candidate.candidate_id,
            "scenario_id": scenario_id,
            "mass_scale": mass_scale,
            "cg_x_shift_mac": cg_x_shift_mac,
            "incidence_bias_deg": incidence_bias_deg,
            "control_eff": control_eff,
            "drag_factor": drag_factor,
            "trim_success": False,
            "alpha_deg": onp.nan,
            "delta_a_deg": onp.nan,
            "delta_e_deg": onp.nan,
            "delta_r_deg": onp.nan,
            "sink_rate_mps": onp.nan,
            "L_over_D": onp.nan,
            "CL": onp.nan,
            "D": onp.nan,
            "alpha_margin_deg": onp.nan,
            "cl_margin_to_cap": onp.nan,
            "delta_e_util": onp.nan,
            "delta_r_util": onp.nan,
            "delta_a_util": onp.nan,
        }

    alpha_num = float(to_scalar(solution(alpha_deg)))
    delta_a_num = float(to_scalar(solution(delta_a_deg)))
    delta_e_num = float(to_scalar(solution(delta_e_deg)))
    delta_r_num = float(to_scalar(solution(delta_r_deg)))
    aero_num = solution(aero)
    drag_num = float(to_scalar(aero_num["D"])) * drag_factor
    lift_num = float(to_scalar(aero_num["L"]))
    cl_num = float(to_scalar(aero_num["CL"]))
    sink_rate_num = drag_num * V_NOM_MPS / max(weight_n, 1e-8)
    l_over_d_num = lift_num / max(drag_num, 1e-8)

    return {
        "candidate_id": candidate.candidate_id,
        "scenario_id": scenario_id,
        "mass_scale": mass_scale,
        "cg_x_shift_mac": cg_x_shift_mac,
        "incidence_bias_deg": incidence_bias_deg,
        "control_eff": control_eff,
        "drag_factor": drag_factor,
        "trim_success": True,
        "alpha_deg": alpha_num,
        "delta_a_deg": delta_a_num,
        "delta_e_deg": delta_e_num,
        "delta_r_deg": delta_r_num,
        "sink_rate_mps": sink_rate_num,
        "L_over_D": l_over_d_num,
        "CL": cl_num,
        "D": drag_num,
        "alpha_margin_deg": max_alpha_cap - alpha_num,
        "cl_margin_to_cap": max_cl_cap - cl_num,
        "delta_e_util": abs(delta_e_num) / max(abs(DELTA_E_MAX_DEG), 1e-8),
        "delta_r_util": abs(delta_r_num) / max(abs(DELTA_R_MAX_DEG), 1e-8),
        "delta_a_util": abs(delta_a_num) / max(abs(DELTA_A_MAX_DEG), 1e-8),
    }


def sink_cvar(values: onp.ndarray, tail_fraction: float = 0.20) -> float:
    if values.size == 0:
        return float("nan")
    tail_count = max(1, int(onp.ceil(tail_fraction * values.size)))
    sorted_desc = onp.sort(values)[::-1]
    return float(onp.mean(sorted_desc[:tail_count]))


def run_robust_postcheck(
    candidates: list[Candidate],
    scenarios_df: pd.DataFrame,
    config: WorkflowConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    robust_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for scenario in scenarios_df.to_dict(orient="records"):
            robust_rows.append(
                trim_candidate_under_scenario(
                    candidate=candidate,
                    scenario_row=scenario,
                    config=config,
                )
            )

    robust_scenarios_df = pd.DataFrame(robust_rows)
    summary_rows: list[dict[str, Any]] = []
    objective_by_candidate = {candidate.candidate_id: candidate.objective for candidate in candidates}

    for candidate in candidates:
        candidate_df = robust_scenarios_df[
            robust_scenarios_df["candidate_id"] == candidate.candidate_id
        ]
        feasible_df = candidate_df[candidate_df["trim_success"] == True]

        scenario_count = max(len(candidate_df), 1)
        feasible_rate = len(feasible_df) / scenario_count
        sink_values = feasible_df["sink_rate_mps"].dropna().to_numpy(dtype=float)

        sink_mean = float(onp.mean(sink_values)) if sink_values.size else float("nan")
        sink_std = float(onp.std(sink_values)) if sink_values.size else float("nan")
        sink_worst = float(onp.max(sink_values)) if sink_values.size else float("nan")
        sink_cvar_20 = sink_cvar(sink_values, tail_fraction=0.20)
        penalty_value = sink_cvar_20 if onp.isfinite(sink_cvar_20) else 1e6
        selection_score = (1.0 - feasible_rate) * 1e3 + penalty_value

        summary_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feasible_rate": feasible_rate,
                "sink_mean": sink_mean,
                "sink_std": sink_std,
                "sink_worst": sink_worst,
                "sink_cvar_20": sink_cvar_20,
                "max_delta_e_util_worst": float(candidate_df["delta_e_util"].max(skipna=True))
                if not candidate_df.empty
                else float("nan"),
                "max_alpha_worst": float(candidate_df["alpha_deg"].max(skipna=True))
                if not candidate_df.empty
                else float("nan"),
                "min_alpha_margin_worst": float(
                    candidate_df["alpha_margin_deg"].min(skipna=True)
                )
                if not candidate_df.empty
                else float("nan"),
                "min_cl_margin_worst": float(candidate_df["cl_margin_to_cap"].min(skipna=True))
                if not candidate_df.empty
                else float("nan"),
                "selection_score": selection_score,
                "_objective_nominal": objective_by_candidate[candidate.candidate_id],
            }
        )

    robust_summary_df = pd.DataFrame(summary_rows)
    if robust_summary_df.empty:
        robust_summary_df["is_selected"] = False
        return robust_scenarios_df, robust_summary_df

    robust_summary_df["_sink_cvar_sort"] = robust_summary_df["sink_cvar_20"].fillna(onp.inf)
    ordered = robust_summary_df.sort_values(
        by=["feasible_rate", "_sink_cvar_sort", "_objective_nominal"],
        ascending=[False, True, True],
    )
    selected_candidate_id = int(ordered.iloc[0]["candidate_id"])
    robust_summary_df["is_selected"] = robust_summary_df["candidate_id"] == selected_candidate_id
    robust_summary_df = robust_summary_df.drop(columns=["_sink_cvar_sort", "_objective_nominal"])
    return robust_scenarios_df, robust_summary_df


def save_workflow_workbook(
    config: WorkflowConfig,
    candidates: list[Candidate],
    robust_scenarios_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
    selected_candidate: Candidate | None,
) -> Path:
    workflow_path = RESULTS_DIR / "nausicaa_workflow.xlsx"
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    run_info_rows = [
        {"Key": "code_version", "Value": get_git_version()},
        {"Key": "timestamp_utc", "Value": timestamp_utc},
        {"Key": "n_starts", "Value": config.n_starts},
        {"Key": "keep_top_k", "Value": config.keep_top_k},
        {"Key": "random_seed", "Value": config.random_seed},
        {"Key": "n_scenarios", "Value": config.n_scenarios},
        {"Key": "scenario_seed", "Value": config.scenario_seed},
        {"Key": "mass_scale_min", "Value": config.mass_scale_min},
        {"Key": "mass_scale_max", "Value": config.mass_scale_max},
        {"Key": "cg_x_shift_mac_min", "Value": config.cg_x_shift_mac_min},
        {"Key": "cg_x_shift_mac_max", "Value": config.cg_x_shift_mac_max},
        {"Key": "incidence_bias_deg_min", "Value": config.incidence_bias_deg_min},
        {"Key": "incidence_bias_deg_max", "Value": config.incidence_bias_deg_max},
        {"Key": "control_eff_min", "Value": config.control_eff_min},
        {"Key": "control_eff_max", "Value": config.control_eff_max},
        {"Key": "drag_factor_min", "Value": config.drag_factor_min},
        {"Key": "drag_factor_max", "Value": config.drag_factor_max},
        {"Key": "stall_alpha_margin_deg", "Value": config.stall_alpha_margin_deg},
        {"Key": "cl_margin", "Value": config.cl_margin},
        {"Key": "max_trim_util_fraction", "Value": config.max_trim_util_fraction},
    ]
    run_info_df = pd.DataFrame(run_info_rows)

    candidate_rows = []
    for candidate in candidates:
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "objective": candidate.objective,
                "wing_span_m": candidate.wing_span_m,
                "wing_chord_m": candidate.wing_chord_m,
                "tail_arm_m": candidate.tail_arm_m,
                "htail_span_m": candidate.htail_span_m,
                "vtail_height_m": candidate.vtail_height_m,
                "wing_AR": candidate.wing_span_m ** 2 / max(candidate.wing_area_m2, 1e-8),
                "alpha_deg": candidate.alpha_deg,
                "delta_a_deg": candidate.delta_a_deg,
                "delta_e_deg": candidate.delta_e_deg,
                "delta_r_deg": candidate.delta_r_deg,
                "sink_rate_mps": candidate.sink_rate_mps,
                "L_over_D": candidate.l_over_d,
                "mass_total_kg": candidate.mass_total_kg,
                "ballast_mass_kg": candidate.ballast_mass_kg,
                "static_margin": candidate.static_margin,
                "vh": candidate.vh,
                "vv": candidate.vv,
                "roll_tau_s": candidate.roll_tau_s,
                "roll_rate_ss_radps": candidate.roll_rate_ss_radps,
                "roll_accel0_rad_s2": candidate.roll_accel0_rad_s2,
                "max_servo_util": candidate.max_servo_util,
            }
        )
    candidates_df = pd.DataFrame(candidate_rows)

    scenario_columns = [
        "candidate_id",
        "scenario_id",
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "control_eff",
        "drag_factor",
        "trim_success",
        "alpha_deg",
        "delta_a_deg",
        "delta_e_deg",
        "delta_r_deg",
        "sink_rate_mps",
        "L_over_D",
        "CL",
        "D",
        "alpha_margin_deg",
        "cl_margin_to_cap",
        "delta_e_util",
        "delta_r_util",
        "delta_a_util",
    ]
    robust_scenarios_df = robust_scenarios_df.reindex(columns=scenario_columns)

    summary_columns = [
        "candidate_id",
        "feasible_rate",
        "sink_mean",
        "sink_std",
        "sink_worst",
        "sink_cvar_20",
        "max_delta_e_util_worst",
        "max_alpha_worst",
        "min_alpha_margin_worst",
        "min_cl_margin_worst",
        "selection_score",
        "is_selected",
    ]
    robust_summary_df = robust_summary_df.reindex(columns=summary_columns)

    correlation_columns = [
        "candidate_id",
        "scenario_id",
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "control_eff",
        "drag_factor",
        "alpha_deg",
        "delta_a_deg",
        "delta_e_deg",
        "delta_r_deg",
        "sink_rate_mps",
        "L_over_D",
        "CL",
        "D",
        "alpha_margin_deg",
        "cl_margin_to_cap",
        "delta_e_util",
        "delta_r_util",
        "delta_a_util",
    ]
    correlation_data_df = robust_scenarios_df[
        robust_scenarios_df["trim_success"] == True
    ].reindex(columns=correlation_columns)

    numeric_df = correlation_data_df.select_dtypes(include=["number"]).copy()
    if not numeric_df.empty:
        numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
        correlation_matrix_df = numeric_df.corr(numeric_only=True)
    else:
        correlation_matrix_df = pd.DataFrame()

    with pd.ExcelWriter(workflow_path) as writer:
        run_info_df.to_excel(writer, sheet_name="RunInfo", index=False)
        candidates_df.to_excel(writer, sheet_name="Candidates", index=False)
        robust_scenarios_df.to_excel(writer, sheet_name="RobustScenarios", index=False)
        robust_summary_df.to_excel(writer, sheet_name="RobustSummary", index=False)
        correlation_data_df.to_excel(writer, sheet_name="CorrelationData", index=False)
        correlation_matrix_df.to_excel(writer, sheet_name="CorrelationMatrix", index=True)

        if selected_candidate is not None:
            if selected_candidate.summary_rows is not None:
                pd.DataFrame(selected_candidate.summary_rows).to_excel(
                    writer,
                    sheet_name="Summary",
                    index=False,
                )
            if selected_candidate.geometry_rows is not None:
                pd.DataFrame(selected_candidate.geometry_rows).to_excel(
                    writer,
                    sheet_name="Geometry",
                    index=False,
                )
            if selected_candidate.mass_rows is not None:
                pd.DataFrame(selected_candidate.mass_rows).to_excel(
                    writer,
                    sheet_name="MassBreakdown",
                    index=False,
                )
            if selected_candidate.aero_rows is not None:
                pd.DataFrame(selected_candidate.aero_rows).to_excel(
                    writer,
                    sheet_name="Aerodynamics",
                    index=False,
                )
            if selected_candidate.constraint_rows is not None:
                pd.DataFrame(selected_candidate.constraint_rows).to_excel(
                    writer,
                    sheet_name="Constraints",
                    index=False,
                )
            if selected_candidate.boundary_rows is not None:
                pd.DataFrame(selected_candidate.boundary_rows).to_excel(
                    writer,
                    sheet_name="DesignVarBounds",
                    index=False,
                )
            if selected_candidate.design_points_rows is not None:
                pd.DataFrame(selected_candidate.design_points_rows).to_excel(
                    writer,
                    sheet_name="DesignPoints",
                    index=False,
                )

    return workflow_path


def export_selected_candidate(candidate: Candidate) -> tuple[PathMap, PathMap]:
    if (
        candidate.summary_rows is None
        or candidate.geometry_rows is None
        or candidate.mass_rows is None
        or candidate.aero_rows is None
        or candidate.constraint_rows is None
    ):
        return {}, {}

    output_paths = save_results(
        summary_rows=candidate.summary_rows,
        geometry_rows=candidate.geometry_rows,
        mass_rows=candidate.mass_rows,
        aero_rows=candidate.aero_rows,
        constraint_rows=candidate.constraint_rows,
        design_points_rows=candidate.design_points_rows,
        boundary_rows=candidate.boundary_rows,
    )

    figure_paths: PathMap = {}
    if (
        candidate.airplane is not None
        and candidate.mass_props is not None
        and candidate.total_mass is not None
    ):
        try:
            figure_paths = make_plots(
                airplane=candidate.airplane,
                mass_props=candidate.mass_props,
                total_mass=candidate.total_mass,
            )
        except Exception as exc:
            print(f"[WARN] Plot generation failed: {exc}", flush=True)

    print_console_report(
        summary_rows=candidate.summary_rows,
        geometry_rows=candidate.geometry_rows,
        constraint_rows=candidate.constraint_rows,
        boundary_rows=candidate.boundary_rows,
        output_paths=output_paths,
        figure_paths=figure_paths,
    )
    return output_paths, figure_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nausicaa glider design optimizer")
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Enable multistart and robust post-check workflow",
    )
    parser.add_argument("--n-starts", type=int, default=WorkflowConfig.n_starts)
    parser.add_argument("--keep-top-k", type=int, default=WorkflowConfig.keep_top_k)
    parser.add_argument("--random-seed", type=int, default=WorkflowConfig.random_seed)
    parser.add_argument("--n-scenarios", type=int, default=WorkflowConfig.n_scenarios)
    parser.add_argument("--scenario-seed", type=int, default=WorkflowConfig.scenario_seed)
    parser.add_argument("--dedup-span-m", type=float, default=WorkflowConfig.dedup_span_m)
    parser.add_argument("--dedup-chord-m", type=float, default=WorkflowConfig.dedup_chord_m)
    parser.add_argument(
        "--dedup-tail-arm-m",
        type=float,
        default=WorkflowConfig.dedup_tail_arm_m,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_workflow = RUN_WORKFLOW or args.workflow

    if use_workflow:
        workflow_config = WorkflowConfig(
            n_starts=max(1, int(args.n_starts)),
            keep_top_k=max(1, int(args.keep_top_k)),
            random_seed=int(args.random_seed),
            n_scenarios=max(1, int(args.n_scenarios)),
            scenario_seed=int(args.scenario_seed),
            dedup_span_m=max(1e-9, float(args.dedup_span_m)),
            dedup_chord_m=max(1e-9, float(args.dedup_chord_m)),
            dedup_tail_arm_m=max(1e-9, float(args.dedup_tail_arm_m)),
        )
        print(
            (
                "Starting workflow mode "
                f"(n_starts={workflow_config.n_starts}, "
                f"keep_top_k={workflow_config.keep_top_k}, "
                f"n_scenarios={workflow_config.n_scenarios})"
            ),
            flush=True,
        )

        candidates = run_multistart(workflow_config)
        if not candidates:
            print("No feasible candidate found in multistart run", flush=True)
            return

        scenarios_df = sample_scenarios(workflow_config)
        robust_scenarios_df, robust_summary_df = run_robust_postcheck(
            candidates=candidates,
            scenarios_df=scenarios_df,
            config=workflow_config,
        )

        selected_ids = robust_summary_df.loc[
            robust_summary_df["is_selected"] == True,
            "candidate_id",
        ]
        if selected_ids.empty:
            selected_candidate = min(candidates, key=lambda item: item.objective)
        else:
            selected_candidate_id = int(selected_ids.iloc[0])
            selected_candidate = next(
                item for item in candidates if item.candidate_id == selected_candidate_id
            )

        workflow_path = save_workflow_workbook(
            config=workflow_config,
            candidates=candidates,
            robust_scenarios_df=robust_scenarios_df,
            robust_summary_df=robust_summary_df,
            selected_candidate=selected_candidate,
        )
        _output_paths, _figure_paths = export_selected_candidate(selected_candidate)
        print(f"Workflow workbook saved: {workflow_path}", flush=True)
        print(f"Selected candidate id: {selected_candidate.candidate_id}", flush=True)
        return

    candidate = legacy_single_run_main(
        init_override=default_initial_guess(),
        ipopt_options=None,
        export_outputs=True,
    )
    if candidate is None:
        return


if __name__ == "__main__":
    main()
