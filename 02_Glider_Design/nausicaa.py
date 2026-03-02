from __future__ import annotations

import argparse
import copy
import json
import subprocess
from dataclasses import asdict, dataclass
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

GENERATE_POLARS = False
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
V_TURN_MPS = 4.0
V_NOM_MPS = 5.0

# Manoeuvre definition (coordinated, banked turn feasibility)
TURN_BANK_DEG = 50.0
WALL_CLEARANCE_M = 0.50
TURN_DEFLECTION_UTIL_MAX = 0.80
TURN_LATERAL_TRIM_TOL_CL = 0.02
TURN_LATERAL_TRIM_TOL_CN = 0.02
# Manoeuvre agility target: time to reach design bank angle at V_TURN_MPS.
# Converted to a minimum steady-state roll-rate requirement.
BANK_ENTRY_TIME_S = 0.8

# Stall / margin settings for manoeuvre case
TURN_ALPHA_MARGIN_DEG = 4.0
TURN_CL_CAP = 1.00
K_LEVEL_TURN = 0.95

# Trim operating-point envelope
DESIGN_SPEED_MPS = V_NOM_MPS
ALPHA_MIN_DEG = -4.0
ALPHA_MAX_DEG = 8.0
STALL_ALPHA_LIMIT_DEG = 14.0
ALPHA_MAX_TURN_DEG = STALL_ALPHA_LIMIT_DEG - TURN_ALPHA_MARGIN_DEG
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
NOSE_X_M = -0.11

# Vertical reference definitions (meters)
WING_ROOT_LOWER_SURFACE_Z_M = 0.004
VTAIL_ROOT_LOWER_SURFACE_Z_M = 0.004
HTAIL_ROOT_LOWER_SURFACE_Z_M = -0.004

# Wing design bounds
WING_SPAN_MIN_M = 0.30
WING_SPAN_MAX_M = 2.00
WING_CHORD_MIN_M = 0.10
WING_CHORD_MAX_M = 0.30

# Tail / boom design bounds
BOOM_LENGTH_MIN_M = 0.35
BOOM_LENGTH_MAX_M = 0.80
# Tail-arm position bounds are kept broad; boom-length constraint is authoritative.
TAIL_ARM_MIN_M = 0.20
TAIL_ARM_MAX_M = 0.80
HT_SPAN_MIN_M = 0.20
HT_SPAN_MAX_M = 0.50
VT_HEIGHT_MIN_M = 0.10
VT_HEIGHT_MAX_M = 0.30

# Tail aspect-ratio assumptions
HT_AR = 4.0
VT_AR = 2.0

# Mesh density used for AeroSandbox lifting surfaces
N_WING_XSECS = 7
N_TAIL_XSECS = 5

# Control-surface geometry
AILERON_ETA_INBOARD = 0.30
AILERON_ETA_OUTBOARD = 0.85
AILERON_CHORD_FRACTION = 0.30
ELEVATOR_CHORD_FRACTION = 0.30
RUDDER_CHORD_FRACTION = 0.35

# Mass model assumptions (Depron/XPS foam)
# Datasheets provide compressive stress at 10% deformation (sigma_d10),
# so use a conservative secant stiffness indicator: E_sec10 = sigma_d10 / 0.10.
FOAM_RHO_G3_KG_M3 = 40.0
FOAM_SIGMA_D10_G3_PA = 100e3
FOAM_ESEC10_G3_PA = FOAM_SIGMA_D10_G3_PA / 0.10

FOAM_RHO_G6_KG_M3 = 33.0
FOAM_SIGMA_D10_G6_PA = 150e3
FOAM_ESEC10_G6_PA = FOAM_SIGMA_D10_G6_PA / 0.10

WING_THICKNESS_M = 0.006
TAIL_THICKNESS_M = 0.003
VTAIL_THICKNESS_M = TAIL_THICKNESS_M

# Use density by sheet thickness (wing=6 mm, tails=3 mm)
WING_DENSITY_KG_M3 = FOAM_RHO_G6_KG_M3
TAIL_DENSITY_KG_M3 = FOAM_RHO_G3_KG_M3

# Effective stiffness indicator for rigid-aero wing deflection proxy.
WING_E_SECANT_PA = FOAM_ESEC10_G6_PA

FUSE_RADIUS_M = 0.002
# Boom: rod-only mass model
BOOM_TUBE_OUTER_DIAMETER_M = 0.004
BOOM_TUBE_INNER_DIAMETER_M = 0.002
BOOM_ROD_DENSITY_KG_M3 = 1400.0
BOOM_EXTRA_MASS_KG = 0.0
BOOM_END_BEFORE_ELEV_FRAC = 0.70

# Wing spar (carry-through) + filament tape reinforcement
WING_SPAR_ENABLE = True
WING_SPAR_X_FRAC = 0.30
WING_SPAR_OD_M = BOOM_TUBE_OUTER_DIAMETER_M
WING_SPAR_ID_M = BOOM_TUBE_INNER_DIAMETER_M
WING_SPAR_RHO_KG_M3 = BOOM_ROD_DENSITY_KG_M3
WING_SPAR_E_FLEX_PA = 25e9
WING_SPAR_Z_FROM_LOWER_M = 0.002
WING_SPAR_SLOT_W_M = 0.004
WING_SPAR_SLOT_H_M = 0.004
SLOT_VOID_MASS_FRAC_MAX = 0.05

TAPE_ENABLE_WING = True
TAPE_ENABLE_TAIL = True
TAPE_WIDTH_M = 0.050
TAPE_THICKNESS_M = 0.00013
TAPE_AREAL_DENSITY_KG_M2 = 0.12
TAPE_E_EFFECTIVE_PA = 10e9
TAPE_EFFICIENCY = 1.0
VTAIL_TAPE_Y_OFFSET_M = 0.5 * VTAIL_THICKNESS_M + 0.5 * TAPE_THICKNESS_M

# Discrete hardware modules
CENTRE_MODULE_MASS_KG = 0.032
TAIL_MODULE_MASS_KG = 0.004
CENTRE_MODULE_VOLUME_M3 = 0.00002206
TAIL_MODULE_VOLUME_M3 = 0.00000361
CENTRE_MODULE_RHO_KG_M3 = CENTRE_MODULE_MASS_KG / CENTRE_MODULE_VOLUME_M3
TAIL_MODULE_RHO_KG_M3 = TAIL_MODULE_MASS_KG / TAIL_MODULE_VOLUME_M3
TAIL_SUPPORT_MASS_KG = 0.001
# CAD-derived core CG locations for module solids excluding mounts
CENTRE_CORE_X_OFFSET_FROM_0p3C_M = 0.0256
CENTRE_CORE_Z_CG_M = -0.00727
TAIL_CORE_X_OFFSET_FROM_BOOM_END_M = 0.0231
TAIL_CORE_Z_CG_M = -0.0005
# X-location rules for aft hardware
TAIL_GEAR_X_OFFSET_FROM_HTAIL_LE_M = -0.00894
TAIL_SUPPORT_Z_CG_M = -0.00587
VTAIL_MOUNT_T_M = 0.002
VTAIL_MOUNT_LB_M = 0.019
VTAIL_MOUNT_LT_M = 0.0095
VTAIL_MOUNT_H_M = 0.0215
VTAIL_MOUNT_X0_OFFSET_FROM_BOOM_END_M = 0.0383
VTAIL_MOUNT_ROOT_LOWER_Z_M = VTAIL_ROOT_LOWER_SURFACE_Z_M

GLUE_FRACTION = 0.08
BALLAST_MAX_KG = 0.025

# Avionics / hardware masses (kg)
BATTERY_MASS_KG = 0.0090
BATTERY_DIM_X_M = 0.047
BATTERY_DIM_Y_M = 0.017
BATTERY_DIM_Z_M = 0.005
RECEIVER_MASS_KG = 0.0050
FLIGHT_CONTROLLER_MASS_KG = 0.0050
REGULATOR_MASS_KG = 0.0004
SERVO_MASS_KG = 0.0022

# Servo layout (4 total): 2 aileron + elevator + rudder
AILERON_SERVO_SPAN_FRAC = 0.40
AILERON_SERVO_X_CHORD_FRAC = 0.30
AILERON_SERVO_Z_OFFSET_M = 0.0
SERVO_CENTERLINE_BASE_Z_M = -0.0065
ELEVATOR_SERVO_X_OFFSET_FROM_CENTRE_CORE_M = 0.0
ELEVATOR_SERVO_Z_OFFSET_FROM_AVIONICS_M = 0.0
RUDDER_SERVO_X_OFFSET_FROM_CENTRE_CORE_M = 0.0
RUDDER_SERVO_Z_OFFSET_FROM_AVIONICS_M = 0.0

# Battery sliding range
BATTERY_X_MAX_FRAC = 0.60
BATTERY_FORE_OFFSET_FROM_CENTRE_MODULE_M = 0.035
AVIONICS_Z_CG_M = -0.008
REGULATOR_X_OFFSET_FROM_BATTERY_M = 0.040
RECEIVER_X_OFFSET_FROM_REGULATOR_M = 0.035
FLIGHT_CONTROLLER_X_OFFSET_FROM_BATTERY_M = 0.015

# Static-stability design window
STATIC_MARGIN_MIN = 0.05
STATIC_MARGIN_MAX = 0.15
VH_MIN = 0.50
VH_MAX = 1.00
VV_MIN = 0.03
VV_MAX = 0.08

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
CL_DELTA_A_PROXY_MAX = 0.45
CLP_NEG_EPS = 1e-4
INCLUDE_SERVO_RATE_IN_BANK_ENTRY = True
SERVO_RATE_DEG_S = 400.0
CL_DELTA_A_FD_STEP_DEG = 2.0

# Servo sizing assumptions
# Update from servo datasheet (N*m)
SERVO_TORQUE_LIMIT_NM = 0.009
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
STRUCT_DEFLECTION_WEIGHT = 5
ROLL_TAU_WEIGHT_IN_OBJECTIVE = 0.05
WING_DEFLECTION_ALLOW_FRAC = 0.08
# Horizontal-tail stiffness proxy (soft regularizer)
HT_LOAD_FRACTION = 0.25            # k_H in F = k_H * n_turn * W (start 0.20-0.35)
HT_DEFLECTION_ALLOW_FRAC = 0.08    # allowed tip deflection fraction of semispan
HT_STRUCT_DEFLECTION_WEIGHT = 1
HTAIL_E_SECANT_PA = FOAM_ESEC10_G3_PA  # tail foam secant modulus (3 mm Depron proxy)
SOFTPLUS_K = 25.0
BOUNDARY_HIT_REL_TOL = 1e-3
BOUNDARY_HIT_ABS_TOL = 1e-6
ACTIVE_SET_ABS_TOL = 1e-3

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
    n_starts: int = 30
    keep_top_k: int = 10
    random_seed: int = 20
    n_scenarios: int = 50
    scenario_seed: int = 30
    dedup_span_m: float = 0.01
    dedup_chord_m: float = 0.005
    dedup_tail_arm_m: float = 0.01
    mass_scale_min: float = 0.90
    mass_scale_max: float = 1.10
    cg_x_shift_mac_min: float = -0.06
    cg_x_shift_mac_max: float = 0.06
    incidence_bias_deg_min: float = -2.0
    incidence_bias_deg_max: float = 2.0
    # Legacy scalar control effectiveness (kept for backward compatibility only)
    control_eff_min: float = 0.70
    control_eff_max: float = 1.00
    # Per-axis control effectiveness in robust Monte-Carlo
    eff_a_min: float = 0.85
    eff_a_max: float = 1.00
    eff_e_min: float = 0.85
    eff_e_max: float = 1.00
    eff_r_min: float = 0.85
    eff_r_max: float = 1.00
    # Neutral-bias uncertainty (deg)
    bias_a_deg_min: float = -3.0
    bias_a_deg_max: float = 3.0
    bias_e_deg_min: float = -3.0
    bias_e_deg_max: float = 3.0
    bias_r_deg_min: float = -3.0
    bias_r_deg_max: float = 3.0
    # Inertia uncertainty scales
    ixx_scale_min: float = 0.90
    ixx_scale_max: float = 1.10
    iyy_scale_min: float = 0.90
    iyy_scale_max: float = 1.10
    izz_scale_min: float = 0.90
    izz_scale_max: float = 1.10
    # Structural stiffness / thickness uncertainty scales
    wing_E_scale_min: float = 0.80
    wing_E_scale_max: float = 1.30
    htail_E_scale_min: float = 0.80
    htail_E_scale_max: float = 1.30
    wing_thickness_scale_min: float = 1.00
    wing_thickness_scale_max: float = 1.00
    tail_thickness_scale_min: float = 1.00
    tail_thickness_scale_max: float = 1.00
    # Actuator-rate proxy settings
    servo_rate_deg_s: float = SERVO_RATE_DEG_S
    nom_trim_time_s: float = 1.5
    turn_trim_time_s: float = BANK_ENTRY_TIME_S
    rate_util_fraction: float = 0.80
    # Updraft disturbance model
    w_gust_nom_min: float = -0.30
    w_gust_nom_max: float = 0.30
    w_gust_turn_min: float = -0.20
    w_gust_turn_max: float = 0.20
    drag_factor_min: float = 1.00
    drag_factor_max: float = 1.25
    # Nominal-point robust margins
    stall_alpha_margin_deg: float = 2.0
    cl_margin: float = 0.15
    # Turn-point robust margins
    turn_alpha_margin_deg: float = TURN_ALPHA_MARGIN_DEG
    turn_cl_margin: float = 0.05
    turn_deflection_util: float = TURN_DEFLECTION_UTIL_MAX
    # Legacy trim utilization bound (used by nominal robust trim commands)
    max_trim_util_fraction: float = 1.00


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
    active_constraint_rows: ReportRows | None = None
    boundary_rows: ReportRows | None = None
    design_points_rows: ReportRows | None = None
    solver_stats: dict[str, Any] | None = None
    wing_area_m2: float = float("nan")
    wing_mac_m: float = float("nan")
    airfoil_label: str = ""


def default_initial_guess() -> dict[str, float]:
    return {
        "wing_span_m": 0.78,
        "wing_chord_m": 0.22,
        "tail_arm_m": 0.45,
        "htail_span_m": 0.38,
        "vtail_height_m": 0.16,
        "alpha_nom_deg": 5.0,
        "delta_a_nom_deg": 0.0,
        "delta_e_nom_deg": 1.0,
        "delta_r_nom_deg": 0.0,
        "alpha_turn_deg": 7.0,
        "delta_a_turn_deg": 0.0,
        "delta_e_turn_deg": 3.5,
        "delta_r_turn_deg": 0.0,
        # Legacy aliases retained for backward compatibility with old init files.
        "alpha_deg": 5.0,
        "delta_a_deg": 0.0,
        "delta_e_deg": 1.0,
        "delta_r_deg": 0.0,
    }


_AIRFOIL_CACHE: tuple[asb.Airfoil, str] | None = None
_LAST_SOLVE_FAILURE_REASON: str | None = None


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
    wing_root_centerline_z_m = WING_ROOT_LOWER_SURFACE_Z_M + 0.5 * WING_THICKNESS_M
    for eta in onp.linspace(0.0, 1.0, N_WING_XSECS):
        # Straight rectangular planform with fixed dihedral
        y_le = eta * span_m / 2.0
        z_le = wing_root_centerline_z_m + y_le * np.tan(np.radians(DIHEDRAL_DEG))
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
    htail_root_centerline_z_m = HTAIL_ROOT_LOWER_SURFACE_Z_M + 0.5 * TAIL_THICKNESS_M
    for eta in onp.linspace(0.0, 1.0, N_TAIL_XSECS):
        y_le = eta * span_m / 2.0
        xsecs.append(
            asb.WingXSec(
                xyz_le=[tail_arm_m, y_le, htail_root_centerline_z_m],
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
        z_le = VTAIL_ROOT_LOWER_SURFACE_Z_M + eta * height_m
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


def build_fuselage(boom_end_x_m: Scalar) -> asb.Fuselage:
    # Lightweight boom/avionics tray representation with nose and tail stations
    tail_x_m = boom_end_x_m
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


def tube_area_m2(od_m: Scalar, id_m: Scalar) -> Scalar:
    return (np.pi / 4.0) * (od_m**2 - id_m**2)


def tube_I_m4(od_m: Scalar, id_m: Scalar) -> Scalar:
    # Second moment of area about tube centroid for bending in either transverse plane.
    return (np.pi / 64.0) * (od_m**4 - id_m**4)


def mass_properties_rect_prism(
    mass_kg: Scalar,
    dim_x_m: Scalar,
    dim_y_m: Scalar,
    dim_z_m: Scalar,
    x_cg_m: Scalar,
    y_cg_m: Scalar,
    z_cg_m: Scalar,
) -> asb.MassProperties:
    # Uniform rectangular prism with body axes aligned to global x/y/z.
    m = mass_kg
    i_xx = (m / 12.0) * (dim_y_m**2 + dim_z_m**2)
    i_yy = (m / 12.0) * (dim_x_m**2 + dim_z_m**2)
    i_zz = (m / 12.0) * (dim_x_m**2 + dim_y_m**2)
    return asb.MassProperties(
        mass=m,
        x_cg=x_cg_m,
        y_cg=y_cg_m,
        z_cg=z_cg_m,
        Ixx=i_xx,
        Iyy=i_yy,
        Izz=i_zz,
        Ixy=0.0,
        Ixz=0.0,
        Iyz=0.0,
    )


def mass_properties_spanwise_tube(
    length_m: Scalar,
    od_m: Scalar,
    id_m: Scalar,
    density_kg_m3: float,
    x_cg_m: Scalar,
    y_cg_m: Scalar,
    z_cg_m: Scalar,
) -> asb.MassProperties:
    # Tube axis aligned with global y (spanwise).
    area_m2 = tube_area_m2(od_m, id_m)
    mass_kg = density_kg_m3 * area_m2 * length_m

    r_outer_m = 0.5 * od_m
    r_inner_m = 0.5 * id_m
    r2 = r_outer_m**2 + r_inner_m**2

    i_yy = 0.5 * mass_kg * r2
    i_xx = (mass_kg / 12.0) * (length_m**2 + 3.0 * r2)
    i_zz = i_xx

    return asb.MassProperties(
        mass=mass_kg,
        x_cg=x_cg_m,
        y_cg=y_cg_m,
        z_cg=z_cg_m,
        Ixx=i_xx,
        Iyy=i_yy,
        Izz=i_zz,
        Ixy=0.0,
        Ixz=0.0,
        Iyz=0.0,
    )


def mass_properties_x_axis_tube(
    length_m: Scalar,
    od_m: Scalar,
    id_m: Scalar,
    density_kg_m3: float,
    x_cg_m: Scalar,
    y_cg_m: Scalar,
    z_cg_m: Scalar,
) -> asb.MassProperties:
    # Tube axis aligned with global x.
    area_m2 = tube_area_m2(od_m, id_m)
    mass_kg = density_kg_m3 * area_m2 * length_m

    r_outer_m = 0.5 * od_m
    r_inner_m = 0.5 * id_m
    r2 = r_outer_m**2 + r_inner_m**2

    i_xx = 0.5 * mass_kg * r2
    i_yy = (mass_kg / 12.0) * (length_m**2 + 3.0 * r2)
    i_zz = i_yy

    return asb.MassProperties(
        mass=mass_kg,
        x_cg=x_cg_m,
        y_cg=y_cg_m,
        z_cg=z_cg_m,
        Ixx=i_xx,
        Iyy=i_yy,
        Izz=i_zz,
        Ixy=0.0,
        Ixz=0.0,
        Iyz=0.0,
    )


def trapezoid_area_m2(base_bottom_m: Scalar, base_top_m: Scalar, height_m: Scalar) -> Scalar:
    return 0.5 * (base_bottom_m + base_top_m) * height_m


def trapezoid_centroid_from_base_m(
    base_bottom_m: Scalar,
    base_top_m: Scalar,
    height_m: Scalar,
) -> Scalar:
    # Distance from the bottom base to the area centroid.
    return height_m * (base_bottom_m + 2.0 * base_top_m) / (
        3.0 * (base_bottom_m + base_top_m)
    )


def polygon_area_centroid_moments_2d(
    vertices_xy: list[tuple[float, float]],
) -> tuple[float, float, float, float, float, float]:
    # Signed-area polygon formulas for centroid and second moments.
    area_twice = 0.0
    c_x_num = 0.0
    c_y_num = 0.0
    i_xx_origin = 0.0
    i_yy_origin = 0.0
    i_xy_origin = 0.0

    n = len(vertices_xy)
    for i in range(n):
        x0, y0 = vertices_xy[i]
        x1, y1 = vertices_xy[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area_twice += cross
        c_x_num += (x0 + x1) * cross
        c_y_num += (y0 + y1) * cross
        i_xx_origin += (y0**2 + y0 * y1 + y1**2) * cross
        i_yy_origin += (x0**2 + x0 * x1 + x1**2) * cross
        i_xy_origin += (2.0 * x0 * y0 + x0 * y1 + x1 * y0 + 2.0 * x1 * y1) * cross

    area = 0.5 * area_twice
    if abs(area) < 1e-12:
        raise ValueError("Degenerate polygon area in moment computation.")

    c_x = c_x_num / (6.0 * area)
    c_y = c_y_num / (6.0 * area)
    i_xx_origin /= 12.0
    i_yy_origin /= 12.0
    i_xy_origin /= 24.0

    if area < 0.0:
        area = -area
        i_xx_origin = -i_xx_origin
        i_yy_origin = -i_yy_origin
        i_xy_origin = -i_xy_origin

    i_xx_centroid = i_xx_origin - area * c_y**2
    i_yy_centroid = i_yy_origin - area * c_x**2
    i_xy_centroid = i_xy_origin - area * c_x * c_y
    return area, c_x, c_y, i_xx_centroid, i_yy_centroid, i_xy_centroid


def mass_properties_isosceles_trapezoid_prism(
    mass_kg: Scalar,
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
    thickness_m: float,
    x_cg_m: Scalar,
    y_cg_m: Scalar,
    z_cg_m: Scalar,
    height_axis: Literal["y", "z"],
) -> asb.MassProperties:
    # Exact prism inertia from 2D trapezoid polygon moments + extrusion thickness.
    verts = [
        (-0.5 * float(base_bottom_m), 0.0),
        (+0.5 * float(base_bottom_m), 0.0),
        (+0.5 * float(base_top_m), float(height_m)),
        (-0.5 * float(base_top_m), float(height_m)),
    ]
    area, _cx, _cy, i_uu_area, i_vv_area, i_uv_area = polygon_area_centroid_moments_2d(verts)
    t = float(thickness_m)
    i_u = mass_kg * (i_uu_area / area + t**2 / 12.0)
    i_v = mass_kg * (i_vv_area / area + t**2 / 12.0)
    i_w = mass_kg * ((i_uu_area + i_vv_area) / area)
    i_uv = -mass_kg * (i_uv_area / area)

    if height_axis == "y":
        return asb.MassProperties(
            mass=mass_kg,
            x_cg=x_cg_m,
            y_cg=y_cg_m,
            z_cg=z_cg_m,
            Ixx=i_u,
            Iyy=i_v,
            Izz=i_w,
            Ixy=i_uv,
            Ixz=0.0,
            Iyz=0.0,
        )

    return asb.MassProperties(
        mass=mass_kg,
        x_cg=x_cg_m,
        y_cg=y_cg_m,
        z_cg=z_cg_m,
        Ixx=i_u,
        Iyy=i_w,
        Izz=i_v,
        Ixy=0.0,
        Ixz=i_uv,
        Iyz=0.0,
    )


def mean_abs_span_location_uniform(span_m: Scalar) -> Scalar:
    # E[|y|] for a symmetric uniform spanwise distribution.
    return 0.25 * span_m


def dihedral_mean_z_offset_m(span_m: Scalar, dihedral_deg: float) -> Scalar:
    return mean_abs_span_location_uniform(span_m) * np.tan(np.radians(dihedral_deg))


def combine_mass_properties(mps: list[asb.MassProperties]) -> asb.MassProperties:
    if len(mps) == 0:
        return asb.MassProperties(
            mass=0.0,
            x_cg=0.0,
            y_cg=0.0,
            z_cg=0.0,
            Ixx=0.0,
            Iyy=0.0,
            Izz=0.0,
            Ixy=0.0,
            Ixz=0.0,
            Iyz=0.0,
        )

    total_mass = 0.0
    x_moment = 0.0
    y_moment = 0.0
    z_moment = 0.0
    for mp in mps:
        m = mp.mass
        total_mass = total_mass + m
        x_moment = x_moment + m * mp.xyz_cg[0]
        y_moment = y_moment + m * mp.xyz_cg[1]
        z_moment = z_moment + m * mp.xyz_cg[2]

    x_cg = x_moment / total_mass
    y_cg = y_moment / total_mass
    z_cg = z_moment / total_mass

    i_xx = 0.0
    i_yy = 0.0
    i_zz = 0.0
    i_xy = 0.0
    i_xz = 0.0
    i_yz = 0.0
    for mp in mps:
        m = mp.mass
        dx = mp.xyz_cg[0] - x_cg
        dy = mp.xyz_cg[1] - y_cg
        dz = mp.xyz_cg[2] - z_cg
        inertia_tensor = mp.inertia_tensor

        i_xx = i_xx + inertia_tensor[0, 0] + m * (dy**2 + dz**2)
        i_yy = i_yy + inertia_tensor[1, 1] + m * (dx**2 + dz**2)
        i_zz = i_zz + inertia_tensor[2, 2] + m * (dx**2 + dy**2)
        i_xy = i_xy + inertia_tensor[0, 1] - m * dx * dy
        i_xz = i_xz + inertia_tensor[0, 2] - m * dx * dz
        i_yz = i_yz + inertia_tensor[1, 2] - m * dy * dz

    return asb.MassProperties(
        mass=total_mass,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg,
        Ixx=i_xx,
        Iyy=i_yy,
        Izz=i_zz,
        Ixy=i_xy,
        Ixz=i_xz,
        Iyz=i_yz,
    )


def scale_mass_properties(mp: asb.MassProperties, scale: Scalar) -> asb.MassProperties:
    inertia_tensor = mp.inertia_tensor
    return asb.MassProperties(
        mass=scale * mp.mass,
        x_cg=mp.xyz_cg[0],
        y_cg=mp.xyz_cg[1],
        z_cg=mp.xyz_cg[2],
        Ixx=scale * inertia_tensor[0, 0],
        Iyy=scale * inertia_tensor[1, 1],
        Izz=scale * inertia_tensor[2, 2],
        Ixy=scale * inertia_tensor[0, 1],
        Ixz=scale * inertia_tensor[0, 2],
        Iyz=scale * inertia_tensor[1, 2],
    )


def composite_EI_flapwise(
    chord_m: Scalar,
    foam_thickness_m: Scalar,
    e_foam_pa: Scalar,
    spar_od_m: Scalar,
    spar_id_m: Scalar,
    e_spar_pa: Scalar,
    spar_z_from_lower_m: Scalar,
    include_spar: bool,
    tape_width_m: Scalar,
    tape_thickness_m: Scalar,
    e_tape_pa: Scalar,
    include_tape: bool,
) -> tuple[Scalar, Scalar]:
    # E-weighted transformed section; z measured from the lower surface.
    t = np.maximum(foam_thickness_m, 1e-9)
    c = np.maximum(chord_m, 1e-9)

    a_foam = c * t
    z_foam = 0.5 * t
    i_foam = c * (t**3) / 12.0

    a_spar = tube_area_m2(spar_od_m, spar_id_m)
    z_spar = spar_z_from_lower_m
    i_spar = tube_I_m4(spar_od_m, spar_id_m)

    tape_w_m = np.minimum(np.maximum(tape_width_m, 0.0), c)
    tape_t_m = np.maximum(tape_thickness_m, 1e-9)
    a_tape = tape_w_m * tape_t_m
    i_tape = tape_w_m * (tape_t_m**3) / 12.0
    z_tape_bottom = 0.5 * tape_t_m
    z_tape_top = t - 0.5 * tape_t_m

    s_spar = 1.0 if include_spar else 0.0
    s_tape = 1.0 if include_tape else 0.0

    ea_sum = (
        e_foam_pa * a_foam
        + s_spar * (e_spar_pa * a_spar)
        + s_tape * (e_tape_pa * a_tape)
        + s_tape * (e_tape_pa * a_tape)
    )
    eaz_sum = (
        e_foam_pa * a_foam * z_foam
        + s_spar * (e_spar_pa * a_spar * z_spar)
        + s_tape * (e_tape_pa * a_tape * z_tape_bottom)
        + s_tape * (e_tape_pa * a_tape * z_tape_top)
    )
    z0 = eaz_sum / np.maximum(ea_sum, 1e-16)

    ei = (
        e_foam_pa * (i_foam + a_foam * (z_foam - z0) ** 2)
        + s_spar * (e_spar_pa * (i_spar + a_spar * (z_spar - z0) ** 2))
        + s_tape * (e_tape_pa * (i_tape + a_tape * (z_tape_bottom - z0) ** 2))
        + s_tape * (e_tape_pa * (i_tape + a_tape * (z_tape_top - z0) ** 2))
    )
    return ei, z0


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
    boom_end_x_m: Scalar,
    vtail_chord_m: Scalar,
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
        density_kg_m3=TAIL_DENSITY_KG_M3,
        thickness_m=TAIL_THICKNESS_M,
        span_axis="y",
    )
    mass_props["vtail_surfaces"] = flat_plate_mass_properties(
        surface=vtail,
        density_kg_m3=TAIL_DENSITY_KG_M3,
        thickness_m=TAIL_THICKNESS_M,
        span_axis="z",
    )

    wing_span_m = surface_span(wing, span_axis="y")
    z_wing_lower_m = WING_ROOT_LOWER_SURFACE_Z_M
    x_spar_m = WING_SPAR_X_FRAC * wing_chord_m
    z_spar_m = z_wing_lower_m + WING_SPAR_Z_FROM_LOWER_M
    z_dihedral_mean_m = dihedral_mean_z_offset_m(wing_span_m, DIHEDRAL_DEG)

    if WING_SPAR_ENABLE:
        mass_props["wing_spar_tube"] = mass_properties_spanwise_tube(
            length_m=wing_span_m,
            od_m=WING_SPAR_OD_M,
            id_m=WING_SPAR_ID_M,
            density_kg_m3=WING_SPAR_RHO_KG_M3,
            x_cg_m=x_spar_m,
            y_cg_m=0.0,
            z_cg_m=z_spar_m + z_dihedral_mean_m,
        )

        slot_vol_m3 = WING_SPAR_SLOT_W_M * WING_SPAR_SLOT_H_M * wing_span_m
        # Geometric foam mass represented by the slot volume.
        slot_geometric_mass_kg = WING_DENSITY_KG_M3 * slot_vol_m3
        wing_skin_mass_kg = mass_props["wing"].mass
        # Capped removed mass to avoid excessive subtraction in thin-wing cases.
        slot_removed_mass_kg = np.minimum(
            slot_geometric_mass_kg,
            SLOT_VOID_MASS_FRAC_MAX * np.maximum(wing_skin_mass_kg, 0.0),
        )
        z_slot_cg_m = z_wing_lower_m + 0.5 * WING_SPAR_SLOT_H_M
        mass_props["wing_spar_slot_void"] = mass_properties_rect_prism(
            mass_kg=-slot_removed_mass_kg,
            dim_x_m=WING_SPAR_SLOT_W_M,
            dim_y_m=wing_span_m,
            dim_z_m=WING_SPAR_SLOT_H_M,
            x_cg_m=x_spar_m,
            y_cg_m=0.0,
            z_cg_m=z_slot_cg_m + z_dihedral_mean_m,
        )

    if TAPE_ENABLE_WING:
        tape_w_wing_m = np.minimum(TAPE_WIDTH_M, wing_chord_m)
        tape_area_one_side_m2 = wing_span_m * tape_w_wing_m
        tape_mass_one_side_kg = TAPE_AREAL_DENSITY_KG_M2 * tape_area_one_side_m2
        mass_props["wing_tape_bottom"] = mass_properties_rect_prism(
            mass_kg=tape_mass_one_side_kg,
            dim_x_m=tape_w_wing_m,
            dim_y_m=wing_span_m,
            dim_z_m=TAPE_THICKNESS_M,
            x_cg_m=x_spar_m,
            y_cg_m=0.0,
            z_cg_m=z_wing_lower_m + 0.5 * TAPE_THICKNESS_M + z_dihedral_mean_m,
        )
        mass_props["wing_tape_top"] = mass_properties_rect_prism(
            mass_kg=tape_mass_one_side_kg,
            dim_x_m=tape_w_wing_m,
            dim_y_m=wing_span_m,
            dim_z_m=TAPE_THICKNESS_M,
            x_cg_m=x_spar_m,
            y_cg_m=0.0,
            z_cg_m=(
                z_wing_lower_m
                + WING_THICKNESS_M
                - 0.5 * TAPE_THICKNESS_M
                + z_dihedral_mean_m
            ),
        )

    if TAPE_ENABLE_TAIL:
        x_le_tail_m = tail_arm_m

        htail_span_m = surface_span(htail, span_axis="y")
        htail_area_m2 = htail.area()
        htail_chord_est_m = htail_area_m2 / np.maximum(htail_span_m, 1e-8)
        tape_w_htail_m = np.minimum(TAPE_WIDTH_M, htail_chord_est_m)
        tape_area_one_side_ht_m2 = htail_span_m * tape_w_htail_m
        tape_mass_one_side_ht_kg = TAPE_AREAL_DENSITY_KG_M2 * tape_area_one_side_ht_m2
        z_htail_lower_m = HTAIL_ROOT_LOWER_SURFACE_Z_M
        htail_tape_bottom = mass_properties_rect_prism(
            mass_kg=tape_mass_one_side_ht_kg,
            dim_x_m=tape_w_htail_m,
            dim_y_m=htail_span_m,
            dim_z_m=TAPE_THICKNESS_M,
            x_cg_m=x_le_tail_m + 0.5 * tape_w_htail_m,
            y_cg_m=0.0,
            z_cg_m=z_htail_lower_m + 0.5 * TAPE_THICKNESS_M,
        )
        htail_tape_top = mass_properties_rect_prism(
            mass_kg=tape_mass_one_side_ht_kg,
            dim_x_m=tape_w_htail_m,
            dim_y_m=htail_span_m,
            dim_z_m=TAPE_THICKNESS_M,
            x_cg_m=x_le_tail_m + 0.5 * tape_w_htail_m,
            y_cg_m=0.0,
            z_cg_m=z_htail_lower_m + TAIL_THICKNESS_M - 0.5 * TAPE_THICKNESS_M,
        )
        mass_props["htail_tape"] = combine_mass_properties(
            [htail_tape_bottom, htail_tape_top]
        )

        vtail_span_m = surface_span(vtail, span_axis="z")
        vtail_chord_use_m = vtail_chord_m
        tape_w_vtail_m = np.minimum(TAPE_WIDTH_M, vtail_chord_use_m)
        tape_area_one_side_vt_m2 = vtail_span_m * tape_w_vtail_m
        tape_mass_one_side_vt_kg = TAPE_AREAL_DENSITY_KG_M2 * tape_area_one_side_vt_m2
        z_vtail_mid_m = VTAIL_ROOT_LOWER_SURFACE_Z_M + 0.5 * vtail_span_m
        vtail_tape_side_a = mass_properties_rect_prism(
            mass_kg=tape_mass_one_side_vt_kg,
            dim_x_m=tape_w_vtail_m,
            dim_y_m=TAPE_THICKNESS_M,
            dim_z_m=vtail_span_m,
            x_cg_m=x_le_tail_m + 0.5 * tape_w_vtail_m,
            y_cg_m=VTAIL_TAPE_Y_OFFSET_M,
            z_cg_m=z_vtail_mid_m,
        )
        vtail_tape_side_b = mass_properties_rect_prism(
            mass_kg=tape_mass_one_side_vt_kg,
            dim_x_m=tape_w_vtail_m,
            dim_y_m=TAPE_THICKNESS_M,
            dim_z_m=vtail_span_m,
            x_cg_m=x_le_tail_m + 0.5 * tape_w_vtail_m,
            y_cg_m=-VTAIL_TAPE_Y_OFFSET_M,
            z_cg_m=z_vtail_mid_m,
        )
        mass_props["vtail_tape"] = combine_mass_properties(
            [vtail_tape_side_a, vtail_tape_side_b]
        )

    # Fixed onboard components
    mass_props["linkages"] = point_mass(0.001, x_m=0.5 * tail_arm_m)

    x_centre_core = 0.3 * wing_chord_m + CENTRE_CORE_X_OFFSET_FROM_0p3C_M
    centre_mount_thickness_m = 0.002
    centre_mount_base_bottom_m = 0.0261
    centre_mount_base_top_m = 0.0115
    centre_mount_height_m = 0.086
    centre_mount_area_m2 = trapezoid_area_m2(
        centre_mount_base_bottom_m,
        centre_mount_base_top_m,
        centre_mount_height_m,
    )
    centre_mount_volume_m3 = centre_mount_area_m2 * centre_mount_thickness_m
    centre_mount_mass_kg = CENTRE_MODULE_RHO_KG_M3 * centre_mount_volume_m3
    centre_mount_ybar_m = trapezoid_centroid_from_base_m(
        centre_mount_base_bottom_m,
        centre_mount_base_top_m,
        centre_mount_height_m,
    )
    centre_mount_x0_fwd_m = 0.3 * wing_chord_m + 0.013069
    centre_mount_x0_aft_m = 0.3 * wing_chord_m + 0.0557
    centre_mount_z_root_bottom_m = 0.002
    tan_dihedral = np.tan(np.radians(DIHEDRAL_DEG))

    centre_mount_components: MassPropertiesMap = {}
    for y_sign, side_label in ((1.0, "R"), (-1.0, "L")):
        y_mount_m = y_sign * centre_mount_ybar_m
        z_mount_m = (
            centre_mount_z_root_bottom_m
            + 0.5 * centre_mount_thickness_m
            + np.abs(y_mount_m) * tan_dihedral
        )
        centre_mount_components[f"wing_mount_{side_label}_fwd"] = (
            mass_properties_isosceles_trapezoid_prism(
                mass_kg=centre_mount_mass_kg,
                base_bottom_m=centre_mount_base_bottom_m,
                base_top_m=centre_mount_base_top_m,
                height_m=centre_mount_height_m,
                thickness_m=centre_mount_thickness_m,
                x_cg_m=centre_mount_x0_fwd_m + 0.5 * centre_mount_base_bottom_m,
                y_cg_m=y_mount_m,
                z_cg_m=z_mount_m,
                height_axis="y",
            )
        )
        centre_mount_components[f"wing_mount_{side_label}_aft"] = (
            mass_properties_isosceles_trapezoid_prism(
                mass_kg=centre_mount_mass_kg,
                base_bottom_m=centre_mount_base_bottom_m,
                base_top_m=centre_mount_base_top_m,
                height_m=centre_mount_height_m,
                thickness_m=centre_mount_thickness_m,
                x_cg_m=centre_mount_x0_aft_m + 0.5 * centre_mount_base_bottom_m,
                y_cg_m=y_mount_m,
                z_cg_m=z_mount_m,
                height_axis="y",
            )
        )

    centre_mounts_total = combine_mass_properties(list(centre_mount_components.values()))
    centre_core_mass_kg = CENTRE_MODULE_MASS_KG - centre_mounts_total.mass
    centre_core_mass_num = to_float_if_possible(centre_core_mass_kg)
    if centre_core_mass_num is not None and centre_core_mass_num <= 0.0:
        centre_mounts_mass_num = to_float_if_possible(centre_mounts_total.mass)
        assert centre_mounts_mass_num is not None
        centre_scale = 0.90 * CENTRE_MODULE_MASS_KG / max(centre_mounts_mass_num, 1e-12)
        for name in list(centre_mount_components.keys()):
            centre_mount_components[name] = scale_mass_properties(
                centre_mount_components[name],
                centre_scale,
            )
        centre_mounts_total = combine_mass_properties(list(centre_mount_components.values()))
        centre_core_mass_kg = CENTRE_MODULE_MASS_KG - centre_mounts_total.mass

    mass_props.update(centre_mount_components)

    mass_props["centre_module_core"] = point_mass(
        centre_core_mass_kg,
        x_m=x_centre_core,
        z_m=CENTRE_CORE_Z_CG_M,
    )

    # Battery as a dedicated CG-trim slider.
    # Foremost position is tied to centre-module CG minus 35 mm.
    battery_eta = opti.variable(
        init_guess=0.60,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    x_batt_min = x_centre_core - BATTERY_FORE_OFFSET_FROM_CENTRE_MODULE_M
    x_batt_max = BATTERY_X_MAX_FRAC * wing_chord_m
    x_batt = x_batt_min + battery_eta * (x_batt_max - x_batt_min)
    mass_props["battery"] = mass_properties_rect_prism(
        mass_kg=BATTERY_MASS_KG,
        dim_x_m=BATTERY_DIM_X_M,
        dim_y_m=BATTERY_DIM_Y_M,
        dim_z_m=BATTERY_DIM_Z_M,
        x_cg_m=x_batt,
        y_cg_m=0.0,
        z_cg_m=AVIONICS_Z_CG_M,
    )

    # Remaining avionics keep fixed spacing from battery.
    mass_props["flight_controller"] = point_mass(
        FLIGHT_CONTROLLER_MASS_KG,
        x_m=x_batt + FLIGHT_CONTROLLER_X_OFFSET_FROM_BATTERY_M,
        z_m=AVIONICS_Z_CG_M,
    )
    x_regulator = x_batt + REGULATOR_X_OFFSET_FROM_BATTERY_M
    mass_props["regulator"] = point_mass(
        REGULATOR_MASS_KG,
        x_m=x_regulator,
        z_m=AVIONICS_Z_CG_M,
    )
    mass_props["receiver"] = point_mass(
        RECEIVER_MASS_KG,
        x_m=x_regulator + RECEIVER_X_OFFSET_FROM_REGULATOR_M,
        z_m=AVIONICS_Z_CG_M,
    )

    # Tail gear (tail support) from H-tail LE root.
    x_tail_support = tail_arm_m + TAIL_GEAR_X_OFFSET_FROM_HTAIL_LE_M
    x_tail_core = boom_end_x_m - TAIL_CORE_X_OFFSET_FROM_BOOM_END_M
    tail_mount_thickness_m = 0.0015
    tail_mount_base_bottom_m = 0.019
    tail_mount_base_top_m = 0.008
    tail_mount_height_m = 0.034
    tail_mount_area_m2 = trapezoid_area_m2(
        tail_mount_base_bottom_m,
        tail_mount_base_top_m,
        tail_mount_height_m,
    )
    tail_mount_volume_m3 = tail_mount_area_m2 * tail_mount_thickness_m
    tail_mount_mass_kg = TAIL_MODULE_RHO_KG_M3 * tail_mount_volume_m3
    tail_mount_ybar_m = trapezoid_centroid_from_base_m(
        tail_mount_base_bottom_m,
        tail_mount_base_top_m,
        tail_mount_height_m,
    )
    tail_mount_x0_m = boom_end_x_m - 0.0277
    tail_mount_z_upper_m = -0.0025
    tail_mount_z_cg_m = tail_mount_z_upper_m - 0.5 * tail_mount_thickness_m

    tail_mount_components: MassPropertiesMap = {}
    for y_sign, side_label in ((1.0, "R"), (-1.0, "L")):
        tail_mount_components[f"htail_mount_{side_label}"] = mass_properties_isosceles_trapezoid_prism(
            mass_kg=tail_mount_mass_kg,
            base_bottom_m=tail_mount_base_bottom_m,
            base_top_m=tail_mount_base_top_m,
            height_m=tail_mount_height_m,
            thickness_m=tail_mount_thickness_m,
            x_cg_m=tail_mount_x0_m + 0.5 * tail_mount_base_bottom_m,
            y_cg_m=y_sign * tail_mount_ybar_m,
            z_cg_m=tail_mount_z_cg_m,
            height_axis="y",
        )

    vtail_mount_area_m2 = trapezoid_area_m2(
        VTAIL_MOUNT_LB_M,
        VTAIL_MOUNT_LT_M,
        VTAIL_MOUNT_H_M,
    )
    vtail_mount_zbar_m = trapezoid_centroid_from_base_m(
        VTAIL_MOUNT_LB_M,
        VTAIL_MOUNT_LT_M,
        VTAIL_MOUNT_H_M,
    )
    vtail_mount_mass_kg = TAIL_MODULE_RHO_KG_M3 * vtail_mount_area_m2 * VTAIL_MOUNT_T_M
    vtail_mount_x0_m = boom_end_x_m - VTAIL_MOUNT_X0_OFFSET_FROM_BOOM_END_M
    vtail_mount_x_cg_m = vtail_mount_x0_m + 0.5 * VTAIL_MOUNT_LB_M
    vtail_mount_z_cg_m = VTAIL_MOUNT_ROOT_LOWER_Z_M + vtail_mount_zbar_m
    tail_mount_components["vtail_mount"] = mass_properties_isosceles_trapezoid_prism(
        mass_kg=vtail_mount_mass_kg,
        base_bottom_m=VTAIL_MOUNT_LB_M,
        base_top_m=VTAIL_MOUNT_LT_M,
        height_m=VTAIL_MOUNT_H_M,
        thickness_m=VTAIL_MOUNT_T_M,
        x_cg_m=vtail_mount_x_cg_m,
        y_cg_m=0.0,
        z_cg_m=vtail_mount_z_cg_m,
        height_axis="z",
    )

    tail_mounts_total = combine_mass_properties(list(tail_mount_components.values()))
    tail_core_mass_kg = TAIL_MODULE_MASS_KG - tail_mounts_total.mass
    tail_core_mass_num = to_float_if_possible(tail_core_mass_kg)
    if tail_core_mass_num is not None and tail_core_mass_num <= 0.0:
        tail_mounts_mass_num = to_float_if_possible(tail_mounts_total.mass)
        assert tail_mounts_mass_num is not None
        tail_scale = 0.90 * TAIL_MODULE_MASS_KG / max(tail_mounts_mass_num, 1e-12)
        for name in list(tail_mount_components.keys()):
            tail_mount_components[name] = scale_mass_properties(
                tail_mount_components[name],
                tail_scale,
            )
        tail_mounts_total = combine_mass_properties(list(tail_mount_components.values()))
        tail_core_mass_kg = TAIL_MODULE_MASS_KG - tail_mounts_total.mass

    mass_props.update(tail_mount_components)

    mass_props["tail_module_core"] = point_mass(
        tail_core_mass_kg,
        x_m=x_tail_core,
        z_m=TAIL_CORE_Z_CG_M,
    )

    mass_props["tail_support"] = point_mass(
        TAIL_SUPPORT_MASS_KG,
        x_m=x_tail_support,
        z_m=TAIL_SUPPORT_Z_CG_M,
    )

    # Tail-control servos mounted near the centre module in the fuselage centerline.
    mass_props["servo_elevator"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_centre_core + ELEVATOR_SERVO_X_OFFSET_FROM_CENTRE_CORE_M,
        y_m=0.0,
        z_m=SERVO_CENTERLINE_BASE_Z_M + ELEVATOR_SERVO_Z_OFFSET_FROM_AVIONICS_M,
    )
    mass_props["servo_rudder"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_centre_core + RUDDER_SERVO_X_OFFSET_FROM_CENTRE_CORE_M,
        y_m=0.0,
        z_m=SERVO_CENTERLINE_BASE_Z_M + RUDDER_SERVO_Z_OFFSET_FROM_AVIONICS_M,
    )

    y_servo = AILERON_SERVO_SPAN_FRAC * wing_span_m
    wing_root_centerline_z_m = WING_ROOT_LOWER_SURFACE_Z_M + 0.5 * WING_THICKNESS_M
    z_servo = (
        wing_root_centerline_z_m
        + np.abs(y_servo) * np.tan(np.radians(DIHEDRAL_DEG))
        + AILERON_SERVO_Z_OFFSET_M
    )
    x_aileron_servo = AILERON_SERVO_X_CHORD_FRAC * wing_chord_m
    mass_props["servo_aileron_R"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_aileron_servo,
        y_m=y_servo,
        z_m=z_servo,
    )
    mass_props["servo_aileron_L"] = point_mass(
        SERVO_MASS_KG,
        x_m=x_aileron_servo,
        y_m=-y_servo,
        z_m=z_servo,
    )

    boom_length_m = np.maximum(boom_end_x_m - NOSE_X_M, 0.05)
    boom_x_cg_m = 0.5 * (NOSE_X_M + boom_end_x_m)
    boom_tube = mass_properties_x_axis_tube(
        length_m=boom_length_m,
        od_m=BOOM_TUBE_OUTER_DIAMETER_M,
        id_m=BOOM_TUBE_INNER_DIAMETER_M,
        density_kg_m3=BOOM_ROD_DENSITY_KG_M3,
        x_cg_m=boom_x_cg_m,
        y_cg_m=0.0,
        z_cg_m=0.0,
    )
    if BOOM_EXTRA_MASS_KG > 0.0:
        mass_props["boom"] = combine_mass_properties(
            [boom_tube, point_mass(BOOM_EXTRA_MASS_KG, x_m=boom_x_cg_m)]
        )
    else:
        mass_props["boom"] = boom_tube

    # Ballast is optimized to close CG/stability constraints
    ballast_mass_kg = opti.variable(
        init_guess=0.0,
        lower_bound=0.0,
        upper_bound=BALLAST_MAX_KG,
    )
    mass_props["ballast"] = point_mass(
        ballast_mass_kg,
        x_m=0.30 * wing_chord_m,
        z_m=0.0,
    )

    subtotal = combine_mass_properties(list(mass_props.values()))
    mass_props["glue"] = scale_mass_properties(subtotal, GLUE_FRACTION)
    total_mass = combine_mass_properties([subtotal, mass_props["glue"]])

    return mass_props, total_mass, ballast_mass_kg, battery_eta


def aileron_effectiveness_proxy(
    aero: AeroMap,
    eta_inboard: float,
    eta_outboard: float,
    chord_fraction: float,
) -> Scalar:
    # Quick control-power proxy: Cl_delta_a ~ CLa * tau * outboard leverage
    c_l_alpha = np.maximum(np.abs(aero["CLa"]), 1e-3)
    span_factor = np.maximum(0.5 * (eta_outboard ** 2 - eta_inboard ** 2), 1e-4)
    tau_aileron = 0.9 * chord_fraction
    cl_delta_a = c_l_alpha * tau_aileron * span_factor
    return np.clip(cl_delta_a, 1e-3, CL_DELTA_A_PROXY_MAX)


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


def stable_softplus(x: Scalar, sharpness: float) -> Scalar:
    tx = sharpness * x
    return (1.0 / sharpness) * (
        np.fmax(tx, 0.0) + np.log1p(np.exp(-np.fabs(tx)))
    )


def build_trim_constraints_and_metrics(
    *,
    opti: asb.Opti,
    airplane: asb.Airplane,
    xyz_ref: Any,
    velocity_mps: float,
    alpha_deg: Scalar,
    mass_kg: Scalar,
    mode: Literal["nominal", "turn"],
    bank_angle_deg: float = 0.0,
    lift_k: float = 1.0,
    cl_cap: float | None = None,
    enforce_lateral_trim: bool = False,
    use_coordinated_turn: bool = False,
    atmosphere: asb.Atmosphere | None = None,
) -> dict[str, Any]:
    phi_turn_rad = np.radians(bank_angle_deg)
    n_load_factor = 1.0 / np.cos(phi_turn_rad)
    turn_denom_raw = lift_k * G * np.tan(phi_turn_rad)
    turn_denom = np.sign(turn_denom_raw) * np.maximum(np.abs(turn_denom_raw), 1e-8)
    yaw_rate_rad_s = (
        turn_denom / np.maximum(velocity_mps, 1e-8)
        if use_coordinated_turn
        else 0.0
    )

    atmos_use = atmosphere if atmosphere is not None else asb.Atmosphere(altitude=0.0)
    op_point = asb.OperatingPoint(
        atmosphere=atmos_use,
        velocity=velocity_mps,
        alpha=alpha_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=yaw_rate_rad_s,
    )
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=xyz_ref,
    ).run_with_stability_derivatives(
        alpha=True,
        beta=True,
        p=True,
        q=True,
        r=True,
    )

    constraints: list[Scalar] = [
        aero["L"] >= lift_k * n_load_factor * mass_kg * G,
        aero["Cm"] == 0.0,
    ]
    if cl_cap is not None:
        constraints.append(aero["CL"] <= cl_cap)
    if enforce_lateral_trim:
        constraints.extend(
            [
                aero["Cl"] == 0.0,
                aero["Cn"] == 0.0,
            ]
        )
    elif mode == "turn":
        constraints.extend(
            [
                aero["Cl"] >= -TURN_LATERAL_TRIM_TOL_CL,
                aero["Cl"] <= TURN_LATERAL_TRIM_TOL_CL,
                aero["Cn"] >= -TURN_LATERAL_TRIM_TOL_CN,
                aero["Cn"] <= TURN_LATERAL_TRIM_TOL_CN,
            ]
        )

    turn_radius_m = float("inf")
    if abs(float(bank_angle_deg)) > 1e-12:
        turn_radius_m = velocity_mps**2 / turn_denom

    return {
        "mode": mode,
        "op_point": op_point,
        "aero": aero,
        "constraints": constraints,
        "bank_angle_rad": phi_turn_rad,
        "n_load_factor": n_load_factor,
        "yaw_rate_rad_s": yaw_rate_rad_s,
        "turn_radius_m": turn_radius_m,
    }


def cl_delta_a_finite_difference(
    *,
    airplane_base: asb.Airplane,
    xyz_ref: list[float],
    velocity_mps: float,
    alpha_deg: float,
    delta_a_center_deg: float,
    delta_e_deg: float,
    delta_r_deg: float,
    yaw_rate_rad_s: float,
    step_deg: float = CL_DELTA_A_FD_STEP_DEG,
    atmosphere: asb.Atmosphere | None = None,
) -> float:
    step_deg_abs = max(abs(float(step_deg)), 1e-6)
    step_rad = float(onp.radians(step_deg_abs))
    atmos_use = atmosphere if atmosphere is not None else asb.Atmosphere(altitude=0.0)

    def eval_cl_at(delta_a_eval_deg: float) -> float:
        airplane_eval = airplane_base.with_control_deflections(
            {
                "aileron": delta_a_eval_deg,
                "elevator": float(delta_e_deg),
                "rudder": float(delta_r_deg),
            }
        )
        op_point_eval = asb.OperatingPoint(
            atmosphere=atmos_use,
            velocity=float(velocity_mps),
            alpha=float(alpha_deg),
            beta=0.0,
            p=0.0,
            q=0.0,
            r=float(yaw_rate_rad_s),
        )
        aero_eval = asb.AeroBuildup(
            airplane=airplane_eval,
            op_point=op_point_eval,
            xyz_ref=xyz_ref,
        ).run_with_stability_derivatives(
            alpha=True,
            beta=True,
            p=True,
            q=True,
            r=True,
        )
        return float(to_scalar(aero_eval["Cl"]))

    cl_plus = eval_cl_at(float(delta_a_center_deg) + step_deg_abs)
    cl_minus = eval_cl_at(float(delta_a_center_deg) - step_deg_abs)
    return (cl_plus - cl_minus) / (2.0 * step_rad)

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

    slack_lower = (value_f - lower) if lower is not None else None
    slack_upper = (upper - value_f) if upper is not None else None
    is_eq = (
        lower is not None
        and upper is not None
        and abs(upper - lower) <= max(tol, 1e-12)
    )
    scale = max(
        1.0,
        abs(float(value_f)),
        abs(float(lower)) if lower is not None else 0.0,
        abs(float(upper)) if upper is not None else 0.0,
    )

    residual: float | None = None
    abs_residual: float | None = None
    margin: float | None = None
    norm_margin: float | None = None
    is_active = False
    constraint_type = "eq" if is_eq else "ineq"

    if is_eq:
        residual = float(value_f - upper)
        abs_residual = abs(residual)
        margin = float(tol - abs_residual)
        norm_margin = float(margin / scale)
        is_active = abs_residual <= max(tol, ACTIVE_SET_ABS_TOL)
    else:
        if lower is not None and upper is not None:
            margin = float(min(slack_lower, slack_upper))
        elif lower is not None:
            margin = float(slack_lower)
        elif upper is not None:
            margin = float(slack_upper)
        else:
            margin = float("nan")
        norm_margin = float(margin / scale) if not onp.isnan(margin) else float("nan")
        is_active = (not onp.isnan(margin) and margin <= ACTIVE_SET_ABS_TOL) or (not passed)

    return {
        "Constraint": name,
        "Value": value_f,
        "Lower": lower,
        "Upper": upper,
        "Tolerance": tol,
        "Pass": passed,
        "Type": constraint_type,
        "SlackLower": slack_lower,
        "SlackUpper": slack_upper,
        "Residual": residual,
        "AbsResidual": abs_residual,
        "Margin": margin,
        "NormMargin": norm_margin,
        "IsActive": bool(is_active),
    }


def build_active_constraints_rows(
    constraint_rows: ReportRows,
    max_rows: int = 25,
) -> ReportRows:
    if not constraint_rows:
        return []

    constraint_df = pd.DataFrame(constraint_rows)
    if constraint_df.empty:
        return []

    active_mask = pd.Series(False, index=constraint_df.index)
    if "IsActive" in constraint_df.columns:
        active_mask = active_mask | constraint_df["IsActive"].fillna(False).astype(bool)
    if "Pass" in constraint_df.columns:
        active_mask = active_mask | (~constraint_df["Pass"].fillna(False).astype(bool))

    active_df = constraint_df.loc[active_mask].copy()
    if active_df.empty:
        return []

    if "NormMargin" in active_df.columns:
        active_df["__norm_margin_sort"] = pd.to_numeric(
            active_df["NormMargin"],
            errors="coerce",
        )
        active_df = active_df.sort_values(
            by="__norm_margin_sort",
            ascending=True,
            na_position="last",
        )
        active_df = active_df.drop(columns="__norm_margin_sort")

    if max_rows > 0:
        active_df = active_df.head(max_rows)

    return active_df.to_dict(orient="records")


def _serialize_stat_value(value: Any) -> ReportValue:
    if isinstance(value, onp.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return str(value)


def flatten_stats(stats: dict[str, Any]) -> ReportRows:
    rows: ReportRows = []

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            if not obj:
                if prefix:
                    rows.append({"Key": prefix, "Value": "{}"})
                return
            for key, value in obj.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(next_prefix, value)
            return

        if isinstance(obj, (list, tuple)):
            if not obj:
                rows.append({"Key": prefix, "Value": "[]"})
                return
            for idx, value in enumerate(obj):
                walk(f"{prefix}[{idx}]", value)
            return

        rows.append(
            {
                "Key": prefix if prefix else "value",
                "Value": _serialize_stat_value(obj),
            }
        )

    walk("", stats if isinstance(stats, dict) else {})
    rows.sort(key=lambda row: str(row.get("Key", "")))
    return rows


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
        component_name = name
        if name == "centre_module_core":
            component_name = "centre module core"
        elif name == "tail_module_core":
            component_name = "tail module core"
        elif name.startswith("wing_mount_"):
            component_name = name.replace("wing_mount_", "wing mount ").replace("_", " ")
        elif name.startswith("htail_mount_"):
            component_name = name.replace("htail_mount_", "htail mount ").replace("_", " ")
        elif name == "vtail_mount":
            component_name = "vtail mount"
        elif name == "tail_support":
            component_name = "tail support"
        rows.append(
            {
                "Component": component_name,
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


def weighted_cg_from_components(
    mass_props: MassPropertiesMap,
) -> tuple[float, float, float, float]:
    # Independent weighted-CG reconstruction for audit/consistency checks.
    total_mass = 0.0
    x_moment = 0.0
    y_moment = 0.0
    z_moment = 0.0
    for component in mass_props.values():
        mass = float(to_scalar(component.mass))
        x_cg = float(to_scalar(component.xyz_cg[0]))
        y_cg = float(to_scalar(component.xyz_cg[1]))
        z_cg = float(to_scalar(component.xyz_cg[2]))
        total_mass += mass
        x_moment += mass * x_cg
        y_moment += mass * y_cg
        z_moment += mass * z_cg

    if abs(total_mass) < 1e-12:
        return float("nan"), float("nan"), float("nan"), total_mass

    return (
        x_moment / total_mass,
        y_moment / total_mass,
        z_moment / total_mass,
        total_mass,
    )


def save_results(
    summary_rows: ReportRows,
    geometry_rows: ReportRows,
    mass_rows: ReportRows,
    aero_rows: ReportRows,
    constraint_rows: ReportRows,
    design_points_rows: ReportRows | None = None,
    boundary_rows: ReportRows | None = None,
    active_constraint_rows: ReportRows | None = None,
    solver_stats: dict[str, Any] | None = None,
) -> PathMap:
    # Persist a compact CSV plus a multi-sheet workbook
    summary_df = pd.DataFrame(summary_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    mass_df = pd.DataFrame(mass_rows)
    aero_df = pd.DataFrame(aero_rows)
    constraints_df = pd.DataFrame(constraint_rows)
    solver_stats_df = pd.DataFrame(
        flatten_stats(solver_stats or {}),
        columns=["Key", "Value"],
    )
    active_constraints_rows = (
        active_constraint_rows
        if active_constraint_rows is not None
        else build_active_constraints_rows(constraint_rows)
    )
    active_constraints_df = pd.DataFrame(active_constraints_rows)
    if active_constraints_df.empty:
        active_constraints_df = pd.DataFrame(columns=constraints_df.columns)

    csv_path = RESULTS_DIR / "nausicaa_results.csv"
    xlsx_path = RESULTS_DIR / "nausicaa_results.xlsx"

    summary_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        geometry_df.to_excel(writer, sheet_name="Geometry", index=False)
        mass_df.to_excel(writer, sheet_name="MassBreakdown", index=False)
        aero_df.to_excel(writer, sheet_name="Aerodynamics", index=False)
        constraints_df.to_excel(writer, sheet_name="Constraints", index=False)
        active_constraints_df.to_excel(
            writer,
            sheet_name="ActiveConstraints",
            index=False,
        )
        solver_stats_df.to_excel(writer, sheet_name="SolverStats", index=False)
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
    print("  - Rectangular wing, fixed dihedral, shared boom reference", flush=True)
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
        f"  Total CG = ({fmt('total_cg_x_m', 4)}, {fmt('total_cg_y_m', 4)}, "
        f"{fmt('total_cg_z_m', 4)}) m",
        flush=True,
    )
    print(
        f"  CG check error = ({fmt('total_cg_x_error_m', 2)}, "
        f"{fmt('total_cg_y_error_m', 2)}, {fmt('total_cg_z_error_m', 2)}) m",
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
        f"  Boom length = {to_scalar(geometry.get('boom_length_m', 0.0)):.3f} m | "
        f"H-tail span = {to_scalar(geometry.get('htail_span_m', 0.0)):.3f} m | "
        f"V-tail height = {to_scalar(geometry.get('vtail_height_m', 0.0)):.3f} m",
        flush=True,
    )
    print("\nStructural diagnostics:", flush=True)
    print(
        f"  Wing semispan L = {fmt('wing_struct_semispan_m', 3)} m | "
        f"half-load = {fmt('wing_struct_half_load_n', 3)} N | "
        f"E = {fmt('wing_struct_E_pa', 2)} Pa | "
        f"t = {fmt('wing_struct_thickness_m', 6)} m",
        flush=True,
    )
    print(
        f"  H-tail semispan L = {fmt('htail_struct_semispan_m', 3)} m | "
        f"half-load = {fmt('htail_struct_half_load_n', 3)} N | "
        f"E = {fmt('htail_struct_E_pa', 2)} Pa | "
        f"t = {fmt('htail_struct_thickness_m', 6)} m",
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
    global _LAST_SOLVE_FAILURE_REASON
    _LAST_SOLVE_FAILURE_REASON = None

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

    # Nominal trim variables (used by objective)
    alpha_nom_deg = opti.variable(
        init_guess=init_value("alpha_nom_deg", init_value("alpha_deg", 4.0)),
        lower_bound=ALPHA_MIN_DEG,
        upper_bound=ALPHA_MAX_DEG,
    )
    delta_a_nom_deg = opti.variable(
        init_guess=init_value("delta_a_nom_deg", init_value("delta_a_deg", 0.0)),
        lower_bound=DELTA_A_MIN_DEG,
        upper_bound=DELTA_A_MAX_DEG,
    )
    delta_e_nom_deg = opti.variable(
        init_guess=init_value("delta_e_nom_deg", init_value("delta_e_deg", 0.0)),
        lower_bound=DELTA_E_MIN_DEG,
        upper_bound=DELTA_E_MAX_DEG,
    )
    delta_r_nom_deg = opti.variable(
        init_guess=init_value("delta_r_nom_deg", init_value("delta_r_deg", 0.0)),
        lower_bound=DELTA_R_MIN_DEG,
        upper_bound=DELTA_R_MAX_DEG,
    )

    # Manoeuvre (turn) trim variables (used by manoeuvre feasibility constraints)
    alpha_turn_deg = opti.variable(
        init_guess=init_value("alpha_turn_deg", init_value("alpha_deg", 7.0)),
        lower_bound=ALPHA_MIN_DEG,
        upper_bound=ALPHA_MAX_TURN_DEG,
    )
    delta_a_turn_deg = opti.variable(
        init_guess=init_value("delta_a_turn_deg", init_value("delta_a_deg", 0.0)),
        lower_bound=DELTA_A_MIN_DEG,
        upper_bound=DELTA_A_MAX_DEG,
    )
    delta_e_turn_deg = opti.variable(
        init_guess=init_value("delta_e_turn_deg", init_value("delta_e_deg", 0.0)),
        lower_bound=DELTA_E_MIN_DEG,
        upper_bound=DELTA_E_MAX_DEG,
    )
    delta_r_turn_deg = opti.variable(
        init_guess=init_value("delta_r_turn_deg", init_value("delta_r_deg", 0.0)),
        lower_bound=DELTA_R_MIN_DEG,
        upper_bound=DELTA_R_MAX_DEG,
    )

    # Primary geometry design variables
    wing_span_m = opti.variable(
        init_guess=init_value("wing_span_m", 0.80),
        lower_bound=WING_SPAN_MIN_M,
        upper_bound=WING_SPAN_MAX_M,
    )
    wing_chord_m = opti.variable(
        init_guess=init_value("wing_chord_m", 0.20),
        lower_bound=WING_CHORD_MIN_M,
        upper_bound=WING_CHORD_MAX_M,
    )
    tail_arm_m = opti.variable(
        init_guess=init_value("tail_arm_m", 0.70),
        lower_bound=TAIL_ARM_MIN_M,
        upper_bound=TAIL_ARM_MAX_M,
    )
    htail_span_m = opti.variable(
        init_guess=init_value("htail_span_m", 0.30),
        lower_bound=HT_SPAN_MIN_M,
        upper_bound=HT_SPAN_MAX_M,
    )
    vtail_height_m = opti.variable(
        init_guess=init_value("vtail_height_m", 0.20),
        lower_bound=VT_HEIGHT_MIN_M,
        upper_bound=VT_HEIGHT_MAX_M,
    )
    # Define boom by its aft endpoint, then derive boom length from nose station.
    boom_end_x_m = tail_arm_m + BOOM_END_BEFORE_ELEV_FRAC * (htail_span_m / HT_AR)
    boom_length_m = boom_end_x_m - NOSE_X_M

    variable_map = {
        "alpha_nom_deg": alpha_nom_deg,
        "delta_a_nom_deg": delta_a_nom_deg,
        "delta_e_nom_deg": delta_e_nom_deg,
        "delta_r_nom_deg": delta_r_nom_deg,
        "alpha_turn_deg": alpha_turn_deg,
        "delta_a_turn_deg": delta_a_turn_deg,
        "delta_e_turn_deg": delta_e_turn_deg,
        "delta_r_turn_deg": delta_r_turn_deg,
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
    fuselage = build_fuselage(boom_end_x_m=boom_end_x_m)

    airplane_base = asb.Airplane(
        name="Nausicaa",
        wings=[wing, htail, vtail],
        fuselages=[fuselage],
    )
    airplane_nom = airplane_base.with_control_deflections(
        {
            "aileron": delta_a_nom_deg,
            "elevator": delta_e_nom_deg,
            "rudder": delta_r_nom_deg,
        }
    )
    airplane_turn = airplane_base.with_control_deflections(
        {
            "aileron": delta_a_turn_deg,
            "elevator": delta_e_turn_deg,
            "rudder": delta_r_turn_deg,
        }
    )

    atmos = asb.Atmosphere(altitude=0.0)

    mass_props, total_mass, ballast_mass_kg, battery_eta = build_mass_model(
        opti=opti,
        wing=wing,
        htail=htail,
        vtail=vtail,
        wing_chord_m=wing_chord_m,
        tail_arm_m=tail_arm_m,
        boom_end_x_m=boom_end_x_m,
        vtail_chord_m=vtail_chord_m,
    )

    trim_nom = build_trim_constraints_and_metrics(
        opti=opti,
        airplane=airplane_nom,
        xyz_ref=total_mass.xyz_cg,
        velocity_mps=V_NOM_MPS,
        alpha_deg=alpha_nom_deg,
        mass_kg=total_mass.mass,
        mode="nominal",
        bank_angle_deg=0.0,
        lift_k=1.0,
        cl_cap=MAX_CL_AT_DESIGN_POINT,
        enforce_lateral_trim=True,
        use_coordinated_turn=False,
        atmosphere=atmos,
    )
    trim_turn = build_trim_constraints_and_metrics(
        opti=opti,
        airplane=airplane_turn,
        xyz_ref=total_mass.xyz_cg,
        velocity_mps=V_TURN_MPS,
        alpha_deg=alpha_turn_deg,
        mass_kg=total_mass.mass,
        mode="turn",
        bank_angle_deg=TURN_BANK_DEG,
        lift_k=K_LEVEL_TURN,
        cl_cap=TURN_CL_CAP,
        enforce_lateral_trim=False,
        use_coordinated_turn=True,
        atmosphere=atmos,
    )

    op_point_nom = trim_nom["op_point"]
    op_point_turn = trim_turn["op_point"]
    aero_nom = trim_nom["aero"]
    aero_turn = trim_turn["aero"]

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

    cl_delta_a_nom = aileron_effectiveness_proxy(
        aero=aero_nom,
        eta_inboard=AILERON_ETA_INBOARD,
        eta_outboard=AILERON_ETA_OUTBOARD,
        chord_fraction=AILERON_CHORD_FRACTION,
    )
    cl_delta_a_turn = aileron_effectiveness_proxy(
        aero=aero_turn,
        eta_inboard=AILERON_ETA_INBOARD,
        eta_outboard=AILERON_ETA_OUTBOARD,
        chord_fraction=AILERON_CHORD_FRACTION,
    )

    # Linearized roll response estimates
    q_dyn = 0.5 * RHO * V_NOM_MPS ** 2
    q_dyn_turn = 0.5 * RHO * V_TURN_MPS ** 2
    i_xx = total_mass.inertia_tensor[0, 0]
    clp_mag = np.maximum(np.abs(aero_nom["Clp"]), 1e-5)
    clp_mag_turn = np.maximum(np.abs(aero_turn["Clp"]), 1e-5)
    delta_a_max_rad = np.radians(DELTA_A_MAX_DEG)
    delta_a_turn_cmd_deg = TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG
    delta_a_turn_rate_limited_deg = SERVO_RATE_DEG_S * BANK_ENTRY_TIME_S
    if INCLUDE_SERVO_RATE_IN_BANK_ENTRY:
        delta_a_turn_eff_deg = np.fmin(
            delta_a_turn_cmd_deg,
            delta_a_turn_rate_limited_deg,
        )
    else:
        delta_a_turn_eff_deg = delta_a_turn_cmd_deg
    delta_a_turn_rad = np.radians(delta_a_turn_eff_deg)

    roll_accel0_rad_s2 = (
        q_dyn
        * wing_area_m2
        * wing_span_m
        * np.abs(cl_delta_a_nom)
        * delta_a_max_rad
        / i_xx
    )

    roll_tau_s = (2.0 * i_xx * V_NOM_MPS) / np.maximum(
        q_dyn * wing_area_m2 * wing_span_m ** 2 * clp_mag,
        1e-8,
    )
    roll_tau_turn_s = (2.0 * i_xx * V_TURN_MPS) / np.maximum(
        q_dyn_turn * wing_area_m2 * wing_span_m ** 2 * clp_mag_turn,
        1e-8,
    )

    roll_rate_ss_radps = (
        2.0
        * V_NOM_MPS
        / np.maximum(wing_span_m, 1e-8)
        * np.abs(cl_delta_a_nom)
        * delta_a_max_rad
        / clp_mag
    )
    roll_rate_ss_turn_radps = (
        2.0
        * V_TURN_MPS
        / np.maximum(wing_span_m, 1e-8)
        * np.abs(cl_delta_a_turn)
        * delta_a_turn_rad
        / clp_mag_turn
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

    q_dyn_max = 0.5 * RHO * max(V_NOM_MPS, V_TURN_MPS) ** 2
    hinge_moment_aileron_nm = estimate_servo_hinge_moment(
        q_dyn=q_dyn_max,
        control_area_m2=0.5 * aileron_area_m2,
        mean_chord_m=aileron_chord_m,
        deflection_deg=DELTA_A_MAX_DEG,
    )
    hinge_moment_elevator_nm = estimate_servo_hinge_moment(
        q_dyn=q_dyn_max,
        control_area_m2=elevator_area_m2,
        mean_chord_m=elevator_chord_m,
        deflection_deg=DELTA_E_MAX_DEG,
    )
    hinge_moment_rudder_nm = estimate_servo_hinge_moment(
        q_dyn=q_dyn_max,
        control_area_m2=rudder_area_m2,
        mean_chord_m=rudder_chord_m,
        deflection_deg=DELTA_R_MAX_DEG,
    )

    # Apply servo safety factor to rated torque
    servo_torque_available_nm = (
        SERVO_TORQUE_LIMIT_NM / SERVO_SAFETY_FACTOR
        * (CONTROL_HORN_ARM_M / SERVO_ARM_M)
        * LINKAGE_EFFICIENCY
    )

    # Penalize unnecessary trim deflections
    trim_effort = (
        delta_e_nom_deg ** 2
        + 0.3 * delta_r_nom_deg ** 2
        + 0.15 * delta_a_nom_deg ** 2
    )

    # Arena footprint for coordinated banked turn feasibility
    phi_turn_rad = trim_turn["bank_angle_rad"]
    turn_radius_m = trim_turn["turn_radius_m"]
    tau_turn_eff_s = np.maximum(roll_tau_turn_s, 1e-4)
    bank_entry_dt_s = np.fmax(BANK_ENTRY_TIME_S - tau_turn_eff_s, 0.0)
    bank_entry_phi_capture_proxy_rad = roll_rate_ss_turn_radps * bank_entry_dt_s
    bank_entry_margin_rad = bank_entry_phi_capture_proxy_rad - phi_turn_rad
    bank_entry_phi_achieved_rad = roll_rate_ss_turn_radps * (
        BANK_ENTRY_TIME_S
        - tau_turn_eff_s * (1.0 - np.exp(-BANK_ENTRY_TIME_S / tau_turn_eff_s))
    )

    # Structural flexibility proxy: each half-wing is a semispan cantilever
    # with half of the total lift, modeled as a uniform line load.
    n_turn = trim_turn["n_load_factor"]
    L_semispan_m = 0.5 * wing_span_m
    weight_n = total_mass.mass * G
    e_foam_wing_pa = WING_E_SECANT_PA
    e_spar_wing_pa = WING_SPAR_E_FLEX_PA
    e_tape_pa = TAPE_EFFICIENCY * TAPE_E_EFFECTIVE_PA
    tape_w_wing_m = np.minimum(TAPE_WIDTH_M, wing_chord_m)
    ei_wing_nm2, _wing_z0_m = composite_EI_flapwise(
        chord_m=wing_chord_m,
        foam_thickness_m=WING_THICKNESS_M,
        e_foam_pa=e_foam_wing_pa,
        spar_od_m=WING_SPAR_OD_M,
        spar_id_m=WING_SPAR_ID_M,
        e_spar_pa=e_spar_wing_pa,
        spar_z_from_lower_m=WING_SPAR_Z_FROM_LOWER_M,
        include_spar=WING_SPAR_ENABLE,
        tape_width_m=tape_w_wing_m,
        tape_thickness_m=TAPE_THICKNESS_M,
        e_tape_pa=e_tape_pa,
        include_tape=TAPE_ENABLE_WING,
    )
    wing_total_lift_n = n_turn * weight_n
    wing_half_load_n = 0.5 * wing_total_lift_n
    wing_line_load_n_m = wing_half_load_n / np.maximum(L_semispan_m, 1e-9)
    delta_tip_m = (
        wing_line_load_n_m
        * (L_semispan_m ** 4)
        / (8.0 * np.maximum(ei_wing_nm2, 1e-12))
    )
    delta_allow_m = WING_DEFLECTION_ALLOW_FRAC * L_semispan_m
    delta_over_m = stable_softplus(delta_tip_m - delta_allow_m, SOFTPLUS_K)
    struct_deflection_penalty = STRUCT_DEFLECTION_WEIGHT * (
        delta_over_m / np.maximum(delta_allow_m, 1e-6)
    ) ** 2

    # Horizontal-tail flexibility proxy: each half-tail is treated the same way.
    L_ht_semispan_m = 0.5 * htail_span_m
    ei_ht_nm2, _ht_z0_m = composite_EI_flapwise(
        chord_m=htail_chord_m,
        foam_thickness_m=TAIL_THICKNESS_M,
        e_foam_pa=HTAIL_E_SECANT_PA,
        spar_od_m=WING_SPAR_OD_M,
        spar_id_m=WING_SPAR_ID_M,
        e_spar_pa=WING_SPAR_E_FLEX_PA,
        spar_z_from_lower_m=0.0,
        include_spar=False,
        tape_width_m=np.minimum(TAPE_WIDTH_M, htail_chord_m),
        tape_thickness_m=TAPE_THICKNESS_M,
        e_tape_pa=e_tape_pa,
        include_tape=TAPE_ENABLE_TAIL,
    )

    # Load model 1: F_ht_design = k_H * n_turn * W
    F_ht_design_n = HT_LOAD_FRACTION * n_turn * weight_n
    ht_half_load_n = 0.5 * F_ht_design_n
    ht_line_load_n_m = ht_half_load_n / np.maximum(L_ht_semispan_m, 1e-9)

    delta_ht_tip_m = (
        ht_line_load_n_m
        * (L_ht_semispan_m ** 4)
        / (8.0 * np.maximum(ei_ht_nm2, 1e-12))
    )

    delta_ht_allow_m = HT_DEFLECTION_ALLOW_FRAC * L_ht_semispan_m
    delta_ht_over_m = stable_softplus(delta_ht_tip_m - delta_ht_allow_m, SOFTPLUS_K)

    htail_deflection_penalty = HT_STRUCT_DEFLECTION_WEIGHT * (
        delta_ht_over_m / np.maximum(delta_ht_allow_m, 1e-6)
    ) ** 2

    # Roll agility regularizer: reward margin below hard roll time-constant cap.
    roll_tau_penalty = ROLL_TAU_WEIGHT_IN_OBJECTIVE * (roll_tau_s / MAX_ROLL_TAU_S) ** 2

    # Objective: nominal-speed sink with light penalties on mass, ballast, and trim effort
    objective = (
        sink_rate_nom_mps
        + MASS_WEIGHT_IN_OBJECTIVE * total_mass.mass
        + BALLAST_WEIGHT_IN_OBJECTIVE * ballast_mass_kg
        + CONTROL_TRIM_WEIGHT * trim_effort
        + struct_deflection_penalty
        + htail_deflection_penalty
        + roll_tau_penalty
    )
    opti.minimize(objective)

    # Feasibility constraints
    opti.subject_to(
        [
            # Nominal-speed trim/performance constraints
            *trim_nom["constraints"],
            aero_nom["D"] >= 1e-3,
            l_over_d >= MIN_L_OVER_D,
            opti.bounded(
                MIN_WING_LOADING_N_M2,
                wing_loading_n_m2,
                MAX_WING_LOADING_N_M2,
            ),
            reynolds_wing >= MIN_RE_WING,
            opti.bounded(STATIC_MARGIN_MIN, static_margin, STATIC_MARGIN_MAX),
            opti.bounded(BOOM_LENGTH_MIN_M, boom_length_m, BOOM_LENGTH_MAX_M),
            opti.bounded(VH_MIN, tail_volume_horizontal, VH_MAX),
            opti.bounded(VV_MIN, tail_volume_vertical, VV_MAX),
            aero_nom["Clb"] <= CLB_MAX,
            aero_nom["Cnb"] >= CNB_MIN,
            aero_nom["Cmq"] <= CMQ_MAX,
            aero_nom["Clp"] <= -CLP_NEG_EPS,
            aero_turn["Clp"] <= -CLP_NEG_EPS,
            roll_rate_ss_radps >= MIN_ROLL_RATE_RAD_S,
            roll_accel0_rad_s2 >= MIN_ROLL_ACCEL_RAD_S2,
            roll_tau_s <= MAX_ROLL_TAU_S,
            total_mass.inertia_tensor[0, 0] >= 1e-8,
            total_mass.inertia_tensor[1, 1] >= 1e-8,
            total_mass.inertia_tensor[2, 2] >= 1e-8,
            # Coarse feasibility screen; not used as a primary sizing driver.
            hinge_moment_aileron_nm <= servo_torque_available_nm,
            hinge_moment_elevator_nm <= servo_torque_available_nm,
            hinge_moment_rudder_nm <= servo_torque_available_nm,
            # Manoeuvre-speed constraints
            turn_radius_m + 0.5 * wing_span_m + WALL_CLEARANCE_M <= 0.5 * ARENA_WIDTH_M,
            *trim_turn["constraints"],
            opti.bounded(
                -TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG,
                delta_a_turn_deg,
                TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG,
            ),
            opti.bounded(
                -TURN_DEFLECTION_UTIL_MAX * DELTA_E_MAX_DEG,
                delta_e_turn_deg,
                TURN_DEFLECTION_UTIL_MAX * DELTA_E_MAX_DEG,
            ),
            opti.bounded(
                -TURN_DEFLECTION_UTIL_MAX * DELTA_R_MAX_DEG,
                delta_r_turn_deg,
                TURN_DEFLECTION_UTIL_MAX * DELTA_R_MAX_DEG,
            ),
            bank_entry_margin_rad >= 0.0,
        ]
    )

    # IPOPT setup
    plugin_options = {"print_time": False, "verbose": False}
    solver_options = {
        "max_iter": 1000,
        "check_derivatives_for_naninf": "no",
        "hessian_approximation": "limited-memory",
        "print_level": 0,
        "sb": "yes",
    }
    if ipopt_options is not None:
        solver_options.update(ipopt_options)
    opti.solver("ipopt", plugin_options, solver_options)

    def debug_guess(label: str, expr: Scalar) -> None:
        try:
            if isinstance(expr, (int, float, onp.floating, onp.integer)):
                value = to_scalar(expr)
            elif hasattr(opti, "debug") and hasattr(opti.debug, "value"):
                value = to_scalar(opti.debug.value(expr, opti.initial()))
            else:
                value = to_scalar(opti.value(expr, opti.initial()))
            print(f"[DEBUG INIT] {label} = {value:.6g}", flush=True)
        except Exception as exc:
            print(f"[DEBUG INIT] {label} unavailable ({exc})", flush=True)

    debug_guess("Ixx", total_mass.inertia_tensor[0, 0])
    debug_guess("aero_nom_Clp", aero_nom["Clp"])
    debug_guess("aero_turn_Clp", aero_turn["Clp"])
    debug_guess("cl_delta_a_nom_proxy", cl_delta_a_nom)
    debug_guess("cl_delta_a_turn_proxy", cl_delta_a_turn)
    debug_guess("roll_tau_s", roll_tau_s)
    debug_guess("roll_tau_turn_s", roll_tau_turn_s)
    debug_guess("roll_accel0_rad_s2", roll_accel0_rad_s2)
    debug_guess("roll_rate_ss_turn_radps", roll_rate_ss_turn_radps)
    debug_guess("delta_a_turn_eff_deg", delta_a_turn_eff_deg)
    debug_guess("bank_entry_dt_s", bank_entry_dt_s)
    debug_guess("bank_entry_margin_rad", bank_entry_margin_rad)
    debug_guess("bank_entry_phi_capture_proxy_rad", bank_entry_phi_capture_proxy_rad)
    debug_guess("bank_entry_phi_achieved_rad", bank_entry_phi_achieved_rad)

    print("Starting optimization...", flush=True)
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        _LAST_SOLVE_FAILURE_REASON = str(exc)
        print(f"\n[SOLVE FAILED] {exc}", flush=True)
        try:
            opti.debug.show_infeasibilities()
        except Exception as infeas_exc:
            print(f"[WARN] show_infeasibilities() failed: {infeas_exc}", flush=True)
        print("No feasible design was found with the current settings", flush=True)
        return None
    solve_stats: dict[str, Any] = {}
    try:
        stats_obj = solution.stats() if hasattr(solution, "stats") else {}
        solve_stats = stats_obj if isinstance(stats_obj, dict) else {}
    except Exception:
        solve_stats = {}

    # Numeric post-processing for reports and exports
    airplane_base_num = solution(airplane_base)
    airplane_num = solution(airplane_nom)
    wing_num = copy.deepcopy(airplane_num.wings[0])
    htail_num = copy.deepcopy(airplane_num.wings[1])
    vtail_num = copy.deepcopy(airplane_num.wings[2])

    mass_props_num = solution(mass_props)
    total_mass_num = solution(total_mass)
    aero_nom_num = solution(aero_nom)
    aero_turn_num = solution(aero_turn)
    i_xx_num = to_scalar(total_mass_num.inertia_tensor[0, 0])
    if (not onp.isfinite(i_xx_num)) or i_xx_num <= 0.0:
        raise ValueError(
            f"Invalid solved inertia: Ixx={i_xx_num:.6e} kg*m^2 (must be positive)."
        )

    objective_num = to_scalar(solution(objective))
    alpha_nom_num = to_scalar(solution(alpha_nom_deg))
    delta_a_nom_num = to_scalar(solution(delta_a_nom_deg))
    delta_e_nom_num = to_scalar(solution(delta_e_nom_deg))
    delta_r_nom_num = to_scalar(solution(delta_r_nom_deg))
    alpha_turn_num = to_scalar(solution(alpha_turn_deg))
    delta_a_turn_num = to_scalar(solution(delta_a_turn_deg))
    delta_e_turn_num = to_scalar(solution(delta_e_turn_deg))
    delta_r_turn_num = to_scalar(solution(delta_r_turn_deg))
    wing_span_design_num = to_scalar(solution(wing_span_m))
    wing_chord_design_num = to_scalar(solution(wing_chord_m))
    tail_arm_design_num = to_scalar(solution(tail_arm_m))
    boom_end_x_design_num = to_scalar(solution(boom_end_x_m))
    boom_length_design_num = to_scalar(solution(boom_length_m))
    htail_span_design_num = to_scalar(solution(htail_span_m))
    vtail_height_design_num = to_scalar(solution(vtail_height_m))
    battery_eta_num = to_scalar(solution(battery_eta))

    sink_rate_num = to_scalar(solution(sink_rate_nom_mps))
    l_over_d_num = to_scalar(solution(l_over_d))
    mass_total_num = to_scalar(total_mass_num.mass)
    total_cg_x_num = to_scalar(total_mass_num.x_cg)
    total_cg_y_num = to_scalar(total_mass_num.y_cg)
    total_cg_z_num = to_scalar(total_mass_num.z_cg)
    weighted_cg_x_num, weighted_cg_y_num, weighted_cg_z_num, component_mass_sum_num = (
        weighted_cg_from_components(mass_props_num)
    )
    total_cg_x_error_num = total_cg_x_num - weighted_cg_x_num
    total_cg_y_error_num = total_cg_y_num - weighted_cg_y_num
    total_cg_z_error_num = total_cg_z_num - weighted_cg_z_num
    ballast_mass_num = to_scalar(solution(ballast_mass_kg))
    ballast_mass_num = max(0.0, ballast_mass_num)
    x_centre_core_num = 0.3 * wing_chord_design_num + CENTRE_CORE_X_OFFSET_FROM_0p3C_M
    battery_x_min_num = x_centre_core_num - BATTERY_FORE_OFFSET_FROM_CENTRE_MODULE_M
    battery_x_num = to_scalar(
        battery_x_min_num
        + battery_eta_num
        * (BATTERY_X_MAX_FRAC * wing_chord_design_num - battery_x_min_num)
    )

    static_margin_num = to_scalar(solution(static_margin))
    tail_volume_h_num = to_scalar(solution(tail_volume_horizontal))
    tail_volume_v_num = to_scalar(solution(tail_volume_vertical))

    wing_loading_num = to_scalar(solution(wing_loading_n_m2))
    reynolds_num = to_scalar(solution(reynolds_wing))

    roll_rate_num = to_scalar(solution(roll_rate_ss_radps))
    roll_rate_turn_num = to_scalar(solution(roll_rate_ss_turn_radps))
    roll_accel_num = to_scalar(solution(roll_accel0_rad_s2))
    roll_tau_num = to_scalar(solution(roll_tau_s))
    roll_tau_turn_num = to_scalar(solution(roll_tau_turn_s))
    try:
        delta_a_turn_eff_deg_num = to_scalar(solution(delta_a_turn_eff_deg))
    except Exception:
        delta_a_turn_eff_deg_num = to_scalar(delta_a_turn_eff_deg)
    bank_entry_margin_deg_num = np.degrees(to_scalar(solution(bank_entry_margin_rad)))
    bank_entry_phi_capture_proxy_deg_num = np.degrees(
        to_scalar(solution(bank_entry_phi_capture_proxy_rad))
    )
    bank_entry_phi_achieved_deg_num = np.degrees(
        to_scalar(solution(bank_entry_phi_achieved_rad))
    )
    delta_tip_num = to_scalar(solution(delta_tip_m))
    delta_allow_num = to_scalar(solution(delta_allow_m))
    wing_ei_num = to_scalar(solution(ei_wing_nm2))
    delta_ht_tip_num = to_scalar(solution(delta_ht_tip_m))
    delta_ht_allow_num = to_scalar(solution(delta_ht_allow_m))
    htail_ei_num = to_scalar(solution(ei_ht_nm2))
    delta_ratio_num = delta_tip_num / max(0.5 * float(wing_span_design_num), 1e-9)
    delta_ht_ratio_num = delta_ht_tip_num / max(0.5 * float(htail_span_design_num), 1e-9)
    struct_deflection_penalty_num = to_scalar(solution(struct_deflection_penalty))
    htail_deflection_penalty_num = to_scalar(solution(htail_deflection_penalty))
    roll_tau_penalty_num = to_scalar(solution(roll_tau_penalty))

    hinge_aileron_num = to_scalar(solution(hinge_moment_aileron_nm))
    hinge_elevator_num = to_scalar(solution(hinge_moment_elevator_nm))
    hinge_rudder_num = to_scalar(solution(hinge_moment_rudder_nm))

    max_servo_utilization = max(
        hinge_aileron_num,
        hinge_elevator_num,
        hinge_rudder_num,
    ) / servo_torque_available_nm

    turn_radius_num = float(
        V_TURN_MPS**2 / (K_LEVEL_TURN * G * onp.tan(onp.radians(TURN_BANK_DEG)))
    )
    turn_footprint_lhs_num = (
        turn_radius_num
        + 0.5 * float(wing_span_design_num)
        + WALL_CLEARANCE_M
    )
    n_turn_num = float(1.0 / onp.cos(onp.radians(TURN_BANK_DEG)))
    turn_lift_required_num = K_LEVEL_TURN * n_turn_num * mass_total_num * G
    yaw_rate_turn_num = float(
        K_LEVEL_TURN * G * onp.tan(onp.radians(TURN_BANK_DEG)) / max(V_TURN_MPS, 1e-8)
    )

    xyz_ref_num = [float(total_cg_x_num), float(total_cg_y_num), float(total_cg_z_num)]
    nominal_cl_delta_a_fd = cl_delta_a_finite_difference(
        airplane_base=airplane_base_num,
        xyz_ref=xyz_ref_num,
        velocity_mps=V_NOM_MPS,
        alpha_deg=float(alpha_nom_num),
        delta_a_center_deg=float(delta_a_nom_num),
        delta_e_deg=float(delta_e_nom_num),
        delta_r_deg=float(delta_r_nom_num),
        yaw_rate_rad_s=0.0,
        step_deg=CL_DELTA_A_FD_STEP_DEG,
        atmosphere=asb.Atmosphere(altitude=0.0),
    )
    turn_cl_delta_a_fd = cl_delta_a_finite_difference(
        airplane_base=airplane_base_num,
        xyz_ref=xyz_ref_num,
        velocity_mps=V_TURN_MPS,
        alpha_deg=float(alpha_turn_num),
        delta_a_center_deg=float(delta_a_turn_num),
        delta_e_deg=float(delta_e_turn_num),
        delta_r_deg=float(delta_r_turn_num),
        yaw_rate_rad_s=yaw_rate_turn_num,
        step_deg=CL_DELTA_A_FD_STEP_DEG,
        atmosphere=asb.Atmosphere(altitude=0.0),
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
    boom_length_num = boom_length_design_num
    tail_x_num = boom_end_x_design_num
    fuselage_length_num = tail_x_num - NOSE_X_M
    wing_struct_semispan_num = 0.5 * wing_span_num
    wing_struct_half_load_num = 0.5 * n_turn_num * mass_total_num * G
    wing_struct_line_load_num = wing_struct_half_load_num / max(wing_struct_semispan_num, 1e-9)
    htail_struct_semispan_num = 0.5 * htail_span_num
    htail_struct_half_load_num = 0.5 * HT_LOAD_FRACTION * n_turn_num * mass_total_num * G
    htail_struct_line_load_num = htail_struct_half_load_num / max(htail_struct_semispan_num, 1e-9)

    boundary_rows = [
        design_variable_boundary_record(
            name="alpha_nom_deg",
            value=alpha_nom_num,
            lower=ALPHA_MIN_DEG,
            upper=ALPHA_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_a_nom_deg",
            value=delta_a_nom_num,
            lower=DELTA_A_MIN_DEG,
            upper=DELTA_A_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_e_nom_deg",
            value=delta_e_nom_num,
            lower=DELTA_E_MIN_DEG,
            upper=DELTA_E_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_r_nom_deg",
            value=delta_r_nom_num,
            lower=DELTA_R_MIN_DEG,
            upper=DELTA_R_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="alpha_turn_deg",
            value=alpha_turn_num,
            lower=ALPHA_MIN_DEG,
            upper=ALPHA_MAX_TURN_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_a_turn_deg",
            value=delta_a_turn_num,
            lower=DELTA_A_MIN_DEG,
            upper=DELTA_A_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_e_turn_deg",
            value=delta_e_turn_num,
            lower=DELTA_E_MIN_DEG,
            upper=DELTA_E_MAX_DEG,
            unit="deg",
        ),
        design_variable_boundary_record(
            name="delta_r_turn_deg",
            value=delta_r_turn_num,
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
            name="boom_length_m",
            value=boom_length_design_num,
            lower=BOOM_LENGTH_MIN_M,
            upper=BOOM_LENGTH_MAX_M,
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
        {"Metric": "bank_entry_time_s", "Value": BANK_ENTRY_TIME_S, "Unit": "s"},
        {"Metric": "arena_width_m", "Value": ARENA_WIDTH_M, "Unit": "m"},
        {"Metric": "wall_clearance_m", "Value": WALL_CLEARANCE_M, "Unit": "m"},
        {"Metric": "boom_length_m", "Value": boom_length_num, "Unit": "m"},
        {"Metric": "fuselage_tail_x_m", "Value": tail_x_num, "Unit": "m"},
        {"Metric": "fuselage_length_m", "Value": fuselage_length_num, "Unit": "m"},
        {"Metric": "alpha_trim_deg", "Value": alpha_nom_num, "Unit": "deg"},
        {"Metric": "delta_a_trim_deg", "Value": delta_a_nom_num, "Unit": "deg"},
        {"Metric": "delta_e_trim_deg", "Value": delta_e_nom_num, "Unit": "deg"},
        {"Metric": "delta_r_trim_deg", "Value": delta_r_nom_num, "Unit": "deg"},
        {"Metric": "alpha_turn_deg", "Value": alpha_turn_num, "Unit": "deg"},
        {"Metric": "delta_a_turn_deg", "Value": delta_a_turn_num, "Unit": "deg"},
        {"Metric": "delta_e_turn_deg", "Value": delta_e_turn_num, "Unit": "deg"},
        {"Metric": "delta_r_turn_deg", "Value": delta_r_turn_num, "Unit": "deg"},
        {"Metric": "mass_total_kg", "Value": mass_total_num, "Unit": "kg"},
        {"Metric": "mass_total_g", "Value": mass_total_num * 1e3, "Unit": "g"},
        {
            "Metric": "mass_total_lbm",
            "Value": mass_total_num / 0.45359237,
            "Unit": "lbm",
        },
        {"Metric": "total_cg_x_m", "Value": total_cg_x_num, "Unit": "m"},
        {"Metric": "total_cg_y_m", "Value": total_cg_y_num, "Unit": "m"},
        {"Metric": "total_cg_z_m", "Value": total_cg_z_num, "Unit": "m"},
        {"Metric": "total_cg_x_weighted_m", "Value": weighted_cg_x_num, "Unit": "m"},
        {"Metric": "total_cg_y_weighted_m", "Value": weighted_cg_y_num, "Unit": "m"},
        {"Metric": "total_cg_z_weighted_m", "Value": weighted_cg_z_num, "Unit": "m"},
        {"Metric": "total_cg_x_error_m", "Value": total_cg_x_error_num, "Unit": "m"},
        {"Metric": "total_cg_y_error_m", "Value": total_cg_y_error_num, "Unit": "m"},
        {"Metric": "total_cg_z_error_m", "Value": total_cg_z_error_num, "Unit": "m"},
        {"Metric": "mass_component_sum_kg", "Value": component_mass_sum_num, "Unit": "kg"},
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
            "Metric": "roll_rate_ss_turn_radps",
            "Value": roll_rate_turn_num,
            "Unit": "rad/s",
        },
        {
            "Metric": "roll_tau_turn_s",
            "Value": roll_tau_turn_num,
            "Unit": "s",
        },
        {
            "Metric": "delta_a_turn_eff_deg",
            "Value": delta_a_turn_eff_deg_num,
            "Unit": "deg",
        },
        {
            "Metric": "bank_entry_phi_capture_proxy_deg",
            "Value": bank_entry_phi_capture_proxy_deg_num,
            "Unit": "deg",
        },
        {
            "Metric": "bank_entry_margin_deg",
            "Value": bank_entry_margin_deg_num,
            "Unit": "deg",
        },
        {
            "Metric": "bank_entry_phi_raw_deg",
            "Value": bank_entry_phi_achieved_deg_num,
            "Unit": "deg",
        },
        {
            "Metric": "bank_entry_phi_achieved_deg",
            "Value": bank_entry_phi_achieved_deg_num,
            "Unit": "deg",
        },
        {
            "Metric": "nominal_Cl_delta_a_fd",
            "Value": nominal_cl_delta_a_fd,
            "Unit": "1/rad",
        },
        {
            "Metric": "turn_Cl_delta_a_fd",
            "Value": turn_cl_delta_a_fd,
            "Unit": "1/rad",
        },
        {
            "Metric": "roll_accel0_rad_s2",
            "Value": roll_accel_num,
            "Unit": "rad/s^2",
        },
        {"Metric": "roll_tau_s", "Value": roll_tau_num, "Unit": "s"},
        {"Metric": "wing_tip_deflection_proxy_m", "Value": delta_tip_num, "Unit": "m"},
        {"Metric": "wing_tip_deflection_proxy_allow_m", "Value": delta_allow_num, "Unit": "m"},
        {
            "Metric": "wing_tip_deflection_proxy_semispan_fraction",
            "Value": delta_ratio_num,
            "Unit": "-",
        },
        {
            "Metric": "wing_tip_deflection_proxy_over_allow",
            "Value": delta_tip_num / max(delta_allow_num, 1e-9),
            "Unit": "-",
        },
        {"Metric": "wing_struct_semispan_m", "Value": wing_struct_semispan_num, "Unit": "m"},
        {"Metric": "wing_struct_half_load_n", "Value": wing_struct_half_load_num, "Unit": "N"},
        {"Metric": "wing_struct_line_load_n_m", "Value": wing_struct_line_load_num, "Unit": "N/m"},
        {"Metric": "wing_struct_E_pa", "Value": WING_E_SECANT_PA, "Unit": "Pa"},
        {"Metric": "wing_struct_EI_Nm2", "Value": wing_ei_num, "Unit": "N*m^2"},
        {"Metric": "wing_struct_thickness_m", "Value": WING_THICKNESS_M, "Unit": "m"},
        {
            "Metric": "htail_tip_deflection_proxy_m",
            "Value": delta_ht_tip_num,
            "Unit": "m",
        },
        {
            "Metric": "htail_tip_deflection_proxy_allow_m",
            "Value": delta_ht_allow_num,
            "Unit": "m",
        },
        {
            "Metric": "htail_tip_deflection_proxy_semispan_fraction",
            "Value": delta_ht_ratio_num,
            "Unit": "-",
        },
        {
            "Metric": "htail_tip_deflection_proxy_over_allow",
            "Value": delta_ht_tip_num / max(delta_ht_allow_num, 1e-9),
            "Unit": "-",
        },
        {"Metric": "htail_struct_semispan_m", "Value": htail_struct_semispan_num, "Unit": "m"},
        {"Metric": "htail_struct_half_load_n", "Value": htail_struct_half_load_num, "Unit": "N"},
        {"Metric": "htail_struct_line_load_n_m", "Value": htail_struct_line_load_num, "Unit": "N/m"},
        {"Metric": "htail_struct_E_pa", "Value": HTAIL_E_SECANT_PA, "Unit": "Pa"},
        {"Metric": "htail_struct_EI_Nm2", "Value": htail_ei_num, "Unit": "N*m^2"},
        {"Metric": "htail_struct_thickness_m", "Value": TAIL_THICKNESS_M, "Unit": "m"},
        {
            "Metric": "objective_struct_deflection_proxy_penalty",
            "Value": struct_deflection_penalty_num,
            "Unit": "-",
        },
        {
            "Metric": "objective_htail_deflection_proxy_penalty",
            "Value": htail_deflection_penalty_num,
            "Unit": "-",
        },
        {
            "Metric": "objective_roll_tau_penalty",
            "Value": roll_tau_penalty_num,
            "Unit": "-",
        },
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
        {"Parameter": "boom_length_m", "Value": boom_length_num, "Unit": "m"},
        {"Parameter": "fuselage_tail_x_m", "Value": tail_x_num, "Unit": "m"},
        {"Parameter": "fuselage_length_m", "Value": fuselage_length_num, "Unit": "m"},
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
            "Alpha turn bound",
            alpha_turn_num,
            upper=ALPHA_MAX_TURN_DEG,
        ),
        constraint_record("Turn CL cap", aero_turn_num["CL"], upper=TURN_CL_CAP),
        constraint_record(
            "Turn Lift >= K_LEVEL_TURN*n*Weight",
            aero_turn_num["L"],
            lower=turn_lift_required_num,
        ),
        constraint_record(
            "Turn Trim Cm == 0",
            aero_turn_num["Cm"],
            lower=0.0,
            upper=0.0,
            tol=1e-3,
        ),
        constraint_record(
            "Turn Trim Cl tolerance",
            aero_turn_num["Cl"],
            lower=-TURN_LATERAL_TRIM_TOL_CL,
            upper=TURN_LATERAL_TRIM_TOL_CL,
        ),
        constraint_record(
            "Turn Trim Cn tolerance",
            aero_turn_num["Cn"],
            lower=-TURN_LATERAL_TRIM_TOL_CN,
            upper=TURN_LATERAL_TRIM_TOL_CN,
        ),
        constraint_record(
            "Turn footprint in width",
            turn_footprint_lhs_num,
            upper=0.5 * ARENA_WIDTH_M,
        ),
        constraint_record(
            "Turn bank-entry capture proxy",
            bank_entry_phi_capture_proxy_deg_num,
            lower=TURN_BANK_DEG,
        ),
        constraint_record("Turn bank-entry margin", bank_entry_margin_deg_num, lower=0.0),
        constraint_record("Turn bank-entry phi raw (diag)", bank_entry_phi_achieved_deg_num),
        constraint_record("H-tail tip deflection proxy (diag)", delta_ht_tip_num),
        constraint_record("H-tail tip deflection proxy allow (diag)", delta_ht_allow_num),
        constraint_record("H-tail deflection proxy penalty (diag)", htail_deflection_penalty_num),
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
        constraint_record("Clp nominal <= -eps", aero_nom_num["Clp"], upper=-CLP_NEG_EPS),
        constraint_record("Clp turn <= -eps", aero_turn_num["Clp"], upper=-CLP_NEG_EPS),
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
            abs(delta_a_turn_num),
            upper=TURN_DEFLECTION_UTIL_MAX * DELTA_A_MAX_DEG,
        ),
        constraint_record(
            "Turn elevator trim utilization",
            abs(delta_e_turn_num),
            upper=TURN_DEFLECTION_UTIL_MAX * DELTA_E_MAX_DEG,
        ),
        constraint_record(
            "Turn rudder trim utilization",
            abs(delta_r_turn_num),
            upper=TURN_DEFLECTION_UTIL_MAX * DELTA_R_MAX_DEG,
        ),
    ]
    active_constraint_rows = build_active_constraints_rows(constraint_rows)

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
        {
            "DesignPoint": "Settings",
            "Metric": "BANK_ENTRY_TIME_S",
            "Value": BANK_ENTRY_TIME_S,
            "Unit": "s",
        },
        {"DesignPoint": "Arena", "Metric": "ARENA_LENGTH_M", "Value": ARENA_LENGTH_M, "Unit": "m"},
        {"DesignPoint": "Arena", "Metric": "ARENA_WIDTH_M", "Value": ARENA_WIDTH_M, "Unit": "m"},
        {"DesignPoint": "Arena", "Metric": "ARENA_HEIGHT_M", "Value": ARENA_HEIGHT_M, "Unit": "m"},
        {
            "DesignPoint": "Nominal",
            "Metric": "alpha_nom_deg",
            "Value": alpha_nom_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "delta_a_nom_deg",
            "Value": delta_a_nom_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "delta_e_nom_deg",
            "Value": delta_e_nom_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Nominal",
            "Metric": "delta_r_nom_deg",
            "Value": delta_r_nom_num,
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
            "DesignPoint": "Nominal",
            "Metric": "Cl_delta_a_fd",
            "Value": nominal_cl_delta_a_fd,
            "Unit": "1/rad",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "alpha_turn_deg",
            "Value": alpha_turn_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_a_turn_deg",
            "Value": delta_a_turn_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_e_turn_deg",
            "Value": delta_e_turn_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_r_turn_deg",
            "Value": delta_r_turn_num,
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
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "Cl_delta_a_fd",
            "Value": turn_cl_delta_a_fd,
            "Unit": "1/rad",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "roll_tau_turn_s",
            "Value": roll_tau_turn_num,
            "Unit": "s",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "delta_a_turn_eff_deg",
            "Value": delta_a_turn_eff_deg_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "bank_entry_phi_capture_proxy_deg",
            "Value": bank_entry_phi_capture_proxy_deg_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "bank_entry_margin_deg",
            "Value": bank_entry_margin_deg_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "bank_entry_phi_raw_deg",
            "Value": bank_entry_phi_achieved_deg_num,
            "Unit": "deg",
        },
        {
            "DesignPoint": "Manoeuvre",
            "Metric": "bank_entry_phi_achieved_deg",
            "Value": bank_entry_phi_achieved_deg_num,
            "Unit": "deg",
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
    aero_scalar_map["nominal_Cl_delta_a_fd"] = float(nominal_cl_delta_a_fd)
    aero_scalar_map["turn_Cl_delta_a_fd"] = float(turn_cl_delta_a_fd)

    candidate = Candidate(
        candidate_id=-1,
        objective=float(to_scalar(objective_num)),
        wing_span_m=float(to_scalar(wing_span_num)),
        wing_chord_m=float(to_scalar(wing_chord_num)),
        tail_arm_m=float(to_scalar(tail_arm_num)),
        htail_span_m=float(to_scalar(htail_span_num)),
        vtail_height_m=float(to_scalar(vtail_height_num)),
        alpha_deg=float(to_scalar(alpha_nom_num)),
        delta_a_deg=float(to_scalar(delta_a_nom_num)),
        delta_e_deg=float(to_scalar(delta_e_nom_num)),
        delta_r_deg=float(to_scalar(delta_r_nom_num)),
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
        active_constraint_rows=active_constraint_rows,
        boundary_rows=boundary_rows,
        design_points_rows=design_points_rows,
        solver_stats=solve_stats,
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
        active_constraint_rows=active_constraint_rows,
        solver_stats=solve_stats,
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
) -> tuple[Candidate | None, str | None]:
    candidate = legacy_single_run_main(
        init_override=init,
        ipopt_options=ipopt_options,
        export_outputs=False,
    )
    if candidate is None:
        return candidate, (_LAST_SOLVE_FAILURE_REASON or "solve_failed_or_infeasible")
    return candidate, None


def sample_initial_guess(rng: onp.random.Generator) -> dict[str, float]:
    center = default_initial_guess()

    def sample_bounded(
        key: str,
        lower: float,
        upper: float,
        sigma_frac: float,
        min_sigma: float,
    ) -> float:
        midpoint = 0.5 * (lower + upper)
        center_value = float(center.get(key, midpoint))
        sigma = max(min_sigma, sigma_frac * (upper - lower))
        value = float(rng.normal(center_value, sigma))
        return float(onp.clip(value, lower, upper))

    # Mostly exploit around a known-good seed, but keep some global exploration.
    use_global_exploration = bool(rng.uniform() < 0.20)
    if use_global_exploration:
        alpha_nom_deg = float(rng.uniform(ALPHA_MIN_DEG, ALPHA_MAX_DEG))
        delta_a_nom_deg = float(rng.uniform(DELTA_A_MIN_DEG, DELTA_A_MAX_DEG))
        delta_e_nom_deg = float(rng.uniform(DELTA_E_MIN_DEG, DELTA_E_MAX_DEG))
        delta_r_nom_deg = float(rng.uniform(DELTA_R_MIN_DEG, DELTA_R_MAX_DEG))
        alpha_turn_deg = float(rng.uniform(ALPHA_MIN_DEG, ALPHA_MAX_TURN_DEG))
        delta_a_turn_deg = float(rng.uniform(DELTA_A_MIN_DEG, DELTA_A_MAX_DEG))
        delta_e_turn_deg = float(rng.uniform(DELTA_E_MIN_DEG, DELTA_E_MAX_DEG))
        delta_r_turn_deg = float(rng.uniform(DELTA_R_MIN_DEG, DELTA_R_MAX_DEG))
        wing_span_m = float(rng.uniform(WING_SPAN_MIN_M, WING_SPAN_MAX_M))
        wing_chord_m = float(rng.uniform(WING_CHORD_MIN_M, WING_CHORD_MAX_M))
        tail_arm_m = float(rng.uniform(TAIL_ARM_MIN_M, TAIL_ARM_MAX_M))
        htail_span_m = float(rng.uniform(HT_SPAN_MIN_M, HT_SPAN_MAX_M))
        vtail_height_m = float(rng.uniform(VT_HEIGHT_MIN_M, VT_HEIGHT_MAX_M))
    else:
        wing_span_m = sample_bounded(
            "wing_span_m",
            WING_SPAN_MIN_M,
            WING_SPAN_MAX_M,
            sigma_frac=0.05,
            min_sigma=0.01,
        )
        wing_chord_m = sample_bounded(
            "wing_chord_m",
            WING_CHORD_MIN_M,
            WING_CHORD_MAX_M,
            sigma_frac=0.06,
            min_sigma=0.003,
        )
        tail_arm_m = sample_bounded(
            "tail_arm_m",
            TAIL_ARM_MIN_M,
            TAIL_ARM_MAX_M,
            sigma_frac=0.05,
            min_sigma=0.01,
        )
        htail_span_m = sample_bounded(
            "htail_span_m",
            HT_SPAN_MIN_M,
            HT_SPAN_MAX_M,
            sigma_frac=0.06,
            min_sigma=0.004,
        )
        vtail_height_m = sample_bounded(
            "vtail_height_m",
            VT_HEIGHT_MIN_M,
            VT_HEIGHT_MAX_M,
            sigma_frac=0.06,
            min_sigma=0.003,
        )
        alpha_nom_deg = sample_bounded(
            "alpha_nom_deg",
            ALPHA_MIN_DEG,
            ALPHA_MAX_DEG,
            sigma_frac=0.08,
            min_sigma=0.4,
        )
        delta_a_nom_deg = sample_bounded(
            "delta_a_nom_deg",
            DELTA_A_MIN_DEG,
            DELTA_A_MAX_DEG,
            sigma_frac=0.07,
            min_sigma=1.2,
        )
        delta_e_nom_deg = sample_bounded(
            "delta_e_nom_deg",
            DELTA_E_MIN_DEG,
            DELTA_E_MAX_DEG,
            sigma_frac=0.07,
            min_sigma=1.2,
        )
        delta_r_nom_deg = sample_bounded(
            "delta_r_nom_deg",
            DELTA_R_MIN_DEG,
            DELTA_R_MAX_DEG,
            sigma_frac=0.07,
            min_sigma=1.2,
        )
        alpha_turn_deg = sample_bounded(
            "alpha_turn_deg",
            ALPHA_MIN_DEG,
            ALPHA_MAX_TURN_DEG,
            sigma_frac=0.08,
            min_sigma=0.5,
        )
        delta_a_turn_deg = sample_bounded(
            "delta_a_turn_deg",
            DELTA_A_MIN_DEG,
            DELTA_A_MAX_DEG,
            sigma_frac=0.08,
            min_sigma=1.5,
        )
        delta_e_turn_deg = sample_bounded(
            "delta_e_turn_deg",
            DELTA_E_MIN_DEG,
            DELTA_E_MAX_DEG,
            sigma_frac=0.08,
            min_sigma=1.5,
        )
        delta_r_turn_deg = sample_bounded(
            "delta_r_turn_deg",
            DELTA_R_MIN_DEG,
            DELTA_R_MAX_DEG,
            sigma_frac=0.08,
            min_sigma=1.5,
        )

    return {
        "wing_span_m": wing_span_m,
        "wing_chord_m": wing_chord_m,
        "tail_arm_m": tail_arm_m,
        "htail_span_m": htail_span_m,
        "vtail_height_m": vtail_height_m,
        "alpha_nom_deg": alpha_nom_deg,
        "delta_a_nom_deg": delta_a_nom_deg,
        "delta_e_nom_deg": delta_e_nom_deg,
        "delta_r_nom_deg": delta_r_nom_deg,
        "alpha_turn_deg": alpha_turn_deg,
        "delta_a_turn_deg": delta_a_turn_deg,
        "delta_e_turn_deg": delta_e_turn_deg,
        "delta_r_turn_deg": delta_r_turn_deg,
        # Legacy aliases retained for backward compatibility with old init files.
        "alpha_deg": alpha_nom_deg,
        "delta_a_deg": delta_a_nom_deg,
        "delta_e_deg": delta_e_nom_deg,
        "delta_r_deg": delta_r_nom_deg,
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


def run_multistart(config: WorkflowConfig) -> tuple[list[Candidate], pd.DataFrame]:
    rng = onp.random.default_rng(config.random_seed)
    feasible_candidates: list[Candidate] = []
    all_starts_rows: list[dict[str, Any]] = []

    for start_index in range(config.n_starts):
        start_id = start_index + 1
        # First start is a deterministic warm seed; remaining starts are stochastic.
        init = default_initial_guess() if start_index == 0 else sample_initial_guess(rng)
        candidate, failure_reason = build_and_solve_once(init=init, ipopt_options=None)
        if candidate is None:
            print(f"[multistart] start {start_id}/{config.n_starts} failed", flush=True)
            all_starts_rows.append(
                {
                    "start_index": start_id,
                    "success": False,
                    "status": "failed",
                    "objective": float("nan"),
                    "failure_reason": failure_reason or "solve_failed_or_infeasible",
                    "kept_after_dedup": False,
                    "kept_rank": float("nan"),
                }
            )
            continue
        candidate.candidate_id = start_id
        feasible_candidates.append(candidate)
        all_starts_rows.append(
            {
                "start_index": start_id,
                "success": True,
                "status": "feasible",
                "objective": float(candidate.objective),
                "failure_reason": "",
                "kept_after_dedup": False,
                "kept_rank": float("nan"),
            }
        )
        print(
            (
                f"[multistart] start {start_id}/{config.n_starts} feasible "
                f"(objective={candidate.objective:.5f})"
            ),
            flush=True,
        )

    if not feasible_candidates:
        all_starts_df = pd.DataFrame(all_starts_rows).sort_values(by="start_index")
        return [], all_starts_df

    deduped: list[Candidate] = []
    for candidate in sorted(feasible_candidates, key=lambda item: item.objective):
        if any(candidates_are_duplicates(candidate, kept, config) for kept in deduped):
            continue
        deduped.append(candidate)

    deduped = sorted(deduped, key=lambda item: item.objective)[: config.keep_top_k]
    kept_rank_by_start: dict[int, int] = {}
    for idx, candidate in enumerate(deduped, start=1):
        kept_rank_by_start[int(candidate.candidate_id)] = idx
        candidate.candidate_id = idx

    for row in all_starts_rows:
        start_id = int(row["start_index"])
        if bool(row.get("success", False)) and start_id in kept_rank_by_start:
            row["kept_after_dedup"] = True
            row["kept_rank"] = kept_rank_by_start[start_id]
            row["status"] = "kept"
        elif bool(row.get("success", False)):
            row["status"] = "dropped_by_dedup_or_rank"

    all_starts_df = pd.DataFrame(all_starts_rows).sort_values(by="start_index")
    return deduped, all_starts_df


def sample_scenarios(config: WorkflowConfig) -> pd.DataFrame:
    rng = onp.random.default_rng(config.scenario_seed)
    scenario_count = max(config.n_scenarios, 0)
    eff_a = rng.uniform(config.eff_a_min, config.eff_a_max, scenario_count)
    eff_e = rng.uniform(config.eff_e_min, config.eff_e_max, scenario_count)
    eff_r = rng.uniform(config.eff_r_min, config.eff_r_max, scenario_count)
    control_eff_legacy = (eff_a + eff_e + eff_r) / 3.0

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
            # Per-axis effectiveness and neutral-bias uncertainty
            "eff_a": eff_a,
            "eff_e": eff_e,
            "eff_r": eff_r,
            "bias_a_deg": rng.uniform(config.bias_a_deg_min, config.bias_a_deg_max, scenario_count),
            "bias_e_deg": rng.uniform(config.bias_e_deg_min, config.bias_e_deg_max, scenario_count),
            "bias_r_deg": rng.uniform(config.bias_r_deg_min, config.bias_r_deg_max, scenario_count),
            # Legacy scalar control effectiveness (kept for compatibility)
            "control_eff": control_eff_legacy,
            # Inertia uncertainty scales
            "ixx_scale": rng.uniform(config.ixx_scale_min, config.ixx_scale_max, scenario_count),
            "iyy_scale": rng.uniform(config.iyy_scale_min, config.iyy_scale_max, scenario_count),
            "izz_scale": rng.uniform(config.izz_scale_min, config.izz_scale_max, scenario_count),
            # Structural uncertainty scales
            "wing_E_scale": rng.uniform(
                config.wing_E_scale_min,
                config.wing_E_scale_max,
                scenario_count,
            ),
            "htail_E_scale": rng.uniform(
                config.htail_E_scale_min,
                config.htail_E_scale_max,
                scenario_count,
            ),
            "wing_thickness_scale": rng.uniform(
                config.wing_thickness_scale_min,
                config.wing_thickness_scale_max,
                scenario_count,
            ),
            "tail_thickness_scale": rng.uniform(
                config.tail_thickness_scale_min,
                config.tail_thickness_scale_max,
                scenario_count,
            ),
            # Updraft disturbance model
            "w_gust_nom": rng.uniform(config.w_gust_nom_min, config.w_gust_nom_max, scenario_count),
            "w_gust_turn": rng.uniform(config.w_gust_turn_min, config.w_gust_turn_max, scenario_count),
            "drag_factor": rng.uniform(
                config.drag_factor_min,
                config.drag_factor_max,
                scenario_count,
            ),
        }
    )


def compute_alpha_gust_deg(w_gust_mps: float, v_mps: float) -> float:
    return float(onp.degrees(onp.arctan(float(w_gust_mps) / max(float(v_mps), 1e-6))))


def servo_command_bounds(
    delta_min_deg: float,
    delta_max_deg: float,
    eff_axis: float,
    bias_axis_deg: float,
    util_fraction: float,
) -> tuple[float, float]:
    eff_safe = max(abs(float(eff_axis)), 1e-3)
    delta_min_u = float(util_fraction) * float(delta_min_deg)
    delta_max_u = float(util_fraction) * float(delta_max_deg)
    u_min = (delta_min_u - float(bias_axis_deg)) / eff_safe
    u_max = (delta_max_u - float(bias_axis_deg)) / eff_safe
    if u_min > u_max:
        u_min, u_max = u_max, u_min
    return float(u_min), float(u_max)


def struct_tip_deflection_proxy(
    total_force_n: float,
    span_m: float,
    chord_m: float,
    thickness_m: float,
    e_secant_pa: float,
    e_scale: float,
    allow_frac: float,
    thickness_scale: float = 1.0,
) -> dict[str, float]:
    # Use a semispan cantilever with half of the total lifting force.
    semispan_m = 0.5 * max(float(span_m), 1e-9)
    thickness_eff_m = max(float(thickness_m) * max(float(thickness_scale), 1e-6), 1e-6)
    i_plate_m4 = max(float(chord_m), 1e-9) * (thickness_eff_m ** 3) / 12.0
    e_eff_pa = max(float(e_secant_pa) * max(float(e_scale), 1e-6), 1e-6)
    half_force_n = 0.5 * float(total_force_n)
    line_load_n_m = half_force_n / max(semispan_m, 1e-9)

    delta_tip_m = (
        line_load_n_m
        * (semispan_m ** 4)
        / (8.0 * e_eff_pa * max(i_plate_m4, 1e-12))
    )
    delta_allow_m = max(float(allow_frac) * semispan_m, 1e-9)
    defl_over_allow = delta_tip_m / delta_allow_m
    return {
        "delta_tip_m": float(delta_tip_m),
        "delta_allow_m": float(delta_allow_m),
        "defl_over_allow": float(defl_over_allow),
        "semispan_m": float(semispan_m),
        "half_force_n": float(half_force_n),
        "line_load_n_m": float(line_load_n_m),
        "thickness_m": float(thickness_eff_m),
        "e_pa": float(e_eff_pa),
        "i_m4": float(i_plate_m4),
    }


def struct_tip_deflection_proxy_composite(
    total_force_n: float,
    span_m: float,
    chord_m: float,
    foam_thickness_m: float,
    e_foam_pa: float,
    e_foam_scale: float,
    allow_frac: float,
    thickness_scale: float,
    include_spar: bool,
    spar_od_m: float,
    spar_id_m: float,
    e_spar_pa: float,
    spar_z_from_lower_m: float,
    include_tape: bool,
    tape_width_m: float,
    tape_thickness_m: float,
    e_tape_pa: float,
) -> dict[str, float]:
    semispan_m = 0.5 * max(float(span_m), 1e-9)
    t_eff_m = max(float(foam_thickness_m) * max(float(thickness_scale), 1e-6), 1e-6)
    e_foam_eff_pa = max(float(e_foam_pa) * max(float(e_foam_scale), 1e-6), 1e-6)
    chord_eff_m = max(float(chord_m), 1e-9)
    tape_w_m = min(float(tape_width_m), chord_eff_m)

    ei_nm2, z0_m = composite_EI_flapwise(
        chord_m=chord_eff_m,
        foam_thickness_m=t_eff_m,
        e_foam_pa=e_foam_eff_pa,
        spar_od_m=float(spar_od_m),
        spar_id_m=float(spar_id_m),
        e_spar_pa=float(e_spar_pa),
        spar_z_from_lower_m=float(spar_z_from_lower_m),
        include_spar=bool(include_spar),
        tape_width_m=tape_w_m,
        tape_thickness_m=float(tape_thickness_m),
        e_tape_pa=float(e_tape_pa),
        include_tape=bool(include_tape),
    )

    half_force_n = 0.5 * float(total_force_n)
    line_load_n_m = half_force_n / max(semispan_m, 1e-9)
    delta_tip_m = line_load_n_m * (semispan_m**4) / (8.0 * max(float(ei_nm2), 1e-12))
    delta_allow_m = max(float(allow_frac) * semispan_m, 1e-9)
    defl_over_allow = delta_tip_m / delta_allow_m

    i_plate_m4 = chord_eff_m * (t_eff_m**3) / 12.0

    return {
        "delta_tip_m": float(delta_tip_m),
        "delta_allow_m": float(delta_allow_m),
        "defl_over_allow": float(defl_over_allow),
        "semispan_m": float(semispan_m),
        "half_force_n": float(half_force_n),
        "line_load_n_m": float(line_load_n_m),
        "thickness_m": float(t_eff_m),
        "e_foam_pa": float(e_foam_eff_pa),
        "EI_Nm2": float(ei_nm2),
        "z0_m": float(z0_m),
        # Legacy aliases kept for compatibility with existing downstream tables.
        "e_pa": float(e_foam_eff_pa),
        "i_m4": float(i_plate_m4),
    }


def trim_candidate_at_point(
    candidate: Candidate,
    scenario_row: dict[str, Any],
    config: WorkflowConfig,
    point: Literal["nom", "turn"],
) -> dict[str, Any]:
    mass_scale = float(scenario_row["mass_scale"])
    cg_x_shift_mac = float(scenario_row["cg_x_shift_mac"])
    incidence_bias_deg = float(scenario_row["incidence_bias_deg"])
    eff_a = float(scenario_row.get("eff_a", scenario_row.get("control_eff", 1.0)))
    eff_e = float(scenario_row.get("eff_e", scenario_row.get("control_eff", 1.0)))
    eff_r = float(scenario_row.get("eff_r", scenario_row.get("control_eff", 1.0)))
    bias_a_deg = float(scenario_row.get("bias_a_deg", 0.0))
    bias_e_deg = float(scenario_row.get("bias_e_deg", 0.0))
    bias_r_deg = float(scenario_row.get("bias_r_deg", 0.0))
    ixx_scale = float(scenario_row.get("ixx_scale", 1.0))
    iyy_scale = float(scenario_row.get("iyy_scale", 1.0))
    izz_scale = float(scenario_row.get("izz_scale", 1.0))
    wing_e_scale = float(scenario_row.get("wing_E_scale", 1.0))
    htail_e_scale = float(scenario_row.get("htail_E_scale", 1.0))
    wing_thickness_scale = float(scenario_row.get("wing_thickness_scale", 1.0))
    tail_thickness_scale = float(scenario_row.get("tail_thickness_scale", 1.0))
    drag_factor = float(scenario_row["drag_factor"])

    if point == "nom":
        velocity_mps = V_NOM_MPS
        n_req_struct = 1.0
        w_gust_mps = float(scenario_row.get("w_gust_nom", 0.0))
        alpha_upper_bound = float(ALPHA_MAX_DEG)
        cl_cap = float(MAX_CL_AT_DESIGN_POINT)
        util_fraction = float(config.max_trim_util_fraction)
        bank_angle_deg = 0.0
        lift_k = 1.0
        use_coordinated_turn = False
        trim_time_s = float(config.nom_trim_time_s)
        include_sink_in_objective = True
    else:
        velocity_mps = V_TURN_MPS
        phi_turn_rad = float(onp.radians(TURN_BANK_DEG))
        n_turn = float(1.0 / onp.cos(phi_turn_rad))
        # Structural proxy continues to use full turn load, matching main model.
        n_req_struct = n_turn
        w_gust_mps = float(scenario_row.get("w_gust_turn", 0.0))
        alpha_upper_bound = float(ALPHA_MAX_TURN_DEG)
        cl_cap = float(TURN_CL_CAP)
        util_fraction = float(config.turn_deflection_util)
        bank_angle_deg = float(TURN_BANK_DEG)
        lift_k = float(K_LEVEL_TURN)
        use_coordinated_turn = True
        trim_time_s = float(config.turn_trim_time_s)
        include_sink_in_objective = False

    # Disturbance modeled as vertical gust -> AoA perturbation; spanwise shear-induced roll bias excluded at optimiser fidelity.
    alpha_gust_deg = compute_alpha_gust_deg(w_gust_mps=w_gust_mps, v_mps=velocity_mps)

    u_a_min, u_a_max = servo_command_bounds(
        DELTA_A_MIN_DEG,
        DELTA_A_MAX_DEG,
        eff_a,
        bias_a_deg,
        util_fraction,
    )
    u_e_min, u_e_max = servo_command_bounds(
        DELTA_E_MIN_DEG,
        DELTA_E_MAX_DEG,
        eff_e,
        bias_e_deg,
        util_fraction,
    )
    u_r_min, u_r_max = servo_command_bounds(
        DELTA_R_MIN_DEG,
        DELTA_R_MAX_DEG,
        eff_r,
        bias_r_deg,
        util_fraction,
    )

    rate_limit_deg = float(
        config.rate_util_fraction
        * config.servo_rate_deg_s
        * max(trim_time_s, 1e-6)
    )

    u_a_init = (float(candidate.delta_a_deg) - bias_a_deg) / max(abs(eff_a), 1e-3)
    u_e_init = (float(candidate.delta_e_deg) - bias_e_deg) / max(abs(eff_e), 1e-3)
    u_r_init = (float(candidate.delta_r_deg) - bias_r_deg) / max(abs(eff_r), 1e-3)

    opti = asb.Opti()
    alpha_trim_deg = opti.variable(
        init_guess=candidate.alpha_deg,
        lower_bound=ALPHA_MIN_DEG,
        upper_bound=alpha_upper_bound,
    )
    u_a_deg = opti.variable(
        init_guess=float(onp.clip(u_a_init, u_a_min, u_a_max)),
        lower_bound=u_a_min,
        upper_bound=u_a_max,
    )
    u_e_deg = opti.variable(
        init_guess=float(onp.clip(u_e_init, u_e_min, u_e_max)),
        lower_bound=u_e_min,
        upper_bound=u_e_max,
    )
    u_r_deg = opti.variable(
        init_guess=float(onp.clip(u_r_init, u_r_min, u_r_max)),
        lower_bound=u_r_min,
        upper_bound=u_r_max,
    )

    delta_a_deg = bias_a_deg + eff_a * u_a_deg
    delta_e_deg = bias_e_deg + eff_e * u_e_deg
    delta_r_deg = bias_r_deg + eff_r * u_r_deg

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
    boom_end_x_m = candidate.tail_arm_m + BOOM_END_BEFORE_ELEV_FRAC * htail_chord_m
    fuselage = build_fuselage(
        boom_end_x_m=boom_end_x_m,
    )

    airplane_base = asb.Airplane(
        name=f"Nausicaa candidate {candidate.candidate_id}",
        wings=[wing, htail, vtail],
        fuselages=[fuselage],
    )
    airplane = airplane_base.with_control_deflections(
        {
            "aileron": delta_a_deg,
            "elevator": delta_e_deg,
            "rudder": delta_r_deg,
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

    trim_metrics = build_trim_constraints_and_metrics(
        opti=opti,
        airplane=airplane,
        xyz_ref=xyz_ref,
        velocity_mps=velocity_mps,
        alpha_deg=alpha_trim_deg + incidence_bias_deg + alpha_gust_deg,
        mass_kg=mass_scale * candidate.mass_total_kg,
        mode="nominal" if point == "nom" else "turn",
        bank_angle_deg=bank_angle_deg,
        lift_k=lift_k,
        cl_cap=cl_cap,
        enforce_lateral_trim=(point == "nom"),
        use_coordinated_turn=use_coordinated_turn,
        atmosphere=asb.Atmosphere(altitude=0.0),
    )
    op_point = trim_metrics["op_point"]
    aero = trim_metrics["aero"]

    weight_n = mass_scale * candidate.mass_total_kg * G
    drag_with_factor = aero["D"] * drag_factor
    sink_rate_mps = drag_with_factor * velocity_mps / np.maximum(weight_n, 1e-8)
    trim_penalty = u_e_deg ** 2 + 0.3 * u_r_deg ** 2 + 0.15 * u_a_deg ** 2

    if include_sink_in_objective:
        opti.minimize(sink_rate_mps + CONTROL_TRIM_WEIGHT * trim_penalty)
    else:
        opti.minimize(CONTROL_TRIM_WEIGHT * trim_penalty)

    constraints = [
        *trim_metrics["constraints"],
        opti.bounded(u_a_min, u_a_deg, u_a_max),
        opti.bounded(u_e_min, u_e_deg, u_e_max),
        opti.bounded(u_r_min, u_r_deg, u_r_max),
        opti.bounded(-rate_limit_deg, u_a_deg - u_a_init, rate_limit_deg),
        opti.bounded(-rate_limit_deg, u_e_deg - u_e_init, rate_limit_deg),
        opti.bounded(-rate_limit_deg, u_r_deg - u_r_init, rate_limit_deg),
    ]
    opti.subject_to(constraints)
    opti.solver(
        "ipopt",
        {"print_time": False, "verbose": False},
        {
            "max_iter": 500,
            "hessian_approximation": "limited-memory",
            "print_level": 0,
            "sb": "yes",
        },
    )

    ixx_nom = float("nan")
    iyy_nom = float("nan")
    izz_nom = float("nan")
    if candidate.total_mass is not None:
        try:
            inertia_tensor_nom = onp.asarray(candidate.total_mass.inertia_tensor, dtype=float)
            ixx_nom = float(abs(inertia_tensor_nom[0, 0]))
            iyy_nom = float(abs(inertia_tensor_nom[1, 1]))
            izz_nom = float(abs(inertia_tensor_nom[2, 2]))
        except Exception:
            pass

    ixx_scaled = mass_scale * ixx_scale * ixx_nom if onp.isfinite(ixx_nom) else float("nan")
    iyy_scaled = mass_scale * iyy_scale * iyy_nom if onp.isfinite(iyy_nom) else float("nan")
    izz_scaled = mass_scale * izz_scale * izz_nom if onp.isfinite(izz_nom) else float("nan")

    nan_result = {
        f"{point}_success": False,
        f"{point}_alpha_deg": onp.nan,
        f"{point}_u_a_deg": onp.nan,
        f"{point}_u_e_deg": onp.nan,
        f"{point}_u_r_deg": onp.nan,
        f"{point}_u_a_init": u_a_init,
        f"{point}_u_e_init": u_e_init,
        f"{point}_u_r_init": u_r_init,
        f"{point}_u_delta_a": onp.nan,
        f"{point}_u_delta_e": onp.nan,
        f"{point}_u_delta_r": onp.nan,
        f"{point}_delta_a_deg": onp.nan,
        f"{point}_delta_e_deg": onp.nan,
        f"{point}_delta_r_deg": onp.nan,
        f"{point}_sink_rate_mps": onp.nan,
        f"{point}_L_over_D": onp.nan,
        f"{point}_CL": onp.nan,
        f"{point}_D": onp.nan,
        f"{point}_Cl_delta_a_fd": onp.nan,
        f"{point}_alpha_margin_deg": onp.nan,
        f"{point}_cl_margin_to_cap": onp.nan,
        f"{point}_util_a": onp.nan,
        f"{point}_util_e": onp.nan,
        f"{point}_util_r": onp.nan,
        f"{point}_u_rate_util_a": onp.nan,
        f"{point}_u_rate_util_e": onp.nan,
        f"{point}_u_rate_util_r": onp.nan,
        f"{point}_w_gust": w_gust_mps,
        f"{point}_roll_tau_s": onp.nan,
        f"{point}_roll_accel0": onp.nan,
        f"{point}_roll_rate_ss": onp.nan,
        f"{point}_wing_tip_deflection_proxy_m": onp.nan,
        f"{point}_wing_tip_deflection_proxy_allow_m": onp.nan,
        f"{point}_wing_deflection_proxy_over_allow": onp.nan,
        f"{point}_wing_struct_EI_Nm2": onp.nan,
        f"{point}_htail_tip_deflection_proxy_m": onp.nan,
        f"{point}_htail_tip_deflection_proxy_allow_m": onp.nan,
        f"{point}_htail_deflection_proxy_over_allow": onp.nan,
        f"{point}_htail_struct_EI_Nm2": onp.nan,
        f"{point}_ixx": ixx_scaled,
        f"{point}_iyy": iyy_scaled,
        f"{point}_izz": izz_scaled,
    }

    try:
        solution = opti.solve()
    except RuntimeError:
        return nan_result

    alpha_num = float(to_scalar(solution(alpha_trim_deg)))
    u_a_num = float(to_scalar(solution(u_a_deg)))
    u_e_num = float(to_scalar(solution(u_e_deg)))
    u_r_num = float(to_scalar(solution(u_r_deg)))
    delta_a_num = float(to_scalar(solution(delta_a_deg)))
    delta_e_num = float(to_scalar(solution(delta_e_deg)))
    delta_r_num = float(to_scalar(solution(delta_r_deg)))
    aero_num = solution(aero)
    drag_num = float(to_scalar(aero_num["D"])) * drag_factor
    lift_num = float(to_scalar(aero_num["L"]))
    cl_num = float(to_scalar(aero_num["CL"]))
    sink_rate_num = drag_num * velocity_mps / max(weight_n, 1e-8)
    l_over_d_num = lift_num / max(drag_num, 1e-8)
    clp_mag = max(abs(float(to_scalar(aero_num["Clp"]))), 1e-5)
    cl_delta_a_proxy = float(
        to_scalar(
            aileron_effectiveness_proxy(
                aero=aero_num,
                eta_inboard=AILERON_ETA_INBOARD,
                eta_outboard=AILERON_ETA_OUTBOARD,
                chord_fraction=AILERON_CHORD_FRACTION,
            )
        )
    )
    cl_delta_a_fd = float(
        cl_delta_a_finite_difference(
            airplane_base=airplane_base,
            xyz_ref=[float(xyz_ref[0]), float(xyz_ref[1]), float(xyz_ref[2])],
            velocity_mps=float(velocity_mps),
            alpha_deg=float(alpha_num + incidence_bias_deg + alpha_gust_deg),
            delta_a_center_deg=float(delta_a_num),
            delta_e_deg=float(delta_e_num),
            delta_r_deg=float(delta_r_num),
            yaw_rate_rad_s=float(to_scalar(trim_metrics["yaw_rate_rad_s"])),
            step_deg=CL_DELTA_A_FD_STEP_DEG,
            atmosphere=asb.Atmosphere(altitude=0.0),
        )
    )
    cl_delta_a_mag = abs(cl_delta_a_fd) if onp.isfinite(cl_delta_a_fd) else abs(cl_delta_a_proxy)
    delta_a_rate_rad = float(
        onp.radians(
            (TURN_DEFLECTION_UTIL_MAX if point == "turn" else 1.0) * DELTA_A_MAX_DEG
        )
    )
    q_dyn = 0.5 * RHO * (velocity_mps ** 2)
    wing_area_m2 = float(to_scalar(wing.area()))
    roll_rate_ss = (
        2.0
        * velocity_mps
        / max(candidate.wing_span_m, 1e-8)
        * cl_delta_a_mag
        * delta_a_rate_rad
        / clp_mag
    )
    if onp.isfinite(ixx_scaled):
        roll_accel0 = (
            q_dyn
            * wing_area_m2
            * candidate.wing_span_m
            * cl_delta_a_mag
            * delta_a_rate_rad
            / max(ixx_scaled, 1e-8)
        )
        roll_tau_s = (
            2.0
            * max(ixx_scaled, 1e-8)
            * velocity_mps
            / max(q_dyn * wing_area_m2 * (candidate.wing_span_m ** 2) * clp_mag, 1e-8)
        )
    else:
        roll_accel0 = float("nan")
        roll_tau_s = float("nan")

    wing_struct = struct_tip_deflection_proxy_composite(
        total_force_n=n_req_struct * weight_n,
        span_m=candidate.wing_span_m,
        chord_m=candidate.wing_chord_m,
        foam_thickness_m=WING_THICKNESS_M,
        e_foam_pa=WING_E_SECANT_PA,
        e_foam_scale=wing_e_scale,
        allow_frac=WING_DEFLECTION_ALLOW_FRAC,
        thickness_scale=wing_thickness_scale,
        include_spar=WING_SPAR_ENABLE,
        spar_od_m=float(WING_SPAR_OD_M),
        spar_id_m=float(WING_SPAR_ID_M),
        e_spar_pa=float(WING_SPAR_E_FLEX_PA),
        spar_z_from_lower_m=float(WING_SPAR_Z_FROM_LOWER_M),
        include_tape=TAPE_ENABLE_WING,
        tape_width_m=float(TAPE_WIDTH_M),
        tape_thickness_m=float(TAPE_THICKNESS_M),
        e_tape_pa=float(TAPE_EFFICIENCY * TAPE_E_EFFECTIVE_PA),
    )
    htail_struct = struct_tip_deflection_proxy_composite(
        total_force_n=HT_LOAD_FRACTION * n_req_struct * weight_n,
        span_m=candidate.htail_span_m,
        chord_m=float(to_scalar(htail_chord_m)),
        foam_thickness_m=TAIL_THICKNESS_M,
        e_foam_pa=HTAIL_E_SECANT_PA,
        e_foam_scale=htail_e_scale,
        allow_frac=HT_DEFLECTION_ALLOW_FRAC,
        thickness_scale=tail_thickness_scale,
        include_spar=False,
        spar_od_m=float(WING_SPAR_OD_M),
        spar_id_m=float(WING_SPAR_ID_M),
        e_spar_pa=float(WING_SPAR_E_FLEX_PA),
        spar_z_from_lower_m=0.0,
        include_tape=TAPE_ENABLE_TAIL,
        tape_width_m=float(TAPE_WIDTH_M),
        tape_thickness_m=float(TAPE_THICKNESS_M),
        e_tape_pa=float(TAPE_EFFICIENCY * TAPE_E_EFFECTIVE_PA),
    )

    return {
        # Keep robust feasibility aligned with main optimization:
        # turn-point structural deflection proxy remains diagnostic/soft, not a hard pass-fail gate.
        f"{point}_success": True,
        f"{point}_alpha_deg": alpha_num,
        f"{point}_u_a_deg": u_a_num,
        f"{point}_u_e_deg": u_e_num,
        f"{point}_u_r_deg": u_r_num,
        f"{point}_u_a_init": u_a_init,
        f"{point}_u_e_init": u_e_init,
        f"{point}_u_r_init": u_r_init,
        f"{point}_u_delta_a": u_a_num - u_a_init,
        f"{point}_u_delta_e": u_e_num - u_e_init,
        f"{point}_u_delta_r": u_r_num - u_r_init,
        f"{point}_delta_a_deg": delta_a_num,
        f"{point}_delta_e_deg": delta_e_num,
        f"{point}_delta_r_deg": delta_r_num,
        f"{point}_sink_rate_mps": sink_rate_num if point == "nom" else onp.nan,
        f"{point}_L_over_D": l_over_d_num,
        f"{point}_CL": cl_num,
        f"{point}_D": drag_num,
        f"{point}_Cl_delta_a_fd": cl_delta_a_fd,
        f"{point}_alpha_margin_deg": alpha_upper_bound - alpha_num,
        f"{point}_cl_margin_to_cap": cl_cap - cl_num,
        f"{point}_util_a": abs(delta_a_num) / max(abs(DELTA_A_MAX_DEG), 1e-8),
        f"{point}_util_e": abs(delta_e_num) / max(abs(DELTA_E_MAX_DEG), 1e-8),
        f"{point}_util_r": abs(delta_r_num) / max(abs(DELTA_R_MAX_DEG), 1e-8),
        f"{point}_u_rate_util_a": abs(u_a_num - u_a_init) / max(rate_limit_deg, 1e-8),
        f"{point}_u_rate_util_e": abs(u_e_num - u_e_init) / max(rate_limit_deg, 1e-8),
        f"{point}_u_rate_util_r": abs(u_r_num - u_r_init) / max(rate_limit_deg, 1e-8),
        f"{point}_w_gust": w_gust_mps,
        f"{point}_roll_tau_s": float(roll_tau_s),
        f"{point}_roll_accel0": float(roll_accel0),
        f"{point}_roll_rate_ss": float(roll_rate_ss),
        f"{point}_wing_tip_deflection_proxy_m": wing_struct["delta_tip_m"],
        f"{point}_wing_tip_deflection_proxy_allow_m": wing_struct["delta_allow_m"],
        f"{point}_wing_deflection_proxy_over_allow": wing_struct["defl_over_allow"],
        f"{point}_wing_struct_semispan_m": wing_struct["semispan_m"],
        f"{point}_wing_struct_half_load_n": wing_struct["half_force_n"],
        f"{point}_wing_struct_E_pa": wing_struct["e_pa"],
        f"{point}_wing_struct_EI_Nm2": wing_struct["EI_Nm2"],
        f"{point}_wing_struct_thickness_m": wing_struct["thickness_m"],
        f"{point}_htail_tip_deflection_proxy_m": htail_struct["delta_tip_m"],
        f"{point}_htail_tip_deflection_proxy_allow_m": htail_struct["delta_allow_m"],
        f"{point}_htail_deflection_proxy_over_allow": htail_struct["defl_over_allow"],
        f"{point}_htail_struct_semispan_m": htail_struct["semispan_m"],
        f"{point}_htail_struct_half_load_n": htail_struct["half_force_n"],
        f"{point}_htail_struct_E_pa": htail_struct["e_pa"],
        f"{point}_htail_struct_EI_Nm2": htail_struct["EI_Nm2"],
        f"{point}_htail_struct_thickness_m": htail_struct["thickness_m"],
        f"{point}_ixx": ixx_scaled,
        f"{point}_iyy": iyy_scaled,
        f"{point}_izz": izz_scaled,
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
    scenarios_df = scenarios_df.drop(
        columns=["dw_dy", "nom_dw_dy", "turn_dw_dy", "nom_Cl_bias", "turn_Cl_bias"],
        errors="ignore",
    )
    robust_rows: list[dict[str, Any]] = []
    scenario_rows = scenarios_df.to_dict(orient="records")

    for candidate in candidates:
        for scenario in scenario_rows:
            nom_row = trim_candidate_at_point(
                candidate=candidate,
                scenario_row=scenario,
                config=config,
                point="nom",
            )
            turn_row = trim_candidate_at_point(
                candidate=candidate,
                scenario_row=scenario,
                config=config,
                point="turn",
            )
            both_success = bool(
                nom_row.get("nom_success", False) and turn_row.get("turn_success", False)
            )
            combined_row: dict[str, Any] = {
                "candidate_id": candidate.candidate_id,
                **scenario,
                **nom_row,
                **turn_row,
                "both_success": both_success,
                # Legacy compatibility alias
                "trim_success": both_success,
            }
            robust_rows.append(combined_row)

    robust_scenarios_df = pd.DataFrame(robust_rows)
    summary_rows: list[dict[str, Any]] = []
    objective_by_candidate = {candidate.candidate_id: candidate.objective for candidate in candidates}

    def col_max(df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return float("nan")
        values = df[column].dropna().to_numpy(dtype=float)
        return float(onp.max(values)) if values.size else float("nan")

    def col_min(df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return float("nan")
        values = df[column].dropna().to_numpy(dtype=float)
        return float(onp.min(values)) if values.size else float("nan")

    for candidate in candidates:
        candidate_df = robust_scenarios_df[
            robust_scenarios_df["candidate_id"] == candidate.candidate_id
        ]
        feasible_df = candidate_df[candidate_df["both_success"] == True]

        scenario_count = max(len(candidate_df), 1)
        feasible_rate = len(feasible_df) / scenario_count
        sink_values = feasible_df["nom_sink_rate_mps"].dropna().to_numpy(dtype=float)

        sink_mean = float(onp.mean(sink_values)) if sink_values.size else float("nan")
        sink_std = float(onp.std(sink_values)) if sink_values.size else float("nan")
        sink_worst = float(onp.max(sink_values)) if sink_values.size else float("nan")
        sink_cvar_20 = sink_cvar(sink_values, tail_fraction=0.20)
        penalty_value = sink_cvar_20 if onp.isfinite(sink_cvar_20) else 1e6
        selection_score = (1.0 - feasible_rate) * 1e3 + penalty_value

        turn_util_values = onp.array(
            [
                col_max(feasible_df, "turn_util_a"),
                col_max(feasible_df, "turn_util_e"),
                col_max(feasible_df, "turn_util_r"),
            ],
            dtype=float,
        )
        turn_util_values = turn_util_values[onp.isfinite(turn_util_values)]
        max_turn_util_worst = float(onp.max(turn_util_values)) if turn_util_values.size else float("nan")

        roll_tau_values = onp.array(
            [
                col_max(feasible_df, "nom_roll_tau_s"),
                col_max(feasible_df, "turn_roll_tau_s"),
            ],
            dtype=float,
        )
        roll_tau_values = roll_tau_values[onp.isfinite(roll_tau_values)]
        max_roll_tau_worst = float(onp.max(roll_tau_values)) if roll_tau_values.size else float("nan")

        summary_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feasible_rate": feasible_rate,
                "sink_mean": sink_mean,
                "sink_std": sink_std,
                "sink_worst": sink_worst,
                "sink_cvar_20": sink_cvar_20,
                "max_turn_util_worst": max_turn_util_worst,
                "min_turn_alpha_margin_worst": col_min(feasible_df, "turn_alpha_margin_deg"),
                "max_turn_wing_deflection_proxy_over_allow_worst": col_max(
                    feasible_df,
                    "turn_wing_deflection_proxy_over_allow",
                ),
                "max_roll_tau_worst": max_roll_tau_worst,
                # Legacy-compatible summary aliases
                "max_delta_e_util_worst": col_max(feasible_df, "nom_util_e"),
                "max_alpha_worst": col_max(feasible_df, "nom_alpha_deg"),
                "min_alpha_margin_worst": col_min(feasible_df, "nom_alpha_margin_deg"),
                "min_cl_margin_worst": col_min(feasible_df, "nom_cl_margin_to_cap"),
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


def build_worst_case_report(
    candidates: list[Candidate],
    robust_scenarios_df: pd.DataFrame,
) -> pd.DataFrame:
    core_columns = ["candidate_id", "metric", "mode", "value", "scenario_id"]
    scenario_trace_candidates = [
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "eff_a",
        "eff_e",
        "eff_r",
        "bias_a_deg",
        "bias_e_deg",
        "bias_r_deg",
        "ixx_scale",
        "iyy_scale",
        "izz_scale",
        "wing_E_scale",
        "htail_E_scale",
        "wing_thickness_scale",
        "tail_thickness_scale",
        "w_gust_nom",
        "w_gust_turn",
        "drag_factor",
        "both_success",
    ]
    key_output_candidates = [
        "nom_alpha_deg",
        "nom_alpha_margin_deg",
        "nom_CL",
        "nom_cl_margin_to_cap",
        "nom_sink_rate_mps",
        "nom_util_a",
        "nom_util_e",
        "nom_util_r",
        "nom_roll_tau_s",
        "turn_alpha_deg",
        "turn_alpha_margin_deg",
        "turn_CL",
        "turn_cl_margin_to_cap",
        "turn_util_a",
        "turn_util_e",
        "turn_util_r",
        "turn_wing_deflection_proxy_over_allow",
        "turn_htail_deflection_proxy_over_allow",
        "turn_roll_tau_s",
    ]
    scenario_trace_columns = [
        col for col in scenario_trace_candidates if col in robust_scenarios_df.columns
    ]
    key_output_columns = [
        col for col in key_output_candidates if col in robust_scenarios_df.columns
    ]
    report_columns = core_columns + scenario_trace_columns + key_output_columns

    metric_specs: list[dict[str, Any]] = [
        {"metric": "Nom sink rate", "mode": "max", "kind": "single", "column": "nom_sink_rate_mps"},
        {"metric": "Nom alpha margin", "mode": "min", "kind": "single", "column": "nom_alpha_margin_deg"},
        {"metric": "Nom CL margin to cap", "mode": "min", "kind": "single", "column": "nom_cl_margin_to_cap"},
        {"metric": "Turn alpha margin", "mode": "min", "kind": "single", "column": "turn_alpha_margin_deg"},
        {"metric": "Turn util a", "mode": "max", "kind": "single", "column": "turn_util_a"},
        {"metric": "Turn util e", "mode": "max", "kind": "single", "column": "turn_util_e"},
        {"metric": "Turn util r", "mode": "max", "kind": "single", "column": "turn_util_r"},
        {
            "metric": "Turn wing deflection proxy over allow",
            "mode": "max",
            "kind": "single",
            "column": "turn_wing_deflection_proxy_over_allow",
        },
        {
            "metric": "Turn htail deflection proxy over allow",
            "mode": "max",
            "kind": "single",
            "column": "turn_htail_deflection_proxy_over_allow",
        },
        {
            "metric": "Max roll tau",
            "mode": "max",
            "kind": "row_max",
            "columns": ["nom_roll_tau_s", "turn_roll_tau_s"],
        },
    ]

    report_rows: list[dict[str, Any]] = []

    def make_empty_row(candidate_id: int, metric: str, mode: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "metric": metric,
            "mode": mode,
            "value": float("nan"),
            "scenario_id": float("nan"),
        }
        for col in scenario_trace_columns:
            row[col] = float("nan")
        for col in key_output_columns:
            row[col] = float("nan")
        return row

    for candidate in candidates:
        candidate_df = robust_scenarios_df[
            robust_scenarios_df["candidate_id"] == candidate.candidate_id
        ]
        if "both_success" in candidate_df.columns:
            feasible_df = candidate_df[candidate_df["both_success"] == True]
        elif "trim_success" in candidate_df.columns:
            feasible_df = candidate_df[candidate_df["trim_success"] == True]
        else:
            feasible_df = candidate_df.iloc[0:0]

        for spec in metric_specs:
            row = make_empty_row(
                candidate_id=candidate.candidate_id,
                metric=spec["metric"],
                mode=spec["mode"],
            )
            if feasible_df.empty:
                report_rows.append(row)
                continue

            source_index: Any = None
            metric_value = float("nan")
            if spec["kind"] == "single":
                metric_column = spec["column"]
                if metric_column in feasible_df.columns:
                    metric_series = feasible_df[metric_column].dropna()
                    if not metric_series.empty:
                        if spec["mode"] == "max":
                            source_index = metric_series.idxmax()
                        else:
                            source_index = metric_series.idxmin()
                        metric_value = float(metric_series.loc[source_index])
            elif spec["kind"] == "row_max":
                roll_columns = [col for col in spec["columns"] if col in feasible_df.columns]
                if roll_columns:
                    row_max_series = feasible_df[roll_columns].max(axis=1, skipna=True)
                    row_max_series = row_max_series.dropna()
                    if not row_max_series.empty:
                        source_index = row_max_series.idxmax()
                        metric_value = float(row_max_series.loc[source_index])

            if source_index is None or source_index not in feasible_df.index:
                report_rows.append(row)
                continue

            source_row = feasible_df.loc[source_index]
            row["value"] = metric_value
            if "scenario_id" in feasible_df.columns:
                row["scenario_id"] = source_row["scenario_id"]
            for col in scenario_trace_columns:
                row[col] = source_row[col]
            for col in key_output_columns:
                row[col] = source_row[col]
            report_rows.append(row)

    worst_case_report_df = pd.DataFrame(report_rows)
    if worst_case_report_df.empty:
        return pd.DataFrame(columns=report_columns)
    return worst_case_report_df.reindex(columns=report_columns)


def save_workflow_workbook(
    config: WorkflowConfig,
    candidates: list[Candidate],
    robust_scenarios_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
    all_starts_df: pd.DataFrame | None,
    selected_candidate: Candidate | None,
) -> Path:
    workflow_path = RESULTS_DIR / "nausicaa_workflow.xlsx"
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    run_info_rows = [
        {"Key": "code_version", "Value": get_git_version()},
        {"Key": "timestamp_utc", "Value": timestamp_utc},
    ]
    run_info_rows.extend(
        {"Key": key, "Value": value} for key, value in asdict(config).items()
    )
    run_info_df = pd.DataFrame(run_info_rows)

    candidate_rows = []
    for candidate in candidates:
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "objective": candidate.objective,
                "wing_span_m": candidate.wing_span_m,
                "wing_chord_m": candidate.wing_chord_m,
                "boom_length_m": (
                    candidate.tail_arm_m
                    + BOOM_END_BEFORE_ELEV_FRAC * (candidate.htail_span_m / HT_AR)
                    - NOSE_X_M
                ),
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

    preferred_scenario_columns = [
        "candidate_id",
        "scenario_id",
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "eff_a",
        "eff_e",
        "eff_r",
        "bias_a_deg",
        "bias_e_deg",
        "bias_r_deg",
        "ixx_scale",
        "iyy_scale",
        "izz_scale",
        "wing_E_scale",
        "htail_E_scale",
        "wing_thickness_scale",
        "tail_thickness_scale",
        "w_gust_nom",
        "w_gust_turn",
        "control_eff",
        "drag_factor",
        "both_success",
        "trim_success",
        "nom_success",
        "turn_success",
        "nom_alpha_deg",
        "nom_u_a_deg",
        "nom_u_e_deg",
        "nom_u_r_deg",
        "nom_delta_a_deg",
        "nom_delta_e_deg",
        "nom_delta_r_deg",
        "nom_sink_rate_mps",
        "nom_L_over_D",
        "nom_CL",
        "nom_D",
        "nom_alpha_margin_deg",
        "nom_cl_margin_to_cap",
        "nom_util_a",
        "nom_util_e",
        "nom_util_r",
        "nom_u_rate_util_a",
        "nom_u_rate_util_e",
        "nom_u_rate_util_r",
        "nom_roll_tau_s",
        "nom_roll_accel0",
        "nom_roll_rate_ss",
        "turn_alpha_deg",
        "turn_u_a_deg",
        "turn_u_e_deg",
        "turn_u_r_deg",
        "turn_delta_a_deg",
        "turn_delta_e_deg",
        "turn_delta_r_deg",
        "turn_CL",
        "turn_D",
        "turn_alpha_margin_deg",
        "turn_cl_margin_to_cap",
        "turn_util_a",
        "turn_util_e",
        "turn_util_r",
        "turn_u_rate_util_a",
        "turn_u_rate_util_e",
        "turn_u_rate_util_r",
        "turn_roll_tau_s",
        "turn_roll_accel0",
        "turn_roll_rate_ss",
        "turn_wing_deflection_proxy_over_allow",
        "turn_htail_deflection_proxy_over_allow",
    ]
    extra_scenario_columns = sorted(
        [col for col in robust_scenarios_df.columns if col not in preferred_scenario_columns]
    )
    robust_scenarios_df = robust_scenarios_df.reindex(
        columns=preferred_scenario_columns + extra_scenario_columns
    )

    preferred_summary_columns = [
        "candidate_id",
        "feasible_rate",
        "sink_mean",
        "sink_std",
        "sink_worst",
        "sink_cvar_20",
        "max_turn_util_worst",
        "min_turn_alpha_margin_worst",
        "max_turn_wing_deflection_proxy_over_allow_worst",
        "max_roll_tau_worst",
        "max_delta_e_util_worst",
        "max_alpha_worst",
        "min_alpha_margin_worst",
        "min_cl_margin_worst",
        "selection_score",
        "is_selected",
    ]
    extra_summary_columns = sorted(
        [col for col in robust_summary_df.columns if col not in preferred_summary_columns]
    )
    robust_summary_df = robust_summary_df.reindex(
        columns=preferred_summary_columns + extra_summary_columns
    )
    worst_case_report_df = build_worst_case_report(
        candidates=candidates,
        robust_scenarios_df=robust_scenarios_df,
    )

    if "both_success" in robust_scenarios_df.columns:
        correlation_mask = robust_scenarios_df["both_success"] == True
    elif "trim_success" in robust_scenarios_df.columns:
        correlation_mask = robust_scenarios_df["trim_success"] == True
    else:
        correlation_mask = onp.zeros(len(robust_scenarios_df), dtype=bool)

    correlation_columns = [
        "candidate_id",
        "scenario_id",
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "eff_a",
        "eff_e",
        "eff_r",
        "bias_a_deg",
        "bias_e_deg",
        "bias_r_deg",
        "wing_E_scale",
        "htail_E_scale",
        "w_gust_nom",
        "w_gust_turn",
        "drag_factor",
        "nom_alpha_deg",
        "nom_delta_a_deg",
        "nom_delta_e_deg",
        "nom_delta_r_deg",
        "nom_sink_rate_mps",
        "nom_L_over_D",
        "nom_CL",
        "nom_D",
        "nom_alpha_margin_deg",
        "nom_cl_margin_to_cap",
        "turn_alpha_deg",
        "turn_delta_a_deg",
        "turn_delta_e_deg",
        "turn_delta_r_deg",
        "turn_CL",
        "turn_D",
        "turn_alpha_margin_deg",
        "turn_cl_margin_to_cap",
        "turn_util_a",
        "turn_util_e",
        "turn_util_r",
        "turn_wing_deflection_proxy_over_allow",
        "turn_htail_deflection_proxy_over_allow",
    ]
    correlation_data_df = robust_scenarios_df[
        correlation_mask
    ].reindex(columns=correlation_columns)

    numeric_df = correlation_data_df.select_dtypes(include=["number"]).copy()
    if not numeric_df.empty:
        numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
        correlation_matrix_df = numeric_df.corr(numeric_only=True)
    else:
        correlation_matrix_df = pd.DataFrame()

    with pd.ExcelWriter(workflow_path) as writer:
        run_info_df.to_excel(writer, sheet_name="RunInfo", index=False)
        if all_starts_df is not None:
            preferred_all_starts_columns = [
                "start_index",
                "success",
                "status",
                "objective",
                "failure_reason",
                "kept_after_dedup",
                "kept_rank",
            ]
            extra_all_starts_columns = sorted(
                [col for col in all_starts_df.columns if col not in preferred_all_starts_columns]
            )
            all_starts_df = all_starts_df.reindex(
                columns=preferred_all_starts_columns + extra_all_starts_columns
            )
            all_starts_df.to_excel(writer, sheet_name="AllStarts", index=False)
        candidates_df.to_excel(writer, sheet_name="Candidates", index=False)
        robust_scenarios_df.to_excel(writer, sheet_name="RobustScenarios", index=False)
        robust_summary_df.to_excel(writer, sheet_name="RobustSummary", index=False)
        worst_case_report_df.to_excel(writer, sheet_name="WorstCaseReport", index=False)
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
            selected_active_rows = selected_candidate.active_constraint_rows
            if (
                selected_active_rows is None
                and selected_candidate.constraint_rows is not None
            ):
                selected_active_rows = build_active_constraints_rows(
                    selected_candidate.constraint_rows
                )
            if selected_active_rows is not None:
                selected_active_df = pd.DataFrame(selected_active_rows)
                if selected_active_df.empty and selected_candidate.constraint_rows is not None:
                    selected_active_df = pd.DataFrame(
                        columns=pd.DataFrame(selected_candidate.constraint_rows).columns
                    )
                selected_active_df.to_excel(
                    writer,
                    sheet_name="ActiveConstraints",
                    index=False,
                )
            if selected_candidate.solver_stats is not None:
                pd.DataFrame(
                    flatten_stats(selected_candidate.solver_stats),
                    columns=["Key", "Value"],
                ).to_excel(
                    writer,
                    sheet_name="SolverStats",
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
        active_constraint_rows=candidate.active_constraint_rows,
        solver_stats=candidate.solver_stats,
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

        candidates, all_starts_df = run_multistart(workflow_config)
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
            all_starts_df=all_starts_df,
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
