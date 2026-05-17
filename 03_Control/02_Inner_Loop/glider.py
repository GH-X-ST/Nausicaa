from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from A_model_parameters.mass_properties_estimate import (
    INERTIA_B as ESTIMATED_INERTIA_B,
    MASS_KG as ESTIMATED_MASS_KG,
    R_CG_BUILD_M as ESTIMATED_R_CG_BUILD_M,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Indices and surface codes
# 2) Geometry dataclasses
# 3) Strip-table builders
# 4) Baseline glider factory
# =============================================================================

# =============================================================================
# 1) Indices and Surface Codes
# =============================================================================
# Control input indices
AILERON = 0
ELEVATOR = 1
RUDDER = 2

# Surface codes
WING = 0
HORIZONTAL_TAIL = 1
VERTICAL_TAIL = 2


# =============================================================================
# 2) Geometry Dataclasses
# =============================================================================
# Geometry containers keep body-axis locations and strip assumptions explicit.
@dataclass(frozen=True)
class ControlSurface:
    name: str
    chord_fraction: float
    eta_start: float
    eta_end: float
    input_axis: int
    input_sign: float = 1.0


@dataclass(frozen=True)
class LiftingSurface:
    name: str
    code: int
    root_le_b: np.ndarray
    chord_m: float
    span_m: float
    # Per-panel upward angle from the centreline, not total left-to-right dihedral.
    dihedral_deg: float
    strip_count: int
    symmetric: bool
    vertical: bool
    cd0: float
    alpha0: float
    induced_drag_efficiency: float
    control_surface: ControlSurface | None = None


@dataclass(frozen=True)
class Glider:
    mass_kg: float
    inertia_b: np.ndarray
    s_ref_m2: float
    b_ref_m: float
    c_ref_m: float
    drag_area_fuse_m2: float
    surfaces: tuple[LiftingSurface, ...]
    r_strip_b: np.ndarray
    area_strip_m2: np.ndarray
    chord_strip_m: np.ndarray
    aspect_ratio_strip: np.ndarray
    span_axis_b: np.ndarray
    surface_normal_b: np.ndarray
    control_mix: np.ndarray
    cd0_strip: np.ndarray
    alpha0_strip: np.ndarray
    efficiency_strip: np.ndarray
    flap_scale_strip: np.ndarray
    surface_code: np.ndarray


# =============================================================================
# 3) Strip-Table Builders
# =============================================================================
def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _flap_scale(chord_fraction: float) -> float:
    theta_f = np.arccos(2.0 * chord_fraction - 1.0)
    return 1.0 - (theta_f - np.sin(theta_f)) / np.pi


def _horizontal_strip_rows(
    surface: LiftingSurface,
) -> list[dict[str, np.ndarray | float | int]]:
    # Horizontal surfaces use mirrored left and right strip sets
    dihedral_rad = np.deg2rad(surface.dihedral_deg)
    half_span_m = 0.5 * surface.span_m
    # Projected semispan for eta-based control coverage
    projected_semispan_m = half_span_m * np.cos(dihedral_rad)
    strip_span_m = half_span_m / surface.strip_count
    strip_centers_m = (np.arange(surface.strip_count) + 0.5) * strip_span_m
    rows: list[dict[str, np.ndarray | float | int]] = []
    control = surface.control_surface
    # Symmetric surfaces are mirrored about the centerline
    for side_sign in (1.0, -1.0) if surface.symmetric else (1.0,):
        # Span and surface-normal directions follow body axes
        span_axis_b = _unit(
            np.array([0.0, side_sign * np.cos(dihedral_rad), -np.sin(dihedral_rad)])
        )
        surface_normal_b = _unit(
            np.array(
                [
                    0.0,
                    -side_sign * np.sin(dihedral_rad),
                    -np.cos(dihedral_rad),
                ]
            )
        )
        for s_m in strip_centers_m:
            # Strip centers lie on the local lifting-surface reference line
            y_b = side_sign * s_m * np.cos(dihedral_rad)
            z_b = surface.root_le_b[2] - s_m * np.sin(dihedral_rad)
            eta = abs(y_b) / max(projected_semispan_m, 1e-12)
            control_mix = np.zeros(3)
            flap_scale_strip = 0.0
            if control is not None and control.eta_start <= eta <= control.eta_end:
                # Positive local deflection increases section incidence
                control_sign = control.input_sign
                if control.input_axis == AILERON:
                    control_sign *= -1.0 if side_sign > 0.0 else 1.0
                control_mix[control.input_axis] = control_sign
                flap_scale_strip = _flap_scale(control.chord_fraction)
            rows.append(
                {
                    "r_strip_b": np.array(
                        [
                            surface.root_le_b[0] - 0.25 * surface.chord_m,
                            y_b,
                            z_b,
                        ]
                    ),
                    "area_strip_m2": surface.chord_m * strip_span_m,
                    "chord_strip_m": surface.chord_m,
                    "aspect_ratio_strip": surface.span_m / surface.chord_m,
                    "span_axis_b": span_axis_b,
                    "surface_normal_b": surface_normal_b,
                    "control_mix": control_mix,
                    "cd0_strip": surface.cd0,
                    "alpha0_strip": surface.alpha0,
                    "efficiency_strip": surface.induced_drag_efficiency,
                    "flap_scale_strip": flap_scale_strip,
                    "surface_code": surface.code,
                }
            )
    return rows


def _vertical_strip_rows(
    surface: LiftingSurface,
) -> list[dict[str, np.ndarray | float | int]]:
    # Vertical-tail strips are stacked along body z
    strip_span_m = surface.span_m / surface.strip_count
    strip_centers_m = (np.arange(surface.strip_count) + 0.5) * strip_span_m
    # Vertical-tail span points upward in the body z-down frame
    span_axis_b = np.array([0.0, 0.0, -1.0])
    surface_normal_b = np.array([0.0, -1.0, 0.0])
    rows: list[dict[str, np.ndarray | float | int]] = []
    control = surface.control_surface
    for s_m in strip_centers_m:
        control_mix = np.zeros(3)
        flap_scale_strip = 0.0
        if control is not None:
            control_mix[control.input_axis] = 1.0
            flap_scale_strip = _flap_scale(control.chord_fraction)
        rows.append(
            {
                "r_strip_b": np.array(
                    [
                        surface.root_le_b[0] - 0.25 * surface.chord_m,
                        0.0,
                        surface.root_le_b[2] - s_m,
                    ]
                ),
                "area_strip_m2": surface.chord_m * strip_span_m,
                "chord_strip_m": surface.chord_m,
                "aspect_ratio_strip": 2.0 * surface.span_m / surface.chord_m,
                "span_axis_b": span_axis_b,
                "surface_normal_b": surface_normal_b,
                "control_mix": control_mix,
                "cd0_strip": surface.cd0,
                "alpha0_strip": surface.alpha0,
                "efficiency_strip": surface.induced_drag_efficiency,
                "flap_scale_strip": flap_scale_strip,
                "surface_code": surface.code,
            }
        )
    return rows


def _flatten_strip_table(
    surfaces: tuple[LiftingSurface, ...],
) -> dict[str, np.ndarray]:
    # Fixed-shape strip table for runtime evaluation
    rows: list[dict[str, np.ndarray | float | int]] = []
    for surface in surfaces:
        if surface.vertical:
            rows.extend(_vertical_strip_rows(surface))
        else:
            rows.extend(_horizontal_strip_rows(surface))
    return {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in (
            "r_strip_b",
            "area_strip_m2",
            "chord_strip_m",
            "aspect_ratio_strip",
            "span_axis_b",
            "surface_normal_b",
            "control_mix",
            "cd0_strip",
            "alpha0_strip",
            "efficiency_strip",
            "flap_scale_strip",
            "surface_code",
        )
    }


# =============================================================================
# 4) Baseline Glider Factory
# =============================================================================
def build_nausicaa_glider() -> Glider:
    # Manufactured measurements use the wing leading edge as the x=0 datum.
    mass_kg = float(ESTIMATED_MASS_KG)
    x_cg_from_wing_le_m = float(ESTIMATED_R_CG_BUILD_M[0])
    z_cg_from_fuselage_rod_m = float(ESTIMATED_R_CG_BUILD_M[2])
    wing_ac_from_wing_le_m = 0.0413
    htail_ac_from_wing_le_m = 0.426
    vtail_ac_from_wing_le_m = 0.433
    inertia_b = ESTIMATED_INERTIA_B.copy()
    wing_span_m = 0.764
    wing_chord_m = 0.165
    htail_span_m = 0.364
    htail_chord_m = 0.091
    vtail_span_m = 0.119
    vtail_chord_m = 0.059
    # Measured dihedral is the full left-to-right included angle. Strip
    # geometry below uses the per-panel upward angle from the centreline.
    total_dihedral_deg = 9.28
    dihedral_deg = 0.5 * total_dihedral_deg
    wing_thickness_m = 0.0060
    htail_thickness_m = 0.0030
    wing_bottom_z_from_fuselage_rod_m = -0.0085
    htail_top_z_from_fuselage_rod_m = 0.0045
    vtail_bottom_z_from_fuselage_rod_m = -0.0050
    # Root leading-edge locations are CG-centered in body axes
    wing_ac_rounding_error_m = wing_ac_from_wing_le_m - 0.25 * wing_chord_m
    if abs(wing_ac_rounding_error_m) > 1e-4:
        raise ValueError("measured wing AC is inconsistent with the wing-LE x datum")
    wing_root_le_from_datum_m = 0.0
    htail_root_le_from_datum_m = htail_ac_from_wing_le_m - 0.25 * htail_chord_m
    vtail_root_le_from_datum_m = vtail_ac_from_wing_le_m - 0.25 * vtail_chord_m
    wing_root_le_z_from_fuselage_rod_m = (
        wing_bottom_z_from_fuselage_rod_m - 0.5 * wing_thickness_m
    )
    htail_root_le_z_from_fuselage_rod_m = (
        htail_top_z_from_fuselage_rod_m + 0.5 * htail_thickness_m
    )
    vtail_root_le_z_from_fuselage_rod_m = vtail_bottom_z_from_fuselage_rod_m
    # The measured wing AC is kept explicit for audit; its rounded value implies
    # an LE offset of 0.00005 m from the declared datum.
    wing_root_le_b = np.array(
        [
            x_cg_from_wing_le_m - wing_root_le_from_datum_m,
            0.0,
            wing_root_le_z_from_fuselage_rod_m - z_cg_from_fuselage_rod_m,
        ]
    )
    htail_root_le_b = np.array(
        [
            x_cg_from_wing_le_m - htail_root_le_from_datum_m,
            0.0,
            htail_root_le_z_from_fuselage_rod_m - z_cg_from_fuselage_rod_m,
        ]
    )
    vtail_root_le_b = np.array(
        [
            x_cg_from_wing_le_m - vtail_root_le_from_datum_m,
            0.0,
            vtail_root_le_z_from_fuselage_rod_m - z_cg_from_fuselage_rod_m,
        ]
    )
    wing = LiftingSurface(
        name="wing",
        code=WING,
        root_le_b=wing_root_le_b,
        chord_m=wing_chord_m,
        span_m=wing_span_m,
        dihedral_deg=dihedral_deg,
        strip_count=6,
        symmetric=True,
        vertical=False,
        cd0=0.018,
        alpha0=0.001,
        induced_drag_efficiency=0.82,
        control_surface=ControlSurface(
            name="aileron",
            chord_fraction=0.30,
            eta_start=0.30,
            eta_end=0.85,
            input_axis=AILERON,
            input_sign=1.0,
        ),
    )
    htail = LiftingSurface(
        name="horizontal_tail",
        code=HORIZONTAL_TAIL,
        root_le_b=htail_root_le_b,
        chord_m=htail_chord_m,
        span_m=htail_span_m,
        dihedral_deg=0.0,
        strip_count=4,
        symmetric=True,
        vertical=False,
        cd0=0.020,
        alpha0=0.0,
        induced_drag_efficiency=0.78,
        control_surface=ControlSurface(
            name="elevator",
            chord_fraction=0.30,
            eta_start=0.0,
            eta_end=1.0,
            input_axis=ELEVATOR,
            input_sign=-1.0,
        ),
    )
    vtail = LiftingSurface(
        name="vertical_tail",
        code=VERTICAL_TAIL,
        root_le_b=vtail_root_le_b,
        chord_m=vtail_chord_m,
        span_m=vtail_span_m,
        dihedral_deg=0.0,
        strip_count=4,
        symmetric=False,
        vertical=True,
        cd0=0.020,
        alpha0=0.0,
        induced_drag_efficiency=0.75,
        control_surface=ControlSurface(
            name="rudder",
            chord_fraction=0.35,
            eta_start=0.0,
            eta_end=1.0,
            input_axis=RUDDER,
            input_sign=1.0,
        ),
    )
    surfaces = (wing, htail, vtail)
    strip_table = _flatten_strip_table(surfaces)
    return Glider(
        mass_kg=mass_kg,
        inertia_b=inertia_b,
        s_ref_m2=wing_span_m * wing_chord_m,
        b_ref_m=wing_span_m,
        c_ref_m=wing_chord_m,
        drag_area_fuse_m2=4.89753346075483e-05,
        surfaces=surfaces,
        r_strip_b=strip_table["r_strip_b"],
        area_strip_m2=strip_table["area_strip_m2"],
        chord_strip_m=strip_table["chord_strip_m"],
        aspect_ratio_strip=strip_table["aspect_ratio_strip"],
        span_axis_b=strip_table["span_axis_b"],
        surface_normal_b=strip_table["surface_normal_b"],
        control_mix=strip_table["control_mix"],
        cd0_strip=strip_table["cd0_strip"],
        alpha0_strip=strip_table["alpha0_strip"],
        efficiency_strip=strip_table["efficiency_strip"],
        flap_scale_strip=strip_table["flap_scale_strip"],
        surface_code=strip_table["surface_code"].astype(int),
    )
