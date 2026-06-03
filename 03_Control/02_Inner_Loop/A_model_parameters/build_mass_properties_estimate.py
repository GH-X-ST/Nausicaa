from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


INNER_LOOP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INNER_LOOP_DIR))

from glider import build_nausicaa_glider


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Shared frame constants and dataclasses
# 2) Inertia helpers
# 3) Surface and component mass elements
# 4) Manufactured estimate case
# 5) Output writer
# =============================================================================


# =============================================================================
# 1) Shared Frame Constants and Dataclasses
# =============================================================================
# Build-frame convention for component audit:
# x: aft from wing leading edge; y: starboard; z: down from fuselage rod centre.
# Simulation body axes use x forward, y starboard, z down, so only x flips.
BUILD_TO_BODY = np.diag([-1.0, 1.0, 1.0])
AS_BUILT_MASS_KG = 0.13356
AS_BUILT_WING_SPAN_M = 0.764
AS_BUILT_WING_CHORD_M = 0.165
AS_BUILT_TOTAL_DIHEDRAL_DEG = 9.28
# Code geometry uses the upward angle of each wing panel from the centreline.
AS_BUILT_PANEL_DIHEDRAL_DEG = 0.5 * AS_BUILT_TOTAL_DIHEDRAL_DEG
AS_BUILT_FUSELAGE_LENGTH_M = 0.568
AS_BUILT_HTAIL_SPAN_M = 0.364
AS_BUILT_HTAIL_CHORD_M = 0.091
AS_BUILT_VTAIL_HEIGHT_M = 0.119
AS_BUILT_VTAIL_CHORD_M = 0.059
AS_BUILT_WING_AC_M = 0.0413
AS_BUILT_HTAIL_AC_M = 0.426
AS_BUILT_VTAIL_AC_M = 0.433
AS_BUILT_X_CG_FROM_WING_LE_M = 0.126
AS_BUILT_Y_CG_M = 0.0
AS_BUILT_WING_BOTTOM_CENTER_Z_M = -0.0085
AS_BUILT_HTAIL_TOP_Z_M = 0.0045
AS_BUILT_VTAIL_BOTTOM_Z_M = -0.0050
AS_BUILT_FUSELAGE_NOSE_X_FROM_WING_LE_M = -0.100

WING_THICKNESS_M = 0.0060
HTAIL_THICKNESS_M = 0.0030
VTAIL_THICKNESS_M = 0.0030

# These constants mirror the detailed mass build-up in 02_Glider_Design/nausicaa.py,
# while positions are converted into the manufactured build frame above.
DESIGN_WING_BOTTOM_Z_UP_M = 0.004
DESIGN_HTAIL_BOTTOM_Z_UP_M = -0.004
DESIGN_VTAIL_BOTTOM_Z_UP_M = 0.004
DESIGN_HTAIL_TOP_Z_UP_M = DESIGN_HTAIL_BOTTOM_Z_UP_M + HTAIL_THICKNESS_M

WING_DENSITY_KG_M3 = 33.0
TAIL_DENSITY_KG_M3 = 40.0
BOOM_TUBE_OUTER_DIAMETER_M = 0.004
BOOM_TUBE_INNER_DIAMETER_M = 0.002
# Bench-measured line masses are stored as kg/m; the source measurements were g/cm.
GRAM_PER_CM_TO_KG_PER_M = 0.1
FUSELAGE_ROD_LINEAR_MASS_KG_PER_M = 0.145 * GRAM_PER_CM_TO_KG_PER_M

WING_SPAR_X_FROM_WING_LE_M = 0.043
WING_SPAR_OD_M = 0.004
WING_SPAR_ID_M = 0.003
WING_SPAR_LINEAR_MASS_KG_PER_M = 0.0828 * GRAM_PER_CM_TO_KG_PER_M
# Spar adhesive is kept local to the spar line: total spar-line mass is
# 1.3 times the measured bare spar mass until a separate glue measurement exists.
WING_SPAR_GLUE_MASS_FACTOR = 1.30
WING_SPAR_Z_FROM_LOWER_M = 0.002
WING_SPAR_SLOT_W_M = 0.004
WING_SPAR_SLOT_H_M = 0.004
SLOT_VOID_MASS_FRAC_MAX = 0.05

TAPE_THICKNESS_M = 0.00013
TAPE_AREAL_DENSITY_KG_M2 = 0.175
WING_TAPE_CHORDWISE_WIDTH_M = 0.015
WING_TAPE_X_FROM_WING_LE_M = 0.0425
WING_TAPE_BOTTOM_SPAN_START_M = 0.108
WING_TAPE_TOP_SPAN_START_M = 0.250
TAIL_TAPE_CHORDWISE_WIDTH_M = 0.020
HTAIL_TAPE_X_FROM_HTAIL_LE_M = 0.051
VTAIL_TAPE_X_FROM_VTAIL_LE_M = 0.038
VTAIL_TAPE_LENGTH_PER_SIDE_M = 0.090
VTAIL_TAPE_Y_OFFSET_M = 0.5 * VTAIL_THICKNESS_M + 0.5 * TAPE_THICKNESS_M

CENTRE_MODULE_MASS_KG = 0.028
TAIL_MOUNT_AS_BUILT_MASS_KG = 0.003
TAIL_SUPPORT_MASS_KG = 0.001
CENTRE_MODULE_VOLUME_M3 = 0.00002349
TAIL_MODULE_VOLUME_M3 = 0.00000361
CENTRE_MODULE_RHO_KG_M3 = CENTRE_MODULE_MASS_KG / CENTRE_MODULE_VOLUME_M3
TAIL_MODULE_RHO_KG_M3 = 0.004 / TAIL_MODULE_VOLUME_M3

CENTRE_CORE_X_OFFSET_FROM_0P3C_M = 0.0243
CENTRE_CORE_Z_CG_UP_M = -0.00745
TAIL_CORE_X_OFFSET_FROM_BOOM_END_M = 0.0231
TAIL_CORE_Z_CG_UP_M = -0.0005
TAIL_GEAR_X_OFFSET_FROM_HTAIL_LE_M = -0.00894
TAIL_SUPPORT_Z_CG_UP_M = -0.00587

WING_MOUNT_T_M = 0.002
WING_MOUNT_LB_M = 0.0261
WING_MOUNT_LT_M = 0.0115
WING_MOUNT_H_M = 0.086
WING_MOUNT_X0_FWD_OFFSET_FROM_SPAR_M = -0.01305
WING_MOUNT_X0_AFT_OFFSET_FROM_SPAR_M = 0.0589
WING_MOUNT_ROOT_BOTTOM_Z_UP_M = 0.002
WING_SPAR_JOINER_W_M = 0.006
WING_SPAR_JOINER_H_M = 0.005
WING_SPAR_JOINER_SPAN_M = 0.02868
WING_SPAR_JOINER_HOLE_D_M = 0.004

HTAIL_MOUNT_T_M = 0.0015
HTAIL_MOUNT_LB_M = 0.019
HTAIL_MOUNT_LT_M = 0.008
HTAIL_MOUNT_H_M = 0.034
HTAIL_MOUNT_X0_OFFSET_FROM_BOOM_END_M = 0.0277
HTAIL_MOUNT_Z_UPPER_UP_M = -0.0025

VTAIL_MOUNT_T_M = 0.002
VTAIL_MOUNT_LB_M = 0.019
VTAIL_MOUNT_LT_M = 0.0095
VTAIL_MOUNT_H_M = 0.0215
VTAIL_MOUNT_X0_OFFSET_FROM_BOOM_END_M = 0.0383

RECEIVER_MASS_KG = 0.0089
RECEIVER_SIZE_X_M = 0.040
RECEIVER_SIZE_Y_M = 0.015
RECEIVER_SIZE_Z_M = 0.0097
RECEIVER_FRONT_X_FROM_WING_LE_M = 0.101
RECEIVER_TOP_Z_M = 0.008
TAIL_PUSHROD_MASS_KG = 0.002
TAIL_PUSHROD_X_FROM_WING_LE_M = AS_BUILT_X_CG_FROM_WING_LE_M
TAIL_PUSHROD_Y_OFFSET_M = 0.027
TAIL_PUSHROD_Z_M = 0.0
VICON_MARKER_MASS_KG = 0.0016
VICON_MARKER_RADIUS_M = 0.0075
VICON_WING_LE_MARKER_Z_M = -0.012
VICON_WINGTIP_MARKER_X_FROM_WING_LE_M = 0.045

CALIBRATION_RESIDUAL_TARGET_MASS_KG = AS_BUILT_MASS_KG
CALIBRATION_RESIDUAL_TARGET_X_CG_M = AS_BUILT_X_CG_FROM_WING_LE_M


@dataclass(frozen=True)
class MassElement:
    name: str
    mass_kg: float
    r_cg_m: np.ndarray
    inertia_m: np.ndarray


@dataclass(frozen=True)
class MassEstimate:
    mass_kg: float
    r_cg_m: np.ndarray
    r_cg_b: np.ndarray
    inertia_b: np.ndarray


# =============================================================================
# 2) Inertia Helpers
# =============================================================================
def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _frame_from_axes(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
) -> np.ndarray:
    # Local element frames are rebuilt from measured build-frame directions.
    x_dir_m = _unit(np.asarray(x_axis_m, dtype=float).reshape(3))
    y_dir_m = _unit(np.asarray(y_axis_m, dtype=float).reshape(3))
    z_dir_m = _unit(np.cross(x_dir_m, y_dir_m))
    y_dir_m = _unit(np.cross(z_dir_m, x_dir_m))
    return np.column_stack((x_dir_m, y_dir_m, z_dir_m))


def assert_as_built_glider_parameters() -> None:
    glider = build_nausicaa_glider()
    wing, htail, vtail = glider.surfaces
    # Mass and CG are generated by this builder. They are not asserted through
    # glider.py here, otherwise a changed measured target would be blocked by
    # the previously generated constants before this script can refresh them.
    checks = {
        "wing_span_m": (wing.span_m, AS_BUILT_WING_SPAN_M),
        "wing_chord_m": (wing.chord_m, AS_BUILT_WING_CHORD_M),
        "wing_panel_dihedral_deg": (wing.dihedral_deg, AS_BUILT_PANEL_DIHEDRAL_DEG),
        "wing_total_dihedral_deg": (
            2.0 * wing.dihedral_deg,
            AS_BUILT_TOTAL_DIHEDRAL_DEG,
        ),
        "htail_span_m": (htail.span_m, AS_BUILT_HTAIL_SPAN_M),
        "htail_chord_m": (htail.chord_m, AS_BUILT_HTAIL_CHORD_M),
        "vtail_height_m": (vtail.span_m, AS_BUILT_VTAIL_HEIGHT_M),
        "vtail_chord_m": (vtail.chord_m, AS_BUILT_VTAIL_CHORD_M),
        "wing_ac_m": (0.25 * wing.chord_m, AS_BUILT_WING_AC_M),
        "htail_ac_m": (
            wing.root_le_b[0] - htail.root_le_b[0] + 0.25 * htail.chord_m,
            AS_BUILT_HTAIL_AC_M,
        ),
        "vtail_ac_m": (
            wing.root_le_b[0] - vtail.root_le_b[0] + 0.25 * vtail.chord_m,
            AS_BUILT_VTAIL_AC_M,
        ),
    }
    for name, (actual, expected) in checks.items():
        if not np.isclose(actual, expected, atol=1.0e-4):
            raise ValueError(
                f"{name}={actual:.9g} does not match as-built {expected:.9g}"
            )


def point_mass(
    name: str,
    mass_kg: float,
    r_cg_m: np.ndarray,
) -> MassElement:
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=np.zeros((3, 3)),
    )


def solid_sphere(
    name: str,
    mass_kg: float,
    radius_m: float,
    r_cg_m: np.ndarray,
) -> MassElement:
    # Vicon marker balls are small but explicit, so their own spherical inertia
    # is kept rather than collapsing them to point masses.
    inertia_scalar = 0.4 * float(mass_kg) * float(radius_m) ** 2
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=inertia_scalar * np.eye(3),
    )


def rectangular_prism(
    name: str,
    mass_kg: float,
    size_x_m: float,
    size_y_m: float,
    size_z_m: float,
    r_cg_m: np.ndarray,
    c_ml: np.ndarray,
) -> MassElement:
    # Prism inertia is computed in its local frame, then rotated to build axes.
    i_local = (float(mass_kg) / 12.0) * np.diag(
        [
            size_y_m**2 + size_z_m**2,
            size_x_m**2 + size_z_m**2,
            size_x_m**2 + size_y_m**2,
        ]
    )
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=c_ml @ i_local @ c_ml.T,
    )


def tube_area_m2(
    outer_diameter_m: float,
    inner_diameter_m: float,
) -> float:
    return 0.25 * np.pi * (outer_diameter_m**2 - inner_diameter_m**2)


def tube_density_from_linear_mass_kg_m3(
    linear_mass_kg_per_m: float,
    outer_diameter_m: float,
    inner_diameter_m: float,
) -> float:
    return linear_mass_kg_per_m / tube_area_m2(outer_diameter_m, inner_diameter_m)


def x_axis_hollow_tube(
    name: str,
    length_m: float,
    outer_diameter_m: float,
    inner_diameter_m: float,
    density_kg_m3: float,
    r_cg_m: np.ndarray,
) -> MassElement:
    area_m2 = tube_area_m2(outer_diameter_m, inner_diameter_m)
    mass_kg = density_kg_m3 * area_m2 * length_m
    r_outer_m = 0.5 * outer_diameter_m
    r_inner_m = 0.5 * inner_diameter_m
    r2_m2 = r_outer_m**2 + r_inner_m**2
    inertia_m = np.diag(
        [
            0.5 * mass_kg * r2_m2,
            (mass_kg / 12.0) * (length_m**2 + 3.0 * r2_m2),
            (mass_kg / 12.0) * (length_m**2 + 3.0 * r2_m2),
        ]
    )
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=inertia_m,
    )


def y_axis_hollow_tube(
    name: str,
    length_m: float,
    outer_diameter_m: float,
    inner_diameter_m: float,
    density_kg_m3: float,
    r_cg_m: np.ndarray,
) -> MassElement:
    area_m2 = tube_area_m2(outer_diameter_m, inner_diameter_m)
    mass_kg = density_kg_m3 * area_m2 * length_m
    r_outer_m = 0.5 * outer_diameter_m
    r_inner_m = 0.5 * inner_diameter_m
    r2_m2 = r_outer_m**2 + r_inner_m**2
    inertia_m = np.diag(
        [
            (mass_kg / 12.0) * (length_m**2 + 3.0 * r2_m2),
            0.5 * mass_kg * r2_m2,
            (mass_kg / 12.0) * (length_m**2 + 3.0 * r2_m2),
        ]
    )
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=inertia_m,
    )


def combine_named_elements(
    name: str,
    elements: list[MassElement],
) -> MassElement:
    mass_kg = sum(element.mass_kg for element in elements)
    r_cg_m = sum(element.mass_kg * element.r_cg_m for element in elements) / mass_kg
    inertia_m = np.zeros((3, 3))
    for element in elements:
        dr_m = element.r_cg_m - r_cg_m
        inertia_m += element.inertia_m + element.mass_kg * (
            np.dot(dr_m, dr_m) * np.eye(3) - np.outer(dr_m, dr_m)
        )
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=r_cg_m,
        inertia_m=inertia_m,
    )


def scale_mass_element(
    name: str,
    element: MassElement,
    scale: float,
) -> MassElement:
    return MassElement(
        name=name,
        mass_kg=scale * element.mass_kg,
        r_cg_m=element.r_cg_m.copy(),
        inertia_m=scale * element.inertia_m,
    )


def trapezoid_area_m2(
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
) -> float:
    return 0.5 * (base_bottom_m + base_top_m) * height_m


def trapezoid_centroid_from_base_m(
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
) -> float:
    return height_m * (base_bottom_m + 2.0 * base_top_m) / (
        3.0 * (base_bottom_m + base_top_m)
    )


def polygon_area_centroid_moments_2d(
    vertices_uv: list[tuple[float, float]],
) -> tuple[float, float, float, float, float, float]:
    area_twice = 0.0
    c_u_num = 0.0
    c_v_num = 0.0
    i_uu_origin = 0.0
    i_vv_origin = 0.0
    i_uv_origin = 0.0
    for index, (u0, v0) in enumerate(vertices_uv):
        u1, v1 = vertices_uv[(index + 1) % len(vertices_uv)]
        cross = u0 * v1 - u1 * v0
        area_twice += cross
        c_u_num += (u0 + u1) * cross
        c_v_num += (v0 + v1) * cross
        i_uu_origin += (v0**2 + v0 * v1 + v1**2) * cross
        i_vv_origin += (u0**2 + u0 * u1 + u1**2) * cross
        i_uv_origin += (2.0 * u0 * v0 + u0 * v1 + u1 * v0 + 2.0 * u1 * v1) * cross

    area_m2 = 0.5 * area_twice
    if abs(area_m2) < 1.0e-12:
        raise ValueError("Degenerate trapezoid area in component inertia.")

    c_u_m = c_u_num / (6.0 * area_m2)
    c_v_m = c_v_num / (6.0 * area_m2)
    i_uu_origin /= 12.0
    i_vv_origin /= 12.0
    i_uv_origin /= 24.0

    if area_m2 < 0.0:
        area_m2 = -area_m2
        i_uu_origin = -i_uu_origin
        i_vv_origin = -i_vv_origin
        i_uv_origin = -i_uv_origin

    i_uu_centroid = i_uu_origin - area_m2 * c_v_m**2
    i_vv_centroid = i_vv_origin - area_m2 * c_u_m**2
    i_uv_centroid = i_uv_origin - area_m2 * c_u_m * c_v_m
    return area_m2, c_u_m, c_v_m, i_uu_centroid, i_vv_centroid, i_uv_centroid


def right_trapezoid_vertices_uv(
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
    straight_side: str,
) -> list[tuple[float, float]]:
    if straight_side == "aft":
        return [
            (0.0, 0.0),
            (base_bottom_m, 0.0),
            (base_bottom_m, height_m),
            (base_bottom_m - base_top_m, height_m),
        ]
    return [
        (0.0, 0.0),
        (base_bottom_m, 0.0),
        (base_top_m, height_m),
        (0.0, height_m),
    ]


def right_trapezoid_centroid_u_from_bottom_fore_m(
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
    straight_side: str,
) -> float:
    vertices_uv = right_trapezoid_vertices_uv(
        base_bottom_m=base_bottom_m,
        base_top_m=base_top_m,
        height_m=height_m,
        straight_side=straight_side,
    )
    _area, c_u_m, _c_v_m, _i_uu, _i_vv, _i_uv = polygon_area_centroid_moments_2d(
        vertices_uv
    )
    return c_u_m


def isosceles_trapezoid_prism(
    name: str,
    mass_kg: float,
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
    thickness_m: float,
    r_cg_m: np.ndarray,
    height_axis: str,
) -> MassElement:
    vertices_uv = [
        (-0.5 * base_bottom_m, 0.0),
        (0.5 * base_bottom_m, 0.0),
        (0.5 * base_top_m, height_m),
        (-0.5 * base_top_m, height_m),
    ]
    area_m2, _c_u, _c_v, i_uu_area, i_vv_area, i_uv_area = (
        polygon_area_centroid_moments_2d(vertices_uv)
    )
    i_u = mass_kg * (i_uu_area / area_m2 + thickness_m**2 / 12.0)
    i_v = mass_kg * (i_vv_area / area_m2 + thickness_m**2 / 12.0)
    i_w = mass_kg * ((i_uu_area + i_vv_area) / area_m2)
    i_uv = -mass_kg * (i_uv_area / area_m2)

    if height_axis == "y":
        inertia_m = np.array(
            [[i_u, i_uv, 0.0], [i_uv, i_v, 0.0], [0.0, 0.0, i_w]]
        )
    else:
        inertia_m = np.array(
            [[i_u, 0.0, i_uv], [0.0, i_w, 0.0], [i_uv, 0.0, i_v]]
        )
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=inertia_m,
    )


def right_trapezoid_prism(
    name: str,
    mass_kg: float,
    base_bottom_m: float,
    base_top_m: float,
    height_m: float,
    thickness_m: float,
    r_cg_m: np.ndarray,
    height_axis: str,
    straight_side: str,
) -> MassElement:
    vertices_uv = right_trapezoid_vertices_uv(
        base_bottom_m=base_bottom_m,
        base_top_m=base_top_m,
        height_m=height_m,
        straight_side=straight_side,
    )
    area_m2, _c_u, _c_v, i_uu_area, i_vv_area, i_uv_area = (
        polygon_area_centroid_moments_2d(vertices_uv)
    )
    i_u = mass_kg * (i_uu_area / area_m2 + thickness_m**2 / 12.0)
    i_v = mass_kg * (i_vv_area / area_m2 + thickness_m**2 / 12.0)
    i_w = mass_kg * ((i_uu_area + i_vv_area) / area_m2)
    i_uv = -mass_kg * (i_uv_area / area_m2)

    if height_axis == "y":
        inertia_m = np.array(
            [[i_u, i_uv, 0.0], [i_uv, i_v, 0.0], [0.0, 0.0, i_w]]
        )
    else:
        inertia_m = np.array(
            [[i_u, 0.0, i_uv], [0.0, i_w, 0.0], [i_uv, 0.0, i_v]]
        )
    return MassElement(
        name=name,
        mass_kg=float(mass_kg),
        r_cg_m=np.asarray(r_cg_m, dtype=float).reshape(3),
        inertia_m=inertia_m,
    )


def combine_mass_elements(
    elements: list[MassElement],
) -> MassEstimate:
    mass_kg = sum(element.mass_kg for element in elements)
    r_cg_m = sum(element.mass_kg * element.r_cg_m for element in elements) / mass_kg
    inertia_m = np.zeros((3, 3))
    for element in elements:
        dr_m = element.r_cg_m - r_cg_m
        # Parallel-axis shift reports inertia about the assembled vehicle CG.
        inertia_m += element.inertia_m + element.mass_kg * (
            np.dot(dr_m, dr_m) * np.eye(3) - np.outer(dr_m, dr_m)
        )
    return MassEstimate(
        mass_kg=mass_kg,
        r_cg_m=r_cg_m,
        r_cg_b=BUILD_TO_BODY @ r_cg_m,
        inertia_b=BUILD_TO_BODY @ inertia_m @ BUILD_TO_BODY.T,
    )


# =============================================================================
# 3) Surface and Component Mass Elements
# =============================================================================
def horizontal_surface_elements(
    name_prefix: str,
    root_le_m: np.ndarray,
    chord_m: float,
    span_m: float,
    dihedral_deg: float,
    mass_kg: float,
    thickness_m: float,
) -> list[MassElement]:
    # Horizontal surfaces are split left/right so dihedral mass offsets are explicit.
    half_span_m = 0.5 * span_m
    quarter_span_m = 0.25 * span_m
    dihedral_rad = np.deg2rad(dihedral_deg)
    half_mass_kg = 0.5 * mass_kg
    root_le_m = np.asarray(root_le_m, dtype=float).reshape(3)
    elements: list[MassElement] = []
    for side_name, side_sign in (("R", 1.0), ("L", -1.0)):
        span_axis_m = np.array(
            [0.0, side_sign * np.cos(dihedral_rad), -np.sin(dihedral_rad)]
        )
        c_ml = _frame_from_axes(np.array([1.0, 0.0, 0.0]), span_axis_m)
        r_cg_m = np.array(
            [
                root_le_m[0] + 0.5 * chord_m,
                side_sign * quarter_span_m * np.cos(dihedral_rad),
                root_le_m[2] - quarter_span_m * np.sin(dihedral_rad),
            ]
        )
        elements.append(
            rectangular_prism(
                name=f"{name_prefix}_{side_name}_half",
                mass_kg=half_mass_kg,
                size_x_m=chord_m,
                size_y_m=half_span_m,
                size_z_m=thickness_m,
                r_cg_m=r_cg_m,
                c_ml=c_ml,
            )
        )
    return elements


def vertical_surface_element(
    root_le_m: np.ndarray,
    chord_m: float,
    height_m: float,
    mass_kg: float,
    thickness_m: float,
) -> MassElement:
    # The vertical tail height points upward, i.e. negative build-frame z.
    c_ml = _frame_from_axes(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    )
    root_le_m = np.asarray(root_le_m, dtype=float).reshape(3)
    r_cg_m = np.array(
        [
            root_le_m[0] + 0.5 * chord_m,
            0.0,
            root_le_m[2] - 0.5 * height_m,
        ]
    )
    return rectangular_prism(
        name="vertical_tail",
        mass_kg=mass_kg,
        size_x_m=chord_m,
        size_y_m=height_m,
        size_z_m=thickness_m,
        r_cg_m=r_cg_m,
        c_ml=c_ml,
    )


def battery_element(
    wing_bottom_center_z_m: float,
) -> MassElement:
    # Manufactured battery measurement, 2026-05-16. The front face is at the
    # wing LE datum and the battery is centred laterally.
    mass_kg = 0.00938
    length_x_m = 0.048
    width_y_m = 0.016
    height_z_m = 0.0065
    bottom_offset_below_wing_bottom_m = 0.028
    r_cg_m = np.array(
        [
            0.5 * length_x_m,
            0.0,
            (
                wing_bottom_center_z_m
                + bottom_offset_below_wing_bottom_m
                - 0.5 * height_z_m
            ),
        ]
    )
    return rectangular_prism(
        name="battery",
        mass_kg=mass_kg,
        size_x_m=length_x_m,
        size_y_m=width_y_m,
        size_z_m=height_z_m,
        r_cg_m=r_cg_m,
        c_ml=np.eye(3),
    )


def aileron_servo_elements(
    wing_bottom_center_z_m: float,
    wing_dihedral_deg: float,
) -> list[MassElement]:
    # Wing servos are point masses. The span station follows the dihedral wing,
    # so the local wing bottom rises with span.
    mass_kg = 0.00400
    span_station_m = 0.200
    chord_station_m = 0.057
    centre_below_wing_bottom_m = 0.005
    dihedral_rad = np.deg2rad(wing_dihedral_deg)
    elements: list[MassElement] = []
    for name, side_sign in (
        ("aileron_L_servo", -1.0),
        ("aileron_R_servo", 1.0),
    ):
        local_wing_bottom_z_m = (
            wing_bottom_center_z_m - span_station_m * np.sin(dihedral_rad)
        )
        r_cg_m = np.array(
            [
                chord_station_m,
                side_sign * span_station_m * np.cos(dihedral_rad),
                local_wing_bottom_z_m + centre_below_wing_bottom_m,
            ]
        )
        elements.append(point_mass(name, mass_kg, r_cg_m))
    return elements


def tail_servo_elements(
    wing_bottom_center_z_m: float,
) -> list[MassElement]:
    # Rudder is mounted on the same side as the right aileron servo. Elevator is
    # mounted symmetrically on the opposite side.
    mass_kg = 0.00400
    chord_station_m = 0.120
    lateral_station_m = 0.015
    centre_below_wing_bottom_m = 0.005
    z_m = wing_bottom_center_z_m + centre_below_wing_bottom_m
    return [
        point_mass(
            name,
            mass_kg,
            np.array([chord_station_m, side_sign * lateral_station_m, z_m]),
        )
        for name, side_sign in (
            ("rudder_servo", 1.0),
            ("elevator_servo", -1.0),
        )
    ]


def tail_pushrod_elements() -> list[MassElement]:
    # Rudder/elevator pushrods are unresolved slender parts at this stage, so
    # each is represented as a point mass at the measured x_CG station.
    return [
        point_mass(
            name,
            TAIL_PUSHROD_MASS_KG,
            np.array(
                [
                    TAIL_PUSHROD_X_FROM_WING_LE_M,
                    side_sign * TAIL_PUSHROD_Y_OFFSET_M,
                    TAIL_PUSHROD_Z_M,
                ]
            ),
        )
        for name, side_sign in (
            ("rudder_pushrod", 1.0),
            ("elevator_pushrod", -1.0),
        )
    ]


def wing_design_z_up_to_build_z_down(z_up_m: float) -> float:
    # Wing-mounted CAD offsets are converted relative to the measured wing bottom.
    return AS_BUILT_WING_BOTTOM_CENTER_Z_M - (
        float(z_up_m) - DESIGN_WING_BOTTOM_Z_UP_M
    )


def htail_design_z_up_to_build_z_down(z_up_m: float) -> float:
    # The measured HT top replaces the old design top datum.
    return AS_BUILT_HTAIL_TOP_Z_M - (float(z_up_m) - DESIGN_HTAIL_TOP_Z_UP_M)


def vtail_design_z_up_to_build_z_down(z_up_m: float) -> float:
    # The measured VT bottom replaces the old design lower-root datum.
    return AS_BUILT_VTAIL_BOTTOM_Z_M - (
        float(z_up_m) - DESIGN_VTAIL_BOTTOM_Z_UP_M
    )


def wing_dihedral_z_up_offset_m(span_m: float, dihedral_deg: float) -> float:
    # `dihedral_deg` is the per-panel upward angle; for the full wing average
    # absolute span station is one quarter of the full span.
    return 0.25 * span_m * np.tan(np.deg2rad(dihedral_deg))


def wing_tape_segment_span_length_m(span_start_m: float) -> float:
    length_m = 0.5 * AS_BUILT_WING_SPAN_M - span_start_m
    if length_m <= 0.0:
        raise ValueError("Wing tape span start must be inboard of the wing tip.")
    return length_m


def wing_tape_segment_dihedral_z_up_offset_m(
    span_start_m: float,
    span_length_m: float,
) -> float:
    y_abs_cg_m = span_start_m + 0.5 * span_length_m
    return y_abs_cg_m * np.tan(np.deg2rad(AS_BUILT_PANEL_DIHEDRAL_DEG))


def foam_surface_elements() -> list[MassElement]:
    wing_mass_kg = (
        WING_DENSITY_KG_M3
        * AS_BUILT_WING_SPAN_M
        * AS_BUILT_WING_CHORD_M
        * WING_THICKNESS_M
    )
    htail_mass_kg = (
        TAIL_DENSITY_KG_M3
        * AS_BUILT_HTAIL_SPAN_M
        * AS_BUILT_HTAIL_CHORD_M
        * HTAIL_THICKNESS_M
    )
    vtail_mass_kg = (
        TAIL_DENSITY_KG_M3
        * AS_BUILT_VTAIL_HEIGHT_M
        * AS_BUILT_VTAIL_CHORD_M
        * VTAIL_THICKNESS_M
    )
    wing_root_le_m = np.array(
        [
            0.0,
            0.0,
            AS_BUILT_WING_BOTTOM_CENTER_Z_M - 0.5 * WING_THICKNESS_M,
        ]
    )
    htail_root_le_m = np.array(
        [
            AS_BUILT_HTAIL_AC_M - 0.25 * AS_BUILT_HTAIL_CHORD_M,
            0.0,
            AS_BUILT_HTAIL_TOP_Z_M + 0.5 * HTAIL_THICKNESS_M,
        ]
    )
    vtail_root_le_m = np.array(
        [
            AS_BUILT_VTAIL_AC_M - 0.25 * AS_BUILT_VTAIL_CHORD_M,
            0.0,
            AS_BUILT_VTAIL_BOTTOM_Z_M,
        ]
    )
    return (
        horizontal_surface_elements(
            name_prefix="wing_foam",
            root_le_m=wing_root_le_m,
            chord_m=AS_BUILT_WING_CHORD_M,
            span_m=AS_BUILT_WING_SPAN_M,
            dihedral_deg=AS_BUILT_PANEL_DIHEDRAL_DEG,
            mass_kg=wing_mass_kg,
            thickness_m=WING_THICKNESS_M,
        )
        + horizontal_surface_elements(
            name_prefix="horizontal_tail_foam",
            root_le_m=htail_root_le_m,
            chord_m=AS_BUILT_HTAIL_CHORD_M,
            span_m=AS_BUILT_HTAIL_SPAN_M,
            dihedral_deg=0.0,
            mass_kg=htail_mass_kg,
            thickness_m=HTAIL_THICKNESS_M,
        )
        + [
            vertical_surface_element(
                root_le_m=vtail_root_le_m,
                chord_m=AS_BUILT_VTAIL_CHORD_M,
                height_m=AS_BUILT_VTAIL_HEIGHT_M,
                mass_kg=vtail_mass_kg,
                thickness_m=VTAIL_THICKNESS_M,
            )
        ]
    )


def wing_spar_and_tape_elements() -> list[MassElement]:
    x_spar_m = WING_SPAR_X_FROM_WING_LE_M
    z_dihedral_up_m = wing_dihedral_z_up_offset_m(
        span_m=AS_BUILT_WING_SPAN_M,
        dihedral_deg=AS_BUILT_PANEL_DIHEDRAL_DEG,
    )
    z_spar_m = wing_design_z_up_to_build_z_down(
        DESIGN_WING_BOTTOM_Z_UP_M + WING_SPAR_Z_FROM_LOWER_M + z_dihedral_up_m
    )
    spar_density_kg_m3 = tube_density_from_linear_mass_kg_m3(
        WING_SPAR_LINEAR_MASS_KG_PER_M,
        WING_SPAR_OD_M,
        WING_SPAR_ID_M,
    )
    spar = y_axis_hollow_tube(
        name="wing_spar_tube",
        length_m=AS_BUILT_WING_SPAN_M,
        outer_diameter_m=WING_SPAR_OD_M,
        inner_diameter_m=WING_SPAR_ID_M,
        density_kg_m3=spar_density_kg_m3,
        r_cg_m=np.array([x_spar_m, 0.0, z_spar_m]),
    )
    # Spar bond adhesive is measured as a local build assumption: total
    # spar-line mass equals 1.3 times the bare measured tube mass.
    spar_glue = scale_mass_element(
        "wing_spar_glue",
        spar,
        WING_SPAR_GLUE_MASS_FACTOR - 1.0,
    )

    slot_volume_m3 = WING_SPAR_SLOT_W_M * WING_SPAR_SLOT_H_M * AS_BUILT_WING_SPAN_M
    slot_geometric_mass_kg = WING_DENSITY_KG_M3 * slot_volume_m3
    wing_skin_mass_kg = (
        WING_DENSITY_KG_M3
        * AS_BUILT_WING_SPAN_M
        * AS_BUILT_WING_CHORD_M
        * WING_THICKNESS_M
    )
    slot_mass_kg = -min(
        slot_geometric_mass_kg,
        SLOT_VOID_MASS_FRAC_MAX * max(wing_skin_mass_kg, 0.0),
    )
    z_slot_m = wing_design_z_up_to_build_z_down(
        DESIGN_WING_BOTTOM_Z_UP_M + 0.5 * WING_SPAR_SLOT_H_M + z_dihedral_up_m
    )
    slot = rectangular_prism(
        name="wing_spar_slot_void",
        mass_kg=slot_mass_kg,
        size_x_m=WING_SPAR_SLOT_W_M,
        size_y_m=AS_BUILT_WING_SPAN_M,
        size_z_m=WING_SPAR_SLOT_H_M,
        r_cg_m=np.array([x_spar_m, 0.0, z_slot_m]),
        c_ml=np.eye(3),
    )

    # Tape "width" measurements are interpreted chordwise. Wing tape starts
    # at x=35 mm with centre at x=42.5 mm, so the model uses a 15 mm strip.
    x_tape_m = WING_TAPE_X_FROM_WING_LE_M
    bottom_tape_span_m = wing_tape_segment_span_length_m(
        WING_TAPE_BOTTOM_SPAN_START_M
    )
    top_tape_span_m = wing_tape_segment_span_length_m(WING_TAPE_TOP_SPAN_START_M)
    bottom_tape_mass_kg = (
        TAPE_AREAL_DENSITY_KG_M2 * bottom_tape_span_m * WING_TAPE_CHORDWISE_WIDTH_M
    )
    top_tape_mass_kg = (
        TAPE_AREAL_DENSITY_KG_M2 * top_tape_span_m * WING_TAPE_CHORDWISE_WIDTH_M
    )
    z_tape_bottom_m = wing_design_z_up_to_build_z_down(
        DESIGN_WING_BOTTOM_Z_UP_M
        + 0.5 * TAPE_THICKNESS_M
        + wing_tape_segment_dihedral_z_up_offset_m(
            WING_TAPE_BOTTOM_SPAN_START_M,
            bottom_tape_span_m,
        )
    )
    z_tape_top_m = wing_design_z_up_to_build_z_down(
        DESIGN_WING_BOTTOM_Z_UP_M
        + WING_THICKNESS_M
        - 0.5 * TAPE_THICKNESS_M
        + wing_tape_segment_dihedral_z_up_offset_m(
            WING_TAPE_TOP_SPAN_START_M,
            top_tape_span_m,
        )
    )
    tape_bottom = combine_named_elements(
        "wing_tape_bottom",
        [
            rectangular_prism(
                name="wing_tape_bottom_R",
                mass_kg=bottom_tape_mass_kg,
                size_x_m=WING_TAPE_CHORDWISE_WIDTH_M,
                size_y_m=bottom_tape_span_m,
                size_z_m=TAPE_THICKNESS_M,
                r_cg_m=np.array(
                    [
                        x_tape_m,
                        WING_TAPE_BOTTOM_SPAN_START_M + 0.5 * bottom_tape_span_m,
                        z_tape_bottom_m,
                    ]
                ),
                c_ml=np.eye(3),
            ),
            rectangular_prism(
                name="wing_tape_bottom_L",
                mass_kg=bottom_tape_mass_kg,
                size_x_m=WING_TAPE_CHORDWISE_WIDTH_M,
                size_y_m=bottom_tape_span_m,
                size_z_m=TAPE_THICKNESS_M,
                r_cg_m=np.array(
                    [
                        x_tape_m,
                        -(WING_TAPE_BOTTOM_SPAN_START_M + 0.5 * bottom_tape_span_m),
                        z_tape_bottom_m,
                    ]
                ),
                c_ml=np.eye(3),
            ),
        ],
    )
    tape_top = combine_named_elements(
        "wing_tape_top",
        [
            rectangular_prism(
                name="wing_tape_top_R",
                mass_kg=top_tape_mass_kg,
                size_x_m=WING_TAPE_CHORDWISE_WIDTH_M,
                size_y_m=top_tape_span_m,
                size_z_m=TAPE_THICKNESS_M,
                r_cg_m=np.array(
                    [
                        x_tape_m,
                        WING_TAPE_TOP_SPAN_START_M + 0.5 * top_tape_span_m,
                        z_tape_top_m,
                    ]
                ),
                c_ml=np.eye(3),
            ),
            rectangular_prism(
                name="wing_tape_top_L",
                mass_kg=top_tape_mass_kg,
                size_x_m=WING_TAPE_CHORDWISE_WIDTH_M,
                size_y_m=top_tape_span_m,
                size_z_m=TAPE_THICKNESS_M,
                r_cg_m=np.array(
                    [
                        x_tape_m,
                        -(WING_TAPE_TOP_SPAN_START_M + 0.5 * top_tape_span_m),
                        z_tape_top_m,
                    ]
                ),
                c_ml=np.eye(3),
            ),
        ],
    )
    return [spar, spar_glue, slot, tape_bottom, tape_top]


def tail_tape_elements() -> list[MassElement]:
    htail_root_le_x_m = AS_BUILT_HTAIL_AC_M - 0.25 * AS_BUILT_HTAIL_CHORD_M
    vtail_root_le_x_m = AS_BUILT_VTAIL_AC_M - 0.25 * AS_BUILT_VTAIL_CHORD_M

    # Tail tape centre positions are explicit bench measurements. The stated
    # starts and centres imply 20 mm chordwise tape strips on both HT and VT.
    htail_tape_mass_kg = (
        TAPE_AREAL_DENSITY_KG_M2
        * AS_BUILT_HTAIL_SPAN_M
        * TAIL_TAPE_CHORDWISE_WIDTH_M
    )
    x_htail_tape_m = htail_root_le_x_m + HTAIL_TAPE_X_FROM_HTAIL_LE_M
    htail_bottom = rectangular_prism(
        name="htail_tape_bottom",
        mass_kg=htail_tape_mass_kg,
        size_x_m=TAIL_TAPE_CHORDWISE_WIDTH_M,
        size_y_m=AS_BUILT_HTAIL_SPAN_M,
        size_z_m=TAPE_THICKNESS_M,
        r_cg_m=np.array(
            [
                x_htail_tape_m,
                0.0,
                htail_design_z_up_to_build_z_down(
                    DESIGN_HTAIL_BOTTOM_Z_UP_M + 0.5 * TAPE_THICKNESS_M
                ),
            ]
        ),
        c_ml=np.eye(3),
    )
    htail_top = rectangular_prism(
        name="htail_tape_top",
        mass_kg=htail_tape_mass_kg,
        size_x_m=TAIL_TAPE_CHORDWISE_WIDTH_M,
        size_y_m=AS_BUILT_HTAIL_SPAN_M,
        size_z_m=TAPE_THICKNESS_M,
        r_cg_m=np.array(
            [
                x_htail_tape_m,
                0.0,
                htail_design_z_up_to_build_z_down(
                    DESIGN_HTAIL_BOTTOM_Z_UP_M
                    + HTAIL_THICKNESS_M
                    - 0.5 * TAPE_THICKNESS_M
                ),
            ]
        ),
        c_ml=np.eye(3),
    )

    vtail_tape_mass_kg = (
        TAPE_AREAL_DENSITY_KG_M2
        * VTAIL_TAPE_LENGTH_PER_SIDE_M
        * TAIL_TAPE_CHORDWISE_WIDTH_M
    )
    x_vtail_tape_m = vtail_root_le_x_m + VTAIL_TAPE_X_FROM_VTAIL_LE_M
    z_vtail_tape_m = vtail_design_z_up_to_build_z_down(
        DESIGN_VTAIL_BOTTOM_Z_UP_M
        + AS_BUILT_VTAIL_HEIGHT_M
        - 0.5 * VTAIL_TAPE_LENGTH_PER_SIDE_M
    )
    vtail_side_a = rectangular_prism(
        name="vtail_tape_side_a",
        mass_kg=vtail_tape_mass_kg,
        size_x_m=TAIL_TAPE_CHORDWISE_WIDTH_M,
        size_y_m=TAPE_THICKNESS_M,
        size_z_m=VTAIL_TAPE_LENGTH_PER_SIDE_M,
        r_cg_m=np.array([x_vtail_tape_m, VTAIL_TAPE_Y_OFFSET_M, z_vtail_tape_m]),
        c_ml=np.eye(3),
    )
    vtail_side_b = rectangular_prism(
        name="vtail_tape_side_b",
        mass_kg=vtail_tape_mass_kg,
        size_x_m=TAIL_TAPE_CHORDWISE_WIDTH_M,
        size_y_m=TAPE_THICKNESS_M,
        size_z_m=VTAIL_TAPE_LENGTH_PER_SIDE_M,
        r_cg_m=np.array([x_vtail_tape_m, -VTAIL_TAPE_Y_OFFSET_M, z_vtail_tape_m]),
        c_ml=np.eye(3),
    )
    return [htail_bottom, htail_top, vtail_side_a, vtail_side_b]


def centre_module_elements() -> list[MassElement]:
    wing_mount_area_m2 = trapezoid_area_m2(
        WING_MOUNT_LB_M,
        WING_MOUNT_LT_M,
        WING_MOUNT_H_M,
    )
    wing_mount_mass_kg = (
        CENTRE_MODULE_RHO_KG_M3 * wing_mount_area_m2 * WING_MOUNT_T_M
    )
    wing_mount_ybar_m = trapezoid_centroid_from_base_m(
        WING_MOUNT_LB_M,
        WING_MOUNT_LT_M,
        WING_MOUNT_H_M,
    )
    wing_mount_xbar_m = right_trapezoid_centroid_u_from_bottom_fore_m(
        base_bottom_m=WING_MOUNT_LB_M,
        base_top_m=WING_MOUNT_LT_M,
        height_m=WING_MOUNT_H_M,
        straight_side="aft",
    )
    # The original mount offsets were defined relative to the quarter-chord
    # spar line; here that reference is replaced by the measured spar x-line.
    wing_mount_x0_fwd_m = (
        WING_SPAR_X_FROM_WING_LE_M + WING_MOUNT_X0_FWD_OFFSET_FROM_SPAR_M
    )
    wing_mount_x0_aft_m = (
        WING_SPAR_X_FROM_WING_LE_M + WING_MOUNT_X0_AFT_OFFSET_FROM_SPAR_M
    )
    tan_dihedral = np.tan(np.deg2rad(AS_BUILT_PANEL_DIHEDRAL_DEG))
    elements: list[MassElement] = []
    for side_label, side_sign in (("R", 1.0), ("L", -1.0)):
        y_mount_m = side_sign * wing_mount_ybar_m
        z_mount_up_m = (
            WING_MOUNT_ROOT_BOTTOM_Z_UP_M
            + 0.5 * WING_MOUNT_T_M
            + abs(y_mount_m) * tan_dihedral
        )
        z_mount_m = wing_design_z_up_to_build_z_down(z_mount_up_m)
        mount_fwd = right_trapezoid_prism(
            name=f"wing_mount_{side_label}_fwd",
            mass_kg=wing_mount_mass_kg,
            base_bottom_m=WING_MOUNT_LB_M,
            base_top_m=WING_MOUNT_LT_M,
            height_m=WING_MOUNT_H_M,
            thickness_m=WING_MOUNT_T_M,
            r_cg_m=np.array(
                [wing_mount_x0_fwd_m + wing_mount_xbar_m, y_mount_m, z_mount_m]
            ),
            height_axis="y",
            straight_side="aft",
        )
        mount_aft = right_trapezoid_prism(
            name=f"wing_mount_{side_label}_aft",
            mass_kg=wing_mount_mass_kg,
            base_bottom_m=WING_MOUNT_LB_M,
            base_top_m=WING_MOUNT_LT_M,
            height_m=WING_MOUNT_H_M,
            thickness_m=WING_MOUNT_T_M,
            r_cg_m=np.array(
                [wing_mount_x0_aft_m + wing_mount_xbar_m, y_mount_m, z_mount_m]
            ),
            height_axis="y",
            straight_side="aft",
        )
        for mount in (mount_fwd, mount_aft):
            # Mirrored right-trapezoid mounts reverse their local y-axis, so
            # their intrinsic Ixy terms must reverse sign for symmetry.
            mount.inertia_m[0, 1] *= side_sign
            mount.inertia_m[1, 0] *= side_sign
            elements.append(mount)

    joiner_x_m = wing_mount_x0_fwd_m + 0.5 * WING_MOUNT_LB_M
    joiner_z_m = wing_design_z_up_to_build_z_down(
        DESIGN_WING_BOTTOM_Z_UP_M + WING_SPAR_Z_FROM_LOWER_M
    )
    joiner_outer = rectangular_prism(
        name="wing_mount_spar_joiner_outer",
        mass_kg=(
            CENTRE_MODULE_RHO_KG_M3
            * WING_SPAR_JOINER_W_M
            * WING_SPAR_JOINER_SPAN_M
            * WING_SPAR_JOINER_H_M
        ),
        size_x_m=WING_SPAR_JOINER_W_M,
        size_y_m=WING_SPAR_JOINER_SPAN_M,
        size_z_m=WING_SPAR_JOINER_H_M,
        r_cg_m=np.array([joiner_x_m, 0.0, joiner_z_m]),
        c_ml=np.eye(3),
    )
    joiner_hole = scale_mass_element(
        name="wing_mount_spar_joiner_hole",
        element=y_axis_hollow_tube(
            name="wing_mount_spar_joiner_hole_positive",
            length_m=WING_SPAR_JOINER_SPAN_M,
            outer_diameter_m=WING_SPAR_JOINER_HOLE_D_M,
            inner_diameter_m=0.0,
            density_kg_m3=CENTRE_MODULE_RHO_KG_M3,
            r_cg_m=np.array([joiner_x_m, 0.0, joiner_z_m]),
        ),
        scale=-1.0,
    )
    joiner = combine_named_elements(
        "wing_mount_spar_joiner_fwd",
        [joiner_outer, joiner_hole],
    )
    elements.append(joiner)

    mounts_total = combine_named_elements("centre_module_mounts", elements)
    centre_core_mass_kg = CENTRE_MODULE_MASS_KG - mounts_total.mass_kg
    if centre_core_mass_kg <= 0.0:
        raise ValueError("Centre-module mounts exceed the centre-module mass.")
    x_centre_core_m = 0.25 * AS_BUILT_WING_CHORD_M + CENTRE_CORE_X_OFFSET_FROM_0P3C_M
    elements.append(
        point_mass(
            "centre_module_core",
            centre_core_mass_kg,
            np.array(
                [
                    x_centre_core_m,
                    0.0,
                    wing_design_z_up_to_build_z_down(CENTRE_CORE_Z_CG_UP_M),
                ]
            ),
        )
    )
    return elements


def tail_mount_and_skid_elements() -> list[MassElement]:
    htail_root_le_x_m = AS_BUILT_HTAIL_AC_M - 0.25 * AS_BUILT_HTAIL_CHORD_M
    vtail_root_le_x_m = AS_BUILT_VTAIL_AC_M - 0.25 * AS_BUILT_VTAIL_CHORD_M
    boom_end_x_m = htail_root_le_x_m + 0.70 * AS_BUILT_HTAIL_CHORD_M

    htail_mount_area_m2 = trapezoid_area_m2(
        HTAIL_MOUNT_LB_M,
        HTAIL_MOUNT_LT_M,
        HTAIL_MOUNT_H_M,
    )
    htail_mount_mass_kg = (
        TAIL_MODULE_RHO_KG_M3 * htail_mount_area_m2 * HTAIL_MOUNT_T_M
    )
    htail_mount_ybar_m = trapezoid_centroid_from_base_m(
        HTAIL_MOUNT_LB_M,
        HTAIL_MOUNT_LT_M,
        HTAIL_MOUNT_H_M,
    )
    htail_mount_x0_m = boom_end_x_m - HTAIL_MOUNT_X0_OFFSET_FROM_BOOM_END_M
    htail_mount_z_m = htail_design_z_up_to_build_z_down(
        HTAIL_MOUNT_Z_UPPER_UP_M - 0.5 * HTAIL_MOUNT_T_M
    )
    elements: list[MassElement] = []
    for side_label, side_sign in (("R", 1.0), ("L", -1.0)):
        elements.append(
            isosceles_trapezoid_prism(
                name=f"htail_mount_{side_label}",
                mass_kg=htail_mount_mass_kg,
                base_bottom_m=HTAIL_MOUNT_LB_M,
                base_top_m=HTAIL_MOUNT_LT_M,
                height_m=HTAIL_MOUNT_H_M,
                thickness_m=HTAIL_MOUNT_T_M,
                r_cg_m=np.array(
                    [
                        htail_mount_x0_m + 0.5 * HTAIL_MOUNT_LB_M,
                        side_sign * htail_mount_ybar_m,
                        htail_mount_z_m,
                    ]
                ),
                height_axis="y",
            )
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
    elements.append(
        isosceles_trapezoid_prism(
            name="vtail_mount",
            mass_kg=vtail_mount_mass_kg,
            base_bottom_m=VTAIL_MOUNT_LB_M,
            base_top_m=VTAIL_MOUNT_LT_M,
            height_m=VTAIL_MOUNT_H_M,
            thickness_m=VTAIL_MOUNT_T_M,
            r_cg_m=np.array(
                [
                    vtail_mount_x0_m + 0.5 * VTAIL_MOUNT_LB_M,
                    0.0,
                    vtail_design_z_up_to_build_z_down(
                        DESIGN_VTAIL_BOTTOM_Z_UP_M + vtail_mount_zbar_m
                    ),
                ]
            ),
            height_axis="z",
        )
    )

    mount_total = combine_named_elements("tail_mount_without_core", elements)
    tail_core_mass_kg = TAIL_MOUNT_AS_BUILT_MASS_KG - mount_total.mass_kg
    if tail_core_mass_kg <= 0.0:
        raise ValueError("Tail mounts exceed the specified tail-mount mass.")
    elements.append(
        point_mass(
            "tail_module_core",
            tail_core_mass_kg,
            np.array(
                [
                    boom_end_x_m - TAIL_CORE_X_OFFSET_FROM_BOOM_END_M,
                    0.0,
                    htail_design_z_up_to_build_z_down(TAIL_CORE_Z_CG_UP_M),
                ]
            ),
        )
    )
    elements.append(
        point_mass(
            "tail_skid",
            TAIL_SUPPORT_MASS_KG,
            np.array(
                [
                    htail_root_le_x_m + TAIL_GEAR_X_OFFSET_FROM_HTAIL_LE_M,
                    0.0,
                    htail_design_z_up_to_build_z_down(TAIL_SUPPORT_Z_CG_UP_M),
                ]
            ),
        )
    )
    return elements


def fuselage_element() -> MassElement:
    fuselage_density_kg_m3 = tube_density_from_linear_mass_kg_m3(
        FUSELAGE_ROD_LINEAR_MASS_KG_PER_M,
        BOOM_TUBE_OUTER_DIAMETER_M,
        BOOM_TUBE_INNER_DIAMETER_M,
    )
    return x_axis_hollow_tube(
        name="fuselage_rod",
        length_m=AS_BUILT_FUSELAGE_LENGTH_M,
        outer_diameter_m=BOOM_TUBE_OUTER_DIAMETER_M,
        inner_diameter_m=BOOM_TUBE_INNER_DIAMETER_M,
        density_kg_m3=fuselage_density_kg_m3,
        r_cg_m=np.array(
            [
                AS_BUILT_FUSELAGE_NOSE_X_FROM_WING_LE_M
                + 0.5 * AS_BUILT_FUSELAGE_LENGTH_M,
                0.0,
                0.0,
            ]
        ),
    )


def named_avionics_elements() -> list[MassElement]:
    # Receiver dimensions are interpreted as x-length, y-width, z-height.
    # The measured front and top faces define the CG in the build frame.
    receiver_cg_m = np.array(
        [
            RECEIVER_FRONT_X_FROM_WING_LE_M + 0.5 * RECEIVER_SIZE_X_M,
            0.0,
            RECEIVER_TOP_Z_M + 0.5 * RECEIVER_SIZE_Z_M,
        ]
    )
    return [
        rectangular_prism(
            name="receiver",
            mass_kg=RECEIVER_MASS_KG,
            size_x_m=RECEIVER_SIZE_X_M,
            size_y_m=RECEIVER_SIZE_Y_M,
            size_z_m=RECEIVER_SIZE_Z_M,
            r_cg_m=receiver_cg_m,
            c_ml=np.eye(3),
        )
    ]


def vicon_marker_elements() -> list[MassElement]:
    half_span_m = 0.5 * AS_BUILT_WING_SPAN_M
    dihedral_rad = np.deg2rad(AS_BUILT_PANEL_DIHEDRAL_DEG)
    wingtip_y_edge_m = half_span_m * np.cos(dihedral_rad)
    # The wingtip markers are offset outward by one marker radius in y, and
    # their z-centres follow the current as-built wingtip bottom datum.
    wingtip_z_m = AS_BUILT_WING_BOTTOM_CENTER_Z_M - half_span_m * np.sin(
        dihedral_rad
    )
    return [
        solid_sphere(
            "vicon_marker_fuselage_front",
            VICON_MARKER_MASS_KG,
            VICON_MARKER_RADIUS_M,
            np.array(
                [
                    AS_BUILT_FUSELAGE_NOSE_X_FROM_WING_LE_M
                    - VICON_MARKER_RADIUS_M,
                    0.0,
                    0.0,
                ]
            ),
        ),
        solid_sphere(
            "vicon_marker_wing_le_front",
            VICON_MARKER_MASS_KG,
            VICON_MARKER_RADIUS_M,
            np.array(
                [
                    -VICON_MARKER_RADIUS_M,
                    0.0,
                    VICON_WING_LE_MARKER_Z_M,
                ]
            ),
        ),
        solid_sphere(
            "vicon_marker_wingtip_R",
            VICON_MARKER_MASS_KG,
            VICON_MARKER_RADIUS_M,
            np.array(
                [
                    VICON_WINGTIP_MARKER_X_FROM_WING_LE_M,
                    wingtip_y_edge_m + VICON_MARKER_RADIUS_M,
                    wingtip_z_m,
                ]
            ),
        ),
        solid_sphere(
            "vicon_marker_wingtip_L",
            VICON_MARKER_MASS_KG,
            VICON_MARKER_RADIUS_M,
            np.array(
                [
                    VICON_WINGTIP_MARKER_X_FROM_WING_LE_M,
                    -(wingtip_y_edge_m + VICON_MARKER_RADIUS_M),
                    wingtip_z_m,
                ]
            ),
        ),
    ]


def calibration_residual_element(elements: list[MassElement]) -> MassElement:
    subtotal = combine_named_elements("subtotal_before_calibration_residual", elements)
    residual_mass_kg = CALIBRATION_RESIDUAL_TARGET_MASS_KG - subtotal.mass_kg
    if residual_mass_kg <= 0.0:
        raise ValueError("Calibration residual mass must be positive.")
    residual_x_m = (
        CALIBRATION_RESIDUAL_TARGET_MASS_KG * CALIBRATION_RESIDUAL_TARGET_X_CG_M
        - subtotal.mass_kg * subtotal.r_cg_m[0]
    ) / residual_mass_kg
    # The residual represents unmodelled glue/small manufacturing mass. It is
    # calibrated only to total mass and x_CG, so y and z are intentionally zero.
    return point_mass(
        "calibration_residual",
        residual_mass_kg,
        np.array([residual_x_m, 0.0, 0.0]),
    )


# =============================================================================
# 4) Manufactured Estimate Case
# =============================================================================
def build_baseline_elements() -> list[MassElement]:
    assert_as_built_glider_parameters()
    # Component build-up datum: x=0 at wing LE, y=0 at the centreline, and
    # z=0 at the fuselage rod centre. Positive z points downward.
    elements = (
        foam_surface_elements()
        + wing_spar_and_tape_elements()
        + tail_tape_elements()
        + centre_module_elements()
        + tail_mount_and_skid_elements()
        + [
            fuselage_element(),
            battery_element(AS_BUILT_WING_BOTTOM_CENTER_Z_M),
        ]
        + named_avionics_elements()
        + aileron_servo_elements(
            wing_bottom_center_z_m=AS_BUILT_WING_BOTTOM_CENTER_Z_M,
            wing_dihedral_deg=AS_BUILT_PANEL_DIHEDRAL_DEG,
        )
        + tail_servo_elements(AS_BUILT_WING_BOTTOM_CENTER_Z_M)
        + tail_pushrod_elements()
    )
    elements += vicon_marker_elements()
    # The residual is the final calibration term: it matches the complete
    # manufactured mass and x_CG after all measured components are included.
    elements.append(calibration_residual_element(elements))
    return elements


def build_baseline_estimate() -> MassEstimate:
    return combine_mass_elements(build_baseline_elements())


# =============================================================================
# 5) Output Writer
# =============================================================================
# The generated module is plain constants so runtime imports do not rerun the
# component build-up. Re-run this script after measured component updates.
def write_mass_properties_module(
    output_path: Path,
    estimate: MassEstimate,
) -> None:
    lines = [
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        "# Generated by build_mass_properties_estimate.py from deterministic hand estimates.",
        "# R_CG_BUILD_M uses x aft from wing LE, y starboard, z down from fuselage rod centre.",
        "# INERTIA_B is about the assembled CG in simulation body-axis orientation.",
        "",
        "# =============================================================================",
        "# SECTION MAP",
        "# =============================================================================",
        "# 1) Generated mass properties",
        "# =============================================================================",
        "",
        "",
        "# =============================================================================",
        "# 1) Generated Mass Properties",
        "# =============================================================================",
        f"MASS_KG = {estimate.mass_kg:.16f}",
        (
            "R_CG_BUILD_M = np.array("
            f"{estimate.r_cg_m.tolist()}, dtype=float)"
        ),
        (
            "R_CG_B = np.array("
            f"{estimate.r_cg_b.tolist()}, dtype=float)"
        ),
        (
            "INERTIA_B = np.array("
            f"{estimate.inertia_b.tolist()}, dtype=float)"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def component_position_rows_cm(
    elements: list[MassElement],
) -> list[dict[str, float | str]]:
    # Human audit rows use cm and grams so bench measurements can be checked
    # directly against the manufacturing notes.
    rows: list[dict[str, float | str]] = []
    for element in elements:
        rows.append(
            {
                "component": element.name,
                "mass_g": 1000.0 * element.mass_kg,
                "x_cm": 100.0 * element.r_cg_m[0],
                "y_cm": 100.0 * element.r_cg_m[1],
                "z_cm": 100.0 * element.r_cg_m[2],
            }
        )
    return rows


def format_component_position_table_cm(elements: list[MassElement]) -> str:
    rows = component_position_rows_cm(elements)
    lines = [
        "component                         mass_g       x_cm       y_cm       z_cm",
        "------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            (
                f"{str(row['component']):<32}"
                f"{float(row['mass_g']):>8.3f}"
                f"{float(row['x_cm']):>11.3f}"
                f"{float(row['y_cm']):>11.3f}"
                f"{float(row['z_cm']):>11.3f}"
            )
        )
    return "\n".join(lines)


def main() -> None:
    elements = build_baseline_elements()
    estimate = combine_mass_elements(elements)
    output_path = Path(__file__).with_name("mass_properties_estimate.py")
    write_mass_properties_module(output_path, estimate)
    print(format_component_position_table_cm(elements))
    print()
    print(f"wrote {output_path}")
    print(f"mass_kg = {estimate.mass_kg:.9f}")
    print(f"r_cg_build_m = {estimate.r_cg_m.tolist()}")
    print(f"inertia_b = {estimate.inertia_b.tolist()}")


if __name__ == "__main__":
    main()
