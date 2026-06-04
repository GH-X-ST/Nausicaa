from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from A_model_parameters.mass_properties_estimate import (
    INERTIA_B as ESTIMATED_INERTIA_B,
    MASS_KG as ESTIMATED_MASS_KG,
    R_CG_BUILD_M as ESTIMATED_R_CG_BUILD_M,
)
from A_model_parameters.neutral_dry_air_calibration import (
    CALIBRATION_ACTIVE as NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE,
    CD0_STRIP_SCALE as NEUTRAL_DRY_AIR_CD0_STRIP_SCALE,
    DELTA_A_TRIM_RAD as NEUTRAL_DRY_AIR_DELTA_A_TRIM_RAD,
    DELTA_E_TRIM_RAD as NEUTRAL_DRY_AIR_DELTA_E_TRIM_RAD,
    DELTA_R_TRIM_RAD as NEUTRAL_DRY_AIR_DELTA_R_TRIM_RAD,
    DRAG_AREA_FUSE_SCALE as NEUTRAL_DRY_AIR_DRAG_AREA_FUSE_SCALE,
    EFFICIENCY_STRIP_SCALE as NEUTRAL_DRY_AIR_EFFICIENCY_STRIP_SCALE,
    PITCH_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_PITCH_MOMENT_BIAS_COEFF,
    POST_STALL_DRAG_RESIDUAL_COEFF as NEUTRAL_DRY_AIR_POST_STALL_DRAG_RESIDUAL_COEFF,
    POST_STALL_LIFT_RESIDUAL_COEFF as NEUTRAL_DRY_AIR_POST_STALL_LIFT_RESIDUAL_COEFF,
    POST_STALL_PITCH_DAMPING_COEFF as NEUTRAL_DRY_AIR_POST_STALL_PITCH_DAMPING_COEFF,
    POST_STALL_PITCH_MOMENT_COEFF as NEUTRAL_DRY_AIR_POST_STALL_PITCH_MOMENT_COEFF,
    POST_STALL_DRAG_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_DRAG_RBF_COEFFS,
    POST_STALL_LIFT_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_LIFT_RBF_COEFFS,
    POST_STALL_PITCH_DAMPING_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_PITCH_DAMPING_RBF_COEFFS,
    POST_STALL_PITCH_MOMENT_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_PITCH_MOMENT_RBF_COEFFS,
    POST_STALL_RBF_ALPHA_CENTERS_DEG as NEUTRAL_DRY_AIR_POST_STALL_RBF_ALPHA_CENTERS_DEG,
    POST_STALL_RBF_ALPHA_WIDTH_DEG as NEUTRAL_DRY_AIR_POST_STALL_RBF_ALPHA_WIDTH_DEG,
    POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG as NEUTRAL_DRY_AIR_POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG,
    POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG as NEUTRAL_DRY_AIR_POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG,
    POST_STALL_ROLL_MOMENT_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_ROLL_MOMENT_RBF_COEFFS,
    POST_STALL_SIDE_FORCE_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_SIDE_FORCE_RBF_COEFFS,
    POST_STALL_YAW_MOMENT_RBF_COEFFS as NEUTRAL_DRY_AIR_POST_STALL_YAW_MOMENT_RBF_COEFFS,
    ROLL_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_ROLL_MOMENT_BIAS_COEFF,
    ROLL_MOMENT_BETA_COEFF as NEUTRAL_DRY_AIR_ROLL_MOMENT_BETA_COEFF,
    ROLL_MOMENT_P_HAT_COEFF as NEUTRAL_DRY_AIR_ROLL_MOMENT_P_HAT_COEFF,
    ROLL_MOMENT_R_HAT_COEFF as NEUTRAL_DRY_AIR_ROLL_MOMENT_R_HAT_COEFF,
    SIDE_FORCE_BETA_COEFF as NEUTRAL_DRY_AIR_SIDE_FORCE_BETA_COEFF,
    SIDE_FORCE_BIAS_COEFF as NEUTRAL_DRY_AIR_SIDE_FORCE_BIAS_COEFF,
    SIDE_FORCE_P_HAT_COEFF as NEUTRAL_DRY_AIR_SIDE_FORCE_P_HAT_COEFF,
    SIDE_FORCE_R_HAT_COEFF as NEUTRAL_DRY_AIR_SIDE_FORCE_R_HAT_COEFF,
    TRANSITION_ROLL_MOMENT_BETA_COEFF as NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_BETA_COEFF,
    TRANSITION_ROLL_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_BIAS_COEFF,
    TRANSITION_ROLL_MOMENT_P_HAT_COEFF as NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_P_HAT_COEFF,
    TRANSITION_ROLL_MOMENT_R_HAT_COEFF as NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_R_HAT_COEFF,
    TRANSITION_SIDE_FORCE_BETA_COEFF as NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_BETA_COEFF,
    TRANSITION_SIDE_FORCE_BIAS_COEFF as NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_BIAS_COEFF,
    TRANSITION_SIDE_FORCE_P_HAT_COEFF as NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_P_HAT_COEFF,
    TRANSITION_SIDE_FORCE_R_HAT_COEFF as NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_R_HAT_COEFF,
    TRANSITION_YAW_MOMENT_BETA_COEFF as NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_BETA_COEFF,
    TRANSITION_YAW_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_BIAS_COEFF,
    TRANSITION_YAW_MOMENT_P_HAT_COEFF as NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_P_HAT_COEFF,
    TRANSITION_YAW_MOMENT_R_HAT_COEFF as NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_R_HAT_COEFF,
    YAW_MOMENT_BETA_COEFF as NEUTRAL_DRY_AIR_YAW_MOMENT_BETA_COEFF,
    YAW_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_YAW_MOMENT_BIAS_COEFF,
    YAW_MOMENT_P_HAT_COEFF as NEUTRAL_DRY_AIR_YAW_MOMENT_P_HAT_COEFF,
    YAW_MOMENT_R_HAT_COEFF as NEUTRAL_DRY_AIR_YAW_MOMENT_R_HAT_COEFF,
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
    neutral_surface_trim_rad: np.ndarray
    side_force_bias_coeff: float
    side_force_beta_coeff: float
    side_force_p_hat_coeff: float
    side_force_r_hat_coeff: float
    roll_moment_bias_coeff: float
    roll_moment_beta_coeff: float
    roll_moment_p_hat_coeff: float
    roll_moment_r_hat_coeff: float
    pitch_moment_bias_coeff: float
    yaw_moment_bias_coeff: float
    yaw_moment_beta_coeff: float
    yaw_moment_p_hat_coeff: float
    yaw_moment_r_hat_coeff: float
    transition_side_force_bias_coeff: float
    transition_side_force_beta_coeff: float
    transition_side_force_p_hat_coeff: float
    transition_side_force_r_hat_coeff: float
    transition_roll_moment_bias_coeff: float
    transition_roll_moment_beta_coeff: float
    transition_roll_moment_p_hat_coeff: float
    transition_roll_moment_r_hat_coeff: float
    transition_yaw_moment_bias_coeff: float
    transition_yaw_moment_beta_coeff: float
    transition_yaw_moment_p_hat_coeff: float
    transition_yaw_moment_r_hat_coeff: float
    post_stall_lift_residual_coeff: float
    post_stall_drag_residual_coeff: float
    post_stall_pitch_moment_coeff: float
    post_stall_pitch_damping_coeff: float
    post_stall_residual_blend_start_alpha_rad: float
    post_stall_residual_blend_full_alpha_rad: float
    post_stall_residual_surface_alpha_centers_rad: np.ndarray
    post_stall_residual_surface_width_rad: float
    post_stall_lift_surface_coeff: np.ndarray
    post_stall_drag_surface_coeff: np.ndarray
    post_stall_pitch_moment_surface_coeff: np.ndarray
    post_stall_pitch_damping_surface_coeff: np.ndarray
    post_stall_side_force_surface_coeff: np.ndarray
    post_stall_roll_moment_surface_coeff: np.ndarray
    post_stall_yaw_moment_surface_coeff: np.ndarray
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
    cd0_scale = NEUTRAL_DRY_AIR_CD0_STRIP_SCALE if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 1.0
    efficiency_scale = NEUTRAL_DRY_AIR_EFFICIENCY_STRIP_SCALE if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 1.0
    drag_area_fuse_scale = NEUTRAL_DRY_AIR_DRAG_AREA_FUSE_SCALE if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 1.0
    side_force_bias_coeff = NEUTRAL_DRY_AIR_SIDE_FORCE_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    side_force_beta_coeff = NEUTRAL_DRY_AIR_SIDE_FORCE_BETA_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    side_force_p_hat_coeff = NEUTRAL_DRY_AIR_SIDE_FORCE_P_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    side_force_r_hat_coeff = NEUTRAL_DRY_AIR_SIDE_FORCE_R_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    roll_moment_bias_coeff = (
        NEUTRAL_DRY_AIR_ROLL_MOMENT_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    roll_moment_beta_coeff = NEUTRAL_DRY_AIR_ROLL_MOMENT_BETA_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    roll_moment_p_hat_coeff = NEUTRAL_DRY_AIR_ROLL_MOMENT_P_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    roll_moment_r_hat_coeff = NEUTRAL_DRY_AIR_ROLL_MOMENT_R_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    pitch_moment_bias_coeff = (
        NEUTRAL_DRY_AIR_PITCH_MOMENT_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    yaw_moment_bias_coeff = (
        NEUTRAL_DRY_AIR_YAW_MOMENT_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    yaw_moment_beta_coeff = NEUTRAL_DRY_AIR_YAW_MOMENT_BETA_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    yaw_moment_p_hat_coeff = NEUTRAL_DRY_AIR_YAW_MOMENT_P_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    yaw_moment_r_hat_coeff = NEUTRAL_DRY_AIR_YAW_MOMENT_R_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    transition_side_force_bias_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_side_force_beta_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_BETA_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_side_force_p_hat_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_P_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_side_force_r_hat_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_SIDE_FORCE_R_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_roll_moment_bias_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_roll_moment_beta_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_BETA_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_roll_moment_p_hat_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_P_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_roll_moment_r_hat_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_ROLL_MOMENT_R_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_yaw_moment_bias_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_BIAS_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_yaw_moment_beta_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_BETA_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_yaw_moment_p_hat_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_P_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    transition_yaw_moment_r_hat_coeff = (
        NEUTRAL_DRY_AIR_TRANSITION_YAW_MOMENT_R_HAT_COEFF if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE else 0.0
    )
    post_stall_lift_residual_coeff = (
        NEUTRAL_DRY_AIR_POST_STALL_LIFT_RESIDUAL_COEFF
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else 0.0
    )
    post_stall_drag_residual_coeff = (
        NEUTRAL_DRY_AIR_POST_STALL_DRAG_RESIDUAL_COEFF
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else 0.0
    )
    post_stall_pitch_moment_coeff = (
        NEUTRAL_DRY_AIR_POST_STALL_PITCH_MOMENT_COEFF
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else 0.0
    )
    post_stall_pitch_damping_coeff = (
        NEUTRAL_DRY_AIR_POST_STALL_PITCH_DAMPING_COEFF
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else 0.0
    )
    post_stall_residual_blend_start_alpha_rad = np.deg2rad(
        NEUTRAL_DRY_AIR_POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else 12.0
    )
    post_stall_residual_blend_full_alpha_rad = np.deg2rad(
        NEUTRAL_DRY_AIR_POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else 20.0
    )
    post_stall_residual_surface_alpha_centers_rad = np.deg2rad(
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_RBF_ALPHA_CENTERS_DEG, dtype=float)
    )
    post_stall_residual_surface_width_rad = np.deg2rad(float(NEUTRAL_DRY_AIR_POST_STALL_RBF_ALPHA_WIDTH_DEG))
    post_stall_lift_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_LIFT_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros_like(post_stall_residual_surface_alpha_centers_rad)
    )
    post_stall_drag_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_DRAG_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros_like(post_stall_residual_surface_alpha_centers_rad)
    )
    post_stall_pitch_moment_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_PITCH_MOMENT_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros_like(post_stall_residual_surface_alpha_centers_rad)
    )
    post_stall_pitch_damping_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_PITCH_DAMPING_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros_like(post_stall_residual_surface_alpha_centers_rad)
    )
    lateral_surface_shape = (4, post_stall_residual_surface_alpha_centers_rad.size)
    post_stall_side_force_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_SIDE_FORCE_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros(lateral_surface_shape, dtype=float)
    ).reshape(lateral_surface_shape)
    post_stall_roll_moment_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_ROLL_MOMENT_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros(lateral_surface_shape, dtype=float)
    ).reshape(lateral_surface_shape)
    post_stall_yaw_moment_surface_coeff = (
        np.asarray(NEUTRAL_DRY_AIR_POST_STALL_YAW_MOMENT_RBF_COEFFS, dtype=float)
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros(lateral_surface_shape, dtype=float)
    ).reshape(lateral_surface_shape)
    neutral_surface_trim_rad = (
        np.array(
            [
                NEUTRAL_DRY_AIR_DELTA_A_TRIM_RAD,
                NEUTRAL_DRY_AIR_DELTA_E_TRIM_RAD,
                NEUTRAL_DRY_AIR_DELTA_R_TRIM_RAD,
            ],
            dtype=float,
        )
        if NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE
        else np.zeros(3, dtype=float)
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
        cd0=0.018 * cd0_scale,
        alpha0=0.001,
        induced_drag_efficiency=0.82 * efficiency_scale,
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
        cd0=0.020 * cd0_scale,
        alpha0=np.deg2rad(-3.0),
        induced_drag_efficiency=0.78 * efficiency_scale,
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
        cd0=0.020 * cd0_scale,
        alpha0=0.0,
        induced_drag_efficiency=0.75 * efficiency_scale,
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
        drag_area_fuse_m2=4.89753346075483e-05 * drag_area_fuse_scale,
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
        neutral_surface_trim_rad=neutral_surface_trim_rad,
        side_force_bias_coeff=float(side_force_bias_coeff),
        side_force_beta_coeff=float(side_force_beta_coeff),
        side_force_p_hat_coeff=float(side_force_p_hat_coeff),
        side_force_r_hat_coeff=float(side_force_r_hat_coeff),
        roll_moment_bias_coeff=float(roll_moment_bias_coeff),
        roll_moment_beta_coeff=float(roll_moment_beta_coeff),
        roll_moment_p_hat_coeff=float(roll_moment_p_hat_coeff),
        roll_moment_r_hat_coeff=float(roll_moment_r_hat_coeff),
        pitch_moment_bias_coeff=float(pitch_moment_bias_coeff),
        yaw_moment_bias_coeff=float(yaw_moment_bias_coeff),
        yaw_moment_beta_coeff=float(yaw_moment_beta_coeff),
        yaw_moment_p_hat_coeff=float(yaw_moment_p_hat_coeff),
        yaw_moment_r_hat_coeff=float(yaw_moment_r_hat_coeff),
        transition_side_force_bias_coeff=float(transition_side_force_bias_coeff),
        transition_side_force_beta_coeff=float(transition_side_force_beta_coeff),
        transition_side_force_p_hat_coeff=float(transition_side_force_p_hat_coeff),
        transition_side_force_r_hat_coeff=float(transition_side_force_r_hat_coeff),
        transition_roll_moment_bias_coeff=float(transition_roll_moment_bias_coeff),
        transition_roll_moment_beta_coeff=float(transition_roll_moment_beta_coeff),
        transition_roll_moment_p_hat_coeff=float(transition_roll_moment_p_hat_coeff),
        transition_roll_moment_r_hat_coeff=float(transition_roll_moment_r_hat_coeff),
        transition_yaw_moment_bias_coeff=float(transition_yaw_moment_bias_coeff),
        transition_yaw_moment_beta_coeff=float(transition_yaw_moment_beta_coeff),
        transition_yaw_moment_p_hat_coeff=float(transition_yaw_moment_p_hat_coeff),
        transition_yaw_moment_r_hat_coeff=float(transition_yaw_moment_r_hat_coeff),
        post_stall_lift_residual_coeff=float(post_stall_lift_residual_coeff),
        post_stall_drag_residual_coeff=float(post_stall_drag_residual_coeff),
        post_stall_pitch_moment_coeff=float(post_stall_pitch_moment_coeff),
        post_stall_pitch_damping_coeff=float(post_stall_pitch_damping_coeff),
        post_stall_residual_blend_start_alpha_rad=float(post_stall_residual_blend_start_alpha_rad),
        post_stall_residual_blend_full_alpha_rad=float(post_stall_residual_blend_full_alpha_rad),
        post_stall_residual_surface_alpha_centers_rad=np.asarray(
            post_stall_residual_surface_alpha_centers_rad,
            dtype=float,
        ),
        post_stall_residual_surface_width_rad=float(post_stall_residual_surface_width_rad),
        post_stall_lift_surface_coeff=np.asarray(post_stall_lift_surface_coeff, dtype=float),
        post_stall_drag_surface_coeff=np.asarray(post_stall_drag_surface_coeff, dtype=float),
        post_stall_pitch_moment_surface_coeff=np.asarray(post_stall_pitch_moment_surface_coeff, dtype=float),
        post_stall_pitch_damping_surface_coeff=np.asarray(post_stall_pitch_damping_surface_coeff, dtype=float),
        post_stall_side_force_surface_coeff=np.asarray(post_stall_side_force_surface_coeff, dtype=float),
        post_stall_roll_moment_surface_coeff=np.asarray(post_stall_roll_moment_surface_coeff, dtype=float),
        post_stall_yaw_moment_surface_coeff=np.asarray(post_stall_yaw_moment_surface_coeff, dtype=float),
        surface_code=strip_table["surface_code"].astype(int),
    )
