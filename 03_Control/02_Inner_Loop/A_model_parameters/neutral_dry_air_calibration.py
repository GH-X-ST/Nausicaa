"""Neutral dry-air calibration constants from real open-loop launch replay.

The active constants below remain the older accepted N07 neutral fit. Newer
staged 0.10 s replay diagnostics now fit regime-separated longitudinal pitch
moment terms before compact post-stall lift/drag cleanup. Lateral terms remain
zero by default and are only claim-bearing if a separate held-out diagnostic
gate accepts them.
"""

from __future__ import annotations


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Calibration metadata
# 2) Active aerodynamic correction
# =============================================================================

# =============================================================================
# 1) Calibration Metadata
# =============================================================================
CALIBRATION_ACTIVE = True
CALIBRATION_ID = "neutral_dry_air_aligned_0p20_N07"
SOURCE_PREP_RUN = "03_Control/05_Results/glider_model_calibration_prep/N07_neutral_aligned_0p20_longitudinal_fit"
SOURCE_THROW_COUNT = 80
HELDOUT_POLICY = "randomised_stratified_by_session_label"
HELDOUT_SEED = 606

# =============================================================================
# 2) Active Aerodynamic Correction
# =============================================================================
# Fitted from aligned replay with static neutral trims disabled. Surface trims
# remain zero unless a physical servo/surface zero offset is measured. As-built
# moment asymmetry should first enter through explicit aerodynamic moment-bias
# coefficients.
CD0_STRIP_SCALE = 3.0
DRAG_AREA_FUSE_SCALE = 5.0
EFFICIENCY_STRIP_SCALE = 0.31
ROLL_MOMENT_BIAS_COEFF = 0.0
# Legacy global Cm offset. Kept for compatibility with older candidate files;
# the current residual SysID uses the regime-separated pitch terms below.
PITCH_MOMENT_BIAS_COEFF = 0.0
ATTACHED_PITCH_MOMENT_BIAS_COEFF = 0.0
TRANSITION_PITCH_MOMENT_BIAS_COEFF = 0.0
YAW_MOMENT_BIAS_COEFF = 0.0
# Attached-flow lateral-directional residual derivatives. They are kept as
# explicit compatibility/diagnostic fields, but the default neutral SysID claim
# path leaves them frozen at zero.
SIDE_FORCE_BIAS_COEFF = 0.0
SIDE_FORCE_BETA_COEFF = 0.0
SIDE_FORCE_P_HAT_COEFF = 0.0
SIDE_FORCE_R_HAT_COEFF = 0.0
ROLL_MOMENT_BETA_COEFF = 0.0
ROLL_MOMENT_P_HAT_COEFF = 0.0
ROLL_MOMENT_R_HAT_COEFF = 0.0
YAW_MOMENT_BETA_COEFF = 0.0
YAW_MOMENT_P_HAT_COEFF = 0.0
YAW_MOMENT_R_HAT_COEFF = 0.0
# Transition-only lateral deltas. They are multiplied by a bounded
# transition-window weight between residual blend start/full alpha, but remain
# disabled unless a diagnostic run explicitly enables them.
TRANSITION_SIDE_FORCE_BIAS_COEFF = 0.0
TRANSITION_SIDE_FORCE_BETA_COEFF = 0.0
TRANSITION_SIDE_FORCE_P_HAT_COEFF = 0.0
TRANSITION_SIDE_FORCE_R_HAT_COEFF = 0.0
TRANSITION_ROLL_MOMENT_BIAS_COEFF = 0.0
TRANSITION_ROLL_MOMENT_BETA_COEFF = 0.0
TRANSITION_ROLL_MOMENT_P_HAT_COEFF = 0.0
TRANSITION_ROLL_MOMENT_R_HAT_COEFF = 0.0
TRANSITION_YAW_MOMENT_BIAS_COEFF = 0.0
TRANSITION_YAW_MOMENT_BETA_COEFF = 0.0
TRANSITION_YAW_MOMENT_P_HAT_COEFF = 0.0
TRANSITION_YAW_MOMENT_R_HAT_COEFF = 0.0
# Compact post-stall longitudinal residual coefficients are separate from the
# attached-flow strip model. The residual blender is calibrated separately from
# coefficient values so transition evidence cannot move attached-flow terms.
POST_STALL_LIFT_RESIDUAL_COEFF = 0.0
POST_STALL_DRAG_RESIDUAL_COEFF = 0.0
POST_STALL_PITCH_MOMENT_COEFF = 0.0
POST_STALL_PITCH_DAMPING_COEFF = 0.0
POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG = 12.0
POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG = 20.0
# Compact neutral SysID residual-surface basis. These coefficients are kept at
# zero in the active calibration until held-out replay passes the promotion gate.
POST_STALL_RBF_ALPHA_CENTERS_DEG = (20.0, 45.0, 70.0)
POST_STALL_RBF_ALPHA_WIDTH_DEG = 15.0
POST_STALL_LIFT_RBF_COEFFS = (0.0, 0.0, 0.0)
POST_STALL_DRAG_RBF_COEFFS = (0.0, 0.0, 0.0)
POST_STALL_PITCH_MOMENT_RBF_COEFFS = (0.0, 0.0, 0.0)
POST_STALL_PITCH_DAMPING_RBF_COEFFS = (0.0, 0.0, 0.0)
# Rows are bias, beta, p_hat, r_hat; columns are the alpha RBF centres above.
POST_STALL_SIDE_FORCE_RBF_COEFFS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
POST_STALL_ROLL_MOMENT_RBF_COEFFS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
POST_STALL_YAW_MOMENT_RBF_COEFFS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
DELTA_A_TRIM_RAD = 0.0
DELTA_E_TRIM_RAD = 0.0
DELTA_R_TRIM_RAD = 0.0
