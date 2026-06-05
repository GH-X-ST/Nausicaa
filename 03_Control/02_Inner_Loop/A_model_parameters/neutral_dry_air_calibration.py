"""Active neutral dry-air calibration constants for simulation and real flight.

The active model is the selected compact residual-calibrated replay alignment
promoted from the n30 neutral open-loop throw set. It includes the longitudinal
terms plus the compact coupling terms from the selected replay row; this is
still replay alignment rather than a full aerodynamic SysID claim. The
comparison-only theory baseline is archived as JSON next to this file and is
not imported by runtime code.
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
CALIBRATION_ID = "neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_v1"
SOURCE_PREP_RUN = "03_Control/05_Results/glider_model_calibration_prep/n30_compact_joint_sweep_from_active_v1"
SOURCE_DIAGNOSTIC_RUN = "03_Control/05_Results/glider_model_calibration_prep/n30_cmq_wide_diagnostic_v1"
SOURCE_SELECTED_CANDIDATE = "joint_0270_post_stall_Cn_p_1.5"
SOURCE_THROW_COUNT = 105
SOURCE_FILTERED_THROW_COUNT = 74
HELDOUT_POLICY = "randomised_stratified_by_session_label"
HELDOUT_COUNT = 11
HELDOUT_SEED = 606
CLAIM_BOUNDARY = "compact_residual_calibrated_replay_alignment_with_selected_coupling_terms"

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
ATTACHED_PITCH_MOMENT_BIAS_COEFF = 0.11309832420327923
TRANSITION_PITCH_MOMENT_BIAS_COEFF = 0.05711558897899738
YAW_MOMENT_BIAS_COEFF = 0.0
# Attached-flow lateral-directional residual derivatives. The selected compact
# replay row promotes CY_beta only; the remaining attached terms stay at zero.
SIDE_FORCE_BIAS_COEFF = 0.0
SIDE_FORCE_BETA_COEFF = -1.9802669025044202
SIDE_FORCE_P_HAT_COEFF = 0.0
SIDE_FORCE_R_HAT_COEFF = 0.0
ROLL_MOMENT_BETA_COEFF = 0.0
ROLL_MOMENT_P_HAT_COEFF = 0.0
ROLL_MOMENT_R_HAT_COEFF = 0.0
YAW_MOMENT_BETA_COEFF = 0.0
YAW_MOMENT_P_HAT_COEFF = 0.0
YAW_MOMENT_R_HAT_COEFF = 0.0
# Transition-only lateral deltas. They are multiplied by a bounded
# transition-window weight between residual blend start/full alpha. The active
# model includes the selected transition CY_r and Cn_p replay terms only.
TRANSITION_SIDE_FORCE_BIAS_COEFF = 0.0
TRANSITION_SIDE_FORCE_BETA_COEFF = 0.0
TRANSITION_SIDE_FORCE_P_HAT_COEFF = 0.0
TRANSITION_SIDE_FORCE_R_HAT_COEFF = -3.0
TRANSITION_ROLL_MOMENT_BIAS_COEFF = 0.0
TRANSITION_ROLL_MOMENT_BETA_COEFF = 0.0
TRANSITION_ROLL_MOMENT_P_HAT_COEFF = 0.0
TRANSITION_ROLL_MOMENT_R_HAT_COEFF = 0.0
TRANSITION_YAW_MOMENT_BIAS_COEFF = 0.0
TRANSITION_YAW_MOMENT_BETA_COEFF = 0.0
TRANSITION_YAW_MOMENT_P_HAT_COEFF = -0.1461483136170422
TRANSITION_YAW_MOMENT_R_HAT_COEFF = 0.0
# Compact post-stall longitudinal residual coefficients are separate from the
# attached-flow strip model. The residual blender is calibrated separately from
# coefficient values so transition evidence cannot move attached-flow terms.
POST_STALL_LIFT_RESIDUAL_COEFF = 0.0
POST_STALL_DRAG_RESIDUAL_COEFF = 0.0
POST_STALL_PITCH_MOMENT_COEFF = 0.07585874586245771
POST_STALL_PITCH_DAMPING_COEFF = 4.0
POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG = 12.0
POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG = 22.0
# Compact neutral SysID residual-surface basis. The active model includes only
# the selected post-stall Cn_p 20 deg term from the compact replay row.
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
    (-0.07119920085255141, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
DELTA_A_TRIM_RAD = 0.0
DELTA_E_TRIM_RAD = 0.0
DELTA_R_TRIM_RAD = 0.0
