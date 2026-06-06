"""Active neutral dry-air and surface-effectiveness calibration constants.

The active model is the conservative 0.040 s local Pareto replay promotion from
the n30 neutral open-loop throw set plus the current-model pulse-ladder and
stage-wise control-surface effectiveness studies. It keeps the compact
replay-alignment family, promotes the stage-5 transition-blend longitudinal
base, adds the selected local yaw-beta plus post-stall roll-damping correction,
and applies a physics-informed conservative alpha-regime surface-effectiveness
schedule. This remains replay alignment and conservative surface-authority
regularisation rather than a full aerodynamic SysID claim. The comparison-only
theory baseline is archived as JSON next to this file and is not imported by
runtime code.
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
CALIBRATION_ID = "neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal"
SOURCE_PREP_RUN = "03_Control/05_Results/glider_model_calibration_prep/n30_joint_pareto_040_local_promising_v1"
SOURCE_HEAVY_JOINT_PARETO_RUN = "03_Control/05_Results/glider_model_calibration_prep/n30_joint_pareto_040_heavy_v1"
SOURCE_LOCAL_PARETO_RUN = "03_Control/05_Results/glider_model_calibration_prep/n30_joint_pareto_040_local_promising_v1"
SOURCE_CONTROL_SURFACE_EFFECTIVENESS_RUN = "03_Control/05_Results/control_surface_effectiveness/control_surface_effectiveness_v3_1_current_model_surface_refit"
SOURCE_STAGEWISE_CONTROL_SURFACE_EFFECTIVENESS_RUN = "03_Control/05_Results/control_surface_effectiveness/cse_v3_2_stage_fit"
SOURCE_CONSTRAINED_CONTROL_SURFACE_EFFECTIVENESS_RUN = "03_Control/05_Results/control_surface_effectiveness/cse_v3_3_constrained_stage_schedule"
SOURCE_SELECTED_CANDIDATE = "jp040local_L00_proposal_stage_5_stage5_tran_local_yaw_beta_s0p75_local_post_stall_Cl_r_s0p6"
SOURCE_SELECTED_LONGITUDINAL_BASE = "proposal_stage_5_stage5_transition_blend"
SOURCE_SELECTED_LATERAL_BUNDLE = "local_yaw_beta_s0p75+local_post_stall_Cl_r_s0p6"
SOURCE_SELECTED_SURFACE_SCALE_CANDIDATE = "physics_informed_conservative_alpha_regime_schedule"
SOURCE_THROW_COUNT = 105
SOURCE_FILTERED_THROW_COUNT = 9
HELDOUT_POLICY = "randomised_stratified_by_session_label"
HELDOUT_COUNT = 14
HELDOUT_SEED = 606
ALIGNMENT_WINDOW_S = 0.040
CLAIM_BOUNDARY = "compact_residual_calibrated_replay_alignment_with_40ms_local_pareto_transition_blend_yaw_beta_post_stall_clr_and_physics_informed_conservative_alpha_regime_surface_effectiveness_schedule"

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
# replay row promotes CY_beta; the 0.040 s local Pareto audit promotes the
# conservative attached yaw-beta correction from the selected S5/yaw0.75/Cl_r0.60
# candidate.
SIDE_FORCE_BIAS_COEFF = 0.0
SIDE_FORCE_BETA_COEFF = -1.9802669025044202
SIDE_FORCE_P_HAT_COEFF = 0.0
SIDE_FORCE_R_HAT_COEFF = 0.0
ROLL_MOMENT_BETA_COEFF = 0.0
ROLL_MOMENT_P_HAT_COEFF = 0.0
ROLL_MOMENT_R_HAT_COEFF = 0.0
YAW_MOMENT_BETA_COEFF = -0.034325897691662534
YAW_MOMENT_P_HAT_COEFF = 0.0
YAW_MOMENT_R_HAT_COEFF = 0.0
# Transition-only lateral deltas. They are multiplied by a bounded
# transition-window weight between residual blend start/full alpha. The active
# model includes the selected transition CY_r/Cn_p replay terms plus a tiny
# Cn_beta lateral replay correction from the frozen-baseline 105-throw check.
TRANSITION_SIDE_FORCE_BIAS_COEFF = 0.0
TRANSITION_SIDE_FORCE_BETA_COEFF = 0.0
TRANSITION_SIDE_FORCE_P_HAT_COEFF = 0.0
TRANSITION_SIDE_FORCE_R_HAT_COEFF = -3.0
TRANSITION_ROLL_MOMENT_BIAS_COEFF = 0.0
TRANSITION_ROLL_MOMENT_BETA_COEFF = 0.0
TRANSITION_ROLL_MOMENT_P_HAT_COEFF = 0.0
TRANSITION_ROLL_MOMENT_R_HAT_COEFF = 0.0
TRANSITION_YAW_MOMENT_BIAS_COEFF = 0.0
TRANSITION_YAW_MOMENT_BETA_COEFF = -0.01467582447
TRANSITION_YAW_MOMENT_P_HAT_COEFF = -0.1461483136170422
TRANSITION_YAW_MOMENT_R_HAT_COEFF = 0.0
# Compact post-stall longitudinal residual coefficients are separate from the
# attached-flow strip model. The residual blender is calibrated separately from
# coefficient values so transition evidence cannot move attached-flow terms.
POST_STALL_LIFT_RESIDUAL_COEFF = 0.0
POST_STALL_DRAG_RESIDUAL_COEFF = 0.0
POST_STALL_PITCH_MOMENT_COEFF = 0.07585874586245771
POST_STALL_PITCH_DAMPING_COEFF = 4.0
POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG = 14.0
POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG = 18.0
# Compact neutral SysID residual-surface basis. The active model includes only
# the inherited compact post-stall Cn terms plus the selected local post-stall
# Cl_r 20 deg correction from the conservative 0.040 s Pareto promotion.
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
    (-0.46003383312128726, 0.0, 0.0),
)
POST_STALL_YAW_MOMENT_RBF_COEFFS = (
    (0.0, 0.0, 0.0),
    (-0.01870860145, 0.0, 0.0),
    (-0.07119920085255141, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
DELTA_A_TRIM_RAD = 0.0
DELTA_E_TRIM_RAD = 0.0
DELTA_R_TRIM_RAD = 0.0
# Control-surface pulse-ladder and stage-schedule update. These scale
# aerodynamic effectiveness in the strip model, not the measured/commanded
# physical surface angle. Command conversion, hardware mapping, servo signs, and
# neutral trims stay unchanged. The v3.1 scalar replay gate remains historical
# evidence; the active nominal model below uses a conservative alpha-regime
# schedule informed by v3.2/v3.3 replay diagnostics and high-AoA literature.
CONTROL_SURFACE_EFFECTIVENESS_MODEL = "alpha_regime_scheduled_v1"
CONTROL_SURFACE_EFFECTIVENESS_MODEL_STATUS = (
    "physics_informed_replay_regularized_conservative_nominal_not_exact_derivative_sysid"
)
CONTROL_SURFACE_EFFECTIVENESS_REGIMES = ("normal", "transition", "post_stall")
CONTROL_SURFACE_EFFECTIVENESS_AXES = ("aileron", "elevator", "rudder")
CONTROL_SURFACE_EFFECTIVENESS_SCHEDULE = (
    (0.85, 0.75, 0.85),
    (0.55, 0.55, 0.55),
    (0.45, 0.45, 0.40),
)
# Retained for backward-compatible diagnostics that read the v3.1 scalar result.
# Runtime code uses CONTROL_SURFACE_EFFECTIVENESS_SCHEDULE instead.
DELTA_A_AERO_EFFECTIVENESS_SCALE = 0.65
DELTA_E_AERO_EFFECTIVENESS_SCALE = 0.70
DELTA_R_AERO_EFFECTIVENESS_SCALE = 0.45
