"""Neutral dry-air calibration constants from real open-loop launch replay.

The active constants below remain the older accepted N07 neutral fit. Newer
staged 0.10 s replay diagnostics can fit simple loss scales plus Cm0/Cl0/Cn0
moment-bias terms, but the latest rich n30 run did not pass the acceptance gate,
so those candidates are not promoted here.
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
PITCH_MOMENT_BIAS_COEFF = 0.0
YAW_MOMENT_BIAS_COEFF = 0.0
# Post-stall residual coefficients are separate from the attached-flow strip
# model. They are smoothly activated across the 12-20 deg transition band.
POST_STALL_LIFT_RESIDUAL_COEFF = 0.0
POST_STALL_DRAG_RESIDUAL_COEFF = 0.0
POST_STALL_PITCH_MOMENT_COEFF = 0.0
POST_STALL_PITCH_DAMPING_COEFF = 0.0
DELTA_A_TRIM_RAD = 0.0
DELTA_E_TRIM_RAD = 0.0
DELTA_R_TRIM_RAD = 0.0
