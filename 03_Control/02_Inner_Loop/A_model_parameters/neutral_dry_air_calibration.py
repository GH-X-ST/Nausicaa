"""Neutral dry-air calibration constants from real open-loop launch replay.

The constants below come from the neutral_30 open-loop throws only, using a
0.20 s first-motion alignment window. They deliberately change only simple
bare-airframe loss terms before any pulse/control-effectiveness fitting.
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
# Fitted from aligned replay with static neutral trims disabled. Free trim
# fitting improved training residuals but worsened held-out lateral error, so
# trims remain zero until physical trim/asymmetry is confirmed separately.
CD0_STRIP_SCALE = 3.0
DRAG_AREA_FUSE_SCALE = 5.0
EFFICIENCY_STRIP_SCALE = 0.31
DELTA_A_TRIM_RAD = 0.0
DELTA_E_TRIM_RAD = 0.0
DELTA_R_TRIM_RAD = 0.0
