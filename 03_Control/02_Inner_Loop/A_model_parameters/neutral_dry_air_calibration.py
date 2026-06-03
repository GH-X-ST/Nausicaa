"""Neutral dry-air calibration constants from measured real-launch replay.

The constants below are a first grey-box correction from the 80 valid neutral
throws collected across five launch sessions. They deliberately change only
simple aerodynamic loss terms before any pulse/control-effectiveness fitting.
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
CALIBRATION_ID = "neutral_dry_air_measured_launch_replay_N04"
SOURCE_PREP_RUN = "03_Control/05_Results/glider_model_calibration_prep/N03_neutral_80_measured_launch_replay"
SOURCE_THROW_COUNT = 80
HELDOUT_POLICY = "randomised_stratified_by_session_label"
HELDOUT_SEED = 604

# =============================================================================
# 2) Active Aerodynamic Correction
# =============================================================================
# The real neutral glider sank faster and travelled less far than the symmetric
# baseline when replayed from the exact measured launch states. These factors
# increase profile/lumped drag and modestly reduce induced-drag efficiency.
CD0_STRIP_SCALE = 6.0
DRAG_AREA_FUSE_SCALE = 10.0
EFFICIENCY_STRIP_SCALE = 0.75

