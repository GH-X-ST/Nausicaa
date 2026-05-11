from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge
# 2) Public implementation-wrapper re-exports
# =============================================================================

# =============================================================================
# 1) Import Path Bridge
# =============================================================================
_PRIMITIVES = Path(__file__).resolve().parents[1] / "03_Primitives"
if str(_PRIMITIVES) not in sys.path:
    # Compatibility bridge preserves older inner-loop imports after the controller layout split.
    sys.path.insert(0, str(_PRIMITIVES))


# =============================================================================
# 2) Public Implementation-Wrapper Re-Exports
# =============================================================================
# Re-export aliases keep older inner-loop imports stable after latency moved to primitives.
from latency import (  # noqa: E402,F401
    COMMAND_LEVELS,
    CommandToSurfaceConfig as ImplementationWrapperConfig,
    CommandToSurfaceLayer as ImplementationCommandWrapper,
    LatencyEnvelope,
    SurfaceLimit,
    actuator_tau_s,
    aggregate_targets_to_surface_degrees,
    angle_to_command_norm,
    command_norm_to_angle,
    feedback_delay_s,
    half_response_s,
    quantise_command_norm,
)
