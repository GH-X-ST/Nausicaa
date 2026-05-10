from __future__ import annotations

import sys
from pathlib import Path


_PRIMITIVES = Path(__file__).resolve().parents[1] / "03_Primitives"
if str(_PRIMITIVES) not in sys.path:
    sys.path.insert(0, str(_PRIMITIVES))

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
    half_response_s,
    quantise_command_norm,
)
