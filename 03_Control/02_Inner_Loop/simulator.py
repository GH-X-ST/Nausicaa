from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge
# 2) Public rollout re-exports
# =============================================================================

# =============================================================================
# 1) Import Path Bridge
# =============================================================================
_CONTROL = Path(__file__).resolve().parents[1]
for rel in ("03_Primitives", "04_Scenarios"):
    path = _CONTROL / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =============================================================================
# 2) Public Rollout Re-Exports
# =============================================================================
from rollout import (  # noqa: E402,F401
    RolloutConfig as SimulationConfig,
    RolloutResult as SimulationResult,
    rk4_step,
    simulate_primitive,
    violation_reason,
    write_log as write_result_log,
)
