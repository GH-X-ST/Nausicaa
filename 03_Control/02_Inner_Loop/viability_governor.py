from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge
# 2) Public governor re-exports
# =============================================================================

# =============================================================================
# 1) Import Path Bridge
# =============================================================================
_CONTROL = Path(__file__).resolve().parents[1]
for rel in ("03_Primitives", "04_Scenarios"):
    path = _CONTROL / rel
    if str(path) not in sys.path:
        # Compatibility bridge exposes the primitive governor through the old inner-loop module.
        sys.path.insert(0, str(path))


# =============================================================================
# 2) Public Governor Re-Exports
# =============================================================================
# Re-export aliases keep governor imports stable across the inner-loop to primitives split.
from governor import (  # noqa: E402,F401
    GovernorDecision,
    GovernorLimits,
    ViabilityGovernor,
)
