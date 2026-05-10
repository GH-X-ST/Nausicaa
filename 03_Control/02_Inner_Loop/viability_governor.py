from __future__ import annotations

import sys
from pathlib import Path


_CONTROL = Path(__file__).resolve().parents[1]
for rel in ("03_Primitives", "04_Scenarios"):
    path = _CONTROL / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from governor import (  # noqa: E402,F401
    GovernorDecision,
    GovernorLimits,
    ViabilityGovernor,
)
