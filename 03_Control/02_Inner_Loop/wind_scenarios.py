from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge
# 2) Public updraft-model re-exports
# =============================================================================

# =============================================================================
# 1) Import Path Bridge
# =============================================================================
_SCENARIOS = Path(__file__).resolve().parents[1] / "04_Scenarios"
if str(_SCENARIOS) not in sys.path:
    # Compatibility bridge keeps wind/updraft imports stable after scenario refactor.
    sys.path.insert(0, str(_SCENARIOS))


# =============================================================================
# 2) Public Updraft-Model Re-Exports
# =============================================================================
from updraft_models import (  # noqa: E402,F401
    AnalyticDebugProxy,
    AnnularGPGridWindField,
    GaussianVarWindField,
    WindField,
    load_updraft_model,
)
