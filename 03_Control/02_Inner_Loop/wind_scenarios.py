from __future__ import annotations

import sys
from pathlib import Path


_SCENARIOS = Path(__file__).resolve().parents[1] / "04_Scenarios"
if str(_SCENARIOS) not in sys.path:
    sys.path.insert(0, str(_SCENARIOS))

from updraft_models import (  # noqa: E402,F401
    AnalyticDebugProxy,
    AnnularGPGridWindField,
    GaussianVarWindField,
    WindField,
    load_updraft_model,
)
