from __future__ import annotations

import sys
from pathlib import Path


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTROL_ROOT.parents[0]
for rel in (
    "02_Inner_Loop",
    "03_Primitives",
    "04_Scenarios",
):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
