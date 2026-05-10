from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "03_Control/02_Inner_Loop",
    "03_Control/03_Primitives",
    "03_Control/04_Scenarios",
):
    path = REPO_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
