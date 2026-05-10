from __future__ import annotations

import sys
from pathlib import Path


_SCENARIOS = Path(__file__).resolve().parents[1] / "04_Scenarios"
if str(_SCENARIOS) not in sys.path:
    sys.path.insert(0, str(_SCENARIOS))

from run_batch import main  # noqa: E402


if __name__ == "__main__":
    main()
