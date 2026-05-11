from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge
# 2) Batch-runner entry point
# =============================================================================

# =============================================================================
# 1) Import Path Bridge
# =============================================================================
_SCENARIOS = Path(__file__).resolve().parents[1] / "04_Scenarios"
if str(_SCENARIOS) not in sys.path:
    # Legacy validation entry point delegates to the scenario batch runner.
    sys.path.insert(0, str(_SCENARIOS))


# =============================================================================
# 2) Batch-Runner Entry Point
# =============================================================================
# The legacy validation command intentionally shares the batch runner implementation.
from run_batch import main  # noqa: E402


if __name__ == "__main__":
    main()
