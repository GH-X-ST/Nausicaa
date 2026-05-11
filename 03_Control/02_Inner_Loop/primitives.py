from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path bridge
# 2) Public primitive re-exports
# =============================================================================

# =============================================================================
# 1) Import Path Bridge
# =============================================================================
_PRIMITIVES = Path(__file__).resolve().parents[1] / "03_Primitives"
if str(_PRIMITIVES) not in sys.path:
    sys.path.insert(0, str(_PRIMITIVES))


# =============================================================================
# 2) Public Primitive Re-Exports
# =============================================================================
from primitive import (  # noqa: E402,F401
    EntryConditionResult,
    FlightPrimitive,
    PrimitiveContext,
    base_entry_conditions,
    build_primitive_context,
)
from templates import (  # noqa: E402,F401
    BankReversalPrimitive,
    NominalGlidePrimitive,
    RecoveryPrimitive,
)
