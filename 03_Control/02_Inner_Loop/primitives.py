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
    # Compatibility bridge preserves older inner-loop primitive imports.
    sys.path.insert(0, str(_PRIMITIVES))


# =============================================================================
# 2) Public Primitive Re-Exports
# =============================================================================
# Re-export aliases preserve the pre-refactor primitive import path used by older scripts.
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
