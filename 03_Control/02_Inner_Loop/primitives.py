from __future__ import annotations

import sys
from pathlib import Path


_PRIMITIVES = Path(__file__).resolve().parents[1] / "03_Primitives"
if str(_PRIMITIVES) not in sys.path:
    sys.path.insert(0, str(_PRIMITIVES))

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
