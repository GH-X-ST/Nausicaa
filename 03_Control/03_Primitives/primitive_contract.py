from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Primitive contract dataclasses
# 2) Primitive validation
# 3) Primitive audit rows
# =============================================================================


# =============================================================================
# 1) Primitive Contract Dataclasses
# =============================================================================
# These contracts describe primitive metadata only; they do not implement
# primitive controllers, trajectories, rollout, or storage.
PrimitiveFamily = Literal[
    "glide",
    "bank",
    "recovery",
    "agile_reversal",
]


@dataclass(frozen=True)
class PrimitiveEntrySet:
    name: str
    description: str
    lower: dict[str, float]
    upper: dict[str, float]


@dataclass(frozen=True)
class PrimitiveExitCheck:
    name: str
    description: str
    required: bool


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    family: PrimitiveFamily
    duration_s: float
    entry_set: PrimitiveEntrySet
    exit_checks: tuple[PrimitiveExitCheck, ...]
    metadata: dict[str, str]


# =============================================================================
# 2) Primitive Validation
# =============================================================================
def allowed_primitive_families() -> tuple[str, ...]:
    """Return all mandatory primitive families."""

    return ("glide", "bank", "recovery", "agile_reversal")


def _validate_entry_set(entry_set: PrimitiveEntrySet) -> None:
    if not entry_set.name:
        raise ValueError("primitive entry set must have a nonempty name.")
    if not entry_set.description:
        raise ValueError("primitive entry set must have a nonempty description.")
    if not entry_set.lower or not entry_set.upper:
        raise ValueError("primitive entry set must have nonempty lower and upper bounds.")
    if set(entry_set.lower) != set(entry_set.upper):
        raise ValueError("primitive entry set lower and upper keys must match.")
    for key in entry_set.lower:
        lower = float(entry_set.lower[key])
        upper = float(entry_set.upper[key])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("primitive entry set bounds must be finite.")
        if lower > upper:
            raise ValueError(f"primitive entry set lower bound exceeds upper bound for {key}.")


def validate_primitive_spec(spec: PrimitiveSpec) -> None:
    """Check finite duration, known family, and nonempty entry and exit definitions."""

    if not spec.name:
        raise ValueError("primitive spec must have a nonempty name.")
    if spec.family not in allowed_primitive_families():
        raise ValueError(f"unknown primitive family: {spec.family}.")
    if not np.isfinite(float(spec.duration_s)) or float(spec.duration_s) <= 0.0:
        raise ValueError("primitive duration_s must be finite and positive.")
    _validate_entry_set(spec.entry_set)
    if not spec.exit_checks:
        raise ValueError("primitive spec must define at least one exit check.")
    for check in spec.exit_checks:
        if not check.name or not check.description:
            raise ValueError("primitive exit checks must have names and descriptions.")


# =============================================================================
# 3) Primitive Audit Rows
# =============================================================================
def primitive_spec_row(spec: PrimitiveSpec) -> dict[str, object]:
    """Return a CSV-ready primitive specification row."""

    validate_primitive_spec(spec)
    return {
        "primitive_name": spec.name,
        "primitive_family": spec.family,
        "duration_s": float(spec.duration_s),
        "entry_set_name": spec.entry_set.name,
        "entry_variables": ",".join(sorted(spec.entry_set.lower.keys())),
        "required_exit_checks": ",".join(
            check.name for check in spec.exit_checks if check.required
        ),
        "optional_exit_checks": ",".join(
            check.name for check in spec.exit_checks if not check.required
        ),
        "metadata": str(spec.metadata),
    }
