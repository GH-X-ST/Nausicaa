from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glide_primitive import build_glide_primitive_spec
from primitive_contract import (
    PrimitiveEntrySet,
    PrimitiveExitCheck,
    PrimitiveSpec,
    validate_primitive_spec,
)
from primitive_interface import evaluate_entry_set
from recovery_primitive import build_recovery_primitive_spec
from state_contract import as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants
# 2) Primitive Metadata
# 3) Terminal Handoff Proxy
# =============================================================================


# =============================================================================
# 1) Constants
# =============================================================================
PRIMITIVE_FAMILY = "agile_reversal"
AGGRESSIVE_PROXY_SOURCE = (
    "build_glide_primitive_spec/build_recovery_primitive_spec + evaluate_entry_set"
)


# =============================================================================
# 2) Primitive Metadata
# =============================================================================
def build_aggressive_reversal_primitive_spec(config: object) -> PrimitiveSpec:
    """Return metadata and entry/exit checks for the aggressive reversal candidate."""

    duration_s = float(getattr(config, "t_final_s"))
    entry_set = PrimitiveEntrySet(
        name="aggressive_reversal_w0_entry",
        description="W0 aggressive reversal entry set in SI units and radians",
        lower={
            "x_w": 1.2,
            "y_w": 0.3,
            "z_w": 0.8,
            "speed_m_s": 5.0,
            "alpha_rad": -0.25,
            "beta_rad": -0.25,
            "phi": -0.60,
            "theta": -0.45,
            "p": -2.0,
            "q": -2.0,
            "r": -2.0,
        },
        upper={
            "x_w": 1.45,
            "y_w": 4.1,
            "z_w": 2.6,
            "speed_m_s": 7.8,
            "alpha_rad": 0.45,
            "beta_rad": 0.30,
            "phi": 0.60,
            "theta": 0.60,
            "p": 2.0,
            "q": 2.0,
            "r": 2.0,
        },
    )
    exit_checks = (
        PrimitiveExitCheck(
            name="finite_state",
            description="full aggressive reversal state history remains finite",
            required=True,
        ),
        PrimitiveExitCheck(
            name="true_safe_margin",
            description="minimum true-safety margin stays nonnegative",
            required=True,
        ),
        PrimitiveExitCheck(
            name="rollout_success",
            description="plant replay remains finite and inside true safety volume",
            required=True,
        ),
    )
    spec = PrimitiveSpec(
        name="aggressive_reversal_w0",
        family=PRIMITIVE_FAMILY,
        duration_s=duration_s,
        entry_set=entry_set,
        exit_checks=exit_checks,
        metadata={
            "simulation_boundary_branch": "true",
            "terminal_recoverable_proxy_source": AGGRESSIVE_PROXY_SOURCE,
            "high_incidence_validation_claim": "false",
            "real_flight_validation_claim": "false",
            "updraft_validation_claim": "false",
        },
    )
    validate_primitive_spec(spec)
    return spec


# =============================================================================
# 3) Terminal Handoff Proxy
# =============================================================================
def _entry_passes(spec: PrimitiveSpec, state: np.ndarray) -> bool:
    try:
        checks = evaluate_entry_set(state, spec.entry_set)
    except ValueError:
        return False
    return bool(all(check.pass_check for check in checks))


def terminal_aggressive_recoverable_proxy(x_terminal: np.ndarray) -> bool:
    """Return True if terminal state is compatible with glide or recovery handoff."""

    try:
        state = as_state_vector(x_terminal)
    except ValueError:
        return False
    if not np.all(np.isfinite(state)):
        return False
    return bool(
        _entry_passes(build_glide_primitive_spec(), state)
        or _entry_passes(build_recovery_primitive_spec(), state)
    )
