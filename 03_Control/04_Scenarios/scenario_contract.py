from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Scenario contract dataclass
# 2) Scenario validation
# 3) Scenario audit rows
# =============================================================================


# =============================================================================
# 1) Scenario Contract Dataclass
# =============================================================================
# Scenario metadata records wind and latency choices only; wind fields and
# rollout logic are intentionally outside this contract layer.
WindMode = Literal["none", "cg", "panel"]
LatencyCase = Literal["none", "actuator_lag_only", "nominal", "conservative"]

WIND_MODES = ("none", "cg", "panel")
LATENCY_CASES = ("none", "actuator_lag_only", "nominal", "conservative")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    wind_mode: WindMode
    latency_case: LatencyCase
    dt_s: float
    t_final_s: float
    seed: int
    use_true_safe_bounds: bool
    description: str


# =============================================================================
# 2) Scenario Validation
# =============================================================================
def validate_scenario_spec(spec: ScenarioSpec) -> None:
    """Check finite timing, allowed wind mode, allowed latency case, and seed."""

    if not spec.name:
        raise ValueError("scenario spec must have a nonempty name.")
    if spec.wind_mode not in WIND_MODES:
        raise ValueError(f"unknown wind_mode: {spec.wind_mode}.")
    if spec.latency_case not in LATENCY_CASES:
        raise ValueError(f"unknown latency_case: {spec.latency_case}.")
    if not np.isfinite(float(spec.dt_s)) or float(spec.dt_s) <= 0.0:
        raise ValueError("scenario dt_s must be finite and positive.")
    if not np.isfinite(float(spec.t_final_s)) or float(spec.t_final_s) <= 0.0:
        raise ValueError("scenario t_final_s must be finite and positive.")
    if float(spec.t_final_s) < float(spec.dt_s):
        raise ValueError("scenario t_final_s must be at least dt_s.")
    if not isinstance(spec.seed, int) or spec.seed < 0:
        raise ValueError("scenario seed must be a nonnegative integer.")
    if not isinstance(spec.use_true_safe_bounds, bool):
        raise ValueError("scenario use_true_safe_bounds must be a bool.")
    if not spec.description:
        raise ValueError("scenario spec must have a nonempty description.")


# =============================================================================
# 3) Scenario Audit Rows
# =============================================================================
def scenario_spec_row(spec: ScenarioSpec) -> dict[str, object]:
    """Return a CSV-ready scenario row."""

    validate_scenario_spec(spec)
    return {
        "scenario_name": spec.name,
        "wind_mode": spec.wind_mode,
        "latency_case": spec.latency_case,
        "dt_s": float(spec.dt_s),
        "t_final_s": float(spec.t_final_s),
        "seed": int(spec.seed),
        "use_true_safe_bounds": bool(spec.use_true_safe_bounds),
        "description": spec.description,
    }
