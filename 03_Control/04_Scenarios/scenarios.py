from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency import CommandToSurfaceConfig
from linearisation import STATE_INDEX
from templates import (
    BankReversalPrimitive,
    NominalGlidePrimitive,
    RecoveryPrimitive,
)
from updraft_models import WindField, build_randomised_wind_field, load_updraft_model


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Scenario dataclass
# 2) Entry-state builders
# 3) Scenario factory
# 4) Scenario lists
# 5) Updraft loader fallback
# =============================================================================

# =============================================================================
# 1) Scenario Dataclass
# =============================================================================
@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    primitive: object
    wind_model_name: str
    wind_model: WindField | None
    wind_mode: str
    latency_config: CommandToSurfaceConfig
    x0: np.ndarray
    wind_param_label: str = ""
    candidate_primitives: tuple[object, ...] = ()


# =============================================================================
# 2) Entry-State Builders
# =============================================================================
def trimmed_state_for_arena(x_trim: np.ndarray, altitude_m: float = 2.7) -> np.ndarray:
    x0 = np.asarray(x_trim, dtype=float).reshape(15).copy()
    x0[STATE_INDEX["x_w"]] = 4.2
    x0[STATE_INDEX["y_w"]] = 2.4
    x0[STATE_INDEX["z_w"]] = float(altitude_m)
    return x0


def arena_feasible_entry_state(x_trim: np.ndarray, altitude_m: float = 2.7) -> np.ndarray:
    x0 = trimmed_state_for_arena(x_trim, altitude_m=altitude_m)
    # Indoor full-duration cases start near the upstream safe-volume edge
    x0[STATE_INDEX["x_w"]] = 1.45
    return x0


def recovery_entry_state(x_trim: np.ndarray) -> np.ndarray:
    x0 = trimmed_state_for_arena(x_trim, altitude_m=2.7)
    # Recovery scenarios start from a small attitude and rate disturbance
    x0[STATE_INDEX["phi"]] = np.deg2rad(12.0)
    x0[STATE_INDEX["theta"]] += np.deg2rad(3.0)
    x0[STATE_INDEX["p"]] = np.deg2rad(-2.0)
    x0[STATE_INDEX["q"]] = np.deg2rad(1.0)
    x0[STATE_INDEX["r"]] = np.deg2rad(1.0)
    return x0


# =============================================================================
# 3) Scenario Factory
# =============================================================================
def build_scenario(
    scenario_id: str,
    x_trim: np.ndarray,
    repo_root,
    seed: int = 1,
) -> ScenarioDefinition:
    base = trimmed_state_for_arena(x_trim)
    full_base = arena_feasible_entry_state(x_trim)
    short_glide = NominalGlidePrimitive(duration_s=0.20)
    short_bank = BankReversalPrimitive(duration_s=0.35, bank_angle_rad=np.deg2rad(8.0))
    short_recovery = RecoveryPrimitive(duration_s=0.30)
    full_glide = NominalGlidePrimitive(duration_s=0.85)
    full_bank_left = BankReversalPrimitive(duration_s=0.85, bank_angle_rad=np.deg2rad(10.0))
    full_bank_right = BankReversalPrimitive(duration_s=0.85, bank_angle_rad=np.deg2rad(-10.0))
    full_recovery = RecoveryPrimitive(duration_s=0.85)
    # Measured updraft stress horizons stay inside the indoor safety envelope
    updraft_glide = NominalGlidePrimitive(duration_s=0.34)
    four_fan_updraft_glide = NominalGlidePrimitive(duration_s=0.26)
    governor_recovery = RecoveryPrimitive(duration_s=0.50)
    nominal = CommandToSurfaceConfig(mode="nominal")
    low = CommandToSurfaceConfig(mode="low")
    high = CommandToSurfaceConfig(mode="high")
    robust = CommandToSurfaceConfig(mode="robust_upper")

    if scenario_id == "s0_no_wind":
        return ScenarioDefinition(scenario_id, short_glide, "none", None, "panel", nominal, base)
    if scenario_id == "s1_latency_nominal_no_wind":
        return ScenarioDefinition(scenario_id, short_bank, "none", None, "panel", nominal, base)
    if scenario_id == "s1_latency_robust_upper_no_wind":
        return ScenarioDefinition(scenario_id, short_bank, "none", None, "panel", robust, base)
    if scenario_id == "s6_single_gaussian_var":
        wind = load_updraft_model("single_gaussian_var", repo_root=repo_root)
        return ScenarioDefinition(scenario_id, short_glide, wind.name, wind, "panel", nominal, base)
    if scenario_id == "s6_four_gaussian_var":
        wind = load_updraft_model("four_gaussian_var", repo_root=repo_root)
        return ScenarioDefinition(scenario_id, short_glide, wind.name, wind, "panel", nominal, base)
    if scenario_id == "s7_single_annular_gp":
        wind = load_updraft_model("single_annular_gp_grid", repo_root=repo_root)
        return ScenarioDefinition(scenario_id, short_glide, wind.name, wind, "panel", nominal, base)
    if scenario_id == "s7_four_annular_gp":
        wind = load_updraft_model("four_annular_gp_grid", repo_root=repo_root)
        return ScenarioDefinition(scenario_id, short_glide, wind.name, wind, "panel", nominal, base)
    if scenario_id == "s11_governor_rejection":
        low = trimmed_state_for_arena(x_trim, altitude_m=0.10)
        return ScenarioDefinition(scenario_id, short_recovery, "none", None, "panel", nominal, low)

    full_no_wind = {
        "s4_full_nominal_glide_no_wind": full_glide,
        "s4_full_bank_reversal_left_no_wind": full_bank_left,
        "s4_full_bank_reversal_right_no_wind": full_bank_right,
        "s4_full_recovery_no_wind": full_recovery,
    }
    if scenario_id in full_no_wind:
        x0 = recovery_entry_state(x_trim) if "recovery" in scenario_id else full_base
        x0 = x0.copy()
        x0[STATE_INDEX["x_w"]] = 1.45
        return ScenarioDefinition(
            scenario_id,
            full_no_wind[scenario_id],
            "none",
            None,
            "panel",
            nominal,
            x0,
            "none",
        )

    latency_cases = {
        "s4_latency_low_nominal_glide": (full_glide, low, full_base),
        "s4_latency_nominal_nominal_glide": (full_glide, nominal, full_base),
        "s4_latency_high_nominal_glide": (full_glide, high, full_base),
        "s4_latency_low_bank_reversal_left": (full_bank_left, low, full_base),
        "s4_latency_nominal_bank_reversal_left": (full_bank_left, nominal, full_base),
        "s4_latency_high_bank_reversal_left": (full_bank_left, high, full_base),
        "s4_latency_low_recovery": (full_recovery, low, recovery_entry_state(x_trim)),
        "s4_latency_nominal_recovery": (full_recovery, nominal, recovery_entry_state(x_trim)),
        "s4_latency_high_recovery": (full_recovery, high, recovery_entry_state(x_trim)),
        "s4_latency_robust_upper_bank_reversal_left": (full_bank_left, robust, full_base),
    }
    if scenario_id in latency_cases:
        primitive, latency_config, x0 = latency_cases[scenario_id]
        x0 = x0.copy()
        x0[STATE_INDEX["x_w"]] = 1.45
        return ScenarioDefinition(
            scenario_id,
            primitive,
            "none",
            None,
            "panel",
            latency_config,
            x0,
            "none",
        )

    updraft_cases = {
        "s4_gaussian_single_panel": ("single_gaussian_var", "panel"),
        "s4_gaussian_single_cg": ("single_gaussian_var", "cg"),
        "s4_gaussian_four_panel": ("four_gaussian_var", "panel"),
        "s4_annular_single_panel": ("single_annular_gp_grid", "panel"),
        "s4_annular_single_cg": ("single_annular_gp_grid", "cg"),
        "s4_annular_four_panel": ("four_annular_gp_grid", "panel"),
    }
    if scenario_id in updraft_cases:
        model_name, wind_mode = updraft_cases[scenario_id]
        wind = _load_or_proxy(model_name, repo_root)
        primitive = four_fan_updraft_glide if "four" in model_name else updraft_glide
        return ScenarioDefinition(
            scenario_id,
            primitive,
            wind.name,
            wind,
            wind_mode,
            nominal,
            full_base,
            getattr(wind, "source", ""),
        )

    if scenario_id == "s4_gaussian_single_panel_randomised":
        wind = _load_or_proxy("single_gaussian_var", repo_root)
        wind, label = build_randomised_wind_field(wind, seed=seed, enabled=True)
        return ScenarioDefinition(
            scenario_id,
            updraft_glide,
            wind.name,
            wind,
            "panel",
            nominal,
            full_base,
            label,
        )

    if scenario_id == "s4_governor_selection":
        x0 = full_base.copy()
        # High bank rejects nominal entries while leaving recovery admissible
        x0[STATE_INDEX["phi"]] = np.deg2rad(70.0)
        candidates = (full_glide, full_bank_left, governor_recovery)
        return ScenarioDefinition(
            scenario_id,
            governor_recovery,
            "none",
            None,
            "panel",
            nominal,
            x0,
            "none",
            candidates,
        )
    raise ValueError(f"Unknown scenario_id '{scenario_id}'.")


# =============================================================================
# 4) Scenario Lists
# =============================================================================
def batch_scenarios() -> tuple[str, ...]:
    return (
        "s0_no_wind",
        "s1_latency_nominal_no_wind",
        "s1_latency_robust_upper_no_wind",
        "s6_single_gaussian_var",
        "s6_four_gaussian_var",
        "s7_single_annular_gp",
        "s7_four_annular_gp",
        "s11_governor_rejection",
        "s4_full_nominal_glide_no_wind",
        "s4_full_bank_reversal_left_no_wind",
        "s4_full_recovery_no_wind",
        "s4_latency_low_bank_reversal_left",
        "s4_latency_nominal_bank_reversal_left",
        "s4_latency_high_bank_reversal_left",
        "s4_gaussian_single_panel",
        "s4_annular_single_panel",
        "s4_gaussian_single_panel_randomised",
        "s4_governor_selection",
    )


def s4_audit_scenarios() -> tuple[str, ...]:
    return (
        "s4_full_nominal_glide_no_wind",
        "s4_full_bank_reversal_left_no_wind",
        "s4_full_bank_reversal_right_no_wind",
        "s4_full_recovery_no_wind",
        "s4_latency_low_bank_reversal_left",
        "s4_latency_nominal_bank_reversal_left",
        "s4_latency_high_bank_reversal_left",
        "s4_gaussian_single_panel",
        "s4_gaussian_single_cg",
        "s4_gaussian_four_panel",
        "s4_annular_single_panel",
        "s4_annular_single_cg",
        "s4_annular_four_panel",
        "s4_gaussian_single_panel_randomised",
        "s4_governor_selection",
    )


# =============================================================================
# 5) Updraft Loader Fallback
# =============================================================================
def _load_or_proxy(model_name: str, repo_root) -> WindField:
    try:
        return load_updraft_model(model_name, repo_root=repo_root)
    except FileNotFoundError:
        return load_updraft_model("analytic_debug_proxy", repo_root=repo_root)
