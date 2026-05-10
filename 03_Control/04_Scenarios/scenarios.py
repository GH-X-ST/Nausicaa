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
from updraft_models import WindField, load_updraft_model


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    primitive: object
    wind_model_name: str
    wind_model: WindField | None
    wind_mode: str
    latency_config: CommandToSurfaceConfig
    x0: np.ndarray


def trimmed_state_for_arena(x_trim: np.ndarray, altitude_m: float = 2.7) -> np.ndarray:
    x0 = np.asarray(x_trim, dtype=float).reshape(15).copy()
    x0[STATE_INDEX["x_w"]] = 4.2
    x0[STATE_INDEX["y_w"]] = 2.4
    x0[STATE_INDEX["z_w"]] = float(altitude_m)
    return x0


def recovery_entry_state(x_trim: np.ndarray) -> np.ndarray:
    x0 = trimmed_state_for_arena(x_trim, altitude_m=2.7)
    x0[STATE_INDEX["phi"]] = np.deg2rad(12.0)
    x0[STATE_INDEX["theta"]] += np.deg2rad(3.0)
    x0[STATE_INDEX["p"]] = np.deg2rad(-2.0)
    x0[STATE_INDEX["q"]] = np.deg2rad(1.0)
    x0[STATE_INDEX["r"]] = np.deg2rad(1.0)
    return x0


def build_scenario(
    scenario_id: str,
    x_trim: np.ndarray,
    repo_root,
) -> ScenarioDefinition:
    base = trimmed_state_for_arena(x_trim)
    short_glide = NominalGlidePrimitive(duration_s=0.20)
    short_bank = BankReversalPrimitive(duration_s=0.35, bank_angle_rad=np.deg2rad(8.0))
    short_recovery = RecoveryPrimitive(duration_s=0.30)
    nominal = CommandToSurfaceConfig(mode="nominal")
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
    raise ValueError(f"Unknown scenario_id '{scenario_id}'.")


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
    )
