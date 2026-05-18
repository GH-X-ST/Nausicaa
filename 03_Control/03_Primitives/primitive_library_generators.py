from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitive_library_schema import (
    PRIMITIVE_FAMILIES,
    PrimitiveCandidateSpec,
    PrimitiveLibraryConfig,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Candidate timing and start states
# 2) Candidate inventory
# 3) Command profile generation
# =============================================================================


# =============================================================================
# 1) Candidate Timing and Start States
# =============================================================================
@dataclass(frozen=True)
class CommandProfile:
    u_norm_requested: np.ndarray
    phase_labels: tuple[str, ...]


def build_start_state(
    start_condition: str,
    speed_m_s: float = 6.5,
    altitude_m: float = 1.8,
) -> np.ndarray:
    """Return a canonical 15-state initial condition for library replay."""

    if start_condition == "favourable":
        x_w, y_w, z_w = 1.30, 2.20, float(altitude_m)
    elif start_condition == "mid_arena":
        x_w, y_w, z_w = 3.90, 2.20, float(altitude_m)
    else:
        raise ValueError(f"unsupported first-pass start condition: {start_condition}")
    state = np.zeros(15, dtype=float)
    state[0:3] = (x_w, y_w, z_w)
    state[6] = float(speed_m_s)
    return state


def _horizon_s(family: str, target_heading_deg: float | None) -> float:
    if family == "glide":
        return 0.50
    if family == "recovery":
        return 0.64
    if family == "mild_bank":
        return 0.60
    if target_heading_deg is not None and float(target_heading_deg) >= 30.0:
        return 0.95
    return 0.65


# =============================================================================
# 2) Candidate Inventory
# =============================================================================
def primitive_candidate_inventory(
    config: PrimitiveLibraryConfig,
) -> tuple[PrimitiveCandidateSpec, ...]:
    """Generate deterministic candidate specs for the first library pass."""

    candidates: list[PrimitiveCandidateSpec] = []
    for family in config.families:
        if family not in PRIMITIVE_FAMILIES:
            raise ValueError(f"unknown primitive family: {family}")
        targets: tuple[float | None, ...]
        targets = (None,) if family in ("glide", "recovery", "mild_bank") else config.targets_deg
        for target in targets:
            for start in config.start_conditions:
                for direction in config.direction_signs:
                    environments = _environment_pairs(config)
                    for updraft_config, wind_fidelity in environments:
                        target_label = "none" if target is None else f"{int(target):03d}"
                        variant_id = (
                            f"{family}_{target_label}_{start}_{updraft_config}_"
                            f"{wind_fidelity}_d{int(direction):+d}"
                        ).replace("+", "p").replace("-", "m")
                        parent = f"{family}_{target_label}"
                        candidates.append(
                            PrimitiveCandidateSpec(
                                primitive_id=variant_id,
                                parent_primitive_id=parent,
                                variant_id=variant_id,
                                family=family,
                                target_heading_deg=target,
                                updraft_config=updraft_config,
                                wind_fidelity=wind_fidelity,
                                start_condition=start,
                                direction_sign=int(direction),
                                horizon_s=_horizon_s(family, target),
                            )
                        )
    return tuple(candidates)


def _environment_pairs(config: PrimitiveLibraryConfig) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    if "none" in config.updraft_configs and "W0" in config.wind_fidelities:
        pairs.append(("none", "W0"))
    for updraft_config in config.updraft_configs:
        if updraft_config == "none":
            continue
        for wind_fidelity in config.wind_fidelities:
            if wind_fidelity in ("W1", "W2"):
                pairs.append((updraft_config, wind_fidelity))
    return tuple(pairs)


# =============================================================================
# 3) Command Profile Generation
# =============================================================================
def generate_command_profile(
    spec: PrimitiveCandidateSpec,
    time_s: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return bounded normalised command requests and phase labels."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    if time.size == 0:
        raise ValueError("time_s must contain at least one sample.")
    phase = tuple(_phase_label(float(t), float(time[-1]), spec.family) for t in time)
    command = np.zeros((time.size, 3), dtype=float)
    direction = float(np.sign(spec.direction_sign) or 1.0)
    tau = time / max(float(time[-1]), 1e-12)

    if spec.family == "glide":
        command[:, 1] = -0.04
    elif spec.family == "recovery":
        command[:, 1] = -0.12 * np.exp(-3.0 * tau)
    elif spec.family == "mild_bank":
        pulse = np.sin(np.pi * np.clip(tau, 0.0, 1.0))
        command[:, 0] = 0.22 * direction * pulse
        command[:, 2] = 0.08 * direction * pulse
        command[:, 1] = -0.03
    elif spec.family == "canyon_steep_bank":
        pulse = np.sin(np.pi * np.clip(tau, 0.0, 1.0))
        command[:, 0] = 0.55 * direction * pulse
        command[:, 2] = 0.18 * direction * pulse
        command[:, 1] = -0.06
    elif spec.family == "wingover_lite":
        climb = np.sin(np.pi * np.clip(tau, 0.0, 1.0))
        command[:, 0] = 0.38 * direction * climb
        command[:, 2] = 0.22 * direction * climb
        command[:, 1] = -0.18 * np.where(tau < 0.45, 1.0 - tau, 0.25)
    elif spec.family == "bank_yaw_energy_retaining":
        pulse = np.sin(np.pi * np.clip(tau, 0.0, 1.0))
        command[:, 0] = 0.36 * direction * pulse
        command[:, 2] = 0.34 * direction * pulse
        command[:, 1] = -0.08 * (1.0 - 0.5 * tau)
    else:
        raise ValueError(f"unknown primitive family: {spec.family}")
    return np.clip(command, -1.0, 1.0), phase


def _phase_label(t_s: float, t_final_s: float, family: str) -> str:
    tau = t_s / max(t_final_s, 1e-12)
    if family in ("glide", "recovery"):
        return family
    if tau < 0.25:
        return "entry"
    if tau < 0.70:
        return "turn"
    return "exit"
