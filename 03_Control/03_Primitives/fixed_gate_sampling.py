from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from episode_schema import ENTRY_SOURCE_VALUES, FAN_BRANCH_VALUES, W_LAYER_VALUES
from fixed_gate_contract import (
    FIXED_LAUNCH_GATE,
    LAUNCH_TOLERANCE_SHELLS,
    launch_gate_admission_status,
    state_to_launch_gate_record,
)
from state_contract import STATE_NAMES, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Sampling constants and configuration
# 2) Fixed-gate and shell sampling
# 3) Reachable downstream sampling
# 4) W0/W1 and W2/W3 validation helpers
# =============================================================================


REACHABLE_TERMINAL_COLUMNS = (
    "x_terminal_w_m",
    "y_terminal_w_m",
    "z_terminal_w_m",
    "phi_terminal_rad",
    "theta_terminal_rad",
    "psi_terminal_rad",
    "u_terminal_m_s",
    "v_terminal_m_s",
    "w_terminal_m_s",
    "p_terminal_rad_s",
    "q_terminal_rad_s",
    "r_terminal_rad_s",
)

SAMPLE_STATE_COLUMNS = (
    "x_w_m",
    "y_w_m",
    "z_w_m",
    "phi_rad",
    "theta_rad",
    "psi_rad",
    "u_m_s",
    "v_m_s",
    "w_m_s",
    "p_rad_s",
    "q_rad_s",
    "r_rad_s",
    "delta_a_rad",
    "delta_e_rad",
    "delta_r_rad",
)


@dataclass(frozen=True)
class FixedGateSamplingConfig:
    total_count: int
    random_seed: int = 20260521
    launch_gate_fraction: float = 0.70
    reachable_or_shell_fraction: float = 0.20
    diagnostic_fraction: float = 0.10
    sampling_round: str = "fixed_gate_round_0"


def sampling_allocation_counts(config: FixedGateSamplingConfig) -> dict[str, int]:
    """Return deterministic source counts with diagnostic rows capped at 10%."""

    total = int(config.total_count)
    if total < 0:
        raise ValueError("total_count must be nonnegative.")
    diagnostic = min(int(round(total * float(config.diagnostic_fraction))), int(np.floor(total * 0.10)))
    launch = int(round(total * float(config.launch_gate_fraction)))
    local = max(0, total - launch - diagnostic)
    if local < int(np.floor(total * 0.20)) and total >= 10:
        needed = int(np.floor(total * 0.20)) - local
        launch = max(0, launch - needed)
        local += needed
    return {
        "launch_gate_main": launch,
        "local_or_reachable": local,
        "diagnostic_broad_only": diagnostic,
    }


def sample_fixed_gate_states(
    config: FixedGateSamplingConfig,
    *,
    fan_branch: str,
    W_layer: str,
    reachable_source_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sample fixed-gate mission starts and labelled non-claim diagnostic rows."""

    _validate_branch_and_layer(fan_branch, W_layer)
    rng = np.random.default_rng(int(config.random_seed))
    counts = sampling_allocation_counts(config)
    rows: list[dict[str, object]] = []
    for index in range(int(counts["launch_gate_main"])):
        rows.append(
            _sample_row(
                rng,
                index,
                config,
                fan_branch=fan_branch,
                W_layer=W_layer,
                entry_source="launch_gate_main",
                bounds=FIXED_LAUNCH_GATE,
            )
        )

    local_count = int(counts["local_or_reachable"])
    reachable = build_reachable_downstream_states(
        reachable_source_rows,
        max_rows=local_count,
        config=config,
        fan_branch=fan_branch,
        W_layer=W_layer,
        start_index=len(rows),
    ) if reachable_source_rows is not None and not reachable_source_rows.empty else pd.DataFrame()
    rows.extend(reachable.to_dict("records"))
    remaining_local = local_count - int(len(reachable))
    shell = LAUNCH_TOLERANCE_SHELLS["local_robustness_shell"]
    for local_index in range(max(0, remaining_local)):
        rows.append(
            _sample_row(
                rng,
                len(rows),
                config,
                fan_branch=fan_branch,
                W_layer=W_layer,
                entry_source="local_robustness_shell",
                bounds=shell,
            )
        )

    for diagnostic_index in range(int(counts["diagnostic_broad_only"])):
        rows.append(
            _diagnostic_row(
                rng,
                len(rows),
                config,
                fan_branch=fan_branch,
                W_layer=W_layer,
                diagnostic_index=diagnostic_index,
            )
        )
    frame = pd.DataFrame(rows)
    validate_entry_sources(frame)
    return frame


def build_reachable_downstream_states(
    accepted_rollouts: pd.DataFrame | None,
    *,
    max_rows: int,
    config: FixedGateSamplingConfig,
    fan_branch: str,
    W_layer: str,
    start_index: int = 0,
) -> pd.DataFrame:
    """Derive downstream primitive starts from accepted fixed-gate rollouts.

    This intentionally refuses arbitrary centre/end-arena rows. A reachable
    downstream state must name an accepted source rollout whose own entry source
    was the fixed launch gate or tolerance shell.
    """

    if accepted_rollouts is None or accepted_rollouts.empty or int(max_rows) <= 0:
        return pd.DataFrame(columns=_sample_columns())
    validate_reachable_source_rollouts(accepted_rollouts)
    rows: list[dict[str, object]] = []
    source = accepted_rollouts.copy().reset_index(drop=True)
    for local_index, (_, rollout) in enumerate(source.head(int(max_rows)).iterrows()):
        state = _terminal_state_from_rollout(rollout)
        record = _base_record(
            state,
            sample_index=int(start_index) + local_index,
            config=config,
            fan_branch=fan_branch,
            W_layer=W_layer,
            entry_source="reachable_downstream",
        )
        record["reachable_provenance_id"] = str(rollout.get("trial_descriptor_id", rollout.get("primitive_rollout_id", f"source_{local_index:06d}")))
        record["source_entry_source"] = str(rollout.get("entry_source"))
        record["source_rollout_accepted"] = True
        rows.append(record)
    frame = pd.DataFrame(rows)
    validate_entry_sources(frame)
    return frame


def validate_reachable_source_rollouts(accepted_rollouts: pd.DataFrame) -> None:
    _require_columns(accepted_rollouts, {"entry_source"}, "accepted_rollouts")
    accepted_column = "accepted" if "accepted" in accepted_rollouts.columns else "success_flag"
    if accepted_column not in accepted_rollouts.columns:
        raise ValueError("accepted_rollouts must contain accepted or success_flag.")
    invalid_entry = set(accepted_rollouts["entry_source"].astype(str)).difference(
        {"launch_gate_main", "launch_gate_tolerance_shell"}
    )
    if invalid_entry:
        raise ValueError(f"reachable downstream sources must start from fixed gate/shell, got {sorted(invalid_entry)}.")
    accepted = accepted_rollouts[accepted_column].astype(bool)
    if not bool(accepted.all()):
        raise ValueError("reachable downstream sources must be accepted rollouts.")
    missing_terminal = sorted(set(REACHABLE_TERMINAL_COLUMNS).difference(accepted_rollouts.columns))
    if missing_terminal:
        raise ValueError(f"accepted_rollouts missing terminal state columns: {missing_terminal}")


def validate_entry_sources(frame: pd.DataFrame) -> None:
    _require_columns(frame, {"entry_source", "fan_branch", "W_layer"}, "fixed_gate_samples")
    invalid_source = sorted(set(frame["entry_source"].astype(str)).difference(ENTRY_SOURCE_VALUES))
    if invalid_source:
        raise ValueError(f"unsupported entry_source values: {invalid_source}")
    invalid_branch = sorted(set(frame["fan_branch"].astype(str)).difference(FAN_BRANCH_VALUES))
    if invalid_branch:
        raise ValueError(f"unsupported fan_branch values: {invalid_branch}")
    invalid_layer = sorted(set(frame["W_layer"].astype(str)).difference(W_LAYER_VALUES))
    if invalid_layer:
        raise ValueError(f"unsupported W_layer values: {invalid_layer}")


def build_fixed_gate_w0_w1_candidate_rows(
    samples: pd.DataFrame,
    *,
    primitive_families: tuple[str, ...],
) -> pd.DataFrame:
    """Build a paired W0/W1 candidate index without filtering W1 by W0 success."""

    validate_entry_sources(samples)
    rows: list[dict[str, object]] = []
    for _, sample in samples.iterrows():
        for W_layer in ("W0", "W1"):
            for primitive_family in primitive_families:
                rows.append(
                    {
                        **sample.to_dict(),
                        "W_layer": W_layer,
                        "test_environment_mode": _environment_mode(str(sample["fan_branch"]), W_layer),
                        "primitive_family": str(primitive_family),
                        "candidate_id": f"{sample['sample_id']}__{W_layer}__{primitive_family}",
                        "w1_scheduled_independent_of_w0_success": True,
                    }
                )
    frame = pd.DataFrame(rows)
    validate_w1_independent_of_w0(frame)
    return frame


def validate_w1_independent_of_w0(candidate_rows: pd.DataFrame) -> None:
    _require_columns(candidate_rows, {"paired_sample_key", "W_layer", "fan_branch", "primitive_family"}, "candidate_rows")
    w0 = candidate_rows[candidate_rows["W_layer"].astype(str) == "W0"]
    w1 = candidate_rows[candidate_rows["W_layer"].astype(str) == "W1"]
    keys = ["paired_sample_key", "fan_branch", "primitive_family"]
    w0_keys = set(map(tuple, w0[keys].astype(str).to_numpy()))
    w1_keys = set(map(tuple, w1[keys].astype(str).to_numpy()))
    missing = sorted(w0_keys.symmetric_difference(w1_keys))
    if missing:
        raise ValueError(f"W0/W1 paired scheduling mismatch; W1 must not be filtered by W0 success: {missing[:5]}")


def select_focused_replay_cases(
    source_rows: pd.DataFrame,
    *,
    target_W_layer: str,
    max_cases: int,
) -> pd.DataFrame:
    """Select W2/W3 replay cases from W1/medoid evidence only."""

    if target_W_layer not in {"W2", "W3"}:
        raise ValueError("target_W_layer must be W2 or W3.")
    _require_columns(source_rows, {"W_layer", "entry_source", "fan_branch"}, "source_rows")
    frame = source_rows.copy()
    if target_W_layer == "W2":
        eligible = frame[frame["W_layer"].astype(str).eq("W1")].copy()
    else:
        eligible = frame[frame["W_layer"].astype(str).isin({"W1", "W2"})].copy()
    if "recommended_use" in eligible.columns:
        eligible = eligible[eligible["recommended_use"].astype(str).isin({"thesis", "hardware", "diagnostic"})]
    if "is_medoid" in eligible.columns:
        eligible = eligible[eligible["is_medoid"].astype(bool) | eligible["W_layer"].astype(str).eq("W1")]
    eligible = eligible[~eligible["entry_source"].astype(str).eq("diagnostic_broad_only")].copy()
    if eligible.empty:
        return pd.DataFrame(columns=[*source_rows.columns, "source_W_layer", "W_layer", "focused_replay_source"])
    eligible = eligible.sort_values([col for col in ("fan_branch", "primitive_family", "sample_id") if col in eligible.columns])
    selected = eligible.head(int(max_cases)).copy()
    selected["source_W_layer"] = selected["W_layer"].astype(str)
    selected["W_layer"] = str(target_W_layer)
    selected["focused_replay_source"] = "selected_W1_or_medoid_case"
    return selected.reset_index(drop=True)


def _sample_row(
    rng: np.random.Generator,
    sample_index: int,
    config: FixedGateSamplingConfig,
    *,
    fan_branch: str,
    W_layer: str,
    entry_source: str,
    bounds: object,
) -> dict[str, object]:
    speed = _uniform(rng, bounds.speed_m_s)
    state = np.zeros(15, dtype=float)
    state[0] = _uniform(rng, bounds.x_w_m)
    state[1] = _uniform(rng, bounds.y_w_m)
    state[2] = _uniform(rng, bounds.z_w_m)
    state[3] = _uniform(rng, bounds.phi_rad)
    state[4] = _uniform(rng, bounds.theta_rad)
    state[5] = _uniform(rng, bounds.psi_rad)
    state[6] = speed
    return _base_record(state, sample_index=sample_index, config=config, fan_branch=fan_branch, W_layer=W_layer, entry_source=entry_source)


def _diagnostic_row(
    rng: np.random.Generator,
    sample_index: int,
    config: FixedGateSamplingConfig,
    *,
    fan_branch: str,
    W_layer: str,
    diagnostic_index: int,
) -> dict[str, object]:
    state = np.zeros(15, dtype=float)
    state[0] = _uniform(rng, (1.2, 6.6))
    state[1] = _uniform(rng, (0.0, 4.4))
    state[2] = _uniform(rng, (0.7, 3.1))
    state[3] = np.deg2rad(_uniform(rng, (-45.0, 45.0)))
    state[4] = np.deg2rad(_uniform(rng, (-45.0, 45.0)))
    state[5] = np.deg2rad(_uniform(rng, (-90.0, 90.0)))
    state[6] = _uniform(rng, (3.0, 8.0))
    record = _base_record(state, sample_index=sample_index, config=config, fan_branch=fan_branch, W_layer=W_layer, entry_source="diagnostic_broad_only")
    record["diagnostic_case_index"] = int(diagnostic_index)
    return record


def _base_record(
    state: np.ndarray,
    *,
    sample_index: int,
    config: FixedGateSamplingConfig,
    fan_branch: str,
    W_layer: str,
    entry_source: str,
) -> dict[str, object]:
    x = as_state_vector(state)
    sample_id = f"fg_{int(config.random_seed)}_{fan_branch}_{W_layer}_{int(sample_index):06d}"
    paired_key = f"fg_pair_{int(config.random_seed)}_{fan_branch}_{int(sample_index):06d}"
    record = {
        "sample_id": sample_id,
        "paired_sample_key": paired_key,
        "fan_branch": str(fan_branch),
        "W_layer": str(W_layer),
        "entry_source": str(entry_source),
        "launch_gate_id": FIXED_LAUNCH_GATE.launch_gate_id,
        "random_seed": int(config.random_seed),
        "sampling_round": str(config.sampling_round),
        "initial_state_vector": ";".join(f"{value:.12g}" for value in x),
        "reachable_provenance_id": "",
        "source_entry_source": "",
        "source_rollout_accepted": False,
    }
    for name, value in zip(SAMPLE_STATE_COLUMNS, x, strict=True):
        record[name] = float(value)
    record.update(state_to_launch_gate_record(x))
    if entry_source == "reachable_downstream":
        record["initial_state_admission_status"] = "reachable_downstream"
    elif entry_source == "local_robustness_shell" and launch_gate_admission_status(x) == "admitted_tolerance_shell":
        record["initial_state_admission_status"] = "admitted_tolerance_shell"
    return record


def _terminal_state_from_rollout(rollout: pd.Series) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    values = [rollout[column] for column in REACHABLE_TERMINAL_COLUMNS]
    state[0:12] = np.asarray(values, dtype=float)
    return state


def _sample_columns() -> list[str]:
    return [
        "sample_id",
        "paired_sample_key",
        "fan_branch",
        "W_layer",
        "entry_source",
        "launch_gate_id",
        "random_seed",
        "sampling_round",
        "initial_state_vector",
        *SAMPLE_STATE_COLUMNS,
    ]


def _validate_branch_and_layer(fan_branch: str, W_layer: str) -> None:
    if str(fan_branch) not in FAN_BRANCH_VALUES:
        raise ValueError("fan_branch must be single_fan_branch or four_fan_branch.")
    if str(W_layer) not in W_LAYER_VALUES:
        raise ValueError("W_layer must be one of W0, W1, W2, W3, real.")


def _environment_mode(fan_branch: str, W_layer: str) -> str:
    if fan_branch == "single_fan_branch":
        return "W0_single_fan_branch" if W_layer == "W0" else f"{W_layer}_single_fan"
    return "W0_four_fan_branch" if W_layer == "W0" else f"{W_layer}_four_fan"


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
