from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE, as_state_vector

LAUNCH_GATE_ROLL_LIMIT_DEG = 20.0
LAUNCH_GATE_PITCH_MIN_DEG = -10.0
LAUNCH_GATE_PITCH_MAX_DEG = 20.0
LAUNCH_GATE_YAW_LIMIT_DEG = 20.0
LAUNCH_GATE_SPEED_MIN_M_S = 3.0
LAUNCH_GATE_SPEED_MAX_M_S = 8.0
LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S = 1.5
LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S = 0.5
LAUNCH_GATE_Z_W_M = (1.3, 1.8)
LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S = 1.2
LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S = 1.2
LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S = 1.8


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) State sample dataclass
# 2) Deterministic archive samplers
# 3) Measured-log compatibility helpers
# =============================================================================


# =============================================================================
# 1) State Sample Dataclass
# =============================================================================
@dataclass(frozen=True)
class ArchiveStateSample:
    state_vector: np.ndarray
    start_state_family: str
    state_sample_source: str
    paired_start_key: str
    state_envelope_label: str
    previous_primitive_status: str
    state_sample_detail: str
    synthetic_previous_primitive_id: str
    synthetic_time_since_launch_s: float
    state_sampling_seed: int
    launch_gate_compliant: bool
    state_sampling_version: str = "mixed_primitive_start_v3_launch_vw_bound"
    measured_log_source: str = ""
    measured_log_row_index: int | str = ""


# =============================================================================
# 2) Deterministic Archive Samplers
# =============================================================================
def archive_state_sample_for_row(
    row_index: int,
    *,
    seed: int,
    W_layer: str,
    environment_mode: str,
) -> ArchiveStateSample:
    """Return a deterministic mixed primitive-start sample for archive rows."""

    paired_key = f"start_{int(row_index) // 2:07d}"
    family = start_state_family_for_row(row_index)
    return archive_state_sample_for_family(
        start_state_family=family,
        paired_start_key=paired_key,
        sample_index=int(row_index),
        seed=seed,
        W_layer=W_layer,
        environment_mode=environment_mode,
    )


def archive_state_sample_for_family(
    *,
    start_state_family: str,
    paired_start_key: str,
    sample_index: int,
    seed: int,
    W_layer: str,
    environment_mode: str,
) -> ArchiveStateSample:
    """Return a deterministic state sample for an explicit paired start family."""

    family = str(start_state_family)
    paired_key = str(paired_start_key)
    sample_seed = _stable_seed(seed, f"{paired_key}:{family}")
    rng = np.random.default_rng(sample_seed)
    if family == "launch_gate":
        state = _launch_gate_state(rng)
        label = "approved_launch_gate"
        source = "synthetic_launch_gate"
        detail = "deterministic_mixed_start_launch_gate"
        previous_status = "launch_start"
        previous_primitive_id = ""
        time_since_launch_s = 0.0
    elif family == "inflight_nominal":
        state = _inflight_nominal_state(rng)
        label = "local_primitive_envelope"
        source = "rollout_exit_resampled"
        detail = "deterministic_mixed_start_inflight_nominal"
        previous_status = "clean_exit"
        previous_primitive_id = _synthetic_previous_primitive_id(sample_index)
        time_since_launch_s = float(rng.uniform(0.8, 2.2))
    elif family == "inflight_lift_region":
        state = _inflight_lift_region_state(rng)
        label = "lift_region"
        source = "synthetic_inflight"
        detail = "deterministic_mixed_start_lift_region"
        previous_status = "clean_exit"
        previous_primitive_id = _synthetic_previous_primitive_id(sample_index)
        time_since_launch_s = float(rng.uniform(0.6, 2.6))
    elif family == "inflight_boundary_near":
        side = "x_max" if int(sample_index) % 2 == 0 else "y_min"
        state = _boundary_near_state(rng, side=side)
        label = "boundary_near"
        source = "stress_sample"
        detail = f"{side}_terminal_useful_exit_detail_legacy_boundary_terminal"
        previous_status = "boundary_terminal"
        previous_primitive_id = _synthetic_previous_primitive_id(sample_index)
        time_since_launch_s = float(rng.uniform(0.8, 2.8))
    else:
        state = _inflight_recovery_edge_state(rng)
        label = "recovery_edge"
        source = "stress_sample"
        detail = "deterministic_mixed_start_recovery_edge"
        previous_status = "recovery_edge"
        previous_primitive_id = _synthetic_previous_primitive_id(sample_index)
        time_since_launch_s = float(rng.uniform(0.4, 2.0))
    del W_layer, environment_mode
    return ArchiveStateSample(
        state_vector=as_state_vector(state),
        start_state_family=family,
        state_sample_source=source,
        paired_start_key=paired_key,
        state_envelope_label=label,
        previous_primitive_status=previous_status,
        state_sample_detail=detail,
        synthetic_previous_primitive_id=previous_primitive_id,
        synthetic_time_since_launch_s=time_since_launch_s,
        state_sampling_seed=int(sample_seed),
        launch_gate_compliant=state_is_launch_gate_compliant(state),
    )


def archive_state_sample_row(sample: ArchiveStateSample) -> dict[str, object]:
    """Return state sample metadata and expanded canonical state columns."""

    state = as_state_vector(sample.state_vector)
    row = {
        "start_state_family": sample.start_state_family,
        "state_sample_source": sample.state_sample_source,
        "paired_start_key": sample.paired_start_key,
        "state_envelope_label": sample.state_envelope_label,
        "previous_primitive_status": sample.previous_primitive_status,
        "state_sample_detail": sample.state_sample_detail,
        "synthetic_previous_primitive_id": sample.synthetic_previous_primitive_id,
        "synthetic_time_since_launch_s": float(sample.synthetic_time_since_launch_s),
        "state_sampling_seed": int(sample.state_sampling_seed),
        "launch_gate_compliant": bool(sample.launch_gate_compliant),
        "state_sampling_version": sample.state_sampling_version,
        "measured_log_source": sample.measured_log_source,
        "measured_log_row_index": sample.measured_log_row_index,
        "initial_state_vector_json": json.dumps(
            [float(value) for value in state],
            separators=(",", ":"),
        ),
    }
    row.update({f"initial_{name}": float(state[index]) for index, name in enumerate(STATE_NAMES)})
    return row


def start_state_family_for_row(row_index: int) -> str:
    """Return the deterministic 40/25/15/10/10 primitive-start family assignment."""

    slot = int(row_index) % 20
    if slot < 8:
        return "launch_gate"
    if slot < 13:
        return "inflight_nominal"
    if slot < 16:
        return "inflight_lift_region"
    if slot < 18:
        return "inflight_boundary_near"
    return "inflight_recovery_edge"


def state_is_launch_gate_compliant(state: np.ndarray) -> bool:
    """Return whether a state lies inside the approved physical release gate."""

    x = as_state_vector(state)
    speed = float(np.linalg.norm(x[6:9]))
    return bool(
        1.2 <= x[STATE_INDEX["x_w"]] <= 1.4
        and 1.8 <= x[STATE_INDEX["y_w"]] <= 2.2
        and LAUNCH_GATE_Z_W_M[0] <= x[STATE_INDEX["z_w"]] <= LAUNCH_GATE_Z_W_M[1]
        and np.deg2rad(-LAUNCH_GATE_ROLL_LIMIT_DEG)
        <= x[STATE_INDEX["phi"]]
        <= np.deg2rad(LAUNCH_GATE_ROLL_LIMIT_DEG)
        and np.deg2rad(LAUNCH_GATE_PITCH_MIN_DEG)
        <= x[STATE_INDEX["theta"]]
        <= np.deg2rad(LAUNCH_GATE_PITCH_MAX_DEG)
        and np.deg2rad(-LAUNCH_GATE_YAW_LIMIT_DEG)
        <= x[STATE_INDEX["psi"]]
        <= np.deg2rad(LAUNCH_GATE_YAW_LIMIT_DEG)
        and LAUNCH_GATE_SPEED_MIN_M_S <= speed <= LAUNCH_GATE_SPEED_MAX_M_S
        and -LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S
        <= x[STATE_INDEX["v"]]
        <= LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S
        and -LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S
        <= x[STATE_INDEX["w"]]
        <= LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S
        and -LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S
        <= x[STATE_INDEX["p"]]
        <= LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S
        and -LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S
        <= x[STATE_INDEX["q"]]
        <= LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S
        and -LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S
        <= x[STATE_INDEX["r"]]
        <= LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S
    )


def _launch_gate_state(rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=float)
    state[STATE_INDEX["x_w"]] = float(rng.uniform(1.2, 1.4))
    state[STATE_INDEX["y_w"]] = float(rng.uniform(1.8, 2.2))
    state[STATE_INDEX["z_w"]] = float(rng.uniform(*LAUNCH_GATE_Z_W_M))
    state[STATE_INDEX["phi"]] = float(
        np.deg2rad(rng.uniform(-LAUNCH_GATE_ROLL_LIMIT_DEG, LAUNCH_GATE_ROLL_LIMIT_DEG))
    )
    state[STATE_INDEX["theta"]] = float(
        np.deg2rad(rng.uniform(LAUNCH_GATE_PITCH_MIN_DEG, LAUNCH_GATE_PITCH_MAX_DEG))
    )
    state[STATE_INDEX["psi"]] = float(
        np.deg2rad(rng.uniform(-LAUNCH_GATE_YAW_LIMIT_DEG, LAUNCH_GATE_YAW_LIMIT_DEG))
    )
    speed = float(rng.uniform(LAUNCH_GATE_SPEED_MIN_M_S, LAUNCH_GATE_SPEED_MAX_M_S))
    v_side = float(
        rng.uniform(-LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S, LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S)
    )
    w_body = float(
        rng.uniform(
            -LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
            LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
        )
    )
    state[STATE_INDEX["u"]] = float(np.sqrt(max(speed * speed - v_side * v_side - w_body * w_body, 0.0)))
    state[STATE_INDEX["v"]] = v_side
    state[STATE_INDEX["w"]] = w_body
    state[STATE_INDEX["p"]] = float(
        rng.uniform(-LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S, LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S)
    )
    state[STATE_INDEX["q"]] = float(
        rng.uniform(-LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S, LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S)
    )
    state[STATE_INDEX["r"]] = float(
        rng.uniform(-LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S, LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S)
    )
    return state


def _local_envelope_state(rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=float)
    state[STATE_INDEX["x_w"]] = float(rng.uniform(1.6, 5.8))
    state[STATE_INDEX["y_w"]] = float(rng.uniform(0.5, 3.9))
    state[STATE_INDEX["z_w"]] = float(rng.uniform(0.9, 2.8))
    state[STATE_INDEX["phi"]] = float(np.deg2rad(rng.uniform(-18.0, 18.0)))
    state[STATE_INDEX["theta"]] = float(np.deg2rad(rng.uniform(-10.0, 10.0)))
    state[STATE_INDEX["psi"]] = float(np.deg2rad(rng.uniform(-25.0, 25.0)))
    state[STATE_INDEX["u"]] = float(rng.uniform(3.0, 8.2))
    state[STATE_INDEX["v"]] = float(rng.uniform(-0.35, 0.35))
    state[STATE_INDEX["w"]] = float(rng.uniform(-0.25, 0.25))
    return _with_inflight_rates_and_surfaces(state, rng, rate_scale=0.35, surface_scale=0.22)


def _inflight_nominal_state(rng: np.random.Generator) -> np.ndarray:
    return _local_envelope_state(rng)


def _inflight_lift_region_state(rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=float)
    state[STATE_INDEX["x_w"]] = float(rng.uniform(2.0, 4.3))
    state[STATE_INDEX["y_w"]] = float(rng.uniform(1.2, 3.2))
    state[STATE_INDEX["z_w"]] = float(rng.uniform(1.2, 2.4))
    state[STATE_INDEX["phi"]] = float(np.deg2rad(rng.uniform(-22.0, 22.0)))
    state[STATE_INDEX["theta"]] = float(np.deg2rad(rng.uniform(-12.0, 12.0)))
    state[STATE_INDEX["psi"]] = float(np.deg2rad(rng.uniform(-35.0, 35.0)))
    state[STATE_INDEX["u"]] = float(rng.uniform(3.2, 8.0))
    state[STATE_INDEX["v"]] = float(rng.uniform(-0.30, 0.30))
    state[STATE_INDEX["w"]] = float(rng.uniform(-0.22, 0.22))
    return _with_inflight_rates_and_surfaces(state, rng, rate_scale=0.30, surface_scale=0.20)


def _boundary_near_state(rng: np.random.Generator, *, side: str) -> np.ndarray:
    state = _local_envelope_state(rng)
    if side == "x_max":
        state[STATE_INDEX["x_w"]] = float(rng.uniform(6.35, 6.58))
    elif side == "y_min":
        state[STATE_INDEX["y_w"]] = float(rng.uniform(0.04, 0.20))
    else:
        raise ValueError("unknown boundary side.")
    state[STATE_INDEX["u"]] = float(rng.uniform(3.0, 8.0))
    return state


def _inflight_recovery_edge_state(rng: np.random.Generator) -> np.ndarray:
    state = _local_envelope_state(rng)
    state[STATE_INDEX["phi"]] = float(np.deg2rad(rng.uniform(-42.0, 42.0)))
    state[STATE_INDEX["theta"]] = float(np.deg2rad(rng.uniform(-28.0, 28.0)))
    state[STATE_INDEX["u"]] = float(rng.uniform(2.2, 5.2))
    state[STATE_INDEX["v"]] = float(rng.uniform(-0.45, 0.45))
    state[STATE_INDEX["w"]] = float(rng.uniform(-0.35, 0.35))
    return _with_inflight_rates_and_surfaces(state, rng, rate_scale=0.70, surface_scale=0.35)


def _with_inflight_rates_and_surfaces(
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    rate_scale: float,
    surface_scale: float,
) -> np.ndarray:
    result = np.asarray(state, dtype=float).reshape(STATE_SIZE).copy()
    result[STATE_INDEX["p"]] = float(rng.uniform(-rate_scale, rate_scale))
    result[STATE_INDEX["q"]] = float(rng.uniform(-0.7 * rate_scale, 0.7 * rate_scale))
    result[STATE_INDEX["r"]] = float(rng.uniform(-0.8 * rate_scale, 0.8 * rate_scale))
    result[STATE_INDEX["delta_a"]] = float(rng.uniform(-surface_scale, surface_scale))
    result[STATE_INDEX["delta_e"]] = float(rng.uniform(-surface_scale, surface_scale))
    result[STATE_INDEX["delta_r"]] = float(rng.uniform(-surface_scale, surface_scale))
    return result


def _synthetic_previous_primitive_id(row_index: int) -> str:
    primitive_ids = (
        "glide",
        "recovery",
        "lift_entry",
        "lift_dwell_arc",
        "mild_turn_left",
        "mild_turn_right",
        "energy_retaining_bank",
        "safe_exit_or_recovery_handoff",
    )
    return primitive_ids[int(row_index) % len(primitive_ids)]


def _clip_state_to_plausible_envelope(state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=float).reshape(STATE_SIZE).copy()
    result[STATE_INDEX["x_w"]] = np.clip(result[STATE_INDEX["x_w"]], 1.4, 6.2)
    result[STATE_INDEX["y_w"]] = np.clip(result[STATE_INDEX["y_w"]], 0.2, 4.1)
    result[STATE_INDEX["z_w"]] = np.clip(result[STATE_INDEX["z_w"]], 0.7, 3.1)
    result[STATE_INDEX["u"]] = np.clip(result[STATE_INDEX["u"]], 3.2, 7.5)
    return result


def _stable_seed(seed: int, key: str) -> int:
    value = int(seed) * 1_000_003
    for char in str(key):
        value = (value * 33 + ord(char)) % (2**32 - 1)
    return int(value)


# =============================================================================
# 3) Measured-Log Compatibility Helpers
# =============================================================================
def measured_log_state_sample_rows(path: Path) -> list[ArchiveStateSample]:
    """Read measured-log style state rows when future sim-real logs are available."""

    frame = pd.read_csv(path)
    required = set(STATE_NAMES)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"measured log is missing state columns: {sorted(missing)}")
    rows: list[ArchiveStateSample] = []
    for index, row in frame.iterrows():
        state = np.asarray([float(row[name]) for name in STATE_NAMES], dtype=float)
        rows.append(
            ArchiveStateSample(
                state_vector=as_state_vector(state),
                start_state_family=str(row.get("start_state_family", "measured_log")),
                state_sample_source="measured_log",
                paired_start_key=str(row.get("paired_start_key", f"measured_{index:07d}")),
                state_envelope_label=str(row.get("state_envelope_label", "local_primitive_envelope")),
                previous_primitive_status=str(row.get("previous_primitive_status", "unknown")),
                state_sample_detail=str(row.get("state_sample_detail", "measured_log_compatible")),
                synthetic_previous_primitive_id=str(row.get("synthetic_previous_primitive_id", "")),
                synthetic_time_since_launch_s=float(row.get("synthetic_time_since_launch_s", 0.0)),
                state_sampling_seed=int(row.get("state_sampling_seed", index)),
                launch_gate_compliant=state_is_launch_gate_compliant(state),
                measured_log_source=Path(path).as_posix(),
                measured_log_row_index=int(index),
            )
        )
    return rows


def measured_log_schema_row() -> dict[str, object]:
    """Return the expected measured-log schema without requiring measured logs."""

    return {
        "required_state_columns": ",".join(STATE_NAMES),
        "optional_metadata_columns": "paired_start_key,state_envelope_label,start_state_family,previous_primitive_status,episode_id",
        "state_sample_source": "measured_log",
    }
