from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE, as_state_vector


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
    state_sample_source: str
    paired_start_key: str
    state_envelope_label: str
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
    """Return a deterministic launch/envelope state sample for archive rows."""

    paired_key = f"start_{int(row_index) // 2:07d}"
    selector = int(row_index) % 5
    rng = np.random.default_rng(_stable_seed(seed, paired_key))
    if selector == 0:
        state = _launch_distribution_state(rng)
        label = "approved_launch_distribution"
        source = "deterministic_launch_distribution"
    elif selector == 1:
        state = _local_envelope_state(rng)
        label = "local_primitive_envelope"
        source = "deterministic_local_envelope"
    elif selector == 2:
        state = _boundary_near_state(rng, side="x_max")
        label = "boundary_near_x"
        source = "deterministic_boundary_near"
    elif selector == 3:
        state = _boundary_near_state(rng, side="y_min")
        label = "boundary_near_y"
        source = "deterministic_boundary_near"
    else:
        state = _paired_comparison_state(rng, W_layer=W_layer, environment_mode=environment_mode)
        label = "paired_start_comparison"
        source = "deterministic_paired_start"
    return ArchiveStateSample(
        state_vector=as_state_vector(state),
        state_sample_source=source,
        paired_start_key=paired_key,
        state_envelope_label=label,
    )


def archive_state_sample_row(sample: ArchiveStateSample) -> dict[str, object]:
    """Return state sample metadata and expanded canonical state columns."""

    state = as_state_vector(sample.state_vector)
    row = {
        "state_sample_source": sample.state_sample_source,
        "paired_start_key": sample.paired_start_key,
        "state_envelope_label": sample.state_envelope_label,
        "measured_log_source": sample.measured_log_source,
        "measured_log_row_index": sample.measured_log_row_index,
        "initial_state_vector_json": json.dumps(
            [float(value) for value in state],
            separators=(",", ":"),
        ),
    }
    row.update({f"initial_{name}": float(state[index]) for index, name in enumerate(STATE_NAMES)})
    return row


def _launch_distribution_state(rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=float)
    state[STATE_INDEX["x_w"]] = float(rng.normal(2.0, 0.12))
    state[STATE_INDEX["y_w"]] = float(rng.normal(2.2, 0.18))
    state[STATE_INDEX["z_w"]] = float(rng.normal(1.55, 0.08))
    state[STATE_INDEX["phi"]] = float(np.deg2rad(rng.normal(0.0, 3.0)))
    state[STATE_INDEX["theta"]] = float(np.deg2rad(rng.normal(0.0, 2.5)))
    state[STATE_INDEX["psi"]] = float(np.deg2rad(rng.normal(0.0, 4.0)))
    state[STATE_INDEX["u"]] = float(rng.normal(5.8, 0.25))
    state[STATE_INDEX["v"]] = float(rng.normal(0.0, 0.08))
    state[STATE_INDEX["w"]] = float(rng.normal(0.0, 0.05))
    return _clip_state_to_plausible_envelope(state)


def _local_envelope_state(rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=float)
    state[STATE_INDEX["x_w"]] = float(rng.uniform(1.6, 5.8))
    state[STATE_INDEX["y_w"]] = float(rng.uniform(0.5, 3.9))
    state[STATE_INDEX["z_w"]] = float(rng.uniform(0.9, 2.8))
    state[STATE_INDEX["phi"]] = float(np.deg2rad(rng.uniform(-18.0, 18.0)))
    state[STATE_INDEX["theta"]] = float(np.deg2rad(rng.uniform(-10.0, 10.0)))
    state[STATE_INDEX["psi"]] = float(np.deg2rad(rng.uniform(-25.0, 25.0)))
    state[STATE_INDEX["u"]] = float(rng.uniform(3.8, 7.2))
    state[STATE_INDEX["v"]] = float(rng.uniform(-0.25, 0.25))
    state[STATE_INDEX["w"]] = float(rng.uniform(-0.15, 0.15))
    return state


def _boundary_near_state(rng: np.random.Generator, *, side: str) -> np.ndarray:
    state = _local_envelope_state(rng)
    if side == "x_max":
        state[STATE_INDEX["x_w"]] = float(rng.uniform(6.35, 6.58))
    elif side == "y_min":
        state[STATE_INDEX["y_w"]] = float(rng.uniform(0.04, 0.20))
    else:
        raise ValueError("unknown boundary side.")
    state[STATE_INDEX["u"]] = float(rng.uniform(4.5, 6.5))
    return state


def _paired_comparison_state(
    rng: np.random.Generator,
    *,
    W_layer: str,
    environment_mode: str,
) -> np.ndarray:
    del W_layer, environment_mode
    state = _launch_distribution_state(rng)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.2
    return state


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
                state_sample_source="measured_log_compatible",
                paired_start_key=str(row.get("paired_start_key", f"measured_{index:07d}")),
                state_envelope_label=str(row.get("state_envelope_label", "measured_log")),
                measured_log_source=Path(path).as_posix(),
                measured_log_row_index=int(index),
            )
        )
    return rows


def measured_log_schema_row() -> dict[str, object]:
    """Return the expected measured-log schema without requiring measured logs."""

    return {
        "required_state_columns": ",".join(STATE_NAMES),
        "optional_metadata_columns": "paired_start_key,state_envelope_label,episode_id",
        "state_sample_source": "measured_log_compatible",
    }
