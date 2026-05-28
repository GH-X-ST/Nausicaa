from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CONTROL_ROOT = Path(__file__).resolve().parents[1]
INNER_LOOP_ROOT = CONTROL_ROOT / "02_Inner_Loop"
if str(INNER_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(INNER_LOOP_ROOT))

from linearisation import LinearModel, linearise_operating_point, linearise_trim  # noqa: E402
from prim_cat import PrimitiveDefinition  # noqa: E402
from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE  # noqa: E402
from trim_solver import TrimTarget  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) LQR linearisation contract
# 2) Primitive reference construction
# 3) Linearisation and controllability helpers
# 4) Serialisation helpers
# =============================================================================


# =============================================================================
# 1) LQR Linearisation Contract
# =============================================================================
LQR_LINEARISATION_VERSION = "reduced_order_lqr_linearisation_v1"
LQR_LOCAL_OPERATING_POINT_VERSION = "gain_scheduled_passive_speed_operating_point_v2"
LQR_LOCAL_OPERATING_SPEED_GRID_M_S = (
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
)
LQR_FEASIBLE_TRIM_SPEED_GRID_M_S = (
    4.8,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
)
LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S = 5.0
LQR_STATE_MASK = (
    "phi",
    "theta",
    "psi",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "delta_a",
    "delta_e",
    "delta_r",
)
ZERO_POSITION_GAIN_STATES = ("x_w", "y_w", "z_w")


@dataclass(frozen=True)
class LQRReference:
    primitive_id: str
    reference_id: str
    reference_state_vector: tuple[float, ...]
    reference_command_vector: tuple[float, float, float]
    linearisation_source: str
    reference_note: str


@dataclass(frozen=True)
class LQRLinearisation:
    primitive_id: str
    linearisation_id: str
    linearisation_source: str
    reference: LQRReference
    state_mask: tuple[str, ...]
    zero_position_gain_states: tuple[str, ...]
    a_full: tuple[tuple[float, ...], ...]
    b_full: tuple[tuple[float, ...], ...]
    a_reduced: tuple[tuple[float, ...], ...]
    b_reduced: tuple[tuple[float, ...], ...]
    full_controllability_rank: int
    full_state_size: int
    reduced_controllability_rank: int
    reduced_state_size: int
    finite_ab_check: str
    reduced_order_lqr: bool = True
    gain_expansion_policy: str = "expand_reduced_gain_to_3x15_with_zero_position_gains"


# =============================================================================
# 2) Primitive Reference Construction
# =============================================================================
def build_lqr_reference(
    primitive: PrimitiveDefinition,
    *,
    trim_model: LinearModel,
    local_reference_speed_m_s: float,
    reference_pitch_bias_rad: float = 0.0,
    reference_bank_bias_rad: float = 0.0,
    reference_roll_rate_bias_rad_s: float = 0.0,
    reference_speed_bias_m_s: float = 0.0,
) -> LQRReference:
    """Return a local reference state and nominal command for one primitive.

    The active LQR implementation is gain-scheduled by local entry speed for
    model validity.  Longitudinal speed is not a hard tracking objective in the
    primitive command law, and speed-reference bias is retained only as a
    backwards-compatible audit field.
    """

    model = trim_model
    x_ref = np.asarray(model.x_trim, dtype=float).reshape(STATE_SIZE).copy()
    u_ref = np.asarray(model.u_trim, dtype=float).reshape(3).copy()
    requested_speed = float(local_reference_speed_m_s)
    x_ref = _state_with_body_speed_preserving_alpha(x_ref, requested_speed)
    note = f"local_speed_operating_reference_{requested_speed:.3f}_m_s"

    if primitive.primitive_id == "glide":
        x_ref[STATE_INDEX["theta"]] += -0.03
        note = "trim_with_small_glide_pitch_bias"
    elif primitive.primitive_id == "recovery":
        x_ref[STATE_INDEX["phi"]] = 0.0
        x_ref[STATE_INDEX["theta"]] += 0.02
        note = "trim_with_level_attitude_recovery_bias"
    elif primitive.primitive_id == "lift_entry":
        x_ref[STATE_INDEX["theta"]] += 0.025
        note = "trim_with_small_lift_entry_pitch_bias"
    elif primitive.primitive_id == "lift_dwell_arc":
        x_ref[STATE_INDEX["phi"]] = 0.22
        x_ref[STATE_INDEX["theta"]] += 0.015
        note = "trim_with_moderate_bank_dwell_bias"
    elif primitive.primitive_id == "mild_turn_left":
        x_ref[STATE_INDEX["phi"]] = -0.20
        note = "trim_with_left_mild_turn_bank_bias"
    elif primitive.primitive_id == "mild_turn_right":
        x_ref[STATE_INDEX["phi"]] = 0.20
        note = "trim_with_right_mild_turn_bank_bias"
    elif primitive.primitive_id == "energy_retaining_bank":
        x_ref[STATE_INDEX["phi"]] = 0.16
        x_ref[STATE_INDEX["theta"]] += -0.015
        note = "trim_with_energy_retaining_bank_bias"
    elif primitive.primitive_id == "safe_exit_or_recovery_handoff":
        x_ref[STATE_INDEX["phi"]] = 0.0
        x_ref[STATE_INDEX["theta"]] += 0.03
        note = "trim_with_safe_exit_recovery_bias"
    # Archive compatibility only. These retired launch_capture_* IDs are not
    # returned by the active primitive catalogue and must not enter new evidence.
    elif primitive.primitive_id == "launch_capture_glide_stabilise":
        x_ref[STATE_INDEX["phi"]] = 0.0
        x_ref[STATE_INDEX["theta"]] += -0.04
        note = "trim_with_launch_capture_glide_stabilise_bias"
    elif primitive.primitive_id == "launch_capture_lift_seek":
        x_ref[STATE_INDEX["phi"]] = 0.0
        x_ref[STATE_INDEX["theta"]] += 0.045
        note = "trim_with_launch_capture_lift_seek_bias"
    elif primitive.primitive_id == "launch_capture_energy_build":
        x_ref[STATE_INDEX["phi"]] = 0.0
        x_ref[STATE_INDEX["theta"]] += -0.065
        note = "trim_with_launch_capture_energy_build_bias"
    elif primitive.primitive_id == "launch_capture_shallow_left":
        x_ref[STATE_INDEX["phi"]] = -0.20
        x_ref[STATE_INDEX["theta"]] += -0.015
        note = "trim_with_launch_capture_shallow_left_bias"
    elif primitive.primitive_id == "launch_capture_shallow_right":
        x_ref[STATE_INDEX["phi"]] = 0.20
        x_ref[STATE_INDEX["theta"]] += -0.015
        note = "trim_with_launch_capture_shallow_right_bias"
    elif primitive.primitive_id == "launch_capture_safe_handoff":
        x_ref[STATE_INDEX["phi"]] = 0.0
        x_ref[STATE_INDEX["theta"]] += 0.035
        note = "trim_with_launch_capture_safe_handoff_bias"

    pitch_bias = float(reference_pitch_bias_rad)
    bank_bias = float(reference_bank_bias_rad)
    roll_rate_bias = float(reference_roll_rate_bias_rad_s)
    requested_speed_bias = float(reference_speed_bias_m_s)
    if pitch_bias or bank_bias or roll_rate_bias or requested_speed_bias:
        x_ref[STATE_INDEX["theta"]] += pitch_bias
        x_ref[STATE_INDEX["phi"]] += bank_bias
        x_ref[STATE_INDEX["p"]] += roll_rate_bias
        note = (
            f"{note}_plus_tuned_reference_bias"
            f"_pitch_{pitch_bias:+.4f}_bank_{bank_bias:+.4f}"
            f"_rollrate_{roll_rate_bias:+.4f}"
            f"_speed_bias_ignored_{requested_speed_bias:+.4f}"
        )

    reference_id = _stable_id(
        "lqr_ref",
        primitive.primitive_id,
        x_ref.tolist(),
        u_ref.tolist(),
        note,
    )
    return LQRReference(
        primitive_id=primitive.primitive_id,
        reference_id=reference_id,
        reference_state_vector=tuple(float(value) for value in x_ref),
        reference_command_vector=tuple(float(value) for value in u_ref),
        linearisation_source=LQR_LOCAL_OPERATING_POINT_VERSION,
        reference_note=note,
    )


# =============================================================================
# 3) Linearisation and Controllability Helpers
# =============================================================================
def build_lqr_linearisation(
    primitive: PrimitiveDefinition,
    *,
    trim_model: LinearModel | None = None,
    local_reference_speed_m_s: float | None = None,
    reference_pitch_bias_rad: float = 0.0,
    reference_bank_bias_rad: float = 0.0,
    reference_roll_rate_bias_rad_s: float = 0.0,
    reference_speed_bias_m_s: float = 0.0,
) -> LQRLinearisation:
    """Build the active full/reduced linear model used for LQR synthesis."""

    if local_reference_speed_m_s is None and trim_model is None:
        raise ValueError("build_lqr_linearisation_requires_local_reference_speed_m_s")
    requested_speed = (
        float(local_reference_speed_m_s)
        if local_reference_speed_m_s is not None
        else float(np.linalg.norm(np.asarray(trim_model.x_trim, dtype=float)[6:9]))
    )
    model = trim_model or local_linear_model_for_speed(requested_speed)
    reference = build_lqr_reference(
        primitive,
        trim_model=model,
        local_reference_speed_m_s=requested_speed,
        reference_pitch_bias_rad=reference_pitch_bias_rad,
        reference_bank_bias_rad=reference_bank_bias_rad,
        reference_roll_rate_bias_rad_s=reference_roll_rate_bias_rad_s,
        reference_speed_bias_m_s=reference_speed_bias_m_s,
    )
    a_full = np.asarray(model.a, dtype=float).reshape(STATE_SIZE, STATE_SIZE)
    b_full = np.asarray(model.b, dtype=float).reshape(STATE_SIZE, 3)
    finite_status = (
        "finite"
        if np.all(np.isfinite(a_full)) and np.all(np.isfinite(b_full))
        else "nonfinite"
    )
    mask_indices = tuple(STATE_INDEX[name] for name in LQR_STATE_MASK)
    a_reduced = a_full[np.ix_(mask_indices, mask_indices)]
    b_reduced = b_full[np.ix_(mask_indices, range(3))]
    full_rank = controllability_rank(a_full, b_full)
    reduced_rank = controllability_rank(a_reduced, b_reduced)
    linearisation_id = _stable_id(
        "lqr_lin",
        primitive.primitive_id,
        reference.reference_id,
        _rounded_matrix_payload(a_reduced),
        _rounded_matrix_payload(b_reduced),
        LQR_STATE_MASK,
    )
    return LQRLinearisation(
        primitive_id=primitive.primitive_id,
        linearisation_id=linearisation_id,
        linearisation_source=reference.linearisation_source,
        reference=reference,
        state_mask=LQR_STATE_MASK,
        zero_position_gain_states=ZERO_POSITION_GAIN_STATES,
        a_full=_tuple_matrix(a_full),
        b_full=_tuple_matrix(b_full),
        a_reduced=_tuple_matrix(a_reduced),
        b_reduced=_tuple_matrix(b_reduced),
        full_controllability_rank=int(full_rank),
        full_state_size=STATE_SIZE,
        reduced_controllability_rank=int(reduced_rank),
        reduced_state_size=len(LQR_STATE_MASK),
        finite_ab_check=finite_status,
    )


def nearest_lqr_operating_speed_m_s(speed_m_s: float) -> float:
    """Return the scheduled local speed-grid point closest to the current state."""

    speed = float(speed_m_s)
    if not np.isfinite(speed):
        raise ValueError("local operating speed must be finite.")
    grid = np.asarray(LQR_LOCAL_OPERATING_SPEED_GRID_M_S, dtype=float)
    return float(grid[int(np.argmin(np.abs(grid - speed)))])


def lqr_speed_bin_id(speed_m_s: float) -> str:
    """Return the stable audit label for a local LQR speed bin."""

    speed = nearest_lqr_operating_speed_m_s(float(speed_m_s))
    return f"speed_bin_{speed:.1f}_m_s".replace(".", "p")


def local_speed_from_state_vector(state_vector: np.ndarray) -> float:
    """Return the body-frame speed magnitude used for local LQR scheduling."""

    state = np.asarray(state_vector, dtype=float).reshape(STATE_SIZE)
    return float(np.linalg.norm(state[6:9]))


@lru_cache(maxsize=64)
def local_linear_model_for_speed(speed_m_s: float) -> LinearModel:
    """Return a cached local operating-point linear model for one speed-grid point."""

    operating_speed = nearest_lqr_operating_speed_m_s(float(speed_m_s))
    base_trim_speed = _nearest_feasible_trim_speed_m_s(operating_speed)
    base = _feasible_trim_model_for_speed(base_trim_speed)
    x_op = _state_with_body_speed_preserving_alpha(np.asarray(base.x_trim, dtype=float), operating_speed)
    target = TrimTarget(speed_m_s=operating_speed)
    return linearise_operating_point(
        x_operating=x_op,
        u_operating=np.asarray(base.u_trim, dtype=float),
        target=target,
    )


@lru_cache(maxsize=32)
def _feasible_trim_model_for_speed(speed_m_s: float) -> LinearModel:
    speed = _nearest_feasible_trim_speed_m_s(float(speed_m_s))
    return linearise_trim(target=TrimTarget(speed_m_s=speed))


def _nearest_feasible_trim_speed_m_s(speed_m_s: float) -> float:
    grid = np.asarray(LQR_FEASIBLE_TRIM_SPEED_GRID_M_S, dtype=float)
    return float(grid[int(np.argmin(np.abs(grid - float(speed_m_s))))])


def _state_with_body_speed_preserving_alpha(state: np.ndarray, speed_m_s: float) -> np.ndarray:
    result = np.asarray(state, dtype=float).reshape(STATE_SIZE).copy()
    speed = float(speed_m_s)
    alpha = float(np.arctan2(result[STATE_INDEX["w"]], result[STATE_INDEX["u"]]))
    result[STATE_INDEX["u"]] = speed * float(np.cos(alpha))
    result[STATE_INDEX["v"]] = 0.0
    result[STATE_INDEX["w"]] = speed * float(np.sin(alpha))
    return result


def controllability_rank(a_matrix: np.ndarray, b_matrix: np.ndarray) -> int:
    """Return the numerical controllability rank for an LTI pair."""

    a = np.asarray(a_matrix, dtype=float)
    b = np.asarray(b_matrix, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != a.shape[1] or b.shape[0] != a.shape[0]:
        raise ValueError("A and B must be compatible two-dimensional matrices.")
    blocks = []
    current = b.copy()
    for _ in range(a.shape[0]):
        blocks.append(current)
        current = a @ current
    controllability = np.concatenate(blocks, axis=1)
    return int(np.linalg.matrix_rank(controllability, tol=1e-8))


def reduced_state_indices() -> tuple[int, ...]:
    return tuple(STATE_INDEX[name] for name in LQR_STATE_MASK)


# =============================================================================
# 4) Serialisation Helpers
# =============================================================================
def lqr_linearisation_row(linearisation: LQRLinearisation) -> dict[str, object]:
    row = asdict(linearisation)
    row.pop("a_full")
    row.pop("b_full")
    row.pop("a_reduced")
    row.pop("b_reduced")
    row["state_mask"] = json.dumps(linearisation.state_mask, separators=(",", ":"))
    row["zero_position_gain_states"] = json.dumps(
        linearisation.zero_position_gain_states,
        separators=(",", ":"),
    )
    row["reference_state_vector"] = json.dumps(
        list(linearisation.reference.reference_state_vector),
        separators=(",", ":"),
    )
    row["reference_command_vector"] = json.dumps(
        list(linearisation.reference.reference_command_vector),
        separators=(",", ":"),
    )
    row["reference_id"] = linearisation.reference.reference_id
    row["reference_note"] = linearisation.reference.reference_note
    return row


def _tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=float))


def _rounded_matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(matrix, dtype=float), decimals=10).tolist()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]
    return f"{prefix}_{digest}"
