from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Library constants
# 2) Data containers
# 3) Geometry and envelope metrics
# 4) Candidate classification
# =============================================================================


# =============================================================================
# 1) Library Constants
# =============================================================================
TARGET_LADDER_DEG = (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
PRIMITIVE_FAMILIES = (
    "glide",
    "recovery",
    "mild_bank",
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
UPDRAFT_CONFIGS = ("none", "U1_single_fan", "U4_four_fan")
WIND_FIDELITIES = ("W0", "W1", "W2", "W3")
START_CONDITIONS = ("favourable", "mid_arena", "lift_sector", "random_stress")
CANDIDATE_CLASSES = (
    "w0_standalone_commandable",
    "w0_updraft_pending_target_candidate",
    "updraft_assisted_commandable",
    "boundary_evidence",
)
EVALUATION_STATUSES = (
    "evaluated",
    "not_evaluated_model_missing",
    "model_unavailable",
)
Z_OUTLET_M = 0.330
TRUE_SAFE_BOUNDS_M = {
    "x_w": (1.2, 6.6),
    "y_w": (0.0, 4.4),
    "z_w": (0.4, 3.5),
}


# =============================================================================
# 2) Data Containers
# =============================================================================
@dataclass(frozen=True)
class PrimitiveLibraryConfig:
    dt_s: float = 0.02
    run_id: int = 1
    targets_deg: tuple[float, ...] = (15.0, 30.0)
    families: tuple[str, ...] = PRIMITIVE_FAMILIES
    updraft_configs: tuple[str, ...] = UPDRAFT_CONFIGS
    wind_fidelities: tuple[str, ...] = ("W0", "W1", "W2")
    start_conditions: tuple[str, ...] = ("favourable", "mid_arena")
    direction_signs: tuple[int, ...] = (1,)
    z_outlet_m: float = Z_OUTLET_M


@dataclass(frozen=True)
class PrimitiveCandidateSpec:
    primitive_id: str
    parent_primitive_id: str
    variant_id: str
    family: str
    target_heading_deg: Optional[float]
    updraft_config: str
    wind_fidelity: str
    start_condition: str
    direction_sign: int
    horizon_s: float = 0.60


@dataclass(frozen=True)
class PrimitiveEvidenceRow:
    primitive_id: str
    parent_primitive_id: str
    variant_id: str
    envelope_group_id: str
    target_heading_deg: Optional[float]
    family: str
    updraft_config: str
    wind_fidelity: str
    start_condition: str
    environment_label: str
    direction_sign: int
    evaluation_status: str
    wind_model_available: bool
    wind_model_name: str
    wind_model_source: str
    z_outlet_m: float
    z_fan_min_m: float
    z_fan_max_m: float
    terminal_heading_change_deg: float
    terminal_heading_error_deg: float
    heading_band_pass: bool
    path_length_xy_m: float
    path_length_3d_m: float
    forward_displacement_m: float
    lateral_displacement_m: float
    xy_bounding_box_area_m2: float
    turn_footprint_proxy_m2: float
    entry_clearance_required_x_plus_m: float
    entry_clearance_required_x_minus_m: float
    entry_clearance_required_y_plus_m: float
    entry_clearance_required_y_minus_m: float
    floor_margin_required_m: float
    ceiling_margin_required_m: float
    margin_consumption_x_m: float
    margin_consumption_y_m: float
    margin_consumption_z_m: float
    speed_min_m_s: float
    terminal_speed_m_s: float
    specific_energy_initial_m: float
    specific_energy_terminal_m: float
    energy_residual_m: float
    alpha_max_deg: float
    beta_max_deg: float
    rate_max_rad_s: float
    saturation_fraction: float
    true_safe_trajectory: bool
    min_true_margin_m: float
    floor_margin_min_m: float
    ceiling_margin_min_m: float
    recovery_class: str
    candidate_class: str
    failure_label: str
    active_limiting_mechanism: str
    wind_query_region: str
    lift_belief_condition: str
    governor_condition: str
    within_existing_envelope: bool
    nearest_existing_primitive_id: str
    normalised_distance_to_nearest_envelope: float
    coverage_region_id: str
    marginal_coverage_gain: float
    library_growth_trigger: bool
    growth_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# =============================================================================
# 3) Geometry and Envelope Metrics
# =============================================================================
def target_heading_band_deg(target_heading_deg: float) -> tuple[float, float]:
    """Return the accepted terminal-heading band in degrees."""

    target = float(target_heading_deg)
    tolerance = max(2.0, 0.10 * abs(target))
    return target - tolerance, target + tolerance


def _finite_positions(position_w_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(position_w_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("position_w_m must have shape (N, 3).")
    return positions[np.all(np.isfinite(positions), axis=1)]


def path_metrics(position_w_m: np.ndarray) -> dict[str, float]:
    """Return footprint and displacement metrics for public z-up positions."""

    positions = _finite_positions(position_w_m)
    if positions.shape[0] < 2:
        return {
            "path_length_xy_m": 0.0,
            "path_length_3d_m": 0.0,
            "forward_displacement_m": 0.0,
            "lateral_displacement_m": 0.0,
            "xy_bounding_box_area_m2": 0.0,
            "turn_footprint_proxy_m2": 0.0,
        }
    delta = np.diff(positions, axis=0)
    xy = positions[:, :2]
    x_span = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    y_span = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    area = x_span * y_span
    return {
        "path_length_xy_m": float(np.sum(np.linalg.norm(delta[:, :2], axis=1))),
        "path_length_3d_m": float(np.sum(np.linalg.norm(delta, axis=1))),
        "forward_displacement_m": float(positions[-1, 0] - positions[0, 0]),
        "lateral_displacement_m": float(positions[-1, 1] - positions[0, 1]),
        "xy_bounding_box_area_m2": float(area),
        "turn_footprint_proxy_m2": float(max(area, x_span * 0.05)),
    }


def entry_clearance_metrics(
    position_w_m: np.ndarray,
    true_safe_bounds: object = TRUE_SAFE_BOUNDS_M,
) -> dict[str, float]:
    """Return start-state clearances required to keep the logged path true-safe."""

    positions = _finite_positions(position_w_m)
    if positions.shape[0] == 0:
        return {
            "entry_clearance_required_x_plus_m": float("nan"),
            "entry_clearance_required_x_minus_m": float("nan"),
            "entry_clearance_required_y_plus_m": float("nan"),
            "entry_clearance_required_y_minus_m": float("nan"),
            "floor_margin_required_m": float("nan"),
            "ceiling_margin_required_m": float("nan"),
            "margin_consumption_x_m": float("nan"),
            "margin_consumption_y_m": float("nan"),
            "margin_consumption_z_m": float("nan"),
        }
    start = positions[0]
    x_bounds, y_bounds, z_bounds = _bounds_tuple(true_safe_bounds)
    return {
        "entry_clearance_required_x_plus_m": float(np.max(positions[:, 0] - start[0])),
        "entry_clearance_required_x_minus_m": float(np.max(start[0] - positions[:, 0])),
        "entry_clearance_required_y_plus_m": float(np.max(positions[:, 1] - start[1])),
        "entry_clearance_required_y_minus_m": float(np.max(start[1] - positions[:, 1])),
        "floor_margin_required_m": float(np.min(positions[:, 2]) - z_bounds[0]),
        "ceiling_margin_required_m": float(z_bounds[1] - np.max(positions[:, 2])),
        "margin_consumption_x_m": float(
            max(np.max(positions[:, 0] - start[0]), np.max(start[0] - positions[:, 0]))
            / max(1e-12, min(start[0] - x_bounds[0], x_bounds[1] - start[0]))
        ),
        "margin_consumption_y_m": float(
            max(np.max(positions[:, 1] - start[1]), np.max(start[1] - positions[:, 1]))
            / max(1e-12, min(start[1] - y_bounds[0], y_bounds[1] - start[1]))
        ),
        "margin_consumption_z_m": float(
            max(np.max(positions[:, 2] - start[2]), np.max(start[2] - positions[:, 2]))
            / max(1e-12, min(start[2] - z_bounds[0], z_bounds[1] - start[2]))
        ),
    }


def classify_wind_query_region(
    z_w_m: np.ndarray,
    model_z_axis_m: np.ndarray | None,
    z_outlet_m: float = Z_OUTLET_M,
) -> str:
    """Classify outlet-relative updraft lookup height against model support."""

    z_w = np.asarray(z_w_m, dtype=float).reshape(-1)
    if z_w.size == 0 or not np.all(np.isfinite(z_w)):
        return "unknown"
    if model_z_axis_m is None:
        return "unknown"
    z_axis = np.asarray(model_z_axis_m, dtype=float).reshape(-1)
    if z_axis.size == 0 or not np.all(np.isfinite(z_axis)):
        return "unknown"
    z_fan = z_w - float(z_outlet_m)
    if np.min(z_fan) >= np.min(z_axis) and np.max(z_fan) <= np.max(z_axis):
        return "measured"
    if np.max(z_fan) < np.min(z_axis) or np.min(z_fan) > np.max(z_axis):
        return "extrapolated"
    return "clipped"


def _bounds_tuple(bounds: object) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if isinstance(bounds, dict):
        return bounds["x_w"], bounds["y_w"], bounds["z_w"]
    return bounds.x_w_m, bounds.y_w_m, bounds.z_w_m


# =============================================================================
# 4) Candidate Classification
# =============================================================================
def classify_candidate(row_inputs: dict[str, object]) -> tuple[str, str, str]:
    """Return candidate_class, failure_label, and active limiting mechanism."""

    status = str(row_inputs.get("evaluation_status", "evaluated"))
    if status != "evaluated":
        return "not_evaluated", status, "model_unavailable"

    target = row_inputs.get("target_heading_deg")
    heading_required = target is not None and np.isfinite(float(target))
    heading_pass = bool(row_inputs.get("heading_band_pass", not heading_required))
    true_safe = bool(row_inputs.get("true_safe_trajectory", False))
    finite = bool(row_inputs.get("finite_replay", True))
    terminal_speed = float(row_inputs.get("terminal_speed_m_s", np.nan))
    speed_min = float(row_inputs.get("speed_min_m_s", np.nan))
    alpha = float(row_inputs.get("alpha_max_deg", np.inf))
    beta = float(row_inputs.get("beta_max_deg", np.inf))
    rate = float(row_inputs.get("rate_max_rad_s", np.inf))
    saturation = float(row_inputs.get("saturation_fraction", np.inf))
    wind_fidelity = str(row_inputs.get("wind_fidelity", "W0"))
    lift_belief = str(row_inputs.get("lift_belief_condition", "none"))
    dry_recovery = str(row_inputs.get("recovery_class", "")) == "dry_recoverable"
    updraft_recovery = str(row_inputs.get("recovery_class", "")) == "updraft_recoverable"

    non_catastrophic = (
        np.isfinite(terminal_speed)
        and np.isfinite(speed_min)
        and terminal_speed >= 3.5
        and speed_min >= 3.0
        and alpha <= 65.0
        and beta <= 35.0
        and rate <= 6.0
        and saturation < 0.60
    )
    dry_speed = terminal_speed >= 5.0 and speed_min >= 4.0

    if not finite:
        return "boundary_evidence", "nonfinite_replay", "numerical_failure"
    if not true_safe:
        return "boundary_evidence", "true_safety_violation", "safety_limited"
    if heading_required and not heading_pass:
        return "boundary_evidence", "target_miss", "turn_authority_limited"
    if not non_catastrophic:
        return "boundary_evidence", "exposure_or_speed_limit", "exposure_limited"
    if wind_fidelity == "W0" and dry_speed and dry_recovery:
        return "w0_standalone_commandable", "success", "none"
    if wind_fidelity == "W0":
        return "w0_updraft_pending_target_candidate", "dry_recovery_pending", "updraft_condition_required"
    if updraft_recovery and lift_belief not in ("none", "missing", "model_unavailable"):
        return "updraft_assisted_commandable", "success", "none"
    return "boundary_evidence", "updraft_recovery_not_proven", "recovery_limited"
