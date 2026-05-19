from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds
from dense_archive_schema import (
    BASELINE_FAMILIES,
    DENSE_TURNING_FAMILIES,
    DIRECTION_SIGNS,
    DRY_RUN_CANDIDATE_COLUMNS,
    ENVIRONMENT_ROLE_BY_FAMILY,
    SAMPLING_RANGES,
    SAMPLING_STRATA_COLUMNS,
    START_CLASSES,
    START_CLASS_FRACTIONS,
    START_STATE_MANIFEST_COLUMNS,
    TARGET_LADDER_DEG,
    TEST_ENVIRONMENT_MODE,
    DenseArchivePlanConfig,
    build_target_direction_plan,
)
from updraft_models import FOUR_FAN_CENTERS_XY, SINGLE_FAN_CENTER_XY


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Sampling Count Helpers
# 2) Sampling Strata Summary
# 3) Pilot Start-State Generation
# 4) Dry-Run Candidate Inventory
# =============================================================================


# =============================================================================
# 1) Sampling Count Helpers
# =============================================================================
def _start_class_counts(sample_count: int) -> dict[str, int]:
    raw_counts = {name: float(sample_count) * START_CLASS_FRACTIONS[name] for name in START_CLASSES}
    counts = {name: int(np.floor(raw_counts[name])) for name in START_CLASSES}
    remainder = int(sample_count) - sum(counts.values())
    order = sorted(
        START_CLASSES,
        key=lambda name: (raw_counts[name] - counts[name], START_CLASS_FRACTIONS[name]),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _pilot_group_count() -> int:
    return (
        len(DENSE_TURNING_FAMILIES) * len(TARGET_LADDER_DEG) * len(DIRECTION_SIGNS)
        + len(BASELINE_FAMILIES) * len(DIRECTION_SIGNS)
    )


def _pilot_class_counts(config: DenseArchivePlanConfig) -> dict[str, int]:
    per_group = _start_class_counts(int(config.pilot_start_states_per_family_target_direction))
    return {name: int(per_group[name] * _pilot_group_count()) for name in START_CLASSES}


def _range_text(values: tuple[float, float]) -> str:
    return f"[{float(values[0]):.3g}, {float(values[1]):.3g}]"


# =============================================================================
# 2) Sampling Strata Summary
# =============================================================================
def build_sampling_strata_summary(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Return explicit sampling ranges and requested fractions."""

    pilot_counts = _pilot_class_counts(config)
    rows: list[dict[str, object]] = []
    for start_class in START_CLASSES:
        ranges = SAMPLING_RANGES[start_class]
        rows.append(
            {
                "start_class": start_class,
                "fraction_requested": START_CLASS_FRACTIONS[start_class],
                "pilot_sample_count": pilot_counts[start_class],
                "minimum_archive_count_reference": int(
                    round(
                        float(config.minimum_start_states_per_family_target_direction)
                        * START_CLASS_FRACTIONS[start_class]
                    )
                ),
                "x_range_m": _range_text(ranges["x_range_m"]),
                "y_range_m": _range_text(ranges["y_range_m"]),
                "z_range_m": _range_text(ranges["z_range_m"]),
                "speed_range_m_s": _range_text(ranges["speed_range_m_s"]),
                "phi_range_deg": _range_text(ranges["phi_range_deg"]),
                "theta_range_deg": _range_text(ranges["theta_range_deg"]),
                "psi_range_deg": _range_text(ranges["psi_range_deg"]),
                "p_range_rad_s": _range_text(ranges["p_range_rad_s"]),
                "q_range_rad_s": _range_text(ranges["q_range_rad_s"]),
                "r_range_rad_s": _range_text(ranges["r_range_rad_s"]),
                "updraft_radius_range_m": _range_text(ranges["updraft_radius_range_m"]),
                "special_rule": ranges["special_rule"],
                "test_environment_mode": config.test_environment_mode,
                "no_rollout_performed": True,
            }
        )
    return pd.DataFrame(rows, columns=SAMPLING_STRATA_COLUMNS)


# =============================================================================
# 3) Pilot Start-State Generation
# =============================================================================
def build_start_state_manifest(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Return deterministic pilot start states for the Phase B scaffold."""

    master_rng = np.random.default_rng(int(config.random_seed))
    plan = build_target_direction_plan(config)
    rows: list[dict[str, object]] = []
    sample_index = 0
    for plan_row in plan.to_dict(orient="records"):
        counts = _start_class_counts(int(config.pilot_start_states_per_family_target_direction))
        for start_class in START_CLASSES:
            for class_index in range(counts[start_class]):
                row_seed = int(master_rng.integers(1, np.iinfo(np.int32).max))
                sample_index += 1
                rows.append(
                    _build_start_state_row(
                        sample_index=sample_index,
                        seed=row_seed,
                        start_class=start_class,
                        class_index=class_index,
                        class_count=counts[start_class],
                        plan_row=plan_row,
                        config=config,
                    )
                )
    frame = pd.DataFrame(rows, columns=START_STATE_MANIFEST_COLUMNS)
    if not bool(frame["true_safe_start"].all()):
        raise RuntimeError("pilot start-state generation produced an unsafe row.")
    return frame


def _build_start_state_row(
    *,
    sample_index: int,
    seed: int,
    start_class: str,
    class_index: int,
    class_count: int,
    plan_row: dict[str, object],
    config: DenseArchivePlanConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    center = _reference_center(start_class, rng)
    x_w, y_w, radius, azimuth, sector = _sample_position_xy(
        start_class=start_class,
        rng=rng,
        center=center,
        class_index=class_index,
        class_count=class_count,
    )
    ranges = SAMPLING_RANGES[start_class]
    z_w = _uniform(rng, ranges["z_range_m"])
    speed = _uniform(rng, ranges["speed_range_m_s"])
    phi = np.deg2rad(_uniform(rng, ranges["phi_range_deg"]))
    theta = np.deg2rad(_uniform(rng, ranges["theta_range_deg"]))
    psi = np.deg2rad(_uniform(rng, ranges["psi_range_deg"]))
    p_rate = _uniform(rng, ranges["p_range_rad_s"])
    q_rate = _uniform(rng, ranges["q_range_rad_s"])
    r_rate = _uniform(rng, ranges["r_range_rad_s"])
    position = np.array([x_w, y_w, z_w], dtype=float)
    target = _target_value(plan_row["target_heading_deg"])

    # The scaffold records body-axis speed without projecting through yaw; no
    # dynamics or world-velocity replay is performed in this planning pass.
    return {
        "sample_id": f"s{int(config.run_id):03d}_pilot_{sample_index:06d}",
        "seed": int(seed),
        "sampling_round": "pilot_round_0",
        "start_class": start_class,
        "family": plan_row["family"],
        "target_heading_deg": target,
        "direction_sign": int(plan_row["direction_sign"]),
        "environment_role": plan_row["environment_role"],
        "test_environment_mode": config.test_environment_mode,
        "count_basis": plan_row["count_basis"],
        "x_w_m": x_w,
        "y_w_m": y_w,
        "z_w_m": z_w,
        "speed_m_s": speed,
        "phi_rad": float(phi),
        "theta_rad": float(theta),
        "psi_rad": float(psi),
        "u_m_s": speed,
        "v_m_s": 0.0,
        "w_m_s": 0.0,
        "p_rad_s": p_rate,
        "q_rad_s": q_rate,
        "r_rad_s": r_rate,
        "updraft_config": "none",
        "updraft_center_x_m": float(center[0]),
        "updraft_center_y_m": float(center[1]),
        "updraft_relative_radius_m": radius,
        "updraft_relative_azimuth_rad": azimuth,
        "updraft_relative_height_m": float(z_w),
        "updraft_sector_label": sector,
        "left_wing_lift_exposure_preference": _wing_preference(azimuth, left=True),
        "right_wing_lift_exposure_preference": _wing_preference(azimuth, left=False),
        "wing_exposure_bookkeeping_status": "geometry_only_w0_no_wind_query",
        "true_safe_start": bool(inside_bounds(position, TRUE_SAFE_BOUNDS)),
        "start_generation_status": "generated_inside_true_safe_bounds",
        "no_rollout_performed": True,
    }


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _reference_center(start_class: str, rng: np.random.Generator) -> tuple[float, float]:
    if start_class == "lift_sector":
        centers = (SINGLE_FAN_CENTER_XY,) + tuple(FOUR_FAN_CENTERS_XY)
        return centers[int(rng.integers(0, len(centers)))]
    return SINGLE_FAN_CENTER_XY


def _sample_position_xy(
    *,
    start_class: str,
    rng: np.random.Generator,
    center: tuple[float, float],
    class_index: int,
    class_count: int,
) -> tuple[float, float, float, float, str]:
    if start_class == "lift_sector":
        required_edge_rows = int(ceil(0.5 * float(class_count)))
        edge_or_ring = class_index < required_edge_rows
        radius_range = (0.65, 1.15) if edge_or_ring else (0.0, 0.45)
        radius = _uniform(rng, radius_range)
        azimuth = _uniform(rng, (-np.pi, np.pi))
        x_w = float(center[0] + radius * np.cos(azimuth))
        y_w = float(center[1] + radius * np.sin(azimuth))
        x_w, y_w = _clip_xy_to_true_safe(x_w, y_w)
        actual_radius = float(np.hypot(x_w - center[0], y_w - center[1]))
        actual_azimuth = float(np.arctan2(y_w - center[1], x_w - center[0]))
        sector = "edge_ring_reference" if edge_or_ring else "core_reference"
        return x_w, y_w, actual_radius, actual_azimuth, sector

    if start_class == "random_stress":
        x_w, y_w = _sample_boundary_near_xy(rng)
    else:
        ranges = SAMPLING_RANGES[start_class]
        x_w = _uniform(rng, ranges["x_range_m"])
        y_w = _uniform(rng, ranges["y_range_m"])
    radius = float(np.hypot(x_w - center[0], y_w - center[1]))
    azimuth = float(np.arctan2(y_w - center[1], x_w - center[0]))
    return x_w, y_w, radius, azimuth, f"{start_class}_reference"


def _clip_xy_to_true_safe(x_w: float, y_w: float) -> tuple[float, float]:
    # A small margin avoids edge equality from roundoff while staying in the
    # public z-up true-safe box used by the existing primitive pipeline.
    margin = 0.05
    x_min, x_max = TRUE_SAFE_BOUNDS.x_w_m
    y_min, y_max = TRUE_SAFE_BOUNDS.y_w_m
    return (
        float(np.clip(x_w, x_min + margin, x_max - margin)),
        float(np.clip(y_w, y_min + margin, y_max - margin)),
    )


def _sample_boundary_near_xy(rng: np.random.Generator) -> tuple[float, float]:
    side = int(rng.integers(0, 4))
    x_min, x_max = TRUE_SAFE_BOUNDS.x_w_m
    y_min, y_max = TRUE_SAFE_BOUNDS.y_w_m
    if side == 0:
        x_w = _uniform(rng, (x_min + 0.05, x_min + 0.45))
        y_w = _uniform(rng, (y_min + 0.10, y_max - 0.10))
    elif side == 1:
        x_w = _uniform(rng, (x_max - 0.45, x_max - 0.05))
        y_w = _uniform(rng, (y_min + 0.10, y_max - 0.10))
    elif side == 2:
        x_w = _uniform(rng, (x_min + 0.10, x_max - 0.10))
        y_w = _uniform(rng, (y_min + 0.05, y_min + 0.45))
    else:
        x_w = _uniform(rng, (x_min + 0.10, x_max - 0.10))
        y_w = _uniform(rng, (y_max - 0.45, y_max - 0.05))
    return float(x_w), float(y_w)


def _target_value(value: object) -> object:
    if pd.isna(value) or value == "":
        return ""
    return float(value)


def _target_key(value: object) -> str:
    if pd.isna(value) or value == "":
        return "none"
    return f"{float(value):.1f}"


def _wing_preference(azimuth_rad: float, *, left: bool) -> str:
    if abs(float(np.sin(azimuth_rad))) < 0.10:
        return "balanced_reference"
    left_has_lift = float(np.sin(azimuth_rad)) > 0.0
    if left:
        return "higher_reference_lift" if left_has_lift else "lower_reference_lift"
    return "lower_reference_lift" if left_has_lift else "higher_reference_lift"


# =============================================================================
# 4) Dry-Run Candidate Inventory
# =============================================================================
def build_dry_run_candidate_inventory(
    config: DenseArchivePlanConfig,
    start_states: pd.DataFrame,
) -> pd.DataFrame:
    """Return pilot candidate rows without replaying dynamics."""

    plan = build_target_direction_plan(config)
    plan_rows = {
        (row["family"], _target_key(row["target_heading_deg"]), int(row["direction_sign"])): row
        for row in plan.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for row in start_states.to_dict(orient="records"):
        key = (row["family"], _target_key(row["target_heading_deg"]), int(row["direction_sign"]))
        plan_row = plan_rows[key]
        candidate_id = _candidate_id(config, row)
        rows.append(
            {
                "candidate_id": candidate_id,
                "sample_id": row["sample_id"],
                "seed": int(row["seed"]),
                "sampling_round": row["sampling_round"],
                "family": row["family"],
                "target_heading_deg": _target_value(row["target_heading_deg"]),
                "direction_sign": int(row["direction_sign"]),
                "start_class": row["start_class"],
                "environment_role": row["environment_role"],
                "test_environment_mode": row["test_environment_mode"],
                "count_basis": row["count_basis"],
                "planned_min_trial_count": int(plan_row["planned_min_trial_count"]),
                "planned_target_trial_count": int(plan_row["planned_target_trial_count"]),
                "pilot_trial_count": int(plan_row["pilot_trial_count"]),
                "planned_replay_status": "not_replayed_in_this_task",
                "planned_result_path": (
                    "03_Control/05_Results/10_dense_archive_planning/"
                    f"{int(config.run_id):03d}/not_replayed/{candidate_id}.csv"
                ),
                "no_rollout_performed": True,
                "notes": "planning inventory only; no primitive replay or rollout performed",
            }
        )
    return pd.DataFrame(rows, columns=DRY_RUN_CANDIDATE_COLUMNS)


def _candidate_id(config: DenseArchivePlanConfig, row: dict[str, object]) -> str:
    target = _target_key(row["target_heading_deg"])
    direction = f"d{int(row['direction_sign']):+d}".replace("+", "p").replace("-", "m")
    return (
        f"s{int(config.run_id):03d}_{row['family']}_{target}_{direction}_{row['sample_id']}"
        .replace(".", "p")
    )
