from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds
from dense_archive_schema import (
    BASELINE_FAMILIES,
    BRANCH_DECISION_SCOPE,
    DENSE_TURNING_FAMILIES,
    DIRECTION_SIGNS,
    DRY_RUN_CANDIDATE_COLUMNS,
    ENVIRONMENT_ROLE_BY_FAMILY,
    FAN_BRANCH_METADATA,
    FAN_LAYOUTS,
    LATENCY_ACCEPTANCE_ROLE,
    LATENCY_CASE_PLANNED,
    LATENCY_MODEL_STATUS,
    NO_CROSS_BRANCH_FLAGS,
    SAMPLING_RANGES,
    SAMPLING_STRATA_COLUMNS,
    START_CLASSES,
    START_CLASS_FRACTIONS,
    START_STATE_MANIFEST_COLUMNS,
    TARGET_LADDER_DEG,
    DenseArchivePlanConfig,
    build_target_environment_plan,
    branch_start_group_count,
)
from updraft_models import FOUR_FAN_CENTERS_XY, SINGLE_FAN_CENTER_XY, load_updraft_model
from wing_wind_descriptors import wing_wind_descriptor_row


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Sampling Count Helpers
# 2) Wind Descriptor Helpers
# 3) Sampling Strata Summary
# 4) Pilot Start-State Generation
# 5) Dry-Run Candidate Inventory
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
        key=lambda name: (-(raw_counts[name] - counts[name]), START_CLASSES.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _pilot_class_counts_per_branch(config: DenseArchivePlanConfig) -> dict[str, int]:
    per_group = _start_class_counts(int(config.pilot_start_states_per_family_target_direction))
    return {name: int(per_group[name] * branch_start_group_count()) for name in START_CLASSES}


def _range_text(values: tuple[float, float]) -> str:
    return f"[{float(values[0]):.3g}, {float(values[1]):.3g}]"


def _branch_w1_values(fan_layout: str) -> dict[str, str]:
    meta = FAN_BRANCH_METADATA[fan_layout]
    return {
        "layout_branch_id": str(meta["layout_branch_id"]),
        "fan_config_id": str(meta["w1_fan_config_id"]),
        "updraft_model_id": str(meta["w1_updraft_model_id"]),
        "w0_environment_mode": str(meta["w0_environment_mode"]),
        "w1_environment_mode": str(meta["w1_environment_mode"]),
    }


# =============================================================================
# 2) Wind Descriptor Helpers
# =============================================================================
def _load_w1_wind_models() -> dict[str, object]:
    model_ids = sorted(
        {
            str(metadata["w1_updraft_model_id"])
            for metadata in FAN_BRANCH_METADATA.values()
        }
    )
    models: dict[str, object] = {}
    for model_id in model_ids:
        if model_id == "analytic_debug_proxy":
            raise ValueError(
                "analytic_debug_proxy is forbidden for dense planning descriptors."
            )
        model = load_updraft_model(model_id)
        if getattr(model, "name", "") == "analytic_debug_proxy":
            raise ValueError(
                "analytic_debug_proxy is forbidden for dense planning descriptors."
            )
        models[model_id] = model
    return models


def _model_source(wind_model: object) -> str:
    return str(getattr(wind_model, "source", "unknown_model_source"))


def _descriptor_from_state(
    *,
    row: dict[str, object],
    fan_layout: str,
    fan_config_id: str,
    environment_mode: str,
    model_id: str,
    wind_model: object | None,
    dry_air: bool,
) -> dict[str, object]:
    source = "dry_air_zero_wind" if dry_air else _model_source(wind_model)
    return wing_wind_descriptor_row(
        wind_field=None if dry_air else wind_model,
        x_w_m=float(row["x_w_m"]),
        y_w_m=float(row["y_w_m"]),
        z_w_m=float(row["z_w_m"]),
        phi_rad=float(row["phi_rad"]),
        theta_rad=float(row["theta_rad"]),
        psi_rad=float(row["psi_rad"]),
        fan_layout=fan_layout,
        fan_config_id=fan_config_id,
        environment_mode=environment_mode,
        model_id=model_id,
        model_source=source,
        dry_air=dry_air,
    )


# =============================================================================
# 3) Sampling Strata Summary
# =============================================================================
def build_sampling_strata_summary(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Return branch-local sampling ranges and requested fractions."""

    pilot_counts = _pilot_class_counts_per_branch(config)
    rows: list[dict[str, object]] = []
    for fan_layout in FAN_LAYOUTS:
        branch = _branch_w1_values(fan_layout)
        for start_class in START_CLASSES:
            ranges = SAMPLING_RANGES[start_class]
            rows.append(
                {
                    "fan_layout": fan_layout,
                    "layout_branch_id": branch["layout_branch_id"],
                    "fan_config_id": branch["fan_config_id"],
                    "updraft_model_id": branch["updraft_model_id"],
                    "start_class": start_class,
                    "fraction_requested": START_CLASS_FRACTIONS[start_class],
                    "pilot_sample_count": pilot_counts[start_class],
                    "floor_archive_count_reference": int(
                        round(
                            float(config.w1_floor_start_states_per_family_target_direction)
                            * START_CLASS_FRACTIONS[start_class]
                        )
                    ),
                    "target_archive_count_reference": int(
                        round(
                            float(config.w1_target_start_states_per_family_target_direction)
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
                    "special_rule": _special_rule(start_class, str(ranges["special_rule"])),
                    "branch_layout_note": f"{fan_layout} samples use branch-local fan geometry",
                    "layout_specific_sampling_required": True,
                    "no_cross_branch_merge": True,
                    "no_rollout_performed": True,
                }
            )
    return pd.DataFrame(rows, columns=SAMPLING_STRATA_COLUMNS)


def _special_rule(start_class: str, base_rule: str) -> str:
    if start_class == "lift_sector":
        return f"{base_rule}; at least half of pilot samples are edge/ring labelled within branch"
    return base_rule


# =============================================================================
# 4) Pilot Start-State Generation
# =============================================================================
def build_start_state_manifest(config: DenseArchivePlanConfig) -> pd.DataFrame:
    """Return branch-local exact starts for the paired W0/W1 scaffold."""

    master_rng = np.random.default_rng(int(config.random_seed))
    plan = _start_state_group_plan(config)
    wind_models = _load_w1_wind_models()
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
                        wind_models=wind_models,
                    )
                )
    frame = pd.DataFrame(rows, columns=START_STATE_MANIFEST_COLUMNS)
    if not bool(frame["true_safe_start"].all()):
        raise RuntimeError("pilot start-state generation produced an unsafe row.")
    return frame


def _start_state_group_plan(config: DenseArchivePlanConfig) -> pd.DataFrame:
    plan = build_target_environment_plan(config)
    w1_rows = plan[plan["test_environment_mode"].astype(str).str.startswith("W1_")].copy()
    return w1_rows.reset_index(drop=True)


def _build_start_state_row(
    *,
    sample_index: int,
    seed: int,
    start_class: str,
    class_index: int,
    class_count: int,
    plan_row: dict[str, object],
    config: DenseArchivePlanConfig,
    wind_models: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    fan_layout = str(plan_row["fan_layout"])
    branch = _branch_w1_values(fan_layout)
    center = _reference_center(fan_layout, start_class, rng)
    x_w, y_w, radius, azimuth, sector = _sample_position_xy(
        fan_layout=fan_layout,
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
    paired_key = _paired_sample_key(config, sample_index, plan_row, start_class)
    paired_modes = f"{branch['w0_environment_mode']};{branch['w1_environment_mode']}"

    # Start rows are exact branch-local states. They are not replay outcomes and
    # deliberately store body-axis speed with yaw logged separately.
    row = {
        "sample_id": f"s{int(config.run_id):03d}_{fan_layout}_pilot_{sample_index:06d}",
        "paired_sample_key": paired_key,
        "seed": int(seed),
        "sampling_round": "pilot_round_0",
        "fan_layout": fan_layout,
        "layout_branch_id": branch["layout_branch_id"],
        "fan_config_id": branch["fan_config_id"],
        "updraft_model_id": branch["updraft_model_id"],
        "branch_seed_family": f"{int(config.random_seed)}_{fan_layout}",
        "start_class": start_class,
        "family": plan_row["family"],
        "target_heading_deg": target,
        "direction_sign": int(plan_row["direction_sign"]),
        "environment_role": plan_row["environment_role"],
        "paired_environment_modes": paired_modes,
        "first_validity_gate_environment": plan_row["first_validity_gate_environment"],
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
        "updraft_center_x_m": float(center[0]),
        "updraft_center_y_m": float(center[1]),
        "updraft_relative_radius_m": radius,
        "updraft_relative_azimuth_rad": azimuth,
        "updraft_relative_height_m": float(z_w),
        "updraft_sector_label": sector,
        "left_wing_lift_exposure_preference": _wing_preference(azimuth, left=True),
        "right_wing_lift_exposure_preference": _wing_preference(azimuth, left=False),
        "wing_exposure_bookkeeping_status": "branch_layout_wing_wind_descriptor_logged",
        "true_safe_start": bool(inside_bounds(position, TRUE_SAFE_BOUNDS)),
        "start_generation_status": "generated_inside_true_safe_bounds",
        "layout_specific_sample_generated": True,
        "no_rollout_performed": True,
    }
    model_id = str(branch["updraft_model_id"])
    row.update(
        _descriptor_from_state(
            row=row,
            fan_layout=fan_layout,
            fan_config_id=str(branch["fan_config_id"]),
            environment_mode=str(branch["w1_environment_mode"]),
            model_id=model_id,
            wind_model=wind_models[model_id],
            dry_air=False,
        )
    )
    return row


def _paired_sample_key(
    config: DenseArchivePlanConfig,
    sample_index: int,
    plan_row: dict[str, object],
    start_class: str,
) -> str:
    target = _target_key(plan_row["target_heading_deg"])
    direction = f"d{int(plan_row['direction_sign']):+d}".replace("+", "p").replace("-", "m")
    return (
        f"s{int(config.run_id):03d}_{plan_row['fan_layout']}_{plan_row['family']}_"
        f"{target}_{direction}_{start_class}_{sample_index:06d}"
    ).replace(".", "p")


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _reference_center(
    fan_layout: str,
    start_class: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if fan_layout == "single_fan":
        return SINGLE_FAN_CENTER_XY
    centers = tuple(FOUR_FAN_CENTERS_XY)
    if start_class == "lift_sector":
        return centers[int(rng.integers(0, len(centers)))]
    return centers[int(rng.integers(0, len(centers)))]


def _sample_position_xy(
    *,
    fan_layout: str,
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
        sector = f"{fan_layout}_edge_ring_reference" if edge_or_ring else f"{fan_layout}_core_reference"
        return x_w, y_w, actual_radius, actual_azimuth, sector

    if start_class == "random_stress":
        x_w, y_w = _sample_boundary_near_xy(rng)
    else:
        ranges = SAMPLING_RANGES[start_class]
        x_w = _uniform(rng, ranges["x_range_m"])
        y_w = _uniform(rng, ranges["y_range_m"])
    radius = float(np.hypot(x_w - center[0], y_w - center[1]))
    azimuth = float(np.arctan2(y_w - center[1], x_w - center[0]))
    return x_w, y_w, radius, azimuth, f"{fan_layout}_{start_class}_reference"


def _clip_xy_to_true_safe(x_w: float, y_w: float) -> tuple[float, float]:
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
        return _uniform(rng, (x_min + 0.05, x_min + 0.45)), _uniform(rng, (y_min + 0.10, y_max - 0.10))
    if side == 1:
        return _uniform(rng, (x_max - 0.45, x_max - 0.05)), _uniform(rng, (y_min + 0.10, y_max - 0.10))
    if side == 2:
        return _uniform(rng, (x_min + 0.10, x_max - 0.10)), _uniform(rng, (y_min + 0.05, y_min + 0.45))
    return _uniform(rng, (x_min + 0.10, x_max - 0.10)), _uniform(rng, (y_max - 0.45, y_max - 0.05))


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
# 5) Dry-Run Candidate Inventory
# =============================================================================
def build_dry_run_candidate_inventory(
    config: DenseArchivePlanConfig,
    start_states: pd.DataFrame,
) -> pd.DataFrame:
    """Return branch-local paired W0/W1 candidate rows without replaying dynamics."""

    plan = build_target_environment_plan(config)
    wind_models = _load_w1_wind_models()
    plan_rows = {
        (
            row["fan_layout"],
            row["test_environment_mode"],
            row["family"],
            _target_key(row["target_heading_deg"]),
            int(row["direction_sign"]),
        ): row
        for row in plan.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for row in start_states.to_dict(orient="records"):
        meta = FAN_BRANCH_METADATA[str(row["fan_layout"])]
        for environment_mode in (meta["w0_environment_mode"], meta["w1_environment_mode"]):
            key = (
                row["fan_layout"],
                environment_mode,
                row["family"],
                _target_key(row["target_heading_deg"]),
                int(row["direction_sign"]),
            )
            plan_row = plan_rows[key]
            rows.append(_candidate_row(config, row, plan_row, wind_models))
    return pd.DataFrame(rows, columns=DRY_RUN_CANDIDATE_COLUMNS)


def _candidate_row(
    config: DenseArchivePlanConfig,
    start_row: dict[str, object],
    plan_row: dict[str, object],
    wind_models: dict[str, object],
) -> dict[str, object]:
    acceptance = _acceptance_interpretation(plan_row)
    candidate_id = _candidate_id(config, start_row, plan_row)
    model_id = str(plan_row["updraft_model_id"])
    dry_air = str(plan_row["test_environment_mode"]).startswith("W0_")
    descriptor = _descriptor_from_state(
        row=start_row,
        fan_layout=str(start_row["fan_layout"]),
        fan_config_id=str(plan_row["fan_config_id"]),
        environment_mode=str(plan_row["test_environment_mode"]),
        model_id=model_id,
        wind_model=None if dry_air else wind_models[model_id],
        dry_air=dry_air,
    )
    return {
        "candidate_id": candidate_id,
        "sample_id": start_row["sample_id"],
        "paired_sample_key": start_row["paired_sample_key"],
        "seed": int(start_row["seed"]),
        "sampling_round": start_row["sampling_round"],
        "fan_layout": start_row["fan_layout"],
        "layout_branch_id": start_row["layout_branch_id"],
        "fan_config_id": plan_row["fan_config_id"],
        "updraft_model_id": plan_row["updraft_model_id"],
        "test_environment_mode": plan_row["test_environment_mode"],
        "paired_environment_mode": plan_row["paired_environment_mode"],
        "family": start_row["family"],
        "target_heading_deg": _target_value(start_row["target_heading_deg"]),
        "direction_sign": int(start_row["direction_sign"]),
        "start_class": start_row["start_class"],
        "environment_role": start_row["environment_role"],
        "validity_gate_role": plan_row["validity_gate_role"],
        "first_validity_gate_environment": plan_row["first_validity_gate_environment"],
        "w0_failure_policy": plan_row["w0_failure_policy"],
        "acceptance_interpretation": acceptance,
        **descriptor,
        "count_basis": plan_row["count_basis"],
        "planned_floor_trial_count": int(plan_row["planned_floor_trial_count"]),
        "planned_target_trial_count": int(plan_row["planned_target_trial_count"]),
        "pilot_trial_count": int(plan_row["pilot_start_count"]),
        "latency_case_planned": LATENCY_CASE_PLANNED,
        "latency_acceptance_role": LATENCY_ACCEPTANCE_ROLE,
        "latency_model_status": LATENCY_MODEL_STATUS,
        "planned_replay_status": "not_replayed_in_this_task",
        "planned_result_path": (
            "03_Control/05_Results/10_dense_archive_planning/"
            f"{int(config.run_id):03d}/not_replayed/{candidate_id}.csv"
        ),
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        **NO_CROSS_BRANCH_FLAGS,
        "no_rollout_performed": True,
        "notes": "planning inventory only; no primitive replay, rollout, or active latency performed",
    }


def _acceptance_interpretation(plan_row: dict[str, object]) -> str:
    if (
        str(plan_row["test_environment_mode"]).startswith("W0_")
        and plan_row["environment_role"] == "updraft_assisted"
    ):
        return "ablation_only_not_rejection"
    if (
        str(plan_row["test_environment_mode"]).startswith("W1_")
        and plan_row["environment_role"] == "updraft_assisted"
    ):
        return "first_validity_gate"
    return str(plan_row["validity_gate_role"])


def _candidate_id(
    config: DenseArchivePlanConfig,
    start_row: dict[str, object],
    plan_row: dict[str, object],
) -> str:
    target = _target_key(start_row["target_heading_deg"])
    direction = f"d{int(start_row['direction_sign']):+d}".replace("+", "p").replace("-", "m")
    return (
        f"s{int(config.run_id):03d}_{start_row['fan_layout']}_{plan_row['test_environment_mode']}_"
        f"{start_row['family']}_{target}_{direction}_{start_row['sample_id']}"
    ).replace(".", "p")
