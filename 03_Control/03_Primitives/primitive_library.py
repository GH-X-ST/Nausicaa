from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import clip_normalised_command, normalised_command_to_surface_rad
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from primitive_library_generators import build_start_state, generate_command_profile
from primitive_library_schema import (
    PrimitiveCandidateSpec,
    PrimitiveEvidenceRow,
    PrimitiveLibraryConfig,
    classify_candidate,
    classify_wind_query_region,
    entry_clearance_metrics,
    path_metrics,
    target_heading_band_deg,
)
from rollout import rk4_step


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data containers
# 2) Replay helpers
# 3) Evidence-row construction
# 4) Library summaries
# =============================================================================


# =============================================================================
# 1) Data Containers
# =============================================================================
@dataclass(frozen=True)
class WindModelInfo:
    available: bool
    model: object | None
    name: str
    source: str
    z_axis_m: np.ndarray | None
    evaluation_status: str = "evaluated"


@dataclass(frozen=True)
class CandidateEvaluation:
    spec: PrimitiveCandidateSpec
    row: PrimitiveEvidenceRow
    time_s: np.ndarray
    x_ref: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    phase: tuple[str, ...]


# =============================================================================
# 2) Replay Helpers
# =============================================================================
def evaluate_candidate(
    spec: PrimitiveCandidateSpec,
    config: PrimitiveLibraryConfig,
    wind_info: WindModelInfo,
    aircraft: object | None = None,
) -> CandidateEvaluation:
    """Replay one candidate and return scalar primitive-library evidence."""

    time_s = np.arange(int(round(spec.horizon_s / config.dt_s)) + 1, dtype=float) * config.dt_s
    state0 = build_start_state(spec.start_condition, altitude_m=1.8)
    u_req, phase = generate_command_profile(spec, time_s)
    if not wind_info.available and spec.wind_fidelity in ("W1", "W2"):
        return _not_evaluated_result(spec, config, wind_info, time_s, state0, u_req, phase)

    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    sample_count = time_s.size
    x_log = np.empty((sample_count, 15), dtype=float)
    u_applied = np.empty((sample_count, 3), dtype=float)
    delta_cmd = np.empty((sample_count, 3), dtype=float)
    x_log[0] = state0
    wind_mode = {"W0": "none", "W1": "cg", "W2": "panel"}.get(spec.wind_fidelity, "none")

    for index in range(sample_count):
        u_applied[index] = clip_normalised_command(u_req[index])
        delta_cmd[index] = normalised_command_to_surface_rad(u_applied[index])
        if index == sample_count - 1:
            break
        x_log[index + 1] = rk4_step(
            x_log[index],
            delta_cmd[index],
            float(config.dt_s),
            aircraft_model,
            wind_info.model,
            wind_mode,
            (0.06, 0.06, 0.06),
        )
        if not np.all(np.isfinite(x_log[index + 1])):
            x_log = x_log[: index + 2]
            u_req = u_req[: index + 2]
            u_applied = u_applied[: index + 2]
            delta_cmd = delta_cmd[: index + 2]
            time_s = time_s[: index + 2]
            phase = phase[: index + 2]
            break

    row = build_evidence_row(
        spec=spec,
        config=config,
        wind_info=wind_info,
        time_s=time_s,
        x_ref=x_log,
        u_norm_requested=u_req,
        u_norm_applied=u_applied,
    )
    return CandidateEvaluation(
        spec=spec,
        row=row,
        time_s=time_s,
        x_ref=x_log,
        u_norm_requested=u_req,
        u_norm_applied=u_applied,
        delta_cmd_rad=delta_cmd,
        phase=phase,
    )


def _not_evaluated_result(
    spec: PrimitiveCandidateSpec,
    config: PrimitiveLibraryConfig,
    wind_info: WindModelInfo,
    time_s: np.ndarray,
    state0: np.ndarray,
    u_req: np.ndarray,
    phase: tuple[str, ...],
) -> CandidateEvaluation:
    applied = np.array([clip_normalised_command(row) for row in u_req], dtype=float)
    delta = np.array([normalised_command_to_surface_rad(row) for row in applied], dtype=float)
    x_ref = np.repeat(state0.reshape(1, 15), time_s.size, axis=0)
    row = build_evidence_row(
        spec=spec,
        config=config,
        wind_info=wind_info,
        time_s=time_s,
        x_ref=x_ref,
        u_norm_requested=u_req,
        u_norm_applied=applied,
    )
    return CandidateEvaluation(
        spec=spec,
        row=row,
        time_s=time_s,
        x_ref=x_ref,
        u_norm_requested=u_req,
        u_norm_applied=applied,
        delta_cmd_rad=delta,
        phase=phase,
    )


# =============================================================================
# 3) Evidence-Row Construction
# =============================================================================
def build_evidence_row(
    spec: PrimitiveCandidateSpec,
    config: PrimitiveLibraryConfig,
    wind_info: WindModelInfo,
    time_s: np.ndarray,
    x_ref: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
) -> PrimitiveEvidenceRow:
    """Return one primitive evidence row with scalar metrics."""

    state = np.asarray(x_ref, dtype=float)
    positions = state[:, 0:3]
    speed = np.linalg.norm(state[:, 6:9], axis=1)
    alpha = np.rad2deg(np.arctan2(state[:, 8], np.maximum(state[:, 6], 1e-12)))
    beta = np.zeros_like(speed)
    valid = speed > 1e-9
    beta[valid] = np.rad2deg(np.arcsin(np.clip(state[valid, 7] / speed[valid], -1.0, 1.0)))
    rates = np.linalg.norm(state[:, 9:12], axis=1)
    yaw_unwrapped = np.unwrap(state[:, 5])
    direction = float(np.sign(spec.direction_sign) or 1.0)
    heading = float(direction * np.rad2deg(yaw_unwrapped[-1] - yaw_unwrapped[0]))
    target = spec.target_heading_deg
    if target is None:
        heading_error = 0.0
        heading_pass = True
    else:
        low, high = target_heading_band_deg(float(target))
        heading_error = float(abs(heading - float(target)))
        heading_pass = bool(low <= heading <= high)

    path = path_metrics(positions)
    clearance = entry_clearance_metrics(positions, TRUE_SAFE_BOUNDS)
    margins = _margin_metrics(positions)
    finite = bool(np.all(np.isfinite(state)))
    true_safe = bool(finite and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in positions))
    saturation_fraction = _saturation_fraction(u_norm_requested, u_norm_applied)
    z_fan = positions[:, 2] - float(config.z_outlet_m)
    energy_initial = float(positions[0, 2] + speed[0] ** 2 / (2.0 * 9.81))
    energy_terminal = float(positions[-1, 2] + speed[-1] ** 2 / (2.0 * 9.81))
    recovery_class = _recovery_class(spec, speed, true_safe, wind_info)
    row_inputs = {
        "evaluation_status": wind_info.evaluation_status,
        "target_heading_deg": target,
        "heading_band_pass": heading_pass,
        "true_safe_trajectory": true_safe,
        "finite_replay": finite,
        "terminal_speed_m_s": float(speed[-1]),
        "speed_min_m_s": float(np.nanmin(speed)),
        "alpha_max_deg": float(np.nanmax(np.abs(alpha))),
        "beta_max_deg": float(np.nanmax(np.abs(beta))),
        "rate_max_rad_s": float(np.nanmax(rates)),
        "saturation_fraction": saturation_fraction,
        "wind_fidelity": spec.wind_fidelity,
        "recovery_class": recovery_class,
        "lift_belief_condition": _lift_belief(spec, wind_info),
    }
    candidate_class, failure_label, limiting = classify_candidate(row_inputs)
    growth = _growth_fields(spec, candidate_class, failure_label, heading_error)
    return PrimitiveEvidenceRow(
        primitive_id=spec.primitive_id,
        parent_primitive_id=spec.parent_primitive_id,
        variant_id=spec.variant_id,
        envelope_group_id=growth["envelope_group_id"],
        target_heading_deg=target,
        family=spec.family,
        updraft_config=spec.updraft_config,
        wind_fidelity=spec.wind_fidelity,
        start_condition=spec.start_condition,
        environment_label=f"{spec.updraft_config}_{spec.wind_fidelity}",
        direction_sign=int(spec.direction_sign),
        evaluation_status=wind_info.evaluation_status,
        wind_model_available=bool(wind_info.available),
        wind_model_name=wind_info.name,
        wind_model_source=wind_info.source,
        z_outlet_m=float(config.z_outlet_m),
        z_fan_min_m=float(np.nanmin(z_fan)),
        z_fan_max_m=float(np.nanmax(z_fan)),
        terminal_heading_change_deg=heading,
        terminal_heading_error_deg=heading_error,
        heading_band_pass=heading_pass,
        **path,
        **clearance,
        speed_min_m_s=float(np.nanmin(speed)),
        terminal_speed_m_s=float(speed[-1]),
        specific_energy_initial_m=energy_initial,
        specific_energy_terminal_m=energy_terminal,
        energy_residual_m=float(energy_terminal - energy_initial),
        alpha_max_deg=float(np.nanmax(np.abs(alpha))),
        beta_max_deg=float(np.nanmax(np.abs(beta))),
        rate_max_rad_s=float(np.nanmax(rates)),
        saturation_fraction=saturation_fraction,
        true_safe_trajectory=true_safe,
        min_true_margin_m=margins["min_true_margin_m"],
        floor_margin_min_m=margins["floor_margin_min_m"],
        ceiling_margin_min_m=margins["ceiling_margin_min_m"],
        recovery_class=recovery_class,
        candidate_class=candidate_class,
        failure_label=failure_label,
        active_limiting_mechanism=limiting,
        wind_query_region=classify_wind_query_region(positions[:, 2], wind_info.z_axis_m, config.z_outlet_m),
        lift_belief_condition=_lift_belief(spec, wind_info),
        governor_condition=_governor_condition(candidate_class, spec.wind_fidelity),
        within_existing_envelope=bool(growth["within_existing_envelope"]),
        nearest_existing_primitive_id=str(growth["nearest_existing_primitive_id"]),
        normalised_distance_to_nearest_envelope=float(growth["normalised_distance_to_nearest_envelope"]),
        coverage_region_id=str(growth["coverage_region_id"]),
        marginal_coverage_gain=float(growth["marginal_coverage_gain"]),
        library_growth_trigger=bool(growth["library_growth_trigger"]),
        growth_reason=str(growth["growth_reason"]),
    )


def _margin_metrics(positions: np.ndarray) -> dict[str, float]:
    rows = [position_margin_m(position, TRUE_SAFE_BOUNDS) for position in positions if np.all(np.isfinite(position))]
    if not rows:
        return {"min_true_margin_m": np.nan, "floor_margin_min_m": np.nan, "ceiling_margin_min_m": np.nan}
    return {
        "min_true_margin_m": float(min(row["min_margin_m"] for row in rows)),
        "floor_margin_min_m": float(min(row["floor_margin_m"] for row in rows)),
        "ceiling_margin_min_m": float(min(row["ceiling_margin_m"] for row in rows)),
    }


def _saturation_fraction(requested: np.ndarray, applied: np.ndarray) -> float:
    clipped = np.any(np.abs(np.asarray(requested) - np.asarray(applied)) > 1e-12, axis=1)
    return float(np.count_nonzero(clipped) / max(1, clipped.size))


def _recovery_class(spec: PrimitiveCandidateSpec, speed: np.ndarray, true_safe: bool, wind_info: WindModelInfo) -> str:
    if not true_safe:
        return "not_recoverable"
    terminal = float(speed[-1])
    minimum = float(np.nanmin(speed))
    if terminal >= 5.0 and minimum >= 4.0:
        return "dry_recoverable"
    if spec.wind_fidelity in ("W1", "W2") and wind_info.available and terminal >= 3.5 and minimum >= 3.0:
        return "updraft_recoverable"
    if terminal >= 3.5 and minimum >= 3.0:
        return "updraft_pending"
    return "not_recoverable"


def _lift_belief(spec: PrimitiveCandidateSpec, wind_info: WindModelInfo) -> str:
    if spec.wind_fidelity == "W0":
        return "none"
    if not wind_info.available:
        return "model_unavailable"
    return f"{spec.updraft_config}_{spec.wind_fidelity}_available"


def _governor_condition(candidate_class: str, wind_fidelity: str) -> str:
    if candidate_class == "updraft_assisted_commandable" and wind_fidelity in ("W1", "W2"):
        return "pending_W3"
    if candidate_class in ("w0_standalone_commandable", "w0_updraft_pending_target_candidate"):
        return "entry_and_recovery_check_required"
    if candidate_class == "not_evaluated":
        return "model_unavailable"
    return "reject"


def _growth_fields(
    spec: PrimitiveCandidateSpec,
    candidate_class: str,
    failure_label: str,
    heading_error_deg: float,
) -> dict[str, object]:
    target = "none" if spec.target_heading_deg is None else f"{int(spec.target_heading_deg):03d}"
    coverage = f"{spec.family}|{target}|{spec.start_condition}|{spec.updraft_config}|{spec.wind_fidelity}"
    commandable = candidate_class in (
        "w0_standalone_commandable",
        "w0_updraft_pending_target_candidate",
        "updraft_assisted_commandable",
    )
    growth_trigger = not commandable and candidate_class != "not_evaluated"
    if commandable:
        growth_reason = "none"
    elif candidate_class == "not_evaluated":
        growth_reason = "model_unavailable"
    elif failure_label == "target_miss":
        growth_reason = "target_not_covered"
    elif failure_label == "true_safety_violation":
        growth_reason = "safety_limited"
    else:
        growth_reason = "recovery_limited"
    return {
        "envelope_group_id": f"{spec.parent_primitive_id}|{spec.start_condition}|{spec.wind_fidelity}",
        "within_existing_envelope": commandable and spec.family in ("glide", "recovery", "mild_bank"),
        "nearest_existing_primitive_id": spec.parent_primitive_id if commandable else "none",
        "normalised_distance_to_nearest_envelope": float(min(10.0, heading_error_deg / 30.0)),
        "coverage_region_id": coverage,
        "marginal_coverage_gain": 1.0 if commandable else 0.0,
        "library_growth_trigger": growth_trigger,
        "growth_reason": growth_reason,
    }


# =============================================================================
# 4) Library Summaries
# =============================================================================
def group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return lightweight primitive-envelope grouping summaries."""

    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["family"],
            row["target_heading_deg"],
            row["candidate_class"],
            row["updraft_config"],
            row["wind_fidelity"],
            row["start_condition"],
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group_rows in groups.items():
        heading_errors = [float(row["terminal_heading_error_deg"]) for row in group_rows if np.isfinite(float(row["terminal_heading_error_deg"]))]
        path_lengths = [float(row["path_length_xy_m"]) for row in group_rows if np.isfinite(float(row["path_length_xy_m"]))]
        footprints = [float(row["turn_footprint_proxy_m2"]) for row in group_rows if np.isfinite(float(row["turn_footprint_proxy_m2"]))]
        energies = [float(row["energy_residual_m"]) for row in group_rows if np.isfinite(float(row["energy_residual_m"]))]
        speeds = [float(row["terminal_speed_m_s"]) for row in group_rows if np.isfinite(float(row["terminal_speed_m_s"]))]
        failures = [str(row["failure_label"]) for row in group_rows]
        group_status = _group_status(group_rows)
        summary.append(
            {
                "family": key[0],
                "target_heading_deg": key[1],
                "candidate_class": key[2],
                "updraft_config": key[3],
                "wind_fidelity": key[4],
                "start_condition": key[5],
                "row_count": len(group_rows),
                "best_heading_error_deg": min(heading_errors) if heading_errors else np.nan,
                "best_path_length_xy_m": min(path_lengths) if path_lengths else np.nan,
                "best_footprint_m2": min(footprints) if footprints else np.nan,
                "best_energy_residual_m": max(energies) if energies else np.nan,
                "best_terminal_speed_m_s": max(speeds) if speeds else np.nan,
                "pass_fraction_true_safe": float(np.mean([bool(row["true_safe_trajectory"]) for row in group_rows])),
                "pass_fraction_heading_band": float(np.mean([bool(row["heading_band_pass"]) for row in group_rows])),
                "dominant_failure_label": max(set(failures), key=failures.count),
                "library_growth_trigger_count": int(sum(bool(row["library_growth_trigger"]) for row in group_rows)),
                "group_status": group_status,
            }
        )
    return summary


def _group_status(group_rows: list[dict[str, object]]) -> str:
    classes = {str(row["candidate_class"]) for row in group_rows}
    if classes == {"not_evaluated"}:
        return "not_evaluated_model_unavailable"
    if any(bool(row["within_existing_envelope"]) for row in group_rows):
        return "widening_existing_envelope"
    if any(bool(row["library_growth_trigger"]) for row in group_rows):
        return "requires_library_growth"
    return "remaining_boundary_evidence"
