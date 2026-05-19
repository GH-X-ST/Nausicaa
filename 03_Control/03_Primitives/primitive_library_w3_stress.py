from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import clip_normalised_command, normalised_command_to_surface_rad
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from logging_contract import command_dataframe, trajectory_dataframe
from primitive_library_generators import (
    build_start_state,
    generate_command_profile,
    primitive_candidate_inventory,
)
from primitive_library_schema import (
    PrimitiveCandidateSpec,
    PrimitiveLibraryConfig,
    classify_wind_query_region,
    entry_clearance_metrics,
    path_metrics,
    target_heading_band_deg,
)
from rollout import rk4_step
from updraft_models import SINGLE_FAN_CENTER_XY, load_updraft_model


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data containers and constants
# 2) Source loading and trial generation
# 3) Wind and command perturbation helpers
# 4) Trial replay and metrics
# 5) Candidate and coverage summaries
# =============================================================================


# =============================================================================
# 1) Data Containers and Constants
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = REPO_ROOT / "03_Control"
CAMPAIGN = "09_primitive_library"
DEFAULT_RANDOM_SEED = 20260526
TRIAL_STATUS_EVALUATED = "evaluated"
TRIAL_STATUS_ENTRY_INVALID = "entry_state_outside_true_safety"

ALLOWED_CANDIDATE_W3_STATUSES = (
    "w3_supported",
    "w3_marginal",
    "w3_rejected_entry_envelope",
    "w3_rejected_safety_or_recovery",
    "w3_boundary_evidence",
)
COVERAGE_STATUS_BY_CANDIDATE_STATUS = {
    "w3_supported": "w3_supported_pending_governor",
    "w3_marginal": "w3_marginal_needs_refinement",
    "w3_rejected_entry_envelope": "w3_rejected_governor_entry",
    "w3_rejected_safety_or_recovery": "w3_rejected_boundary",
    "w3_boundary_evidence": "w3_rejected_boundary",
}
RECOMMENDED_NEXT_STEP_BY_COVERAGE_STATUS = {
    "w3_supported_pending_governor": "send_to_governor_seed",
    "w3_marginal_needs_refinement": "refine_seed_before_governor",
    "w3_rejected_governor_entry": "keep_as_boundary_evidence",
    "w3_rejected_boundary": "keep_as_boundary_evidence",
    "not_in_w3_plan": "no_action_not_in_w3",
}


@dataclass(frozen=True)
class PerturbedWindModel:
    base_model: object
    strength_scale: float
    center_shift_xy_m: tuple[float, float]
    width_scale: float
    reference_center_xy_m: tuple[float, float]

    @property
    def name(self) -> str:
        base_name = getattr(self.base_model, "name", "unknown_wind")
        return f"{base_name}_selected_w3_perturbed"

    @property
    def source(self) -> str:
        base_source = getattr(self.base_model, "source", "unknown_source")
        return (
            f"{base_source}; selected_w3_strength={self.strength_scale:.6g}; "
            f"center_shift={self.center_shift_xy_m}; width_scale={self.width_scale:.6g}"
        )

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3).copy()
        center = np.asarray(self.reference_center_xy_m, dtype=float).reshape(2)
        shift = np.asarray(self.center_shift_xy_m, dtype=float).reshape(2)
        width = max(float(self.width_scale), 1e-12)
        points[:, :2] = center + (points[:, :2] - center - shift) / width
        return float(self.strength_scale) * np.asarray(self.base_model(points), dtype=float)


@dataclass(frozen=True)
class W3TrialResult:
    summary: dict[str, object]
    time_s: np.ndarray
    x_ref: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    phase_labels: tuple[str, ...]


@dataclass(frozen=True)
class W3TrialLog:
    summary: dict[str, object]
    time_s: np.ndarray
    x_ref: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray

    def trajectory_dataframe(self) -> pd.DataFrame:
        return trajectory_dataframe(self.time_s, self.x_ref)

    def command_dataframe(self) -> pd.DataFrame:
        return command_dataframe(
            self.time_s,
            self.u_norm_requested,
            self.u_norm_applied,
            self.delta_cmd_rad,
        )


# =============================================================================
# 2) Source Loading and Trial Generation
# =============================================================================
def load_w3_plan(result_root: Path, source_run_id: int = 3) -> pd.DataFrame:
    """Load the selected run-003 W3 plan and validate controlling columns."""

    root = Path(result_root)
    path = root / f"{source_run_id:03d}" / "metrics" / f"w3_stress_plan_s{source_run_id:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing selected W3 plan: {path}")
    plan = pd.read_csv(path)
    required = {
        "w3_plan_id",
        "source_primitive_id",
        "w3_role",
        "family",
        "target_heading_deg",
        "updraft_config",
        "wind_fidelity",
        "start_condition",
        "stress_seed_count",
        "start_position_perturbation_m",
        "speed_perturbation_m_s",
        "attitude_perturbation_deg",
        "updraft_strength_scale_range",
        "updraft_center_shift_m",
        "updraft_width_scale_range",
        "latency_perturbation_s",
    }
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise ValueError(f"malformed W3 plan; missing columns: {missing}")
    if plan.empty:
        raise ValueError("selected W3 plan is empty.")
    return plan


def load_source_evidence(
    result_root: Path,
    evidence_run_id: int = 2,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Load the run-002 manifest and primitive evidence table."""

    root = Path(result_root)
    manifest_path = (
        root / f"{evidence_run_id:03d}" / "manifests" / f"primitive_library_manifest_s{evidence_run_id:03d}.json"
    )
    evidence_path = (
        root / f"{evidence_run_id:03d}" / "metrics" / f"primitive_evidence_library_s{evidence_run_id:03d}.csv"
    )
    missing = [str(path) for path in (manifest_path, evidence_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run-{evidence_run_id:03d} source evidence: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    evidence = pd.read_csv(evidence_path)
    if "primitive_id" not in evidence.columns:
        raise ValueError("source evidence is missing primitive_id.")
    return manifest, evidence


def build_w3_trial_table(
    w3_plan: pd.DataFrame,
    seeds_per_candidate: int | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate deterministic nominal and perturbed W3 trial rows."""

    if w3_plan.empty:
        raise ValueError("w3_plan must contain selected candidates.")
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []
    for _, plan_row in w3_plan.iterrows():
        planned_count = int(plan_row["stress_seed_count"])
        seed_count = planned_count if seeds_per_candidate is None else min(planned_count, int(seeds_per_candidate))
        seed_count = min(seed_count, 25)
        if seed_count <= 0:
            raise ValueError("stress seed count must be positive.")
        for seed_index in range(seed_count):
            rows.append(
                _trial_row_from_plan(
                    plan_row,
                    seed_index=seed_index,
                    random_seed=int(random_seed),
                    rng=rng,
                )
            )
    return pd.DataFrame(rows)


def _trial_row_from_plan(
    plan_row: pd.Series,
    *,
    seed_index: int,
    random_seed: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    position_limit = float(plan_row["start_position_perturbation_m"])
    speed_limit = float(plan_row["speed_perturbation_m_s"])
    attitude_limit = float(plan_row["attitude_perturbation_deg"])
    strength_range = _parse_range(plan_row["updraft_strength_scale_range"])
    width_range = _parse_range(plan_row["updraft_width_scale_range"])
    center_limit = float(plan_row["updraft_center_shift_m"])
    latency_limit = float(plan_row["latency_perturbation_s"])

    if seed_index == 0:
        perturbations = {
            "start_dx_m": 0.0,
            "start_dy_m": 0.0,
            "start_dz_m": 0.0,
            "speed_perturbation_m_s": 0.0,
            "phi_perturbation_deg": 0.0,
            "theta_perturbation_deg": 0.0,
            "psi_perturbation_deg": 0.0,
            "updraft_strength_scale": 1.0,
            "updraft_center_shift_x_m": 0.0,
            "updraft_center_shift_y_m": 0.0,
            "updraft_width_scale": 1.0,
            "command_latency_s": 0.0,
        }
    else:
        perturbations = {
            "start_dx_m": float(rng.uniform(-position_limit, position_limit)),
            "start_dy_m": float(rng.uniform(-position_limit, position_limit)),
            "start_dz_m": float(rng.uniform(-position_limit, position_limit)),
            "speed_perturbation_m_s": float(rng.uniform(-speed_limit, speed_limit)),
            "phi_perturbation_deg": float(rng.uniform(-attitude_limit, attitude_limit)),
            "theta_perturbation_deg": float(rng.uniform(-attitude_limit, attitude_limit)),
            "psi_perturbation_deg": float(rng.uniform(-attitude_limit, attitude_limit)),
            "updraft_strength_scale": float(rng.uniform(*strength_range)),
            "updraft_center_shift_x_m": float(rng.uniform(-center_limit, center_limit)),
            "updraft_center_shift_y_m": float(rng.uniform(-center_limit, center_limit)),
            "updraft_width_scale": float(rng.uniform(*width_range)),
            "command_latency_s": float(rng.uniform(0.0, latency_limit)),
        }
    row = dict(plan_row)
    row.update(
        {
            "stress_seed_index": int(seed_index),
            "random_seed": int(random_seed),
            **perturbations,
        }
    )
    return row


def _parse_range(value: object) -> tuple[float, float]:
    text = str(value)
    if ":" not in text:
        number = float(text)
        return number, number
    low, high = text.split(":", maxsplit=1)
    return float(low), float(high)


# =============================================================================
# 3) Wind and Command Perturbation Helpers
# =============================================================================
def apply_command_latency(
    u_norm_requested: np.ndarray,
    dt_s: float,
    command_latency_s: float,
) -> np.ndarray:
    """Delay requested normalised commands while preserving array shape."""

    requested = np.asarray(u_norm_requested, dtype=float)
    if requested.ndim != 2 or requested.shape[1] != 3:
        raise ValueError("u_norm_requested must have shape (N, 3).")
    n_delay = int(round(float(command_latency_s) / float(dt_s)))
    if n_delay <= 0:
        return requested.copy()
    delayed = np.empty_like(requested)
    fill_count = min(n_delay, requested.shape[0])
    delayed[:fill_count] = requested[0]
    if fill_count < requested.shape[0]:
        delayed[fill_count:] = requested[:-fill_count]
    return delayed


def _wind_mode(wind_fidelity: str) -> str:
    return {"W0": "none", "W1": "cg", "W2": "panel"}.get(str(wind_fidelity), "none")


def _build_wind_model(trial_row: dict[str, object]) -> tuple[object | None, np.ndarray | None, bool, str]:
    wind_fidelity = str(trial_row["wind_fidelity"])
    updraft_config = str(trial_row["updraft_config"])
    if wind_fidelity == "W0" or updraft_config == "none":
        return None, None, True, "none"
    model_name = {
        "U1_single_fan": "single_gaussian_var",
        "U4_four_fan": "four_gaussian_var",
    }.get(updraft_config)
    if model_name is None:
        raise ValueError(f"unsupported updraft configuration for W3 stress: {updraft_config}")
    base = _load_cached_wind_model(model_name)
    center = _reference_center_xy(base)
    perturbed = PerturbedWindModel(
        base_model=base,
        strength_scale=float(trial_row["updraft_strength_scale"]),
        center_shift_xy_m=(
            float(trial_row["updraft_center_shift_x_m"]),
            float(trial_row["updraft_center_shift_y_m"]),
        ),
        width_scale=float(trial_row["updraft_width_scale"]),
        reference_center_xy_m=center,
    )
    z_axis = getattr(base, "z_axis_m", None)
    return perturbed, None if z_axis is None else np.asarray(z_axis, dtype=float), True, "selected_w3_wrapper"


@lru_cache(maxsize=4)
def _load_cached_wind_model(model_name: str) -> object:
    return load_updraft_model(model_name, repo_root=REPO_ROOT)


def _reference_center_xy(base_model: object) -> tuple[float, float]:
    centers = getattr(base_model, "fan_centers_xy", None)
    if centers is None:
        return SINGLE_FAN_CENTER_XY
    arr = np.asarray(centers, dtype=float).reshape(-1, 2)
    center = np.mean(arr, axis=0)
    return float(center[0]), float(center[1])


# =============================================================================
# 4) Trial Replay and Metrics
# =============================================================================
def evaluate_w3_trial(
    trial_row: dict[str, object],
    source_evidence: pd.DataFrame,
    config: PrimitiveLibraryConfig,
    aircraft: object | None = None,
) -> W3TrialResult:
    """Replay one selected W3 trial and return scalar evidence plus logs."""

    spec = _reconstruct_spec(trial_row, config)
    source_row = _source_row(source_evidence, str(trial_row["source_primitive_id"]))
    time_s = _time_vector(spec.horizon_s, config.dt_s)
    x0 = build_start_state(
        spec.start_condition,
        speed_m_s=6.5 + float(trial_row["speed_perturbation_m_s"]),
        altitude_m=1.8,
    )
    x0[0] += float(trial_row["start_dx_m"])
    x0[1] += float(trial_row["start_dy_m"])
    x0[2] += float(trial_row["start_dz_m"])
    x0[3] += np.deg2rad(float(trial_row["phi_perturbation_deg"]))
    x0[4] += np.deg2rad(float(trial_row["theta_perturbation_deg"]))
    x0[5] += np.deg2rad(float(trial_row["psi_perturbation_deg"]))

    u_profile, phase = generate_command_profile(spec, time_s)
    requested = apply_command_latency(
        u_profile,
        dt_s=float(config.dt_s),
        command_latency_s=float(trial_row["command_latency_s"]),
    )
    applied = np.array([clip_normalised_command(row) for row in requested], dtype=float)
    delta_cmd = np.array([normalised_command_to_surface_rad(row) for row in applied], dtype=float)

    if not inside_bounds(x0[0:3], TRUE_SAFE_BOUNDS):
        summary = _invalid_entry_summary(trial_row, source_row, spec, x0, time_s, requested, applied, delta_cmd)
        return W3TrialResult(
            summary=summary,
            time_s=np.array([0.0]),
            x_ref=x0.reshape(1, 15),
            u_norm_requested=requested[:1],
            u_norm_applied=applied[:1],
            delta_cmd_rad=delta_cmd[:1],
            phase_labels=phase[:1],
        )

    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    wind_model, model_z_axis_m, width_scale_applied, width_scale_note = _build_wind_model(trial_row)
    wind_mode = _wind_mode(spec.wind_fidelity)
    x_log = np.empty((time_s.size, 15), dtype=float)
    x_log[0] = x0
    final_index = time_s.size - 1
    for index in range(time_s.size - 1):
        x_next = rk4_step(
            x_log[index],
            delta_cmd[index],
            float(config.dt_s),
            aircraft_model,
            wind_model,
            wind_mode,
            (0.06, 0.06, 0.06),
        )
        x_log[index + 1] = x_next
        if not np.all(np.isfinite(x_next)):
            final_index = index + 1
            break
    x_log = x_log[: final_index + 1]
    time_s = time_s[: final_index + 1]
    requested = requested[: final_index + 1]
    applied = applied[: final_index + 1]
    delta_cmd = delta_cmd[: final_index + 1]
    phase = phase[: final_index + 1]

    summary = _trial_summary(
        trial_row,
        source_row,
        spec,
        time_s,
        x_log,
        requested,
        applied,
        delta_cmd,
        model_z_axis_m,
        width_scale_applied,
        width_scale_note,
    )
    return W3TrialResult(
        summary=summary,
        time_s=time_s,
        x_ref=x_log,
        u_norm_requested=requested,
        u_norm_applied=applied,
        delta_cmd_rad=delta_cmd,
        phase_labels=phase,
    )


def _reconstruct_spec(trial_row: dict[str, object], config: PrimitiveLibraryConfig) -> PrimitiveCandidateSpec:
    source_id = str(trial_row["source_primitive_id"])
    inventory = {spec.primitive_id: spec for spec in primitive_candidate_inventory(config)}
    if source_id not in inventory:
        raise ValueError(f"selected W3 candidate cannot be reconstructed exactly: {source_id}")
    spec = inventory[source_id]
    if spec.family != str(trial_row["family"]):
        raise ValueError(f"family mismatch while reconstructing {source_id}")
    return spec


def _source_row(source_evidence: pd.DataFrame, source_primitive_id: str) -> dict[str, object]:
    rows = source_evidence[source_evidence["primitive_id"].astype(str) == source_primitive_id]
    if rows.empty:
        raise ValueError(f"source primitive not found in run-002 evidence: {source_primitive_id}")
    return rows.iloc[0].to_dict()


def _time_vector(horizon_s: float, dt_s: float) -> np.ndarray:
    step_count = int(round(float(horizon_s) / float(dt_s)))
    return np.arange(step_count + 1, dtype=float) * float(dt_s)


def _invalid_entry_summary(
    trial_row: dict[str, object],
    source_row: dict[str, object],
    spec: PrimitiveCandidateSpec,
    x0: np.ndarray,
    time_s: np.ndarray,
    requested: np.ndarray,
    applied: np.ndarray,
    delta_cmd: np.ndarray,
) -> dict[str, object]:
    return _base_summary(
        trial_row,
        source_row,
        spec,
        time_s=np.array([0.0]),
        x_log=x0.reshape(1, 15),
        requested=requested[:1],
        applied=applied[:1],
        delta_cmd=delta_cmd[:1],
        model_z_axis_m=None,
        width_scale_applied=True,
        width_scale_note="entry_not_replayed",
        trial_evaluation_status=TRIAL_STATUS_ENTRY_INVALID,
        trial_success=False,
        failure_label="entry_invalid",
        active_limiting_mechanism="entry_envelope_limited",
    )


def _trial_summary(
    trial_row: dict[str, object],
    source_row: dict[str, object],
    spec: PrimitiveCandidateSpec,
    time_s: np.ndarray,
    x_log: np.ndarray,
    requested: np.ndarray,
    applied: np.ndarray,
    delta_cmd: np.ndarray,
    model_z_axis_m: np.ndarray | None,
    width_scale_applied: bool,
    width_scale_note: str,
) -> dict[str, object]:
    metrics = _base_summary(
        trial_row,
        source_row,
        spec,
        time_s=time_s,
        x_log=x_log,
        requested=requested,
        applied=applied,
        delta_cmd=delta_cmd,
        model_z_axis_m=model_z_axis_m,
        width_scale_applied=width_scale_applied,
        width_scale_note=width_scale_note,
        trial_evaluation_status=TRIAL_STATUS_EVALUATED,
        trial_success=False,
        failure_label="success",
        active_limiting_mechanism="none",
    )
    failure_label, limiting = _trial_failure(metrics)
    metrics["failure_label"] = failure_label
    metrics["active_limiting_mechanism"] = limiting
    metrics["trial_success"] = failure_label == "success"
    return metrics


def _base_summary(
    trial_row: dict[str, object],
    source_row: dict[str, object],
    spec: PrimitiveCandidateSpec,
    *,
    time_s: np.ndarray,
    x_log: np.ndarray,
    requested: np.ndarray,
    applied: np.ndarray,
    delta_cmd: np.ndarray,
    model_z_axis_m: np.ndarray | None,
    width_scale_applied: bool,
    width_scale_note: str,
    trial_evaluation_status: str,
    trial_success: bool,
    failure_label: str,
    active_limiting_mechanism: str,
) -> dict[str, object]:
    positions = np.asarray(x_log[:, 0:3], dtype=float)
    speed, alpha_deg, beta_deg = _speed_alpha_beta(x_log)
    rates = np.linalg.norm(np.asarray(x_log[:, 9:12], dtype=float), axis=1)
    yaw = np.unwrap(np.asarray(x_log[:, 5], dtype=float))
    direction = float(np.sign(spec.direction_sign) or 1.0)
    terminal_heading = float(direction * np.rad2deg(yaw[-1] - yaw[0])) if yaw.size else float("nan")
    if spec.target_heading_deg is None:
        heading_error = 0.0
        heading_pass = True
    else:
        low, high = target_heading_band_deg(float(spec.target_heading_deg))
        heading_error = float(abs(terminal_heading - float(spec.target_heading_deg)))
        heading_pass = bool(low <= terminal_heading <= high)
    path = path_metrics(positions)
    clearance = entry_clearance_metrics(positions, TRUE_SAFE_BOUNDS)
    margin = _margin_metrics(positions)
    finite = bool(np.all(np.isfinite(x_log)))
    true_safe = bool(finite and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in positions))
    saturation_fraction = _saturation_fraction(requested, applied)
    command_bridge = _command_bridge_verified(requested, applied, delta_cmd)
    energy_initial = float(positions[0, 2] + speed[0] ** 2 / (2.0 * 9.81)) if speed.size else float("nan")
    energy_terminal = float(positions[-1, 2] + speed[-1] ** 2 / (2.0 * 9.81)) if speed.size else float("nan")
    return {
        "w3_plan_id": trial_row["w3_plan_id"],
        "source_primitive_id": trial_row["source_primitive_id"],
        "w3_role": trial_row["w3_role"],
        "family": trial_row["family"],
        "target_heading_deg": trial_row["target_heading_deg"],
        "updraft_config": trial_row["updraft_config"],
        "wind_fidelity": trial_row["wind_fidelity"],
        "start_condition": trial_row["start_condition"],
        "stress_seed_index": int(trial_row["stress_seed_index"]),
        "random_seed": int(trial_row["random_seed"]),
        "trial_evaluation_status": trial_evaluation_status,
        "trial_success": bool(trial_success),
        "failure_label": failure_label,
        "active_limiting_mechanism": active_limiting_mechanism,
        "finite_replay": finite,
        "true_safe_trajectory": true_safe,
        "heading_band_pass": heading_pass,
        "terminal_heading_change_deg": terminal_heading,
        "terminal_heading_error_deg": heading_error,
        **path,
        "entry_clearance_required_x_plus_m": clearance["entry_clearance_required_x_plus_m"],
        "entry_clearance_required_y_plus_m": clearance["entry_clearance_required_y_plus_m"],
        "margin_consumption_x_m": clearance["margin_consumption_x_m"],
        "margin_consumption_y_m": clearance["margin_consumption_y_m"],
        "speed_min_m_s": _nanmin(speed),
        "terminal_speed_m_s": float(speed[-1]) if speed.size else float("nan"),
        "specific_energy_initial_m": energy_initial,
        "specific_energy_terminal_m": energy_terminal,
        "energy_residual_m": float(energy_terminal - energy_initial),
        "alpha_max_deg": _nanmax(np.abs(alpha_deg)),
        "beta_max_deg": _nanmax(np.abs(beta_deg)),
        "rate_max_rad_s": _nanmax(rates),
        "saturation_fraction": saturation_fraction,
        "min_true_margin_m": margin["min_true_margin_m"],
        "floor_margin_min_m": margin["floor_margin_min_m"],
        "ceiling_margin_min_m": margin["ceiling_margin_min_m"],
        "wind_query_region": classify_wind_query_region(positions[:, 2], model_z_axis_m),
        "updraft_strength_scale": float(trial_row["updraft_strength_scale"]),
        "updraft_center_shift_x_m": float(trial_row["updraft_center_shift_x_m"]),
        "updraft_center_shift_y_m": float(trial_row["updraft_center_shift_y_m"]),
        "updraft_width_scale": float(trial_row["updraft_width_scale"]),
        "width_scale_applied": bool(width_scale_applied),
        "width_scale_note": width_scale_note,
        "command_latency_s": float(trial_row["command_latency_s"]),
        "start_dx_m": float(trial_row["start_dx_m"]),
        "start_dy_m": float(trial_row["start_dy_m"]),
        "start_dz_m": float(trial_row["start_dz_m"]),
        "speed_perturbation_m_s": float(trial_row["speed_perturbation_m_s"]),
        "phi_perturbation_deg": float(trial_row["phi_perturbation_deg"]),
        "theta_perturbation_deg": float(trial_row["theta_perturbation_deg"]),
        "psi_perturbation_deg": float(trial_row["psi_perturbation_deg"]),
        "command_bridge_verified": command_bridge,
        "coverage_region_id": source_row.get("coverage_region_id", ""),
        "coverage_status_s002": source_row.get("coverage_status", ""),
        "candidate_class_s002": source_row.get("candidate_class", ""),
    }


def _trial_failure(metrics: dict[str, object]) -> tuple[str, str]:
    if not bool(metrics["finite_replay"]):
        return "nonfinite_replay", "numerical_failure"
    if not bool(metrics["true_safe_trajectory"]):
        return "true_safety_violation", "safety_limited"
    if not bool(metrics["command_bridge_verified"]):
        return "command_bridge_failed", "interface_contract_limited"
    if float(metrics["terminal_speed_m_s"]) < 3.5 or float(metrics["speed_min_m_s"]) < 3.0:
        return "speed_recovery_limit", "recovery_limited"
    if (
        float(metrics["alpha_max_deg"]) > 65.0
        or float(metrics["beta_max_deg"]) > 35.0
        or float(metrics["rate_max_rad_s"]) > 6.0
    ):
        return "exposure_limit", "exposure_limited"
    if float(metrics["saturation_fraction"]) >= 0.60:
        return "saturation_limit", "actuator_saturation_limited"
    target = metrics["target_heading_deg"]
    if pd.notna(target) and not bool(metrics["heading_band_pass"]):
        return "target_miss", "turn_authority_limited"
    return "success", "none"


def _speed_alpha_beta(x_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.asarray(x_log[:, 6:9], dtype=float)
    speed = np.linalg.norm(velocity, axis=1)
    alpha_deg = np.rad2deg(np.arctan2(velocity[:, 2], np.maximum(velocity[:, 0], 1e-12)))
    beta = np.zeros_like(speed)
    valid = speed > 1e-9
    beta[valid] = np.rad2deg(np.arcsin(np.clip(velocity[valid, 1] / speed[valid], -1.0, 1.0)))
    return speed, alpha_deg, beta


def _margin_metrics(positions: np.ndarray) -> dict[str, float]:
    rows = [
        position_margin_m(position, TRUE_SAFE_BOUNDS)
        for position in np.asarray(positions, dtype=float)
        if np.all(np.isfinite(position))
    ]
    if not rows:
        return {"min_true_margin_m": float("nan"), "floor_margin_min_m": float("nan"), "ceiling_margin_min_m": float("nan")}
    return {
        "min_true_margin_m": float(min(row["min_margin_m"] for row in rows)),
        "floor_margin_min_m": float(min(row["floor_margin_m"] for row in rows)),
        "ceiling_margin_min_m": float(min(row["ceiling_margin_m"] for row in rows)),
    }


def _saturation_fraction(requested: np.ndarray, applied: np.ndarray) -> float:
    clipped = np.any(np.abs(np.asarray(requested) - np.asarray(applied)) > 1e-12, axis=1)
    return float(np.count_nonzero(clipped) / max(1, clipped.size))


def _command_bridge_verified(requested: np.ndarray, applied: np.ndarray, delta_cmd: np.ndarray) -> bool:
    expected_applied = np.array([clip_normalised_command(row) for row in requested], dtype=float)
    expected_delta = np.array([normalised_command_to_surface_rad(row) for row in expected_applied], dtype=float)
    return bool(np.allclose(applied, expected_applied) and np.allclose(delta_cmd, expected_delta))


def _nanmin(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


# =============================================================================
# 5) Candidate and Coverage Summaries
# =============================================================================
def run_w3_stress_trials(
    w3_plan: pd.DataFrame,
    source_evidence: pd.DataFrame,
    config: PrimitiveLibraryConfig,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[W3TrialLog]]:
    """Evaluate all selected W3 trials and return trial/candidate/coverage evidence."""

    trial_table = build_w3_trial_table(w3_plan, random_seed=random_seed)
    return run_w3_stress_trial_table(trial_table, source_evidence, config)


def run_w3_stress_trial_table(
    trial_table: pd.DataFrame,
    source_evidence: pd.DataFrame,
    config: PrimitiveLibraryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[W3TrialLog]]:
    """Evaluate a pre-built W3 trial table."""

    aircraft = adapt_glider(build_nausicaa_glider())
    results: list[W3TrialResult] = []
    for _, row in trial_table.iterrows():
        results.append(evaluate_w3_trial(row.to_dict(), source_evidence, config, aircraft=aircraft))
    trial_summary = pd.DataFrame([result.summary for result in results])
    candidate_summary = build_candidate_summary(trial_summary)
    coverage_update = build_coverage_update(candidate_summary)
    logs = [
        W3TrialLog(
            summary=result.summary,
            time_s=result.time_s,
            x_ref=result.x_ref,
            u_norm_requested=result.u_norm_requested,
            u_norm_applied=result.u_norm_applied,
            delta_cmd_rad=result.delta_cmd_rad,
        )
        for result in results
    ]
    return trial_summary, candidate_summary, coverage_update, logs


def build_candidate_summary(trial_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_id, group in trial_summary.groupby("source_primitive_id", sort=False):
        evaluated = group[group["trial_evaluation_status"] == TRIAL_STATUS_EVALUATED]
        failures = group[group["failure_label"] != "success"]
        trial_count = int(len(group))
        evaluated_count = int(len(evaluated))
        success_count = int(group["trial_success"].astype(bool).sum())
        success_fraction = float(success_count / evaluated_count) if evaluated_count else 0.0
        nominal = group[group["stress_seed_index"].astype(int) == 0]
        nominal_success = bool(nominal["trial_success"].astype(bool).iloc[0]) if not nominal.empty else False
        entry_invalid_count = int((group["failure_label"] == "entry_invalid").sum())
        hard_safety_count = int((evaluated["failure_label"] == "true_safety_violation").sum())
        worst_failure_label = _worst_failure_label(group)
        dominant_label = _dominant_value(failures["failure_label"], default="success")
        dominant_mechanism = _dominant_value(failures["active_limiting_mechanism"], default="none")
        status = classify_candidate_w3_status(
            trial_count=trial_count,
            evaluated_trial_count=evaluated_count,
            entry_invalid_count=entry_invalid_count,
            trial_success_fraction=success_fraction,
            nominal_trial_success=nominal_success,
            hard_true_safety_violation_count=hard_safety_count,
            dominant_failure_label=dominant_label,
            dominant_limiting_mechanism=dominant_mechanism,
        )
        first = group.iloc[0]
        rows.append(
            {
                "source_primitive_id": source_id,
                "w3_plan_id": first["w3_plan_id"],
                "w3_role": first["w3_role"],
                "family": first["family"],
                "target_heading_deg": first["target_heading_deg"],
                "updraft_config": first["updraft_config"],
                "wind_fidelity": first["wind_fidelity"],
                "start_condition": first["start_condition"],
                "trial_count": trial_count,
                "evaluated_trial_count": evaluated_count,
                "entry_invalid_count": entry_invalid_count,
                "finite_replay_count": int(group["finite_replay"].astype(bool).sum()),
                "true_safe_count": int(group["true_safe_trajectory"].astype(bool).sum()),
                "trial_success_count": success_count,
                "trial_success_fraction": success_fraction,
                "nominal_trial_success": nominal_success,
                "hard_true_safety_violation_count": hard_safety_count,
                "worst_failure_label": worst_failure_label,
                "dominant_failure_label": dominant_label,
                "dominant_limiting_mechanism": dominant_mechanism,
                "heading_band_pass_fraction": _true_fraction(evaluated, "heading_band_pass"),
                "finite_fraction": _true_fraction(group, "finite_replay"),
                "true_safe_fraction": _true_fraction(group, "true_safe_trajectory"),
                "median_terminal_speed_m_s": _median(evaluated, "terminal_speed_m_s"),
                "min_terminal_speed_m_s": _min(evaluated, "terminal_speed_m_s"),
                "median_energy_residual_m": _median(evaluated, "energy_residual_m"),
                "min_true_margin_min_m": _min(evaluated, "min_true_margin_m"),
                "max_alpha_deg": _max(evaluated, "alpha_max_deg"),
                "max_beta_deg": _max(evaluated, "beta_max_deg"),
                "max_rate_rad_s": _max(evaluated, "rate_max_rad_s"),
                "max_saturation_fraction": _max(evaluated, "saturation_fraction"),
                "candidate_w3_status": status,
                "candidate_w3_recommendation": _candidate_recommendation(status),
                "coverage_region_id": first["coverage_region_id"],
                "coverage_status_s002": first["coverage_status_s002"],
            }
        )
    return pd.DataFrame(rows)


def classify_candidate_w3_status(
    *,
    trial_count: int,
    evaluated_trial_count: int,
    entry_invalid_count: int,
    trial_success_fraction: float,
    nominal_trial_success: bool,
    hard_true_safety_violation_count: int,
    dominant_failure_label: str,
    dominant_limiting_mechanism: str,
) -> str:
    required_evaluated = min(20, int(trial_count))
    if (
        bool(nominal_trial_success)
        and int(evaluated_trial_count) >= required_evaluated
        and float(trial_success_fraction) >= 0.80
        and int(hard_true_safety_violation_count) == 0
    ):
        return "w3_supported"
    if bool(nominal_trial_success) or float(trial_success_fraction) >= 0.50:
        return "w3_marginal"
    if int(entry_invalid_count) > max(0, int(trial_count) - int(entry_invalid_count)):
        return "w3_rejected_entry_envelope"
    if str(dominant_failure_label) == "entry_invalid" or str(dominant_limiting_mechanism) == "entry_envelope_limited":
        return "w3_rejected_entry_envelope"
    if str(dominant_failure_label) in {
        "true_safety_violation",
        "speed_recovery_limit",
        "exposure_limit",
        "saturation_limit",
        "nonfinite_replay",
    }:
        return "w3_rejected_safety_or_recovery"
    if str(dominant_limiting_mechanism) in {
        "safety_limited",
        "recovery_limited",
        "exposure_limited",
        "actuator_saturation_limited",
        "numerical_failure",
    }:
        return "w3_rejected_safety_or_recovery"
    return "w3_boundary_evidence"


def build_coverage_update(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in candidate_summary.iterrows():
        status = str(row["candidate_w3_status"])
        coverage_status = COVERAGE_STATUS_BY_CANDIDATE_STATUS.get(status, "w3_rejected_boundary")
        rows.append(
            {
                "coverage_region_id": row["coverage_region_id"],
                "source_primitive_id": row["source_primitive_id"],
                "w3_role": row["w3_role"],
                "coverage_decision_s003": "covered_send_to_w3",
                "candidate_w3_status": status,
                "trial_success_fraction": float(row["trial_success_fraction"]),
                "coverage_status_s004": coverage_status,
                "recommended_next_step": RECOMMENDED_NEXT_STEP_BY_COVERAGE_STATUS[coverage_status],
            }
        )
    return pd.DataFrame(rows)


def _true_fraction(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    return float(df[column].astype(bool).mean())


def _median(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").median())


def _min(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").min())


def _max(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").max())


def _dominant_value(series: pd.Series, default: str) -> str:
    if series.empty:
        return default
    counts = series.astype(str).value_counts(dropna=False)
    return str(counts.index[0])


def _worst_failure_label(group: pd.DataFrame) -> str:
    severity = (
        "nonfinite_replay",
        "true_safety_violation",
        "entry_invalid",
        "command_bridge_failed",
        "speed_recovery_limit",
        "exposure_limit",
        "saturation_limit",
        "target_miss",
        "success",
    )
    labels = set(str(value) for value in group["failure_label"])
    for label in severity:
        if label in labels:
            return label
    return sorted(labels)[0] if labels else "unknown"


def _candidate_recommendation(status: str) -> str:
    if status == "w3_supported":
        return "send_to_governor_seed"
    if status == "w3_marginal":
        return "refine_seed_before_governor"
    return "keep_as_boundary_evidence"
