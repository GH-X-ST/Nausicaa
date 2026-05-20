from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds  # noqa: E402
from command_contract import (  # noqa: E402
    clip_normalised_command,
    normalised_command_to_surface_rad,
)
from dense_archive_clustering import select_cluster_representatives  # noqa: E402
from dense_archive_envelope_maps import (  # noqa: E402
    EnvelopeMapConfig,
    build_envelope_map,
)
from dense_archive_schema import BRANCH_DECISION_SCOPE, CAMPAIGN  # noqa: E402
from dense_archive_trial_logging import (  # noqa: E402
    DENSE_TRIAL_DESCRIPTOR_COLUMNS,
    dense_trial_descriptor_row,
    dense_trial_match_key,
)
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from latency import (  # noqa: E402
    actuator_tau_for_case,
    latency_acceptance_scope,
    latency_adjusted_command_sample,
    latency_audit_fields_from_case,
    latency_case_config,
    latency_pass_label_for_single_run,
)
from primitive_library_generators import generate_command_profile  # noqa: E402
from primitive_library_schema import PrimitiveCandidateSpec  # noqa: E402
from rollout import rk4_step  # noqa: E402
from updraft_models import load_updraft_model  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Configuration
# 2) Planning-Table Loading and Trial Selection
# 3) Pilot Replay
# 4) Output Writers and Public Runner
# 5) CLI
# =============================================================================


# =============================================================================
# 1) Paths and Configuration
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN
NO_CLAIM_TEXT = (
    "This is a Sun 24 5k-20k pilot sweep for descriptor/logging/storage "
    "validation only; it is not a production W0/W1 archive, not a final "
    "envelope claim, not W2/W3/W4/W5 evidence, not mission evaluation, not "
    "hardware validation, and not sim-to-real transfer."
)


@dataclass(frozen=True)
class DensePilotSweepConfig:
    run_id: int = 9
    planning_run_id: int = 8
    max_trials: int = 5000
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    random_seed: int = 20260524
    write_trial_logs: bool = False
    result_root: Path | None = None
    overwrite: bool = False
    branch_selection_rule: str = "round_robin_by_layout_branch_id"
    environment_selection_rule: str = "round_robin_by_test_environment_mode"


@dataclass(frozen=True)
class DensePilotSweepOutputs:
    root: Path
    trial_descriptors_csv: Path
    envelope_map_csv: Path
    cluster_representatives_csv: Path
    manifest_json: Path
    report_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "trial_descriptors_csv": self.trial_descriptors_csv,
            "envelope_map_csv": self.envelope_map_csv,
            "cluster_representatives_csv": self.cluster_representatives_csv,
            "manifest_json": self.manifest_json,
            "report_md": self.report_md,
        }


def _active_result_root(config: DensePilotSweepConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _output_paths(config: DensePilotSweepConfig) -> DensePilotSweepOutputs:
    root = _active_result_root(config) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    return DensePilotSweepOutputs(
        root=root,
        trial_descriptors_csv=root / "metrics" / f"dense_pilot_trial_descriptors_{suffix}.csv",
        envelope_map_csv=root / "metrics" / f"dense_pilot_envelope_map_{suffix}.csv",
        cluster_representatives_csv=root
        / "metrics"
        / f"dense_pilot_cluster_representatives_{suffix}.csv",
        manifest_json=root / "manifests" / f"dense_pilot_sweep_manifest_{suffix}.json",
        report_md=root / "reports" / f"dense_pilot_sweep_report_{suffix}.md",
    )


def _planning_paths(config: DensePilotSweepConfig) -> tuple[Path, Path]:
    root = _active_result_root(config) / f"{int(config.planning_run_id):03d}" / "metrics"
    suffix = f"s{int(config.planning_run_id):03d}"
    return (
        root / f"equal_branch_start_state_manifest_pilot_{suffix}.csv",
        root / f"equal_branch_dry_run_candidate_inventory_pilot_{suffix}.csv",
    )


# =============================================================================
# 2) Planning-Table Loading and Trial Selection
# =============================================================================
def _load_planning_tables(config: DensePilotSweepConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_path, candidate_path = _planning_paths(config)
    if not start_path.exists():
        raise FileNotFoundError(f"missing planning start-state table: {start_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing planning candidate table: {candidate_path}")
    return pd.read_csv(start_path), pd.read_csv(candidate_path)


def _select_pilot_candidates(
    candidates: pd.DataFrame,
    config: DensePilotSweepConfig,
) -> list[dict[str, object]]:
    if int(config.max_trials) <= 0 or candidates.empty:
        return []
    rows = sorted(
        candidates.to_dict(orient="records"),
        key=lambda row: (_selection_key(row), _text(row.get("candidate_id", ""))),
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(_selection_key(row), []).append(row)

    selected: list[dict[str, object]] = []
    keys = sorted(groups)
    while len(selected) < int(config.max_trials) and keys:
        remaining: list[tuple[object, ...]] = []
        for key in keys:
            group = groups[key]
            if group and len(selected) < int(config.max_trials):
                selected.append(group.pop(0))
            if group:
                remaining.append(key)
        keys = remaining
    return selected


def _selection_key(row: MappingLike) -> tuple[object, ...]:
    return (
        _text(row.get("layout_branch_id", "")),
        _text(row.get("test_environment_mode", "")),
        _text(row.get("family", "")),
        _target_sort_value(row.get("target_heading_deg", "")),
        _direction_int(row.get("direction_sign", 1)),
        _text(row.get("start_class", "")),
    )


# =============================================================================
# 3) Pilot Replay
# =============================================================================
def _run_pilot_replays(
    start_states: pd.DataFrame,
    candidates: list[dict[str, object]],
    config: DensePilotSweepConfig,
) -> pd.DataFrame:
    start_rows = {
        _text(row["sample_id"]): row
        for row in start_states.to_dict(orient="records")
    }
    aircraft = adapt_glider(build_nausicaa_glider())
    latency_config = latency_case_config(config.latency_case)
    actuator_tau = actuator_tau_for_case(latency_config)
    wind_cache: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for index, candidate_row in enumerate(candidates):
        sample_id = _text(candidate_row["sample_id"])
        if sample_id not in start_rows:
            raise KeyError(f"candidate sample_id missing from start manifest: {sample_id}")
        start_row = start_rows[sample_id]
        rows.append(
            _replay_one_candidate(
                start_row=start_row,
                candidate_row=candidate_row,
                config=config,
                replay_seed=int(config.random_seed) + index,
                aircraft=aircraft,
                latency_config=latency_config,
                actuator_tau=actuator_tau,
                wind_cache=wind_cache,
            )
        )
    return pd.DataFrame(rows, columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS)


def _replay_one_candidate(
    *,
    start_row: dict[str, object],
    candidate_row: dict[str, object],
    config: DensePilotSweepConfig,
    replay_seed: int,
    aircraft: object,
    latency_config: object,
    actuator_tau: tuple[float, float, float],
    wind_cache: dict[str, object],
) -> dict[str, object]:
    time_s = _time_grid(float(config.dt_s), float(config.horizon_s))
    spec = _candidate_spec(candidate_row, config.horizon_s)
    u_requested, _ = generate_command_profile(spec, time_s)
    x0 = _initial_state_from_start(start_row)
    wind_model = _wind_model_for_candidate(candidate_row, wind_cache)
    wind_mode = "none" if _is_w0(candidate_row) else "panel"
    replay = _integrate_open_loop(
        x0=x0,
        time_s=time_s,
        u_requested=u_requested,
        config=config,
        latency_config=latency_config,
        actuator_tau=actuator_tau,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
    )
    failure_label, descriptor_status = _pilot_failure(
        replay["x_ref"],
        candidate_row,
        EnvelopeMapConfig().heading_error_success_deg,
    )
    accepted = failure_label == "success"
    latency_fields = latency_audit_fields_from_case(
        latency_config,
        active_actuator_tau_s=actuator_tau,
    )
    latency_fields.update(
        {
            "latency_acceptance_scope": latency_acceptance_scope(config.latency_case),
            "latency_pass_label": latency_pass_label_for_single_run(
                config.latency_case,
                accepted,
            ),
            "state_feedback_delay_applied": False,
        }
    )
    match_key = dense_trial_match_key(
        layout_branch_id=_text(candidate_row.get("layout_branch_id", "")),
        fan_layout=_text(candidate_row.get("fan_layout", "")),
        fan_config_id=_text(candidate_row.get("fan_config_id", "")),
        test_environment_mode=_text(candidate_row.get("test_environment_mode", "")),
        paired_environment_mode=_text(candidate_row.get("paired_environment_mode", "")),
        candidate_id=_text(candidate_row.get("candidate_id", "")),
        sample_id=_text(candidate_row.get("sample_id", "")),
        paired_sample_key=_text(candidate_row.get("paired_sample_key", "")),
        seed=_csv_scalar(candidate_row.get("seed", "")),
        replay_seed=replay_seed,
        latency_case=config.latency_case,
    )
    return dense_trial_descriptor_row(
        start_row=start_row,
        candidate_row=candidate_row,
        time_s=replay["time_s"],
        x_ref=replay["x_ref"],
        u_norm_requested=replay["u_requested"],
        u_norm_effective_target=replay["u_effective"],
        u_norm_applied=replay["u_applied"],
        delta_cmd_rad=replay["delta_cmd_rad"],
        latency_fields=latency_fields,
        failure_label=failure_label,
        governor_rejection_cause="none",
        robustness_label="not_evaluated",
        sim_real_match_key=match_key,
        descriptor_status=descriptor_status,
        replay_seed=replay_seed,
        lift_exposure_m_s=_lift_exposure(replay["x_ref"], wind_model, _is_w0(candidate_row)),
        sim_real_transfer_result="not_evaluated",
        wind_model_z_axis_m=getattr(wind_model, "z_axis_m", None),
    )


def _integrate_open_loop(
    *,
    x0: np.ndarray,
    time_s: np.ndarray,
    u_requested: np.ndarray,
    config: DensePilotSweepConfig,
    latency_config: object,
    actuator_tau: tuple[float, float, float],
    aircraft: object,
    wind_model: object,
    wind_mode: str,
) -> dict[str, np.ndarray]:
    sample_count = time_s.size
    x_log = np.empty((sample_count, 15), dtype=float)
    u_effective = np.empty((sample_count, 3), dtype=float)
    u_applied = np.empty((sample_count, 3), dtype=float)
    delta_cmd = np.empty((sample_count, 3), dtype=float)
    x_log[0] = x0
    final_index = sample_count - 1

    for index, sample_time in enumerate(time_s):
        u_effective[index] = latency_adjusted_command_sample(
            time_s,
            u_requested,
            float(sample_time),
            latency_config,
        )
        u_applied[index] = clip_normalised_command(u_effective[index])
        delta_cmd[index] = normalised_command_to_surface_rad(u_applied[index])
        if config.latency_case == "none":
            x_log[index, 12:15] = delta_cmd[index]
        if index == sample_count - 1:
            break
        x_log[index + 1] = rk4_step(
            x_log[index],
            delta_cmd[index],
            float(config.dt_s),
            aircraft,
            wind_model,
            wind_mode,
            actuator_tau,
        )
        if not np.all(np.isfinite(x_log[index + 1])):
            final_index = index + 1
            u_effective[final_index] = u_effective[index]
            u_applied[final_index] = u_applied[index]
            delta_cmd[final_index] = delta_cmd[index]
            break
        if not inside_bounds(x_log[index + 1, 0:3], TRUE_SAFE_BOUNDS):
            final_index = index + 1
            u_effective[final_index] = u_effective[index]
            u_applied[final_index] = u_applied[index]
            delta_cmd[final_index] = delta_cmd[index]
            break

    terminal = final_index + 1
    return {
        "time_s": time_s[:terminal],
        "x_ref": x_log[:terminal],
        "u_requested": u_requested[:terminal],
        "u_effective": u_effective[:terminal],
        "u_applied": u_applied[:terminal],
        "delta_cmd_rad": delta_cmd[:terminal],
    }


def _pilot_failure(
    x_ref: np.ndarray,
    candidate_row: MappingLike,
    heading_error_success_deg: float,
) -> tuple[str, str]:
    state = np.asarray(x_ref, dtype=float)
    if not np.all(np.isfinite(state)):
        return "nonfinite_state", "nonfinite_state"
    if not all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in state[:, 0:3]):
        return "true_safety_violation", "entry_invalid"
    target = _float_or_nan(candidate_row.get("target_heading_deg", ""))
    if np.isfinite(target):
        direction = float(_direction_int(candidate_row.get("direction_sign", 1)))
        yaw_deg = np.rad2deg(np.unwrap(state[:, 5]))
        heading_change = float(direction * (yaw_deg[-1] - yaw_deg[0]))
        heading_error = abs(heading_change - float(target))
        if heading_error > float(heading_error_success_deg):
            return "target_miss", "replay_evaluated"
    return "success", "replay_evaluated"


def _candidate_spec(
    candidate_row: MappingLike,
    horizon_s: float,
) -> PrimitiveCandidateSpec:
    target = _float_or_nan(candidate_row.get("target_heading_deg", ""))
    wind_fidelity = "W0" if _is_w0(candidate_row) else "W1"
    return PrimitiveCandidateSpec(
        primitive_id=_text(candidate_row.get("candidate_id", "")),
        parent_primitive_id=_text(candidate_row.get("family", "")),
        variant_id=_text(candidate_row.get("candidate_id", "")),
        family=_text(candidate_row.get("family", "")),
        target_heading_deg=float(target) if np.isfinite(target) else None,
        updraft_config=_text(candidate_row.get("test_environment_mode", "")),
        wind_fidelity=wind_fidelity,
        start_condition=_text(candidate_row.get("start_class", "")),
        direction_sign=_direction_int(candidate_row.get("direction_sign", 1)),
        horizon_s=float(horizon_s),
    )


def _initial_state_from_start(start_row: MappingLike) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0:3] = (
        _float_or_nan(start_row.get("x_w_m")),
        _float_or_nan(start_row.get("y_w_m")),
        _float_or_nan(start_row.get("z_w_m")),
    )
    state[3:6] = (
        _float_or_nan(start_row.get("phi_rad")),
        _float_or_nan(start_row.get("theta_rad")),
        _float_or_nan(start_row.get("psi_rad")),
    )
    state[6:9] = (
        _float_or_nan(start_row.get("u_m_s")),
        _float_or_nan(start_row.get("v_m_s")),
        _float_or_nan(start_row.get("w_m_s")),
    )
    state[9:12] = (
        _float_or_nan(start_row.get("p_rad_s")),
        _float_or_nan(start_row.get("q_rad_s")),
        _float_or_nan(start_row.get("r_rad_s")),
    )
    return state


def _wind_model_for_candidate(
    candidate_row: MappingLike,
    wind_cache: dict[str, object],
) -> object:
    if _is_w0(candidate_row):
        return None
    model_id = _text(candidate_row.get("updraft_model_id", ""))
    if model_id == "analytic_debug_proxy":
        raise ValueError("analytic_debug_proxy is forbidden for dense pilot sweeps.")
    if model_id not in wind_cache:
        wind_model = load_updraft_model(model_id)
        if getattr(wind_model, "name", "") == "analytic_debug_proxy":
            raise ValueError("analytic_debug_proxy is forbidden for dense pilot sweeps.")
        wind_cache[model_id] = wind_model
    return wind_cache[model_id]


def _lift_exposure(
    x_ref: np.ndarray,
    wind_model: object,
    dry_air: bool,
) -> np.ndarray:
    if dry_air or wind_model is None:
        return np.zeros(np.asarray(x_ref).shape[0], dtype=float)
    wind = np.asarray(wind_model(np.asarray(x_ref, dtype=float)[:, 0:3]), dtype=float)
    if wind.shape != (np.asarray(x_ref).shape[0], 3):
        return np.full(np.asarray(x_ref).shape[0], np.nan)
    return wind[:, 2]


# =============================================================================
# 4) Output Writers and Public Runner
# =============================================================================
def run_dense_archive_pilot_sweep(
    *,
    run_id: int = 9,
    planning_run_id: int = 8,
    max_trials: int = 5000,
    latency_case: str = "nominal",
    overwrite: bool = False,
    random_seed: int = 20260524,
    write_trial_logs: bool = False,
    result_root: Path | None = None,
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
) -> dict[str, Path]:
    """Run the Sun 24 paired W0/W1 pilot sweep and write scaffold outputs."""

    config = DensePilotSweepConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        max_trials=int(max_trials),
        latency_case=str(latency_case),
        dt_s=float(dt_s),
        horizon_s=float(horizon_s),
        random_seed=int(random_seed),
        write_trial_logs=bool(write_trial_logs),
        result_root=result_root,
        overwrite=bool(overwrite),
    )
    _validate_config(config)
    outputs = _output_paths(config)
    _prepare_output_tree(outputs, config.overwrite)
    start_states, candidate_inventory = _load_planning_tables(config)
    selected = _select_pilot_candidates(candidate_inventory, config)
    descriptors = _run_pilot_replays(start_states, selected, config)
    envelope = build_envelope_map(descriptors)
    clusters = select_cluster_representatives(descriptors, envelope)

    descriptors.to_csv(outputs.trial_descriptors_csv, index=False)
    envelope.to_csv(outputs.envelope_map_csv, index=False)
    clusters.to_csv(outputs.cluster_representatives_csv, index=False)
    manifest = _manifest(config, outputs, int(len(descriptors)))
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    return outputs.as_dict()


def _validate_config(config: DensePilotSweepConfig) -> None:
    if int(config.max_trials) < 0:
        raise ValueError("max_trials must be nonnegative.")
    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive.")
    if not np.isfinite(float(config.horizon_s)) or float(config.horizon_s) <= 0.0:
        raise ValueError("horizon_s must be finite and positive.")
    latency_case_config(config.latency_case)


def _prepare_output_tree(outputs: DensePilotSweepOutputs, overwrite: bool) -> None:
    if outputs.root.exists() and not overwrite:
        raise ValueError(f"output directory already exists: {outputs.root}")
    if outputs.root.exists() and overwrite:
        _clear_output_tree(outputs.root)
    for path in (
        outputs.trial_descriptors_csv.parent,
        outputs.manifest_json.parent,
        outputs.report_md.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clear_output_tree(root: Path) -> None:
    root_resolved = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if root_resolved not in path.resolve().parents and path.resolve() != root_resolved:
            raise RuntimeError(f"refusing to clear path outside output root: {path}")
        if path.is_file() or path.is_symlink():
            path.unlink()


def _manifest(
    config: DensePilotSweepConfig,
    outputs: DensePilotSweepOutputs,
    trial_count: int,
) -> dict[str, object]:
    return {
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "max_trials_requested": int(config.max_trials),
        "trial_count_executed": int(trial_count),
        "latency_case": str(config.latency_case),
        "dt_s": float(config.dt_s),
        "horizon_s": float(config.horizon_s),
        "random_seed": int(config.random_seed),
        "write_trial_logs": bool(config.write_trial_logs),
        "branch_selection_rule": str(config.branch_selection_rule),
        "environment_selection_rule": str(config.environment_selection_rule),
        "pilot_sweep_performed": True,
        "production_dense_archive_performed": False,
        "w0_full_archive_performed": False,
        "w1_full_archive_performed": False,
        "envelope_map_scaffold_implemented": True,
        "clustering_scaffold_implemented": True,
        "hardware_or_mission_claim": False,
        "branch_local_decisions_only": True,
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        "output_files": {
            "dense_pilot_trial_descriptors": _path_text(outputs.trial_descriptors_csv),
            "dense_pilot_envelope_map": _path_text(outputs.envelope_map_csv),
            "dense_pilot_cluster_representatives": _path_text(
                outputs.cluster_representatives_csv
            ),
            "dense_pilot_sweep_manifest": _path_text(outputs.manifest_json),
            "dense_pilot_sweep_report": _path_text(outputs.report_md),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Sun 24 Dense Archive Pilot Sweep Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Planning run id: `{manifest['planning_run_id']}`",
        f"- Trial count executed: `{manifest['trial_count_executed']}`",
        f"- Latency case: `{manifest['latency_case']}`",
        f"- Branch-local decisions only: `{str(manifest['branch_local_decisions_only']).lower()}`",
        f"- Hardware or mission claim: `{str(manifest['hardware_or_mission_claim']).lower()}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 5) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=9)
    parser.add_argument("--planning-run-id", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=5000)
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260524)
    parser.add_argument("--write-trial-logs", action="store_true")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_dense_archive_pilot_sweep(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        max_trials=args.max_trials,
        latency_case=args.latency_case,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        write_trial_logs=args.write_trial_logs,
        result_root=args.result_root,
        dt_s=args.dt_s,
        horizon_s=args.horizon_s,
    )
    print(f"dense_archive_pilot_sweep_outputs={_path_text(paths['root'])}")
    return 0


MappingLike = dict[str, object] | pd.Series


def _time_grid(dt_s: float, horizon_s: float) -> np.ndarray:
    step_count = int(round(float(horizon_s) / float(dt_s)))
    return np.arange(step_count + 1, dtype=float) * float(dt_s)


def _is_w0(row: MappingLike) -> bool:
    return _text(row.get("test_environment_mode", "")).startswith("W0_")


def _target_sort_value(value: object) -> tuple[int, float | str]:
    numeric = _float_or_nan(value)
    if np.isfinite(numeric):
        return 0, float(numeric)
    return 1, "none"


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.strip() == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _direction_int(value: object) -> int:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric) or numeric == 0.0:
        return 1
    return -1 if numeric < 0.0 else 1


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return str(value)


def _csv_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _path_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
