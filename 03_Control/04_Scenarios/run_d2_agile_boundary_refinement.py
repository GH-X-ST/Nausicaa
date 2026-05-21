from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agile_trajectory_optimisation import (  # noqa: E402
    AGILE_TRAJECTORY_GENERATION_METHOD,
    AgileTrajectoryRequest,
    AgileTrajectoryResult,
    trajectory_metrics,
)
from agile_tvlqr import rollout_reference_feedback  # noqa: E402
from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    ensure_directory,
    file_sha256,
    filesystem_path,
    read_table_partition,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
)
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from latency import latency_pass_label_for_single_run  # noqa: E402
from run_agile_trajectory_optimisation import (  # noqa: E402
    D2_ARCHIVE_ROWS_PER_SECOND_ESTIMATE,
    D2_SCALE_ROWS,
    DEFAULT_REFERENCE_ROOT,
    NO_OVERCLAIMING_TEXT,
)
from updraft_models import load_updraft_model  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Configuration and Paths
# 2) Runtime Budget and Reference Loading
# 3) D2 Archive Execution
# 4) Aggregation, Plots, and Reports
# 5) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Configuration and Paths
# =============================================================================
DEFAULT_D2_ROOT = CONTROL_DIR / "05_Results" / "14_d2_agile_boundary_refinement"
DEFAULT_D1A_ARCHIVE_ROOT = CONTROL_DIR / "05_Results" / "12_paired_w0_w1_archive"
D2_DESCRIPTOR_COLUMNS = (
    "d2_trial_id",
    "trajectory_id",
    "trajectory_generation_method",
    "optimizer_status",
    "tvlqr_status",
    "controller_config_id",
    "agile_evidence_class",
    "layout_branch_id",
    "fan_layout",
    "test_environment_mode",
    "family",
    "target_heading_deg",
    "direction_sign",
    "source_sample_id",
    "reference_horizon_s",
    "terminal_heading_target_deg",
    "achieved_heading_deg",
    "terminal_heading_error_deg",
    "terminal_speed_m_s",
    "height_loss_m",
    "minimum_safety_margin_m",
    "actuator_saturation_fraction",
    "closed_loop_tracking_error_rms",
    "latency_case",
    "latency_pass_label",
    "success_flag",
    "failure_label",
    "closed_loop_replay_performed",
    "open_loop_optimizer_success_used_for_acceptance",
    "command_template_success_used_for_acceptance",
    "wall_floor_ceiling_safety_hard",
    "speed_height_loss_recoverable_only_when_logged",
)


@dataclass(frozen=True)
class D2AgileBoundaryConfig:
    run_id: int = 18
    reference_run_id: int = 17
    d1a_archive_run_id: int = 16
    result_root: Path | None = None
    reference_root: Path | None = None
    d1a_archive_root: Path | None = None
    d2_scale_class: str = "auto"
    partition_rows: int = 2500
    workers: str | int = 8
    max_workers: int | None = 8
    resume: bool = True
    latency_case: str = "nominal"
    runtime_budget_hours: float = 5.0
    runtime_buffer_fraction: float = 0.20
    build_plots: bool = False
    build_reports: bool = False
    storage_format: str = "csv_gz"
    compression_level: int = 1
    random_seed: int = 20260530
    row_count_override: int | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class D2AgileBoundaryOutputs:
    root: Path
    manifest_json: Path
    table_manifest_json: Path
    descriptors_path: Path
    aggregate_csv: Path
    report_md: Path
    figures_dir: Path


def _active_result_root(config: D2AgileBoundaryConfig) -> Path:
    return DEFAULT_D2_ROOT if config.result_root is None else Path(config.result_root)


def _active_reference_root(config: D2AgileBoundaryConfig) -> Path:
    return DEFAULT_REFERENCE_ROOT if config.reference_root is None else Path(config.reference_root)


def _active_d1a_archive_root(config: D2AgileBoundaryConfig) -> Path:
    return DEFAULT_D1A_ARCHIVE_ROOT if config.d1a_archive_root is None else Path(config.d1a_archive_root)


def _outputs(config: D2AgileBoundaryConfig) -> D2AgileBoundaryOutputs:
    root = _active_result_root(config) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    extension = "csv.gz" if resolve_storage_format(config.storage_format) == "csv_gz" else "csv"
    return D2AgileBoundaryOutputs(
        root=root,
        manifest_json=root / "manifests" / f"d2_agile_boundary_manifest_{suffix}.json",
        table_manifest_json=root / "manifests" / f"table_manifest_{suffix}.json",
        descriptors_path=root / "tables" / "d2_trial_descriptors" / f"part-00000.{extension}",
        aggregate_csv=root / "metrics_summary" / f"d2_agile_boundary_summary_{suffix}.csv",
        report_md=root / "reports" / f"d2_agile_boundary_report_{suffix}.md",
        figures_dir=root / "figures",
    )


# =============================================================================
# 2) Runtime Budget and Reference Loading
# =============================================================================
def _reference_run_root(config: D2AgileBoundaryConfig) -> Path:
    return _active_reference_root(config) / f"{int(config.reference_run_id):03d}"


def _reference_manifest_path(config: D2AgileBoundaryConfig) -> Path:
    suffix = f"s{int(config.reference_run_id):03d}"
    return _reference_run_root(config) / "manifests" / f"agile_reference_manifest_{suffix}.json"


def _reference_budget_state_path(config: D2AgileBoundaryConfig) -> Path:
    suffix = f"s{int(config.reference_run_id):03d}"
    return _reference_run_root(config) / "manifests" / f"runtime_budget_state_{suffix}.json"


def _reference_index_path(config: D2AgileBoundaryConfig) -> Path:
    suffix = f"s{int(config.reference_run_id):03d}"
    return _reference_run_root(config) / "metrics" / f"agile_reference_index_{suffix}.csv"


def _reference_table_manifest_path(config: D2AgileBoundaryConfig) -> Path:
    suffix = f"s{int(config.reference_run_id):03d}"
    return _reference_run_root(config) / "manifests" / f"table_manifest_{suffix}.json"


def _load_references(config: D2AgileBoundaryConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    manifest = json.loads(filesystem_path(_reference_manifest_path(config)).read_text(encoding="ascii"))
    budget = json.loads(filesystem_path(_reference_budget_state_path(config)).read_text(encoding="ascii"))
    index = pd.read_csv(filesystem_path(_reference_index_path(config)))
    table_manifest = json.loads(filesystem_path(_reference_table_manifest_path(config)).read_text(encoding="ascii"))
    frames = []
    for item in table_manifest.get("tables", []):
        path = _resolve_reference_table_path(str(item["relative_path"]), _reference_run_root(config))
        frames.append(read_table_partition(path, storage_format=str(item["storage_format"])))
    samples = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return index, samples, manifest, budget


def _selected_scale(config: D2AgileBoundaryConfig, reference_manifest: dict[str, object]) -> str:
    if str(config.d2_scale_class) == "auto":
        value = str(reference_manifest.get("selected_d2_scale_class", "primary"))
    else:
        value = str(config.d2_scale_class)
    if value not in D2_SCALE_ROWS:
        raise ValueError("d2_scale_class must be auto, primary, or fast.")
    return value


def _planned_row_count(config: D2AgileBoundaryConfig, scale: str) -> int:
    if config.row_count_override is not None:
        return int(config.row_count_override)
    return int(D2_SCALE_ROWS[str(scale)])


def _would_exceed_budget(
    budget: dict[str, object],
    *,
    budget_hours: float,
    buffer_fraction: float,
    projected_stage_s: float,
) -> bool:
    allowed = float(budget_hours) * 3600.0 * (1.0 - float(buffer_fraction))
    elapsed = time.time() - float(budget["runtime_budget_started_epoch_s"])
    return elapsed + float(projected_stage_s) > allowed


# =============================================================================
# 3) D2 Archive Execution
# =============================================================================
def build_d2_descriptors(
    reference_index: pd.DataFrame,
    reference_samples: pd.DataFrame,
    *,
    row_count: int,
    latency_case: str,
    random_seed: int,
) -> pd.DataFrame:
    if str(latency_case) != "nominal":
        raise ValueError("D2 acceptance requires latency_case='nominal'.")
    if reference_index.empty or reference_samples.empty:
        raise RuntimeError("D2 archive requires completed reference index and samples.")
    aircraft = adapt_glider(build_nausicaa_glider())
    wind_cache: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    selected = _select_reference_rows(reference_index, int(row_count), int(random_seed))
    sample_groups = {
        str(trajectory_id): group.sort_values("sample_index")
        for trajectory_id, group in reference_samples.groupby("trajectory_id", sort=False)
    }
    for archive_index, ref_row in enumerate(selected):
        trajectory_id = str(ref_row["trajectory_id"])
        samples = sample_groups[trajectory_id]
        trajectory = _trajectory_from_rows(ref_row, samples)
        gains = _gains_from_samples(samples)
        wind_model, wind_mode = _wind_for_reference(ref_row, wind_cache)
        rollout = rollout_reference_feedback(
            trajectory,
            gains,
            aircraft=aircraft,
            wind_model=wind_model,
            wind_mode=wind_mode,
            dt_s=float(trajectory.request.dt_s),
            latency_case=latency_case,
        )
        rows.append(
            _descriptor_row(
                archive_index=archive_index,
                reference_row=ref_row,
                trajectory=trajectory,
                rollout=rollout,
                latency_case=latency_case,
            )
        )
    return pd.DataFrame(rows, columns=D2_DESCRIPTOR_COLUMNS)


def _select_reference_rows(
    reference_index: pd.DataFrame,
    row_count: int,
    random_seed: int,
) -> list[dict[str, object]]:
    frame = reference_index.copy()
    frame = frame[frame["trajectory_generation_method"].astype(str).eq(AGILE_TRAJECTORY_GENERATION_METHOD)]
    frame = frame[frame["agile_evidence_class"].astype(str).str.contains("nominal_latency", na=False)]
    if frame.empty:
        raise RuntimeError("no closed-loop trajectory-optimised references available.")
    rng = np.random.default_rng(int(random_seed))
    groups = [group for _, group in frame.groupby(["layout_branch_id", "family", "target_heading_deg"], sort=True)]
    rows: list[dict[str, object]] = []
    while len(rows) < int(row_count):
        for group in groups:
            if len(rows) >= int(row_count):
                break
            idx = int(rng.integers(0, len(group)))
            rows.append(group.iloc[idx].to_dict())
    return rows


def _trajectory_from_rows(ref_row: dict[str, object], samples: pd.DataFrame) -> AgileTrajectoryResult:
    x_ref = samples[[f"x_ref_{index:02d}" for index in range(15)]].to_numpy(dtype=float)
    u_norm_ref = samples[
        ["u_norm_ref_delta_a", "u_norm_ref_delta_e", "u_norm_ref_delta_r"]
    ].to_numpy(dtype=float)
    delta_ref = samples[
        ["delta_cmd_ref_rad_delta_a", "delta_cmd_ref_rad_delta_e", "delta_cmd_ref_rad_delta_r"]
    ].to_numpy(dtype=float)
    time_s = samples["time_s"].to_numpy(dtype=float)
    request = AgileTrajectoryRequest(
        trajectory_id=str(ref_row["trajectory_id"]),
        family=str(ref_row["family"]),
        target_heading_deg=float(ref_row["target_heading_deg"]),
        direction_sign=int(float(ref_row["direction_sign"])),
        x0=x_ref[0],
        layout_branch_id=str(ref_row["layout_branch_id"]),
        fan_layout=str(ref_row["fan_layout"]),
        test_environment_mode=str(ref_row["test_environment_mode"]),
        updraft_model_id=str(ref_row["updraft_model_id"]),
        sample_id=str(ref_row["sample_id"]),
        dt_s=float(time_s[1] - time_s[0]) if len(time_s) > 1 else 0.02,
        horizon_s=float(time_s[-1] - time_s[0]) if len(time_s) else 0.02,
    )
    return AgileTrajectoryResult(
        request=request,
        trajectory_id=str(ref_row["trajectory_id"]),
        trajectory_generation_method=str(ref_row["trajectory_generation_method"]),
        optimizer_status=str(ref_row["optimizer_status"]),
        optimizer_message=str(ref_row.get("optimizer_message", "")),
        optimizer_success=bool(ref_row.get("optimizer_success", False)),
        optimizer_iterations=int(float(ref_row.get("optimizer_iterations", 0))),
        optimizer_wall_time_s=float(ref_row.get("optimizer_wall_time_s", 0.0)),
        objective_cost=float(ref_row.get("objective_cost", np.nan)),
        heading_cost=float(ref_row.get("heading_cost", np.nan)),
        speed_loss_cost=float(ref_row.get("speed_loss_cost", np.nan)),
        height_loss_cost=float(ref_row.get("height_loss_cost", np.nan)),
        saturation_cost=float(ref_row.get("saturation_cost", np.nan)),
        safety_status=str(ref_row.get("safety_status", "")),
        time_s=time_s,
        x_ref=x_ref,
        u_norm_ref=u_norm_ref,
        delta_cmd_ref_rad=delta_ref,
        achieved_heading_deg=float(ref_row.get("achieved_heading_deg", np.nan)),
        terminal_heading_error_deg=float(ref_row.get("terminal_heading_error_deg", np.nan)),
        terminal_speed_m_s=float(ref_row.get("terminal_speed_m_s", np.nan)),
        height_loss_m=float(ref_row.get("height_loss_m", np.nan)),
        minimum_safety_margin_m=float(ref_row.get("minimum_safety_margin_m", np.nan)),
        actuator_saturation_fraction=float(ref_row.get("actuator_saturation_fraction", np.nan)),
        open_loop_success_flag=bool(ref_row.get("open_loop_success_flag", False)),
        failure_label=str(ref_row.get("failure_label", "")),
    )


def _gains_from_samples(samples: pd.DataFrame) -> np.ndarray:
    gains = np.empty((len(samples), 3, 15), dtype=float)
    for command_index in range(3):
        for state_index in range(15):
            gains[:, command_index, state_index] = samples[f"k_{command_index}_{state_index:02d}"].to_numpy(dtype=float)
    return gains


def _descriptor_row(
    *,
    archive_index: int,
    reference_row: dict[str, object],
    trajectory: AgileTrajectoryResult,
    rollout: dict[str, object],
    latency_case: str,
) -> dict[str, object]:
    time_s = np.asarray(rollout["time_s"], dtype=float)
    x_closed = np.asarray(rollout["x_closed_loop"], dtype=float)
    u_requested = np.asarray(rollout["u_norm_requested"], dtype=float)
    metrics = trajectory_metrics(time_s, x_closed, u_requested, trajectory.request)
    failure_label = _failure_label(metrics, trajectory)
    success = failure_label == "success"
    return {
        "d2_trial_id": f"d2_trial_{archive_index:08d}",
        "trajectory_id": str(trajectory.trajectory_id),
        "trajectory_generation_method": str(reference_row["trajectory_generation_method"]),
        "optimizer_status": str(reference_row["optimizer_status"]),
        "tvlqr_status": str(reference_row["tvlqr_status"]),
        "controller_config_id": str(reference_row["controller_config_id"]),
        "agile_evidence_class": str(reference_row["agile_evidence_class"]),
        "layout_branch_id": str(reference_row["layout_branch_id"]),
        "fan_layout": str(reference_row["fan_layout"]),
        "test_environment_mode": str(reference_row["test_environment_mode"]),
        "family": str(reference_row["family"]),
        "target_heading_deg": float(reference_row["target_heading_deg"]),
        "direction_sign": int(float(reference_row["direction_sign"])),
        "source_sample_id": str(reference_row["sample_id"]),
        "reference_horizon_s": float(reference_row["reference_horizon_s"]),
        "terminal_heading_target_deg": float(reference_row["terminal_heading_target_deg"]),
        "achieved_heading_deg": float(metrics["achieved_heading_deg"]),
        "terminal_heading_error_deg": float(metrics["terminal_heading_error_deg"]),
        "terminal_speed_m_s": float(metrics["terminal_speed_m_s"]),
        "height_loss_m": float(metrics["height_loss_m"]),
        "minimum_safety_margin_m": float(metrics["minimum_safety_margin_m"]),
        "actuator_saturation_fraction": float(metrics["actuator_saturation_fraction"]),
        "closed_loop_tracking_error_rms": _tracking_error_rms(
            time_s,
            x_closed,
            trajectory.time_s,
            trajectory.x_ref,
        ),
        "latency_case": str(latency_case),
        "latency_pass_label": latency_pass_label_for_single_run(latency_case, success),
        "success_flag": bool(success),
        "failure_label": failure_label,
        "closed_loop_replay_performed": True,
        "open_loop_optimizer_success_used_for_acceptance": False,
        "command_template_success_used_for_acceptance": False,
        "wall_floor_ceiling_safety_hard": True,
        "speed_height_loss_recoverable_only_when_logged": True,
    }


# =============================================================================
# 4) Aggregation, Plots, and Reports
# =============================================================================
def build_d2_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    grouped = descriptors.groupby(
        ["layout_branch_id", "fan_layout", "test_environment_mode", "family", "target_heading_deg"],
        dropna=False,
    )
    rows = []
    for key, group in grouped:
        branch, fan, environment, family, target = key
        success = group["success_flag"].astype(bool)
        rows.append(
            {
                "layout_branch_id": branch,
                "fan_layout": fan,
                "test_environment_mode": environment,
                "family": family,
                "target_heading_deg": float(target),
                "trial_count": int(len(group)),
                "success_count": int(success.sum()),
                "success_rate": float(success.mean()) if len(group) else 0.0,
                "minimum_safety_margin_m_min": float(group["minimum_safety_margin_m"].min()),
                "terminal_heading_error_deg_median": float(group["terminal_heading_error_deg"].median()),
                "latency_pass_count": int(group["latency_pass_label"].astype(str).eq("nominal_pass").sum()),
            }
        )
    return pd.DataFrame(rows)


def final_classification(descriptors: pd.DataFrame) -> str:
    successes = descriptors[descriptors["success_flag"].astype(bool)]
    if successes.empty:
        return "D2_high_angle_empty_tested_envelope_current_model"
    high = successes[pd.to_numeric(successes["target_heading_deg"], errors="coerce") >= 45.0]
    if not high.empty:
        return "ready_for_W2_complex_updraft_replay"
    return "D2_ready_with_small_turn_only_strategy"


# =============================================================================
# 5) Public Runner and CLI
# =============================================================================
def run_d2_agile_boundary_refinement(config: D2AgileBoundaryConfig) -> D2AgileBoundaryOutputs:
    _validate_config(config)
    outputs = _outputs(config)
    _prepare_output_root(outputs, config)
    reference_index, reference_samples, reference_manifest, budget = _load_references(config)
    scale = _selected_scale(config, reference_manifest)
    row_count = _planned_row_count(config, scale)
    projected_archive_s = float(row_count) / D2_ARCHIVE_ROWS_PER_SECOND_ESTIMATE + 180.0
    if (
        str(reference_manifest.get("runtime_budget_status", "")) == "D2_execution_blocked_by_runtime_budget"
        or _would_exceed_budget(
            budget,
            budget_hours=float(config.runtime_budget_hours),
            buffer_fraction=float(config.runtime_buffer_fraction),
            projected_stage_s=projected_archive_s,
        )
    ):
        manifest = _manifest(
            config,
            outputs,
            status="D2_execution_blocked_by_runtime_budget",
            scale=scale,
            row_count=0,
            classification="D2_execution_blocked_by_runtime_budget",
        )
        manifest["runtime_budget_blocked_before_archive"] = True
        _write_json(outputs.manifest_json, manifest)
        _write_report(outputs.report_md, manifest)
        return outputs

    profile_count = min(64, int(row_count))
    profile_started = time.perf_counter()
    archive_profile = build_d2_descriptors(
        reference_index,
        reference_samples,
        row_count=profile_count,
        latency_case=str(config.latency_case),
        random_seed=int(config.random_seed),
    )
    profile_elapsed = max(time.perf_counter() - profile_started, 1.0e-12)
    measured_rows_per_second = float(profile_count) / profile_elapsed
    projected_archive_s = float(row_count) / max(measured_rows_per_second, 1.0e-12) + 180.0
    if row_count > profile_count and _would_exceed_budget(
        budget,
        budget_hours=float(config.runtime_budget_hours),
        buffer_fraction=float(config.runtime_buffer_fraction),
        projected_stage_s=projected_archive_s,
    ):
        manifest = _manifest(
            config,
            outputs,
            status="D2_execution_blocked_by_runtime_budget",
            scale=scale,
            row_count=0,
            classification="D2_execution_blocked_by_runtime_budget",
        )
        manifest.update(
            {
                "runtime_budget_blocked_before_archive": True,
                "archive_profile_row_count": int(profile_count),
                "archive_profile_elapsed_s": float(profile_elapsed),
                "archive_profile_rows_per_second": float(measured_rows_per_second),
                "projected_archive_runtime_s": float(projected_archive_s),
            }
        )
        _write_json(outputs.manifest_json, manifest)
        _write_report(outputs.report_md, manifest)
        return outputs

    descriptors = archive_profile
    if row_count > profile_count:
        descriptors = build_d2_descriptors(
            reference_index,
            reference_samples,
            row_count=row_count,
            latency_case=str(config.latency_case),
            random_seed=int(config.random_seed),
        )
    summary = build_d2_summary(descriptors)
    classification = final_classification(descriptors)
    partition = write_table_partition(
        descriptors,
        outputs.descriptors_path,
        storage_format=config.storage_format,
        compression_level=int(config.compression_level),
    )
    table_manifest = TableManifest(
        run_id=int(config.run_id),
        root=_path_text(outputs.root),
        storage_format=resolve_storage_format(config.storage_format),
        tables=(partition,),
    )
    write_table_manifest(outputs.table_manifest_json, table_manifest)
    ensure_directory(outputs.aggregate_csv.parent)
    summary.to_csv(filesystem_path(outputs.aggregate_csv), index=False)
    if config.build_plots:
        _write_plots(outputs, summary)
    manifest = _manifest(
        config,
        outputs,
        status="complete",
        scale=scale,
        row_count=len(descriptors),
        classification=classification,
    )
    manifest.update(
        {
            "descriptor_sha256": partition.checksum_sha256,
            "aggregate_sha256": file_sha256(outputs.aggregate_csv),
            "success_count": int(descriptors["success_flag"].astype(bool).sum()),
            "branch_local_decisions_only": True,
            "w1_selected_independently_of_w0_success": True,
            "nominal_latency_closed_loop_acceptance_only": True,
        }
    )
    _write_json(outputs.manifest_json, manifest)
    if config.build_reports:
        _write_report(outputs.report_md, manifest)
    return outputs


def _validate_config(config: D2AgileBoundaryConfig) -> None:
    if int(config.run_id) <= 0 or int(config.reference_run_id) <= 0:
        raise ValueError("run_id and reference_run_id must be positive.")
    if str(config.latency_case) != "nominal":
        raise ValueError("D2 acceptance requires --latency-case nominal.")
    if int(config.partition_rows) <= 0:
        raise ValueError("partition_rows must be positive.")
    resolve_storage_format(config.storage_format)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=18)
    parser.add_argument("--reference-run-id", type=int, default=17)
    parser.add_argument("--d1a-archive-run-id", type=int, default=16)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--d1a-archive-root", type=Path, default=None)
    parser.add_argument("--d2-scale-class", default="auto")
    parser.add_argument("--partition-rows", type=int, default=2500)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--runtime-budget-hours", type=float, default=5.0)
    parser.add_argument("--runtime-buffer-fraction", type=float, default=0.20)
    parser.add_argument("--build-plots", action="store_true")
    parser.add_argument("--build-reports", action="store_true")
    parser.add_argument("--storage-format", default="csv_gz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_d2_agile_boundary_refinement(
        D2AgileBoundaryConfig(
            run_id=args.run_id,
            reference_run_id=args.reference_run_id,
            d1a_archive_run_id=args.d1a_archive_run_id,
            result_root=args.result_root,
            reference_root=args.reference_root,
            d1a_archive_root=args.d1a_archive_root,
            d2_scale_class=args.d2_scale_class,
            partition_rows=args.partition_rows,
            workers=args.workers,
            max_workers=args.max_workers,
            resume=bool(args.resume),
            latency_case=args.latency_case,
            runtime_budget_hours=args.runtime_budget_hours,
            runtime_buffer_fraction=args.runtime_buffer_fraction,
            build_plots=bool(args.build_plots),
            build_reports=bool(args.build_reports),
            storage_format=args.storage_format,
            overwrite=bool(args.overwrite),
        )
    )
    return 0


def _prepare_output_root(outputs: D2AgileBoundaryOutputs, config: D2AgileBoundaryConfig) -> None:
    root = filesystem_path(outputs.root)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    if config.overwrite:
        return
    existing = [path for path in root.rglob("*") if path.is_file()]
    if existing:
        raise RuntimeError(f"D2 output root is non-empty: {outputs.root}")


def _resolve_reference_table_path(path_text: str, run_root: Path) -> Path:
    path = Path(path_text)
    candidates = (path, run_root / "tables" / path, run_root / path, REPO_ROOT / path)
    for candidate in candidates:
        if filesystem_path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"missing reference table: {path_text}")


def _wind_for_reference(ref_row: dict[str, object], wind_cache: dict[str, object]) -> tuple[object, str]:
    if str(ref_row["test_environment_mode"]).startswith("W0"):
        return None, "none"
    model_id = str(ref_row.get("updraft_model_id", ""))
    if not model_id:
        return None, "none"
    if model_id not in wind_cache:
        wind_cache[model_id] = load_updraft_model(model_id)
    return wind_cache[model_id], "panel"


def _failure_label(metrics: dict[str, float], trajectory: AgileTrajectoryResult) -> str:
    if metrics["minimum_safety_margin_m"] < 0.0:
        return "true_safety_violation"
    if metrics["terminal_heading_error_deg"] > trajectory.request.heading_success_tolerance_deg:
        return "target_miss"
    if metrics["terminal_speed_m_s"] < trajectory.request.minimum_terminal_speed_m_s:
        return "terminal_speed_below_recoverable_limit"
    if metrics["height_loss_m"] > trajectory.request.maximum_recoverable_height_loss_m:
        return "height_loss_above_recoverable_limit"
    return "success"


def _tracking_error_rms(
    time_s: np.ndarray,
    states: np.ndarray,
    reference_time_s: np.ndarray,
    reference_states: np.ndarray,
) -> float:
    errors = []
    for index, time_value in enumerate(time_s):
        reference = np.array(
            [
                np.interp(float(time_value), reference_time_s, reference_states[:, column])
                for column in range(15)
            ],
            dtype=float,
        )
        errors.append(float(np.linalg.norm(states[index] - reference)))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def _manifest(
    config: D2AgileBoundaryConfig,
    outputs: D2AgileBoundaryOutputs,
    *,
    status: str,
    scale: str,
    row_count: int,
    classification: str,
) -> dict[str, object]:
    return {
        "status": str(status),
        "run_id": int(config.run_id),
        "reference_run_id": int(config.reference_run_id),
        "d2_scale_class": str(scale),
        "row_count": int(row_count),
        "final_d2_classification": str(classification),
        "runtime_budget_hours": float(config.runtime_budget_hours),
        "runtime_buffer_fraction": float(config.runtime_buffer_fraction),
        "no_overclaiming_statement": NO_OVERCLAIMING_TEXT,
        "d1a_d1b_outputs_immutable": True,
        "wall_floor_ceiling_safety_hard": True,
        "speed_height_loss_recoverable_only_when_logged": True,
        "closed_loop_replay_required_for_success": True,
        "open_loop_optimizer_success_used_for_acceptance": False,
        "command_template_success_used_for_acceptance": False,
        "output_files": {
            "manifest_json": _path_text(outputs.manifest_json),
            "table_manifest_json": _path_text(outputs.table_manifest_json),
            "descriptors_path": _path_text(outputs.descriptors_path),
            "aggregate_csv": _path_text(outputs.aggregate_csv),
            "report_md": _path_text(outputs.report_md),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    filesystem_path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# D2 Agile Boundary Refinement",
        "",
        str(manifest["no_overclaiming_statement"]),
        "",
        f"- Status: `{manifest['status']}`",
        f"- Final classification: `{manifest['final_d2_classification']}`",
        f"- Row count: `{manifest['row_count']}`",
        f"- Closed-loop success only: `{manifest['closed_loop_replay_required_for_success']}`",
        "",
    ]
    ensure_directory(path.parent)
    filesystem_path(path).write_text("\n".join(lines), encoding="ascii")


def _write_plots(outputs: D2AgileBoundaryOutputs, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_directory(outputs.figures_dir)
    if summary.empty:
        return
    pivot = summary.pivot_table(
        index="family",
        columns="target_heading_deg",
        values="success_rate",
        aggfunc="mean",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(8, 3.6))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(pivot.columns)), [f"{float(value):.0f}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_xlabel("target heading change (deg)")
    ax.set_title("D2 nominal-latency closed-loop success rate")
    fig.colorbar(image, ax=ax, label="success rate")
    fig.tight_layout()
    suffix = outputs.manifest_json.stem.split("_")[-1]
    fig.savefig(filesystem_path(outputs.figures_dir / f"d2_agile_success_heatmap_{suffix}.png"), dpi=160)
    plt.close(fig)


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
