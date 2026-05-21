from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    AGILE_FAMILIES,
    AGILE_TARGETS_BY_FAMILY,
    AGILE_TRAJECTORY_GENERATION_METHOD,
    AgileTrajectoryRequest,
    AgileTrajectoryResult,
    command_template_reference,
    optimise_agile_trajectory,
    result_index_row,
    result_sample_rows,
)
from agile_tvlqr import TVLQRResult, synthesize_tvlqr_for_trajectory  # noqa: E402
from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
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
from updraft_models import load_updraft_model  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Configuration and Paths
# 2) Runtime Budget Gate
# 3) D1a Row Loading and Request Selection
# 4) Reference Execution and Output
# 5) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Configuration and Paths
# =============================================================================
DEFAULT_REFERENCE_ROOT = CONTROL_DIR / "05_Results" / "13_agile_trajectory_optimisation"
DEFAULT_D1A_ARCHIVE_ROOT = CONTROL_DIR / "05_Results" / "12_paired_w0_w1_archive"
NO_OVERCLAIMING_TEXT = (
    "D2 trajectory-optimised simulation boundary refinement only; success is judged "
    "from nominal-latency closed-loop TVLQR/local-feedback rollout, not open-loop "
    "optimiser rollout or command-template pulses. No production-floor completion, "
    "W2/W3/W4/W5 completion, mission success, hardware readiness, or sim-to-real "
    "completion claim is made."
)
D2_ARCHIVE_ROWS_PER_SECOND_ESTIMATE = 18.0
D2_SCALE_ROWS = {"primary": 50000, "fast": 20000}
PROFILE_REFERENCE_MIN = 16
PROFILE_REFERENCE_DEFAULT = 32
REFERENCE_LIBRARY_CHOICES = (512, 256, 128)


@dataclass(frozen=True)
class AgileReferenceConfig:
    reference_run_id: int = 17
    d1a_archive_run_id: int = 16
    d1a_planning_run_id: int = 15
    reference_root: Path | None = None
    d1a_archive_root: Path | None = None
    profile_reference_count: int = PROFILE_REFERENCE_DEFAULT
    profile_only: bool = False
    reference_library_size: str | int = "auto"
    build_tvlqr: bool = False
    runtime_budget_hours: float = 5.0
    runtime_buffer_fraction: float = 0.20
    storage_format: str = "csv_gz"
    compression_level: int = 1
    random_seed: int = 20260530
    overwrite: bool = False


@dataclass(frozen=True)
class AgileReferenceOutputs:
    root: Path
    manifest_json: Path
    budget_state_json: Path
    profile_csv: Path
    reference_index_csv: Path
    reference_samples_path: Path
    table_manifest_json: Path
    report_md: Path


def _active_reference_root(config: AgileReferenceConfig) -> Path:
    return DEFAULT_REFERENCE_ROOT if config.reference_root is None else Path(config.reference_root)


def _active_d1a_archive_root(config: AgileReferenceConfig) -> Path:
    return DEFAULT_D1A_ARCHIVE_ROOT if config.d1a_archive_root is None else Path(config.d1a_archive_root)


def _reference_run_root(config: AgileReferenceConfig) -> Path:
    return _active_reference_root(config) / f"{int(config.reference_run_id):03d}"


def _d1a_run_root(config: AgileReferenceConfig) -> Path:
    return _active_d1a_archive_root(config) / f"{int(config.d1a_archive_run_id):03d}"


def _outputs(config: AgileReferenceConfig) -> AgileReferenceOutputs:
    root = _reference_run_root(config)
    suffix = f"s{int(config.reference_run_id):03d}"
    extension = "csv.gz" if resolve_storage_format(config.storage_format) == "csv_gz" else "csv"
    return AgileReferenceOutputs(
        root=root,
        manifest_json=root / "manifests" / f"agile_reference_manifest_{suffix}.json",
        budget_state_json=root / "manifests" / f"runtime_budget_state_{suffix}.json",
        profile_csv=root / "metrics" / f"agile_reference_profile_{suffix}.csv",
        reference_index_csv=root / "metrics" / f"agile_reference_index_{suffix}.csv",
        reference_samples_path=root / "tables" / "agile_reference_samples" / f"part-00000.{extension}",
        table_manifest_json=root / "manifests" / f"table_manifest_{suffix}.json",
        report_md=root / "reports" / f"agile_reference_report_{suffix}.md",
    )


# =============================================================================
# 2) Runtime Budget Gate
# =============================================================================
def profile_runtime_gate(
    profile: pd.DataFrame,
    *,
    budget_hours: float,
    buffer_fraction: float,
    elapsed_s: float = 0.0,
) -> dict[str, object]:
    if profile.empty:
        return {
            "selected_reference_library_size": 0,
            "selected_d2_scale_class": "blocked",
            "runtime_budget_status": "blocked_no_profile_rows",
            "projections": [],
        }
    allowed_s = float(budget_hours) * 3600.0 * (1.0 - float(buffer_fraction))
    seconds_per_reference = float(profile["seconds_per_reference"].mean())
    projections: list[dict[str, object]] = []
    selected_refs = 0
    selected_scale = "blocked"
    selected_projection = float("inf")
    for refs in REFERENCE_LIBRARY_CHOICES:
        total = (
            float(elapsed_s)
            + seconds_per_reference * float(refs)
            + float(D2_SCALE_ROWS["primary"]) / D2_ARCHIVE_ROWS_PER_SECOND_ESTIMATE
            + 180.0
        )
        feasible = bool(total <= allowed_s)
        projections.append(
            {
                "reference_library_size": int(refs),
                "d2_scale_class": "primary",
                "projected_total_runtime_s": float(total),
                "feasible_under_buffered_budget": feasible,
            }
        )
        if feasible and selected_refs == 0:
            selected_refs = int(refs)
            selected_scale = "primary"
            selected_projection = float(total)
    fast_total = (
        float(elapsed_s)
        + seconds_per_reference * 128.0
        + float(D2_SCALE_ROWS["fast"]) / D2_ARCHIVE_ROWS_PER_SECOND_ESTIMATE
        + 180.0
    )
    fast_feasible = bool(fast_total <= allowed_s)
    projections.append(
        {
            "reference_library_size": 128,
            "d2_scale_class": "fast",
            "projected_total_runtime_s": float(fast_total),
            "feasible_under_buffered_budget": fast_feasible,
        }
    )
    if selected_refs == 0 and fast_feasible:
        selected_refs = 128
        selected_scale = "fast"
        selected_projection = float(fast_total)
    status = "pass" if selected_refs else "D2_execution_blocked_by_runtime_budget"
    return {
        "selected_reference_library_size": int(selected_refs),
        "selected_d2_scale_class": str(selected_scale),
        "selected_projected_total_runtime_s": float(selected_projection),
        "runtime_budget_status": status,
        "allowed_runtime_with_buffer_s": float(allowed_s),
        "seconds_per_reference_mean": seconds_per_reference,
        "optimizer_success_rate": _rate(profile, "optimizer_success"),
        "tvlqr_success_rate": _fraction_equal(profile, "tvlqr_status", "tvlqr_synthesised"),
        "local_feedback_fallback_rate": _fraction_equal(
            profile,
            "tvlqr_status",
            "local_feedback_approx",
        ),
        "closed_loop_nominal_pass_rate": _rate(profile, "closed_loop_success_flag"),
        "projections": projections,
    }


def _load_or_create_budget_state(outputs: AgileReferenceOutputs, config: AgileReferenceConfig) -> dict[str, object]:
    if filesystem_path(outputs.budget_state_json).exists():
        return json.loads(filesystem_path(outputs.budget_state_json).read_text(encoding="ascii"))
    state = {
        "runtime_budget_started_epoch_s": time.time(),
        "runtime_budget_hours": float(config.runtime_budget_hours),
        "runtime_buffer_fraction": float(config.runtime_buffer_fraction),
    }
    _write_json(outputs.budget_state_json, state)
    return state


def _elapsed_budget_s(state: dict[str, object]) -> float:
    return float(time.time() - float(state["runtime_budget_started_epoch_s"]))


# =============================================================================
# 3) D1a Row Loading and Request Selection
# =============================================================================
def load_d1a_trial_rows(config: AgileReferenceConfig) -> pd.DataFrame:
    root = _d1a_run_root(config)
    suffix = f"s{int(config.d1a_archive_run_id):03d}"
    manifest_path = root / "manifests" / f"table_manifest_{suffix}.json"
    payload = json.loads(filesystem_path(manifest_path).read_text(encoding="ascii"))
    frames = []
    for item in payload.get("tables", []):
        path = _resolve_table_path(str(item["relative_path"]), root)
        frames.append(read_table_partition(path, storage_format=str(item["storage_format"])))
    if not frames:
        raise RuntimeError("D1a archive table manifest contains no trial partitions.")
    return pd.concat(frames, ignore_index=True)


def build_agile_reference_requests(
    trials: pd.DataFrame,
    *,
    reference_count: int,
    random_seed: int,
) -> list[AgileTrajectoryRequest]:
    candidates = _eligible_trial_rows(trials)
    if candidates.empty:
        raise RuntimeError("no eligible D1a agile rows for D2 reference generation.")
    selected = _round_robin_select(candidates, max(1, int(reference_count)))
    requests = []
    for index, row in enumerate(selected.to_dict(orient="records")):
        requests.append(_request_from_trial_row(row, index=index, random_seed=random_seed))
    return requests


def build_profile_requests(
    trials: pd.DataFrame,
    *,
    profile_reference_count: int,
    random_seed: int,
) -> list[AgileTrajectoryRequest]:
    count = max(PROFILE_REFERENCE_MIN, min(PROFILE_REFERENCE_DEFAULT, int(profile_reference_count)))
    candidates = _eligible_trial_rows(trials)
    selected = _profile_family_select(candidates, count)
    return [
        _request_from_trial_row(row, index=index, random_seed=random_seed)
        for index, row in enumerate(selected.to_dict(orient="records"))
    ]


# =============================================================================
# 4) Reference Execution and Output
# =============================================================================
def execute_references(
    requests: list[AgileTrajectoryRequest],
    *,
    build_tvlqr: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[AgileTrajectoryResult], list[TVLQRResult | None]]:
    aircraft = adapt_glider(build_nausicaa_glider())
    wind_cache: dict[str, object] = {}
    index_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    trajectory_results: list[AgileTrajectoryResult] = []
    feedback_results: list[TVLQRResult | None] = []
    for request in requests:
        wind_model, wind_mode = _wind_for_request(request, wind_cache)
        result = optimise_agile_trajectory(
            request,
            aircraft=aircraft,
            wind_model=wind_model,
            wind_mode=wind_mode,
        )
        feedback = None
        if build_tvlqr:
            feedback = synthesize_tvlqr_for_trajectory(
                result,
                aircraft=aircraft,
                wind_model=wind_model,
                wind_mode=wind_mode,
                dt_s=float(request.dt_s),
            )
        row = result_index_row(result)
        if feedback is None:
            row.update(
                {
                    "controller_config_id": "",
                    "tvlqr_status": "not_requested",
                    "agile_evidence_class": "trajectory_optimised_open_loop",
                    "closed_loop_success_flag": False,
                    "closed_loop_tracking_error_rms": np.nan,
                    "latency_case": "",
                    "latency_pass_label": "",
                    "seconds_per_reference": float(result.optimizer_wall_time_s),
                }
            )
            sample_rows.extend(result_sample_rows(result))
        else:
            row.update(
                {
                    "controller_config_id": feedback.controller_config_id,
                    "tvlqr_status": feedback.tvlqr_status,
                    "agile_evidence_class": feedback.agile_evidence_class,
                    "closed_loop_success_flag": bool(feedback.closed_loop_success_flag),
                    "closed_loop_failure_label": feedback.closed_loop_failure_label,
                    "closed_loop_tracking_error_rms": feedback.closed_loop_tracking_error_rms,
                    "closed_loop_achieved_heading_deg": feedback.achieved_heading_deg,
                    "closed_loop_terminal_heading_error_deg": feedback.terminal_heading_error_deg,
                    "closed_loop_minimum_safety_margin_m": feedback.minimum_safety_margin_m,
                    "latency_case": feedback.latency_case,
                    "latency_pass_label": feedback.latency_pass_label,
                    "seconds_per_reference": float(
                        result.optimizer_wall_time_s + feedback.synthesis_wall_time_s
                    ),
                }
            )
            sample_rows.extend(
                result_sample_rows(
                    result,
                    controller_config_id=feedback.controller_config_id,
                    k_feedback=feedback.k_feedback,
                )
            )
        index_rows.append(row)
        trajectory_results.append(result)
        feedback_results.append(feedback)
    return (
        pd.DataFrame(index_rows),
        pd.DataFrame(sample_rows),
        trajectory_results,
        feedback_results,
    )


def write_reference_outputs(
    outputs: AgileReferenceOutputs,
    config: AgileReferenceConfig,
    index: pd.DataFrame,
    samples: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    ensure_directory(outputs.reference_index_csv.parent)
    index.to_csv(filesystem_path(outputs.reference_index_csv), index=False)
    partition = write_table_partition(
        samples,
        outputs.reference_samples_path,
        storage_format=config.storage_format,
        compression_level=int(config.compression_level),
    )
    table_manifest = TableManifest(
        run_id=int(config.reference_run_id),
        root=_path_text(outputs.root),
        storage_format=resolve_storage_format(config.storage_format),
        tables=(partition,),
    )
    write_table_manifest(outputs.table_manifest_json, table_manifest)
    manifest["reference_index_sha256"] = file_sha256(outputs.reference_index_csv)
    manifest["reference_samples_sha256"] = partition.checksum_sha256
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)


# =============================================================================
# 5) Public Runner and CLI
# =============================================================================
def run_agile_trajectory_optimisation(
    config: AgileReferenceConfig,
    *,
    trial_rows: pd.DataFrame | None = None,
) -> AgileReferenceOutputs:
    _validate_config(config)
    outputs = _outputs(config)
    _prepare_output_root(outputs, config)
    budget_state = _load_or_create_budget_state(outputs, config)
    trials = load_d1a_trial_rows(config) if trial_rows is None else trial_rows.copy()

    if config.profile_only:
        requests = build_profile_requests(
            trials,
            profile_reference_count=int(config.profile_reference_count),
            random_seed=int(config.random_seed),
        )
        profile, _, _, _ = execute_references(requests, build_tvlqr=bool(config.build_tvlqr))
        ensure_directory(outputs.profile_csv.parent)
        profile.to_csv(filesystem_path(outputs.profile_csv), index=False)
        gate = profile_runtime_gate(
            profile,
            budget_hours=float(config.runtime_budget_hours),
            buffer_fraction=float(config.runtime_buffer_fraction),
            elapsed_s=_elapsed_budget_s(budget_state),
        )
        manifest = _base_manifest(config, outputs, status="profile_complete")
        manifest.update(gate)
        _write_json(outputs.manifest_json, manifest)
        _write_report(outputs.report_md, manifest)
        return outputs

    profile = _load_or_run_profile(config, outputs, budget_state, trials)
    gate = profile_runtime_gate(
        profile,
        budget_hours=float(config.runtime_budget_hours),
        buffer_fraction=float(config.runtime_buffer_fraction),
        elapsed_s=_elapsed_budget_s(budget_state),
    )
    selected_count = _selected_reference_count(config.reference_library_size, gate)
    if selected_count <= 0:
        manifest = _base_manifest(
            config,
            outputs,
            status="D2_execution_blocked_by_runtime_budget",
        )
        manifest.update(gate)
        _write_json(outputs.manifest_json, manifest)
        _write_report(outputs.report_md, manifest)
        return outputs
    if _would_exceed_budget(
        budget_state,
        float(config.runtime_budget_hours),
        float(config.runtime_buffer_fraction),
        float(gate["seconds_per_reference_mean"]) * selected_count,
    ):
        manifest = _base_manifest(
            config,
            outputs,
            status="D2_execution_blocked_by_runtime_budget",
        )
        manifest.update(gate)
        _write_json(outputs.manifest_json, manifest)
        _write_report(outputs.report_md, manifest)
        return outputs

    requests = build_agile_reference_requests(
        trials,
        reference_count=selected_count,
        random_seed=int(config.random_seed),
    )
    index, samples, _, _ = execute_references(requests, build_tvlqr=bool(config.build_tvlqr))
    manifest = _base_manifest(config, outputs, status="complete")
    manifest.update(gate)
    manifest.update(
        {
            "selected_reference_library_size": int(selected_count),
            "reference_count": int(len(index)),
            "sample_row_count": int(len(samples)),
            "optimizer_success_count": int(index["optimizer_success"].astype(bool).sum()),
            "closed_loop_success_count": int(index["closed_loop_success_flag"].astype(bool).sum()),
            "reference_generation_method": AGILE_TRAJECTORY_GENERATION_METHOD,
            "d2_success_judged_from": "nominal_latency_closed_loop_tvlqr_or_labelled_local_feedback",
            "command_templates_role": "initial_guess_or_ablation_only",
        }
    )
    write_reference_outputs(outputs, config, index, samples, manifest)
    return outputs


def _validate_config(config: AgileReferenceConfig) -> None:
    if int(config.reference_run_id) <= 0:
        raise ValueError("reference_run_id must be positive.")
    if int(config.d1a_archive_run_id) <= 0 or int(config.d1a_planning_run_id) <= 0:
        raise ValueError("D1a run ids must be positive.")
    if float(config.runtime_budget_hours) <= 0.0:
        raise ValueError("runtime_budget_hours must be positive.")
    if not 0.0 <= float(config.runtime_buffer_fraction) < 1.0:
        raise ValueError("runtime_buffer_fraction must be in [0, 1).")
    resolve_storage_format(config.storage_format)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-id", type=int, default=17)
    parser.add_argument("--d1a-archive-run-id", type=int, default=16)
    parser.add_argument("--d1a-planning-run-id", type=int, default=15)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--d1a-archive-root", type=Path, default=None)
    parser.add_argument("--profile-reference-count", type=int, default=PROFILE_REFERENCE_DEFAULT)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--reference-library-size", default="auto")
    parser.add_argument("--build-tvlqr", action="store_true")
    parser.add_argument("--runtime-budget-hours", type=float, default=5.0)
    parser.add_argument("--runtime-buffer-fraction", type=float, default=0.20)
    parser.add_argument("--storage-format", default="csv_gz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_agile_trajectory_optimisation(
        AgileReferenceConfig(
            reference_run_id=args.reference_run_id,
            d1a_archive_run_id=args.d1a_archive_run_id,
            d1a_planning_run_id=args.d1a_planning_run_id,
            reference_root=args.reference_root,
            d1a_archive_root=args.d1a_archive_root,
            profile_reference_count=args.profile_reference_count,
            profile_only=bool(args.profile_only),
            reference_library_size=args.reference_library_size,
            build_tvlqr=bool(args.build_tvlqr),
            runtime_budget_hours=args.runtime_budget_hours,
            runtime_buffer_fraction=args.runtime_buffer_fraction,
            storage_format=args.storage_format,
            overwrite=bool(args.overwrite),
        )
    )
    return 0


def _prepare_output_root(outputs: AgileReferenceOutputs, config: AgileReferenceConfig) -> None:
    root = Path(outputs.root)
    if not filesystem_path(root).exists():
        filesystem_path(root).mkdir(parents=True, exist_ok=True)
        return
    if config.overwrite:
        return
    allowed_existing = {
        outputs.manifest_json.resolve(),
        outputs.budget_state_json.resolve(),
        outputs.profile_csv.resolve(),
        outputs.report_md.resolve(),
    }
    existing_files = [path for path in root.rglob("*") if path.is_file()]
    unexpected = [path for path in existing_files if path.resolve() not in allowed_existing]
    if unexpected:
        raise RuntimeError(f"reference output root is non-empty: {outputs.root}")


def _load_or_run_profile(
    config: AgileReferenceConfig,
    outputs: AgileReferenceOutputs,
    budget_state: dict[str, object],
    trials: pd.DataFrame,
) -> pd.DataFrame:
    if filesystem_path(outputs.profile_csv).exists():
        return pd.read_csv(filesystem_path(outputs.profile_csv))
    requests = build_profile_requests(
        trials,
        profile_reference_count=int(config.profile_reference_count),
        random_seed=int(config.random_seed),
    )
    profile, _, _, _ = execute_references(requests, build_tvlqr=bool(config.build_tvlqr))
    ensure_directory(outputs.profile_csv.parent)
    profile.to_csv(filesystem_path(outputs.profile_csv), index=False)
    gate = profile_runtime_gate(
        profile,
        budget_hours=float(config.runtime_budget_hours),
        buffer_fraction=float(config.runtime_buffer_fraction),
        elapsed_s=_elapsed_budget_s(budget_state),
    )
    manifest = _base_manifest(config, outputs, status="profile_complete")
    manifest.update(gate)
    _write_json(outputs.manifest_json, manifest)
    return profile


def _selected_reference_count(value: str | int, gate: dict[str, object]) -> int:
    if str(value) == "auto":
        return int(gate["selected_reference_library_size"])
    selected = int(value)
    if selected not in REFERENCE_LIBRARY_CHOICES:
        raise ValueError("reference_library_size must be auto, 512, 256, or 128.")
    return selected


def _would_exceed_budget(
    state: dict[str, object],
    budget_hours: float,
    buffer_fraction: float,
    projected_stage_s: float,
) -> bool:
    allowed = float(budget_hours) * 3600.0 * (1.0 - float(buffer_fraction))
    return _elapsed_budget_s(state) + float(projected_stage_s) > allowed


def _eligible_trial_rows(trials: pd.DataFrame) -> pd.DataFrame:
    frame = trials.copy()
    frame = frame[frame["family"].astype(str).isin(AGILE_FAMILIES)].copy()
    frame["target_heading_deg"] = pd.to_numeric(frame["target_heading_deg"], errors="coerce")
    keep = []
    for _, row in frame.iterrows():
        target = float(row["target_heading_deg"])
        keep.append(target in AGILE_TARGETS_BY_FAMILY[str(row["family"])])
    frame = frame[pd.Series(keep, index=frame.index)].copy()
    frame = frame.sort_values(
        [
            "family",
            "target_heading_deg",
            "layout_branch_id",
            "test_environment_mode",
            "direction_sign",
            "sample_id",
        ],
        kind="mergesort",
    )
    return frame


def _round_robin_select(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    group_columns = ["family", "target_heading_deg", "layout_branch_id", "direction_sign"]
    groups = [group for _, group in frame.groupby(group_columns, sort=True)]
    rows = []
    cursor = 0
    while len(rows) < count and groups:
        group = groups[cursor % len(groups)]
        local_index = len([item for item in rows if item.get("_group") == cursor % len(groups)])
        if local_index < len(group):
            row = group.iloc[local_index].to_dict()
            row["_group"] = cursor % len(groups)
            rows.append(row)
        if all(len([item for item in rows if item.get("_group") == idx]) >= len(group) for idx, group in enumerate(groups)):
            break
        cursor += 1
    if len(rows) < count:
        repeats = frame.to_dict(orient="records")
        idx = 0
        while len(rows) < count:
            rows.append(dict(repeats[idx % len(repeats)]))
            idx += 1
    for row in rows:
        row.pop("_group", None)
    return pd.DataFrame(rows[:count])


def _profile_family_select(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("no eligible D1a agile rows for D2 reference profiling.")
    family_frames = {
        family: _profile_sorted_family(frame[frame["family"].astype(str).eq(family)])
        for family in AGILE_FAMILIES
    }
    missing = [family for family, family_frame in family_frames.items() if family_frame.empty]
    if missing:
        raise RuntimeError(f"profile batch cannot cover required agile families: {missing}")
    rows: list[dict[str, object]] = []
    family_offsets = {family: 0 for family in AGILE_FAMILIES}
    cursor = 0
    while len(rows) < int(count):
        family = AGILE_FAMILIES[cursor % len(AGILE_FAMILIES)]
        family_frame = family_frames[family]
        offset = family_offsets[family] % len(family_frame)
        rows.append(family_frame.iloc[offset].to_dict())
        family_offsets[family] += 1
        cursor += 1
    return pd.DataFrame(rows)


def _profile_sorted_family(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    active = frame.copy()
    active["profile_target_priority"] = active["target_heading_deg"].map(_profile_target_priority)
    return active.sort_values(
        [
            "profile_target_priority",
            "target_heading_deg",
            "layout_branch_id",
            "test_environment_mode",
            "direction_sign",
            "sample_id",
        ],
        kind="mergesort",
    ).drop(columns=["profile_target_priority"])


def _profile_target_priority(target_heading_deg: object) -> int:
    target = float(target_heading_deg)
    if target in (15.0, 30.0):
        return 0
    if target >= 45.0:
        return 1
    return 2


def _request_from_trial_row(
    row: dict[str, object],
    *,
    index: int,
    random_seed: int,
) -> AgileTrajectoryRequest:
    x0 = np.zeros(15, dtype=float)
    x0[0:3] = [float(row["x0_w_m"]), float(row["y0_w_m"]), float(row["z0_w_m"])]
    x0[3:6] = [float(row["phi0_rad"]), float(row["theta0_rad"]), float(row["psi0_rad"])]
    x0[6:9] = [float(row["u0_m_s"]), float(row["v0_m_s"]), float(row["w0_m_s"])]
    x0[9:12] = [float(row["p0_rad_s"]), float(row["q0_rad_s"]), float(row["r0_rad_s"])]
    if not inside_bounds(x0[0:3], TRUE_SAFE_BOUNDS):
        raise ValueError("selected D1a row has unsafe initial state.")
    target = float(row["target_heading_deg"])
    horizon = 0.95 if target <= 45.0 else 1.25
    branch = str(row.get("layout_branch_id", ""))
    env = str(row.get("test_environment_mode", ""))
    family = str(row["family"])
    direction = int(float(row["direction_sign"]))
    trajectory_id = (
        f"d2_ref_{branch}_{env}_{family}_{int(target):03d}_"
        f"d{direction:+d}_{index:04d}"
    ).replace("+", "p").replace("-", "m")
    return AgileTrajectoryRequest(
        trajectory_id=trajectory_id,
        family=family,
        target_heading_deg=target,
        direction_sign=direction,
        x0=x0,
        layout_branch_id=branch,
        fan_layout=str(row.get("fan_layout", "")),
        test_environment_mode=env,
        updraft_model_id=str(row.get("updraft_model_id", "")),
        sample_id=str(row.get("sample_id", "")),
        horizon_s=horizon,
        command_knot_count=5,
        max_iterations=12,
        random_seed=int(random_seed) + int(index),
    )


def _wind_for_request(
    request: AgileTrajectoryRequest,
    wind_cache: dict[str, object],
) -> tuple[object, str]:
    if str(request.test_environment_mode).startswith("W0"):
        return None, "none"
    model_id = str(request.updraft_model_id)
    if not model_id:
        return None, "none"
    if model_id not in wind_cache:
        wind_cache[model_id] = load_updraft_model(model_id)
    return wind_cache[model_id], "panel"


def _resolve_table_path(path_text: str, run_root: Path) -> Path:
    path = Path(path_text)
    candidates = (path, run_root / "tables" / path, run_root / path, REPO_ROOT / path)
    for candidate in candidates:
        if filesystem_path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"missing D1a table partition: {path_text}")


def _base_manifest(
    config: AgileReferenceConfig,
    outputs: AgileReferenceOutputs,
    *,
    status: str,
) -> dict[str, object]:
    return {
        "status": str(status),
        "reference_run_id": int(config.reference_run_id),
        "d1a_archive_run_id": int(config.d1a_archive_run_id),
        "d1a_planning_run_id": int(config.d1a_planning_run_id),
        "runtime_budget_hours": float(config.runtime_budget_hours),
        "runtime_buffer_fraction": float(config.runtime_buffer_fraction),
        "no_overclaiming_statement": NO_OVERCLAIMING_TEXT,
        "d1a_d1b_outputs_immutable": True,
        "wall_floor_ceiling_safety_hard": True,
        "speed_height_loss_recoverable_only_when_logged": True,
        "success_judgement": "nominal_latency_closed_loop_tvlqr_or_labelled_local_feedback_only",
        "command_template_role": "initial_guess_or_ablation_only",
        "output_files": {
            "manifest_json": _path_text(outputs.manifest_json),
            "profile_csv": _path_text(outputs.profile_csv),
            "reference_index_csv": _path_text(outputs.reference_index_csv),
            "reference_samples_path": _path_text(outputs.reference_samples_path),
            "table_manifest_json": _path_text(outputs.table_manifest_json),
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
        "# D2 Agile Reference Generation",
        "",
        str(manifest["no_overclaiming_statement"]),
        "",
        f"- Status: `{manifest['status']}`",
        f"- Runtime budget status: `{manifest.get('runtime_budget_status', '')}`",
        f"- Selected references: `{manifest.get('selected_reference_library_size', '')}`",
        f"- Selected D2 scale: `{manifest.get('selected_d2_scale_class', '')}`",
        f"- Success judged from: `{manifest['success_judgement']}`",
        "",
    ]
    ensure_directory(path.parent)
    filesystem_path(path).write_text("\n".join(lines), encoding="ascii")


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(frame[column].astype(bool).mean())


def _fraction_equal(frame: pd.DataFrame, column: str, label: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(frame[column].astype(str).eq(label).mean())


if __name__ == "__main__":
    raise SystemExit(main())
