from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    file_sha256,
    filesystem_path,
    read_table_partition,
)
from run_dense_archive_pilot_sweep import (  # noqa: E402
    DensePilotSweepConfig,
    _run_pilot_replays,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Configuration and Paths
# 2) Manifest and Package Validation
# 3) Trial Loading and Summary Tables
# 4) Reproducibility Selection and Rerun
# 5) Audit Artifact Writing and CLI
# =============================================================================


# =============================================================================
# 1) Configuration and Paths
# =============================================================================
DEFAULT_ARCHIVE_ROOT = CONTROL_DIR / "05_Results" / "12_paired_w0_w1_archive"
PLANNING_CAMPAIGN = "10_dense_archive_planning"
NO_OVERCLAIMING_TEXT = (
    "D1b audit of D1a thesis-scale paired W0/W1 simulation evidence only; "
    "no production-floor completion, W2/W3/W4/W5 completion, mission success, "
    "hardware readiness, or sim-to-real completion claim is made."
)
AGILE_FAMILIES = (
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
TARGET_LADDER_DEG = (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
BRANCH_ENVIRONMENTS = {
    "single_fan_branch": {
        "fan_layout": "single_fan",
        "w0": "W0_single_fan_branch",
        "w1": "W1_single_fan",
    },
    "four_fan_branch": {
        "fan_layout": "four_fan",
        "w0": "W0_four_fan_branch",
        "w1": "W1_four_fan",
    },
}
REPRO_COMPARE_CATEGORICAL = (
    "success_flag",
    "failure_label",
    "latency_pass_label",
    "descriptor_status",
)
REPRO_COMPARE_NUMERIC = (
    "terminal_speed_m_s",
    "heading_error_deg",
    "min_true_margin_m",
    "energy_residual_m",
)


@dataclass(frozen=True)
class D1bAuditConfig:
    archive_run_id: int = 16
    planning_run_id: int = 15
    d1a_evidence_class: str = "thesis_primary"
    result_root: Path | None = None
    audit_root: Path | None = None
    build_plots: bool = False
    build_reproducibility_plan: bool = False
    execute_reproducibility_rerun: bool = False
    observed_wall_time_min: float | None = None
    expected_w0_trials_per_environment: int = 25000
    expected_w1_trials_per_environment: int = 100000
    reproducibility_total_rows: int = 2500
    reproducibility_random_seed: int = 20260528
    reproducibility_numeric_tolerance: float = 1.0e-9


@dataclass(frozen=True)
class D1bAuditOutputs:
    root: Path
    manifest_json: Path
    branch_environment_summary_csv: Path
    w1_target_ladder_summary_csv: Path
    agile_family_ladder_summary_csv: Path
    w0_failed_w1_valid_summary_csv: Path
    latency_acceptance_summary_csv: Path
    runtime_storage_summary_csv: Path
    spatial_envelope_grid_csv: Path
    updraft_relative_envelope_grid_csv: Path
    wing_exposure_envelope_grid_csv: Path
    recommendation_md: Path
    figures_dir: Path
    reproducibility_dir: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "branch_environment_summary_csv": self.branch_environment_summary_csv,
            "w1_target_ladder_summary_csv": self.w1_target_ladder_summary_csv,
            "agile_family_ladder_summary_csv": self.agile_family_ladder_summary_csv,
            "w0_failed_w1_valid_summary_csv": self.w0_failed_w1_valid_summary_csv,
            "latency_acceptance_summary_csv": self.latency_acceptance_summary_csv,
            "runtime_storage_summary_csv": self.runtime_storage_summary_csv,
            "spatial_envelope_grid_csv": self.spatial_envelope_grid_csv,
            "updraft_relative_envelope_grid_csv": self.updraft_relative_envelope_grid_csv,
            "wing_exposure_envelope_grid_csv": self.wing_exposure_envelope_grid_csv,
            "recommendation_md": self.recommendation_md,
            "figures_dir": self.figures_dir,
            "reproducibility_dir": self.reproducibility_dir,
        }


def _active_archive_root(config: D1bAuditConfig) -> Path:
    return DEFAULT_ARCHIVE_ROOT if config.result_root is None else Path(config.result_root)


def _archive_run_root(config: D1bAuditConfig) -> Path:
    return _active_archive_root(config) / f"{int(config.archive_run_id):03d}"


def _planning_run_root(config: D1bAuditConfig) -> Path:
    return (
        _active_archive_root(config).parent
        / PLANNING_CAMPAIGN
        / f"{int(config.planning_run_id):03d}"
    )


def _audit_root(config: D1bAuditConfig) -> Path:
    if config.audit_root is not None:
        return Path(config.audit_root)
    return _archive_run_root(config) / "d1b_audit"


def _outputs(config: D1bAuditConfig) -> D1bAuditOutputs:
    root = _audit_root(config)
    suffix = f"s{int(config.archive_run_id):03d}"
    return D1bAuditOutputs(
        root=root,
        manifest_json=root / f"d1b_audit_manifest_{suffix}.json",
        branch_environment_summary_csv=root / f"d1b_branch_environment_summary_{suffix}.csv",
        w1_target_ladder_summary_csv=root / f"d1b_w1_target_ladder_summary_{suffix}.csv",
        agile_family_ladder_summary_csv=root
        / f"d1b_agile_family_ladder_summary_{suffix}.csv",
        w0_failed_w1_valid_summary_csv=root / f"d1b_w0_failed_w1_valid_summary_{suffix}.csv",
        latency_acceptance_summary_csv=root / f"d1b_latency_acceptance_summary_{suffix}.csv",
        runtime_storage_summary_csv=root / f"d1b_runtime_storage_summary_{suffix}.csv",
        spatial_envelope_grid_csv=root / f"d1b_w1_spatial_envelope_grid_{suffix}.csv",
        updraft_relative_envelope_grid_csv=root
        / f"d1b_w1_updraft_relative_envelope_grid_{suffix}.csv",
        wing_exposure_envelope_grid_csv=root
        / f"d1b_w1_wing_exposure_envelope_grid_{suffix}.csv",
        recommendation_md=root / f"d2_w2_readiness_recommendation_{suffix}.md",
        figures_dir=root / "figures",
        reproducibility_dir=root / "reproducibility_rerun",
    )


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Manifest and Package Validation
# =============================================================================
def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _prepare_output_root(path: Path) -> None:
    if filesystem_path(path).exists() and any(filesystem_path(path).iterdir()):
        raise RuntimeError(f"D1b audit output root is non-empty: {path}")
    filesystem_path(path).mkdir(parents=True, exist_ok=True)


def _load_contract_manifests(config: D1bAuditConfig) -> dict[str, dict[str, object]]:
    archive_root = _archive_run_root(config)
    planning_root = _planning_run_root(config)
    suffix = f"s{int(config.archive_run_id):03d}"
    planning_suffix = f"s{int(config.planning_run_id):03d}"
    paths = {
        "planning": planning_root
        / "manifests"
        / f"paired_w0_w1_planning_manifest_{planning_suffix}.json",
        "archive": archive_root / "manifests" / f"paired_w0_w1_manifest_{suffix}.json",
        "progress": archive_root / "manifests" / f"paired_w0_w1_progress_{suffix}.json",
        "table": archive_root / "manifests" / f"table_manifest_{suffix}.json",
        "upload": archive_root / "upload_package" / "final_manifest.json",
        "governor": archive_root
        / "compressed_governor_package"
        / "governor_package_manifest.json",
    }
    missing = [name for name, path in paths.items() if not filesystem_path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing D1a audit manifest(s): {missing}")
    return {name: _read_json(path) for name, path in paths.items()}


def _validate_d1a_contract(
    config: D1bAuditConfig,
    manifests: dict[str, dict[str, object]],
) -> dict[str, object]:
    archive = manifests["archive"]
    planning = manifests["planning"]
    progress = manifests["progress"]
    expected_counts = _expected_environment_counts(config)
    errors: list[str] = []

    _require_equal(errors, "planning.run_id", planning.get("run_id"), config.planning_run_id)
    _require_equal(errors, "archive.run_id", archive.get("run_id"), config.archive_run_id)
    _require_equal(errors, "archive.planning_run_id", archive.get("planning_run_id"), config.planning_run_id)
    _require_equal(errors, "archive.d1a_evidence_class", archive.get("d1a_evidence_class"), config.d1a_evidence_class)
    _require_equal(errors, "planning.d1a_evidence_class", planning.get("d1a_evidence_class"), config.d1a_evidence_class)
    _require_equal(errors, "archive.d1a_target_contract", archive.get("d1a_target_contract"), "updated_thesis_scale_v1")
    _require_equal(errors, "archive.chunk_manifest_count", archive.get("chunk_manifest_count"), 100)
    _require_equal(errors, "progress.status", progress.get("status"), "complete")
    _require_equal(errors, "progress.completed_chunk_count", progress.get("completed_chunk_count"), 100)
    _require_equal(errors, "progress.pending_chunk_count", progress.get("pending_chunk_count"), 0)
    _require_equal(errors, "progress.failed_chunk_count", progress.get("failed_chunk_count"), 0)

    actual_counts = {
        str(key): int(value)
        for key, value in dict(archive.get("trial_count_by_environment", {})).items()
    }
    if actual_counts != expected_counts:
        errors.append(
            "trial_count_by_environment mismatch: "
            f"actual={actual_counts}, expected={expected_counts}"
        )
    _require_equal(errors, "archive.trial_count_total", archive.get("trial_count_total"), sum(expected_counts.values()))
    _require_true(errors, "archive.w1_scheduled_independent_of_w0_success", archive)
    _require_true(errors, "archive.single_fan_and_four_fan_never_merged", archive)
    _require_true(errors, "archive.branch_local_decisions_only", archive)
    _require_false(errors, "archive.governor_package_contains_w0_candidates", archive)
    _require_false(errors, "archive.governor_artifacts_scan_raw_tables", archive)
    _require_true(errors, "archive.governor_package_branch_local_only", archive)
    _require_false(errors, "archive.w2_w3_w4_w5_performed", archive)
    _require_false(errors, "archive.hardware_or_mission_claim", archive)
    _require_false(errors, "archive.sim_to_real_transfer_claim", archive)
    errors.extend(_no_overclaiming_errors(str(archive.get("no_overclaiming_statement", ""))))

    if errors:
        raise RuntimeError("D1a audit contract validation failed: " + "; ".join(errors))
    return {
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "planning_manifest_valid": True,
        "archive_manifest_valid": True,
        "progress_manifest_valid": True,
    }


def _expected_environment_counts(config: D1bAuditConfig) -> dict[str, int]:
    return {
        "W0_single_fan_branch": int(config.expected_w0_trials_per_environment),
        "W0_four_fan_branch": int(config.expected_w0_trials_per_environment),
        "W1_single_fan": int(config.expected_w1_trials_per_environment),
        "W1_four_fan": int(config.expected_w1_trials_per_environment),
    }


def _require_equal(errors: list[str], name: str, actual: object, expected: object) -> None:
    if actual != expected:
        errors.append(f"{name}={actual!r}, expected {expected!r}")


def _require_true(errors: list[str], name: str, payload: dict[str, object]) -> None:
    if payload.get(name.rsplit(".", 1)[-1]) is not True:
        errors.append(f"{name} is not true")


def _require_false(errors: list[str], name: str, payload: dict[str, object]) -> None:
    if payload.get(name.rsplit(".", 1)[-1]) is not False:
        errors.append(f"{name} is not false")


def _no_overclaiming_errors(statement: str) -> list[str]:
    errors: list[str] = []
    text = str(statement).lower()
    required_denials = (
        "no production-floor completion",
        "w2/w3/w4/w5 completion",
        "mission success",
        "hardware readiness",
        "sim-to-real completion",
    )
    for phrase in required_denials:
        if phrase not in text:
            errors.append(f"no-overclaiming statement missing denial: {phrase}")
    return errors


def _validate_table_manifest(
    config: D1bAuditConfig,
    manifest: dict[str, object],
) -> dict[str, object]:
    archive_root = _archive_run_root(config)
    partitions = list(manifest.get("tables", []))
    if not partitions:
        raise RuntimeError("D1a table manifest has no partitions.")
    rows = 0
    bytes_total = 0
    for item in partitions:
        partition = dict(item)
        path = _resolve_table_path(str(partition["relative_path"]), archive_root)
        checksum = file_sha256(path)
        if checksum != str(partition["checksum_sha256"]):
            raise RuntimeError(f"checksum mismatch for D1a partition: {path}")
        rows += int(partition["row_count"])
        bytes_total += int(partition["byte_count"])
    return {
        "table_manifest_validation_status": "pass",
        "table_manifest_partition_count": len(partitions),
        "table_manifest_row_count": rows,
        "table_manifest_byte_count": bytes_total,
    }


def _validate_packages(config: D1bAuditConfig) -> dict[str, object]:
    archive_root = _archive_run_root(config)
    upload_root = archive_root / "upload_package"
    governor_root = archive_root / "compressed_governor_package"
    if not filesystem_path(upload_root).exists():
        raise RuntimeError("missing upload package.")
    if not filesystem_path(governor_root).exists():
        raise RuntimeError("missing compressed governor package.")
    upload_table_files = _files_with_path_part(upload_root, "tables")
    governor_table_files = _files_with_path_part(governor_root, "tables")
    if upload_table_files:
        raise RuntimeError("upload package contains raw tables.")
    if governor_table_files:
        raise RuntimeError("governor package contains raw tables.")
    governor_w0_files = [
        path for path in _iter_files(governor_root)
        if "W0_" in path.name or "W0_" in path.as_posix()
    ]
    if governor_w0_files:
        raise RuntimeError("governor package contains W0 candidates or metadata.")
    return {
        "upload_package_raw_tables_included": False,
        "governor_package_raw_tables_included": False,
        "governor_package_contains_w0_candidates": False,
        "upload_package_file_count": len(_iter_files(upload_root)),
        "governor_package_file_count": len(_iter_files(governor_root)),
    }


# =============================================================================
# 3) Trial Loading and Summary Tables
# =============================================================================
def _load_trial_rows(config: D1bAuditConfig, table_manifest: dict[str, object]) -> pd.DataFrame:
    archive_root = _archive_run_root(config)
    frames: list[pd.DataFrame] = []
    for item in table_manifest.get("tables", []):
        partition = dict(item)
        path = _resolve_table_path(str(partition["relative_path"]), archive_root)
        frames.append(read_table_partition(path, storage_format=str(partition["storage_format"])))
    if not frames:
        raise RuntimeError("D1b audit found no trial partitions to load.")
    return pd.concat(frames, ignore_index=True)


def build_branch_environment_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in trials.groupby(
        ["layout_branch_id", "fan_layout", "test_environment_mode"],
        sort=True,
        dropna=False,
    ):
        success = _success_series(group)
        rows.append(
            {
                "layout_branch_id": keys[0],
                "fan_layout": keys[1],
                "test_environment_mode": keys[2],
                "row_count": int(len(group)),
                "success_count": int(success.sum()),
                "failure_count": int(len(group) - success.sum()),
                "nominal_pass_count": _count_label(group, "latency_pass_label", "nominal_pass"),
                "latency_label_counts_json": _counts_json(group, "latency_pass_label"),
                "latency_case_counts_json": _counts_json(group, "latency_case"),
            }
        )
    return pd.DataFrame(rows)


def build_w1_target_ladder_summary(trials: pd.DataFrame) -> pd.DataFrame:
    w1 = _w1_rows(trials)
    rows = [
        _ladder_row(keys, group, include_start_class=True)
        for keys, group in w1.groupby(
            [
                "fan_layout",
                "layout_branch_id",
                "family",
                "target_heading_deg",
                "direction_sign",
                "start_class",
            ],
            sort=True,
            dropna=False,
        )
    ]
    return pd.DataFrame(rows)


def build_agile_family_ladder_summary(trials: pd.DataFrame) -> pd.DataFrame:
    w1 = _w1_rows(trials)
    branch_rows = (
        w1[["fan_layout", "layout_branch_id"]]
        .drop_duplicates()
        .sort_values(["fan_layout", "layout_branch_id"])
        .to_dict(orient="records")
    )
    rows: list[dict[str, object]] = []
    for branch in branch_rows:
        branch_filter = (
            w1["fan_layout"].astype(str).eq(str(branch["fan_layout"]))
            & w1["layout_branch_id"].astype(str).eq(str(branch["layout_branch_id"]))
        )
        for family in AGILE_FAMILIES:
            for target in TARGET_LADDER_DEG:
                for direction in (-1, 1):
                    group = w1[
                        branch_filter
                        & w1["family"].astype(str).eq(family)
                        & pd.to_numeric(w1["target_heading_deg"], errors="coerce").eq(target)
                        & pd.to_numeric(w1["direction_sign"], errors="coerce").eq(direction)
                    ]
                    rows.append(
                        _ladder_row(
                            (
                                branch["fan_layout"],
                                branch["layout_branch_id"],
                                family,
                                target,
                                direction,
                            ),
                            group,
                            include_start_class=False,
                        )
                    )
    return pd.DataFrame(rows)


def build_latency_acceptance_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in trials.groupby(
        ["test_environment_mode", "latency_case", "latency_pass_label"],
        sort=True,
        dropna=False,
    ):
        rows.append(
            {
                "test_environment_mode": keys[0],
                "latency_case": keys[1],
                "latency_pass_label": keys[2],
                "trial_count": int(len(group)),
                "interpretation": (
                    "nominal open-loop evidence is not hardware-ready "
                    "delayed-state-feedback evidence"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_runtime_storage_summary(
    config: D1bAuditConfig,
    archive_manifest: dict[str, object],
    table_validation: dict[str, object],
) -> pd.DataFrame:
    archive_root = _archive_run_root(config)
    rows = int(archive_manifest.get("trial_count_total", 0))
    wall_time_s = (
        float(config.observed_wall_time_min) * 60.0
        if config.observed_wall_time_min is not None
        else math.nan
    )
    rows_per_s = rows / wall_time_s if wall_time_s and math.isfinite(wall_time_s) else math.nan
    sizes = {
        "archive_visible": _tree_size(archive_root),
        "trial_partitions": _tree_size(archive_root / "tables"),
        "upload_package": _tree_size(archive_root / "upload_package"),
        "compressed_governor_package": _tree_size(archive_root / "compressed_governor_package"),
        "d1b_audit": _tree_size(_audit_root(config)),
    }
    return pd.DataFrame(
        [
            {
                "archive_run_id": int(config.archive_run_id),
                "planning_run_id": int(config.planning_run_id),
                "trial_count_total": rows,
                "observed_wall_time_min": config.observed_wall_time_min,
                "observed_rows_per_second": rows_per_s,
                "selected_worker_count": archive_manifest.get("selected_worker_count"),
                "worker_fallback_reason": archive_manifest.get("worker_fallback_reason"),
                "storage_format": archive_manifest.get("storage_format"),
                "table_manifest_partition_count": table_validation[
                    "table_manifest_partition_count"
                ],
                "table_manifest_byte_count": table_validation["table_manifest_byte_count"],
                "trial_partition_size_mib": _mib(sizes["trial_partitions"][1]),
                "visible_archive_size_mib": _mib(sizes["archive_visible"][1]),
                "upload_package_size_mib": _mib(sizes["upload_package"][1]),
                "governor_package_size_mib": _mib(sizes["compressed_governor_package"][1]),
                "d1b_audit_size_mib": _mib(sizes["d1b_audit"][1]),
            }
        ]
    )


def build_spatial_envelope_grid(trials: pd.DataFrame) -> pd.DataFrame:
    w1 = _w1_rows(trials).copy()
    w1["x_bin_m"] = _bin_series(w1, "x0_w_m", 0.50, "x")
    w1["y_bin_m"] = _bin_series(w1, "y0_w_m", 0.50, "y")
    w1["z_bin_m"] = _bin_series(w1, "z0_w_m", 0.25, "z")
    return _grid_summary(w1, ["layout_branch_id", "fan_layout", "test_environment_mode", "x_bin_m", "y_bin_m", "z_bin_m"])


def build_updraft_relative_envelope_grid(trials: pd.DataFrame) -> pd.DataFrame:
    w1 = _w1_rows(trials).copy()
    w1["radius_bin_m"] = _bin_series(w1, "updraft_relative_radius_m", 0.25, "r")
    w1["azimuth_bin_deg"] = _bin_series(
        w1.assign(
            _azimuth_deg=pd.to_numeric(
                w1["updraft_relative_azimuth_rad"],
                errors="coerce",
            )
            * 180.0
            / math.pi
        ),
        "_azimuth_deg",
        30.0,
        "az",
    )
    return _grid_summary(w1, ["layout_branch_id", "fan_layout", "test_environment_mode", "radius_bin_m", "azimuth_bin_deg"])


def build_wing_exposure_envelope_grid(trials: pd.DataFrame) -> pd.DataFrame:
    w1 = _w1_rows(trials).copy()
    w1["wing_mean_bin_m_s"] = _bin_series(w1, "w_wing_mean_m_s", 0.10, "ww")
    w1["delta_w_lr_bin_m_s"] = _bin_series(w1, "delta_w_lr_m_s", 0.10, "dw")
    return _grid_summary(w1, ["layout_branch_id", "fan_layout", "test_environment_mode", "wing_mean_bin_m_s", "delta_w_lr_bin_m_s"])


def _ladder_row(
    keys: tuple[object, ...],
    group: pd.DataFrame,
    *,
    include_start_class: bool,
) -> dict[str, object]:
    if include_start_class:
        fan_layout, branch, family, target, direction, start_class = keys
    else:
        fan_layout, branch, family, target, direction = keys
        start_class = "all"
    success = _success_series(group)
    nominal = _label_series(group, "latency_pass_label", "nominal_pass")
    trial_count = int(len(group))
    success_count = int(success.sum()) if trial_count else 0
    nominal_count = int(nominal.sum()) if trial_count else 0
    return {
        "fan_layout": fan_layout,
        "layout_branch_id": branch,
        "family": family,
        "target_heading_deg": float(target),
        "direction_sign": int(direction),
        "start_class": start_class,
        "trial_count": trial_count,
        "success_count": success_count,
        "success_rate": _rate(success_count, trial_count),
        "nominal_pass_count": nominal_count,
        "nominal_pass_rate": _rate(nominal_count, trial_count),
        "failure_label_distribution_json": _counts_json(group, "failure_label"),
        "min_true_margin_m_min": _nanmin(group, "min_true_margin_m"),
        "min_true_margin_m_median": _nanmedian(group, "min_true_margin_m"),
        "energy_residual_m_median": _nanmedian(group, "energy_residual_m"),
        "heading_error_deg_median": _nanmedian(group, "heading_error_deg"),
        "lift_dwell_fraction_median": _nanmedian(group, "lift_dwell_fraction"),
    }


def _grid_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        success = _success_series(group)
        nominal = _label_series(group, "latency_pass_label", "nominal_pass")
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "trial_count": int(len(group)),
                "success_count": int(success.sum()),
                "success_rate": _rate(int(success.sum()), int(len(group))),
                "nominal_pass_count": int(nominal.sum()),
                "nominal_pass_rate": _rate(int(nominal.sum()), int(len(group))),
                "median_heading_error_deg": _nanmedian(group, "heading_error_deg"),
                "median_energy_residual_m": _nanmedian(group, "energy_residual_m"),
                "median_lift_dwell_fraction": _nanmedian(group, "lift_dwell_fraction"),
                "min_true_margin_m_min": _nanmin(group, "min_true_margin_m"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _w1_rows(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[trials["test_environment_mode"].astype(str).str.startswith("W1_")]


# =============================================================================
# 4) Reproducibility Selection and Rerun
# =============================================================================
def select_reproducibility_rows(
    trials: pd.DataFrame,
    *,
    total_rows: int = 2500,
) -> pd.DataFrame:
    rows_per_branch = int(total_rows) // 2
    if rows_per_branch * 2 != int(total_rows):
        raise ValueError("reproducibility total_rows must be even.")
    w0_per_branch = int(round(rows_per_branch * 0.20))
    w1_per_branch = rows_per_branch - w0_per_branch
    selections: list[pd.DataFrame] = []
    for branch, envs in BRANCH_ENVIRONMENTS.items():
        branch_rows = trials[trials["layout_branch_id"].astype(str).eq(branch)]
        w0 = branch_rows[branch_rows["test_environment_mode"].astype(str).eq(envs["w0"])]
        w1 = branch_rows[branch_rows["test_environment_mode"].astype(str).eq(envs["w1"])]
        selections.append(_stratified_selection(w0, w0_per_branch))
        selections.append(_stratified_selection(w1, w1_per_branch))
    selected = pd.concat(selections, ignore_index=True)
    if len(selected) != int(total_rows):
        raise RuntimeError(
            f"reproducibility selection row count mismatch: {len(selected)}"
        )
    selected = selected.copy()
    selected["reproducibility_selection_id"] = [
        f"repro_s016_{index:05d}" for index in range(len(selected))
    ]
    return selected


def _stratified_selection(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(frame) < int(count):
        raise RuntimeError("not enough rows for reproducibility selection.")
    work = frame.copy()
    status = work["failure_label"].astype(str)
    status = status.where(~_success_series(work), "success")
    work["_repro_status"] = status
    work["_stable_hash"] = [
        _stable_hash(row)
        for row in work[
            [
                "layout_branch_id",
                "test_environment_mode",
                "family",
                "target_heading_deg",
                "direction_sign",
                "start_class",
                "_repro_status",
                "trial_descriptor_id",
            ]
        ].to_dict(orient="records")
    ]
    strata = [
        "family",
        "target_heading_deg",
        "direction_sign",
        "start_class",
        "_repro_status",
    ]
    grouped = list(work.groupby(strata, sort=True, dropna=False))
    quotas = _proportional_quotas([len(group) for _, group in grouped], int(count))
    selected_parts = []
    for quota, (_keys, group) in zip(quotas, grouped):
        if quota <= 0:
            continue
        selected_parts.append(group.sort_values("_stable_hash").head(int(quota)))
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != int(count):
        raise RuntimeError("stratified reproducibility selection quota mismatch.")
    return selected.drop(columns=["_repro_status", "_stable_hash"])


def _proportional_quotas(sizes: list[int], total: int) -> list[int]:
    raw = [float(total) * float(size) / float(sum(sizes)) for size in sizes]
    quotas = [int(math.floor(value)) for value in raw]
    remaining = int(total) - sum(quotas)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - quotas[index], sizes[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return quotas


def execute_reproducibility_rerun(
    config: D1bAuditConfig,
    selected: pd.DataFrame,
    original_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rerun_frames: list[pd.DataFrame] = []
    planning_rows = _load_planning_rows_for_selection(config, selected)
    for (branch, environment), env_selection in selected.groupby(
        ["layout_branch_id", "test_environment_mode"],
        sort=True,
    ):
        candidate_rows = planning_rows["candidates"]
        start_rows = planning_rows["starts"]
        candidate_ids = set(env_selection["candidate_id"].astype(str))
        sample_ids = set(env_selection["sample_id"].astype(str))
        candidates = candidate_rows[candidate_rows["candidate_id"].astype(str).isin(candidate_ids)]
        starts = start_rows[
            start_rows["layout_branch_id"].astype(str).eq(str(branch))
            & start_rows["test_environment_mode"].astype(str).eq(str(environment))
            & start_rows["sample_id"].astype(str).isin(sample_ids)
        ]
        candidates = candidates.sort_values("candidate_id").to_dict(orient="records")
        replay_config = DensePilotSweepConfig(
            run_id=int(config.archive_run_id),
            planning_run_id=int(config.planning_run_id),
            max_trials=len(candidates),
            latency_case="nominal",
            random_seed=int(config.reproducibility_random_seed),
        )
        rerun_frames.append(_run_pilot_replays(starts, candidates, replay_config))
    rerun = pd.concat(rerun_frames, ignore_index=True)
    comparison = compare_reproducibility_rows(
        selected,
        rerun,
        tolerance=float(config.reproducibility_numeric_tolerance),
    )
    return rerun, comparison


def _load_planning_rows_for_selection(
    config: D1bAuditConfig,
    selected: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    planning_root = _planning_run_root(config)
    suffix = f"s{int(config.planning_run_id):03d}"
    table_manifest = _read_json(planning_root / "manifests" / f"table_manifest_{suffix}.json")
    candidate_ids = set(selected["candidate_id"].astype(str))
    sample_keys = set(
        zip(
            selected["layout_branch_id"].astype(str),
            selected["test_environment_mode"].astype(str),
            selected["sample_id"].astype(str),
        )
    )
    candidate_frames: list[pd.DataFrame] = []
    start_frames: list[pd.DataFrame] = []
    for item in table_manifest.get("tables", []):
        partition = dict(item)
        path = _resolve_table_path(str(partition["relative_path"]), planning_root)
        table_name = str(partition["table_name"])
        if table_name == "candidate_index":
            frame = read_table_partition(path, storage_format=str(partition["storage_format"]))
            filtered = frame[frame["candidate_id"].astype(str).isin(candidate_ids)]
            if not filtered.empty:
                candidate_frames.append(filtered)
        elif table_name == "start_states":
            frame = read_table_partition(path, storage_format=str(partition["storage_format"]))
            keys = list(
                zip(
                    frame["layout_branch_id"].astype(str),
                    frame["test_environment_mode"].astype(str),
                    frame["sample_id"].astype(str),
                )
            )
            mask = pd.Series([key in sample_keys for key in keys], index=frame.index)
            filtered = frame[mask]
            if not filtered.empty:
                start_frames.append(filtered)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    starts = pd.concat(start_frames, ignore_index=True)
    if len(candidates) != len(selected):
        raise RuntimeError("planning candidate match count does not equal selection.")
    if len(starts) != len(selected):
        raise RuntimeError("planning start-state match count does not equal selection.")
    return {"candidates": candidates, "starts": starts}


def compare_reproducibility_rows(
    selected: pd.DataFrame,
    rerun: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    keys = ["layout_branch_id", "test_environment_mode", "candidate_id", "sample_id"]
    original = selected.set_index(keys, drop=False)
    replay = rerun.set_index(keys, drop=False)
    rows = []
    for key, original_row in original.iterrows():
        rerun_row = replay.loc[key]
        if isinstance(rerun_row, pd.DataFrame):
            rerun_row = rerun_row.iloc[0]
        categorical_match = all(
            str(original_row[field]) == str(rerun_row[field])
            for field in REPRO_COMPARE_CATEGORICAL
            if field in original_row.index and field in rerun_row.index
        )
        numeric_max_abs_diff = 0.0
        numeric_match = True
        for field in REPRO_COMPARE_NUMERIC:
            if field not in original_row.index or field not in rerun_row.index:
                continue
            diff = _numeric_diff(original_row[field], rerun_row[field])
            if math.isfinite(diff):
                numeric_max_abs_diff = max(numeric_max_abs_diff, diff)
                numeric_match = numeric_match and diff <= float(tolerance)
        rows.append(
            {
                "layout_branch_id": key[0],
                "test_environment_mode": key[1],
                "candidate_id": key[2],
                "sample_id": key[3],
                "categorical_match": bool(categorical_match),
                "numeric_match": bool(numeric_match),
                "numeric_max_abs_diff": numeric_max_abs_diff,
                "reproducibility_match": bool(categorical_match and numeric_match),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 5) Audit Artifact Writing and CLI
# =============================================================================
def run_paired_w0_w1_d1b_audit(
    *,
    archive_run_id: int = 16,
    planning_run_id: int = 15,
    d1a_evidence_class: str = "thesis_primary",
    result_root: Path | None = None,
    audit_root: Path | None = None,
    build_plots: bool = False,
    build_reproducibility_plan: bool = False,
    execute_reproducibility_rerun: bool = False,
    observed_wall_time_min: float | None = None,
) -> dict[str, Path]:
    config = D1bAuditConfig(
        archive_run_id=int(archive_run_id),
        planning_run_id=int(planning_run_id),
        d1a_evidence_class=str(d1a_evidence_class),
        result_root=result_root,
        audit_root=audit_root,
        build_plots=bool(build_plots),
        build_reproducibility_plan=bool(build_reproducibility_plan),
        execute_reproducibility_rerun=bool(execute_reproducibility_rerun),
        observed_wall_time_min=observed_wall_time_min,
    )
    return run_d1b_audit(config).as_dict()


def run_d1b_audit(config: D1bAuditConfig) -> D1bAuditOutputs:
    outputs = _outputs(config)
    _prepare_output_root(outputs.root)
    manifests = _load_contract_manifests(config)
    contract_validation = _validate_d1a_contract(config, manifests)
    table_validation = _validate_table_manifest(config, manifests["table"])
    package_validation = _validate_packages(config)
    trials = _load_trial_rows(config, manifests["table"])
    _validate_loaded_counts(trials, contract_validation["expected_counts"])

    branch_summary = build_branch_environment_summary(trials)
    ladder_summary = build_w1_target_ladder_summary(trials)
    agile_summary = build_agile_family_ladder_summary(trials)
    latency_summary = build_latency_acceptance_summary(trials)
    runtime_storage = build_runtime_storage_summary(
        config,
        manifests["archive"],
        table_validation,
    )
    spatial_grid = build_spatial_envelope_grid(trials)
    updraft_grid = build_updraft_relative_envelope_grid(trials)
    wing_grid = build_wing_exposure_envelope_grid(trials)
    w0_failed = _read_existing_w0_failed_summary(config)

    _write_csv(outputs.branch_environment_summary_csv, branch_summary)
    _write_csv(outputs.w1_target_ladder_summary_csv, ladder_summary)
    _write_csv(outputs.agile_family_ladder_summary_csv, agile_summary)
    _write_csv(outputs.w0_failed_w1_valid_summary_csv, w0_failed)
    _write_csv(outputs.latency_acceptance_summary_csv, latency_summary)
    _write_csv(outputs.runtime_storage_summary_csv, runtime_storage)
    _write_csv(outputs.spatial_envelope_grid_csv, spatial_grid)
    _write_csv(outputs.updraft_relative_envelope_grid_csv, updraft_grid)
    _write_csv(outputs.wing_exposure_envelope_grid_csv, wing_grid)

    reproducibility_payload = {"status": "not_requested", "mismatch_count": None}
    if config.build_reproducibility_plan or config.execute_reproducibility_rerun:
        filesystem_path(outputs.reproducibility_dir).mkdir(parents=True, exist_ok=True)
        selected = select_reproducibility_rows(
            trials,
            total_rows=int(config.reproducibility_total_rows),
        )
        _write_csv(
            outputs.reproducibility_dir
            / f"d1b_reproducibility_selection_s{int(config.archive_run_id):03d}.csv",
            selected,
        )
        reproducibility_payload = _reproducibility_manifest(
            config,
            selected,
            status="planned",
        )
        if config.execute_reproducibility_rerun:
            rerun, comparison = execute_reproducibility_rerun(config, selected, trials)
            _write_csv(
                outputs.reproducibility_dir
                / f"d1b_reproducibility_rerun_descriptors_s{int(config.archive_run_id):03d}.csv",
                rerun,
            )
            _write_csv(
                outputs.reproducibility_dir
                / f"d1b_reproducibility_comparison_s{int(config.archive_run_id):03d}.csv",
                comparison,
            )
            mismatch_count = int((~comparison["reproducibility_match"]).sum())
            reproducibility_payload = _reproducibility_manifest(
                config,
                selected,
                status="executed",
                comparison=comparison,
            )
            reproducibility_payload["mismatch_count"] = mismatch_count
        _write_json(
            outputs.reproducibility_dir
            / f"d1b_reproducibility_manifest_s{int(config.archive_run_id):03d}.json",
            reproducibility_payload,
        )

    if config.build_plots:
        _write_plots(outputs, branch_summary, agile_summary, w0_failed, runtime_storage)

    runtime_storage = build_runtime_storage_summary(
        config,
        manifests["archive"],
        table_validation,
    )
    _write_csv(outputs.runtime_storage_summary_csv, runtime_storage)

    classification = (
        "ready_for_D2_boundary_refinement"
        if int(reproducibility_payload.get("mismatch_count") or 0) == 0
        else "D1b_audit_blocked_by_reproducibility_mismatch"
    )
    recommendation = _readiness_recommendation(
        classification=classification,
        agile_summary=agile_summary,
        w0_failed=w0_failed,
    )
    filesystem_path(outputs.recommendation_md).write_text(recommendation, encoding="ascii")
    manifest = _audit_manifest(
        config=config,
        manifests=manifests,
        contract_validation=contract_validation,
        table_validation=table_validation,
        package_validation=package_validation,
        branch_summary=branch_summary,
        runtime_storage=runtime_storage,
        reproducibility=reproducibility_payload,
        classification=classification,
        outputs=outputs,
    )
    _write_json(outputs.manifest_json, manifest)
    return outputs


def _validate_loaded_counts(trials: pd.DataFrame, expected: dict[str, int]) -> None:
    actual = {
        str(key): int(value)
        for key, value in trials["test_environment_mode"].value_counts().to_dict().items()
    }
    if actual != expected:
        raise RuntimeError(f"loaded D1a row counts mismatch: {actual} != {expected}")


def _read_existing_w0_failed_summary(config: D1bAuditConfig) -> pd.DataFrame:
    suffix = f"s{int(config.archive_run_id):03d}"
    path = (
        _archive_run_root(config)
        / "metrics_summary"
        / f"w0_failed_w1_valid_summary_{suffix}.csv"
    )
    if not filesystem_path(path).exists():
        return pd.DataFrame(
            columns=["paired_summary_label", "layout_branch_id", "fan_layout", "trial_count"]
        )
    return pd.read_csv(filesystem_path(path))


def _reproducibility_manifest(
    config: D1bAuditConfig,
    selected: pd.DataFrame,
    *,
    status: str,
    comparison: pd.DataFrame | None = None,
) -> dict[str, object]:
    by_branch = selected.groupby("layout_branch_id").size().to_dict()
    by_environment = selected.groupby("test_environment_mode").size().to_dict()
    payload: dict[str, object] = {
        "status": status,
        "selection_row_count": int(len(selected)),
        "selection_by_branch": {str(key): int(value) for key, value in by_branch.items()},
        "selection_by_environment": {
            str(key): int(value) for key, value in by_environment.items()
        },
        "selection_rule": (
            "1 percent per branch, preserving approximately 250 W0 and "
            "1000 W1 rows per branch for thesis-primary D1a"
        ),
        "numeric_tolerance": float(config.reproducibility_numeric_tolerance),
        "mismatch_count": None,
    }
    if comparison is not None:
        payload["comparison_row_count"] = int(len(comparison))
        payload["mismatch_count"] = int((~comparison["reproducibility_match"]).sum())
    return payload


def _audit_manifest(
    *,
    config: D1bAuditConfig,
    manifests: dict[str, dict[str, object]],
    contract_validation: dict[str, object],
    table_validation: dict[str, object],
    package_validation: dict[str, object],
    branch_summary: pd.DataFrame,
    runtime_storage: pd.DataFrame,
    reproducibility: dict[str, object],
    classification: str,
    outputs: D1bAuditOutputs,
) -> dict[str, object]:
    runtime_row = runtime_storage.iloc[0].to_dict()
    return {
        "archive_run_id": int(config.archive_run_id),
        "planning_run_id": int(config.planning_run_id),
        "code_commit": _git_commit(),
        "d1a_evidence_class": str(config.d1a_evidence_class),
        "d1a_target_contract": manifests["archive"].get("d1a_target_contract"),
        "row_counts_by_environment": contract_validation["actual_counts"],
        "row_counts_by_branch": manifests["archive"].get("trial_count_by_branch"),
        "chunk_count": manifests["archive"].get("chunk_manifest_count"),
        "completion_status": manifests["progress"].get("status"),
        "checksum_table_manifest_validation_status": table_validation[
            "table_manifest_validation_status"
        ],
        "storage_format": manifests["archive"].get("storage_format"),
        "storage_size": {
            "trial_partition_size_mib": runtime_row["trial_partition_size_mib"],
            "visible_archive_size_mib": runtime_row["visible_archive_size_mib"],
            "upload_package_size_mib": runtime_row["upload_package_size_mib"],
            "governor_package_size_mib": runtime_row["governor_package_size_mib"],
            "d1b_audit_size_mib": runtime_row["d1b_audit_size_mib"],
        },
        "execution_time": {
            "observed_wall_time_min": runtime_row["observed_wall_time_min"],
            "observed_rows_per_second": runtime_row["observed_rows_per_second"],
            "worker_count": runtime_row["selected_worker_count"],
            "fallback_reason": runtime_row["worker_fallback_reason"],
        },
        "package_integrity": package_validation,
        "no_overclaiming_statement": NO_OVERCLAIMING_TEXT,
        "reproducibility_check": reproducibility,
        "branch_environment_summary_rows": int(len(branch_summary)),
        "output_files": {
            key: _path_text(path)
            for key, path in outputs.as_dict().items()
            if key not in {"root", "figures_dir", "reproducibility_dir"}
        },
        "final_d1b_classification": classification,
    }


def _readiness_recommendation(
    *,
    classification: str,
    agile_summary: pd.DataFrame,
    w0_failed: pd.DataFrame,
) -> str:
    winners = agile_summary[agile_summary["success_count"].astype(int) > 0].copy()
    winners = winners.sort_values(["success_rate", "success_count"], ascending=False).head(12)
    higher_targets = agile_summary[
        pd.to_numeric(agile_summary["target_heading_deg"], errors="coerce") >= 45.0
    ]
    higher_success = int(higher_targets["success_count"].astype(int).sum())
    w0_lines = [
        f"- {row['paired_summary_label']}: {int(row['trial_count'])}"
        for row in w0_failed.to_dict(orient="records")
    ]
    winner_lines = [
        (
            f"- {row['layout_branch_id']} {row['family']} "
            f"{float(row['target_heading_deg']):.0f} deg dir {int(row['direction_sign'])}: "
            f"{int(row['success_count'])}/{int(row['trial_count'])} "
            f"({float(row['success_rate']):.4f})"
        )
        for row in winners.to_dict(orient="records")
    ]
    return "\n".join(
        [
            "# D2/W2 Readiness Recommendation",
            "",
            f"Classification: `{classification}`.",
            "",
            "D1a is sufficient to proceed to D2 boundary refinement. It remains "
            "D1a thesis-scale simulation evidence only and does not complete W2, "
            "W3, W4, W5, mission, hardware, or sim-to-real validation.",
            "",
            "## W1 Target-Ladder Findings",
            *(winner_lines or ["- No agile W1 successes were found."]),
            f"- Agile targets at 45 deg and above produced {higher_success} W1 successes.",
            "",
            "## W0-Failed / W1-Valid Counts",
            *(w0_lines or ["- No W0-failed/W1-valid summary rows were available."]),
            "",
            "## Recommended Next Run",
            "- Run D2 boundary refinement before W2 complex-updraft replay.",
            "- Use 5,000 to 20,000 extra W1-focused or paired cases per fan-layout branch.",
            "- Concentrate D2 on 15 deg and weak 30 deg agile cases, W1 boundary cells, "
            "W0/W1 disagreements, updraft-edge samples, and safety-margin boundaries.",
            "- Prepare W2 with W1 winners, boundary cases, and W0/W1 disagreement cases, "
            "but execute W2 only after D2 confirms a stable branch-local shortlist.",
            "- W2 reduced preflight count should be 20,000 to 40,000 complex-updraft "
            "cases per fan-layout branch.",
            "",
            "## Stop Rules",
            "- Move from D2 to W2 only when branch-local W1 boundary cells are stable "
            "enough to choose W2 representatives without cross-layout promotion.",
            "- Move beyond W2 only after nominal-latency W2 candidates preserve safety "
            "margins under complex updraft variation.",
            "- Do not escalate to W3/W4/W5 claims without conservative-latency stress, "
            "mission evaluation, and real-flight evidence.",
            "",
        ]
    )


def _write_plots(
    outputs: D1bAuditOutputs,
    branch_summary: pd.DataFrame,
    agile_summary: pd.DataFrame,
    w0_failed: pd.DataFrame,
    runtime_storage: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filesystem_path(outputs.figures_dir).mkdir(parents=True, exist_ok=True)
    _plot_w1_heatmap(agile_summary, outputs.figures_dir / "d1b_w1_agile_success_heatmap_s016.png")
    _plot_agile_ladder(agile_summary, outputs.figures_dir / "d1b_agile_target_ladder_s016.png")
    _plot_w0_failed(w0_failed, outputs.figures_dir / "d1b_w0_failed_w1_valid_s016.png")
    _plot_runtime_storage(runtime_storage, outputs.figures_dir / "d1b_runtime_storage_s016.png")
    plt.close("all")


def _plot_w1_heatmap(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = frame.pivot_table(
        index="family",
        columns="target_heading_deg",
        values="success_rate",
        aggfunc="mean",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(pivot.columns)), [f"{float(value):.0f}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_xlabel("target heading change (deg)")
    ax.set_title("W1 agile-family success rate")
    fig.colorbar(image, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(filesystem_path(path), dpi=160)


def _plot_agile_ladder(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = (
        frame.groupby(["family", "target_heading_deg"], sort=True)["success_count"]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for family, group in summary.groupby("family", sort=True):
        ax.plot(group["target_heading_deg"], group["success_count"], marker="o", label=family)
    ax.set_xlabel("target heading change (deg)")
    ax.set_ylabel("W1 success count")
    ax.legend(fontsize=8)
    ax.set_title("Agile target ladder")
    fig.tight_layout()
    fig.savefig(filesystem_path(path), dpi=160)


def _plot_w0_failed(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = frame["fan_layout"].astype(str).to_list()
    values = frame["trial_count"].astype(int).to_list()
    ax.bar(labels, values)
    ax.set_ylabel("trial count")
    ax.set_title("W0 failed / W1 valid")
    fig.tight_layout()
    fig.savefig(filesystem_path(path), dpi=160)


def _plot_runtime_storage(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    row = frame.iloc[0]
    labels = ["partitions", "upload", "governor"]
    values = [
        row["trial_partition_size_mib"],
        row["upload_package_size_mib"],
        row["governor_package_size_mib"],
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(labels, values)
    ax.set_ylabel("MiB")
    ax.set_title("D1a/D1b storage summary")
    fig.tight_layout()
    fig.savefig(filesystem_path(path), dpi=160)


def _resolve_table_path(path_text: str, run_root: Path) -> Path:
    path = Path(path_text)
    candidates = (
        path,
        run_root / "tables" / path,
        run_root / path,
        REPO_ROOT / path,
    )
    for candidate in candidates:
        if filesystem_path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"missing table partition: {path_text}")


def _iter_files(root: Path) -> list[Path]:
    if not filesystem_path(root).exists():
        return []
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(filesystem_path(root)):
        for filename in filenames:
            files.append(Path(dirpath) / filename)
    return files


def _files_with_path_part(root: Path, part: str) -> list[Path]:
    return [path for path in _iter_files(root) if str(part) in path.parts]


def _tree_size(root: Path) -> tuple[int, int]:
    files = _iter_files(root)
    total = 0
    for path in files:
        total += int(path.stat().st_size)
    return len(files), total


def _mib(byte_count: object) -> float:
    return float(byte_count) / 1024.0 / 1024.0


def _success_series(frame: pd.DataFrame) -> pd.Series:
    return frame["success_flag"].astype(str).str.lower().isin({"true", "1", "1.0"})


def _label_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    return frame[column].astype(str).eq(str(label))


def _count_label(frame: pd.DataFrame, column: str, label: str) -> int:
    return int(_label_series(frame, column, label).sum())


def _counts_json(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return "{}"
    counts = frame[column].astype(str).value_counts().sort_index().to_dict()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def _rate(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _nanmedian(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median()) if values.notna().any() else math.nan


def _nanmin(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.min()) if values.notna().any() else math.nan


def _bin_series(frame: pd.DataFrame, column: str, width: float, prefix: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    labels: list[str] = []
    for value in values:
        if not math.isfinite(float(value)):
            labels.append(f"{prefix}_nan")
            continue
        lower = math.floor(float(value) / float(width)) * float(width)
        upper = lower + float(width)
        labels.append(f"{prefix}[{lower:.2f},{upper:.2f})")
    return pd.Series(labels, index=frame.index)


def _numeric_diff(a: object, b: object) -> float:
    left = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
    if pd.isna(left) and pd.isna(right):
        return math.nan
    if pd.isna(left) or pd.isna(right):
        return math.inf
    return abs(float(left) - float(right))


def _stable_hash(row: dict[str, object]) -> str:
    text = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    head = REPO_ROOT / ".git" / "HEAD"
    if not filesystem_path(head).exists():
        return None
    text = filesystem_path(head).read_text(encoding="ascii").strip()
    if text.startswith("ref:"):
        ref = REPO_ROOT / ".git" / text.split(" ", 1)[1]
        if filesystem_path(ref).exists():
            return filesystem_path(ref).read_text(encoding="ascii").strip()
    return text if text else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-run-id", type=int, default=16)
    parser.add_argument("--planning-run-id", type=int, default=15)
    parser.add_argument("--d1a-evidence-class", default="thesis_primary")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--build-plots", action="store_true")
    parser.add_argument("--build-reproducibility-plan", action="store_true")
    parser.add_argument("--execute-reproducibility-rerun", action="store_true")
    parser.add_argument("--observed-wall-time-min", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_paired_w0_w1_d1b_audit(
        archive_run_id=args.archive_run_id,
        planning_run_id=args.planning_run_id,
        d1a_evidence_class=args.d1a_evidence_class,
        result_root=args.result_root,
        audit_root=args.audit_root,
        build_plots=args.build_plots,
        build_reproducibility_plan=args.build_reproducibility_plan,
        execute_reproducibility_rerun=args.execute_reproducibility_rerun,
        observed_wall_time_min=args.observed_wall_time_min,
    )
    print(f"paired_w0_w1_d1b_audit_outputs={paths['root'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
