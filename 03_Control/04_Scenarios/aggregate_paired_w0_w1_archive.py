from __future__ import annotations

import argparse
import json
import math
import shutil
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

from dense_archive_artifacts import (  # noqa: E402
    export_diagnostic_slice as export_descriptor_slice,
    finalize_upload_package,
    reset_package_dir,
    write_governor_branch_package,
    write_json,
)
from dense_archive_chunking import TIMING_FIELDS  # noqa: E402
from dense_archive_envelope_maps import build_envelope_map  # noqa: E402
from dense_archive_runtime import (  # noqa: E402
    GOVERNOR_PACKAGE_SCHEMA_VERSION,
    GPU_ACCELERATION_ASSESSMENT,
    runtime_manifest_fields,
)
from dense_archive_schema import BRANCH_DECISION_SCOPE  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    file_sha256,
    read_table_partition,
    resolve_storage_format,
    write_table_manifest,
)
from run_paired_w0_w1_archive_chunked import RECOMMENDED_PAIRED_PROOF_COMMAND  # noqa: E402
from run_paired_w0_w1_partitioned_planning import (  # noqa: E402
    PAIRED_ENVIRONMENT_MODES,
    PAIRED_SCALE_MODES,
    SIMULATION_STAGE,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Data Containers
# 2) Chunk Verification and Loading
# 3) Summary Builders
# 4) Artifact Writers
# 5) Public Aggregation and Diagnostic Slice CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "12_paired_w0_w1_archive"
W0_W1_ENVIRONMENT_PAIRS = {
    "single_fan_branch": ("W0_single_fan_branch", "W1_single_fan", "single_fan"),
    "four_fan_branch": ("W0_four_fan_branch", "W1_four_fan", "four_fan"),
}
PAIRED_IDENTITY_KEYS = (
    "paired_sample_key",
    "layout_branch_id",
    "fan_layout",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "seed",
    "latency_case",
)
PAIRED_AUDIT_ONLY_FIELDS = (
    "candidate_id",
    "sample_id",
    "test_environment_mode",
    "replay_seed",
)
NO_CLAIM_TEXT = (
    "Paired W0/W1 proof aggregation only; W1 is evaluated independently of W0 "
    "success, single_fan_branch and four_fan_branch remain branch-local, and no "
    "W2/W3/W4/W5, mission, hardware, or sim-to-real claim is made."
)


@dataclass(frozen=True)
class PairedAggregationConfig:
    run_id: int = 14
    planning_run_id: int = 13
    result_root: Path | None = None
    storage_format: str = "auto"
    paired_scale_mode: str = "proof"
    active_environment_modes: tuple[str, ...] = PAIRED_ENVIRONMENT_MODES
    latency_case: str = "nominal"
    expected_trials_per_environment: int | None = None
    build_upload_package: bool = False
    build_governor_package: bool = False
    profile_source: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class PairedAggregationOutputs:
    root: Path
    manifest_json: Path
    table_manifest_json: Path
    report_md: Path
    branch_environment_counts_csv: Path
    failure_summary_csv: Path
    paired_comparison_summary_csv: Path
    w0_failed_w1_valid_summary_csv: Path
    w1_nominal_latency_envelope_summary_csv: Path
    chunk_manifest_summary_csv: Path
    schema_summary_csv: Path
    upload_package_dir: Path
    governor_package_dir: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "table_manifest_json": self.table_manifest_json,
            "report_md": self.report_md,
            "branch_environment_counts_csv": self.branch_environment_counts_csv,
            "failure_summary_csv": self.failure_summary_csv,
            "paired_comparison_summary_csv": self.paired_comparison_summary_csv,
            "w0_failed_w1_valid_summary_csv": self.w0_failed_w1_valid_summary_csv,
            "w1_nominal_latency_envelope_summary_csv": (
                self.w1_nominal_latency_envelope_summary_csv
            ),
            "chunk_manifest_summary_csv": self.chunk_manifest_summary_csv,
            "schema_summary_csv": self.schema_summary_csv,
            "upload_package_dir": self.upload_package_dir,
            "governor_package_dir": self.governor_package_dir,
        }


def _active_result_root(config: PairedAggregationConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _run_root(config: PairedAggregationConfig) -> Path:
    return _active_result_root(config) / f"{int(config.run_id):03d}"


def _outputs(config: PairedAggregationConfig) -> PairedAggregationOutputs:
    root = _run_root(config)
    suffix = f"s{int(config.run_id):03d}"
    return PairedAggregationOutputs(
        root=root,
        manifest_json=root / "manifests" / f"paired_w0_w1_manifest_{suffix}.json",
        table_manifest_json=root / "manifests" / f"table_manifest_{suffix}.json",
        report_md=root / "reports" / f"paired_w0_w1_report_{suffix}.md",
        branch_environment_counts_csv=root
        / "metrics_summary"
        / f"paired_branch_environment_counts_{suffix}.csv",
        failure_summary_csv=root
        / "metrics_summary"
        / f"paired_failure_summary_{suffix}.csv",
        paired_comparison_summary_csv=root
        / "metrics_summary"
        / f"paired_comparison_summary_{suffix}.csv",
        w0_failed_w1_valid_summary_csv=root
        / "metrics_summary"
        / f"w0_failed_w1_valid_summary_{suffix}.csv",
        w1_nominal_latency_envelope_summary_csv=root
        / "metrics_summary"
        / f"w1_nominal_latency_envelope_summary_{suffix}.csv",
        chunk_manifest_summary_csv=root
        / "metrics_summary"
        / f"paired_chunk_manifest_summary_{suffix}.csv",
        schema_summary_csv=root / "schema" / f"paired_schema_summary_{suffix}.csv",
        upload_package_dir=root / "upload_package",
        governor_package_dir=root / "compressed_governor_package",
    )


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validate_config(config: PairedAggregationConfig) -> None:
    if str(config.paired_scale_mode) not in PAIRED_SCALE_MODES:
        raise ValueError("paired_scale_mode must be 'proof' or 'production'.")
    if not config.active_environment_modes:
        raise ValueError("active_environment_modes must not be empty.")
    unknown = set(config.active_environment_modes).difference(PAIRED_ENVIRONMENT_MODES)
    if unknown:
        raise ValueError(f"unknown active environment modes: {sorted(unknown)}")
    if str(config.paired_scale_mode) == "production":
        required = {"W1_single_fan", "W1_four_fan"}
        if not required.issubset(set(config.active_environment_modes)):
            raise ValueError("production aggregation requires both W1 branches active.")


def _validate_aggregation_outputs(
    outputs: PairedAggregationOutputs,
    *,
    overwrite: bool,
    build_upload_package: bool,
    build_governor_package: bool,
) -> None:
    planned = [
        outputs.manifest_json,
        outputs.table_manifest_json,
        outputs.report_md,
        outputs.branch_environment_counts_csv,
        outputs.failure_summary_csv,
        outputs.paired_comparison_summary_csv,
        outputs.w0_failed_w1_valid_summary_csv,
        outputs.w1_nominal_latency_envelope_summary_csv,
        outputs.chunk_manifest_summary_csv,
        outputs.schema_summary_csv,
    ]
    if build_upload_package:
        planned.append(outputs.upload_package_dir)
    if build_governor_package:
        planned.append(outputs.governor_package_dir)
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        names = ", ".join(_path_text(path) for path in existing[:5])
        raise RuntimeError(f"paired aggregation outputs already exist: {names}")


# =============================================================================
# 2) Chunk Verification and Loading
# =============================================================================
def _load_chunk_manifests(root: Path) -> pd.DataFrame:
    paths = sorted((root / "chunk_manifests").rglob("chunk-*.json"))
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"malformed paired chunk manifest: {path}") from exc
        payload["_manifest_path"] = _path_text(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def _verify_and_load_chunks(
    manifests: pd.DataFrame,
    config: PairedAggregationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifests.empty:
        raise FileNotFoundError("missing paired W0/W1 chunk manifests.")
    _verify_manifest_schedule(manifests, config)
    verified_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    sort_columns = ["layout_branch_id", "test_environment_mode", "chunk_index"]
    for row in manifests.sort_values(sort_columns).to_dict(orient="records"):
        _verify_manifest_row(row, config)
        partition_path = _resolve_repo_path(str(row["partition_path"]))
        if not partition_path.exists():
            raise FileNotFoundError(
                f"missing paired chunk partition before aggregation: {partition_path}"
            )
        checksum = file_sha256(partition_path)
        if checksum != str(row["checksum_sha256"]):
            raise RuntimeError(
                "paired chunk checksum mismatch before aggregation: "
                f"{row['layout_branch_id']}:{row['test_environment_mode']}:"
                f"{row['chunk_index']}"
            )
        frame = read_table_partition(partition_path, storage_format=str(row["storage_format"]))
        if int(len(frame)) != int(row["row_count"]):
            raise RuntimeError(
                "paired chunk row count mismatch before aggregation: "
                f"{row['layout_branch_id']}:{row['test_environment_mode']}:"
                f"{row['chunk_index']}"
            )
        verified = dict(row)
        verified["_resolved_partition_path"] = str(partition_path)
        verified["_actual_byte_count"] = int(partition_path.stat().st_size)
        verified["_actual_checksum_sha256"] = checksum
        verified["_partition_columns"] = tuple(str(column) for column in frame.columns)
        verified_rows.append(verified)
        frames.append(frame)
    descriptors = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame(verified_rows), descriptors


def _verify_manifest_schedule(
    manifests: pd.DataFrame,
    config: PairedAggregationConfig,
) -> None:
    required = {
        "status",
        "run_id",
        "planning_run_id",
        "layout_branch_id",
        "test_environment_mode",
        "chunk_index",
        "chunk_count",
        "chunk_size",
        "storage_format",
        "latency_case",
        "dt_s",
        "horizon_s",
        "row_count",
        "partition_path",
        "checksum_sha256",
        *TIMING_FIELDS,
    }
    missing = sorted(required.difference(manifests.columns))
    if missing:
        raise RuntimeError(f"paired chunk manifests missing required fields: {missing}")
    incomplete = manifests[~manifests["status"].astype(str).eq("complete")]
    if not incomplete.empty:
        raise RuntimeError("not all paired chunk manifests are complete.")
    for environment_mode in config.active_environment_modes:
        env = manifests[manifests["test_environment_mode"].astype(str).eq(environment_mode)]
        if env.empty:
            raise RuntimeError(f"missing chunk manifests for environment: {environment_mode}")
        for branch_id, group in env.groupby("layout_branch_id", sort=True):
            expected_count = int(group["chunk_count"].iloc[0])
            if set(group["chunk_count"].astype(int)) != {expected_count}:
                raise RuntimeError(f"inconsistent chunk_count for {branch_id}:{environment_mode}")
            actual_indices = sorted(int(value) for value in group["chunk_index"].to_list())
            if actual_indices != list(range(expected_count)):
                raise RuntimeError(f"missing chunk indices for {branch_id}:{environment_mode}")


def _verify_manifest_row(row: dict[str, object], config: PairedAggregationConfig) -> None:
    if int(row["run_id"]) != int(config.run_id):
        raise RuntimeError("paired chunk manifest run_id mismatch.")
    if int(row["planning_run_id"]) != int(config.planning_run_id):
        raise RuntimeError("paired chunk manifest planning_run_id mismatch.")
    if str(row["test_environment_mode"]) not in PAIRED_ENVIRONMENT_MODES:
        raise RuntimeError("paired chunk manifest environment mode mismatch.")
    if str(row["test_environment_mode"]) not in config.active_environment_modes:
        raise RuntimeError("paired chunk manifest is outside active environment modes.")
    if str(row["storage_format"]) != resolve_storage_format(config.storage_format):
        raise RuntimeError("paired chunk manifest storage_format mismatch.")
    if str(row["latency_case"]) != str(config.latency_case):
        raise RuntimeError("paired chunk manifest latency_case mismatch.")
    if str(row["status"]) != "complete":
        raise RuntimeError("paired chunk manifest is not complete.")
    if int(row["row_count"]) <= 0:
        raise RuntimeError("paired chunk manifest row_count must be positive.")
    for field in TIMING_FIELDS:
        value = float(row[field])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"paired chunk manifest invalid timing field: {field}")


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    return REPO_ROOT / path_text


def _verify_counts(descriptors: pd.DataFrame, config: PairedAggregationConfig) -> None:
    if descriptors.empty:
        raise RuntimeError("paired aggregation has no descriptor rows.")
    env_counts = descriptors["test_environment_mode"].astype(str).value_counts().to_dict()
    for environment_mode in config.active_environment_modes:
        if int(env_counts.get(environment_mode, 0)) == 0:
            raise RuntimeError(f"missing descriptor rows for environment: {environment_mode}")
    if config.expected_trials_per_environment is None:
        return
    expected = int(config.expected_trials_per_environment)
    for environment_mode in config.active_environment_modes:
        actual = int(env_counts.get(environment_mode, 0))
        if actual != expected:
            raise RuntimeError(
                "paired aggregation rejected environment count: "
                f"{environment_mode} actual={actual}, expected={expected}"
            )


def _table_manifest_from_chunks(
    manifests: pd.DataFrame,
    config: PairedAggregationConfig,
) -> TableManifest:
    partitions: list[TablePartition] = []
    for row in manifests.to_dict(orient="records"):
        path = Path(str(row["_resolved_partition_path"]))
        partitions.append(
            TablePartition(
                table_name="trial_outcomes",
                relative_path=_path_text(path),
                storage_format=str(row["storage_format"]),
                row_count=int(row["row_count"]),
                byte_count=int(row["_actual_byte_count"]),
                columns=tuple(row["_partition_columns"]),
                checksum_sha256=str(row["_actual_checksum_sha256"]),
            )
        )
    return TableManifest(
        run_id=int(config.run_id),
        root=_path_text(_run_root(config)),
        storage_format=resolve_storage_format(config.storage_format),
        tables=tuple(partitions),
    )


# =============================================================================
# 3) Summary Builders
# =============================================================================
def _branch_environment_counts(descriptors: pd.DataFrame) -> pd.DataFrame:
    return (
        descriptors.groupby(
            ["layout_branch_id", "fan_layout", "test_environment_mode"],
            dropna=False,
        )
        .size()
        .reset_index(name="trial_count")
        .sort_values(["layout_branch_id", "test_environment_mode"])
    )


def _failure_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "layout_branch_id",
        "test_environment_mode",
        "family",
        "target_heading_deg",
        "direction_sign",
        "start_class",
        "failure_label",
    ]
    return (
        descriptors.groupby(keys, dropna=False)
        .size()
        .reset_index(name="trial_count")
        .sort_values(keys)
    )


def _paired_comparison_summary(descriptors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    for branch_id, (w0_mode, w1_mode, fan_layout) in W0_W1_ENVIRONMENT_PAIRS.items():
        branch = descriptors[descriptors["layout_branch_id"].astype(str).eq(branch_id)]
        w0 = branch[branch["test_environment_mode"].astype(str).eq(w0_mode)].copy()
        w1 = branch[branch["test_environment_mode"].astype(str).eq(w1_mode)].copy()
        merged = _paired_merge(w0, w1)
        w1_valid = _w1_nominal_accepted(w1)
        w0_failed_w1_valid = _w0_failed_w1_valid_count(merged)
        label = (
            "W0_failed_W1_valid_single_fan"
            if branch_id == "single_fan_branch"
            else "W0_failed_W1_valid_four_fan"
        )
        rows.append(
            {
                "layout_branch_id": branch_id,
                "fan_layout": fan_layout,
                "w0_environment_mode": w0_mode,
                "w1_environment_mode": w1_mode,
                "w0_trial_count": int(len(w0)),
                "w1_trial_count": int(len(w1)),
                "paired_join_count": int(len(merged)),
                "w0_success_w1_valid_count": _w0_success_w1_valid_count(merged),
                "w0_failed_w1_valid_count": int(w0_failed_w1_valid),
                "w1_nominal_valid_count": int(w1_valid.sum()),
                "paired_summary_label": label,
                "paired_key_contract": "|".join(PAIRED_IDENTITY_KEYS),
                "paired_join_key_contract": "|".join(PAIRED_IDENTITY_KEYS),
            }
        )
        cross_rows.append(
            {
                "paired_summary_label": label,
                "layout_branch_id": branch_id,
                "fan_layout": fan_layout,
                "trial_count": int(w0_failed_w1_valid),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cross_rows)


def _paired_merge(w0: pd.DataFrame, w1: pd.DataFrame) -> pd.DataFrame:
    if w0.empty or w1.empty:
        return pd.DataFrame()
    missing = [
        column
        for column in PAIRED_IDENTITY_KEYS
        if column not in w0.columns or column not in w1.columns
    ]
    if missing:
        raise RuntimeError(f"paired identity columns missing: {missing}")
    left = w0.copy()
    right = w1.copy()
    return left.merge(right, on=list(PAIRED_IDENTITY_KEYS), suffixes=("_w0", "_w1"), how="inner")


def _w0_success_w1_valid_count(merged: pd.DataFrame) -> int:
    if merged.empty:
        return 0
    w0_success = _success_from_columns(merged, suffix="_w0")
    w1_valid = _w1_nominal_from_columns(merged, suffix="_w1")
    return int((w0_success & w1_valid).sum())


def _w0_failed_w1_valid_count(merged: pd.DataFrame) -> int:
    if merged.empty:
        return 0
    w0_failed = ~_success_from_columns(merged, suffix="_w0")
    w1_valid = _w1_nominal_from_columns(merged, suffix="_w1")
    return int((w0_failed & w1_valid).sum())


def _success_from_columns(frame: pd.DataFrame, *, suffix: str = "") -> pd.Series:
    success_col = f"success_flag{suffix}"
    failure_col = f"failure_label{suffix}"
    if success_col in frame.columns:
        return _bool_series(frame[success_col])
    return frame[failure_col].astype(str).eq("success")


def _w1_nominal_accepted(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return (
        _success_from_columns(frame)
        & frame["latency_case"].astype(str).eq("nominal")
        & frame["latency_pass_label"].astype(str).eq("nominal_pass")
    )


def _w1_nominal_from_columns(frame: pd.DataFrame, *, suffix: str) -> pd.Series:
    latency_case = (
        frame[f"latency_case{suffix}"]
        if f"latency_case{suffix}" in frame.columns
        else frame["latency_case"]
    )
    latency_pass = (
        frame[f"latency_pass_label{suffix}"]
        if f"latency_pass_label{suffix}" in frame.columns
        else frame["latency_pass_label"]
    )
    return (
        _success_from_columns(frame, suffix=suffix)
        & latency_case.astype(str).eq("nominal")
        & latency_pass.astype(str).eq("nominal_pass")
    )


def _w1_nominal_latency_descriptors(descriptors: pd.DataFrame) -> pd.DataFrame:
    w1 = descriptors[
        descriptors["test_environment_mode"].astype(str).isin({"W1_single_fan", "W1_four_fan"})
    ].copy()
    if w1.empty:
        return w1
    return w1[_w1_nominal_accepted(w1)].copy()


def _schema_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_name": "trial_outcomes",
                "column_name": str(column),
                "dtype": str(descriptors[column].dtype),
            }
            for column in descriptors.columns
        ]
    )


def _seed_stability_summary(descriptors: pd.DataFrame) -> dict[str, object]:
    key_columns = [
        "paired_sample_key",
        "layout_branch_id",
        "fan_layout",
        "family",
        "target_heading_deg",
        "direction_sign",
        "start_class",
    ]
    missing = [
        column
        for column in key_columns + ["seed", "test_environment_mode"]
        if column not in descriptors.columns
    ]
    if missing:
        raise RuntimeError(f"paired seed-stability columns missing: {missing}")
    unstable = 0
    paired_groups = 0
    for _key, group in descriptors.groupby(key_columns, dropna=False):
        modes = set(group["test_environment_mode"].astype(str))
        if any(mode.startswith("W0_") for mode in modes) and any(
            mode.startswith("W1_") for mode in modes
        ):
            paired_groups += 1
            if group["seed"].astype(str).nunique() != 1:
                unstable += 1
    return {
        "paired_seed_stability_checked": True,
        "paired_identity_seed_field": "seed",
        "paired_seed_stable_across_w0_w1": unstable == 0,
        "paired_seed_stability_group_count": int(paired_groups),
        "paired_seed_instability_count": int(unstable),
    }


def _validate_seed_stability(descriptors: pd.DataFrame) -> None:
    summary = _seed_stability_summary(descriptors)
    if not bool(summary["paired_seed_stable_across_w0_w1"]):
        raise RuntimeError("paired identity seed is not stable across W0/W1 rows.")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


# =============================================================================
# 4) Artifact Writers
# =============================================================================
def _load_progress_manifest(config: PairedAggregationConfig) -> dict[str, object]:
    path = (
        _run_root(config)
        / "manifests"
        / f"paired_w0_w1_progress_s{int(config.run_id):03d}.json"
    )
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="ascii"))


def _load_profile_payload(config: PairedAggregationConfig) -> dict[str, object]:
    if config.profile_source is not None and Path(config.profile_source).exists():
        return json.loads(Path(config.profile_source).read_text(encoding="ascii"))
    default = (
        _active_result_root(config)
        / "profiles"
        / f"paired_planning_s{int(config.planning_run_id):03d}"
        / f"paired_w0_w1_profile_s{int(config.planning_run_id):03d}.json"
    )
    if default.exists():
        return json.loads(default.read_text(encoding="ascii"))
    return {}


def _worker_profile_metadata(
    progress: dict[str, object],
    profile: dict[str, object],
) -> dict[str, object]:
    return {
        "selected_worker_count": progress.get(
            "selected_worker_count",
            profile.get("selected_worker_count"),
        ),
        "os_cpu_count": progress.get("os_cpu_count", profile.get("os_cpu_count")),
        "memory_total_gb": progress.get("memory_total_gb", profile.get("memory_total_gb")),
        "memory_safety_margin_gb": progress.get(
            "memory_safety_margin_gb",
            profile.get("memory_safety_margin_gb"),
        ),
        "estimated_worker_memory_gb": progress.get(
            "estimated_worker_memory_gb",
            profile.get("estimated_worker_memory_gb"),
        ),
        "worker_fallback_reason": progress.get(
            "worker_fallback_reason",
            profile.get("worker_fallback_reason", "none"),
        ),
        "rows_per_second_by_worker_count": progress.get(
            "rows_per_second_by_worker_count",
            profile.get("rows_per_second_by_worker_count", {}),
        ),
    }


def _manifest(
    *,
    config: PairedAggregationConfig,
    outputs: PairedAggregationOutputs,
    descriptors: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    table_manifest: TableManifest,
    paired_summary: pd.DataFrame,
    w0_failed_w1_valid: pd.DataFrame,
    w1_envelope: pd.DataFrame,
    progress: dict[str, object],
    profile: dict[str, object],
) -> dict[str, object]:
    env_counts = (
        descriptors["test_environment_mode"].astype(str).value_counts().sort_index().to_dict()
    )
    branch_counts = (
        descriptors["layout_branch_id"].astype(str).value_counts().sort_index().to_dict()
    )
    metadata = _worker_profile_metadata(progress, profile)
    return {
        **runtime_manifest_fields(
            simulation_stage=SIMULATION_STAGE,
            environment_mode="multiple",
            branch_decision_scope=BRANCH_DECISION_SCOPE,
        ),
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "paired_scale_mode": str(config.paired_scale_mode),
        "active_environment_modes": list(config.active_environment_modes),
        "trial_count_total": int(len(descriptors)),
        "trial_count_by_environment": {
            key: int(value) for key, value in env_counts.items()
        },
        "trial_count_by_branch": {key: int(value) for key, value in branch_counts.items()},
        "expected_trials_per_environment": config.expected_trials_per_environment,
        "storage_format": resolve_storage_format(config.storage_format),
        "latency_case": str(config.latency_case),
        "chunk_manifest_count": int(len(chunk_summary)),
        "table_manifest_partition_count": int(len(table_manifest.tables)),
        "paired_summary_row_count": int(len(paired_summary)),
        "w0_failed_w1_valid_summary": w0_failed_w1_valid.to_dict(orient="records"),
        "w1_nominal_latency_envelope_cell_count": int(len(w1_envelope)),
        "paired_w0_w1_aggregated": True,
        "w1_scheduled_independent_of_w0_success": True,
        "branch_local_decisions_only": True,
        "single_fan_and_four_fan_never_merged": True,
        "full_w1_production_claim": False,
        "w2_w3_w4_w5_performed": False,
        "hardware_or_mission_claim": False,
        "sim_to_real_transfer_claim": False,
        "governor_artifacts_scan_raw_tables": False,
        "governor_package_contains_w0_candidates": False,
        "governor_package_branch_local_only": True,
        "w1_acceptance_latency_rule": "latency_case=nominal and latency_pass_label=nominal_pass",
        "paired_comparison_keys": list(PAIRED_IDENTITY_KEYS),
        "paired_identity_keys": list(PAIRED_IDENTITY_KEYS),
        "paired_join_keys": list(PAIRED_IDENTITY_KEYS),
        "paired_audit_only_fields": list(PAIRED_AUDIT_ONLY_FIELDS),
        "paired_summary_labels": [
            "W0_failed_W1_valid_single_fan",
            "W0_failed_W1_valid_four_fan",
        ],
        "gpu_acceleration_assessment": GPU_ACCELERATION_ASSESSMENT,
        "recommended_paired_proof_command": RECOMMENDED_PAIRED_PROOF_COMMAND,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        "selected_worker_count": metadata["selected_worker_count"],
        "os_cpu_count": metadata["os_cpu_count"],
        "memory_total_gb": metadata["memory_total_gb"],
        "memory_safety_margin_gb": metadata["memory_safety_margin_gb"],
        "estimated_worker_memory_gb": metadata["estimated_worker_memory_gb"],
        "worker_fallback_reason": metadata["worker_fallback_reason"],
        "profile_rows_per_second_by_worker_count": metadata[
            "rows_per_second_by_worker_count"
        ],
        "profile_estimated_runtime_s_by_workers": profile.get(
            "estimated_runtime_s_by_workers",
            profile.get("estimated_runtime_s_by_worker_count", {}),
        ),
        "governor_package_schema_version": GOVERNOR_PACKAGE_SCHEMA_VERSION,
        "output_files": {
            "manifest": _path_text(outputs.manifest_json),
            "table_manifest": _path_text(outputs.table_manifest_json),
            "report": _path_text(outputs.report_md),
            "branch_environment_counts": _path_text(outputs.branch_environment_counts_csv),
            "failure_summary": _path_text(outputs.failure_summary_csv),
            "paired_comparison_summary": _path_text(outputs.paired_comparison_summary_csv),
            "w0_failed_w1_valid_summary": _path_text(
                outputs.w0_failed_w1_valid_summary_csv
            ),
            "w1_nominal_latency_envelope_summary": _path_text(
                outputs.w1_nominal_latency_envelope_summary_csv
            ),
            "chunk_manifest_summary": _path_text(outputs.chunk_manifest_summary_csv),
            "schema_summary": _path_text(outputs.schema_summary_csv),
            "upload_package": _path_text(outputs.upload_package_dir),
            "compressed_governor_package": _path_text(outputs.governor_package_dir),
        },
        **_seed_stability_summary(descriptors),
    }


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Paired W0/W1 Archive Aggregation Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Planning run id: `{manifest['planning_run_id']}`",
        f"- Trial count total: `{manifest['trial_count_total']}`",
        f"- Trial count by environment: `{manifest['trial_count_by_environment']}`",
        f"- Selected worker count: `{manifest.get('selected_worker_count', '')}`",
        f"- Worker fallback reason: `{manifest.get('worker_fallback_reason', '')}`",
        f"- W1 acceptance rule: `{manifest['w1_acceptance_latency_rule']}`",
        f"- GPU assessment: {manifest['gpu_acceleration_assessment']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def _write_upload_package(
    *,
    outputs: PairedAggregationOutputs,
    manifest: dict[str, object],
    table_manifest: TableManifest,
    branch_counts: pd.DataFrame,
    failure_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    w0_failed_w1_valid: pd.DataFrame,
    w1_envelope: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    schema_summary: pd.DataFrame,
    profile: dict[str, object],
    descriptors: pd.DataFrame,
) -> None:
    package = reset_package_dir(outputs.upload_package_dir)
    write_json(package / "final_manifest.json", manifest)
    shutil.copy2(outputs.report_md, package / "final_report.md")
    write_json(
        package / "table_manifest_summary.json",
        {
            "run_id": table_manifest.run_id,
            "root": table_manifest.root,
            "storage_format": table_manifest.storage_format,
            "partition_count": len(table_manifest.tables),
            "row_count": sum(part.row_count for part in table_manifest.tables),
            "byte_count": sum(part.byte_count for part in table_manifest.tables),
        },
    )
    chunk_summary.to_csv(package / "chunk_manifest_summary.csv", index=False)
    branch_counts.to_csv(package / "branch_environment_counts.csv", index=False)
    failure_summary.to_csv(package / "failure_summary.csv", index=False)
    paired_summary.to_csv(package / "paired_comparison_summary.csv", index=False)
    w0_failed_w1_valid.to_csv(package / "w0_failed_w1_valid_summary.csv", index=False)
    w1_envelope.to_csv(package / "w1_nominal_latency_envelope_summary.csv", index=False)
    write_json(package / "profiling_summary.json", profile)
    pd.DataFrame(
        [
            {
                "field": key,
                "value": json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value,
            }
            for key, value in profile.items()
        ]
    ).to_csv(package / "profiling_summary.csv", index=False)
    schema_summary.to_csv(package / "schema_summary.csv", index=False)
    (package / "command_history.md").write_text(
        "\n".join(
            [
                "# Paired W0/W1 Command History",
                "",
                "Recommended paired proof execution:",
                "",
                f"```powershell\n{RECOMMENDED_PAIRED_PROOF_COMMAND}\n```",
                "",
            ]
        ),
        encoding="ascii",
    )
    for environment_mode in manifest.get("active_environment_modes", PAIRED_ENVIRONMENT_MODES):
        preview = (
            descriptors[
                descriptors["test_environment_mode"].astype(str).eq(environment_mode)
            ]
            .sort_values("trial_descriptor_id")
            .head(1000)
        )
        preview.to_csv(package / f"{environment_mode}_preview.csv", index=False)
    finalize_upload_package(package)


def _write_governor_package(
    *,
    outputs: PairedAggregationOutputs,
    descriptors: pd.DataFrame,
    worker_profile_metadata: dict[str, object],
    storage_format: str,
) -> dict[str, object]:
    root = reset_package_dir(outputs.governor_package_dir)
    accepted = _w1_nominal_latency_descriptors(descriptors)
    metadata_by_fan: dict[str, object] = {}
    for branch_id, (_w0_mode, w1_mode, fan_layout) in W0_W1_ENVIRONMENT_PAIRS.items():
        branch_rows = accepted[
            accepted["test_environment_mode"].astype(str).eq(w1_mode)
            & accepted["layout_branch_id"].astype(str).eq(branch_id)
        ].copy()
        envelope_cells = build_envelope_map(branch_rows)
        candidate_columns = [
            column
            for column in (
                "candidate_id",
                "paired_sample_key",
                "layout_branch_id",
                "fan_layout",
                "test_environment_mode",
                "family",
                "target_heading_deg",
                "direction_sign",
                "start_class",
                "latency_case",
                "latency_pass_label",
                "fan_config_id",
                "updraft_model_id",
            )
            if column in branch_rows.columns
        ]
        candidates = (
            branch_rows[candidate_columns]
            .drop_duplicates()
            .sort_values(candidate_columns[:1] or ["candidate_id"])
            if candidate_columns
            else pd.DataFrame()
        )
        viability = pd.DataFrame(
            [
                {
                    "layout_branch_id": branch_id,
                    "fan_layout": fan_layout,
                    "test_environment_mode": w1_mode,
                    "minimum_success_fraction": 1.0,
                    "required_latency_case": "nominal",
                    "required_latency_pass_label": "nominal_pass",
                    "branch_decision_scope": BRANCH_DECISION_SCOPE,
                }
            ]
        )
        latency = pd.DataFrame(
            [
                {
                    "layout_branch_id": branch_id,
                    "fan_layout": fan_layout,
                    "test_environment_mode": w1_mode,
                    "latency_case": "nominal",
                    "accepted_latency_pass_label": "nominal_pass",
                }
            ]
        )
        model_columns = [
            column
            for column in (
                "layout_branch_id",
                "fan_layout",
                "test_environment_mode",
                "fan_config_id",
                "updraft_model_id",
                "timing_model_version",
            )
            if column in branch_rows.columns
        ]
        model_ids = (
            branch_rows[model_columns].drop_duplicates()
            if model_columns
            else pd.DataFrame(
                [{"layout_branch_id": branch_id, "fan_layout": fan_layout, "test_environment_mode": w1_mode}]
            )
        )
        metadata_by_fan[fan_layout] = write_governor_branch_package(
            root=root,
            fan_layout=fan_layout,
            environment_mode=w1_mode,
            envelope_cells=envelope_cells,
            candidate_representatives=candidates,
            viability_thresholds=viability,
            latency_metadata=latency,
            model_ids=model_ids,
            worker_profile_metadata=worker_profile_metadata,
            storage_format=storage_format,
            compression_level=1,
        )
    write_json(
        root / "governor_package_manifest.json",
        {
            "governor_package_schema_version": GOVERNOR_PACKAGE_SCHEMA_VERSION,
            "simulation_stage": SIMULATION_STAGE,
            "raw_tables_included": False,
            "governor_artifacts_scan_raw_tables": False,
            "governor_package_contains_w0_candidates": False,
            "governor_package_branch_local_only": True,
            "branch_local_decisions_only": True,
            "metadata_by_fan_layout": metadata_by_fan,
            "worker_profile_metadata": worker_profile_metadata,
        },
    )
    finalize_upload_package(root)
    return metadata_by_fan


# =============================================================================
# 5) Public Aggregation and Diagnostic Slice CLI
# =============================================================================
def aggregate_paired_w0_w1_archive(
    *,
    run_id: int = 14,
    planning_run_id: int = 13,
    result_root: Path | None = None,
    storage_format: str = "auto",
    paired_scale_mode: str = "proof",
    active_environment_modes: tuple[str, ...] = PAIRED_ENVIRONMENT_MODES,
    latency_case: str = "nominal",
    expected_trials_per_environment: int | None = None,
    build_upload_package: bool = False,
    build_governor_package: bool = False,
    profile_source: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    config = PairedAggregationConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        result_root=result_root,
        storage_format=str(storage_format),
        paired_scale_mode=str(paired_scale_mode),
        active_environment_modes=tuple(active_environment_modes),
        latency_case=str(latency_case),
        expected_trials_per_environment=expected_trials_per_environment,
        build_upload_package=bool(build_upload_package),
        build_governor_package=bool(build_governor_package),
        profile_source=profile_source,
        overwrite=bool(overwrite),
    )
    resolve_storage_format(config.storage_format)
    _validate_config(config)
    outputs = _outputs(config)
    _validate_aggregation_outputs(
        outputs,
        overwrite=bool(config.overwrite),
        build_upload_package=bool(config.build_upload_package),
        build_governor_package=bool(config.build_governor_package),
    )
    for path in (
        outputs.manifest_json.parent,
        outputs.report_md.parent,
        outputs.branch_environment_counts_csv.parent,
        outputs.schema_summary_csv.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    raw_chunk_summary = _load_chunk_manifests(outputs.root)
    chunk_summary, descriptors = _verify_and_load_chunks(raw_chunk_summary, config)
    _verify_counts(descriptors, config)
    _validate_seed_stability(descriptors)

    branch_counts = _branch_environment_counts(descriptors)
    failure_summary = _failure_summary(descriptors)
    paired_summary, w0_failed_w1_valid = _paired_comparison_summary(descriptors)
    w1_envelope = build_envelope_map(_w1_nominal_latency_descriptors(descriptors))
    schema_summary = _schema_summary(descriptors)
    table_manifest = _table_manifest_from_chunks(chunk_summary, config)
    progress = _load_progress_manifest(config)
    profile = _load_profile_payload(config)
    manifest = _manifest(
        config=config,
        outputs=outputs,
        descriptors=descriptors,
        chunk_summary=chunk_summary,
        table_manifest=table_manifest,
        paired_summary=paired_summary,
        w0_failed_w1_valid=w0_failed_w1_valid,
        w1_envelope=w1_envelope,
        progress=progress,
        profile=profile,
    )

    write_table_manifest(outputs.table_manifest_json, table_manifest)
    branch_counts.to_csv(outputs.branch_environment_counts_csv, index=False)
    failure_summary.to_csv(outputs.failure_summary_csv, index=False)
    paired_summary.to_csv(outputs.paired_comparison_summary_csv, index=False)
    w0_failed_w1_valid.to_csv(outputs.w0_failed_w1_valid_summary_csv, index=False)
    w1_envelope.to_csv(outputs.w1_nominal_latency_envelope_summary_csv, index=False)
    chunk_summary.to_csv(outputs.chunk_manifest_summary_csv, index=False)
    schema_summary.to_csv(outputs.schema_summary_csv, index=False)
    write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    if config.build_upload_package:
        _write_upload_package(
            outputs=outputs,
            manifest=manifest,
            table_manifest=table_manifest,
            branch_counts=branch_counts,
            failure_summary=failure_summary,
            paired_summary=paired_summary,
            w0_failed_w1_valid=w0_failed_w1_valid,
            w1_envelope=w1_envelope,
            chunk_summary=chunk_summary,
            schema_summary=schema_summary,
            profile=profile,
            descriptors=descriptors,
        )
    if config.build_governor_package:
        _write_governor_package(
            outputs=outputs,
            descriptors=descriptors,
            worker_profile_metadata=_worker_profile_metadata(progress, profile),
            storage_format=resolve_storage_format(config.storage_format),
        )
    return outputs.as_dict()


def export_diagnostic_slice(
    *,
    run_id: int = 14,
    result_root: Path | None = None,
    layout_branch_id: str | None = None,
    test_environment_mode: str | None = None,
    failure_label: str | None = None,
    max_rows: int = 5000,
    output_path: Path | None = None,
) -> Path:
    config = PairedAggregationConfig(run_id=int(run_id), result_root=result_root)
    root = _run_root(config)
    manifests = _load_chunk_manifests(root)
    frames: list[pd.DataFrame] = []
    for row in manifests.to_dict(orient="records"):
        if str(row.get("status", "")) != "complete":
            continue
        path = _resolve_repo_path(str(row["partition_path"]))
        frames.append(read_table_partition(path, storage_format=str(row["storage_format"])))
    descriptors = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    destination = output_path or (
        root / "upload_package" / f"paired_diagnostic_slice_s{int(run_id):03d}.csv"
    )
    return export_descriptor_slice(
        descriptors,
        output_path=destination,
        layout_branch_id=layout_branch_id,
        test_environment_mode=test_environment_mode,
        failure_label=failure_label,
        max_rows=int(max_rows),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=14)
    parser.add_argument("--planning-run-id", type=int, default=13)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--paired-scale-mode", choices=PAIRED_SCALE_MODES, default="proof")
    parser.add_argument("--active-environment-modes", nargs="*", default=list(PAIRED_ENVIRONMENT_MODES))
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--expected-trials-per-environment", type=int, default=None)
    parser.add_argument("--build-upload-package", action="store_true")
    parser.add_argument("--build-governor-package", action="store_true")
    parser.add_argument("--profile-source", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-slice", action="store_true")
    parser.add_argument("--slice-layout-branch-id", default=None)
    parser.add_argument("--slice-test-environment-mode", default=None)
    parser.add_argument("--slice-failure-label", default=None)
    parser.add_argument("--slice-max-rows", type=int, default=5000)
    parser.add_argument("--slice-output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.export_slice:
        path = export_diagnostic_slice(
            run_id=args.run_id,
            result_root=args.result_root,
            layout_branch_id=args.slice_layout_branch_id,
            test_environment_mode=args.slice_test_environment_mode,
            failure_label=args.slice_failure_label,
            max_rows=args.slice_max_rows,
            output_path=args.slice_output_path,
        )
        print(f"paired_w0_w1_diagnostic_slice={_path_text(path)}")
        return 0
    paths = aggregate_paired_w0_w1_archive(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        result_root=args.result_root,
        storage_format=args.storage_format,
        paired_scale_mode=args.paired_scale_mode,
        active_environment_modes=tuple(args.active_environment_modes),
        latency_case=args.latency_case,
        expected_trials_per_environment=args.expected_trials_per_environment,
        build_upload_package=args.build_upload_package,
        build_governor_package=args.build_governor_package,
        profile_source=args.profile_source,
        overwrite=args.overwrite,
    )
    print(f"paired_w0_w1_aggregation_outputs={_path_text(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
