from __future__ import annotations

import argparse
import json
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

from dense_archive_cluster_diagnostics import (  # noqa: E402
    DenseClusterDiagnosticConfig,
    build_cluster_diagnostics,
)
from dense_archive_clustering import cluster_key, select_cluster_representatives  # noqa: E402
from dense_archive_envelope_maps import build_envelope_map  # noqa: E402
from dense_archive_schema import BRANCH_DECISION_SCOPE  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    file_sha256,
    read_table_partition,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
)
from run_w0_dense_archive_chunked import (  # noqa: E402
    GPU_ACCELERATION_ASSESSMENT,
    PRODUCTION_COMMAND,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Data Containers
# 2) Chunk and Partition Loading
# 3) Summary and Upload Package Writers
# 4) Public Aggregation and Diagnostic Export CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "11_w0_dense_archive"
W0_BRANCH_IDS = ("single_fan_branch", "four_fan_branch")
UPLOAD_PACKAGE_MAX_BYTES = 25 * 1024 * 1024
NO_CLAIM_TEXT = (
    "W0 dense deterministic archive aggregation only; no W1/W2/W3/W4/W5 "
    "evidence, mission evaluation, hardware validation, or sim-to-real transfer "
    "is claimed."
)


@dataclass(frozen=True)
class W0AggregationConfig:
    run_id: int = 13
    planning_run_id: int = 12
    result_root: Path | None = None
    expected_trials_total: int = 500000
    expected_trials_per_branch: int = 250000
    storage_format: str = "auto"
    archive_scale_mode: str = "strict"
    build_upload_package: bool = False
    profile_source: Path | None = None


@dataclass(frozen=True)
class W0AggregationOutputs:
    root: Path
    manifest_json: Path
    table_manifest_json: Path
    report_md: Path
    branch_counts_csv: Path
    failure_summary_csv: Path
    envelope_summary_csv: Path
    cluster_representatives_csv: Path
    cluster_diagnostics_csv: Path
    chunk_manifest_summary_csv: Path
    schema_summary_csv: Path
    upload_package_dir: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "table_manifest_json": self.table_manifest_json,
            "report_md": self.report_md,
            "branch_counts_csv": self.branch_counts_csv,
            "failure_summary_csv": self.failure_summary_csv,
            "envelope_summary_csv": self.envelope_summary_csv,
            "cluster_representatives_csv": self.cluster_representatives_csv,
            "cluster_diagnostics_csv": self.cluster_diagnostics_csv,
            "chunk_manifest_summary_csv": self.chunk_manifest_summary_csv,
            "schema_summary_csv": self.schema_summary_csv,
            "upload_package_dir": self.upload_package_dir,
        }


def _active_result_root(config: W0AggregationConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _run_root(config: W0AggregationConfig) -> Path:
    return _active_result_root(config) / f"{int(config.run_id):03d}"


def _outputs(config: W0AggregationConfig) -> W0AggregationOutputs:
    root = _run_root(config)
    suffix = f"s{int(config.run_id):03d}"
    return W0AggregationOutputs(
        root=root,
        manifest_json=root / "manifests" / f"w0_dense_archive_manifest_{suffix}.json",
        table_manifest_json=root / "manifests" / f"table_manifest_{suffix}.json",
        report_md=root / "reports" / f"w0_dense_archive_report_{suffix}.md",
        branch_counts_csv=root / "metrics_summary" / f"w0_branch_counts_{suffix}.csv",
        failure_summary_csv=root / "metrics_summary" / f"w0_failure_summary_{suffix}.csv",
        envelope_summary_csv=root / "metrics_summary" / f"w0_envelope_summary_{suffix}.csv",
        cluster_representatives_csv=root
        / "metrics_summary"
        / f"w0_cluster_representatives_{suffix}.csv",
        cluster_diagnostics_csv=root
        / "metrics_summary"
        / f"w0_cluster_diagnostics_summary_{suffix}.csv",
        chunk_manifest_summary_csv=root
        / "metrics_summary"
        / f"w0_chunk_manifest_summary_{suffix}.csv",
        schema_summary_csv=root / "schema" / f"w0_schema_summary_{suffix}.csv",
        upload_package_dir=root / "upload_package",
    )


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Chunk and Partition Loading
# =============================================================================
def _load_chunk_manifests(root: Path) -> pd.DataFrame:
    paths = sorted((root / "chunk_manifests").rglob("chunk-*.json"))
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="ascii"))
        payload["_manifest_path"] = _path_text(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def _verify_chunk_manifests(
    manifests: pd.DataFrame,
    config: W0AggregationConfig,
) -> None:
    if manifests.empty:
        raise FileNotFoundError("missing W0 chunk manifests.")
    incomplete = manifests[~manifests["status"].astype(str).eq("complete")]
    if not incomplete.empty:
        raise RuntimeError("not all chunk manifests are complete.")
    for branch_id in W0_BRANCH_IDS:
        branch = manifests[manifests["layout_branch_id"].astype(str).eq(branch_id)]
        if branch.empty:
            raise RuntimeError(f"missing chunk manifests for branch: {branch_id}")
        expected_count = int(branch["chunk_count"].iloc[0])
        actual_indices = sorted(int(value) for value in branch["chunk_index"].to_list())
        if actual_indices != list(range(expected_count)):
            raise RuntimeError(f"missing chunk indices for branch: {branch_id}")


def _load_trial_outcomes(manifests: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in manifests.to_dict(orient="records"):
        path = _resolve_repo_path(str(row["partition_path"]))
        frames.append(read_table_partition(path, storage_format=str(row["storage_format"])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    return REPO_ROOT / path_text


def _verify_counts(descriptors: pd.DataFrame, config: W0AggregationConfig) -> None:
    counts = descriptors["layout_branch_id"].astype(str).value_counts().to_dict()
    total = int(len(descriptors))
    if str(config.archive_scale_mode) == "strict":
        if total != int(config.expected_trials_total):
            raise RuntimeError(
                "strict W0 aggregation rejected total count: "
                f"actual={total}, expected={int(config.expected_trials_total)}"
            )
        for branch_id in W0_BRANCH_IDS:
            if int(counts.get(branch_id, 0)) != int(config.expected_trials_per_branch):
                raise RuntimeError(
                    "strict W0 aggregation rejected branch count: "
                    f"{branch_id} actual={int(counts.get(branch_id, 0))}, "
                    f"expected={int(config.expected_trials_per_branch)}"
                )


def _table_manifest_from_chunks(
    manifests: pd.DataFrame,
    config: W0AggregationConfig,
) -> TableManifest:
    partitions: list[TablePartition] = []
    for row in manifests.to_dict(orient="records"):
        path = _resolve_repo_path(str(row["partition_path"]))
        frame_columns = tuple(read_table_partition(path).columns)
        partitions.append(
            TablePartition(
                table_name="trial_outcomes",
                relative_path=_path_text(path),
                storage_format=str(row["storage_format"]),
                row_count=int(row["row_count"]),
                byte_count=int(path.stat().st_size),
                columns=frame_columns,
                checksum_sha256=file_sha256(path),
            )
        )
    return TableManifest(
        run_id=int(config.run_id),
        root=_path_text(_run_root(config)),
        storage_format=resolve_storage_format(config.storage_format),
        tables=tuple(partitions),
    )


# =============================================================================
# 3) Summary and Upload Package Writers
# =============================================================================
def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# W0 Dense Archive Aggregation Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Planning run id: `{manifest['planning_run_id']}`",
        f"- Trial count total: `{manifest['trial_count_total']}`",
        f"- Trial count by branch: `{manifest['trial_count_by_branch']}`",
        f"- Archive scale mode: `{manifest['archive_scale_mode']}`",
        f"- Storage format: `{manifest['storage_format']}`",
        f"- Selected worker count: `{manifest.get('selected_worker_count', '')}`",
        f"- Worker fallback reason: `{manifest.get('worker_fallback_reason', '')}`",
        f"- GPU assessment: {manifest['gpu_acceleration_assessment']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def _branch_counts(descriptors: pd.DataFrame) -> pd.DataFrame:
    return (
        descriptors.groupby(["layout_branch_id", "fan_layout", "test_environment_mode"], dropna=False)
        .size()
        .reset_index(name="trial_count")
        .sort_values(["layout_branch_id", "test_environment_mode"])
    )


def _failure_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "layout_branch_id",
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


def _schema_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"table_name": "trial_outcomes", "column_name": str(column), "dtype": str(descriptors[column].dtype)}
            for column in descriptors.columns
        ]
    )


def _load_progress_manifest(config: W0AggregationConfig) -> dict[str, object]:
    path = _run_root(config) / "manifests" / f"w0_dense_archive_progress_s{int(config.run_id):03d}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="ascii"))


def _load_profile_payload(config: W0AggregationConfig) -> dict[str, object]:
    if config.profile_source is not None and Path(config.profile_source).exists():
        return json.loads(Path(config.profile_source).read_text(encoding="ascii"))
    default = (
        DEFAULT_RESULT_ROOT
        / "profiles"
        / f"planning_s{int(config.planning_run_id):03d}"
        / f"w0_profile_s{int(config.planning_run_id):03d}.json"
    )
    if default.exists():
        return json.loads(default.read_text(encoding="ascii"))
    return {}


def _manifest(
    *,
    config: W0AggregationConfig,
    outputs: W0AggregationOutputs,
    descriptors: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    envelope: pd.DataFrame,
    clusters: pd.DataFrame,
    diagnostics: pd.DataFrame,
    progress: dict[str, object],
    profile: dict[str, object],
) -> dict[str, object]:
    branch_counts = descriptors["layout_branch_id"].astype(str).value_counts().sort_index().to_dict()
    return {
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "trial_count_total": int(len(descriptors)),
        "trial_count_by_branch": {key: int(value) for key, value in branch_counts.items()},
        "expected_trials_total": int(config.expected_trials_total),
        "expected_trials_per_branch": int(config.expected_trials_per_branch),
        "archive_scale_mode": str(config.archive_scale_mode),
        "storage_format": resolve_storage_format(config.storage_format),
        "chunk_manifest_count": int(len(chunk_summary)),
        "envelope_cell_count": int(len(envelope)),
        "cluster_representative_count": int(len(clusters)),
        "cluster_diagnostic_row_count": int(len(diagnostics)),
        "w0_dense_archive_aggregated": True,
        "w0_full_archive_performed": int(len(descriptors)) == int(config.expected_trials_total),
        "w1_w2_w3_w4_w5_performed": False,
        "hardware_or_mission_claim": False,
        "sim_to_real_transfer_claim": False,
        "branch_local_decisions_only": True,
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        "selected_worker_count": progress.get("selected_worker_count"),
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
        "profile_rows_per_second_by_worker_count": profile.get("rows_per_second_by_worker_count", {}),
        "profile_estimated_500k_runtime_s_by_workers": profile.get(
            "estimated_500k_runtime_s_by_workers",
            {},
        ),
        "gpu_acceleration_assessment": GPU_ACCELERATION_ASSESSMENT,
        "recommended_production_command": PRODUCTION_COMMAND,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        "output_files": {
            "manifest": _path_text(outputs.manifest_json),
            "table_manifest": _path_text(outputs.table_manifest_json),
            "report": _path_text(outputs.report_md),
            "branch_counts": _path_text(outputs.branch_counts_csv),
            "failure_summary": _path_text(outputs.failure_summary_csv),
            "envelope_summary": _path_text(outputs.envelope_summary_csv),
            "cluster_representatives": _path_text(outputs.cluster_representatives_csv),
            "cluster_diagnostics": _path_text(outputs.cluster_diagnostics_csv),
            "chunk_manifest_summary": _path_text(outputs.chunk_manifest_summary_csv),
            "schema_summary": _path_text(outputs.schema_summary_csv),
            "upload_package": _path_text(outputs.upload_package_dir),
        },
    }


def _write_upload_package(
    *,
    outputs: W0AggregationOutputs,
    manifest: dict[str, object],
    table_manifest: TableManifest,
    branch_counts: pd.DataFrame,
    failure_summary: pd.DataFrame,
    envelope: pd.DataFrame,
    diagnostics: pd.DataFrame,
    clusters: pd.DataFrame,
    descriptors: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    schema_summary: pd.DataFrame,
    profile: dict[str, object],
) -> None:
    package = outputs.upload_package_dir
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    _write_json(package / "final_manifest.json", manifest)
    shutil.copy2(outputs.report_md, package / "final_report.md")
    _write_json(
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
    branch_counts.to_csv(package / "branch_counts.csv", index=False)
    failure_summary.to_csv(package / "failure_summary.csv", index=False)
    envelope.to_csv(package / "envelope_summary.csv", index=False)
    diagnostics.to_csv(package / "cluster_diagnostics_summary.csv", index=False)
    if not clusters.empty and "candidate_role" in clusters.columns:
        boundary = clusters[
            clusters["candidate_role"].astype(str).eq("boundary_representative")
        ].head(1000)
    else:
        boundary = pd.DataFrame()
    boundary.to_csv(package / "top_mixed_boundary_examples.csv", index=False)
    _top_examples(descriptors, success=False).to_csv(package / "top_failure_examples.csv", index=False)
    _top_examples(descriptors, success=True).to_csv(package / "top_success_examples.csv", index=False)
    _write_json(package / "profiling_summary.json", profile)
    pd.DataFrame(
        [
            {"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
            for key, value in profile.items()
        ]
    ).to_csv(package / "profiling_summary.csv", index=False)
    schema_summary.to_csv(package / "schema_summary.csv", index=False)
    (package / "command_history.md").write_text(
        "\n".join(
            [
                "# W0 Command History",
                "",
                "Recommended production execution:",
                "",
                f"```powershell\n{PRODUCTION_COMMAND}\n```",
                "",
                "Recommended strict aggregation:",
                "",
                "```powershell",
                "python 03_Control/04_Scenarios/aggregate_w0_dense_archive.py --run-id 13 --planning-run-id 12 --expected-trials-total 500000 --expected-trials-per-branch 250000 --archive-scale-mode strict --storage-format auto --build-upload-package",
                "```",
                "",
            ]
        ),
        encoding="ascii",
    )
    for branch_id in W0_BRANCH_IDS:
        descriptors[descriptors["layout_branch_id"].astype(str).eq(branch_id)].sort_values(
            "trial_descriptor_id"
        ).head(1000).to_csv(package / f"{branch_id}_preview.csv", index=False)
    _check_upload_package_sizes(package)


def _top_examples(descriptors: pd.DataFrame, *, success: bool) -> pd.DataFrame:
    mask = descriptors["success_flag"].astype(bool).eq(bool(success))
    columns = [
        "trial_descriptor_id",
        "layout_branch_id",
        "family",
        "target_heading_deg",
        "direction_sign",
        "start_class",
        "success_flag",
        "failure_label",
        "heading_error_deg",
        "energy_residual_m",
        "min_true_margin_m",
        "saturation_fraction",
    ]
    available = [column for column in columns if column in descriptors.columns]
    return descriptors[mask][available].sort_values("trial_descriptor_id").head(1000)


def _check_upload_package_sizes(package: Path) -> None:
    oversized = [
        path
        for path in package.rglob("*")
        if path.is_file() and path.stat().st_size > UPLOAD_PACKAGE_MAX_BYTES
    ]
    if oversized:
        names = ", ".join(_path_text(path) for path in oversized)
        raise RuntimeError(f"upload package file exceeds 25 MB: {names}")


# =============================================================================
# 4) Public Aggregation and Diagnostic Export CLI
# =============================================================================
def aggregate_w0_dense_archive(
    *,
    run_id: int = 13,
    planning_run_id: int = 12,
    result_root: Path | None = None,
    expected_trials_total: int = 500000,
    expected_trials_per_branch: int = 250000,
    storage_format: str = "auto",
    archive_scale_mode: str = "strict",
    build_upload_package: bool = False,
    profile_source: Path | None = None,
) -> dict[str, Path]:
    """Aggregate completed W0 chunks into compact summaries and manifests."""

    config = W0AggregationConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        result_root=result_root,
        expected_trials_total=int(expected_trials_total),
        expected_trials_per_branch=int(expected_trials_per_branch),
        storage_format=str(storage_format),
        archive_scale_mode=str(archive_scale_mode),
        build_upload_package=bool(build_upload_package),
        profile_source=profile_source,
    )
    resolve_storage_format(config.storage_format)
    outputs = _outputs(config)
    for path in (
        outputs.manifest_json.parent,
        outputs.report_md.parent,
        outputs.branch_counts_csv.parent,
        outputs.schema_summary_csv.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    chunk_summary = _load_chunk_manifests(outputs.root)
    _verify_chunk_manifests(chunk_summary, config)
    descriptors = _load_trial_outcomes(chunk_summary)
    _verify_counts(descriptors, config)

    envelope = build_envelope_map(descriptors)
    clusters = select_cluster_representatives(descriptors, envelope)
    diagnostics = build_cluster_diagnostics(
        descriptors,
        envelope,
        clusters,
        DenseClusterDiagnosticConfig(min_trials_total=int(config.expected_trials_total)),
    )
    branch_counts = _branch_counts(descriptors)
    failure_summary = _failure_summary(descriptors)
    schema_summary = _schema_summary(descriptors)
    table_manifest = _table_manifest_from_chunks(chunk_summary, config)
    progress = _load_progress_manifest(config)
    profile = _load_profile_payload(config)
    manifest = _manifest(
        config=config,
        outputs=outputs,
        descriptors=descriptors,
        chunk_summary=chunk_summary,
        envelope=envelope,
        clusters=clusters,
        diagnostics=diagnostics,
        progress=progress,
        profile=profile,
    )

    write_table_manifest(outputs.table_manifest_json, table_manifest)
    branch_counts.to_csv(outputs.branch_counts_csv, index=False)
    failure_summary.to_csv(outputs.failure_summary_csv, index=False)
    envelope.to_csv(outputs.envelope_summary_csv, index=False)
    clusters.to_csv(outputs.cluster_representatives_csv, index=False)
    diagnostics.to_csv(outputs.cluster_diagnostics_csv, index=False)
    chunk_summary.to_csv(outputs.chunk_manifest_summary_csv, index=False)
    schema_summary.to_csv(outputs.schema_summary_csv, index=False)
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    if config.build_upload_package:
        _write_upload_package(
            outputs=outputs,
            manifest=manifest,
            table_manifest=table_manifest,
            branch_counts=branch_counts,
            failure_summary=failure_summary,
            envelope=envelope,
            diagnostics=diagnostics,
            clusters=clusters,
            descriptors=descriptors,
            chunk_summary=chunk_summary,
            schema_summary=schema_summary,
            profile=profile,
        )
    return outputs.as_dict()


def export_diagnostic_slice(
    *,
    run_id: int = 13,
    result_root: Path | None = None,
    layout_branch_id: str | None = None,
    failure_label: str | None = None,
    cluster_key_filter: str | None = None,
    max_rows: int = 5000,
    output_path: Path | None = None,
) -> Path:
    """Export a small targeted diagnostic slice from ignored full partitions."""

    config = W0AggregationConfig(run_id=int(run_id), result_root=result_root)
    root = _run_root(config)
    manifests = _load_chunk_manifests(root)
    descriptors = _load_trial_outcomes(manifests)
    frame = descriptors.copy()
    if layout_branch_id:
        frame = frame[frame["layout_branch_id"].astype(str).eq(str(layout_branch_id))]
    if failure_label:
        frame = frame[frame["failure_label"].astype(str).eq(str(failure_label))]
    if cluster_key_filter:
        frame["_cluster_key"] = [cluster_key(row) for row in frame.to_dict(orient="records")]
        frame = frame[frame["_cluster_key"].astype(str).eq(str(cluster_key_filter))]
    frame = frame.sort_values("trial_descriptor_id").head(int(max_rows))
    destination = output_path or (
        root
        / "upload_package"
        / f"diagnostic_slice_s{int(run_id):03d}.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=13)
    parser.add_argument("--planning-run-id", type=int, default=12)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--expected-trials-total", type=int, default=500000)
    parser.add_argument("--expected-trials-per-branch", type=int, default=250000)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--archive-scale-mode", default="strict")
    parser.add_argument("--build-upload-package", action="store_true")
    parser.add_argument("--profile-source", type=Path, default=None)
    parser.add_argument("--export-slice", action="store_true")
    parser.add_argument("--slice-layout-branch-id", default=None)
    parser.add_argument("--slice-failure-label", default=None)
    parser.add_argument("--slice-cluster-key", default=None)
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
            failure_label=args.slice_failure_label,
            cluster_key_filter=args.slice_cluster_key,
            max_rows=args.slice_max_rows,
            output_path=args.slice_output_path,
        )
        print(f"w0_diagnostic_slice={_path_text(path)}")
        return 0
    paths = aggregate_w0_dense_archive(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        result_root=args.result_root,
        expected_trials_total=args.expected_trials_total,
        expected_trials_per_branch=args.expected_trials_per_branch,
        storage_format=args.storage_format,
        archive_scale_mode=args.archive_scale_mode,
        build_upload_package=args.build_upload_package,
        profile_source=args.profile_source,
    )
    print(f"w0_dense_archive_aggregation_outputs={_path_text(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
