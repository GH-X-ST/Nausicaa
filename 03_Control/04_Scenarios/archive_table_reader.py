from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dense_archive_table_io import (
    filesystem_path,
    load_table_manifest,
    read_table_partition,
)
from evidence_status import is_thesis_eligible_status, source_is_retired


@dataclass(frozen=True)
class ArchiveTableSourceInfo:
    source_path: str
    source_kind: str
    manifest_path: str
    run_manifest_path: str
    row_count_loaded: int
    row_count_manifested: int
    storage_format: str
    run_stage: str
    claim_status: str
    rollout_backend: str
    rows_requested: int
    evidence_eligible: bool
    selected_controller_registry_path: str
    registry_backed_row_count: int
    missing_controller_row_count: int
    blocked_ratio: float
    missing_controller_ratio: float
    archive_evidence_status: str
    evidence_eligibility_reason: str


def read_archive_table(source: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read archive rows from a manifest, run root, or table partition."""

    frame, _ = read_archive_table_with_info(source, max_rows=max_rows)
    return frame


def read_archive_table_with_info(
    source: Path,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, ArchiveTableSourceInfo]:
    """Read archive rows and return compact source metadata."""

    path = Path(source)
    fs_path = filesystem_path(path)
    if fs_path.is_dir():
        manifest = path / "manifests" / "table_manifest.json"
        if filesystem_path(manifest).is_file():
            frame = _read_manifest(manifest, max_rows=max_rows)
            return frame, _source_info(
                source=path,
                source_kind="run_root",
                manifest_path=manifest,
                frame=frame,
            )
        partitions = sorted(
            item for item in fs_path.rglob("*") if item.is_file() and _is_partition_name(item.name)
        )
        frame = _concat_partitions(partitions, max_rows=max_rows)
        return frame, _source_info(
            source=path,
            source_kind="partition_directory",
            manifest_path=None,
            frame=frame,
        )
    if path.name == "table_manifest.json":
        frame = _read_manifest(path, max_rows=max_rows)
        return frame, _source_info(
            source=path,
            source_kind="table_manifest",
            manifest_path=path,
            frame=frame,
        )
    if _is_partition_name(path.name):
        frame = read_table_partition(path)
        frame = frame.head(max_rows) if max_rows is not None else frame
        return frame, _source_info(
            source=path,
            source_kind="single_partition",
            manifest_path=None,
            frame=frame,
        )
    raise ValueError(f"unsupported archive table source: {path}")


def _read_manifest(path: Path, *, max_rows: int | None) -> pd.DataFrame:
    manifest = load_table_manifest(path)
    root = Path(manifest.root)
    frames = []
    remaining = None if max_rows is None else int(max_rows)
    for partition in manifest.tables:
        partition_path = root / "tables" / partition.relative_path
        frame = read_table_partition(partition_path, storage_format=partition.storage_format)
        if remaining is not None:
            frame = frame.head(remaining)
            remaining -= len(frame)
        frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _concat_partitions(paths: list[Path], *, max_rows: int | None) -> pd.DataFrame:
    frames = []
    remaining = None if max_rows is None else int(max_rows)
    for path in paths:
        frame = read_table_partition(Path(path))
        if remaining is not None:
            frame = frame.head(remaining)
            remaining -= len(frame)
        frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _is_partition_name(name: str) -> bool:
    lower = str(name).lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz") or lower.endswith(".parquet")


def _source_info(
    *,
    source: Path,
    source_kind: str,
    manifest_path: Path | None,
    frame: pd.DataFrame,
) -> ArchiveTableSourceInfo:
    run_manifest_path = _run_manifest_path(manifest_path)
    run_manifest = _read_json(run_manifest_path)
    table_manifest = load_table_manifest(manifest_path) if manifest_path is not None else None
    row_count_manifested = (
        sum(int(partition.row_count) for partition in table_manifest.tables)
        if table_manifest is not None
        else int(len(frame))
    )
    storage_format = (
        str(table_manifest.storage_format)
        if table_manifest is not None
        else _storage_format_from_frame_source(source)
    )
    rows_requested = int(run_manifest.get("rows_requested", row_count_manifested))
    claim_status = str(run_manifest.get("claim_status", "unknown"))
    run_stage = str(run_manifest.get("run_stage", run_manifest.get("stage", "unknown")))
    rollout_backend = str(run_manifest.get("rollout_backend", ""))
    if not rollout_backend and run_stage in {"R8_W2_model_backed_replay", "R9_W3_generalisation_replay"}:
        rollout_backend = "model_backed_lqr"
    if not rollout_backend:
        rollout_backend = "unknown"
    nested_source_info = run_manifest.get("archive_source_info", {})
    nested_registry_path = (
        str(nested_source_info.get("selected_controller_registry_path", ""))
        if isinstance(nested_source_info, dict)
        else ""
    )
    selected_controller_registry_path = str(
        run_manifest.get("selected_controller_registry", "")
        or nested_registry_path
        or _first_nonempty(frame, "registry_path")
        or _first_nonempty(frame, "source_registry_path")
    )
    archive_evidence_status, evidence_eligibility_reason, counts = _archive_evidence_status(
        source=source,
        frame=frame,
        run_manifest=run_manifest,
        manifest_path=manifest_path,
        rows_requested=rows_requested,
        rollout_backend=rollout_backend,
        selected_controller_registry_path=selected_controller_registry_path,
    )
    evidence_eligible = is_thesis_eligible_status(archive_evidence_status)
    return ArchiveTableSourceInfo(
        source_path=Path(source).as_posix(),
        source_kind=str(source_kind),
        manifest_path="" if manifest_path is None else Path(manifest_path).as_posix(),
        run_manifest_path="" if run_manifest_path is None else Path(run_manifest_path).as_posix(),
        row_count_loaded=int(len(frame)),
        row_count_manifested=int(row_count_manifested),
        storage_format=storage_format,
        run_stage=run_stage,
        claim_status=claim_status,
        rollout_backend=rollout_backend,
        rows_requested=rows_requested,
        evidence_eligible=evidence_eligible,
        selected_controller_registry_path=selected_controller_registry_path,
        registry_backed_row_count=int(counts["registry_backed_row_count"]),
        missing_controller_row_count=int(counts["missing_controller_row_count"]),
        blocked_ratio=float(counts["blocked_ratio"]),
        missing_controller_ratio=float(counts["missing_controller_ratio"]),
        archive_evidence_status=archive_evidence_status,
        evidence_eligibility_reason=evidence_eligibility_reason,
    )


def _archive_evidence_status(
    *,
    source: Path,
    frame: pd.DataFrame,
    run_manifest: dict[str, object],
    manifest_path: Path | None,
    rows_requested: int,
    rollout_backend: str,
    selected_controller_registry_path: str,
) -> tuple[str, str, dict[str, float]]:
    row_count = int(len(frame))
    controller_selection = _string_series(frame, "controller_selection_status")
    outcome_class = _string_series(frame, "outcome_class")
    controller_evidence = _string_series(frame, "controller_evidence_status")
    registry_backed = controller_selection.isin(
        ["W0_W1_registry_selected", "W2_verified_registry_replay", "W3_verified_registry_replay"]
    )
    nonblocked = outcome_class.ne("blocked")
    missing_controller = controller_evidence.str.contains("missing", na=False) | controller_selection.str.contains(
        "missing",
        na=False,
    )
    blocked_ratio = float(outcome_class.eq("blocked").mean()) if row_count else 1.0
    missing_controller_ratio = float(missing_controller.mean()) if row_count else 1.0
    counts = {
        "registry_backed_row_count": float((registry_backed & nonblocked).sum()),
        "missing_controller_row_count": float(missing_controller.sum()),
        "blocked_ratio": blocked_ratio,
        "missing_controller_ratio": missing_controller_ratio,
    }
    if source_is_retired(source):
        return "retired_not_active", "blocked_retired_source", counts
    if manifest_path is None:
        return "smoke_incomplete", "debug_smoke_incomplete", counts
    if bool(run_manifest.get("dry_run_schedule", False)):
        return "smoke_incomplete", "debug_smoke_incomplete", counts
    if rollout_backend != "model_backed_lqr":
        return "smoke_incomplete", "debug_smoke_incomplete", counts
    manifest_status = str(run_manifest.get("archive_evidence_status", ""))
    replay_stage = str(run_manifest.get("stage", run_manifest.get("run_stage", ""))) in {
        "R8_W2_model_backed_replay",
        "R9_W3_generalisation_replay",
    }
    if not selected_controller_registry_path:
        return "blocked", "blocked_missing_selected_registry", counts
    if row_count == 0:
        return "blocked", "blocked_no_rows", counts
    if counts["registry_backed_row_count"] <= 0:
        return "blocked", "blocked_no_registry_backed_nonblocked_rows", counts
    if not _candidate_metadata_complete(frame, registry_backed & nonblocked):
        return "blocked", "blocked_missing_candidate_metadata", counts
    registry_statuses = {
        value
        for value in _string_series(frame, "registry_status")[registry_backed & nonblocked].str.strip()
        if value and value.lower() not in {"nan", "none", "null"}
    }
    if not registry_statuses:
        return "blocked", "blocked_missing_registry_status", counts
    if "retired_not_active" in registry_statuses:
        return "retired_not_active", "blocked_retired_source", counts
    if "blocked" in registry_statuses:
        return "blocked", "blocked_registry_status_not_eligible", counts
    if "smoke_incomplete" in registry_statuses:
        return "smoke_incomplete", "debug_smoke_incomplete", counts
    if not registry_statuses.issubset({"complete", "accepted_fallback"}):
        return "blocked", "blocked_registry_status_not_eligible", counts
    if missing_controller_ratio > 0.05:
        return "blocked", "blocked_high_missing_controller_ratio", counts
    if not replay_stage and blocked_ratio > 0.60:
        return "blocked", "blocked_high_blocked_ratio", counts
    if replay_stage:
        if blocked_ratio > 0.70:
            return "blocked", "blocked_high_blocked_ratio", counts
        if manifest_status in {"complete", "accepted_fallback"}:
            return manifest_status, f"eligible_verified_replay_{manifest_status}", counts
        if manifest_status in {"blocked", "retired_not_active"}:
            return manifest_status, str(run_manifest.get("evidence_eligibility_reason", "blocked_replay_source")), counts
        return "smoke_incomplete", "debug_smoke_incomplete", counts
    if manifest_status == "accepted_fallback" or "accepted_fallback" in registry_statuses:
        return "accepted_fallback", "eligible_registry_backed_accepted_fallback", counts
    if manifest_status == "complete" and registry_statuses == {"complete"}:
        return "complete", "eligible_registry_backed_complete", counts
    return "smoke_incomplete", "debug_smoke_incomplete", counts


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series([""] * len(frame))).astype(str)


def _candidate_metadata_complete(frame: pd.DataFrame, mask: pd.Series) -> bool:
    if frame.empty or not bool(mask.any()):
        return False
    required = (
        "candidate_weight_label",
        "lqr_Q_weights_json",
        "lqr_R_weights_json",
        "lqr_gain_checksum",
        "linearisation_id",
    )
    for column in required:
        values = _string_series(frame, column)[mask].str.strip().str.lower()
        if values.empty or values.isin(["", "nan", "none", "null"]).any():
            return False
    return True


def _first_nonempty(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = frame[column].astype(str).str.strip()
    values = values[~values.str.lower().isin(["", "nan", "none", "null"])]
    if values.empty:
        return ""
    return str(values.iloc[0])


def _run_manifest_path(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None
    manifest_dir = Path(manifest_path).parent
    for name in (
        "run_manifest.json",
        "w2_replay_manifest.json",
        "w3_generalisation_manifest.json",
        "selector_report_manifest.json",
    ):
        candidate = manifest_dir / name
        if filesystem_path(candidate).is_file():
            return candidate
    return None


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None or not filesystem_path(path).is_file():
        return {}
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _storage_format_from_frame_source(source: Path) -> str:
    lower = Path(source).name.lower()
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".csv.gz"):
        return "csv_gz"
    if lower.endswith(".csv"):
        return "csv"
    return "unknown"
