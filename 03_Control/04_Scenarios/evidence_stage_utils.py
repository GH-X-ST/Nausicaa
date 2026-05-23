from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from dense_archive_runtime import (
    MAX_GENERATED_FILE_SIZE_MB,
    PREFERRED_GENERATED_FILE_SIZE_MB,
)
from dense_archive_table_io import filesystem_path


STAGE_STATUS_VALUES = (
    "complete",
    "accepted_fallback",
    "smoke_incomplete",
    "blocked",
    "retired_not_active",
)
CLAIM_BOUNDARY_TEXT = (
    "No controller-performance, mission-success, hardware-readiness, "
    "real-flight-transfer, full W2 survival, full W3 robustness, or "
    "environment-generalisation claim is made by this stage."
)


@dataclass(frozen=True)
class StageEvidenceStatus:
    stage: str
    status: str
    run_root: str
    row_count: int
    table_manifest_path: str
    fallback_reason: str
    blocked_ratio: float
    approximate_ratio: float
    file_size_status: str
    coverage_status: str
    surrogate_status: str
    runtime_projection_s: float
    claim_status: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def stage_status(
    *,
    stage: str,
    status: str,
    run_root: Path | str = "",
    row_count: int = 0,
    table_manifest_path: Path | str = "",
    fallback_reason: str = "",
    blocked_ratio: float = 0.0,
    approximate_ratio: float = 0.0,
    file_size_status: str = "not_checked",
    coverage_status: str = "not_checked",
    surrogate_status: str = "not_checked",
    runtime_projection_s: float = 0.0,
    claim_status: str = "simulation_only_no_claim",
) -> StageEvidenceStatus:
    if status not in STAGE_STATUS_VALUES:
        raise ValueError(f"stage status must be one of {STAGE_STATUS_VALUES}.")
    return StageEvidenceStatus(
        stage=str(stage),
        status=str(status),
        run_root="" if not run_root else Path(run_root).as_posix(),
        row_count=int(row_count),
        table_manifest_path="" if not table_manifest_path else Path(table_manifest_path).as_posix(),
        fallback_reason=str(fallback_reason),
        blocked_ratio=float(blocked_ratio),
        approximate_ratio=float(approximate_ratio),
        file_size_status=str(file_size_status),
        coverage_status=str(coverage_status),
        surrogate_status=str(surrogate_status),
        runtime_projection_s=float(runtime_projection_s),
        claim_status=str(claim_status),
    )


def write_evidence_status_manifest(
    path: Path,
    *,
    statuses: list[StageEvidenceStatus],
    run_label: str,
    run_root: Path | str = "",
    metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "run_label": str(run_label),
        "run_root": "" if not run_root else Path(run_root).as_posix(),
        "stage_statuses": [status.as_row() for status in statuses],
        "claim_boundary": CLAIM_BOUNDARY_TEXT,
        "metadata": {} if metadata is None else dict(metadata),
    }
    output = filesystem_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def runtime_projection_s(
    *,
    total_rows: int,
    first_chunk_rows: int,
    first_chunk_seconds: float,
    actual_worker_count: int,
    safety_factor: float = 1.25,
) -> float:
    rows = max(1, int(total_rows))
    chunk_rows = max(1, int(first_chunk_rows))
    workers = max(1, int(actual_worker_count))
    total_chunks = int(math.ceil(rows / chunk_rows))
    return float(math.ceil(total_chunks / workers) * float(first_chunk_seconds) * float(safety_factor))


def write_file_size_audit(run_root: Path, output_path: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    status = "pass"
    for path in sorted(Path(run_root).rglob("*")):
        if path.is_file():
            byte_count = int(filesystem_path(path).stat().st_size)
            size_mb = float(byte_count) / (1024.0 * 1024.0)
            above_100mb = bool(size_mb > MAX_GENERATED_FILE_SIZE_MB)
            above_75mb = bool(size_mb > PREFERRED_GENERATED_FILE_SIZE_MB)
            if above_100mb:
                status = "fail_oversized_file"
            rows.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "byte_count": byte_count,
                    "size_mb": size_mb,
                    "above_75mb": above_75mb,
                    "above_100mb": above_100mb,
                    "under_100mb": not above_100mb,
                }
            )
    filesystem_path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filesystem_path(output_path), index=False)
    return rows, status


def write_coverage_summary(
    path: Path,
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in frame.columns:
            rows.append(
                {
                    "coverage_axis": column,
                    "coverage_value": "missing_column",
                    "row_count": 0,
                    "coverage_note": "missing",
                }
            )
            continue
        counts = frame[column].fillna("").astype(str).value_counts(dropna=False)
        if counts.empty:
            rows.append(
                {
                    "coverage_axis": column,
                    "coverage_value": "empty",
                    "row_count": 0,
                    "coverage_note": "empty",
                }
            )
        for value, count in counts.items():
            rows.append(
                {
                    "coverage_axis": column,
                    "coverage_value": str(value),
                    "row_count": int(count),
                    "coverage_note": "observed",
                }
            )
    summary = pd.DataFrame(rows)
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(filesystem_path(path), index=False)
    missing_axes = {
        row["coverage_axis"]
        for row in rows
        if row["coverage_note"] in {"missing", "empty"}
    }
    status = "pass" if not missing_axes and not frame.empty else "partial_missing_axes"
    return summary, status


def write_blocked_approximate_ratio_summary(
    path: Path,
    frame: pd.DataFrame,
    *,
    blocked_columns: tuple[str, ...] = ("outcome_class", "surrogate_binding_status"),
    approximate_columns: tuple[str, ...] = (
        "environment_adjustment_status",
        "environment_adjustment_limitations",
    ),
) -> dict[str, object]:
    row_count = int(len(frame))
    if row_count <= 0:
        payload = {
            "row_count": 0,
            "blocked_count": 0,
            "approximate_count": 0,
            "blocked_ratio": 1.0,
            "approximate_ratio": 0.0,
        }
    else:
        blocked_mask = pd.Series(False, index=frame.index)
        for column in blocked_columns:
            if column in frame.columns:
                blocked_mask = blocked_mask | frame[column].astype(str).str.contains("blocked", case=False, na=False)
        approximate_mask = pd.Series(False, index=frame.index)
        for column in approximate_columns:
            if column in frame.columns:
                approximate_mask = approximate_mask | frame[column].astype(str).str.contains("approximate", case=False, na=False)
        blocked_count = int(blocked_mask.sum())
        approximate_count = int(approximate_mask.sum())
        payload = {
            "row_count": row_count,
            "blocked_count": blocked_count,
            "approximate_count": approximate_count,
            "blocked_ratio": float(blocked_count / row_count),
            "approximate_ratio": float(approximate_count / row_count),
        }
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(filesystem_path(path), index=False)
    return payload


def write_claim_boundary_report(
    path: Path,
    *,
    stage: str,
    status: str,
    claim_status: str,
    blocked_claims: tuple[str, ...],
) -> None:
    lines = [
        f"# {stage} Claim Boundary",
        "",
        f"- Stage status: `{status}`",
        f"- Claim status: `{claim_status}`",
        f"- Blocked claims: `{'; '.join(blocked_claims)}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
    ]
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text("\n".join(lines), encoding="ascii")


def status_from_blocked_ratio(
    *,
    target_status: str,
    blocked_ratio: float,
    fallback_threshold: float,
    partial_threshold: float,
    fallback_status: str = "accepted_fallback",
) -> str:
    if target_status not in {"complete", "accepted_fallback"}:
        return target_status
    if float(blocked_ratio) > float(partial_threshold):
        return "blocked"
    if float(blocked_ratio) > float(fallback_threshold):
        return "smoke_incomplete" if target_status == fallback_status else fallback_status
    return target_status
