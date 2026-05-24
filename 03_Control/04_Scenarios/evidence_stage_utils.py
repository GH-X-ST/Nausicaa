from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB
from dense_archive_table_io import filesystem_path


STAGE_STATUS_VALUES = (
    "complete",
    "weak",
    "failed",
    "rejected",
    "blocked",
    "retired_not_active",
)
CLAIM_BOUNDARY_TEXT = (
    "This stage may claim only corrected W01/W2/W3 workflow readiness or fixed-LQR "
    "survival replay status. It makes no governor, hardware, transfer, robustness, "
    "or mission-success claim."
)


@dataclass(frozen=True)
class StageEvidenceStatus:
    stage: str
    status: str
    run_root: str
    row_count: int
    table_manifest_path: str
    blocked_reason: str
    file_size_status: str
    coverage_status: str
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
    blocked_reason: str = "",
    file_size_status: str = "not_checked",
    coverage_status: str = "not_checked",
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
        blocked_reason=str(blocked_reason),
        file_size_status=str(file_size_status),
        coverage_status=str(coverage_status),
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


def write_file_size_audit(run_root: Path, output_path: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    status = "pass"
    for path in sorted(Path(run_root).rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(run_root).as_posix()
        byte_count = int(filesystem_path(path).stat().st_size)
        size_mb = float(byte_count) / (1024.0 * 1024.0)
        under_100mb = bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB)
        if not under_100mb:
            status = "fail_push_safety_audit"
        rows.append(
            {
                "relative_path": relative_path,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "under_100mb": under_100mb,
                "dense_table_partition": "/tables/" in f"/{relative_path}",
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
        values = frame.get(column, pd.Series(dtype=str)).astype(str)
        for value, count in values.value_counts(dropna=False).items():
            rows.append({"coverage_axis": column, "coverage_value": str(value), "row_count": int(count)})
        if values.empty:
            rows.append({"coverage_axis": column, "coverage_value": "empty", "row_count": 0})
    summary = pd.DataFrame(rows)
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(filesystem_path(path), index=False)
    status = "pass" if rows and not frame.empty else "partial_missing_axes"
    return summary, status


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
