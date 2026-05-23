from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from archive_table_reader import read_archive_table  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from state_contract import STATE_NAMES  # noqa: E402


@dataclass(frozen=True)
class W2ReplayConfig:
    run_id: int
    output_root: Path
    source_archive: Path | None = None


def parse_args(argv: list[str] | None = None) -> W2ReplayConfig:
    parser = argparse.ArgumentParser(description="Write a temp W2 replay scaffold manifest.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, default=None)
    args = parser.parse_args(argv)
    return W2ReplayConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        source_archive=None if args.source_archive is None else Path(args.source_archive),
    )


def run_w2_replay_scaffold(config: W2ReplayConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"w2_replay_{config.run_id:03d}"
    for rel in ("manifests", "reports", "metrics", "tables"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    replay_rows = _replay_rows_from_source(config.source_archive)
    replay_status = (
        "mixed_start_w2_replay_smoke_from_source"
        if replay_rows
        else "blocked_until_approved_R6_archive_exists"
    )
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R8_W2_replay_scaffold",
        "source_archive": "" if config.source_archive is None else Path(config.source_archive).as_posix(),
        "W_layer": "W2",
        "surrogate_family": "gp_corrected_annular_gaussian",
        "required_latency_mechanisms": ["state_feedback_delay", "command_delay", "actuator_lag"],
        "replay_status": replay_status,
        "R8_W2_replay_complete": False,
        "replayed_row_count": len(replay_rows),
        "claim_status": "simulation_only_w2_scaffold_no_survival_claim",
        "blocked_claims": ["W2_survival", "hardware_readiness", "real_flight_transfer"],
    }
    manifest_path = run_root / "manifests" / "w2_replay_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    replay_table = run_root / "tables" / "w2_replay_rows.csv"
    pd.DataFrame(replay_rows).to_csv(filesystem_path(replay_table), index=False)
    filesystem_path(run_root / "reports" / "w2_replay_report.md").write_text(
        "# W2 Replay Scaffold\n\nNo W2 survival or hardware-readiness claim is made.\n",
        encoding="ascii",
    )
    _write_file_size_audit(run_root)
    return {"run_root": run_root, "manifest": manifest_path, "replay_table": replay_table}


def _replay_rows_from_source(source_archive: Path | None) -> list[dict[str, object]]:
    if source_archive is None:
        return []
    frame = read_archive_table(Path(source_archive), max_rows=256)
    if frame.empty:
        return []
    rows = []
    for label in ("accepted", "weak", "boundary_terminal", "failed", "rejected"):
        subset = frame[frame["outcome_class"] == label] if "outcome_class" in frame else frame.iloc[0:0]
        if subset.empty:
            continue
        row = subset.iloc[0].to_dict()
        rows.append(_w2_replay_row(row))
    blocked = frame[frame["outcome_class"] == "blocked"] if "outcome_class" in frame else frame.iloc[0:0]
    if not blocked.empty:
        rows.append(_w2_replay_row(blocked.iloc[0].to_dict(), audit_only=True))
    return rows


def _w2_replay_row(row: dict[str, object], *, audit_only: bool = False) -> dict[str, object]:
    source_outcome = str(row.get("outcome_class", "blocked"))
    label_by_source = {
        "accepted": "w2_survived",
        "weak": "w2_weak",
        "boundary_terminal": "w2_boundary_terminal",
        "failed": "w2_failed",
        "rejected": "w2_failed",
        "blocked": "w2_blocked",
    }
    result = {
        "source_rollout_id": row.get("rollout_id", ""),
        "source_outcome_class": source_outcome,
        "w2_replay_label": "w2_blocked" if audit_only else label_by_source.get(source_outcome, "w2_blocked"),
        "w2_replay_audit_only": bool(audit_only),
        "start_state_family": row.get("start_state_family", ""),
        "state_envelope_label": row.get("state_envelope_label", ""),
        "previous_primitive_status": row.get("previous_primitive_status", ""),
        "primitive_id": row.get("primitive_id", ""),
        "latency_case": "nominal",
        "boundary_use_class": row.get("boundary_use_class", ""),
        "claim_status": "simulation_only_w2_replay_smoke_no_survival_claim",
    }
    result.update(
        {
            f"entry_{name}": float(row.get(f"initial_{name}", 0.0))
            for name in STATE_NAMES
        }
    )
    return result


def _write_file_size_audit(run_root: Path) -> None:
    rows = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            rows.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "byte_count": int(size),
                    "under_100mb": bool(size <= 100 * 1024 * 1024),
                }
            )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    run_w2_replay_scaffold(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
