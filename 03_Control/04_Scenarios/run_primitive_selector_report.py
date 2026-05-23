from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from archive_table_reader import read_archive_table  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from env_ctx import ENV_CONTEXT_COLUMNS, EnvironmentContext  # noqa: E402
from prim_cat import active_primitive_catalogue  # noqa: E402
from prim_model import fit_primitive_outcome_model  # noqa: E402
from prim_select import primitive_selection_row, select_primitive  # noqa: E402
from state_contract import STATE_NAMES  # noqa: E402


@dataclass(frozen=True)
class SelectorReportConfig:
    run_id: int
    archive_table: Path
    output_root: Path
    governor_mode: str = "continuation"
    k_neighbours: int = 3
    max_rows: int = 32


def parse_args(argv: list[str] | None = None) -> SelectorReportConfig:
    parser = argparse.ArgumentParser(description="Run a temp selector report from archive rows.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--archive-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--governor-mode", choices=("continuation", "terminal_episode"), default="continuation")
    parser.add_argument("--k-neighbours", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=32)
    args = parser.parse_args(argv)
    return SelectorReportConfig(
        run_id=int(args.run_id),
        archive_table=Path(args.archive_table),
        output_root=Path(args.output_root),
        governor_mode=str(args.governor_mode),
        k_neighbours=int(args.k_neighbours),
        max_rows=int(args.max_rows),
    )


def run_primitive_selector_report(config: SelectorReportConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"selector_report_{config.run_id:03d}"
    for rel in ("manifests", "tables", "reports", "metrics"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    frame = read_archive_table(config.archive_table, max_rows=int(config.max_rows))
    rows = frame.to_dict(orient="records")
    model = fit_primitive_outcome_model(rows, k_neighbours=config.k_neighbours)
    decisions = []
    primitives = active_primitive_catalogue()
    for row in rows[: min(8, len(rows))]:
        context = _context_from_row(row)
        state = _state_from_row(row)
        result = select_primitive(
            context=context,
            model=model,
            catalogue=primitives,
            current_state=state,
            governor_mode=config.governor_mode,
            max_uncertainty=1_000_000.0,
        )
        out = primitive_selection_row(result)
        out["source_rollout_id"] = row.get("rollout_id", "")
        out.update(_canonical_entry_state_columns(row))
        out.update(_derived_report_group_columns(row))
        decisions.append(out)
    decision_path = run_root / "tables" / "selector_decisions.csv"
    pd.DataFrame(decisions).to_csv(filesystem_path(decision_path), index=False)
    manifest = {
        "run_id": int(config.run_id),
        "archive_table": Path(config.archive_table).as_posix(),
        "training_row_count": int(model.fitted_row_count),
        "validation_split_type": "derived_mixed_start_smoke_groups",
        "validation_split_columns": _available_split_columns(rows),
        "governor_mode": str(config.governor_mode),
        "claim_status": "simulation_only_selector_report_smoke",
        "blocked_claims": ["controller_performance", "hardware_readiness", "real_flight_transfer"],
    }
    manifest_path = run_root / "manifests" / "selector_report_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    _write_validation_split_summary(run_root / "metrics" / "validation_split_summary.csv", rows)
    filesystem_path(run_root / "reports" / "selector_report.md").write_text(
        "# Primitive Selector Smoke Report\n\nNo performance, transfer, or hardware-readiness claim is made.\n",
        encoding="ascii",
    )
    _write_file_size_audit(run_root)
    return {"run_root": run_root, "manifest": manifest_path, "decision_table": decision_path}


def _context_from_row(row: dict[str, object]) -> EnvironmentContext:
    return EnvironmentContext(**{name: row[f"context_{name}"] for name in ENV_CONTEXT_COLUMNS})


def _state_from_row(row: dict[str, object]) -> np.ndarray:
    names = ("x_w", "y_w", "z_w", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "delta_a", "delta_e", "delta_r")
    return np.asarray([float(row.get(f"initial_{name}", 0.0)) for name in names], dtype=float)


def _canonical_entry_state_columns(row: dict[str, object]) -> dict[str, float]:
    return {
        f"entry_{name}": float(row.get(f"initial_{name}", 0.0))
        for name in STATE_NAMES
    }


def _derived_report_group_columns(row: dict[str, object]) -> dict[str, object]:
    return {
        "start_state_family": row.get("start_state_family", ""),
        "state_envelope_label": row.get("state_envelope_label", ""),
        "previous_primitive_status": row.get("previous_primitive_status", ""),
        "source_W_layer": row.get("W_layer", ""),
        "source_environment_instance_id": row.get(
            "environment_instance_environment_id",
            row.get("environment_id", ""),
        ),
        "source_primitive_id": row.get("primitive_id", ""),
        "source_latency_case": row.get("latency_case", ""),
        "source_outcome_class": row.get("outcome_class", ""),
        "source_boundary_use_class": row.get("boundary_use_class", ""),
    }


def _available_split_columns(rows: list[dict[str, object]]) -> list[str]:
    candidates = [
        "start_state_family",
        "state_envelope_label",
        "environment_instance_environment_id",
        "paired_start_key",
        "primitive_id",
        "W_layer",
        "latency_case",
        "outcome_class",
        "boundary_use_class",
    ]
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return [name for name in candidates if name in keys]


def _write_validation_split_summary(path: Path, rows: list[dict[str, object]]) -> None:
    columns = _available_split_columns(rows)
    if not rows or not columns:
        pd.DataFrame(columns=["row_count"]).to_csv(filesystem_path(path), index=False)
        return
    group_columns = columns[:4]
    frame = (
        pd.DataFrame(rows)
        .groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    frame.to_csv(filesystem_path(path), index=False)


def _write_file_size_audit(run_root: Path) -> None:
    rows = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            rows.append({"path": path.relative_to(run_root).as_posix(), "byte_count": int(size), "under_100mb": bool(size <= 100 * 1024 * 1024)})
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    run_primitive_selector_report(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
