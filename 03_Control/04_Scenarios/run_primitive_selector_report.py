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

from archive_table_reader import read_archive_table_with_info  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from controller_registry import controller_from_evidence_row  # noqa: E402
from evidence_stage_utils import (  # noqa: E402
    write_blocked_approximate_ratio_summary,
    write_claim_boundary_report,
    write_coverage_summary,
    write_file_size_audit,
)
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
    governor_modes: tuple[str, ...] = ("continuation", "terminal_episode")
    k_neighbours: int = 3
    max_rows: int = 0
    evaluation_max_rows: int = 4096


def parse_args(argv: list[str] | None = None) -> SelectorReportConfig:
    parser = argparse.ArgumentParser(description="Run a temp selector report from archive rows.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--archive-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--governor-mode", choices=("continuation", "terminal_episode"), default=None)
    parser.add_argument("--governor-modes", default="continuation,terminal_episode")
    parser.add_argument("--k-neighbours", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--evaluation-max-rows", type=int, default=4096)
    args = parser.parse_args(argv)
    return SelectorReportConfig(
        run_id=int(args.run_id),
        archive_table=Path(args.archive_table),
        output_root=Path(args.output_root),
        governor_mode=str(args.governor_mode or "continuation"),
        governor_modes=(
            (str(args.governor_mode),)
            if args.governor_mode
            else _split_csv(args.governor_modes)
        ),
        k_neighbours=int(args.k_neighbours),
        max_rows=int(args.max_rows),
        evaluation_max_rows=int(args.evaluation_max_rows),
    )


def run_primitive_selector_report(config: SelectorReportConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"sel_{config.run_id:03d}"
    for rel in ("manifests", "tables", "reports", "metrics"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    max_rows = None if int(config.max_rows) <= 0 else int(config.max_rows)
    frame, source_info = read_archive_table_with_info(config.archive_table, max_rows=max_rows)
    rows = frame.to_dict(orient="records")
    model = fit_primitive_outcome_model(rows, k_neighbours=config.k_neighbours)
    if model.fitted_row_count <= 0:
        raise ValueError("selector report has empty training data.")
    decisions = []
    candidate_rows = []
    primitives = active_primitive_catalogue()
    controller_registry = _controller_registry_from_rows(rows)
    evaluation_frame, evaluation_metadata = _evaluation_frame(
        frame,
        max_rows=int(config.evaluation_max_rows),
    )
    decision_input_rows = evaluation_frame.to_dict(orient="records")
    for row in decision_input_rows:
        for governor_mode in config.governor_modes:
            context = _context_from_row(row)
            state = _state_from_row(row)
            result = select_primitive(
                context=context,
                model=model,
                catalogue=primitives,
                current_state=state,
                governor_mode=governor_mode,
                max_uncertainty=1_000_000.0,
                controller_registry=controller_registry,
                require_controller_registry=True,
            )
            out = primitive_selection_row(result)
            out["source_rollout_id"] = row.get("rollout_id", "")
            out["governor_mode"] = governor_mode
            out.update(_canonical_entry_state_columns(row))
            out.update(_derived_report_group_columns(row))
            decisions.append(out)
            for candidate in result.decisions:
                candidate_row = {
                    "source_rollout_id": row.get("rollout_id", ""),
                    "governor_mode": governor_mode,
                    **candidate.__dict__,
                    "energy_residual_m": row.get("energy_residual_m", 0.0),
                    "lift_dwell_time_s": row.get("lift_dwell_time_s", 0.0),
                    "minimum_wall_margin_m": row.get("minimum_wall_margin_m", 0.0),
                    "minimum_speed_m_s": row.get("minimum_speed_m_s", 0.0),
                    "continuation_valid": row.get("continuation_valid", False),
                    "episode_terminal_useful": row.get("episode_terminal_useful", False),
                }
                candidate_row.update(_derived_report_group_columns(row))
                candidate_rows.append(candidate_row)
    decision_path = run_root / "tables" / "selector_decisions.csv"
    pd.DataFrame(decisions).to_csv(filesystem_path(decision_path), index=False)
    candidate_path = run_root / "tables" / "selector_candidate_log.csv"
    pd.DataFrame(candidate_rows).to_csv(filesystem_path(candidate_path), index=False)
    _write_rejection_summary(run_root / "metrics" / "rejection_summary.csv", candidate_rows)
    _write_selected_summary(run_root / "metrics" / "selected_primitive_summary.csv", decisions)
    _write_validation_split_summary(run_root / "metrics" / "validation_split_summary.csv", rows)
    _write_split_manifest(run_root / "manifests" / "train_validation_test_split_manifest.json", rows)
    _write_confusion_matrices(run_root / "metrics" / "confusion_matrix_by_primitive_w_layer.csv", decisions)
    _write_calibration_scores(run_root / "metrics" / "calibration_brier_scores.csv", candidate_rows)
    _write_regression_errors(run_root / "metrics" / "regression_error_summary.csv", candidate_rows)
    write_coverage_summary(
        run_root / "metrics" / "coverage_summary.csv",
        pd.DataFrame(decisions),
        columns=(
            "start_state_family",
            "state_envelope_label",
            "previous_primitive_status",
            "source_environment_instance_id",
            "source_primitive_id",
            "source_W_layer",
            "source_latency_case",
            "source_outcome_class",
            "source_boundary_use_class",
            "source_continuation_valid",
            "source_episode_terminal_useful",
            "source_archive_evidence_status",
            "source_evidence_eligibility_reason",
            "source_registry_status",
            "governor_mode",
        ),
    )
    ratio_summary = write_blocked_approximate_ratio_summary(
        run_root / "metrics" / "blocked_or_approximate_ratio_summary.csv",
        frame,
    )
    file_audit, file_status = write_file_size_audit(
        run_root,
        run_root / "metrics" / "file_size_audit.csv",
    )
    evaluation_row_count = int(len(decision_input_rows))
    evaluation_strategy = str(evaluation_metadata["evaluation_strategy"])
    declared_bounded_ok = bool(
        evaluation_strategy == "declared_stratified_subset"
        and evaluation_row_count >= min(512, max(1, source_info.row_count_loaded))
    )
    full_evaluation_ok = evaluation_strategy == "full_source"
    r7_complete = bool(
        source_info.evidence_eligible
        and model.fitted_row_count == source_info.row_count_manifested
        and set(config.governor_modes) == {"continuation", "terminal_episode"}
        and candidate_rows
        and file_status == "pass"
        and (full_evaluation_ok or declared_bounded_ok)
    )
    stage_status = "complete" if r7_complete and full_evaluation_ok else (
        "fallback" if r7_complete and declared_bounded_ok else "partial"
    )
    manifest = {
        "run_id": int(config.run_id),
        "archive_table": Path(config.archive_table).as_posix(),
        "archive_source_info": source_info.__dict__,
        "training_row_count": int(model.fitted_row_count),
        "decision_row_count": int(len(decisions)),
        "candidate_row_count": int(len(candidate_rows)),
        "full_training_row_count": int(model.fitted_row_count),
        "evaluation_row_count": int(len(decision_input_rows)),
        "evaluation_strategy": evaluation_strategy,
        "evaluation_strata": evaluation_metadata["evaluation_strata"],
        "bounded_evaluation_reason": evaluation_metadata["bounded_evaluation_reason"],
        "source_manifest_path": source_info.manifest_path,
        "validation_split_type": "derived_mixed_start_groups",
        "validation_split_columns": _available_split_columns(rows),
        "governor_modes": list(config.governor_modes),
        "feature_schema_version": (
            model.records[0].feature_schema_version if model.records else "unfitted"
        ),
        "max_rows": int(config.max_rows),
        "evaluation_max_rows": int(config.evaluation_max_rows),
        "R7_selector_report_complete": r7_complete,
        "stage_status": stage_status,
        "archive_evidence_status": source_info.archive_evidence_status,
        "evidence_eligibility_reason": source_info.evidence_eligibility_reason,
        "blocked_ratio": float(ratio_summary["blocked_ratio"]),
        "file_size_status": file_status,
        "claim_status": (
            "simulation_only_selector_report_from_R6_archive"
            if source_info.evidence_eligible
            else "simulation_only_selector_report_smoke_or_subset"
        ),
        "blocked_claims": ["controller_performance", "hardware_readiness", "real_flight_transfer"],
    }
    manifest_path = run_root / "manifests" / "selector_report_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    filesystem_path(run_root / "reports" / "selector_report.md").write_text(
        "# Primitive Selector Report\n\nNo performance, transfer, or hardware-readiness claim is made.\n",
        encoding="ascii",
    )
    write_claim_boundary_report(
        run_root / "reports" / "claim_boundary_report.md",
        stage="R7 selector report",
        status=str(manifest["stage_status"]),
        claim_status=str(manifest["claim_status"]),
        blocked_claims=tuple(str(item) for item in manifest["blocked_claims"]),
    )
    return {
        "run_root": run_root,
        "manifest": manifest_path,
        "decision_table": decision_path,
        "candidate_table": candidate_path,
        "file_size_audit": run_root / "metrics" / "file_size_audit.csv",
    }


def _split_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must contain at least one value.")
    unknown = set(values) - {"continuation", "terminal_episode"}
    if unknown:
        raise ValueError(f"unknown governor modes: {sorted(unknown)}")
    return values


def _context_from_row(row: dict[str, object]) -> EnvironmentContext:
    return EnvironmentContext(**{name: row[f"context_{name}"] for name in ENV_CONTEXT_COLUMNS})


def _evaluation_frame(frame: pd.DataFrame, *, max_rows: int) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty:
        return frame.copy(), {
            "evaluation_strategy": "empty",
            "evaluation_strata": [],
            "bounded_evaluation_reason": "empty_source",
        }
    if int(max_rows) <= 0 or len(frame) <= int(max_rows):
        return frame.copy(), {
            "evaluation_strategy": "full_source",
            "evaluation_strata": _available_split_columns(frame.to_dict(orient="records")),
            "bounded_evaluation_reason": "",
        }
    strata_columns = [
        column
        for column in (
            "start_state_family",
            "state_envelope_label",
            "previous_primitive_status",
            "environment_instance_environment_id",
            "primitive_id",
            "W_layer",
            "latency_case",
            "outcome_class",
            "boundary_use_class",
        )
        if column in frame.columns
    ]
    if not strata_columns:
        return frame.head(int(max_rows)).copy(), {
            "evaluation_strategy": "bounded_head_fallback",
            "evaluation_strata": [],
            "bounded_evaluation_reason": "no_stratification_columns_available",
        }
    grouped = frame.groupby(strata_columns, dropna=False, sort=False)
    selected_indices: list[int] = []
    groups = [group for _, group in grouped]
    cursor = 0
    while len(selected_indices) < int(max_rows) and groups:
        group = groups[cursor % len(groups)]
        offset = cursor // len(groups)
        if offset < len(group):
            selected_indices.append(int(group.index[offset]))
        cursor += 1
        if cursor > int(max_rows) * max(2, len(groups) + 1):
            break
    return frame.loc[selected_indices].copy(), {
        "evaluation_strategy": "declared_stratified_subset",
        "evaluation_strata": strata_columns,
        "bounded_evaluation_reason": (
            f"source_rows_{len(frame)}_exceed_evaluation_max_rows_{int(max_rows)}"
        ),
    }


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
        "source_continuation_valid": row.get("continuation_valid", ""),
        "source_episode_terminal_useful": row.get("episode_terminal_useful", ""),
        "source_archive_evidence_status": row.get("archive_evidence_status", ""),
        "source_evidence_eligibility_reason": row.get("evidence_eligibility_reason", ""),
        "source_registry_status": row.get("registry_status", ""),
        "source_registry_claim_status": row.get("registry_claim_status", ""),
        "source_registry_path": row.get("registry_path", ""),
    }


def _controller_registry_from_rows(rows: list[dict[str, object]]) -> dict:
    registry = {}
    for row in rows:
        primitive_id = str(row.get("primitive_id", ""))
        if not primitive_id or primitive_id in registry:
            continue
        try:
            registry[primitive_id] = controller_from_evidence_row(row)
        except Exception:
            continue
    return registry


def _available_split_columns(rows: list[dict[str, object]]) -> list[str]:
    candidates = [
        "start_state_family",
        "state_envelope_label",
        "environment_instance_environment_id",
        "implementation_instance_implementation_instance_id",
        "plant_instance_plant_instance_id",
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


def _write_rejection_summary(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        pd.DataFrame(columns=["governor_mode", "rejection_reason", "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    summary = frame.groupby(["governor_mode", "rejection_reason"], dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_selected_summary(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        pd.DataFrame(columns=["governor_mode", "selected_primitive_id", "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    summary = frame.groupby(["governor_mode", "selected_primitive_id"], dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_split_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "split_manifest_version": "lqr_outcome_model_split_v1",
        "split_policy": "deterministic_index_modulo_70_15_15",
        "row_count": len(rows),
        "train_count": sum(1 for index, _ in enumerate(rows) if index % 20 < 14),
        "validation_count": sum(1 for index, _ in enumerate(rows) if 14 <= index % 20 < 17),
        "test_count": sum(1 for index, _ in enumerate(rows) if index % 20 >= 17),
        "stratification_fields": [
            "primitive_id",
            "controller_id",
            "W_layer",
            "latency_case",
            "start_state_family",
            "outcome_class",
        ],
    }
    filesystem_path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _write_confusion_matrices(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        pd.DataFrame(columns=["source_primitive_id", "source_W_layer", "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    frame["predicted_outcome_class"] = frame["decision_status"].map(
        lambda value: "accepted" if value == "selected_viable_primitive" else "rejected"
    )
    columns = ["source_primitive_id", "source_W_layer", "source_outcome_class", "predicted_outcome_class"]
    summary = frame.groupby(columns, dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_calibration_scores(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        pd.DataFrame(columns=["probability_name", "brier_score"]).to_csv(filesystem_path(path), index=False)
        return
    output = []
    source_outcome = frame.get("source_outcome_class", pd.Series([""] * len(frame))).astype(str)
    targets = {
        "probability_accepted": source_outcome == "accepted",
        "probability_weak": source_outcome == "weak",
        "probability_failed": source_outcome.isin(["failed", "rejected", "blocked"]),
        "probability_continuation_valid": _bool_series(
            frame.get("source_continuation_valid", frame.get("continuation_valid", pd.Series([False] * len(frame))))
        ),
        "probability_episode_terminal_useful": _bool_series(
            frame.get(
                "source_episode_terminal_useful",
                frame.get("episode_terminal_useful", pd.Series([False] * len(frame))),
            )
        ),
    }
    for name in (
        "probability_accepted",
        "probability_weak",
        "probability_failed",
        "probability_continuation_valid",
        "probability_episode_terminal_useful",
    ):
        if name not in frame.columns:
            continue
        target = targets[name].astype(float)
        pred = frame[name].astype(float)
        output.append(
            {
                "probability_name": name,
                "brier_score": float(((pred - target) ** 2).mean()),
                "mean_prediction": float(pred.mean()),
                "target_rate": float(target.mean()),
            }
        )
    pd.DataFrame(output).to_csv(filesystem_path(path), index=False)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})


def _write_regression_errors(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    metrics = [
        ("energy_residual_m", "predicted_energy_residual_m"),
        ("lift_dwell_time_s", "predicted_lift_dwell_time_s"),
        ("minimum_wall_margin_m", "predicted_minimum_wall_margin_m"),
    ]
    output = []
    for truth, pred in metrics:
        if truth not in frame.columns or pred not in frame.columns:
            output.append({"metric": truth, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "row_count": 0})
            continue
        delta = frame[pred].astype(float) - frame[truth].astype(float)
        output.append(
            {
                "metric": truth,
                "mae": float(delta.abs().mean()),
                "rmse": float((delta.pow(2).mean()) ** 0.5),
                "bias": float(delta.mean()),
                "row_count": int(len(delta)),
            }
        )
    pd.DataFrame(output).to_csv(filesystem_path(path), index=False)


def main(argv: list[str] | None = None) -> int:
    run_primitive_selector_report(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
