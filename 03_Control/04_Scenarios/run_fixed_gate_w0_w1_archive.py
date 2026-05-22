from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    filesystem_path,
    resolve_storage_format,
    table_extension,
    write_table_manifest,
    write_table_partition,
)
from fixed_gate_code_path_map import active_code_path_text, code_path_map_frame  # noqa: E402
from fixed_gate_primitive_rollout import (  # noqa: E402
    FixedGatePrimitiveRolloutConfig,
    build_archive_move_on_gates,
    build_rollout_outcome_summary,
    build_w0_w1_pairing_audit,
    run_fixed_gate_primitive_rollouts,
)
from fixed_gate_sampling import (  # noqa: E402
    FixedGateSamplingConfig,
    build_fixed_gate_w0_w1_candidate_rows,
    sample_fixed_gate_states,
    validate_w1_independent_of_w0,
)
from primitive_roles import active_mission_primitive_families  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and public runner
# 2) Sampling and candidate preparation
# 3) Output writers and manifests
# 4) CLI
# =============================================================================


CAMPAIGN = "11_fixed_gate_repeated_launch"
PASS_NAME = "fixed_gate_w0_w1_primitive_rollout_archive"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN
BRANCHES = ("single_fan_branch", "four_fan_branch")
SMOKE_SCALE_ROWS_PER_BRANCH_LIMIT = 200
DENSE_ROLLOUT_ROW_THRESHOLD = 10_000
DENSE_CANDIDATE_ROW_THRESHOLD = 5_000
DENSE_RUNTIME_MIN_THRESHOLD = 30.0
DENSE_TABLE_SIZE_MB_THRESHOLD = 250.0
SMOKE_PURPOSES = {"smoke", "readiness", "smoke_readiness"}
CHUNKED_RUNNER_NAME = "run_fixed_gate_w0_w1_archive_chunked.py"


# =============================================================================
# 1) Paths and Public Runner
# =============================================================================
def run_fixed_gate_w0_w1_archive(
    *,
    run_id: int,
    rows_per_branch: int = 40,
    seed: int = 20260521,
    fan_branch: str = "all",
    w_layers: str | tuple[str, ...] = "W0,W1",
    latency_case: str = "nominal",
    controller_mode: str = "both",
    reachable_source_csv: Path | None = None,
    result_root: Path | None = None,
    storage_format: str = "auto",
    run_purpose: str = "smoke",
    expected_runtime_min: float | None = None,
    expected_uncompressed_table_mb: float | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    selected_branches = _branches(fan_branch)
    selected_layers = _layers(w_layers)
    scale_classification = classify_fixed_gate_w0_w1_run_scale(
        rows_per_branch=int(rows_per_branch),
        branch_count=len(selected_branches),
        controller_mode=str(controller_mode),
        run_purpose=str(run_purpose),
        expected_runtime_min=expected_runtime_min,
        expected_uncompressed_table_mb=expected_uncompressed_table_mb,
    )
    _assert_simple_runner_smoke_scale(scale_classification)

    root = _archive_root(run_id, result_root)
    paths = _prepare_tree(root, overwrite=overwrite)
    reachable_sources = _read_optional_reachable_sources(reachable_source_csv)
    samples = _sample_archive_states(
        rows_per_branch=int(rows_per_branch),
        seed=int(seed),
        branches=selected_branches,
        reachable_sources=reachable_sources,
    )
    candidates = build_fixed_gate_w0_w1_candidate_rows(
        samples,
        primitive_families=active_mission_primitive_families(),
    )
    candidates = candidates[candidates["W_layer"].astype(str).isin(selected_layers)].copy()
    candidates = _limit_candidate_rows(candidates, rows_per_branch=int(rows_per_branch), branches=selected_branches)
    validate_w1_independent_of_w0(candidates) if set(selected_layers) == {"W0", "W1"} else None

    rollout_rows = run_fixed_gate_primitive_rollouts(
        candidates,
        FixedGatePrimitiveRolloutConfig(
            latency_case=str(latency_case),
            random_seed=int(seed),
            controller_mode=str(controller_mode),
            feedback_mode="instant_state_feedback" if str(controller_mode) in {"both", "feedback_stabilised_primitive"} else "open_loop",
            allow_open_loop_diagnostic=True,
        ),
    )
    pairing_audit = build_w0_w1_pairing_audit(candidates, rollout_rows)
    outcome_summary = build_rollout_outcome_summary(rollout_rows)
    move_on_gates = build_archive_move_on_gates(rollout_rows)
    code_path_map = code_path_map_frame()

    output_paths = _write_outputs(
        paths=paths,
        run_id=int(run_id),
        storage_format=storage_format,
        samples=samples,
        candidates=candidates,
        rollout_rows=rollout_rows,
        pairing_audit=pairing_audit,
        outcome_summary=outcome_summary,
        code_path_map=code_path_map,
        move_on_gates=move_on_gates,
    )
    manifest = _manifest(
        run_id=int(run_id),
        seed=int(seed),
        rows_per_branch=int(rows_per_branch),
        fan_branch=fan_branch,
        selected_layers=selected_layers,
        latency_case=latency_case,
        controller_mode=controller_mode,
        run_purpose=run_purpose,
        scale_classification=scale_classification,
        samples=samples,
        candidates=candidates,
        rollout_rows=rollout_rows,
        pairing_audit=pairing_audit,
        outcome_summary=outcome_summary,
        move_on_gates=move_on_gates,
        reachable_source_csv=reachable_source_csv,
        output_paths=output_paths,
    )
    output_paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    output_paths["report_md"].write_text(_report(manifest), encoding="ascii")
    return output_paths


def run_fixed_gate_w0_w1_archive_scaffold(
    *,
    run_id: int,
    sample_count_per_branch: int = 20,
    result_root: Path | None = None,
    storage_format: str = "auto",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Compatibility wrapper for callers that used the v11 scaffold name."""

    return run_fixed_gate_w0_w1_archive(
        run_id=run_id,
        rows_per_branch=sample_count_per_branch,
        result_root=result_root,
        storage_format=storage_format,
        run_purpose="smoke",
        overwrite=overwrite,
    )


def classify_fixed_gate_w0_w1_run_scale(
    *,
    rows_per_branch: int,
    branch_count: int,
    controller_mode: str,
    run_purpose: str = "smoke",
    expected_runtime_min: float | None = None,
    expected_uncompressed_table_mb: float | None = None,
) -> dict[str, object]:
    """Classify whether the simple fixed-gate runner may execute.

    The legacy runner is retained for smoke/readiness checks only. Dense,
    archive, or thesis-scale work must use the chunked runner so it cannot
    silently accumulate full rollout tables in memory.
    """

    branch_count = max(1, int(branch_count))
    planned_candidate_rows = int(rows_per_branch) * branch_count
    rollout_multiplier = 2 if str(controller_mode) == "both" else 1
    planned_rollout_rows = planned_candidate_rows * rollout_multiplier
    reasons: list[str] = []
    if int(rows_per_branch) > SMOKE_SCALE_ROWS_PER_BRANCH_LIMIT:
        reasons.append(f"rows_per_branch>{SMOKE_SCALE_ROWS_PER_BRANCH_LIMIT}")
    if planned_rollout_rows >= DENSE_ROLLOUT_ROW_THRESHOLD:
        reasons.append(f"planned_rollout_rows>={DENSE_ROLLOUT_ROW_THRESHOLD}")
    if planned_candidate_rows >= DENSE_CANDIDATE_ROW_THRESHOLD:
        reasons.append(f"planned_candidate_rows>={DENSE_CANDIDATE_ROW_THRESHOLD}")
    if expected_runtime_min is not None and float(expected_runtime_min) > DENSE_RUNTIME_MIN_THRESHOLD:
        reasons.append(f"expected_runtime_min>{DENSE_RUNTIME_MIN_THRESHOLD:g}")
    if expected_uncompressed_table_mb is not None and float(expected_uncompressed_table_mb) > DENSE_TABLE_SIZE_MB_THRESHOLD:
        reasons.append(f"expected_uncompressed_table_mb>{DENSE_TABLE_SIZE_MB_THRESHOLD:g}")
    if str(run_purpose).strip().lower() not in SMOKE_PURPOSES:
        reasons.append(f"run_purpose={run_purpose}")
    return {
        "runner": "simple_smoke_runner",
        "run_purpose": str(run_purpose),
        "rows_per_branch": int(rows_per_branch),
        "branch_count": int(branch_count),
        "controller_mode": str(controller_mode),
        "planned_candidate_rows": int(planned_candidate_rows),
        "planned_rollout_rows": int(planned_rollout_rows),
        "expected_runtime_min": None if expected_runtime_min is None else float(expected_runtime_min),
        "expected_uncompressed_table_mb": (
            None if expected_uncompressed_table_mb is None else float(expected_uncompressed_table_mb)
        ),
        "dense_threshold_reasons": reasons,
        "scale_class": "smoke_scale" if not reasons else "dense_archive_or_thesis_scale",
        "allowed_in_simple_runner": not reasons,
        "handoff_runner": CHUNKED_RUNNER_NAME,
    }


def _assert_simple_runner_smoke_scale(scale_classification: dict[str, object]) -> None:
    if bool(scale_classification["allowed_in_simple_runner"]):
        return
    reasons = ", ".join(str(reason) for reason in scale_classification["dense_threshold_reasons"])
    raise RuntimeError(
        "run_fixed_gate_w0_w1_archive.py is smoke-scale only and refused this dense/archive/thesis-scale request "
        f"({reasons}). Use {CHUNKED_RUNNER_NAME} for chunked, resumable, compressed, branch-local execution."
    )


# =============================================================================
# 2) Sampling and Candidate Preparation
# =============================================================================
def _sample_archive_states(
    *,
    rows_per_branch: int,
    seed: int,
    branches: tuple[str, ...],
    reachable_sources: pd.DataFrame | None,
) -> pd.DataFrame:
    primitive_count = max(1, len(active_mission_primitive_families()))
    # W0/W1 are paired per sample/family, so choose enough samples to satisfy
    # the requested approximate branch row count without dropping W1 rows.
    sample_count = max(1, int(np.ceil(int(rows_per_branch) / (2.0 * primitive_count))))
    frames = []
    for branch_index, branch in enumerate(branches):
        frames.append(
            sample_fixed_gate_states(
                FixedGateSamplingConfig(
                    total_count=sample_count,
                    random_seed=int(seed) + int(branch_index),
                ),
                fan_branch=branch,
                W_layer="W1",
                reachable_source_rows=reachable_sources,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _limit_candidate_rows(candidates: pd.DataFrame, *, rows_per_branch: int, branches: tuple[str, ...]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    selected = []
    for branch in branches:
        branch_rows = candidates[candidates["fan_branch"].astype(str).eq(branch)].copy()
        if branch_rows.empty:
            continue
        # Keep complete paired W0/W1 sample/family groups rather than slicing
        # through the W-ladder and accidentally making W1 dependent on W0.
        group_keys = ["paired_sample_key", "primitive_family"]
        ordered_groups = list(branch_rows.groupby(group_keys, sort=True, dropna=False))
        rows: list[pd.DataFrame] = []
        for _, group in ordered_groups:
            rows.append(group)
            if sum(len(item) for item in rows) >= int(rows_per_branch):
                break
        selected.append(pd.concat(rows, ignore_index=True))
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=candidates.columns)


def _read_optional_reachable_sources(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def _branches(value: str) -> tuple[str, ...]:
    if str(value) == "all":
        return BRANCHES
    if str(value) not in BRANCHES:
        raise ValueError("fan_branch must be single_fan_branch, four_fan_branch, or all.")
    return (str(value),)


def _layers(value: str | tuple[str, ...]) -> tuple[str, ...]:
    items = value if isinstance(value, tuple) else tuple(item.strip() for item in str(value).split(",") if item.strip())
    invalid = sorted(set(items).difference({"W0", "W1"}))
    if invalid:
        raise ValueError(f"w_layers for the W0/W1 archive must contain only W0/W1, got {invalid}.")
    return tuple(items)


# =============================================================================
# 3) Output Writers and Manifests
# =============================================================================
def _archive_root(run_id: int, result_root: Path | None) -> Path:
    return (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "003_fixed_gate_w0_w1_proof_archive"


def _prepare_tree(root: Path, *, overwrite: bool) -> dict[str, Path]:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    paths = {
        "root": root,
        "tables": root / "tables",
        "metrics": root / "metrics",
        "manifests": root / "manifests",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_outputs(
    *,
    paths: dict[str, Path],
    run_id: int,
    storage_format: str,
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    rollout_rows: pd.DataFrame,
    pairing_audit: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    code_path_map: pd.DataFrame,
    move_on_gates: dict[str, object],
) -> dict[str, Path]:
    fmt = resolve_storage_format(storage_format)
    extension = table_extension(fmt)
    branch_coverage = _branch_coverage_summary(rollout_rows, move_on_gates)
    partitions = (
        write_table_partition(samples, paths["tables"] / "fixed_gate_samples" / f"part-00000.{extension}", storage_format=fmt),
        write_table_partition(candidates, paths["tables"] / "candidate_index" / f"part-00000.{extension}", storage_format=fmt),
        write_table_partition(rollout_rows, paths["tables"] / "primitive_rollout_rows" / f"part-00000.{extension}", storage_format=fmt),
    )
    table_manifest = paths["manifests"] / "table_manifest.json"
    write_table_manifest(
        table_manifest,
        TableManifest(run_id=int(run_id), root=paths["root"].as_posix(), storage_format=fmt, tables=partitions),
    )

    output_paths = {
        "root": paths["root"],
        "table_manifest_json": table_manifest,
        "samples_csv": paths["metrics"] / "fixed_gate_samples.csv",
        "candidate_index_csv": paths["metrics"] / "fixed_gate_w0_w1_candidate_index.csv",
        "rollout_rows_csv": paths["metrics"] / "fixed_gate_w0_w1_primitive_rollout_rows.csv",
        "diagnostic_rows_csv": paths["metrics"] / "fixed_gate_w0_w1_diagnostic_rows.csv",
        "partial_feedback_rows_csv": paths["metrics"] / "fixed_gate_w0_w1_partial_feedback_rows.csv",
        "mission_candidate_rows_csv": paths["metrics"] / "fixed_gate_w0_w1_mission_candidate_rows.csv",
        "pairing_audit_csv": paths["metrics"] / "fixed_gate_w0_w1_pairing_audit.csv",
        "outcome_summary_csv": paths["metrics"] / "fixed_gate_w0_w1_outcome_summary.csv",
        "branch_coverage_summary_csv": paths["metrics"] / "fixed_gate_w0_w1_branch_coverage_summary.csv",
        "code_path_map_csv": paths["metrics"] / "active_deprecated_code_path_map.csv",
        "manifest_json": paths["manifests"] / "fixed_gate_w0_w1_archive_manifest.json",
        "report_md": paths["reports"] / "fixed_gate_w0_w1_archive_report.md",
        "branch_coverage_report_md": paths["reports"] / "fixed_gate_w0_w1_branch_coverage_report.md",
    }
    samples.to_csv(filesystem_path(output_paths["samples_csv"]), index=False)
    candidates.to_csv(filesystem_path(output_paths["candidate_index_csv"]), index=False)
    rollout_rows.to_csv(filesystem_path(output_paths["rollout_rows_csv"]), index=False)
    _role_rows(rollout_rows, {"ablation_diagnostic", "boundary_diagnostic"}).to_csv(
        filesystem_path(output_paths["diagnostic_rows_csv"]),
        index=False,
    )
    _role_rows(rollout_rows, {"partial_feedback", "blocked_partial"}).to_csv(
        filesystem_path(output_paths["partial_feedback_rows_csv"]),
        index=False,
    )
    _role_rows(rollout_rows, {"mission_candidate"}).to_csv(
        filesystem_path(output_paths["mission_candidate_rows_csv"]),
        index=False,
    )
    pairing_audit.to_csv(filesystem_path(output_paths["pairing_audit_csv"]), index=False)
    outcome_summary.to_csv(filesystem_path(output_paths["outcome_summary_csv"]), index=False)
    branch_coverage.to_csv(filesystem_path(output_paths["branch_coverage_summary_csv"]), index=False)
    code_path_map.to_csv(filesystem_path(output_paths["code_path_map_csv"]), index=False)
    output_paths["branch_coverage_report_md"].write_text(
        _branch_coverage_report(branch_coverage, move_on_gates),
        encoding="ascii",
    )
    return output_paths


def _role_rows(frame: pd.DataFrame, roles: set[str]) -> pd.DataFrame:
    if frame.empty or "evidence_role" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame["evidence_role"].astype(str).isin(roles)].copy()


def _branch_coverage_summary(rollout_rows: pd.DataFrame, move_on_gates: dict[str, object]) -> pd.DataFrame:
    """Return compact v11.2 branch-coverage evidence diagnostics.

    The summary is deliberately descriptive only. It reports where accepted
    fixed-gate partial-feedback evidence exists and where rows remain weak or
    rejected, without changing acceptance checks or promoting diagnostic rows.
    """

    columns = (
        "summary_section",
        "fan_branch",
        "W_layer",
        "primitive_family",
        "failure_label",
        "evidence_role",
        "accepted",
        "row_count",
        "minimum_margin_min_m",
        "minimum_margin_mean_m",
        "energy_residual_min_m",
        "energy_residual_mean_m",
        "energy_residual_max_m",
        "archive_prepared_status",
        "mission_evidence_ready_status",
    )
    if rollout_rows.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    frame = rollout_rows.copy()
    accepted = frame["accepted"].astype(bool) if "accepted" in frame.columns else pd.Series(False, index=frame.index)

    rows.extend(
        _count_section(
            frame,
            "row_count_by_branch_layer",
            ["fan_branch", "W_layer"],
            move_on_gates,
        )
    )
    w1_measured = frame[
        frame["W_layer"].astype(str).eq("W1")
        & frame["wind_mode"].astype(str).ne("none")
        & frame["wind_descriptor_status"].astype(str).eq("wind_model_evaluated")
        & ~frame["wind_descriptor_model_source"].astype(str).str.contains("dry_air", na=False)
    ].copy()
    rows.extend(
        _count_section(
            w1_measured,
            "non_dry_w1_measured_updraft_by_branch",
            ["fan_branch", "W_layer"],
            move_on_gates,
        )
    )
    accepted_partial = frame[
        accepted
        & frame["evidence_role"].astype(str).eq("partial_feedback")
    ].copy()
    rows.extend(
        _count_section(
            accepted_partial,
            "accepted_partial_feedback_by_branch_layer_primitive",
            ["fan_branch", "W_layer", "primitive_family"],
            move_on_gates,
        )
    )
    rejected = frame[~accepted].copy()
    rows.extend(
        _count_section(
            rejected,
            "rejection_failure_counts_by_branch_primitive",
            ["fan_branch", "primitive_family", "failure_label", "evidence_role"],
            move_on_gates,
        )
    )
    rows.extend(
        _metric_section(
            accepted_partial,
            "accepted_partial_feedback_margin_energy_summary",
            ["fan_branch", "W_layer", "primitive_family"],
            move_on_gates,
        )
    )
    weak = frame[
        ~accepted
        & frame["evidence_role"].astype(str).eq("partial_feedback")
        & pd.to_numeric(frame["minimum_margin_m"], errors="coerce").notna()
    ].copy()
    rows.extend(
        _metric_section(
            weak,
            "weak_partial_feedback_margin_energy_summary",
            ["fan_branch", "W_layer", "primitive_family", "failure_label"],
            move_on_gates,
        )
    )
    rows.append(
        {
            "summary_section": "readiness_status",
            "fan_branch": "all",
            "W_layer": "W0,W1",
            "primitive_family": "all",
            "failure_label": "",
            "evidence_role": "",
            "accepted": "",
            "row_count": int(len(frame)),
            "minimum_margin_min_m": np.nan,
            "minimum_margin_mean_m": np.nan,
            "energy_residual_min_m": np.nan,
            "energy_residual_mean_m": np.nan,
            "energy_residual_max_m": np.nan,
            "archive_prepared_status": str(move_on_gates.get("archive_prepared", "")),
            "mission_evidence_ready_status": str(move_on_gates.get("mission_evidence_ready", "")),
        }
    )
    return pd.DataFrame(rows, columns=columns)


def _count_section(
    frame: pd.DataFrame,
    section: str,
    group_columns: list[str],
    move_on_gates: dict[str, object],
) -> list[dict[str, object]]:
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, dropna=False).size().reset_index(name="row_count")
    for record in grouped.to_dict(orient="records"):
        rows.append(
            {
                "summary_section": section,
                "fan_branch": str(record.get("fan_branch", "")),
                "W_layer": str(record.get("W_layer", "")),
                "primitive_family": str(record.get("primitive_family", "")),
                "failure_label": str(record.get("failure_label", "")),
                "evidence_role": str(record.get("evidence_role", "")),
                "accepted": str(record.get("accepted", "")),
                "row_count": int(record["row_count"]),
                "minimum_margin_min_m": np.nan,
                "minimum_margin_mean_m": np.nan,
                "energy_residual_min_m": np.nan,
                "energy_residual_mean_m": np.nan,
                "energy_residual_max_m": np.nan,
                "archive_prepared_status": str(move_on_gates.get("archive_prepared", "")),
                "mission_evidence_ready_status": str(move_on_gates.get("mission_evidence_ready", "")),
            }
        )
    return rows


def _metric_section(
    frame: pd.DataFrame,
    section: str,
    group_columns: list[str],
    move_on_gates: dict[str, object],
) -> list[dict[str, object]]:
    if frame.empty:
        return []
    metric_frame = frame.copy()
    metric_frame["minimum_margin_m"] = pd.to_numeric(metric_frame["minimum_margin_m"], errors="coerce")
    metric_frame["energy_residual_m"] = pd.to_numeric(metric_frame["energy_residual_m"], errors="coerce")
    grouped = (
        metric_frame.groupby(group_columns, dropna=False)
        .agg(
            row_count=("sample_id", "size"),
            minimum_margin_min_m=("minimum_margin_m", "min"),
            minimum_margin_mean_m=("minimum_margin_m", "mean"),
            energy_residual_min_m=("energy_residual_m", "min"),
            energy_residual_mean_m=("energy_residual_m", "mean"),
            energy_residual_max_m=("energy_residual_m", "max"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for record in grouped.to_dict(orient="records"):
        rows.append(
            {
                "summary_section": section,
                "fan_branch": str(record.get("fan_branch", "")),
                "W_layer": str(record.get("W_layer", "")),
                "primitive_family": str(record.get("primitive_family", "")),
                "failure_label": str(record.get("failure_label", "")),
                "evidence_role": str(record.get("evidence_role", "")),
                "accepted": str(record.get("accepted", "")),
                "row_count": int(record["row_count"]),
                "minimum_margin_min_m": _finite_or_nan(record["minimum_margin_min_m"]),
                "minimum_margin_mean_m": _finite_or_nan(record["minimum_margin_mean_m"]),
                "energy_residual_min_m": _finite_or_nan(record["energy_residual_min_m"]),
                "energy_residual_mean_m": _finite_or_nan(record["energy_residual_mean_m"]),
                "energy_residual_max_m": _finite_or_nan(record["energy_residual_max_m"]),
                "archive_prepared_status": str(move_on_gates.get("archive_prepared", "")),
                "mission_evidence_ready_status": str(move_on_gates.get("mission_evidence_ready", "")),
            }
        )
    return rows


def _branch_coverage_report(summary: pd.DataFrame, move_on_gates: dict[str, object]) -> str:
    accepted = summary[
        summary["summary_section"].astype(str).eq("accepted_partial_feedback_by_branch_layer_primitive")
    ].copy()
    weak = summary[
        summary["summary_section"].astype(str).eq("weak_partial_feedback_margin_energy_summary")
    ].copy()
    return "\n".join(
        [
            "# Fixed-Gate W0/W1 Branch-Coverage Diagnosis",
            "",
            f"- Archive-prepared status: `{move_on_gates.get('archive_prepared', '')}`",
            f"- Mission-evidence-ready status: `{move_on_gates.get('mission_evidence_ready', '')}`",
            f"- W0 rows by branch: `{move_on_gates.get('w0_row_count_by_branch', {})}`",
            f"- W1 rows by branch: `{move_on_gates.get('w1_row_count_by_branch', {})}`",
            f"- Non-dry W1 measured-updraft rows by branch: `{move_on_gates.get('w1_measured_updraft_row_count_by_branch', {})}`",
            f"- Accepted W0 partial-feedback rows: `{move_on_gates.get('accepted_w0_partial_feedback_row_count', 0)}`",
            f"- Accepted W1 partial-feedback rows: `{move_on_gates.get('accepted_w1_partial_feedback_row_count', 0)}`",
            f"- Accepted branch/layer/primitive groups: `{len(accepted)}`",
            f"- Weak partial-feedback groups: `{len(weak)}`",
            "",
            "Weak rows are non-accepted partial-feedback rows with finite true-margin metrics.",
            "This report is diagnostic only and does not loosen entry, exit, safety, latency, wind, or governor-compatible checks.",
            "",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, hardware-ready agile-turn, or full delayed-state-feedback claim is made.",
            "",
        ]
    )


def _finite_or_nan(value: object) -> float:
    number = float(value)
    return number if np.isfinite(number) else float("nan")


def _manifest(
    *,
    run_id: int,
    seed: int,
    rows_per_branch: int,
    fan_branch: str,
    selected_layers: tuple[str, ...],
    latency_case: str,
    controller_mode: str,
    run_purpose: str,
    scale_classification: dict[str, object],
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    rollout_rows: pd.DataFrame,
    pairing_audit: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    move_on_gates: dict[str, object],
    reachable_source_csv: Path | None,
    output_paths: dict[str, Path],
) -> dict[str, object]:
    del outcome_summary
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "active_mission_path": active_code_path_text(),
        "seed": int(seed),
        "rows_per_branch_requested": int(rows_per_branch),
        "fan_branch": str(fan_branch),
        "w_layers": list(selected_layers),
        "latency_case": str(latency_case),
        "controller_mode": str(controller_mode),
        "run_purpose": str(run_purpose),
        "scale_classification": scale_classification,
        "reachable_source_csv": "" if reachable_source_csv is None else str(reachable_source_csv),
        "sample_row_count": int(len(samples)),
        "candidate_row_count": int(len(candidates)),
        "rollout_row_count": int(len(rollout_rows)),
        "pairing_audit_row_count": int(len(pairing_audit)),
        "W1_independent_of_W0_success": bool(pairing_audit["w1_scheduled_independent_of_w0_success"].all()) if not pairing_audit.empty else False,
        "branch_local_separation": True,
        "rollout_execution_performed": True,
        "mission_feedback_path_status": move_on_gates["feedback_path_status"],
        "feedback_stabilised_primitive_available": bool(move_on_gates["partial_feedback_row_count"] > 0),
        "w0_row_count_by_branch": move_on_gates["w0_row_count_by_branch"],
        "w1_row_count_by_branch": move_on_gates["w1_row_count_by_branch"],
        "w1_measured_updraft_row_count": move_on_gates["w1_measured_updraft_row_count"],
        "w1_measured_updraft_row_count_by_branch": move_on_gates["w1_measured_updraft_row_count_by_branch"],
        "accepted_w0_partial_feedback_row_count": move_on_gates["accepted_w0_partial_feedback_row_count"],
        "accepted_w1_partial_feedback_row_count": move_on_gates["accepted_w1_partial_feedback_row_count"],
        "open_loop_rows_promoted_to_mission_candidate": False,
        "open_loop_rows_promoted_to_partial_feedback": False,
        "move_on_gates": move_on_gates,
        "code_ready_status": move_on_gates["code_ready"],
        "archive_prepared_status": move_on_gates["archive_prepared"],
        "mission_evidence_ready_status": move_on_gates["mission_evidence_ready"],
        "archive_readiness_status": move_on_gates["archive_prepared"],
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Fixed-gate W0/W1 evidence execution with explicit diagnostic/blocked hierarchy; "
            "no real-flight transfer, mission success, same-flight recapture, perching, "
            "all-arena validity, or hardware-ready agile claim is made."
        ),
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }


def _report(manifest: dict[str, object]) -> str:
    gates = manifest["move_on_gates"]
    return "\n".join(
        [
            "# Fixed-Gate W0/W1 Primitive Rollout Archive",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            f"- Sample rows: `{manifest['sample_row_count']}`",
            f"- Candidate rows: `{manifest['candidate_row_count']}`",
            f"- Rollout rows: `{manifest['rollout_row_count']}`",
            f"- Mission-candidate rows: `{gates['mission_candidate_row_count']}`",
            f"- Partial-feedback rows: `{gates['partial_feedback_row_count']}`",
            f"- Accepted W0 partial-feedback rows: `{gates['accepted_w0_partial_feedback_row_count']}`",
            f"- Accepted W1 partial-feedback rows: `{gates['accepted_w1_partial_feedback_row_count']}`",
            f"- Blocked-partial rows: `{gates['blocked_partial_row_count']}`",
            f"- Diagnostic open-loop rows: `{gates['ablation_diagnostic_row_count']}`",
            f"- W0 rows by branch: `{gates['w0_row_count_by_branch']}`",
            f"- W1 rows by branch: `{gates['w1_row_count_by_branch']}`",
            f"- Non-dry W1 measured-updraft rows: `{gates['w1_measured_updraft_row_count']}`",
            f"- W1 measured-updraft rows by branch: `{gates['w1_measured_updraft_row_count_by_branch']}`",
            f"- Code-ready status: `{manifest['code_ready_status']}`",
            f"- Archive-prepared status: `{manifest['archive_prepared_status']}`",
            f"- Mission-evidence-ready status: `{manifest['mission_evidence_ready_status']}`",
            f"- Feedback path status: `{manifest['mission_feedback_path_status']}`",
            "- W1 remains scheduled independently of W0 success.",
            "",
            "Open-loop rollout rows are ablation diagnostics only and are not governor-package or mission-facing evidence.",
            "",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


# =============================================================================
# 4) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--rows-per-branch", type=int, default=40)
    parser.add_argument("--sample-count-per-branch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--fan-branch", choices=("single_fan_branch", "four_fan_branch", "all"), default="all")
    parser.add_argument("--w-layers", default="W0,W1")
    parser.add_argument("--latency-case", choices=("none", "actuator_lag_only", "nominal", "conservative"), default="nominal")
    parser.add_argument("--controller-mode", choices=("open_loop_rollout", "feedback_stabilised_primitive", "both"), default="both")
    parser.add_argument("--reachable-source-csv", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--run-purpose", default="smoke")
    parser.add_argument("--expected-runtime-min", type=float, default=None)
    parser.add_argument("--expected-uncompressed-table-mb", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows_per_branch = args.rows_per_branch if args.sample_count_per_branch is None else args.sample_count_per_branch
    run_fixed_gate_w0_w1_archive(
        run_id=args.run_id,
        rows_per_branch=rows_per_branch,
        seed=args.seed,
        fan_branch=args.fan_branch,
        w_layers=args.w_layers,
        latency_case=args.latency_case,
        controller_mode=args.controller_mode,
        reachable_source_csv=args.reachable_source_csv,
        result_root=args.result_root,
        storage_format=args.storage_format,
        run_purpose=args.run_purpose,
        expected_runtime_min=args.expected_runtime_min,
        expected_uncompressed_table_mb=args.expected_uncompressed_table_mb,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
