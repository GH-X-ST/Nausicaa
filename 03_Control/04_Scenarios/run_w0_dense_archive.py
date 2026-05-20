from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_cluster_diagnostics import (  # noqa: E402
    DenseClusterDiagnosticConfig,
    build_cluster_diagnostics,
)
from dense_archive_clustering import select_cluster_representatives  # noqa: E402
from dense_archive_envelope_maps import build_envelope_map  # noqa: E402
from dense_archive_schema import BRANCH_DECISION_SCOPE, CAMPAIGN  # noqa: E402
from run_dense_archive_pilot_sweep import (  # noqa: E402
    _check_planning_inputs_exist,
    _run_pilot_replays,
    _same_or_contained,
    _target_sort_value,
    _text,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths, Constants, and Data Containers
# 2) Planning Loading and W0 Selection
# 3) Guardrails and Scale Semantics
# 4) Output Writers
# 5) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths, Constants, and Data Containers
# =============================================================================
W0_CAMPAIGN = "11_w0_dense_archive"
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / W0_CAMPAIGN
PLANNING_CAMPAIGN = CAMPAIGN
ARCHIVE_SCALE_MODES = frozenset({"strict", "reduced"})
W0_ENVIRONMENT_MODES = ("W0_single_fan_branch", "W0_four_fan_branch")
NO_CLAIM_TEXT = (
    "Mon 25 W0 dense archive execution only; no W1 production archive, "
    "W2/W3/W4/W5 evidence, mission evaluation, hardware validation, or "
    "sim-to-real transfer is claimed."
)


@dataclass(frozen=True)
class W0DenseArchiveConfig:
    run_id: int = 12
    planning_run_id: int = 10
    max_trials: int = 500000
    floor_trials_per_branch: int = 150000
    target_trials_per_branch: int = 250000
    archive_scale_mode: str = "strict"
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    random_seed: int = 20260525
    result_root: Path | None = None
    overwrite: bool = False
    write_trial_logs: bool = False
    branch_selection_rule: str = "equal_w0_branch_quota"
    environment_selection_rule: str = "w0_only_paired_branch_baselines"

    @property
    def floor_trials_total(self) -> int:
        return 2 * int(self.floor_trials_per_branch)

    @property
    def target_trials_total(self) -> int:
        return 2 * int(self.target_trials_per_branch)


@dataclass(frozen=True)
class W0DenseArchiveOutputs:
    root: Path
    trial_descriptors_csv: Path
    envelope_map_csv: Path
    cluster_representatives_csv: Path
    cluster_diagnostics_csv: Path
    manifest_json: Path
    report_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "trial_descriptors_csv": self.trial_descriptors_csv,
            "envelope_map_csv": self.envelope_map_csv,
            "cluster_representatives_csv": self.cluster_representatives_csv,
            "cluster_diagnostics_csv": self.cluster_diagnostics_csv,
            "manifest_json": self.manifest_json,
            "report_md": self.report_md,
        }


def _active_result_root(config: W0DenseArchiveConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _planning_result_root(config: W0DenseArchiveConfig) -> Path:
    return _active_result_root(config).parent / PLANNING_CAMPAIGN


def _output_paths(config: W0DenseArchiveConfig) -> W0DenseArchiveOutputs:
    root = _active_result_root(config) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    return W0DenseArchiveOutputs(
        root=root,
        trial_descriptors_csv=root / "metrics" / f"w0_dense_trial_descriptors_{suffix}.csv",
        envelope_map_csv=root / "metrics" / f"w0_dense_envelope_map_{suffix}.csv",
        cluster_representatives_csv=root
        / "metrics"
        / f"w0_dense_cluster_representatives_{suffix}.csv",
        cluster_diagnostics_csv=root
        / "metrics"
        / f"w0_dense_cluster_diagnostics_{suffix}.csv",
        manifest_json=root / "manifests" / f"w0_dense_archive_manifest_{suffix}.json",
        report_md=root / "reports" / f"w0_dense_archive_report_{suffix}.md",
    )


def _planning_paths(config: W0DenseArchiveConfig) -> tuple[Path, Path]:
    root = _planning_result_root(config) / f"{int(config.planning_run_id):03d}" / "metrics"
    suffix = f"s{int(config.planning_run_id):03d}"
    return (
        root / f"equal_branch_start_state_manifest_pilot_{suffix}.csv",
        root / f"equal_branch_dry_run_candidate_inventory_pilot_{suffix}.csv",
    )


def _planning_run_root(config: W0DenseArchiveConfig) -> Path:
    return _planning_result_root(config) / f"{int(config.planning_run_id):03d}"


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Planning Loading and W0 Selection
# =============================================================================
def _load_planning_tables(config: W0DenseArchiveConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_path, candidate_path = _planning_paths(config)
    _check_planning_inputs_exist(start_path, candidate_path)
    return pd.read_csv(start_path), pd.read_csv(candidate_path)


def _filter_w0_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    return candidates[
        candidates["test_environment_mode"].astype(str).isin(W0_ENVIRONMENT_MODES)
    ].copy()


def _select_w0_candidates(
    w0_candidates: pd.DataFrame,
    config: W0DenseArchiveConfig,
) -> list[dict[str, object]]:
    if int(config.max_trials) <= 0 or w0_candidates.empty:
        return []
    branch_quota = max(0, int(config.max_trials) // len(W0_ENVIRONMENT_MODES))
    rows: list[dict[str, object]] = []
    for mode in W0_ENVIRONMENT_MODES:
        branch_rows = w0_candidates[
            w0_candidates["test_environment_mode"].astype(str).eq(mode)
        ].to_dict(orient="records")
        rows.extend(sorted(branch_rows, key=_w0_selection_key)[:branch_quota])
    return sorted(rows, key=_w0_selection_key)


def _w0_selection_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        _text(row.get("layout_branch_id", "")),
        _text(row.get("test_environment_mode", "")),
        _text(row.get("family", "")),
        _target_sort_value(row.get("target_heading_deg", "")),
        _direction_int(row.get("direction_sign", 1)),
        _text(row.get("start_class", "")),
        _text(row.get("candidate_id", "")),
    )


def _count_by_branch(rows: pd.DataFrame | list[dict[str, object]]) -> dict[str, int]:
    if isinstance(rows, list):
        values = [_text(row.get("layout_branch_id", "")) for row in rows]
        return {branch: int(values.count(branch)) for branch in sorted(set(values))}
    if rows.empty or "layout_branch_id" not in rows.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in rows["layout_branch_id"].value_counts().sort_index().to_dict().items()
    }


# =============================================================================
# 3) Guardrails and Scale Semantics
# =============================================================================
def _validate_config(config: W0DenseArchiveConfig) -> None:
    if str(config.archive_scale_mode) not in ARCHIVE_SCALE_MODES:
        raise ValueError("archive_scale_mode must be one of: strict, reduced.")
    if int(config.max_trials) < 0:
        raise ValueError("max_trials must be nonnegative.")
    for name, value in (
        ("floor_trials_per_branch", config.floor_trials_per_branch),
        ("target_trials_per_branch", config.target_trials_per_branch),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(config.target_trials_per_branch) < int(config.floor_trials_per_branch):
        raise ValueError("target_trials_per_branch must be at least floor_trials_per_branch.")
    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive.")
    if not np.isfinite(float(config.horizon_s)) or float(config.horizon_s) <= 0.0:
        raise ValueError("horizon_s must be finite and positive.")


def _validate_output_guardrails(
    config: W0DenseArchiveConfig,
    outputs: W0DenseArchiveOutputs,
) -> None:
    output_root = outputs.root.resolve()
    planning_root = _planning_run_root(config).resolve()
    if _same_or_contained(output_root, planning_root) or _same_or_contained(
        planning_root,
        output_root,
    ):
        raise ValueError(
            "refusing output/planning overlap after path resolution: "
            f"output_root={output_root}, planning_root={planning_root}"
        )


def _same_or_contained(path: Path, container: Path) -> bool:
    return path == container or container in path.parents


def _check_output_available(outputs: W0DenseArchiveOutputs, overwrite: bool) -> None:
    if outputs.root.exists() and not bool(overwrite):
        raise ValueError(
            f"output directory already exists and overwrite=False: {outputs.root}"
        )


def _w0_scale_status(
    *,
    selected_by_branch: dict[str, int],
    config: W0DenseArchiveConfig,
    reduced_mode: bool = False,
) -> str:
    selected_total = int(sum(selected_by_branch.values()))
    if selected_total == 0:
        return "no_trials_selected"
    branch_floor_met = all(
        int(selected_by_branch.get(branch, 0)) >= int(config.floor_trials_per_branch)
        for branch in ("single_fan_branch", "four_fan_branch")
    )
    branch_target_met = all(
        int(selected_by_branch.get(branch, 0)) >= int(config.target_trials_per_branch)
        for branch in ("single_fan_branch", "four_fan_branch")
    )
    if branch_target_met and selected_total >= int(config.target_trials_total):
        return "meets_user_500k_target"
    if branch_floor_met and selected_total >= int(config.floor_trials_total):
        return "meets_w0_floor_below_user_target"
    if reduced_mode:
        return "reduced_below_w0_floor"
    return "below_w0_floor"


def _enforce_scale_gate(
    config: W0DenseArchiveConfig,
    status: str,
) -> None:
    if str(config.archive_scale_mode) == "strict" and status != "meets_user_500k_target":
        raise RuntimeError(
            "strict W0 archive scale gate rejected before output creation: "
            f"w0_scale_status={status}, floor_trials_per_branch={config.floor_trials_per_branch}, "
            f"target_trials_per_branch={config.target_trials_per_branch}"
        )


# =============================================================================
# 4) Output Writers
# =============================================================================
def _prepare_output_tree(outputs: W0DenseArchiveOutputs, overwrite: bool) -> None:
    _check_output_available(outputs, overwrite)
    if outputs.root.exists() and overwrite:
        _clear_output_tree(outputs.root)
    for path in (
        outputs.trial_descriptors_csv.parent,
        outputs.manifest_json.parent,
        outputs.report_md.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clear_output_tree(root: Path) -> None:
    root_resolved = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        resolved = path.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise RuntimeError(f"refusing to clear path outside output root: {path}")
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir() and path != root:
            try:
                path.rmdir()
            except OSError:
                pass


def _manifest(
    config: W0DenseArchiveConfig,
    outputs: W0DenseArchiveOutputs,
    *,
    candidate_rows_available_total: int,
    available_by_branch: dict[str, int],
    selected_by_branch: dict[str, int],
    trial_count: int,
    w0_scale_status: str,
    clustering_strategy_status: str,
) -> dict[str, object]:
    reduced = str(config.archive_scale_mode) == "reduced" and w0_scale_status != "meets_user_500k_target"
    return {
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "candidate_rows_available_total": int(candidate_rows_available_total),
        "available_w0_rows_total": int(sum(available_by_branch.values())),
        "available_w0_rows_by_branch": dict(available_by_branch),
        "selected_w0_rows_total": int(sum(selected_by_branch.values())),
        "selected_w0_rows_by_branch": dict(selected_by_branch),
        "selected_trial_count": int(sum(selected_by_branch.values())),
        "trial_count_executed": int(trial_count),
        "floor_trials_per_branch": int(config.floor_trials_per_branch),
        "floor_trials_total": int(config.floor_trials_total),
        "target_trials_per_branch": int(config.target_trials_per_branch),
        "target_trials_total": int(config.target_trials_total),
        "max_trials_requested": int(config.max_trials),
        "archive_scale_mode": str(config.archive_scale_mode),
        "w0_scale_status": str(w0_scale_status),
        "latency_case": str(config.latency_case),
        "dt_s": float(config.dt_s),
        "horizon_s": float(config.horizon_s),
        "random_seed": int(config.random_seed),
        "write_trial_logs": bool(config.write_trial_logs),
        "branch_selection_rule": str(config.branch_selection_rule),
        "environment_selection_rule": str(config.environment_selection_rule),
        "w0_dense_archive_performed": True,
        "w0_full_archive_performed": w0_scale_status == "meets_user_500k_target",
        "reduced_w0_archive_performed": bool(reduced),
        "production_w1_archive_performed": False,
        "w2_w3_w4_w5_performed": False,
        "hardware_or_mission_claim": False,
        "sim_to_real_transfer_claim": False,
        "branch_local_decisions_only": True,
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        "clustering_strategy_status": str(clustering_strategy_status),
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        "output_files": {
            "w0_dense_trial_descriptors": _path_text(outputs.trial_descriptors_csv),
            "w0_dense_envelope_map": _path_text(outputs.envelope_map_csv),
            "w0_dense_cluster_representatives": _path_text(
                outputs.cluster_representatives_csv
            ),
            "w0_dense_cluster_diagnostics": _path_text(outputs.cluster_diagnostics_csv),
            "w0_dense_archive_manifest": _path_text(outputs.manifest_json),
            "w0_dense_archive_report": _path_text(outputs.report_md),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Mon 25 W0 Dense Archive Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Planning run id: `{manifest['planning_run_id']}`",
        f"- Archive scale mode: `{manifest['archive_scale_mode']}`",
        f"- W0 scale status: `{manifest['w0_scale_status']}`",
        f"- Available W0 rows total: `{manifest['available_w0_rows_total']}`",
        f"- Selected W0 rows total: `{manifest['selected_w0_rows_total']}`",
        f"- Selected W0 rows by branch: `{manifest['selected_w0_rows_by_branch']}`",
        f"- Floor trials per branch: `{manifest['floor_trials_per_branch']}`",
        f"- Floor trials total: `{manifest['floor_trials_total']}`",
        f"- Target trials per branch: `{manifest['target_trials_per_branch']}`",
        f"- Target trials total: `{manifest['target_trials_total']}`",
        f"- W0 full archive performed: `{str(manifest['w0_full_archive_performed']).lower()}`",
        f"- Reduced W0 archive performed: `{str(manifest['reduced_w0_archive_performed']).lower()}`",
        f"- Production W1 archive performed: `{str(manifest['production_w1_archive_performed']).lower()}`",
        f"- W2/W3/W4/W5 performed: `{str(manifest['w2_w3_w4_w5_performed']).lower()}`",
        f"- Hardware or mission claim: `{str(manifest['hardware_or_mission_claim']).lower()}`",
        f"- Sim-to-real transfer claim: `{str(manifest['sim_to_real_transfer_claim']).lower()}`",
        f"- Clustering strategy status: `{manifest['clustering_strategy_status']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 5) Public Runner and CLI
# =============================================================================
def run_w0_dense_archive(
    *,
    run_id: int = 12,
    planning_run_id: int = 10,
    max_trials: int = 500000,
    floor_trials_per_branch: int = 150000,
    target_trials_per_branch: int = 250000,
    archive_scale_mode: str = "strict",
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    overwrite: bool = False,
    random_seed: int = 20260525,
    write_trial_logs: bool = False,
    result_root: Path | None = None,
) -> dict[str, Path]:
    """Run a W0-only dense archive pass from branch-separated planning rows."""

    config = W0DenseArchiveConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        max_trials=int(max_trials),
        floor_trials_per_branch=int(floor_trials_per_branch),
        target_trials_per_branch=int(target_trials_per_branch),
        archive_scale_mode=str(archive_scale_mode),
        latency_case=str(latency_case),
        dt_s=float(dt_s),
        horizon_s=float(horizon_s),
        random_seed=int(random_seed),
        result_root=result_root,
        overwrite=bool(overwrite),
        write_trial_logs=bool(write_trial_logs),
    )
    _validate_config(config)
    outputs = _output_paths(config)
    _validate_output_guardrails(config, outputs)
    _check_output_available(outputs, config.overwrite)
    start_path, candidate_path = _planning_paths(config)
    _check_planning_inputs_exist(start_path, candidate_path)
    start_states, candidates = _load_planning_tables(config)
    w0_candidates = _filter_w0_candidates(candidates)
    selected = _select_w0_candidates(w0_candidates, config)
    available_by_branch = _count_by_branch(w0_candidates)
    selected_by_branch = _count_by_branch(selected)
    scale_status = _w0_scale_status(
        selected_by_branch=selected_by_branch,
        config=config,
        reduced_mode=str(config.archive_scale_mode) == "reduced",
    )
    _enforce_scale_gate(config, scale_status)

    descriptors = _run_pilot_replays(start_states, selected, config)
    envelope = build_envelope_map(descriptors)
    clusters = select_cluster_representatives(descriptors, envelope)
    diagnostics = build_cluster_diagnostics(
        descriptors,
        envelope,
        clusters,
        DenseClusterDiagnosticConfig(min_trials_total=int(config.floor_trials_total)),
    )
    clustering_status = _dominant_clustering_status(diagnostics)
    manifest = _manifest(
        config,
        outputs,
        candidate_rows_available_total=int(len(candidates)),
        available_by_branch=available_by_branch,
        selected_by_branch=selected_by_branch,
        trial_count=int(len(descriptors)),
        w0_scale_status=scale_status,
        clustering_strategy_status=clustering_status,
    )

    _prepare_output_tree(outputs, config.overwrite)
    descriptors.to_csv(outputs.trial_descriptors_csv, index=False)
    envelope.to_csv(outputs.envelope_map_csv, index=False)
    clusters.to_csv(outputs.cluster_representatives_csv, index=False)
    diagnostics.to_csv(outputs.cluster_diagnostics_csv, index=False)
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    return outputs.as_dict()


def _dominant_clustering_status(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "insufficient_data_for_clustering_decision"
    priority = (
        "insufficient_data_for_clustering_decision",
        "needs_cluster_quota_strategy",
        "needs_boundary_augmented_strategy",
        "needs_adaptive_binning_strategy",
        "baseline_sufficient_for_w0_inspection",
    )
    statuses = set(diagnostics["clustering_strategy_status"].astype(str))
    for status in priority:
        if status in statuses:
            return status
    return "insufficient_data_for_clustering_decision"


def _direction_int(value: object) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return 0
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=12)
    parser.add_argument("--planning-run-id", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=500000)
    parser.add_argument("--floor-trials-per-branch", type=int, default=150000)
    parser.add_argument("--target-trials-per-branch", type=int, default=250000)
    parser.add_argument("--archive-scale-mode", default="strict")
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260525)
    parser.add_argument("--write-trial-logs", action="store_true")
    parser.add_argument("--result-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_w0_dense_archive(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        max_trials=args.max_trials,
        floor_trials_per_branch=args.floor_trials_per_branch,
        target_trials_per_branch=args.target_trials_per_branch,
        archive_scale_mode=args.archive_scale_mode,
        latency_case=args.latency_case,
        dt_s=args.dt_s,
        horizon_s=args.horizon_s,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        write_trial_logs=args.write_trial_logs,
        result_root=args.result_root,
    )
    print(f"w0_dense_archive_outputs={_path_text(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
