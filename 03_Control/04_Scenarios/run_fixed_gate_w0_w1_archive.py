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
    reachable_source_csv: Path | None = None,
    result_root: Path | None = None,
    storage_format: str = "auto",
    overwrite: bool = False,
) -> dict[str, Path]:
    root = _archive_root(run_id, result_root)
    paths = _prepare_tree(root, overwrite=overwrite)
    selected_branches = _branches(fan_branch)
    selected_layers = _layers(w_layers)
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
            controller_mode="open_loop_rollout",
            feedback_mode="open_loop",
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
    )
    manifest = _manifest(
        run_id=int(run_id),
        seed=int(seed),
        rows_per_branch=int(rows_per_branch),
        fan_branch=fan_branch,
        selected_layers=selected_layers,
        latency_case=latency_case,
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
        overwrite=overwrite,
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
) -> dict[str, Path]:
    fmt = resolve_storage_format(storage_format)
    extension = table_extension(fmt)
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
        "pairing_audit_csv": paths["metrics"] / "fixed_gate_w0_w1_pairing_audit.csv",
        "outcome_summary_csv": paths["metrics"] / "fixed_gate_w0_w1_outcome_summary.csv",
        "code_path_map_csv": paths["metrics"] / "active_deprecated_code_path_map.csv",
        "manifest_json": paths["manifests"] / "fixed_gate_w0_w1_archive_manifest.json",
        "report_md": paths["reports"] / "fixed_gate_w0_w1_archive_report.md",
    }
    samples.to_csv(filesystem_path(output_paths["samples_csv"]), index=False)
    candidates.to_csv(filesystem_path(output_paths["candidate_index_csv"]), index=False)
    rollout_rows.to_csv(filesystem_path(output_paths["rollout_rows_csv"]), index=False)
    pairing_audit.to_csv(filesystem_path(output_paths["pairing_audit_csv"]), index=False)
    outcome_summary.to_csv(filesystem_path(output_paths["outcome_summary_csv"]), index=False)
    code_path_map.to_csv(filesystem_path(output_paths["code_path_map_csv"]), index=False)
    return output_paths


def _manifest(
    *,
    run_id: int,
    seed: int,
    rows_per_branch: int,
    fan_branch: str,
    selected_layers: tuple[str, ...],
    latency_case: str,
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
        "reachable_source_csv": "" if reachable_source_csv is None else str(reachable_source_csv),
        "sample_row_count": int(len(samples)),
        "candidate_row_count": int(len(candidates)),
        "rollout_row_count": int(len(rollout_rows)),
        "pairing_audit_row_count": int(len(pairing_audit)),
        "W1_independent_of_W0_success": bool(pairing_audit["w1_scheduled_independent_of_w0_success"].all()) if not pairing_audit.empty else False,
        "branch_local_separation": True,
        "rollout_execution_performed": True,
        "mission_feedback_path_status": move_on_gates["feedback_path_status"],
        "feedback_stabilised_primitive_available": False,
        "open_loop_rows_promoted_to_mission_candidate": False,
        "move_on_gates": move_on_gates,
        "archive_readiness_status": move_on_gates["archive_readiness"],
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
            f"- Diagnostic open-loop rows: `{gates['ablation_diagnostic_row_count']}`",
            f"- Archive readiness: `{manifest['archive_readiness_status']}`",
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
    parser.add_argument("--reachable-source-csv", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--storage-format", default="auto")
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
        reachable_source_csv=args.reachable_source_csv,
        result_root=args.result_root,
        storage_format=args.storage_format,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
