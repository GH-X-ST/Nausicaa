from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_schema import (  # noqa: E402
    BRANCH_DECISION_SCOPE,
    CAMPAIGN,
    DenseArchivePlanConfig,
    build_archive_count_manifest,
    build_target_environment_plan,
)
from dense_start_state_sampling import (  # noqa: E402
    build_dry_run_candidate_inventory,
    build_sampling_strata_summary,
    build_start_state_manifest,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Data Containers
# 2) Guardrails
# 3) Output Writers
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN
PROTECTED_DEFAULT_RUN_IDS = frozenset({7, 8, 9})
NO_CLAIM_TEXT = (
    "Expanded planning for a proper Sun-24 20k pilot gate only; no W0/W1 "
    "production archive, W2/W3/W4/W5 evidence, mission evaluation, hardware "
    "validation, or sim-to-real transfer was performed."
)


@dataclass(frozen=True)
class ExpandedDensePlanningConfig:
    run_id: int = 10
    source_planning_run_id: int = 8
    result_root: Path | None = None
    pilot_start_states_per_family_target_direction: int = 75
    random_seed: int = 20260524
    overwrite: bool = False
    required_min_candidate_rows: int = 20000
    branch_selection_rule: str = "equal_branch_local_planning"
    environment_selection_rule: str = "paired_w0_w1_environment_modes"


@dataclass(frozen=True)
class ExpandedDensePlanningOutputs:
    root: Path
    manifest_json: Path
    target_environment_plan_csv: Path
    sampling_strata_summary_csv: Path
    start_state_manifest_csv: Path
    dry_run_candidate_inventory_csv: Path
    report_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "target_environment_plan_csv": self.target_environment_plan_csv,
            "sampling_strata_summary_csv": self.sampling_strata_summary_csv,
            "start_state_manifest_csv": self.start_state_manifest_csv,
            "dry_run_candidate_inventory_csv": self.dry_run_candidate_inventory_csv,
            "report_md": self.report_md,
        }


def _active_result_root(config: ExpandedDensePlanningConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _output_paths(config: ExpandedDensePlanningConfig) -> ExpandedDensePlanningOutputs:
    root = _active_result_root(config) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    return ExpandedDensePlanningOutputs(
        root=root,
        manifest_json=root
        / "manifests"
        / f"expanded_dense_archive_planning_manifest_{suffix}.json",
        target_environment_plan_csv=root
        / "metrics"
        / f"equal_branch_target_environment_plan_{suffix}.csv",
        sampling_strata_summary_csv=root
        / "metrics"
        / f"equal_branch_sampling_strata_summary_{suffix}.csv",
        start_state_manifest_csv=root
        / "metrics"
        / f"equal_branch_start_state_manifest_pilot_{suffix}.csv",
        dry_run_candidate_inventory_csv=root
        / "metrics"
        / f"equal_branch_dry_run_candidate_inventory_pilot_{suffix}.csv",
        report_md=root
        / "reports"
        / f"expanded_dense_archive_planning_report_{suffix}.md",
    )


def _source_planning_root(config: ExpandedDensePlanningConfig) -> Path:
    return _active_result_root(config) / f"{int(config.source_planning_run_id):03d}"


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Guardrails
# =============================================================================
def _validate_config(config: ExpandedDensePlanningConfig) -> None:
    if int(config.run_id) <= 0:
        raise ValueError("run_id must be positive.")
    if int(config.source_planning_run_id) <= 0:
        raise ValueError("source_planning_run_id must be positive.")
    if int(config.run_id) == int(config.source_planning_run_id):
        raise ValueError("run_id must differ from source_planning_run_id.")
    if int(config.pilot_start_states_per_family_target_direction) <= 0:
        raise ValueError("pilot_start_states_per_family_target_direction must be positive.")
    if int(config.required_min_candidate_rows) <= 0:
        raise ValueError("required_min_candidate_rows must be positive.")


def _validate_output_guardrails(
    config: ExpandedDensePlanningConfig,
    outputs: ExpandedDensePlanningOutputs,
) -> None:
    if config.result_root is None and int(config.run_id) <= max(PROTECTED_DEFAULT_RUN_IDS):
        raise ValueError(
            "default result-root expanded planning run_id must be greater than 009."
        )
    if (
        config.result_root is None
        and bool(config.overwrite)
        and int(config.run_id) in PROTECTED_DEFAULT_RUN_IDS
    ):
        raise ValueError("refusing overwrite for protected default dense-planning runs.")

    output_root = outputs.root.resolve()
    source_root = _source_planning_root(config).resolve()
    if _same_or_contained(output_root, source_root) or _same_or_contained(
        source_root,
        output_root,
    ):
        raise ValueError(
            "refusing output/source planning overlap after path resolution: "
            f"output_root={output_root}, source_root={source_root}"
        )
    if not _source_planning_root(config).exists():
        raise FileNotFoundError(f"missing source planning run: {_source_planning_root(config)}")


def _same_or_contained(path: Path, container: Path) -> bool:
    return path == container or container in path.parents


def _check_output_available(
    outputs: ExpandedDensePlanningOutputs,
    overwrite: bool,
) -> None:
    if outputs.root.exists() and not bool(overwrite):
        raise ValueError(
            f"output directory already exists and overwrite=False: {outputs.root}"
        )


def _source_file_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[_path_text(path)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def _prepare_output_tree(
    config: ExpandedDensePlanningConfig,
    outputs: ExpandedDensePlanningOutputs,
) -> None:
    _check_output_available(outputs, config.overwrite)
    if outputs.root.exists() and config.overwrite:
        if config.result_root is None and int(config.run_id) in PROTECTED_DEFAULT_RUN_IDS:
            raise ValueError("refusing to clear a protected default dense-planning run.")
        _clear_output_tree(outputs.root)
    for path in (
        outputs.manifest_json.parent,
        outputs.target_environment_plan_csv.parent,
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


# =============================================================================
# 3) Output Writers
# =============================================================================
def _output_file_manifest(outputs: ExpandedDensePlanningOutputs) -> dict[str, str]:
    return {
        "expanded_dense_archive_planning_manifest": _path_text(outputs.manifest_json),
        "equal_branch_target_environment_plan": _path_text(
            outputs.target_environment_plan_csv
        ),
        "equal_branch_sampling_strata_summary": _path_text(
            outputs.sampling_strata_summary_csv
        ),
        "equal_branch_start_state_manifest_pilot": _path_text(
            outputs.start_state_manifest_csv
        ),
        "equal_branch_dry_run_candidate_inventory_pilot": _path_text(
            outputs.dry_run_candidate_inventory_csv
        ),
        "expanded_dense_archive_planning_report": _path_text(outputs.report_md),
    }


def _manifest(
    config: ExpandedDensePlanningConfig,
    outputs: ExpandedDensePlanningOutputs,
    start_row_count: int,
    candidate_row_count: int,
    source_preserved: bool,
) -> dict[str, object]:
    plan_config = _plan_config(config)
    manifest = build_archive_count_manifest(plan_config)
    manifest.update(
        {
            "source_planning_run_id": int(config.source_planning_run_id),
            "pilot_start_states_per_family_target_direction": int(
                config.pilot_start_states_per_family_target_direction
            ),
            "start_state_rows_all_branches": int(start_row_count),
            "candidate_rows_all_branches": int(candidate_row_count),
            "required_min_candidate_rows": int(config.required_min_candidate_rows),
            "expanded_planning_performed": True,
            "ready_for_20k_pilot": bool(
                int(candidate_row_count) >= int(config.required_min_candidate_rows)
            ),
            "production_w0_archive_performed": False,
            "production_w1_archive_performed": False,
            "pilot_sweep_performed": False,
            "hardware_or_mission_claim": False,
            "sim_to_real_transfer_claim": False,
            "branch_local_decisions_only": True,
            "source_run008_preserved": bool(source_preserved),
            "branch_selection_rule": str(config.branch_selection_rule),
            "environment_selection_rule": str(config.environment_selection_rule),
            "branch_decision_scope": BRANCH_DECISION_SCOPE,
            "no_overclaiming_statement": NO_CLAIM_TEXT,
            "output_files": _output_file_manifest(outputs),
        }
    )
    return manifest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Expanded Dense Archive Planning Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Source planning run id: `{manifest['source_planning_run_id']}`",
        f"- Pilot starts per family-target-direction: `{manifest['pilot_start_states_per_family_target_direction']}`",
        f"- Start-state rows all branches: `{manifest['start_state_rows_all_branches']}`",
        f"- Candidate rows all branches: `{manifest['candidate_rows_all_branches']}`",
        f"- Required minimum candidate rows: `{manifest['required_min_candidate_rows']}`",
        f"- Ready for 20k pilot: `{str(manifest['ready_for_20k_pilot']).lower()}`",
        f"- Source run 008 preserved: `{str(manifest['source_run008_preserved']).lower()}`",
        f"- Branch-local decisions only: `{str(manifest['branch_local_decisions_only']).lower()}`",
        f"- Production W0 archive performed: `{str(manifest['production_w0_archive_performed']).lower()}`",
        f"- Production W1 archive performed: `{str(manifest['production_w1_archive_performed']).lower()}`",
        f"- Pilot sweep performed: `{str(manifest['pilot_sweep_performed']).lower()}`",
        f"- Hardware or mission claim: `{str(manifest['hardware_or_mission_claim']).lower()}`",
        f"- Sim-to-real transfer claim: `{str(manifest['sim_to_real_transfer_claim']).lower()}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_expanded_dense_archive_planning(
    *,
    run_id: int = 10,
    source_planning_run_id: int = 8,
    result_root: Path | None = None,
    pilot_start_states_per_family_target_direction: int = 75,
    random_seed: int = 20260524,
    overwrite: bool = False,
    required_min_candidate_rows: int = 20000,
) -> dict[str, Path]:
    """Write an expanded branch-local planning run for the proper 20k pilot gate."""

    config = ExpandedDensePlanningConfig(
        run_id=int(run_id),
        source_planning_run_id=int(source_planning_run_id),
        result_root=result_root,
        pilot_start_states_per_family_target_direction=int(
            pilot_start_states_per_family_target_direction
        ),
        random_seed=int(random_seed),
        overwrite=bool(overwrite),
        required_min_candidate_rows=int(required_min_candidate_rows),
    )
    _validate_config(config)
    outputs = _output_paths(config)
    _validate_output_guardrails(config, outputs)
    _check_output_available(outputs, config.overwrite)
    source_before = _source_file_snapshot(_source_planning_root(config))

    plan_config = _plan_config(config)
    target_plan = build_target_environment_plan(plan_config)
    sampling_summary = build_sampling_strata_summary(plan_config)
    start_states = build_start_state_manifest(plan_config)
    candidate_inventory = build_dry_run_candidate_inventory(plan_config, start_states)
    candidate_count = int(len(candidate_inventory))
    if candidate_count < int(config.required_min_candidate_rows):
        raise RuntimeError(
            "expanded planning generated too few candidates for the proper 20k pilot: "
            f"candidate_rows={candidate_count}, "
            f"required_min_candidate_rows={int(config.required_min_candidate_rows)}"
        )

    source_preserved = source_before == _source_file_snapshot(_source_planning_root(config))
    manifest = _manifest(
        config,
        outputs,
        start_row_count=int(len(start_states)),
        candidate_row_count=candidate_count,
        source_preserved=source_preserved,
    )

    _prepare_output_tree(config, outputs)
    target_plan.to_csv(outputs.target_environment_plan_csv, index=False)
    sampling_summary.to_csv(outputs.sampling_strata_summary_csv, index=False)
    start_states.to_csv(outputs.start_state_manifest_csv, index=False)
    candidate_inventory.to_csv(outputs.dry_run_candidate_inventory_csv, index=False)
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    return outputs.as_dict()


def _plan_config(config: ExpandedDensePlanningConfig) -> DenseArchivePlanConfig:
    return DenseArchivePlanConfig(
        run_id=int(config.run_id),
        random_seed=int(config.random_seed),
        pilot_start_states_per_family_target_direction=int(
            config.pilot_start_states_per_family_target_direction
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=10)
    parser.add_argument("--source-planning-run-id", type=int, default=8)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument(
        "--pilot-starts-per-family-target-direction",
        type=int,
        default=75,
    )
    parser.add_argument("--random-seed", type=int, default=20260524)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--required-min-candidate-rows", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_expanded_dense_archive_planning(
        run_id=args.run_id,
        source_planning_run_id=args.source_planning_run_id,
        result_root=args.result_root,
        pilot_start_states_per_family_target_direction=(
            args.pilot_starts_per_family_target_direction
        ),
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        required_min_candidate_rows=args.required_min_candidate_rows,
    )
    print(f"expanded_dense_archive_planning_outputs={_path_text(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
