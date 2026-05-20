from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from hashlib import sha256
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
    CAMPAIGN,
    FORBIDDEN_CLAIMS,
    PASS_NAME,
    PROTECTED_PATHS,
    RECOMMENDED_NEXT_STEP,
    SOURCE_RUN007_MANIFEST,
    SOURCE_STAGE0_MANIFEST,
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
# 2) Stage 0, Run 007, and Protected-Hash Checks
# 3) Output Writers and Report
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN
REQUIRED_STAGE0_STATUS = "passed"


@dataclass(frozen=True)
class DenseArchivePlanningOutputs:
    root: Path
    manifest_json: Path
    target_environment_plan_csv: Path
    sampling_strata_summary_csv: Path
    start_state_manifest_csv: Path
    dry_run_candidate_inventory_csv: Path
    planning_report_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "target_environment_plan_csv": self.target_environment_plan_csv,
            "sampling_strata_summary_csv": self.sampling_strata_summary_csv,
            "start_state_manifest_csv": self.start_state_manifest_csv,
            "dry_run_candidate_inventory_csv": self.dry_run_candidate_inventory_csv,
            "planning_report_md": self.planning_report_md,
        }


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _output_paths(run_id: int) -> DenseArchivePlanningOutputs:
    root = RESULT_ROOT / f"{int(run_id):03d}"
    suffix = f"s{int(run_id):03d}"
    return DenseArchivePlanningOutputs(
        root=root,
        manifest_json=root / "manifests" / f"equal_branch_paired_archive_count_manifest_{suffix}.json",
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
        planning_report_md=root
        / "reports"
        / f"equal_branch_paired_dense_archive_planning_report_{suffix}.md",
    )


def _prepare_output_tree(outputs: DenseArchivePlanningOutputs, overwrite: bool) -> None:
    if outputs.root.exists() and not overwrite:
        raise ValueError(f"output directory already exists: {outputs.root}")
    if outputs.root.exists() and overwrite:
        _clear_output_tree(outputs.root)
    for path in (
        outputs.manifest_json.parent,
        outputs.target_environment_plan_csv.parent,
        outputs.planning_report_md.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clear_output_tree(root: Path) -> None:
    # Only run-008 generated files are cleared. Directories are retained to
    # avoid Windows/OneDrive directory locks during repeated audit runs.
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()


# =============================================================================
# 2) Stage 0, Run 007, and Protected-Hash Checks
# =============================================================================
def _read_stage0_gate_status() -> str:
    path = REPO_ROOT / SOURCE_STAGE0_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"missing Stage 0 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="ascii"))
    status = str(manifest.get("overall_stage0_gate_status", "missing"))
    if status != REQUIRED_STAGE0_STATUS:
        raise RuntimeError(
            "Phase B equal-branch planning requires Stage 0 overall gate to be passed; "
            f"saw {status!r}."
        )
    return status


def _run007_preserved() -> bool:
    path = REPO_ROOT / SOURCE_RUN007_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"missing run 007 planning manifest: {path}")
    manifest = json.loads(path.read_text(encoding="ascii"))
    return int(manifest.get("run_id", -1)) == 7


def _protected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in PROTECTED_PATHS:
        root = REPO_ROOT / relative
        if not root.exists():
            hashes[relative] = "missing"
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[_repo_relative(path)] = sha256(path.read_bytes()).hexdigest()
    return hashes


# =============================================================================
# 3) Output Writers and Report
# =============================================================================
def _output_file_manifest(outputs: DenseArchivePlanningOutputs) -> dict[str, str]:
    return {
        "equal_branch_paired_archive_count_manifest": _repo_relative(outputs.manifest_json),
        "equal_branch_target_environment_plan": _repo_relative(outputs.target_environment_plan_csv),
        "equal_branch_sampling_strata_summary": _repo_relative(outputs.sampling_strata_summary_csv),
        "equal_branch_start_state_manifest_pilot": _repo_relative(outputs.start_state_manifest_csv),
        "equal_branch_dry_run_candidate_inventory_pilot": _repo_relative(
            outputs.dry_run_candidate_inventory_csv
        ),
        "equal_branch_paired_dense_archive_planning_report": _repo_relative(outputs.planning_report_md),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Phase B Task 1.1 Equal Fan Branch Planning Report",
        "",
        f"- Campaign: `{manifest['campaign']}`",
        f"- Pass name: `{manifest['pass_name']}`",
        f"- Stage 0 gate status seen: `{manifest['stage0_gate_status_seen']}`",
        f"- Run 007 preserved: `{str(manifest['run007_preserved']).lower()}`",
        f"- Protected hash status: `{manifest['protected_hash_check_status']}`",
        "- This pass created a branch-separated paired W0/W1 planning scaffold only.",
        "- Wing-scale wind descriptor logging is implemented for planning rows only.",
        "- Dense-trial descriptor schema is implemented for future descriptor rows only;",
        "  dense-trial descriptor execution was not performed.",
        "- No rollout, primitive replay, dense archive execution, active latency implementation,",
        "  envelope mapping, clustering, mission evaluation, or sim-to-real transfer was performed.",
        "",
        "## Branch Count Contract",
        "",
        f"- W1 floor total per branch: `{manifest['w1_floor_total_trials_per_branch']}`",
        f"- W1 target total per branch: `{manifest['w1_target_total_trials_per_branch']}`",
        f"- W1 floor total all branches: `{manifest['w1_floor_total_trials_all_branches']}`",
        f"- W1 target total all branches: `{manifest['w1_target_total_trials_all_branches']}`",
        f"- W0 floor total per branch: `{manifest['w0_floor_total_trials_per_branch']}`",
        f"- W0 target total per branch: `{manifest['w0_target_total_trials_per_branch']}`",
        f"- W0 floor total all branches: `{manifest['w0_floor_total_trials_all_branches']}`",
        f"- W0 target total all branches: `{manifest['w0_target_total_trials_all_branches']}`",
        f"- Combined floor total all branches: `{manifest['combined_floor_total_trials_all_branches']}`",
        f"- Combined target total all branches: `{manifest['combined_target_total_trials_all_branches']}`",
        f"- Pilot start rows all branches: `{manifest['pilot_start_state_rows_all_branches']}`",
        f"- Pilot candidate rows all branches: `{manifest['pilot_candidate_rows_all_branches']}`",
        "",
        "## Forbidden Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in FORBIDDEN_CLAIMS)
    lines.extend(["", "## Next Step", "", RECOMMENDED_NEXT_STEP, ""])
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_dense_archive_planning(
    *,
    run_id: int = 8,
    overwrite: bool = False,
    random_seed: int = 20260520,
    pilot_start_states_per_family_target_direction: int = 10,
) -> dict[str, Path]:
    """Write equal fan-branch dense archive planning outputs without rollouts."""

    if int(run_id) != 8:
        raise ValueError("Phase B Task 1.1 equal-branch planning outputs must use run_id=8.")
    if int(pilot_start_states_per_family_target_direction) <= 0:
        raise ValueError("pilot_start_states_per_family_target_direction must be positive.")

    config = DenseArchivePlanConfig(
        run_id=int(run_id),
        random_seed=int(random_seed),
        pilot_start_states_per_family_target_direction=int(
            pilot_start_states_per_family_target_direction
        ),
    )
    stage0_status = _read_stage0_gate_status()
    run007_preserved = _run007_preserved()
    protected_before = _protected_hashes()
    outputs = _output_paths(config.run_id)
    _prepare_output_tree(outputs, overwrite=bool(overwrite))

    target_plan = build_target_environment_plan(config)
    sampling_summary = build_sampling_strata_summary(config)
    start_states = build_start_state_manifest(config)
    candidate_inventory = build_dry_run_candidate_inventory(config, start_states)

    target_plan.to_csv(outputs.target_environment_plan_csv, index=False)
    sampling_summary.to_csv(outputs.sampling_strata_summary_csv, index=False)
    start_states.to_csv(outputs.start_state_manifest_csv, index=False)
    candidate_inventory.to_csv(outputs.dry_run_candidate_inventory_csv, index=False)

    protected_after = _protected_hashes()
    protected_status = "unchanged" if protected_before == protected_after else "changed"
    manifest = build_archive_count_manifest(config)
    manifest.update(
        {
            "stage0_gate_status_seen": stage0_status,
            "run007_preserved": bool(run007_preserved),
            "output_files": _output_file_manifest(outputs),
            "protected_hash_check_status": protected_status,
            "protected_hash_count_before": len(protected_before),
            "protected_hash_count_after": len(protected_after),
            "no_rollout_performed": True,
            "planning_only_scope": True,
        }
    )
    _write_report(outputs.planning_report_md, manifest)
    _write_json(outputs.manifest_json, manifest)

    if protected_status != "unchanged":
        raise RuntimeError("protected hashes changed while writing run 008 planning outputs.")
    return outputs.as_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260520)
    parser.add_argument("--pilot-starts-per-family-target-direction", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_dense_archive_planning(
        run_id=args.run_id,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        pilot_start_states_per_family_target_direction=args.pilot_starts_per_family_target_direction,
    )
    print(f"dense_archive_planning_outputs={_repo_relative(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
