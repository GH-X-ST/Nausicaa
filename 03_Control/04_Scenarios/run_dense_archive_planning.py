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
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_schema import (  # noqa: E402
    CAMPAIGN,
    FORBIDDEN_CLAIMS,
    PASS_NAME,
    PROTECTED_STAGE0_PATHS,
    RECOMMENDED_NEXT_STEP,
    SOURCE_STAGE0_MANIFEST,
    DenseArchivePlanConfig,
    build_archive_count_manifest,
    build_target_direction_plan,
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
# 2) Stage 0 and Protected-Hash Checks
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
    target_direction_plan_csv: Path
    sampling_strata_summary_csv: Path
    start_state_manifest_csv: Path
    dry_run_candidate_inventory_csv: Path
    planning_report_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "target_direction_plan_csv": self.target_direction_plan_csv,
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
        manifest_json=root / "manifests" / f"archive_count_manifest_{suffix}.json",
        target_direction_plan_csv=root / "metrics" / f"target_direction_plan_{suffix}.csv",
        sampling_strata_summary_csv=root / "metrics" / f"sampling_strata_summary_{suffix}.csv",
        start_state_manifest_csv=root / "metrics" / f"start_state_manifest_pilot_{suffix}.csv",
        dry_run_candidate_inventory_csv=root / "metrics" / f"dry_run_candidate_inventory_pilot_{suffix}.csv",
        planning_report_md=root / "reports" / f"dense_archive_planning_report_{suffix}.md",
    )


def _prepare_output_tree(outputs: DenseArchivePlanningOutputs, overwrite: bool) -> None:
    if outputs.root.exists() and not overwrite:
        raise ValueError(f"output directory already exists: {outputs.root}")
    if outputs.root.exists() and overwrite:
        _clear_output_tree(outputs.root)
    for path in (
        outputs.manifest_json.parent,
        outputs.target_direction_plan_csv.parent,
        outputs.planning_report_md.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clear_output_tree(root: Path) -> None:
    # The runner is only allowed to clear files in its own Phase B output tree.
    # Keeping directories avoids Windows/OneDrive locks while still replacing
    # every generated artifact deterministically on --overwrite.
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()


# =============================================================================
# 2) Stage 0 and Protected-Hash Checks
# =============================================================================
def _read_stage0_gate_status() -> str:
    path = REPO_ROOT / SOURCE_STAGE0_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"missing Stage 0 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="ascii"))
    status = str(manifest.get("overall_stage0_gate_status", "missing"))
    if status != REQUIRED_STAGE0_STATUS:
        raise RuntimeError(
            "Phase B dense archive planning requires Stage 0 overall gate to be passed; "
            f"saw {status!r}."
        )
    return status


def _protected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in PROTECTED_STAGE0_PATHS:
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
        "archive_count_manifest": _repo_relative(outputs.manifest_json),
        "target_direction_plan": _repo_relative(outputs.target_direction_plan_csv),
        "sampling_strata_summary": _repo_relative(outputs.sampling_strata_summary_csv),
        "start_state_manifest_pilot": _repo_relative(outputs.start_state_manifest_csv),
        "dry_run_candidate_inventory_pilot": _repo_relative(outputs.dry_run_candidate_inventory_csv),
        "dense_archive_planning_report": _repo_relative(outputs.planning_report_md),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Phase B Task 1 Dense Archive Planning Report",
        "",
        f"- Campaign: `{manifest['campaign']}`",
        f"- Pass name: `{manifest['pass_name']}`",
        f"- Stage 0 gate status seen: `{manifest['stage0_gate_status_seen']}`",
        "- Stage 0 source evidence remains protected and was not edited by this pass.",
        "- This pass created a Phase B planning scaffold only.",
        "- Full W0 dense execution was not performed.",
        "- W1/W2/W3 robustness was not performed.",
        "- Objective one and objective two were not attempted.",
        "- Real flight transfer was not attempted.",
        "",
        "## Corrected Count Contract",
        "",
        f"- Minimum W0 turning trials: `{manifest['minimum_w0_turning_trials']}`",
        f"- Target W0 turning trials: `{manifest['target_w0_turning_trials']}`",
        f"- Minimum glide/recovery W0 trials: `{manifest['minimum_w0_baseline_trials']}`",
        f"- Target glide/recovery W0 trials: `{manifest['target_w0_baseline_trials']}`",
        f"- Minimum W0 total trials: `{manifest['minimum_w0_total_trials']}`",
        f"- Target W0 total trials: `{manifest['target_w0_total_trials']}`",
        f"- Target W0 total trial range: `{manifest['target_w0_total_trial_range']}`",
        f"- Pilot candidate count: `{manifest['pilot_total_candidate_count']}`",
        "",
        "The pilot count controls only the small deterministic pilot manifests. It does not",
        "replace the 2000 minimum or 5000 target starts per turning family-target-direction",
        "recorded in the archive count manifest.",
        "",
        "## Forbidden Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in FORBIDDEN_CLAIMS)
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            RECOMMENDED_NEXT_STEP,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_dense_archive_planning(
    *,
    run_id: int = 7,
    overwrite: bool = False,
    random_seed: int = 20260520,
    pilot_start_states_per_family_target_direction: int = 10,
) -> dict[str, Path]:
    """Write Phase B dense archive planning outputs without running rollouts."""

    if int(run_id) != 7:
        raise ValueError("Phase B Task 1 dense archive planning outputs must use run_id=7.")
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
    protected_before = _protected_hashes()
    outputs = _output_paths(config.run_id)
    _prepare_output_tree(outputs, overwrite=bool(overwrite))

    target_plan = build_target_direction_plan(config)
    sampling_summary = build_sampling_strata_summary(config)
    start_states = build_start_state_manifest(config)
    candidate_inventory = build_dry_run_candidate_inventory(config, start_states)

    target_plan.to_csv(outputs.target_direction_plan_csv, index=False)
    sampling_summary.to_csv(outputs.sampling_strata_summary_csv, index=False)
    start_states.to_csv(outputs.start_state_manifest_csv, index=False)
    candidate_inventory.to_csv(outputs.dry_run_candidate_inventory_csv, index=False)

    protected_after = _protected_hashes()
    protected_status = "unchanged" if protected_before == protected_after else "changed"
    manifest = build_archive_count_manifest(config)
    manifest.update(
        {
            "stage0_gate_status_seen": stage0_status,
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
        raise RuntimeError("protected Stage 0 hashes changed while writing Phase B planning outputs.")
    return outputs.as_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=7)
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
