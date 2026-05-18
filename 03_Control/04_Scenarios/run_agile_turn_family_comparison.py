from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
TESTS_DIR = CONTROL_DIR / "tests"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agile_turn_family_comparison import (
    AGILE_TURN_CAMPAIGN,
    ARCHIVED_BOUNDARY_REFERENCE,
    DEFAULT_TARGETS_DEG,
    FAMILY_NAMES,
    TARGET_HORIZON_GRID_S,
    AgileTurnFamilyConfig,
    AgileTurnCandidateResult,
    AgileTurnFamilyComparisonResult,
    acceptance_thresholds,
    candidate_ranking_key,
    compare_agile_turn_families,
    family_inventory,
    heading_band_deg,
    target_ladder_deg,
)
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from logging_contract import command_dataframe, trajectory_dataframe
from result_paths import make_result_tree


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Shared constants and cleanup helpers
# 2) Output helpers
# 3) Comparison runner
# 4) CLI
# =============================================================================


# =============================================================================
# 1) Shared Constants and Cleanup Helpers
# =============================================================================
DEFAULT_RESULTS_ROOT = CONTROL_DIR / "05_Results"
OLD_STEM = "aggressive" + "_reversal"
OLD_CAMPAIGN = "07_" + OLD_STEM + "_ocp"
BOUNDARY_REFERENCE_NOTE = (
    "Archived high-alpha/perch-like boundary reference is preserved exactly as "
    "negative evidence; it is not an active reusable agile-turn family."
)
NO_OVERCLAIM_FLAGS = {
    "actual_agile_turn_family_comparison_implemented": True,
    "actual_agile_reversal_primitive_implemented": False,
    "updraft_validation_claim": False,
    "w1_w2_w3_updraft_validation_claim": False,
    "real_flight_validation_claim": False,
    "ocp_implemented": False,
    "tvlqr_implemented": False,
    "governor_implemented": False,
    "outer_loop_implemented": False,
    "vicon_implemented": False,
    "hardware_implemented": False,
    "high_incidence_validation_claim": False,
    "raw_normalised_commands_enter_state_derivative": False,
}


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _run_suffix(run_id: int) -> str:
    return f"s{run_id:03d}"


def _target_token(target_heading_deg: float) -> str:
    return f"{int(round(float(target_heading_deg))):03d}"


def _obsolete_active_paths() -> list[Path]:
    test_names = (
        "shapes",
        "smoke",
        "target_ladder",
        "30deg_energy",
    )
    return [
        PRIMITIVES_DIR / f"{OLD_STEM}_ocp.py",
        PRIMITIVES_DIR / f"{OLD_STEM}_primitive.py",
        SCENARIOS_DIR / f"run_{OLD_STEM}_search.py",
        *[TESTS_DIR / f"test_{OLD_STEM}_{name}.py" for name in test_names],
    ]


def _inventory_found_paths() -> dict[str, list[str]]:
    paths = _obsolete_active_paths()
    result_dirs = [
        DEFAULT_RESULTS_ROOT / OLD_CAMPAIGN / "001",
        DEFAULT_RESULTS_ROOT / OLD_CAMPAIGN / "002",
        DEFAULT_RESULTS_ROOT / AGILE_TURN_CAMPAIGN / "001",
    ]
    return {
        "obsolete_active_files_found": [
            _repo_relative(path) for path in paths if path.exists()
        ],
        "generated_result_directories_found": [
            _repo_relative(path) for path in result_dirs if path.exists()
        ],
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict[str, dict[str, object]]:
    if not root.exists():
        return {}
    hashes: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            hashes[rel] = {"sha256": _hash_file(path), "size_bytes": path.stat().st_size}
    return hashes


def _clear_generated_tree(root: Path) -> list[str]:
    """Delete generated files while tolerating sync-managed empty directories."""

    deleted: list[str] = []
    if not root.exists():
        return deleted
    resolved = root.resolve()
    allowed_parent = (DEFAULT_RESULTS_ROOT.resolve())
    if allowed_parent not in resolved.parents and resolved != allowed_parent:
        raise ValueError("refusing to clear generated files outside results root.")
    all_paths = list(root.rglob("*"))
    for path in all_paths:
        if path.is_file():
            path.unlink()
            deleted.append(_repo_relative(path))
    for path in sorted((path for path in all_paths if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
            deleted.append(_repo_relative(path))
        except OSError:
            pass
    if root.exists():
        try:
            root.rmdir()
            deleted.append(_repo_relative(root))
        except OSError:
            try:
                root.rename(root.with_name(f"{root.name}_cleanup_empty"))
                root.with_name(f"{root.name}_cleanup_empty").rmdir()
                deleted.append(_repo_relative(root))
            except OSError:
                pass
    return deleted


def _perform_cleanup() -> dict[str, object]:
    archive_root = REPO_ROOT / ARCHIVED_BOUNDARY_REFERENCE
    previous_cleanup = None
    previous_manifest_dir = DEFAULT_RESULTS_ROOT / AGILE_TURN_CAMPAIGN / "001" / "manifests"
    if previous_manifest_dir.exists():
        previous_files = sorted(previous_manifest_dir.glob("agile_turn_cleanup_manifest_*.json"))
        if previous_files:
            previous_cleanup = json.loads(previous_files[-1].read_text(encoding="ascii"))
    before_inventory = _inventory_found_paths()
    if previous_cleanup is not None:
        old_inventory = previous_cleanup.get("pre_cleanup_inventory", {})
        before_inventory = {
            "obsolete_active_files_found": sorted(
                set(before_inventory["obsolete_active_files_found"])
                | set(old_inventory.get("obsolete_active_files_found", []))
            ),
            "generated_result_directories_found": sorted(
                set(before_inventory["generated_result_directories_found"])
                | set(old_inventory.get("generated_result_directories_found", []))
            ),
        }
    archive_hashes_before = (
        previous_cleanup.get("archive_hashes_before")
        if previous_cleanup is not None
        else _hash_tree(archive_root)
    )
    deleted: list[str] = []

    for path in _obsolete_active_paths():
        if path.exists():
            path.unlink()
            deleted.append(_repo_relative(path))

    deleted.extend(_clear_generated_tree(DEFAULT_RESULTS_ROOT / OLD_CAMPAIGN / "001"))
    deleted.extend(_clear_generated_tree(DEFAULT_RESULTS_ROOT / AGILE_TURN_CAMPAIGN / "001"))
    if previous_cleanup is not None:
        deleted = sorted(set(previous_cleanup.get("deleted_paths", [])) | set(deleted))
    archive_hashes_after_cleanup = _hash_tree(archive_root)
    return {
        "pre_cleanup_inventory": before_inventory,
        "deleted_paths": deleted,
        "archive_reference_path": ARCHIVED_BOUNDARY_REFERENCE,
        "archive_hashes_before": archive_hashes_before,
        "archive_hashes_after_cleanup": archive_hashes_after_cleanup,
        "archive_preserved_after_cleanup": archive_hashes_before == archive_hashes_after_cleanup,
    }


# =============================================================================
# 2) Output Helpers
# =============================================================================
def _json_safe(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return _repo_relative(value)
    return str(value)


def _best_for_family(
    result: AgileTurnFamilyComparisonResult,
    family_name: str,
) -> AgileTurnCandidateResult:
    family_candidates = [
        candidate for candidate in result.family_results
        if candidate.family_name == family_name
    ]
    if not family_candidates:
        raise ValueError(f"no candidates found for family {family_name}.")
    return max(family_candidates, key=lambda candidate: candidate_ranking_key(candidate.metrics))


def _write_best_candidate_logs(
    paths: dict[str, Path],
    run_id: int,
    result: AgileTurnFamilyComparisonResult,
    output_paths: dict[str, Path],
) -> None:
    suffix = _run_suffix(run_id)
    for family_name in family_inventory():
        candidate = _best_for_family(result, family_name)
        target = _target_token(result.target_heading_deg)
        base = f"agile_turn_{family_name}_target_{target}_{suffix}"
        trajectory_csv = paths["metrics"] / f"{base}_trajectory.csv"
        commands_csv = paths["metrics"] / f"{base}_commands.csv"
        trajectory_dataframe(candidate.time_s, candidate.x_ref).assign(
            phase=list(candidate.phase)
        ).to_csv(trajectory_csv, index=False)
        command_dataframe(
            candidate.time_s,
            candidate.u_norm_requested,
            candidate.u_norm_applied,
            candidate.delta_cmd_rad,
        ).assign(phase=list(candidate.phase)).to_csv(commands_csv, index=False)
        output_paths[f"{base}_trajectory_csv"] = trajectory_csv
        output_paths[f"{base}_commands_csv"] = commands_csv


def _best_candidate_id(
    candidates: tuple[AgileTurnCandidateResult, ...],
    field: str,
) -> str:
    filtered = [candidate for candidate in candidates if bool(candidate.metrics[field])]
    if not filtered:
        return ""
    selected = max(filtered, key=lambda candidate: candidate_ranking_key(candidate.metrics))
    return str(selected.metrics["candidate_id"])


def _target_summary_row(result: AgileTurnFamilyComparisonResult) -> dict[str, object]:
    candidates = result.family_results
    selected = result.selected_candidate
    best_any = max(candidates, key=lambda candidate: candidate.metrics["peak_heading_change_deg"])
    lower, upper = heading_band_deg(result.target_heading_deg)
    return {
        "target_heading_deg": float(result.target_heading_deg),
        "selected_candidate_id": "" if selected is None else selected.metrics["candidate_id"],
        "selected_family": "" if selected is None else selected.family_name,
        "selected_horizon_s": "" if selected is None else selected.metrics["horizon_s"],
        "commandable_target_found": selected is not None,
        "best_commandable_candidate_id": _best_candidate_id(candidates, "commandable_target_candidate"),
        "best_safe_partial_candidate_id": _best_candidate_id(candidates, "safe_partial_turn_evidence"),
        "best_accurate_boundary_candidate_id": _best_candidate_id(candidates, "accurate_boundary_evidence"),
        "best_any_heading_candidate_id": best_any.metrics["candidate_id"],
        "best_any_heading_failure_label": best_any.metrics["failure_label"],
        "terminal_heading_band": f"{lower:.1f} to {upper:.1f} deg",
        "shortest_successful_horizon_s": "" if selected is None else selected.metrics["horizon_s"],
        "escalation_allowed": False,
        "escalation_reason": "pending_30deg_commandable_check",
        "selection_reason": result.notes,
    }


def _family_summary_rows(results: tuple[AgileTurnFamilyComparisonResult, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_name in family_inventory():
        candidates = [
            candidate
            for result in results
            for candidate in result.family_results
            if candidate.family_name == family_name
        ]
        best = max(candidates, key=lambda candidate: candidate_ranking_key(candidate.metrics))
        if any(bool(candidate.metrics["commandable_target_candidate"]) for candidate in candidates):
            status = "selected_for_next_stage"
        elif any(bool(candidate.metrics["safe_partial_turn_evidence"]) for candidate in candidates):
            status = "retained_as_thesis_discussion_evidence"
        else:
            status = "rejected_for_active_primitive"
        rows.append(
            {
                "family_name": family_name,
                "family_status": status,
                "best_candidate_id": best.metrics["candidate_id"],
                "best_candidate_class": best.metrics["candidate_class"],
                "best_terminal_heading_change_deg": best.metrics["terminal_heading_change_deg"],
                "best_terminal_speed_m_s": best.metrics["terminal_speed_m_s"],
                "best_active_limiting_mechanism": best.metrics["active_limiting_mechanism"],
            }
        )
    return rows


def _write_report(
    path: Path,
    manifest: dict[str, object],
    target_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Agile Turn Precision Ladder Cleanup Report",
        "",
        "This W0/no-wind pass enforces terminal-heading target bands for commandable agile-turn labels.",
        BOUNDARY_REFERENCE_NOTE,
        "",
        "No OCP, TVLQR, governor, outer loop, updraft validation, real-flight, hardware,",
        "or high-incidence validation claim is made from this pass.",
        "",
        "## Target Summary",
        "",
        "| target_deg | commandable | selected_family | selected_horizon_s | best_safe_partial | best_boundary | escalation_reason |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in target_rows:
        lines.append(
            "| {target_heading_deg:.0f} | {commandable_target_found} | {selected_family} | "
            "{selected_horizon_s} | {best_safe_partial_candidate_id} | "
            "{best_accurate_boundary_candidate_id} | {escalation_reason} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Family Status",
            "",
            "| family | status | best_class | best_terminal_heading_deg | limiter |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in family_rows:
        lines.append(
            "| {family_name} | {family_status} | {best_candidate_class} | "
            "{best_terminal_heading_change_deg:.3f} | {best_active_limiting_mechanism} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Cleanup",
            "",
            f"- Archived boundary reference preserved: `{manifest['archived_perch_reference_preserved']}`.",
            f"- Old branch active: `{manifest['old_perch_like_branch_active']}`.",
            f"- Fixed target ladder: `{manifest['fixed_target_ladder_deg']}`.",
            f"- No 20-degree bin: `{manifest['no_20deg_bin']}`.",
            f"- Command bridge: `{manifest['command_bridge']}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 3) Comparison Runner
# =============================================================================
def run_comparison(
    run_id: int = 1,
    targets_deg: tuple[float, ...] = DEFAULT_TARGETS_DEG,
    overwrite: bool = False,
    escalate: bool = False,
) -> dict[str, Path]:
    """Run target-band agile-turn comparison and write cleanup evidence."""

    cleanup_info = _perform_cleanup() if overwrite else {
        "pre_cleanup_inventory": _inventory_found_paths(),
        "deleted_paths": [],
        "archive_reference_path": ARCHIVED_BOUNDARY_REFERENCE,
        "archive_hashes_before": _hash_tree(REPO_ROOT / ARCHIVED_BOUNDARY_REFERENCE),
        "archive_hashes_after_cleanup": _hash_tree(REPO_ROOT / ARCHIVED_BOUNDARY_REFERENCE),
        "archive_preserved_after_cleanup": True,
    }
    paths = make_result_tree(DEFAULT_RESULTS_ROOT, AGILE_TURN_CAMPAIGN, run_id, overwrite=overwrite)
    suffix = _run_suffix(run_id)
    aircraft = adapt_glider(build_nausicaa_glider())
    requested_targets = tuple(float(target) for target in targets_deg)
    if any(not any(np.isclose(target, ladder) for ladder in target_ladder_deg()) for target in requested_targets):
        raise ValueError("targets must be drawn from the fixed precision ladder.")
    if any(target not in DEFAULT_TARGETS_DEG for target in requested_targets) and not escalate:
        raise ValueError("higher targets require --escalate and a commandable 30 deg candidate.")

    active_targets = [target for target in requested_targets if target in DEFAULT_TARGETS_DEG]
    results: list[AgileTurnFamilyComparisonResult] = []
    candidate_rows: list[dict[str, object]] = []
    output_paths: dict[str, Path] = {}
    for target in active_targets:
        config = AgileTurnFamilyConfig(
            t_final_s=TARGET_HORIZON_GRID_S[target][0],
            target_heading_deg=target,
        )
        result = compare_agile_turn_families(config, families=FAMILY_NAMES, aircraft=aircraft)
        results.append(result)
        candidate_rows.extend({"run_id": suffix, **row} for row in result.ranking_rows)
        _write_best_candidate_logs(paths, run_id, result, output_paths)

    target_rows = [_target_summary_row(result) for result in results]
    commandable_30 = any(
        row["target_heading_deg"] == 30.0 and bool(row["commandable_target_found"])
        for row in target_rows
    )
    for row in target_rows:
        row["escalation_allowed"] = bool(commandable_30)
        row["escalation_reason"] = (
            "30deg_commandable_target_candidate_found"
            if commandable_30
            else "30deg_not_commandable_safe_partial_or_boundary_only"
        )
    family_rows = _family_summary_rows(tuple(results))

    candidate_summary_csv = paths["metrics"] / f"agile_turn_candidate_summary_{suffix}.csv"
    target_summary_csv = paths["metrics"] / f"agile_turn_target_summary_{suffix}.csv"
    family_summary_csv = paths["metrics"] / f"agile_turn_family_summary_{suffix}.csv"
    pd.DataFrame(candidate_rows).to_csv(candidate_summary_csv, index=False)
    pd.DataFrame(target_rows).to_csv(target_summary_csv, index=False)
    pd.DataFrame(family_rows).to_csv(family_summary_csv, index=False)
    output_paths.update(
        {
            "candidate_summary_csv": candidate_summary_csv,
            "target_summary_csv": target_summary_csv,
            "family_summary_csv": family_summary_csv,
        }
    )

    manifest_json = paths["manifests"] / f"agile_turn_family_comparison_manifest_{suffix}.json"
    cleanup_manifest_json = paths["manifests"] / f"agile_turn_cleanup_manifest_{suffix}.json"
    report_md = paths["reports"] / f"agile_turn_family_comparison_report_{suffix}.md"
    output_paths["manifest_json"] = manifest_json
    output_paths["cleanup_manifest_json"] = cleanup_manifest_json
    output_paths["report_md"] = report_md
    regenerated_files = {key: _repo_relative(path) for key, path in output_paths.items()}
    archive_hashes_after_regeneration = _hash_tree(REPO_ROOT / ARCHIVED_BOUNDARY_REFERENCE)
    archive_preserved = (
        cleanup_info["archive_hashes_before"]
        == cleanup_info["archive_hashes_after_cleanup"]
        == archive_hashes_after_regeneration
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": suffix,
        "campaign": AGILE_TURN_CAMPAIGN,
        "fixed_target_ladder_deg": list(target_ladder_deg()),
        "no_20deg_bin": True,
        "old_perch_like_branch_active": False,
        "archived_perch_reference_preserved": bool(archive_preserved),
        "archived_boundary_reference": ARCHIVED_BOUNDARY_REFERENCE,
        "active_family_inventory": list(family_inventory()),
        "targets_requested_deg": list(requested_targets),
        "targets_run_deg": [result.target_heading_deg for result in results],
        "target_horizon_grid_s": {str(key): list(value) for key, value in TARGET_HORIZON_GRID_S.items()},
        "acceptance_thresholds": acceptance_thresholds(),
        "escalation_allowed": bool(commandable_30),
        "escalation_requested": bool(escalate),
        "escalation_targets_run_deg": [],
        "escalation_reason": (
            "30deg_commandable_target_candidate_found"
            if commandable_30
            else "30deg_not_commandable_safe_partial_or_boundary_only"
        ),
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad",
        "state_derivative_command_input": "delta_cmd_rad",
        "output_files": regenerated_files,
        **NO_OVERCLAIM_FLAGS,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=_json_safe), encoding="ascii")
    cleanup_manifest = {
        **cleanup_info,
        "archive_hashes_after_regeneration": archive_hashes_after_regeneration,
        "archive_preserved_byte_for_byte": bool(archive_preserved),
        "regenerated_files": regenerated_files,
        "forbidden_token_grep_expected_exit_code": 1,
    }
    cleanup_manifest_json.write_text(
        json.dumps(cleanup_manifest, indent=2, default=_json_safe),
        encoding="ascii",
    )
    _write_report(report_md, manifest, target_rows, family_rows)
    output_paths["root"] = paths["root"]
    return output_paths


# =============================================================================
# 4) CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the precision-ladder agile-turn comparison.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--targets", nargs="+", type=float, default=list(DEFAULT_TARGETS_DEG))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--escalate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_comparison(
        run_id=args.run_id,
        targets_deg=tuple(args.targets),
        overwrite=args.overwrite,
        escalate=args.escalate,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['manifest_json']}")
    print(f"cleanup_manifest={outputs['cleanup_manifest_json']}")
    print(f"report={outputs['report_md']}")
    print(f"targets_run_deg={manifest['targets_run_deg']}")
    print(f"escalation_allowed={manifest['escalation_allowed']}")
    print(f"escalation_reason={manifest['escalation_reason']}")


if __name__ == "__main__":
    main()
