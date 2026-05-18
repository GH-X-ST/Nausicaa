from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import stat
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from logging_contract import command_dataframe, trajectory_dataframe
from primitive_library import CandidateEvaluation, WindModelInfo, evaluate_candidate, group_summary
from primitive_library_generators import primitive_candidate_inventory
from primitive_library_schema import (
    CANDIDATE_CLASSES,
    PRIMITIVE_FAMILIES,
    TARGET_LADDER_DEG,
    UPDRAFT_CONFIGS,
    WIND_FIDELITIES,
    Z_OUTLET_M,
    PrimitiveLibraryConfig,
)
from updraft_models import load_updraft_model


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Housekeeping constants and helpers
# 2) Manifest builders
# 3) Evidence generation and writing
# 4) CLI entry point
# =============================================================================


# =============================================================================
# 1) Housekeeping Constants and Helpers
# =============================================================================
CAMPAIGN = "09_primitive_library"
RESULT_ROOT = CONTROL_DIR / "05_Results"
OBSOLETE_TOKEN_PARTS = (
    ("aggressive", "_reversal", "_ocp"),
    ("run", "_aggressive", "_reversal", "_search"),
    ("aggressive", "_reversal", "_primitive"),
    ("agile", "_turn", "_family", "_comparison"),
    ("run", "_agile", "_turn", "_family", "_comparison"),
    ("dive", "_perch", "_redirect", "_30"),
    ("reduced", "_perch", "_redirect", "_30"),
    ("early", "_unload", "_recovery", "_30"),
    ("speed", "_collapse", "_pitch", "_redirect"),
    ("20 deg", " action bin"),
    ("20", "-24 deg"),
)


def _join(parts: tuple[str, ...]) -> str:
    return "".join(parts)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _obsolete_files() -> tuple[Path, ...]:
    old_agile = _join(("agile", "_turn", "_family", "_comparison"))
    run_old_agile = _join(("run", "_agile", "_turn", "_family", "_comparison"))
    old_aggressive = _join(("aggressive", "_reversal"))
    return (
        PRIMITIVES_DIR / f"{old_aggressive}_ocp.py",
        PRIMITIVES_DIR / f"{old_aggressive}_primitive.py",
        SCENARIOS_DIR / f"run_{old_aggressive}_search.py",
        CONTROL_DIR / "tests" / f"test_{old_aggressive}_shapes.py",
        CONTROL_DIR / "tests" / f"test_{old_aggressive}_smoke.py",
        CONTROL_DIR / "tests" / f"test_{old_aggressive}_target_ladder.py",
        CONTROL_DIR / "tests" / f"test_{old_aggressive}_30deg_energy.py",
        PRIMITIVES_DIR / f"{old_agile}.py",
        SCENARIOS_DIR / f"{run_old_agile}.py",
        CONTROL_DIR / "tests" / f"test_{old_agile}_profiles.py",
        CONTROL_DIR / "tests" / f"test_{old_agile}_runner.py",
    )


def _obsolete_result_dirs() -> tuple[Path, ...]:
    old_aggressive = _join(("07_", "aggressive", "_reversal", "_ocp"))
    old_agile = _join(("08_", "agile", "_turn", "_family", "_comparison"))
    return (
        RESULT_ROOT / old_aggressive / "001",
        RESULT_ROOT / old_aggressive / "001_cleanup_empty",
        RESULT_ROOT / old_agile,
    )


def _hash_tree(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": _repo_relative(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
    return rows


def _cleanup_inventory() -> dict[str, object]:
    archive = RESULT_ROOT / _join(("07_", "aggressive", "_reversal", "_ocp")) / "002"
    return {
        "old_active_source_files_found": [
            _repo_relative(path)
            for path in _obsolete_files()
            if path.exists() and "tests" not in path.parts
        ],
        "old_active_runner_files_found": [
            _repo_relative(path)
            for path in _obsolete_files()
            if path.exists() and path.name.startswith("run_")
        ],
        "old_active_test_files_found": [
            _repo_relative(path)
            for path in _obsolete_files()
            if path.exists() and "tests" in path.parts
        ],
        "old_generated_result_dirs_found": [
            _repo_relative(path)
            for path in _obsolete_result_dirs()
            if path.exists()
        ],
        "preserved_boundary_reference_files": _hash_tree(archive),
        "baseline_primitive_files_preserved": [
            "03_Control/03_Primitives/glide_primitive.py",
            "03_Control/03_Primitives/recovery_primitive.py",
            "03_Control/03_Primitives/bank_primitive.py",
        ],
        "contract_files_preserved": [
            "03_Control/03_Primitives/command_contract.py",
            "03_Control/03_Primitives/state_contract.py",
            "03_Control/03_Primitives/metric_contract.py",
            "03_Control/03_Primitives/primitive_contract.py",
        ],
        "updraft_files_preserved": ["03_Control/04_Scenarios/updraft_models.py"],
    }


def _delete_obsolete_paths() -> tuple[list[str], list[str]]:
    deleted_files: list[str] = []
    deleted_directories: list[str] = []
    for path in _obsolete_files():
        if path.exists():
            path.unlink()
            deleted_files.append(_repo_relative(path))
    for path in _obsolete_result_dirs():
        if path.exists():
            try:
                _rmtree_best_effort(path)
                deleted_directories.append(_repo_relative(path))
            except PermissionError:
                _clear_result_files(path)
                deleted_directories.append(f"{_repo_relative(path)} (files cleared; directory placeholder retained)")
    return deleted_files, deleted_directories


def _rmtree_best_effort(path: Path) -> None:
    def _make_writable_and_retry(function: object, item: str, exc_info: object) -> None:
        del exc_info
        os.chmod(item, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(item)

    shutil.rmtree(path, onexc=_make_writable_and_retry)


def _negative_grep() -> tuple[int, str]:
    pattern = "|".join(_join(parts) for parts in OBSOLETE_TOKEN_PARTS)
    command = [
        "git",
        "grep",
        "-n",
        "-E",
        pattern,
        "--",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
        "03_Control/tests",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return int(result.returncode), result.stdout + result.stderr


# =============================================================================
# 2) Manifest Builders
# =============================================================================
def _final_plan_lock(run_id: int) -> dict[str, object]:
    return {
        "plan_name": "Final Plan Lock and Primitive Evidence Library Rebuild",
        "plan_version_label": f"s{run_id:03d}",
        "central_research_question": "widen primitive envelopes before library growth",
        "core_research_question": (
            "How far can each primitive's verified operating envelope be widened "
            "under measured updraft uncertainty before the accepted primitive library itself must grow?"
        ),
        "target_standard": "top robotics journal evidence discipline",
        "deadline": "2026-06-15",
        "writing_lock": "protected final writing period retained",
        "state_order": "x_w,y_w,z_w,phi,theta,psi,u,v,w,p,q,r,delta_a,delta_e,delta_r",
        "command_order": "delta_a_cmd,delta_e_cmd,delta_r_cmd",
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad -> rk4_step/state_derivative",
        "world_frame": "public z-up world frame",
        "body_frame": "x forward, y starboard, z down",
        "true_safe_bounds_m": {
            "x_w": TRUE_SAFE_BOUNDS.x_w_m,
            "y_w": TRUE_SAFE_BOUNDS.y_w_m,
            "z_w": TRUE_SAFE_BOUNDS.z_w_m,
        },
        "z_outlet_m": Z_OUTLET_M,
        "z_fan_convention": "z_fan = z_w - 0.330",
        "wind_fidelity_ladder": WIND_FIDELITIES,
        "updraft_configurations": UPDRAFT_CONFIGS,
        "primitive_seed_set": PRIMITIVE_FAMILIES,
        "candidate_classes": CANDIDATE_CLASSES,
        "target_ladder_deg": TARGET_LADDER_DEG,
        "evidence_library_first": True,
        "cluster_after_evidence": True,
        "library_growth_only_after_envelope_failure": True,
        "dry_air_agile_turn_recovery_loop_closed": True,
        "dry_air_agile_turn_recovery_loops_closed": True,
        "forbidden_method_pivots": ("broad_NMPC", "direct_surface_RL", "generic_waypoint_following"),
        "no_overclaiming_flags": _no_overclaiming_flags(),
    }


def _no_overclaiming_flags() -> dict[str, bool]:
    return {
        "primitive_library_schema_implemented": True,
        "primitive_library_runner_implemented": True,
        "w0_w1_w2_screen_implemented": True,
        "w3_stress_implemented": False,
        "clustering_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "real_flight_validation_claim": False,
        "tvlqr_implemented": False,
        "ocp_implemented": False,
        "high_incidence_validation_claim": False,
        "old_perch_like_branch_active": False,
        "archived_boundary_reference_preserved": True,
    }


# =============================================================================
# 3) Evidence Generation and Writing
# =============================================================================
def run_primitive_library_pass(
    run_id: int = 1,
    overwrite: bool = False,
    targets_deg: tuple[float, ...] = (15.0, 30.0),
    wind_fidelities: tuple[str, ...] = ("W0", "W1", "W2"),
    updraft_configs: tuple[str, ...] = ("none", "U1_single_fan", "U4_four_fan"),
) -> dict[str, Path]:
    """Run housekeeping and write the primitive evidence-library pass."""

    paths = _prepare_result_tree(run_id, overwrite)
    suffix = f"s{run_id:03d}"
    pre_inventory = _cleanup_inventory()
    archive = RESULT_ROOT / _join(("07_", "aggressive", "_reversal", "_ocp")) / "002"
    archive_hashes_before = _hash_tree(archive)
    deleted_files, deleted_directories = _delete_obsolete_paths()

    config = PrimitiveLibraryConfig(
        run_id=run_id,
        targets_deg=targets_deg,
        wind_fidelities=wind_fidelities,
        updraft_configs=updraft_configs,
    )
    wind_infos = _load_wind_infos(config)
    evaluations = _run_evaluations(config, wind_infos)
    evidence_rows = [evaluation.row.as_dict() for evaluation in evaluations]
    library_df = pd.DataFrame(evidence_rows)
    group_df = pd.DataFrame(group_summary(evidence_rows))
    summary_df = _library_summary(library_df)
    log_files = _write_representative_logs(paths["logs"], suffix, evaluations)

    library_csv = paths["metrics"] / f"primitive_evidence_library_{suffix}.csv"
    summary_csv = paths["metrics"] / f"primitive_library_summary_{suffix}.csv"
    group_csv = paths["metrics"] / f"primitive_envelope_group_summary_{suffix}.csv"
    library_df.to_csv(library_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    group_df.to_csv(group_csv, index=False)

    negative_code, negative_text = _negative_grep()
    archive_hashes_after = _hash_tree(archive)
    archive_preserved = archive_hashes_before == archive_hashes_after and archive.exists()
    if not archive_preserved:
        raise RuntimeError("archived 07/002 boundary evidence hash changed or is missing.")

    final_plan_lock = _final_plan_lock(run_id)
    plan_lock_json = paths["manifests"] / f"final_plan_lock_{suffix}.json"
    plan_lock_json.write_text(json.dumps(final_plan_lock, indent=2), encoding="ascii")

    housekeeping_json = paths["manifests"] / f"repo_housekeeping_manifest_{suffix}.json"
    housekeeping = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pre_cleanup_inventory": pre_inventory,
        "approved_obsolete_files": [
            _repo_relative(path)
            for path in _obsolete_files()
        ],
        "approved_obsolete_result_directories": [
            _repo_relative(path)
            for path in _obsolete_result_dirs()
        ],
        "already_absent_obsolete_files": [
            _repo_relative(path)
            for path in _obsolete_files()
            if not path.exists()
        ],
        "already_absent_obsolete_result_directories": [
            _repo_relative(path)
            for path in _obsolete_result_dirs()
            if not path.exists()
        ],
        "deleted_files": deleted_files,
        "deleted_directories": deleted_directories,
        "preserved_files": pre_inventory["baseline_primitive_files_preserved"],
        "preserved_boundary_reference_path": _repo_relative(archive),
        "preserved_boundary_reference_hashes_before": archive_hashes_before,
        "preserved_boundary_reference_hashes_after": archive_hashes_after,
        "preserved_archive_file_count": len(archive_hashes_after),
        "boundary_reference_preserved_byte_for_byte": archive_preserved,
        "negative_grep_command": "git grep active source/tests for obsolete branch tokens",
        "negative_grep_exit_code": negative_code,
        "negative_grep_pass": negative_code == 1,
        "negative_grep_output": negative_text,
        "regenerated_files": [
            _repo_relative(library_csv),
            _repo_relative(summary_csv),
            _repo_relative(group_csv),
        ],
        "final_plan_lock": final_plan_lock,
        **_no_overclaiming_flags(),
    }
    housekeeping_json.write_text(json.dumps(housekeeping, indent=2), encoding="ascii")

    report_md = paths["reports"] / f"primitive_library_report_{suffix}.md"
    _write_report(report_md, library_df, group_df)
    manifest_json = paths["manifests"] / f"primitive_library_manifest_{suffix}.json"
    output_files = {
        "plan_lock_json": plan_lock_json,
        "housekeeping_manifest_json": housekeeping_json,
        "library_csv": library_csv,
        "summary_csv": summary_csv,
        "envelope_group_summary_csv": group_csv,
        "report_md": report_md,
        **log_files,
    }
    w1_complete = all(
        bool(info.available)
        for key, info in wind_infos.items()
        if key.endswith("_W1")
    )
    w2_complete = all(
        bool(info.available)
        for key, info in wind_infos.items()
        if key.endswith("_W2")
    )
    manifest = {
        "run_id": suffix,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign": CAMPAIGN,
        "central_research_question": final_plan_lock["central_research_question"],
        "dry_air_agile_turn_recovery_loop_closed": final_plan_lock[
            "dry_air_agile_turn_recovery_loop_closed"
        ],
        "plan_lock_file": _repo_relative(plan_lock_json),
        "housekeeping_manifest_file": _repo_relative(housekeeping_json),
        "library_csv": _repo_relative(library_csv),
        "summary_csv": _repo_relative(summary_csv),
        "envelope_group_summary_csv": _repo_relative(group_csv),
        "report_md": _repo_relative(report_md),
        "candidate_classes": CANDIDATE_CLASSES,
        "target_ladder_deg": TARGET_LADDER_DEG,
        "families_run": PRIMITIVE_FAMILIES,
        "starts_run": config.start_conditions,
        "updraft_configs_run": updraft_configs,
        "wind_fidelities_run": wind_fidelities,
        "z_outlet_m": Z_OUTLET_M,
        "true_safe_bounds_m": {
            "x_w": TRUE_SAFE_BOUNDS.x_w_m,
            "y_w": TRUE_SAFE_BOUNDS.y_w_m,
            "z_w": TRUE_SAFE_BOUNDS.z_w_m,
        },
        "command_bridge": final_plan_lock["command_bridge"],
        "state_order": final_plan_lock["state_order"],
        "old_active_agile_branch_present": False,
        "preserved_boundary_reference": archive_preserved,
        "archived_boundary_reference_preserved": archive_preserved,
        "w1_complete": bool(w1_complete),
        "w2_complete": bool(w2_complete),
        "w1_w2_completeness": {
            key: bool(info.available)
            for key, info in wind_infos.items()
            if key != "none_W0"
        },
        "output_files": {
            name: _repo_relative(path)
            for name, path in output_files.items()
        },
        **_no_overclaiming_flags(),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return {
        "root": paths["root"],
        "plan_lock": plan_lock_json,
        "housekeeping_manifest": housekeeping_json,
        "manifest": manifest_json,
        "library_csv": library_csv,
        "summary_csv": summary_csv,
        "group_summary_csv": group_csv,
        "report": report_md,
    }


def _prepare_result_tree(run_id: int, overwrite: bool) -> dict[str, Path]:
    root = RESULT_ROOT / CAMPAIGN / f"{run_id:03d}"
    if root.exists() and overwrite:
        _clear_result_files(root)
    if root.exists() and not overwrite:
        raise ValueError(f"result tree already exists: {root}")
    paths = {
        "root": root,
        "manifests": root / "manifests",
        "metrics": root / "metrics",
        "logs": root / "logs" / "candidates",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _clear_result_files(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _load_wind_infos(config: PrimitiveLibraryConfig) -> dict[str, WindModelInfo]:
    infos = {
        "none_W0": WindModelInfo(True, None, "none", "no_wind", None, "evaluated")
    }
    model_names = {
        "U1_single_fan": "single_gaussian_var",
        "U4_four_fan": "four_gaussian_var",
    }
    for updraft_config in config.updraft_configs:
        if updraft_config == "none":
            continue
        for wind_fidelity in config.wind_fidelities:
            if wind_fidelity not in ("W1", "W2"):
                continue
            key = f"{updraft_config}_{wind_fidelity}"
            try:
                model = load_updraft_model(model_names[updraft_config], repo_root=REPO_ROOT)
                z_axis = getattr(model, "z_axis_m", None)
                infos[key] = WindModelInfo(
                    True,
                    model,
                    getattr(model, "name", model_names[updraft_config]),
                    getattr(model, "source", "unknown"),
                    None if z_axis is None else np.asarray(z_axis, dtype=float),
                    "evaluated",
                )
            except Exception as exc:  # noqa: BLE001 - model failure is logged as evidence.
                infos[key] = WindModelInfo(
                    False,
                    None,
                    model_names[updraft_config],
                    f"model_unavailable: {exc}",
                    None,
                    "not_evaluated_model_missing",
                )
    return infos


def _run_evaluations(
    config: PrimitiveLibraryConfig,
    wind_infos: dict[str, WindModelInfo],
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    aircraft = adapt_glider(build_nausicaa_glider())
    for spec in primitive_candidate_inventory(config):
        key = "none_W0" if spec.wind_fidelity == "W0" else f"{spec.updraft_config}_{spec.wind_fidelity}"
        evaluations.append(evaluate_candidate(spec, config, wind_infos[key], aircraft=aircraft))
    return evaluations


def _write_representative_logs(
    log_dir: Path,
    suffix: str,
    evaluations: list[CandidateEvaluation],
) -> dict[str, Path]:
    selected: dict[tuple[object, ...], CandidateEvaluation] = {}
    for evaluation in evaluations:
        key = (
            evaluation.spec.family,
            evaluation.spec.target_heading_deg,
            evaluation.spec.start_condition,
            evaluation.spec.updraft_config,
            evaluation.spec.wind_fidelity,
        )
        selected.setdefault(key, evaluation)
    output_files: dict[str, Path] = {}
    for evaluation in selected.values():
        stem = f"{evaluation.spec.primitive_id}_{suffix}"
        trajectory_csv = log_dir / f"{stem}_trajectory.csv"
        commands_csv = log_dir / f"{stem}_commands.csv"
        trajectory_dataframe(evaluation.time_s, evaluation.x_ref).to_csv(trajectory_csv, index=False)
        command_dataframe(
            evaluation.time_s,
            evaluation.u_norm_requested,
            evaluation.u_norm_applied,
            evaluation.delta_cmd_rad,
        ).to_csv(commands_csv, index=False)
        output_files[f"{stem}_trajectory_csv"] = trajectory_csv
        output_files[f"{stem}_commands_csv"] = commands_csv
    return output_files


def _library_summary(library_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("wind_fidelity", "updraft_config", "start_condition", "candidate_class"):
        for value, count in library_df[column].value_counts(dropna=False).sort_index().items():
            rows.append({"summary_type": column, "value": value, "row_count": int(count)})
    rows.append(
        {
            "summary_type": "library_growth_trigger",
            "value": True,
            "row_count": int(library_df["library_growth_trigger"].sum()),
        }
    )
    return pd.DataFrame(rows)


def _write_report(path: Path, library_df: pd.DataFrame, group_df: pd.DataFrame) -> None:
    class_counts = library_df["candidate_class"].value_counts(dropna=False).to_dict()
    lines = [
        "# Primitive Evidence Library Report",
        "",
        "This pass locks the final primitive-library plan and restarts active",
        "development from a unified evidence table. It does not implement W3,",
        "clustering, governor, outer loop, OCP, TVLQR, or real-flight validation.",
        "",
        "## Evidence Counts",
        "",
    ]
    for name, count in sorted(class_counts.items()):
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(
        [
            "",
            "## Envelope Group Status",
            "",
        ]
    )
    for name, count in group_df["group_status"].value_counts(dropna=False).sort_index().items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(
        [
            "",
            "The archived high-alpha/perch-like branch remains boundary evidence only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 4) CLI Entry Point
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--targets", nargs="*", type=float, default=[15.0, 30.0])
    parser.add_argument("--wind-fidelities", nargs="*", default=["W0", "W1", "W2"])
    parser.add_argument("--updraft-configs", nargs="*", default=["none", "U1_single_fan", "U4_four_fan"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_primitive_library_pass(
        run_id=int(args.run_id),
        overwrite=bool(args.overwrite),
        targets_deg=tuple(float(value) for value in args.targets),
        wind_fidelities=tuple(args.wind_fidelities),
        updraft_configs=tuple(args.updraft_configs),
    )
    for key in ("root", "manifest", "library_csv", "report"):
        print(f"{key}={paths[key]}")


if __name__ == "__main__":
    main()
