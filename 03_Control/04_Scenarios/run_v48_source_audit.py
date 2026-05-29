from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import file_sha256, filesystem_path, load_table_manifest  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.8"
SOURCE_AUDIT_VERSION = "v48_full_loop_source_audit_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/R11_validation/001")
DEFAULT_W01_ROOT = Path("03_Control/05_Results/R5_dense/015")
DEFAULT_W2_ROOT = Path("03_Control/05_Results/R6_archived/015")
DEFAULT_W3_ROOT = Path("03_Control/05_Results/R7_survival/013")
DEFAULT_POST_W3_ROOT = Path("03_Control/05_Results/R8_library_size_study/001")
DEFAULT_OUTCOME_SMOKE_ROOT = Path("03_Control/05_Results/R8_outcome/001")
DEFAULT_GOVERNOR_SMOKE_ROOT = Path("03_Control/05_Results/B_runtime_smoke/001")
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "memory_improves_performance_without_full_loop_evidence",
    "formal_LQR_tree_funnel_region_of_attraction",
    "post_hoc_controller_retuning",
)


@dataclass(frozen=True)
class V48SourceAuditConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    w01_root: Path = DEFAULT_W01_ROOT
    w2_root: Path = DEFAULT_W2_ROOT
    w3_root: Path = DEFAULT_W3_ROOT
    post_w3_root: Path = DEFAULT_POST_W3_ROOT
    outcome_smoke_root: Path = DEFAULT_OUTCOME_SMOKE_ROOT
    governor_smoke_root: Path = DEFAULT_GOVERNOR_SMOKE_ROOT


def run_v48_source_audit(config: V48SourceAuditConfig) -> dict[str, object]:
    """Audit that the v4.8 pass is consuming complete v4.7 source evidence."""

    run_root = Path(config.output_root)
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    checks = [
        _audit_w01(config.w01_root),
        _audit_w2(config.w2_root, expected_w01_root=config.w01_root),
        _audit_w3(config.w3_root, expected_w2_root=config.w2_root),
        _audit_post_w3(config.post_w3_root, expected_w3_root=config.w3_root),
        _audit_smoke_root(config.outcome_smoke_root, "outcome_model_manifest.json", "simulation_only_outcome_model_smoke"),
        _audit_smoke_root(config.governor_smoke_root, "governor_smoke_manifest.json", "simulation_only_governor_smoke_no_performance_claim"),
    ]
    blockers = [reason for check in checks for reason in check["blockers"]]
    status = "source_audit_pass" if not blockers else "blocked_source_audit_failed"
    manifest = {
        "manifest_version": SOURCE_AUDIT_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "source_roots": {
            "w01_root": config.w01_root.as_posix(),
            "w2_root": config.w2_root.as_posix(),
            "w3_root": config.w3_root.as_posix(),
            "post_w3_root": config.post_w3_root.as_posix(),
            "outcome_smoke_root": config.outcome_smoke_root.as_posix(),
            "governor_smoke_root": config.governor_smoke_root.as_posix(),
        },
        "checks": checks,
        "blockers": blockers,
        "claim_status": "simulation_only_source_audit_for_full_loop_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "source_audit.json", manifest)
    _write_csv(run_root / "metrics" / "source_audit_summary.csv", pd.DataFrame(checks))
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {"status": status, "blockers": blockers, "run_root": run_root.as_posix()}


def _audit_w01(root: Path) -> dict[str, object]:
    blockers: list[str] = []
    run_manifest = _read_json_or_empty(root / "manifests" / "run_manifest.json")
    bundle = _read_json_or_empty(root / "manifests" / "frozen_w01_controller_bundle.json")
    row_count = _table_row_count(root)
    file_audit = _source_file_size_status(root)
    start_counts = dict(run_manifest.get("per_start_family_row_counts", {}))
    expected_start_counts = {
        "launch_gate": 30720,
        "inflight_nominal": 19200,
        "inflight_lift_region": 11520,
        "inflight_boundary_near": 7680,
        "inflight_recovery_edge": 7680,
    }
    _expect(blockers, filesystem_path(root).is_dir(), "missing_W01_015_root")
    _expect(blockers, int(run_manifest.get("rows_requested", 0)) == 76800, "W01_rows_requested_not_76800")
    _expect(blockers, int(row_count) == 76800, "W01_table_row_count_not_76800")
    _expect(blockers, int(run_manifest.get("candidate_count", 0)) == 32, "W01_candidate_count_not_32")
    _expect(blockers, int(run_manifest.get("paired_tests_per_candidate", 0)) == 100, "W01_paired_tests_not_100")
    _expect(blockers, start_counts == expected_start_counts, "W01_start_family_mix_not_exact_40_25_15_10_10")
    _expect(blockers, bool(run_manifest.get("no_small_library_selection", False)), "W01_small_library_selection_not_disabled")
    _expect(blockers, int(bundle.get("ready_count", 0)) == 256, "W01_ready_frozen_controller_count_not_256")
    _expect(blockers, int(bundle.get("variant_count", 0)) == 256, "W01_variant_count_not_256")
    _expect(blockers, not bool(bundle.get("physical_K_only_active_replay_allowed", True)), "W01_physical_K_only_replay_allowed")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "W01_source_file_size_audit_above_100mb")
    return {
        "stage": "W01",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": int(row_count),
        "variant_count": int(bundle.get("variant_count", 0)),
        "ready_count": int(bundle.get("ready_count", 0)),
        "representative_count": 0,
        **file_audit,
        "blockers": blockers,
    }


def _audit_w2(root: Path, *, expected_w01_root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "w2_survival_manifest.json")
    registry = _read_json_or_empty(root / "manifests" / "w2_survivor_registry.json")
    row_count = _table_row_count(root)
    file_audit = _source_file_size_status(root)
    _expect(blockers, str(manifest.get("status", "")) == "w2_dense_survival_pass", "W2_status_not_dense_survival_pass")
    _expect(blockers, int(row_count) == 51200, "W2_table_row_count_not_51200")
    _expect(blockers, int(manifest.get("variant_count", 0)) == 256, "W2_variant_count_not_256")
    _expect(blockers, int(manifest.get("ready_controller_count", 0)) == 256, "W2_ready_controller_count_not_256")
    _expect(blockers, sorted(manifest.get("w2_environment_modes", [])) == ["annular_gp_four", "annular_gp_single"], "W2_environment_modes_not_single_four_annular")
    _expect(blockers, bool(manifest.get("fixed_lqr_replay_only", False)), "W2_not_fixed_lqr_replay_only")
    _expect(blockers, not bool(manifest.get("mutates_Q_R_K_reference_horizon_entry_set_or_entry_role", True)), "W2_mutation_flag_true")
    _expect(blockers, str(manifest.get("source_w01_root", "")) == expected_w01_root.as_posix(), "W2_source_W01_root_mismatch")
    _expect(blockers, str(registry.get("status", "")) == "survived_variants_available", "W2_survivor_registry_not_available")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "W2_source_file_size_audit_above_100mb")
    return {
        "stage": "W2",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": int(row_count),
        "variant_count": int(manifest.get("variant_count", 0)),
        "ready_count": int(manifest.get("ready_controller_count", 0)),
        "survivor_count": int(registry.get("survivor_count", 0)),
        "representative_count": 0,
        **file_audit,
        "blockers": blockers,
    }


def _audit_w3(root: Path, *, expected_w2_root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "w3_survival_manifest.json")
    registry = _read_json_or_empty(root / "manifests" / "w3_survivor_registry.json")
    row_count = _table_row_count(root)
    file_audit = _source_file_size_status(root)
    _expect(blockers, str(manifest.get("status", "")) == "complete", "W3_status_not_complete")
    _expect(blockers, int(row_count) == 51200, "W3_table_row_count_not_51200")
    _expect(blockers, str(manifest.get("input_root", "")) == expected_w2_root.as_posix(), "W3_source_W2_root_mismatch")
    _expect(blockers, bool(manifest.get("fixed_lqr_replay_only", False)), "W3_not_fixed_lqr_replay_only")
    _expect(blockers, not bool(manifest.get("mutates_Q_R_K_reference_horizon_entry_set_or_entry_role", True)), "W3_mutation_flag_true")
    _expect(blockers, str(registry.get("status", "")) == "w3_survivors_available", "W3_survivor_registry_not_available")
    _expect(blockers, int(registry.get("survivor_count", 0)) > 0, "W3_survivor_count_zero")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "W3_source_file_size_audit_above_100mb")
    return {
        "stage": "W3",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": int(row_count),
        "variant_count": int(manifest.get("survivor_count", 0)),
        "ready_count": int(manifest.get("survivor_count", 0)),
        "survivor_count": int(registry.get("survivor_count", 0)),
        "representative_count": 0,
        **file_audit,
        "blockers": blockers,
    }


def _audit_post_w3(root: Path, *, expected_w3_root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "post_w3_cluster_manifest.json")
    library = _read_json_or_empty(root / "manifests" / "final_compact_primitive_library.json")
    representatives = list(library.get("representatives", []))
    file_audit = _source_file_size_status(root)
    _expect(blockers, str(manifest.get("status", "")) == "complete", "post_W3_cluster_status_not_complete")
    _expect(blockers, str(library.get("source_w3_root", "")) == expected_w3_root.as_posix(), "post_W3_source_W3_root_mismatch")
    _expect(blockers, int(library.get("representative_count", 0)) == 29, "post_W3_representative_count_not_29")
    _expect(blockers, bool(library.get("no_controller_mutation", False)), "post_W3_no_controller_mutation_false")
    _expect(blockers, bool(library.get("continuation_and_terminal_evidence_separate", False)), "post_W3_terminal_continuation_not_separate")
    for representative in representatives:
        if str(representative.get("w3_variant_status", "")) != "survived":
            blockers.append("post_W3_executable_representative_not_survived")
            break
        for key in ("controller_id", "primitive_variant_id", "K_gain_checksum", "augmented_gain_checksum", "source_w2_root"):
            if not representative.get(key):
                blockers.append(f"post_W3_representative_missing_{key}")
                break
    l10 = filesystem_path(root / "reports" / "l10_move_on_check.md")
    _expect(blockers, l10.is_file() and "Governor smoke allowed: `True`" in l10.read_text(encoding="ascii"), "post_W3_l10_does_not_allow_governor_smoke")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "post_W3_source_file_size_audit_above_100mb")
    return {
        "stage": "post_W3",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": 0,
        "variant_count": 0,
        "ready_count": 0,
        "survivor_count": int(manifest.get("source_w3_survivor_count", 0)),
        "representative_count": int(library.get("representative_count", 0)),
        **file_audit,
        "blockers": blockers,
    }


def _audit_smoke_root(root: Path, manifest_name: str, expected_claim_status: str) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / manifest_name)
    file_audit = _source_file_size_status(root)
    _expect(blockers, str(manifest.get("status", "")) == "complete", f"{manifest_name}_status_not_complete")
    _expect(blockers, str(manifest.get("claim_status", "")) == expected_claim_status, f"{manifest_name}_claim_status_unexpected")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, f"{manifest_name}_source_file_size_audit_above_100mb")
    return {
        "stage": manifest_name.replace("_manifest.json", ""),
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": 0,
        "variant_count": 0,
        "ready_count": 0,
        "representative_count": int(manifest.get("representative_count", 0)),
        **file_audit,
        "blockers": blockers,
    }


def _table_row_count(root: Path) -> int:
    manifest_path = root / "manifests" / "table_manifest.json"
    if not filesystem_path(manifest_path).is_file():
        return 0
    manifest = load_table_manifest(manifest_path)
    return int(sum(int(partition.row_count) for partition in manifest.tables))


def _source_file_size_status(root: Path) -> dict[str, object]:
    audit_path = filesystem_path(root / "metrics" / "file_size_audit.csv")
    if not audit_path.is_file():
        return {
            "file_size_audit_status": "missing_file_size_audit",
            "largest_source_file_size_mb": 0.0,
            "source_file_above_100mb_count": 0,
        }
    frame = pd.read_csv(audit_path)
    if frame.empty:
        return {
            "file_size_audit_status": "empty_file_size_audit",
            "largest_source_file_size_mb": 0.0,
            "source_file_above_100mb_count": 0,
        }
    above_100 = int(frame["above_100mb"].astype(bool).sum()) if "above_100mb" in frame.columns else 0
    return {
        "file_size_audit_status": "below_100mb" if above_100 == 0 else "above_100mb_blocked",
        "largest_source_file_size_mb": float(pd.to_numeric(frame["size_mb"], errors="coerce").fillna(0.0).max()),
        "source_file_above_100mb_count": above_100,
    }


def _read_json_or_empty(path: Path) -> dict[str, object]:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return {}
    return json.loads(fs_path.read_text(encoding="ascii"))


def _expect(blockers: list[str], condition: bool, reason: str) -> None:
    if not bool(condition):
        blockers.append(str(reason))


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    blockers = list(manifest.get("blockers", []))
    lines = [
        "# v4.8 Source Audit",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Blocker count: `{len(blockers)}`",
        "- Outcome/governor `001` roots are smoke-only and are not treated as full-loop evidence.",
        "- Claim boundary: simulation-only source audit; no upstream rerun or controller redesign.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{reason}`" for reason in blockers] or ["- `none`"])
    lines.append("")
    filesystem_path(run_root / "reports" / "source_audit.md").write_text("\n".join(lines), encoding="ascii")


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_fs).as_posix()
        byte_count = int(path.stat().st_size)
        size_mb = float(byte_count) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                "dense_table_partition": False,
                "sha256": file_sha256(path),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v4.8 full-loop source roots.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--w01-root", type=Path, default=DEFAULT_W01_ROOT)
    parser.add_argument("--w2-root", type=Path, default=DEFAULT_W2_ROOT)
    parser.add_argument("--w3-root", type=Path, default=DEFAULT_W3_ROOT)
    parser.add_argument("--post-w3-root", type=Path, default=DEFAULT_POST_W3_ROOT)
    parser.add_argument("--outcome-smoke-root", type=Path, default=DEFAULT_OUTCOME_SMOKE_ROOT)
    parser.add_argument("--governor-smoke-root", type=Path, default=DEFAULT_GOVERNOR_SMOKE_ROOT)
    parser.add_argument("--allow-retired-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_retired_diagnostic:
        print(json.dumps({"status": "blocked", "blocked_reason": "retired_diagnostic_requires_explicit_allow_retired_diagnostic"}, indent=2, sort_keys=True))
        return 1
    result = run_v48_source_audit(
        V48SourceAuditConfig(
            output_root=args.output_root,
            w01_root=args.w01_root,
            w2_root=args.w2_root,
            w3_root=args.w3_root,
            post_w3_root=args.post_w3_root,
            outcome_smoke_root=args.outcome_smoke_root,
            governor_smoke_root=args.governor_smoke_root,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "source_audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
