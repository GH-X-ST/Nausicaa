from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from prim_cat import ACTIVE_PRIMITIVE_IDS, LAUNCH_CAPTURE_PRIMITIVE_IDS, active_primitive_catalogue  # noqa: E402
from primitive_variant_registry import ENTRY_ROLE_BY_PRIMITIVE_ID  # noqa: E402
from primitive_timing_contract import (  # noqa: E402
    PRIMITIVE_TIMING_CONTRACT_VERSION,
    primitive_timing_contract_row,
    primitive_timing_contract_status,
)
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.3"
SOURCE_AUDIT_VERSION = "v53_r5_only_launch_aware_source_audit"
REQUIRED_DOCS = (
    "Glider_Control_Project_Plan.md",
    "Skills.md",
    "Python Coding Instruction.txt",
    "Python Coding to CODEX.txt",
    "MATLAB Coding.txt",
    "housekeeping_and_naming_rules.md",
    "Daily_Schedule.txt",
    "R5_R10_Full_Evidence_Execution_Plan.md",
    "CODEX_R9_launch_gate_coverage_repair_guidance.md",
    "PR.txt",
)
RESULT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0")
ARCHIVE_ROOT = RESULT_ROOT / "archive" / "diagnostic_not_passed_v410" / "001"
ACTIVE_DIRS = ("02_Inner_Loop", "03_Primitives", "04_Scenarios")
RETIRED_NAME_FRAGMENTS = (
    "v48",
    "v49",
    "v410",
    "full_loop_validation",
    "governor_calibration",
    "validation_figures",
    "post_w3_cluster_merge",
    "post_w3_governor_smoke",
    "episodic_lift_belief",
)
SUPERSEDED_PATH_FRAGMENTS = (
    "full_loop_validation",
    "governor_calibration",
    "post_w3_cluster",
    "v48",
    "v49",
    "v410",
)
SUPERSEDED_TEXT_FRAGMENTS = (
    "LQR-Stabilised Contextual Primitive v4.5",
    "LQR-Stabilised Contextual Primitive v4.7",
    "LQR-Stabilised Contextual Primitive v4.8",
    "LQR-Stabilised Contextual Primitive v4.9",
    "LQR-Stabilised Contextual Primitive v4.10",
    "run_v48",
    "run_v49",
    "run_v410",
    "post_w3_compact_representative_library_v1",
    "full_loop_validation",
    "governor_calibration",
)
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "full_autonomy",
    "memory_improvement",
)


@dataclass(frozen=True)
class SourceAuditConfig:
    repo_root: Path = Path(".")
    docs_root: Path = Path("docs")
    control_root: Path = Path("03_Control")
    result_root: Path = RESULT_ROOT
    archive_root: Path = ARCHIVE_ROOT
    write_archive: bool = True
    dry_run: bool = False


def run_v411_source_audit(config: SourceAuditConfig) -> dict[str, object]:
    repo_root = Path(config.repo_root)
    docs = inspect_required_docs(repo_root / config.docs_root)
    inventory = build_control_inventory(repo_root / config.control_root)
    superseded_roots = discover_superseded_result_roots(repo_root / config.result_root)
    archive_manifest = None
    if config.write_archive:
        archive_manifest = write_diagnostic_not_passed_archive(
            archive_root=repo_root / config.archive_root,
            docs=docs,
            inventory=inventory,
            superseded_roots=superseded_roots,
        )
    blockers = _audit_blockers(
        repo_root=repo_root,
        docs=docs,
        inventory=inventory,
        superseded_roots=superseded_roots,
        archive_manifest=archive_manifest,
        archive_root=repo_root / config.archive_root,
    )
    status = "ready" if not blockers else "blocked"
    payload = {
        "manifest_version": SOURCE_AUDIT_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "dry_run": bool(config.dry_run),
        "required_doc_count": len(REQUIRED_DOCS),
        "missing_docs": [row["doc_name"] for row in docs if not row["readable"]],
        "inventory_count": len(inventory),
        "active_path_count": sum(row["classification"].startswith("active_") for row in inventory),
        "retired_path_count": sum(row["classification"] == "retired_not_active" for row in inventory),
        "superseded_result_root_count": len(superseded_roots),
        "archive_root": (repo_root / config.archive_root).as_posix(),
        "archive_manifest": None if archive_manifest is None else archive_manifest.get("manifest_path", ""),
        "blockers": blockers,
        "primitive_timing_contract": primitive_timing_contract_row(),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    if not config.dry_run:
        _write_json(repo_root / config.archive_root / "v411_source_audit_result.json", payload)
    return payload


def inspect_required_docs(docs_root: Path) -> list[dict[str, object]]:
    rows = []
    for name in REQUIRED_DOCS:
        path = Path(docs_root) / name
        try:
            text = path.read_text(encoding="utf-8")
            rows.append(
                {
                    "doc_name": name,
                    "relative_path": path.as_posix(),
                    "readable": True,
                    "byte_count": int(path.stat().st_size),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "doc_name": name,
                    "relative_path": path.as_posix(),
                    "readable": False,
                    "byte_count": 0,
                    "sha256": "",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
    return rows


def build_control_inventory(control_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = Path(control_root)
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rows.append(
            {
                "relative_path": rel,
                "classification": classify_control_file(rel),
                "byte_count": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    return rows


def classify_control_file(relative_path: str) -> str:
    rel = relative_path.replace("\\", "/")
    lower = rel.lower()
    if "/05_results/" in f"/{lower}/" or lower.startswith("05_results/"):
        return "generated_result"
    if lower.startswith("99_archive/"):
        return "retired_not_active"
    if "/tests/" in f"/{lower}/" or lower.startswith("tests/"):
        return "test_fixture"
    if "scratch" in lower or lower.endswith(".tmp"):
        return "local_only_scratch"
    if any(fragment in lower for fragment in RETIRED_NAME_FRAGMENTS):
        return "retired_not_active"
    if lower.startswith("02_inner_loop/") or lower.endswith("state_sampling.py"):
        return "active_foundation"
    if lower.startswith("03_primitives/") or lower.startswith("04_scenarios/"):
        return "active_repair_cycle"
    return "active_foundation"


def discover_superseded_result_roots(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = Path(result_root)
    if not root.exists():
        return rows
    for candidate in sorted(root.glob("*/*")):
        if not candidate.is_dir():
            continue
        rel = candidate.relative_to(root).as_posix()
        if rel.startswith("archive/"):
            continue
        reason = _superseded_result_reason(candidate, rel)
        if reason:
            rows.append(
                {
                    "result_root": candidate.as_posix(),
                    "result_root_relative": rel,
                    "diagnostic_archive_status": "diagnostic_not_passed",
                    "rejection_reason": reason,
                    "byte_count": _tree_size_bytes(candidate),
                    "file_count": _tree_file_count(candidate),
                }
            )
    return rows


def write_diagnostic_not_passed_archive(
    *,
    archive_root: Path,
    docs: list[dict[str, object]],
    inventory: list[dict[str, object]],
    superseded_roots: list[dict[str, object]],
) -> dict[str, object]:
    root = Path(archive_root)
    for subdir in ("manifests", "metrics", "reports"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    _write_csv(root / "metrics" / "required_docs_audit.csv", docs)
    _write_csv(root / "metrics" / "active_retired_control_inventory.csv", inventory)
    _write_csv(root / "archived_result_roots.csv", superseded_roots)
    _write_csv(root / "file_size_audit.csv", _file_size_audit_rows(root))
    _write_csv(root / "metrics" / "file_size_audit.csv", _file_size_audit_rows(root))
    manifest = {
        "manifest_version": "diagnostic_not_passed_v410_archive_manifest_v5",
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "diagnostic_not_passed",
        "archive_root": root.as_posix(),
        "superseded_result_root_count": len(superseded_roots),
        "required_docs": docs,
        "inventory_counts": _inventory_counts(inventory),
        "archived_result_roots_csv": (root / "archived_result_roots.csv").as_posix(),
        "file_size_audit_csv": (root / "metrics" / "file_size_audit.csv").as_posix(),
        "claim_status": "diagnostic_archive_only_not_v5_validation_evidence",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "manifests" / "diagnostic_not_passed_manifest.json", manifest)
    (root / "reports" / "diagnostic_not_passed_report.md").write_text(
        "\n".join(
            [
                "# Diagnostic Not Passed Archive",
                "",
                f"- Status: `{manifest['status']}`",
                f"- Superseded roots: `{len(superseded_roots)}`",
                "- Evidence in this archive is rejected for v5.3 launch-aware R5 claims.",
                "- No hardware-readiness, real-flight transfer, mission-success, autonomy, or memory-improvement claim is allowed.",
                "",
            ]
        ),
        encoding="ascii",
    )
    manifest["manifest_path"] = (root / "manifest.json").as_posix()
    return manifest


def _audit_blockers(
    *,
    repo_root: Path,
    docs: list[dict[str, object]],
    inventory: list[dict[str, object]],
    superseded_roots: list[dict[str, object]],
    archive_manifest: dict[str, object] | None,
    archive_root: Path,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    missing_docs = [row["doc_name"] for row in docs if not row["readable"]]
    if missing_docs:
        blockers.append({"blocker_id": "missing_required_docs", "details": ",".join(missing_docs)})
    blockers.extend(_primitive_timing_blockers())
    blockers.extend(_launch_aware_catalogue_blockers())
    blockers.extend(_active_source_blockers(repo_root=repo_root, inventory=inventory))
    if archive_manifest is None and not (archive_root / "manifest.json").is_file():
        blockers.append({"blocker_id": "missing_diagnostic_not_passed_archive", "details": archive_root.as_posix()})
    if not superseded_roots:
        blockers.append({"blocker_id": "no_rejected_evidence_discovered", "details": "expected archived v4.10-style evidence"})
    return blockers


def _primitive_timing_blockers() -> list[dict[str, object]]:
    blockers = []
    for primitive in active_primitive_catalogue():
        status, reason = primitive_timing_contract_status(
            finite_horizon_s=primitive.finite_horizon_s,
            controller_input_slots_per_primitive=primitive.controller_input_slots_per_primitive,
            controller_input_update_period_s=primitive.controller_input_update_period_s,
            primitive_timing_contract_version=primitive.primitive_timing_contract_version,
        )
        if status != "compliant":
            blockers.append(
                {
                    "blocker_id": "active_primitive_timing_contract_noncompliant",
                    "details": f"{primitive.primitive_id}:{reason}",
                }
            )
    return blockers


def _launch_aware_catalogue_blockers() -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    required_launch_capture = {
        "launch_capture_glide_stabilise",
        "launch_capture_lift_seek",
        "launch_capture_energy_build",
        "launch_capture_shallow_left",
        "launch_capture_shallow_right",
        "launch_capture_safe_handoff",
    }
    if len(ACTIVE_PRIMITIVE_IDS) != 14:
        blockers.append({"blocker_id": "active_primitive_count_not_14", "details": str(len(ACTIVE_PRIMITIVE_IDS))})
    if set(LAUNCH_CAPTURE_PRIMITIVE_IDS) != required_launch_capture:
        blockers.append(
            {
                "blocker_id": "launch_capture_ids_not_exact_required_set",
                "details": ",".join(sorted(set(LAUNCH_CAPTURE_PRIMITIVE_IDS) ^ required_launch_capture)),
            }
        )
    for primitive_id in sorted(required_launch_capture):
        if primitive_id not in ACTIVE_PRIMITIVE_IDS:
            blockers.append({"blocker_id": "launch_capture_id_missing_from_active_catalogue", "details": primitive_id})
        if ENTRY_ROLE_BY_PRIMITIVE_ID.get(primitive_id) != "launch_capable":
            blockers.append({"blocker_id": "launch_capture_entry_role_not_launch_capable", "details": primitive_id})
    for primitive_id in ("glide", "lift_entry", "lift_dwell_arc", "mild_turn_left", "mild_turn_right", "energy_retaining_bank"):
        if ENTRY_ROLE_BY_PRIMITIVE_ID.get(primitive_id) != "inflight_only":
            blockers.append({"blocker_id": "inflight_primitive_relabelled_launch_capable", "details": primitive_id})
    for primitive_id in ("recovery", "safe_exit_or_recovery_handoff"):
        if ENTRY_ROLE_BY_PRIMITIVE_ID.get(primitive_id) != "terminal_or_recovery":
            blockers.append({"blocker_id": "recovery_primitive_entry_role_changed", "details": primitive_id})
    return blockers


def _active_source_blockers(*, repo_root: Path, inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    active_paths = [
        repo_root / "03_Control" / str(row["relative_path"])
        for row in inventory
        if str(row["classification"]).startswith("active_")
    ]
    for path in active_paths:
        text = _read_text_or_empty(path)
        rel = path.relative_to(repo_root).as_posix()
        if rel.endswith("run_v411_source_audit.py"):
            continue
        if "LiftBeliefGrid" in text or "episodic_lift_belief" in text:
            blockers.append({"blocker_id": "active_2d_memory_reference", "details": rel})
        if "run_v48_source_audit" in text or "run_v49_source_audit" in text or "run_v410_source_audit" in text:
            blockers.append({"blocker_id": "active_old_validation_import", "details": rel})
        if "run_full_loop_validation" in text or "run_governor_calibration" in text:
            blockers.append({"blocker_id": "active_retired_validation_or_calibration_import", "details": rel})
        if "post_w3_cluster" in text and "run_post_w3_library_size_study" not in rel:
            blockers.append({"blocker_id": "active_single_post_w3_library_reference", "details": rel})
    governor = _read_text_or_empty(repo_root / "03_Control/04_Scenarios/viability_governor.py")
    if "exploration_bonus_weight=0.0," in governor or "applied_after_viability_filter" not in governor:
        blockers.append({"blocker_id": "pure_exploitation_without_safe_exploration", "details": "viability_governor.py"})
    if set(LIBRARY_SIZE_CASE_IDS) != {"heavy_cluster", "balanced_cluster", "light_cluster", "no_cluster_no_merge"}:
        blockers.append({"blocker_id": "missing_four_library_size_cases", "details": ",".join(LIBRARY_SIZE_CASE_IDS)})
    return blockers


def _superseded_result_reason(root: Path, relative: str) -> str:
    lower_rel = relative.lower()
    reasons = [f"path:{fragment}" for fragment in SUPERSEDED_PATH_FRAGMENTS if fragment in lower_rel]
    manifest_paths = sorted((root / "manifests").glob("*.json")) if (root / "manifests").is_dir() else []
    manifest_paths.extend(sorted(root.glob("*.json")))
    metadata_text = []
    for path in manifest_paths[:20]:
        text = _read_text_or_empty(path)
        metadata_text.append(text)
        reasons.extend(f"manifest:{fragment}" for fragment in SUPERSEDED_TEXT_FRAGMENTS if fragment in text)
    combined_metadata = "\n".join(metadata_text)
    if metadata_text and PRIMITIVE_TIMING_CONTRACT_VERSION not in combined_metadata and "v411" not in combined_metadata.lower():
        reasons.append("manifest_missing_v411_timing_contract")
    if _root_has_long_horizon_tables(root):
        reasons.append("table_long_primitive_horizon")
    return ";".join(sorted(set(reasons)))


def _root_has_long_horizon_tables(root: Path) -> bool:
    metric_root = root / "metrics"
    if not metric_root.is_dir():
        return False
    for path in sorted(metric_root.glob("*.csv"))[:25]:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader):
                    if index > 200:
                        break
                    for key in ("finite_horizon_s", "variant_finite_horizon_s"):
                        if key in row and row[key]:
                            try:
                                value = float(row[key])
                            except ValueError:
                                continue
                            if abs(value - 0.100) > 1e-9:
                                return True
        except Exception:
            continue
    return False


def _file_size_audit_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        byte_count = int(path.stat().st_size)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": byte_count / float(1024 * 1024),
                "push_allowed": byte_count <= 100 * 1024 * 1024,
            }
        )
    return rows


def _inventory_counts(inventory: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in inventory:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _tree_size_bytes(root: Path) -> int:
    return sum(int(path.stat().st_size) for path in root.rglob("*") if path.is_file())


def _tree_file_count(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict v4.11 source audit rerun-readiness gate.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--docs-root", type=Path, default=Path("docs"))
    parser.add_argument("--control-root", type=Path, default=Path("03_Control"))
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-write-archive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_v411_source_audit(
        SourceAuditConfig(
            repo_root=args.repo_root,
            docs_root=args.docs_root,
            control_root=args.control_root,
            result_root=args.result_root,
            archive_root=args.archive_root,
            dry_run=args.dry_run,
            write_archive=not args.no_write_archive,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
