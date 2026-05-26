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
from dense_archive_table_io import file_sha256, filesystem_path  # noqa: E402
from run_v48_source_audit import (  # noqa: E402
    BLOCKED_CLAIMS as V48_BLOCKED_CLAIMS,
    DEFAULT_GOVERNOR_SMOKE_ROOT,
    DEFAULT_POST_W3_ROOT,
    DEFAULT_W01_ROOT,
    DEFAULT_W2_ROOT,
    DEFAULT_W3_ROOT,
    _audit_post_w3,
    _audit_w01,
    _audit_w2,
    _audit_w3,
    _expect,
    _read_json_or_empty,
    _source_file_size_status,
)


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.9"
SOURCE_AUDIT_VERSION = "v49_paired_full_loop_source_audit_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation/003")
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/outcome_model/002")
DEFAULT_FULL_LOOP_INFRASTRUCTURE_ROOTS = (
    Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation/001"),
    Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation/002"),
)
BLOCKED_CLAIMS = tuple(
    dict.fromkeys(
        [
            *V48_BLOCKED_CLAIMS,
            "full_autonomy",
            "sim_real_transfer",
            "memory_improves_performance_without_paired_v49_evidence",
        ]
    )
)


@dataclass(frozen=True)
class V49SourceAuditConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    w01_root: Path = DEFAULT_W01_ROOT
    w2_root: Path = DEFAULT_W2_ROOT
    w3_root: Path = DEFAULT_W3_ROOT
    post_w3_root: Path = DEFAULT_POST_W3_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    governor_smoke_root: Path = DEFAULT_GOVERNOR_SMOKE_ROOT
    full_loop_infrastructure_roots: tuple[Path, ...] = DEFAULT_FULL_LOOP_INFRASTRUCTURE_ROOTS


def run_v49_source_audit(config: V49SourceAuditConfig) -> dict[str, object]:
    """Audit frozen v4.7/v4.8 sources for paired v4.9 full-loop validation."""

    run_root = Path(config.output_root)
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    checks = [
        _audit_w01(config.w01_root),
        _audit_w2(config.w2_root, expected_w01_root=config.w01_root),
        _audit_w3(config.w3_root, expected_w2_root=config.w2_root),
        _audit_post_w3(config.post_w3_root, expected_w3_root=config.w3_root),
        _audit_outcome_model(config.outcome_root, expected_post_w3_root=config.post_w3_root),
        _audit_governor_smoke(config.governor_smoke_root, expected_post_w3_root=config.post_w3_root),
    ]
    checks.extend(_audit_full_loop_infrastructure(root) for root in config.full_loop_infrastructure_roots)

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
            "outcome_root": config.outcome_root.as_posix(),
            "governor_smoke_root": config.governor_smoke_root.as_posix(),
            "full_loop_infrastructure_roots": [root.as_posix() for root in config.full_loop_infrastructure_roots],
        },
        "checks": checks,
        "blockers": blockers,
        "v48_full_loop_roots_role": "infrastructure_evidence_not_final_memory_comparison",
        "rejects_smoke_fixture_diagnostic_superseded_sources": True,
        "claim_status": "simulation_only_source_audit_for_paired_full_loop_memory_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "source_audit.json", manifest)
    _write_csv(run_root / "metrics" / "source_audit_summary.csv", pd.DataFrame(checks))
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {"status": status, "blockers": blockers, "run_root": run_root.as_posix()}


def _audit_outcome_model(root: Path, *, expected_post_w3_root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "outcome_model_manifest.json")
    file_audit = _source_file_size_status(root)
    table = filesystem_path(root / "metrics" / "outcome_model_table.csv")
    row_count = 0 if not table.is_file() else int(len(pd.read_csv(table)))
    _expect(blockers, str(manifest.get("status", "")) == "complete", "outcome_model_002_status_not_complete")
    _expect(
        blockers,
        str(manifest.get("claim_status", "")) == "simulation_only_outcome_model_for_full_loop_validation",
        "outcome_model_002_claim_status_unexpected",
    )
    _expect(blockers, int(manifest.get("representative_count", 0)) == 29, "outcome_model_representative_count_not_29")
    _expect(blockers, int(manifest.get("outcome_row_count", 0)) == 29, "outcome_model_row_count_manifest_not_29")
    _expect(blockers, row_count == 29, "outcome_model_table_row_count_not_29")
    _expect(
        blockers,
        str(manifest.get("compact_library_path", "")).startswith(expected_post_w3_root.as_posix()),
        "outcome_model_source_compact_library_mismatch",
    )
    _expect(blockers, bool(manifest.get("continuation_and_terminal_evidence_separate", False)), "outcome_model_terminal_continuation_not_separate")
    _expect(blockers, not bool(manifest.get("controller_mutation_allowed", True)), "outcome_model_controller_mutation_allowed")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "outcome_model_source_file_size_audit_above_100mb")
    return {
        "stage": "outcome_model_002",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": row_count,
        "variant_count": int(manifest.get("representative_count", 0)),
        "ready_count": int(manifest.get("outcome_row_count", 0)),
        "survivor_count": 0,
        "representative_count": int(manifest.get("representative_count", 0)),
        "source_role": "frozen_advisory_outcome_model",
        **file_audit,
        "blockers": blockers,
    }


def _audit_governor_smoke(root: Path, *, expected_post_w3_root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "governor_smoke_manifest.json")
    file_audit = _source_file_size_status(root)
    _expect(blockers, str(manifest.get("status", "")) == "complete", "governor_smoke_status_not_complete")
    _expect(
        blockers,
        str(manifest.get("claim_status", "")) == "simulation_only_governor_smoke_no_performance_claim",
        "governor_smoke_claim_status_unexpected",
    )
    _expect(
        blockers,
        str(manifest.get("compact_library_path", "")).startswith(expected_post_w3_root.as_posix()),
        "governor_smoke_source_compact_library_mismatch",
    )
    _expect(blockers, sorted(manifest.get("governor_modes", [])) == ["continuation_mode", "terminal_episode_mode"], "governor_smoke_modes_incomplete")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "governor_smoke_source_file_size_audit_above_100mb")
    return {
        "stage": "governor_smoke_001",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": int(manifest.get("candidate_row_count", 0)),
        "variant_count": 0,
        "ready_count": int(manifest.get("selection_row_count", 0)),
        "representative_count": 0,
        "source_role": "infrastructure_smoke_for_governor_interface",
        **file_audit,
        "blockers": blockers,
    }


def _audit_full_loop_infrastructure(root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "full_loop_validation_manifest.json")
    file_audit = _source_file_size_status(root)
    _expect(blockers, str(manifest.get("status", "")) == "complete", f"full_loop_{root.name}_status_not_complete")
    _expect(
        blockers,
        str(manifest.get("claim_status", "")) == "simulation_only_full_loop_validation",
        f"full_loop_{root.name}_claim_status_unexpected",
    )
    _expect(blockers, not bool(manifest.get("controller_mutation_allowed", True)), f"full_loop_{root.name}_controller_mutation_allowed")
    _expect(blockers, not bool(manifest.get("retuning_allowed", True)), f"full_loop_{root.name}_retuning_allowed")
    _expect(blockers, int(manifest.get("episode_count", 0)) > 0, f"full_loop_{root.name}_episode_count_zero")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, f"full_loop_{root.name}_source_file_size_audit_above_100mb")
    return {
        "stage": f"full_loop_{root.name}",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": int(manifest.get("episode_count", 0)),
        "variant_count": 0,
        "ready_count": 0,
        "representative_count": int(manifest.get("compact_representative_count", 0)),
        "source_role": "infrastructure_evidence_not_final_memory_comparison",
        **file_audit,
        "blockers": blockers,
    }


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    blockers = list(manifest.get("blockers", []))
    lines = [
        "# v4.9 Source Audit",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Blocker count: `{len(blockers)}`",
        "- W01/W2/W3/post-W3/outcome/governor roots are frozen sources, not regenerated here.",
        "- Full-loop `001` and `002` are classified as infrastructure evidence, not final paired memory evidence.",
        "- Claim boundary: simulation-only paired full-loop memory validation source audit.",
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
        if not path.is_file() or path.name == "file_size_audit.csv":
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
    parser = argparse.ArgumentParser(description="Audit v4.9 paired full-loop validation source roots.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--w01-root", type=Path, default=DEFAULT_W01_ROOT)
    parser.add_argument("--w2-root", type=Path, default=DEFAULT_W2_ROOT)
    parser.add_argument("--w3-root", type=Path, default=DEFAULT_W3_ROOT)
    parser.add_argument("--post-w3-root", type=Path, default=DEFAULT_POST_W3_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--governor-smoke-root", type=Path, default=DEFAULT_GOVERNOR_SMOKE_ROOT)
    parser.add_argument("--allow-retired-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_retired_diagnostic:
        print(json.dumps({"status": "blocked", "blocked_reason": "retired_diagnostic_requires_explicit_allow_retired_diagnostic"}, indent=2, sort_keys=True))
        return 1
    result = run_v49_source_audit(
        V49SourceAuditConfig(
            output_root=args.output_root,
            w01_root=args.w01_root,
            w2_root=args.w2_root,
            w3_root=args.w3_root,
            post_w3_root=args.post_w3_root,
            outcome_root=args.outcome_root,
            governor_smoke_root=args.governor_smoke_root,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "source_audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
