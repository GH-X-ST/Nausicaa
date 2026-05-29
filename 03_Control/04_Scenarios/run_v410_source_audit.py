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
from run_v49_source_audit import (  # noqa: E402
    BLOCKED_CLAIMS as V49_BLOCKED_CLAIMS,
    DEFAULT_GOVERNOR_SMOKE_ROOT,
    DEFAULT_OUTCOME_ROOT,
    DEFAULT_POST_W3_ROOT,
    DEFAULT_W01_ROOT,
    DEFAULT_W2_ROOT,
    DEFAULT_W3_ROOT,
    _audit_governor_smoke,
    _audit_outcome_model,
)
from run_v48_source_audit import (  # noqa: E402
    _audit_post_w3,
    _audit_w01,
    _audit_w2,
    _audit_w3,
    _expect,
    _read_json_or_empty,
    _source_file_size_status,
)


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.10"
SOURCE_AUDIT_VERSION = "v410_governor_calibration_source_audit_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/R11_validation/006")
DEFAULT_FULL_LOOP_ROOT = Path("03_Control/05_Results/R11_validation/005")
BLOCKED_CLAIMS = tuple(
    dict.fromkeys(
        [
            *V49_BLOCKED_CLAIMS,
            "W01_W2_W3_rerun_superiority",
            "memory_improvement_without_paired_heldout_evidence",
        ]
    )
)


@dataclass(frozen=True)
class V410SourceAuditConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    w01_root: Path = DEFAULT_W01_ROOT
    w2_root: Path = DEFAULT_W2_ROOT
    w3_root: Path = DEFAULT_W3_ROOT
    post_w3_root: Path = DEFAULT_POST_W3_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    governor_smoke_root: Path = DEFAULT_GOVERNOR_SMOKE_ROOT
    full_loop_root: Path = DEFAULT_FULL_LOOP_ROOT


def run_v410_source_audit(config: V410SourceAuditConfig) -> dict[str, object]:
    """Audit frozen source evidence for v4.10 governor calibration."""

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
        _audit_full_loop_005(config.full_loop_root),
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
            "outcome_root": config.outcome_root.as_posix(),
            "governor_smoke_root": config.governor_smoke_root.as_posix(),
            "full_loop_root": config.full_loop_root.as_posix(),
        },
        "checks": checks,
        "blockers": blockers,
        "source_policy": "frozen_sources_only_no_W01_W2_W3_rerun_or_mutation",
        "claim_status": "simulation_only_source_audit_for_governor_calibration",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "source_audit.json", manifest)
    _write_csv(run_root / "metrics" / "source_audit_summary.csv", pd.DataFrame(checks))
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {"status": status, "blockers": blockers, "run_root": run_root.as_posix()}


def _audit_full_loop_005(root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = _read_json_or_empty(root / "manifests" / "full_loop_validation_manifest.json")
    file_audit = _source_file_size_status(root)
    summary_path = filesystem_path(root / "metrics" / "memory_ablation_summary.csv")
    summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    _expect(blockers, str(manifest.get("project_title_version", "")) == "LQR-Stabilised Contextual Primitive v4.9", "full_loop_005_not_v49")
    _expect(blockers, str(manifest.get("status", "")) == "complete", "full_loop_005_status_not_complete")
    _expect(blockers, int(manifest.get("paired_episode_count", 0)) >= 200, "full_loop_005_paired_episode_count_below_200")
    _expect(blockers, int(manifest.get("episode_count", 0)) >= 1200, "full_loop_005_episode_count_below_1200")
    _expect(blockers, str(manifest.get("source_w01_root", "")).endswith("/w01_dense/015"), "full_loop_005_source_W01_mismatch")
    _expect(blockers, str(manifest.get("source_w2_root", "")).endswith("/w2_survival/015"), "full_loop_005_source_W2_mismatch")
    _expect(blockers, str(manifest.get("source_w3_root", "")).endswith("/w3_survival/013"), "full_loop_005_source_W3_mismatch")
    _expect(blockers, not bool(manifest.get("controller_mutation_allowed", True)), "full_loop_005_controller_mutation_allowed")
    _expect(blockers, not bool(manifest.get("retuning_allowed", True)), "full_loop_005_retuning_allowed")
    _expect(blockers, file_audit["source_file_above_100mb_count"] == 0, "full_loop_005_source_file_size_audit_above_100mb")
    if not summary.empty:
        terminal_rates = set(round(float(value), 6) for value in summary["terminal_useful_rate"])
        hard_rates = set(round(float(value), 6) for value in summary["hard_failure_rate"])
        no_viable = set(int(float(value)) for value in summary["no_viable_primitive_count"])
    else:
        terminal_rates = set()
        hard_rates = set()
        no_viable = set()
        blockers.append("full_loop_005_memory_summary_missing")
    return {
        "stage": "full_loop_005",
        "root": root.as_posix(),
        "status": "pass" if not blockers else "blocked",
        "row_count": int(manifest.get("episode_count", 0)),
        "variant_count": 0,
        "ready_count": 0,
        "representative_count": int(manifest.get("compact_representative_count", 0)),
        "source_role": "frozen_v49_paired_full_loop_evidence_for_calibration_diagnosis",
        "terminal_rate_values": ";".join(str(value) for value in sorted(terminal_rates)),
        "hard_failure_rate_values": ";".join(str(value) for value in sorted(hard_rates)),
        "no_viable_count_values": ";".join(str(value) for value in sorted(no_viable)),
        **file_audit,
        "blockers": blockers,
    }


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    blockers = list(manifest.get("blockers", []))
    lines = [
        "# v4.10 Source Audit",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Blocker count: `{len(blockers)}`",
        "- Frozen source chain: W01 `015`, W2 `015`, W3 `013`, post-W3 `001`, outcome `002`, full-loop `005`.",
        "- This audit allows governor calibration only; it does not allow W01/W2/W3 reruns or controller mutation.",
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
    parser = argparse.ArgumentParser(description="Audit v4.10 governor-calibration source roots.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--w01-root", type=Path, default=DEFAULT_W01_ROOT)
    parser.add_argument("--w2-root", type=Path, default=DEFAULT_W2_ROOT)
    parser.add_argument("--w3-root", type=Path, default=DEFAULT_W3_ROOT)
    parser.add_argument("--post-w3-root", type=Path, default=DEFAULT_POST_W3_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--governor-smoke-root", type=Path, default=DEFAULT_GOVERNOR_SMOKE_ROOT)
    parser.add_argument("--full-loop-root", type=Path, default=DEFAULT_FULL_LOOP_ROOT)
    parser.add_argument("--allow-retired-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_retired_diagnostic:
        print(json.dumps({"status": "blocked", "blocked_reason": "retired_diagnostic_requires_explicit_allow_retired_diagnostic"}, indent=2, sort_keys=True))
        return 1
    result = run_v410_source_audit(
        V410SourceAuditConfig(
            output_root=args.output_root,
            w01_root=args.w01_root,
            w2_root=args.w2_root,
            w3_root=args.w3_root,
            post_w3_root=args.post_w3_root,
            outcome_root=args.outcome_root,
            governor_smoke_root=args.governor_smoke_root,
            full_loop_root=args.full_loop_root,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "source_audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
