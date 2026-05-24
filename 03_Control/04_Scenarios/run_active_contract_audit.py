from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTROL_ROOT.parent
ACTIVE_STATUS_VALUES = (
    "complete",
    "accepted_fallback",
    "smoke_incomplete",
    "blocked",
    "retired_not_active",
)


@dataclass(frozen=True)
class AuditFinding:
    path: str
    check: str
    detail: str


def run_active_contract_audit(repo_root: Path | None = None) -> list[AuditFinding]:
    """Return active-code findings against the current LQR project contract."""

    root = Path(repo_root or REPO_ROOT)
    findings: list[AuditFinding] = []
    findings.extend(_audit_source_priority(root))
    findings.extend(_audit_validation_environment(root))
    findings.extend(_audit_status_contract(root))
    findings.extend(_audit_forbidden_methods(root))
    findings.extend(_audit_boundary_outcome_contract(root))
    findings.extend(_audit_surrogate_ladder(root))
    findings.extend(_audit_selected_controller_provenance(root))
    findings.extend(_audit_replay_only_contract(root))
    findings.extend(_audit_dense_runtime_contract(root))
    findings.extend(_audit_no_stale_active_generated_evidence(root))
    findings.extend(_audit_r6_readiness_behavior(root))
    return findings


def _audit_validation_environment(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    expected_python = (root / ".venv" / "Scripts" / "python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        findings.append(
            _finding(
                root,
                actual_python,
                "validation_environment",
                f"active audit must run with {expected_python}",
            )
        )

    active_docs = (
        root / "docs" / "local_validation_environment.md",
        root / "docs" / "housekeeping_and_naming_rules.md",
        root / "docs" / "Skills.md",
        root / "docs" / "Python Coding Instruction.txt",
    )
    for path in active_docs:
        text = _read(path)
        normalized = " ".join(text.split())
        for token in (".venv\\Scripts\\python.exe", "requirements-control-dev.txt"):
            if token not in text:
                findings.append(_finding(root, path, "validation_environment", f"missing {token}"))
        if "Paul_Li_FYP" in text and "not the active validation environment" not in normalized:
            findings.append(
                _finding(
                    root,
                    path,
                    "validation_environment",
                    "old Conda environment appears without active rejection wording",
                )
            )
    return findings


def _audit_source_priority(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    plan = root / "docs" / "Glider_Control_Project_Plan.md"
    housekeeping = root / "docs" / "housekeeping_and_naming_rules.md"
    python_guidance = root / "docs" / "Python Coding Instruction.txt"
    for path, tokens in (
        (plan, ("active local controller is time-invariant LQR", "outcome_class               accepted / weak / failed / rejected / blocked")),
        (housekeeping, ("03_Control/05_Results/", "100 MB", "retired_not_active")),
        (python_guidance, ("continuation_valid", "episode_terminal_useful", "boundary_use_class")),
    ):
        text = _read(path)
        for token in tokens:
            if token not in text:
                findings.append(_finding(root, path, "source_priority", f"missing {token}"))
    return findings


def _audit_status_contract(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    status_path = root / "03_Control" / "04_Scenarios" / "evidence_stage_utils.py"
    text = _read(status_path)
    for value in ACTIVE_STATUS_VALUES:
        if f'"{value}"' not in text:
            findings.append(_finding(root, status_path, "status_enum", f"missing {value}"))
    for old_value in ('"fallback"', '"partial"', '"deferred"'):
        if old_value in text:
            findings.append(_finding(root, status_path, "status_enum", f"old status {old_value}"))

    for rel in (
        "03_Control/04_Scenarios/run_primitive_selector_report.py",
        "03_Control/04_Scenarios/run_w2_replay.py",
        "03_Control/04_Scenarios/run_w3_generalisation.py",
    ):
        path = root / rel
        text = _read(path)
        for pattern in (
            r"stage_status\s*=\s*[\"']fallback[\"']",
            r"stage_status\s*=\s*[\"']partial[\"']",
            r"target_status\s*=\s*[\"']fallback[\"']",
            r"target_status\s*=\s*[\"']partial[\"']",
            r"return\s+[\"']fallback[\"']",
            r"return\s+[\"']partial[\"']",
            r"else\s+[\"']fallback[\"']",
            r"else\s+[\"']partial[\"']",
        ):
            if re.search(pattern, text):
                findings.append(_finding(root, path, "status_enum", f"old stage status pattern {pattern}"))
    return findings


def _audit_forbidden_methods(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    patterns = {
        "pd_pid_fallback": re.compile(r"\b(PD/PID|PD fallback|PID fallback|bounded[-_ ]PD|bounded[-_ ]feedback)\b", re.I),
        "forbidden_controller_family": re.compile(r"\b(TVLQR|MPC|LQR[-_ ]?tree)\b", re.I),
        "reachable_chain": re.compile(r"\breachable[-_ ]?chain\b", re.I),
        "online_fan_layout_branch": re.compile(r"\bonline fan[-_ ]?layout\b", re.I),
        "pd_pid_controller_literal": re.compile(r"controller_family\s*=\s*[\"']p(?:d|id)[\"']", re.I),
    }
    for path in _active_python_files(root, include_tests=False):
        text = _read(path)
        for check, pattern in patterns.items():
            if pattern.search(text):
                findings.append(_finding(root, path, check, "forbidden active-method token"))
    return findings


def _audit_boundary_outcome_contract(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    forbidden = (
        r"outcome_class\s*=\s*[\"']boundary_terminal[\"']",
        r"return\s+[\"']boundary_terminal[\"']",
        r"OUTCOME_CLASSES\s*=.*boundary_terminal",
    )
    for path in _active_python_files(root, include_tests=False):
        text = _read(path)
        for pattern in forbidden:
            if re.search(pattern, text):
                findings.append(_finding(root, path, "boundary_outcome", f"forbidden pattern {pattern}"))
    schema = _read(root / "03_Control" / "03_Primitives" / "primitive_evidence_schema.py")
    for token in (
        'OUTCOME_CLASSES = ("accepted", "weak", "failed", "rejected", "blocked")',
        'BOUNDARY_USE_CLASSES = ("continuation_valid", "episode_terminal_useful", "hard_failure", "blocked")',
        "continuation_valid",
        "episode_terminal_useful",
    ):
        if token not in schema:
            findings.append(_finding(root, root / "03_Control/03_Primitives/primitive_evidence_schema.py", "boundary_outcome", f"missing {token}"))
    return findings


def _audit_surrogate_ladder(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    path = root / "03_Control" / "04_Scenarios" / "env_surrogate.py"
    text = _read(path)
    for token in (
        "W0_requires_dry_air_zero_wind",
        "W1_requires_gaussian_plume_surrogate",
        "W2_requires_gp_corrected_annular_gaussian_surrogate",
        "W3_requires_randomised_gp_corrected_annular_gaussian_surrogate",
        "blocked_no_fallback",
    ):
        if token not in text:
            findings.append(_finding(root, path, "surrogate_ladder", f"missing {token}"))
    return findings


def _audit_selected_controller_provenance(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    checks = (
        (
            root / "03_Control" / "03_Primitives" / "controller_registry.py",
            (
                "SelectedControllerRecord",
                "load_selected_controller_records",
                "controller_from_evidence_row",
                "W0_W1_registry_selected",
            ),
        ),
        (
            root / "03_Control" / "04_Scenarios" / "run_ctx_archive.py",
            (
                "selected_controller_registry",
                "load_selected_controller_records",
                "selected_record.row_metadata",
                "blocked_missing_selected_registry",
            ),
        ),
        (
            root / "03_Control" / "04_Scenarios" / "run_lqr_tuning_sweep.py",
            (
                "selected_lqr_controllers.csv",
                "selected_lqr_controllers.json",
                "registry_status",
                "lqr_gain_checksum",
                "candidate_weight_label",
            ),
        ),
    )
    for path, tokens in checks:
        text = _read(path)
        for token in tokens:
            if token not in text:
                findings.append(_finding(root, path, "selected_controller_registry", f"missing {token}"))
    return findings


def _audit_replay_only_contract(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for rel, expected_status in (
        ("03_Control/04_Scenarios/run_w2_replay.py", "W2_verified_registry_replay"),
        ("03_Control/04_Scenarios/run_w3_generalisation.py", "W3_verified_registry_replay"),
    ):
        path = root / rel
        text = _read(path)
        for token in (
            "controller_from_evidence_row",
            expected_status,
            "actual_model_backed_replay",
            "accepted_fallback",
            "smoke_incomplete",
        ):
            if token not in text:
                findings.append(_finding(root, path, "replay_only", f"missing {token}"))
        for forbidden in ("synthesize_lqr_controller", "candidate_weight_specs", "write_selected_controller_registry"):
            if forbidden in text:
                findings.append(_finding(root, path, "replay_only", f"retuning path {forbidden}"))
    return findings


def _audit_dense_runtime_contract(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for rel in (
        "03_Control/04_Scenarios/run_ctx_archive.py",
        "03_Control/04_Scenarios/run_lqr_tuning_sweep.py",
        "03_Control/04_Scenarios/run_w2_replay.py",
        "03_Control/04_Scenarios/run_w3_generalisation.py",
    ):
        path = root / rel
        text = _read(path)
        for token in (
            "write_table_partition",
            "write_table_manifest",
            "write_file_size_audit",
            "chunk",
            "checksum",
        ):
            if token not in text:
                findings.append(_finding(root, path, "dense_runtime", f"missing {token}"))
    return findings


def _audit_r6_readiness_behavior(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    python_path = Path(sys.executable)
    if "WindowsApps" in python_path.as_posix():
        return [
            AuditFinding(
                path=str(python_path),
                check="r6_readiness_behavior",
                detail="active audit is running under the WindowsApps Python launcher; use a real repo interpreter",
            )
        ]

    script = root / "03_Control" / "04_Scenarios" / "run_lqr_tuning_sweep.py"
    with tempfile.TemporaryDirectory(prefix="nausicaa_r6_readiness_") as tmp:
        tmp_root = Path(tmp)
        dry_cmd = _r6_tuning_cmd(root, script, tmp_root, run_id=901, dry_run=True)
        command_findings = _run_and_check(root, dry_cmd, "r6_dry_run")
        if command_findings:
            return command_findings
        dry_root = tmp_root / "tune_901"
        for rel in (
            "manifests/run_manifest.json",
            "metrics/runtime_summary.csv",
            "metrics/chunk_summary.csv",
        ):
            if not (dry_root / rel).is_file():
                findings.append(_finding(root, dry_root / rel, "r6_dry_run", "missing compact dry-run output"))
        if any((dry_root / "tables").rglob("*")):
            findings.append(_finding(root, dry_root / "tables", "r6_dry_run", "dry-run wrote table files"))

        smoke_cmd = _r6_tuning_cmd(root, script, tmp_root, run_id=902, dry_run=False)
        command_findings = _run_and_check(root, smoke_cmd, "r6_smoke")
        if command_findings:
            return command_findings
        smoke_root = tmp_root / "tune_902"
        required = (
            "manifests/run_manifest.json",
            "manifests/table_manifest.json",
            "manifests/selected_lqr_controllers.json",
            "metrics/runtime_summary.csv",
            "metrics/chunk_summary.csv",
            "metrics/coverage_summary.csv",
            "metrics/file_size_audit.csv",
            "metrics/selected_lqr_controllers.csv",
            "chunk_manifests/lqr_tuning_rows/c00000.json",
            "tables/lqr_tuning_rows/c00000.csv.gz",
        )
        for rel in required:
            if not (smoke_root / rel).is_file():
                findings.append(_finding(root, smoke_root / rel, "r6_smoke", "missing readiness smoke output"))
        first_partition = smoke_root / "tables" / "lqr_tuning_rows" / "c00000.csv.gz"
        first_manifest = smoke_root / "chunk_manifests" / "lqr_tuning_rows" / "c00000.json"
        if first_partition.is_file() and first_manifest.is_file():
            before = (first_partition.stat().st_mtime_ns, first_manifest.stat().st_mtime_ns)
            command_findings = _run_and_check(root, smoke_cmd, "r6_resume")
            if command_findings:
                return command_findings
            after = (first_partition.stat().st_mtime_ns, first_manifest.stat().st_mtime_ns)
            if before != after:
                findings.append(_finding(root, first_partition, "r6_resume", "resume rewrote a validated chunk"))
            with first_partition.open("ab") as handle:
                handle.write(b"corrupt")
            repair_cmd = [*smoke_cmd, "--repair-incomplete"]
            command_findings = _run_and_check(root, repair_cmd, "r6_repair")
            if command_findings:
                return command_findings
            if not first_partition.is_file() or not first_manifest.is_file():
                findings.append(_finding(root, first_partition, "r6_repair", "repair did not regenerate corrupt chunk"))
    return findings


def _r6_tuning_cmd(root: Path, script: Path, output_root: Path, *, run_id: int, dry_run: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--run-id",
        str(run_id),
        "--output-root",
        str(output_root),
        "--rows",
        "8",
        "--candidate-count",
        "1",
        "--paired-tests-per-candidate",
        "1",
        "--candidate-chunk-size",
        "4",
        "--workers",
        "1",
        "--max-workers",
        "1",
        "--storage-format",
        "csv_gz",
        "--compression-level",
        "1",
        "--resume",
    ]
    if dry_run:
        cmd.append("--dry-run-schedule")
    return cmd


def _run_and_check(root: Path, cmd: list[str], check: str) -> list[AuditFinding]:
    result = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    detail = (result.stderr or result.stdout or "command failed").strip().splitlines()
    return [
        AuditFinding(
            path=Path(cmd[1]).relative_to(root).as_posix(),
            check=check,
            detail=detail[-1] if detail else "command failed",
        )
    ]


def _audit_no_stale_active_generated_evidence(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    results_root = root / "03_Control" / "05_Results"
    if not results_root.exists():
        return findings
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "03_Control/05_Results/.gitkeep":
            continue
        findings.append(_finding(root, path, "stale_active_generated_evidence", "active generated evidence file must be regenerated, not tracked"))
    return findings


def _active_python_files(root: Path, *, include_tests: bool) -> list[Path]:
    files: list[Path] = []
    for path in (root / "03_Control").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if "/99_Archive/" in rel or "/05_Results/" in rel:
            continue
        if rel == "03_Control/04_Scenarios/run_active_contract_audit.py":
            continue
        if not include_tests and "/tests/" in rel:
            continue
        files.append(path)
    return sorted(files)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _finding(root: Path, path: Path, check: str, detail: str) -> AuditFinding:
    try:
        rel_path = path.relative_to(root).as_posix()
    except ValueError:
        rel_path = path.as_posix()
    return AuditFinding(
        path=rel_path,
        check=str(check),
        detail=str(detail),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit active 03_Control code against the current LQR contract.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    findings = run_active_contract_audit(Path(args.repo_root))
    for finding in findings:
        print(f"{finding.path}: {finding.check}: {finding.detail}")
    if findings:
        print(f"active contract audit failed with {len(findings)} finding(s)")
        return 1
    print("active contract audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
