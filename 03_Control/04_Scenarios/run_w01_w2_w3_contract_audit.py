from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402


AUDIT_VERSION = "w01_w2_w3_contract_audit_v1"


@dataclass(frozen=True)
class AuditFinding:
    path: str
    check: str
    detail: str


FORBIDDEN_ACTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("selected_controller_registry", re.compile(r"selected[_-]controller[_-]registry", re.I)),
    ("selected_lqr_controllers", re.compile(r"selected[_-]lqr[_-]controllers", re.I)),
    ("accepted_fallback", re.compile(r"\baccepted_fallback\b", re.I)),
    ("r6_r7_r8_r9_stage_story", re.compile(r"\bR[6789]\b|\br6_1\b|\bR61\b", re.I)),
    ("finalist_shortlist", re.compile(r"\bfinalist\b|\bshortlist\b|12\s*[-/]\s*24", re.I)),
    ("pre_w3_medoid", re.compile(r"\bmedoid\b|\bclustering_output\b", re.I)),
    ("hardware_shortlist", re.compile(r"hardware[_-]shortlist", re.I)),
    ("policy_governor_output", re.compile(r"stale[_-]policy[_-]output|stale[_-]governor[_-]output", re.I)),
    ("reachable_chain", re.compile(r"reachable[-_ ]?chain", re.I)),
    ("fan_layout_active_mode", re.compile(r"['\"]fan_shift['\"]|['\"]power_scale['\"]", re.I)),
    ("pd_pid_fallback", re.compile(r"\bPD/PID fallback\b|\bPID fallback\b|\bPD fallback\b", re.I)),
)

ACTIVE_SCAN_ROOTS = (
    Path("03_Control/02_Inner_Loop"),
    Path("03_Control/03_Primitives"),
    Path("03_Control/04_Scenarios"),
    Path("03_Control/tests"),
)

SKIP_FILES = {
    Path("03_Control/04_Scenarios/run_w01_w2_w3_contract_audit.py"),
    Path("03_Control/04_Scenarios/run_active_contract_audit.py"),
    Path("03_Control/04_Scenarios/run_post_w3_library_size_study.py"),
    Path("03_Control/04_Scenarios/" + "run_post_w3_" + "cluster_merge.py"),
}

R5_R10_STAGE_TOKEN_ALLOWED_FILES = {
    Path("03_Control/04_Scenarios/run_r5_r10_pipeline.py"),
    Path("03_Control/04_Scenarios/run_w3_survival.py"),
    Path("03_Control/04_Scenarios/run_repeated_launch_learning_curve.py"),
    Path("03_Control/04_Scenarios/run_changed_case_validation.py"),
}


def run_w01_w2_w3_contract_audit(repo_root: Path | str = Path(".")) -> list[AuditFinding]:
    root = Path(repo_root)
    findings: list[AuditFinding] = []
    findings.extend(_scan_forbidden_tokens(root))
    findings.extend(_audit_active_results(root))
    findings.extend(_audit_w2_w3_fixed_replay(root))
    findings.extend(_audit_controller_registry_stub(root))
    findings.extend(_audit_file_sizes(root))
    return findings


def _scan_forbidden_tokens(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for rel_root in ACTIVE_SCAN_ROOTS:
        scan_root = root / rel_root
        if not filesystem_path(scan_root).exists():
            continue
        for path in sorted(filesystem_path(scan_root).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt"}:
                continue
            rel = path.relative_to(filesystem_path(root))
            if Path(rel.as_posix()) in SKIP_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name, pattern in FORBIDDEN_ACTIVE_PATTERNS:
                if name == "r6_r7_r8_r9_stage_story" and Path(rel.as_posix()) in R5_R10_STAGE_TOKEN_ALLOWED_FILES:
                    continue
                if pattern.search(text):
                    findings.append(AuditFinding(rel.as_posix(), name, "forbidden active workflow token present"))
    return findings


def _audit_active_results(root: Path) -> list[AuditFinding]:
    result_root = root / "03_Control" / "05_Results"
    if not filesystem_path(result_root).exists():
        return []
    findings: list[AuditFinding] = []
    forbidden_parts = {
        "r6",
        "r7",
        "r8",
        "r9",
        "selected_lqr_controllers.csv",
        "selected_lqr_controllers.json",
        "medoid_cluster_report.md",
        "hardware_shortlist.csv",
    }
    for path in sorted(filesystem_path(result_root).rglob("*")):
        rel = path.relative_to(filesystem_path(root)).as_posix()
        parts = {part.lower() for part in Path(rel).parts}
        if forbidden_parts.intersection(parts):
            findings.append(AuditFinding(rel, "stale_active_result_path", "stale active result path must be removed"))
    return findings


def _audit_w2_w3_fixed_replay(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for rel in (
        Path("03_Control/04_Scenarios/run_w2_survival.py"),
        Path("03_Control/04_Scenarios/run_w3_survival.py"),
    ):
        path = root / rel
        if not filesystem_path(path).is_file():
            findings.append(AuditFinding(rel.as_posix(), "missing_survival_scaffold", "required fixed-LQR scaffold is missing"))
            continue
        text = filesystem_path(path).read_text(encoding="utf-8", errors="ignore")
        for token in ("synthesize_lqr_controller", "candidate_weight_specs", "default_lqr_weight_spec", "lqr_tuning"):
            if token in text:
                findings.append(AuditFinding(rel.as_posix(), "w2_w3_retuning_dependency", f"forbidden dependency {token}"))
        if "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role" not in text:
            findings.append(AuditFinding(rel.as_posix(), "missing_no_mutation_claim", "fixed-LQR no-mutation boundary missing"))
    return findings


def _audit_controller_registry_stub(root: Path) -> list[AuditFinding]:
    rel = Path("03_Control/03_Primitives/controller_registry.py")
    path = root / rel
    if not filesystem_path(path).is_file():
        return []
    text = filesystem_path(path).read_text(encoding="utf-8", errors="ignore")
    forbidden = ("write_selected", "load_selected", "selected_lqr", "accepted_fallback")
    return [
        AuditFinding(rel.as_posix(), "controller_registry_not_non_active", f"forbidden token {token}")
        for token in forbidden
        if token in text
    ]


def _audit_file_sizes(root: Path) -> list[AuditFinding]:
    result_root = root / "03_Control" / "05_Results"
    if not filesystem_path(result_root).exists():
        return []
    findings: list[AuditFinding] = []
    max_bytes = int(MAX_GENERATED_FILE_SIZE_MB * 1024 * 1024)
    for path in sorted(filesystem_path(result_root).rglob("*")):
        if path.is_file() and path.stat().st_size > max_bytes:
            rel = path.relative_to(filesystem_path(root)).as_posix()
            findings.append(AuditFinding(rel, "file_size_over_100mb", f"{path.stat().st_size} bytes"))
    return findings


def _payload(findings: list[AuditFinding]) -> dict[str, object]:
    return {
        "audit_version": AUDIT_VERSION,
        "status": "pass" if not findings else "fail",
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    findings = run_w01_w2_w3_contract_audit(Path("."))
    print(json.dumps(_payload(findings), indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
