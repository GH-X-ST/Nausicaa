from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = REPO_ROOT / "03_Control"
RESULT_ROOT = CONTROL_ROOT / "05_Results"


def _join(parts: tuple[str, ...]) -> str:
    return "".join(parts)


def _old_archive_root(run_id: str) -> Path:
    campaign = _join(("07", "_aggressive", "_reversal", "_ocp"))
    return RESULT_ROOT / campaign / run_id


def _old_family_root() -> Path:
    campaign = _join(("08", "_agile", "_turn", "_family", "_comparison"))
    return RESULT_ROOT / campaign


def _old_family_module() -> str:
    return _join(("agile", "_turn", "_family", "_comparison"))


def _forbidden_pattern() -> str:
    tokens = (
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
    return "|".join(_join(parts) for parts in tokens)


def test_obsolete_active_branch_files_are_absent() -> None:
    module = _old_family_module()
    obsolete_paths = (
        CONTROL_ROOT / "03_Primitives" / f"{module}.py",
        CONTROL_ROOT / "04_Scenarios" / f"run_{module}.py",
        CONTROL_ROOT / "tests" / f"test_{module}_profiles.py",
        CONTROL_ROOT / "tests" / f"test_{module}_runner.py",
    )

    for path in obsolete_paths:
        assert not path.exists()

    assert not (_old_archive_root("001")).exists()
    assert not _old_family_root().exists()


def test_archived_boundary_evidence_is_preserved_and_manifested() -> None:
    archive_root = _old_archive_root("002")
    manifest_path = (
        RESULT_ROOT
        / "09_primitive_library"
        / "001"
        / "manifests"
        / "repo_housekeeping_manifest_s001.json"
    )

    assert archive_root.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    assert manifest["boundary_reference_preserved_byte_for_byte"] is True
    assert manifest["preserved_archive_file_count"] > 0
    assert manifest["final_plan_lock"]["dry_air_agile_turn_recovery_loop_closed"] is True
    assert manifest["final_plan_lock"]["no_overclaiming_flags"]["w3_stress_implemented"] is False


def test_forbidden_retired_branch_tokens_are_absent_from_active_source() -> None:
    command = [
        "git",
        "grep",
        "-n",
        "-E",
        _forbidden_pattern(),
        "--",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
        "03_Control/tests",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
