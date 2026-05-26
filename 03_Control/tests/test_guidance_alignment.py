from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_controlling_docs_and_active_runner_share_w01_contract_language() -> None:
    controlling = "\n".join(
        _read(path)
        for path in (
            "docs/Glider_Control_Project_Plan.md",
            "docs/Daily_Schedule.txt",
            "docs/Skills.md",
            "docs/Python Coding Instruction.txt",
            "docs/MATLAB Coding.txt",
            "docs/housekeeping_and_naming_rules.md",
        )
    )
    runner = _read("03_Control/04_Scenarios/run_lqr_w01_dense_chunked.py")

    for token in (
        "W0",
        "W1",
        "W2",
        "W3",
        "100 MB",
        "resume",
        "checksum",
    ):
        assert token in controlling
    for token in (
        "R5_W0_W1_robust_randomised_primitive_synthesis",
        "w1_randomised_single",
        "w1_randomised_four",
        "launch_gate",
        "inflight_recovery_edge",
        "W3_replay_only",
        "pd_pid_fallback_allowed",
    ):
        assert token in runner
