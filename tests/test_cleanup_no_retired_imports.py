from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TOKENS = (
    "turn_trajectory_" + "optimisation",
    "run_agile_trajectory_" + "optimisation",
    "run_agile_template_" + "search",
    "phase2_" + "tvlqr",
    "phase2_latency_" + "recovery",
    "tight_turn_" + "phase2",
    "ocp" + "030",
    "ocp" + "30",
    "from " + "tvlqr",
    "import " + "tvlqr",
    "Trajectory" + "Primitive",
    "solve_discrete_" + "tvlqr",
    "linearise_trajectory_finite_" + "difference",
)


def _active_python_files() -> list[Path]:
    roots = (REPO_ROOT / "03_Control", REPO_ROOT / "tests")
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(files)


def test_no_retired_agile_or_old_tvlqr_references_remain() -> None:
    offenders: list[str] = []
    for path in _active_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("ClosedLoopTrajectoryPrimitive", "")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{token!r} in {rel}")

    assert offenders == []
