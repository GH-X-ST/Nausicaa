from __future__ import annotations

import sys
from pathlib import Path

import pytest


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTROL_ROOT.parents[0]
for rel in (
    "02_Inner_Loop",
    "03_Primitives",
    "04_Scenarios",
):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SLOW_INTEGRATION_TEST_FILES = {
    "test_ctx_archive_smoke.py",
    "test_ctx_episode_smoke.py",
    "test_lqr_contextual_archive_dry_run.py",
    "test_lqr_tuning_dry_run_contract.py",
    "test_post_w3_pipeline.py",
    "test_r5_transition_qr_training.py",
    "test_report_scaffolds.py",
    "test_v411_repair_cycle.py",
    "test_w2_survival.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow pipeline/archive integration tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark pipeline-executing tests as slow integration tests.

    These tests are valuable before dense evidence regeneration, but they run
    miniature archive/replay pipelines and should not be part of the default
    fast regression pass after every small edit.
    """

    slow = pytest.mark.slow(reason="pipeline/archive integration test")
    skip_slow = pytest.mark.skip(reason="slow pipeline/archive test; pass --run-slow to execute")
    run_slow = bool(config.getoption("--run-slow"))
    for item in items:
        if Path(str(item.fspath)).name in SLOW_INTEGRATION_TEST_FILES:
            item.add_marker(slow)
            if not run_slow:
                item.add_marker(skip_slow)
