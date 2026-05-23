from __future__ import annotations

import json
from pathlib import Path

from run_lqr_contextual_archive import parse_args, main


def test_lqr_contextual_archive_dry_run_schedule(tmp_path: Path) -> None:
    config = parse_args(
        [
            "--run-id",
            "131",
            "--output-root",
            str(tmp_path),
            "--dry-run-schedule",
            "--stop-after-chunks",
            "1",
            "--workers",
            "1",
            "--max-workers",
            "1",
        ]
    )
    assert config.rollout_backend == "model_backed_lqr"
    assert config.dry_run_schedule is True
    assert main(
        [
            "--run-id",
            "131",
            "--output-root",
            str(tmp_path),
            "--dry-run-schedule",
            "--stop-after-chunks",
            "1",
            "--workers",
            "1",
            "--max-workers",
            "1",
        ]
    ) == 0
    manifest = json.loads((tmp_path / "r131" / "manifests" / "run_manifest.json").read_text())
    assert manifest["rollout_backend"] == "model_backed_lqr"
    assert manifest["dry_run_schedule"] is True
