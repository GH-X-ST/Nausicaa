from __future__ import annotations

import json
from pathlib import Path

from run_lqr_tuning_sweep import LQRTuningSweepConfig, run_lqr_tuning_sweep


def test_lqr_tuning_dry_run_records_go_no_go_contract(tmp_path: Path) -> None:
    result = run_lqr_tuning_sweep(
        LQRTuningSweepConfig(
            run_id=141,
            output_root=tmp_path,
            rows=32,
            candidate_count=8,
            paired_tests_per_candidate=25,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            dry_run_schedule=True,
        )
    )

    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="ascii"))
    assert manifest["raw_K_tuning_allowed"] is False
    assert manifest["W0_W1_tune_controller_ids"] is True
    assert manifest["W2_W3_replay_only"] is True
    assert "valid_lqr_synthesis" in manifest["hard_gates"]
