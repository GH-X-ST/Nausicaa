from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_ctx_episode_smoke import EpisodeSmokeConfig, run_contextual_episode_smoke


def test_episode_smoke_writes_temp_only_feedback_rows(tmp_path: Path) -> None:
    result = run_contextual_episode_smoke(
        EpisodeSmokeConfig(
            run_id=72,
            episode_count=10,
            seed=72,
            governor_mode="terminal_episode",
            output_root=tmp_path,
        )
    )

    run_root = Path(result["run_root"])
    manifest = json.loads(
        (run_root / "manifests" / "episode_smoke_manifest.json").read_text()
    )
    episode_log = pd.read_csv(run_root / "tables" / "episode_log.csv")

    assert manifest["claim_status"] == "simulation_only_episode_smoke_no_performance_claim"
    assert manifest["rollout_backend"] == "model_backed_feedback"
    assert len(episode_log) == 10
    assert set(episode_log["evidence_role"]) == {"feedback_rollout_candidate"}
    assert set(episode_log["selector_governor_mode"]) == {"terminal_episode"}
    assert set(episode_log["memory_label"]) == {
        "episodic_lift_belief_smoke_no_improvement_claim"
    }
    assert "belief_before_local_lift_m_s" in episode_log.columns
    assert "belief_after_local_lift_m_s" in episode_log.columns
    assert (run_root / "tables" / "belief_snapshots.csv").is_file()

    result_entries = [
        path.relative_to("03_Control/05_Results").as_posix()
        for path in Path("03_Control/05_Results").rglob("*")
    ]
    assert result_entries == [".gitkeep"]
