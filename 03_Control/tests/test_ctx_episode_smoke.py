from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from run_ctx_episode_smoke import EpisodeSmokeConfig, run_contextual_episode_smoke


def _results_entries_are_placeholder_or_allowed(entries: list[str]) -> bool:
    allowed_root = os.environ.get("NAUSICAA_ALLOW_LOCAL_EVIDENCE_ROOT", "").strip()
    if not allowed_root:
        return all(
            entry == ".gitkeep"
            or entry == "lqr_contextual_v1_0"
            or entry == "lqr_contextual_v1_0/w01_dense"
            or entry.startswith("lqr_contextual_v1_0/w01_dense/")
            or entry == "lqr_contextual_v1_0/w2_survival"
            or entry.startswith("lqr_contextual_v1_0/w2_survival/")
            or entry == "lqr_contextual_v1_0/w3_survival"
            or entry.startswith("lqr_contextual_v1_0/w3_survival/")
            or entry == "lqr_contextual_v1_0/post_w3_cluster"
            or entry.startswith("lqr_contextual_v1_0/post_w3_cluster/")
            or entry == "lqr_contextual_v1_0/post_w3_library_size_study"
            or entry.startswith("lqr_contextual_v1_0/post_w3_library_size_study/")
            or entry == "lqr_contextual_v1_0/outcome_model"
            or entry.startswith("lqr_contextual_v1_0/outcome_model/")
            or entry == "lqr_contextual_v1_0/repeated_launch_validation"
            or entry.startswith("lqr_contextual_v1_0/repeated_launch_validation/")
            or entry == "lqr_contextual_v1_0/changed_case_validation"
            or entry.startswith("lqr_contextual_v1_0/changed_case_validation/")
            or entry == "lqr_contextual_v1_0/r5_r10_pipeline"
            or entry.startswith("lqr_contextual_v1_0/r5_r10_pipeline/")
            or entry == "lqr_contextual_v1_0/archive"
            or entry.startswith("lqr_contextual_v1_0/archive/")
            or entry == "lqr_contextual_v1_0/governor_smoke"
            or entry.startswith("lqr_contextual_v1_0/governor_smoke/")
            or entry == "lqr_contextual_v1_0/full_loop_validation"
            or entry.startswith("lqr_contextual_v1_0/full_loop_validation/")
            or entry == "lqr_contextual_v1_0/figures"
            or entry.startswith("lqr_contextual_v1_0/figures/")
            or entry == "lqr_contextual_v1_0/governor_calibration"
            or entry.startswith("lqr_contextual_v1_0/governor_calibration/")
            for entry in entries
        )

    root = Path(allowed_root)
    if root.is_absolute():
        try:
            allowed_prefix = root.resolve().relative_to(
                Path.cwd() / "03_Control" / "05_Results"
            )
        except ValueError:
            return False
    else:
        root_text = root.as_posix().rstrip("/")
        prefix = "03_Control/05_Results/"
        if root_text.startswith(prefix):
            root_text = root_text[len(prefix) :]
        allowed_prefix = Path(root_text)

    allowed_text = allowed_prefix.as_posix().rstrip("/")
    allowed_parents = {
        parent.as_posix()
        for parent in Path(allowed_text).parents
        if parent.as_posix() != "."
    }
    return all(
        entry == ".gitkeep"
        or entry == allowed_text
        or entry in allowed_parents
        or entry.startswith(f"{allowed_text}/")
        for entry in entries
    )


def test_episode_smoke_writes_temp_only_lqr_rows(tmp_path: Path) -> None:
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
    assert manifest["rollout_backend"] == "model_backed_lqr"
    assert len(episode_log) == 10
    assert set(episode_log["evidence_role"]).issubset({"lqr_rollout_candidate", "blocked_lqr_synthesis"})
    assert set(episode_log["selector_governor_mode"]) == {"terminal_episode"}
    assert set(episode_log["memory_label"]) == {
        "directional_residual_lift_belief_smoke_no_improvement_claim"
    }
    assert "belief_before_local_lift_residual_m_s" in episode_log.columns
    assert "belief_after_local_lift_residual_m_s" in episode_log.columns
    assert (run_root / "tables" / "belief_snapshots.csv").is_file()

    result_entries = [
        path.relative_to("03_Control/05_Results").as_posix()
        for path in Path("03_Control/05_Results").rglob("*")
    ]
    assert _results_entries_are_placeholder_or_allowed(result_entries)
