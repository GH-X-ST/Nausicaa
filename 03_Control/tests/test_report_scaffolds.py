from __future__ import annotations

from pathlib import Path

from run_post_w3_cluster_merge import run_post_w3_cluster_merge
from run_w2_survival import W2SurvivalConfig, run_w2_survival
from run_w3_survival import W3SurvivalConfig, run_w3_survival


def test_w2_w3_scaffolds_are_blocked_without_required_survivors(tmp_path: Path) -> None:
    w2 = run_w2_survival(W2SurvivalConfig(run_id=1, input_root=tmp_path / "missing", output_root=tmp_path / "w2"))
    w3 = run_w3_survival(W3SurvivalConfig(run_id=1, input_root=Path(w2["run_root"]), output_root=tmp_path / "w3"))
    post = run_post_w3_cluster_merge(input_root=Path(w3["run_root"]))

    assert w2["status"] == "blocked"
    assert w3["status"] == "blocked"
    assert post["status"] == "blocked"
    w2_manifest = (Path(w2["run_root"]) / "manifests" / "w2_survival_manifest.json").read_text(encoding="ascii")
    w3_manifest = (Path(w3["run_root"]) / "manifests" / "w3_survival_manifest.json").read_text(encoding="ascii")
    assert "w2_survivor_registry.json" in w2_manifest
    assert "ready_for_fixed_lqr_replay" in w3_manifest
    assert "fixed_lqr_replay_only" in w2_manifest
    assert "input_contract" in w3_manifest


def test_w2_w3_scaffolds_do_not_import_retuning_dependencies() -> None:
    for path in (
        Path("03_Control/04_Scenarios/run_w2_survival.py"),
        Path("03_Control/04_Scenarios/run_w3_survival.py"),
        Path("03_Control/03_Primitives/frozen_w01_controller_bundle.py"),
    ):
        text = path.read_text(encoding="ascii")
        for token in (
            "synthesize_lqr_controller",
            "candidate_weight_specs",
            "default_lqr_weight_spec",
            "lqr_tuning",
            "solve_discrete_are",
            "solve_continuous_are",
        ):
            assert token not in text
