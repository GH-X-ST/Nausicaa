from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="v4.10 governor calibration path is retired behind v4.11 diagnostic gates")

from episode_selector import select_compact_representative
from run_full_loop_validation import FullLoopValidationConfig, run_full_loop_validation
from run_governor_calibration import GovernorCalibrationConfig, run_governor_calibration
from run_v410_source_audit import V410SourceAuditConfig, run_v410_source_audit
from run_v410_validation_figures import run_v410_validation_figures
from viability_governor import DEFAULT_GOVERNOR_CONFIG, governor_score


@pytest.fixture(scope="module")
def v410_calibration_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("v410_calibration")
    result = run_governor_calibration(
        GovernorCalibrationConfig(
            run_id=1,
            output_root=base / "governor_calibration",
            ranking_config_count=4,
            top_config_count=2,
            context_sample_count=12,
            calibration_episodes_per_policy=1,
            seed=410,
        )
    )
    assert result["status"] == "complete"
    return base / "governor_calibration" / "001"


def test_v410_source_audit_accepts_frozen_chain(tmp_path: Path) -> None:
    result = run_v410_source_audit(V410SourceAuditConfig(output_root=tmp_path / "full_loop_validation" / "006"))

    assert result["status"] == "source_audit_pass"


def test_default_governor_config_reproduces_v49_scores() -> None:
    continuation = governor_score(
        governor_mode="continuation_mode",
        continuation_probability=0.4,
        terminal_useful_probability=0.2,
        hard_failure_risk=0.3,
        expected_updraft_gain_proxy_m=0.1,
        expected_lift_dwell_time_s=0.5,
        wall_margin_m=0.6,
        belief_local_lift_m_s=0.7,
        governor_config=DEFAULT_GOVERNOR_CONFIG,
    )
    terminal = governor_score(
        governor_mode="terminal_episode_mode",
        continuation_probability=0.4,
        terminal_useful_probability=0.2,
        hard_failure_risk=0.3,
        expected_updraft_gain_proxy_m=0.1,
        expected_lift_dwell_time_s=0.5,
        wall_margin_m=0.6,
        belief_local_lift_m_s=0.7,
        governor_config=DEFAULT_GOVERNOR_CONFIG,
    )

    assert continuation == pytest.approx(0.4 - 0.3 * 0.2 - 0.8 * 0.3 + 0.04 * 0.1 + 0.03 * 0.5 + 0.05 * 0.7)
    assert terminal == pytest.approx(1.10 * 0.2 + 0.25 * 0.4 - 0.75 * 0.3 + 0.05 * 0.1 + 0.04 * 0.5 + 0.05 * 0.7)


def test_belief_weight_changes_score_when_belief_nonzero() -> None:
    zero = governor_score(
        governor_mode="continuation_mode",
        continuation_probability=0.2,
        terminal_useful_probability=0.1,
        hard_failure_risk=0.2,
        expected_updraft_gain_proxy_m=0.0,
        expected_lift_dwell_time_s=0.0,
        wall_margin_m=0.1,
        belief_local_lift_m_s=0.0,
        governor_config=DEFAULT_GOVERNOR_CONFIG,
    )
    nonzero = governor_score(
        governor_mode="continuation_mode",
        continuation_probability=0.2,
        terminal_useful_probability=0.1,
        hard_failure_risk=0.2,
        expected_updraft_gain_proxy_m=0.0,
        expected_lift_dwell_time_s=0.0,
        wall_margin_m=0.1,
        belief_local_lift_m_s=1.0,
        governor_config=DEFAULT_GOVERNOR_CONFIG,
    )

    assert nonzero - zero == pytest.approx(DEFAULT_GOVERNOR_CONFIG.belief_weight)


def test_candidate_diagnostics_include_memory_and_rank_fields() -> None:
    representatives = [
        {
            "compact_library_id": "lib_a",
            "primitive_variant_id": "var_a",
            "primitive_id": "glide",
            "entry_role": "launch_capable",
            "controller_id": "ctrl_a",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
        },
        {
            "compact_library_id": "lib_b",
            "primitive_variant_id": "var_b",
            "primitive_id": "glide",
            "entry_role": "launch_capable",
            "controller_id": "ctrl_b",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
        },
    ]
    outcomes = {
        "var_a": {"continuation_probability": 0.3, "terminal_useful_probability": 0.1, "hard_failure_risk": 0.1},
        "var_b": {"continuation_probability": 0.2, "terminal_useful_probability": 0.1, "hard_failure_risk": 0.1},
    }
    context = {
        "context_id": "ctx",
        "W_layer": "W0",
        "environment_mode": "dry_air",
        "start_state_family": "launch_gate",
        "latency_case": "none",
        "wall_margin_m": 1.0,
        "floor_margin_m": 1.0,
        "ceiling_margin_m": 1.0,
    }

    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcomes,
        context=context,
        governor_mode="continuation_mode",
        policy_id="context_plus_memory_lambda_0_8",
        belief_features={"belief_local_lift_m_s": 0.5, "belief_mean_lift_m_s": 0.2, "belief_max_lift_m_s": 0.8},
    )

    assert selected is not None
    for row in rows:
        assert "base_score_without_memory" in row
        assert "memory_score_component" in row
        assert "score_margin_to_selected" in row
        assert "rank_without_memory" in row
        assert "rank_with_memory" in row
        assert row["governor_config_id"] == DEFAULT_GOVERNOR_CONFIG.config_id


def test_calibration_writes_frozen_config_and_preserves_compact_ids(v410_calibration_root: Path) -> None:
    payload = json.loads((v410_calibration_root / "manifests" / "frozen_governor_config.json").read_text(encoding="ascii"))
    summary = pd.read_csv(v410_calibration_root / "metrics" / "calibration_config_summary.csv")
    audit = pd.read_csv(v410_calibration_root / "metrics" / "file_size_audit.csv")

    assert payload["status"] == "selected"
    assert payload["controller_mutation_allowed"] is False
    assert payload["retuning_allowed"] is False
    assert payload["governor_config"]["config_id"] in set(summary["governor_config_id"])
    assert (v410_calibration_root / "cal" / "c000").is_dir()
    assert not (v410_calibration_root / "calibration_full_loop").exists()
    assert audit["relative_path"].str.contains("calibration_full_loop").sum() == 0
    assert audit["relative_path_length"].max() <= 90
    assert audit["path_within_140_chars"].astype(bool).all()


def test_heldout_uses_frozen_config_and_disjoint_episode_keys(v410_calibration_root: Path, tmp_path: Path) -> None:
    frozen_path = v410_calibration_root / "manifests" / "frozen_governor_config.json"
    result = run_full_loop_validation(
        FullLoopValidationConfig(
            run_id=6,
            output_root=tmp_path / "full_loop_validation",
            episodes_per_policy=2,
            seed=4106,
            source_audit_version="v410",
            governor_config_path=frozen_path,
            resume=True,
            repair_incomplete=True,
        )
    )
    heldout_root = tmp_path / "full_loop_validation" / "006"
    heldout_schedule = pd.read_csv(heldout_root / "metrics" / "episode_schedule.csv")
    calibration_schedule = pd.read_csv(v410_calibration_root / "metrics" / "calibration_episode_schedule.csv")
    calibration_keys = set(calibration_schedule["common_random_key"].astype(str))

    assert result["status"] == "complete"
    assert not calibration_keys.intersection(set(heldout_schedule["common_random_key"].astype(str)))
    manifest = json.loads((heldout_root / "manifests" / "full_loop_validation_manifest.json").read_text(encoding="ascii"))
    assert manifest["governor_config_id"] == json.loads(frozen_path.read_text(encoding="ascii"))["governor_config"]["config_id"]
    assert manifest["project_title_version"] == "LQR-Stabilised Contextual Primitive v4.10"


def test_v410_figures_and_tables_are_generated(v410_calibration_root: Path, tmp_path: Path) -> None:
    frozen_path = v410_calibration_root / "manifests" / "frozen_governor_config.json"
    result = run_full_loop_validation(
        FullLoopValidationConfig(
            run_id=6,
            output_root=tmp_path / "full_loop_validation",
            episodes_per_policy=1,
            seed=4206,
            source_audit_version="v410",
            governor_config_path=frozen_path,
            resume=True,
            repair_incomplete=True,
        )
    )
    assert result["status"] == "complete"
    figure_result = run_v410_validation_figures(v410_calibration_root, tmp_path / "full_loop_validation" / "006")

    assert figure_result["status"] == "complete"
    for name in (
        "v410_memory_ablation_bar.png",
        "v410_paired_delta_plot.png",
        "v410_governor_rejection_summary.png",
        "v410_belief_evolution.png",
        "v410_outcome_prediction_alignment.png",
    ):
        assert (v410_calibration_root / "09_figures" / name).is_file()
    audit = pd.read_csv(v410_calibration_root / "metrics" / "file_size_audit.csv")
    assert not audit["above_100mb"].astype(bool).any()
    assert audit["push_allowed"].astype(bool).all()
