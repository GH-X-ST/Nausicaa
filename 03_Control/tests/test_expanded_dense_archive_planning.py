from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = REPO_ROOT / "03_Control"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_dense_archive_pilot_sweep as pilot_runner  # noqa: E402
import run_expanded_dense_archive_planning as expanded_runner  # noqa: E402


EXPECTED_MODES = {
    "W0_single_fan_branch",
    "W1_single_fan",
    "W0_four_fan_branch",
    "W1_four_fan",
}


@pytest.fixture(scope="module")
def expanded_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    result_root = tmp_path_factory.mktemp("expanded_dense_archive_planning")
    _write_source_run(result_root)
    return expanded_runner.run_expanded_dense_archive_planning(
        run_id=10,
        source_planning_run_id=8,
        result_root=result_root,
        pilot_start_states_per_family_target_direction=75,
        required_min_candidate_rows=20000,
    )


def test_expanded_planning_generates_at_least_20k_candidates_with_default_75(
    expanded_outputs: dict[str, Path],
) -> None:
    manifest = _read_manifest(expanded_outputs)
    starts = pd.read_csv(expanded_outputs["start_state_manifest_csv"])
    candidates = pd.read_csv(expanded_outputs["dry_run_candidate_inventory_csv"])

    assert manifest["pilot_start_states_per_family_target_direction"] == 75
    assert manifest["start_state_rows_all_branches"] == 10200
    assert manifest["candidate_rows_all_branches"] == 20400
    assert manifest["required_min_candidate_rows"] == 20000
    assert manifest["ready_for_20k_pilot"] is True
    assert len(starts) == 10200
    assert len(candidates) == 20400


def test_expanded_planning_output_names_are_pilot_runner_compatible(
    expanded_outputs: dict[str, Path],
) -> None:
    result_root = expanded_outputs["root"].parent
    config = pilot_runner.DensePilotSweepConfig(
        run_id=11,
        planning_run_id=10,
        result_root=result_root,
    )
    start_path, candidate_path = pilot_runner._planning_paths(config)

    assert expanded_outputs["manifest_json"].name == (
        "expanded_dense_archive_planning_manifest_s010.json"
    )
    assert start_path == expanded_outputs["start_state_manifest_csv"]
    assert candidate_path == expanded_outputs["dry_run_candidate_inventory_csv"]
    assert start_path.exists()
    assert candidate_path.exists()


def test_expanded_planning_preserves_branch_and_environment_coverage(
    expanded_outputs: dict[str, Path],
) -> None:
    starts = pd.read_csv(expanded_outputs["start_state_manifest_csv"])
    candidates = pd.read_csv(expanded_outputs["dry_run_candidate_inventory_csv"])

    assert set(starts["fan_layout"]) == {"single_fan", "four_fan"}
    assert starts["fan_layout"].value_counts().to_dict() == {
        "single_fan": 5100,
        "four_fan": 5100,
    }
    assert set(candidates["test_environment_mode"]) == EXPECTED_MODES
    assert candidates["test_environment_mode"].value_counts().to_dict() == {
        "W0_single_fan_branch": 5100,
        "W1_single_fan": 5100,
        "W0_four_fan_branch": 5100,
        "W1_four_fan": 5100,
    }
    for column in (
        "no_cross_branch_promotion",
        "no_cross_branch_rejection",
        "no_cross_branch_cluster_merge",
        "no_cross_branch_safety_justification",
    ):
        assert set(candidates[column]) == {True}


def test_expanded_planning_manifest_has_no_archive_or_hardware_claims(
    expanded_outputs: dict[str, Path],
) -> None:
    manifest = _read_manifest(expanded_outputs)
    report = expanded_outputs["report_md"].read_text(encoding="ascii")

    assert manifest["expanded_planning_performed"] is True
    assert manifest["production_w0_archive_performed"] is False
    assert manifest["production_w1_archive_performed"] is False
    assert manifest["pilot_sweep_performed"] is False
    assert manifest["hardware_or_mission_claim"] is False
    assert manifest["sim_to_real_transfer_claim"] is False
    assert manifest["branch_local_decisions_only"] is True
    assert manifest["source_run008_preserved"] is True
    assert "no W0/W1 production archive" in report


def test_expanded_planning_rejects_protected_run_ids_without_writing() -> None:
    with pytest.raises(ValueError, match="greater than 009"):
        expanded_runner.run_expanded_dense_archive_planning(
            run_id=9,
            source_planning_run_id=8,
        )


def test_expanded_planning_rejects_existing_output_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    _write_source_run(result_root)
    (result_root / "010").mkdir(parents=True)

    def fail_if_generating(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        raise AssertionError("planning generation should not run")

    monkeypatch.setattr(expanded_runner, "build_target_environment_plan", fail_if_generating)
    with pytest.raises(ValueError, match="overwrite=False"):
        expanded_runner.run_expanded_dense_archive_planning(
            run_id=10,
            source_planning_run_id=8,
            result_root=result_root,
        )


def test_expanded_planning_candidate_count_threshold_raises_before_writing(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    _write_source_run(result_root)

    with pytest.raises(RuntimeError, match="too few candidates"):
        expanded_runner.run_expanded_dense_archive_planning(
            run_id=10,
            source_planning_run_id=8,
            result_root=result_root,
            pilot_start_states_per_family_target_direction=1,
            required_min_candidate_rows=20000,
        )

    assert not (result_root / "010").exists()


def test_existing_pilot_runner_can_load_expanded_planning_tables_from_temp_root(
    expanded_outputs: dict[str, Path],
) -> None:
    config = pilot_runner.DensePilotSweepConfig(
        run_id=11,
        planning_run_id=10,
        max_trials=20000,
        result_root=expanded_outputs["root"].parent,
    )
    starts, candidates = pilot_runner._load_planning_tables(config)
    selected = pilot_runner._select_pilot_candidates(candidates, config)

    assert len(starts) == 10200
    assert len(candidates) == 20400
    assert len(selected) == 20000
    assert (
        pilot_runner._pilot_scale_status(
            available_count=len(candidates),
            selected_count=len(selected),
            max_trials=20000,
        )
        == "meets_sun24_minimum"
    )


def test_expanded_planning_cli_has_complete_import_path(tmp_path: Path) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    _write_source_run(result_root)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCENARIOS_DIR / "run_expanded_dense_archive_planning.py"),
            "--run-id",
            "10",
            "--source-planning-run-id",
            "8",
            "--result-root",
            str(result_root),
            "--pilot-starts-per-family-target-direction",
            "1",
            "--required-min-candidate-rows",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "expanded_dense_archive_planning_outputs=" in completed.stdout
    assert (
        result_root
        / "010"
        / "metrics"
        / "equal_branch_dry_run_candidate_inventory_pilot_s010.csv"
    ).exists()


def _read_manifest(paths: dict[str, Path]) -> dict[str, object]:
    return json.loads(paths["manifest_json"].read_text(encoding="ascii"))


def _write_source_run(result_root: Path) -> None:
    source = result_root / "008" / "manifests"
    source.mkdir(parents=True)
    (source / "source_marker.json").write_text('{"run_id": 8}\n', encoding="ascii")
