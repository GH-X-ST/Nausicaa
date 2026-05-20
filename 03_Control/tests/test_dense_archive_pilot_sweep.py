from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_dense_archive_pilot_sweep as pilot_runner


TRACKED_RESULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "05_Results"
    / "10_dense_archive_planning"
)


class FakeWindModel:
    name = "fake_measured_w1"
    source = "fake_test_source"
    z_axis_m = np.array([0.0, 3.0])

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        return np.column_stack(
            [
                np.zeros(points.shape[0]),
                np.zeros(points.shape[0]),
                np.full(points.shape[0], 0.1),
            ]
        )


def test_pilot_runner_outputs_are_isolated_and_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    _write_planning_tables(result_root, _start_rows(), _candidate_rows())
    default_run_root = TRACKED_RESULT_ROOT / "109"
    assert not default_run_root.exists()
    load_calls: list[str] = []
    dt_calls: list[float] = []

    _patch_lightweight_replay(monkeypatch, load_calls=load_calls, dt_calls=dt_calls)
    paths = pilot_runner.run_dense_archive_pilot_sweep(
        run_id=109,
        planning_run_id=8,
        max_trials=4,
        result_root=result_root,
        overwrite=True,
        dt_s=0.02,
        horizon_s=0.04,
    )

    assert not default_run_root.exists()
    assert paths["trial_descriptors_csv"] == (
        result_root / "109" / "metrics" / "dense_pilot_trial_descriptors_s109.csv"
    )
    assert paths["envelope_map_csv"] == (
        result_root / "109" / "metrics" / "dense_pilot_envelope_map_s109.csv"
    )
    assert paths["cluster_representatives_csv"] == (
        result_root
        / "109"
        / "metrics"
        / "dense_pilot_cluster_representatives_s109.csv"
    )
    assert paths["manifest_json"] == (
        result_root / "109" / "manifests" / "dense_pilot_sweep_manifest_s109.json"
    )
    assert paths["report_md"] == (
        result_root / "109" / "reports" / "dense_pilot_sweep_report_s109.md"
    )
    for path in paths.values():
        assert Path(path).exists()
        assert result_root in Path(path).resolve().parents or Path(path) == result_root

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert manifest["run_id"] == 109
    assert manifest["planning_run_id"] == 8
    assert manifest["available_planning_candidate_count"] == 4
    assert manifest["selected_trial_count"] == 4
    assert manifest["max_trials_requested"] == 4
    assert manifest["sun24_min_pilot_trials"] == 5000
    assert (
        manifest["pilot_scale_status"]
        == "reduced_below_sun24_minimum_due_to_max_trials"
    )
    assert manifest["trial_count_executed"] == 4
    assert manifest["latency_case"] == "nominal"
    assert manifest["dt_s"] == 0.02
    assert manifest["horizon_s"] == 0.04
    assert manifest["pilot_sweep_performed"] is True
    assert manifest["production_dense_archive_performed"] is False
    assert manifest["w0_full_archive_performed"] is False
    assert manifest["w1_full_archive_performed"] is False
    assert manifest["envelope_map_scaffold_implemented"] is True
    assert manifest["clustering_scaffold_implemented"] is True
    assert manifest["hardware_or_mission_claim"] is False
    assert manifest["branch_local_decisions_only"] is True
    report = paths["report_md"].read_text(encoding="ascii")
    assert "not a production W0/W1 archive" in report
    assert "Pilot scale status" in report

    descriptors = pd.read_csv(paths["trial_descriptors_csv"])
    labels = dict(zip(descriptors["sample_id"], descriptors["failure_label"]))
    assert labels["sample_blank"] == "success"
    assert labels["sample_target"] == "target_miss"
    assert labels["sample_safety"] == "true_safety_violation"
    assert labels["sample_nonfinite"] == "nonfinite_state"
    assert set(descriptors["descriptor_status"]) == {
        "replay_evaluated",
        "entry_invalid",
        "nonfinite_state",
    }
    assert load_calls == ["fake_w1_model"]
    assert set(dt_calls) == {0.02}

    envelope = pd.read_csv(paths["envelope_map_csv"])
    assert "latency_case" in envelope.columns
    assert "evaluated_trial_count" in envelope.columns
    clusters = pd.read_csv(paths["cluster_representatives_csv"])
    assert "cluster_key" in clusters.columns


def test_pilot_runner_reports_available_row_scale_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    _write_planning_tables(result_root, _start_rows(), _candidate_rows())
    _patch_lightweight_replay(monkeypatch)

    paths = pilot_runner.run_dense_archive_pilot_sweep(
        run_id=114,
        planning_run_id=8,
        max_trials=5000,
        result_root=result_root,
        overwrite=True,
        horizon_s=0.04,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    report = paths["report_md"].read_text(encoding="ascii")
    assert manifest["available_planning_candidate_count"] == 4
    assert manifest["selected_trial_count"] == 4
    assert manifest["max_trials_requested"] == 5000
    assert manifest["sun24_min_pilot_trials"] == 5000
    assert (
        manifest["pilot_scale_status"]
        == "reduced_below_sun24_minimum_due_to_available_planning_rows"
    )
    assert "Available planning candidates: `4`" in report
    assert (
        "Pilot scale status: "
        "`reduced_below_sun24_minimum_due_to_available_planning_rows`"
    ) in report


def test_branch_environment_round_robin_selection_is_deterministic() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row("a1", "s1", "single_fan_branch", "single_fan", "W0_single_fan_branch"),
            _candidate_row("a2", "s2", "single_fan_branch", "single_fan", "W0_single_fan_branch"),
            _candidate_row("b1", "s3", "single_fan_branch", "single_fan", "W1_single_fan"),
            _candidate_row("b2", "s4", "single_fan_branch", "single_fan", "W1_single_fan"),
            _candidate_row("c1", "s5", "four_fan_branch", "four_fan", "W0_four_fan_branch"),
            _candidate_row("c2", "s6", "four_fan_branch", "four_fan", "W0_four_fan_branch"),
            _candidate_row("d1", "s7", "four_fan_branch", "four_fan", "W1_four_fan"),
            _candidate_row("d2", "s8", "four_fan_branch", "four_fan", "W1_four_fan"),
        ]
    )
    config = pilot_runner.DensePilotSweepConfig(max_trials=4)

    selected_a = pilot_runner._select_pilot_candidates(candidates, config)
    selected_b = pilot_runner._select_pilot_candidates(candidates.sample(frac=1.0), config)

    assert [row["candidate_id"] for row in selected_a] == ["c1", "d1", "a1", "b1"]
    assert [row["candidate_id"] for row in selected_b] == ["c1", "d1", "a1", "b1"]


def test_resolved_output_planning_overlap_rejected_and_siblings_allowed(
    tmp_path: Path,
) -> None:
    config = pilot_runner.DensePilotSweepConfig(
        run_id=9,
        planning_run_id=8,
        result_root=tmp_path,
    )

    pilot_runner._validate_output_guardrails(
        config,
        _outputs_at(tmp_path / "009"),
    )
    with pytest.raises(ValueError, match="output/planning overlap"):
        pilot_runner._validate_output_guardrails(config, _outputs_at(tmp_path / "008"))
    with pytest.raises(ValueError, match="output/planning overlap"):
        pilot_runner._validate_output_guardrails(
            config,
            _outputs_at(tmp_path / "008" / "nested"),
        )
    with pytest.raises(ValueError, match="output/planning overlap"):
        pilot_runner._validate_output_guardrails(config, _outputs_at(tmp_path))


def test_default_sibling_planning_and_output_runs_are_allowed() -> None:
    config = pilot_runner.DensePilotSweepConfig(run_id=9, planning_run_id=8)

    pilot_runner._validate_output_guardrails(
        config,
        pilot_runner._output_paths(config),
    )


def test_output_exists_without_overwrite_raises_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    _write_planning_tables(result_root, _start_rows(), _candidate_rows())
    output_root = result_root / "112"
    output_root.mkdir(parents=True)
    replay_called = False

    def fail_if_replayed(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        nonlocal replay_called
        replay_called = True
        raise AssertionError("replay should not run when output exists")

    monkeypatch.setattr(pilot_runner, "_run_pilot_replays", fail_if_replayed)
    with pytest.raises(ValueError, match="overwrite=False"):
        pilot_runner.run_dense_archive_pilot_sweep(
            run_id=112,
            planning_run_id=8,
            max_trials=1,
            result_root=result_root,
            overwrite=False,
        )

    assert replay_called is False


def test_missing_planning_csvs_create_no_output_directory(tmp_path: Path) -> None:
    result_root = tmp_path / "10_dense_archive_planning"

    with pytest.raises(FileNotFoundError, match="missing planning"):
        pilot_runner.run_dense_archive_pilot_sweep(
            run_id=113,
            planning_run_id=8,
            max_trials=1,
            result_root=result_root,
            overwrite=True,
        )

    assert not (result_root / "113").exists()


def test_run_id_and_default_root_guardrails() -> None:
    with pytest.raises(ValueError, match="must differ"):
        pilot_runner.run_dense_archive_pilot_sweep(run_id=8, planning_run_id=8)
    with pytest.raises(ValueError, match="greater than planning_run_id"):
        pilot_runner.run_dense_archive_pilot_sweep(run_id=7, planning_run_id=8)
    with pytest.raises(ValueError, match="protected planning run"):
        pilot_runner.run_dense_archive_pilot_sweep(
            run_id=8,
            planning_run_id=7,
            overwrite=True,
        )


def test_pilot_scale_status_priority() -> None:
    assert (
        pilot_runner._pilot_scale_status(
            available_count=10,
            selected_count=0,
            max_trials=5000,
        )
        == "no_trials_selected"
    )
    assert (
        pilot_runner._pilot_scale_status(
            available_count=6000,
            selected_count=5000,
            max_trials=5000,
        )
        == "meets_sun24_minimum"
    )
    assert (
        pilot_runner._pilot_scale_status(
            available_count=100,
            selected_count=4,
            max_trials=4,
        )
        == "reduced_below_sun24_minimum_due_to_max_trials"
    )
    assert (
        pilot_runner._pilot_scale_status(
            available_count=100,
            selected_count=4,
            max_trials=5000,
        )
        == "reduced_below_sun24_minimum_due_to_available_planning_rows"
    )
    with pytest.raises(RuntimeError, match="internal consistency"):
        pilot_runner._pilot_scale_status(
            available_count=6000,
            selected_count=4999,
            max_trials=6000,
        )


def test_analytic_debug_proxy_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "10_dense_archive_planning"
    starts = [_start_row("sample_proxy", 2.0, target=30.0)]
    candidates = [
        _candidate_row(
            "candidate_proxy",
            "sample_proxy",
            "single_fan_branch",
            "single_fan",
            "W1_single_fan",
            updraft_model_id="analytic_debug_proxy",
            target=30.0,
        )
    ]
    _write_planning_tables(result_root, starts, candidates)
    _patch_lightweight_replay(monkeypatch)

    with pytest.raises(ValueError, match="analytic_debug_proxy"):
        pilot_runner.run_dense_archive_pilot_sweep(
            run_id=110,
            planning_run_id=8,
            max_trials=1,
            result_root=result_root,
            overwrite=True,
        )


def _outputs_at(root: Path) -> pilot_runner.DensePilotSweepOutputs:
    return pilot_runner.DensePilotSweepOutputs(
        root=root,
        trial_descriptors_csv=root / "metrics" / "trial.csv",
        envelope_map_csv=root / "metrics" / "envelope.csv",
        cluster_representatives_csv=root / "metrics" / "clusters.csv",
        manifest_json=root / "manifests" / "manifest.json",
        report_md=root / "reports" / "report.md",
    )


def _patch_lightweight_replay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_calls: list[str] | None = None,
    dt_calls: list[float] | None = None,
) -> None:
    monkeypatch.setattr(pilot_runner, "build_nausicaa_glider", lambda: object())
    monkeypatch.setattr(pilot_runner, "adapt_glider", lambda aircraft: aircraft)

    def fake_load(model_id: str) -> FakeWindModel:
        if load_calls is not None:
            load_calls.append(str(model_id))
        return FakeWindModel()

    def fake_rk4_step(
        x: np.ndarray,
        delta_cmd_rad: np.ndarray,
        dt_s: float,
        aircraft: object,
        wind_model: object,
        wind_mode: str,
        actuator_tau_s: tuple[float, float, float],
    ) -> np.ndarray:
        del delta_cmd_rad, aircraft, wind_model, wind_mode, actuator_tau_s
        if dt_calls is not None:
            dt_calls.append(float(dt_s))
        state = np.asarray(x, dtype=float).copy()
        marker = round(float(state[0]), 1)
        if marker == 2.3:
            state[0] = 7.0
        elif marker == 2.4:
            state[0] = np.nan
        else:
            state[0] += 0.01
        return state

    monkeypatch.setattr(pilot_runner, "load_updraft_model", fake_load)
    monkeypatch.setattr(pilot_runner, "rk4_step", fake_rk4_step)


def _write_planning_tables(
    result_root: Path,
    starts: list[dict[str, object]],
    candidates: list[dict[str, object]],
) -> None:
    metrics = result_root / "008" / "metrics"
    metrics.mkdir(parents=True)
    pd.DataFrame(starts).to_csv(
        metrics / "equal_branch_start_state_manifest_pilot_s008.csv",
        index=False,
    )
    pd.DataFrame(candidates).to_csv(
        metrics / "equal_branch_dry_run_candidate_inventory_pilot_s008.csv",
        index=False,
    )


def _start_rows() -> list[dict[str, object]]:
    return [
        _start_row("sample_blank", 2.0, target=""),
        _start_row("sample_target", 2.2, target=30.0),
        _start_row("sample_safety", 2.3, target=30.0),
        _start_row("sample_nonfinite", 2.4, target=30.0),
    ]


def _candidate_rows() -> list[dict[str, object]]:
    return [
        _candidate_row(
            "candidate_blank",
            "sample_blank",
            "single_fan_branch",
            "single_fan",
            "W0_single_fan_branch",
            family="glide",
            target="",
        ),
        _candidate_row(
            "candidate_target",
            "sample_target",
            "single_fan_branch",
            "single_fan",
            "W1_single_fan",
            target=30.0,
        ),
        _candidate_row(
            "candidate_safety",
            "sample_safety",
            "four_fan_branch",
            "four_fan",
            "W0_four_fan_branch",
            target=30.0,
        ),
        _candidate_row(
            "candidate_nonfinite",
            "sample_nonfinite",
            "four_fan_branch",
            "four_fan",
            "W1_four_fan",
            target=30.0,
        ),
    ]


def _start_row(sample_id: str, x_w_m: float, *, target: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 101,
        "sampling_round": "pilot_round_0",
        "fan_layout": "single_fan",
        "layout_branch_id": "single_fan_branch",
        "fan_config_id": "single_fan_nominal_updraft",
        "updraft_model_id": "fake_w1_model",
        "start_class": "favourable",
        "family": "mild_bank",
        "target_heading_deg": target,
        "direction_sign": 1,
        "environment_role": "dry_air_capable",
        "x_w_m": x_w_m,
        "y_w_m": 1.0,
        "z_w_m": 1.0,
        "speed_m_s": 6.0,
        "phi_rad": 0.0,
        "theta_rad": 0.0,
        "psi_rad": 0.0,
        "u_m_s": 6.0,
        "v_m_s": 0.0,
        "w_m_s": 0.0,
        "p_rad_s": 0.0,
        "q_rad_s": 0.0,
        "r_rad_s": 0.0,
        "updraft_center_x_m": 3.0,
        "updraft_center_y_m": 2.0,
        "updraft_relative_radius_m": 0.5,
        "updraft_relative_azimuth_rad": 0.0,
        "updraft_relative_height_m": 0.67,
        "updraft_sector_label": "test",
    }
    row.update(_wing_fields())
    return row


def _candidate_row(
    candidate_id: str,
    sample_id: str,
    layout_branch_id: str,
    fan_layout: str,
    test_environment_mode: str,
    *,
    family: str = "mild_bank",
    target: object = 30.0,
    updraft_model_id: str = "fake_w1_model",
) -> dict[str, object]:
    paired = "W1_single_fan" if test_environment_mode.startswith("W0_") else "W0_single_fan_branch"
    if fan_layout == "four_fan":
        paired = "W1_four_fan" if test_environment_mode.startswith("W0_") else "W0_four_fan_branch"
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "seed": 101,
        "sampling_round": "pilot_round_0",
        "fan_layout": fan_layout,
        "layout_branch_id": layout_branch_id,
        "fan_config_id": f"{fan_layout}_config",
        "updraft_model_id": "no_updraft_dry_air"
        if test_environment_mode.startswith("W0_")
        else updraft_model_id,
        "test_environment_mode": test_environment_mode,
        "paired_environment_mode": paired,
        "family": family,
        "target_heading_deg": target,
        "direction_sign": 1,
        "start_class": "favourable",
        "environment_role": "dry_air_capable",
        "validity_gate_role": "pilot",
        "acceptance_interpretation": "pilot",
        "branch_decision_scope": "branch_local_only_no_cross_layout_decision_transfer",
    }
    row.update(_wing_fields())
    return row


def _wing_fields() -> dict[str, object]:
    return {
        "wind_descriptor_status": "wind_model_evaluated",
        "wind_descriptor_environment_mode": "W1_single_fan",
        "wind_descriptor_model_id": "fake_w1_model",
        "wind_descriptor_model_source": "fake",
        "w_cg_m_s": 0.1,
        "w_wing_mean_m_s": 0.1,
        "w_left_m_s": 0.11,
        "w_right_m_s": 0.09,
        "delta_w_lr_m_s": 0.02,
        "w_panel_max_m_s": 0.12,
        "w_panel_min_m_s": 0.08,
        "spanwise_w_gradient_m_s_per_m": 0.01,
        "local_updraft_uncertainty_m_s": np.nan,
        "local_updraft_uncertainty_status": "not_available",
        "wing_panel_sample_count": 11,
    }
