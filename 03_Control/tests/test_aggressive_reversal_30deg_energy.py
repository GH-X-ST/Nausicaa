from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggressive_reversal_ocp import (
    RELAXED_TERMINAL_SPEED_M_S,
    SEED_FAMILIES,
    STRICT_TERMINAL_SPEED_M_S,
    THIRTY_DEG_SEED_FAMILIES,
    AggressiveReversalOcpConfig,
    descent_required_to_speed_m,
    energy_audit_for_trajectory,
    metrics_for_candidate,
    phase_labels_for_family,
    phase_seed_command_profile,
    phase_shape_audit,
    recoverable_speed_from_energy_m_s,
    seed_family_inventory_for_target,
    specific_energy_height_m,
    target_config,
)
from run_aggressive_reversal_search import run_search


def _state_from_speed_z_heading(
    speed_m_s: np.ndarray,
    z_w_m: np.ndarray,
    heading_deg: np.ndarray,
) -> np.ndarray:
    state = np.zeros((speed_m_s.size, 15), dtype=float)
    state[:, 0] = np.linspace(1.25, 2.10, speed_m_s.size)
    state[:, 1] = 2.2
    state[:, 2] = z_w_m
    state[:, 6] = speed_m_s
    state[:, 5] = np.deg2rad(heading_deg)
    return state


def test_energy_height_and_descent_to_speed_calculations() -> None:
    state = _state_from_speed_z_heading(
        np.array([6.0, 4.0]),
        np.array([1.5, 1.2]),
        np.array([0.0, 5.0]),
    )

    energy = specific_energy_height_m(state)

    assert energy[0] == pytest.approx(1.5 + 6.0**2 / (2.0 * 9.81))
    assert descent_required_to_speed_m(4.0, 5.0) == pytest.approx(9.0 / 19.62)
    assert descent_required_to_speed_m(5.0, 4.0) == 0.0
    assert recoverable_speed_from_energy_m_s(energy[1], 0.8) > 4.0


def test_30deg_seed_families_are_target_specific_and_bounded() -> None:
    config = target_config(30.0)
    time_s = np.arange(int(round(config.t_final_s / config.dt_s)) + 1) * config.dt_s

    assert seed_family_inventory_for_target(30.0) == SEED_FAMILIES
    for family_name in THIRTY_DEG_SEED_FAMILIES:
        command = phase_seed_command_profile(config, time_s, family_name)
        phase = phase_labels_for_family(family_name, time_s, config.t_final_s)

        assert command.shape == (time_s.size, 3)
        assert np.all(np.isfinite(command))
        assert np.max(np.abs(command)) <= 1.0
        assert "slow_redirect" in phase
        assert "unload_descend" in phase


def test_unload_exit_shape_evidence_uses_descent_and_speed_gain() -> None:
    config = AggressiveReversalOcpConfig(target_heading_deg=30.0)
    speed = np.array([6.5, 6.7, 6.4, 5.8, 4.6, 4.5, 4.7, 4.8, 5.2, 5.35])
    z_w = np.array([1.8, 1.74, 1.82, 2.02, 2.05, 2.02, 1.95, 1.82, 1.72, 1.64])
    heading = np.array([0.0, 0.0, 2.0, 6.0, 14.0, 22.0, 28.0, 30.0, 30.0, 30.0])
    phase = (
        "entry",
        "pre_dive_accelerate",
        "pitch_brake",
        "pitch_brake",
        "slow_redirect",
        "slow_redirect",
        "heading_capture",
        "unload_descend",
        "unload_descend",
        "exit_glide",
    )
    state = _state_from_speed_z_heading(speed, z_w, heading)

    shape = phase_shape_audit(config, state, phase, terminal_recoverable=True)
    energy = energy_audit_for_trajectory(state, phase)

    assert shape["unload_descent"] is True
    assert shape["unload_speed_gain"] is True
    assert shape["unload_exit_descent_m"] >= 0.15
    assert shape["unload_exit_speed_gain_m_s"] >= 0.50
    assert energy["ideal_descent_to_5ms_m"] == 0.0


def test_heading_without_unload_recovery_is_speed_collapse_pitch_redirect() -> None:
    config = AggressiveReversalOcpConfig(target_heading_deg=30.0)
    time_s = np.arange(10, dtype=float) * config.dt_s
    speed = np.linspace(6.5, 2.2, time_s.size)
    z_w = np.linspace(1.8, 2.1, time_s.size)
    heading = np.linspace(0.0, 30.0, time_s.size)
    phase = (
        "entry",
        "pitch_brake",
        "pitch_brake",
        "slow_redirect",
        "slow_redirect",
        "slow_redirect",
        "heading_capture",
        "unload_descend",
        "unload_descend",
        "exit_glide",
    )
    state = _state_from_speed_z_heading(speed, z_w, heading)
    command = np.zeros((time_s.size, 3), dtype=float)

    metrics = metrics_for_candidate(config, time_s, state, command, command, phase)

    assert metrics["heading_success"] is True
    assert metrics["manoeuvre_shape_class"] == "speed_collapse_pitch_redirect"
    assert metrics["notes"] == "speed_collapse_pitch_redirect"
    assert metrics["primitive_success"] is False
    assert metrics["terminal_speed_m_s"] < RELAXED_TERMINAL_SPEED_M_S


def test_run002_records_only_30deg_energy_and_phase_audits(monkeypatch, tmp_path: Path) -> None:
    import run_aggressive_reversal_search as runner

    monkeypatch.setattr(runner, "DEFAULT_RESULTS_ROOT", tmp_path)
    outputs = run_search(
        run_id=2,
        overwrite=True,
        targets_deg=(30.0,),
        max_ipopt_iter=1,
        ocp_max_cpu_time_s=0.05,
        ocp_node_count=1,
    )
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="ascii"))
    summary = pd.read_csv(outputs["summary_csv"])
    row = summary.iloc[0]
    phase_audit = pd.read_csv(
        outputs["root"] / "metrics" / "aggressive_reversal_target_030_phase_audit_s002.csv"
    )

    assert manifest["targets_completed_deg"] == [30.0]
    assert manifest["thirty_degree_physics_first_refinement_only"] is True
    assert row["target_heading_deg"] == 30.0
    assert row["selected_family"] in SEED_FAMILIES
    assert "manoeuvre_shape_class" in summary.columns
    assert "active_tradeoff" in summary.columns
    assert "specific_energy_lost_m" in summary.columns
    assert set(
        [
            "speed_m_s",
            "z_w",
            "specific_energy_height_m",
            "alpha_deg",
            "theta_deg",
            "heading_change_deg",
            "phase",
        ]
    ).issubset(phase_audit.columns)
    if float(row["terminal_speed_m_s"]) < STRICT_TERMINAL_SPEED_M_S:
        assert str(row["strict_30deg_primitive_success"]) != "True"
    assert not any(Path(value).is_absolute() for value in manifest["output_files"].values())
