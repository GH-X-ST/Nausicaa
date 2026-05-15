from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.patches import Circle


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "03_Control" / "05_Results"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

SCENARIO_DIR = REPO_ROOT / "03_Control" / "04_Scenarios"
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from arena import ArenaConfig, tracker_bounds  # noqa: E402
from plotting import (  # noqa: E402
    ATTITUDE_PLOT_SPECS,
    build_composite_figure,
    build_figure_c_flight_state_alpha_beta,
    build_figure_e_2d_trajectory_geometry,
    FIGURE_STEMS,
    generate_scenario_figure,
    load_scenario_plot_data,
)


def test_generate_scenario_figure_writes_five_canonical_pngs(tmp_path: Path) -> None:
    paths = generate_scenario_figure(
        "s0_no_wind",
        seed=2,
        output_root=tmp_path,
        save_png=True,
        save_pdf=False,
    )
    expected_keys = {f"{label}_png" for label in FIGURE_STEMS}
    assert set(paths) == expected_keys
    for label, stem in FIGURE_STEMS.items():
        path = paths[f"{label}_png"]
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.name == f"{stem}.png"
    result_root = paths["A_png"].parent
    assert result_root.name == "002"
    assert result_root.parent.name == "01_basic_no_wind_glide"
    assert result_root.parents[1].name == "00_smoke"
    assert (result_root / "actual_rollout.csv").exists()
    assert (result_root / "actual_metrics.csv").exists()
    assert (result_root / "control_reference_rollout.csv").exists()
    assert (result_root / "environment_reference_rollout.csv").exists()
    assert (result_root / "manifest.json").exists()
    manifest = json.loads((result_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures"]["png"] == {
        label: f"{stem}.png"
        for label, stem in FIGURE_STEMS.items()
    }
    assert not list(result_root.glob("*.pdf"))
    assert not (result_root / "digital_results").exists()
    assert not (result_root / "analysis_data").exists()
    assert not (result_root / "figures").exists()
    assert "s0" not in result_root.parent.name.lower()
    assert plt.get_fignums() == []


def test_generate_recovery_scenario_figure_uses_standard_result_layout(tmp_path: Path) -> None:
    paths = generate_scenario_figure(
        "s4_full_recovery_no_wind",
        seed=1,
        output_root=tmp_path,
        save_png=True,
        save_pdf=False,
    )
    result_root = paths["A_png"].parent

    assert result_root.name == "001"
    assert result_root.parent.name == "04_recovery_no_wind"
    assert result_root.parents[1].name == "03_primitives"
    assert (result_root / "actual_rollout.csv").exists()
    assert (result_root / "actual_metrics.csv").exists()
    assert (result_root / "control_reference_rollout.csv").exists()
    assert (result_root / "environment_reference_rollout.csv").exists()
    assert (result_root / "manifest.json").exists()
    assert not (result_root / "analysis_data").exists()
    assert not (result_root / "figures").exists()


def test_figure_c_attitude_panels_use_pitch_roll_yaw_order(tmp_path: Path) -> None:
    data = load_scenario_plot_data("s0_no_wind", seed=2, output_root=tmp_path)
    fig = build_figure_c_flight_state_alpha_beta(data)
    try:
        assert [spec[0] for spec in ATTITUDE_PLOT_SPECS] == ["theta", "phi", "psi"]
        assert [ax.get_ylabel() for ax in fig.axes[:3]] == [
            r"Pitch $\theta$ (deg)",
            r"Roll $\phi$ (deg)",
            r"Yaw $\psi$ (deg)",
        ]
    finally:
        plt.close(fig)


def test_composite_figure_uses_tracker_limit_axes(tmp_path: Path) -> None:
    data = load_scenario_plot_data("s0_no_wind", seed=2, output_root=tmp_path)
    fig = build_composite_figure(data)
    try:
        ax3d = fig.axes[0]
        bounds = {
            "x_w": (0.0, 8.0),
            "y_w": (0.0, 4.8),
            "z_w": (0.0, 3.5),
        }

        # The plotted axes follow the measured tracker box; the larger room is not drawn.
        assert ax3d.get_xlim() == pytest.approx(bounds["x_w"])
        assert ax3d.get_ylim() == pytest.approx(bounds["y_w"])
        assert ax3d.get_zlim() == pytest.approx(bounds["z_w"])
        assert tracker_bounds(ArenaConfig()) == bounds
        labels = {line.get_label() for line in ax3d.lines}
        assert "Tracker limit" in labels
        assert "True safety volume" in labels
        assert "Room context" not in labels
        safety_line = next(
            line for line in ax3d.lines if line.get_label() == "True safety volume"
        )
        assert mcolors.to_hex(safety_line.get_color()) == "#595959"
        assert safety_line.get_linestyle() == ":"
        assert safety_line.get_linewidth() == 0.75

        command_axes = fig.axes[2:5]
        expected_limits = ((-26.0, 22.0), (-30.0, 22.0), (-35.0, 28.0))
        for ax, limits in zip(command_axes, expected_limits):
            dotted_y = sorted(
                {
                    round(float(line.get_ydata()[0]), 6)
                    for line in ax.lines
                    if line.get_linestyle() == ":"
                    and np.allclose(line.get_ydata(), line.get_ydata()[0])
                }
            )
            assert dotted_y == pytest.approx(sorted(limits))
    finally:
        plt.close(fig)


def test_figure_e_keeps_layout_without_no_wind_heat_map(tmp_path: Path) -> None:
    data = load_scenario_plot_data("s0_no_wind", seed=5, output_root=tmp_path)
    fig = build_figure_e_2d_trajectory_geometry(data)
    try:
        assert not [
            collection
            for ax in fig.axes
            for collection in ax.collections
            if isinstance(collection, QuadMesh)
        ]
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_figure_e_keeps_wind_case_to_fan_outlet_geometry(tmp_path: Path) -> None:
    data = load_scenario_plot_data("s4_gaussian_single_panel", seed=1, output_root=tmp_path)
    fig = build_figure_e_2d_trajectory_geometry(data)
    try:
        assert not [
            collection
            for ax in fig.axes
            for collection in ax.collections
            if isinstance(collection, QuadMesh)
        ]
        assert len(fig.axes) == 2
        assert [
            patch
            for patch in fig.axes[0].patches
            if isinstance(patch, Circle) and patch.get_label() == "Fan outlet"
        ]
        safety_patches = [
            patch
            for ax in fig.axes
            for patch in ax.patches
            if patch.get_label() == "True safety volume"
        ]
        assert safety_patches
        assert all(
            mcolors.to_hex(patch.get_edgecolor()) == "#595959"
            for patch in safety_patches
        )
        assert all(patch.get_linestyle() == ":" for patch in safety_patches)
        assert all(patch.get_linewidth() == 0.75 for patch in safety_patches)
    finally:
        plt.close(fig)
