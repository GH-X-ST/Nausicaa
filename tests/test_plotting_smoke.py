from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "03_Control" / "05_Results"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

SCENARIO_DIR = REPO_ROOT / "03_Control" / "04_Scenarios"
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from arena import ArenaConfig, tracker_bounds  # noqa: E402
from plotting import (  # noqa: E402
    build_composite_figure,
    generate_scenario_figure,
    load_scenario_plot_data,
)


def test_generate_scenario_figure_writes_png(tmp_path: Path) -> None:
    paths = generate_scenario_figure(
        "s0_no_wind",
        seed=2,
        output_root=tmp_path,
        save_png=True,
        save_pdf=False,
    )
    assert set(paths) == {"png"}
    assert paths["png"].exists()
    assert paths["png"].stat().st_size > 0
    assert paths["png"].name == "flight_trajectory_and_control_commands.png"
    result_root = paths["png"].parents[1]
    assert result_root.parent.name == "flight_case_results"
    assert paths["png"].parent.name == "figures"
    analysis_dir = result_root / "analysis_data"
    assert (analysis_dir / "actual_rollout.csv").exists()
    assert (analysis_dir / "actual_metrics.csv").exists()
    assert (analysis_dir / "control_reference_rollout.csv").exists()
    assert (analysis_dir / "environment_reference_rollout.csv").exists()
    assert (analysis_dir / "manifest.json").exists()
    assert not list(result_root.rglob("*.pdf"))
    assert not (result_root / "digital_results").exists()
    assert "s0" not in result_root.name.lower()
    assert result_root.name.startswith("baseline_no_wind_glide")


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
    finally:
        plt.close(fig)
