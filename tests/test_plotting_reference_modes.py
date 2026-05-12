from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "03_Control" / "05_Results"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plotting import build_composite_figure, load_scenario_plot_data  # noqa: E402


def test_control_and_environment_references_have_distinct_styles(tmp_path: Path) -> None:
    data = load_scenario_plot_data("s0_no_wind", seed=3, output_root=tmp_path)
    assert {series.key for series in data.references} == {
        "control_reference",
        "environment_reference",
    }

    fig = build_composite_figure(data)
    try:
        ax3d = fig.axes[0]
        by_label = {
            line.get_label(): line
            for line in ax3d.lines
            if line.get_label()
            in {"Control reference", "Environment reference"}
        }
        actual_collections = [
            collection
            for collection in ax3d.collections
            if isinstance(collection, Line3DCollection)
            and collection.get_label() == "Actual"
        ]
        assert actual_collections
        assert set(by_label) == {"Control reference", "Environment reference"}
        assert mcolors.to_hex(by_label["Control reference"].get_color()) == "#e66a2c"
        assert mcolors.to_hex(by_label["Environment reference"].get_color()) == "#7b2cbf"
        assert by_label["Control reference"].get_alpha() == 0.55
        assert by_label["Environment reference"].get_alpha() == 0.55
        assert by_label["Control reference"].get_linewidth() == 1.25
        assert by_label["Environment reference"].get_linewidth() == 1.25
        assert by_label["Control reference"].get_linestyle() != "-"
        assert by_label["Environment reference"].get_linestyle() != "-"
        assert by_label["Control reference"].get_linestyle() != by_label["Environment reference"].get_linestyle()

        actual_colors = actual_collections[0].get_colors()
        assert mcolors.to_hex(actual_colors[0]) == "#20b6c7"
        assert mcolors.to_hex(actual_colors[-1]) == "#00244c"
    finally:
        plt.close(fig)
