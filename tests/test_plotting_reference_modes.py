from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


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
            in {"Actual", "Control reference", "Environment reference"}
        }
        assert set(by_label) == {"Actual", "Control reference", "Environment reference"}
        assert by_label["Actual"].get_linestyle() == "-"
        assert by_label["Control reference"].get_linestyle() != "-"
        assert by_label["Environment reference"].get_linestyle() != "-"
        assert by_label["Control reference"].get_linestyle() != by_label["Environment reference"].get_linestyle()
    finally:
        plt.close(fig)
