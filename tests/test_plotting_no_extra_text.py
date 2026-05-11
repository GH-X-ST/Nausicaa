from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Annotation


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "03_Control" / "05_Results"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plotting import build_composite_figure, load_scenario_plot_data  # noqa: E402


def test_composite_figure_has_no_titles_or_annotations(tmp_path: Path) -> None:
    data = load_scenario_plot_data("s0_no_wind", seed=4, output_root=tmp_path)
    fig = build_composite_figure(data)
    try:
        assert fig._suptitle is None
        assert not [text for text in fig.texts if text.get_text()]
        assert all(ax.get_title() == "" for ax in fig.axes)
        assert not [
            artist
            for ax in fig.axes
            for artist in ax.get_children()
            if isinstance(artist, Annotation) and artist.get_text()
        ]
    finally:
        plt.close(fig)
