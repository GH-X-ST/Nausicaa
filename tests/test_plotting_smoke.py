from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "03_Control" / "05_Results"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from plotting import generate_scenario_figure  # noqa: E402


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
