from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "03_Control/04_Scenarios",
    "03_Control/05_Results",
):
    path = REPO_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plotting import _normalise_actual_outputs, scenario_output_paths  # noqa: E402
from run_one import _relative_output_path  # noqa: E402


def test_relative_output_path_uses_redirected_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "redirected_results"
    path = output_root / "metrics" / "case_seed1.csv"

    relative = _relative_output_path(path, output_root)

    assert relative == "metrics/case_seed1.csv"


def test_normalised_metrics_use_analysis_data_file_names(tmp_path: Path) -> None:
    scenario_id = "s0_no_wind"
    seed = 7
    output_root = tmp_path / "results"
    runner_root = output_root / "flight_case_results" / "case_seed_007" / "_run"
    paths = scenario_output_paths(scenario_id, seed, output_root)
    metrics_dir = runner_root / "metrics"
    logs_dir = runner_root / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    raw_log = logs_dir / f"{scenario_id}_seed{seed}.csv"
    raw_metrics = metrics_dir / f"{scenario_id}_seed{seed}.csv"
    raw_candidates = metrics_dir / f"{scenario_id}_seed{seed}_governor_candidates.csv"
    raw_log.write_text("t_s\n0.0\n", encoding="utf-8")
    raw_metrics.write_text("scenario_id\ns0_no_wind\n", encoding="utf-8")
    raw_candidates.write_text("primitive_name,selected\nnominal_glide,True\n", encoding="utf-8")

    row = {
        "scenario_id": scenario_id,
        "seed": seed,
        "log_path": str(raw_log.resolve()),
        "log_path_relative": str(raw_log.resolve()),
        "candidate_table_path": str(raw_candidates.resolve()),
        "tracking_error_rms": 0.0,
    }

    friendly_row = _normalise_actual_outputs(paths, runner_root, scenario_id, seed, row)

    assert friendly_row["actual_rollout_file"] == "actual_rollout.csv"
    assert friendly_row["actual_metrics_file"] == "actual_metrics.csv"
    assert friendly_row["log_path"] == "actual_rollout.csv"
    assert friendly_row["log_path_relative"] == "actual_rollout.csv"
    assert friendly_row["candidate_table_path"] == "governor_candidates.csv"
    assert (paths.digital_dir / "governor_candidates.csv").exists()

    with paths.actual_metrics.open(newline="", encoding="utf-8") as handle:
        metrics_row = next(csv.DictReader(handle))
    assert metrics_row["log_path"] == "actual_rollout.csv"
    assert metrics_row["candidate_table_path"] == "governor_candidates.csv"

    metrics_text = paths.actual_metrics.read_text(encoding="utf-8")
    assert "_run" not in metrics_text
    assert str(tmp_path) not in metrics_text
    assert str(tmp_path).replace("\\", "/") not in metrics_text
    assert not re.search(r"[A-Za-z]:\\", metrics_text)
    assert "/tmp/" not in metrics_text

