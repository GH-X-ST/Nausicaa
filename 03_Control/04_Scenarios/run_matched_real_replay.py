from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MATCHED_REPLAY_COLUMNS = (
    "real_episode_id",
    "matched_replay_id",
    "real_initial_state",
    "sim_initial_state",
    "real_outcome",
    "sim_outcome",
    "energy_residual_error",
    "capture_success_match",
    "dwell_time_error",
    "termination_match",
    "transfer_label",
)


def build_matched_replay_table(real_episode_summary: pd.DataFrame, sim_episode_summary: pd.DataFrame) -> pd.DataFrame:
    if real_episode_summary.empty:
        return pd.DataFrame(columns=MATCHED_REPLAY_COLUMNS)
    rows: list[dict[str, object]] = []
    sim_by_match = {
        str(row.get("matched_replay_id", row.get("episode_id", ""))): row
        for _, row in sim_episode_summary.iterrows()
    }
    for _, real in real_episode_summary.iterrows():
        real_id = str(real["episode_id"])
        sim = sim_by_match.get(real_id)
        if sim is None:
            rows.append(_not_tested_row(real))
            continue
        rows.append(_comparison_row(real, sim))
    return pd.DataFrame(rows, columns=MATCHED_REPLAY_COLUMNS)


def write_matched_replay_table(
    *,
    real_episode_summary_csv: Path,
    sim_episode_summary_csv: Path,
    output_csv: Path,
) -> Path:
    real = pd.read_csv(real_episode_summary_csv)
    sim = pd.read_csv(sim_episode_summary_csv)
    table = build_matched_replay_table(real, sim)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return Path(output_csv)


def _not_tested_row(real: pd.Series) -> dict[str, object]:
    return {
        "real_episode_id": str(real["episode_id"]),
        "matched_replay_id": "",
        "real_initial_state": str(real.get("initial_state_vector", "")),
        "sim_initial_state": "",
        "real_outcome": str(real.get("termination_cause", "")),
        "sim_outcome": "",
        "energy_residual_error": float("nan"),
        "capture_success_match": False,
        "dwell_time_error": float("nan"),
        "termination_match": False,
        "transfer_label": "not_tested",
    }


def _comparison_row(real: pd.Series, sim: pd.Series) -> dict[str, object]:
    capture_match = bool(real.get("lift_capture_success", False)) == bool(sim.get("lift_capture_success", False))
    termination_match = str(real.get("termination_cause", "")) == str(sim.get("termination_cause", ""))
    energy_error = float(sim.get("energy_residual_m", 0.0)) - float(real.get("energy_residual_m", 0.0))
    dwell_error = float(sim.get("lift_dwell_time_s", 0.0)) - float(real.get("lift_dwell_time_s", 0.0))
    label = "supported" if capture_match and termination_match else ("partial" if capture_match else "negative")
    return {
        "real_episode_id": str(real["episode_id"]),
        "matched_replay_id": str(sim.get("episode_id", "")),
        "real_initial_state": str(real.get("initial_state_vector", "")),
        "sim_initial_state": str(sim.get("initial_state_vector", "")),
        "real_outcome": str(real.get("termination_cause", "")),
        "sim_outcome": str(sim.get("termination_cause", "")),
        "energy_residual_error": float(energy_error),
        "capture_success_match": bool(capture_match),
        "dwell_time_error": float(dwell_error),
        "termination_match": bool(termination_match),
        "transfer_label": label,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-episode-summary-csv", type=Path, required=True)
    parser.add_argument("--sim-episode-summary-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    write_matched_replay_table(
        real_episode_summary_csv=args.real_episode_summary_csv,
        sim_episode_summary_csv=args.sim_episode_summary_csv,
        output_csv=args.output_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
