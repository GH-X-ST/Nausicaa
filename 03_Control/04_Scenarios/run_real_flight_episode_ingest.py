from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from episode_logging import write_episode_log
from episode_schema import (
    EPISODE_BELIEF_UPDATE_COLUMNS,
    EPISODE_GOVERNOR_QUERY_COLUMNS,
    EPISODE_PRIMITIVE_STEP_COLUMNS,
    EPISODE_SUMMARY_COLUMNS,
)
from fixed_gate_contract import FIXED_LAUNCH_GATE, launch_gate_admission_status


REQUIRED_REAL_LOG_COLUMNS = (
    "time_s",
    "vicon_valid",
    "controller_ready",
    "x_w_m",
    "y_w_m",
    "z_w_m",
    "phi_rad",
    "theta_rad",
    "psi_rad",
    "u_m_s",
    "v_m_s",
    "w_m_s",
)


def find_episode_start_row(real_log: pd.DataFrame, *, consecutive_frames: int = 5, min_speed_m_s: float = 2.5) -> int | None:
    _require_columns(real_log, set(REQUIRED_REAL_LOG_COLUMNS), "real flight log")
    valid_run = 0
    for index, row in real_log.reset_index(drop=True).iterrows():
        state = _state_from_row(row)
        speed = float(np.linalg.norm(state[6:9]))
        ready = bool(row["vicon_valid"]) and bool(row["controller_ready"]) and speed >= float(min_speed_m_s)
        gate_ok = launch_gate_admission_status(state) in {"admitted_main_gate", "admitted_tolerance_shell"}
        valid_run = valid_run + 1 if ready and gate_ok else 0
        if valid_run >= int(consecutive_frames):
            return int(index - consecutive_frames + 1)
    return None


def ingest_real_flight_episode_log(
    *,
    real_log_csv: Path,
    output_root: Path,
    episode_id: str,
    policy_id: str = "real_flight_logged_policy",
    fan_branch: str = "single_fan_branch",
    overwrite: bool = False,
) -> dict[str, Path]:
    real_log = pd.read_csv(real_log_csv)
    start_index = find_episode_start_row(real_log)
    if start_index is None:
        summary = pd.DataFrame([_rejected_summary(episode_id, policy_id, fan_branch, "vicon_lost")], columns=EPISODE_SUMMARY_COLUMNS)
    else:
        state = _state_from_row(real_log.iloc[int(start_index)])
        energy_initial = _specific_energy_height_m(state)
        summary = pd.DataFrame(
            [
                {
                    **_rejected_summary(episode_id, policy_id, fan_branch, "operator_abort"),
                    "initial_state_vector": ";".join(f"{value:.12g}" for value in state),
                    "initial_state_admission_status": launch_gate_admission_status(state),
                    "energy_initial_m": energy_initial,
                    "energy_final_m": energy_initial,
                    "minimum_speed_m_s": float(np.linalg.norm(state[6:9])),
                    "termination_cause": "operator_abort",
                    "simulation_or_real": "real",
                    "claim_status": "real_flight_evidence",
                }
            ],
            columns=EPISODE_SUMMARY_COLUMNS,
        )
    outputs = write_episode_log(
        output_root,
        episode_summary=summary,
        primitive_steps=pd.DataFrame(columns=EPISODE_PRIMITIVE_STEP_COLUMNS),
        governor_queries=pd.DataFrame(columns=EPISODE_GOVERNOR_QUERY_COLUMNS),
        belief_updates=pd.DataFrame(columns=EPISODE_BELIEF_UPDATE_COLUMNS),
        manifest_extra={
            "campaign": "11_fixed_gate_repeated_launch",
            "pass_name": "real_flight_episode_ingest",
            "real_log_csv": str(real_log_csv),
            "real_flight_transfer_claim": False,
        },
        overwrite=overwrite,
    )
    return outputs.as_dict()


def _rejected_summary(episode_id: str, policy_id: str, fan_branch: str, termination_cause: str) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "policy_id": policy_id,
        "fan_branch": fan_branch,
        "W_layer": "real",
        "launch_gate_id": FIXED_LAUNCH_GATE.launch_gate_id,
        "initial_state_vector": "",
        "initial_state_admission_status": "invalid_state",
        "belief_id_before": "",
        "belief_id_after": "",
        "primitive_sequence": "",
        "candidate_query_count": 0,
        "governor_accept_count": 0,
        "governor_reject_count": 0,
        "lift_capture_success": False,
        "lift_capture_time_s": float("nan"),
        "lift_dwell_time_s": 0.0,
        "energy_initial_m": float("nan"),
        "energy_final_m": float("nan"),
        "energy_residual_m": float("nan"),
        "minimum_margin_m": 0.0,
        "minimum_speed_m_s": 0.0,
        "maximum_abs_phi_deg": 0.0,
        "maximum_abs_theta_deg": 0.0,
        "termination_cause": termination_cause,
        "simulation_or_real": "real",
        "matched_replay_id": "",
        "claim_status": "instrumentation_limited",
    }


def _state_from_row(row: pd.Series) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = float(row["x_w_m"])
    state[1] = float(row["y_w_m"])
    state[2] = float(row["z_w_m"])
    state[3] = float(row["phi_rad"])
    state[4] = float(row["theta_rad"])
    state[5] = float(row["psi_rad"])
    state[6] = float(row["u_m_s"])
    state[7] = float(row["v_m_s"])
    state[8] = float(row["w_m_s"])
    return state


def _specific_energy_height_m(state: np.ndarray) -> float:
    return float(state[2] + float(np.linalg.norm(state[6:9])) ** 2 / (2.0 * 9.81))


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-log-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--policy-id", default="real_flight_logged_policy")
    parser.add_argument("--fan-branch", default="single_fan_branch")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ingest_real_flight_episode_log(
        real_log_csv=args.real_log_csv,
        output_root=args.output_root,
        episode_id=args.episode_id,
        policy_id=args.policy_id,
        fan_branch=args.fan_branch,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
