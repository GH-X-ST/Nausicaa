from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("D_Transmitter_Test")
DEFAULT_OUTPUT_PREFIX = "post_transition_e2e"
DEFAULT_SURFACE_ORDER = ["Aileron_L", "Aileron_R", "Rudder", "Elevator"]
DEFAULT_STATE_LEVELS = 3
DEFAULT_SEED: int | None = 3 # DEFAULT_SEED: int | None = None


@dataclass(frozen=True)
class MatchingConfig:
    offset_search_min_s: float = 0.15
    offset_search_max_s: float = 0.35
    offset_bin_width_s: float = 5e-4
    offset_refine_half_window_s: float = 1.5e-3
    trainer_match_window_s: float = 0.03
    receiver_match_window_s: float = 0.05
    min_stable_pulses: int = 2
    pre_match_tolerance_s: float = 0.002


@dataclass(frozen=True)
class TrainerSurfaceCalibration:
    pulse_bias_us: float
    frame_delta: int
    pulse_tolerance_us: float


@dataclass(frozen=True)
class PulseSurfaceCalibration:
    pulse_bias_us: float
    pulse_tolerance_us: float


@dataclass(frozen=True)
class LocalAlignmentModel:
    base_offset_s: float
    residual_slope_per_frame: float
    residual_intercept_s: float


def predict_alignment_offset_s(
    alignment_model: LocalAlignmentModel,
    board_frame_index: float,
) -> float:
    if not np.isfinite(board_frame_index):
        return float(alignment_model.base_offset_s)
    return float(
        alignment_model.base_offset_s
        + alignment_model.residual_intercept_s
        + alignment_model.residual_slope_per_frame * float(board_frame_index)
    )


def latest_logger_folder(root: Path) -> Path:
    candidates = [path for path in root.rglob("*") if path.is_dir() and path.name.endswith("TransmitterLogger")]
    if not candidates:
        raise FileNotFoundError(f"No '*TransmitterLogger' folder found under: {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def logger_folder_from_seed(root: Path, seed: int) -> Path:
    logger_folder = root / f"Seed_{int(seed)}_Transmitter_TransmitterLogger"
    if not logger_folder.is_dir():
        raise FileNotFoundError(f"Seed logger folder not found: {logger_folder}")
    return logger_folder


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def kmeans_1d(values: np.ndarray, k: int) -> np.ndarray:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.linspace(0.0, 1.0, max(1, k))

    unique_values = np.unique(finite_values)
    if unique_values.size <= k:
        centers = unique_values.astype(float)
        if centers.size < k:
            centers = np.pad(centers, (0, k - centers.size), mode="edge")
        return np.sort(centers)

    quantiles = np.linspace(0.10, 0.90, k)
    centers = np.quantile(finite_values, quantiles)
    for _ in range(50):
        labels = classify_by_centers(finite_values, centers)
        next_centers = centers.copy()
        for label in range(k):
            if np.any(labels == label):
                next_centers[label] = float(np.mean(finite_values[labels == label]))
        next_centers = np.sort(next_centers)
        if np.allclose(next_centers, centers):
            break
        centers = next_centers
    return np.sort(centers)


def classify_by_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    center_array = np.asarray(centers, dtype=float)
    distances = np.abs(value_array[:, None] - center_array[None, :])
    return np.argmin(distances, axis=1)


def compute_state_level_count(host_surface: pd.DataFrame) -> int:
    unique_positions = np.unique(pd.to_numeric(host_surface["position_norm"], errors="coerce").dropna().to_numpy(dtype=float))
    return max(2, min(DEFAULT_STATE_LEVELS, unique_positions.size))


def build_host_state_table(host_dispatch: pd.DataFrame, surface_name: str) -> tuple[pd.DataFrame, Dict[float, int]]:
    surface_rows = host_dispatch.loc[host_dispatch["surface_name"] == surface_name, [
        "sample_sequence",
        "command_sequence",
        "scheduled_time_s",
        "command_dispatch_s",
        "position_norm",
    ]].drop_duplicates(subset=["sample_sequence"]).sort_values("sample_sequence", kind="stable").copy()

    positions = np.sort(surface_rows["position_norm"].dropna().unique())
    position_to_state = {float(position): state_index for state_index, position in enumerate(positions)}
    surface_rows["state"] = surface_rows["position_norm"].map(position_to_state)
    surface_rows = surface_rows.dropna(subset=["state"]).copy()
    surface_rows["state"] = surface_rows["state"].astype(int)
    return surface_rows.reset_index(drop=True), position_to_state


def build_host_transition_table(host_surface_states: pd.DataFrame) -> pd.DataFrame:
    if host_surface_states.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    previous_state = int(host_surface_states.iloc[0]["state"])
    for row_index in range(1, len(host_surface_states)):
        row = host_surface_states.iloc[row_index]
        current_state = int(row["state"])
        if current_state == previous_state:
            continue
        rows.append({
            "sample_sequence": int(row["sample_sequence"]),
            "command_sequence": int(row["command_sequence"]),
            "scheduled_time_s": float(row["scheduled_time_s"]),
            "command_dispatch_s": float(row["command_dispatch_s"]),
            "prev_state": previous_state,
            "new_state": current_state,
        })
        previous_state = current_state
    return pd.DataFrame(rows)


def build_commit_transition_table(
    board_commit_log: pd.DataFrame,
    host_surface_states: pd.DataFrame,
    ppm_column_name: str,
) -> pd.DataFrame:
    if board_commit_log.empty or host_surface_states.empty:
        return pd.DataFrame()

    commit_rows = board_commit_log.loc[:, ["sample_sequence", "commit_time_s", "receive_to_commit_us", "frame_index", ppm_column_name]].copy()
    commit_rows = commit_rows.rename(columns={ppm_column_name: "expected_pulse_us"})
    commit_rows = commit_rows.merge(
        host_surface_states.loc[:, ["sample_sequence", "state"]],
        on="sample_sequence",
        how="inner",
    )
    commit_rows = commit_rows.sort_values("sample_sequence", kind="stable").reset_index(drop=True)
    if commit_rows.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    previous_state = int(commit_rows.iloc[0]["state"])
    previous_pulse = float(commit_rows.iloc[0]["expected_pulse_us"])
    for row_index in range(1, len(commit_rows)):
        row = commit_rows.iloc[row_index]
        current_state = int(row["state"])
        current_pulse = float(row["expected_pulse_us"])
        if current_state == previous_state:
            previous_pulse = current_pulse
            continue
        rows.append({
            "sample_sequence": int(row["sample_sequence"]),
            "commit_time_s": float(row["commit_time_s"]),
            "receive_to_commit_us": float(row["receive_to_commit_us"]) if "receive_to_commit_us" in commit_rows.columns else np.nan,
            "board_frame_index": int(row["frame_index"]) if np.isfinite(row["frame_index"]) else np.nan,
            "prev_state": previous_state,
            "new_state": current_state,
            "previous_pulse_us": previous_pulse,
            "expected_pulse_us": current_pulse,
        })
        previous_state = current_state
        previous_pulse = current_pulse
    return pd.DataFrame(rows)


def build_board_rx_lookup(board_rx_log: pd.DataFrame) -> pd.DataFrame:
    if board_rx_log.empty:
        return pd.DataFrame(columns=["surface_name", "sample_sequence", "command_sequence", "board_rx_s", "received_position_norm"])
    lookup = board_rx_log.loc[:, ["surface_name", "sample_sequence", "command_sequence", "rx_time_s", "received_position_norm"]].copy()
    lookup = lookup.rename(columns={"rx_time_s": "board_rx_s"})
    lookup["surface_name"] = lookup["surface_name"].astype(str)
    return lookup.drop_duplicates(subset=["surface_name", "sample_sequence", "command_sequence"], keep="first").reset_index(drop=True)


def build_capture_run_table(capture_rows: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    if capture_rows.empty:
        return pd.DataFrame()

    capture_rows = capture_rows.sort_values("time_s", kind="stable").copy()
    pulse_values = pd.to_numeric(capture_rows["pulse_us"], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(pulse_values)
    capture_rows = capture_rows.loc[valid_mask].copy()
    pulse_values = pulse_values[valid_mask]
    if capture_rows.empty:
        return pd.DataFrame()

    capture_rows["state"] = classify_by_centers(pulse_values, centers)

    runs: list[dict] = []
    state_values = capture_rows["state"].to_numpy(dtype=int)
    time_values = pd.to_numeric(capture_rows["time_s"], errors="coerce").to_numpy(dtype=float)
    pulse_values = pd.to_numeric(capture_rows["pulse_us"], errors="coerce").to_numpy(dtype=float)

    run_start = 0
    for row_index in range(1, len(capture_rows) + 1):
        at_end = row_index >= len(capture_rows)
        if not at_end and state_values[row_index] == state_values[run_start]:
            continue
        run_slice = slice(run_start, row_index)
        runs.append({
            "state": int(state_values[run_start]),
            "start_index": run_start,
            "stop_index": row_index - 1,
            "length": row_index - run_start,
            "start_time_s": float(time_values[run_start]),
            "stop_time_s": float(time_values[row_index - 1]),
            "pulse_median_us": float(np.median(pulse_values[run_slice])),
        })
        run_start = row_index

    return pd.DataFrame(runs)


def build_stable_transition_table(
    capture_rows: pd.DataFrame,
    centers: np.ndarray,
    min_stable_pulses: int,
) -> pd.DataFrame:
    run_table = build_capture_run_table(capture_rows, centers)
    if run_table.empty:
        return pd.DataFrame()

    stable_runs = run_table.loc[run_table["length"] >= min_stable_pulses].reset_index(drop=True)
    if len(stable_runs) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for row_index in range(1, len(stable_runs)):
        previous_run = stable_runs.iloc[row_index - 1]
        current_run = stable_runs.iloc[row_index]
        if int(previous_run["state"]) == int(current_run["state"]):
            continue
        rows.append({
            "prev_state": int(previous_run["state"]),
            "new_state": int(current_run["state"]),
            "time_s": float(current_run["start_time_s"]),
            "previous_pulse_us": float(previous_run["pulse_median_us"]),
            "new_pulse_us": float(current_run["pulse_median_us"]),
            "new_state_run_length": int(current_run["length"]),
        })
    return pd.DataFrame(rows)


def estimate_trainer_surface_calibration(
    board_commit_log: pd.DataFrame,
    trainer_surface_rows: pd.DataFrame,
    ppm_column_name: str,
    alignment_model: LocalAlignmentModel,
) -> TrainerSurfaceCalibration:
    if board_commit_log.empty or trainer_surface_rows.empty:
        return TrainerSurfaceCalibration(pulse_bias_us=0.0, frame_delta=0, pulse_tolerance_us=12.0)

    trainer_rows = trainer_surface_rows.sort_values("time_s", kind="stable").reset_index(drop=True)
    trainer_time = pd.to_numeric(trainer_rows["time_s"], errors="coerce").to_numpy(dtype=float)
    trainer_pulse = pd.to_numeric(trainer_rows["pulse_us"], errors="coerce").to_numpy(dtype=float)
    trainer_frame_index = pd.to_numeric(trainer_rows.get("frame_index"), errors="coerce").to_numpy(dtype=float)
    if trainer_time.size == 0:
        return TrainerSurfaceCalibration(pulse_bias_us=0.0, frame_delta=0, pulse_tolerance_us=12.0)

    pulse_differences: list[float] = []
    frame_differences: list[float] = []
    for _, commit_row in board_commit_log.iterrows():
        board_frame_index = float(commit_row["frame_index"]) if "frame_index" in commit_row.index else np.nan
        commit_time_s = float(commit_row["commit_time_s"]) + predict_alignment_offset_s(alignment_model, board_frame_index)
        search_index = int(np.searchsorted(trainer_time, commit_time_s, side="left"))
        if search_index >= trainer_time.size:
            continue
        expected_pulse_us = float(commit_row[ppm_column_name])
        observed_pulse_us = float(trainer_pulse[search_index])
        if not np.isfinite(expected_pulse_us) or not np.isfinite(observed_pulse_us):
            continue
        pulse_differences.append(observed_pulse_us - expected_pulse_us)
        observed_frame_index = float(trainer_frame_index[search_index]) if trainer_frame_index.size > search_index else np.nan
        if np.isfinite(board_frame_index) and np.isfinite(observed_frame_index):
            frame_differences.append(observed_frame_index - board_frame_index)

    if pulse_differences:
        pulse_difference_array = np.asarray(pulse_differences, dtype=float)
        pulse_bias_us = float(np.median(pulse_difference_array))
        pulse_residual_us = np.abs(pulse_difference_array - pulse_bias_us)
        pulse_tolerance_us = float(max(12.0, 2.5 * np.quantile(pulse_residual_us, 0.95)))
    else:
        pulse_bias_us = 0.0
        pulse_tolerance_us = 12.0

    if frame_differences:
        frame_delta = int(round(float(np.median(np.asarray(frame_differences, dtype=float)))))
    else:
        frame_delta = 0

    return TrainerSurfaceCalibration(
        pulse_bias_us=pulse_bias_us,
        frame_delta=frame_delta,
        pulse_tolerance_us=pulse_tolerance_us,
    )


def estimate_receiver_surface_calibration(
    board_commit_log: pd.DataFrame,
    receiver_surface_rows: pd.DataFrame,
    ppm_column_name: str,
    alignment_model: LocalAlignmentModel,
) -> PulseSurfaceCalibration:
    if board_commit_log.empty or receiver_surface_rows.empty:
        return PulseSurfaceCalibration(pulse_bias_us=0.0, pulse_tolerance_us=12.0)

    receiver_rows = receiver_surface_rows.sort_values("time_s", kind="stable").reset_index(drop=True)
    receiver_time = pd.to_numeric(receiver_rows["time_s"], errors="coerce").to_numpy(dtype=float)
    receiver_pulse = pd.to_numeric(receiver_rows["pulse_us"], errors="coerce").to_numpy(dtype=float)
    if receiver_time.size == 0:
        return PulseSurfaceCalibration(pulse_bias_us=0.0, pulse_tolerance_us=12.0)

    pulse_differences: list[float] = []
    for _, commit_row in board_commit_log.iterrows():
        expected_pulse_us = float(commit_row[ppm_column_name])
        if not np.isfinite(expected_pulse_us):
            continue
        board_frame_index = float(commit_row["frame_index"]) if "frame_index" in commit_row.index else np.nan
        search_time_s = float(commit_row["commit_time_s"]) + predict_alignment_offset_s(alignment_model, board_frame_index) + 0.01
        search_index = int(np.searchsorted(receiver_time, search_time_s, side="left"))
        if search_index >= receiver_time.size:
            continue
        observed_pulse_us = float(receiver_pulse[search_index])
        if not np.isfinite(observed_pulse_us):
            continue
        pulse_differences.append(observed_pulse_us - expected_pulse_us)

    if pulse_differences:
        pulse_difference_array = np.asarray(pulse_differences, dtype=float)
        pulse_bias_us = float(np.median(pulse_difference_array))
        pulse_residual_us = np.abs(pulse_difference_array - pulse_bias_us)
        pulse_tolerance_us = float(max(12.0, 2.5 * np.quantile(pulse_residual_us, 0.95)))
    else:
        pulse_bias_us = 0.0
        pulse_tolerance_us = 12.0

    return PulseSurfaceCalibration(
        pulse_bias_us=pulse_bias_us,
        pulse_tolerance_us=pulse_tolerance_us,
    )


def match_trainer_transition_from_capture(
    trainer_surface_rows: pd.DataFrame,
    trainer_calibration: TrainerSurfaceCalibration,
    commit_capture_time_s: float,
    previous_pulse_us: float,
    expected_pulse_us: float,
    board_frame_index: float,
    config: MatchingConfig,
) -> pd.Series | None:
    if trainer_surface_rows.empty or not np.isfinite(expected_pulse_us):
        return None

    trainer_rows = trainer_surface_rows
    predicted_frame_index = (
        int(round(float(board_frame_index) + trainer_calibration.frame_delta))
        if np.isfinite(board_frame_index)
        else None
    )

    def row_matches(row: pd.Series) -> bool:
        observed_pulse_us = float(row["pulse_us"])
        if not np.isfinite(observed_pulse_us):
            return False
        corrected_pulse_us = observed_pulse_us - trainer_calibration.pulse_bias_us
        return abs(corrected_pulse_us - float(expected_pulse_us)) <= trainer_calibration.pulse_tolerance_us

    candidate_rows = trainer_rows.loc[
        (pd.to_numeric(trainer_rows["time_s"], errors="coerce") >= commit_capture_time_s - config.pre_match_tolerance_s)
        & (pd.to_numeric(trainer_rows["time_s"], errors="coerce") <= commit_capture_time_s + config.trainer_match_window_s)
    ].copy()

    scored_candidates: list[tuple[float, float, int]] = []
    for row_index, trainer_row in candidate_rows.iterrows():
        if not row_matches(trainer_row):
            continue
        prev_matches_old = True
        if np.isfinite(previous_pulse_us) and row_index > 0:
            previous_observed_pulse_us = float(trainer_rows.iloc[row_index - 1]["pulse_us"])
            if np.isfinite(previous_observed_pulse_us):
                corrected_previous_pulse_us = previous_observed_pulse_us - trainer_calibration.pulse_bias_us
                prev_matches_old = abs(corrected_previous_pulse_us - float(previous_pulse_us)) <= trainer_calibration.pulse_tolerance_us

        frame_penalty = 0.0
        if predicted_frame_index is not None and "frame_index" in trainer_row.index:
            observed_frame_index = float(trainer_row["frame_index"])
            if np.isfinite(observed_frame_index):
                frame_penalty = abs(observed_frame_index - predicted_frame_index)

        score = (
            0.0 if prev_matches_old else 1.0,
            frame_penalty,
            float(trainer_row["time_s"]),
        )
        scored_candidates.append((score, row_index, int(prev_matches_old)))

    if scored_candidates:
        _, best_index, _ = min(scored_candidates, key=lambda item: item[0])
        return trainer_rows.iloc[best_index]

    return None


def match_output_transition_from_capture(
    capture_surface_rows: pd.DataFrame,
    pulse_calibration: PulseSurfaceCalibration,
    commit_capture_time_s: float,
    previous_pulse_us: float,
    expected_pulse_us: float,
    match_window_s: float,
    pre_match_tolerance_s: float,
) -> pd.Series | None:
    if capture_surface_rows.empty or not np.isfinite(expected_pulse_us):
        return None

    capture_rows = capture_surface_rows.reset_index(drop=True)
    time_values = pd.to_numeric(capture_rows["time_s"], errors="coerce")
    candidate_rows = capture_rows.loc[
        (time_values >= commit_capture_time_s - pre_match_tolerance_s)
        & (time_values <= commit_capture_time_s + match_window_s)
    ].copy()

    for row_index, capture_row in candidate_rows.iterrows():
        observed_pulse_us = float(capture_row["pulse_us"])
        if not np.isfinite(observed_pulse_us):
            continue
        corrected_pulse_us = observed_pulse_us - pulse_calibration.pulse_bias_us
        if abs(corrected_pulse_us - float(expected_pulse_us)) > pulse_calibration.pulse_tolerance_us:
            continue

        prev_matches_old = True
        if np.isfinite(previous_pulse_us) and row_index > 0:
            previous_observed_pulse_us = float(capture_rows.iloc[row_index - 1]["pulse_us"])
            if np.isfinite(previous_observed_pulse_us):
                corrected_previous_pulse_us = previous_observed_pulse_us - pulse_calibration.pulse_bias_us
                prev_matches_old = abs(corrected_previous_pulse_us - float(previous_pulse_us)) <= pulse_calibration.pulse_tolerance_us
        if prev_matches_old:
            return capture_row

    return None


def match_trainer_transition_from_state_table(
    trainer_transition_table: pd.DataFrame,
    commit_capture_time_s: float,
    prev_state: int,
    new_state: int,
    config: MatchingConfig,
) -> pd.Series | None:
    if trainer_transition_table.empty:
        return None

    trainer_rows = trainer_transition_table.reset_index(drop=True)
    candidate_rows = trainer_rows.loc[
        (pd.to_numeric(trainer_rows["time_s"], errors="coerce") >= commit_capture_time_s - 1e-6)
        & (pd.to_numeric(trainer_rows["time_s"], errors="coerce") <= commit_capture_time_s + config.trainer_match_window_s)
    ].copy()
    for _, trainer_row in candidate_rows.iterrows():
        if int(trainer_row["prev_state"]) == prev_state and int(trainer_row["new_state"]) == new_state:
            return trainer_row
    return None


def compute_candidate_alignment_differences(
    commit_transition_table: pd.DataFrame,
    trainer_transition_table: pd.DataFrame,
    search_min_s: float,
    search_max_s: float,
) -> np.ndarray:
    if commit_transition_table.empty or trainer_transition_table.empty:
        return np.array([], dtype=float)

    trainer_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for pair, rows in trainer_transition_table.groupby(["prev_state", "new_state"], sort=False):
        trainer_by_pair[(int(pair[0]), int(pair[1]))] = pd.to_numeric(rows["time_s"], errors="coerce").to_numpy(dtype=float)

    differences: list[np.ndarray] = []
    for pair, rows in commit_transition_table.groupby(["prev_state", "new_state"], sort=False):
        trainer_times = trainer_by_pair.get((int(pair[0]), int(pair[1])))
        if trainer_times is None or trainer_times.size == 0:
            continue
        commit_times = pd.to_numeric(rows["commit_time_s"], errors="coerce").to_numpy(dtype=float)
        pairwise_diff = trainer_times[None, :] - commit_times[:, None]
        valid_mask = np.isfinite(pairwise_diff) & (pairwise_diff >= search_min_s) & (pairwise_diff <= search_max_s)
        if np.any(valid_mask):
            differences.append(pairwise_diff[valid_mask])

    if not differences:
        return np.array([], dtype=float)
    return np.concatenate(differences)


def compute_reference_alignment_differences(
    board_commit_log: pd.DataFrame,
    reference_capture: pd.DataFrame,
    search_min_s: float,
    search_max_s: float,
) -> np.ndarray:
    if board_commit_log.empty or reference_capture.empty:
        return np.array([], dtype=float)

    commit_times = pd.to_numeric(board_commit_log["commit_time_s"], errors="coerce").to_numpy(dtype=float)
    reference_times = pd.to_numeric(reference_capture["time_s"], errors="coerce").to_numpy(dtype=float)
    commit_times = commit_times[np.isfinite(commit_times)]
    reference_times = reference_times[np.isfinite(reference_times)]
    if commit_times.size == 0 or reference_times.size == 0:
        return np.array([], dtype=float)

    differences: list[np.ndarray] = []
    for commit_time_s in commit_times:
        lower_index = int(np.searchsorted(reference_times, commit_time_s + search_min_s, side="left"))
        upper_index = int(np.searchsorted(reference_times, commit_time_s + search_max_s, side="right"))
        if upper_index <= lower_index:
            continue
        differences.append(reference_times[lower_index:upper_index] - commit_time_s)

    if not differences:
        return np.array([], dtype=float)
    return np.concatenate(differences)


def estimate_alignment_offset(candidate_differences: np.ndarray, config: MatchingConfig) -> tuple[float, pd.DataFrame]:
    if candidate_differences.size == 0:
        raise RuntimeError("Could not estimate an alignment offset because no valid trainer-transition candidates were found.")

    bin_edges = np.arange(
        config.offset_search_min_s,
        config.offset_search_max_s + config.offset_bin_width_s,
        config.offset_bin_width_s,
    )
    hist, bin_edges = np.histogram(candidate_differences, bins=bin_edges)
    best_bin_index = int(np.argmax(hist))
    best_bin_start = float(bin_edges[best_bin_index])
    best_bin_stop = float(bin_edges[best_bin_index + 1])

    primary_cluster_mask = (candidate_differences >= best_bin_start) & (candidate_differences < best_bin_stop)
    primary_cluster = candidate_differences[primary_cluster_mask]
    cluster_center = float(np.median(primary_cluster))

    refined_mask = np.abs(candidate_differences - cluster_center) <= config.offset_refine_half_window_s
    refined_cluster = candidate_differences[refined_mask]
    offset_seconds = float(np.median(refined_cluster))

    diagnostics = pd.DataFrame([
        {
            "metric": "candidate_count",
            "value": int(candidate_differences.size),
        },
        {
            "metric": "mode_bin_start_s",
            "value": best_bin_start,
        },
        {
            "metric": "mode_bin_stop_s",
            "value": best_bin_stop,
        },
        {
            "metric": "mode_bin_count",
            "value": int(hist[best_bin_index]),
        },
        {
            "metric": "refined_cluster_count",
            "value": int(refined_cluster.size),
        },
        {
            "metric": "alignment_offset_s",
            "value": offset_seconds,
        },
        {
            "metric": "alignment_offset_ms",
            "value": offset_seconds * 1e3,
        },
    ])
    return offset_seconds, diagnostics


def candidate_offsets_from_differences(candidate_differences: np.ndarray, config: MatchingConfig) -> list[float]:
    if candidate_differences.size == 0:
        return []

    bin_edges = np.arange(
        config.offset_search_min_s,
        config.offset_search_max_s + config.offset_bin_width_s,
        config.offset_bin_width_s,
    )
    hist, bin_edges = np.histogram(candidate_differences, bins=bin_edges)
    top_indices = np.argsort(hist)[::-1]
    offsets: list[float] = []
    for bin_index in top_indices:
        if hist[bin_index] <= 0:
            continue
        bin_start = float(bin_edges[bin_index])
        bin_stop = float(bin_edges[bin_index + 1])
        cluster = candidate_differences[(candidate_differences >= bin_start) & (candidate_differences < bin_stop)]
        if cluster.size == 0:
            continue
        candidate_offset = float(np.median(cluster))
        if any(abs(candidate_offset - prior_offset) < config.offset_bin_width_s for prior_offset in offsets):
            continue
        offsets.append(candidate_offset)
        if len(offsets) >= 8:
            break
    return offsets


def score_alignment_offset(
    offset_s: float,
    host_transition_tables: dict[str, pd.DataFrame],
    commit_transition_tables: dict[str, pd.DataFrame],
    trainer_transition_tables: dict[str, pd.DataFrame],
    config: MatchingConfig,
) -> tuple[tuple[float, ...], dict[str, float]]:
    match_count = 0
    negative_count = 0
    latencies_ms: list[float] = []

    for surface_name, host_transition_table in host_transition_tables.items():
        commit_transition_table = commit_transition_tables.get(surface_name, pd.DataFrame())
        trainer_transition_table = trainer_transition_tables.get(surface_name, pd.DataFrame())
        if host_transition_table.empty or commit_transition_table.empty or trainer_transition_table.empty:
            continue

        trainer_rows = trainer_transition_table.reset_index(drop=True)
        trainer_cursor = 0
        commit_rows = commit_transition_table.set_index("sample_sequence", drop=False)
        for _, host_row in host_transition_table.iterrows():
            sample_sequence = int(host_row["sample_sequence"])
            if sample_sequence not in commit_rows.index:
                continue
            commit_row = commit_rows.loc[sample_sequence]
            if isinstance(commit_row, pd.DataFrame):
                commit_row = commit_row.iloc[0]
            commit_capture_time_s = float(commit_row["commit_time_s"]) + offset_s
            prev_state = int(commit_row["prev_state"])
            new_state = int(commit_row["new_state"])

            while trainer_cursor < len(trainer_rows) and float(trainer_rows.iloc[trainer_cursor]["time_s"]) < commit_capture_time_s - 1e-6:
                trainer_cursor += 1

            trainer_search_index = trainer_cursor
            matched_latency_ms = np.nan
            while trainer_search_index < len(trainer_rows):
                trainer_row = trainer_rows.iloc[trainer_search_index]
                trainer_time_s = float(trainer_row["time_s"])
                if trainer_time_s > commit_capture_time_s + config.trainer_match_window_s:
                    break
                if int(trainer_row["prev_state"]) == prev_state and int(trainer_row["new_state"]) == new_state:
                    matched_latency_ms = (trainer_time_s - commit_capture_time_s) * 1e3
                    trainer_cursor = trainer_search_index + 1
                    break
                trainer_search_index += 1

            if np.isfinite(matched_latency_ms):
                match_count += 1
                latencies_ms.append(float(matched_latency_ms))
                if matched_latency_ms < -0.25:
                    negative_count += 1

    if latencies_ms:
        latency_array = np.asarray(latencies_ms, dtype=float)
        median_latency_ms = float(np.median(latency_array))
        p95_latency_ms = float(np.quantile(latency_array, 0.95))
        std_latency_ms = float(np.std(latency_array))
    else:
        median_latency_ms = float("inf")
        p95_latency_ms = float("inf")
        std_latency_ms = float("inf")

    plausibility_penalty = 0
    if not np.isfinite(median_latency_ms) or median_latency_ms < 0.0 or median_latency_ms > 8.0:
        plausibility_penalty += 1
    if not np.isfinite(p95_latency_ms) or p95_latency_ms > 15.0:
        plausibility_penalty += 1

    score = (
        float(plausibility_penalty),
        float(negative_count),
        abs(median_latency_ms - 3.0),
        p95_latency_ms,
        -float(match_count),
        median_latency_ms,
        std_latency_ms,
    )
    diagnostics = {
        "offset_s": offset_s,
        "trainer_match_count": match_count,
        "trainer_negative_count": negative_count,
        "trainer_plausibility_penalty": plausibility_penalty,
        "trainer_commit_to_trainer_median_ms": median_latency_ms,
        "trainer_commit_to_trainer_p95_ms": p95_latency_ms,
        "trainer_commit_to_trainer_std_ms": std_latency_ms,
    }
    return score, diagnostics


def estimate_alignment_offset_with_trainer_scoring(
    reference_candidate_differences: np.ndarray,
    transition_candidate_differences: np.ndarray,
    host_transition_tables: dict[str, pd.DataFrame],
    commit_transition_tables: dict[str, pd.DataFrame],
    trainer_transition_tables: dict[str, pd.DataFrame],
    config: MatchingConfig,
) -> tuple[float, pd.DataFrame]:
    candidate_offsets = candidate_offsets_from_differences(reference_candidate_differences, config)
    for transition_offset in candidate_offsets_from_differences(transition_candidate_differences, config):
        if any(abs(transition_offset - candidate_offset) < config.offset_bin_width_s for candidate_offset in candidate_offsets):
            continue
        candidate_offsets.append(transition_offset)

    if not candidate_offsets:
        combined = np.concatenate([
            reference_candidate_differences if reference_candidate_differences.size > 0 else np.array([], dtype=float),
            transition_candidate_differences if transition_candidate_differences.size > 0 else np.array([], dtype=float),
        ])
        if combined.size == 0:
            raise RuntimeError("Could not estimate an alignment offset because no valid candidates were found.")
        fallback_offset, fallback_diagnostics = estimate_alignment_offset(combined, config)
        fallback_diagnostics["selection"] = "fallback_histogram"
        return fallback_offset, fallback_diagnostics

    best_score = None
    best_diagnostics: dict[str, float] | None = None
    for candidate_offset in candidate_offsets:
        score, diagnostics = score_alignment_offset(
            candidate_offset,
            host_transition_tables,
            commit_transition_tables,
            trainer_transition_tables,
            config,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_diagnostics = diagnostics

    assert best_diagnostics is not None
    alignment_rows = [
        {"metric": "selection", "value": "trainer_scored_mode"},
        {"metric": "alignment_offset_s", "value": float(best_diagnostics["offset_s"])},
        {"metric": "alignment_offset_ms", "value": float(best_diagnostics["offset_s"]) * 1e3},
        {"metric": "trainer_match_count", "value": int(best_diagnostics["trainer_match_count"])},
        {"metric": "trainer_negative_count", "value": int(best_diagnostics["trainer_negative_count"])},
        {"metric": "trainer_commit_to_trainer_median_ms", "value": best_diagnostics["trainer_commit_to_trainer_median_ms"]},
        {"metric": "trainer_commit_to_trainer_p95_ms", "value": best_diagnostics["trainer_commit_to_trainer_p95_ms"]},
        {"metric": "trainer_commit_to_trainer_std_ms", "value": best_diagnostics["trainer_commit_to_trainer_std_ms"]},
        {"metric": "reference_candidate_count", "value": int(reference_candidate_differences.size)},
        {"metric": "transition_candidate_count", "value": int(transition_candidate_differences.size)},
    ]
    for candidate_index, candidate_offset in enumerate(candidate_offsets, start=1):
        alignment_rows.append({"metric": f"candidate_offset_{candidate_index}_ms", "value": candidate_offset * 1e3})
    return float(best_diagnostics["offset_s"]), pd.DataFrame(alignment_rows)


def estimate_local_alignment_model(
    board_commit_log: pd.DataFrame,
    reference_capture: pd.DataFrame,
    base_offset_s: float,
) -> tuple[LocalAlignmentModel, pd.DataFrame]:
    if board_commit_log.empty or reference_capture.empty:
        model = LocalAlignmentModel(base_offset_s=float(base_offset_s), residual_slope_per_frame=0.0, residual_intercept_s=0.0)
        diagnostics = pd.DataFrame([
            {"metric": "local_alignment_sample_count", "value": 0},
            {"metric": "local_alignment_residual_slope_us_per_frame", "value": 0.0},
            {"metric": "local_alignment_residual_intercept_ms", "value": 0.0},
        ])
        return model, diagnostics

    commit_rows = board_commit_log.copy()
    to_numeric(commit_rows, ["commit_time_s", "frame_index"])
    commit_rows = commit_rows.dropna(subset=["commit_time_s", "frame_index"]).sort_values("frame_index", kind="stable")
    reference_times = pd.to_numeric(reference_capture["time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if commit_rows.empty or reference_times.size == 0:
        model = LocalAlignmentModel(base_offset_s=float(base_offset_s), residual_slope_per_frame=0.0, residual_intercept_s=0.0)
        diagnostics = pd.DataFrame([
            {"metric": "local_alignment_sample_count", "value": 0},
            {"metric": "local_alignment_residual_slope_us_per_frame", "value": 0.0},
            {"metric": "local_alignment_residual_intercept_ms", "value": 0.0},
        ])
        return model, diagnostics

    frame_index_values: list[float] = []
    residual_values: list[float] = []
    for _, commit_row in commit_rows.iterrows():
        target_time_s = float(commit_row["commit_time_s"]) + float(base_offset_s)
        nearest_index = int(np.searchsorted(reference_times, target_time_s, side="left"))
        candidate_indices = [candidate for candidate in (nearest_index - 1, nearest_index, nearest_index + 1) if 0 <= candidate < reference_times.size]
        if not candidate_indices:
            continue
        nearest_time_s = min(candidate_indices, key=lambda index: abs(reference_times[index] - target_time_s))
        matched_reference_time_s = float(reference_times[nearest_time_s])
        residual_s = matched_reference_time_s - target_time_s
        if abs(residual_s) > 0.01:
            continue
        frame_index_values.append(float(commit_row["frame_index"]))
        residual_values.append(float(residual_s))

    if len(frame_index_values) < 3:
        model = LocalAlignmentModel(base_offset_s=float(base_offset_s), residual_slope_per_frame=0.0, residual_intercept_s=0.0)
        diagnostics = pd.DataFrame([
            {"metric": "local_alignment_sample_count", "value": int(len(frame_index_values))},
            {"metric": "local_alignment_residual_slope_us_per_frame", "value": 0.0},
            {"metric": "local_alignment_residual_intercept_ms", "value": 0.0},
        ])
        return model, diagnostics

    frame_index_array = np.asarray(frame_index_values, dtype=float)
    residual_array = np.asarray(residual_values, dtype=float)
    fit_mask = np.isfinite(frame_index_array) & np.isfinite(residual_array)
    frame_index_array = frame_index_array[fit_mask]
    residual_array = residual_array[fit_mask]

    if frame_index_array.size >= 8:
        initial_slope, initial_intercept = np.polyfit(frame_index_array, residual_array, deg=1)
        residual_error = residual_array - (initial_slope * frame_index_array + initial_intercept)
        mad = float(np.median(np.abs(residual_error - np.median(residual_error))))
        if np.isfinite(mad) and mad > 0.0:
            robust_mask = np.abs(residual_error) <= max(3.5 * mad, 0.0015)
            if np.count_nonzero(robust_mask) >= 3:
                frame_index_array = frame_index_array[robust_mask]
                residual_array = residual_array[robust_mask]

    residual_slope_per_frame, residual_intercept_s = np.polyfit(frame_index_array, residual_array, deg=1)
    model = LocalAlignmentModel(
        base_offset_s=float(base_offset_s),
        residual_slope_per_frame=float(residual_slope_per_frame),
        residual_intercept_s=float(residual_intercept_s),
    )

    fitted_residual = residual_slope_per_frame * frame_index_array + residual_intercept_s
    residual_error = residual_array - fitted_residual
    diagnostics = pd.DataFrame([
        {"metric": "local_alignment_sample_count", "value": int(frame_index_array.size)},
        {"metric": "local_alignment_residual_slope_us_per_frame", "value": float(residual_slope_per_frame) * 1e6},
        {"metric": "local_alignment_residual_intercept_ms", "value": float(residual_intercept_s) * 1e3},
        {"metric": "local_alignment_residual_p95_ms", "value": float(np.quantile(np.abs(residual_error), 0.95)) * 1e3},
        {"metric": "local_alignment_residual_max_ms", "value": float(np.max(np.abs(residual_error))) * 1e3},
        {"metric": "local_alignment_residual_start_ms", "value": float(np.polyval([residual_slope_per_frame, residual_intercept_s], frame_index_array.min())) * 1e3},
        {"metric": "local_alignment_residual_end_ms", "value": float(np.polyval([residual_slope_per_frame, residual_intercept_s], frame_index_array.max())) * 1e3},
    ])
    return model, diagnostics


def match_surface_transitions(
    surface_name: str,
    host_transition_table: pd.DataFrame,
    commit_transition_table: pd.DataFrame,
    trainer_transition_table: pd.DataFrame,
    trainer_surface_rows: pd.DataFrame,
    trainer_calibration: TrainerSurfaceCalibration,
    receiver_surface_rows: pd.DataFrame,
    receiver_calibration: PulseSurfaceCalibration,
    receiver_transition_table: pd.DataFrame,
    board_rx_lookup: pd.DataFrame,
    alignment_model: LocalAlignmentModel,
    config: MatchingConfig,
) -> pd.DataFrame:
    if host_transition_table.empty or commit_transition_table.empty:
        return pd.DataFrame()

    host_transition_table = host_transition_table.copy()
    commit_transition_table = commit_transition_table.copy()

    rows: list[dict] = []
    receiver_cursor = 0
    receiver_rows = receiver_transition_table.reset_index(drop=True)

    for _, host_row in host_transition_table.iterrows():
        sample_sequence = int(host_row["sample_sequence"])
        commit_rows = commit_transition_table.loc[commit_transition_table["sample_sequence"] == sample_sequence]
        if commit_rows.empty:
            continue
        commit_row = commit_rows.iloc[0]
        board_rx_rows = board_rx_lookup.loc[
            (board_rx_lookup["surface_name"] == surface_name)
            & (board_rx_lookup["sample_sequence"] == sample_sequence)
            & (board_rx_lookup["command_sequence"] == int(host_row["command_sequence"]))
        ]
        if board_rx_rows.empty:
            board_rx_s = np.nan
            received_position_norm = np.nan
        else:
            board_rx_row = board_rx_rows.iloc[0]
            board_rx_s = float(board_rx_row["board_rx_s"])
            received_position_norm = float(board_rx_row["received_position_norm"])

        board_frame_index = float(commit_row["board_frame_index"]) if "board_frame_index" in commit_row.index else np.nan
        event_alignment_offset_s = predict_alignment_offset_s(alignment_model, board_frame_index)
        scheduled_capture_time_s = float(host_row["scheduled_time_s"]) + event_alignment_offset_s
        dispatch_capture_time_s = float(host_row["command_dispatch_s"]) + event_alignment_offset_s
        commit_capture_time_s = float(commit_row["commit_time_s"]) + event_alignment_offset_s
        prev_state = int(commit_row["prev_state"])
        new_state = int(commit_row["new_state"])

        matched_trainer = match_trainer_transition_from_capture(
            trainer_surface_rows,
            trainer_calibration,
            commit_capture_time_s,
            float(commit_row["previous_pulse_us"]) if "previous_pulse_us" in commit_row.index else np.nan,
            float(commit_row["expected_pulse_us"]),
            board_frame_index,
            config,
        )
        if matched_trainer is None:
            matched_trainer = match_trainer_transition_from_state_table(
                trainer_transition_table,
                commit_capture_time_s,
                prev_state,
                new_state,
                config,
            )

        matched_receiver = match_output_transition_from_capture(
            receiver_surface_rows,
            receiver_calibration,
            commit_capture_time_s,
            float(commit_row["previous_pulse_us"]) if "previous_pulse_us" in commit_row.index else np.nan,
            float(commit_row["expected_pulse_us"]),
            config.receiver_match_window_s,
            config.pre_match_tolerance_s,
        )
        if matched_receiver is None:
            while receiver_cursor < len(receiver_rows) and float(receiver_rows.iloc[receiver_cursor]["time_s"]) < commit_capture_time_s - 1e-6:
                receiver_cursor += 1
            receiver_search_index = receiver_cursor
            while receiver_search_index < len(receiver_rows):
                receiver_row = receiver_rows.iloc[receiver_search_index]
                receiver_time_s = float(receiver_row["time_s"])
                if receiver_time_s > commit_capture_time_s + config.receiver_match_window_s:
                    break
                if int(receiver_row["prev_state"]) == prev_state and int(receiver_row["new_state"]) == new_state:
                    matched_receiver = receiver_row
                    receiver_cursor = receiver_search_index + 1
                    break
                receiver_search_index += 1

        trainer_time_s = float(matched_trainer["time_s"]) if matched_trainer is not None else np.nan
        receiver_time_s = float(matched_receiver["time_s"]) if matched_receiver is not None else np.nan
        if np.isfinite(trainer_time_s) and trainer_time_s < commit_capture_time_s and trainer_time_s >= commit_capture_time_s - config.pre_match_tolerance_s:
            trainer_time_s = commit_capture_time_s
        if np.isfinite(receiver_time_s) and receiver_time_s < commit_capture_time_s and receiver_time_s >= commit_capture_time_s - config.pre_match_tolerance_s:
            receiver_time_s = commit_capture_time_s

        rows.append({
            "surface_name": surface_name,
            "sample_sequence": sample_sequence,
            "command_sequence": int(host_row["command_sequence"]),
            "prev_state": prev_state,
            "new_state": new_state,
            "scheduled_time_s": float(host_row["scheduled_time_s"]),
            "command_dispatch_s": float(host_row["command_dispatch_s"]),
            "board_rx_s": board_rx_s,
            "commit_time_s": float(commit_row["commit_time_s"]),
            "receive_to_commit_us": float(commit_row["receive_to_commit_us"]) if np.isfinite(commit_row["receive_to_commit_us"]) else np.nan,
            "received_position_norm": received_position_norm,
            "scheduled_capture_time_s": scheduled_capture_time_s,
            "dispatch_capture_time_s": dispatch_capture_time_s,
            "commit_capture_time_s": commit_capture_time_s,
            "board_frame_index": board_frame_index,
            "alignment_offset_s": event_alignment_offset_s,
            "expected_pulse_us": float(commit_row["expected_pulse_us"]),
            "trainer_transition_s": trainer_time_s,
            "receiver_transition_s": receiver_time_s,
            "trainer_transition_found": int(np.isfinite(trainer_time_s)),
            "receiver_transition_found": int(np.isfinite(receiver_time_s)),
        })

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    events["dispatch_to_rx_latency_s"] = events["board_rx_s"] - events["command_dispatch_s"]
    events["rx_to_commit_latency_s"] = np.where(
        np.isfinite(pd.to_numeric(events["receive_to_commit_us"], errors="coerce")),
        pd.to_numeric(events["receive_to_commit_us"], errors="coerce") / 1e6,
        events["commit_time_s"] - events["board_rx_s"],
    )
    events["commit_to_trainer_latency_s"] = events["trainer_transition_s"] - events["commit_capture_time_s"]
    events["commit_to_receiver_latency_s"] = events["receiver_transition_s"] - events["commit_capture_time_s"]
    events["dispatch_to_receiver_latency_s"] = events["receiver_transition_s"] - events["dispatch_capture_time_s"]
    events["scheduled_to_receiver_latency_s"] = events["receiver_transition_s"] - events["scheduled_capture_time_s"]
    return events


def latency_stats(values: pd.Series) -> dict[str, float]:
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return {
            "count": 0,
            "mean_s": np.nan,
            "median_s": np.nan,
            "p95_s": np.nan,
            "p99_s": np.nan,
            "max_s": np.nan,
        }
    return {
        "count": int(numeric_values.size),
        "mean_s": float(np.mean(numeric_values)),
        "median_s": float(np.median(numeric_values)),
        "p95_s": float(np.quantile(numeric_values, 0.95)),
        "p99_s": float(np.quantile(numeric_values, 0.99)),
        "max_s": float(np.max(numeric_values)),
    }


def build_surface_summary(events: pd.DataFrame, surface_names: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for surface_name in surface_names:
        surface_events = events.loc[events["surface_name"] == surface_name].copy()
        row = {
            "SurfaceName": surface_name,
            "TransitionCount": int(len(surface_events)),
            "TrainerMatchedCount": int(surface_events["trainer_transition_found"].sum()) if not surface_events.empty else 0,
            "ReceiverMatchedCount": int(surface_events["receiver_transition_found"].sum()) if not surface_events.empty else 0,
            "TrainerMissCount": int(len(surface_events) - surface_events["trainer_transition_found"].sum()) if not surface_events.empty else 0,
            "ReceiverMissCount": int(len(surface_events) - surface_events["receiver_transition_found"].sum()) if not surface_events.empty else 0,
        }
        for prefix, column_name in [
            ("CommitToTrainer", "commit_to_trainer_latency_s"),
            ("CommitToReceiver", "commit_to_receiver_latency_s"),
            ("DispatchToReceiver", "dispatch_to_receiver_latency_s"),
            ("ScheduledToReceiver", "scheduled_to_receiver_latency_s"),
        ]:
            stats = latency_stats(surface_events[column_name]) if column_name in surface_events.columns else latency_stats(pd.Series(dtype=float))
            for suffix, value in stats.items():
                row[f"{prefix}{suffix.capitalize()}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_overall_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric_name in [
        "commit_to_trainer_latency_s",
        "commit_to_receiver_latency_s",
        "dispatch_to_receiver_latency_s",
        "scheduled_to_receiver_latency_s",
    ]:
        stats = latency_stats(events[metric_name]) if metric_name in events.columns else latency_stats(pd.Series(dtype=float))
        rows.append({"metric": metric_name, **stats})
    return pd.DataFrame(rows)


def infer_surface_names(host_dispatch: pd.DataFrame) -> list[str]:
    if "surface_name" not in host_dispatch.columns:
        return DEFAULT_SURFACE_ORDER.copy()
    found_names = host_dispatch["surface_name"].astype(str).drop_duplicates().tolist()
    ordered_names = [name for name in DEFAULT_SURFACE_ORDER if name in found_names]
    remaining_names = [name for name in found_names if name not in ordered_names]
    return ordered_names + remaining_names


def build_ppm_column_lookup(surface_names: list[str]) -> dict[str, str]:
    return {
        surface_name: f"ppm_ch{surface_index + 1}_us"
        for surface_index, surface_name in enumerate(surface_names)
    }


def run_post_processing(logger_folder: Path, output_prefix: str, config: MatchingConfig) -> dict[str, Path]:
    host_dispatch = read_csv(logger_folder / "host_dispatch_log.csv")
    board_rx = read_csv(logger_folder / "board_rx_log.csv")
    board_commit = read_csv(logger_folder / "board_commit_log.csv")
    reference_capture = read_csv(logger_folder / "reference_capture.csv")
    trainer_capture = read_csv(logger_folder / "trainer_ppm_capture.csv")
    receiver_capture = read_csv(logger_folder / "receiver_capture.csv")

    to_numeric(
        host_dispatch,
        ["sample_sequence", "command_sequence", "scheduled_time_s", "command_dispatch_s", "position_norm"],
    )
    to_numeric(
        board_rx,
        ["sample_sequence", "command_sequence", "rx_time_s", "received_position_norm"],
    )
    to_numeric(
        board_commit,
        ["sample_sequence", "commit_time_s", "receive_to_commit_us"] + [f"ppm_ch{channel_index}_us" for channel_index in range(1, 9)],
    )
    to_numeric(reference_capture, ["time_s"])
    to_numeric(trainer_capture, ["time_s", "pulse_us", "frame_index"])
    to_numeric(receiver_capture, ["time_s", "pulse_us"])
    board_rx["surface_name"] = board_rx["surface_name"].astype(str)

    surface_names = infer_surface_names(host_dispatch)
    ppm_columns = build_ppm_column_lookup(surface_names)
    board_rx_lookup = build_board_rx_lookup(board_rx)

    host_state_tables: dict[str, pd.DataFrame] = {}
    host_transition_tables: dict[str, pd.DataFrame] = {}
    commit_transition_tables: dict[str, pd.DataFrame] = {}
    trainer_transition_tables: dict[str, pd.DataFrame] = {}
    trainer_capture_tables: dict[str, pd.DataFrame] = {}
    receiver_capture_tables: dict[str, pd.DataFrame] = {}
    receiver_transition_tables: dict[str, pd.DataFrame] = {}
    center_summary_rows: list[dict] = []
    candidate_differences: list[np.ndarray] = []

    for surface_name in surface_names:
        ppm_column_name = ppm_columns[surface_name]
        if ppm_column_name not in board_commit.columns:
            continue

        host_surface_states, _ = build_host_state_table(host_dispatch, surface_name)
        if host_surface_states.empty:
            continue
        state_level_count = compute_state_level_count(host_surface_states)
        host_transition_table = build_host_transition_table(host_surface_states)
        commit_transition_table = build_commit_transition_table(board_commit, host_surface_states, ppm_column_name)

        trainer_surface_rows = trainer_capture.loc[trainer_capture["surface_name"] == surface_name, ["time_s", "pulse_us"]].copy()
        receiver_surface_rows = receiver_capture.loc[receiver_capture["surface_name"] == surface_name, ["time_s", "pulse_us"]].copy()
        trainer_centers = kmeans_1d(trainer_surface_rows["pulse_us"].to_numpy(dtype=float), state_level_count)
        receiver_centers = kmeans_1d(receiver_surface_rows["pulse_us"].to_numpy(dtype=float), state_level_count)
        trainer_transition_table = build_stable_transition_table(trainer_surface_rows, trainer_centers, config.min_stable_pulses)
        receiver_transition_table = build_stable_transition_table(receiver_surface_rows, receiver_centers, config.min_stable_pulses)

        host_state_tables[surface_name] = host_surface_states
        host_transition_tables[surface_name] = host_transition_table
        commit_transition_tables[surface_name] = commit_transition_table
        trainer_transition_tables[surface_name] = trainer_transition_table
        trainer_capture_tables[surface_name] = trainer_capture.loc[
            trainer_capture["surface_name"] == surface_name,
            ["time_s", "pulse_us", "frame_index"],
        ].copy().sort_values("time_s", kind="stable").reset_index(drop=True)
        receiver_capture_tables[surface_name] = receiver_capture.loc[
            receiver_capture["surface_name"] == surface_name,
            ["time_s", "pulse_us"],
        ].copy().sort_values("time_s", kind="stable").reset_index(drop=True)
        receiver_transition_tables[surface_name] = receiver_transition_table

        candidate_differences.append(
            compute_candidate_alignment_differences(
                commit_transition_table,
                trainer_transition_table,
                config.offset_search_min_s,
                config.offset_search_max_s,
            )
        )

        for state_index, center_value in enumerate(trainer_centers):
            center_summary_rows.append({
                "surface_name": surface_name,
                "capture": "trainer",
                "state_index": state_index,
                "center_us": float(center_value),
            })
        for state_index, center_value in enumerate(receiver_centers):
            center_summary_rows.append({
                "surface_name": surface_name,
                "capture": "receiver",
                "state_index": state_index,
                "center_us": float(center_value),
            })

    transition_candidate_differences = np.concatenate([values for values in candidate_differences if values.size > 0]) if any(
        values.size > 0 for values in candidate_differences
    ) else np.array([], dtype=float)
    reference_candidate_differences = compute_reference_alignment_differences(
        board_commit,
        reference_capture,
        config.offset_search_min_s,
        config.offset_search_max_s,
    )
    alignment_offset_s, alignment_diagnostics = estimate_alignment_offset_with_trainer_scoring(
        reference_candidate_differences,
        transition_candidate_differences,
        host_transition_tables,
        commit_transition_tables,
        trainer_transition_tables,
        config,
    )
    alignment_model, local_alignment_diagnostics = estimate_local_alignment_model(
        board_commit,
        reference_capture,
        alignment_offset_s,
    )
    alignment_diagnostics = pd.concat([alignment_diagnostics, local_alignment_diagnostics], ignore_index=True)

    trainer_calibrations: dict[str, TrainerSurfaceCalibration] = {}
    receiver_calibrations: dict[str, PulseSurfaceCalibration] = {}
    for surface_name in surface_names:
        ppm_column_name = ppm_columns.get(surface_name)
        trainer_surface_rows = trainer_capture_tables.get(surface_name, pd.DataFrame())
        receiver_surface_rows = receiver_capture_tables.get(surface_name, pd.DataFrame())
        if ppm_column_name is None or ppm_column_name not in board_commit.columns:
            continue
        trainer_calibration = estimate_trainer_surface_calibration(
            board_commit,
            trainer_surface_rows,
            ppm_column_name,
            alignment_model,
        )
        trainer_calibrations[surface_name] = trainer_calibration
        receiver_calibration = estimate_receiver_surface_calibration(
            board_commit,
            receiver_surface_rows,
            ppm_column_name,
            alignment_model,
        )
        receiver_calibrations[surface_name] = receiver_calibration
        center_summary_rows.append({
            "surface_name": surface_name,
            "capture": "trainer_calibration",
            "state_index": -1,
            "center_us": float(trainer_calibration.pulse_bias_us),
        })
        center_summary_rows.append({
            "surface_name": surface_name,
            "capture": "trainer_frame_delta",
            "state_index": -1,
            "center_us": float(trainer_calibration.frame_delta),
        })
        center_summary_rows.append({
            "surface_name": surface_name,
            "capture": "trainer_tolerance",
            "state_index": -1,
            "center_us": float(trainer_calibration.pulse_tolerance_us),
        })
        center_summary_rows.append({
            "surface_name": surface_name,
            "capture": "receiver_calibration",
            "state_index": -1,
            "center_us": float(receiver_calibration.pulse_bias_us),
        })
        center_summary_rows.append({
            "surface_name": surface_name,
            "capture": "receiver_tolerance",
            "state_index": -1,
            "center_us": float(receiver_calibration.pulse_tolerance_us),
        })

    event_tables: list[pd.DataFrame] = []
    for surface_name in surface_names:
        if surface_name not in host_transition_tables or surface_name not in commit_transition_tables:
            continue
        event_table = match_surface_transitions(
            surface_name,
            host_transition_tables[surface_name],
            commit_transition_tables[surface_name],
            trainer_transition_tables.get(surface_name, pd.DataFrame()),
            trainer_capture_tables.get(surface_name, pd.DataFrame()),
            trainer_calibrations.get(surface_name, TrainerSurfaceCalibration(0.0, 0, 12.0)),
            receiver_capture_tables.get(surface_name, pd.DataFrame()),
            receiver_calibrations.get(surface_name, PulseSurfaceCalibration(0.0, 12.0)),
            receiver_transition_tables.get(surface_name, pd.DataFrame()),
            board_rx_lookup,
            alignment_model,
            config,
        )
        if not event_table.empty:
            event_tables.append(event_table)

    if not event_tables:
        raise RuntimeError("No post-processed transition events could be matched.")

    events = pd.concat(event_tables, ignore_index=True)
    surface_summary = build_surface_summary(events, surface_names)
    overall_summary = build_overall_summary(events)
    centers_table = pd.DataFrame(center_summary_rows)

    outputs = {
        "events": logger_folder / f"{output_prefix}_events.csv",
        "surface_summary": logger_folder / f"{output_prefix}_surface_summary.csv",
        "overall_summary": logger_folder / f"{output_prefix}_overall_summary.csv",
        "alignment": logger_folder / f"{output_prefix}_alignment.csv",
        "state_centers": logger_folder / f"{output_prefix}_state_centers.csv",
    }
    events.to_csv(outputs["events"], index=False)
    surface_summary.to_csv(outputs["surface_summary"], index=False)
    overall_summary.to_csv(outputs["overall_summary"], index=False)
    alignment_diagnostics.to_csv(outputs["alignment"], index=False)
    centers_table.to_csv(outputs["state_centers"], index=False)

    print(f"Logger folder: {logger_folder}")
    print(f"Alignment offset: {alignment_offset_s * 1e3:.3f} ms")
    print(f"Transition events: {len(events)}")
    print(f"Trainer matched: {int(events['trainer_transition_found'].sum())} / {len(events)}")
    print(f"Receiver matched: {int(events['receiver_transition_found'].sum())} / {len(events)}")
    if "scheduled_to_receiver_latency_s" in events.columns:
        receiver_stats = latency_stats(events["scheduled_to_receiver_latency_s"])
        print(
            "Scheduled-to-receiver latency: "
            f"median {receiver_stats['median_s'] * 1e3:.3f} ms, "
            f"p95 {receiver_stats['p95_s'] * 1e3:.3f} ms, "
            f"max {receiver_stats['max_s'] * 1e3:.3f} ms"
        )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Improved transmitter E2E post-processing based on stable state transitions "
            "and capture-alignment estimation."
        )
    )
    parser.add_argument("--logger-folder", type=Path, default=None, help="Specific *_TransmitterLogger folder to process.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed number used to resolve D_Transmitter_Test\\Seed_<N>_Transmitter_TransmitterLogger.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Search root when --logger-folder is omitted.")
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX, help="Prefix used for generated CSV files.")
    parser.add_argument("--offset-search-min-ms", type=float, default=150.0, help="Minimum trainer alignment-candidate offset in ms.")
    parser.add_argument("--offset-search-max-ms", type=float, default=350.0, help="Maximum trainer alignment-candidate offset in ms.")
    parser.add_argument("--offset-bin-ms", type=float, default=0.5, help="Histogram bin width for offset estimation in ms.")
    parser.add_argument("--offset-refine-ms", type=float, default=1.5, help="Half-window around the winning offset mode in ms.")
    parser.add_argument("--trainer-match-window-ms", type=float, default=30.0, help="Maximum commit-to-trainer transition latency in ms after alignment.")
    parser.add_argument("--receiver-match-window-ms", type=float, default=50.0, help="Maximum commit-to-receiver transition latency in ms after alignment.")
    parser.add_argument("--min-stable-pulses", type=int, default=2, help="Minimum consecutive pulses required to accept a capture state change.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.logger_folder is not None:
        logger_folder = args.logger_folder
    elif args.seed is not None:
        logger_folder = logger_folder_from_seed(args.root, int(args.seed))
    else:
        logger_folder = latest_logger_folder(args.root)
    config = MatchingConfig(
        offset_search_min_s=args.offset_search_min_ms / 1e3,
        offset_search_max_s=args.offset_search_max_ms / 1e3,
        offset_bin_width_s=args.offset_bin_ms / 1e3,
        offset_refine_half_window_s=args.offset_refine_ms / 1e3,
        trainer_match_window_s=args.trainer_match_window_ms / 1e3,
        receiver_match_window_s=args.receiver_match_window_ms / 1e3,
        min_stable_pulses=max(1, int(args.min_stable_pulses)),
    )
    run_post_processing(logger_folder, args.output_prefix, config)


if __name__ == "__main__":
    main()
