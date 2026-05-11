from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from Plot_Transmitter_Test import build_workbook_bundle, plot_transmitter_summary_figure


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and Latency Column Maps
# 2) Logger and CSV Loading Helpers
# 3) Signal, Summary, and Workbook Builders
# 4) Post-Processor Event Preparation
# 5) CLI Entry Point
# =============================================================================

# =============================================================================
# 1) Constants and Latency Column Maps
# =============================================================================
DEFAULT_SEED: int | None = 1
DEFAULT_PLOT_MODE = "post"
# Mode selects the provenance of event latencies: Python transition matching or
# the older MATLAB shared-clock path.
DEFAULT_EVENT_PREFIX_BY_MODE = {
    "post": "post_transition_e2e",
    "matlab": "e2e_shared_clock",
}

# These columns are seconds-valued post-processed latency stages; names differ
# between workbook summary prefixes and event-level CSV columns.
SUMMARY_PREFIX_TO_COLUMN = {
    "HostSchedulingDelay": "host_scheduling_delay_s",
    "ComputerToArduinoRxLatency": "dispatch_to_rx_latency_s",
    "ArduinoReceiveToPpmCommitLatency": "rx_to_commit_latency_s",
    "PpmToReceiverLatency": "true_ppm_to_receiver_latency_s",
    "ScheduledToReceiverLatency": "true_scheduled_to_receiver_latency_s",
    "AnchorToPpmLatency": "anchor_to_ppm_latency_s",
    "AnchorToReceiverLatency": "anchor_to_receiver_latency_s",
    "DispatchToReceiverLatency": "true_dispatch_to_receiver_latency_s",
}


# =============================================================================
# 2) Logger and CSV Loading Helpers
# =============================================================================
def _latest_logger_folder(root: Path) -> Path:
    candidates = [path for path in root.rglob("*") if path.is_dir() and path.name.endswith("TransmitterLogger")]
    if not candidates:
        raise FileNotFoundError(f"No '*TransmitterLogger' folder found under: {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _logger_folder_from_seed(root: Path, seed: int) -> Path:
    logger_folder = root / f"Seed_{int(seed)}_Transmitter_TransmitterLogger"
    if not logger_folder.is_dir():
        raise FileNotFoundError(f"Seed logger folder not found: {logger_folder}")
    return logger_folder


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _latency_stats(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "SampleCount": 0,
            "Mean_s": np.nan,
            "Std_s": np.nan,
            "Median_s": np.nan,
            "P95_s": np.nan,
            "P99_s": np.nan,
            "Max_s": np.nan,
        }
    return {
        "SampleCount": int(values.size),
        "Mean_s": float(np.mean(values)),
        "Std_s": float(np.std(values, ddof=0)) if values.size >= 2 else np.nan,
        "Median_s": float(np.percentile(values, 50)),
        "P95_s": float(np.percentile(values, 95)),
        "P99_s": float(np.percentile(values, 99)),
        "Max_s": float(np.max(values)),
    }


# =============================================================================
# 3) Signal, Summary, and Workbook Builders
# =============================================================================
def _build_input_signal(events: pd.DataFrame, surfaces: List[str]) -> pd.DataFrame:
    rows = []
    grouped = events.groupby("sample_index", sort=True)
    for sample_index, group in grouped:
        # A single host command can expand to one row per surface; medians keep
        # the reconstructed input signal robust to duplicate telemetry rows.
        row = {
            "sample_index": sample_index,
            "time_s": float(np.nanmedian(group["command_dispatch_s"])),
            "scheduled_time_s": float(np.nanmedian(group["scheduled_time_s"])),
            "command_write_start_s": float(np.nanmedian(group["command_dispatch_s"])),
            "command_write_stop_s": float(np.nanmedian(group["command_dispatch_s"])),
        }
        base_position = float(np.nanmedian(group["position_norm"]))
        row["base_command_deg"] = 180.0 * (base_position - 0.5)
        for surface in surfaces:
            sub = group[group["surface_name"] == surface]
            pos = float(np.nanmedian(sub["position_norm"])) if not sub.empty else np.nan
            row[f"{surface}_desired_deg"] = 180.0 * (pos - 0.5) if np.isfinite(pos) else np.nan
            row[f"{surface}_command_position"] = pos
            row[f"{surface}_command_saturated"] = False
        rows.append(row)
    input_df = pd.DataFrame(rows).sort_values("sample_index", kind="stable").reset_index(drop=True)
    return input_df.drop(columns=["sample_index"])


def _build_post_input_signal(
    host_dispatch: pd.DataFrame,
    surfaces: List[str],
) -> pd.DataFrame:
    rows = []
    grouped = host_dispatch.groupby("sample_index", sort=True)
    for sample_index, group in grouped:
        row = {
            "sample_index": sample_index,
            "time_s": float(np.nanmedian(group["command_dispatch_s"])),
            "scheduled_time_s": float(np.nanmedian(group["scheduled_time_s"])),
            "command_write_start_s": float(np.nanmedian(group["command_dispatch_s"])),
            "command_write_stop_s": float(np.nanmedian(group["command_dispatch_s"])),
        }
        base_position = float(np.nanmedian(group["position_norm"]))
        row["base_command_deg"] = 180.0 * (base_position - 0.5)
        for surface in surfaces:
            sub = group[group["surface_name"] == surface]
            pos = float(np.nanmedian(sub["position_norm"])) if not sub.empty else np.nan
            row[f"{surface}_desired_deg"] = 180.0 * (pos - 0.5) if np.isfinite(pos) else np.nan
            row[f"{surface}_command_position"] = pos
            row[f"{surface}_command_saturated"] = False
        rows.append(row)
    input_df = pd.DataFrame(rows).sort_values("sample_index", kind="stable").reset_index(drop=True)
    return input_df.drop(columns=["sample_index"])


def _build_surface_wide_sheet(
    events: pd.DataFrame,
    surfaces: List[str],
    time_source_column: str,
    per_surface_columns: Dict[str, str],
) -> pd.DataFrame:
    sample_base = (
        events.groupby("sample_index", as_index=False)
        .agg(time_s=(time_source_column, "median"))
        .sort_values("sample_index", kind="stable")
    )

    out = sample_base.copy()
    for surface in surfaces:
        surface_events = events[events["surface_name"] == surface].copy()
        keep = ["sample_index", "command_sequence", "scheduled_time_s", "command_dispatch_s"] + list(
            per_surface_columns.values()
        )
        keep = [column for column in keep if column in surface_events.columns]
        surface_events = surface_events[keep].drop_duplicates(subset=["sample_index"], keep="first")

        rename = {}
        if "command_sequence" in surface_events.columns:
            rename["command_sequence"] = f"{surface}_command_sequence"
        if "scheduled_time_s" in surface_events.columns:
            rename["scheduled_time_s"] = f"{surface}_scheduled_time_s"
        if "command_dispatch_s" in surface_events.columns:
            rename["command_dispatch_s"] = f"{surface}_command_dispatch_s"
        for out_suffix, src_column in per_surface_columns.items():
            if src_column in surface_events.columns:
                rename[src_column] = f"{surface}_{out_suffix}"

        surface_events = surface_events.rename(columns=rename)
        out = out.merge(surface_events, on="sample_index", how="left")

    return out.drop(columns=["sample_index"])


def _build_latency_summary(events: pd.DataFrame, surfaces: List[str]) -> pd.DataFrame:
    rows = []
    for surface in surfaces:
        group = events[events["surface_name"] == surface]
        row: Dict[str, float] = {"SurfaceName": surface, "IsActive": True}
        for prefix, column in SUMMARY_PREFIX_TO_COLUMN.items():
            stats = _latency_stats(group[column]) if column in group.columns else _latency_stats(pd.Series(dtype=float))
            for stat_name, value in stats.items():
                row[f"{prefix}{stat_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _build_integrity_summary(events: pd.DataFrame, surfaces: List[str]) -> pd.DataFrame:
    rows = []
    for surface in surfaces:
        group = events[events["surface_name"] == surface]
        dispatch_count = int(group.shape[0])
        matched_rx = int(np.isfinite(group["board_rx_s"]).sum()) if "board_rx_s" in group.columns else 0
        matched_commit = int(np.isfinite(group["commit_time_s"]).sum()) if "commit_time_s" in group.columns else 0
        matched_receiver = int(np.isfinite(group["receiver_time_s"]).sum()) if "receiver_time_s" in group.columns else 0
        dropped_before_commit = max(0, dispatch_count - matched_commit)
        rows.append(
            {
                "SurfaceName": surface,
                "IsActive": True,
                "DispatchedCommandCount": dispatch_count,
                "MatchedRxCount": matched_rx,
                "UnmatchedRxCount": max(0, dispatch_count - matched_rx),
                "MatchedCommitCount": matched_commit,
                "DroppedBeforeCommitCount": dropped_before_commit,
                "MatchedReceiverCount": matched_receiver,
                "UnmatchedReceiverCount": max(0, dispatch_count - matched_receiver),
            }
        )
    return pd.DataFrame(rows)


def _build_critical_settings(run_label: str, logger_folder: Path) -> pd.DataFrame:
    settings = [
        ("Run", "RunLabel", run_label),
        ("Run", "Status", "completed"),
        ("Run", "OutputFolder", str(logger_folder.parent.resolve())),
        ("Run", "LoggerFolder", str(logger_folder.resolve())),
        ("Run", "ArduinoBoard", "Uno"),
        ("Command", "Mode", "all"),
        ("Profile", "Type", "latency_step_train"),
        ("LogicAnalyzer", "SampleRateHz", "4000000"),
        ("TrainerPPM", "FrameLengthUs", "20000"),
        ("Matching", "Mode", "shared_clock_e2e"),
        ("Matching", "AnchorPriority", "D4_then_D5"),
        ("Analysis", "LatencySummarySource", "e2e_shared_clock_matlab"),
    ]
    return pd.DataFrame(settings, columns=["Category", "Setting", "Value"])


def _build_post_critical_settings(run_label: str, logger_folder: Path) -> pd.DataFrame:
    settings = [
        ("Run", "RunLabel", run_label),
        ("Run", "Status", "completed"),
        ("Run", "OutputFolder", str(logger_folder.parent.resolve())),
        ("Run", "LoggerFolder", str(logger_folder.resolve())),
        ("Run", "ArduinoBoard", "Uno"),
        ("Command", "Mode", "all"),
        ("Profile", "Type", "latency_step_train"),
        ("LogicAnalyzer", "SampleRateHz", "4000000"),
        ("TrainerPPM", "FrameLengthUs", "20000"),
        ("Matching", "Mode", "stable_transition_post"),
        ("Matching", "AnchorPriority", "estimated_global_alignment"),
        ("Analysis", "LatencySummarySource", "post_transition_e2e_python"),
    ]
    return pd.DataFrame(settings, columns=["Category", "Setting", "Value"])


def _empty_profile_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EventIndex": pd.Series(dtype=float),
            "EventLabel": pd.Series(dtype=str),
            "StartTime_s": pd.Series(dtype=float),
            "StopTime_s": pd.Series(dtype=float),
            "TargetDeflection_deg": pd.Series(dtype=float),
        }
    )


def _resolve_writable_workbook_path(requested_path: Path) -> Path:
    if not requested_path.exists():
        return requested_path
    try:
        with requested_path.open("a", encoding="utf-8"):
            pass
        return requested_path
    except OSError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return requested_path.with_name(f"{requested_path.stem}_locked_{timestamp}{requested_path.suffix}")


def _trim_window(
    events: pd.DataFrame,
    receiver_capture: pd.DataFrame,
    trim_start_s: float,
    trim_end_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, float | None]:
    trim_start_s = max(0.0, float(trim_start_s))
    trim_end_s = max(0.0, float(trim_end_s))
    if trim_start_s <= 0.0 and trim_end_s <= 0.0:
        return events, receiver_capture, None, None

    time_column = "scheduled_time_s" if "scheduled_time_s" in events.columns else "command_dispatch_s"
    if time_column not in events.columns:
        return events, receiver_capture, None, None

    event_time = pd.to_numeric(events[time_column], errors="coerce")
    finite_time = event_time[np.isfinite(event_time.to_numpy(dtype=float))]
    if finite_time.empty:
        return events, receiver_capture, None, None

    start_time = float(finite_time.min()) + trim_start_s
    stop_time = float(finite_time.max()) - trim_end_s
    if stop_time <= start_time:
        return events, receiver_capture, None, None

    event_mask = event_time.between(start_time, stop_time, inclusive="both")
    trimmed_events = events.loc[event_mask].copy()

    trimmed_receiver = receiver_capture.copy()
    if "time_s" in trimmed_receiver.columns:
        receiver_time = pd.to_numeric(trimmed_receiver["time_s"], errors="coerce")
        receiver_mask = receiver_time.between(start_time, stop_time, inclusive="both")
        trimmed_receiver = trimmed_receiver.loc[receiver_mask].copy()

    return trimmed_events, trimmed_receiver, start_time, stop_time


def _compute_host_trim_window(
    host_dispatch: pd.DataFrame,
    trim_start_s: float,
    trim_end_s: float,
) -> tuple[float, float] | tuple[None, None]:
    time_column = "scheduled_time_s" if "scheduled_time_s" in host_dispatch.columns else "command_dispatch_s"
    if host_dispatch.empty or time_column not in host_dispatch.columns:
        return None, None

    dispatch_time = pd.to_numeric(host_dispatch[time_column], errors="coerce")
    finite_dispatch = dispatch_time[np.isfinite(dispatch_time.to_numpy(dtype=float))]
    if finite_dispatch.empty:
        return None, None

    start_time = float(finite_dispatch.min()) + max(0.0, float(trim_start_s))
    stop_time = float(finite_dispatch.max()) - max(0.0, float(trim_end_s))
    if stop_time <= start_time:
        return None, None
    return start_time, stop_time


def _trim_frame_by_time(
    frame: pd.DataFrame,
    time_column: str,
    start_time_s: float | None,
    stop_time_s: float | None,
) -> pd.DataFrame:
    if (
        frame.empty
        or start_time_s is None
        or stop_time_s is None
        or time_column not in frame.columns
    ):
        return frame

    time_values = pd.to_numeric(frame[time_column], errors="coerce")
    mask = time_values.between(start_time_s, stop_time_s, inclusive="both")
    return frame.loc[mask].copy()


# =============================================================================
# 4) Post-Processor Event Preparation
# =============================================================================
def _shift_time_columns(
    frame: pd.DataFrame,
    origin_s: float | None,
    column_names: Iterable[str],
) -> pd.DataFrame:
    if frame.empty or origin_s is None:
        return frame

    shifted = frame.copy()
    for column_name in column_names:
        if column_name not in shifted.columns:
            continue
        shifted[column_name] = pd.to_numeric(shifted[column_name], errors="coerce") - float(origin_s)
    return shifted


def _prepare_post_events(events: pd.DataFrame) -> pd.DataFrame:
    post_events = events.copy()
    if "sample_index" not in post_events.columns:
        post_events["sample_index"] = pd.to_numeric(post_events["sample_sequence"], errors="coerce")
    if "position_norm" not in post_events.columns:
        post_events["position_norm"] = np.clip((pd.to_numeric(post_events["expected_pulse_us"], errors="coerce") - 1000.0) / 1000.0, 0.0, 1.0)
    if "host_scheduling_delay_s" not in post_events.columns:
        post_events["host_scheduling_delay_s"] = (
            pd.to_numeric(post_events["command_dispatch_s"], errors="coerce")
            - pd.to_numeric(post_events["scheduled_time_s"], errors="coerce")
        )
    if "board_rx_s" not in post_events.columns:
        post_events["board_rx_s"] = np.nan
    if "dispatch_to_rx_latency_s" not in post_events.columns:
        post_events["dispatch_to_rx_latency_s"] = np.nan
    if "rx_to_commit_latency_s" not in post_events.columns:
        post_events["rx_to_commit_latency_s"] = np.nan
    if "ppm_time_s" not in post_events.columns:
        post_events["ppm_time_s"] = pd.to_numeric(post_events.get("commit_capture_time_s"), errors="coerce")
    if "receiver_time_s" not in post_events.columns:
        post_events["receiver_time_s"] = pd.to_numeric(post_events.get("receiver_transition_s"), errors="coerce")
    if "true_ppm_to_receiver_latency_s" not in post_events.columns:
        commit_to_receiver = pd.to_numeric(post_events.get("commit_to_receiver_latency_s"), errors="coerce")
        if np.isfinite(commit_to_receiver).any():
            post_events["true_ppm_to_receiver_latency_s"] = commit_to_receiver
        else:
            post_events["true_ppm_to_receiver_latency_s"] = (
                pd.to_numeric(post_events["receiver_time_s"], errors="coerce")
                - pd.to_numeric(post_events["ppm_time_s"], errors="coerce")
            )
    if "true_scheduled_to_receiver_latency_s" not in post_events.columns:
        post_events["true_scheduled_to_receiver_latency_s"] = pd.to_numeric(
            post_events.get("scheduled_to_receiver_latency_s"),
            errors="coerce",
        )
    if "true_dispatch_to_receiver_latency_s" not in post_events.columns:
        post_events["true_dispatch_to_receiver_latency_s"] = pd.to_numeric(
            post_events.get("dispatch_to_receiver_latency_s"),
            errors="coerce",
        )
    if "commit_time_s" not in post_events.columns:
        post_events["commit_time_s"] = pd.to_numeric(post_events.get("commit_capture_time_s"), errors="coerce")
    if "ppm_pulse_us" not in post_events.columns:
        post_events["ppm_pulse_us"] = pd.to_numeric(post_events.get("expected_pulse_us"), errors="coerce")
    if "receiver_pulse_us" not in post_events.columns:
        post_events["receiver_pulse_us"] = np.nan
    return post_events


def _build_post_integrity_summary(events: pd.DataFrame, surfaces: List[str]) -> pd.DataFrame:
    rows = []
    for surface in surfaces:
        group = events[events["surface_name"] == surface]
        transition_count = int(group.shape[0])
        matched_receiver = int(pd.to_numeric(group.get("receiver_transition_found"), errors="coerce").fillna(0).sum())
        matched_trainer = int(pd.to_numeric(group.get("trainer_transition_found"), errors="coerce").fillna(0).sum())
        rows.append(
            {
                "SurfaceName": surface,
                "IsActive": True,
                "DispatchedCommandCount": transition_count,
                "MatchedRxCount": transition_count,
                "UnmatchedRxCount": 0,
                "MatchedCommitCount": transition_count,
                "DroppedBeforeCommitCount": 0,
                "MatchedReceiverCount": matched_receiver,
                "UnmatchedReceiverCount": max(0, transition_count - matched_receiver),
                "MatchedTrainerCount": matched_trainer,
                "UnmatchedTrainerCount": max(0, transition_count - matched_trainer),
            }
        )
    return pd.DataFrame(rows)


def _write_e2e_workbook(
    logger_folder: Path,
    event_prefix: str,
    workbook_path: Path,
    trim_start_s: float,
    trim_end_s: float,
    source_mode: str,
) -> Path:
    host_dispatch = _read_csv(logger_folder / "host_dispatch_log.csv")
    if source_mode == "post":
        events = _read_csv(logger_folder / f"{event_prefix}_events.csv")
        alignment = _read_csv(logger_folder / f"{event_prefix}_alignment.csv")
    else:
        events = _read_csv(logger_folder / f"{event_prefix}_event_latency.csv")
        alignment = pd.DataFrame()
    receiver_capture = _read_csv(logger_folder / "receiver_capture.csv")

    _to_numeric(
        host_dispatch,
        [
            "sample_index",
            "command_sequence",
            "sample_sequence",
            "scheduled_time_s",
            "command_dispatch_s",
            "position_norm",
        ],
    )
    _to_numeric(
        events,
        [
            "sample_index",
            "command_sequence",
            "sample_sequence",
            "scheduled_time_s",
            "command_dispatch_s",
            "position_norm",
            "board_rx_s",
            "commit_time_s",
            "ppm_time_s",
            "receiver_time_s",
            "host_scheduling_delay_s",
            "dispatch_to_rx_latency_s",
            "rx_to_commit_latency_s",
            "ppm_to_receiver_latency_s",
            "true_ppm_to_receiver_latency_s",
            "scheduled_to_receiver_latency_s",
            "true_scheduled_to_receiver_latency_s",
            "true_dispatch_to_receiver_latency_s",
            "expected_pulse_us",
            "trainer_transition_s",
            "receiver_transition_s",
        ],
    )
    _to_numeric(receiver_capture, ["time_s", "pulse_us", "sample_index", "sample_rate_hz"])
    events["surface_name"] = events["surface_name"].astype(str)
    receiver_capture["surface_name"] = receiver_capture["surface_name"].astype(str)
    if source_mode == "post":
        events = _prepare_post_events(events)

    if "host_scheduling_delay_s" not in events.columns:
        events["host_scheduling_delay_s"] = events["command_dispatch_s"] - events["scheduled_time_s"]

    surfaces = sorted(events["surface_name"].dropna().unique().tolist())
    run_label = logger_folder.name.replace("_TransmitterLogger", "")

    if source_mode == "post":
        critical_settings = _build_post_critical_settings(run_label, logger_folder)
        alignment_offset_series = alignment.loc[alignment["metric"] == "alignment_offset_s", "value"]
        alignment_offset_s = float(pd.to_numeric(alignment_offset_series, errors="coerce").iloc[0]) if not alignment_offset_series.empty else 0.0
        host_trim_start_s, host_trim_stop_s = _compute_host_trim_window(host_dispatch, trim_start_s, trim_end_s)
        host_dispatch_time_column = "scheduled_time_s" if "scheduled_time_s" in host_dispatch.columns else "command_dispatch_s"
        event_time_column = "scheduled_time_s" if "scheduled_time_s" in events.columns else "command_dispatch_s"
        host_dispatch = _trim_frame_by_time(host_dispatch, host_dispatch_time_column, host_trim_start_s, host_trim_stop_s)
        events = _trim_frame_by_time(events, event_time_column, host_trim_start_s, host_trim_stop_s)
        receiver_capture = _trim_frame_by_time(
            receiver_capture,
            "time_s",
            None if host_trim_start_s is None else host_trim_start_s + alignment_offset_s,
            None if host_trim_stop_s is None else host_trim_stop_s + alignment_offset_s,
        )
        host_dispatch = _shift_time_columns(
            host_dispatch,
            host_trim_start_s,
            ["scheduled_time_s", "command_dispatch_s"],
        )
        events = _shift_time_columns(
            events,
            host_trim_start_s,
            [
                "scheduled_time_s",
                "command_dispatch_s",
                "board_rx_s",
                "commit_time_s",
                "scheduled_capture_time_s",
                "dispatch_capture_time_s",
                "commit_capture_time_s",
                "trainer_transition_s",
                "receiver_transition_s",
                "ppm_time_s",
                "receiver_time_s",
            ],
        )
        receiver_capture = _shift_time_columns(receiver_capture, host_trim_start_s, ["time_s"])
        input_signal = _build_post_input_signal(host_dispatch, surfaces)
    else:
        critical_settings = _build_critical_settings(run_label, logger_folder)
        events, receiver_capture, host_trim_start_s, _ = _trim_window(events, receiver_capture, trim_start_s, trim_end_s)
        events = _shift_time_columns(
            events,
            host_trim_start_s,
            [
                "scheduled_time_s",
                "command_dispatch_s",
                "board_rx_s",
                "commit_time_s",
                "ppm_time_s",
                "receiver_time_s",
            ],
        )
        receiver_capture = _shift_time_columns(receiver_capture, host_trim_start_s, ["time_s"])
        input_signal = _build_input_signal(events, surfaces)
    host_scheduling_delay = _build_surface_wide_sheet(
        events,
        surfaces,
        time_source_column="command_dispatch_s",
        per_surface_columns={"host_scheduling_delay_s": "host_scheduling_delay_s"},
    )
    computer_to_rx = _build_surface_wide_sheet(
        events,
        surfaces,
        time_source_column="board_rx_s",
        per_surface_columns={
            "arduino_rx_s": "board_rx_s",
            "computer_to_arduino_rx_latency_s": "dispatch_to_rx_latency_s",
        },
    )
    rx_to_commit = _build_surface_wide_sheet(
        events,
        surfaces,
        time_source_column="commit_time_s",
        per_surface_columns={
            "arduino_rx_s": "board_rx_s",
            "ppm_commit_s": "commit_time_s",
            "arduino_receive_to_ppm_commit_latency_s": "rx_to_commit_latency_s",
        },
    )
    ppm_to_receiver = _build_surface_wide_sheet(
        events,
        surfaces,
        time_source_column="receiver_time_s",
        per_surface_columns={
            "trainer_ppm_s": "ppm_time_s",
            "receiver_response_s": "receiver_time_s",
            "ppm_to_receiver_latency_s": "true_ppm_to_receiver_latency_s",
        },
    )
    scheduled_to_receiver = _build_surface_wide_sheet(
        events,
        surfaces,
        time_source_column="receiver_time_s",
        per_surface_columns={
            "receiver_response_s": "receiver_time_s",
            "scheduled_to_receiver_latency_s": "true_scheduled_to_receiver_latency_s",
        },
    )
    latency_summary = _build_latency_summary(events, surfaces)
    if source_mode == "post":
        integrity_summary = _build_post_integrity_summary(events, surfaces)
    else:
        integrity_summary = _build_integrity_summary(events, surfaces)
    profile_events = _empty_profile_events()

    workbook_path = _resolve_writable_workbook_path(workbook_path)
    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="w") as writer:
        critical_settings.to_excel(writer, sheet_name="CriticalSettings", index=False)
        input_signal.to_excel(writer, sheet_name="InputSignal", index=False)
        host_scheduling_delay.to_excel(writer, sheet_name="HostSchedulingDelay", index=False)
        computer_to_rx.to_excel(writer, sheet_name="ComputerToArduinoRxLatency", index=False)
        rx_to_commit.to_excel(writer, sheet_name="ArduinoRxToPpmCommitLatency", index=False)
        ppm_to_receiver.to_excel(writer, sheet_name="PpmToReceiverLatency", index=False)
        scheduled_to_receiver.to_excel(writer, sheet_name="ScheduledToReceiverLatency", index=False)
        latency_summary.to_excel(writer, sheet_name="LatencySummary", index=False)
        integrity_summary.to_excel(writer, sheet_name="IntegritySummary", index=False)
        profile_events.to_excel(writer, sheet_name="ProfileEvents", index=False)
        receiver_capture.to_excel(writer, sheet_name="ReceiverCapture", index=False)

    return workbook_path


# =============================================================================
# 5) CLI Entry Point
# =============================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build transmitter E2E comparable Excel + plot from MATLAB shared-clock E2E outputs."
        )
    )
    parser.add_argument("--logger-folder", type=str, default="", help="*_TransmitterLogger folder path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed number used to resolve D_Transmitter_Test\\Seed_<N>_Transmitter_TransmitterLogger.")
    parser.add_argument("--root-folder", type=str, default="D_Transmitter_Test", help="Search root when logger folder omitted.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(DEFAULT_EVENT_PREFIX_BY_MODE.keys()),
        default=DEFAULT_PLOT_MODE,
        help=(
            "Explicit plotting source mode. "
            "'post' uses Transmitter_Test_E2E_Post.py outputs; "
            "'matlab' uses Transmitter_Test_E2E.m shared-clock outputs."
        ),
    )
    parser.add_argument(
        "--event-prefix",
        type=str,
        default="",
        help=(
            "Optional explicit event prefix override. "
            "If omitted, the code uses the mode mapping declared near the top of this file."
        ),
    )
    parser.add_argument("--trim-start-s", type=float, default=10.0, help="Trim this many seconds from the start of event timeline.")
    parser.add_argument("--trim-end-s", type=float, default=10.0, help="Trim this many seconds from the end of event timeline.")
    return parser.parse_args()


def _resolve_event_source(logger_folder: Path, event_prefix: str, source_mode: str) -> tuple[Path, str]:
    matlab_event_path = logger_folder / f"{event_prefix}_event_latency.csv"
    post_event_path = logger_folder / f"{event_prefix}_events.csv"
    if source_mode == "post":
        if post_event_path.is_file():
            return post_event_path, "post"
        raise FileNotFoundError(
            "Missing post-processing event file for plotting.\n"
            f"Expected: {post_event_path}\n"
            "Run Transmitter_Test_E2E_Post.py first, or switch to --mode matlab."
        )
    if source_mode == "matlab":
        if matlab_event_path.is_file():
            return matlab_event_path, "matlab"
        raise FileNotFoundError(
            "Missing MATLAB shared-clock event file for plotting.\n"
            f"Expected: {matlab_event_path}\n"
            "Run Transmitter_Test_E2E.m first, or switch to --mode post."
        )
    if post_event_path.is_file():
        return post_event_path, "post"
    if matlab_event_path.is_file():
        return matlab_event_path, "matlab"
    raise FileNotFoundError(
        "Missing event file for plotting.\n"
        f"Expected one of:\n  {post_event_path}\n  {matlab_event_path}"
    )


def main():
    args = _parse_args()
    root = Path(args.root_folder).resolve()
    if args.logger_folder:
        logger_folder = Path(args.logger_folder).resolve()
    elif args.seed is not None:
        logger_folder = _logger_folder_from_seed(root, int(args.seed)).resolve()
    else:
        logger_folder = _latest_logger_folder(root)

    source_mode = str(args.mode)
    event_prefix = args.event_prefix.strip() or DEFAULT_EVENT_PREFIX_BY_MODE[source_mode]
    required_event_path, source_mode = _resolve_event_source(logger_folder, event_prefix, source_mode)
    probe = pd.read_csv(required_event_path, nrows=10)
    if source_mode == "matlab":
        required_columns = {
            "is_true_e2e",
            "true_ppm_to_receiver_latency_s",
            "true_scheduled_to_receiver_latency_s",
            "true_dispatch_to_receiver_latency_s",
        }
        schema_name = "MATLAB shared-clock E2E"
    else:
        required_columns = {
            "trainer_transition_s",
            "receiver_transition_s",
            "scheduled_to_receiver_latency_s",
            "dispatch_to_receiver_latency_s",
        }
        schema_name = "post transition E2E"
    missing_columns = sorted(required_columns - set(probe.columns))
    if missing_columns:
        raise ValueError(
            f"{schema_name} event file schema is incomplete.\n"
            f"Missing columns: {', '.join(missing_columns)}\n"
            f"File: {required_event_path}"
        )

    run_label = logger_folder.name.replace("_TransmitterLogger", "")
    workbook_dir = logger_folder.parent
    if source_mode == "post":
        workbook_path = workbook_dir / f"{run_label}_{event_prefix}.xlsx"
    else:
        workbook_path = workbook_dir / f"{run_label}.xlsx"
    workbook_path = _write_e2e_workbook(
        logger_folder,
        event_prefix,
        workbook_path,
        trim_start_s=float(args.trim_start_s),
        trim_end_s=float(args.trim_end_s),
        source_mode=source_mode,
    )

    out_dir = workbook_dir / "A_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{workbook_path.stem}.png"

    workbook_bundle = build_workbook_bundle(workbook_path)
    plot_transmitter_summary_figure(workbook_path, out_path, workbook_bundle)

    print("E2E workbook and plot generated")
    print(f"  Workbook: {workbook_path.resolve()}")
    print(f"  Figure:   {out_path.resolve()}")


if __name__ == "__main__":
    main()
