from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from Plot_Transmitter_Test import build_workbook_bundle, plot_transmitter_summary_figure


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


def _latest_logger_folder(root: Path) -> Path:
    candidates = [path for path in root.rglob("*") if path.is_dir() and path.name.endswith("TransmitterLogger")]
    if not candidates:
        raise FileNotFoundError(f"No '*TransmitterLogger' folder found under: {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


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


def _build_input_signal(events: pd.DataFrame, surfaces: List[str]) -> pd.DataFrame:
    rows = []
    grouped = events.groupby("sample_index", sort=True)
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trim_start_s = max(0.0, float(trim_start_s))
    trim_end_s = max(0.0, float(trim_end_s))
    if trim_start_s <= 0.0 and trim_end_s <= 0.0:
        return events, receiver_capture

    time_column = "command_dispatch_s" if "command_dispatch_s" in events.columns else "scheduled_time_s"
    if time_column not in events.columns:
        return events, receiver_capture

    event_time = pd.to_numeric(events[time_column], errors="coerce")
    finite_time = event_time[np.isfinite(event_time.to_numpy(dtype=float))]
    if finite_time.empty:
        return events, receiver_capture

    start_time = float(finite_time.min()) + trim_start_s
    stop_time = float(finite_time.max()) - trim_end_s
    if stop_time <= start_time:
        return events, receiver_capture

    event_mask = event_time.between(start_time, stop_time, inclusive="both")
    trimmed_events = events.loc[event_mask].copy()

    trimmed_receiver = receiver_capture.copy()
    if "time_s" in trimmed_receiver.columns:
        receiver_time = pd.to_numeric(trimmed_receiver["time_s"], errors="coerce")
        receiver_mask = receiver_time.between(start_time, stop_time, inclusive="both")
        trimmed_receiver = trimmed_receiver.loc[receiver_mask].copy()

    return trimmed_events, trimmed_receiver


def _write_e2e_workbook(
    logger_folder: Path,
    event_prefix: str,
    workbook_path: Path,
    trim_start_s: float,
    trim_end_s: float,
) -> Path:
    events = _read_csv(logger_folder / f"{event_prefix}_event_latency.csv")
    receiver_capture = _read_csv(logger_folder / "receiver_capture.csv")

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
            "scheduled_to_receiver_latency_s",
        ],
    )
    _to_numeric(receiver_capture, ["time_s", "pulse_us", "sample_index", "sample_rate_hz"])
    events["surface_name"] = events["surface_name"].astype(str)
    receiver_capture["surface_name"] = receiver_capture["surface_name"].astype(str)
    events, receiver_capture = _trim_window(events, receiver_capture, trim_start_s, trim_end_s)

    if "host_scheduling_delay_s" not in events.columns:
        events["host_scheduling_delay_s"] = events["command_dispatch_s"] - events["scheduled_time_s"]

    surfaces = sorted(events["surface_name"].dropna().unique().tolist())
    run_label = logger_folder.name.replace("_TransmitterLogger", "")

    critical_settings = _build_critical_settings(run_label, logger_folder)
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build transmitter E2E comparable Excel + plot from MATLAB shared-clock E2E outputs."
        )
    )
    parser.add_argument("--logger-folder", type=str, default="", help="*_TransmitterLogger folder path.")
    parser.add_argument("--root-folder", type=str, default="D_Transmitter_Test", help="Search root when logger folder omitted.")
    parser.add_argument("--event-prefix", type=str, default="e2e_shared_clock", help="Prefix used by Transmitter_Test_E2E outputs.")
    parser.add_argument("--trim-start-s", type=float, default=10.0, help="Trim this many seconds from the start of event timeline.")
    parser.add_argument("--trim-end-s", type=float, default=10.0, help="Trim this many seconds from the end of event timeline.")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = Path(args.root_folder).resolve()
    if args.logger_folder:
        logger_folder = Path(args.logger_folder).resolve()
    else:
        logger_folder = _latest_logger_folder(root)

    required_event_path = logger_folder / f"{args.event_prefix}_event_latency.csv"
    if not required_event_path.is_file():
        raise FileNotFoundError(
            "Missing E2E event file. Run Transmitter_Test_E2E.m first, then rerun this plot script.\n"
            f"Expected: {required_event_path}"
        )
    probe = pd.read_csv(required_event_path, nrows=10)
    required_true_columns = {
        "is_true_e2e",
        "true_ppm_to_receiver_latency_s",
        "true_scheduled_to_receiver_latency_s",
        "true_dispatch_to_receiver_latency_s",
    }
    missing_columns = sorted(required_true_columns - set(probe.columns))
    if missing_columns:
        raise ValueError(
            "E2E event file schema is incomplete for the new MATLAB pipeline.\n"
            f"Missing columns: {', '.join(missing_columns)}\n"
            f"File: {required_event_path}"
        )

    run_label = logger_folder.name.replace("_TransmitterLogger", "")
    workbook_dir = logger_folder.parent
    workbook_path = workbook_dir / f"{run_label}.xlsx"
    workbook_path = _write_e2e_workbook(
        logger_folder,
        args.event_prefix,
        workbook_path,
        trim_start_s=float(args.trim_start_s),
        trim_end_s=float(args.trim_end_s),
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
