from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import Plot_Transmitter_Test as transmitter_plot
import Plot_Transmitter_Test_E2E as base_plot


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Nano Transmitter Plot Constants
# 2) Logger and Settings Builders
# 3) Plot-Module Patching
# 4) CLI Entry Point
# =============================================================================

# =============================================================================
# 1) Nano Transmitter Plot Constants
# =============================================================================
DEFAULT_SEED: int | None = 2
DEFAULT_PLOT_MODE = "post"
# Reuse the base E2E modes so Nano figures compare the same latency definitions
# as the Uno transmitter plots.
DEFAULT_EVENT_PREFIX_BY_MODE = base_plot.DEFAULT_EVENT_PREFIX_BY_MODE


# =============================================================================
# 2) Logger and Settings Builders
# =============================================================================
def _logger_folder_from_seed(root: Path, seed: int) -> Path:
    logger_folder = root / f"Seed_{int(seed)}_Nano_Transmitter_TransmitterLogger"
    if not logger_folder.is_dir():
        raise FileNotFoundError(f"Seed logger folder not found: {logger_folder}")
    return logger_folder


def _build_critical_settings(run_label: str, logger_folder: Path) -> pd.DataFrame:
    # The Nano plotter synthesizes CriticalSettings for CSV-only post runs so
    # the shared plotting code still records board, frame, and matching context.
    settings = [
        ("Run", "RunLabel", run_label),
        ("Run", "Status", "completed"),
        ("Run", "OutputFolder", str(logger_folder.parent.resolve())),
        ("Run", "LoggerFolder", str(logger_folder.resolve())),
        ("Run", "ArduinoBoard", "Nano33IoT"),
        ("Command", "Mode", "all"),
        ("Profile", "Type", "latency_vector_step_train"),
        ("LogicAnalyzer", "SampleRateHz", "4000000"),
        ("TrainerPPM", "FrameLengthUs", "20000"),
        ("Matching", "Mode", "shared_clock_e2e"),
        ("Matching", "AnchorPriority", "D2_then_D3"),
        ("Analysis", "LatencySummarySource", "e2e_shared_clock_matlab"),
    ]
    return pd.DataFrame(settings, columns=["Category", "Setting", "Value"])


def _build_post_critical_settings(run_label: str, logger_folder: Path) -> pd.DataFrame:
    settings = [
        ("Run", "RunLabel", run_label),
        ("Run", "Status", "completed"),
        ("Run", "OutputFolder", str(logger_folder.parent.resolve())),
        ("Run", "LoggerFolder", str(logger_folder.resolve())),
        ("Run", "ArduinoBoard", "Nano33IoT"),
        ("Command", "Mode", "all"),
        ("Profile", "Type", "latency_vector_step_train"),
        ("LogicAnalyzer", "SampleRateHz", "4000000"),
        ("TrainerPPM", "FrameLengthUs", "20000"),
        ("Matching", "Mode", "stable_transition_post"),
        ("Matching", "AnchorPriority", "estimated_global_alignment"),
        ("Analysis", "LatencySummarySource", "post_transition_e2e_python"),
    ]
    return pd.DataFrame(settings, columns=["Category", "Setting", "Value"])


# =============================================================================
# 3) Plot-Module Patching
# =============================================================================
def _patch_plot_modules() -> None:
    # The base transmitter plotter is parameterized by module-level hooks; patch
    # only those hooks so the Nano path keeps the same workbook/figure layout.
    base_plot._logger_folder_from_seed = _logger_folder_from_seed
    base_plot._build_critical_settings = _build_critical_settings
    base_plot._build_post_critical_settings = _build_post_critical_settings

    transmitter_plot.LATENCY_METRICS = [
        {
            "sheet": "HostSchedulingDelay",
            "suffix": "_host_scheduling_delay_s",
            "label": "Scheduled to dispatch",
            "summary_prefix": "HostSchedulingDelay",
            "color": "#264653",
        },
        {
            "sheet": "ComputerToArduinoRxLatency",
            "suffix": "_computer_to_arduino_rx_latency_s",
            "label": "Dispatch to Nano RX",
            "summary_prefix": "ComputerToArduinoRxLatency",
            "color": "#2a9d8f",
        },
        {
            "sheet": "ArduinoRxToPpmCommitLatency",
            "suffix": "_arduino_receive_to_ppm_commit_latency_s",
            "label": "Nano RX to PPM commit",
            "summary_prefix": "ArduinoReceiveToPpmCommitLatency",
            "color": "#e9c46a",
        },
        {
            "sheet": "PpmToReceiverLatency",
            "suffix": "_ppm_to_receiver_latency_s",
            "label": "PPM to receiver PWM",
            "summary_prefix": "PpmToReceiverLatency",
            "color": "#f4a261",
        },
        {
            "sheet": "ScheduledToReceiverLatency",
            "suffix": "_scheduled_to_receiver_latency_s",
            "label": "Scheduled to receiver PWM",
            "summary_prefix": "ScheduledToReceiverLatency",
            "color": "#e76f51",
        },
    ]


# =============================================================================
# 4) CLI Entry Point
# =============================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build Nano 33 IoT transmitter E2E workbook + plot."
    )
    parser.add_argument("--logger-folder", type=str, default="", help="*_TransmitterLogger folder path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Explicit seed number used to resolve D_Transmitter_Test\\Seed_<N>_Nano_Transmitter_TransmitterLogger.",
    )
    parser.add_argument("--root-folder", type=str, default="D_Transmitter_Test", help="Search root when logger folder omitted.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(DEFAULT_EVENT_PREFIX_BY_MODE.keys()),
        default=DEFAULT_PLOT_MODE,
        help=(
            "Explicit plotting source mode. "
            "'post' uses Transmitter_Test_E2E_Post.py outputs; "
            "'matlab' uses MATLAB shared-clock E2E outputs if present."
        ),
    )
    parser.add_argument(
        "--event-prefix",
        type=str,
        default="",
        help="Optional explicit event prefix override.",
    )
    parser.add_argument("--trim-start-s", type=float, default=10.0, help="Trim this many seconds from the start of event timeline.")
    parser.add_argument("--trim-end-s", type=float, default=10.0, help="Trim this many seconds from the end of event timeline.")
    return parser.parse_args()


def main():
    _patch_plot_modules()

    args = _parse_args()
    root = Path(args.root_folder).resolve()
    if args.logger_folder:
        logger_folder = Path(args.logger_folder).resolve()
        selection_source = f"logger folder {logger_folder}"
    elif args.seed is not None:
        logger_folder = _logger_folder_from_seed(root, int(args.seed)).resolve()
        selection_source = f"seed {int(args.seed)}"
    else:
        raise ValueError(
            "Explicitly provide --seed <N> or --logger-folder <PATH> when plotting Nano runs."
        )

    source_mode = str(args.mode)
    event_prefix = args.event_prefix.strip() or DEFAULT_EVENT_PREFIX_BY_MODE[source_mode]
    required_event_path, source_mode = base_plot._resolve_event_source(logger_folder, event_prefix, source_mode)
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
    workbook_path = base_plot._write_e2e_workbook(
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

    workbook_bundle = transmitter_plot.build_workbook_bundle(workbook_path)
    transmitter_plot.plot_transmitter_summary_figure(workbook_path, out_path, workbook_bundle)

    print("Nano E2E workbook and plot generated")
    print(f"  Source:   {selection_source}")
    print(f"  Logger:   {logger_folder.resolve()}")
    print(f"  Workbook: {workbook_path.resolve()}")
    print(f"  Figure:   {out_path.resolve()}")


if __name__ == "__main__":
    main()
