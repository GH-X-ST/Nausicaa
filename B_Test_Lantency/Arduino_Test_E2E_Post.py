from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("C_Arduino_Test")
DEFAULT_OUTPUT_PREFIX = "e2e_output"
DEFAULT_SEED: int | None = 5
DEFAULT_SAMPLE_TIME_SECONDS = 0.02
DEFAULT_CLOCK_MAP_MODE = "command"
DEFAULT_MATCHING_MODE = "apply_anchored"
DEFAULT_MAX_OUTPUT_ASSOCIATION_SECONDS = 0.05
DEFAULT_REFERENCE_ASSOCIATION_WINDOW_SECONDS = 0.02
DEFAULT_TRANSITION_PULSE_THRESHOLD_US = 4.0
DEFAULT_TARGET_PULSE_TOLERANCE_US = 40.0
DEFAULT_PREVIOUS_PULSE_TOLERANCE_US = 25.0
DEFAULT_PRE_ANCHOR_SLACK_SECONDS = 0.005
DEFAULT_MAXIMUM_APPLY_TO_OUTPUT_SECONDS = 0.05
DEFAULT_SIGROK_SAMPLE_RATE_HZ = 4_000_000.0
DEFAULT_SIGROK_CHANNELS = [0, 1, 2, 3, 4]
DEFAULT_SIGROK_CHANNEL_NAMES = ["D0", "D1", "D2", "D3", "D4"]
DEFAULT_OUTPUT_CHANNELS = [0, 1, 2, 3]
DEFAULT_REFERENCE_CHANNEL: int | None = None
DEFAULT_MINIMUM_PULSE_US = 800.0
DEFAULT_MAXIMUM_PULSE_US = 2200.0
DEFAULT_REFERENCE_DEBOUNCE_US = 100.0
BOOL_COLUMNS = [
    "is_transition_command",
    "matched_output_transition",
    "is_valid_e2e",
    "IsActive",
]


@dataclass(frozen=True)
class LogicAnalyzerConfig:
    sample_rate_hz: float = DEFAULT_SIGROK_SAMPLE_RATE_HZ
    channels: tuple[int, ...] = tuple(DEFAULT_SIGROK_CHANNELS)
    channel_names: tuple[str, ...] = tuple(DEFAULT_SIGROK_CHANNEL_NAMES)
    output_channels: tuple[int, ...] = tuple(DEFAULT_OUTPUT_CHANNELS)
    reference_channel: int | None = DEFAULT_REFERENCE_CHANNEL
    minimum_pulse_us: float = DEFAULT_MINIMUM_PULSE_US
    maximum_pulse_us: float = DEFAULT_MAXIMUM_PULSE_US
    reference_debounce_us: float = DEFAULT_REFERENCE_DEBOUNCE_US


@dataclass(frozen=True)
class AlignmentMetadata:
    output_time_offset_s: float = 0.0
    output_time_drift_slope: float = 0.0
    output_time_drift_baseline_s: float = 0.0


def latest_logger_folder(root: Path) -> Path:
    candidates = [path for path in root.glob("*_ArduinoLogger") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No '*_ArduinoLogger' folder found under: {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def logger_folder_from_seed(root: Path, seed: int) -> Path:
    controller = root / f"Seed_{int(seed)}_Controller_ArduinoLogger"
    if controller.is_dir():
        return controller
    generic = root / f"Seed_{int(seed)}_ArduinoLogger"
    if generic.is_dir():
        return generic
    raise FileNotFoundError(f"Seed logger folder not found for seed {seed} under: {root}")


def resolve_logger_folder(logger_folder: str | None, seed: int | None) -> Path:
    if logger_folder:
        path = Path(logger_folder)
        if not path.is_absolute():
            path = DEFAULT_ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"Logger folder not found: {path}")
        return path
    chosen_seed = DEFAULT_SEED if seed is None else seed
    if chosen_seed is not None:
        return logger_folder_from_seed(DEFAULT_ROOT, chosen_seed)
    return latest_logger_folder(DEFAULT_ROOT)


def strip_logger_suffix(logger_folder: Path) -> str:
    name = logger_folder.name
    if name.endswith("_ArduinoLogger"):
        return name[: -len("_ArduinoLogger")]
    return name


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def convert_column_to_numeric(column: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(column):
        return pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    text = column.astype(str).str.strip().replace({"true": "1", "false": "0", "True": "1", "False": "0"})
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=float)


def normalize_time_column(raw_time: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    raw_time = np.asarray(raw_time, dtype=float).reshape(-1)
    time_s = raw_time.copy()
    finite = raw_time[np.isfinite(raw_time)]
    if finite.size and np.nanmax(np.abs(finite)) > 1e4:
        time_s = raw_time / float(sample_rate_hz)
    return time_s


def build_input_signal(host_dispatch_log: pd.DataFrame, surface_names: list[str], sample_time_seconds: float) -> pd.DataFrame:
    dispatch = host_dispatch_log.sort_values(["command_sequence", "surface_name"], kind="stable").copy()
    seq = pd.Index(pd.unique(pd.to_numeric(dispatch["command_sequence"], errors="coerce"))).dropna().astype(int)
    input_signal = pd.DataFrame({"command_sequence": seq.to_numpy(dtype=int)})

    if "scheduled_time_s" in dispatch.columns:
        schedule_by_seq = dispatch.groupby("command_sequence", sort=False)["scheduled_time_s"].median()
        input_signal["scheduled_time_s"] = input_signal["command_sequence"].map(schedule_by_seq)
    else:
        input_signal["scheduled_time_s"] = (input_signal["command_sequence"] - int(input_signal["command_sequence"].min())) * sample_time_seconds

    if "command_dispatch_s" in dispatch.columns:
        dispatch_by_seq = dispatch.groupby("command_sequence", sort=False)["command_dispatch_s"].median()
        input_signal["command_dispatch_s"] = input_signal["command_sequence"].map(dispatch_by_seq)
        if not np.isfinite(pd.to_numeric(input_signal["command_dispatch_s"], errors="coerce")).any():
            dispatch_us = dispatch.groupby("command_sequence", sort=False)["command_dispatch_us"].median()
            origin_us = float(pd.to_numeric(dispatch_us, errors="coerce").min())
            input_signal["command_dispatch_s"] = (input_signal["command_sequence"].map(dispatch_us) - origin_us) / 1e6
    else:
        dispatch_us = dispatch.groupby("command_sequence", sort=False)["command_dispatch_us"].median()
        origin_us = float(pd.to_numeric(dispatch_us, errors="coerce").min())
        input_signal["command_dispatch_s"] = (input_signal["command_sequence"].map(dispatch_us) - origin_us) / 1e6

    input_signal["time_s"] = input_signal["command_dispatch_s"]
    input_signal["command_write_start_s"] = input_signal["command_dispatch_s"]
    input_signal["command_write_stop_s"] = input_signal["command_dispatch_s"]
    input_signal["base_command_deg"] = np.nan

    for idx, command_sequence in enumerate(input_signal["command_sequence"].to_numpy(dtype=int)):
        subset = dispatch.loc[pd.to_numeric(dispatch["command_sequence"], errors="coerce") == command_sequence].copy()
        surface_positions = pd.to_numeric(subset["position_norm"], errors="coerce").to_numpy(dtype=float)
        finite_positions = surface_positions[np.isfinite(surface_positions)]
        if finite_positions.size and (np.nanmax(finite_positions) - np.nanmin(finite_positions)) <= 1e-9:
            input_signal.at[idx, "base_command_deg"] = 180.0 * (float(np.nanmedian(finite_positions)) - 0.5)
        for surface_name in surface_names:
            surface_subset = subset.loc[subset["surface_name"].astype(str) == surface_name]
            if surface_subset.empty:
                pos = np.nan
            else:
                pos = float(np.nanmedian(pd.to_numeric(surface_subset["position_norm"], errors="coerce")))
            input_signal.at[idx, f"{surface_name}_desired_deg"] = 180.0 * (pos - 0.5) if np.isfinite(pos) else np.nan
            input_signal.at[idx, f"{surface_name}_command_position"] = pos
            input_signal.at[idx, f"{surface_name}_command_saturated"] = 0
    return input_signal


def load_profile_events(logger_folder: Path) -> pd.DataFrame:
    workbook_path = logger_folder.parent / f"{strip_logger_suffix(logger_folder)}.xlsx"
    if not workbook_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_excel(workbook_path, sheet_name="ProfileEvents")
    except Exception:
        return pd.DataFrame()


def parse_sigrok_sample_rate(sample_rate_text: str, default_sample_rate_hz: float) -> float:
    token = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([kmg]?)\s*(?:hz)?\s*$", sample_rate_text, re.IGNORECASE)
    if not token:
        return float(default_sample_rate_hz)
    value = float(token.group(1))
    scale = token.group(2).lower()
    factor = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}.get(scale, 1.0)
    return value * factor


def parse_sigrok_raw_metadata(metadata_text: str, default_sample_rate_hz: float) -> tuple[float, list[str], int]:
    sample_rate_hz = float(default_sample_rate_hz)
    unit_size_bytes = 1
    probe_map: dict[int, str] = {}
    for raw_line in metadata_text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("samplerate="):
            sample_rate_hz = parse_sigrok_sample_rate(line.split("=", 1)[1], default_sample_rate_hz)
        elif line.lower().startswith("unitsize="):
            try:
                unit_size_bytes = int(round(float(line.split("=", 1)[1])))
            except ValueError:
                unit_size_bytes = 1
        else:
            probe_match = re.match(r"^probe(\d+)=(.*)$", line, re.IGNORECASE)
            if probe_match:
                probe_map[int(probe_match.group(1))] = probe_match.group(2).strip()
    if probe_map:
        probe_names = [""] * max(probe_map)
        for probe_index, probe_name in probe_map.items():
            probe_names[probe_index - 1] = probe_name
    else:
        probe_names = []
    return sample_rate_hz, probe_names, unit_size_bytes


def read_sigrok_logic_transitions(raw_capture_path: Path, logic_config: LogicAnalyzerConfig) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    if not raw_capture_path.is_file():
        raise FileNotFoundError(f"Missing sigrok raw capture: {raw_capture_path}")
    with zipfile.ZipFile(raw_capture_path) as archive:
        names = archive.namelist()
        if "metadata" not in names:
            raise ValueError(f"Sigrok capture is missing metadata: {raw_capture_path}")
        metadata_text = archive.read("metadata").decode("utf-8", errors="replace")
        sample_rate_hz, probe_names, unit_size_bytes = parse_sigrok_raw_metadata(metadata_text, logic_config.sample_rate_hz)
        if unit_size_bytes != 1:
            raise ValueError(f"Unsupported sigrok unitsize={unit_size_bytes} in {raw_capture_path}")

        logic_entries = []
        for name in names:
            file_name = Path(name).name
            match = re.match(r"^logic-\d+-(\d+)$", file_name)
            if match:
                logic_entries.append((int(match.group(1)), name))
        if not logic_entries:
            raise ValueError(f"No sigrok logic chunks found in {raw_capture_path}")
        logic_entries.sort(key=lambda item: item[0])

        sample_offset = 0
        last_byte: int | None = None
        change_sample_indices: list[np.ndarray] = []
        change_bytes: list[np.ndarray] = []
        for _, entry_name in logic_entries:
            chunk = np.frombuffer(archive.read(entry_name), dtype=np.uint8)
            if chunk.size == 0:
                continue
            change_mask = np.zeros(chunk.size, dtype=bool)
            if last_byte is None:
                change_mask[0] = True
            else:
                change_mask[0] = chunk[0] != last_byte
            if chunk.size > 1:
                change_mask[1:] = chunk[1:] != chunk[:-1]
            change_sample_indices.append((sample_offset + np.flatnonzero(change_mask)).astype(np.int64))
            change_bytes.append(chunk[change_mask])
            sample_offset += int(chunk.size)
            last_byte = int(chunk[-1])

    if not change_sample_indices:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint8), sample_rate_hz, probe_names
    return (
        np.concatenate(change_sample_indices),
        np.concatenate(change_bytes),
        sample_rate_hz,
        probe_names,
    )


def find_matching_probe_index(probe_names: list[str], logic_config: LogicAnalyzerConfig, channel_index: int) -> int:
    configured_name = logic_config.channel_names[channel_index]
    configured_number = str(int(logic_config.channels[channel_index]))
    candidates = [configured_name, f"D{configured_number}", configured_number]
    probe_lookup = {str(name): idx for idx, name in enumerate(probe_names)}
    for candidate in candidates:
        if candidate in probe_lookup:
            return probe_lookup[candidate]
    raise ValueError(f"Probe {configured_name} was not found in raw sigrok metadata")


def read_sigrok_raw_capture_as_logic_state(raw_capture_path: Path, logic_config: LogicAnalyzerConfig) -> dict[str, object]:
    change_sample_index, change_bytes, sample_rate_hz, probe_names = read_sigrok_logic_transitions(raw_capture_path, logic_config)
    channel_names = list(logic_config.channel_names)
    state_matrix = np.zeros((change_sample_index.size, len(channel_names)), dtype=float)
    for channel_index, _ in enumerate(channel_names):
        probe_index = find_matching_probe_index(probe_names, logic_config, channel_index)
        state_matrix[:, channel_index] = ((change_bytes >> probe_index) & 1).astype(float)
    return {
        "sample_index": change_sample_index.astype(np.int64),
        "sample_rate_hz": float(sample_rate_hz),
        "channel_names": channel_names,
        "state_matrix": state_matrix,
    }


def pair_edge_samples(rising_samples: np.ndarray, falling_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rising_samples.size == 0 or falling_samples.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pulse_start_samples: list[int] = []
    pulse_sample_counts: list[int] = []
    fall_index = 0
    for rise in rising_samples:
        while fall_index < falling_samples.size and falling_samples[fall_index] <= rise:
            fall_index += 1
        if fall_index >= falling_samples.size:
            break
        pulse_start_samples.append(int(rise))
        pulse_sample_counts.append(int(falling_samples[fall_index] - rise))
        fall_index += 1
    return np.asarray(pulse_start_samples, dtype=np.int64), np.asarray(pulse_sample_counts, dtype=np.int64)


def decode_pulse_polarity(
    sample_index: np.ndarray,
    channel_states: np.ndarray,
    sample_rate_hz: float,
    minimum_pulse_us: float,
    maximum_pulse_us: float,
    active_high: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transitions = np.diff(channel_states)
    if active_high:
        edge_start = sample_index[1:][transitions > 0]
        edge_stop = sample_index[1:][transitions < 0]
    else:
        edge_start = sample_index[1:][transitions < 0]
        edge_stop = sample_index[1:][transitions > 0]
    start_samples, sample_counts = pair_edge_samples(edge_start.astype(np.int64), edge_stop.astype(np.int64))
    pulse_us = 1e6 * sample_counts.astype(float) / float(sample_rate_hz)
    valid = np.isfinite(pulse_us) & (pulse_us >= minimum_pulse_us) & (pulse_us <= maximum_pulse_us)
    return start_samples[valid], sample_counts[valid], pulse_us[valid]


def decode_pulse_channel(
    sample_index: np.ndarray,
    channel_states: np.ndarray,
    sample_rate_hz: float,
    minimum_pulse_us: float,
    maximum_pulse_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_high, count_high, pulse_high = decode_pulse_polarity(
        sample_index,
        channel_states,
        sample_rate_hz,
        minimum_pulse_us,
        maximum_pulse_us,
        True,
    )
    start_low, count_low, pulse_low = decode_pulse_polarity(
        sample_index,
        channel_states,
        sample_rate_hz,
        minimum_pulse_us,
        maximum_pulse_us,
        False,
    )
    if start_low.size > start_high.size:
        return start_low, count_low, pulse_low
    return start_high, count_high, pulse_high


def apply_sample_debounce(edge_samples: np.ndarray, debounce_samples: int) -> np.ndarray:
    if edge_samples.size == 0:
        return edge_samples
    kept = [int(edge_samples[0])]
    for edge_sample in edge_samples[1:]:
        if int(edge_sample) - kept[-1] > debounce_samples:
            kept.append(int(edge_sample))
    return np.asarray(kept, dtype=np.int64)


def resolve_logic_state_channel_column_index(logic_state: dict[str, object], logic_config: LogicAnalyzerConfig, role_channel: int) -> int:
    try:
        config_index = logic_config.channels.index(role_channel)
    except ValueError as exc:
        raise ValueError(f"Configured logic channel {role_channel} was not found") from exc
    configured_name = logic_config.channel_names[config_index]
    channel_names = [str(name) for name in logic_state["channel_names"]]
    if configured_name not in channel_names:
        raise ValueError(f"Logic-state export is missing configured channel {configured_name}")
    return channel_names.index(configured_name)


def build_empty_pulse_capture_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["surface_name", "time_s", "pulse_us", "sample_index", "sample_count", "sample_rate_hz"])


def build_empty_reference_capture_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["time_s", "sample_index", "sample_rate_hz"])


def extract_output_capture(logic_state: dict[str, object], logic_config: LogicAnalyzerConfig, surface_names: list[str]) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    sample_index = np.asarray(logic_state["sample_index"], dtype=np.int64)
    state_matrix = np.asarray(logic_state["state_matrix"], dtype=float)
    sample_rate_hz = float(logic_state["sample_rate_hz"])
    for surface_idx, surface_name in enumerate(surface_names[: len(logic_config.output_channels)]):
        col = resolve_logic_state_channel_column_index(logic_state, logic_config, logic_config.output_channels[surface_idx])
        start_samples, sample_counts, pulse_us = decode_pulse_channel(
            sample_index,
            state_matrix[:, col],
            sample_rate_hz,
            logic_config.minimum_pulse_us,
            logic_config.maximum_pulse_us,
        )
        if start_samples.size == 0:
            continue
        block = pd.DataFrame({
            "surface_name": [surface_name] * start_samples.size,
            "time_s": start_samples.astype(float) / sample_rate_hz,
            "pulse_us": pulse_us.astype(float),
            "sample_index": start_samples.astype(float),
            "sample_count": sample_counts.astype(float),
            "sample_rate_hz": np.full(start_samples.size, sample_rate_hz, dtype=float),
        })
        blocks.append(block)
    if not blocks:
        return build_empty_pulse_capture_table()
    out = pd.concat(blocks, ignore_index=True)
    return out.sort_values(["surface_name", "sample_index"], kind="stable").reset_index(drop=True)


def extract_reference_capture(logic_state: dict[str, object], logic_config: LogicAnalyzerConfig) -> pd.DataFrame:
    if logic_config.reference_channel is None:
        return build_empty_reference_capture_table()
    col = resolve_logic_state_channel_column_index(logic_state, logic_config, logic_config.reference_channel)
    sample_index = np.asarray(logic_state["sample_index"], dtype=np.int64)
    states = np.asarray(logic_state["state_matrix"], dtype=float)[:, col]
    edge_samples = sample_index[1:][np.diff(states) > 0]
    debounce_samples = int(round(logic_config.reference_debounce_us * float(logic_state["sample_rate_hz"]) / 1e6))
    edge_samples = apply_sample_debounce(edge_samples, debounce_samples)
    return pd.DataFrame({
        "time_s": edge_samples.astype(float) / float(logic_state["sample_rate_hz"]),
        "sample_index": edge_samples.astype(float),
        "sample_rate_hz": np.full(edge_samples.size, float(logic_state["sample_rate_hz"]), dtype=float),
    })


def load_existing_alignment_metadata(event_path: Path) -> AlignmentMetadata:
    if not event_path.is_file():
        return AlignmentMetadata()
    events = pd.read_csv(event_path, usecols=lambda column: column in {"output_time_offset_s", "output_time_drift_slope", "output_time_drift_baseline_s"})
    if events.empty:
        return AlignmentMetadata()

    def first_finite(column: str) -> float:
        if column not in events.columns:
            return 0.0
        values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(values[0]) if values.size else 0.0

    return AlignmentMetadata(
        output_time_offset_s=first_finite("output_time_offset_s"),
        output_time_drift_slope=first_finite("output_time_drift_slope"),
        output_time_drift_baseline_s=first_finite("output_time_drift_baseline_s"),
    )


def unapply_alignment(frame: pd.DataFrame, time_column: str, alignment: AlignmentMetadata) -> pd.DataFrame:
    if frame.empty or time_column not in frame.columns:
        return frame
    out = frame.copy()
    time_values = pd.to_numeric(out[time_column], errors="coerce").to_numpy(dtype=float)
    slope = float(alignment.output_time_drift_slope)
    baseline = float(alignment.output_time_drift_baseline_s)
    offset = float(alignment.output_time_offset_s)
    if np.isfinite(slope) and np.isfinite(baseline) and abs(1.0 - slope) > 1e-12:
        time_values = (time_values + baseline) / (1.0 - slope)
    if np.isfinite(offset):
        time_values = time_values + offset
    out[time_column] = time_values
    return out


def normalize_output_capture_table(raw_table: pd.DataFrame, logic_config: LogicAnalyzerConfig, surface_names: list[str]) -> pd.DataFrame:
    if raw_table.empty:
        return build_empty_pulse_capture_table()
    variable_names = list(raw_table.columns)
    canonical_names = [re.sub(r"[^a-zA-Z0-9]", "", str(name)).lower() for name in variable_names]

    time_idx = next((idx for idx, name in enumerate(canonical_names) if "time" in name), None)
    pulse_idx = next((idx for idx, name in enumerate(canonical_names) if name in {"pulseus", "pulsewidthus", "widthus"}), None)
    surface_idx = next((idx for idx, name in enumerate(canonical_names) if name in {"surfacename", "channelname"}), None)
    if time_idx is None or pulse_idx is None or surface_idx is None:
        return build_empty_pulse_capture_table()

    time_s = normalize_time_column(convert_column_to_numeric(raw_table.iloc[:, time_idx]), logic_config.sample_rate_hz)
    pulse_us = convert_column_to_numeric(raw_table.iloc[:, pulse_idx])
    surface_name = raw_table.iloc[:, surface_idx].astype(str).to_numpy()

    valid = (
        np.isfinite(time_s)
        & np.isfinite(pulse_us)
        & (pulse_us >= logic_config.minimum_pulse_us)
        & (pulse_us <= logic_config.maximum_pulse_us)
        & np.isin(surface_name, np.asarray(surface_names, dtype=object))
    )
    if not np.any(valid):
        return build_empty_pulse_capture_table()
    out = pd.DataFrame({
        "surface_name": surface_name[valid],
        "time_s": time_s[valid],
        "pulse_us": pulse_us[valid],
        "sample_index": np.nan,
        "sample_count": np.nan,
        "sample_rate_hz": np.nan,
    })
    return out.sort_values(["surface_name", "time_s"], kind="stable").reset_index(drop=True)


def normalize_reference_capture_table(raw_table: pd.DataFrame, debounce_us: float) -> pd.DataFrame:
    if raw_table.empty:
        return build_empty_reference_capture_table()
    variable_names = list(raw_table.columns)
    canonical_names = [re.sub(r"[^a-zA-Z0-9]", "", str(name)).lower() for name in variable_names]
    time_idx = next((idx for idx, name in enumerate(canonical_names) if "time" in name), None)
    if time_idx is None:
        return build_empty_reference_capture_table()
    time_s = normalize_time_column(convert_column_to_numeric(raw_table.iloc[:, time_idx]), 1.0)
    valid = np.isfinite(time_s)
    if not np.any(valid):
        return build_empty_reference_capture_table()
    out = pd.DataFrame({
        "time_s": time_s[valid],
        "sample_index": np.nan,
        "sample_rate_hz": np.nan,
    }).sort_values("time_s", kind="stable")
    if len(out) >= 2:
        keep = np.concatenate([[True], np.diff(out["time_s"].to_numpy(dtype=float)) > debounce_us / 1e6])
        out = out.loc[keep].copy()
    return out.reset_index(drop=True)


def import_analyzer_capture(logger_folder: Path, surface_names: list[str], logic_config: LogicAnalyzerConfig, output_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_label = strip_logger_suffix(logger_folder)
    raw_capture_path = logger_folder / f"{run_label}_sigrok_raw.sr"
    if raw_capture_path.is_file():
        logic_state = read_sigrok_raw_capture_as_logic_state(raw_capture_path, logic_config)
        reference_capture = extract_reference_capture(logic_state, logic_config)
        output_capture = extract_output_capture(logic_state, logic_config, surface_names)
        if output_capture.empty:
            raise ValueError(f"No valid PWM pulses were decoded from {raw_capture_path}")
        return reference_capture, output_capture

    output_capture_path = logger_folder / "output_capture.csv"
    reference_capture_path = logger_folder / "reference_capture.csv"
    output_capture = normalize_output_capture_table(read_csv(output_capture_path, required=False), logic_config, surface_names)
    reference_capture = normalize_reference_capture_table(read_csv(reference_capture_path, required=False), logic_config.reference_debounce_us)
    if not output_capture.empty:
        alignment = load_existing_alignment_metadata(logger_folder / f"{output_prefix}_events.csv")
        output_capture = unapply_alignment(output_capture, "time_s", alignment)
        reference_capture = unapply_alignment(reference_capture, "time_s", alignment)
        return reference_capture, output_capture
    raise FileNotFoundError(f"No analyser capture artifact found in {logger_folder}")


def load_arduino_logs(logger_folder: Path, sample_time_seconds: float, logic_config: LogicAnalyzerConfig, output_prefix: str) -> dict[str, object]:
    host_dispatch_log = read_csv(logger_folder / "host_dispatch_log.csv")
    board_command_log = read_csv(logger_folder / "board_command_log.csv")
    host_sync_roundtrip = read_csv(logger_folder / "host_sync_roundtrip.csv")

    host_dispatch_log["surface_name"] = host_dispatch_log["surface_name"].astype(str)
    board_command_log["surface_name"] = board_command_log["surface_name"].astype(str)
    surface_names = [str(name) for name in pd.unique(host_dispatch_log["surface_name"])]
    input_signal = build_input_signal(host_dispatch_log, surface_names, sample_time_seconds)
    profile_events = load_profile_events(logger_folder)
    reference_capture, output_capture = import_analyzer_capture(logger_folder, surface_names, logic_config, output_prefix)
    return {
        "hostDispatchLog": host_dispatch_log,
        "boardCommandLog": board_command_log,
        "hostSyncRoundTrip": host_sync_roundtrip,
        "surfaceNames": surface_names,
        "inputSignal": input_signal,
        "profileEvents": profile_events,
        "referenceCapture": reference_capture,
        "outputCapture": output_capture,
    }


def latency_stats(values: Iterable[float]) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
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
        "SampleCount": int(series.size),
        "Mean_s": float(np.mean(series)),
        "Std_s": float(np.std(series, ddof=0)) if series.size >= 2 else np.nan,
        "Median_s": float(np.median(series)),
        "P95_s": float(np.percentile(series, 95)),
        "P99_s": float(np.percentile(series, 99)),
        "Max_s": float(np.max(series)),
    }


def estimate_clock_map(sync_roundtrip_log: pd.DataFrame, joined: pd.DataFrame, clock_map_mode: str) -> tuple[float, float]:
    host_tx_us = pd.to_numeric(sync_roundtrip_log.get("host_tx_us"), errors="coerce").to_numpy(dtype=float)
    host_rx_us = pd.to_numeric(sync_roundtrip_log.get("host_rx_us"), errors="coerce").to_numpy(dtype=float)
    board_rx_us = pd.to_numeric(sync_roundtrip_log.get("board_rx_us"), errors="coerce").to_numpy(dtype=float)
    board_tx_us = pd.to_numeric(sync_roundtrip_log.get("board_tx_us"), errors="coerce").to_numpy(dtype=float)
    use_sync = str(clock_map_mode).lower() == "sync"

    if use_sync:
        mask = np.isfinite(host_tx_us) & np.isfinite(host_rx_us) & np.isfinite(board_rx_us) & np.isfinite(board_tx_us)
        if np.count_nonzero(mask) >= 2:
            coeffs = np.polyfit(0.5 * (board_rx_us[mask] + board_tx_us[mask]), 0.5 * (host_tx_us[mask] + host_rx_us[mask]), 1)
            clock_slope = float(coeffs[0])
            clock_intercept = float(coeffs[1])
        else:
            use_sync = False

    if not use_sync:
        joined_rx_us = pd.to_numeric(joined["rx_us"], errors="coerce").to_numpy(dtype=float)
        dispatch_us = pd.to_numeric(joined["command_dispatch_us"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(joined_rx_us) & np.isfinite(dispatch_us)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Unable to estimate the board-to-host clock map from the available timestamps")
        coeffs = np.polyfit(joined_rx_us[mask], dispatch_us[mask], 1)
        round_trip = host_rx_us - host_tx_us
        round_trip = round_trip[np.isfinite(round_trip) & (round_trip >= 0)]
        one_way_us = 0.5 * float(np.nanmedian(round_trip)) if round_trip.size else 0.0
        clock_slope = float(coeffs[0])
        clock_intercept = float(coeffs[1] + one_way_us)

    rx_host_us = clock_slope * pd.to_numeric(joined["rx_us"], errors="coerce").to_numpy(dtype=float) + clock_intercept
    apply_host_us = clock_slope * pd.to_numeric(joined["apply_us"], errors="coerce").to_numpy(dtype=float) + clock_intercept
    dispatch_us = pd.to_numeric(joined["command_dispatch_us"], errors="coerce").to_numpy(dtype=float)
    all_latencies = np.concatenate([rx_host_us - dispatch_us, apply_host_us - dispatch_us])
    all_latencies = all_latencies[np.isfinite(all_latencies)]
    if all_latencies.size:
        minimum_latency_us = float(np.min(all_latencies))
        if minimum_latency_us < 0:
            clock_intercept -= minimum_latency_us
    return clock_slope, clock_intercept


def match_reference_times(anchor_times: np.ndarray, reference_times: np.ndarray, window_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    ref_time = np.full(anchor_times.shape, np.nan, dtype=float)
    ref_index = np.full(anchor_times.shape, np.nan, dtype=float)
    if reference_times.size == 0:
        return ref_time, ref_index
    next_index = 0
    for idx, anchor_time in enumerate(anchor_times):
        if not np.isfinite(anchor_time):
            continue
        while next_index < reference_times.size and reference_times[next_index] < anchor_time - window_seconds:
            next_index += 1
        candidates = [candidate for candidate in (next_index - 1, next_index) if 0 <= candidate < reference_times.size]
        if not candidates:
            continue
        distances = [abs(reference_times[candidate] - anchor_time) for candidate in candidates]
        best_position = int(np.argmin(distances))
        if distances[best_position] <= window_seconds:
            ref_time[idx] = float(reference_times[candidates[best_position]])
            ref_index[idx] = float(candidates[best_position])
    return ref_time, ref_index


def build_output_transition_table(surface_capture: pd.DataFrame, threshold_us: float) -> pd.DataFrame:
    if surface_capture.empty or len(surface_capture) < 2:
        return pd.DataFrame(columns=["time_s", "previous_pulse_us", "pulse_us"])
    ordered = surface_capture.sort_values("time_s", kind="stable").copy()
    pulse = pd.to_numeric(ordered["pulse_us"], errors="coerce").to_numpy(dtype=float)
    time_s = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(dtype=float)
    previous_pulse = pulse[:-1]
    current_pulse = pulse[1:]
    valid = np.isfinite(previous_pulse) & np.isfinite(current_pulse) & (np.abs(current_pulse - previous_pulse) >= threshold_us)
    return pd.DataFrame({
        "time_s": time_s[1:][valid],
        "previous_pulse_us": previous_pulse[valid],
        "pulse_us": current_pulse[valid],
    })


def trim_transition_table_for_alignment(transition_table: pd.DataFrame) -> pd.DataFrame:
    if transition_table.empty or len(transition_table) < 3:
        return transition_table
    time_values = pd.to_numeric(transition_table["time_s"], errors="coerce").to_numpy(dtype=float)
    start_index = 0
    for idx in range(len(time_values) - 1):
        if np.isfinite(time_values[idx]) and np.isfinite(time_values[idx + 1]) and (time_values[idx + 1] - time_values[idx]) <= 1.0:
            start_index = idx
            break
    if start_index > 0:
        return transition_table.iloc[start_index:].reset_index(drop=True)
    return transition_table.reset_index(drop=True)


def estimate_output_time_offset(joined: pd.DataFrame, output_capture: pd.DataFrame, surface_names: list[str], transition_pulse_threshold_us: float) -> tuple[float, pd.DataFrame]:
    rows: list[dict] = []
    offset_samples: list[np.ndarray] = []
    for surface_name in surface_names:
        surface_rows = joined.index[(joined["surface_name"].astype(str) == surface_name) & (joined["is_transition_command"] == 1)].to_numpy()
        if surface_rows.size == 0:
            continue
        surface_capture = output_capture.loc[output_capture["surface_name"].astype(str) == surface_name].copy()
        transition_table = trim_transition_table_for_alignment(build_output_transition_table(surface_capture, transition_pulse_threshold_us))
        sample_count = int(min(surface_rows.size, len(transition_table)))
        if sample_count < 3:
            continue
        surface_offsets = (
            pd.to_numeric(transition_table["time_s"].iloc[:sample_count], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(joined.loc[surface_rows[:sample_count], "board_apply_s"], errors="coerce").to_numpy(dtype=float)
        )
        surface_offsets = surface_offsets[np.isfinite(surface_offsets)]
        if surface_offsets.size == 0:
            continue
        estimated_offset = float(np.percentile(surface_offsets, 1.0))
        rows.append({"SurfaceName": surface_name, "SampleCount": sample_count, "MedianOffset_s": estimated_offset})
        offset_samples.append(surface_offsets)

    surface_offsets = pd.DataFrame(rows, columns=["SurfaceName", "SampleCount", "MedianOffset_s"])
    all_offsets = np.concatenate(offset_samples) if offset_samples else np.zeros(0, dtype=float)
    surface_medians = pd.to_numeric(surface_offsets.get("MedianOffset_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    surface_medians = surface_medians[np.isfinite(surface_medians)]
    if surface_medians.size:
        return float(np.nanmedian(surface_medians)), surface_offsets
    if all_offsets.size:
        return float(np.percentile(all_offsets, 1.0)), surface_offsets
    return 0.0, surface_offsets


def estimate_output_pulse_bias(joined: pd.DataFrame, output_capture: pd.DataFrame, surface_names: list[str], transition_pulse_threshold_us: float) -> tuple[float, pd.DataFrame]:
    rows: list[dict] = []
    bias_samples: list[np.ndarray] = []
    for surface_name in surface_names:
        surface_rows = joined.index[(joined["surface_name"].astype(str) == surface_name) & (joined["is_transition_command"] == 1)].to_numpy()
        if surface_rows.size == 0:
            continue
        surface_capture = output_capture.loc[output_capture["surface_name"].astype(str) == surface_name].copy()
        transition_table = trim_transition_table_for_alignment(build_output_transition_table(surface_capture, transition_pulse_threshold_us))
        sample_count = int(min(surface_rows.size, len(transition_table)))
        if sample_count < 3:
            continue
        expected_bias = (
            pd.to_numeric(transition_table["pulse_us"].iloc[:sample_count], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(joined.loc[surface_rows[:sample_count], "expected_pulse_us"], errors="coerce").to_numpy(dtype=float)
        )
        previous_bias = (
            pd.to_numeric(transition_table["previous_pulse_us"].iloc[:sample_count], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(joined.loc[surface_rows[:sample_count], "previous_expected_pulse_us"], errors="coerce").to_numpy(dtype=float)
        )
        surface_bias = np.concatenate([expected_bias, previous_bias])
        surface_bias = surface_bias[np.isfinite(surface_bias)]
        if surface_bias.size == 0:
            continue
        median_bias = float(np.median(surface_bias))
        rows.append({"SurfaceName": surface_name, "SampleCount": sample_count, "MedianBias_us": median_bias})
        bias_samples.append(surface_bias)

    surface_biases = pd.DataFrame(rows, columns=["SurfaceName", "SampleCount", "MedianBias_us"])
    all_biases = np.concatenate(bias_samples) if bias_samples else np.zeros(0, dtype=float)
    surface_medians = pd.to_numeric(surface_biases.get("MedianBias_us", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    surface_medians = surface_medians[np.isfinite(surface_medians)]
    if surface_medians.size:
        return float(np.nanmedian(surface_medians)), surface_biases
    if all_biases.size:
        return float(np.median(all_biases)), surface_biases
    return 0.0, surface_biases


def estimate_output_time_drift(joined: pd.DataFrame, output_capture: pd.DataFrame, surface_names: list[str], transition_pulse_threshold_us: float) -> tuple[float, float]:
    x_samples: list[np.ndarray] = []
    dt_samples: list[np.ndarray] = []
    minimum_sample_count = 20
    for surface_name in surface_names:
        surface_rows = joined.index[(joined["surface_name"].astype(str) == surface_name) & (joined["is_transition_command"] == 1)].to_numpy()
        if surface_rows.size == 0:
            continue
        surface_capture = output_capture.loc[output_capture["surface_name"].astype(str) == surface_name].copy()
        transition_table = trim_transition_table_for_alignment(build_output_transition_table(surface_capture, transition_pulse_threshold_us))
        sample_count = int(min(surface_rows.size, len(transition_table)))
        if sample_count < 3:
            continue
        x_local = pd.to_numeric(transition_table["time_s"].iloc[:sample_count], errors="coerce").to_numpy(dtype=float)
        dt_local = x_local - pd.to_numeric(joined.loc[surface_rows[:sample_count], "board_apply_s"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_local) & np.isfinite(dt_local)
        if np.any(valid):
            x_samples.append(x_local[valid])
            dt_samples.append(dt_local[valid])

    if not x_samples:
        return 0.0, 0.0
    x = np.concatenate(x_samples)
    dt = np.concatenate(dt_samples)
    if x.size < minimum_sample_count:
        return 0.0, 0.0

    coeffs = np.polyfit(x, dt, 1)
    drift_slope = float(coeffs[0])
    residual = dt - np.polyval(coeffs, x)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return 0.0, 0.0

    if residual.size >= 8:
        residual_median = float(np.nanmedian(residual))
        residual_mad = float(np.nanmedian(np.abs(residual - residual_median)))
        if np.isfinite(residual_mad) and residual_mad > 0:
            robust_mask = np.abs(residual - residual_median) <= max(3.5 * residual_mad, 0.005)
            if np.count_nonzero(robust_mask) >= minimum_sample_count:
                x = x[robust_mask]
                dt = dt[robust_mask]
                coeffs = np.polyfit(x, dt, 1)
                drift_slope = float(coeffs[0])
                residual = dt - np.polyval(coeffs, x)
                residual = residual[np.isfinite(residual)]

    if residual.size == 0:
        return 0.0, 0.0

    drift_span_seconds = abs(drift_slope) * max(0.0, float(np.max(x) - np.min(x)))
    residual_p95_seconds = float(np.percentile(np.abs(residual), 95))
    if abs(drift_slope) > 0.002 or drift_span_seconds > 0.10 or residual_p95_seconds > 0.02:
        return 0.0, 0.0
    return drift_slope, float(np.percentile(residual, 1.0))


def is_candidate_score_better(candidate_score: list[float], best_score: list[float] | None) -> bool:
    if best_score is None:
        return True
    for candidate_value, best_value in zip(candidate_score, best_score):
        if candidate_value < best_value:
            return True
        if candidate_value > best_value:
            return False
    return len(candidate_score) < len(best_score)


def find_next_output_transition(
    transition_table: pd.DataFrame,
    search_index: int,
    anchor_time: float,
    previous_pulse_us: float,
    target_pulse_us: float,
    previous_tolerance_us: float,
    target_tolerance_us: float,
    max_window_seconds: float,
    pre_anchor_slack_seconds: float,
) -> tuple[float, int]:
    if transition_table.empty or not np.isfinite(anchor_time):
        return np.nan, search_index
    time_values = pd.to_numeric(transition_table["time_s"], errors="coerce").to_numpy(dtype=float)
    previous_values = pd.to_numeric(transition_table["previous_pulse_us"], errors="coerce").to_numpy(dtype=float)
    pulse_values = pd.to_numeric(transition_table["pulse_us"], errors="coerce").to_numpy(dtype=float)
    search_window_start = anchor_time - max(0.0, pre_anchor_slack_seconds if np.isfinite(pre_anchor_slack_seconds) else 0.0)
    while search_index < len(transition_table) and time_values[search_index] < search_window_start:
        search_index += 1
    window_end = anchor_time + max_window_seconds
    best_index = np.nan
    best_score: list[float] | None = None
    for idx in range(search_index, len(transition_table)):
        t = float(time_values[idx])
        if t > window_end:
            break
        target_error_us = abs(float(pulse_values[idx]) - target_pulse_us)
        if target_error_us > target_tolerance_us:
            continue
        previous_error_us = 0.0
        previous_matches_old = True
        if np.isfinite(previous_pulse_us):
            previous_error_us = abs(float(previous_values[idx]) - previous_pulse_us)
            if np.isfinite(previous_tolerance_us):
                previous_matches_old = previous_error_us <= previous_tolerance_us
        latency_seconds = t - anchor_time
        candidate_score = [
            float(latency_seconds < 0),
            float(not previous_matches_old),
            float(target_error_us),
            float(previous_error_us),
            float(max(0.0, latency_seconds)),
            float(abs(latency_seconds)),
            float(t),
        ]
        if np.isnan(best_index) or is_candidate_score_better(candidate_score, best_score):
            best_index = float(idx)
            best_score = candidate_score
    if np.isfinite(best_index):
        return best_index, int(best_index) + 1
    return np.nan, search_index


def compute_arduino_e2e(
    logs: dict[str, object],
    clock_map_mode: str,
    matching_mode: str,
    max_output_association_seconds: float,
    reference_association_window_seconds: float,
    transition_pulse_threshold_us: float,
    target_pulse_tolerance_us: float,
    previous_pulse_tolerance_us: float,
    pre_anchor_slack_seconds: float,
    maximum_apply_to_output_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dispatch = logs["hostDispatchLog"].sort_values(["command_sequence", "surface_name"], kind="stable").reset_index(drop=True).copy()
    board = logs["boardCommandLog"].sort_values(["command_sequence", "surface_name"], kind="stable").reset_index(drop=True).copy()
    output_capture = logs["outputCapture"].copy()
    reference_capture = logs["referenceCapture"].copy()
    surface_names = list(logs["surfaceNames"])
    analyzer_alignment: dict[str, object] = {
        "outputTimeOffsetSeconds": 0.0,
        "surfaceOffsetsSeconds": pd.DataFrame(columns=["SurfaceName", "SampleCount", "MedianOffset_s"]),
        "outputTimeDriftSlope": 0.0,
        "outputTimeDriftBaselineSeconds": 0.0,
        "outputPulseBiasUs": 0.0,
        "surfacePulseBiasUs": pd.DataFrame(columns=["SurfaceName", "SampleCount", "MedianBias_us"]),
        "isApplied": False,
    }

    if "receive_to_apply_us" not in board.columns:
        board["receive_to_apply_us"] = pd.to_numeric(board["apply_us"], errors="coerce") - pd.to_numeric(board["rx_us"], errors="coerce")

    joined = dispatch.merge(board, on=["surface_name", "command_sequence"], how="left", sort=False)
    joined["sample_index"] = pd.to_numeric(joined["command_sequence"], errors="coerce")
    origin_us = float(pd.to_numeric(joined["command_dispatch_us"], errors="coerce").min())
    joined["command_dispatch_s"] = (pd.to_numeric(joined["command_dispatch_us"], errors="coerce") - origin_us) / 1e6

    input_signal = logs["inputSignal"]
    if not input_signal.empty and {"command_sequence", "scheduled_time_s"}.issubset(input_signal.columns):
        schedule_lookup = input_signal.set_index("command_sequence")["scheduled_time_s"]
        joined["scheduled_time_s"] = pd.to_numeric(joined["command_sequence"], errors="coerce").map(schedule_lookup)
        if "command_dispatch_s" in input_signal.columns:
            dispatch_lookup = input_signal.set_index("command_sequence")["command_dispatch_s"]
            logged_dispatch = pd.to_numeric(joined["command_sequence"], errors="coerce").map(dispatch_lookup)
            replace_mask = np.isfinite(pd.to_numeric(logged_dispatch, errors="coerce"))
            joined.loc[replace_mask, "command_dispatch_s"] = logged_dispatch.loc[replace_mask]
    if "scheduled_time_s" not in joined.columns or not np.isfinite(pd.to_numeric(joined["scheduled_time_s"], errors="coerce")).any():
        if len(input_signal) >= 2 and "scheduled_time_s" in input_signal.columns:
            delta = float(pd.to_numeric(input_signal["scheduled_time_s"], errors="coerce").iloc[1] - pd.to_numeric(input_signal["scheduled_time_s"], errors="coerce").iloc[0])
        else:
            delta = DEFAULT_SAMPLE_TIME_SECONDS
        joined["scheduled_time_s"] = (pd.to_numeric(joined["command_sequence"], errors="coerce") - float(pd.to_numeric(joined["command_sequence"], errors="coerce").min())) * delta

    joined["host_scheduling_delay_s"] = pd.to_numeric(joined["command_dispatch_s"], errors="coerce") - pd.to_numeric(joined["scheduled_time_s"], errors="coerce")
    clock_slope, clock_intercept = estimate_clock_map(logs["hostSyncRoundTrip"], joined, clock_map_mode)
    joined["board_rx_s"] = (clock_slope * pd.to_numeric(joined["rx_us"], errors="coerce") + clock_intercept - origin_us) / 1e6
    joined["board_apply_s"] = (clock_slope * pd.to_numeric(joined["apply_us"], errors="coerce") + clock_intercept - origin_us) / 1e6
    joined["dispatch_to_rx_latency_s"] = pd.to_numeric(joined["board_rx_s"], errors="coerce") - pd.to_numeric(joined["command_dispatch_s"], errors="coerce")
    joined["dispatch_to_apply_latency_s"] = pd.to_numeric(joined["board_apply_s"], errors="coerce") - pd.to_numeric(joined["command_dispatch_s"], errors="coerce")
    joined["scheduled_to_rx_latency_s"] = pd.to_numeric(joined["board_rx_s"], errors="coerce") - pd.to_numeric(joined["scheduled_time_s"], errors="coerce")
    joined["scheduled_to_apply_latency_s"] = pd.to_numeric(joined["board_apply_s"], errors="coerce") - pd.to_numeric(joined["scheduled_time_s"], errors="coerce")
    joined["rx_to_apply_latency_s"] = pd.to_numeric(joined["receive_to_apply_us"], errors="coerce") / 1e6
    joined["expected_pulse_us"] = pd.to_numeric(joined["pulse_us"], errors="coerce")
    joined["reference_time_s"] = np.nan
    joined["anchor_time_s"] = pd.to_numeric(joined["board_apply_s"], errors="coerce")
    joined["anchor_source"] = "apply"

    if not reference_capture.empty:
        ref_time, _ = match_reference_times(
            pd.to_numeric(joined["board_apply_s"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(reference_capture["time_s"], errors="coerce").to_numpy(dtype=float),
            reference_association_window_seconds,
        )
        joined["reference_time_s"] = ref_time
        if str(matching_mode).lower() == "shared_clock":
            mask = np.isfinite(ref_time)
            joined.loc[mask, "anchor_time_s"] = ref_time[mask]
            joined.loc[mask, "anchor_source"] = "reference"

    joined["previous_expected_pulse_us"] = np.nan
    joined["is_transition_command"] = 0
    for surface_name in surface_names:
        surface_mask = joined["surface_name"].astype(str) == surface_name
        rows = joined.index[surface_mask].to_numpy()
        if rows.size < 2:
            continue
        prev_pulse = pd.to_numeric(joined.loc[rows[:-1], "expected_pulse_us"], errors="coerce").to_numpy(dtype=float)
        curr_pulse = pd.to_numeric(joined.loc[rows[1:], "expected_pulse_us"], errors="coerce").to_numpy(dtype=float)
        transition_mask = np.isfinite(prev_pulse) & np.isfinite(curr_pulse) & (np.abs(curr_pulse - prev_pulse) >= transition_pulse_threshold_us)
        joined.loc[rows[1:], "previous_expected_pulse_us"] = prev_pulse
        joined.loc[rows[1:], "is_transition_command"] = transition_mask.astype(int)

    output_time_offset_seconds, surface_offsets = estimate_output_time_offset(joined, output_capture, surface_names, transition_pulse_threshold_us)
    analyzer_alignment["outputTimeOffsetSeconds"] = output_time_offset_seconds
    analyzer_alignment["surfaceOffsetsSeconds"] = surface_offsets
    if np.isfinite(output_time_offset_seconds) and abs(output_time_offset_seconds) > 1e-6:
        output_capture["time_s"] = pd.to_numeric(output_capture["time_s"], errors="coerce") - output_time_offset_seconds
        if not reference_capture.empty:
            reference_capture["time_s"] = pd.to_numeric(reference_capture["time_s"], errors="coerce") - output_time_offset_seconds
        analyzer_alignment["isApplied"] = True

    output_time_drift_slope, output_time_drift_baseline_seconds = estimate_output_time_drift(joined, output_capture, surface_names, transition_pulse_threshold_us)
    analyzer_alignment["outputTimeDriftSlope"] = output_time_drift_slope
    analyzer_alignment["outputTimeDriftBaselineSeconds"] = output_time_drift_baseline_seconds
    if np.isfinite(output_time_drift_slope) and np.isfinite(output_time_drift_baseline_seconds):
        output_capture["time_s"] = pd.to_numeric(output_capture["time_s"], errors="coerce") - (
            output_time_drift_slope * pd.to_numeric(output_capture["time_s"], errors="coerce") + output_time_drift_baseline_seconds
        )
        if not reference_capture.empty:
            reference_capture["time_s"] = pd.to_numeric(reference_capture["time_s"], errors="coerce") - (
                output_time_drift_slope * pd.to_numeric(reference_capture["time_s"], errors="coerce") + output_time_drift_baseline_seconds
            )
        analyzer_alignment["isApplied"] = True

    output_pulse_bias_us, surface_pulse_biases = estimate_output_pulse_bias(joined, output_capture, surface_names, transition_pulse_threshold_us)
    analyzer_alignment["outputPulseBiasUs"] = output_pulse_bias_us
    analyzer_alignment["surfacePulseBiasUs"] = surface_pulse_biases

    joined["output_time_s"] = np.nan
    joined["output_pulse_us"] = np.nan
    joined["matched_output_transition"] = 0
    joined["apply_to_output_latency_s"] = np.nan
    joined["dispatch_to_output_latency_s"] = np.nan
    joined["scheduled_to_output_latency_s"] = np.nan
    joined["is_valid_e2e"] = 0
    joined["non_realistic_reason"] = ""

    for surface_name in surface_names:
        surface_rows = joined.index[(joined["surface_name"].astype(str) == surface_name) & (joined["is_transition_command"] == 1)].to_numpy()
        surface_capture = output_capture.loc[output_capture["surface_name"].astype(str) == surface_name].copy()
        transition_table = build_output_transition_table(surface_capture, transition_pulse_threshold_us)
        surface_bias_us = float(output_pulse_bias_us)
        if not surface_pulse_biases.empty:
            bias_row = surface_pulse_biases.loc[surface_pulse_biases["SurfaceName"].astype(str) == surface_name]
            if not bias_row.empty and np.isfinite(pd.to_numeric(bias_row["MedianBias_us"], errors="coerce").iloc[0]):
                surface_bias_us = float(pd.to_numeric(bias_row["MedianBias_us"], errors="coerce").iloc[0])
        if np.isfinite(surface_bias_us) and abs(surface_bias_us) > 1e-9 and not transition_table.empty:
            transition_table["previous_pulse_us"] = pd.to_numeric(transition_table["previous_pulse_us"], errors="coerce") - surface_bias_us
            transition_table["pulse_us"] = pd.to_numeric(transition_table["pulse_us"], errors="coerce") - surface_bias_us

        search_index = 0
        for row_index in surface_rows:
            match_index, search_index = find_next_output_transition(
                transition_table,
                search_index,
                float(pd.to_numeric(pd.Series([joined.at[row_index, "anchor_time_s"]]), errors="coerce").iloc[0]),
                float(pd.to_numeric(pd.Series([joined.at[row_index, "previous_expected_pulse_us"]]), errors="coerce").iloc[0]),
                float(pd.to_numeric(pd.Series([joined.at[row_index, "expected_pulse_us"]]), errors="coerce").iloc[0]),
                previous_pulse_tolerance_us,
                target_pulse_tolerance_us,
                max_output_association_seconds,
                pre_anchor_slack_seconds,
            )
            if not np.isfinite(match_index):
                joined.at[row_index, "non_realistic_reason"] = "unmatched_output_transition"
                continue

            matched = transition_table.iloc[int(match_index)]
            joined.at[row_index, "output_time_s"] = float(pd.to_numeric(pd.Series([matched["time_s"]]), errors="coerce").iloc[0])
            joined.at[row_index, "output_pulse_us"] = float(pd.to_numeric(pd.Series([matched["pulse_us"]]), errors="coerce").iloc[0])
            joined.at[row_index, "matched_output_transition"] = 1

            apply_to_output_latency = float(pd.to_numeric(pd.Series([joined.at[row_index, "output_time_s"]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([joined.at[row_index, "board_apply_s"]]), errors="coerce").iloc[0])
            if np.isfinite(apply_to_output_latency) and (-5e-4 < apply_to_output_latency < 0):
                apply_to_output_latency = 0.0
            joined.at[row_index, "apply_to_output_latency_s"] = apply_to_output_latency
            joined.at[row_index, "dispatch_to_output_latency_s"] = float(pd.to_numeric(pd.Series([joined.at[row_index, "output_time_s"]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([joined.at[row_index, "command_dispatch_s"]]), errors="coerce").iloc[0])
            joined.at[row_index, "scheduled_to_output_latency_s"] = float(pd.to_numeric(pd.Series([joined.at[row_index, "output_time_s"]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([joined.at[row_index, "scheduled_time_s"]]), errors="coerce").iloc[0])

            is_valid = int(
                np.isfinite(apply_to_output_latency)
                and apply_to_output_latency >= -1e-6
                and apply_to_output_latency <= maximum_apply_to_output_seconds
            )
            joined.at[row_index, "is_valid_e2e"] = is_valid
            joined.at[row_index, "non_realistic_reason"] = "ok" if is_valid else "apply_to_output_out_of_range"

    events = joined.loc[joined["is_transition_command"] == 1].copy().reset_index(drop=True)
    events["output_time_offset_s"] = float(output_time_offset_seconds)
    events["output_time_drift_slope"] = float(output_time_drift_slope)
    events["output_time_drift_baseline_s"] = float(output_time_drift_baseline_seconds)
    events["output_pulse_bias_us"] = float(output_pulse_bias_us)
    return events, output_capture.reset_index(drop=True), reference_capture.reset_index(drop=True), analyzer_alignment


def build_summaries(events: pd.DataFrame, surface_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_map = [
        ("HostSchedulingDelay", "host_scheduling_delay_s"),
        ("ComputerToArduinoRxLatency", "dispatch_to_rx_latency_s"),
        ("ComputerToArduinoApplyLatency", "dispatch_to_apply_latency_s"),
        ("ScheduledToArduinoRxLatency", "scheduled_to_rx_latency_s"),
        ("ArduinoReceiveToApplyLatency", "rx_to_apply_latency_s"),
        ("ScheduledToApplyLatency", "scheduled_to_apply_latency_s"),
        ("ApplyToOutputLatency", "apply_to_output_latency_s"),
        ("DispatchToOutputLatency", "dispatch_to_output_latency_s"),
        ("ScheduledToOutputLatency", "scheduled_to_output_latency_s"),
    ]

    surface_rows: list[dict] = []
    integrity_rows: list[dict] = []
    for surface_name in surface_names:
        group = events.loc[events["surface_name"].astype(str) == surface_name].copy()
        transition_count = int(pd.to_numeric(group.get("is_transition_command"), errors="coerce").fillna(0).sum()) if not group.empty else 0
        matched_count = int(pd.to_numeric(group.get("matched_output_transition"), errors="coerce").fillna(0).sum()) if not group.empty else 0
        valid_count = int(pd.to_numeric(group.get("is_valid_e2e"), errors="coerce").fillna(0).sum()) if not group.empty else 0
        unmatched_count = transition_count - matched_count
        unmatched_fraction = (unmatched_count / transition_count) if transition_count > 0 else np.nan

        row: dict[str, object] = {
            "SurfaceName": surface_name,
            "IsActive": 1,
            "TransitionCommandCount": transition_count,
            "MatchedOutputTransitionCount": matched_count,
            "UnmatchedOutputTransitionCount": unmatched_count,
            "UnmatchedOutputTransitionFraction": unmatched_fraction,
        }
        for prefix, column_name in metric_map:
            stats = latency_stats(group.get(column_name, pd.Series(dtype=float)))
            for suffix in ["SampleCount", "Mean_s", "Std_s", "Median_s", "P95_s", "P99_s", "Max_s"]:
                row[f"{prefix}{suffix}"] = stats[suffix]
        surface_rows.append(row)

        integrity_rows.append({
            "SurfaceName": surface_name,
            "IsActive": 1,
            "TransitionCommandCount": transition_count,
            "MatchedOutputTransitionCount": matched_count,
            "ValidE2ECount": valid_count,
            "UnmatchedOutputTransitionCount": unmatched_count,
            "UnmatchedOutputTransitionFraction": unmatched_fraction,
        })

    surface_summary = pd.DataFrame(surface_rows)
    overall_rows = []
    for metric_name in [
        "dispatch_to_rx_latency_s",
        "dispatch_to_apply_latency_s",
        "scheduled_to_rx_latency_s",
        "scheduled_to_apply_latency_s",
        "rx_to_apply_latency_s",
        "apply_to_output_latency_s",
        "dispatch_to_output_latency_s",
        "scheduled_to_output_latency_s",
    ]:
        values = pd.to_numeric(events.get(metric_name, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        overall_rows.append({
            "metric": metric_name,
            "count": int(values.size),
            "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)),
            "p95_s": float(np.percentile(values, 95)),
            "p99_s": float(np.percentile(values, 99)),
            "max_s": float(np.max(values)),
        })
    overall_summary = pd.DataFrame(overall_rows, columns=["metric", "count", "mean_s", "median_s", "p95_s", "p99_s", "max_s"])
    integrity_summary = pd.DataFrame(integrity_rows)
    return surface_summary, overall_summary, integrity_summary


def cast_bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in BOOL_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    return out


def export_arduino_e2e(
    events: pd.DataFrame,
    input_signal: pd.DataFrame,
    profile_events: pd.DataFrame,
    output_capture: pd.DataFrame,
    reference_capture: pd.DataFrame,
    surface_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    integrity_summary: pd.DataFrame,
    logger_folder: Path,
    output_prefix: str,
) -> dict[str, Path]:
    logger_folder.mkdir(parents=True, exist_ok=True)
    prefix = logger_folder / output_prefix

    events_out = cast_bool_columns(events)
    surface_summary_out = cast_bool_columns(surface_summary)
    integrity_summary_out = cast_bool_columns(integrity_summary)
    input_signal_out = cast_bool_columns(input_signal)

    event_path = prefix.with_name(prefix.name + "_events.csv")
    surface_summary_path = prefix.with_name(prefix.name + "_surface_summary.csv")
    overall_summary_path = prefix.with_name(prefix.name + "_overall_summary.csv")
    integrity_summary_path = prefix.with_name(prefix.name + "_integrity_summary.csv")
    input_signal_path = prefix.with_name(prefix.name + "_input_signal.csv")
    profile_events_path = prefix.with_name(prefix.name + "_profile_events.csv")

    events_out.to_csv(event_path, index=False)
    surface_summary_out.to_csv(surface_summary_path, index=False)
    overall_summary.to_csv(overall_summary_path, index=False)
    integrity_summary_out.to_csv(integrity_summary_path, index=False)
    input_signal_out.to_csv(input_signal_path, index=False)
    if not profile_events.empty:
        profile_events.to_csv(profile_events_path, index=False)
    if not reference_capture.empty:
        reference_capture.to_csv(logger_folder / "reference_capture.csv", index=False)
    output_capture.to_csv(logger_folder / "output_capture.csv", index=False)
    return {
        "eventPath": event_path,
        "surfaceSummaryPath": surface_summary_path,
        "overallSummaryPath": overall_summary_path,
        "integritySummaryPath": integrity_summary_path,
        "inputSignalPath": input_signal_path,
        "profileEventsPath": profile_events_path if not profile_events.empty else Path(),
    }


def run_post_processing(
    logger_folder: Path,
    output_prefix: str,
    sample_time_seconds: float,
    clock_map_mode: str,
    matching_mode: str,
    max_output_association_seconds: float,
    reference_association_window_seconds: float,
    transition_pulse_threshold_us: float,
    target_pulse_tolerance_us: float,
    previous_pulse_tolerance_us: float,
    pre_anchor_slack_seconds: float,
    maximum_apply_to_output_seconds: float,
) -> dict[str, object]:
    logic_config = LogicAnalyzerConfig()
    logs = load_arduino_logs(logger_folder, sample_time_seconds, logic_config, output_prefix)
    events, output_capture, reference_capture, analyzer_alignment = compute_arduino_e2e(
        logs,
        clock_map_mode,
        matching_mode,
        max_output_association_seconds,
        reference_association_window_seconds,
        transition_pulse_threshold_us,
        target_pulse_tolerance_us,
        previous_pulse_tolerance_us,
        pre_anchor_slack_seconds,
        maximum_apply_to_output_seconds,
    )
    surface_summary, overall_summary, integrity_summary = build_summaries(events, list(logs["surfaceNames"]))
    output_paths = export_arduino_e2e(
        events,
        logs["inputSignal"],
        logs["profileEvents"],
        output_capture,
        reference_capture,
        surface_summary,
        overall_summary,
        integrity_summary,
        logger_folder,
        output_prefix,
    )
    return {
        "events": events,
        "surfaceSummary": surface_summary,
        "overallSummary": overall_summary,
        "integritySummary": integrity_summary,
        "outputCapture": output_capture,
        "referenceCapture": reference_capture,
        "outputPaths": output_paths,
        "analyzerAlignment": analyzer_alignment,
        "surfaceNames": list(logs["surfaceNames"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Python post-processing for Arduino_Test_E2E.")
    parser.add_argument("--logger-folder", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--sample-time-seconds", type=float, default=DEFAULT_SAMPLE_TIME_SECONDS)
    parser.add_argument("--clock-map-mode", type=str, default=DEFAULT_CLOCK_MAP_MODE)
    parser.add_argument("--matching-mode", type=str, default=DEFAULT_MATCHING_MODE)
    parser.add_argument("--max-output-association-seconds", type=float, default=DEFAULT_MAX_OUTPUT_ASSOCIATION_SECONDS)
    parser.add_argument("--reference-association-window-seconds", type=float, default=DEFAULT_REFERENCE_ASSOCIATION_WINDOW_SECONDS)
    parser.add_argument("--transition-pulse-threshold-us", type=float, default=DEFAULT_TRANSITION_PULSE_THRESHOLD_US)
    parser.add_argument("--target-pulse-tolerance-us", type=float, default=DEFAULT_TARGET_PULSE_TOLERANCE_US)
    parser.add_argument("--previous-pulse-tolerance-us", type=float, default=DEFAULT_PREVIOUS_PULSE_TOLERANCE_US)
    parser.add_argument("--pre-anchor-slack-seconds", type=float, default=DEFAULT_PRE_ANCHOR_SLACK_SECONDS)
    parser.add_argument("--maximum-apply-to-output-seconds", type=float, default=DEFAULT_MAXIMUM_APPLY_TO_OUTPUT_SECONDS)
    args = parser.parse_args()

    logger_folder = resolve_logger_folder(args.logger_folder, args.seed)
    result = run_post_processing(
        logger_folder=logger_folder,
        output_prefix=args.output_prefix,
        sample_time_seconds=args.sample_time_seconds,
        clock_map_mode=args.clock_map_mode,
        matching_mode=args.matching_mode,
        max_output_association_seconds=args.max_output_association_seconds,
        reference_association_window_seconds=args.reference_association_window_seconds,
        transition_pulse_threshold_us=args.transition_pulse_threshold_us,
        target_pulse_tolerance_us=args.target_pulse_tolerance_us,
        previous_pulse_tolerance_us=args.previous_pulse_tolerance_us,
        pre_anchor_slack_seconds=args.pre_anchor_slack_seconds,
        maximum_apply_to_output_seconds=args.maximum_apply_to_output_seconds,
    )

    integrity = result["integritySummary"]
    matched = int(pd.to_numeric(integrity.get("MatchedOutputTransitionCount"), errors="coerce").fillna(0).sum()) if not integrity.empty else 0
    valid = int(pd.to_numeric(integrity.get("ValidE2ECount"), errors="coerce").fillna(0).sum()) if not integrity.empty else 0
    total = int(pd.to_numeric(integrity.get("TransitionCommandCount"), errors="coerce").fillna(0).sum()) if not integrity.empty else 0

    print(f"Logger folder: {logger_folder.resolve()}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Transition commands: {total}")
    print(f"Matched output transitions: {matched}")
    print(f"Valid output events: {valid}")
    print(f"Events CSV: {result['outputPaths']['eventPath'].resolve()}")


if __name__ == "__main__":
    main()
