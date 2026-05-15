from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup
# 2) Entry sampling
# 3) Sweep execution
# 4) CLI entry point
# =============================================================================

# =============================================================================
# 1) Import Path Setup
# =============================================================================
def _add_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "03_Control/02_Inner_Loop",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
    ):
        path = repo_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _add_paths()

from arena import ArenaConfig  # noqa: E402
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from governor import ViabilityGovernor  # noqa: E402
from latency import CommandToSurfaceLayer, LatencyEnvelope  # noqa: E402
from linearisation import STATE_INDEX, linearise_trim  # noqa: E402
from metrics import rejected_metrics  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, simulate_primitive, write_log  # noqa: E402
from run_one import _output_dirs  # noqa: E402
from scenarios import build_scenario  # noqa: E402


# =============================================================================
# 2) Entry Sampling
# =============================================================================
@dataclass(frozen=True)
class EntrySweepBounds:
    position_m: tuple[float, float, float] = (0.05, 0.08, 0.06)
    attitude_rad: tuple[float, float, float] = (
        np.deg2rad(4.0),
        np.deg2rad(3.0),
        np.deg2rad(3.0),
    )
    velocity_m_s: tuple[float, float, float] = (0.15, 0.08, 0.08)
    rate_rad_s: tuple[float, float, float] = (
        np.deg2rad(3.0),
        np.deg2rad(3.0),
        np.deg2rad(3.0),
    )
    surface_rad: tuple[float, float, float] = (
        np.deg2rad(2.0),
        np.deg2rad(2.0),
        np.deg2rad(2.0),
    )


def sample_entry_states(
    x_nominal: np.ndarray,
    seed: int,
    sample_count: int,
    bounds: object | None = None,
) -> np.ndarray:
    """Return deterministic randomised entry states in canonical state order."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    nominal = np.asarray(x_nominal, dtype=float).reshape(15)
    limits = _entry_sweep_bounds(bounds)
    rng = np.random.default_rng(int(seed))
    samples = np.repeat(nominal.reshape(1, 15), int(sample_count), axis=0)
    if sample_count == 0:
        return samples
    samples[:, 0:3] += rng.uniform(-1.0, 1.0, size=(sample_count, 3)) * np.asarray(
        limits.position_m
    )
    samples[:, 3:6] += rng.uniform(-1.0, 1.0, size=(sample_count, 3)) * np.asarray(
        limits.attitude_rad
    )
    samples[:, 6:9] += rng.uniform(-1.0, 1.0, size=(sample_count, 3)) * np.asarray(
        limits.velocity_m_s
    )
    samples[:, 9:12] += rng.uniform(-1.0, 1.0, size=(sample_count, 3)) * np.asarray(
        limits.rate_rad_s
    )
    samples[:, 12:15] += rng.uniform(-1.0, 1.0, size=(sample_count, 3)) * np.asarray(
        limits.surface_rad
    )
    samples[:, STATE_INDEX["x_w"]] = np.clip(samples[:, STATE_INDEX["x_w"]], 1.25, 6.45)
    samples[:, STATE_INDEX["y_w"]] = np.clip(samples[:, STATE_INDEX["y_w"]], 0.15, 4.25)
    samples[:, STATE_INDEX["z_w"]] = np.clip(samples[:, STATE_INDEX["z_w"]], 0.90, 2.90)
    return samples


# =============================================================================
# 3) Sweep Execution
# =============================================================================
SWEEP_SUMMARY_STAT_KEYS = (
    "heading_change_deg",
    "min_wall_distance_m",
    "height_change_m",
    "terminal_speed_m_s",
    "max_alpha_deg",
    "saturation_fraction",
)


def run_entry_sweep(
    scenario_id: str,
    primitive: object,
    seed: int,
    sample_count: int,
    output_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Run randomised entry states and return metric rows."""
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(
        x_trim=linear_model.x_trim,
        u_trim=linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    arena = ArenaConfig()
    scenario = build_scenario(scenario_id, linear_model.x_trim, REPO_ROOT, seed=seed)
    if primitive is None:
        selected_primitive = scenario.primitive
    elif primitive is scenario.primitive:
        selected_primitive = scenario.primitive
    else:
        selected_primitive = primitive
    if hasattr(selected_primitive, "x_ref"):
        x_nominal = np.asarray(selected_primitive.x_ref[0], dtype=float)
    else:
        x_nominal = scenario.x0
    samples = sample_entry_states(x_nominal, seed=seed, sample_count=sample_count)
    metrics_dir, log_dir = _output_dirs(output_root)
    stem = _sweep_file_stem(scenario_id, seed)
    metrics_path = metrics_dir / f"{stem}.csv"
    rejection_path = metrics_dir / f"{stem}_rejections.csv"
    governor = ViabilityGovernor(arena_config=arena)
    rows: list[dict[str, object]] = []
    for sample_idx, x0 in enumerate(samples):
        log_path = log_dir / f"{stem}_sample{sample_idx:03d}.csv"
        decision = governor.evaluate(
            scenario_id=scenario_id,
            primitive=selected_primitive,
            x0=x0,
            context=context,
        )
        if not decision.accepted:
            envelope = LatencyEnvelope()
            row = rejected_metrics(
                scenario_id=scenario_id,
                seed=seed,
                wind_model=scenario.wind_model_name,
                wind_mode=scenario.wind_mode,
                wind_param_label=scenario.wind_param_label,
                latency_mode=scenario.latency_config.mode,
                latency_s=None,
                latency_range_s=None,
                primitive_name=selected_primitive.name,
                x0=x0,
                log_path=log_path,
                repo_root=REPO_ROOT,
                arena_config=arena,
                governor_rejection_reason="; ".join(decision.reasons),
                state_feedback_delay_s=None,
                actuator_t10_s=float(envelope.onset_latency_s),
                actuator_t50_nominal_s=float(envelope.half_response_nominal_s),
                actuator_t90_s=float(envelope.actuator_t90_s),
                conservative_actuator_bound_s=float(envelope.conservative_actuator_bound_s),
                vicon_filter_cutoff_hz=float(envelope.vicon_filter_cutoff_hz),
                vicon_filter_model=str(envelope.vicon_filter_model),
            )
        else:
            result = simulate_primitive(
                scenario_id=scenario_id,
                seed=seed,
                primitive=selected_primitive,
                x0=x0,
                context=context,
                aircraft=aircraft,
                wind_model=scenario.wind_model,
                wind_model_name=scenario.wind_model_name,
                wind_mode=scenario.wind_mode,
                command_layer=CommandToSurfaceLayer(
                    config=scenario.latency_config,
                    envelope=LatencyEnvelope(),
                ),
                log_path=log_path,
                repo_root=REPO_ROOT,
                rollout_config=RolloutConfig(),
                arena_config=arena,
                wind_param_label=scenario.wind_param_label,
            )
            write_log(result, log_path)
            row = dict(result.metrics)
        row["sample_index"] = int(sample_idx)
        row["entry_seed"] = int(seed)
        row["run_id"] = f"{scenario_id}_seed{int(seed)}_sample{sample_idx:03d}"
        rows.append(row)
    governor.write_rejection_log(rejection_path)
    _write_rows(metrics_path, rows)
    return rows


def summarise_sweep_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    sample_count = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success", False)))
    rejection_count = sum(
        1 for row in rows if str(row.get("termination_reason", "")) == "governor_rejected"
    )
    summary: dict[str, object] = {
        "sample_count": sample_count,
        "success_count": success_count,
        "success_rate": 0.0 if sample_count == 0 else success_count / sample_count,
        "rejection_count": rejection_count,
    }
    for key in SWEEP_SUMMARY_STAT_KEYS:
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if row.get(key) not in {None, ""} and np.isfinite(float(row[key]))
            ],
            dtype=float,
        )
        if values.size == 0:
            summary[f"{key}_mean"] = None
            summary[f"{key}_min"] = None
            summary[f"{key}_max"] = None
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def _entry_sweep_bounds(value: object | None) -> EntrySweepBounds:
    if value is None:
        return EntrySweepBounds()
    if isinstance(value, EntrySweepBounds):
        return value
    if isinstance(value, dict):
        return EntrySweepBounds(**value)
    return EntrySweepBounds(
        position_m=tuple(getattr(value, "position_m", EntrySweepBounds.position_m)),
        attitude_rad=tuple(getattr(value, "attitude_rad", EntrySweepBounds.attitude_rad)),
        velocity_m_s=tuple(getattr(value, "velocity_m_s", EntrySweepBounds.velocity_m_s)),
        rate_rad_s=tuple(getattr(value, "rate_rad_s", EntrySweepBounds.rate_rad_s)),
        surface_rad=tuple(getattr(value, "surface_rad", EntrySweepBounds.surface_rad)),
    )


def _sweep_file_stem(scenario_id: str, seed: int) -> str:
    parts = scenario_id.split("_")
    target = next(
        (
            parts[index + 1]
            for index, part in enumerate(parts[:-1])
            if part == "target" and parts[index + 1].isdigit()
        ),
        None,
    )
    scenario_tag = target if target is not None else "_".join(parts[:4])
    return f"{scenario_tag}_sweep_s{int(seed):03d}"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 4) CLI Entry Point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    rows = run_entry_sweep(
        scenario_id=args.scenario,
        primitive=None,
        seed=args.seed,
        sample_count=args.samples,
        output_root=args.output_root,
    )
    summary = summarise_sweep_rows(rows)
    print("sweep complete")
    print(f"scenario_id: {args.scenario}")
    print(f"samples: {summary['sample_count']}")
    print(f"success_rate: {summary['success_rate']:.3f}")
    print(f"rejections: {summary['rejection_count']}")
    metrics_dir, _log_dir = _output_dirs(args.output_root)
    print(f"metrics: {metrics_dir / f'{_sweep_file_stem(args.scenario, args.seed)}.csv'}")


if __name__ == "__main__":
    main()
