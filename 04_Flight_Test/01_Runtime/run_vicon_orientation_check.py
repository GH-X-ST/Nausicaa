from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from flight_config import CONTROLLER_ROOT, RESULT_ROOT
from flight_logger import FlightLogger
from run_experiment_sequence import VICON_ATTITUDE_SIGNS, VICON_HOST, VICON_POSITION_OFFSET_M, VICON_YAW_ALIGNMENT_DEG
from vicon_rigid_body import LiveNausicaaViconRigidBody

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import NausicaaViconStateAdapter, ViconArenaFrameTransform  # noqa: E402
from state_contract import STATE_INDEX, state_dataframe_row  # noqa: E402


# =============================================================================
# CLICK-AND-GO SETTINGS
# This script imports the current Vicon host/offset/yaw from run_experiment_sequence.py.
# =============================================================================
VICON_SUBJECT_NAME = "Nausicaa"
SAMPLE_RATE_HZ = 200.0
STEP_DURATION_S = 3.0
STEP_COUNTDOWN_S = 10.0
MIN_TRANSLATION_DELTA_M = 0.15
MIN_ATTITUDE_DELTA_DEG = 8.0
# =============================================================================


@dataclass(frozen=True)
class OrientationStep:
    step_id: str
    instruction: str
    kind: str
    signal_name: str
    expected_sign: int
    min_delta: float


ORIENTATION_STEPS = (
    OrientationStep(
        "neutral",
        "Hold level, nose toward +x/front wall, left wing toward +y, top upward.",
        "reference",
        "",
        0,
        0.0,
    ),
    OrientationStep(
        "move_forward",
        "Translate the whole glider forward toward the front wall (+x_w).",
        "translation",
        "x_w",
        +1,
        MIN_TRANSLATION_DELTA_M,
    ),
    OrientationStep(
        "move_left",
        "Translate the whole glider left (+y_w).",
        "translation",
        "y_w",
        +1,
        MIN_TRANSLATION_DELTA_M,
    ),
    OrientationStep(
        "move_up",
        "Translate the whole glider upward (+z_w).",
        "translation",
        "z_w",
        +1,
        MIN_TRANSLATION_DELTA_M,
    ),
    OrientationStep(
        "pitch_up",
        "Hold the body nearly fixed and pitch nose up.",
        "attitude",
        "theta",
        +1,
        np.deg2rad(MIN_ATTITUDE_DELTA_DEG),
    ),
    OrientationStep(
        "pitch_down",
        "Hold the body nearly fixed and pitch nose down.",
        "attitude",
        "theta",
        -1,
        np.deg2rad(MIN_ATTITUDE_DELTA_DEG),
    ),
    OrientationStep(
        "roll_right",
        "Viewed from behind, roll right wing down / left wing up.",
        "attitude",
        "phi",
        +1,
        np.deg2rad(MIN_ATTITUDE_DELTA_DEG),
    ),
    OrientationStep(
        "roll_left",
        "Viewed from behind, roll left wing down / right wing up.",
        "attitude",
        "phi",
        -1,
        np.deg2rad(MIN_ATTITUDE_DELTA_DEG),
    ),
    OrientationStep(
        "yaw_right",
        "Viewed from above, yaw nose right.",
        "attitude",
        "psi",
        +1,
        np.deg2rad(MIN_ATTITUDE_DELTA_DEG),
    ),
    OrientationStep(
        "yaw_left",
        "Viewed from above, yaw nose left.",
        "attitude",
        "psi",
        -1,
        np.deg2rad(MIN_ATTITUDE_DELTA_DEG),
    ),
)


def run_vicon_orientation_check(
    *,
    vicon_host: str,
    subject_name: str,
    vicon_position_offset_m: tuple[float, float, float],
    vicon_yaw_alignment_deg: float,
    vicon_attitude_signs: tuple[float, float, float],
    sample_rate_hz: float,
    step_duration_s: float,
    step_countdown_s: float,
    run_label: str,
) -> dict[str, object]:
    poll_period_s = 1.0 / float(sample_rate_hz)
    run_root = RESULT_ROOT / "vicon_orientation_check" / run_label
    logger = FlightLogger(run_root)
    reader = LiveNausicaaViconRigidBody(host=vicon_host, subject_name=subject_name)
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=20.0,
        arena_transform=ViconArenaFrameTransform(
            position_offset_m=vicon_position_offset_m,
            yaw_alignment_rad=float(np.deg2rad(vicon_yaw_alignment_deg)),
            attitude_signs=vicon_attitude_signs,
        ),
    )
    logger.write_manifest(
        "vicon_orientation_check_manifest.json",
        {
            "vicon_host": str(vicon_host),
            "subject_name": str(subject_name),
            "vicon_position_offset_m": tuple(float(value) for value in vicon_position_offset_m),
            "vicon_yaw_alignment_deg": float(vicon_yaw_alignment_deg),
            "vicon_attitude_signs_phi_theta_psi": tuple(float(value) for value in vicon_attitude_signs),
            "sample_rate_hz": float(sample_rate_hz),
            "step_duration_s": float(step_duration_s),
            "step_countdown_s": float(step_countdown_s),
            "expected_controller_convention": {
                "positive_x_w": "glider moves toward the front wall",
                "positive_y_w": "glider moves left",
                "positive_z_w": "glider moves upward",
                "positive_theta": "nose-up pitch",
                "positive_phi": "right wing down / left wing up roll",
                "positive_psi": "nose-right yaw",
            },
        },
    )

    print("[CHECK] Vicon rigid-body orientation check")
    print(f"[CHECK] result root: {run_root}")
    print("[CHECK] Critical convention: physical nose-up must produce positive theta.")
    print("[CHECK] Stop and do not fly if any sign check fails.")

    all_samples: dict[str, list[np.ndarray]] = {}
    raw_samples_by_step: dict[str, list[np.ndarray]] = {}
    invalid_reasons: dict[str, int] = {}

    try:
        reader.open()
        for step in ORIENTATION_STEPS:
            print()
            print(f"[STEP] {step.step_id}: {step.instruction}")
            _countdown(step_countdown_s)
            samples, raw_samples = _collect_step(
                reader=reader,
                adapter=adapter,
                logger=logger,
                step=step,
                duration_s=float(step_duration_s),
                poll_period_s=poll_period_s,
                invalid_reasons=invalid_reasons,
            )
            all_samples[step.step_id] = samples
            raw_samples_by_step[step.step_id] = raw_samples
            print(f"[STEP] collected {len(samples)} valid samples")
    finally:
        reader.close()

    result_rows = _evaluate_steps(all_samples)
    for row in result_rows:
        logger.append_metric_row("vicon_orientation_check_summary.csv", row)
    passed = all(bool(row["passed"]) for row in result_rows if row["step_id"] != "neutral")
    result = {
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "run_root": run_root.as_posix(),
        "invalid_reasons": invalid_reasons,
        "summary": result_rows,
    }
    logger.write_manifest("vicon_orientation_check_summary.json", result)
    logger.write_report(
        "vicon_orientation_check_report.md",
        _report_lines(result_rows=result_rows, result=result),
    )
    logger.close()

    print()
    print(f"[RESULT] {result['status'].upper()}")
    for row in result_rows:
        if row["step_id"] == "neutral":
            continue
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{status}] {row['step_id']}: {row['observed_delta']:.3f} "
            f"{row['unit']} expected {row['expected_direction']}"
        )
    print(f"[RESULT] report: {run_root / 'reports' / 'vicon_orientation_check_report.md'}")
    return result


def _countdown(duration_s: float) -> None:
    duration = max(0.0, float(duration_s))
    if duration <= 0.0:
        return
    started = time.perf_counter()
    last_printed = None
    while True:
        remaining = max(0.0, duration - (time.perf_counter() - started))
        remaining_int = int(np.ceil(remaining))
        if remaining_int != last_printed:
            print(f"[STEP] collecting in {remaining_int}s")
            last_printed = remaining_int
        if remaining <= 0.0:
            break
        time.sleep(min(0.2, remaining))
    print("[STEP] collecting now")


def _collect_step(
    *,
    reader: LiveNausicaaViconRigidBody,
    adapter: NausicaaViconStateAdapter,
    logger: FlightLogger,
    step: OrientationStep,
    duration_s: float,
    poll_period_s: float,
    invalid_reasons: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    samples: list[np.ndarray] = []
    raw_samples: list[np.ndarray] = []
    started = time.perf_counter()
    next_print_s = 0.0
    while (time.perf_counter() - started) <= duration_s:
        elapsed_s = time.perf_counter() - started
        sample, status = reader.read_latest()
        if sample is None or not status.valid:
            invalid_reasons[status.reason] = invalid_reasons.get(status.reason, 0) + 1
            if elapsed_s + 1e-12 >= next_print_s:
                print(f"[STEP] waiting for valid Vicon: {status.reason}")
                next_print_s = elapsed_s + 1.0
            time.sleep(poll_period_s)
            continue
        raw_position = np.asarray(sample.position_m, dtype=float).reshape(3)
        state = adapter.update(sample, command_norm=np.zeros(3))
        estimator = adapter.estimator_status()
        samples.append(state.copy())
        raw_samples.append(raw_position.copy())
        logger.append_metric_row(
            "vicon_orientation_check_samples.csv",
            {
                "step_id": step.step_id,
                "t_host_s": time.perf_counter(),
                "frame_number": status.frame_number,
                "vicon_frame_rate_hz": status.frame_rate_hz,
                "vicon_latency_s": status.vicon_latency_s,
                **{f"estimator_{key}": value for key, value in estimator.items()},
                "raw_x_m": float(raw_position[0]),
                "raw_y_m": float(raw_position[1]),
                "raw_z_m": float(raw_position[2]),
                **state_dataframe_row(state),
            },
        )
        time.sleep(poll_period_s)
    return samples, raw_samples


def _evaluate_steps(samples_by_step: dict[str, list[np.ndarray]]) -> list[dict[str, object]]:
    neutral = _mean_state(samples_by_step.get("neutral", []))
    rows: list[dict[str, object]] = [
        {
            "step_id": "neutral",
            "kind": "reference",
            "signal_name": "",
            "observed_delta": 0.0,
            "observed_delta_deg": "",
            "unit": "",
            "expected_direction": "reference",
            "minimum_required_delta": 0.0,
            "passed": bool(neutral is not None),
            "sample_count": len(samples_by_step.get("neutral", [])),
            "failure_reason": "" if neutral is not None else "no_valid_neutral_samples",
        }
    ]
    for step in ORIENTATION_STEPS:
        if step.step_id == "neutral":
            continue
        samples = samples_by_step.get(step.step_id, [])
        sample_count = len(samples)
        if not samples or neutral is None:
            rows.append(_failed_row(step, "no_valid_samples", sample_count))
            continue
        values = np.vstack(samples)
        if step.kind == "translation":
            idx = STATE_INDEX[step.signal_name]
            delta = float(np.mean(values[:, idx]) - neutral[idx])
            unit = "m"
        elif step.kind == "attitude":
            idx = STATE_INDEX[step.signal_name]
            delta = _wrapped_mean_delta(values[:, idx], float(neutral[idx]))
            unit = "rad"
        else:
            rows.append(_failed_row(step, "unknown_step_kind", sample_count))
            continue
        passed = (float(step.expected_sign) * float(delta)) >= float(step.min_delta)
        rows.append(
            {
                "step_id": step.step_id,
                "kind": step.kind,
                "signal_name": step.signal_name,
                "observed_delta": float(delta),
                "observed_delta_deg": float(np.rad2deg(delta)) if step.kind == "attitude" else "",
                "unit": unit,
                "expected_direction": "+" if step.expected_sign > 0 else "-",
                "minimum_required_delta": float(step.min_delta),
                "passed": bool(passed),
                "sample_count": int(sample_count),
                "failure_reason": "" if passed else "wrong_sign_or_too_small_delta",
            }
        )
    return rows


def _mean_state(samples: list[np.ndarray]) -> np.ndarray | None:
    if not samples:
        return None
    return np.mean(np.vstack(samples), axis=0)


def _wrapped_mean_delta(values: np.ndarray, reference: float) -> float:
    deltas = (np.asarray(values, dtype=float) - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.mean(deltas))


def _failed_row(step: OrientationStep, reason: str, sample_count: int) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "signal_name": step.signal_name,
        "observed_delta": 0.0,
        "observed_delta_deg": "",
        "unit": "",
        "expected_direction": "+" if step.expected_sign > 0 else "-",
        "minimum_required_delta": float(step.min_delta),
        "passed": False,
        "sample_count": int(sample_count),
        "failure_reason": str(reason),
    }


def _report_lines(*, result_rows: list[dict[str, object]], result: dict[str, object]) -> list[str]:
    lines = [
        "# Vicon Orientation Check Report",
        f"- Status: `{result['status']}`",
        f"- Result root: `{result['run_root']}`",
        "",
        "Expected convention:",
        "- move forward -> x_w increases",
        "- move left -> y_w increases",
        "- move up -> z_w increases",
        "- nose up -> theta positive",
        "- right wing down / left wing up -> phi positive",
        "- nose right -> psi positive",
        "",
        "| Step | Signal | Observed | Expected | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        if row["step_id"] == "neutral":
            continue
        observed = row["observed_delta"]
        if row.get("unit") == "rad":
            observed_text = f"{float(row.get('observed_delta_deg', 0.0)):.2f} deg"
        else:
            observed_text = f"{float(observed):.3f} m"
        lines.append(
            f"| `{row['step_id']}` | `{row['signal_name']}` | {observed_text} | "
            f"`{row['expected_direction']}` | `{row['passed']}` |"
        )
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided Vicon rigid-body orientation and sign check.")
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--subject-name", default=VICON_SUBJECT_NAME)
    parser.add_argument("--vicon-offset-m", nargs=3, type=float, default=VICON_POSITION_OFFSET_M)
    parser.add_argument("--vicon-yaw-deg", type=float, default=VICON_YAW_ALIGNMENT_DEG)
    parser.add_argument("--vicon-attitude-signs", nargs=3, type=float, default=VICON_ATTITUDE_SIGNS)
    parser.add_argument("--sample-rate-hz", type=float, default=SAMPLE_RATE_HZ)
    parser.add_argument("--step-duration-s", type=float, default=STEP_DURATION_S)
    parser.add_argument("--step-countdown-s", type=float, default=STEP_COUNTDOWN_S)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_vicon_orientation_check(
        vicon_host=str(args.vicon_host),
        subject_name=str(args.subject_name),
        vicon_position_offset_m=tuple(args.vicon_offset_m),
        vicon_yaw_alignment_deg=float(args.vicon_yaw_deg),
        vicon_attitude_signs=tuple(args.vicon_attitude_signs),
        sample_rate_hz=float(args.sample_rate_hz),
        step_duration_s=float(args.step_duration_s),
        step_countdown_s=float(args.step_countdown_s),
        run_label=run_label,
    )


if __name__ == "__main__":
    main()
