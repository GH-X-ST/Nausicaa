"""Prepare real-glider calibration summaries from 04_Flight_Test logs.

This script is intentionally offline. It reads completed flight-test calibration
sessions, extracts neutral-glide and invalid-start metrics, and writes compact
calibration targets for later grey-box model fitting. It does not edit the
flight dynamics model, controller, primitive library, or real-flight runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_SEARCH_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"

VALID_THROW_FIELDS = [
    "session_label",
    "case_id",
    "case_name",
    "throw_id",
    "command_axis",
    "command_value",
    "pulse_start_s",
    "pulse_duration_s",
    "termination_reason",
    "launch_speed_m_s",
    "duration_s",
    "x0_m",
    "y0_m",
    "z0_m",
    "x_end_m",
    "y_end_m",
    "z_end_m",
    "dx_m",
    "dy_m",
    "dz_m",
    "altitude_loss_m",
    "horizontal_distance_m",
    "glide_ratio_x_over_altloss",
    "glide_ratio_horizontal_over_altloss",
    "sink_rate_m_s",
    "mean_speed_m_s",
    "mean_forward_speed_m_s",
    "max_abs_phi_deg",
    "max_abs_theta_deg",
    "max_abs_psi_deg",
    "max_abs_p_rad_s",
    "max_abs_q_rad_s",
    "max_abs_r_rad_s",
    "mean_rate_confidence",
    "min_rate_confidence",
    "spike_downweighted_fraction",
    "body_rate_limited_fraction",
    "sample_count",
]

INVALID_ATTEMPT_FIELDS = [
    "session_label",
    "case_id",
    "case_name",
    "attempt_id",
    "cancellation_reason",
    "launch_gate_reason",
    "trigger_source",
    "speed_m_s",
    "x_w_m",
    "y_w_m",
    "z_w_m",
    "phi_deg",
    "theta_deg",
    "psi_deg",
    "p_rad_s",
    "q_rad_s",
    "r_rad_s",
]

AGGREGATE_FIELDS = [
    "metric",
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "role_for_model_calibration",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.10g}"
        return ""
    return value


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return mean(finite) if finite else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return median(finite) if finite else float("nan")


def _safe_std(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return pstdev(finite) if len(finite) > 1 else 0.0 if finite else float("nan")


def _safe_min(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def _safe_max(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-9:
        return float("nan")
    return numerator / denominator


def _session_summary_path(path: Path) -> Path:
    return path / "manifests" / "glider_calibration_sequence_final_summary.json"


def resolve_session_root(path: Path) -> Path:
    """Resolve either a session root or a directory containing session roots."""
    if _session_summary_path(path).exists():
        return path
    candidates: list[Path] = []
    if path.exists():
        for summary_path in path.rglob("manifests/glider_calibration_sequence_final_summary.json"):
            session_root = summary_path.parents[1]
            summary = _load_json(summary_path)
            if summary.get("block_id") == "neutral_30" and summary.get("total_valid_throw_count", 0):
                candidates.append(session_root)
    if not candidates:
        raise FileNotFoundError(
            f"No completed glider calibration session found under {path}. "
            "Pass --session-root to a folder containing manifests/glider_calibration_sequence_final_summary.json."
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _case_dirs(session_root: Path) -> list[Path]:
    ignored = {"manifests", "metrics", "reports"}
    return sorted(
        child for child in session_root.iterdir() if child.is_dir() and child.name not in ignored
    )


def _throw_summary(throw_dir: Path) -> dict[str, Any]:
    return _load_json(throw_dir / "manifests" / "glider_calibration_throw_summary.json")


def _throw_manifest(throw_dir: Path) -> dict[str, Any]:
    return _load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")


def _valid_throw_dirs(session_root: Path) -> list[Path]:
    throw_dirs: list[Path] = []
    for case_dir in _case_dirs(session_root):
        candidate_dirs = list(case_dir.glob("throw_*")) + list(case_dir.glob("v[0-9]*"))
        for throw_dir in sorted(candidate_dirs):
            summary = _throw_summary(throw_dir)
            if summary.get("valid_throw") is True:
                throw_dirs.append(throw_dir)
    return throw_dirs


def _invalid_attempt_dirs(session_root: Path) -> list[Path]:
    attempt_dirs: list[Path] = []
    for case_dir in _case_dirs(session_root):
        old_invalid_root = case_dir / "invalid_attempts"
        if old_invalid_root.exists():
            attempt_dirs.extend(sorted(old_invalid_root.glob("attempt_*")))
        new_invalid_root = case_dir / "bad"
        if new_invalid_root.exists():
            attempt_dirs.extend(sorted(new_invalid_root.glob("i[0-9]*")))
    return attempt_dirs


def summarize_valid_throw(session_label: str, throw_dir: Path) -> dict[str, Any]:
    summary = _throw_summary(throw_dir)
    manifest = _throw_manifest(throw_dir)
    case = manifest.get("calibration_case", {})
    rows = _read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not rows:
        raise ValueError(f"No state_samples.csv rows found for {throw_dir}")

    first = rows[0]
    last = rows[-1]
    t0 = _float(first, "t_s", 0.0)
    t1 = _float(last, "t_s", t0)
    duration_s = max(0.0, t1 - t0)
    x0, y0, z0 = _float(first, "x_w"), _float(first, "y_w"), _float(first, "z_w")
    x1, y1, z1 = _float(last, "x_w"), _float(last, "y_w"), _float(last, "z_w")
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    altitude_loss_m = z0 - z1
    horizontal_distance_m = math.hypot(dx, dy)

    speeds = [
        math.sqrt(_float(row, "u", 0.0) ** 2 + _float(row, "v", 0.0) ** 2 + _float(row, "w", 0.0) ** 2)
        for row in rows
    ]
    forward_speeds = [_float(row, "u") for row in rows]
    confidence = [_float(row, "estimator_rate_confidence") for row in rows]
    spike_flags = [row.get("estimator_spike_rejected", "").strip().lower() == "true" for row in rows]
    limited_flags = [row.get("estimator_body_rate_limited", "").strip().lower() == "true" for row in rows]

    return {
        "session_label": session_label,
        "case_id": summary.get("case_id", case.get("case_id", "")),
        "case_name": summary.get("case_name", case.get("case_name", "")),
        "throw_id": throw_dir.name,
        "command_axis": case.get("command_axis", ""),
        "command_value": case.get("command_value", ""),
        "pulse_start_s": case.get("pulse_start_s", ""),
        "pulse_duration_s": case.get("pulse_duration_s", ""),
        "termination_reason": summary.get("termination_reason", ""),
        "launch_speed_m_s": summary.get("launch_speed_m_s", float("nan")),
        "duration_s": duration_s,
        "x0_m": x0,
        "y0_m": y0,
        "z0_m": z0,
        "x_end_m": x1,
        "y_end_m": y1,
        "z_end_m": z1,
        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,
        "altitude_loss_m": altitude_loss_m,
        "horizontal_distance_m": horizontal_distance_m,
        "glide_ratio_x_over_altloss": _ratio(dx, altitude_loss_m),
        "glide_ratio_horizontal_over_altloss": _ratio(horizontal_distance_m, altitude_loss_m),
        "sink_rate_m_s": _ratio(altitude_loss_m, duration_s),
        "mean_speed_m_s": _safe_mean(speeds),
        "mean_forward_speed_m_s": _safe_mean(forward_speeds),
        "max_abs_phi_deg": math.degrees(_safe_max([abs(_float(row, "phi")) for row in rows])),
        "max_abs_theta_deg": math.degrees(_safe_max([abs(_float(row, "theta")) for row in rows])),
        "max_abs_psi_deg": math.degrees(_safe_max([abs(_float(row, "psi")) for row in rows])),
        "max_abs_p_rad_s": _safe_max([abs(_float(row, "p")) for row in rows]),
        "max_abs_q_rad_s": _safe_max([abs(_float(row, "q")) for row in rows]),
        "max_abs_r_rad_s": _safe_max([abs(_float(row, "r")) for row in rows]),
        "mean_rate_confidence": _safe_mean(confidence),
        "min_rate_confidence": _safe_min(confidence),
        "spike_downweighted_fraction": _ratio(sum(spike_flags), len(spike_flags)),
        "body_rate_limited_fraction": _ratio(sum(limited_flags), len(limited_flags)),
        "sample_count": len(rows),
    }


def summarize_invalid_attempt(session_label: str, attempt_dir: Path) -> dict[str, Any]:
    summary = _throw_summary(attempt_dir)
    manifest = _throw_manifest(attempt_dir)
    case = manifest.get("calibration_case", {})
    event_rows = _read_csv(attempt_dir / "metrics" / "runtime_events.csv")
    details: dict[str, Any] = {}
    for row in event_rows:
        if "rejected_launch_attempt" in row.get("event", ""):
            try:
                details = json.loads(row.get("details_json", "{}"))
            except json.JSONDecodeError:
                details = {}
    return {
        "session_label": session_label,
        "case_id": summary.get("case_id", case.get("case_id", "")),
        "case_name": summary.get("case_name", case.get("case_name", "")),
        "attempt_id": attempt_dir.name,
        "cancellation_reason": summary.get("cancellation_reason", ""),
        "launch_gate_reason": details.get("launch_gate_reason", ""),
        "trigger_source": details.get("trigger_source", ""),
        "speed_m_s": details.get("speed_m_s", details.get("launch_attempt_speed_m_s", float("nan"))),
        "x_w_m": details.get("x_w_m", float("nan")),
        "y_w_m": details.get("y_w_m", float("nan")),
        "z_w_m": details.get("z_w_m", float("nan")),
        "phi_deg": details.get("phi_deg", float("nan")),
        "theta_deg": details.get("theta_deg", float("nan")),
        "psi_deg": details.get("psi_deg", float("nan")),
        "p_rad_s": details.get("p_rad_s", float("nan")),
        "q_rad_s": details.get("q_rad_s", float("nan")),
        "r_rad_s": details.get("r_rad_s", float("nan")),
    }


def aggregate_valid_metrics(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles = {
        "launch_speed_m_s": "stratify model error versus launch energy; do not fit directly",
        "sink_rate_m_s": "primary neutral-glide target for drag/lift/trim consistency",
        "glide_ratio_x_over_altloss": "primary neutral-glide target for aerodynamic polar and trim",
        "dy_m": "lateral trim/asymmetry diagnostic before fitting aileron/rudder effects",
        "max_abs_phi_deg": "attitude envelope check; should not be used alone as fit objective",
        "max_abs_theta_deg": "pitch/CG/static-margin diagnostic before controller retuning",
        "max_abs_p_rad_s": "roll-rate envelope diagnostic for future damping/control derivative fit",
        "max_abs_q_rad_s": "pitch-rate envelope diagnostic for future damping/control derivative fit",
        "max_abs_r_rad_s": "yaw-rate envelope diagnostic for future damping/control derivative fit",
        "mean_rate_confidence": "state-estimator evidence quality; exclude low-confidence throws from fitting",
        "spike_downweighted_fraction": "state-estimator health diagnostic",
    }
    rows: list[dict[str, Any]] = []
    for metric, role in roles.items():
        values = [_to_float(row.get(metric)) for row in valid_rows]
        finite = [value for value in values if math.isfinite(value)]
        rows.append(
            {
                "metric": metric,
                "count": len(finite),
                "mean": _safe_mean(finite),
                "median": _safe_median(finite),
                "std": _safe_std(finite),
                "min": _safe_min(finite),
                "max": _safe_max(finite),
                "role_for_model_calibration": role,
            }
        )
    return rows


def _to_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_report(
    report_path: Path,
    session_root: Path,
    output_root: Path,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    termination_counts: dict[str, int] = {}
    for row in valid_rows:
        key = str(row.get("termination_reason", ""))
        termination_counts[key] = termination_counts.get(key, 0) + 1

    def agg(metric: str, key: str = "mean") -> float:
        for row in aggregate_rows:
            if row["metric"] == metric:
                return _to_float(row.get(key))
        return float("nan")

    lines = [
        "# Real Glider Calibration Prep Report",
        "",
        f"- source session: `{session_root.as_posix()}`",
        f"- output root: `{output_root.as_posix()}`",
        f"- generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- valid throws: `{len(valid_rows)}`",
        f"- invalid launch attempts: `{len(invalid_rows)}`",
        f"- termination counts: `{termination_counts}`",
        "",
        "## Neutral-Glide Calibration Targets",
        "",
        "These values are evidence targets for a later grey-box fit. They do not by themselves update the simulator.",
        "",
        f"- mean sink rate: `{_format_value(agg('sink_rate_m_s'))}` m/s",
        f"- mean x/altitude-loss glide ratio: `{_format_value(agg('glide_ratio_x_over_altloss'))}`",
        f"- mean launch speed: `{_format_value(agg('launch_speed_m_s'))}` m/s",
        f"- mean lateral displacement: `{_format_value(agg('dy_m'))}` m",
        f"- mean rate-estimator confidence: `{_format_value(agg('mean_rate_confidence'))}`",
        f"- mean spike-downweighted fraction: `{_format_value(agg('spike_downweighted_fraction'))}`",
        "",
        "## Recommended Calibration Order",
        "",
        "1. Use the full neutral set to fit bare-airframe trim/polar consistency first: sink rate, glide ratio, and pitch tendency.",
        "2. Use held-out neutral throws to check that the fitted model predicts terminal wall/floor behaviour, not only mean sink.",
        "3. Use pulse-ladder throws only after the neutral fit is stable, fitting control effectiveness and damping separately.",
        "4. Regenerate R5/R7/R8/R10/R11 only after the model update is fixed and documented.",
        "",
        "## Files Written",
        "",
        "- `metrics/neutral_throw_summary.csv`",
        "- `metrics/neutral_aggregate_summary.csv`",
        "- `metrics/invalid_attempt_summary.csv`",
        "- `manifests/calibration_prep_manifest.json`",
        "",
        "## Claims Not Made",
        "",
        "- No aerodynamic parameter was changed.",
        "- No controller/library evidence was regenerated.",
        "- No zero-shot transfer claim is made from this prep report alone.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration_prep(session_root: Path, output_root: Path, run_label: str | None) -> Path:
    session_root = resolve_session_root(session_root)
    session_label = session_root.name
    label = run_label or f"{session_label}_prep"
    output_dir = output_root / label

    valid_rows = [summarize_valid_throw(session_label, throw_dir) for throw_dir in _valid_throw_dirs(session_root)]
    invalid_rows = [
        summarize_invalid_attempt(session_label, attempt_dir)
        for attempt_dir in _invalid_attempt_dirs(session_root)
    ]
    aggregate_rows = aggregate_valid_metrics(valid_rows)

    _write_csv(output_dir / "metrics" / "neutral_throw_summary.csv", valid_rows, VALID_THROW_FIELDS)
    _write_csv(output_dir / "metrics" / "invalid_attempt_summary.csv", invalid_rows, INVALID_ATTEMPT_FIELDS)
    _write_csv(output_dir / "metrics" / "neutral_aggregate_summary.csv", aggregate_rows, AGGREGATE_FIELDS)

    manifest = {
        "source_session_root": session_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "session_label": session_label,
        "valid_throw_count": len(valid_rows),
        "invalid_attempt_count": len(invalid_rows),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_stage": "offline_prep_only_no_model_mutation",
    }
    manifest_path = output_dir / "manifests" / "calibration_prep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_report(
        output_dir / "reports" / "neutral_glide_calibration_prep_report.md",
        session_root,
        output_dir,
        valid_rows,
        invalid_rows,
        aggregate_rows,
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare offline real-glider calibration summaries from completed 04_Flight_Test logs."
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=DEFAULT_SESSION_SEARCH_ROOT,
        help=(
            "Completed session root, or a directory containing session roots. Defaults to searching "
            "04_Flight_Test/05_Results for the latest neutral_30 session in either old or short layout."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for compact calibration-prep evidence.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional output run label. Defaults to '<session_label>_prep'.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = run_calibration_prep(args.session_root, args.output_root, args.run_label)
    print(f"[DONE] calibration prep written to {output_dir}")


if __name__ == "__main__":
    main()
