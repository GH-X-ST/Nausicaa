from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup
# 2) Feasibility runner
# 3) CSV and CLI helpers
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

from run_one import _output_dirs, run_scenario  # noqa: E402
from run_sweep import agile_random_entry_gate, run_entry_sweep, summarise_sweep_rows  # noqa: E402


# =============================================================================
# 2) Feasibility Runner
# =============================================================================
def run_agile_feasibility(
    seed: int,
    targets_deg: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0, 180.0),
    output_root: str | Path | None = None,
    sweep_samples: int = 50,
    run_sweeps: bool = False,
) -> list[dict[str, object]]:
    """Return fixed-start agile feasibility rows and gated sweep summaries."""
    if sweep_samples < 50 and run_sweeps:
        raise ValueError("agile feasibility sweeps require at least 50 samples by default.")
    rows: list[dict[str, object]] = []
    for target in targets_deg:
        scenario_id = _target_scenario_id(float(target))
        fixed_row = dict(run_scenario(scenario_id, seed=seed, output_root=output_root))
        gate_passed = agile_random_entry_gate(fixed_row)
        row = {
            "target_heading_deg": float(target),
            "scenario_id": scenario_id,
            "fixed_start_success": bool(fixed_row.get("fixed_start_success", False)),
            "actual_heading_change_deg": fixed_row.get("actual_heading_change_deg"),
            "heading_error_deg": fixed_row.get("heading_error_deg"),
            "min_wall_distance_m": fixed_row.get("min_wall_distance_m"),
            "height_change_m": fixed_row.get("height_change_m"),
            "terminal_speed_m_s": fixed_row.get("terminal_speed_m_s"),
            "max_alpha_deg": fixed_row.get("max_alpha_deg"),
            "saturation_fraction": fixed_row.get("saturation_fraction"),
            "exit_recoverable": fixed_row.get("exit_recoverable"),
            "failure_class": fixed_row.get("failure_class"),
            "feasibility_label": fixed_row.get("feasibility_label"),
            "random_entry_gate_passed": gate_passed,
            "updraft_stress_gate_passed": _agile_updraft_gate(fixed_row),
            "sweep_sample_count": 0,
            "sweep_success_rate": None,
            "sweep_rejection_count": None,
        }
        if run_sweeps and gate_passed:
            sweep_rows = run_entry_sweep(
                scenario_id=scenario_id,
                primitive=None,
                seed=seed,
                sample_count=int(sweep_samples),
                output_root=output_root,
            )
            summary = summarise_sweep_rows(sweep_rows)
            row.update(
                {
                    "sweep_sample_count": summary["sample_count"],
                    "sweep_success_rate": summary["success_rate"],
                    "sweep_rejection_count": summary["rejection_count"],
                    "sweep_heading_change_deg_mean": summary["heading_change_deg_mean"],
                    "sweep_min_wall_distance_m_min": summary["min_wall_distance_m_min"],
                    "sweep_height_change_m_mean": summary["height_change_m_mean"],
                    "sweep_terminal_speed_m_s_mean": summary["terminal_speed_m_s_mean"],
                    "sweep_max_alpha_deg_max": summary["max_alpha_deg_max"],
                    "sweep_saturation_fraction_mean": summary["saturation_fraction_mean"],
                }
            )
        rows.append(row)
    largest = _largest_feasible_target(rows)
    for row in rows:
        row["largest_fixed_start_feasible_target_deg"] = largest
    _write_summary(rows, seed=seed, output_root=output_root)
    return rows


def _target_scenario_id(target_heading_deg: float) -> str:
    return f"s9_agile_reversal_left_target_{int(round(target_heading_deg)):03d}_no_wind"


def _agile_updraft_gate(row: dict[str, object]) -> bool:
    return (
        abs(float(row.get("actual_heading_change_deg", row.get("heading_change_deg", 0.0))))
        >= 30.0
        and bool(row.get("fixed_start_success", False))
        and bool(row.get("exit_recoverable", False))
        and float(row.get("min_wall_distance_m", -1.0)) > 0.0
    )


def _largest_feasible_target(rows: list[dict[str, object]]) -> float | None:
    feasible = [
        float(row["target_heading_deg"])
        for row in rows
        if row.get("feasibility_label") == "fixed_start_feasible"
    ]
    return None if not feasible else max(feasible)


# =============================================================================
# 3) CSV and CLI Helpers
# =============================================================================
def _write_summary(
    rows: list[dict[str, object]],
    seed: int,
    output_root: str | Path | None,
) -> None:
    metrics_dir, _log_dir = _output_dirs(output_root)
    path = metrics_dir / f"s9_agile_feasibility_seed{int(seed)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sweep-samples", type=int, default=50)
    parser.add_argument("--run-sweeps", action="store_true")
    args = parser.parse_args()
    rows = run_agile_feasibility(
        seed=args.seed,
        output_root=args.output_root,
        sweep_samples=args.sweep_samples,
        run_sweeps=args.run_sweeps,
    )
    largest = rows[0]["largest_fixed_start_feasible_target_deg"] if rows else None
    print("agile feasibility complete")
    print(f"rows: {len(rows)}")
    print(f"largest_fixed_start_feasible_target_deg: {largest}")


if __name__ == "__main__":
    main()
