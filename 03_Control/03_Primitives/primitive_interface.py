from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from logging_contract import command_dataframe, metric_dataframe, trajectory_dataframe
from metric_contract import empty_metric_row, validate_metric_row
from primitive_contract import (
    PrimitiveEntrySet,
    PrimitiveExitCheck,
    PrimitiveSpec,
    validate_primitive_spec,
)
from result_paths import make_result_tree
from rollout import (
    CommandSchedule,
    RolloutConfig,
    RolloutResult,
    rollout_open_loop_normalised,
)
from state_contract import STATE_INDEX, STATE_NAMES, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data containers and constants
# 2) Entry-set evaluation
# 3) Exit-check evaluation
# 4) Smoke specification and execution
# 5) Output writing
# =============================================================================


# =============================================================================
# 1) Data Containers and Constants
# =============================================================================
PRIMITIVE_INTERFACE_CAMPAIGN = "03_primitive_interface"
SMOKE_NOTES = "primitive_interface_smoke_no_controller"
VALIDATION_COMMANDS = (
    "python -m py_compile "
    "03_Control/03_Primitives/primitive_interface.py "
    "03_Control/04_Scenarios/run_primitive_interface_smoke.py",
    "python 03_Control/04_Scenarios/run_primitive_interface_smoke.py "
    "--run-id 1 --overwrite",
    "python -m pytest -q "
    "03_Control/tests/test_primitive_interface.py "
    "03_Control/tests/test_primitive_interface_smoke.py",
    "python -m pytest -q 03_Control/tests",
)

ENTRY_UNITS = {
    "x_w": "m",
    "y_w": "m",
    "z_w": "m",
    "phi": "rad",
    "theta": "rad",
    "psi": "rad",
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "p": "rad/s",
    "q": "rad/s",
    "r": "rad/s",
    "delta_a": "rad",
    "delta_e": "rad",
    "delta_r": "rad",
    "speed_m_s": "m/s",
    "alpha_rad": "rad",
    "beta_rad": "rad",
    "bank_rad": "rad",
    "pitch_rad": "rad",
    "yaw_rad": "rad",
}


@dataclass(frozen=True)
class EntryCheckResult:
    variable: str
    value: float
    lower: float
    upper: float
    pass_check: bool
    units: str
    reason: str


@dataclass(frozen=True)
class ExitCheckResult:
    name: str
    value: float | bool
    pass_check: bool
    required: bool
    reason: str


@dataclass(frozen=True)
class PrimitiveExecutionConfig:
    dt_s: float
    t_final_s: float
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    seed: int = 1
    scenario_name: str = "primitive_interface_smoke"


@dataclass(frozen=True)
class PrimitiveExecutionResult:
    primitive_spec: PrimitiveSpec
    entry_pass: bool
    exit_pass: bool
    entry_checks: tuple[EntryCheckResult, ...]
    exit_checks: tuple[ExitCheckResult, ...]
    rollout_result: RolloutResult | None
    metrics: dict[str, object]
    notes: str


# =============================================================================
# 2) Entry-Set Evaluation
# =============================================================================
def state_entry_variables(x: np.ndarray) -> dict[str, float]:
    """Return direct and derived entry variables from the canonical 15-state vector."""

    state = as_state_vector(x)
    values = {name: float(state[index]) for index, name in enumerate(STATE_NAMES)}
    u_b = float(state[STATE_INDEX["u"]])
    v_b = float(state[STATE_INDEX["v"]])
    w_b = float(state[STATE_INDEX["w"]])
    speed_m_s = float(np.linalg.norm([u_b, v_b, w_b]))
    if speed_m_s > 1e-12:
        beta_rad = float(np.arcsin(np.clip(v_b / speed_m_s, -1.0, 1.0)))
    else:
        beta_rad = 0.0
    # Entry comparisons use body-axis velocity and radians, matching the plant contract.
    values.update(
        {
            "speed_m_s": speed_m_s,
            "alpha_rad": float(np.arctan2(w_b, u_b)),
            "beta_rad": beta_rad,
            "bank_rad": float(state[STATE_INDEX["phi"]]),
            "pitch_rad": float(state[STATE_INDEX["theta"]]),
            "yaw_rad": float(state[STATE_INDEX["psi"]]),
        }
    )
    return values


def evaluate_entry_set(
    x: np.ndarray,
    entry_set: PrimitiveEntrySet,
) -> tuple[EntryCheckResult, ...]:
    """Evaluate a primitive entry set without modifying the state."""

    if set(entry_set.lower) != set(entry_set.upper):
        raise ValueError("entry set lower and upper keys must match.")
    variables = state_entry_variables(x)
    results: list[EntryCheckResult] = []
    for variable in sorted(entry_set.lower):
        if variable not in variables:
            raise ValueError(f"unknown entry-set variable: {variable}.")
        lower = float(entry_set.lower[variable])
        upper = float(entry_set.upper[variable])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("entry-set bounds must be finite.")
        if lower > upper:
            raise ValueError(f"entry-set lower bound exceeds upper bound for {variable}.")
        value = float(variables[variable])
        pass_check = bool(lower <= value <= upper)
        reason = "inside_bounds" if pass_check else "outside_bounds"
        results.append(
            EntryCheckResult(
                variable=variable,
                value=value,
                lower=lower,
                upper=upper,
                pass_check=pass_check,
                units=ENTRY_UNITS.get(variable, "-"),
                reason=reason,
            )
        )
    return tuple(results)


# =============================================================================
# 3) Exit-Check Evaluation
# =============================================================================
def _minimum_true_safe_margin(x_log: np.ndarray) -> float:
    margins = [
        position_margin_m(position_w, TRUE_SAFE_BOUNDS)["min_margin_m"]
        for position_w in np.asarray(x_log, dtype=float)[:, 0:3]
        if np.all(np.isfinite(position_w))
    ]
    if not margins:
        return float("-inf")
    return float(np.min(margins))


def evaluate_exit_checks(
    spec: PrimitiveSpec,
    rollout_result: RolloutResult | None,
) -> tuple[ExitCheckResult, ...]:
    """Evaluate declared exit checks from rollout evidence."""

    results: list[ExitCheckResult] = []
    for check in spec.exit_checks:
        if rollout_result is None:
            results.append(
                ExitCheckResult(
                    name=check.name,
                    value=False,
                    pass_check=False,
                    required=bool(check.required),
                    reason="no_rollout_evidence",
                )
            )
            continue

        if check.name == "finite_state":
            pass_check = bool(np.all(np.isfinite(rollout_result.x)))
            results.append(
                ExitCheckResult(
                    name=check.name,
                    value=pass_check,
                    pass_check=pass_check,
                    required=bool(check.required),
                    reason="full_state_history_finite"
                    if pass_check
                    else "nonfinite_state_history",
                )
            )
        elif check.name == "true_safe_margin":
            min_margin_m = _minimum_true_safe_margin(rollout_result.x)
            pass_check = bool(min_margin_m >= 0.0)
            results.append(
                ExitCheckResult(
                    name=check.name,
                    value=min_margin_m,
                    pass_check=pass_check,
                    required=bool(check.required),
                    reason="trajectory_inside_true_safe_bounds"
                    if pass_check
                    else "trajectory_outside_true_safe_bounds",
                )
            )
        elif check.name == "rollout_success":
            pass_check = bool(rollout_result.metrics.get("rollout_success", False))
            results.append(
                ExitCheckResult(
                    name=check.name,
                    value=pass_check,
                    pass_check=pass_check,
                    required=bool(check.required),
                    reason="rollout_metric_success"
                    if pass_check
                    else "rollout_metric_failure",
                )
            )
        else:
            results.append(
                ExitCheckResult(
                    name=check.name,
                    value=False,
                    pass_check=False,
                    required=bool(check.required),
                    reason="unknown_exit_check",
                )
            )
    return tuple(results)


# =============================================================================
# 4) Smoke Specification and Execution
# =============================================================================
def build_interface_smoke_spec() -> PrimitiveSpec:
    """Return a contract-only primitive spec for interface smoke testing."""

    entry_set = PrimitiveEntrySet(
        name="glide_interface_smoke_entry",
        description="contract-only finite no-wind smoke entry set",
        lower={
            "speed_m_s": 4.5,
            "x_w": 1.2,
            "y_w": 0.0,
            "z_w": 0.0,
            "phi": -0.35,
            "theta": -0.35,
            "p": -1.0,
            "q": -1.0,
            "r": -1.0,
        },
        upper={
            "speed_m_s": 8.5,
            "x_w": 6.6,
            "y_w": 4.4,
            "z_w": 3.0,
            "phi": 0.35,
            "theta": 0.35,
            "p": 1.0,
            "q": 1.0,
            "r": 1.0,
        },
    )
    exit_checks = (
        PrimitiveExitCheck(
            name="finite_state",
            description="full rollout state history remains finite",
            required=True,
        ),
        PrimitiveExitCheck(
            name="true_safe_margin",
            description="minimum true-safety margin stays nonnegative",
            required=True,
        ),
        PrimitiveExitCheck(
            name="rollout_success",
            description="rollout integrity metric reports success",
            required=True,
        ),
    )
    return PrimitiveSpec(
        name="glide_interface_smoke_contract",
        family="glide",
        duration_s=0.24,
        entry_set=entry_set,
        exit_checks=exit_checks,
        metadata={
            "contract_only": "true",
            "actual_glide_primitive_implemented": "false",
        },
    )


def _safe_initial_metrics(x0: np.ndarray) -> dict[str, float]:
    state = as_state_vector(x0)
    variables = state_entry_variables(state)
    margins = position_margin_m(state[0:3], TRUE_SAFE_BOUNDS)
    return {
        "speed_m_s": variables["speed_m_s"],
        "alpha_deg": float(np.rad2deg(variables["alpha_rad"])),
        "beta_deg": float(np.rad2deg(variables["beta_rad"])),
        "bank_deg": float(np.rad2deg(state[STATE_INDEX["phi"]])),
        "pitch_deg": float(np.rad2deg(state[STATE_INDEX["theta"]])),
        "rate_norm_rad_s": float(np.linalg.norm(state[9:12])),
        "min_true_wall_margin_m": margins["min_wall_margin_m"],
        "min_floor_margin_m": margins["floor_margin_m"],
        "min_ceiling_margin_m": margins["ceiling_margin_m"],
    }


def _metric_from_inputs(
    spec: PrimitiveSpec,
    x0: np.ndarray,
    config: PrimitiveExecutionConfig,
    rollout_result: RolloutResult | None,
    failure_label: str,
    notes: str,
) -> dict[str, object]:
    if rollout_result is not None:
        row = dict(rollout_result.metrics)
    else:
        row = empty_metric_row(include_agile=True)
        initial = _safe_initial_metrics(x0)
        row.update(
            {
                "duration_s": 0.0,
                "initial_speed_m_s": initial["speed_m_s"],
                "terminal_speed_m_s": initial["speed_m_s"],
                "height_change_m": 0.0,
                "min_true_wall_margin_m": initial["min_true_wall_margin_m"],
                "min_floor_margin_m": initial["min_floor_margin_m"],
                "min_ceiling_margin_m": initial["min_ceiling_margin_m"],
                "max_alpha_deg": abs(initial["alpha_deg"]),
                "max_beta_deg": abs(initial["beta_deg"]),
                "max_bank_deg": abs(initial["bank_deg"]),
                "max_pitch_deg": abs(initial["pitch_deg"]),
                "max_rate_rad_s": initial["rate_norm_rad_s"],
                "saturation_fraction": 0.0,
                "saturation_time_s": 0.0,
            }
        )
    row.update(
        {
            "run_id": "",
            "seed": int(config.seed),
            "primitive_name": spec.name,
            "primitive_family": spec.family,
            "scenario_name": config.scenario_name,
            "wind_mode": config.wind_mode,
            "latency_case": config.latency_case,
            "success": False,
            "primitive_success": False,
            "closed_loop_replay_success": False,
            "failure_label": failure_label,
            "notes": notes,
        }
    )
    if rollout_result is None:
        row["finite_state_success"] = bool(np.all(np.isfinite(as_state_vector(x0))))
        row["rollout_success"] = False
    validate_metric_row(row)
    return row


def _rollout_config(config: PrimitiveExecutionConfig) -> RolloutConfig:
    return RolloutConfig(
        dt_s=float(config.dt_s),
        t_final_s=float(config.t_final_s),
        wind_mode=config.wind_mode,
        latency_case=config.latency_case,
        actuator_tau_s=config.actuator_tau_s,
    )


def _execution_failure_label(
    entry_pass: bool,
    exit_pass: bool,
    rollout_result: RolloutResult | None,
) -> str:
    if not entry_pass:
        return "entry_set_violation"
    if rollout_result is not None and rollout_result.failure_label != "not_run":
        return rollout_result.failure_label
    if not exit_pass:
        return "terminal_recovery_limited"
    return "not_run"


def execute_open_loop_primitive_interface(
    spec: PrimitiveSpec,
    x0: np.ndarray,
    command_schedule: CommandSchedule,
    config: PrimitiveExecutionConfig,
    aircraft: object | None = None,
    wind_model: object = None,
) -> PrimitiveExecutionResult:
    """Check entry set, run open-loop rollout if admissible, and report evidence."""

    validate_primitive_spec(spec)
    state = as_state_vector(x0)
    entry_checks = evaluate_entry_set(state, spec.entry_set)
    entry_pass = all(check.pass_check for check in entry_checks)

    rollout_result = None
    exit_checks: tuple[ExitCheckResult, ...] = ()
    if entry_pass:
        rollout_result = rollout_open_loop_normalised(
            state,
            command_schedule,
            _rollout_config(config),
            aircraft=aircraft,
            wind_model=wind_model,
            seed=config.seed,
            scenario_name=config.scenario_name,
        )
        exit_checks = evaluate_exit_checks(spec, rollout_result)

    exit_pass = bool(
        exit_checks
        and all(check.pass_check for check in exit_checks if check.required)
    )
    failure_label = _execution_failure_label(entry_pass, exit_pass, rollout_result)
    notes = SMOKE_NOTES if entry_pass else "entry_set_violation"
    metrics = _metric_from_inputs(
        spec,
        state,
        config,
        rollout_result,
        failure_label,
        notes,
    )
    return PrimitiveExecutionResult(
        primitive_spec=spec,
        entry_pass=entry_pass,
        exit_pass=exit_pass,
        entry_checks=entry_checks,
        exit_checks=exit_checks,
        rollout_result=rollout_result,
        metrics=metrics,
        notes=notes,
    )


# =============================================================================
# 5) Output Writing
# =============================================================================
def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _entry_checks_dataframe(checks: tuple[EntryCheckResult, ...]) -> pd.DataFrame:
    return pd.DataFrame([check.__dict__ for check in checks])


def _exit_checks_dataframe(checks: tuple[ExitCheckResult, ...]) -> pd.DataFrame:
    return pd.DataFrame([check.__dict__ for check in checks])


def _remove_stale_optional_outputs(paths: dict[str, Path], keys: tuple[str, ...]) -> None:
    for key in keys:
        path = paths[key]
        if path.exists():
            path.unlink()


def _interface_status(result: PrimitiveExecutionResult) -> dict[str, bool | str]:
    rollout_ran = result.rollout_result is not None
    entry_checks_pass = bool(result.entry_pass)
    exit_checks_pass = bool(result.exit_pass)
    implementation_flags_clear = True
    interface_checks_pass = bool(
        entry_checks_pass
        and exit_checks_pass
        and rollout_ran
        and bool(result.metrics.get("rollout_success", False))
        and implementation_flags_clear
    )
    return {
        "overall_status": "pass" if interface_checks_pass else "needs_review",
        "interface_checks_pass": interface_checks_pass,
        "entry_checks_pass": entry_checks_pass,
        "exit_checks_pass": exit_checks_pass,
        "rollout_ran": rollout_ran,
    }


def _manifest(
    result: PrimitiveExecutionResult,
    run_id: int,
    output_files: dict[str, Path],
) -> dict[str, Any]:
    status = _interface_status(result)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        **status,
        "primitive_interface_implemented": True,
        "primitive_implemented": False,
        "primitive_controller_implemented": False,
        "controller_implemented": False,
        "actual_glide_primitive_implemented": False,
        "actual_bank_primitive_implemented": False,
        "actual_recovery_primitive_implemented": False,
        "actual_agile_reversal_primitive_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "vicon_implemented": False,
        "hardware_implemented": False,
        "high_incidence_validation_claim": False,
        "primitive_success": bool(result.metrics["primitive_success"]),
        "success": bool(result.metrics["success"]),
        "failure_label": result.metrics["failure_label"],
        "notes": result.notes,
        "command_bridge": "normalised_command_to_surface_rad",
        "state_derivative_command_input": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "output_files": {
            name: _repo_relative(path)
            for name, path in output_files.items()
        },
        "validation_commands": list(VALIDATION_COMMANDS),
    }


def _write_report(
    path: Path,
    result: PrimitiveExecutionResult,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Primitive Interface Smoke Report",
        "",
        "This is primitive-interface execution evidence only. It connects the",
        "primitive metadata contract to the existing open-loop rollout/logging",
        "base, but it is not a real glide primitive result.",
        "",
        "It does not implement a controller, OCP, TVLQR, governor, outer loop,",
        "Vicon interface, hardware path, or high-incidence validation.",
        "",
        "## Status",
        "",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Interface checks pass: `{manifest['interface_checks_pass']}`",
        f"- Entry checks pass: `{manifest['entry_checks_pass']}`",
        f"- Exit checks pass: `{manifest['exit_checks_pass']}`",
        f"- Rollout ran: `{manifest['rollout_ran']}`",
        f"- Final primitive success: `{manifest['primitive_success']}`",
        f"- Final success: `{manifest['success']}`",
        f"- Failure label: `{manifest['failure_label']}`",
        f"- Notes: `{result.notes}`",
        "",
        "## Command Path",
        "",
        "- Requested normalised command: `u_norm_requested`.",
        "- Applied normalised command: `u_norm_applied`, clipped by the rollout layer.",
        "- Plant command: `delta_cmd_rad`, produced by `normalised_command_to_surface_rad`.",
        "- `state_derivative` receives `delta_cmd_rad`, never raw normalised commands.",
        "",
        "## Implementation Flags",
        "",
        f"- Primitive interface implemented: `{manifest['primitive_interface_implemented']}`",
        f"- Actual glide primitive implemented: `{manifest['actual_glide_primitive_implemented']}`",
        f"- Controller implemented: `{manifest['controller_implemented']}`",
        f"- OCP implemented: `{manifest['ocp_implemented']}`",
        f"- TVLQR implemented: `{manifest['tvlqr_implemented']}`",
        f"- Governor implemented: `{manifest['governor_implemented']}`",
        f"- Outer loop implemented: `{manifest['outer_loop_implemented']}`",
        f"- High-incidence validation claim: `{manifest['high_incidence_validation_claim']}`",
        "",
        "## Next Step",
        "",
        "Implement the first actual primitive family, likely glide, using this",
        "interface and keeping primitive success separate from rollout integrity.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def write_primitive_interface_outputs(
    result: PrimitiveExecutionResult,
    result_root: Path,
    campaign: str,
    run_id: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write primitive-interface smoke evidence."""

    paths = make_result_tree(Path(result_root), campaign, run_id, overwrite=overwrite)
    suffix = f"s{run_id:03d}"
    output_paths = {
        "entry_checks_csv": paths["metrics"] / f"entry_checks_{suffix}.csv",
        "exit_checks_csv": paths["metrics"] / f"exit_checks_{suffix}.csv",
        "trajectory_csv": paths["metrics"] / f"trajectory_{suffix}.csv",
        "commands_csv": paths["metrics"] / f"commands_{suffix}.csv",
        "metrics_csv": paths["metrics"] / f"primitive_interface_metrics_{suffix}.csv",
        "manifest_json": paths["manifests"]
        / f"primitive_interface_manifest_{suffix}.json",
        "report_md": paths["reports"] / f"primitive_interface_report_{suffix}.md",
    }
    _entry_checks_dataframe(result.entry_checks).to_csv(
        output_paths["entry_checks_csv"],
        index=False,
    )
    if result.exit_checks:
        _exit_checks_dataframe(result.exit_checks).to_csv(
            output_paths["exit_checks_csv"],
            index=False,
        )
    else:
        _remove_stale_optional_outputs(output_paths, ("exit_checks_csv",))
    if result.rollout_result is not None:
        rollout = result.rollout_result
        trajectory_dataframe(rollout.time_s, rollout.x).to_csv(
            output_paths["trajectory_csv"],
            index=False,
        )
        command_dataframe(
            rollout.time_s,
            rollout.u_norm_requested,
            rollout.u_norm_applied,
            rollout.delta_cmd_rad,
        ).to_csv(output_paths["commands_csv"], index=False)
    else:
        _remove_stale_optional_outputs(
            output_paths,
            ("trajectory_csv", "commands_csv"),
        )
    metric_row = dict(result.metrics)
    metric_row["run_id"] = suffix
    metric_dataframe(metric_row).to_csv(output_paths["metrics_csv"], index=False)
    written_paths = {
        key: path for key, path in output_paths.items() if path.exists()
    }
    manifest_paths = {
        **written_paths,
        "manifest_json": output_paths["manifest_json"],
        "report_md": output_paths["report_md"],
    }
    manifest = _manifest(result, run_id, manifest_paths)
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    written_paths["manifest_json"] = output_paths["manifest_json"]
    _write_report(output_paths["report_md"], result, manifest)
    written_paths["report_md"] = output_paths["report_md"]
    written_paths["root"] = paths["root"]
    return written_paths
