from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from command_contract import COMMAND_NAMES, as_surface_command_rad
from flight_dynamics import AircraftModel, adapt_glider, state_derivative
from glider import build_nausicaa_glider
from linearisation import (
    INPUT_INDEX,
    INPUT_NAMES,
    LATERAL_INPUTS,
    LATERAL_STATES,
    LONGITUDINAL_INPUTS,
    LONGITUDINAL_STATES,
    STATE_INDEX,
    STATE_NAMES as LINEAR_STATE_NAMES,
    LinearModel,
    key_derivatives,
    linearise_trim,
    reduced_model,
)
from result_paths import make_result_tree
from state_contract import STATE_NAMES as CONTRACT_STATE_NAMES
from trim_solver import TrimResult, TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and data containers
# 2) Public case and residual helpers
# 3) Jacobian and reduced-model helpers
# 4) Case execution helpers
# 5) Output writers
# 6) Public audit workflow
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
DEFAULT_RESULTS_ROOT = CONTROL_DIR / "05_Results"
TRIM_LINEARISATION_CAMPAIGN = "02_trim_linearisation"

DYNAMIC_RESIDUAL_NORM_TOL = 1e-6
FINITE_DIFFERENCE_EPS = 1e-6
JACOBIAN_MAX_ABS_TOL = 1e-4
ACTUATOR_JACOBIAN_ABS_TOL = 1e-9
UNIFORM_WIND_MAX_ABS_DIFF_TOL = 1e-9
UNIFORM_WIND_W_M_S = (0.0, 0.0, 0.5)

VALIDATION_COMMANDS = (
    "python -m py_compile "
    "03_Control/04_Scenarios/trim_linearisation_audit.py "
    "03_Control/04_Scenarios/run_trim_linearisation_audit.py",
    "python 03_Control/04_Scenarios/run_trim_linearisation_audit.py "
    "--run-id 1 --overwrite",
    "python -m pytest -q 03_Control/tests/test_trim_linearisation_audit.py",
    "python -m pytest -q 03_Control/tests",
)

FUTURE_CASES = (
    "mild_bank_segment",
    "recovery_segment",
    "agile_reversal_knots",
    "high_alpha_braking_segment",
)

SIGN_EXPECTATIONS = {
    "m_delta_e": "positive",
    "l_delta_a": "positive",
    "n_delta_r": "positive",
}


@dataclass(frozen=True)
class TrimLinearisationAuditCase:
    name: str
    speed_m_s: float
    altitude_m: float = 1.5
    wind_mode: str = "none"
    wind_w_m_s: tuple[float, float, float] | None = None
    rho_kg_m3: float = 1.225
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    required: bool = True


@dataclass(frozen=True)
class _ExecutedCase:
    case: TrimLinearisationAuditCase
    role: str
    target: TrimTarget
    trim_result: TrimResult | None
    x_dot: np.ndarray | None
    linear_model: LinearModel | None
    status: str
    error: str


# =============================================================================
# 2) Public Case and Residual Helpers
# =============================================================================
def build_trim_linearisation_cases() -> tuple[TrimLinearisationAuditCase, ...]:
    """Return deterministic trim and linearisation audit cases."""

    return (
        TrimLinearisationAuditCase(
            name="natural_glide_6p5_none",
            speed_m_s=6.5,
            required=True,
        ),
        TrimLinearisationAuditCase(
            name="natural_glide_5p5_none",
            speed_m_s=5.5,
            required=False,
        ),
        TrimLinearisationAuditCase(
            name="natural_glide_7p5_none",
            speed_m_s=7.5,
            required=False,
        ),
        TrimLinearisationAuditCase(
            name="natural_glide_4p5_none",
            speed_m_s=4.5,
            required=False,
        ),
        TrimLinearisationAuditCase(
            name="uniform_cg_updraft_6p5",
            speed_m_s=6.5,
            wind_mode="cg",
            wind_w_m_s=UNIFORM_WIND_W_M_S,
            required=False,
        ),
    )


def dynamic_residual_from_xdot(x_dot: np.ndarray) -> dict[str, float]:
    """Return trim residual norms, excluding position rates from dynamic residual."""

    derivative = np.asarray(x_dot, dtype=float)
    if derivative.size != len(LINEAR_STATE_NAMES):
        raise ValueError("state derivative must contain 15 values.")
    derivative = derivative.reshape(len(LINEAR_STATE_NAMES))
    return {
        "full_xdot_norm": float(np.linalg.norm(derivative)),
        "position_rate_norm": float(np.linalg.norm(derivative[0:3])),
        "attitude_rate_norm": float(np.linalg.norm(derivative[3:6])),
        "body_accel_norm": float(np.linalg.norm(derivative[6:9])),
        "angular_accel_norm": float(np.linalg.norm(derivative[9:12])),
        "actuator_residual_norm": float(np.linalg.norm(derivative[12:15])),
        "dynamic_residual_norm": float(np.linalg.norm(derivative[3:15])),
    }


# =============================================================================
# 3) Jacobian and Reduced-Model Helpers
# =============================================================================
def finite_difference_linearisation(
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    aircraft: AircraftModel,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
    wind_model: object = None,
    wind_mode: str = "none",
    eps: float = FINITE_DIFFERENCE_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return central finite-difference A and B matrices around a trim point."""

    x0 = np.asarray(x_trim, dtype=float).reshape(len(LINEAR_STATE_NAMES))
    u0 = as_surface_command_rad(u_trim)
    epsilon = float(eps)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("finite-difference epsilon must be finite and positive.")

    a_fd = np.empty((x0.size, x0.size), dtype=float)
    b_fd = np.empty((x0.size, u0.size), dtype=float)
    for column in range(x0.size):
        perturb = np.zeros_like(x0)
        perturb[column] = epsilon
        f_plus = state_derivative(
            x0 + perturb,
            u0,
            aircraft,
            wind_model=wind_model,
            rho=rho_kg_m3,
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
        )
        f_minus = state_derivative(
            x0 - perturb,
            u0,
            aircraft,
            wind_model=wind_model,
            rho=rho_kg_m3,
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
        )
        a_fd[:, column] = (f_plus - f_minus) / (2.0 * epsilon)

    for column in range(u0.size):
        perturb = np.zeros_like(u0)
        perturb[column] = epsilon
        u_plus = as_surface_command_rad(u0 + perturb)
        u_minus = as_surface_command_rad(u0 - perturb)
        f_plus = state_derivative(
            x0,
            u_plus,
            aircraft,
            wind_model=wind_model,
            rho=rho_kg_m3,
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
        )
        f_minus = state_derivative(
            x0,
            u_minus,
            aircraft,
            wind_model=wind_model,
            rho=rho_kg_m3,
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
        )
        b_fd[:, column] = (f_plus - f_minus) / (2.0 * epsilon)
    return a_fd, b_fd


def compare_jacobians(
    a_symbolic: np.ndarray,
    b_symbolic: np.ndarray,
    a_fd: np.ndarray,
    b_fd: np.ndarray,
) -> dict[str, float | bool]:
    """Compare symbolic and finite-difference Jacobians using audit tolerances."""

    a_error = np.asarray(a_symbolic, dtype=float) - np.asarray(a_fd, dtype=float)
    b_error = np.asarray(b_symbolic, dtype=float) - np.asarray(b_fd, dtype=float)
    finite = bool(np.all(np.isfinite(a_error)) and np.all(np.isfinite(b_error)))
    a_max = float(np.max(np.abs(a_error))) if a_error.size else float("nan")
    b_max = float(np.max(np.abs(b_error))) if b_error.size else float("nan")
    a_rms = float(np.sqrt(np.mean(a_error**2))) if a_error.size else float("nan")
    b_rms = float(np.sqrt(np.mean(b_error**2))) if b_error.size else float("nan")
    a_index = int(np.argmax(np.abs(a_error))) if a_error.size else 0
    b_index = int(np.argmax(np.abs(b_error))) if b_error.size else 0
    return {
        "a_max_abs_error": a_max,
        "b_max_abs_error": b_max,
        "a_rms_error": a_rms,
        "b_rms_error": b_rms,
        "a_largest_error_row": float(np.unravel_index(a_index, a_error.shape)[0]),
        "a_largest_error_col": float(np.unravel_index(a_index, a_error.shape)[1]),
        "b_largest_error_row": float(np.unravel_index(b_index, b_error.shape)[0]),
        "b_largest_error_col": float(np.unravel_index(b_index, b_error.shape)[1]),
        "finite": finite,
        "pass": bool(
            finite
            and a_max <= JACOBIAN_MAX_ABS_TOL
            and b_max <= JACOBIAN_MAX_ABS_TOL
        ),
    }


def _controllability_rank(a_matrix: np.ndarray, b_matrix: np.ndarray) -> int:
    block = b_matrix
    power = np.eye(a_matrix.shape[0])
    parts = []
    for _ in range(a_matrix.shape[0]):
        parts.append(power @ block)
        power = a_matrix @ power
    return int(np.linalg.matrix_rank(np.hstack(parts)))


def reduced_model_audit_rows(model: object) -> pd.DataFrame:
    """Return reduced longitudinal and lateral model diagnostics."""

    if not isinstance(model, LinearModel):
        raise TypeError("model must be a LinearModel instance.")
    specs = (
        ("longitudinal", LONGITUDINAL_STATES, LONGITUDINAL_INPUTS, (5, 5), (5, 1)),
        ("lateral", LATERAL_STATES, LATERAL_INPUTS, (6, 6), (6, 2)),
    )
    rows: list[dict[str, object]] = []
    for name, state_names, input_names, expected_a, expected_b in specs:
        a_red, b_red = reduced_model(model, state_names, input_names)
        shape_pass = a_red.shape == expected_a and b_red.shape == expected_b
        finite_entries = bool(np.all(np.isfinite(a_red)) and np.all(np.isfinite(b_red)))
        rows.append(
            {
                "model_name": name,
                "state_names": ",".join(state_names),
                "input_names": ",".join(input_names),
                "a_rows": int(a_red.shape[0]),
                "a_cols": int(a_red.shape[1]),
                "b_rows": int(b_red.shape[0]),
                "b_cols": int(b_red.shape[1]),
                "expected_a_shape": str(expected_a),
                "expected_b_shape": str(expected_b),
                "shape_pass": bool(shape_pass),
                "finite_entries": finite_entries,
                "controllability_rank": _controllability_rank(a_red, b_red),
                "max_real_eigenvalue": float(np.max(np.real(np.linalg.eigvals(a_red)))),
                "rank_is_diagnostic_only": True,
                "eigenvalues_are_diagnostic_only": True,
                "hard_gate_pass": bool(shape_pass and finite_entries),
            }
        )
    return pd.DataFrame(rows)


def uniform_wind_consistency_check(
    x: np.ndarray,
    u_cmd: np.ndarray,
    aircraft: AircraftModel,
    wind_w_m_s: tuple[float, float, float] = UNIFORM_WIND_W_M_S,
) -> dict[str, float | bool]:
    """Compare numeric state derivatives for uniform CG and panel wind modes."""

    state = np.asarray(x, dtype=float).reshape(len(LINEAR_STATE_NAMES))
    command = as_surface_command_rad(u_cmd)
    wind = np.asarray(wind_w_m_s, dtype=float).reshape(3)
    cg_xdot = state_derivative(
        state,
        command,
        aircraft,
        wind_model=wind,
        wind_mode="cg",
    )
    panel_xdot = state_derivative(
        state,
        command,
        aircraft,
        wind_model=wind,
        wind_mode="panel",
    )
    diff = np.asarray(cg_xdot, dtype=float) - np.asarray(panel_xdot, dtype=float)
    max_abs = float(np.max(np.abs(diff)))
    return {
        "wind_x_w_m_s": float(wind[0]),
        "wind_y_w_m_s": float(wind[1]),
        "wind_z_w_m_s": float(wind[2]),
        "max_abs_xdot_diff": max_abs,
        "finite": bool(np.all(np.isfinite(diff))),
        "pass": bool(np.all(np.isfinite(diff)) and max_abs <= UNIFORM_WIND_MAX_ABS_DIFF_TOL),
    }


# =============================================================================
# 4) Case Execution Helpers
# =============================================================================
def _case_role(case: TrimLinearisationAuditCase) -> str:
    if case.name == "natural_glide_4p5_none":
        return "diagnostic"
    if case.wind_w_m_s is not None:
        return "optional_wind"
    if case.required:
        return "required"
    return "optional"


def _target_for_case(case: TrimLinearisationAuditCase) -> TrimTarget:
    wind_model = None
    if case.wind_w_m_s is not None:
        wind_model = np.asarray(case.wind_w_m_s, dtype=float)
    return TrimTarget(
        speed_m_s=float(case.speed_m_s),
        altitude_m=float(case.altitude_m),
        rho_kg_m3=float(case.rho_kg_m3),
        wind_model=wind_model,
        wind_mode=case.wind_mode,
        actuator_tau_s=case.actuator_tau_s,
    )


def _preflight_checks() -> dict[str, bool]:
    state_order_pass = tuple(CONTRACT_STATE_NAMES) == tuple(LINEAR_STATE_NAMES)
    command_order_pass = tuple(COMMAND_NAMES) == tuple(INPUT_NAMES)
    if not state_order_pass:
        raise RuntimeError("state contract order does not match linearisation state order.")
    if not command_order_pass:
        raise RuntimeError("command contract order does not match linearisation input order.")
    return {
        "state_order_pass": state_order_pass,
        "command_order_pass": command_order_pass,
        "trim_solver_importable": callable(solve_straight_trim),
        "linearise_trim_importable": callable(linearise_trim),
        "state_derivative_importable": callable(state_derivative),
    }


def _execute_case(
    case: TrimLinearisationAuditCase,
    aircraft: AircraftModel,
) -> _ExecutedCase:
    role = _case_role(case)
    target = _target_for_case(case)
    try:
        trim_result = solve_straight_trim(aircraft=aircraft, target=target)
        command = as_surface_command_rad(trim_result.u_cmd_trim)
        x_dot = state_derivative(
            trim_result.x_trim,
            command,
            aircraft,
            wind_model=target.wind_model,
            rho=target.rho_kg_m3,
            actuator_tau_s=target.actuator_tau_s,
            wind_mode=target.wind_mode,
        )
        linear_model = None
        if target.wind_model is None:
            linear_model = linearise_trim(
                aircraft=aircraft,
                trim_result=trim_result,
                target=target,
            )
            as_surface_command_rad(linear_model.u_trim)
        status = "converged" if trim_result.converged else "solver_failed"
        return _ExecutedCase(
            case=case,
            role=role,
            target=target,
            trim_result=trim_result,
            x_dot=x_dot,
            linear_model=linear_model,
            status=status,
            error="",
        )
    except Exception as exc:  # noqa: BLE001 - report audit failures without hiding rows.
        return _ExecutedCase(
            case=case,
            role=role,
            target=target,
            trim_result=None,
            x_dot=None,
            linear_model=None,
            status="exception",
            error=f"{type(exc).__name__}: {exc}",
        )


def _empty_state_command_columns(row: dict[str, object]) -> None:
    for name in LINEAR_STATE_NAMES:
        row[f"x_trim_{name}"] = np.nan
    for name in INPUT_NAMES:
        row[f"u_trim_{name}"] = np.nan


def _trim_case_rows(executed_cases: tuple[_ExecutedCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for executed in executed_cases:
        case = executed.case
        row: dict[str, object] = {
            "case_name": case.name,
            "role": executed.role,
            "required": bool(case.required),
            "speed_m_s": float(case.speed_m_s),
            "altitude_m": float(case.altitude_m),
            "wind_mode": case.wind_mode,
            "wind_x_w_m_s": np.nan,
            "wind_y_w_m_s": np.nan,
            "wind_z_w_m_s": np.nan,
            "status": executed.status,
            "converged": False,
            "linearisation_status": "not_run",
            "error": executed.error,
        }
        if case.wind_w_m_s is not None:
            row["wind_x_w_m_s"] = float(case.wind_w_m_s[0])
            row["wind_y_w_m_s"] = float(case.wind_w_m_s[1])
            row["wind_z_w_m_s"] = float(case.wind_w_m_s[2])
        _empty_state_command_columns(row)
        if executed.trim_result is not None:
            trim = executed.trim_result
            command = as_surface_command_rad(trim.u_cmd_trim)
            row.update(
                {
                    "converged": bool(trim.converged),
                    "alpha_rad": float(trim.alpha_rad),
                    "theta_rad": float(trim.theta_rad),
                    "gamma_rad": float(trim.gamma_rad),
                    "sink_rate_m_s": float(trim.sink_rate_m_s),
                    "solver_return_status": str(trim.solver_stats.get("return_status")),
                    "solver_success": bool(trim.solver_stats.get("success", False)),
                    "solver_iter_count": int(trim.solver_stats.get("iter_count", 0)),
                    "x_force_b": float(trim.x_force_b),
                    "y_force_b": float(trim.y_force_b),
                    "z_force_b": float(trim.z_force_b),
                    "l_moment_b": float(trim.l_moment_b),
                    "m_moment_b": float(trim.m_moment_b),
                    "n_moment_b": float(trim.n_moment_b),
                    "linearisation_status": (
                        "linearised"
                        if executed.linear_model is not None
                        else "not_run_nonzero_wind"
                    ),
                }
            )
            for index, name in enumerate(LINEAR_STATE_NAMES):
                row[f"x_trim_{name}"] = float(trim.x_trim[index])
            for index, name in enumerate(INPUT_NAMES):
                row[f"u_trim_{name}"] = float(command[index])
        rows.append(row)
    for future_case in FUTURE_CASES:
        row = {
            "case_name": future_case,
            "role": "future_trajectory_knot",
            "required": False,
            "speed_m_s": np.nan,
            "altitude_m": np.nan,
            "wind_mode": "not_available_yet",
            "status": "not_available_yet",
            "converged": False,
            "linearisation_status": "not_available_yet",
            "error": "",
        }
        _empty_state_command_columns(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _dynamic_residual_rows(executed_cases: tuple[_ExecutedCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for executed in executed_cases:
        if executed.x_dot is None:
            continue
        residual = dynamic_residual_from_xdot(executed.x_dot)
        residual_pass = residual["dynamic_residual_norm"] <= DYNAMIC_RESIDUAL_NORM_TOL
        rows.append(
            {
                "case_name": executed.case.name,
                "role": executed.role,
                "speed_m_s": float(executed.case.speed_m_s),
                "wind_mode": executed.case.wind_mode,
                "dynamic_residual_norm_tol": DYNAMIC_RESIDUAL_NORM_TOL,
                "dynamic_residual_excludes_position_rates": True,
                "hard_gate_case": executed.role in {"required", "optional"},
                "pass": bool(residual_pass),
                **residual,
            }
        )
    return pd.DataFrame(rows)


def _linearisation_rows(executed_cases: tuple[_ExecutedCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for executed in executed_cases:
        model = executed.linear_model
        if model is None:
            continue
        a_shape_pass = model.a.shape == (len(LINEAR_STATE_NAMES), len(LINEAR_STATE_NAMES))
        b_shape_pass = model.b.shape == (len(LINEAR_STATE_NAMES), len(INPUT_NAMES))
        finite_pass = bool(
            np.all(np.isfinite(model.a))
            and np.all(np.isfinite(model.b))
            and np.all(np.isfinite(model.f_trim))
        )
        indexing_pass = (
            tuple(model.state_names) == tuple(LINEAR_STATE_NAMES)
            and tuple(model.input_names) == tuple(INPUT_NAMES)
        )
        residual = dynamic_residual_from_xdot(model.f_trim)
        rows.append(
            {
                "case_name": executed.case.name,
                "role": executed.role,
                "speed_m_s": float(executed.case.speed_m_s),
                "a_rows": int(model.a.shape[0]),
                "a_cols": int(model.a.shape[1]),
                "b_rows": int(model.b.shape[0]),
                "b_cols": int(model.b.shape[1]),
                "a_shape_pass": bool(a_shape_pass),
                "b_shape_pass": bool(b_shape_pass),
                "finite_entries": finite_pass,
                "indexing_pass": bool(indexing_pass),
                "max_real_eigenvalue": float(np.max(np.real(np.linalg.eigvals(model.a)))),
                "open_loop_stability_is_diagnostic_only": True,
                "dynamic_residual_norm": residual["dynamic_residual_norm"],
                "hard_gate_pass": bool(
                    a_shape_pass
                    and b_shape_pass
                    and finite_pass
                    and indexing_pass
                    and residual["dynamic_residual_norm"] <= DYNAMIC_RESIDUAL_NORM_TOL
                ),
            }
        )
    return pd.DataFrame(rows)


def _finite_difference_rows(
    executed_cases: tuple[_ExecutedCase, ...],
    aircraft: AircraftModel,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for executed in executed_cases:
        model = executed.linear_model
        if model is None:
            continue
        a_fd, b_fd = finite_difference_linearisation(
            model.x_trim,
            model.u_trim,
            aircraft,
            rho_kg_m3=executed.case.rho_kg_m3,
            actuator_tau_s=executed.case.actuator_tau_s,
            wind_model=None,
            wind_mode="none",
            eps=FINITE_DIFFERENCE_EPS,
        )
        comparison = compare_jacobians(model.a, model.b, a_fd, b_fd)
        rows.append(
            {
                "case_name": executed.case.name,
                "role": executed.role,
                "speed_m_s": float(executed.case.speed_m_s),
                "eps": FINITE_DIFFERENCE_EPS,
                "jacobian_max_abs_tol": JACOBIAN_MAX_ABS_TOL,
                **comparison,
            }
        )
    return pd.DataFrame(rows)


def _key_derivative_rows(executed_cases: tuple[_ExecutedCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for executed in executed_cases:
        model = executed.linear_model
        if model is None:
            continue
        derivatives = key_derivatives(model)
        for name, value in derivatives.items():
            expected_sign = SIGN_EXPECTATIONS.get(name, "diagnostic_only")
            sign_pass = True
            if expected_sign == "positive":
                sign_pass = value > 0.0
            rows.append(
                {
                    "case_name": executed.case.name,
                    "role": executed.role,
                    "speed_m_s": float(executed.case.speed_m_s),
                    "derivative": name,
                    "value": float(value),
                    "expected_sign": expected_sign,
                    "hard_gate": expected_sign != "diagnostic_only",
                    "pass": bool(sign_pass),
                }
            )
    return pd.DataFrame(rows)


def _reduced_rows(executed_cases: tuple[_ExecutedCase, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for executed in executed_cases:
        model = executed.linear_model
        if model is None:
            continue
        frame = reduced_model_audit_rows(model)
        frame.insert(0, "speed_m_s", float(executed.case.speed_m_s))
        frame.insert(0, "role", executed.role)
        frame.insert(0, "case_name", executed.case.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _actuator_rows(executed_cases: tuple[_ExecutedCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for executed in executed_cases:
        model = executed.linear_model
        if model is None:
            continue
        for surface_name, input_name, tau in zip(
            ("delta_a", "delta_e", "delta_r"),
            INPUT_NAMES,
            executed.case.actuator_tau_s,
            strict=True,
        ):
            state_index = STATE_INDEX[surface_name]
            input_index = INPUT_INDEX[input_name]
            expected_a = -1.0 / float(tau)
            expected_b = 1.0 / float(tau)
            a_value = float(model.a[state_index, state_index])
            b_value = float(model.b[state_index, input_index])
            a_error = abs(a_value - expected_a)
            b_error = abs(b_value - expected_b)
            rows.append(
                {
                    "case_name": executed.case.name,
                    "role": executed.role,
                    "speed_m_s": float(executed.case.speed_m_s),
                    "surface_state": surface_name,
                    "command_input": input_name,
                    "tau_s": float(tau),
                    "a_diag": a_value,
                    "expected_a_diag": expected_a,
                    "a_abs_error": a_error,
                    "b_cmd": b_value,
                    "expected_b_cmd": expected_b,
                    "b_abs_error": b_error,
                    "actuator_jacobian_abs_tol": ACTUATOR_JACOBIAN_ABS_TOL,
                    "pass": bool(
                        a_error <= ACTUATOR_JACOBIAN_ABS_TOL
                        and b_error <= ACTUATOR_JACOBIAN_ABS_TOL
                    ),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 5) Output Writers
# =============================================================================
def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_linear_model_npz(path: Path, executed_cases: tuple[_ExecutedCase, ...]) -> None:
    arrays: dict[str, np.ndarray] = {
        "state_names": np.asarray(LINEAR_STATE_NAMES, dtype=str),
        "input_names": np.asarray(INPUT_NAMES, dtype=str),
    }
    linearised_names = []
    for executed in executed_cases:
        model = executed.linear_model
        if model is None:
            continue
        prefix = executed.case.name
        linearised_names.append(prefix)
        arrays[f"{prefix}_A"] = model.a
        arrays[f"{prefix}_B"] = model.b
        arrays[f"{prefix}_x_trim"] = model.x_trim
        arrays[f"{prefix}_u_trim"] = model.u_trim
        arrays[f"{prefix}_f_trim"] = model.f_trim
    arrays["case_names"] = np.asarray(linearised_names, dtype=str)
    np.savez(path, **arrays)


def _bool_column_all(frame: pd.DataFrame, column: str, default: bool = True) -> bool:
    if frame.empty or column not in frame:
        return default
    return bool(frame[column].astype(bool).all())


def _non_diagnostic_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "role" not in frame:
        return frame
    return frame[frame["role"] != "diagnostic"]


def _overall_status(
    executed_cases: tuple[_ExecutedCase, ...],
    residuals: pd.DataFrame,
    linear_summary: pd.DataFrame,
    fd_summary: pd.DataFrame,
    key_derivatives_frame: pd.DataFrame,
    reduced_frame: pd.DataFrame,
    actuator_frame: pd.DataFrame,
    wind_row: dict[str, float | bool],
) -> tuple[str, list[str], list[str]]:
    needs_review: list[str] = []
    optional_failures: list[str] = []
    for executed in executed_cases:
        if executed.role == "required" and executed.status != "converged":
            needs_review.append(f"{executed.case.name}: required trim did not converge")
        if executed.role == "optional" and executed.status != "converged":
            optional_failures.append(f"{executed.case.name}: optional trim did not converge")
        if executed.role == "optional_wind" and executed.status != "converged":
            optional_failures.append(f"{executed.case.name}: optional wind trim did not converge")

    hard_residuals = residuals[residuals.get("hard_gate_case", False) == True]  # noqa: E712
    if not _bool_column_all(hard_residuals, "pass"):
        needs_review.append("dynamic residual hard gate failed")
    hard_linear = _non_diagnostic_rows(linear_summary)
    if not _bool_column_all(hard_linear, "hard_gate_pass"):
        needs_review.append("linearisation matrix hard gate failed")
    hard_fd = _non_diagnostic_rows(fd_summary)
    if not _bool_column_all(hard_fd, "pass"):
        needs_review.append("finite-difference Jacobian hard gate failed")
    hard_key_derivatives = _non_diagnostic_rows(key_derivatives_frame)
    if hard_key_derivatives.empty or "hard_gate" not in hard_key_derivatives:
        sign_rows = pd.DataFrame()
    else:
        sign_rows = hard_key_derivatives[
            hard_key_derivatives["hard_gate"].astype(bool)
        ]
    if not _bool_column_all(sign_rows, "pass"):
        needs_review.append("project derivative sign convention check failed")
    hard_reduced = _non_diagnostic_rows(reduced_frame)
    if not _bool_column_all(hard_reduced, "hard_gate_pass"):
        needs_review.append("reduced-model shape or finite-entry hard gate failed")
    hard_actuator = _non_diagnostic_rows(actuator_frame)
    if not _bool_column_all(hard_actuator, "pass"):
        needs_review.append("actuator dynamics hard gate failed")
    if not bool(wind_row["pass"]):
        needs_review.append("uniform cg versus panel wind consistency failed")

    if needs_review:
        return "needs_review", needs_review, optional_failures
    if optional_failures:
        return "pass_with_optional_case_failures", needs_review, optional_failures
    return "pass", needs_review, optional_failures


def _manifest(
    run_id: int,
    paths: dict[str, Path],
    preflight: dict[str, bool],
    overall_status: str,
    needs_review: list[str],
    optional_failures: list[str],
    executed_cases: tuple[_ExecutedCase, ...],
    wind_row: dict[str, float | bool],
) -> dict[str, Any]:
    speed_outcomes = [
        {
            "case_name": executed.case.name,
            "role": executed.role,
            "speed_m_s": float(executed.case.speed_m_s),
            "wind_mode": executed.case.wind_mode,
            "status": executed.status,
        }
        for executed in executed_cases
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "output_root": _repo_relative(paths["root"]),
        "overall_status": overall_status,
        "needs_review": needs_review,
        "optional_failures": optional_failures,
        "speed_case_outcomes": speed_outcomes,
        "future_cases": [
            {"case_name": name, "status": "not_available_yet"} for name in FUTURE_CASES
        ],
        "required_trim_case": "natural_glide_6p5_none",
        "trim_audit_implemented": True,
        "linearisation_audit_implemented": True,
        "controller_implemented": False,
        "primitive_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "vicon_implemented": False,
        "hardware_implemented": False,
        "high_incidence_validation_claim": False,
        "dynamic_residual_excludes_position_rates": True,
        "dynamic_residual_norm_tol": DYNAMIC_RESIDUAL_NORM_TOL,
        "finite_difference_eps": FINITE_DIFFERENCE_EPS,
        "jacobian_max_abs_tol": JACOBIAN_MAX_ABS_TOL,
        "actuator_jacobian_abs_tol": ACTUATOR_JACOBIAN_ABS_TOL,
        "uniform_wind_max_abs_diff_tol": UNIFORM_WIND_MAX_ABS_DIFF_TOL,
        "uniform_wind_consistency_pass": bool(wind_row["pass"]),
        "command_interface_to_state_derivative": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "angle_units_internal": "rad",
        "speed_units": "m/s",
        "preflight": preflight,
        "validation_commands": list(VALIDATION_COMMANDS),
        "output_files": {
            name: _repo_relative(path)
            for name, path in paths.items()
            if name != "root"
        },
    }


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    trim_cases: pd.DataFrame,
    residuals: pd.DataFrame,
    fd_summary: pd.DataFrame,
    key_derivatives_frame: pd.DataFrame,
    wind_row: dict[str, float | bool],
) -> None:
    if key_derivatives_frame.empty or "hard_gate" not in key_derivatives_frame:
        sign_rows = pd.DataFrame()
    else:
        sign_rows = key_derivatives_frame[
            key_derivatives_frame["hard_gate"].astype(bool)
        ]
    lines = [
        "# Trim And Linearisation Audit Report",
        "",
        "This is an audit-only trim and linearisation evidence report. It does not",
        "implement a controller, primitive, OCP, TVLQR, governor, outer loop,",
        "Vicon interface, hardware path, or high-incidence validation.",
        "",
        "## Status",
        "",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Required trim case: `{manifest['required_trim_case']}`",
        "- Model-facing command input: `delta_cmd_rad` in radians.",
        "- Raw normalised commands do not enter `state_derivative`.",
        "",
        "## Tolerances",
        "",
        f"- Dynamic residual norm tolerance: `{DYNAMIC_RESIDUAL_NORM_TOL:g}`",
        f"- Finite-difference epsilon: `{FINITE_DIFFERENCE_EPS:g}`",
        f"- Jacobian max absolute error tolerance: `{JACOBIAN_MAX_ABS_TOL:g}`",
        f"- Actuator Jacobian absolute tolerance: `{ACTUATOR_JACOBIAN_ABS_TOL:g}`",
        f"- Uniform wind max derivative difference tolerance: `{UNIFORM_WIND_MAX_ABS_DIFF_TOL:g}`",
        "",
        "## Speed Coverage",
        "",
    ]
    for _, row in trim_cases.iterrows():
        if row["status"] == "not_available_yet":
            continue
        lines.append(
            "- "
            f"`{row['case_name']}`: role `{row['role']}`, "
            f"speed `{row['speed_m_s']}` m/s, status `{row['status']}`"
        )
    lines.extend(
        [
            "",
            "Optional 5.5 m/s and 7.5 m/s trim failures are recorded without failing",
            "the core 6.5 m/s acceptance audit. The 4.5 m/s case is diagnostic only.",
            "",
            "Steady trim is not used to represent the agile post-turn exit. Post-agile",
            "speed variation will be handled later by trajectory-knot linearisation",
            "and primitive-library expansion over entry and exit speed.",
            "",
            "## Dynamic Residuals",
            "",
        ]
    )
    for _, row in residuals.iterrows():
        lines.append(
            "- "
            f"`{row['case_name']}`: dynamic residual "
            f"`{row['dynamic_residual_norm']:.6g}`, pass `{row['pass']}`"
        )
    lines.extend(["", "## Finite-Difference Check", ""])
    for _, row in fd_summary.iterrows():
        lines.append(
            "- "
            f"`{row['case_name']}`: A max `{row['a_max_abs_error']:.6g}`, "
            f"B max `{row['b_max_abs_error']:.6g}`, pass `{row['pass']}`"
        )
    lines.extend(["", "## Derivative Sign Checks", ""])
    for _, row in sign_rows.iterrows():
        lines.append(
            "- "
            f"`{row['case_name']}` `{row['derivative']}` = "
            f"`{row['value']:.6g}`, expected `{row['expected_sign']}`, "
            f"pass `{row['pass']}`"
        )
    lines.extend(
        [
            "",
            "Reduced-model eigenvalues, controllability-style rank, and open-loop",
            "stability are diagnostic only. They do not fail the audit unless the",
            "reduced matrices have wrong shape, nonfinite values, or inconsistent indexing.",
            "",
            "## Uniform Wind Consistency",
            "",
            f"- Wind vector: `{UNIFORM_WIND_W_M_S}` m/s in public z-up world axes.",
            f"- Max absolute derivative difference: `{wind_row['max_abs_xdot_diff']:.6g}`",
            f"- Pass: `{wind_row['pass']}`",
            "",
            "## Future Cases",
            "",
        ]
    )
    for case_name in FUTURE_CASES:
        lines.append(f"- `{case_name}`: `not_available_yet`")
    lines.extend(
        [
            "",
            "## Validation Commands",
            "",
            *[f"- `{command}`" for command in VALIDATION_COMMANDS],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


# =============================================================================
# 6) Public Audit Workflow
# =============================================================================
def run_trim_linearisation_audit(
    output_root: Path | None = None,
    run_id: int = 1,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run the trim and linearisation audit and write evidence artifacts."""

    preflight = _preflight_checks()
    result_root = DEFAULT_RESULTS_ROOT if output_root is None else Path(output_root)
    tree = make_result_tree(
        result_root,
        TRIM_LINEARISATION_CAMPAIGN,
        run_id,
        overwrite=overwrite,
    )
    suffix = f"s{run_id:03d}"
    output_paths = {
        "trim_cases_csv": tree["metrics"] / f"trim_cases_{suffix}.csv",
        "dynamic_residuals_csv": tree["metrics"] / f"dynamic_residuals_{suffix}.csv",
        "linearisation_summary_csv": tree["metrics"]
        / f"linearisation_summary_{suffix}.csv",
        "finite_difference_check_csv": tree["metrics"]
        / f"finite_difference_check_{suffix}.csv",
        "key_derivatives_csv": tree["metrics"] / f"key_derivatives_{suffix}.csv",
        "reduced_model_audit_csv": tree["metrics"]
        / f"reduced_model_audit_{suffix}.csv",
        "actuator_dynamics_csv": tree["metrics"] / f"actuator_dynamics_{suffix}.csv",
        "uniform_wind_consistency_csv": tree["metrics"]
        / f"uniform_wind_consistency_{suffix}.csv",
        "linear_model_npz": tree["logs"] / f"linear_model_{suffix}.npz",
        "manifest_json": tree["manifests"]
        / f"trim_linearisation_manifest_{suffix}.json",
        "report_md": tree["reports"] / f"trim_linearisation_report_{suffix}.md",
    }

    aircraft = adapt_glider(build_nausicaa_glider())
    executed_cases = tuple(
        _execute_case(case, aircraft) for case in build_trim_linearisation_cases()
    )
    trim_cases = _trim_case_rows(executed_cases)
    residuals = _dynamic_residual_rows(executed_cases)
    linear_summary = _linearisation_rows(executed_cases)
    fd_summary = _finite_difference_rows(executed_cases, aircraft)
    key_derivatives_frame = _key_derivative_rows(executed_cases)
    reduced_frame = _reduced_rows(executed_cases)
    actuator_frame = _actuator_rows(executed_cases)

    required_model = next(
        (
            executed.linear_model
            for executed in executed_cases
            if executed.case.name == "natural_glide_6p5_none"
        ),
        None,
    )
    if required_model is None:
        wind_row: dict[str, float | bool] = {
            "wind_x_w_m_s": float(UNIFORM_WIND_W_M_S[0]),
            "wind_y_w_m_s": float(UNIFORM_WIND_W_M_S[1]),
            "wind_z_w_m_s": float(UNIFORM_WIND_W_M_S[2]),
            "max_abs_xdot_diff": float("nan"),
            "finite": False,
            "pass": False,
        }
    else:
        wind_row = uniform_wind_consistency_check(
            required_model.x_trim,
            required_model.u_trim,
            aircraft,
            wind_w_m_s=UNIFORM_WIND_W_M_S,
        )
    wind_frame = pd.DataFrame([wind_row])

    overall_status, needs_review, optional_failures = _overall_status(
        executed_cases,
        residuals,
        linear_summary,
        fd_summary,
        key_derivatives_frame,
        reduced_frame,
        actuator_frame,
        wind_row,
    )
    all_paths = {"root": tree["root"], **output_paths}
    manifest = _manifest(
        run_id,
        all_paths,
        preflight,
        overall_status,
        needs_review,
        optional_failures,
        executed_cases,
        wind_row,
    )

    trim_cases.to_csv(output_paths["trim_cases_csv"], index=False)
    residuals.to_csv(output_paths["dynamic_residuals_csv"], index=False)
    linear_summary.to_csv(output_paths["linearisation_summary_csv"], index=False)
    fd_summary.to_csv(output_paths["finite_difference_check_csv"], index=False)
    key_derivatives_frame.to_csv(output_paths["key_derivatives_csv"], index=False)
    reduced_frame.to_csv(output_paths["reduced_model_audit_csv"], index=False)
    actuator_frame.to_csv(output_paths["actuator_dynamics_csv"], index=False)
    wind_frame.to_csv(output_paths["uniform_wind_consistency_csv"], index=False)
    _write_linear_model_npz(output_paths["linear_model_npz"], executed_cases)
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    _write_report(
        output_paths["report_md"],
        manifest,
        trim_cases,
        residuals,
        fd_summary,
        key_derivatives_frame,
        wind_row,
    )
    output_paths["root"] = tree["root"]
    return output_paths
