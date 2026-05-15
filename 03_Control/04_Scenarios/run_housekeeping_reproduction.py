from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


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
from flight_dynamics import adapt_glider, state_derivative  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from latency import CommandToSurfaceConfig, CommandToSurfaceLayer  # noqa: E402
from linearisation import INPUT_NAMES, STATE_INDEX, STATE_NAMES, key_derivatives, linearise_trim  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, simulate_primitive, write_log  # noqa: E402
from scenarios import arena_feasible_entry_state, recovery_entry_state  # noqa: E402
from templates import BankReversalPrimitive, NominalGlidePrimitive, RecoveryPrimitive  # noqa: E402


HOUSEKEEPING_FIELDS = (
    "seed",
    "check_name",
    "success",
    "failure_reason",
    "state_order_ok",
    "input_order_ok",
    "sign_check_ok",
    "trim_residual_norm",
    "linearisation_finite_difference_error",
    "inside_true_safety_volume",
    "terminal_speed_m_s",
    "max_alpha_deg",
    "max_beta_deg",
    "max_bank_deg",
    "min_wall_distance_m",
    "min_floor_margin_m",
    "min_ceiling_margin_m",
    "saturation_fraction",
    "log_path",
)


def run_housekeeping_reproduction(
    *,
    seed: int,
    output_root: str | Path | None = None,
    quick: bool = False,
) -> dict[str, object]:
    """Run non-agile housekeeping checks and write metrics/manifests."""
    root = Path(output_root) if output_root is not None else _default_output_root(seed)
    metrics_dir = root / "metrics"
    manifests_dir = root / "manifests"
    logs_dir = root / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(linear_model.x_trim, linear_model.u_trim)
    arena = ArenaConfig()

    state_rows = [_state_order_row(seed)]
    sign_rows = [_control_sign_row(seed, linear_model)]
    trim_rows = [_trim_linearisation_row(seed, aircraft, linear_model)]
    smoke_rows = _baseline_smoke_rows(
        seed=seed,
        quick=quick,
        root=root,
        logs_dir=logs_dir,
        aircraft=aircraft,
        context=context,
        linear_model=linear_model,
        arena=arena,
    )

    output_paths = {
        "state_order_audit": metrics_dir / f"state_order_audit_s{seed:03d}.csv",
        "control_sign_audit": metrics_dir / f"control_sign_audit_s{seed:03d}.csv",
        "trim_linearisation_audit": metrics_dir / f"trim_linearisation_audit_s{seed:03d}.csv",
        "baseline_primitive_smoke": metrics_dir / f"baseline_primitive_smoke_s{seed:03d}.csv",
        "manifest": manifests_dir / f"housekeeping_manifest_s{seed:03d}.json",
    }
    _write_rows(output_paths["state_order_audit"], state_rows)
    _write_rows(output_paths["control_sign_audit"], sign_rows)
    _write_rows(output_paths["trim_linearisation_audit"], trim_rows)
    _write_rows(output_paths["baseline_primitive_smoke"], smoke_rows)

    all_rows = state_rows + sign_rows + trim_rows + smoke_rows
    manifest = {
        "ect_layer_sequence": "Cleanup -> Exploration -> Candidate",
        "seed": int(seed),
        "quick": bool(quick),
        "output_root": _rel(root),
        "metrics": {key: _rel(path) for key, path in output_paths.items() if key != "manifest"},
        "logs": sorted(_rel(path) for path in logs_dir.glob("*.csv")),
        "row_count": len(all_rows),
        "all_checks_success": bool(all(row.get("success") is True for row in all_rows)),
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_root": root,
        "manifest": output_paths["manifest"],
        "metrics": output_paths,
        "rows": all_rows,
        "all_checks_success": manifest["all_checks_success"],
    }


def _default_output_root(seed: int) -> Path:
    return REPO_ROOT / "03_Control" / "05_Results" / "00_housekeeping" / f"{seed:03d}"


def _blank_row(seed: int, check_name: str) -> dict[str, object]:
    return {key: "" for key in HOUSEKEEPING_FIELDS} | {
        "seed": int(seed),
        "check_name": check_name,
        "success": False,
        "failure_reason": "",
    }


def _state_order_row(seed: int) -> dict[str, object]:
    expected_state = (
        "x_w",
        "y_w",
        "z_w",
        "phi",
        "theta",
        "psi",
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "delta_a",
        "delta_e",
        "delta_r",
    )
    expected_input = ("delta_a_cmd", "delta_e_cmd", "delta_r_cmd")
    row = _blank_row(seed, "state_order_audit")
    state_ok = tuple(STATE_NAMES) == expected_state
    input_ok = tuple(INPUT_NAMES) == expected_input
    row.update(
        {
            "state_order_ok": state_ok,
            "input_order_ok": input_ok,
            "success": bool(state_ok and input_ok),
            "failure_reason": "" if state_ok and input_ok else "state_or_input_order_mismatch",
        }
    )
    return row


def _control_sign_row(seed: int, linear_model: object) -> dict[str, object]:
    row = _blank_row(seed, "control_sign_audit")
    derivatives = key_derivatives(linear_model)
    sign_ok = (
        derivatives["l_delta_a"] > 0.0
        and derivatives["m_delta_e"] > 0.0
        and derivatives["n_delta_r"] > 0.0
    )
    row.update(
        {
            "sign_check_ok": bool(sign_ok),
            "success": bool(sign_ok),
            "failure_reason": "" if sign_ok else "control_sign_regression",
        }
    )
    return row


def _trim_linearisation_row(seed: int, aircraft: object, linear_model: object) -> dict[str, object]:
    row = _blank_row(seed, "trim_linearisation_audit")
    trim_residual = float(np.linalg.norm(linear_model.f_trim))
    fd_error = _finite_difference_error(aircraft, linear_model)
    success = bool(np.isfinite(trim_residual) and np.isfinite(fd_error) and fd_error < 1e-3)
    row.update(
        {
            "trim_residual_norm": trim_residual,
            "linearisation_finite_difference_error": fd_error,
            "success": success,
            "failure_reason": "" if success else "trim_linearisation_audit_failed",
        }
    )
    return row


def _finite_difference_error(aircraft: object, linear_model: object) -> float:
    eps = 1e-6
    errors: list[float] = []
    for state_name in ("u", "w", "q", "delta_e"):
        idx = STATE_INDEX[state_name]
        perturb = np.zeros(15)
        perturb[idx] = eps
        f_plus = state_derivative(
            linear_model.x_trim + perturb,
            linear_model.u_trim,
            aircraft,
            wind_mode="none",
        )
        f_minus = state_derivative(
            linear_model.x_trim - perturb,
            linear_model.u_trim,
            aircraft,
            wind_mode="none",
        )
        column = (f_plus - f_minus) / (2.0 * eps)
        errors.append(float(np.max(np.abs(column - linear_model.a[:, idx]))))
    return float(max(errors))


def _baseline_smoke_rows(
    *,
    seed: int,
    quick: bool,
    root: Path,
    logs_dir: Path,
    aircraft: object,
    context: object,
    linear_model: object,
    arena: ArenaConfig,
) -> list[dict[str, object]]:
    duration = 0.18 if quick else 0.45
    primitives = (
        NominalGlidePrimitive(duration_s=duration),
        RecoveryPrimitive(duration_s=duration),
        BankReversalPrimitive(duration_s=duration, bank_angle_rad=np.deg2rad(8.0)),
    )
    rows: list[dict[str, object]] = []
    for primitive in primitives:
        x0 = (
            recovery_entry_state(linear_model.x_trim)
            if primitive.name == "recovery"
            else arena_feasible_entry_state(linear_model.x_trim)
        )
        log_path = logs_dir / f"{primitive.name}_s{seed:03d}.csv"
        result = simulate_primitive(
            scenario_id=f"housekeeping_{primitive.name}",
            seed=seed,
            primitive=primitive,
            x0=x0,
            context=context,
            aircraft=aircraft,
            wind_model=None,
            wind_model_name="none",
            wind_mode="none",
            command_layer=CommandToSurfaceLayer(config=CommandToSurfaceConfig(mode="nominal")),
            log_path=log_path,
            repo_root=REPO_ROOT,
            rollout_config=RolloutConfig(max_abs_alpha_rad=np.deg2rad(45.0)),
            arena_config=arena,
        )
        write_log(result, log_path)
        row = _blank_row(seed, primitive.name)
        row.update(
            {
                "success": bool(result.success),
                "failure_reason": result.termination_reason,
                "inside_true_safety_volume": result.metrics.get("inside_safe_volume"),
                "terminal_speed_m_s": result.metrics.get("terminal_speed_m_s"),
                "max_alpha_deg": result.metrics.get("max_alpha_deg"),
                "max_beta_deg": result.metrics.get("max_beta_deg"),
                "max_bank_deg": result.metrics.get("max_bank_deg"),
                "min_wall_distance_m": result.metrics.get("min_wall_distance_m"),
                "min_floor_margin_m": result.metrics.get("min_floor_margin_m"),
                "min_ceiling_margin_m": result.metrics.get("min_ceiling_margin_m"),
                "saturation_fraction": result.metrics.get("saturation_fraction"),
                "log_path": _rel(log_path),
            }
        )
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HOUSEKEEPING_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_housekeeping_reproduction(
        seed=args.seed,
        output_root=args.output_root,
        quick=args.quick,
    )
    print("housekeeping reproduction complete")
    print(f"output_root: {_rel(Path(result['output_root']))}")
    print(f"manifest: {_rel(Path(result['manifest']))}")
    print(f"all_checks_success: {result['all_checks_success']}")


if __name__ == "__main__":
    main()

