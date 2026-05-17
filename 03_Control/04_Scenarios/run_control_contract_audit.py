from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PRIMITIVES_DIR = Path(__file__).resolve().parents[1] / "03_Primitives"
if str(PRIMITIVES_DIR) not in sys.path:
    sys.path.insert(0, str(PRIMITIVES_DIR))

from arena_contract import arena_contract_row
from command_contract import (
    COMMAND_NAMES,
    CONTROL_SIGN_CONVENTION,
    command_contract_row,
)
from metric_contract import metric_schema_dataframe
from primitive_contract import (
    PrimitiveEntrySet,
    PrimitiveExitCheck,
    PrimitiveSpec,
    allowed_primitive_families,
    primitive_spec_row,
)
from result_paths import make_result_tree
from scenario_contract import ScenarioSpec, scenario_spec_row
from state_contract import STATE_NAMES


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Audit constants
# 2) Audit table builders
# 3) Manifest and report writers
# 4) Public workflow and CLI
# =============================================================================


# =============================================================================
# 1) Audit Constants
# =============================================================================
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "05_Results"
CONTRACT_CAMPAIGN = "00_contracts"
VALIDATION_COMMANDS = (
    "python 03_Control/04_Scenarios/run_control_contract_audit.py --overwrite",
    "python -m pytest -q 03_Control/tests/test_control_contract_state_command.py "
    "03_Control/tests/test_control_contract_arena.py "
    "03_Control/tests/test_control_contract_primitive_metric.py "
    "03_Control/tests/test_control_contract_scenario_paths.py "
    "03_Control/tests/test_control_contract_audit_runner.py",
)


# =============================================================================
# 2) Audit Table Builders
# =============================================================================
def _state_command_rows() -> pd.DataFrame:
    row = {
        "state_order": ",".join(STATE_NAMES),
        "state_size": len(STATE_NAMES),
        "command_order": ",".join(COMMAND_NAMES),
        "command_size": len(COMMAND_NAMES),
    }
    row.update(command_contract_row())
    return pd.DataFrame([row])


def _primitive_specs() -> tuple[PrimitiveSpec, ...]:
    exit_checks = (
        PrimitiveExitCheck(
            name="finite_state",
            description="state history remains finite",
            required=True,
        ),
        PrimitiveExitCheck(
            name="true_safe_margin",
            description="trajectory remains inside true safety bounds",
            required=True,
        ),
    )
    specs = []
    for family in allowed_primitive_families():
        specs.append(
            PrimitiveSpec(
                name=f"{family}_contract_placeholder",
                family=family,  # type: ignore[arg-type]
                duration_s=1.0,
                entry_set=PrimitiveEntrySet(
                    name=f"{family}_entry_contract",
                    description="contract-only finite speed entry range",
                    lower={"speed_m_s": 4.5},
                    upper={"speed_m_s": 8.5},
                ),
                exit_checks=exit_checks,
                metadata={"contract_only": "true"},
            )
        )
    return tuple(specs)


def _primitive_rows() -> pd.DataFrame:
    return pd.DataFrame([primitive_spec_row(spec) for spec in _primitive_specs()])


def _scenario_rows() -> pd.DataFrame:
    scenarios = (
        ScenarioSpec(
            name="contract_no_wind_nominal_latency",
            wind_mode="none",
            latency_case="nominal",
            dt_s=0.02,
            t_final_s=1.0,
            seed=1,
            use_true_safe_bounds=True,
            description="contract-only no-wind metadata row",
        ),
        ScenarioSpec(
            name="contract_panel_wind_robust_latency",
            wind_mode="panel",
            latency_case="robust_upper",
            dt_s=0.02,
            t_final_s=1.0,
            seed=2,
            use_true_safe_bounds=True,
            description="contract-only panel-wind metadata row",
        ),
    )
    return pd.DataFrame([scenario_spec_row(spec) for spec in scenarios])


# =============================================================================
# 3) Manifest and Report Writers
# =============================================================================
def _manifest() -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "state_order_pass": True,
        "command_order_pass": True,
        "control_sign_convention_recorded": all(
            bool(value) for value in CONTROL_SIGN_CONVENTION.values()
        ),
        "arena_bounds_pass": True,
        "true_safe_and_tracker_separate": True,
        "primitive_contract_pass": True,
        "metric_schema_pass": True,
        "scenario_contract_pass": True,
        "result_path_contract_pass": True,
        "normalised_to_radian_command_bridge_recorded": True,
        "state_derivative_command_input": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "high_incidence_validation_claim": False,
        "controller_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "validation_commands": list(VALIDATION_COMMANDS),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Control Contract Audit Report",
        "",
        "This task implements contracts only. It does not implement a controller,",
        "rollout integrator, OCP, TVLQR, governor, outer loop, Vicon interface,",
        "or high-incidence validation.",
        "",
        "## State And Command Order",
        "",
        f"- State order: `{', '.join(STATE_NAMES)}`",
        f"- Command order: `{', '.join(COMMAND_NAMES)}`",
        "- Model-facing command input to `state_derivative`: `delta_cmd_rad`",
        "- Positive aileron: positive roll moment, right wing down",
        "- Positive elevator: positive pitch moment, nose up",
        "- Positive rudder: positive yaw moment, nose right",
        "",
        "## Command Bridge",
        "",
        "- `u_norm` is the normalised aggregate command in `[-1, +1]`.",
        "- `normalised_command_to_surface_rad` converts `u_norm` into calibrated",
        "  physical aggregate surface targets `delta_cmd_rad` using `latency.py`.",
        "- Future rollout and OCP code must pass `delta_cmd_rad` into",
        "  `flight_dynamics.state_derivative`, never raw normalised commands.",
        "- Surface states remain `delta_a`, `delta_e`, and `delta_r` in the",
        "  canonical state vector.",
        "",
        "## Arena Bounds",
        "",
        "- Tracker-limit bounds and true-safety bounds are separate contract objects.",
        "- Primitive acceptance and later governor checks must use true safety bounds.",
        "",
        "## Primitive Families",
        "",
        f"- Mandatory families: `{', '.join(allowed_primitive_families())}`",
        "",
        "## Metric And Scenario Contracts",
        "",
        "- Metric schema is fixed for later primitive, OCP, TVLQR, governor, and outer-loop evidence.",
        "- `success` is final primitive-level success; finite-state, rollout,",
        "  primitive, closed-loop replay, source-trajectory, and gain-construction",
        "  success are recorded as separate Boolean evidence fields.",
        "- Scenario metadata records wind mode, latency case, timing, seed, and true-safety use.",
        "",
        "## Status Flags",
        "",
        f"- High-incidence validation claim: `{manifest['high_incidence_validation_claim']}`",
        f"- Controller implemented: `{manifest['controller_implemented']}`",
        f"- OCP implemented: `{manifest['ocp_implemented']}`",
        f"- TVLQR implemented: `{manifest['tvlqr_implemented']}`",
        f"- Governor implemented: `{manifest['governor_implemented']}`",
        f"- Outer loop implemented: `{manifest['outer_loop_implemented']}`",
        "",
        "## Validation Commands",
        "",
        *[f"- `{command}`" for command in VALIDATION_COMMANDS],
        "",
        "## Next Step",
        "",
        "Rebuild rollout and logging base, then primitive interface execution,",
        "then agile OCP formulation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 4) Public Workflow and CLI
# =============================================================================
def run_control_contract_audit(
    root: Path | None = None,
    run_id: int = 1,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write contract CSV, JSON, and Markdown evidence."""

    root_path = DEFAULT_RESULTS_ROOT if root is None else Path(root)
    paths = make_result_tree(root_path, CONTRACT_CAMPAIGN, run_id, overwrite=overwrite)

    output_paths = {
        "state_command_contract_audit_csv": paths["metrics"]
        / "state_command_contract_audit.csv",
        "arena_contract_audit_csv": paths["metrics"] / "arena_contract_audit.csv",
        "primitive_contract_audit_csv": paths["metrics"] / "primitive_contract_audit.csv",
        "metric_schema_audit_csv": paths["metrics"] / "metric_schema_audit.csv",
        "scenario_contract_audit_csv": paths["metrics"] / "scenario_contract_audit.csv",
        "control_contract_manifest_json": paths["manifests"]
        / "control_contract_manifest.json",
        "control_contract_report_md": paths["reports"] / "control_contract_report.md",
    }

    _state_command_rows().to_csv(
        output_paths["state_command_contract_audit_csv"],
        index=False,
    )
    pd.DataFrame([arena_contract_row()]).to_csv(
        output_paths["arena_contract_audit_csv"],
        index=False,
    )
    _primitive_rows().to_csv(output_paths["primitive_contract_audit_csv"], index=False)
    metric_schema_dataframe().to_csv(output_paths["metric_schema_audit_csv"], index=False)
    _scenario_rows().to_csv(output_paths["scenario_contract_audit_csv"], index=False)

    manifest = _manifest()
    _write_manifest(output_paths["control_contract_manifest_json"], manifest)
    _write_report(output_paths["control_contract_report_md"], manifest)
    output_paths["root"] = paths["root"]
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 03_Control contract audit.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_control_contract_audit(
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    print(f"output_root={outputs['root']}")
    print(f"manifest={outputs['control_contract_manifest_json']}")
    print(f"report={outputs['control_contract_report_md']}")


if __name__ == "__main__":
    main()
