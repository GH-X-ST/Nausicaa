from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from command_contract import COMMAND_NAMES, NORMALISED_COMMAND_NAMES
from metric_contract import validate_metric_row
from result_paths import make_result_tree
from rollout import RolloutResult
from state_contract import STATE_NAMES, STATE_SIZE


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Dataframe Builders
# 2) Manifest and Report Writers
# 3) Rollout Output Writer
# =============================================================================


# =============================================================================
# 1) Dataframe Builders
# =============================================================================
def _time_vector(time_s: np.ndarray) -> np.ndarray:
    time = np.asarray(time_s, dtype=float).reshape(-1)
    if time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must contain finite time samples.")
    return time


def trajectory_dataframe(time_s: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    """Return time plus canonical state columns."""

    time = _time_vector(time_s)
    state = np.asarray(x, dtype=float)
    if state.shape != (time.size, STATE_SIZE):
        raise ValueError("x must have shape (N, 15) matching time_s.")
    data: dict[str, np.ndarray] = {"time_s": time}
    for index, name in enumerate(STATE_NAMES):
        data[name] = state[:, index]
    return pd.DataFrame(data)


def _command_matrix(values: np.ndarray, name: str, sample_count: int) -> np.ndarray:
    command = np.asarray(values, dtype=float)
    if command.shape != (sample_count, 3):
        raise ValueError(f"{name} must have shape (N, 3) matching time_s.")
    if not np.all(np.isfinite(command)):
        raise ValueError(f"{name} must contain only finite values.")
    return command


def command_dataframe(
    time_s: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    delta_cmd_rad: np.ndarray,
) -> pd.DataFrame:
    """Return requested/applied normalised commands and physical radian targets."""

    time = _time_vector(time_s)
    requested = _command_matrix(u_norm_requested, "u_norm_requested", time.size)
    applied = _command_matrix(u_norm_applied, "u_norm_applied", time.size)
    command_rad = _command_matrix(delta_cmd_rad, "delta_cmd_rad", time.size)
    data: dict[str, np.ndarray] = {"time_s": time}
    for index, name in enumerate(NORMALISED_COMMAND_NAMES):
        data[f"u_norm_requested_{name}"] = requested[:, index]
    for index, name in enumerate(NORMALISED_COMMAND_NAMES):
        data[f"u_norm_applied_{name}"] = applied[:, index]
    for index, name in enumerate(COMMAND_NAMES):
        data[f"delta_cmd_rad_{name}"] = command_rad[:, index]
    return pd.DataFrame(data)


def metric_dataframe(metrics: Mapping[str, object]) -> pd.DataFrame:
    """Return one validated metric row."""

    validate_metric_row(metrics)
    return pd.DataFrame([dict(metrics)])


# =============================================================================
# 2) Manifest and Report Writers
# =============================================================================
def _manifest(result: RolloutResult, run_id: int, output_files: Mapping[str, Path]) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "success": bool(result.success),
        "failure_label": result.failure_label,
        "notes": result.notes,
        "rollout_implemented": True,
        "controller_implemented": False,
        "primitive_implemented": False,
        "ocp_implemented": False,
        "tvlqr_implemented": False,
        "governor_implemented": False,
        "outer_loop_implemented": False,
        "high_incidence_validation_claim": False,
        "command_bridge": "normalised_command_to_surface_rad",
        "state_derivative_command_input": "delta_cmd_rad",
        "raw_normalised_commands_enter_state_derivative": False,
        "output_files": {name: str(path) for name, path in output_files.items()},
    }


def _write_report(path: Path, result: RolloutResult, manifest: Mapping[str, object]) -> None:
    lines = [
        "# Rollout Smoke Report",
        "",
        "This is an open-loop plant rollout smoke test only. It does not implement",
        "a primitive, controller, OCP, TVLQR, governor, outer loop, Vicon interface,",
        "hardware path, or high-incidence validation.",
        "",
        "## Command Path",
        "",
        "- Requested command: `u_norm_requested`.",
        "- Applied command: `u_norm_applied`, clipped to `[-1, +1]`.",
        "- Plant command: `delta_cmd_rad`, produced by `normalised_command_to_surface_rad`.",
        "- `flight_dynamics.state_derivative` receives `delta_cmd_rad`, never raw normalised commands.",
        "",
        "## Result",
        "",
        f"- Success flag: `{result.success}`",
        f"- Failure label: `{result.failure_label}`",
        f"- Notes: `{result.notes}`",
        f"- Rollout implemented: `{manifest['rollout_implemented']}`",
        f"- Primitive implemented: `{manifest['primitive_implemented']}`",
        f"- Controller implemented: `{manifest['controller_implemented']}`",
        f"- OCP implemented: `{manifest['ocp_implemented']}`",
        f"- TVLQR implemented: `{manifest['tvlqr_implemented']}`",
        f"- Governor implemented: `{manifest['governor_implemented']}`",
        f"- Outer loop implemented: `{manifest['outer_loop_implemented']}`",
        f"- High-incidence validation claim: `{manifest['high_incidence_validation_claim']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# =============================================================================
# 3) Rollout Output Writer
# =============================================================================
def write_rollout_outputs(
    result: RolloutResult,
    result_root: Path,
    campaign: str,
    run_id: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write trajectory, command, metric, manifest, and report outputs."""

    paths = make_result_tree(Path(result_root), campaign, run_id, overwrite=overwrite)
    suffix = f"s{run_id:03d}"
    output_paths = {
        "trajectory_csv": paths["metrics"] / f"trajectory_{suffix}.csv",
        "commands_csv": paths["metrics"] / f"commands_{suffix}.csv",
        "metrics_csv": paths["metrics"] / f"metrics_{suffix}.csv",
        "manifest_json": paths["manifests"] / f"rollout_smoke_manifest_{suffix}.json",
        "report_md": paths["reports"] / f"rollout_smoke_report_{suffix}.md",
    }
    trajectory_dataframe(result.time_s, result.x).to_csv(
        output_paths["trajectory_csv"],
        index=False,
    )
    command_dataframe(
        result.time_s,
        result.u_norm_requested,
        result.u_norm_applied,
        result.delta_cmd_rad,
    ).to_csv(output_paths["commands_csv"], index=False)
    metric_dataframe(result.metrics).to_csv(output_paths["metrics_csv"], index=False)
    manifest = _manifest(result, run_id, output_paths)
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    _write_report(output_paths["report_md"], result, manifest)
    output_paths["root"] = paths["root"]
    return output_paths
