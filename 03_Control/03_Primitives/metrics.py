from __future__ import annotations

from pathlib import Path

import numpy as np

from arena import ArenaConfig, safety_margins
from linearisation import STATE_INDEX


def relative_path(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


def rollout_metrics(
    scenario_id: str,
    seed: int,
    wind_model: str,
    wind_mode: str,
    latency_mode: str,
    primitive_selected: str,
    success: bool,
    termination_reason: str,
    states: np.ndarray,
    log_path: Path,
    repo_root: Path,
    arena_config: ArenaConfig,
    saturation_fraction: float,
    tracking_error_rms: float | None = None,
    governor_rejection_reason: str = "",
) -> dict[str, float | str | bool | int]:
    state_arr = np.asarray(states, dtype=float)
    final = state_arr[-1]
    margins = [safety_margins(row, arena_config) for row in state_arr]
    min_wall = min(float(row["min_wall_distance_m"]) for row in margins)
    inside = all(bool(row["inside_safe_volume"]) for row in margins)
    alpha = np.arctan2(
        state_arr[:, STATE_INDEX["w"]],
        np.maximum(state_arr[:, STATE_INDEX["u"]], 1e-12),
    )
    return {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "wind_model": wind_model,
        "wind_mode": wind_mode,
        "latency_mode": latency_mode,
        "primitive_selected": primitive_selected,
        "success": bool(success),
        "termination_reason": termination_reason,
        "height_change_m": float(final[STATE_INDEX["z_w"]] - state_arr[0, STATE_INDEX["z_w"]]),
        "terminal_speed_m_s": float(np.linalg.norm(final[6:9])),
        "max_alpha_deg": float(np.rad2deg(np.max(np.abs(alpha)))),
        "max_abs_phi_deg": float(
            np.rad2deg(np.max(np.abs(state_arr[:, STATE_INDEX["phi"]])))
        ),
        "min_wall_distance_m": float(min_wall),
        "inside_safe_volume": bool(inside),
        "saturation_fraction": float(saturation_fraction),
        "tracking_error_rms": "" if tracking_error_rms is None else float(tracking_error_rms),
        "governor_rejection_reason": governor_rejection_reason,
        "log_path_relative": relative_path(log_path, repo_root),
    }
