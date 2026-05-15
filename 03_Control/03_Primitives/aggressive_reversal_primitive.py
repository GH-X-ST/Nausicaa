from __future__ import annotations

import numpy as np

from closed_loop_trajectory import ClosedLoopTrajectoryPrimitive


def build_aggressive_reversal_primitive(
    *,
    result: object,
    context: object,
    aircraft: object,
    wind_model: object | None = None,
    wind_mode: str = "none",
    command_layer: object | None = None,
) -> object:
    """Return a ClosedLoopTrajectoryPrimitive-compatible aggressive reversal primitive."""
    del context, aircraft, wind_model, wind_mode, command_layer
    times_s = np.asarray(getattr(result, "times_s"), dtype=float)
    x_ref = np.asarray(getattr(result, "x_ref"), dtype=float)
    u_ff = np.asarray(getattr(result, "u_ff"), dtype=float)
    phase_labels = tuple(str(label) for label in getattr(result, "phase_labels"))
    target = getattr(result, "target")
    k_feedback = getattr(result, "k_feedback", None)
    if k_feedback is None:
        k_feedback = np.zeros((times_s.size, 3, 15), dtype=float)
    metadata = {
        "primitive_family": "aggressive_high_incidence_reversal",
        "target_heading_deg": float(getattr(target, "target_heading_deg")),
        "direction": str(getattr(target, "direction")),
        "phase_labels": phase_labels,
        "model_status": "high_incidence_simulation_surrogate",
        "is_real_flight_claim": False,
    }
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, dict):
        metadata.update({f"metric_{key}": value for key, value in metrics.items()})
    return ClosedLoopTrajectoryPrimitive(
        name=f"aggressive_reversal_{int(round(float(target.target_heading_deg))):03d}_{target.direction}",
        times_s=times_s,
        x_ref=x_ref,
        u_ff=u_ff,
        k_feedback=np.asarray(k_feedback, dtype=float),
        phase_labels=phase_labels,
        metadata=metadata,
    )

