from __future__ import annotations

from pathlib import Path

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from latency import CommandToSurfaceLayer
from linearisation import linearise_trim
from primitive import build_primitive_context
from scenarios import arena_feasible_entry_state
from turn_trajectory_optimisation import (
    OptimisedTurnResult,
    TurnOptimisationConfig,
    TurnTarget,
    build_turn_trajectory_primitive,
    load_selected_turn_primitive,
    save_turn_result,
)


def _synthetic_result() -> tuple[OptimisedTurnResult, object, object]:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    x0 = arena_feasible_entry_state(linear_model.x_trim, altitude_m=2.7)
    times = np.array([0.0, 0.02, 0.04], dtype=float)
    x_ref = np.vstack([x0, x0, x0])
    x_ref[:, 0] += np.array([0.0, 0.13, 0.26])
    u_ff = np.vstack([linear_model.u_trim, linear_model.u_trim, linear_model.u_trim])
    config = TurnOptimisationConfig(n_intervals=2, t_min_s=0.04, t_max_s=0.04)
    result = OptimisedTurnResult(
        success=True,
        failure_reason="",
        feasibility_label="accepted_low_alpha",
        target=TurnTarget(target_heading_deg=0.0, direction="left"),
        config=config,
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        objective_value=0.0,
        metrics={
            "target_heading_deg": 0.0,
            "dynamics_defect_max": 0.0,
            "slack_max": 0.0,
        },
        solver_stats={"initial_guess_name": "unit"},
        nu_ff=np.zeros((3, 3), dtype=float),
    )
    context = build_primitive_context(linear_model.x_trim, linear_model.u_trim)
    return result, aircraft, context


def test_turn_result_converts_to_finite_trajectory_primitive(tmp_path: Path) -> None:
    result, aircraft, context = _synthetic_result()

    primitive = build_turn_trajectory_primitive(
        result=result,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        command_layer=CommandToSurfaceLayer(),
    )
    paths = save_turn_result(result, tmp_path, "unit_turn", primitive=primitive)
    loaded = load_selected_turn_primitive(paths["trajectory"])

    assert primitive.times_s.shape == (3,)
    assert primitive.x_ref.shape == (3, 15)
    assert primitive.u_ff.shape == (3, 3)
    assert primitive.a_mats is not None and primitive.a_mats.shape == (3, 15, 15)
    assert primitive.b_mats is not None and primitive.b_mats.shape == (3, 15, 3)
    assert primitive.k_lqr.shape == (3, 3, 15)
    assert primitive.s_mats is not None and primitive.s_mats.shape == (3, 15, 15)
    assert np.all(np.isfinite(primitive.k_lqr))
    assert loaded.k_lqr.shape == primitive.k_lqr.shape

