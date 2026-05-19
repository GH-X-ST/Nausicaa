from __future__ import annotations

import numpy as np
import pytest

import primitive_library
from command_contract import normalised_command_to_surface_rad
from latency import actuator_tau_for_case, format_actuator_tau_s, latency_case_config
from primitive_library import WindModelInfo, evaluate_candidate
from primitive_library_schema import PrimitiveCandidateSpec, PrimitiveLibraryConfig


def _spec(wind_fidelity: str = "W0") -> PrimitiveCandidateSpec:
    return PrimitiveCandidateSpec(
        primitive_id="latency_test_mild_bank",
        parent_primitive_id="mild_bank_none",
        variant_id="latency_test_mild_bank",
        family="mild_bank",
        target_heading_deg=None,
        updraft_config="none",
        wind_fidelity=wind_fidelity,
        start_condition="favourable",
        direction_sign=1,
        horizon_s=0.40,
    )


def _wind_info(
    available: bool = True,
    evaluation_status: str = "evaluated",
) -> WindModelInfo:
    return WindModelInfo(
        available=available,
        model=None,
        name="none",
        source="test",
        z_axis_m=None,
        evaluation_status=evaluation_status,
    )


def _step_profile(
    spec: PrimitiveCandidateSpec,
    time_s: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    del spec
    commands = np.zeros((time_s.size, 3))
    commands[time_s >= 0.10, 0] = 1.0
    return commands, tuple("test" for _ in time_s)


def _surface_only_step(
    x: np.ndarray,
    delta_cmd_rad: np.ndarray,
    dt_s: float,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    actuator_tau_s: tuple[float, float, float],
) -> np.ndarray:
    del aircraft, wind_model, wind_mode
    state = np.asarray(x, dtype=float).copy()
    tau = np.asarray(actuator_tau_s, dtype=float)
    alpha = 1.0 - np.exp(-float(dt_s) / tau)
    state[12:15] += alpha * (np.asarray(delta_cmd_rad) - state[12:15])
    return state


def _first_surface_crossing(time_s: np.ndarray, surface: np.ndarray) -> float:
    target = normalised_command_to_surface_rad(np.array([1.0, 0.0, 0.0]))[0]
    indices = np.flatnonzero(np.asarray(surface) >= 0.25 * target)
    if indices.size == 0:
        return float("inf")
    return float(time_s[int(indices[0])])


def test_primitive_replay_surface_timing_orders_latency_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primitive_library, "generate_command_profile", _step_profile)
    monkeypatch.setattr(primitive_library, "rk4_step", _surface_only_step)
    crossings: dict[str, float] = {}

    for latency_case in ("none", "actuator_lag_only", "nominal", "conservative"):
        result = evaluate_candidate(
            _spec(),
            PrimitiveLibraryConfig(
                latency_case=latency_case,
                wind_fidelities=("W0",),
                updraft_configs=("none",),
            ),
            _wind_info(),
            aircraft=object(),
        )
        crossings[latency_case] = _first_surface_crossing(
            result.time_s,
            result.x_ref[:, 12],
        )

    assert crossings["none"] == pytest.approx(0.10)
    assert crossings["none"] < crossings["actuator_lag_only"]
    assert crossings["actuator_lag_only"] < crossings["nominal"]
    assert crossings["nominal"] < crossings["conservative"]


def test_primitive_saturation_uses_effective_command_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primitive_library, "generate_command_profile", _step_profile)
    monkeypatch.setattr(primitive_library, "rk4_step", _surface_only_step)

    result = evaluate_candidate(
        _spec(),
        PrimitiveLibraryConfig(
            latency_case="nominal",
            wind_fidelities=("W0",),
            updraft_configs=("none",),
        ),
        _wind_info(),
        aircraft=object(),
    )

    assert not np.allclose(result.u_norm_requested, result.u_norm_effective_target)
    assert np.allclose(result.u_norm_effective_target, result.u_norm_applied)
    assert result.row.saturation_fraction == 0.0


def test_primitive_evidence_latency_metadata_and_single_run_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primitive_library, "generate_command_profile", _step_profile)
    monkeypatch.setattr(primitive_library, "rk4_step", _surface_only_step)

    result = evaluate_candidate(
        _spec(),
        PrimitiveLibraryConfig(
            latency_case="nominal",
            wind_fidelities=("W0",),
            updraft_configs=("none",),
        ),
        _wind_info(),
        aircraft=object(),
    )
    row = result.row
    config = latency_case_config("nominal")

    assert row.latency_case == "nominal"
    assert row.actuator_tau_s == format_actuator_tau_s(config.actuator_tau_s)
    assert row.latency_pass_label == "nominal_pass"
    assert row.latency_acceptance_scope == "command_path_nominal_no_feedback_controller"
    assert row.state_feedback_delay_applied is False
    assert row.command_delay_applied is True
    assert row.actuator_lag_applied is True


def test_primitive_evidence_active_audit_fields_match_replay_tau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(primitive_library, "generate_command_profile", _step_profile)
    monkeypatch.setattr(primitive_library, "rk4_step", _surface_only_step)

    none = evaluate_candidate(
        _spec(),
        PrimitiveLibraryConfig(
            latency_case="none",
            wind_fidelities=("W0",),
            updraft_configs=("none",),
        ),
        _wind_info(),
        aircraft=object(),
    ).row
    assert none.actuator_tau_s == "0.000000000;0.000000000;0.000000000"
    assert none.actuator_t50_s == 0.0
    assert none.actuator_t90_s == 0.0
    assert none.actuator_lag_applied is False

    lag_config = latency_case_config("actuator_lag_only")
    active_tau = actuator_tau_for_case(lag_config)
    lag = evaluate_candidate(
        _spec(),
        PrimitiveLibraryConfig(
            latency_case="actuator_lag_only",
            wind_fidelities=("W0",),
            updraft_configs=("none",),
        ),
        _wind_info(),
        aircraft=object(),
    ).row
    assert lag.actuator_tau_s == format_actuator_tau_s(active_tau)
    assert lag.actuator_t50_s == pytest.approx(max(active_tau) * np.log(2.0))
    assert lag.actuator_t90_s == pytest.approx(max(active_tau) * np.log(10.0))
    assert lag.actuator_lag_applied is True

    for latency_case in ("nominal", "conservative"):
        case_config = latency_case_config(latency_case)
        row = evaluate_candidate(
            _spec(),
            PrimitiveLibraryConfig(
                latency_case=latency_case,
                wind_fidelities=("W0",),
                updraft_configs=("none",),
            ),
            _wind_info(),
            aircraft=object(),
        ).row
        assert row.actuator_tau_s == format_actuator_tau_s(
            actuator_tau_for_case(case_config)
        )
        assert row.actuator_t50_s == pytest.approx(case_config.actuator_t50_s)
        assert row.actuator_t90_s == pytest.approx(case_config.actuator_t90_s)
        assert row.actuator_lag_applied is True


def test_not_evaluated_result_has_latency_metadata_and_tau_format() -> None:
    result = evaluate_candidate(
        _spec(wind_fidelity="W1"),
        PrimitiveLibraryConfig(
            latency_case="nominal",
            wind_fidelities=("W1",),
            updraft_configs=("none",),
        ),
        _wind_info(available=False, evaluation_status="not_evaluated_model_missing"),
        aircraft=object(),
    )
    row = result.row

    assert row.evaluation_status == "not_evaluated_model_missing"
    assert row.latency_case == "nominal"
    assert row.latency_pass_label == "nominal_fail"
    assert row.actuator_tau_s == format_actuator_tau_s(
        latency_case_config("nominal").actuator_tau_s
    )
    assert row.state_feedback_delay_applied is False
    assert row.command_delay_applied is True
    assert row.actuator_lag_applied is False


def test_not_evaluated_actuator_lag_only_uses_active_audit_tau() -> None:
    result = evaluate_candidate(
        _spec(wind_fidelity="W1"),
        PrimitiveLibraryConfig(
            latency_case="actuator_lag_only",
            wind_fidelities=("W1",),
            updraft_configs=("none",),
        ),
        _wind_info(available=False, evaluation_status="not_evaluated_model_missing"),
        aircraft=object(),
    )
    row = result.row
    active_tau = actuator_tau_for_case(latency_case_config("actuator_lag_only"))

    assert row.evaluation_status == "not_evaluated_model_missing"
    assert row.actuator_tau_s == format_actuator_tau_s(active_tau)
    assert row.actuator_t50_s == pytest.approx(max(active_tau) * np.log(2.0))
    assert row.actuator_t90_s == pytest.approx(max(active_tau) * np.log(10.0))
    assert row.actuator_lag_applied is False
