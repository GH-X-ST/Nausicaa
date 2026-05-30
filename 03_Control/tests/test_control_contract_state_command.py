from __future__ import annotations

import numpy as np
import pytest

from command_contract import (
    COMMAND_INDEX,
    COMMAND_NAMES,
    COMMAND_SIZE,
    CONTROL_SIGN_CONVENTION,
    as_normalised_command_vector,
    as_surface_command_rad,
    command_contract_row,
    command_dataframe_row,
    clip_normalised_command,
    normalised_command_dataframe_row,
    normalised_command_to_surface_rad,
    quantise_normalised_command_vector,
    surface_rad_to_normalised_command,
)
from flight_dynamics import adapt_glider, state_derivative
from glider import build_nausicaa_glider
from latency import AGGREGATE_LIMITS, command_norm_to_angle
from linearisation import INPUT_NAMES, STATE_NAMES as LINEARISATION_STATE_NAMES
from state_contract import (
    STATE_INDEX,
    STATE_NAMES,
    STATE_SIZE,
    as_state_vector,
    pack_state,
    state_dataframe_row,
    unpack_state,
)


def test_state_names_match_canonical_order() -> None:
    assert STATE_NAMES == (
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
    assert STATE_SIZE == 15
    assert STATE_INDEX["delta_r"] == 14
    assert STATE_NAMES == LINEARISATION_STATE_NAMES


def test_state_pack_unpack_round_trip() -> None:
    values = {name: float(index) for index, name in enumerate(STATE_NAMES)}
    state = pack_state(values)

    assert np.allclose(state, np.arange(STATE_SIZE, dtype=float))
    assert unpack_state(state) == values
    assert state_dataframe_row(state, prefix="trim_")["trim_theta"] == values["theta"]


def test_state_validation_rejects_wrong_size_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="15"):
        as_state_vector(np.zeros(14))
    bad = np.zeros(15)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        as_state_vector(bad)


def test_pack_state_requires_exact_keys() -> None:
    values = {name: 0.0 for name in STATE_NAMES}
    values.pop("psi")
    with pytest.raises(ValueError, match="missing"):
        pack_state(values)

    values = {name: 0.0 for name in STATE_NAMES}
    values["extra"] = 0.0
    with pytest.raises(ValueError, match="unknown"):
        pack_state(values)


def test_command_names_match_canonical_order() -> None:
    assert COMMAND_NAMES == ("delta_a_cmd", "delta_e_cmd", "delta_r_cmd")
    assert COMMAND_SIZE == 3
    assert COMMAND_INDEX["delta_r_cmd"] == 2
    assert COMMAND_NAMES == INPUT_NAMES


def test_normalised_command_validation_and_clipping() -> None:
    command = as_normalised_command_vector([0.0, -0.5, 0.5])

    assert command.shape == (3,)
    assert np.allclose(clip_normalised_command([-2.0, 0.2, 2.0]), [-1.0, 0.2, 1.0])
    assert normalised_command_dataframe_row(command, prefix="cmd_")[
        "cmd_delta_e_norm"
    ] == -0.5
    with pytest.raises(ValueError, match="3"):
        as_normalised_command_vector(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        as_normalised_command_vector([0.0, np.inf, 0.0])


def test_executable_command_quantisation_uses_20_percent_lattice() -> None:
    command = quantise_normalised_command_vector([-0.91, -0.11, 0.71])

    assert np.allclose(command, [-1.0, -0.2, 0.8])


def test_normalised_to_radian_command_bridge_matches_latency_limits() -> None:
    normalised = np.array([1.0, -1.0, 1.0])
    expected = np.array(
        [
            command_norm_to_angle(1.0, AGGREGATE_LIMITS["delta_a"]),
            command_norm_to_angle(-1.0, AGGREGATE_LIMITS["delta_e"]),
            command_norm_to_angle(1.0, AGGREGATE_LIMITS["delta_r"]),
        ]
    )

    surface_rad = normalised_command_to_surface_rad(normalised)

    assert np.allclose(surface_rad, expected)
    assert np.allclose(surface_rad_to_normalised_command(surface_rad), normalised)
    assert command_dataframe_row(surface_rad, prefix="cmd_")["cmd_delta_e_cmd"] == (
        expected[1]
    )
    with pytest.raises(ValueError, match="calibrated aggregate limits"):
        as_surface_command_rad([10.0, 0.0, 0.0])


def test_state_derivative_consumes_radian_surface_command_order() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["u"]] = 6.0
    delta_cmd_rad = np.array([0.01, -0.02, 0.03])

    derivative = state_derivative(
        state,
        delta_cmd_rad,
        aircraft,
        wind_model=None,
        actuator_tau_s=(1.0, 1.0, 1.0),
    )

    assert np.allclose(derivative[12:15], delta_cmd_rad)


def test_control_sign_convention_is_recorded() -> None:
    for key in ("positive_aileron", "positive_elevator", "positive_rudder"):
        assert key in CONTROL_SIGN_CONVENTION
        assert CONTROL_SIGN_CONVENTION[key]
    row = command_contract_row()
    assert row["command_names"] == "delta_a_cmd,delta_e_cmd,delta_r_cmd"
    assert row["command_units"] == "rad"
    assert row["command_interface_to_state_derivative"] == "delta_cmd_rad"
    assert row["normalised_to_radian_bridge"] == "normalised_command_to_surface_rad"
    assert row["raw_normalised_commands_enter_state_derivative"] is False
    assert row["continuous_lqr_commands_enter_state_derivative"] is False
    assert row["executable_command_quantisation"] == "fixed_20_percent_lattice"
    assert row["normalised_command_min"] == -1.0
    assert row["normalised_command_max"] == 1.0
    assert "delta_a" in row["aggregate_limits"]
