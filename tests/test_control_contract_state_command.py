from __future__ import annotations

import numpy as np
import pytest

from command_contract import (
    COMMAND_INDEX,
    COMMAND_NAMES,
    COMMAND_SIZE,
    CONTROL_SIGN_CONVENTION,
    clip_normalised_command,
    command_contract_row,
    command_dataframe_row,
    as_command_vector,
)
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


def test_command_vector_validation_and_clipping() -> None:
    command = as_command_vector([0.0, -0.5, 0.5])

    assert command.shape == (3,)
    assert np.allclose(clip_normalised_command([-2.0, 0.2, 2.0]), [-1.0, 0.2, 1.0])
    assert command_dataframe_row(command, prefix="cmd_")["cmd_delta_e_cmd"] == -0.5
    with pytest.raises(ValueError, match="3"):
        as_command_vector(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        as_command_vector([0.0, np.inf, 0.0])


def test_control_sign_convention_is_recorded() -> None:
    for key in ("positive_aileron", "positive_elevator", "positive_rudder"):
        assert key in CONTROL_SIGN_CONVENTION
        assert CONTROL_SIGN_CONVENTION[key]
    row = command_contract_row()
    assert row["command_names"] == "delta_a_cmd,delta_e_cmd,delta_r_cmd"
    assert row["normalised_command_min"] == -1.0
    assert row["normalised_command_max"] == 1.0
