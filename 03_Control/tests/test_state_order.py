from __future__ import annotations

from linearisation import INPUT_NAMES, STATE_NAMES


def test_canonical_state_and_command_order() -> None:
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
    assert INPUT_NAMES == ("delta_a_cmd", "delta_e_cmd", "delta_r_cmd")
