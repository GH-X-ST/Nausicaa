from __future__ import annotations

from linearisation import key_derivatives, linearise_trim


def test_control_effectiveness_signs() -> None:
    derivatives = key_derivatives(linearise_trim())
    assert derivatives["l_delta_a"] > 0.0
    assert derivatives["m_delta_e"] > 0.0
    assert derivatives["n_delta_r"] > 0.0
