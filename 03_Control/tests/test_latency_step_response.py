from __future__ import annotations

import numpy as np

from latency import (
    CommandToSurfaceConfig,
    LatencyEnvelope,
    actuator_tau_s,
    half_response_s,
)


def _continuous_half_time(config: CommandToSurfaceConfig) -> float:
    envelope = LatencyEnvelope()
    tau = actuator_tau_s(config, envelope)
    return envelope.onset_latency_s + tau * np.log(2.0)


def test_nominal_latency_half_response() -> None:
    config = CommandToSurfaceConfig(mode="nominal")
    assert np.isclose(_continuous_half_time(config), half_response_s(config), atol=1e-12)


def test_robust_upper_latency_half_response() -> None:
    config = CommandToSurfaceConfig(mode="robust_upper")
    assert np.isclose(_continuous_half_time(config), half_response_s(config), atol=1e-12)
