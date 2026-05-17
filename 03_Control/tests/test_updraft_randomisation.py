from __future__ import annotations

import numpy as np

from updraft_models import (
    build_randomised_wind_field,
    load_updraft_model,
    sample_updraft_randomisation,
    updraft_randomisation_label,
)


def test_updraft_randomisation_is_seed_deterministic() -> None:
    first = sample_updraft_randomisation(seed=11, enabled=True)
    second = sample_updraft_randomisation(seed=11, enabled=True)
    third = sample_updraft_randomisation(seed=12, enabled=True)

    assert first == second
    assert first != third
    assert updraft_randomisation_label(first) == updraft_randomisation_label(second)


def test_randomised_wind_samples_once_outside_call_loop() -> None:
    base = load_updraft_model("analytic_debug_proxy")
    wind, label = build_randomised_wind_field(base, seed=9, enabled=True)
    points = np.array([[4.2, 2.4, 1.5], [4.0, 2.2, 1.3]], dtype=float)

    first = wind(points)
    second = wind(points)

    assert np.allclose(first, second)
    assert "strength_scale" in label
    assert "not_applied" in label

