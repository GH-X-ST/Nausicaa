from __future__ import annotations

import numpy as np

from conftest import REPO_ROOT
from updraft_models import load_updraft_model


def test_available_measured_updraft_models_return_world_wind_shape() -> None:
    points = np.array(
        [
            [4.2, 2.4, 0.2],
            [4.0, 2.0, 1.1],
            [3.0, 3.6, 2.2],
        ],
        dtype=float,
    )
    for name in (
        "single_gaussian_var",
        "four_gaussian_var",
        "single_annular_gp_grid",
        "four_annular_gp_grid",
    ):
        model = load_updraft_model(name, repo_root=REPO_ROOT)
        wind = model(points)
        assert wind.shape == (3, 3)
        assert np.all(np.isfinite(wind))
        assert np.allclose(wind[:, :2], 0.0)
