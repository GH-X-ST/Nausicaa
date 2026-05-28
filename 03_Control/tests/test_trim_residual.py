from __future__ import annotations

import numpy as np

from linearisation import linearise_trim
from trim_solver import TrimTarget


def test_trim_dynamic_residual_excludes_position_rates() -> None:
    model = linearise_trim(target=TrimTarget(speed_m_s=4.8))
    assert float(np.max(np.abs(model.f_trim[3:]))) < 1e-7
