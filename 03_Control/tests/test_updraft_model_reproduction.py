from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import REPO_ROOT
from updraft_models import SINGLE_FAN_CENTER_XY, load_updraft_model


def test_single_gaussian_reproduces_workbook_center_value() -> None:
    path = REPO_ROOT / "01_Thermal" / "B_results" / "single_var_params.xlsx"
    df = pd.read_excel(path, sheet_name="single_var")
    row = df.iloc[0]
    model = load_updraft_model("single_gaussian_var", repo_root=REPO_ROOT)
    point = np.array([[SINGLE_FAN_CENTER_XY[0], SINGLE_FAN_CENTER_XY[1], row["z_m"]]])
    expected = float(row["w0"] + row["A"])
    assert np.isclose(model(point)[0, 2], expected, rtol=1e-10, atol=1e-10)
