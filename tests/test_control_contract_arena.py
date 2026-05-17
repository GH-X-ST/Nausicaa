from __future__ import annotations

import numpy as np
import pytest

from arena import ArenaConfig, safe_bounds, tracker_bounds
from arena_contract import (
    TRACKER_LIMIT_BOUNDS,
    TRUE_SAFE_BOUNDS,
    arena_contract_row,
    as_position_vector,
    classify_position,
    inside_bounds,
    position_margin_m,
)


def test_true_safe_and_tracker_boxes_are_separate() -> None:
    assert TRUE_SAFE_BOUNDS != TRACKER_LIMIT_BOUNDS
    assert TRUE_SAFE_BOUNDS.x_w_m[0] >= TRACKER_LIMIT_BOUNDS.x_w_m[0]
    assert TRUE_SAFE_BOUNDS.x_w_m[1] <= TRACKER_LIMIT_BOUNDS.x_w_m[1]
    assert TRUE_SAFE_BOUNDS.y_w_m[0] >= TRACKER_LIMIT_BOUNDS.y_w_m[0]
    assert TRUE_SAFE_BOUNDS.y_w_m[1] <= TRACKER_LIMIT_BOUNDS.y_w_m[1]
    assert TRUE_SAFE_BOUNDS.z_w_m[0] >= TRACKER_LIMIT_BOUNDS.z_w_m[0]
    assert TRUE_SAFE_BOUNDS.z_w_m[1] <= TRACKER_LIMIT_BOUNDS.z_w_m[1]


def test_contract_bounds_match_arena_module() -> None:
    config = ArenaConfig()
    tracker = tracker_bounds(config)
    true_safe = safe_bounds(config)

    assert TRACKER_LIMIT_BOUNDS.x_w_m == tracker["x_w"]
    assert TRACKER_LIMIT_BOUNDS.y_w_m == tracker["y_w"]
    assert TRACKER_LIMIT_BOUNDS.z_w_m == tracker["z_w"]
    assert TRUE_SAFE_BOUNDS.x_w_m == true_safe["x_w"]
    assert TRUE_SAFE_BOUNDS.y_w_m == true_safe["y_w"]
    assert TRUE_SAFE_BOUNDS.z_w_m == true_safe["z_w"]


def test_position_validation_rejects_bad_inputs() -> None:
    assert np.allclose(as_position_vector([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="3"):
        as_position_vector([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        as_position_vector([1.0, np.nan, 3.0])


def test_position_classification() -> None:
    assert classify_position([2.0, 2.0, 1.0]) == "inside_true_safe"
    assert classify_position([0.5, 2.0, 1.0]) == "outside_true_safe_inside_tracker"
    assert classify_position([-0.1, 2.0, 1.0]) == "outside_tracker"


def test_margins_have_correct_sign() -> None:
    inside = position_margin_m([2.0, 2.0, 1.0], TRUE_SAFE_BOUNDS)
    outside = position_margin_m([0.5, 2.0, 1.0], TRUE_SAFE_BOUNDS)

    assert inside["min_margin_m"] > 0.0
    assert outside["x_min_margin_m"] < 0.0
    assert inside_bounds([2.0, 2.0, 1.0], TRUE_SAFE_BOUNDS)
    assert not inside_bounds([0.5, 2.0, 1.0], TRUE_SAFE_BOUNDS)


def test_arena_contract_row_includes_both_boxes() -> None:
    row = arena_contract_row()

    assert row["tracker_limit_name"] == "tracker_limit"
    assert row["true_safe_name"] == "true_safe"
    assert "8.0" in row["tracker_limit_x_w_m"]
    assert "6.6" in row["true_safe_x_w_m"]
