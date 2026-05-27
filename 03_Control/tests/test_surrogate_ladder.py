from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env_ctx import EnvironmentMetadata
from updraft_models import FOUR_FAN_CENTERS_XY
from env_surrogate import (
    READY_STATUS,
    resolve_surrogate_binding,
    validate_surrogate_ladder,
    wind_field_for_binding,
)


def test_w1_accepts_gaussian_plume_only() -> None:
    metadata = EnvironmentMetadata(
        environment_id="W1_single",
        fan_count=1,
        updraft_model_id="single_gaussian_var",
    )

    binding = resolve_surrogate_binding("W1", metadata)

    assert binding.surrogate_binding_status == READY_STATUS
    assert binding.surrogate_family == "gaussian_plume"
    assert binding.wind_mode == "panel"
    validate_surrogate_ladder(binding)


def test_w0_accepts_dry_air_only() -> None:
    dry = resolve_surrogate_binding(
        "W0",
        EnvironmentMetadata(
            environment_id="W0_dry",
            fan_count=0,
            updraft_model_id="dry_air_zero_wind",
        ),
    )
    invalid = resolve_surrogate_binding(
        "W0",
        EnvironmentMetadata(
            environment_id="bad_W0",
            fan_count=1,
            updraft_model_id="single_gaussian_var",
        ),
    )

    assert dry.surrogate_binding_status == READY_STATUS
    assert dry.surrogate_family == "dry_air_zero_wind"
    assert dry.wind_mode == "none"
    assert invalid.surrogate_binding_status == "blocked"
    assert invalid.blocked_reason == "W0_requires_dry_air_zero_wind"


def test_w1_blocks_annular_gp_surrogate() -> None:
    metadata = EnvironmentMetadata(
        environment_id="bad_W1",
        fan_count=1,
        updraft_model_id="single_annular_gp_grid",
    )

    binding = resolve_surrogate_binding("W1", metadata)

    assert binding.surrogate_binding_status == "blocked"
    assert "W1_requires_gaussian" in binding.blocked_reason
    with pytest.raises(ValueError, match="blocked"):
        validate_surrogate_ladder(binding)


def test_w1_annular_gp_training_mode_accepts_randomised_annular_gp() -> None:
    metadata = EnvironmentMetadata(
        environment_id="W1_annular_train",
        fan_count=1,
        updraft_model_id="single_annular_gp_grid",
        environment_mode="w1_annular_gp_randomised_single",
        randomisation_seed=23,
    )

    binding = resolve_surrogate_binding("W1", metadata, randomisation_seed=23)
    wind = wind_field_for_binding(binding)

    assert binding.surrogate_binding_status == READY_STATUS
    assert binding.surrogate_family == "randomised_gp_corrected_annular_gaussian_training"
    assert "single_layer_annular_gp_randomisation_v1" in binding.randomisation_label
    assert "extra_randomised_wind_wrapper" in binding.randomisation_label
    assert wind is not None


def test_w2_blocks_gaussian_fallback() -> None:
    metadata = EnvironmentMetadata(
        environment_id="bad_W2",
        fan_count=1,
        updraft_model_id="single_gaussian_var",
    )

    binding = resolve_surrogate_binding("W2", metadata)

    assert binding.surrogate_binding_status == "blocked"
    assert "W2_requires" in binding.blocked_reason


def test_w3_is_randomised_gp_corrected_annular_only() -> None:
    metadata = EnvironmentMetadata(
        environment_id="W3_single",
        fan_count=1,
        updraft_model_id="single_annular_gp_grid",
        randomisation_seed=17,
    )

    binding = resolve_surrogate_binding("W3", metadata, randomisation_seed=17)
    wind = wind_field_for_binding(binding)

    assert binding.surrogate_binding_status == READY_STATUS
    assert binding.surrogate_family == "randomised_gp_corrected_annular_gaussian"
    assert "single_layer_annular_gp_randomisation_v1" in binding.randomisation_label
    assert wind is not None
    assert "composed_annular_gp" in wind.source


def test_w3_annular_active_fan_mask_changes_composed_wind() -> None:
    positions = tuple((float(x), float(y)) for x, y in FOUR_FAN_CENTERS_XY)
    one_active = EnvironmentMetadata(
        environment_id="W3_four_one_active",
        fan_count=4,
        fan_positions_m=positions,
        fan_power_scales=(1.0, 1.0, 1.0, 1.0),
        active_fan_mask=(True, False, False, False),
        updraft_model_id="four_annular_gp_grid",
        environment_mode="w3_randomised_four",
        randomisation_seed=17,
    )
    all_active = EnvironmentMetadata(
        environment_id="W3_four_all_active",
        fan_count=4,
        fan_positions_m=positions,
        fan_power_scales=(1.0, 1.0, 1.0, 1.0),
        active_fan_mask=(True, True, True, True),
        updraft_model_id="four_annular_gp_grid",
        environment_mode="w3_randomised_four",
        randomisation_seed=17,
    )

    point = np.asarray([[4.2, 2.4, 1.6]], dtype=float)
    one_wind = wind_field_for_binding(resolve_surrogate_binding("W3", one_active, randomisation_seed=17))
    all_wind = wind_field_for_binding(resolve_surrogate_binding("W3", all_active, randomisation_seed=17))

    assert one_wind is not None
    assert all_wind is not None
    assert not np.isclose(float(one_wind(point)[0, 2]), float(all_wind(point)[0, 2]))


def test_missing_surrogate_is_blocked_not_replaced(tmp_path: Path) -> None:
    metadata = EnvironmentMetadata(
        environment_id="missing_W1",
        fan_count=1,
        updraft_model_id="single_gaussian_var",
    )

    binding = resolve_surrogate_binding("W1", metadata, repo_root=tmp_path)

    assert binding.surrogate_binding_status == "blocked"
    assert "missing_or_unreadable_surrogate" in binding.blocked_reason
