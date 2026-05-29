from __future__ import annotations

import numpy as np

from env_ctx import build_environment_context
from env_instance import (
    EnvironmentRandomisationConfig,
    environment_instance_for_mode,
    environment_metadata_from_instance,
    sample_environment_randomisation,
)
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding
from state_sampling import archive_state_sample_for_row


def _wind_value(instance, point=(4.2, 2.4, 1.6)) -> float:
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(instance.W_layer, metadata, randomisation_seed=instance.randomisation_seed)
    wind = wind_field_for_binding(binding)
    assert wind is not None
    return float(wind(np.asarray([point], dtype=float))[0, 2])


def test_w01_environment_instances_are_deterministic_and_official() -> None:
    first = environment_instance_for_mode("W1", "gaussian_single", 101)
    second = environment_instance_for_mode("W1", "gaussian_single", 101)
    four = environment_instance_for_mode("W1", "gaussian_four", 101)

    assert first == second
    assert first.fan_count == 1
    assert four.fan_count == 4
    assert first.environment_id != four.environment_id


def test_w01_gaussian_single_and_four_change_wind_values() -> None:
    single = environment_instance_for_mode("W1", "gaussian_single", 7)
    four = environment_instance_for_mode("W1", "gaussian_four", 7)

    assert not np.isclose(_wind_value(single), _wind_value(four))


def test_w01_active_annular_gp_training_modes_use_annular_gp_models() -> None:
    single = environment_instance_for_mode("W1", "w1_annular_gp_randomised_single", 7)
    four = environment_instance_for_mode("W1", "w1_annular_gp_randomised_four", 8)

    assert single.fan_count == 1
    assert four.fan_count == 4
    assert single.updraft_model_id == "single_annular_gp_grid"
    assert four.updraft_model_id == "four_annular_gp_grid"
    assert single.residual_field_id == "single_layer_annular_gp_randomisation_no_duplicate_strength_or_shift"
    assert four.residual_field_id == "single_layer_annular_gp_randomisation_no_duplicate_strength_or_shift"
    assert single.updraft_amplitude_scale == 1.0
    assert four.updraft_amplitude_scale == 1.0
    assert single.updraft_centre_shift_m == (0.0, 0.0)
    assert four.updraft_centre_shift_m == (0.0, 0.0)
    assert any(not np.isclose(value, 1.0) for value in single.fan_power_scales)
    assert any(not np.isclose(value, 1.0) for value in four.fan_power_scales)
    assert wind_field_for_binding(
        resolve_surrogate_binding("W1", environment_metadata_from_instance(single), randomisation_seed=7)
    ) is not None


def test_randomisation_defaults_do_not_duplicate_strength_or_position_channels() -> None:
    cfg = EnvironmentRandomisationConfig()

    assert cfg.fan_power_scale_range != (1.0, 1.0)
    assert cfg.amplitude_scale_range == (1.0, 1.0)
    assert cfg.centre_shift_range_m == (0.0, 0.0)


def test_context_uses_conservative_nonzero_uncertainty_for_fitted_wind() -> None:
    instance = environment_instance_for_mode("W1", "gaussian_single", 11)
    state = archive_state_sample_for_row(0, seed=11, W_layer="W1", environment_mode="gaussian_single").state_vector
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding("W1", metadata, randomisation_seed=11)
    context = build_environment_context(
        state,
        wind_field=wind_field_for_binding(binding),
        metadata=metadata,
        latency_case="nominal",
        surrogate_binding=binding,
    )

    assert context.w_local_uncertainty_m_s > 0.0
    assert context.w_local_uncertainty_status in {
        "available_from_model",
        "conservative_fallback_nonzero",
    }


def test_invalid_w1_diagnostic_modes_are_blocked() -> None:
    shifted_mode = "fan" + "_shift"
    powered_mode = "power" + "_scale"
    shifted = environment_instance_for_mode("W1", shifted_mode, 1)
    powered = environment_instance_for_mode("W1", powered_mode, 1)

    assert shifted.instance_status == "blocked"
    assert powered.instance_status == "blocked"


def test_w3_randomisation_mode_selection_is_explicit_for_single_and_four() -> None:
    single = environment_instance_for_mode("W1", "gaussian_single", 3)
    four = environment_instance_for_mode("W1", "gaussian_four", 3)

    assert sample_environment_randomisation(single, 4).environment_mode == "w3_randomised_single"
    assert sample_environment_randomisation(four, 4).environment_mode == "w3_randomised_four"


def test_w3_four_fan_randomisation_can_request_exact_active_fan_count() -> None:
    instance = environment_instance_for_mode(
        "W3",
        "w3_randomised_four",
        9,
        randomisation_config=EnvironmentRandomisationConfig(active_fan_count=2),
    )

    assert instance.fan_count == 4
    assert sum(instance.active_fan_mask) == 2

    no_active = environment_instance_for_mode(
        "W3",
        "w3_randomised_four",
        10,
        randomisation_config=EnvironmentRandomisationConfig(active_fan_count=0),
    )
    assert no_active.fan_count == 4
    assert sum(no_active.active_fan_mask) == 0


def test_r10_nominal_fan_position_policy_keeps_base_positions() -> None:
    instance = environment_instance_for_mode(
        "W3",
        "w3_randomised_four",
        15,
        randomisation_config=EnvironmentRandomisationConfig(
            active_fan_count=4,
            fan_position_policy="fixed_base_positions",
        ),
    )

    assert instance.fan_count == 4
    assert sum(instance.active_fan_mask) == 4
    assert instance.fan_positions_m == ((3.0, 3.6), (5.4, 3.6), (3.0, 1.2), (5.4, 1.2))


def test_r10_arena_wide_fan_position_policy_samples_inside_tracker_footprint() -> None:
    instance = environment_instance_for_mode(
        "W3",
        "w3_randomised_four",
        15,
        randomisation_config=EnvironmentRandomisationConfig(
            active_fan_count=4,
            fan_position_policy="independent_uniform_xy_bounds",
            fan_position_xy_bounds_m=((0.0, 8.0), (0.0, 4.8)),
        ),
    )

    assert instance.fan_count == 4
    assert sum(instance.active_fan_mask) == 4
    assert all(0.0 <= x <= 8.0 and 0.0 <= y <= 4.8 for x, y in instance.fan_positions_m)
    for index, first in enumerate(instance.fan_positions_m):
        for second in instance.fan_positions_m[index + 1 :]:
            assert float(np.linalg.norm(np.asarray(first) - np.asarray(second))) >= 1.0
    assert any(abs(x - nominal_x) > 0.20 or abs(y - nominal_y) > 0.20 for (x, y), (nominal_x, nominal_y) in zip(instance.fan_positions_m, ((3.0, 3.6), (5.4, 3.6), (3.0, 1.2), (5.4, 1.2)), strict=True))


def test_split_environment_randomisation_seeds_keep_layout_and_count_fixed_while_parameters_vary() -> None:
    first = environment_instance_for_mode(
        "W3",
        "w3_randomised_four",
        501,
        randomisation_config=EnvironmentRandomisationConfig(
            active_fan_count=2,
            fan_position_policy="independent_uniform_xy_bounds",
            fan_layout_seed=11,
            active_fan_seed=12,
            fan_parameter_seed=13,
        ),
    )
    second = environment_instance_for_mode(
        "W3",
        "w3_randomised_four",
        502,
        randomisation_config=EnvironmentRandomisationConfig(
            active_fan_count=2,
            fan_position_policy="independent_uniform_xy_bounds",
            fan_layout_seed=11,
            active_fan_seed=12,
            fan_parameter_seed=14,
        ),
    )

    assert first.fan_positions_m == second.fan_positions_m
    assert first.active_fan_mask == second.active_fan_mask
    assert first.fan_power_scales != second.fan_power_scales
    assert first.updraft_width_scale != second.updraft_width_scale
    assert first.local_uncertainty_scale != second.local_uncertainty_scale
