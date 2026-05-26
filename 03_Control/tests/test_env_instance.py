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
