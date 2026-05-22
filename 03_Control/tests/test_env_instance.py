from __future__ import annotations

import numpy as np

from env_ctx import build_environment_context
from env_instance import environment_instance_for_mode, environment_metadata_from_instance
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding
from state_sampling import archive_state_sample_for_row


def _wind_value(instance, point=(4.2, 2.4, 1.6)) -> float:
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(instance.W_layer, metadata, randomisation_seed=instance.randomisation_seed)
    wind = wind_field_for_binding(binding)
    assert wind is not None
    return float(wind(np.asarray([point], dtype=float))[0, 2])


def test_environment_instances_are_deterministic_and_distinct() -> None:
    first = environment_instance_for_mode("W1", "fan_shift", 101)
    second = environment_instance_for_mode("W1", "fan_shift", 101)
    power = environment_instance_for_mode("W1", "power_scale", 101)

    assert first == second
    assert first.environment_id != power.environment_id
    assert first.fan_positions_m != power.fan_positions_m
    assert power.fan_power_scales != tuple(1.0 for _ in power.fan_power_scales)


def test_environment_adjustments_change_wind_values() -> None:
    base = environment_instance_for_mode("W1", "gaussian_four", 7)
    shifted = environment_instance_for_mode("W1", "fan_shift", 7)
    powered = environment_instance_for_mode("W1", "power_scale", 7)

    base_w = _wind_value(base)
    shifted_w = _wind_value(shifted)
    powered_w = _wind_value(powered)

    assert not np.isclose(base_w, shifted_w)
    assert not np.isclose(base_w, powered_w)


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


def test_invalid_environment_mode_is_blocked() -> None:
    instance = environment_instance_for_mode("W1", "unsupported_mode", 1)

    assert instance.instance_status == "blocked"
    assert "unknown_environment_mode" in instance.blocked_reason
