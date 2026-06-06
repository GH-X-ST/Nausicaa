from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from implementation_instance import (
    IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE,
    _clip_surface_rad,
    apply_aileron_asymmetry_to_aircraft,
    adjusted_actuator_tau_s,
    apply_surface_implementation,
    implementation_instance_for_layer,
    implementation_instance_row,
)
from plant_instance import (
    AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE,
    apply_plant_instance_to_aircraft,
    plant_instance_for_layer,
    plant_instance_row,
)


def test_implementation_instance_is_deterministic_and_changes_surface_and_tau() -> None:
    first = implementation_instance_for_layer("W3", 31, latency_case="nominal")
    second = implementation_instance_for_layer("W3", 31, latency_case="nominal")
    nominal = implementation_instance_for_layer("W1", 31, latency_case="nominal")
    command = np.asarray([0.10, -0.08, 0.05], dtype=float)

    assert first == second
    assert implementation_instance_row(first)["implementation_adjustment_status"] == "randomised_applied"
    assert first.aileron_effectiveness_scale == IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
    assert first.elevator_effectiveness_scale == IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
    assert first.rudder_effectiveness_scale == IMPLEMENTATION_SURFACE_EFFECTIVENESS_SCALE
    assert "surface effectiveness scaling retired" in first.implementation_adjustment_limitations
    assert adjusted_actuator_tau_s((0.1, 0.2, 0.3), first) != (0.1, 0.2, 0.3)
    assert not np.allclose(
        apply_surface_implementation(command, first),
        apply_surface_implementation(command, nominal),
    )
    assert not np.allclose(
        apply_surface_implementation(np.zeros(3), first),
        np.zeros(3),
    )
    asymmetric = apply_aileron_asymmetry_to_aircraft(adapt_glider(build_nausicaa_glider()), first)
    nominal_aircraft = apply_aileron_asymmetry_to_aircraft(adapt_glider(build_nausicaa_glider()), nominal)
    assert not np.allclose(asymmetric.control_mix, nominal_aircraft.control_mix)


def test_surface_implementation_never_exceeds_full_authority_clip() -> None:
    oversized = np.asarray([10.0, -10.0, 10.0], dtype=float)
    clipped = _clip_surface_rad(oversized, surface_limit_scale=1.5)
    full_scale = _clip_surface_rad(oversized, surface_limit_scale=1.0)

    assert np.allclose(clipped, full_scale)


def test_plant_instance_is_deterministic_and_changes_aircraft_model() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    first = plant_instance_for_layer("W3", 41, baseline_mass_kg=aircraft.mass_kg)
    second = plant_instance_for_layer("W3", 41, baseline_mass_kg=aircraft.mass_kg)
    adjusted = apply_plant_instance_to_aircraft(aircraft, first)

    assert first == second
    assert "global and axis-specific multipliers applied directly to scheduled surface authority" in first.plant_adjustment_limitations
    assert "cg offset shifts aerodynamic moment arms" in first.plant_adjustment_limitations
    assert plant_instance_row(first)["plant_adjustment_status"] == "randomised_applied"
    assert not np.isclose(adjusted.mass_kg, aircraft.mass_kg)
    assert not np.allclose(adjusted.inertia_b, aircraft.inertia_b)
    assert np.allclose(
        adjusted.r_strip_b,
        aircraft.r_strip_b - np.asarray(first.cg_offset_m, dtype=float).reshape(1, 3),
    )
    assert np.allclose(adjusted.flap_scale_strip, aircraft.flap_scale_strip)
    assert np.allclose(adjusted.control_mix, aircraft.control_mix)
    assert not np.allclose(
        adjusted.control_effectiveness_regime_scales,
        aircraft.control_effectiveness_regime_scales,
    )


def test_w3_plant_surface_effectiveness_uses_global_and_axis_ranges() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    instance = plant_instance_for_layer("W3", 77, baseline_mass_kg=aircraft.mass_kg)
    adjusted = apply_plant_instance_to_aircraft(aircraft, instance)
    nominal = plant_instance_for_layer("W2", 77, baseline_mass_kg=aircraft.mass_kg)

    assert nominal.aileron_control_effectiveness_multiplier == 1.0
    assert nominal.elevator_control_effectiveness_multiplier == 1.0
    assert nominal.rudder_control_effectiveness_multiplier == 1.0
    assert nominal.global_control_effectiveness_multiplier == 1.0
    assert instance.control_effectiveness_perturbation_policy == "global_plus_axis_scheduled_surface_authority_multiplier_v4"
    assert (
        GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
        <= instance.global_control_effectiveness_multiplier
        <= GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
    )
    assert (
        AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
        <= instance.aileron_control_effectiveness_multiplier
        <= AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
    )
    assert (
        ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
        <= instance.elevator_control_effectiveness_multiplier
        <= ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
    )
    assert (
        RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[0]
        <= instance.rudder_control_effectiveness_multiplier
        <= RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE[1]
    )

    base_schedule = np.asarray(aircraft.control_effectiveness_regime_scales, dtype=float)
    adjusted_schedule = np.asarray(adjusted.control_effectiveness_regime_scales, dtype=float)
    expected_multiplier = np.asarray(
        [
            instance.aileron_control_effectiveness_multiplier,
            instance.elevator_control_effectiveness_multiplier,
            instance.rudder_control_effectiveness_multiplier,
        ],
        dtype=float,
    ) * instance.global_control_effectiveness_multiplier
    assert np.allclose(adjusted_schedule, base_schedule * expected_multiplier.reshape(1, 3))
