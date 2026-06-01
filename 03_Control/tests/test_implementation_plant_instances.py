from __future__ import annotations

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from implementation_instance import (
    _clip_surface_rad,
    apply_aileron_asymmetry_to_aircraft,
    adjusted_actuator_tau_s,
    apply_surface_implementation,
    implementation_instance_for_layer,
    implementation_instance_row,
)
from plant_instance import (
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
    assert "cg offset shifts aerodynamic moment arms" in first.plant_adjustment_limitations
    assert plant_instance_row(first)["plant_adjustment_status"] == "randomised_applied"
    assert not np.isclose(adjusted.mass_kg, aircraft.mass_kg)
    assert not np.allclose(adjusted.inertia_b, aircraft.inertia_b)
    assert np.allclose(
        adjusted.r_strip_b,
        aircraft.r_strip_b - np.asarray(first.cg_offset_m, dtype=float).reshape(1, 3),
    )
    assert not np.allclose(adjusted.flap_scale_strip, aircraft.flap_scale_strip)
