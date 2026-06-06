from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from A_model_parameters.mass_properties_estimate import MASS_KG as ESTIMATED_MASS_KG
from flight_dynamics import AircraftModel

CONTROL_EFFECTIVENESS_PERTURBATION_POLICY = "global_plus_axis_scheduled_surface_authority_multiplier_v4"
GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE = (0.75, 1.25)
AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE = (0.85, 1.15)
ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE = (0.85, 1.15)
RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE = (0.85, 1.15)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Plant instance dataclass
# 2) Instance construction
# 3) Aircraft-model adjustment
# =============================================================================


# =============================================================================
# 1) Plant Instance Dataclass
# =============================================================================
@dataclass(frozen=True)
class PlantInstance:
    plant_instance_id: str
    W_layer: str
    plant_randomisation_seed: int
    mass_scale: float
    mass_kg_effective: float
    cg_offset_m: tuple[float, float, float]
    inertia_scale: float
    Ixx_scale: float
    Iyy_scale: float
    Izz_scale: float
    cross_inertia_status: str
    aero_coefficient_scale: float
    surface_calibration_scale: float
    global_control_effectiveness_multiplier: float
    aileron_control_effectiveness_multiplier: float
    elevator_control_effectiveness_multiplier: float
    rudder_control_effectiveness_multiplier: float
    control_effectiveness_perturbation_policy: str
    plant_adjustment_status: str
    plant_adjustment_limitations: str
    claim_status: str = "simulation_only_plant_instance"


# =============================================================================
# 2) Instance Construction
# =============================================================================
def plant_instance_for_layer(
    W_layer: str,
    seed: int,
    *,
    baseline_mass_kg: float = float(ESTIMATED_MASS_KG),
) -> PlantInstance:
    """Return deterministic glider plant metadata for a W layer."""

    layer = str(W_layer).upper()
    rng = np.random.default_rng(int(seed))
    if layer in {"W0", "W1"}:
        return _plant_instance(
            layer=layer,
            seed=seed,
            baseline_mass_kg=baseline_mass_kg,
        )
    if layer == "W2":
        return _plant_instance(
            layer=layer,
            seed=seed,
            baseline_mass_kg=baseline_mass_kg,
            mass_scale=1.0,
            inertia_scale=1.0,
            plant_adjustment_status="deterministic_hardware_aware_nominal",
        )
    if layer == "W3":
        return _plant_instance(
            layer=layer,
            seed=seed,
            baseline_mass_kg=baseline_mass_kg,
            mass_scale=float(rng.uniform(0.95, 1.05)),
            cg_offset_m=tuple(float(value) for value in rng.uniform(-0.01, 0.01, size=3)),
            inertia_scale=float(rng.uniform(0.92, 1.08)),
            Ixx_scale=float(rng.uniform(0.92, 1.08)),
            Iyy_scale=float(rng.uniform(0.92, 1.08)),
            Izz_scale=float(rng.uniform(0.92, 1.08)),
            aero_coefficient_scale=float(rng.uniform(0.95, 1.05)),
            surface_calibration_scale=1.0,
            global_control_effectiveness_multiplier=float(
                rng.uniform(*GLOBAL_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE)
            ),
            aileron_control_effectiveness_multiplier=float(
                rng.uniform(*AILERON_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE)
            ),
            elevator_control_effectiveness_multiplier=float(
                rng.uniform(*ELEVATOR_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE)
            ),
            rudder_control_effectiveness_multiplier=float(
                rng.uniform(*RUDDER_CONTROL_EFFECTIVENESS_MULTIPLIER_RANGE)
            ),
            control_effectiveness_perturbation_policy=CONTROL_EFFECTIVENESS_PERTURBATION_POLICY,
            plant_adjustment_status="randomised_applied",
            plant_adjustment_limitations=(
                "mass inertia aero and cg offset applied; flap-scale surface calibration retained nominal; "
                "global and axis-specific multipliers applied directly to scheduled surface authority; "
                "cg offset shifts aerodynamic moment arms; cross inertia remains not perturbed"
            ),
        )
    return _plant_instance(
        layer=layer,
        seed=seed,
        baseline_mass_kg=baseline_mass_kg,
        plant_adjustment_status="blocked_unknown_W_layer",
        plant_adjustment_limitations=f"unknown W layer {layer}",
    )


def _plant_instance(
    *,
    layer: str,
    seed: int,
    baseline_mass_kg: float,
    mass_scale: float = 1.0,
    cg_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia_scale: float = 1.0,
    Ixx_scale: float = 1.0,
    Iyy_scale: float = 1.0,
    Izz_scale: float = 1.0,
    cross_inertia_status: str = "not_perturbed",
    aero_coefficient_scale: float = 1.0,
    surface_calibration_scale: float = 1.0,
    global_control_effectiveness_multiplier: float = 1.0,
    aileron_control_effectiveness_multiplier: float = 1.0,
    elevator_control_effectiveness_multiplier: float = 1.0,
    rudder_control_effectiveness_multiplier: float = 1.0,
    control_effectiveness_perturbation_policy: str = "nominal_no_control_mix_perturbation",
    plant_adjustment_status: str = "nominal_no_perturbation",
    plant_adjustment_limitations: str = "",
) -> PlantInstance:
    return PlantInstance(
        plant_instance_id=f"{layer}_plant_s{int(seed):06d}",
        W_layer=str(layer),
        plant_randomisation_seed=int(seed),
        mass_scale=float(mass_scale),
        mass_kg_effective=float(baseline_mass_kg) * float(mass_scale),
        cg_offset_m=tuple(float(value) for value in cg_offset_m),
        inertia_scale=float(inertia_scale),
        Ixx_scale=float(Ixx_scale),
        Iyy_scale=float(Iyy_scale),
        Izz_scale=float(Izz_scale),
        cross_inertia_status=str(cross_inertia_status),
        aero_coefficient_scale=float(aero_coefficient_scale),
        surface_calibration_scale=float(surface_calibration_scale),
        global_control_effectiveness_multiplier=float(global_control_effectiveness_multiplier),
        aileron_control_effectiveness_multiplier=float(aileron_control_effectiveness_multiplier),
        elevator_control_effectiveness_multiplier=float(elevator_control_effectiveness_multiplier),
        rudder_control_effectiveness_multiplier=float(rudder_control_effectiveness_multiplier),
        control_effectiveness_perturbation_policy=str(control_effectiveness_perturbation_policy),
        plant_adjustment_status=str(plant_adjustment_status),
        plant_adjustment_limitations=str(plant_adjustment_limitations),
    )


def plant_instance_row(instance: PlantInstance) -> dict[str, object]:
    """Return CSV-ready plant metadata."""

    row = asdict(instance)
    row["cg_offset_m"] = ";".join(f"{float(value):.9g}" for value in instance.cg_offset_m)
    return row


# =============================================================================
# 3) Aircraft-Model Adjustment
# =============================================================================
def apply_plant_instance_to_aircraft(
    aircraft: AircraftModel,
    instance: PlantInstance,
) -> AircraftModel:
    """Return an adjusted aircraft model with only honestly supported perturbations."""

    inertia = np.asarray(aircraft.inertia_b, dtype=float).copy()
    diag_scale = np.asarray(
        [
            float(instance.Ixx_scale),
            float(instance.Iyy_scale),
            float(instance.Izz_scale),
        ],
        dtype=float,
    ) * float(instance.inertia_scale)
    for index in range(3):
        inertia[index, index] *= diag_scale[index]
    inertia_inv = np.linalg.inv(inertia)
    aero_scale = float(instance.aero_coefficient_scale)
    surface_scale = float(instance.surface_calibration_scale)
    cg_offset_b = np.asarray(instance.cg_offset_m, dtype=float).reshape(3)
    axis_control_effectiveness = np.asarray(
        [
            float(instance.aileron_control_effectiveness_multiplier),
            float(instance.elevator_control_effectiveness_multiplier),
            float(instance.rudder_control_effectiveness_multiplier),
        ],
        dtype=float,
    )
    scheduled_surface_multiplier = (
        float(instance.global_control_effectiveness_multiplier) * axis_control_effectiveness
    )
    control_effectiveness_regime_scales = (
        np.asarray(aircraft.control_effectiveness_regime_scales, dtype=float)
        * scheduled_surface_multiplier.reshape(1, 3)
    )
    return replace(
        aircraft,
        mass_kg=float(aircraft.mass_kg) * float(instance.mass_scale),
        inertia_b=inertia,
        inertia_inv_b=inertia_inv,
        r_strip_b=np.asarray(aircraft.r_strip_b, dtype=float) - cg_offset_b.reshape(1, 3),
        cd0_strip=np.asarray(aircraft.cd0_strip, dtype=float) * aero_scale,
        flap_scale_strip=np.asarray(aircraft.flap_scale_strip, dtype=float) * surface_scale,
        control_effectiveness_regime_scales=control_effectiveness_regime_scales,
    )
