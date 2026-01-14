from __future__ import annotations

import aerosandbox as asb

from config import Config
from model.mass_simple import flat_plate_mass_properties

def build_mass_properties(opti: asb.Opti, cfg: Config, geom: dict) -> dict:
    """Assemble aircraft mass properties (structure strips + discrete components)."""
    cst = cfg.constants

    wing = geom["wing"]
    htail = geom["htail"]
    vtail = geom["vtail"]
    x_nose = geom["x_nose"]
    x_tail = geom["x_tail"]

    mass_props: dict[str, asb.MassProperties] = {}

    mass_props["wing"] = flat_plate_mass_properties(
        lifting_surface=geom["wing"],
        density_kg_m3=cst.density_wing,
        thickness_m=cst.wing_thickness,
        span_axis="y",
    )
    
    mass_props["htail_surfaces"] = flat_plate_mass_properties(
        lifting_surface=geom["htail"],
        density_kg_m3=cst.density_wing,
        thickness_m=cst.tail_thickness,
        span_axis="y",
    )
    
    mass_props["vtail_surfaces"] = flat_plate_mass_properties(
        lifting_surface=geom["vtail"],
        density_kg_m3=cst.density_wing,
        thickness_m=cst.tail_thickness,
        span_axis="z",
    )

    # Avionics / fuselage items
    mass_props["linkages"] = asb.MassProperties(mass=0.001, x_cg=x_tail / 2)

    mass_props["Receiver"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.005, x_cg=x_nose + 0.010
    )
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.009 + 0.004, x_cg=x_nose + 0.05
    )
    mass_props["servo"] = asb.mass_properties_from_radius_of_gyration(
        mass=4 * 0.0022, x_cg=x_nose + 0.015
    )
    mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.009 * (x_tail - x_nose), x_cg=(x_nose + x_tail) / 2
    )
    mass_props["pod"] = asb.MassProperties(mass=0.007, x_cg=x_nose + 0.010)

    # Ballast
    mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
        mass=opti.variable(init_guess=0.0, lower_bound=0.0),
        x_cg=opti.variable(init_guess=0.0, lower_bound=x_nose, upper_bound=x_tail),
    )

    # Summation
    mass_props_togw = asb.MassProperties(mass=0.0)
    for mp in mass_props.values():
        mass_props_togw = mass_props_togw + mp

    # Glue weight (8% of total)
    mass_props["glue_weight"] = mass_props_togw * 0.08
    mass_props_togw += mass_props["glue_weight"]

    return {
        "mass_props": mass_props,
        "mass_props_togw": mass_props_togw,
    }