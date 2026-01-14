from __future__ import annotations

import aerosandbox as asb

from config import Config
from utils_git import get_git_version

from model.operating_point import build_operating_point
from model.airfoils import build_airfoils
from model.geometry import build_airplane
from model.mass_model import build_mass_properties
from model.thermal import build_thermal
from model.aero import build_aero
from model.roll_in import add_roll_in_constraints
from model.objective import set_objective_and_constraints

from post.postprocess import extract_solution
from post.export import save_results
from post.plots import make_all_plots


def main() -> None:
    cfg = Config()

    print("CODE_VERSION:", get_git_version())

    # Ensure dirs exist
    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)

    opti = asb.Opti()

    mission = build_operating_point(opti, cfg)
    airfoils = build_airfoils(cfg)
    geom = build_airplane(opti, cfg, airfoils, mission["controls"])
    mass = build_mass_properties(opti, cfg, geom)

    thermal = build_thermal(cfg, mission["r_target"])
    aero_pack = build_aero(
        cfg=cfg,
        airplane=geom["airplane"],
        op_point=mission["op_point"],
        xyz_ref=mass["mass_props_togw"].xyz_cg,
        geom=geom,
        controls=mission["controls"],
    )

    roll = add_roll_in_constraints(
        opti=opti,
        cfg=cfg,
        mission=mission,
        geom=geom,
        mass=mass,
        aero_pack=aero_pack,
        thermal=thermal,
    )

    objective_pack = set_objective_and_constraints(
        opti=opti,
        cfg=cfg,
        mission=mission,
        geom=geom,
        mass=mass,
        thermal=thermal,
        aero_pack=aero_pack,
        roll=roll,
    )

    # Solve
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug

    solved = extract_solution(
        sol=sol,
        mission=mission,
        geom=geom,
        mass=mass,
        thermal=thermal,
        aero_pack=aero_pack,
        roll=roll,
        objective_pack=objective_pack,
    )

    # Plots + export
    make_all_plots(cfg, solved)
    outputs = save_results(cfg, solved)

    print("\nSaved results to:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()