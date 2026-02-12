# run_opt.py
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Tuple

import aerosandbox as asb

from config import Config
from debug_nlp import constraint_count, ipopt_smoke_test, variable_count
from model.aero import build_aero
from model.airfoils import build_airfoils
from model.geometry import build_airplane
from model.mass_model import build_mass_properties
from model.objective import set_objective_and_constraints
from model.operating_point import build_operating_point
from model.roll_in import add_roll_in_constraints
from model.thermal import build_thermal
from post.export import save_results
from post.plots import make_all_plots
from post.postprocess import extract_solution
from utils_git import get_git_version


def _ensure_output_dirs(cfg: Config) -> None:
    """Create output folders if they do not exist."""
    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)


def _mark(opti: asb.Opti, stage: str) -> None:
    """Print the current NLP size (best-effort)."""
    print(
        f"[mark] {stage}: n_x={variable_count(opti)}, n_g={constraint_count(opti)}",
        flush=True,
    )


def _smoke_or_exit(
    opti: asb.Opti,
    *,
    stage: str,
    max_iter: int = 0,
    max_cpu_time: float = 1e-4,
) -> None:
    """Run an IPOPT smoke test. Abort immediately if it fails."""
    result = ipopt_smoke_test(
        opti,
        stage=stage,
        max_iter=max_iter,
        max_cpu_time=max_cpu_time,
        raise_on_fail=False,
        silent=True,
        hard_silence=True,
    )

    if not result.ok:
        raise SystemExit(
            f"[ABORT] Smoke test failed at stage '{stage}'.\n"
            f"Hint: {result.hint}"
            f"Message:\n{result.message}"
        )


def _read_ipopt_options_from_cfg(cfg: Config) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Read solver options from Config if expose them."""
    plugin_opts: dict[str, Any] = {"print_time": False, "verbose": False}
    ipopt_opts: dict[str, Any] = {}

    solver = getattr(cfg, "solver", None)
    if solver is None:
        return plugin_opts, ipopt_opts

    user_plugin = getattr(solver, "plugin_opts", None)
    user_ipopt = getattr(solver, "ipopt_opts", None)

    if isinstance(user_plugin, Mapping):
        plugin_opts.update(dict(user_plugin))
    if isinstance(user_ipopt, Mapping):
        ipopt_opts.update(dict(user_ipopt))

    return plugin_opts, ipopt_opts


def _reset_ipopt_for_full_solve(opti: asb.Opti, cfg: Config) -> None:
    """Reset IPOPT after smoke tests."""
    plugin_opts, cfg_ipopt_opts = _read_ipopt_options_from_cfg(cfg)

    # Defaults that are sensible if cfg doesn't provide anything.
    # (Last-write-wins: cfg overrides defaults.)
    ipopt_opts: dict[str, Any] = {
        "max_iter": 2000,
        "check_derivatives_for_naninf": "yes",
        **cfg_ipopt_opts,
    }

    opti.solver("ipopt", plugin_opts, ipopt_opts)


def main() -> None:
    cfg = Config()
    print("CODE_VERSION:", get_git_version(), flush=True)

    _ensure_output_dirs(cfg)

    opti = asb.Opti()

    print("[1/8] Building operating point...", flush=True)
    mission: MutableMapping[str, Any] = dict(build_operating_point(opti, cfg))
    # _mark(opti, "after operating_point")
    # _smoke_or_exit(opti, stage="after operating_point")

    print("[2/8] Building airfoils...", flush=True)
    airfoils = build_airfoils(cfg)
    # _mark(opti, "after airfoils")
    # _smoke_or_exit(opti, stage="after airfoils")

    print("[3/8] Building geometry...", flush=True)
    geom: MutableMapping[str, Any] = dict(
        build_airplane(opti, cfg, airfoils, mission["controls"])
    )
    # _mark(opti, "after geometry")
    # _smoke_or_exit(opti, stage="after geometry")

    print("[4/8] Building mass model...", flush=True)
    mass: MutableMapping[str, Any] = dict(build_mass_properties(opti, cfg, geom))
    # _mark(opti, "after mass")
    # _smoke_or_exit(opti, stage="after mass")

    print("[5/8] Building thermal model...", flush=True)
    thermal = build_thermal(cfg, mission["r_target"])
    # _mark(opti, "after thermal")
    # _smoke_or_exit(opti, stage="after thermal")

    print("[6/8] Building aerodynamics...", flush=True)
    aero_pack = build_aero(
        cfg=cfg,
        airplane=geom["airplane"],
        op_point=mission["op_point"],
        xyz_ref=mass["mass_props_togw"].xyz_cg,
        geom=geom,
        controls=mission["controls"],
    )
    # _mark(opti, "after aero")
    # _smoke_or_exit(opti, stage="after aero")

    print("[7/8] Building roll-in model...", flush=True)
    roll = add_roll_in_constraints(
        opti=opti,
        cfg=cfg,
        mission=mission,
        geom=geom,
        mass=mass,
        aero_pack=aero_pack,
        thermal=thermal,
    )
    # _mark(opti, "after roll-in")
    # _smoke_or_exit(opti, stage="after roll-in")

    print("[8/8] Setting objective and constraints...", flush=True)
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
    # _mark(opti, "after objective/constraints")
    # c_smoke_or_exit(opti, stage="after objective/constraints")

    # if all smoke tests passed.
    print("Initialisation complete. Resetting IPOPT for full solve...", flush=True)
    _reset_ipopt_for_full_solve(opti, cfg)

    print("Starting solver...", flush=True)
    try:
        sol = opti.solve()
    except RuntimeError as exc:
        print("\n[SOLVE FAILED]", exc, flush=True)
        print("Skipping postprocess/export/plots because no valid solution exists.", flush=True)
        return

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

    make_all_plots(cfg, solved)
    outputs = save_results(cfg, solved)

    print("\nSaved results to:", flush=True)
    for key, path in outputs.items():
        print(f"  {key}: {path}", flush=True)


if __name__ == "__main__":
    main()