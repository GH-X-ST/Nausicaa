from __future__ import annotations

from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp

from config import Config


def build_airfoils(cfg: Config) -> dict[str, asb.Airfoil]:
    """
    Build a dict of airfoils. Includes:
    - Named airfoils (optionally with cached polars)
    - An analytic 'flat_plate' airfoil with a simple geometry definition for plotting.
    """
    airfoil_names = ["ag04", "naca0002", "naca0008", "s1223", "s3021"]
    airfoils: dict[str, asb.Airfoil] = {name: asb.Airfoil(name=name) for name in airfoil_names}

    # flat plate model
    eps = 1e-4  # nondimensional half-thickness for visualization only
    coords = onp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, eps],
            [0.0, eps],
            [0.0, 0.0],
        ]
    )

    flat_plate = asb.Airfoil(name="flat_plate", coordinates=coords)

    def cl(alpha_deg, Re=None, mach=None):
        a = np.radians(alpha_deg)
        return 2.0 * np.sin(a) * np.cos(a)

    def cd(alpha_deg, Re=None, mach=None):
        a = np.radians(alpha_deg)
        return 2.0 * (np.sin(a) ** 2)

    def cm(alpha_deg, Re=None, mach=None):
        # common simplification for a symmetric plate about an appropriate reference.
        return 0.0

    flat_plate.CL_function = cl
    flat_plate.CD_function = cd
    flat_plate.CM_function = cm

    airfoils["flat_plate"] = flat_plate

    # generate polars for *named* airfoils only
    if cfg.airfoils.generate_polars:
        cache_dir: Path = cfg.paths.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        alpha_min = cfg.bounds.alpha_min_deg
        alpha_max = cfg.bounds.alpha_max_deg
        n_alpha = cfg.airfoils.n_alpha
        alphas = np.linspace(alpha_min, alpha_max, n_alpha)

        for af in airfoils.values():
            #  skip polar generation for analytic flat plate
            if af.name == "flat_plate":
                continue

            af.generate_polars(
                cache_filename=str(cache_dir / f"{af.name}.json"),
                alphas=alphas,
            )

    return airfoils