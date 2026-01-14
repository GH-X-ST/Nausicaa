from __future__ import annotations

from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np

from config import Config


def build_airfoils(cfg: Config) -> dict[str, asb.Airfoil]:
    """Build airfoils and generate polars using XFoil only when needed."""
    cache_dir: Path = cfg.paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    airfoil_names = ["ag04", "naca0008", "s1223", "s3021"]
    airfoils = {name: asb.Airfoil(name=name) for name in airfoil_names}

    alphas = np.linspace(-10.0, 10.0, 21)

    for af in airfoils.values():
        cache_file = cache_dir / f"{af.name}.json"
        if not cache_file.exists():
            af.generate_polars(cache_filename=str(cache_file), alphas=alphas)

    return airfoils