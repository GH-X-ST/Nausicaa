from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(".")
    cache_dir: Path = Path("A_cache")
    figures_dir: Path = Path("B_figures")
    results_dir: Path = Path("C_results")


@dataclass(frozen=True)
class AirfoilSettings:
    generate_polars: bool = False
    n_alpha: int = 21


@dataclass(frozen=True)
class FlatPlatePolar:
    cd0: float = 0.02
    k: float = 0.08
    cl_max: float = 1.2


@dataclass(frozen=True)
class Bounds:
    # Operating point
    v_min: float = 0.1
    v_max: float = 10.0
    alpha_min_deg: float = -10.0
    alpha_max_deg: float = 10.0

    # Control surfaces
    delta_a_min_deg: float = -25.0
    delta_a_max_deg: float = 25.0
    delta_r_min_deg: float = -30.0
    delta_r_max_deg: float = 30.0
    delta_e_min_deg: float = -25.0
    delta_e_max_deg: float = 25.0

    # Geometry
    x_nose_max: float = 1e-3
    b_w_min: float = 0.1
    b_w_max: float = 10.0
    gamma_w_min_deg: float = 0.0
    gamma_w_max_deg: float = 20.0
    c_root_w_min: float = 1e-3

    l_ht_min: float = 0.2
    l_ht_max: float = 1.5
    b_ht_min: float = 1e-3
    b_vt_min: float = 1e-3

    # Mass
    togw_min: float = 1e-3

    # Roll-in
    psi0_min_deg: float = -90.0
    psi0_max_deg: float = 90.0
    t_roll_min: float = 0.05
    t_roll_max: float = 20.0
    p_roll_max: float = 1.5  # rad/s
    delta_a_rate_max_deg_s: float = 250.0


@dataclass(frozen=True)
class Constants:
    g: float = 9.81
    rho: float = 1.225
    density_wing: float = 33.0  # Depron foam approx
    wing_thickness: float = 0.006  # 6 mm
    tail_thickness: float = 0.003  # 3 mm


@dataclass(frozen=True)
class Mission:
    bank_angle_deg: float = 35.0
    z_th: float = 1.00


@dataclass(frozen=True)
class Arena:
    x_min: float = 0.0
    x_max: float = 8.0
    y_min: float = 0.0
    y_max: float = 5.0


@dataclass(frozen=True)
class ThermalParams:
    q_v: float = 1.69
    x_center: float = 4.0
    y_center: float = 2.5

    r_th0: float = 0.381
    k_th: float = 0.10
    z0: float = 0.50


@dataclass(frozen=True)
class PlotSettings:
    make_plots: bool = True
    dpi: int = 300


@dataclass(frozen=True)
class RollInSettings:
    n_roll: int = 21


@dataclass(frozen=True)
class Config:
    paths: Paths = Paths()
    airfoils: AirfoilSettings = AirfoilSettings()
    flat_plate: FlatPlatePolar = FlatPlatePolar()
    bounds: Bounds = Bounds()
    constants: Constants = Constants()
    mission: Mission = Mission()
    arena: Arena = Arena()
    thermal: ThermalParams = ThermalParams()
    plot: PlotSettings = PlotSettings()
    roll_in: RollInSettings = RollInSettings()

    # Wing/tails “fixed design choices”
    lambda_w: float = 1.0
    ar_ht: float = 4.0
    lambda_ht: float = 1.0
    ar_vt: float = 2.0
    lambda_vt: float = 1.0

    # Objective tuning
    k_soft: float = 50.0
    w_control_penalty: float = 1e-5