from __future__ import annotations

import aerosandbox.numpy as np

from config import Config


def vertical_velocity_field(
    q_v: float,
    r_th0: float,
    k_th: float,
    r_orbit,
    z_th_local: float,
    z0: float,
    fan_spacing: float,
) -> object:
    """
    2×2 fan thermal model (axisymmetric Gaussian approximation).
    Returns average w along orbit sampled at 4 azimuth points.

    All symbolic-friendly: r_orbit may be an Opti variable expression.
    """
    r_th = r_th0 + k_th * (z_th_local - z0)
    r_th = np.maximum(r_th, 1e-6)

    w_th = q_v / (np.pi * r_th**2)

    fan_centres = [
        (-fan_spacing / 2, -fan_spacing / 2),
        (fan_spacing / 2, -fan_spacing / 2),
        (-fan_spacing / 2, fan_spacing / 2),
        (fan_spacing / 2, fan_spacing / 2),
    ]

    def w_at_xy(x, y):
        w_total = 0.0
        for x_c, y_c in fan_centres:
            r_i = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            w_i = w_th * np.exp(-(r_i / r_th) ** 2)
            w_i = np.where(z_th_local < z0, 0.0, w_i)
            w_total = w_total + w_i
        return w_total

    thetas = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])

    w_sum = 0.0
    for th in thetas:
        x = r_orbit * np.cos(th)
        y = r_orbit * np.sin(th)
        w_sum = w_sum + w_at_xy(x, y)

    return w_sum / len(thetas)


def build_thermal(cfg: Config, r_orbit) -> dict:
    """Build thermal model and return average updraft at orbit radius."""
    th = cfg.thermal
    fan_spacing = 2 * th.r_th0 + 1.7

    w_avg = vertical_velocity_field(
        q_v=th.q_v,
        r_th0=th.r_th0,
        k_th=th.k_th,
        r_orbit=r_orbit,
        z_th_local=cfg.mission.z_th,
        z0=th.z0,
        fan_spacing=fan_spacing,
    )

    return {
        "w": w_avg,
        "fan_spacing": fan_spacing,
        "x_center": th.x_center,
        "y_center": th.y_center,
        "q_v": th.q_v,
        "r_th0": th.r_th0,
        "k_th": th.k_th,
        "z0": th.z0,
    }
