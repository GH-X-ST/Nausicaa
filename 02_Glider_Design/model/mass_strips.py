from __future__ import annotations

import aerosandbox as asb
import aerosandbox.numpy as np


def lifting_surface_massprops_from_strips(
    lifting_surface: asb.Wing,
    density: float,
    thickness: float | None = None,
    span_axis: str = "y",
) -> asb.MassProperties:
    """
    Approximate lifting-surface mass properties using spanwise strips.

    Uses xsecs geometry and distributes point masses at strip centroids.
    Returns mass/inertia about the strip-approximated CG, then scaled to geometric volume*density.
    """
    if thickness is None:
        a_planform = lifting_surface.area()
        v_geom = lifting_surface.volume()
        thickness = v_geom / a_planform

    xyz_le = np.stack([xsec.xyz_le for xsec in lifting_surface.xsecs], axis=0)
    chords = np.array([xsec.chord for xsec in lifting_surface.xsecs])

    x_le = xyz_le[:, 0]
    y_le = xyz_le[:, 1]
    z_le = xyz_le[:, 2]

    if span_axis == "y":
        span_coord = y_le
    elif span_axis == "z":
        span_coord = z_le
    else:
        raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis}")

    dspan = span_coord[1:] - span_coord[:-1]
    c_mid = 0.5 * (chords[:-1] + chords[1:])
    a_strip_half = c_mid * dspan

    x_mid_i = x_le[:-1] + 0.5 * chords[:-1]
    x_mid_ip1 = x_le[1:] + 0.5 * chords[1:]
    x_mid = 0.5 * (x_mid_i + x_mid_ip1)
    y_mid = 0.5 * (y_le[:-1] + y_le[1:])
    z_mid = 0.5 * (z_le[:-1] + z_le[1:])

    m_half = density * thickness * a_strip_half

    if lifting_surface.symmetric and span_axis == "y":
        m_pts = np.concatenate([m_half, m_half], axis=0)
        x_pts = np.concatenate([x_mid, x_mid], axis=0)
        y_pts = np.concatenate([y_mid, -y_mid], axis=0)
        z_pts = np.concatenate([z_mid, z_mid], axis=0)
    else:
        m_pts, x_pts, y_pts, z_pts = m_half, x_mid, y_mid, z_mid

    m_raw = np.sum(m_pts)
    m_target = lifting_surface.volume() * density

    x_cg = np.sum(m_pts * x_pts) / m_raw
    y_cg = np.sum(m_pts * y_pts) / m_raw
    z_cg = np.sum(m_pts * z_pts) / m_raw

    x_rel = x_pts - x_cg
    y_rel = y_pts - y_cg
    z_rel = z_pts - z_cg

    i_xx_raw = np.sum(m_pts * (y_rel**2 + z_rel**2))
    i_yy_raw = np.sum(m_pts * (x_rel**2 + z_rel**2))
    i_zz_raw = np.sum(m_pts * (x_rel**2 + y_rel**2))

    i_xy_raw = -np.sum(m_pts * x_rel * y_rel)
    i_xz_raw = -np.sum(m_pts * x_rel * z_rel)
    i_yz_raw = -np.sum(m_pts * y_rel * z_rel)

    scale = m_target / m_raw

    return asb.MassProperties(
        mass=m_target,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg,
        Ixx=i_xx_raw * scale,
        Iyy=i_yy_raw * scale,
        Izz=i_zz_raw * scale,
        Ixy=i_xy_raw * scale,
        Ixz=i_xz_raw * scale,
        Iyz=i_yz_raw * scale,
    )