from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import aerosandbox as asb
import aerosandbox.numpy as np


SpanAxis = Literal["y", "z"]


@dataclass(frozen=True)
class PlateDims:
    """Equivalent rectangular plate dimensions."""
    span_m: np.ndarray  # full span (y or z direction)
    chord_m: np.ndarray  # equivalent chord (x direction)
    thickness_m: float


def _full_span_from_wing(lifting_surface: asb.Wing, span_axis: SpanAxis) -> np.ndarray:
    """
    Estimate full span from wing cross-section leading-edge coordinates.

    This avoids iterating over CasADi matrices.
    """
    coords = []
    for xsec in lifting_surface.xsecs:
        xyz = xsec.xyz_le
        coords.append(xyz[1] if span_axis == "y" else xyz[2])

    coords = np.array(coords)
    span_half = np.max(np.abs(coords))

    if lifting_surface.symmetric and span_axis == "y":
        return 2.0 * span_half

    # Non-symmetric or vertical tail case: span is max - min
    return np.max(coords) - np.min(coords)


def _equivalent_plate_dims(
    lifting_surface: asb.Wing,
    thickness_m: float,
    span_axis: SpanAxis,
) -> PlateDims:
    """
    Build an equivalent rectangular plate with the same planform area.
    chord_eq = area / span
    """
    span_m = _full_span_from_wing(lifting_surface, span_axis=span_axis)

    # AeroSandbox wing area is usually full area (already includes symmetry).
    area_m2 = lifting_surface.area()
    chord_m = area_m2 / span_m

    return PlateDims(span_m=span_m, chord_m=chord_m, thickness_m=thickness_m)


def _mid_chord_point_xyz(lifting_surface: asb.Wing, span_axis: SpanAxis) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate the mid-chord location using the root and tip mid-chord points,
    averaged.
    """
    xsecs = lifting_surface.xsecs
    if len(xsecs) < 2:
        # Degenerate case: single xsec
        xsec = xsecs[0]
        xyz_le = xsec.xyz_le
        x_cg = xyz_le[0] + 0.5 * xsec.chord
        y_cg = xyz_le[1]
        z_cg = xyz_le[2]
        return x_cg, y_cg, z_cg

    root = xsecs[0]
    tip = xsecs[-1]

    x_root = root.xyz_le[0] + 0.5 * root.chord
    x_tip = tip.xyz_le[0] + 0.5 * tip.chord
    x_cg = 0.5 * (x_root + x_tip)

    if span_axis == "y":
        # Symmetric planform => y_cg = 0 by symmetry
        y_cg = 0.0 if lifting_surface.symmetric else 0.5 * (root.xyz_le[1] + tip.xyz_le[1])
        z_cg = 0.5 * (root.xyz_le[2] + tip.xyz_le[2])
        return x_cg, y_cg, z_cg

    # span_axis == "z" (vertical tail): lies in x-z plane => y_cg ~ 0
    y_cg = 0.5 * (root.xyz_le[1] + tip.xyz_le[1])
    z_root = root.xyz_le[2]
    z_tip = tip.xyz_le[2]
    z_cg = 0.5 * (z_root + z_tip)
    return x_cg, y_cg, z_cg


def flat_plate_mass_properties(
    lifting_surface: asb.Wing,
    density_kg_m3: float,
    thickness_m: float,
    span_axis: SpanAxis,
) -> asb.MassProperties:
    """
    Compute mass properties for a flat-plate lifting surface using the thin
    rectangular plate inertia approximation.

    For span_axis="y" (wing/h-tail): span is along body-y, thickness along body-z.
    For span_axis="z" (v-tail): span is along body-z, thickness along body-y.

    Returns inertia about the component CG in body axes (x,y,z).
    """
    dims = _equivalent_plate_dims(
        lifting_surface=lifting_surface,
        thickness_m=thickness_m,
        span_axis=span_axis,
    )

    # Mass from volume (area * thickness) * density
    area_m2 = lifting_surface.area()
    mass = density_kg_m3 * area_m2 * thickness_m

    # CG location (mid-chord, mid-span)
    x_cg, y_cg, z_cg = _mid_chord_point_xyz(lifting_surface, span_axis=span_axis)

    b = dims.span_m
    c = dims.chord_m
    t = dims.thickness_m

    if span_axis == "y":
        i_xx = (mass / 12.0) * (b**2 + t**2)
        i_yy = (mass / 12.0) * (c**2 + t**2)
        i_zz = (mass / 12.0) * (b**2 + c**2)
    elif span_axis == "z":
        i_xx = (mass / 12.0) * (b**2 + t**2)
        i_zz = (mass / 12.0) * (c**2 + t**2)
        i_yy = (mass / 12.0) * (b**2 + c**2)
    else:
        raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis}")

    return asb.MassProperties(
        mass=mass,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg,
        Ixx=i_xx,
        Iyy=i_yy,
        Izz=i_zz,
        Ixy=0.0,
        Ixz=0.0,
        Iyz=0.0,
    )
