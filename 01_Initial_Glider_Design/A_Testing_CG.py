import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy
import subprocess
import os

### Code version
def get_git_version():

    try:
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            stderr = subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return desc
    except Exception:
        return "unknown"


CODE_VERSION = get_git_version()


###### AeroSandbox Setup

opti = asb.Opti()
# variable_categories_to_freeze = "all",
# freeze_style = "float"

make_plots=True

###### Lifting Surface

### Material
density_wing = 33.0 # kg/m^3 for depron foam

### Airfoil
airfoils = {
    name: asb.Airfoil(name=name,) for name in ["ag04", "naca0008"]
}

for v in airfoils.values():
   v.generate_polars(
       cache_filename = f"cache/{v.name}.json", alphas = np.linspace(-10, 10, 21)
    ) # generating aerodynamic polars using XFoil

###### Lifting Surface

### Material
density_wing = 33.0 # kg/m^3 for depron foam

### Wing
wing_span = 0.5
wing_dihedral_angle_deg = 11
wing_root_chord = 0.1
wing_taper = 0.7

def wing_rot(xyz):

    dihedral_rot = np.rotation_matrix_3D(angle = np.radians(wing_dihedral_angle_deg), axis = "X")

    return dihedral_rot @ np.array(xyz)

def wing_chord(y):

    half_span = wing_span / 2
    tip_chord = wing_taper * wing_root_chord
    spanfrac = np.abs(y) / half_span # 0 at root, 1 at tip

    return (1 - spanfrac) * wing_root_chord + spanfrac * tip_chord

def wing_twist(y):

    return np.zeros_like(y) # no twist

wing_ys = np.sinspace(0, wing_span / 2, 11, reverse_spacing = True) # y station

wing = asb.Wing(name = "Main Wing", symmetric = True,
    xsecs = [asb.WingXSec(xyz_le = wing_rot([-wing_chord(wing_ys[i]), wing_ys[i], 0.0]),
                          chord = wing_chord(wing_ys[i]),
                          airfoil = airfoils["ag04"],
                          twist = wing_twist(wing_ys[i]),
                          )
                          for i in range(np.length(wing_ys))
             ]      
).translate([0.75 * wing_root_chord, 0, 0])


##### Internal Geometry and Weights
mass_props = {}

### Lifting surface centre of gravity
def lifting_surface_planform_cg(wing: asb.Wing, span_axis: str = "y"):

    # extract leading-edge positions and chords from xsecs
    xyz_le = np.stack([xsec.xyz_le for xsec in wing.xsecs], axis=0) # (N, 3)
    chords = np.array([xsec.chord for xsec in wing.xsecs]) # (N,)

    x_le = xyz_le[:, 0]
    y_le = xyz_le[:, 1]
    z_le = xyz_le[:, 2]

    if span_axis == "y":
        span = y_le
    elif span_axis == "z":
        span = z_le
    else:
        raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis}")

    # spanwise strips between stations
    dspan = span[1:] - span[:-1] # strip width
    c_mid = 0.5 * (chords[:-1] + chords[1:]) # average chord

    # surface area
    A_strip_half = c_mid * dspan

    if wing.symmetric and span_axis == "y":
        A_strip = 2.0 * A_strip_half
    else:
        A_strip = A_strip_half

    # centroid x,z of each strip
    x_mid_i = x_le[:-1] + 0.5 * chords[:-1]
    x_mid_ip1 = x_le[1:] + 0.5 * chords[1:]
    x_mid_strip = 0.5 * (x_mid_i + x_mid_ip1)

    z_mid_strip = 0.5 * (z_le[:-1] + z_le[1:])

    A_total = np.sum(A_strip)

    x_cg = np.sum(A_strip * x_mid_strip) / A_total
    z_cg = np.sum(A_strip * z_mid_strip) / A_total

    return x_cg, z_cg

### Wing
x_cg_wing, z_cg_wing = lifting_surface_planform_cg(wing, span_axis="y")

mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
    mass = wing.volume() * density_wing,
    x_cg = x_cg_wing,
    z_cg = z_cg_wing,
    )

x_cg_feather = (0.50 - 0.25) * wing_root_chord
z_cg_feather = (0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1)

### Results
print("Wing area   [m^2]:         ", float(wing.area()))
print("Wing volume [m^3]:         ", float(wing.volume()))
print("Planform CG x [m]:         ", float(x_cg_wing))
print("Planform CG z [m]:         ", float(z_cg_wing))
print("Planform analytic CG x [m]:", float(x_cg_feather))
print("Planform analytic CG z [m]:", float(z_cg_feather))
print("Mass [kg]:                 ", float(mass_props['wing'].mass))