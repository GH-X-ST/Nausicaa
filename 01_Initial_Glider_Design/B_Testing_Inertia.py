import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy

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

### Lifting surface centre of gravity and inertia tensor
def lifting_surface_massprops_from_strips(wing: asb.Wing, density: float, thickness: float = None, span_axis: str = "y",):

    # find thickness
    if thickness is None:
        A_planform = wing.area()
        V_geom     = wing.volume()
        thickness  = V_geom / A_planform

    # extract leading-edge positions and chords from xsecs
    xyz_le = np.stack([xsec.xyz_le for xsec in wing.xsecs], axis = 0)  # (N, 3)
    chords = np.array([xsec.chord for xsec in wing.xsecs])             # (N,)

    x_le = xyz_le[:, 0]
    y_le = xyz_le[:, 1]
    z_le = xyz_le[:, 2]

    if span_axis == "y":
        span_coord = y_le
    elif span_axis == "z":
        span_coord = z_le
    else:
        raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis}")

    # spanwise strips between stations
    dspan = span_coord[1:] - span_coord[:-1]    # (N-1,)
    c_mid = 0.5 * (chords[:-1] + chords[1:])    # (N-1,)

    # half-surface area
    A_strip_half = c_mid * dspan                # (N-1,)

    # centroid of each half-strip in x, y, z
    x_mid_i     = x_le[:-1] + 0.5 * chords[:-1]
    x_mid_ip1   = x_le[1:]  + 0.5 * chords[1:]
    x_mid_strip = 0.5 * (x_mid_i + x_mid_ip1)   # (N-1,)

    y_mid_strip = 0.5 * (y_le[:-1] + y_le[1:])  # (N-1,)
    z_mid_strip = 0.5 * (z_le[:-1] + z_le[1:])  # (N-1,)

    # mass per half-strip
    m_half = density * thickness * A_strip_half # (N-1,)

    # build discrete point masses for full surface
    if wing.symmetric and span_axis == "y":

        # mirror about y = 0
        m_points = np.concatenate([m_half,      m_half      ], axis = 0)
        x_points = np.concatenate([x_mid_strip, x_mid_strip ], axis = 0)
        y_points = np.concatenate([y_mid_strip, -y_mid_strip], axis = 0)
        z_points = np.concatenate([z_mid_strip, z_mid_strip ], axis = 0)

    else:
        m_points = m_half
        x_points = x_mid_strip
        y_points = y_mid_strip
        z_points = z_mid_strip

    # total mass from strips
    M_raw    = np.sum(m_points)

    # Target physical mass from geometry
    M_target = wing.volume() * density
    
    # inertia tensor about CG
    x_cg = np.sum(m_points * x_points) / M_raw
    y_cg = np.sum(m_points * y_points) / M_raw
    z_cg = np.sum(m_points * z_points) / M_raw

    x_rel = x_points - x_cg
    y_rel = y_points - y_cg
    z_rel = z_points - z_cg

    I_xx_raw = np.sum(m_points * (y_rel**2 + z_rel**2))
    I_yy_raw = np.sum(m_points * (x_rel**2 + z_rel**2))
    I_zz_raw = np.sum(m_points * (x_rel**2 + y_rel**2))

    I_xy_raw = -np.sum(m_points * x_rel * y_rel)
    I_xz_raw = -np.sum(m_points * x_rel * z_rel)
    I_yz_raw = -np.sum(m_points * y_rel * z_rel)

    I_xx = I_xx_raw * (M_target / M_raw)
    I_yy = I_yy_raw * (M_target / M_raw)
    I_zz = I_zz_raw * (M_target / M_raw)
    I_xy = I_xy_raw * (M_target / M_raw)
    I_xz = I_xz_raw * (M_target / M_raw)
    I_yz = I_yz_raw * (M_target / M_raw)

    return asb.MassProperties(
        mass = M_target,
        x_cg = x_cg,
        y_cg = y_cg,
        z_cg = z_cg,
        Ixx  = I_xx,
        Iyy  = I_yy,
        Izz  = I_zz,
        Ixy  = I_xy,
        Ixz  = I_xz,
        Iyz  = I_yz,
    )


### Wing
mass_props['wing'] = lifting_surface_massprops_from_strips(
    wing      = wing,
    density   = density_wing,
    span_axis = "y",
)

x_cg_feather = (0.50 - 0.25) * wing_root_chord
z_cg_feather = (0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1)


mp_wing = mass_props['wing']

# CG
x_cg_wing_new = float(mp_wing.xyz_cg[0])
y_cg_wing_new = float(mp_wing.xyz_cg[1])
z_cg_wing_new = float(mp_wing.xyz_cg[2])

# Inertia tensor
Ixx = float(mp_wing.inertia_tensor[0, 0])
Iyy = float(mp_wing.inertia_tensor[1, 1])
Izz = float(mp_wing.inertia_tensor[2, 2])
Ixy = float(mp_wing.inertia_tensor[0, 1])
Ixz = float(mp_wing.inertia_tensor[0, 2])
Iyz = float(mp_wing.inertia_tensor[1, 2])

print("=== Wing Mass Properties (Strip Model) ===")
print(f"Wing area              [m^2]:  {float(wing.area()):.6g}")
print(f"Wing volume            [m^3]:  {float(wing.volume()):.6g}")
print(f"Mass                   [kg]:   {mp_wing.mass:.6g}")

print("\n-- Centre of Gravity --")
print(f"x_cg                   [m]:    {x_cg_wing_new:.6g}")
print(f"y_cg                   [m]:    {y_cg_wing_new:.6g}")
print(f"z_cg                   [m]:    {z_cg_wing_new:.6g}")

print("\n-- Inertia Tensor (about CG) --")
print(f"I_xx                   [kg·m²]: {Ixx:.6g}")
print(f"I_yy                   [kg·m²]: {Iyy:.6g}")
print(f"I_zz                   [kg·m²]: {Izz:.6g}")
print(f"I_xy                   [kg·m²]: {Ixy:.6g}")
print(f"I_xz                   [kg·m²]: {Ixz:.6g}")
print(f"I_yz                   [kg·m²]: {Iyz:.6g}")

print("Planform analytic CG x [m]:", float(x_cg_feather))
print("Planform analytic CG z [m]:", float(z_cg_feather))