###### Initialization

### Imports
import os
import copy
import subprocess

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import aerosandbox.tools.units as u
import pandas as pd

### Code version
def get_git_version() -> str:
    """
    Returns a short git description string (tag/commit), or 'unknown' if not in a repo.
    """
    try:
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return desc
    except Exception:
        return "unknown"

CODE_VERSION = get_git_version()

### AeroSandbox setup
opti = asb.Opti()
# variable_categories_to_freeze = "all"
# freeze_style = "float"

make_plots = True

### Constants
# gravitational acceleration
g = 9.81             # m/s^2

# air density
rho = 1.225          # kg/m^3

# material
density_wing = 33.0  # kg/m^3 for Depron foam


##### Overall Specs

### Operating point
# target turn radius
R_target = opti.variable(init_guess=1.0, lower_bound=0.1, upper_bound=2.5)

# operated height
z_th = 1.00  # m

# operating point (wind axes)
op_point = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=5.0,
        lower_bound=0.1,
        upper_bound=10.0,
        log_transform=True,
    ),
    alpha=opti.variable(
        init_guess=0.0,
        lower_bound=-10.0,
        upper_bound=10.0,
    ),
    beta=0.0,  # coordinated turn
    p=0.0,     # coordinated turn
)

# bank angle
phi_rad = np.arctan(op_point.velocity ** 2 / (g * R_target))
phi = np.degrees(phi_rad)

# load factor
n_load = 1 / np.cos(phi_rad)

# control surface deflections
delta_A_deg = opti.variable(init_guess=0.0, lower_bound=-25.0, upper_bound=25.0)
delta_A_max_rad = np.radians(25.0)
delta_A_eff_rad = 1.00 * delta_A_max_rad
delta_R_deg = opti.variable(init_guess=0.0, lower_bound=-30.0, upper_bound=30.0)
delta_E_deg = opti.variable(init_guess=0.0, lower_bound=-25.0, upper_bound=25.0)

# effective L/D in turn
L_over_D_turn = opti.variable(init_guess=15.0, lower_bound=0.1, log_transform=True)

### Take-off gross weight
design_mass_TOGW = opti.variable(init_guess=0.1, lower_bound=1e-3)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)  # numerical clamp


###### Lifting Surface

### Airfoils
airfoil_names = ["ag04", "naca0008", "s1223", "s3021"]
airfoils = {
    name: asb.Airfoil(name=name)
    for name in airfoil_names
}

# generate aerodynamic polars using XFoil
for airfoil in airfoils.values():
    airfoil.generate_polars(
        cache_filename=f"cache/{airfoil.name}.json",
        alphas=np.linspace(-10.0, 10.0, 21),
    )

### Control Surfaces
aileron_cs = asb.ControlSurface(
    name="aileron",
    symmetric=False,
    hinge_point=0.75,
    trailing_edge=True,
)

rudder_cs = asb.ControlSurface(
    name="rudder",
    symmetric=True,
    hinge_point=0.75,
    trailing_edge=True,
)

elevator_cs = asb.ControlSurface(
    name="elevator",
    symmetric=True,
    hinge_point=0.75,
    trailing_edge=True,
)

##### Vehicle Definition
# datum (0, 0, 0): quarter-chord of main-wing centerline section.

### Nose
x_nose = opti.variable(
    init_guess=-0.1,
    upper_bound=1e-3,
)

### Wing
b_W = opti.variable(  # span
    init_guess=0.3,
    lower_bound=0.1,
    upper_bound=10.0,
)
Gamma_W_deg = opti.variable(  # dihedral
    init_guess=11.0,
    lower_bound=0.0,
    upper_bound=20.0,
)
c_root_W = opti.variable(  # root chord
    init_guess=0.15,
    lower_bound=1e-3,
)

lambda_W = 1.0  # taper ratio

def wing_rotation(xyz):
    """Apply dihedral rotation Γ_W about the x-axis."""
    dihedral_rot = np.rotation_matrix_3D(
        angle=np.radians(Gamma_W_deg),
        axis="X",
    )
    return dihedral_rot @ np.array(xyz)

def chord_W(y):
    """Chord distribution c_W(y) for a simple tapered wing."""
    half_span = b_W / 2
    c_tip_W = lambda_W * c_root_W
    span_fraction = np.abs(y) / half_span  # 0 at root, 1 at tip
    return (1 - span_fraction) * c_root_W + span_fraction * c_tip_W

def twist_W(y):
    """Twist distribution (currently untwisted)."""
    return np.zeros_like(y)

# spanwise stations
y_W_stations = np.sinspace(0, b_W / 2, 11, reverse_spacing=True)

# wing sections with control surfaces
N_W_span = 11
span_fracs = onp.linspace(0.0, 0.5, N_W_span)
wing_xsecs = []

for eta in span_fracs:
    y_W = eta * b_W
    if 0.25 <= eta <= 0.45:
        cs_list = [aileron_cs]
    else:
        cs_list = []

    wing_xsecs.append(
        asb.WingXSec(
            xyz_le=wing_rotation([-chord_W(y_W), y_W, 0.0]),
            chord=chord_W(y_W),
            airfoil=airfoils["s3021"],
            twist=twist_W(y_W),
            control_surfaces=cs_list,
        )
    )

# wing assembly
wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=wing_xsecs,
).translate([0.75 * c_root_W, 0.0, 0.0])

### Horizontal tailplane

l_HT = opti.variable(  # moment arm
    init_guess=0.6,
    lower_bound=0.2,
    upper_bound=1.5,
)

AR_HT = 4.0  # aspect ratio
lambda_HT = 1.0  # taper ratio
b_HT = opti.variable(init_guess=0.15, lower_bound=1e-3)  # span
c_root_HT = 2 * b_HT / (AR_HT * (1 + lambda_HT))  # root chord

def chord_HT(y):
    """Chord distribution c_HT(y) for a simple tapered horizontal tailplane."""
    half_span_HT = b_HT / 2
    c_tip_HT = lambda_HT * c_root_HT
    span_fraction = np.abs(y) / half_span_HT
    return (1 - span_fraction) * c_root_HT + span_fraction * c_tip_HT

def twist_HT(y):
    """Twist distribution (currently untwisted)."""
    return np.zeros_like(y)

# spanwise stations
y_HT_stations = np.sinspace(0, b_HT / 2, 7, reverse_spacing=True)

# horizontal tailplane sections with control surfaces
htail_xsecs = []
for i in range(np.length(y_HT_stations)):
    y_HT = y_HT_stations[i]
    htail_xsecs.append(
        asb.WingXSec(
            xyz_le=[l_HT - chord_HT(y_HT), y_HT, 0.0],
            chord=chord_HT(y_HT),
            twist=twist_HT(y_HT),
            airfoil=airfoils["naca0008"],
            control_surfaces=[elevator_cs],
        )
    )

# horizontal tailplane assembly
htail = asb.Wing(
    name="HTail",
    symmetric=True,
    xsecs=htail_xsecs,
)

# volume coefficient
V_HT = htail.area() * l_HT / (wing.area() * wing.mean_aerodynamic_chord())

### Vertical tailplane
b_VT = opti.variable(init_guess=0.07, lower_bound=1e-3)  # span

l_VT = l_HT  # moment arm
AR_VT = 2.0  # aspect ratio
lambda_VT = 1.0  # taper ratio
c_root_VT = 2 * b_VT / (AR_VT * (1 + lambda_VT))  # root chord

def chord_VT(z):
    """Chord distribution c_VT(y) for a simple tapered vertical tailplane."""
    c_tip_VT = lambda_VT * c_root_VT
    span_fraction = np.abs(z) / b_VT
    return (1 - span_fraction) * c_root_VT + span_fraction * c_tip_VT

def twist_VT(z):
    """Twist distribution (currently untwisted)."""
    return np.zeros_like(z)

# spanwise stations
z_VT_stations = np.sinspace(0, b_VT, 7, reverse_spacing=True)

# vertical tailplane sections with control surfaces
vtail_xsecs = []
for i in range(np.length(z_VT_stations)):
    z_VT = z_VT_stations[i]
    vtail_xsecs.append(
        asb.WingXSec(
            xyz_le=[l_VT - chord_VT(z_VT), 0.0, z_VT],
            chord=chord_VT(z_VT),
            twist=twist_VT(z_VT),
            airfoil=airfoils["naca0008"],
            control_surfaces=[rudder_cs],
        )
    )

# vertical tailplane assembly
vtail = asb.Wing(
    name="VTail",
    symmetric=False,
    xsecs=vtail_xsecs,
)

# volume coefficient
V_VT = vtail.area() * l_VT / (wing.area() * b_W)

### Fuselage
x_tail = np.maximum(l_HT, l_VT)

# fuselage assembly
fuselage = asb.Fuselage(
    name="Fuse",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[x_nose, 0.0, 0.0],
            radius=4e-3 / 2,
        ),
        asb.FuselageXSec(
            xyz_c=[x_tail, 0.0, 0.0],
            radius=4e-3 / 2,
        ),
    ],
)

### Airplane
airplane = asb.Airplane(
    name="Nausicaa",
    wings=[wing, htail, vtail],
    fuselages=[fuselage],
)

# Control surface deflections
airplane = airplane.with_control_deflections(
    {
        "aileron": delta_A_deg,
        "rudder": delta_R_deg,
        "elevator": delta_E_deg,
    }
)

##### Internal Geometry and Weights
mass_props = {}

def lifting_surface_massprops_from_strips(
    lifting_surface: asb.Wing,
    density: float,
    thickness: float = None,
    span_axis: str = "y",
):
    """
    Approximate mass properties of a lifting surface using spanwise strips.
    """

    # infer thickness if not given
    if thickness is None:
        A_planform = lifting_surface.area()
        V_geom = lifting_surface.volume()
        thickness = V_geom / A_planform

    # extract leading-edge positions and chords from xsecs
    xyz_le = np.stack(
        [xsec.xyz_le for xsec in lifting_surface.xsecs], axis=0
    )  # (N, 3)
    chords = np.array(
        [xsec.chord for xsec in lifting_surface.xsecs]
    )  # (N,)

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
    dspan = span_coord[1:] - span_coord[:-1]  # (N-1,)
    c_mid = 0.5 * (chords[:-1] + chords[1:])  # (N-1,)

    # half-surface area
    A_strip_half = c_mid * dspan  # (N-1,)

    # centroid of each half-strip in x, y, z
    x_mid_i = x_le[:-1] + 0.5 * chords[:-1]
    x_mid_ip1 = x_le[1:] + 0.5 * chords[1:]
    x_mid_strip = 0.5 * (x_mid_i + x_mid_ip1)  # (N-1,)
    y_mid_strip = 0.5 * (y_le[:-1] + y_le[1:])  # (N-1,)
    z_mid_strip = 0.5 * (z_le[:-1] + z_le[1:])  # (N-1,)

    # mass per half-strip
    m_half = density * thickness * A_strip_half  # (N-1,)

    # build discrete point masses for full surface
    if lifting_surface.symmetric and span_axis == "y":
        # mirror about y = 0
        m_points = np.concatenate([m_half, m_half], axis=0)
        x_points = np.concatenate([x_mid_strip, x_mid_strip], axis=0)
        y_points = np.concatenate([y_mid_strip, -y_mid_strip], axis=0)
        z_points = np.concatenate([z_mid_strip, z_mid_strip], axis=0)
    else:
        m_points = m_half
        x_points = x_mid_strip
        y_points = y_mid_strip
        z_points = z_mid_strip

    # total mass from strips
    M_raw = np.sum(m_points)

    # target physical mass from geometry
    M_target = lifting_surface.volume() * density

    # centre of gravity
    x_cg = np.sum(m_points * x_points) / M_raw
    y_cg = np.sum(m_points * y_points) / M_raw
    z_cg = np.sum(m_points * z_points) / M_raw

    x_rel = x_points - x_cg
    y_rel = y_points - y_cg
    z_rel = z_points - z_cg

    I_xx_raw = np.sum(m_points * (y_rel ** 2 + z_rel ** 2))
    I_yy_raw = np.sum(m_points * (x_rel ** 2 + z_rel ** 2))
    I_zz_raw = np.sum(m_points * (x_rel ** 2 + y_rel ** 2))

    I_xy_raw = -np.sum(m_points * x_rel * y_rel)
    I_xz_raw = -np.sum(m_points * x_rel * z_rel)
    I_yz_raw = -np.sum(m_points * y_rel * z_rel)

    scale = M_target / M_raw

    I_xx = I_xx_raw * scale
    I_yy = I_yy_raw * scale
    I_zz = I_zz_raw * scale
    I_xy = I_xy_raw * scale
    I_xz = I_xz_raw * scale
    I_yz = I_yz_raw * scale

    return asb.MassProperties(
        mass=M_target,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg,
        Ixx=I_xx,
        Iyy=I_yy,
        Izz=I_zz,
        Ixy=I_xy,
        Ixz=I_xz,
        Iyz=I_yz,
    )

### Lifting surface mass properties
mass_props["wing"] = lifting_surface_massprops_from_strips(
    lifting_surface=wing,
    density=density_wing,
    span_axis="y",
)

mass_props["htail_surfaces"] = lifting_surface_massprops_from_strips(
    lifting_surface=htail,
    density=density_wing,
    span_axis="y",
)

mass_props["vtail_surfaces"] = lifting_surface_massprops_from_strips(
    lifting_surface=vtail,
    density=density_wing,
    span_axis="z",
)

### Avionics and fuselage items
mass_props["linkages"] = asb.MassProperties(
    mass=0.001,
    x_cg=x_tail / 2,
)

mass_props["Receiver"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.005,
    x_cg=x_nose + 0.010,
)

mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.009 + 0.004,
    x_cg=x_nose + 0.05,
)

mass_props["servo"] = asb.mass_properties_from_radius_of_gyration(
    mass=4 * 0.0022,
    x_cg=x_nose + 0.015,
)

mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.009 * (x_tail - x_nose),
    x_cg=(x_nose + x_tail) / 2,
)

mass_props["pod"] = asb.MassProperties(
    mass=0.007,
    x_cg=x_nose + 0.010,
)

mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
    mass=opti.variable(init_guess=0.0, lower_bound=0.0),
    x_cg=opti.variable(
        init_guess=0.0,
        lower_bound=x_nose,
        upper_bound=x_tail,
    ),
)

### Summation
mass_props_TOGW = asb.MassProperties(mass=0.0)
for mp in mass_props.values():
    mass_props_TOGW = mass_props_TOGW + mp

# glue weight (8% of structure)
mass_props["glue_weight"] = mass_props_TOGW * 0.08
mass_props_TOGW += mass_props["glue_weight"]

x_cg_total, y_cg_total, z_cg_total = mass_props_TOGW.xyz_cg


##### Thermal Vertical Velocity Field Model

def vertical_velocity_field(
    Q_v,
    R_th0,
    k_th,
    R_orbit,
    z_th_local,
    z0,
    fan_spacing,
):
    """
    2×2 fan thermal model (axisymmetric Gaussian approximation).

    Q_v       - Vertical volume flux (m^3/s)
    R_th0     - Core radius at z0 (m)
    k_th      - Empirical spreading rate
    z0        - Reference height for R_th0 (m)
    fan_spacing - spacing between each fan (m)
    R_orbit   - orbital radius of the glider (m)
    z_th_local - operating height of the glider (m)
    """

    # core radius as function of height
    R_th = R_th0 + k_th * (z_th_local - z0)
    R_th = np.maximum(R_th, 1e-6)  # avoid negative / zero radius

    # peak vertical velocity w_th(z)
    w_th = Q_v / (np.pi * R_th ** 2)

    # fan centres (2×2 array)
    fan_centres = [
        (-fan_spacing / 2, -fan_spacing / 2),
        (fan_spacing / 2, -fan_spacing / 2),
        (-fan_spacing / 2, fan_spacing / 2),
        (fan_spacing / 2, fan_spacing / 2),
    ]

    def w_at_xy(x, y):
        """Total vertical velocity at a single (x, y) from all four fans."""
        w_total = 0.0
        for x_centre, y_centre in fan_centres:
            R_i = np.sqrt((x - x_centre) ** 2 + (y - y_centre) ** 2)
            w_i = w_th * np.exp(-(R_i / R_th) ** 2)
            # no thermal below z0
            w_i = np.where(z_th_local < z0, 0.0, w_i)
            w_total = w_total + w_i
        return w_total

    # sample 4 azimuth angles around the orbit
    thetas = np.array(
        [
            np.pi / 4,
            3 * np.pi / 4,
            5 * np.pi / 4,
            7 * np.pi / 4,
        ]
    )

    w_sum = 0.0
    for theta in thetas:
        x = R_orbit * np.cos(theta)
        y = R_orbit * np.sin(theta)
        w_sum = w_sum + w_at_xy(x, y)

    # average over azimuth, as an approximation for glider operation
    w_avg = w_sum / len(thetas)
    return w_avg

### Thermal setup (CAMAX30)
Q_v = 1.69  # m^3/s per original estimate
x_center = 4.0
y_center = 2.5

R_th0 = 0.381  # assume core radius equal to fan radius at z0
k_th = 0.10    # typical turbulent plume spreading rate
z0 = 0.50      # reference height near fan centre (m)
fan_spacing = 2 * R_th0 + 0.5

# average w(R, z) along orbit
w = vertical_velocity_field(
    Q_v=Q_v,
    R_th0=R_th0,
    k_th=k_th,
    R_orbit=R_target,
    z_th_local=z_th,
    z0=z0,
    fan_spacing=fan_spacing,
)


##### Aerodynamics and Stability

### Aerodynamic force–moment model
ab = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point,
    xyz_ref=mass_props_TOGW.xyz_cg,
)

### Stability derivatives
aero = ab.run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=True,
    q=True,
    r=True,
)

### Aileron effectiveness
delta_A_fd_deg = 1.0  # finite difference step in degrees

# positive aileron deflection
airplane_plus = airplane.with_control_deflections(
    {
        "aileron": delta_A_deg - delta_A_fd_deg,
        "rudder": delta_R_deg,
        "elevator": delta_E_deg,
    }
)

ab_plus = asb.AeroBuildup(
    airplane=airplane_plus,
    op_point=op_point,
    xyz_ref=mass_props_TOGW.xyz_cg,
)
aero_plus = ab_plus.run()
Cl_plus = aero_plus["Cl"]

# negative aileron deflection
airplane_minus = airplane.with_control_deflections(
    {
        "aileron": delta_A_deg + delta_A_fd_deg,
        "rudder": delta_R_deg,
        "elevator": delta_E_deg,
    }
)

ab_minus = asb.AeroBuildup(
    airplane=airplane_minus,
    op_point=op_point,
    xyz_ref=mass_props_TOGW.xyz_cg,
)
aero_minus = ab_minus.run()
Cl_minus = aero_minus["Cl"]

# compute rolling moment coefficient derivative
Cl_delta_A_per_deg = (Cl_plus - Cl_minus) / (2 * delta_A_fd_deg)
Cl_delta_A = Cl_delta_A_per_deg * (180 / np.pi)

### Performance quantities
L_over_D = aero["L"] / aero["D"]
power_loss = aero["D"] * op_point.velocity
sink_rate = power_loss / g / mass_props_TOGW.mass
K_n = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
climb_rate = w - sink_rate

##### Finalize Optimization Problem

# “softmax” style penalty on climb rate shortfall
k_soft = 50.0
shortfall = (1 / k_soft) * np.log(1 + np.exp(-k_soft * climb_rate))
obj_climb = shortfall ** 2

obj_span = b_W + b_HT + b_VT

obj_control = 1e-5 * (
    delta_E_deg ** 2 + delta_A_deg ** 2 + delta_R_deg ** 2
)

objective = obj_climb + obj_span + obj_control

# tiny penalty on ballast CG position
penalty = (mass_props["ballast"].x_cg / 1e3) ** 2

opti.minimize(objective + penalty)

### Optimization constraints
opti.subject_to(
    [

        # coordinated turn bank-angle bounds
        opti.bounded(5.0, phi, 65.0),

        # aerodynamics (force balance in turn)
        aero["L"] >= n_load * mass_props_TOGW.mass * g,

        # trim and stability constraints
        aero["Cl"] == 0.0,  # trimmed in roll
        aero["Cm"] == 0.0,  # trimmed in pitch
        aero["Cn"] == 0.0,  # trimmed in yaw

        # stability derivatives
        aero["Clb"] <= -0.025,
        aero["Cnb"] >= 0.03,

        opti.bounded(0.04, K_n, 0.10),  # static margin Kn
        opti.bounded(0.40, V_HT, 0.70),
        opti.bounded(0.02, V_VT, 0.04),
    ]
)

### Additional constraints
opti.subject_to(
    [
        L_over_D_turn == L_over_D,
        design_mass_TOGW == mass_props_TOGW.mass,
    ]
)


##### Roll-In Manoeuvre

### Kinematics

# achievable constant roll rate p during roll-in
p_roll_deg = opti.variable(init_guess=60.0, lower_bound=5.0)
p_roll = np.radians(p_roll_deg)

# roll-in duration
t_roll = phi_rad / p_roll

# experiment volume bounds
x_min, x_max = 0.0, 8.0
y_min, y_max = 0.0, 5.0

# worst-case entry condition
x0 = 0.0
y0 = y_center

# initial heading
psi0_deg = opti.variable(init_guess=0.0, lower_bound=-90.0, upper_bound=90.0)
psi0 = np.radians(psi0_deg)

# discretisation settings
N_roll = 41
dt = t_roll / (N_roll - 1)

# state variables along roll-in
psi = opti.variable(init_guess=0.0 * np.ones(N_roll))
x = opti.variable(init_guess=x0 * np.ones(N_roll))
y = opti.variable(init_guess=y0 * np.ones(N_roll))

### Roll-in constraints

opti.subject_to([

    # initial conditions
    psi[0] == psi0,
    x[0]   == x0,
    y[0]   == y0,
])

# forward Euler integration along roll-in
for k_idx in range(N_roll - 1):
    # bank angle during roll-in
    phi_k = p_roll * (k_idx * dt)

    # coordinated heading rate r = g tan(ϕ) / V
    r_k = g * np.tan(phi_k) / op_point.velocity

    opti.subject_to(
        [
            # heading integration
            psi[k_idx + 1] == psi[k_idx] + r_k * dt,

            # position integration
            x[k_idx + 1] == x[k_idx] + op_point.velocity * np.cos(psi[k_idx]) * dt,
            y[k_idx + 1] == y[k_idx] + op_point.velocity * np.sin(psi[k_idx]) * dt,

            # stay inside experimental volume
            opti.bounded(x_min, x[k_idx + 1], x_max),
            opti.bounded(y_min, y[k_idx + 1], y_max),
        ]
    )

### End of roll-in constraints

dx_end = x[-1] - x_center
dy_end = y[-1] - y_center

opti.subject_to(
    [
        # reach the target turn radius R_target
        (x[-1] - x_center) ** 2 + (y[-1] - y_center) ** 2 == R_target ** 2,

        # heading perpendicular to radius at entry
        np.cos(psi[-1]) * dx_end + np.sin(psi[-1]) * dy_end == 0.0,

        # positive yaw rate (left turn about centre)
        -np.cos(psi[-1]) * dy_end + np.sin(psi[-1]) * dx_end >= 0.0,
    ]
)

### Roll-rate capability

# peak roll-rate limit from bang–bang roll assumption with roll damping
L_roll_max = (
    0.5
    * rho
    * op_point.velocity ** 2
    * wing.area()
    * b_W
    * (
        Cl_delta_A * delta_A_eff_rad
        + aero["Clp"] * p_roll
    )
)

# ensure the chosen roll is dynamically achievable
opti.subject_to(L_roll_max >= 0.0)

p_dot_max = L_roll_max / mass_props_TOGW.inertia_tensor[0, 0]

# constraint on p_roll
opti.subject_to(p_roll**2 <= phi_rad * p_dot_max)

p_roll_max = np.sqrt(phi_rad * p_dot_max)

##### Solve Optimization Problem

if __name__ == "__main__":
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug

    s = lambda x: sol.value(x)

    ### Turn symbolic airplane into numeric values
    airplane = sol(airplane)

    # lifting surfaces
    wing = copy.deepcopy(airplane.wings[0])
    htail = copy.deepcopy(airplane.wings[1])
    vtail = copy.deepcopy(airplane.wings[2])

    # fuselage
    fuselage = copy.deepcopy(airplane.fuselages[0])

    # mass properties
    mass_props = sol(mass_props)
    mass_props_TOGW = sol(mass_props_TOGW)

    # performance
    aero = sol(aero)
    Cl_delta_A = sol(Cl_delta_A)
    sink_rate = sol(sink_rate)
    w = sol(w)
    climb_rate = sol(climb_rate)

    # operating point
    op_point = sol(op_point)
    phi = sol(phi)
    phi_rad_val = sol(phi_rad)
    R_target = sol(R_target)
    n_load = sol(n_load)

    # roll-in kinematics
    p_roll_deg = sol(p_roll_deg)
    p_roll = sol(p_roll)
    p_roll_max = sol(p_roll_max)
    t_roll = sol(t_roll)
    psi0_deg = sol(psi0_deg)
    psi0 = sol(psi0)

    psi_roll = sol(psi)
    x_roll = sol(x)
    y_roll = sol(y)

    t_roll_vec = onp.linspace(
        0.0, float(t_roll), int(len(onp.atleast_1d(x_roll)))
    )

    # control surfaces
    aileron_cs = sol(aileron_cs)
    rudder_cs = sol(rudder_cs)
    elevator_cs = sol(elevator_cs)

    # control surface deflections
    delta_E_deg = sol(delta_E_deg)
    delta_A_deg = sol(delta_A_deg)
    delta_R_deg = sol(delta_R_deg)

    # wing and tails
    b_W = sol(b_W)
    Gamma_W_deg = sol(Gamma_W_deg)
    c_root_W = sol(c_root_W)

    l_HT = sol(l_HT)
    b_HT = sol(b_HT)
    c_root_HT = sol(c_root_HT)

    l_VT = sol(l_VT)
    b_VT = sol(b_VT)
    c_root_VT = sol(c_root_VT)

    # thermal parameters
    R_th0 = sol(R_th0)

    # objective and components
    objective_val = sol(objective)
    penalty_val = sol(penalty)
    obj_climb = sol(obj_climb)
    obj_span = sol(obj_span)
    obj_control = sol(obj_control)

    # stability and volume coefficients
    K_n = sol(K_n)
    V_HT = sol(V_HT)
    V_VT = sol(V_VT)
    L_over_D_turn = sol(L_over_D_turn)

    # nose and tail
    x_nose = sol(x_nose)
    x_tail = sol(x_tail)

    # TOGW mass
    design_mass_TOGW = sol(design_mass_TOGW)

    ##### Result Formatting

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    print_title = lambda s_: print(s_.upper().join(["*" * 20] * 2))

    def fmt(x_):
        return f"{s(x_):.6g}"

    ### Optimisation problem summary
    print_title("Objective contribution breakdown")
    for key, val in {
        "Total Objective": fmt(objective),
        "Climb rate": f"{fmt(obj_climb)} ({s(obj_climb) / s(objective) * 100:.1f}%)",
        "Span": f"{fmt(obj_span)} ({s(obj_span) / s(objective) * 100:.1f}%)",
        "Control effect": f"{fmt(obj_control)} ({s(obj_control) / s(objective) * 100:.1f}%)",
    }.items():
        print(f"{key.rjust(25)} = {val}")

    print_title("Design variables")

    def check_var(name, var_value, lb=None, ub=None, atol=1e-6, rtol=1e-3):
        v = float(s(var_value))
        hits = []
        if lb is not None:
            if v <= lb + max(atol, rtol * max(1.0, abs(lb))):
                hits.append("LOWER")
        if ub is not None:
            if v >= ub - max(atol, rtol * max(1.0, abs(ub))):
                hits.append("UPPER")
        status = ", ".join(hits) if hits else "OK"
        print(f"{name:25s} = {v: .6g}  [{status}]")

    check_var("R_target (m)", R_target, lb=0.1, ub=2.5)

    check_var("V (m/s)", op_point.velocity, lb=0.1, ub=10.0)
    check_var("α (deg)", op_point.alpha, lb=-10.0, ub=10.0)
    check_var("ϕ (deg)", phi, lb=5.0, ub=65.0)

    check_var("ψ0 (deg)", psi0_deg, lb=-90.0, ub=90.0)
    check_var("p_roll (deg/s)", p_roll_deg, lb=5.0, ub=720.0)

    check_var("TOGW (kg)", design_mass_TOGW, lb=1e-3)

    check_var("x_nose (m)", x_nose, ub=1e-3)

    check_var("b_W (m)", b_W, lb=0.1, ub=10.0)
    check_var("Γ_W (deg)", Gamma_W_deg, lb=0.0, ub=20.0)
    check_var("c_root_W (m)", c_root_W, lb=1e-3)

    check_var("l_HT (m)", l_HT, lb=0.2, ub=1.5)
    check_var("b_HT (m)", b_HT, lb=1e-3)
    check_var("b_VT (m)", b_VT, lb=1e-3)

    check_var("m_ballast (kg)", mass_props["ballast"].mass, lb=0.0)
    check_var(
        "x_cg_ballast (m)",
        mass_props["ballast"].x_cg,
        lb=x_nose,
        ub=x_tail,
    )

    check_var("δ_E (deg)", delta_E_deg, lb=-25.0, ub=25.0)
    check_var("δ_A (deg)", delta_A_deg, lb=-30.0, ub=30.0)
    check_var("δ_R (deg)", delta_R_deg, lb=-25.0, ub=25.0)

    check_var(
        "Lift (N)",
        aero["L"],
        lb=n_load * mass_props_TOGW.mass * g,
    )
    check_var("C_m", aero["Cm"])
    check_var("C_l", aero["Cl"])
    check_var("Cl,β", aero["Clb"], ub=-0.025)
    check_var("Cn,β", aero["Cnb"], lb=0.03)
    check_var("K_n", K_n, lb=0.04, ub=0.10)
    check_var("V_HT", V_HT, lb=0.40, ub=0.70)
    check_var("V_VT", V_VT, lb=0.02, ub=0.04)

    ### Output summary
    print_title("Outputs")

    # mission & trim state
    print("\n--- Mission & trim state ---")
    for key, val in {
        "R_target (m)": fmt(R_target),
        "V (m/s)": fmt(op_point.velocity),
        "α (deg)": fmt(op_point.alpha),
        "ϕ (deg)": fmt(phi),
        "n": fmt(n_load),
        "C_L (turn)": fmt(aero["CL"]),
    }.items():
        print(f"{key.rjust(25)} = {val}")

    # performance
    print("\n--- Performance ---")
    for key, val in {
        "L/D_turn": fmt(L_over_D_turn),
        "w (m/s)": fmt(w),
        "sink_rate (m/s)": fmt(sink_rate),
        "climb_rate (m/s)": fmt(climb_rate),
    }.items():
        print(f"{key.rjust(25)} = {val}")

    # roll performance
    print("\n--- Roll performance ---")
    for key, val in {
        "p (rad/s)": fmt(p_roll),
        "p_max (rad/s)": fmt(p_roll_max),
    }.items():
        print(f"{key.rjust(25)} = {val}")

    # stability
    print("\n--- Stability ---")
    for key, val in {
        "C_mα": fmt(aero["Cma"]),
        "C_nβ": fmt(aero["Cnb"]),
        "C_m": fmt(aero["Cm"]),
        "K_n": fmt(K_n),
    }.items():
        print(f"{key.rjust(25)} = {val}")

    # mass & geometry
    print("\n--- Mass & geometry ---")
    for key, val in {
        "m_TOGW (kg)": f"{fmt(mass_props_TOGW.mass)} kg "
                       f"({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "I_xx (kg m^2)": f"{fmt(mass_props_TOGW.inertia_tensor[0, 0])} kg/m^2",
        "Wing Re": eng_string(
            op_point.reynolds(sol(wing.mean_aerodynamic_chord()))
        ),
        "CG location": "("
                       + ", ".join([fmt(x_c) for x_c in mass_props_TOGW.xyz_cg])
                       + ") m",
        "b_W (m)": f"{fmt(b_W)} m ({fmt(b_W / u.foot)} ft)",
    }.items():
        print(f"{key.rjust(25)} = {val}")

    ### Mass breakdown
    print_title("Mass props")
    for name, mp in mass_props.items():
        print(
            f"{name.rjust(25)} = {mp.mass * 1e3:.2f} g ({mp.mass / u.oz:.2f} oz)"
        )

    ##### Plotting
    if make_plots:
        ### Three-view
        airplane.draw_three_view(show=False)
        p.show_plot(
            tight_layout=False,
            savefig="figures/three_view.png",
        )

        ### Mass budget pie chart
        fig, ax = plt.subplots(
            figsize=(12, 5),
            subplot_kw=dict(aspect="equal"),
            dpi=300,
        )

        name_remaps = {
            **{
                k: k.replace("_", " ").title()
                for k in mass_props.keys()
            },
        }

        mass_props_to_plot = copy.deepcopy(mass_props)
        if mass_props_to_plot["ballast"].mass < 1e-6:
            mass_props_to_plot.pop("ballast")

        p.pie(
            values=[mp.mass for mp in mass_props_to_plot.values()],
            names=[
                n if n not in name_remaps else name_remaps[n]
                for n in mass_props_to_plot.keys()
            ],
            center_text=(
                f"$\\bf{{Mass\\ Budget}}$\nTOGW: {s(mass_props_TOGW.mass * 1e3):.2f} g"
            ),
            label_format=lambda name, value, percentage: (
                f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%"
            ),
            startangle=110,
            arm_length=30,
            arm_radius=20,
            y_max_labels=1.1,
        )
        p.show_plot(savefig="figures/mass_budget.png")

        ### Thermal and Roll-In Trajectory
        base_cmap = plt.cm.YlOrRd
        colors = base_cmap(onp.linspace(0, 1, 256))
        Nfade = 15
        first_color = colors[Nfade].copy()
        for i in range(Nfade):
            t_ = i / (Nfade - 1)
            colors[i] = (1 - t_) * onp.array([1, 1, 1, 1]) + t_ * first_color
        cmap_white0 = mcolors.ListedColormap(colors)

        fan_centres_plot = [
            (x_center - fan_spacing / 2, y_center - fan_spacing / 2),
            (x_center + fan_spacing / 2, y_center - fan_spacing / 2),
            (x_center - fan_spacing / 2, y_center + fan_spacing / 2),
            (x_center + fan_spacing / 2, y_center + fan_spacing / 2),
        ]

        def vertical_velocity_field_single(
            Q_v_local,
            R_th0_local,
            k_th_local,
            x_local,
            y_local,
            z_local,
            z0_local,
            x_center_local,
            y_center_local,
        ):
            R_local = onp.sqrt(
                (x_local - x_center_local) ** 2
                + (y_local - y_center_local) ** 2
            )
            R_th_local = R_th0_local + k_th_local * (z_local - z0_local)
            R_th_local = onp.maximum(R_th_local, 1e-6)

            w_th_local = Q_v_local / (onp.pi * R_th_local ** 2)
            w_local = w_th_local * onp.exp(-(R_local / R_th_local) ** 2)
            w_local = onp.where(z_local < z0_local, 0.0, w_local)
            return w_local

        def vertical_velocity_field_multi(
            Q_v_local,
            R_th0_local,
            k_th_local,
            x_local,
            y_local,
            z_local,
            z0_local,
            fan_centres_local,
        ):
            w_total_local = 0.0
            for x_c, y_c in fan_centres_local:
                w_total_local += vertical_velocity_field_single(
                    Q_v_local,
                    R_th0_local,
                    k_th_local,
                    x_local,
                    y_local,
                    z_local,
                    z0_local,
                    x_c,
                    y_c,
                )
            return w_total_local

        # grid
        Nx, Ny = 120, 80
        xg = onp.linspace(x_min, x_max, Nx)
        yg = onp.linspace(y_min, y_max, Ny)
        Xg, Yg = onp.meshgrid(xg, yg, indexing="xy")

        z_plot = float(z_th)
        Zg = z_plot * onp.ones_like(Xg)

        W_slice = vertical_velocity_field_multi(
            Q_v_local=Q_v,
            R_th0_local=R_th0,
            k_th_local=k_th,
            x_local=Xg,
            y_local=Yg,
            z_local=Zg,
            z0_local=z0,
            fan_centres_local=fan_centres_plot,
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)

        levels = onp.linspace(W_slice.min(), W_slice.max(), 256)
        cf = ax.contourf(
            Xg,
            Yg,
            W_slice,
            levels=levels,
            cmap=cmap_white0,
            zorder=1,
        )

        # initial thermal radius circles
        theta_circle = onp.linspace(0, 2 * onp.pi, 200)
        for x_c, y_c in fan_centres_plot:
            ax.plot(
                x_c + R_th0 * onp.cos(theta_circle),
                y_c + R_th0 * onp.sin(theta_circle),
                color="k",
                linewidth=1.3,
                zorder=0,
            )

        # target orbit radius R_target
        ax.plot(
            x_center + R_target * onp.cos(theta_circle),
            y_center + R_target * onp.sin(theta_circle),
            color="k",
            linestyle="--",
            linewidth=1.3,
            zorder=1000,
        )

        x_txt = x_center + float(R_target)
        y_txt = y_center

        ax.text(
            x_txt + 0.04 * float(R_target),
            y_txt,
            rf"$R_\mathrm{{target}} = {float(R_target):.2f}\,\mathrm{{m}}$",
            fontsize=11.5,
            color="k",
            verticalalignment="center",
            zorder=1200,
        )

        # roll-in trajectory
        xr = onp.asarray(x_roll, dtype=float)
        yr = onp.asarray(y_roll, dtype=float)
        t_param = onp.linspace(0, 1, len(xr))

        points = onp.array([xr, yr]).T.reshape(-1, 1, 2)
        segments = onp.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap="winter",
            array=t_param,
            linewidth=1.5,
            zorder=1100,
        )
        ax.add_collection(lc)

        line_cmap = mpl.colormaps["winter"]
        c_start = line_cmap(0.0)
        c_end = line_cmap(1.0)

        ax.scatter(
            [xr[0]],
            [yr[0]],
            s=35,
            marker="o",
            facecolor=c_start,
            edgecolor="k",
            linewidth=0.6,
            zorder=1200,
        )

        ax.scatter(
            [xr[-1]],
            [yr[-1]],
            s=55,
            marker="X",
            facecolor=c_end,
            edgecolor="k",
            linewidth=0.6,
            zorder=1200,
        )

        cbar = fig.colorbar(cf, ax=ax, shrink=0.95)
        cbar.set_label(f"w (m/s) at z = {z_plot:.2f} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(
            "figures/thermal_with_rollin_trajectory.png",
            dpi=300,
            bbox_inches="tight",
        )

    ###### Save results to file

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    def to_scalar(x_):
        """Safely convert scalars or small arrays to float for export."""
        try:
            import numpy as _np

            arr = _np.array(x_)
            if arr.shape == ():
                return float(arr)
            return float(arr.flatten()[0])
        except Exception:
            try:
                return float(x_)
            except Exception:
                return x_

    ### Design results (aligned with logbook symbols)
    design_results = {
        # operating point
        "V (m/s)": to_scalar(op_point.velocity),
        "α (deg)": to_scalar(op_point.alpha),
        "ϕ (deg)": to_scalar(phi),
        "R_target (m)": to_scalar(R_target),
        "n": to_scalar(n_load),

        # global mass and performance
        "m_TOGW (kg)": to_scalar(mass_props_TOGW.mass),
        "L/D_turn": to_scalar(L_over_D_turn),
        "sink_rate (m/s)": to_scalar(sink_rate),
        "w (m/s)": to_scalar(w),
        "climb_rate (m/s)": to_scalar(climb_rate),
        "K_n": to_scalar(K_n),
        "V_HT": to_scalar(V_HT),
        "V_VT": to_scalar(V_VT),
        "Re_W": to_scalar(
            op_point.reynolds(sol(wing.mean_aerodynamic_chord()))
        ),

        # geometry
        "x_nose (m)": to_scalar(x_nose),
        "x_tail (m)": to_scalar(x_tail),

        "b_W (m)": to_scalar(b_W),
        "Γ_W (deg)": to_scalar(Gamma_W_deg),
        "c_root_W (m)": to_scalar(c_root_W),
        "λ_W": to_scalar(lambda_W),
        "S_W (m^2)": to_scalar(wing.area()),

        "l_HT (m)": to_scalar(l_HT),
        "b_HT (m)": to_scalar(b_HT),
        "c_root_HT (m)": to_scalar(c_root_HT),
        "λ_HT": to_scalar(lambda_HT),
        "S_HT (m^2)": to_scalar(htail.area()),

        "l_VT (m)": to_scalar(l_VT),
        "b_VT (m)": to_scalar(b_VT),
        "c_root_VT (m)": to_scalar(c_root_VT),
        "λ_VT": to_scalar(lambda_VT),
        "S_VT (m^2)": to_scalar(vtail.area()),

        # control surfaces (hinge positions)
        "hinge_point_A": to_scalar(aileron_cs.hinge_point),
        "hinge_point_R": to_scalar(rudder_cs.hinge_point),
        "hinge_point_E": to_scalar(elevator_cs.hinge_point),

        # control surface deflections
        "δ_A (deg)": to_scalar(delta_A_deg),
        "δ_R (deg)": to_scalar(delta_R_deg),
        "δ_E (deg)": to_scalar(delta_E_deg),

        # roll-in kinematics
        "p (rad/s)": to_scalar(p_roll),
        "p (deg/s)": to_scalar(p_roll_deg),
        "p_max (rad/s)": to_scalar(p_roll_max),
        "t_roll (s)": to_scalar(t_roll),
        "ψ0 (deg)": to_scalar(psi0_deg),
        "Cl,δA (rad^-1)": to_scalar(Cl_delta_A),

        # objective decomposition
        "objective_total": to_scalar(objective_val),
        "objective_climb": to_scalar(obj_climb),
        "objective_span": to_scalar(obj_span),
        "objective_control": to_scalar(obj_control),
        "penalty": to_scalar(penalty_val),

        # CG location
        "x_CG (m)": to_scalar(mass_props_TOGW.xyz_cg[0]),
        "y_CG (m)": to_scalar(mass_props_TOGW.xyz_cg[1]),
        "z_CG (m)": to_scalar(mass_props_TOGW.xyz_cg[2]),

        # inertia tensor about CG
        "I_xx (kg m^2)": to_scalar(
            mass_props_TOGW.inertia_tensor[0, 0]
        ),
        "I_yy (kg m^2)": to_scalar(
            mass_props_TOGW.inertia_tensor[1, 1]
        ),
        "I_zz (kg m^2)": to_scalar(
            mass_props_TOGW.inertia_tensor[2, 2]
        ),
        "I_xy (kg m^2)": to_scalar(
            mass_props_TOGW.inertia_tensor[0, 1]
        ),
        "I_xz (kg m^2)": to_scalar(
            mass_props_TOGW.inertia_tensor[0, 2]
        ),
        "I_yz (kg m^2)": to_scalar(
            mass_props_TOGW.inertia_tensor[1, 2]
        ),
    }

    # component masses
    for name, mp in mass_props.items():
        design_results[f"mass_{name}_kg"] = to_scalar(mp.mass)

    ### Lifting surface leading-edge locations
    def LE_coords(xsec):
        return (
            to_scalar(xsec.xyz_le[0]),
            to_scalar(xsec.xyz_le[1]),
            to_scalar(xsec.xyz_le[2]),
        )

    # wing root and tip LE
    wing_root_LE = LE_coords(wing.xsecs[0])
    wing_tip_LE = LE_coords(wing.xsecs[-1])

    design_results["wing root LE x (m)"] = wing_root_LE[0]
    design_results["wing root LE y (m)"] = wing_root_LE[1]
    design_results["wing root LE z (m)"] = wing_root_LE[2]

    design_results["wing tip LE x (m)"] = wing_tip_LE[0]
    design_results["wing tip LE y (m)"] = wing_tip_LE[1]
    design_results["wing tip LE z (m)"] = wing_tip_LE[2]

    # horizontal tail root and tip LE
    ht_root_LE = LE_coords(htail.xsecs[0])
    ht_tip_LE = LE_coords(htail.xsecs[-1])

    design_results["htail root LE x (m)"] = ht_root_LE[0]
    design_results["htail root LE y (m)"] = ht_root_LE[1]
    design_results["htail root LE z (m)"] = ht_root_LE[2]

    design_results["htail tip LE x (m)"] = ht_tip_LE[0]
    design_results["htail tip LE y (m)"] = ht_tip_LE[1]
    design_results["htail tip LE z (m)"] = ht_tip_LE[2]

    # vertical tail root and tip LE
    vt_root_LE = LE_coords(vtail.xsecs[0])
    vt_tip_LE = LE_coords(vtail.xsecs[-1])

    design_results["vtail root LE x (m)"] = vt_root_LE[0]
    design_results["vtail root LE y (m)"] = vt_root_LE[1]
    design_results["vtail root LE z (m)"] = vt_root_LE[2]

    design_results["vtail tip LE x (m)"] = vt_tip_LE[0]
    design_results["vtail tip LE y (m)"] = vt_tip_LE[1]
    design_results["vtail tip LE z (m)"] = vt_tip_LE[2]

    ### Aerodynamic and stability results
    aero_results = {}

    def extract_component_list(name, component_list, out_dict):
        for idx, comp in enumerate(component_list):
            prefix = f"{name}_comp{idx + 1}"

            out_dict[f"{prefix}_L (N)"] = to_scalar(comp.L)
            out_dict[f"{prefix}_D (N)"] = to_scalar(comp.D)
            out_dict[f"{prefix}_Y (N)"] = to_scalar(comp.Y)

            out_dict[f"{prefix}_l_b"] = to_scalar(comp.l_b)
            out_dict[f"{prefix}_m_b"] = to_scalar(comp.m_b)
            out_dict[f"{prefix}_n_b"] = to_scalar(comp.n_b)

            out_dict[f"{prefix}_span_eff (m)"] = to_scalar(
                comp.span_effective
            )
            out_dict[f"{prefix}_oswald"] = to_scalar(
                comp.oswalds_efficiency
            )

    for key, value in aero.items():
        if not isinstance(value, list):
            aero_results[f"aero_{key}"] = to_scalar(value)
        else:
            if len(value) > 0 and hasattr(value[0], "L") and hasattr(
                value[0], "span_effective"
            ):
                component_name = key.replace(
                    "aero_", ""
                ).replace("_aero_components", "")
                extract_component_list(
                    component_name, value, aero_results
                )
            else:
                for idx, item in enumerate(value):
                    aero_results[f"aero_{key}[{idx}]"] = to_scalar(item)

    all_results = {}
    all_results.update(design_results)
    all_results.update(aero_results)

    results_df = pd.DataFrame(
        list(all_results.items()),
        columns=["Variable", "Value"],
    )

    csv_path = os.path.join(results_dir, "nausicaa_results.csv")
    excel_path = os.path.join(results_dir, "nausicaa_results.xlsx")

    results_df.to_csv(csv_path, index=False)
    results_df.to_excel(excel_path, index=False)

    print(f"\nSaved results to:\n  {csv_path}\n  {excel_path}")

    ##### Save Roll-in Timeseries
    rollin_df = pd.DataFrame(
        {
            "t (s)": onp.asarray(t_roll_vec, dtype=float),
            "x (m)": onp.asarray(x_roll, dtype=float),
            "y (m)": onp.asarray(y_roll, dtype=float),
            "ψ (rad)": onp.asarray(psi_roll, dtype=float),
            "ψ (deg)": onp.degrees(
                onp.asarray(psi_roll, dtype=float)
            ),
        }
    )

    rollin_csv_path = os.path.join(
        results_dir, "nausicaa_rollin_timeseries.csv"
    )
    rollin_excel_path = os.path.join(
        results_dir, "nausicaa_rollin_timeseries.xlsx"
    )

    rollin_df.to_csv(rollin_csv_path, index=False)
    rollin_df.to_excel(rollin_excel_path, index=False)

    print(
        f"Saved roll-in timeseries to:\n  {rollin_csv_path}\n  {rollin_excel_path}"
    )