###### Initialization

### Import
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
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

### Aerosandbox setup
opti = asb.Opti()
# variable_categories_to_freeze = "all",
# freeze_style = "float"

make_plots = True

### Constant
# gravitational acceleration
g = 9.81

# material
density_wing = 33.0 # kg/m^3 for depron foam


##### Overall Specs

### Operating point
# target radius
r_target = opti.variable(init_guess = 1.0, lower_bound = 0.1, upper_bound = 2.5)

# operated height
z_op = 1.00 

# operating point
op_point = asb.OperatingPoint(
    velocity = opti.variable(init_guess = 15.5, lower_bound = 0.1, log_transform = True),
    alpha    = opti.variable(init_guess = 0, lower_bound = -10.0, upper_bound = 10.0),
    beta     = 0.0, # coordinated turn
    p        = 0.0, # coordinated turn
)

# bank angle
phi = np.degrees(np.arctan(op_point.velocity ** 2 / (g * r_target)))

# load factor
n_load = 1 / np.cos(np.radians(phi))

# control surface deflection
aileron_deflection  = opti.variable(init_guess = 0.0, lower_bound = -25.0, upper_bound = 25.0)
rudder_deflection   = opti.variable(init_guess = 0.0, lower_bound = -30.0, upper_bound = 30.0)
elevator_deflection = opti.variable(init_guess = 0.0, lower_bound = -25.0, upper_bound = 25.0)

# effective L/D
LD_turn = opti.variable(init_guess = 15, lower_bound = 0.1, log_transform = True)

### Take off gross weight 
design_mass_TOGW = opti.variable(init_guess = 0.1, lower_bound = 1e-3)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3) # numerical clamp


###### Lifting Surface

### Airfoil
# assign airfoil
airfoils = {name: asb.Airfoil(name = name,) for name in ["ag04", "naca0008", "s1223", "s3021"]}

# generating aerodynamic polars using XFoil
for v in airfoils.values():
    v.generate_polars(cache_filename = f"cache/{v.name}.json", alphas = np.linspace(-10, 10, 21))

### Control Surface
# alieron
aileron_cs = asb.ControlSurface(name = "aileron", symmetric = False, hinge_point = 0.75, trailing_edge = True,)

# rudder
rudder_cs = asb.ControlSurface(name = "rudder", symmetric = True, hinge_point = 0.75, trailing_edge = True,)

# elevator
elevator_cs = asb.ControlSurface(name = "elevator", symmetric = True, hinge_point = 0.75, trailing_edge = True,)


##### Vehicle Definition
# Datum (0, 0, 0) is coincident with the quarter-chord-point of the centerline cross section of the main wing

### Nose
# optimise varibale(s)
x_nose = opti.variable(init_guess = -0.1, upper_bound = 1e-3,)

### Wing
# optimise varibale(s)
wing_span               = opti.variable(init_guess = 0.3, lower_bound = 0.1, upper_bound = 10)
wing_dihedral_angle_deg = opti.variable(init_guess = 11, lower_bound = 0, upper_bound = 20)
wing_root_chord         = opti.variable(init_guess = 0.15, lower_bound = 1e-3,)

# defined varibale(s)
wing_taper              = 1.0

# root
def wing_rot(xyz):
    dihedral_rot = np.rotation_matrix_3D(angle = np.radians(wing_dihedral_angle_deg), axis = "X")
    return dihedral_rot @ np.array(xyz)

# chord
def wing_chord(y):
    half_span = wing_span / 2
    tip_chord = wing_taper * wing_root_chord
    spanfrac = np.abs(y) / half_span # 0 at root, 1 at tip
    return (1 - spanfrac) * wing_root_chord + spanfrac * tip_chord

# twist
def wing_twist(y):
    return np.zeros_like(y)

# y-station
wing_ys = np.sinspace(0, wing_span / 2, 11, reverse_spacing = True)

# control surface
N_span        = 11
span_fracs    = onp.linspace(0.0, 0.5, N_span)
wing_xsecs = []
for i, eta in enumerate(span_fracs):
    y = eta * wing_span
    if (eta >= 0.25) and (eta <= 0.45):
        cs_list = [aileron_cs]
    else:
        cs_list = []
    wing_xsecs.append(
        asb.WingXSec(
            xyz_le  = wing_rot([-wing_chord(y), y, 0.0]),
            chord   = wing_chord(y),
            airfoil = airfoils["s3021"],
            twist   = wing_twist(y),
            control_surfaces = cs_list,
        )
    )

# assembly
wing = asb.Wing(name = "Main Wing", symmetric = True, xsecs = wing_xsecs,).translate([0.75 * wing_root_chord, 0, 0])

### Horizontal tailplane
# optimise varibale(s)
l_ht             = opti.variable(init_guess = 0.6, lower_bound = 0.2, upper_bound = 1.5)

# defined varibale(s)
AR_ht            = 4.0
taper_ht         = 1.0
htail_span       = opti.variable(init_guess = 0.15, lower_bound = 1e-3,)
htail_root_chord = 2 * htail_span / (AR_ht * (1 + taper_ht))

# chord
def htail_chord(y):
    half_span_ht     = htail_span / 2
    htail_tip_chord  = taper_ht * htail_root_chord
    spanfrac = np.abs(y) / half_span_ht
    return (1 - spanfrac) * htail_root_chord + spanfrac * htail_tip_chord

# twist
def htail_twist(y):
    return np.zeros_like(y)

# y-station
htail_ys = np.sinspace(0, htail_span / 2, 7, reverse_spacing=True) # y station

# control surface
htail_xsecs = []
for i in range(np.length(htail_ys)):
    y = htail_ys[i]
    htail_xsecs.append(
        asb.WingXSec(
            xyz_le = [l_ht - htail_chord(y), y, 0.0],
            chord  = htail_chord(y),
            twist  = htail_twist(y),
            airfoil = airfoils["naca0008"],
            control_surfaces = [elevator_cs],
        )
    )

# assembly
htail = asb.Wing(name = "HTail", symmetric = True, xsecs = htail_xsecs,)

# volume coefficient
V_ht = htail.area() * l_ht / (wing.area() * wing.mean_aerodynamic_chord())

### Vertical tailplane
# optimise varibale(s)
vtail_span       = opti.variable(init_guess = 0.07, lower_bound = 1e-3,)

# defined varibale(s)
l_vt             = l_ht
AR_vt            = 2.0
taper_vt         = 1.0
vtail_root_chord = 2 * vtail_span / (AR_vt * (1 + taper_vt))

# chord
def vtail_chord(z):
    vtail_tip_chord = taper_vt * vtail_root_chord
    spanfrac = np.abs(z) / vtail_span
    return (1 - spanfrac) * vtail_root_chord + spanfrac * vtail_tip_chord

# twist
def vtail_twist(z):
    return np.zeros_like(z) # no twist

# z-station
vtail_zs = np.sinspace(0, vtail_span, 7, reverse_spacing=True) # z station

# control surface
vtail_xsecs = []
for i in range(np.length(vtail_zs)):
    z = vtail_zs[i]
    vtail_xsecs.append(
        asb.WingXSec(
            xyz_le  = [l_vt - vtail_chord(z), 0.0, z],
            chord   = vtail_chord(z),
            twist   = vtail_twist(z),
            airfoil = airfoils["naca0008"],
            control_surfaces = [rudder_cs],
        )
    )

# assembly
vtail = asb.Wing(name = "VTail", symmetric = False, xsecs = vtail_xsecs,)

# volume coefficient
V_vt = vtail.area() * l_vt / (wing.area() * wing_span)

### Fuselage
# tail length
x_tail = np.maximum(l_ht, l_vt)

# assembly
fuselage = asb.Fuselage(name = "Fuse",
    xsecs = [asb.FuselageXSec(xyz_c = [x_nose, 0.0, 0.0], radius = 4e-3 / 2),
             asb.FuselageXSec(xyz_c = [x_tail, 0.0, 0.0], radius = 4e-3 / 2)
             ]
)

### Overall
# airplane
airplane = asb.Airplane(
    name      = "Nausicaa",
    wings     = [wing, htail, vtail],
    fuselages = [fuselage]
)

# control surface
airplane = airplane.with_control_deflections({
    "aileron" : aileron_deflection,
    "rudder"  : rudder_deflection,
    "elevator": elevator_deflection,
})

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

### Lifting surface
# wing
mass_props['wing'] = lifting_surface_massprops_from_strips(
    wing      = wing,
    density   = density_wing,
    span_axis = "y",
)

# horizontal tailplane
mass_props["htail_surfaces"] = lifting_surface_massprops_from_strips(
    wing      = htail,
    density   = density_wing,
    span_axis = "y",
)

# vertical tailplane
mass_props["vtail_surfaces"] = lifting_surface_massprops_from_strips(
    wing      = vtail,
    density   = density_wing,
    span_axis = "z",
)

### Avionics
# linkages
mass_props["linkages"] = asb.MassProperties(
    mass = 0.001,
    x_cg = x_tail / 2
)

# receiver
mass_props["Receiver"] = asb.mass_properties_from_radius_of_gyration(
    mass = 0.005,
    x_cg = x_nose + 0.010
)

# battery
mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
    mass = 0.009 + 0.004,
    x_cg = x_nose + 0.05
)

# servo
mass_props["servo"] = asb.mass_properties_from_radius_of_gyration(
    mass = 4 * 0.0022,
    x_cg = x_nose + 0.015
)

### Fuselage
# boom
mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
    mass = 0.009 * (x_tail - x_nose),
    x_cg = (x_nose + x_tail) / 2
)

# pod
mass_props["pod"] = asb.MassProperties(
    mass = 0.007,
    x_cg = x_nose + 0.010
)

# ballast
mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
    mass = opti.variable(init_guess = 0, lower_bound = 0,),
    x_cg = opti.variable(init_guess = 0, lower_bound = x_nose, upper_bound = x_tail),
)

### Summation
# assembly
mass_props_TOGW = asb.MassProperties(mass=0)
for k, v in mass_props.items():
    mass_props_TOGW = mass_props_TOGW + v

# glue weight
mass_props['glue_weight'] = mass_props_TOGW * 0.08
mass_props_TOGW += mass_props['glue_weight']

# centre of gravity
x_cg_total, y_cg_total, z_cg_total = mass_props_TOGW.xyz_cg


##### Thermal Vertical Velocity Field Model

### Gaussian plume model
# 2×2 fan thermal model
# partly hard coded for simplicity
def vertical_velocity_field(Q_v, r_th0, k, r, z, z0, fan_spacing):
    # Q_v         - Vertical volume flux (m^3/s)
    # r_th0       - Core radius at z0 (m)
    # k           - Empirical spreading rate
    # z0          - referemce height for r_th0 (m)
    # fan_spacing - spacing between each fan (m)

    # core radius as function of height
    r_th = r_th0 + k * (z - z0)
    r_th = np.maximum(r_th, 1e-6) # avoid negative radius 

    # peak vertical velocity
    w_th = Q_v / (np.pi * r_th ** 2)

    # fan centres
    fan_centers = [
        (-fan_spacing / 2, -fan_spacing / 2),
        ( fan_spacing / 2, -fan_spacing / 2),
        (-fan_spacing / 2,  fan_spacing / 2),
        ( fan_spacing / 2,  fan_spacing / 2),
    ]

    # total w at a single point (x, y) from all four fans
    def w_at_xy(x, y):
        w_total = 0.0
        for (xc, yc) in fan_centers:
            r_i = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            w_i = w_th * np.exp(-(r_i / r_th) ** 2)
            w_i = np.where(z < z0, 0.0, w_i)
            w_total = w_total + w_i
        return w_total
    
    # sample 4 azimuth angles around the orbit
    thetas = np.array([
        np.pi / 4,
        3 * np.pi / 4,
        5 * np.pi / 4,
        7 * np.pi / 4,
    ])

    # total vertical velocity
    w_sum = 0.0
    for theta in thetas:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        w_sum = w_sum + w_at_xy(x, y)
    
    # take average as assumption for glider operation
    w_avg = w_sum / len(thetas)
    return w_avg

### Setup
# CAMAX30 fan parameters
Q_v      = 1.69
x_center = 4.0
y_center = 2.5

# plume parameters
r_th0 = 0.381 # assume core radius equal to fan radius
k     = 0.10  # typical turbulent plume spreading rate
z0    = 0.50  # reference height at fan centre
fan_spacing = 2 * r_th0 + 0.7

# compute average w(r, z)
w = vertical_velocity_field(Q_v = Q_v, r_th0 = r_th0, k = k, r = r_target, z = z_op, z0 = z0, fan_spacing = fan_spacing,)

##### Aerodynamics and Stability

### Aerodynamic force-moment model
ab = asb.AeroBuildup(
    airplane = airplane,
    op_point = op_point,
    xyz_ref  = mass_props_TOGW.xyz_cg
)

### Stability derivatives
aero = ab.run_with_stability_derivatives(alpha = True, beta = True, p = True, q = True, r = True,)

### Performance quantities
LD            = aero["L"] / aero["D"]
power_loss    = aero["D"] * op_point.velocity
sink_rate     = power_loss / 9.81 / mass_props_TOGW.mass
static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
climb_rate    = w - sink_rate

# how much we fail to meet zero climb
climb_shortfall = np.minimum(0, climb_rate)

##### Finalize Optimization Problem
obj_sink    = 0 * sink_rate
obj_climb   = climb_shortfall ** 2 # only penalize the negative climb rate
obj_mass    = 0 * mass_props_TOGW.mass
obj_span    = 0.01 * (wing_span + htail_span + vtail_span)
obj_control = 1e-5 * (elevator_deflection ** 2 + aileron_deflection ** 2 + rudder_deflection ** 2)

### Objective
objective = obj_climb + obj_span + obj_control
penalty   = (mass_props["ballast"].x_cg / 1e3) ** 2

opti.minimize(objective + penalty)

### Optimization constraints
opti.subject_to([

    # coordinated turn
    opti.bounded(5.0, phi,            65.0),        

    # aerodynamics
    aero["L"]   == n_load * mass_props_TOGW.mass * g, # force balance in a coordinate turn

    # stability
    aero["Cm"]  == 0,                                 # trimmed in pitch
    aero["Cl"]  == 0,                                 # trimmed in roll
    aero["Clb"] <= -0.025,
    opti.bounded(0.04, static_margin, 0.10),
    opti.bounded(0.40, V_ht,          0.70),
    opti.bounded(0.02, V_vt,          0.04),
])

### Additional constraint

opti.subject_to([
    LD_turn == LD,
    design_mass_TOGW == mass_props_TOGW.mass
])


##### Solve Optimization Problem

if __name__ == '__main__': # only run this block when the file is executed directly
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
    s = lambda x: sol.value(x)

    ### Turn symbolic airplane into numeric values
    # airplane
    airplane = sol(airplane)

    # lifting surfaces
    wing  = copy.deepcopy(airplane.wings[0])
    htail = copy.deepcopy(airplane.wings[1])
    vtail = copy.deepcopy(airplane.wings[2])

    # fuselage
    fuselage = copy.deepcopy(airplane.fuselages[0])

    # mass proportions
    mass_props = sol(mass_props)

    # performance
    aero       = sol(aero)
    sink_rate  = sol(sink_rate)
    w          = sol(w)
    climb_rate = sol(climb_rate)

    ### Turn symbolic optimized values into numeric values
    # operating point
    op_point = sol(op_point)
    phi      = sol(phi)
    r_target = sol(r_target)
    n_load   = sol(n_load)

    # control surface
    aileron_cs  = sol(aileron_cs)
    rudder_cs   = sol(rudder_cs)
    elevator_cs = sol(elevator_cs)    

    # control surface deflections
    elevator_deflection = sol(elevator_deflection)
    aileron_deflection  = sol(aileron_deflection)
    rudder_deflection   = sol(rudder_deflection)

    # take off gross weight 
    mass_props_TOGW = sol(mass_props_TOGW)

    # effective L/D
    LD_turn = sol(LD_turn)

    # nose and tail
    x_nose = sol(x_nose)
    x_tail = sol(x_tail)

    # wing
    wing_span               = sol(wing_span)
    wing_dihedral_angle_deg = sol(wing_dihedral_angle_deg)
    wing_root_chord         = sol(wing_root_chord)
    wing_taper              = sol(wing_taper)

    # horizontal tailplane
    l_ht             = sol(l_ht)
    htail_span       = sol(htail_span)
    htail_root_chord = sol(htail_root_chord)

    # vertical tailplane
    l_vt             = sol(l_vt)
    vtail_span       = sol(vtail_span)
    vtail_root_chord = sol(vtail_root_chord)

    ### Turn symbolic optimized problem into numeric values
    # objective
    objective = sol(objective)
    penalty   = sol(penalty)

    # make objective components numeric too
    obj_sink    = sol(obj_sink)
    obj_climb   = sol(obj_climb)
    obj_mass    = sol(obj_mass)
    obj_span    = sol(obj_span)
    obj_control = sol(obj_control)

    # constraints
    static_margin = sol(static_margin)
    V_ht          = sol(V_ht)
    V_vt          = sol(V_vt)

    ##### Result

    ### Help fomatting
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    print_title = lambda s: print(s.upper().join(["*" * 20] * 2))

    def fmt(x):
        return f"{s(x):.6g}"
    
    ### Optimisation problem summary
    # breakdown
    print_title("Objective contribution breakdown")
    for k, v in {
    "Total Objective" : fmt(objective),
    "Sink rate"       : f"{fmt(obj_sink)} ({s(obj_sink) / s(objective) * 100:.1f}%)",
    "Climb rate"      : f"{fmt(obj_climb)} ({s(obj_climb) / s(objective) * 100:.1f}%)",
    "Weight"          : f"{fmt(obj_mass)} ({s(obj_mass) / s(objective) * 100:.1f}%)",
    "Span"            : f"{fmt(obj_span)} ({s(obj_span) / s(objective) * 100:.1f}%)",
    "Control effect"  : f"{fmt(obj_control)} ({s(obj_control) / s(objective) * 100:.1f}%)",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    # boundary
    print_title("Design variable")

    def check_var(name, var_value, lb=None, ub=None, atol=1e-6, rtol=1e-3):
        v = float(s(var_value))
        hits = []
        if lb is not None:
            if v <= lb + max(atol, rtol * max(1.0, abs(lb))):
                hits.append("LOWER")
        if ub is not None:
            if v >= ub - max(atol, rtol * max(1.0, abs(ub))):
                hits.append("UPPER")
        status = " ,".join(hits) if hits else "OK"
        print(f"{name:25s} = {v: .6g}  [{status}]")

    check_var("r_target (m)",      r_target,                   lb = 0.1,    ub = 2.5)

    check_var("V (m/s)",           op_point.velocity,          lb = 0.1,    ub = 15.0)
    check_var("alpha (deg)",       op_point.alpha,             lb = -10.0,  ub = 10.0)
    check_var("phi (deg)",         phi,                        lb = 5.0,    ub = 65.0)

    check_var("TOGW (kg)",         design_mass_TOGW,           lb = 1e-3)

    check_var("x_nose (m)",        x_nose,                                  ub = 1e-3)

    check_var("wing_span",         wing_span,                  lb = 0.1,    ub = 10.0)
    check_var("wing_dihedral_deg", wing_dihedral_angle_deg,    lb = 0.0,    ub = 20.0)
    check_var("wing_root_chord",   wing_root_chord,            lb = 1e-3)

    check_var("l_ht",              l_ht,                       lb = 0.2,    ub = 1.5)
    check_var("htail_span",        htail_span,                 lb = 1e-3,)
    check_var("vtail_span",        vtail_span,                 lb = 1e-3,)

    check_var("ballast_mass",      mass_props["ballast"].mass, lb = 0.0)
    check_var("ballast_x_cg",      mass_props["ballast"].x_cg, lb = x_nose, ub = x_tail)

    check_var("elev_defl (deg)",   elevator_deflection,        lb = -25.0,  ub = 25.0)
    check_var("ail_defl (deg)",    aileron_deflection,         lb = -30.0,  ub = 30.0)
    check_var("rud_defl (deg)",    rudder_deflection,          lb = -25.0,  ub = 25.0)

    check_var("lift (N)",          aero["L"],                  lb = n_load * mass_props_TOGW.mass * 9.81)
    check_var("C_m",               aero["Cm"],)
    check_var("C_l",               aero["Cl"],)
    check_var("C_l_b",             aero["Clb"],                             ub = -0.025)
    check_var("static_margin",     static_margin,              lb = 0.04,   ub = 0.10)
    check_var("V_ht",              V_ht,                       lb = 0.40,   ub = 0.70)
    check_var("V_vt",              V_vt,                       lb = 0.02,   ub = 0.04)
    
    ### Output summary
    print_title("Outputs")
    for k, v in {
        "mass_TOGW"             : f"{fmt(mass_props_TOGW.mass)} kg ({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "L/D (turn)"            : fmt(LD_turn),
        "Bank Angle"            : f"{fmt(phi)} deg",
        "Load Factor"           : fmt(n_load),
        "Turn Airspeed"         : f"{fmt(op_point.velocity)} m/s",
        "Turn AoA"              : f"{fmt(op_point.alpha)} deg",
        "Turn CL"               : fmt(aero['CL']),
        "Sink Rate"             : f"{fmt(sink_rate)} m/s",
        "w"                     : f"{fmt(w)} m/s",
        "Climb Rate"            : f"{fmt(climb_rate)} m/s",
        "Cma"                   : fmt(aero['Cma']),
        "Cnb"                   : fmt(aero['Cnb']),
        "Cm"                    : fmt(aero['Cm']),
        "I_xx"                  : f"{fmt(mass_props_TOGW.inertia_tensor[0, 0])} kg/m^2",
        "Wing Reynolds Number"  : eng_string(op_point.reynolds(sol(wing.mean_aerodynamic_chord()))),
        "CG location"           : "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
        "Wing Span"             : f"{fmt(wing_span)} m ({fmt(wing_span / u.foot)} ft)",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    ### Mass breakdown
    fmtpow = lambda x: fmt(x) + " W"

    print_title("Mass props")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {v.mass * 1e3:.2f} g ({v.mass / u.oz:.2f} oz)")

    ##### Plotting
    if make_plots:
        # geometry
        airplane.draw_three_view(show = False)
        p.show_plot(tight_layout= False, savefig = "figures/three_view.png")

        # mass budget
        fig, ax = plt.subplots(figsize = (12, 5), subplot_kw = dict(aspect = "equal"), dpi = 300)

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
            values = [
                v.mass
                for v in mass_props_to_plot.values()
            ],
            names = [
                n if n not in name_remaps.keys() else name_remaps[n]
                for n in mass_props_to_plot.keys()
            ],
            center_text = f"$\\bf{{Mass\\ Budget}}$\nTOGW: {s(mass_props_TOGW.mass * 1e3):.2f} g",
            label_format = lambda name, value, percentage: f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%",
            startangle = 110,
            arm_length = 30,
            arm_radius = 20,
            y_max_labels = 1.1
        )
        p.show_plot(savefig="figures/mass_budget.png")


    ###### Save results to file

    ### Settings
    import os

    # ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    ### Helper to safely convert scalars
    def to_scalar(x):

        try:
            # try numpy conversion or flattening if needed
            import numpy as _np
            arr = _np.array(x)
            if arr.shape == ():
                return float(arr)
            # if it's a small array, just return first element
            return float(arr.flatten()[0])
        except Exception:
            # last resort
            try:
                return float(x)
            except Exception:
                return x  # leave as-is

    ### Design results

    design_results = {

        # operating point
        "V_operate (m/s)"    : to_scalar(op_point.velocity),
        "alpha (deg)"        : to_scalar(op_point.alpha),
        "phi (deg)"          : to_scalar(phi),
        "r_target (m)"       : to_scalar(r_target),
        "n_load"             : to_scalar(n_load),

        # global mass and performance
        "TOGW (kg)"          : to_scalar(mass_props_TOGW.mass),
        "L/D_turn"           : to_scalar(LD_turn),
        "sink_rate (m/s)"    : to_scalar(sink_rate),
        "w (m/s)"            : to_scalar(w),
        "climb_rate (m/s)"   : to_scalar(climb_rate),
        "static_margin"      : to_scalar(static_margin),
        "V_ht"               : to_scalar(V_ht),
        "V_vt"               : to_scalar(V_vt),
        "Re_w"               : to_scalar(op_point.reynolds(sol(wing.mean_aerodynamic_chord()))),

        # geometry
        "x_nose (m)"         : to_scalar(x_nose),
        "x_tail (m)"         : to_scalar(x_tail),

        "b_w (m)"            : to_scalar(wing_span),
        "dihedral_w (deg)"   : to_scalar(wing_dihedral_angle_deg),
        "c_root_w (m)"       : to_scalar(wing_root_chord),
        "taper_w"            : to_scalar(wing_taper),
        "S_w (m^2)"          : to_scalar(wing.area()),

        "l_ht (m)"           : to_scalar(l_ht),
        "b_ht (m)"           : to_scalar(htail_span),
        "c_root_ht (m)"      : to_scalar(htail_root_chord),
        "taper_ht"           : to_scalar(taper_ht),
        "S_ht (m^2)"         : to_scalar(htail.area()),
        
        "l_vt (m)"           : to_scalar(l_vt),
        "b_vt (m)"           : to_scalar(vtail_span),
        "c_root_vt (m)"      : to_scalar(vtail_root_chord),
        "taper_vt"           : to_scalar(taper_vt),
        "S_vt (m^2)"         : to_scalar(vtail.area()),

        # control surface
        "hinge_point_a"      : to_scalar(aileron_cs.hinge_point),
        "hinge_point_r"      : to_scalar(rudder_cs.hinge_point),
        "hinge_point_e"      : to_scalar(elevator_cs.hinge_point),       

        # control surface deflections
        "delta_a (deg)"      : to_scalar(aileron_deflection),
        "delta_r (deg)"      : to_scalar(rudder_deflection),
        "delta_e (deg)"      : to_scalar(elevator_deflection),

        # objective decomposition
        "objective_total"    : to_scalar(objective),
        "objective_sink"     : to_scalar(obj_sink),
        "objective_climb"    : to_scalar(obj_climb),
        "objective_mass"     : to_scalar(obj_mass),
        "objective_span"     : to_scalar(obj_span),
        "objective_control"  : to_scalar(obj_control),
        "penalty"            : to_scalar(penalty),

        # CG location
        "x_cg (m)"           : to_scalar(mass_props_TOGW.xyz_cg[0]),
        "y_cg (m)"           : to_scalar(mass_props_TOGW.xyz_cg[1]),
        "z_cg (m)"           : to_scalar(mass_props_TOGW.xyz_cg[2]),

        # Inertia tensor (about CG)
        "I_xx (kg m^2)"      : to_scalar(mass_props_TOGW.inertia_tensor[0, 0]),
        "I_yy (kg m^2)"      : to_scalar(mass_props_TOGW.inertia_tensor[1, 1]),
        "I_zz (kg m^2)"      : to_scalar(mass_props_TOGW.inertia_tensor[2, 2]),
        "I_xy (kg m^2)"      : to_scalar(mass_props_TOGW.inertia_tensor[0, 1]),
        "I_xz (kg m^2)"      : to_scalar(mass_props_TOGW.inertia_tensor[0, 2]),
        "I_yz (kg m^2)"      : to_scalar(mass_props_TOGW.inertia_tensor[1, 2]),
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
    wing_tip_LE  = LE_coords(wing.xsecs[-1])

    design_results["wing root LE x (m)"] = wing_root_LE[0]
    design_results["wing root LE y (m)"] = wing_root_LE[1]
    design_results["wing root LE z (m)"] = wing_root_LE[2]

    design_results["wing tip LE x (m)"]  = wing_tip_LE[0]
    design_results["wing tip LE y (m)"]  = wing_tip_LE[1]
    design_results["wing tip LE z (m)"]  = wing_tip_LE[2]

    # horizontal tail root and tip LE
    ht_root_LE = LE_coords(htail.xsecs[0])
    ht_tip_LE  = LE_coords(htail.xsecs[-1])

    design_results["htail root LE x (m)"] = ht_root_LE[0]
    design_results["htail root LE y (m)"] = ht_root_LE[1]
    design_results["htail root LE z (m)"] = ht_root_LE[2]

    design_results["htail tip LE x (m)"]  = ht_tip_LE[0]
    design_results["htail tip LE y (m)"]  = ht_tip_LE[1]
    design_results["htail tip LE z (m)"]  = ht_tip_LE[2]

    # vertical tail root and tip LE
    vt_root_LE = LE_coords(vtail.xsecs[0])
    vt_tip_LE  = LE_coords(vtail.xsecs[-1])

    design_results["vtail root LE x (m)"] = vt_root_LE[0]
    design_results["vtail root LE y (m)"] = vt_root_LE[1]
    design_results["vtail root LE z (m)"] = vt_root_LE[2]

    design_results["vtail tip LE x (m)"]  = vt_tip_LE[0]
    design_results["vtail tip LE y (m)"]  = vt_tip_LE[1]
    design_results["vtail tip LE z (m)"]  = vt_tip_LE[2]

    ### All aerodynamic and stability
    aero_results = {}

    def extract_component_list(name, component_list, out_dict):

        for i, comp in enumerate(component_list):
            prefix = f"{name}_comp{i+1}"

            out_dict[f"{prefix}_L (N)"]        = to_scalar(comp.L)
            out_dict[f"{prefix}_D (N)"]        = to_scalar(comp.D)
            out_dict[f"{prefix}_Y (N)"]        = to_scalar(comp.Y)

            out_dict[f"{prefix}_l_b"]          = to_scalar(comp.l_b)
            out_dict[f"{prefix}_m_b"]          = to_scalar(comp.m_b)
            out_dict[f"{prefix}_n_b"]          = to_scalar(comp.n_b)

            out_dict[f"{prefix}_span_eff (m)"] = to_scalar(comp.span_effective)
            out_dict[f"{prefix}_oswald"]       = to_scalar(comp.oswalds_efficiency)

    # loop through everything in aero
    for key, value in aero.items():

        # normal numeric aero outputs
        if not isinstance(value, list):
            aero_results[f"aero_{key}"] = to_scalar(value)

        # value is a list of AeroComponentResults
        else:
            if len(value) > 0 and hasattr(value[0], "L") and hasattr(value[0], "span_effective"):
                component_name = (
                    key.replace("aero_", "")
                    .replace("_aero_components", "")
                )
                extract_component_list(component_name, value, aero_results)
            else:
                for i, item in enumerate(value):
                    aero_results[f"aero_{key}[{i}]"] = to_scalar(item)

    ### Combine everything into one flat dict
    all_results = {}
    all_results.update(design_results)
    all_results.update(aero_results)

    ### Write to .csv and Excel
    results_df = pd.DataFrame(
        list(all_results.items()),
        columns=["Variable", "Value"]
        )

    csv_path   = os.path.join(results_dir, "nausicaa_results.csv")
    excel_path = os.path.join(results_dir, "nausicaa_results.xlsx")

    results_df.to_csv(csv_path, index=False)
    results_df.to_excel(excel_path, index=False)

    print(f"\nSaved results to:\n  {csv_path}\n  {excel_path}")