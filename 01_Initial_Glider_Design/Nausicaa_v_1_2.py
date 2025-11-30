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


##### Overall Specs

### Operating point
op_point = asb.OperatingPoint(
    velocity = opti.variable(init_guess = 14, lower_bound = 1, log_transform = True),
    alpha = opti.variable(init_guess = 0, lower_bound = -10, upper_bound = 10)
)

### Take off gross weight 
design_mass_TOGW = opti.variable(init_guess = 0.1, lower_bound = 1e-3)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3) # numerical clamp

### Cruise L/D
LD_cruise = opti.variable(init_guess = 15, lower_bound = 0.1, log_transform = True)

### Gravitational acceleration
g = 9.81

##### Vehicle Definition
# Datum (0, 0, 0) is coincident with the quarter-chord-point of the centerline cross section of the main wing

### Nose
x_nose = opti.variable(init_guess = -0.1, upper_bound = 1e-3,)

### Wing
wing_span = opti.variable(init_guess = 0.5, lower_bound = 0.3, upper_bound = 0.7)
wing_dihedral_angle_deg = opti.variable(init_guess = 11, lower_bound = 0, upper_bound = 20)
wing_root_chord = opti.variable(init_guess = 0.15, lower_bound = 1e-3,)
wing_taper = opti.variable(init_guess = 0.5, lower_bound = 0.3, upper_bound = 1)

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

### Horizontal Tailplane
AR_ht = 5.0
taper_ht = 0.7
l_ht = opti.variable(init_guess = 0.6, lower_bound = 0.2, upper_bound = 1.2)
S_ht = 0.12 * wing.area()
# S_ht = opti.variable(init_guess = 0.02, lower_bound = 1e-3,)

b_ht = np.sqrt(AR_ht * S_ht)
half_span_ht = b_ht / 2

def htail_chord(y):

    spanfrac = np.abs(y) / half_span_ht
    c_root_ht = 2 * S_ht / (b_ht * (1 + taper_ht))
    c_tip_ht  = taper_ht * c_root_ht

    return (1 - spanfrac) * c_root_ht + spanfrac * c_tip_ht

def htail_twist(y):

    return np.zeros_like(y) # no twist

htail_ys = np.sinspace(0, half_span_ht, 7, reverse_spacing=True) # y station

htail = asb.Wing(name = "HTail", symmetric = True,
    xsecs = [asb.WingXSec(xyz_le = [l_ht - htail_chord(htail_ys[i]), htail_ys[i], 0.0],
                          chord = htail_chord(htail_ys[i]),
                          twist = htail_twist(htail_ys[i]),
                          airfoil = airfoils["naca0008"],
                          )
                          for i in range(np.length(htail_ys))
             ]
)

V_ht = htail.area() * l_ht / (wing.area() * wing.mean_aerodynamic_chord())

### Vertical Tailplane
AR_vt = 2.0
taper_vt = 0.6
# l_vt = l_ht
l_vt = opti.variable(init_guess = 0.6, lower_bound = 0.2, upper_bound = 1.2)
S_vt = 0.06 * wing.area()
# S_vt = opti.variable(init_guess = 0.01, lower_bound = 1e-3,)

b_vt = np.sqrt(AR_vt * S_vt)

def vtail_chord(z):

    spanfrac = np.abs(z) / b_vt
    c_root_vt = 2 * S_vt / (b_vt * (1 + taper_vt))
    c_tip_vt  = taper_vt * c_root_vt
    return (1 - spanfrac) * c_root_vt + spanfrac * c_tip_vt

def vtail_twist(z):

    return np.zeros_like(z) # no twist

vtail_zs = np.sinspace(0, b_vt, 7, reverse_spacing=True) # z station

vtail = asb.Wing(name = "VTail", symmetric = False,
    xsecs = [asb.WingXSec(xyz_le = [l_vt - vtail_chord(vtail_zs[i]), 0.0, vtail_zs[i], ],
                          chord = vtail_chord(vtail_zs[i]),
                          twist = vtail_twist(vtail_zs[i]),
                          airfoil = airfoils["naca0008"],
                          )
                          for i in range(np.length(vtail_zs))
             ]
)

V_vt = vtail.area() * l_vt / (wing.area() * wing_span)

### Fuselage
x_tail = np.maximum(l_ht, l_vt)

fuselage = asb.Fuselage(name = "Fuse",
    xsecs = [asb.FuselageXSec(xyz_c = [x_nose, 0.0, 0.0], radius = 4e-3 / 2),
             asb.FuselageXSec(xyz_c=[x_tail, 0.0, 0.0], radius = 4e-3 / 2)
             ]
)

### Overall
airplane = asb.Airplane(
    name="Nausicaa",
    wings=[wing, htail, vtail],
    fuselages=[fuselage]
)


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
x_cg_wing, z_cg_wing = lifting_surface_planform_cg(wing, span_axis = "y")

mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
    mass = wing.volume() * density_wing,
    x_cg = x_cg_wing,
    z_cg = z_cg_wing,
    )

### Horizontal tailplane
x_cg_ht, z_cg_ht = lifting_surface_planform_cg(htail, span_axis = "y")

mass_props["htail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass = htail.volume() * density_wing,
    x_cg = x_cg_ht,
    z_cg = z_cg_ht,
)

### Vertical tailplane
x_cg_vt, z_cg_vt = lifting_surface_planform_cg(vtail, span_axis = "z")

mass_props["vtail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass = vtail.volume() * density_wing,
    x_cg = x_cg_vt,
    z_cg = z_cg_vt,
)

### Linkages
mass_props["linkages"] = asb.MassProperties(
    mass = 0.001,
    x_cg = x_tail / 2
)

### Avionics
mass_props["Receiver"] = asb.mass_properties_from_radius_of_gyration(
    mass = 0.005,
    x_cg = x_nose + 0.010
)

mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
    mass = 0.009 + 0.004,
    x_cg = x_nose + 0.05
)

mass_props["servo"] = asb.mass_properties_from_radius_of_gyration(
    mass = 4 * 0.0022,
    x_cg = x_nose + 0.015
)

### Boom
mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
    mass = 0.009 * (x_tail - x_nose),
    x_cg = (x_nose + x_tail) / 2
)

### Pod
mass_props["pod"] = asb.MassProperties(
    mass = 0.007,
    x_cg = (x_nose + 0.75 * wing_root_chord) / 2
)

### Ballast
mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
    mass = opti.variable(init_guess = 0, lower_bound = 0,),
    x_cg = opti.variable(init_guess = 0, lower_bound = x_nose, upper_bound = x_tail),
)

### Summation
mass_props_TOGW = asb.MassProperties(mass=0)
for k, v in mass_props.items():
    mass_props_TOGW = mass_props_TOGW + v

### Glue weight
mass_props['glue_weight'] = mass_props_TOGW * 0.08
mass_props_TOGW += mass_props['glue_weight']

### Centre of gravity
x_cg_total, y_cg_total, z_cg_total = mass_props_TOGW.xyz_cg

### Moment of inertia
J_cg = mass_props_TOGW.inertia_tensor
I_xx = J_cg[0, 0]


##### Aerodynamics and Stability

### Aerodynamic force-moment model
ab = asb.AeroBuildup(airplane = airplane, op_point = op_point, xyz_ref = mass_props_TOGW.xyz_cg)

### Stability derivatives
aero = ab.run_with_stability_derivatives(alpha = True, beta = True, p = True, q = True, r = True,)

### Performance quantities
LD = aero["L"] / aero["D"]
power_loss = aero["D"] * op_point.velocity
sink_rate = power_loss / 9.81 / mass_props_TOGW.mass
static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()


##### Finalize Optimization Problem

### Objective
obj_sink = 0.7 * sink_rate
obj_mass = 2.0 * mass_props_TOGW.mass
obj_inertia = 140.0 * I_xx
obj_wingload = 0.02 * (mass_props_TOGW.mass * g / wing.area())

objective = obj_sink + obj_mass + obj_inertia + obj_wingload
penalty = (mass_props["ballast"].x_cg / 1e3) ** 2

opti.minimize(objective + penalty)

### Optimization constraints
opti.subject_to([
    # aerodynamics
    aero["L"] >= 9.81 * mass_props_TOGW.mass, # lift >= weight
    # stability
    aero["Cm"] == 0,                          # trimmed flight
    aero["Clb"] <= -0.025,
    static_margin >= 0.04,
    static_margin <= 0.10,
    V_ht >= 0.40,
    V_ht <= 0.70,
    V_vt >= 0.02,
    V_vt <= 0.04,
    # material
    x_tail - x_nose < 0.8                     # maximum carbon fibre tube length
])

### Additional constraint
opti.subject_to([
    LD_cruise == LD,
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
    airplane = sol(airplane)
    op_point = sol(op_point)
    mass_props = sol(mass_props)
    mass_props_TOGW = sol(mass_props_TOGW)
    aero = sol(aero)
    
    wing = copy.deepcopy(airplane.wings[0])
    htail = copy.deepcopy(airplane.wings[1])
    vtail = copy.deepcopy(airplane.wings[2])
    fuselage = copy.deepcopy(airplane.fuselages[0])

    ### Make a simplified wing for Athena Vortex Lattice
    wing_lowres = copy.deepcopy(wing)
    xsecs_to_keep = np.arange(len(wing.xsecs)) % 2 == 0
    xsecs_to_keep[0] = True
    xsecs_to_keep[-1] = True
    wing_lowres.xsecs = np.array(wing_lowres.xsecs)[xsecs_to_keep]

    airplane_avl = asb.Airplane(
                wings = [wing_lowres, htail, vtail,],
                fuselages = [fuselage]
            )
    airplane_avl = copy.deepcopy(airplane_avl)
    
    ### Run Athena Vortex Lattice to cross-check stability derivatives
    try:
        avl_aero = asb.AVL(
            airplane = airplane_avl,
            op_point = op_point,
            xyz_ref = mass_props_TOGW.xyz_cg,
            working_directory = "avl_debug"
        ).run()
    except FileNotFoundError:
        class EmptyDict:
            def __getitem__(self, item):
                return "Install AVL to see this."

        avl_aero = EmptyDict()


    ##### Result

    ### Help fomatting
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    print_title = lambda s: print(s.upper().join(["*" * 20] * 2))

    def fmt(x):
        return f"{s(x):.6g}"
    
    ### Output summary
    print_title("Outputs")
    for k, v in {
        "mass_TOGW"             : f"{fmt(mass_props_TOGW.mass)} kg ({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "L/D (actual)"          : fmt(LD_cruise),
        "Cruise Airspeed"       : f"{fmt(op_point.velocity)} m/s",
        "Cruise AoA"            : f"{fmt(op_point.alpha)} deg",
        "Cruise CL"             : fmt(aero['CL']),
        "Sink Rate"             : f"{fmt(sink_rate)} m/s",
        "Cma"                   : fmt(aero['Cma']),
        "Cnb"                   : fmt(aero['Cnb']),
        "Cm"                    : fmt(aero['Cm']),
        "Wing Reynolds Number"  : eng_string(op_point.reynolds(sol(wing.mean_aerodynamic_chord()))),
        "AVL: Cma"              : avl_aero['Cma'],
        "AVL: Cnb"              : avl_aero['Cnb'],
        "AVL: Cm"               : avl_aero['Cm'],
        "AVL: Clb Cnr / Clr Cnb": avl_aero['Clb Cnr / Clr Cnb'],
        "CG location"           : "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
        "Wing Span"             : f"{fmt(wing_span)} m ({fmt(wing_span / u.foot)} ft)",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    ### Mass breakdown
    fmtpow = lambda x: fmt(x) + " W"

    print_title("Mass props")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {v.mass * 1e3:.2f} g ({v.mass / u.oz:.2f} oz)")

    ### Plotting
    if make_plots:
        # geometry
        airplane.draw_three_view(show=False)
        p.show_plot(tight_layout=False, savefig="figures/three_view.png")

        # mass budget
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw=dict(aspect="equal"), dpi=300)

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
            values=[
                v.mass
                for v in mass_props_to_plot.values()
            ],
            names=[
                n if n not in name_remaps.keys() else name_remaps[n]
                for n in mass_props_to_plot.keys()
            ],
            center_text=f"$\\bf{{Mass\\ Budget}}$\nTOGW: {s(mass_props_TOGW.mass * 1e3):.2f} g",
            label_format=lambda name, value, percentage: f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%",
            startangle=110,
            arm_length=30,
            arm_radius=20,
            y_max_labels=1.1
        )
        p.show_plot(savefig="figures/mass_budget.png")