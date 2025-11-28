import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy

opti = asb.Opti(
    # variable_categories_to_freeze = "all",
    # freeze_style = "float"
)

make_plots=True

##### Section: Parameters

wing_method = 'foam'

airfoils = {
    "flat_plate": asb.Airfoil(
        name="flat_plate",
        coordinates=np.array([
            [1.0,  0.0],  # TE (start)
            [0.99999,  1e-6], # upper surface
            [0.9,  1e-6],
            [0.8,  1e-6],
            [0.7,  1e-6],
            [0.6,  1e-6],
            [0.5,  1e-6],
            [0.4,  1e-6],
            [0.3,  1e-6],
            [0.2,  1e-6], 
            [0.1,  1e-6], 
            [0.00001, 1e-6], # upper surface
            [0.0,  0.0],  # LE
            [0.00001, -1e-6], # lower surface
            [0.1, -1e-6],
            [0.2, -1e-6],
            [0.3, -1e-6],
            [0.4, -1e-6],
            [0.5, -1e-6],
            [0.6, -1e-6],
            [0.7, -1e-6],
            [0.8, -1e-6],
            [0.9, -1e-6],
            [0.99999, -1e-6],  # lower surface
            [1.0,  0.0],  # TE (end, same as start)
        ])
    )
}

# for v in airfoils.values():
#    v.generate_polars(
#        cache_filename=f"cache/{v.name}.json",
#         alphas=np.linspace(-10, 10, 21)
#     )

##### Section: Vehicle Overall Specs

op_point = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=14,
        lower_bound=1,
        log_transform=True
    ),
    alpha=opti.variable(
        init_guess=0,
        lower_bound=-10,
        upper_bound=10
    )
)

design_mass_TOGW = opti.variable(
    init_guess=0.1,
    lower_bound=1e-3
)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)

LD_cruise = opti.variable(
    init_guess=15,
    lower_bound=0.1,
    log_transform=True
)

g = 9.81

##### Section: Vehicle Definition

"""
Coordinate system:

Geometry axes. Datum (0, 0, 0) is coincident with the quarter-chord-point of the centerline cross section of the main 
wing.

"""

### Define x-stations
x_nose = opti.variable(
    init_guess=-0.1,
    upper_bound=1e-3,
)

### Wing

wing_span = opti.variable(
    init_guess=0.5,
    lower_bound=0.3,
    upper_bound=0.7
)

wing_dihedral_angle_deg = opti.variable(
    init_guess=11,
    lower_bound=0,
    upper_bound=20
)

wing_root_chord = opti.variable(
    init_guess=0.15,
    lower_bound=1e-3,
)

wing_taper = opti.variable(
    init_guess=0.5,
    lower_bound=0.3,
    upper_bound=1)


def wing_rot(xyz):
    dihedral_rot = np.rotation_matrix_3D(
        angle=np.radians(wing_dihedral_angle_deg),
        axis="X"
    )

    return dihedral_rot @ np.array(xyz)


def wing_chord(y):
    
    half_span = wing_span / 2
    tip_chord = wing_taper * wing_root_chord
    spanfrac = np.abs(y) / half_span   # 0 at root, 1 at tip

    return (1 - spanfrac) * wing_root_chord + spanfrac * tip_chord


def wing_twist(y):
    return np.zeros_like(y)


wing_ys = np.sinspace(
    0,
    wing_span / 2,
    11,
    reverse_spacing=True
)

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=wing_rot([
                -wing_chord(wing_ys[i]),
                wing_ys[i],
                0
            ]),
            chord=wing_chord(wing_ys[i]),
            airfoil=airfoils["flat_plate"],
            twist=wing_twist(wing_ys[i]),
        )
        for i in range(np.length(wing_ys))
    ]
).translate([
    0.75 * wing_root_chord,
    0,
    0
])

## Horizontal Tailplane

AR_ht = 5.0
taper_ht = 0.7

S_ht = 0.12 * wing.area()
# S_ht = opti.variable(
   #  init_guess=0.02,
   #  lower_bound=1e-3
# )

l_ht = opti.variable(
    init_guess=0.6,
    lower_bound=0.2,
    upper_bound=1.2
)

b_ht = np.sqrt(AR_ht * S_ht)
half_span_ht = b_ht / 2

c_root_ht = 2 * S_ht / (b_ht * (1 + taper_ht))
c_tip_ht  = taper_ht * c_root_ht

def htail_chord(y):
    spanfrac = np.abs(y) / half_span_ht
    return (1 - spanfrac) * c_root_ht + spanfrac * c_tip_ht

def htail_twist(y):
    return np.zeros_like(y) 

htail_ys = np.sinspace(
    0,
    half_span_ht,
    7,
    reverse_spacing=True
)

htail = asb.Wing(
    name="HTail",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le = [
                l_ht - htail_chord(htail_ys[i]),   # TE at x = l_ht
                htail_ys[i],
                0.0
            ],
            chord=htail_chord(htail_ys[i]),
            twist=htail_twist(htail_ys[i]),
            airfoil=airfoils["flat_plate"],
        )
        for i in range(np.length(htail_ys))
    ]
)

## Vertical Tailplane

AR_vt = 2.0
taper_vt = 0.6

l_vt = l_ht

S_vt = 0.06 * wing.area()
# S_vt = opti.variable(
    # init_guess=0.01,
    # lower_bound=1e-3
# )

b_vt = np.sqrt(AR_vt * S_vt)

c_root_vt = 2 * S_vt / (b_vt * (1 + taper_vt))
c_tip_vt  = taper_vt * c_root_vt


def vtail_chord(z):
    spanfrac = np.abs(z) / b_vt
    return (1 - spanfrac) * c_root_vt + spanfrac * c_tip_vt

def vtail_twist(z):
    return np.zeros_like(z)

vtail_zs = np.sinspace(
    0,
    b_vt,
    7,
    reverse_spacing=True
)


vtail = asb.Wing(
    name="VTail",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                l_vt - vtail_chord(vtail_zs[i]),   # TE at x = l_vt
                0.0,
                vtail_zs[i],
            ],
            chord=vtail_chord(vtail_zs[i]),
            twist=vtail_twist(vtail_zs[i]),
            airfoil=airfoils["flat_plate"],
        )
        for i in range(np.length(vtail_zs))
    ]
)

## Fuselage

x_tail = np.maximum(l_ht, l_vt)

fuselage = asb.Fuselage(
    name="Fuse",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[x_nose, 0, 0],
            radius=4e-3 / 2
        ),
        asb.FuselageXSec(
            xyz_c=[x_tail, 0, 0],
            radius=4e-3 / 2
        )
    ]
)

## Overall

airplane = asb.Airplane(
    name="Glider",
    wings=[
        wing,
        htail,
        vtail
    ],
    fuselages=[fuselage]
)

##### Section: Internal Geometry and Weights

mass_props = {}

### Lifting bodies
if wing_method == '3d printing':
    # mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
    #     mass=(12e-3 / (0.200 * 0.150)) * wing.area(),
    #     x_cg=
    # )
    raise ValueError

elif wing_method == 'foam':
    # density = 20.8 # kg/m^3, for Foamular 150
    # density = 38.06 # kg/m^3, for hard blue foam
    # density = 4 * u.lbm / u.foot ** 3
    density = 33.0 # kg/m^3 for depron foam
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.area() * 0.003 * density,
        x_cg=(0.500 - 0.25) * wing_root_chord,
        z_cg=(0.03591) * (
                np.sind(wing_dihedral_angle_deg) / np.sind(11)
        ) * (
                     wing_span / 1
             ),
    )
elif wing_method == 'elf':
    # wing_mass = asb.MassProperties(
    #     mass=0.0506 * wing_span ** 2.09,
    #     x_cg=0.5 * root_chord
    # )
    raise ValueError

density = 33.0 # kg/m^3 for depron foam

mass_props["htail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass=htail.area() * 0.003 * density,
    x_cg=htail.xsecs[0].xyz_le[0] + 0.50 * htail.xsecs[0].chord,
)

mass_props["vtail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass=vtail.area() * 0.003 * density,
    x_cg=vtail.xsecs[0].xyz_le[0] + 0.50 * vtail.xsecs[0].chord,
)

mass_props["linkages"] = asb.MassProperties(
    mass=1e-3,
    x_cg=x_tail / 2
)

### Pod and avionics

mass_props["wing_electronics"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.008,
    x_cg=wing.xsecs[0].xyz_le[0] + 0.25 * wing.xsecs[0].chord,
)  # Includes 2 linear servos

mass_props["htail_electronics"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.004,
    x_cg=htail.xsecs[0].xyz_le[0] + 0.25 * htail.xsecs[0].chord,
)  # Includes 1 linear servos

mass_props["vtail_electronics"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.004,
    x_cg=htail.xsecs[0].xyz_le[0] + 0.25 * vtail.xsecs[0].chord,
)  # Includes 1 linear servos

mass_props["Receiver_UBEC"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.0015,
    x_cg=x_nose + 0.05
)  # Includes RC Receiver, UBEC

mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.008,
    x_cg=x_nose + 0.02
)

mass_props["pod"] = asb.MassProperties(
    mass=7e-3,
    x_cg=x_nose + 0.07
    # x_cg=(x_nose + 0.75 * wing_root_chord) / 2
)

mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
    mass=opti.variable(init_guess=0, lower_bound=0),
    x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail),
)

### Boom
mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
    mass=7.0e-3 * ((x_tail - x_nose) / 826e-3),
    x_cg=(x_nose + x_tail) / 2
)

### Summation
mass_props_TOGW = asb.MassProperties(mass=0)
for k, v in mass_props.items():
    mass_props_TOGW = mass_props_TOGW + v

### Add glue weight
mass_props['glue_weight'] = mass_props_TOGW * 0.08
mass_props_TOGW += mass_props['glue_weight']

##### Section: Aerodynamics

ab = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point,
    xyz_ref=mass_props_TOGW.xyz_cg
)
aero = ab.run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=False,
    q=False,
    r=False,
)

opti.subject_to([
    aero["L"] >= 9.81 * mass_props_TOGW.mass,
    aero["Cm"] == 0,
])

LD = aero["L"] / aero["D"]
power_loss = aero["D"] * op_point.velocity
sink_rate = power_loss / 9.81 / mass_props_TOGW.mass

##### Section: Stability
static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()

opti.subject_to(
    static_margin >= 0.04
)
opti.subject_to(
    static_margin <= 0.10
)

##### Section: Finalize Optimization Problem

### Moment of inertia
m_wing = mass_props["wing"].mass
I_xx = (1/12) * m_wing * wing_span**2  # simple roll inertia model

### Turn Radius
wing_loading = mass_props_TOGW.mass * g / wing.area()
# partly redundant with the sink-rate term

### Objective contributions
obj_sink        = 0.7 * sink_rate
obj_mass        = 0.2 * (mass_props_TOGW.mass / 0.100)
obj_inertia     = 7.0 * (I_xx / 0.050)
obj_wingload    = 0.3 * (wing_loading / 15)

### Objective
objective = obj_sink + obj_mass + obj_inertia + obj_wingload
penalty = (mass_props["ballast"].x_cg / 1e3) ** 2

opti.minimize(objective + penalty)

### Additional constraint
opti.subject_to([
    LD_cruise == LD,
    design_mass_TOGW == mass_props_TOGW.mass
])

V_ht = S_ht * l_ht / (wing.area() * wing.mean_aerodynamic_chord())
V_vt = S_vt * l_vt / (wing.area() * wing_span)
# B_w = l_vt * np.radians(wing_dihedral_angle_deg) / (wing_span * aero['CL'])
fuselage_length = x_tail - x_nose

opti.subject_to([
    # x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,  # propeller must extend in front of wing
    fuselage_length < 0.8,                            # due to the length of carbon tube I have
    V_ht >= 0.40,
    V_ht <= 0.70,
    V_vt >= 0.02,
    V_vt <= 0.04,
    aero["Clb"] <= -0.025,
])

if __name__ == '__main__':
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug

    # helper for getting numeric values
    def s(x):
        return sol.value(x)

    airplane        = s(airplane)
    op_point        = s(op_point)
    mass_props      = s(mass_props)
    mass_props_TOGW = s(mass_props_TOGW)
    aero            = s(aero)

    fuselage_length_val = s(fuselage_length)

    sink_val    = s(obj_sink)
    mass_val    = s(obj_mass)
    inertia_val = s(obj_inertia)
    wingload_val= s(obj_wingload)
    total_val   = s(objective)

    print("\n===== OBJECTIVE CONTRIBUTION BREAKDOWN =====")
    print(f"Total Objective    : {total_val:.5f}")
    print("--------------------------------------------")
    print(f"Sink-rate term     : {sink_val:.5f}   ({sink_val/total_val*100:.1f}%)")
    print(f"Mass term          : {mass_val:.5f}   ({mass_val/total_val*100:.1f}%)")
    print(f"Inertia I_xx term   : {inertia_val:.5f}   ({inertia_val/total_val*100:.1f}%)")
    print(f"Wing loading term  : {wingload_val:.5f}   ({wingload_val/total_val*100:.1f}%)")
    print("--------------------------------------------")
    print("Percent values tell you which objective term dominates.\n")

    # --- Bounds / constraint check block starts here ---
    print("\n" + "*" * 20 + " DESIGN VARIABLE BOUNDS CHECK " + "*" * 20)

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

    # precompute some bounds that depend on other vars
    x_nose_val = s(x_nose)
    x_tail_val = s(x_tail)

    # Design variables
    check_var("V (m/s)",              op_point.velocity,        lb=1.0)
    check_var("alpha (deg)",          op_point.alpha,           lb=-10.0, ub=10.0)
    check_var("x_nose",               x_nose,                   ub=1e-3)
    check_var("wing_span",            wing_span,                lb=0.3,   ub=0.8)
    check_var("wing_dihedral_deg",    wing_dihedral_angle_deg,  lb=0.0,   ub=20.0)
    check_var("wing_root_chord",      wing_root_chord,          lb=1e-3)
    check_var("wing_taper",           wing_taper,               lb=0.3,   ub=1.0)

    check_var("l_ht",                 l_ht,                     lb=0.2,   ub=1.2)
    check_var("l_vt",                 l_vt,                     lb=0.2,   ub=1.2)

    check_var("ballast_mass",         mass_props["ballast"].mass, lb=0.0)
    check_var("ballast_x_cg",         mass_props["ballast"].x_cg,
               lb=x_nose_val, ub=x_tail_val)

    # Constraint quantities (non-variable, but nice to see if they are active)
    print("\n" + "*" * 20 + " CONSTRAINT ACTIVITY CHECK " + "*" * 20)

    static_margin_val = s(static_margin)
    V_ht_val = s(V_ht)
    V_vt_val = s(V_vt)
    Clb_val =  s(aero["Clb"])

    def check_quantity(name, value, lb=None, ub=None, atol=1e-6, rtol=1e-3):
        v = float(value)
        hits = []
        if lb is not None and v <= lb + max(atol, rtol * max(1.0, abs(lb))):
            hits.append("LOWER")
        if ub is not None and v >= ub - max(atol, rtol * max(1.0, abs(ub))):
            hits.append("UPPER")
        status = " ,".join(hits) if hits else "OK"
        print(f"{name:25s} = {v: .6g}  [{status}]")

    check_quantity("static_margin",   static_margin_val, lb=0.04, ub=0.10)
    check_quantity("V_ht",            V_ht_val,          lb=0.40, ub=0.70)
    check_quantity("V_vt",            V_vt_val,          lb=0.02, ub=0.04)
    check_quantity("Clb",             Clb_val,           ub=-0.025)  # only upper bound used here
    # --- Bounds / constraint check block ends here ---

    # Get numeric main wing from solved airplane
    main_wing_solved = copy.deepcopy(airplane.wings[0])
    htail_solved     = copy.deepcopy(airplane.wings[1])
    vtail_solved     = copy.deepcopy(airplane.wings[2])
    fuselage_solved  = copy.deepcopy(airplane.fuselages[0])

    # Make a low-res copy of the solved main wing
    wing_lowres = copy.deepcopy(main_wing_solved)
    xsecs_to_keep = np.arange(len(wing_lowres.xsecs)) % 2 == 0
    xsecs_to_keep[0] = True
    xsecs_to_keep[-1] = True
    wing_lowres.xsecs = np.array(wing_lowres.xsecs)[xsecs_to_keep]

    try:
        avl_aero = asb.AVL(
            airplane=asb.Airplane(
                wings=[
                    wing_lowres,
                    htail_solved,
                    vtail_solved,
                ],
                fuselages=[fuselage_solved]
            ),
            op_point=op_point,
            xyz_ref=mass_props_TOGW.xyz_cg,
            working_directory="avl_debug"
        ).run()
    except FileNotFoundError:
        class EmptyDict:
            def __getitem__(self, item):
                return "Install AVL to see this."


        avl_aero = EmptyDict()

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    ##### Section: Printout
    print_title = lambda s: print(s.upper().join(["*" * 20] * 2))


    def fmt(x):
        return f"{s(x):.6g}"


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
        # "AVL: Cma"              : avl_aero['Cma'],
        # "AVL: Cnb"              : avl_aero['Cnb'],
        # "AVL: Cm"               : avl_aero['Cm'],
        # "AVL: Clb Cnr / Clr Cnb": avl_aero['Clb Cnr / Clr Cnb'],
        "CG location"           : "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
        "Wing Span"             : f"{fmt(wing_span)} m ({fmt(wing_span / u.foot)} ft)",
        "Fuselage Length"       : f"{fmt(sol.value(fuselage_length))} m ({fmt(sol.value(fuselage_length) / u.foot)} ft)",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    fmtpow = lambda x: fmt(x) + " W"

    print_title("Mass props")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {v.mass * 1e3:.2f} g ({v.mass / u.oz:.2f} oz)")

    if make_plots:
        ##### Section: Geometry
        airplane.draw_three_view(show=False)
        p.show_plot(tight_layout=False, savefig="figures/three_view.png")

        ##### Section: Mass Budget
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