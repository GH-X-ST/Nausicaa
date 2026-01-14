from __future__ import annotations

import numpy as onp
import aerosandbox as asb
import aerosandbox.numpy as np

from config import Config


def build_airplane(
    opti: asb.Opti,
    cfg: Config,
    airfoils: dict[str, asb.Airfoil],
    controls: dict,
) -> dict:
    """Build airplane geometry and apply control deflections."""
    bnd = cfg.bounds

    # Control surfaces
    aileron_cs = asb.ControlSurface(
        name="aileron", symmetric=False, hinge_point=0.75, trailing_edge=True
    )
    rudder_cs = asb.ControlSurface(
        name="rudder", symmetric=True, hinge_point=0.75, trailing_edge=True
    )
    elevator_cs = asb.ControlSurface(
        name="elevator", symmetric=True, hinge_point=0.75, trailing_edge=True
    )

    # Nose location
    x_nose = opti.variable(init_guess=-0.1, upper_bound=bnd.x_nose_max)

    # Wing
    b_w = opti.variable(init_guess=0.3, lower_bound=bnd.b_w_min, upper_bound=bnd.b_w_max)
    gamma_w_deg = opti.variable(
        init_guess=11.0, lower_bound=bnd.gamma_w_min_deg, upper_bound=bnd.gamma_w_max_deg
    )
    c_root_w = opti.variable(init_guess=0.15, lower_bound=bnd.c_root_w_min)

    lambda_w = cfg.lambda_w

    def wing_rotation(xyz):
        """Apply dihedral rotation Γ_w about x-axis."""
        rot = np.rotation_matrix_3D(angle=np.radians(gamma_w_deg), axis="X")
        return rot @ np.array(xyz)

    def chord_w(y):
        """Simple linear taper."""
        half_span = b_w / 2
        c_tip = lambda_w * c_root_w
        eta = np.abs(y) / half_span
        return (1 - eta) * c_root_w + eta * c_tip

    def twist_w(y):
        return np.zeros_like(y)

    n_w_span = 11
    span_fracs = onp.linspace(0.0, 0.5, n_w_span)
    wing_xsecs = []
    for eta in span_fracs:
        y_w = eta * b_w
        cs_list = [aileron_cs] if 0.25 <= eta <= 0.45 else []
        wing_xsecs.append(
            asb.WingXSec(
                xyz_le=wing_rotation([-chord_w(y_w), y_w, 0.0]),
                chord=chord_w(y_w),
                airfoil=airfoils["s3021"],
                twist=twist_w(y_w),
                control_surfaces=cs_list,
            )
        )

    wing = (
        asb.Wing(name="Main Wing", symmetric=True, xsecs=wing_xsecs)
        .translate([0.75 * c_root_w, 0.0, 0.0])
    )

    # Horizontal tail
    l_ht = opti.variable(init_guess=0.6, lower_bound=bnd.l_ht_min, upper_bound=bnd.l_ht_max)
    b_ht = opti.variable(init_guess=0.15, lower_bound=bnd.b_ht_min)

    ar_ht = cfg.ar_ht
    lambda_ht = cfg.lambda_ht
    c_root_ht = 2 * b_ht / (ar_ht * (1 + lambda_ht))

    def chord_ht(y):
        half_span = b_ht / 2
        c_tip = lambda_ht * c_root_ht
        eta = np.abs(y) / half_span
        return (1 - eta) * c_root_ht + eta * c_tip

    def twist_ht(y):
        return np.zeros_like(y)

    y_ht_stations = onp.sin(onp.linspace(0.0, onp.pi / 2, 7)) * 0.5
    y_ht_stations = y_ht_stations[::-1]
    
    htail_xsecs = []
    for eta in y_ht_stations:
        y_ht = eta * b_ht
        htail_xsecs.append(
            asb.WingXSec(
                xyz_le=[l_ht - chord_ht(y_ht), y_ht, 0.0],
                chord=chord_ht(y_ht),
                twist=twist_ht(y_ht),
                airfoil=airfoils["naca0008"],
                control_surfaces=[elevator_cs],
            )
        )

    htail = asb.Wing(name="HTail", symmetric=True, xsecs=htail_xsecs)

    v_ht = htail.area() * l_ht / (wing.area() * wing.mean_aerodynamic_chord())

    # Vertical tail
    b_vt = opti.variable(init_guess=0.07, lower_bound=bnd.b_vt_min)
    l_vt = l_ht
    ar_vt = cfg.ar_vt
    lambda_vt = cfg.lambda_vt
    c_root_vt = 2 * b_vt / (ar_vt * (1 + lambda_vt))

    def chord_vt(z):
        c_tip = lambda_vt * c_root_vt
        eta = np.abs(z) / b_vt
        return (1 - eta) * c_root_vt + eta * c_tip

    def twist_vt(z):
        return np.zeros_like(z)
    
    z_vt_stations = onp.sin(onp.linspace(0.0, onp.pi / 2, 7))
    z_vt_stations = z_vt_stations[::-1]

    vtail_xsecs = []
    for eta in z_vt_stations:
        z_vt = eta * b_vt
        vtail_xsecs.append(
            asb.WingXSec(
                xyz_le=[l_vt - chord_vt(z_vt), 0.0, z_vt],
                chord=chord_vt(z_vt),
                twist=twist_vt(z_vt),
                airfoil=airfoils["naca0008"],
                control_surfaces=[rudder_cs],
            )
        )

    vtail = asb.Wing(name="VTail", symmetric=False, xsecs=vtail_xsecs)
    v_vt = vtail.area() * l_vt / (wing.area() * b_w)

    # Fuselage
    x_tail = l_ht
    fuselage = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[x_nose, 0.0, 0.0], radius=4e-3 / 2),
            asb.FuselageXSec(xyz_c=[x_tail, 0.0, 0.0], radius=4e-3 / 2),
        ],
    )

    airplane = asb.Airplane(
        name="Nausicaa",
        wings=[wing, htail, vtail],
        fuselages=[fuselage],
    ).with_control_deflections(
        {
            "aileron": controls["delta_a_deg"],
            "rudder": controls["delta_r_deg"],
            "elevator": controls["delta_e_deg"],
        }
    )

    return {
        "airplane": airplane,
        "wing": wing,
        "htail": htail,
        "vtail": vtail,
        "fuselage": fuselage,
        "x_nose": x_nose,
        "x_tail": x_tail,
        "b_w": b_w,
        "gamma_w_deg": gamma_w_deg,
        "c_root_w": c_root_w,
        "lambda_w": lambda_w,
        "l_ht": l_ht,
        "b_ht": b_ht,
        "c_root_ht": c_root_ht,
        "lambda_ht": lambda_ht,
        "l_vt": l_vt,
        "b_vt": b_vt,
        "c_root_vt": c_root_vt,
        "lambda_vt": lambda_vt,
        "v_ht": v_ht,
        "v_vt": v_vt,
        "control_surfaces": {
            "aileron": aileron_cs,
            "rudder": rudder_cs,
            "elevator": elevator_cs,
        },
    }