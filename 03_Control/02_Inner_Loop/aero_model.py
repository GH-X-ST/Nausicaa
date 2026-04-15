from __future__ import annotations

from math import pi

import casadi as ca
import numpy as np

from glider import Glider


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Model constants
# 2) Frame and backend helpers
# 3) Public aerodynamic helpers
# 4) Numeric evaluation path
# 5) Symbolic evaluation path
# =============================================================================

# =============================================================================
# 1) Model Constants
# =============================================================================
# Section-model blend settings
STALL_BLEND_ALPHA_RAD = np.deg2rad(12.0)
STALL_BLEND_WIDTH_RAD = np.deg2rad(3.0)
POST_STALL_DRAG_GAIN = 1.8
EPS = 1e-9


# =============================================================================
# 2) Frame and Backend Helpers
# =============================================================================
def _is_casadi_type(value: object) -> bool:
    return isinstance(value, (ca.SX, ca.MX, ca.DM))


def _is_symbolic_input(*values: object) -> bool:
    return any(_is_casadi_type(value) for value in values)


def _c_wb_numpy(euler_321: np.ndarray) -> np.ndarray:
    # 3-2-1 body-to-world rotation
    phi, theta, psi = euler_321
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    return np.array(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [-s_theta, s_phi * c_theta, c_phi * c_theta],
        ]
    )


def _c_wb_ca(euler_321: ca.SX | ca.MX | ca.DM) -> ca.SX | ca.MX | ca.DM:
    # 3-2-1 body-to-world rotation
    phi = euler_321[0]
    theta = euler_321[1]
    psi = euler_321[2]
    c_phi = ca.cos(phi)
    s_phi = ca.sin(phi)
    c_theta = ca.cos(theta)
    s_theta = ca.sin(theta)
    c_psi = ca.cos(psi)
    s_psi = ca.sin(psi)
    return ca.vertcat(
        ca.hcat(
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ]
        ),
        ca.hcat(
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ]
        ),
        ca.hcat([-s_theta, s_phi * c_theta, c_phi * c_theta]),
    )


def _world_up_to_w_frame_vector(vector_w_up: np.ndarray) -> np.ndarray:
    # Public world frame uses z up
    return np.array([vector_w_up[0], vector_w_up[1], -vector_w_up[2]], dtype=float)


def _world_up_to_w_frame_rows(rows_w_up: np.ndarray) -> np.ndarray:
    # Public world frame uses z up
    rows_w = np.asarray(rows_w_up, dtype=float).copy()
    rows_w[:, 2] *= -1.0
    return rows_w


def _w_frame_to_world_up_rows(rows_w: np.ndarray) -> np.ndarray:
    # Public world frame uses z up
    rows_w_up = np.asarray(rows_w, dtype=float).copy()
    rows_w_up[:, 2] *= -1.0
    return rows_w_up


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, EPS)


def _ca_vector3(value: ca.SX | ca.MX | ca.DM | np.ndarray | list[float]) -> ca.SX | ca.MX | ca.DM:
    if _is_casadi_type(value):
        return ca.reshape(value, 3, 1)
    return ca.DM(np.asarray(value, dtype=float).reshape(3, 1))


def _ca_world_up_to_w_frame(vector_w_up: ca.SX | ca.MX | ca.DM) -> ca.SX | ca.MX | ca.DM:
    return ca.vertcat(vector_w_up[0], vector_w_up[1], -vector_w_up[2])


def _ca_norm(vector: ca.SX | ca.MX | ca.DM) -> ca.SX | ca.MX | ca.DM:
    return ca.sqrt(ca.dot(vector, vector) + EPS)


def _ca_normalize(vector: ca.SX | ca.MX | ca.DM) -> ca.SX | ca.MX | ca.DM:
    return vector / _ca_norm(vector)


def _fabs(value: object) -> object:
    if _is_casadi_type(value):
        return ca.fabs(value)
    return np.abs(value)


def _tanh(value: object) -> object:
    if _is_casadi_type(value):
        return ca.tanh(value)
    return np.tanh(value)


def _sin(value: object) -> object:
    if _is_casadi_type(value):
        return ca.sin(value)
    return np.sin(value)


def _cos(value: object) -> object:
    if _is_casadi_type(value):
        return ca.cos(value)
    return np.cos(value)


# =============================================================================
# 3) Public Aerodynamic Helpers
# =============================================================================
def local_apparent_wind(
    v_cg_b: np.ndarray | ca.SX | ca.MX | ca.DM,
    omega_b: np.ndarray | ca.SX | ca.MX | ca.DM,
    r_strip_b: np.ndarray | ca.SX | ca.MX | ca.DM,
    wind_b: np.ndarray | ca.SX | ca.MX | ca.DM,
) -> np.ndarray | ca.SX | ca.MX | ca.DM:
    # Rigid-body strip velocity minus local wind
    if _is_symbolic_input(v_cg_b, omega_b, r_strip_b, wind_b):
        return _ca_vector3(v_cg_b) + ca.cross(_ca_vector3(omega_b), _ca_vector3(r_strip_b)) - _ca_vector3(wind_b)
    v_cg = np.asarray(v_cg_b, dtype=float)
    omega = np.asarray(omega_b, dtype=float)
    r_strip = np.asarray(r_strip_b, dtype=float)
    wind = np.asarray(wind_b, dtype=float)
    return v_cg + np.cross(omega, r_strip) - wind


def section_aero_coefficients(
    alpha: object,
    delta_local: object,
    aspect_ratio: object,
    cd0: object,
    alpha0: object,
    induced_drag_efficiency: object,
    flap_effectiveness: object,
) -> tuple[object, object, object]:
    # Positive local deflection increases section incidence
    alpha_eff = alpha - alpha0 + flap_effectiveness * delta_local
    # Attached branch uses a finite-wing lift slope
    a_3d = 2.0 * pi * aspect_ratio / (aspect_ratio + 2.0)
    cl_attached = a_3d * alpha_eff
    cd_attached = cd0 + cl_attached * cl_attached / (
        pi * induced_drag_efficiency * aspect_ratio
    )
    cl_post_stall = 2.0 * _sin(alpha_eff) * _cos(alpha_eff)
    cd_post_stall = cd0 + POST_STALL_DRAG_GAIN * _sin(alpha_eff) ** 2
    sigma = 0.5 * (
        1.0
        + _tanh(
            (STALL_BLEND_ALPHA_RAD - _fabs(alpha_eff))
            / STALL_BLEND_WIDTH_RAD
        )
    )
    cl = sigma * cl_attached + (1.0 - sigma) * cl_post_stall
    cd = sigma * cd_attached + (1.0 - sigma) * cd_post_stall
    # Diagnostic output only in v1
    cm = 0.0 * cl
    return cl, cd, cm


# =============================================================================
# 4) Numeric Evaluation Path
# =============================================================================
def _sample_numeric_wind(
    wind_field: object,
    r_cg_w: np.ndarray,
    r_strip_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Numeric wind callbacks use the public z-up world frame
    if wind_field is None:
        wind_strip_w_up = np.zeros_like(r_strip_w)
        wind_cg_w_up = np.zeros(3)
    elif callable(wind_field):
        wind_strip_w_up = np.asarray(wind_field(_w_frame_to_world_up_rows(r_strip_w)), dtype=float)
        wind_cg_w_up = np.asarray(
            wind_field(_w_frame_to_world_up_rows(r_cg_w.reshape(1, 3))),
            dtype=float,
        ).reshape(-1, 3)[0]
    else:
        wind_cg_w_up = np.asarray(wind_field, dtype=float).reshape(3)
        wind_strip_w_up = np.broadcast_to(wind_cg_w_up, r_strip_w.shape)
    return _world_up_to_w_frame_rows(wind_strip_w_up), _world_up_to_w_frame_vector(wind_cg_w_up)


def _evaluate_numeric(
    glider: Glider,
    r_cg_w: np.ndarray,
    euler_321: np.ndarray,
    v_cg_b: np.ndarray,
    omega_b: np.ndarray,
    delta_a: float,
    delta_e: float,
    delta_r: float,
    rho: float,
    wind_field: object,
    return_strips: bool,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    # Euler angles are converted once at the public interface
    c_wb = _c_wb_numpy(euler_321)
    c_bw = c_wb.T
    r_cg_w_frame = _world_up_to_w_frame_vector(r_cg_w)
    # Strip centers in the internal z-down world frame
    r_strip_w = r_cg_w_frame + glider.r_strip_b @ c_wb.T
    # Wind is sampled in world axes and rotated into body axes
    wind_strip_w, wind_cg_w = _sample_numeric_wind(wind_field, r_cg_w_frame, r_strip_w)
    wind_strip_b = wind_strip_w @ c_bw.T
    wind_cg_b = c_bw @ wind_cg_w
    # Strip velocity uses rigid-body kinematics
    v_rot_b = np.cross(np.broadcast_to(omega_b, glider.r_strip_b.shape), glider.r_strip_b)
    v_strip_b = v_cg_b + v_rot_b
    v_rel_b = v_strip_b - wind_strip_b
    # Section-plane state excludes spanwise velocity
    span_speed = np.sum(v_rel_b * glider.span_axis_b, axis=1)
    v_plane_b = v_rel_b - glider.span_axis_b * span_speed[:, None]
    speed_plane = np.linalg.norm(v_plane_b, axis=1)
    speed_plane_safe = np.maximum(speed_plane, EPS)
    speed_total = np.linalg.norm(v_rel_b, axis=1)
    speed_total_safe = np.maximum(speed_total, EPS)
    drag_dir_b = -v_plane_b / speed_plane_safe[:, None]
    # Lift lies in the local section plane
    lift_vec_b = glider.surface_normal_b - np.sum(
        glider.surface_normal_b * drag_dir_b,
        axis=1,
        keepdims=True,
    ) * drag_dir_b
    lift_dir_b = _normalize_rows(lift_vec_b)
    # Section angles come from the local apparent wind
    alpha = np.arctan2(
        -np.sum(v_plane_b * glider.surface_normal_b, axis=1),
        v_plane_b[:, 0],
    )
    beta = np.arcsin(np.clip(span_speed / speed_total_safe, -1.0, 1.0))
    q_bar = 0.5 * rho * speed_plane * speed_plane
    delta_vector = np.array([delta_a, delta_e, delta_r], dtype=float)
    delta_local = glider.control_mix @ delta_vector
    cl, cd, cm = section_aero_coefficients(
        alpha=alpha,
        delta_local=delta_local,
        aspect_ratio=glider.aspect_ratio_strip,
        cd0=glider.cd0_strip,
        alpha0=glider.alpha0_strip,
        induced_drag_efficiency=glider.efficiency_strip,
        flap_effectiveness=glider.flap_scale_strip,
    )
    force_scale = (q_bar * glider.area_strip_m2)[:, None]
    # Strip forces are accumulated in body axes
    f_strip_b = force_scale * (cl[:, None] * lift_dir_b + cd[:, None] * drag_dir_b)
    # Total moment is formed from strip force only
    m_strip_b = np.cross(glider.r_strip_b, f_strip_b)
    v_rel_cg_b = v_cg_b - wind_cg_b
    speed_cg = np.linalg.norm(v_rel_cg_b)
    if speed_cg > EPS:
        # Lumped fuselage parasitic drag
        f_fuse_b = (
            -0.5
            * rho
            * glider.drag_area_fuse_m2
            * speed_cg
            * v_rel_cg_b
        )
    else:
        f_fuse_b = np.zeros(3)
    f_aero_b = np.sum(f_strip_b, axis=0) + f_fuse_b
    m_aero_b = np.sum(m_strip_b, axis=0)
    result: dict[str, np.ndarray | dict[str, np.ndarray]] = {
        "f_aero_b": f_aero_b,
        "m_aero_b": m_aero_b,
    }
    if return_strips:
        # Diagnostic arrays keep fixed shapes in v1
        result["strips"] = {
            "r_strip_b": glider.r_strip_b.copy(),
            "v_rel_b": v_rel_b,
            "alpha": alpha,
            "beta": beta,
            "q_bar": q_bar,
            "delta_local": delta_local,
            "cl": np.asarray(cl, dtype=float),
            "cd": np.asarray(cd, dtype=float),
            "cm": np.asarray(cm, dtype=float),
            "f_strip_b": f_strip_b,
            "m_strip_b": m_strip_b,
            "surface_code": glider.surface_code.copy(),
        }
    return result


# =============================================================================
# 5) Symbolic Evaluation Path
# =============================================================================
def _evaluate_symbolic(
    glider: Glider,
    r_cg_w: ca.SX | ca.MX | ca.DM,
    euler_321: ca.SX | ca.MX | ca.DM,
    v_cg_b: ca.SX | ca.MX | ca.DM,
    omega_b: ca.SX | ca.MX | ca.DM,
    delta_a: ca.SX | ca.MX | ca.DM,
    delta_e: ca.SX | ca.MX | ca.DM,
    delta_r: ca.SX | ca.MX | ca.DM,
    rho: float,
    wind_field: object,
    return_strips: bool,
) -> dict[str, ca.SX | ca.MX | ca.DM | dict[str, ca.SX | ca.MX | ca.DM]]:
    if callable(wind_field):
        raise TypeError("Symbolic evaluation supports only wind_field=None or constant wind.")
    # Euler angles are converted once at the public interface
    c_wb = _c_wb_ca(_ca_vector3(euler_321))
    c_bw = c_wb.T
    v_cg = _ca_vector3(v_cg_b)
    omega = _ca_vector3(omega_b)
    if wind_field is None:
        wind_cg_w = ca.DM.zeros(3, 1)
    else:
        wind_cg_w = _ca_world_up_to_w_frame(_ca_vector3(wind_field))
    wind_cg_b = c_bw @ wind_cg_w
    f_aero_b = ca.DM.zeros(3, 1)
    m_aero_b = ca.DM.zeros(3, 1)
    if return_strips:
        v_rel_rows: list[ca.SX | ca.MX | ca.DM] = []
        alpha_rows: list[ca.SX | ca.MX | ca.DM] = []
        beta_rows: list[ca.SX | ca.MX | ca.DM] = []
        q_bar_rows: list[ca.SX | ca.MX | ca.DM] = []
        delta_rows: list[ca.SX | ca.MX | ca.DM] = []
        cl_rows: list[ca.SX | ca.MX | ca.DM] = []
        cd_rows: list[ca.SX | ca.MX | ca.DM] = []
        cm_rows: list[ca.SX | ca.MX | ca.DM] = []
        f_rows: list[ca.SX | ca.MX | ca.DM] = []
        m_rows: list[ca.SX | ca.MX | ca.DM] = []
    delta_vector = ca.vertcat(delta_a, delta_e, delta_r)
    for idx in range(glider.r_strip_b.shape[0]):
        r_strip = ca.DM(glider.r_strip_b[idx]).reshape((3, 1))
        wind_strip_b = wind_cg_b
        # Strip velocity uses rigid-body kinematics
        v_rel_b = local_apparent_wind(v_cg, omega, r_strip, wind_strip_b)
        span_axis_b = ca.DM(glider.span_axis_b[idx]).reshape((3, 1))
        surface_normal_b = ca.DM(glider.surface_normal_b[idx]).reshape((3, 1))
        # Section-plane state excludes spanwise velocity
        span_speed = ca.dot(v_rel_b, span_axis_b)
        v_plane_b = v_rel_b - span_axis_b * span_speed
        speed_plane = _ca_norm(v_plane_b)
        speed_total = _ca_norm(v_rel_b)
        drag_dir_b = -v_plane_b / speed_plane
        # Lift lies in the local section plane
        lift_vec_b = surface_normal_b - ca.dot(surface_normal_b, drag_dir_b) * drag_dir_b
        lift_dir_b = _ca_normalize(lift_vec_b)
        alpha = ca.atan2(-ca.dot(v_plane_b, surface_normal_b), v_plane_b[0])
        beta_arg = ca.fmin(1.0, ca.fmax(-1.0, span_speed / speed_total))
        beta = ca.asin(beta_arg)
        q_bar = 0.5 * rho * speed_plane * speed_plane
        delta_local = ca.dot(ca.DM(glider.control_mix[idx]), delta_vector)
        cl, cd, cm = section_aero_coefficients(
            alpha=alpha,
            delta_local=delta_local,
            aspect_ratio=float(glider.aspect_ratio_strip[idx]),
            cd0=float(glider.cd0_strip[idx]),
            alpha0=float(glider.alpha0_strip[idx]),
            induced_drag_efficiency=float(glider.efficiency_strip[idx]),
            flap_effectiveness=float(glider.flap_scale_strip[idx]),
        )
        force_scale = q_bar * float(glider.area_strip_m2[idx])
        # Strip forces are accumulated in body axes
        f_strip_b = force_scale * (cl * lift_dir_b + cd * drag_dir_b)
        # Total moment is formed from strip force only
        m_strip_b = ca.cross(r_strip, f_strip_b)
        f_aero_b += f_strip_b
        m_aero_b += m_strip_b
        if return_strips:
            v_rel_rows.append(v_rel_b)
            alpha_rows.append(alpha)
            beta_rows.append(beta)
            q_bar_rows.append(q_bar)
            delta_rows.append(delta_local)
            cl_rows.append(cl)
            cd_rows.append(cd)
            cm_rows.append(cm)
            f_rows.append(f_strip_b)
            m_rows.append(m_strip_b)
    v_rel_cg_b = v_cg - wind_cg_b
    speed_cg = _ca_norm(v_rel_cg_b)
    f_fuse_b = -0.5 * rho * glider.drag_area_fuse_m2 * speed_cg * v_rel_cg_b
    result: dict[str, ca.SX | ca.MX | ca.DM | dict[str, ca.SX | ca.MX | ca.DM]] = {
        "f_aero_b": f_aero_b + f_fuse_b,
        "m_aero_b": m_aero_b,
    }
    if return_strips:
        result["strips"] = {
            "r_strip_b": ca.DM(glider.r_strip_b),
            "v_rel_b": ca.hcat(v_rel_rows).T,
            "alpha": ca.vertcat(*alpha_rows),
            "beta": ca.vertcat(*beta_rows),
            "q_bar": ca.vertcat(*q_bar_rows),
            "delta_local": ca.vertcat(*delta_rows),
            "cl": ca.vertcat(*cl_rows),
            "cd": ca.vertcat(*cd_rows),
            "cm": ca.vertcat(*cm_rows),
            "f_strip_b": ca.hcat(f_rows).T,
            "m_strip_b": ca.hcat(m_rows).T,
            "surface_code": ca.DM(glider.surface_code).reshape((-1, 1)),
        }
    return result


# =============================================================================
# Public Evaluation Entry Point
# =============================================================================
def evaluate_glider_aero(
    glider: Glider,
    r_cg_w: np.ndarray | ca.SX | ca.MX | ca.DM,
    euler_321: np.ndarray | ca.SX | ca.MX | ca.DM,
    v_cg_b: np.ndarray | ca.SX | ca.MX | ca.DM,
    omega_b: np.ndarray | ca.SX | ca.MX | ca.DM,
    delta_a: float | ca.SX | ca.MX | ca.DM,
    delta_e: float | ca.SX | ca.MX | ca.DM,
    delta_r: float | ca.SX | ca.MX | ca.DM,
    rho: float = 1.225,
    wind_field: object = None,
    return_strips: bool = False,
) -> dict[str, object]:
    if _is_symbolic_input(r_cg_w, euler_321, v_cg_b, omega_b, delta_a, delta_e, delta_r):
        return _evaluate_symbolic(
            glider=glider,
            r_cg_w=r_cg_w,
            euler_321=euler_321,
            v_cg_b=v_cg_b,
            omega_b=omega_b,
            delta_a=delta_a,
            delta_e=delta_e,
            delta_r=delta_r,
            rho=rho,
            wind_field=wind_field,
            return_strips=return_strips,
        )
    return _evaluate_numeric(
        glider=glider,
        r_cg_w=np.asarray(r_cg_w, dtype=float).reshape(3),
        euler_321=np.asarray(euler_321, dtype=float).reshape(3),
        v_cg_b=np.asarray(v_cg_b, dtype=float).reshape(3),
        omega_b=np.asarray(omega_b, dtype=float).reshape(3),
        delta_a=float(delta_a),
        delta_e=float(delta_e),
        delta_r=float(delta_r),
        rho=float(rho),
        wind_field=wind_field,
        return_strips=return_strips,
    )
