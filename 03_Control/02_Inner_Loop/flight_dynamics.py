from __future__ import annotations

from dataclasses import dataclass
from math import pi

import casadi as ca
import numpy as np

from glider import Glider


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Model constants
# 2) Runtime dataclasses
# 3) Numeric frame helpers
# 4) Symbolic frame helpers
# 5) Section model
# 6) Aircraft adapter
# 7) Numeric wind and aerodynamic evaluation
# 8) Public numeric state evaluation
# 9) Symbolic dynamics builder
# =============================================================================

# =============================================================================
# 1) Model Constants
# =============================================================================
G_M_S2 = 9.81
EPS = 1e-9
STALL_BLEND_ALPHA_RAD = np.deg2rad(12.0)
STALL_BLEND_WIDTH_RAD = np.deg2rad(3.0)
POST_STALL_DRAG_GAIN = 1.8


# =============================================================================
# 2) Runtime Dataclasses
# =============================================================================
@dataclass(frozen=True)
class AircraftModel:
    mass_kg: float
    inertia_b: np.ndarray
    inertia_inv_b: np.ndarray
    r_strip_b: np.ndarray
    area_strip_m2: np.ndarray
    chord_strip_m: np.ndarray
    aspect_ratio_strip: np.ndarray
    span_axis_b: np.ndarray
    surface_normal_b: np.ndarray
    control_mix: np.ndarray
    cd0_strip: np.ndarray
    alpha0_strip: np.ndarray
    efficiency_strip: np.ndarray
    flap_scale_strip: np.ndarray
    drag_area_fuse_m2: float
    strip_count: int


@dataclass(frozen=True)
class SymbolicDynamicsModel:
    x: ca.SX
    u_cmd: ca.SX
    wind_param: ca.SX | None
    x_dot: ca.SX
    function: ca.Function


# =============================================================================
# 3) Numeric Frame Helpers
# =============================================================================
def _world_up_to_internal_vector(vector_w_up: np.ndarray) -> np.ndarray:
    return np.array([vector_w_up[0], vector_w_up[1], -vector_w_up[2]], dtype=float)


def _internal_to_world_up_vector(vector_w: np.ndarray) -> np.ndarray:
    return np.array([vector_w[0], vector_w[1], -vector_w[2]], dtype=float)


def _internal_to_world_up_rows(rows_w: np.ndarray) -> np.ndarray:
    rows_w_up = np.asarray(rows_w, dtype=float).copy()
    rows_w_up[:, 2] *= -1.0
    return rows_w_up


def _c_wb_numpy(phi: float, theta: float, psi: float) -> np.ndarray:
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


def _t_euler_numpy(phi: float, theta: float) -> np.ndarray:
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    t_theta = np.tan(theta)
    return np.array(
        [
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ]
    )


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, EPS)


# =============================================================================
# 4) Symbolic Frame Helpers
# =============================================================================
def _c_wb_ca(phi: ca.SX, theta: ca.SX, psi: ca.SX) -> ca.SX:
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


def _t_euler_ca(phi: ca.SX, theta: ca.SX) -> ca.SX:
    c_phi = ca.cos(phi)
    s_phi = ca.sin(phi)
    c_theta = ca.cos(theta)
    t_theta = ca.tan(theta)
    return ca.vertcat(
        ca.hcat([1.0, s_phi * t_theta, c_phi * t_theta]),
        ca.hcat([0.0, c_phi, -s_phi]),
        ca.hcat([0.0, s_phi / c_theta, c_phi / c_theta]),
    )


def _ca_vector3(value: ca.SX | np.ndarray | list[float] | tuple[float, ...]) -> ca.SX:
    if isinstance(value, ca.SX):
        return ca.reshape(value, 3, 1)
    return ca.DM(np.asarray(value, dtype=float).reshape(3, 1))


def _ca_internal_from_world_up(vector_w_up: ca.SX) -> ca.SX:
    return ca.vertcat(vector_w_up[0], vector_w_up[1], -vector_w_up[2])


def _ca_world_up_from_internal(vector_w: ca.SX) -> ca.SX:
    return ca.vertcat(vector_w[0], vector_w[1], -vector_w[2])


def _ca_norm(vector: ca.SX) -> ca.SX:
    return ca.sqrt(ca.dot(vector, vector) + EPS)


# =============================================================================
# 5) Section Model
# =============================================================================
def _section_aero_numpy(
    alpha: np.ndarray,
    delta_local: np.ndarray,
    aspect_ratio: np.ndarray,
    cd0: np.ndarray,
    alpha0: np.ndarray,
    efficiency: np.ndarray,
    flap_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_eff = alpha - alpha0 + flap_scale * delta_local
    a_3d = 2.0 * pi * aspect_ratio / (aspect_ratio + 2.0)
    cl_attached = a_3d * alpha_eff
    cd_attached = cd0 + cl_attached**2 / (pi * efficiency * aspect_ratio)
    cl_post_stall = 2.0 * np.sin(alpha_eff) * np.cos(alpha_eff)
    cd_post_stall = cd0 + POST_STALL_DRAG_GAIN * np.sin(alpha_eff) ** 2
    sigma = 0.5 * (
        1.0
        + np.tanh(
            (STALL_BLEND_ALPHA_RAD - np.abs(alpha_eff))
            / STALL_BLEND_WIDTH_RAD
        )
    )
    cl = sigma * cl_attached + (1.0 - sigma) * cl_post_stall
    cd = sigma * cd_attached + (1.0 - sigma) * cd_post_stall
    return cl, cd


def _section_aero_ca(
    alpha: ca.SX,
    delta_local: ca.SX,
    aspect_ratio: float,
    cd0: float,
    alpha0: float,
    efficiency: float,
    flap_scale: float,
) -> tuple[ca.SX, ca.SX]:
    alpha_eff = alpha - alpha0 + flap_scale * delta_local
    a_3d = 2.0 * pi * aspect_ratio / (aspect_ratio + 2.0)
    cl_attached = a_3d * alpha_eff
    cd_attached = cd0 + cl_attached**2 / (pi * efficiency * aspect_ratio)
    cl_post_stall = 2.0 * ca.sin(alpha_eff) * ca.cos(alpha_eff)
    cd_post_stall = cd0 + POST_STALL_DRAG_GAIN * ca.sin(alpha_eff) ** 2
    sigma = 0.5 * (
        1.0
        + ca.tanh(
            (STALL_BLEND_ALPHA_RAD - ca.fabs(alpha_eff))
            / STALL_BLEND_WIDTH_RAD
        )
    )
    cl = sigma * cl_attached + (1.0 - sigma) * cl_post_stall
    cd = sigma * cd_attached + (1.0 - sigma) * cd_post_stall
    return cl, cd


# =============================================================================
# 6) Aircraft Adapter
# =============================================================================
def adapt_glider(glider: Glider) -> AircraftModel:
    inertia_b = np.asarray(glider.inertia_b, dtype=float)
    return AircraftModel(
        mass_kg=float(glider.mass_kg),
        inertia_b=inertia_b,
        inertia_inv_b=np.linalg.inv(inertia_b),
        r_strip_b=np.asarray(glider.r_strip_b, dtype=float),
        area_strip_m2=np.asarray(glider.area_strip_m2, dtype=float),
        chord_strip_m=np.asarray(glider.chord_strip_m, dtype=float),
        aspect_ratio_strip=np.asarray(glider.aspect_ratio_strip, dtype=float),
        span_axis_b=np.asarray(glider.span_axis_b, dtype=float),
        surface_normal_b=np.asarray(glider.surface_normal_b, dtype=float),
        control_mix=np.asarray(glider.control_mix, dtype=float),
        cd0_strip=np.asarray(glider.cd0_strip, dtype=float),
        alpha0_strip=np.asarray(glider.alpha0_strip, dtype=float),
        efficiency_strip=np.asarray(glider.efficiency_strip, dtype=float),
        flap_scale_strip=np.asarray(glider.flap_scale_strip, dtype=float),
        drag_area_fuse_m2=float(glider.drag_area_fuse_m2),
        strip_count=int(glider.r_strip_b.shape[0]),
    )


# =============================================================================
# 7) Numeric Wind and Aerodynamic Evaluation
# =============================================================================
def _sample_numeric_wind(
    wind_model: object,
    r_cg_w: np.ndarray,
    r_strip_w: np.ndarray,
    wind_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if wind_model is None:
        wind_cg_w_up = np.zeros(3)
        wind_strip_w_up = np.zeros_like(r_strip_w)
    elif callable(wind_model):
        wind_cg_w_up = np.asarray(
            wind_model(_internal_to_world_up_rows(r_cg_w.reshape(1, 3))),
            dtype=float,
        ).reshape(-1, 3)[0]
        if wind_mode == "cg":
            wind_strip_w_up = np.broadcast_to(wind_cg_w_up, r_strip_w.shape)
        else:
            wind_strip_w_up = np.asarray(
                wind_model(_internal_to_world_up_rows(r_strip_w)),
                dtype=float,
            )
    else:
        wind_cg_w_up = np.asarray(wind_model, dtype=float).reshape(3)
        wind_strip_w_up = np.broadcast_to(wind_cg_w_up, r_strip_w.shape)
    wind_cg_w = _world_up_to_internal_vector(wind_cg_w_up)
    wind_strip_w = np.asarray(wind_strip_w_up, dtype=float).copy()
    wind_strip_w[:, 2] *= -1.0
    return wind_cg_w, wind_strip_w


def _evaluate_aero_numeric(
    x: np.ndarray,
    aircraft: AircraftModel,
    wind_model: object,
    rho: float,
    wind_mode: str,
) -> dict[str, object]:
    x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r = x
    r_cg_w_up = np.array([x_w, y_w, z_w], dtype=float)
    v_b = np.array([u, v, w], dtype=float)
    omega_b = np.array([p, q, r], dtype=float)
    delta = np.array([delta_a, delta_e, delta_r], dtype=float)
    c_wb = _c_wb_numpy(phi, theta, psi)
    c_bw = c_wb.T
    r_cg_w = _world_up_to_internal_vector(r_cg_w_up)
    r_strip_w = r_cg_w + aircraft.r_strip_b @ c_wb.T
    wind_cg_w, wind_strip_w = _sample_numeric_wind(
        wind_model=wind_model,
        r_cg_w=r_cg_w,
        r_strip_w=r_strip_w,
        wind_mode=wind_mode,
    )
    wind_cg_b = c_bw @ wind_cg_w
    wind_strip_b = wind_strip_w @ c_bw.T
    v_rot_b = np.cross(
        np.broadcast_to(omega_b, aircraft.r_strip_b.shape),
        aircraft.r_strip_b,
    )
    v_strip_b = v_b + v_rot_b
    v_air_strip_b = v_strip_b - wind_strip_b
    span_speed = np.sum(v_air_strip_b * aircraft.span_axis_b, axis=1)
    v_plane_b = v_air_strip_b - aircraft.span_axis_b * span_speed[:, None]
    speed_plane = np.linalg.norm(v_plane_b, axis=1)
    speed_plane_safe = np.maximum(speed_plane, EPS)
    speed_total = np.linalg.norm(v_air_strip_b, axis=1)
    speed_total_safe = np.maximum(speed_total, EPS)
    drag_dir_b = -v_plane_b / speed_plane_safe[:, None]
    lift_vec_b = aircraft.surface_normal_b - np.sum(
        aircraft.surface_normal_b * drag_dir_b,
        axis=1,
        keepdims=True,
    ) * drag_dir_b
    lift_dir_b = _normalize_rows(lift_vec_b)
    alpha_strip = np.arctan2(
        -np.sum(v_plane_b * aircraft.surface_normal_b, axis=1),
        v_plane_b[:, 0],
    )
    beta_strip = np.arcsin(np.clip(span_speed / speed_total_safe, -1.0, 1.0))
    q_bar_strip = 0.5 * rho * speed_plane**2
    delta_local = aircraft.control_mix @ delta
    cl_strip, cd_strip = _section_aero_numpy(
        alpha=alpha_strip,
        delta_local=delta_local,
        aspect_ratio=aircraft.aspect_ratio_strip,
        cd0=aircraft.cd0_strip,
        alpha0=aircraft.alpha0_strip,
        efficiency=aircraft.efficiency_strip,
        flap_scale=aircraft.flap_scale_strip,
    )
    force_scale = (q_bar_strip * aircraft.area_strip_m2)[:, None]
    f_strip_b = force_scale * (
        cl_strip[:, None] * lift_dir_b + cd_strip[:, None] * drag_dir_b
    )
    m_strip_b = np.cross(aircraft.r_strip_b, f_strip_b)
    f_aero_b = np.sum(f_strip_b, axis=0)
    m_aero_b = np.sum(m_strip_b, axis=0)
    v_air_cg_b = v_b - wind_cg_b
    speed_cg = np.linalg.norm(v_air_cg_b)
    if speed_cg > EPS:
        f_fuse_b = (
            -0.5 * rho * aircraft.drag_area_fuse_m2 * speed_cg * v_air_cg_b
        )
    else:
        f_fuse_b = np.zeros(3)
    f_aero_b = f_aero_b + f_fuse_b
    v_air_w = c_wb @ v_air_cg_b
    v_air_w_up = _internal_to_world_up_vector(v_air_w)
    r_dot_w = _internal_to_world_up_vector(c_wb @ v_b)
    alpha_cg = np.arctan2(v_air_cg_b[2], v_air_cg_b[0])
    beta_cg = np.arcsin(np.clip(v_air_cg_b[1] / max(speed_cg, EPS), -1.0, 1.0))
    gamma_rad = np.arctan2(
        v_air_w_up[2],
        max(np.linalg.norm(v_air_w_up[:2]), EPS),
    )
    sink_rate_m_s = -r_dot_w[2]
    return {
        "c_wb": c_wb,
        "f_aero_b": f_aero_b,
        "m_aero_b": m_aero_b,
        "wind_cg_w": _internal_to_world_up_vector(wind_cg_w),
        "wind_strip_w": _internal_to_world_up_rows(wind_strip_w),
        "wind_cg_b": wind_cg_b,
        "v_air_cg_b": v_air_cg_b,
        "speed_m_s": speed_cg,
        "alpha_rad": alpha_cg,
        "beta_rad": beta_cg,
        "gamma_rad": gamma_rad,
        "sink_rate_m_s": sink_rate_m_s,
        "r_dot_w": r_dot_w,
        "strips": {
            "r_strip_b": aircraft.r_strip_b.copy(),
            "r_strip_w": _internal_to_world_up_rows(r_strip_w),
            "wind_strip_b": wind_strip_b,
            "v_air_strip_b": v_air_strip_b,
            "alpha_rad": alpha_strip,
            "beta_rad": beta_strip,
            "q_bar_pa": q_bar_strip,
            "delta_local_rad": delta_local,
            "cl": cl_strip,
            "cd": cd_strip,
            "f_strip_b": f_strip_b,
            "m_strip_b": m_strip_b,
        },
    }


# =============================================================================
# 8) Public Numeric State Evaluation
# =============================================================================
def evaluate_state(
    x: np.ndarray,
    u_cmd: np.ndarray,
    aircraft: AircraftModel,
    wind_model: object = None,
    rho: float = 1.225,
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
    wind_mode: str = "panel",
) -> dict[str, object]:
    del u_cmd
    del actuator_tau_s
    x = np.asarray(x, dtype=float).reshape(15)
    aero = _evaluate_aero_numeric(
        x=x,
        aircraft=aircraft,
        wind_model=wind_model,
        rho=float(rho),
        wind_mode=wind_mode,
    )
    phi, theta, psi = x[3:6]
    v_b = x[6:9]
    omega_b = x[9:12]
    c_bw = aero["c_wb"].T
    gravity_b = c_bw @ np.array([0.0, 0.0, G_M_S2])
    f_total_b = aero["f_aero_b"] + aircraft.mass_kg * gravity_b
    m_total_b = aero["m_aero_b"]
    return {
        "x_force_b": float(f_total_b[0]),
        "y_force_b": float(f_total_b[1]),
        "z_force_b": float(f_total_b[2]),
        "l_moment_b": float(m_total_b[0]),
        "m_moment_b": float(m_total_b[1]),
        "n_moment_b": float(m_total_b[2]),
        "f_total_b": f_total_b,
        "m_total_b": m_total_b,
        "f_aero_b": aero["f_aero_b"],
        "m_aero_b": aero["m_aero_b"],
        "gravity_b": gravity_b,
        "speed_m_s": float(aero["speed_m_s"]),
        "alpha_rad": float(aero["alpha_rad"]),
        "beta_rad": float(aero["beta_rad"]),
        "gamma_rad": float(aero["gamma_rad"]),
        "sink_rate_m_s": float(aero["sink_rate_m_s"]),
        "wind_cg_w": aero["wind_cg_w"],
        "wind_cg_b": aero["wind_cg_b"],
        "r_dot_w": aero["r_dot_w"],
        "state": {
            "x_w": float(x[0]),
            "y_w": float(x[1]),
            "z_w": float(x[2]),
            "phi": float(phi),
            "theta": float(theta),
            "psi": float(psi),
            "u": float(v_b[0]),
            "v": float(v_b[1]),
            "w": float(v_b[2]),
            "p": float(omega_b[0]),
            "q": float(omega_b[1]),
            "r": float(omega_b[2]),
            "delta_a": float(x[12]),
            "delta_e": float(x[13]),
            "delta_r": float(x[14]),
        },
        "strips": aero["strips"],
    }


def state_derivative(
    x: np.ndarray,
    u_cmd: np.ndarray,
    aircraft: AircraftModel,
    wind_model: object = None,
    rho: float = 1.225,
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
    wind_mode: str = "panel",
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(15)
    u_cmd = np.asarray(u_cmd, dtype=float).reshape(3)
    loads = evaluate_state(
        x=x,
        u_cmd=u_cmd,
        aircraft=aircraft,
        wind_model=wind_model,
        rho=rho,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )
    phi, theta = x[3], x[4]
    v_b = x[6:9]
    omega_b = x[9:12]
    delta = x[12:15]
    v_dot_b = loads["f_total_b"] / aircraft.mass_kg - np.cross(omega_b, v_b)
    omega_dot_b = aircraft.inertia_inv_b @ (
        loads["m_total_b"] - np.cross(omega_b, aircraft.inertia_b @ omega_b)
    )
    euler_dot = _t_euler_numpy(phi, theta) @ omega_b
    delta_dot = (u_cmd - delta) / np.asarray(actuator_tau_s, dtype=float)
    return np.concatenate(
        [loads["r_dot_w"], euler_dot, v_dot_b, omega_dot_b, delta_dot]
    )


# =============================================================================
# 9) Symbolic Dynamics Builder
# =============================================================================
def _symbolic_wind_vectors(
    wind_mode: str,
    wind_param: ca.SX | None,
    strip_index: int,
) -> tuple[ca.SX, ca.SX]:
    if wind_mode == "none" or wind_param is None:
        zero = ca.DM.zeros(3, 1)
        return zero, zero
    if wind_mode == "cg":
        wind_cg_w = _ca_internal_from_world_up(_ca_vector3(wind_param))
        return wind_cg_w, wind_cg_w
    wind_cg_w = _ca_internal_from_world_up(_ca_vector3(wind_param[:3]))
    start = 3 + 3 * strip_index
    stop = start + 3
    wind_strip_w = _ca_internal_from_world_up(_ca_vector3(wind_param[start:stop]))
    return wind_cg_w, wind_strip_w


def _state_derivative_symbolic(
    x: ca.SX,
    u_cmd: ca.SX,
    aircraft: AircraftModel,
    rho: float,
    actuator_tau_s: tuple[float, float, float],
    wind_mode: str,
    wind_param: ca.SX | None,
) -> ca.SX:
    phi = x[3]
    theta = x[4]
    psi = x[5]
    u = x[6]
    v = x[7]
    w = x[8]
    p = x[9]
    q = x[10]
    r = x[11]
    delta = ca.reshape(x[12:15], 3, 1)
    v_b = ca.vertcat(u, v, w)
    omega_b = ca.vertcat(p, q, r)
    c_wb = _c_wb_ca(phi, theta, psi)
    c_bw = c_wb.T
    f_aero_b = ca.DM.zeros(3, 1)
    m_aero_b = ca.DM.zeros(3, 1)
    wind_cg_w, _ = _symbolic_wind_vectors(wind_mode, wind_param, 0)
    wind_cg_b = c_bw @ wind_cg_w
    for idx in range(aircraft.strip_count):
        r_strip_b = ca.DM(aircraft.r_strip_b[idx]).reshape((3, 1))
        _, wind_strip_w = _symbolic_wind_vectors(wind_mode, wind_param, idx)
        wind_strip_b = c_bw @ wind_strip_w
        v_air_strip_b = v_b + ca.cross(omega_b, r_strip_b) - wind_strip_b
        span_axis_b = ca.DM(aircraft.span_axis_b[idx]).reshape((3, 1))
        surface_normal_b = ca.DM(aircraft.surface_normal_b[idx]).reshape((3, 1))
        span_speed = ca.dot(v_air_strip_b, span_axis_b)
        v_plane_b = v_air_strip_b - span_axis_b * span_speed
        speed_plane = _ca_norm(v_plane_b)
        drag_dir_b = -v_plane_b / speed_plane
        lift_vec_b = (
            surface_normal_b
            - ca.dot(surface_normal_b, drag_dir_b) * drag_dir_b
        )
        lift_dir_b = lift_vec_b / _ca_norm(lift_vec_b)
        alpha = ca.atan2(-ca.dot(v_plane_b, surface_normal_b), v_plane_b[0])
        delta_local = ca.dot(ca.DM(aircraft.control_mix[idx]), delta)
        cl_strip, cd_strip = _section_aero_ca(
            alpha=alpha,
            delta_local=delta_local,
            aspect_ratio=float(aircraft.aspect_ratio_strip[idx]),
            cd0=float(aircraft.cd0_strip[idx]),
            alpha0=float(aircraft.alpha0_strip[idx]),
            efficiency=float(aircraft.efficiency_strip[idx]),
            flap_scale=float(aircraft.flap_scale_strip[idx]),
        )
        q_bar_strip = 0.5 * rho * speed_plane**2
        f_strip_b = (
            q_bar_strip
            * float(aircraft.area_strip_m2[idx])
            * (cl_strip * lift_dir_b + cd_strip * drag_dir_b)
        )
        f_aero_b += f_strip_b
        m_aero_b += ca.cross(r_strip_b, f_strip_b)
    v_air_cg_b = v_b - wind_cg_b
    speed_cg = _ca_norm(v_air_cg_b)
    f_fuse_b = (
        -0.5
        * rho
        * aircraft.drag_area_fuse_m2
        * speed_cg
        * v_air_cg_b
    )
    gravity_b = c_bw @ ca.DM([0.0, 0.0, G_M_S2])
    f_total_b = f_aero_b + f_fuse_b + aircraft.mass_kg * gravity_b
    v_dot_b = f_total_b / aircraft.mass_kg - ca.cross(omega_b, v_b)
    inertia_b = ca.DM(aircraft.inertia_b)
    inertia_inv_b = ca.DM(aircraft.inertia_inv_b)
    omega_dot_b = inertia_inv_b @ (
        m_aero_b - ca.cross(omega_b, inertia_b @ omega_b)
    )
    euler_dot = _t_euler_ca(phi, theta) @ omega_b
    r_dot_w = _ca_world_up_from_internal(c_wb @ v_b)
    delta_dot = (
        ca.reshape(u_cmd, 3, 1) - delta
    ) / ca.DM(np.asarray(actuator_tau_s, dtype=float).reshape(3, 1))
    return ca.vertcat(r_dot_w, euler_dot, v_dot_b, omega_dot_b, delta_dot)


def build_symbolic_dynamics(
    aircraft: AircraftModel,
    rho: float = 1.225,
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
    wind_mode: str = "panel",
) -> SymbolicDynamicsModel:
    if wind_mode not in {"none", "cg", "panel"}:
        raise ValueError("wind_mode must be 'none', 'cg', or 'panel'.")
    x = ca.SX.sym("x", 15)
    u_cmd = ca.SX.sym("u_cmd", 3)
    if wind_mode == "none":
        wind_param = None
        x_dot = _state_derivative_symbolic(
            x=x,
            u_cmd=u_cmd,
            aircraft=aircraft,
            rho=float(rho),
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
            wind_param=None,
        )
        function = ca.Function("f_dynamics", [x, u_cmd], [x_dot])
    elif wind_mode == "cg":
        wind_param = ca.SX.sym("wind_w", 3)
        x_dot = _state_derivative_symbolic(
            x=x,
            u_cmd=u_cmd,
            aircraft=aircraft,
            rho=float(rho),
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
            wind_param=wind_param,
        )
        function = ca.Function("f_dynamics", [x, u_cmd, wind_param], [x_dot])
    else:
        wind_param = ca.SX.sym("wind_w", 3 * (aircraft.strip_count + 1))
        x_dot = _state_derivative_symbolic(
            x=x,
            u_cmd=u_cmd,
            aircraft=aircraft,
            rho=float(rho),
            actuator_tau_s=actuator_tau_s,
            wind_mode=wind_mode,
            wind_param=wind_param,
        )
        function = ca.Function("f_dynamics", [x, u_cmd, wind_param], [x_dot])
    return SymbolicDynamicsModel(
        x=x,
        u_cmd=u_cmd,
        wind_param=wind_param,
        x_dot=x_dot,
        function=function,
    )
