from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flight_config import CONTROLLER_ROOT

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from directional_residual_lift_belief import (  # noqa: E402
    DirectionalResidualLiftBelief,
    DirectionalResidualObservation,
    directional_residual_lift_cell_lookup,
    directional_residual_lift_spatial_cell_lookup,
    initial_directional_residual_lift_belief,
    query_spatial_flow_belief_features_fast,
    update_directional_residual_lift_belief_batch,
)
from state_contract import STATE_INDEX  # noqa: E402


GRAVITY_M_S2 = 9.80665
MEMORY_QUERY_RADIUS_M = 0.20
PRIMITIVE_HORIZON_S = 0.10
REACHABLE_LOOKAHEAD_M = 0.80
REACHABLE_AZIMUTH_HALF_ANGLE_RAD = math.radians(35.0)
REACHABLE_ELEVATION_HALF_ANGLE_RAD = math.radians(20.0)


@dataclass(frozen=True)
class MemoryUpdateSummary:
    launch_index: int
    observation_count: int
    updated_cell_count: int
    mean_specific_energy_residual_m: float
    mean_positive_updraft_gain_m: float


@dataclass
class RealFlightMemoryState:
    enabled: bool
    launch_index: int = 0
    belief: DirectionalResidualLiftBelief = field(default_factory=initial_directional_residual_lift_belief)

    def reset(self) -> None:
        self.launch_index = 0
        self.belief = initial_directional_residual_lift_belief()

    def cell_count(self) -> int:
        return len(self.belief.cells)

    def candidate_features(
        self,
        representative: dict[str, object],
        outcome: dict[str, object],
        *,
        current_state: np.ndarray,
    ) -> dict[str, object] | None:
        if not self.enabled or not self.belief.cells:
            return _empty_candidate_features(current_state, representative)
        path = _candidate_path_points(current_state, representative, outcome)
        cell_lookup = directional_residual_lift_cell_lookup(self.belief)
        spatial_lookup = directional_residual_lift_spatial_cell_lookup(self.belief)
        query_cache: dict[tuple[float, float, float, float], dict[str, object]] = {}

        utilities: list[float] = []
        confidences: list[float] = []
        uncertainties: list[float] = []
        for point in path:
            features = _query_cached(
                self.belief,
                query_cache,
                cell_lookup=cell_lookup,
                spatial_lookup=spatial_lookup,
                x_w_m=point[0],
                y_w_m=point[1],
                z_w_m=point[2],
                direction_rad=point[3],
                launch_index=self.launch_index,
            )
            utility = _memory_utility(features)
            confidence = _confidence(features)
            utilities.append(utility)
            confidences.append(confidence)
            uncertainties.append(1.0 - confidence)

        exit_x, exit_y, exit_z, exit_direction = path[-1]
        reachable = _reachable_flow_value(
            self.belief,
            cell_lookup=cell_lookup,
            spatial_lookup=spatial_lookup,
            query_cache=query_cache,
            exit_x=exit_x,
            exit_y=exit_y,
            exit_z=exit_z,
            exit_direction=exit_direction,
            launch_index=self.launch_index,
        )
        margins = position_margin_m(np.asarray([exit_x, exit_y, exit_z], dtype=float), TRUE_SAFE_BOUNDS)
        path_utility = float(np.mean(utilities)) if utilities else 0.0
        path_confidence = float(np.mean(confidences)) if confidences else 0.0
        path_uncertainty = float(np.mean(uncertainties)) if uncertainties else 1.0
        front_progress = _front_progress(current_state[STATE_INDEX["x_w"]], max(exit_x, reachable["best_x_w_m"]))
        route_exploitation = max(path_utility, float(reachable["capped_attraction_m"]))
        route_confidence = max(path_confidence, float(reachable["confidence"]))
        information_gain = max(path_uncertainty, float(reachable["mean_uncertainty"]))
        return {
            "belief_version": self.belief.belief_version,
            "belief_candidate_path_residual_memory_active": True,
            "belief_candidate_path_probe_count": len(path),
            "belief_candidate_path_lookahead_s": PRIMITIVE_HORIZON_S,
            "belief_candidate_path_confidence": path_confidence,
            "belief_candidate_path_updraft_residual_uncapped_m": path_utility,
            "belief_candidate_path_specific_energy_residual_uncapped_m": path_utility,
            "belief_candidate_path_specific_energy_residual_cap_m": 1.0,
            "belief_candidate_path_memory_utility_m": path_utility,
            "belief_candidate_path_memory_utility_without_attraction_m": path_utility,
            "belief_candidate_path_exit_x_w_m": exit_x,
            "belief_candidate_path_exit_y_w_m": exit_y,
            "belief_candidate_path_exit_z_w_m": exit_z,
            "belief_candidate_path_exit_direction_rad": exit_direction,
            "belief_candidate_path_exit_wall_margin_m": float(margins["min_wall_margin_m"]),
            "belief_candidate_path_exit_min_margin_m": float(margins["min_margin_m"]),
            "belief_flow_map_grid_resolution_m": 0.1,
            "belief_flow_map_query_radius_m": MEMORY_QUERY_RADIUS_M,
            "belief_flow_map_reachable_attraction_m": float(reachable["capped_attraction_m"]),
            "belief_flow_map_reachable_attraction_raw_m": float(reachable["raw_attraction_m"]),
            "belief_flow_map_reachable_attraction_cap_m": 0.25,
            "belief_flow_map_reachable_attraction_confidence": float(reachable["confidence"]),
            "belief_flow_map_reachable_attraction_query_count": int(reachable["query_count"]),
            "belief_flow_map_reachable_attraction_observation_count": int(reachable["observation_count"]),
            "belief_flow_map_reachable_attraction_best_x_w_m": float(reachable["best_x_w_m"]),
            "belief_flow_map_reachable_attraction_best_y_w_m": float(reachable["best_y_w_m"]),
            "belief_flow_map_reachable_attraction_best_z_w_m": float(reachable["best_z_w_m"]),
            "belief_flow_map_reachable_attraction_lookahead_m": REACHABLE_LOOKAHEAD_M,
            "belief_flow_map_reachable_attraction_azimuth_half_angle_rad": REACHABLE_AZIMUTH_HALF_ANGLE_RAD,
            "belief_flow_map_reachable_attraction_elevation_half_angle_rad": REACHABLE_ELEVATION_HALF_ANGLE_RAD,
            "belief_flow_map_reachable_attraction_geometry": "forward_3d_cone",
            "belief_flow_map_candidate_path_uncertainty": path_uncertainty,
            "belief_flow_map_memory_guided_exploration_uncertainty": information_gain,
            "belief_flow_map_information_gain": information_gain,
            "belief_flow_map_information_gain_path_uncertainty": path_uncertainty,
            "belief_flow_map_information_gain_reachable_uncertainty": float(reachable["mean_uncertainty"]),
            "belief_flow_map_information_gain_query_count": int(reachable["query_count"]),
            "belief_flow_map_information_gain_low_confidence_query_count": int(reachable["low_confidence_query_count"]),
            "belief_flow_map_policy": "real_flight_case_local_specific_energy_map",
            "belief_flow_map_route_policy": "first_primitive_plus_reachable_cone",
            "belief_flow_map_route_horizon_primitives": 1,
            "belief_flow_map_route_probe_count": len(path) + int(reachable["query_count"]),
            "belief_flow_map_route_exploitation_m": float(route_exploitation),
            "belief_flow_map_route_information_gain": float(information_gain),
            "belief_flow_map_route_confidence": float(route_confidence),
            "belief_flow_map_route_uncertainty": float(1.0 - route_confidence),
            "belief_flow_map_route_front_progress": float(front_progress),
            "belief_flow_map_route_safe_fraction": 1.0 if float(margins["min_margin_m"]) >= 0.0 else 0.0,
            "belief_flow_map_route_best_x_w_m": max(exit_x, float(reachable["best_x_w_m"])),
            "belief_flow_map_route_best_y_w_m": float(reachable["best_y_w_m"]),
            "belief_flow_map_route_best_z_w_m": float(reachable["best_z_w_m"]),
        }

    def update_from_decision_records(self, records: list[dict[str, object]]) -> MemoryUpdateSummary:
        if not self.enabled or len(records) < 2:
            return MemoryUpdateSummary(self.launch_index, 0, self.cell_count(), 0.0, 0.0)
        observations: list[DirectionalResidualObservation] = []
        residuals: list[float] = []
        positives: list[float] = []
        for current, following in zip(records[:-1], records[1:], strict=False):
            state0 = np.asarray(current["state"], dtype=float).reshape(15)
            state1 = np.asarray(following["state"], dtype=float).reshape(15)
            energy_delta = _specific_energy_m(state1) - _specific_energy_m(state0)
            expected = float(current.get("expected_energy_residual_m", 0.0) or 0.0)
            residual = float(energy_delta) - float(expected)
            direction = math.atan2(
                float(state1[STATE_INDEX["y_w"]] - state0[STATE_INDEX["y_w"]]),
                float(state1[STATE_INDEX["x_w"]] - state0[STATE_INDEX["x_w"]]),
            )
            midpoint = 0.5 * (state0 + state1)
            observation = DirectionalResidualObservation(
                x_w_m=float(midpoint[STATE_INDEX["x_w"]]),
                y_w_m=float(midpoint[STATE_INDEX["y_w"]]),
                z_w_m=float(midpoint[STATE_INDEX["z_w"]]),
                direction_rad=direction,
                lift_residual_m_s=float((state1[STATE_INDEX["z_w"]] - state0[STATE_INDEX["z_w"]]) / max(1e-9, float(following["t_s"]) - float(current["t_s"]))),
                updraft_gain_residual_m=max(0.0, residual),
                dwell_residual_s=0.10 if residual > 0.0 else 0.0,
                specific_energy_residual_m=residual,
                observation_weight=1.0,
                history_launch_index=int(self.launch_index),
            )
            observations.append(observation)
            residuals.append(residual)
            positives.append(max(0.0, residual))
        self.belief = update_directional_residual_lift_belief_batch(self.belief, observations)
        summary = MemoryUpdateSummary(
            launch_index=int(self.launch_index),
            observation_count=len(observations),
            updated_cell_count=self.cell_count(),
            mean_specific_energy_residual_m=float(np.mean(residuals)) if residuals else 0.0,
            mean_positive_updraft_gain_m=float(np.mean(positives)) if positives else 0.0,
        )
        self.launch_index += 1
        return summary


def _empty_candidate_features(current_state: np.ndarray, representative: dict[str, object]) -> dict[str, object]:
    path = _candidate_path_points(current_state, representative, {})
    exit_x, exit_y, exit_z, exit_direction = path[-1]
    margins = position_margin_m(np.asarray([exit_x, exit_y, exit_z], dtype=float), TRUE_SAFE_BOUNDS)
    return {
        "belief_version": "real_flight_memory_inactive",
        "belief_candidate_path_confidence": 0.0,
        "belief_candidate_path_memory_utility_m": 0.0,
        "belief_candidate_path_memory_utility_without_attraction_m": 0.0,
        "belief_candidate_path_exit_x_w_m": exit_x,
        "belief_candidate_path_exit_y_w_m": exit_y,
        "belief_candidate_path_exit_z_w_m": exit_z,
        "belief_candidate_path_exit_direction_rad": exit_direction,
        "belief_candidate_path_exit_wall_margin_m": float(margins["min_wall_margin_m"]),
        "belief_candidate_path_exit_min_margin_m": float(margins["min_margin_m"]),
    }


def _candidate_path_points(
    state: np.ndarray,
    representative: dict[str, object],
    outcome: dict[str, object],
) -> list[tuple[float, float, float, float]]:
    del outcome
    x0 = float(state[STATE_INDEX["x_w"]])
    y0 = float(state[STATE_INDEX["y_w"]])
    z0 = float(state[STATE_INDEX["z_w"]])
    psi = float(state[STATE_INDEX["psi"]])
    speed = _candidate_speed(state, representative)
    bank = _candidate_bank(representative)
    yaw_rate = GRAVITY_M_S2 * math.tan(float(np.clip(bank, -0.45, 0.45))) / max(1.5, speed)
    end_direction = psi + yaw_rate * PRIMITIVE_HORIZON_S
    dz_dt = float(np.clip(state[STATE_INDEX["w"]], -0.8, 0.8))
    points = []
    for fraction in (0.0, 0.5, 1.0):
        direction = psi + (end_direction - psi) * fraction
        distance = speed * PRIMITIVE_HORIZON_S * fraction
        points.append(
            (
                float(x0 + math.cos(direction) * distance),
                float(y0 + math.sin(direction) * distance),
                float(z0 + dz_dt * PRIMITIVE_HORIZON_S * fraction),
                float(direction),
            )
        )
    return points


def _candidate_speed(state: np.ndarray, representative: dict[str, object]) -> float:
    try:
        value = float(representative.get("local_lqr_reference_speed_m_s", "nan"))
    except (TypeError, ValueError):
        value = float("nan")
    if math.isfinite(value) and value > 0.0:
        return float(np.clip(value, 1.5, 9.0))
    velocity = state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]
    return float(np.clip(np.linalg.norm(velocity), 1.5, 9.0))


def _candidate_bank(representative: dict[str, object]) -> float:
    primitive_id = str(representative.get("primitive_id", ""))
    return {
        "mild_turn_left": -0.20,
        "mild_turn_right": 0.20,
        "lift_dwell_arc": 0.22,
        "energy_retaining_bank": 0.16,
    }.get(primitive_id, 0.0)


def _specific_energy_m(state: np.ndarray) -> float:
    velocity = state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]
    return float(state[STATE_INDEX["z_w"]] + float(np.dot(velocity, velocity)) / (2.0 * GRAVITY_M_S2))


def _query_cached(
    belief: DirectionalResidualLiftBelief,
    cache: dict[tuple[float, float, float, float], dict[str, object]],
    *,
    cell_lookup: dict[tuple[int, int, int, int], object],
    spatial_lookup: dict[tuple[int, int, int], tuple[object, ...]],
    x_w_m: float,
    y_w_m: float,
    z_w_m: float,
    direction_rad: float,
    launch_index: int,
) -> dict[str, object]:
    key = (round(x_w_m, 2), round(y_w_m, 2), round(z_w_m, 2), round(direction_rad, 2))
    if key not in cache:
        cache[key] = dict(
            query_spatial_flow_belief_features_fast(
                belief,
                x_w_m=x_w_m,
                y_w_m=y_w_m,
                z_w_m=z_w_m,
                direction_rad=direction_rad,
                cell_lookup=cell_lookup,  # type: ignore[arg-type]
                spatial_cell_lookup=spatial_lookup,  # type: ignore[arg-type]
                current_history_launch_index=launch_index,
                query_radius_m=MEMORY_QUERY_RADIUS_M,
            )
        )
    return cache[key]


def _memory_utility(features: dict[str, object]) -> float:
    specific = float(features.get("belief_local_specific_energy_residual_m", features.get("belief_local_energy_residual_m", 0.0)) or 0.0)
    updraft = float(features.get("belief_local_updraft_gain_residual_m", 0.0) or 0.0)
    return float(0.75 * specific + 0.25 * updraft)


def _confidence(features: dict[str, object]) -> float:
    effective = float(features.get("belief_effective_observation_count", 0.0) or 0.0)
    uncertainty = float(features.get("belief_uncertainty", 1.0) or 1.0)
    return float(np.clip(max(1.0 - uncertainty, min(1.0, effective / 3.0)), 0.0, 1.0))


def _reachable_flow_value(
    belief: DirectionalResidualLiftBelief,
    *,
    cell_lookup: dict[tuple[int, int, int, int], object],
    spatial_lookup: dict[tuple[int, int, int], tuple[object, ...]],
    query_cache: dict[tuple[float, float, float, float], dict[str, object]],
    exit_x: float,
    exit_y: float,
    exit_z: float,
    exit_direction: float,
    launch_index: int,
) -> dict[str, float | int]:
    best_score = 0.0
    best_confidence = 0.0
    best_observation_count = 0
    best_x, best_y, best_z = exit_x, exit_y, exit_z
    query_count = 0
    uncertainty_sum = 0.0
    low_confidence_count = 0
    for distance_m, azimuth_rad, elevation_rad, weight in (
        (0.25, 0.0, 0.0, 1.0),
        (0.50, 0.0, 0.0, 0.9),
        (0.80, 0.0, 0.0, 0.75),
        (0.50, REACHABLE_AZIMUTH_HALF_ANGLE_RAD, 0.0, 0.55),
        (0.50, -REACHABLE_AZIMUTH_HALF_ANGLE_RAD, 0.0, 0.55),
        (0.50, 0.0, REACHABLE_ELEVATION_HALF_ANGLE_RAD, 0.45),
        (0.50, 0.0, -REACHABLE_ELEVATION_HALF_ANGLE_RAD, 0.45),
    ):
        direction = exit_direction + azimuth_rad
        horizontal_distance = distance_m * math.cos(elevation_rad)
        x = exit_x + math.cos(direction) * horizontal_distance
        y = exit_y + math.sin(direction) * horizontal_distance
        z = exit_z + math.sin(elevation_rad) * distance_m
        if not _inside_bounds(x, y, z):
            continue
        features = _query_cached(
            belief,
            query_cache,
            cell_lookup=cell_lookup,
            spatial_lookup=spatial_lookup,
            x_w_m=x,
            y_w_m=y,
            z_w_m=z,
            direction_rad=direction,
            launch_index=launch_index,
        )
        query_count += 1
        confidence = _confidence(features)
        uncertainty_sum += 1.0 - confidence
        if confidence < 0.5:
            low_confidence_count += 1
        utility = max(0.0, _memory_utility(features))
        score = utility * confidence * weight
        if score > best_score:
            best_score = float(score)
            best_confidence = float(confidence)
            best_observation_count = int(float(features.get("belief_observation_count", 0.0) or 0.0))
            best_x, best_y, best_z = x, y, z
    return {
        "raw_attraction_m": float(best_score),
        "capped_attraction_m": float(np.clip(best_score, 0.0, 0.25)),
        "confidence": float(best_confidence),
        "query_count": int(query_count),
        "observation_count": int(best_observation_count),
        "mean_uncertainty": float(uncertainty_sum / max(1, query_count)),
        "low_confidence_query_count": int(low_confidence_count),
        "best_x_w_m": float(best_x),
        "best_y_w_m": float(best_y),
        "best_z_w_m": float(best_z),
    }


def _inside_bounds(x_w_m: float, y_w_m: float, z_w_m: float) -> bool:
    return (
        TRUE_SAFE_BOUNDS.x_w_m[0] <= x_w_m <= TRUE_SAFE_BOUNDS.x_w_m[1]
        and TRUE_SAFE_BOUNDS.y_w_m[0] <= y_w_m <= TRUE_SAFE_BOUNDS.y_w_m[1]
        and TRUE_SAFE_BOUNDS.z_w_m[0] <= z_w_m <= TRUE_SAFE_BOUNDS.z_w_m[1]
    )


def _front_progress(x0: float, x1: float) -> float:
    return float(np.clip((float(x1) - float(x0)) / max(0.2, TRUE_SAFE_BOUNDS.x_w_m[1] - float(x0)), 0.0, 1.0))
