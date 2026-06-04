from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flight_config import CONTROLLER_ROOT, FlightRuntimeConfig
from real_flight_memory import MemoryUpdateSummary, RealFlightMemoryState

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from arena_contract import TRUE_SAFE_BOUNDS, heading_aligned_wall_margins_m, position_margin_m  # noqa: E402
from command_contract import clip_normalised_command, quantise_normalised_command_vector  # noqa: E402
from episode_selector import select_compact_representative  # noqa: E402
from latency import AGGREGATE_LIMITS, SurfaceLimit  # noqa: E402
from lqr_linearisation import LQR_STATE_MASK, lqr_speed_bin_id, reduced_state_indices  # noqa: E402
from real_flight_io import encode_arduino_command_packet  # noqa: E402
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector  # noqa: E402
from transition_labels import (  # noqa: E402
    classify_state,
    entry_classes_for_state_class,
    required_entry_role_for_state_class,
    start_family_for_state_class,
)
from viability_governor import (  # noqa: E402
    GovernorConfig,
    governor_config_from_row,
)


@dataclass(frozen=True)
class FlightControllerDecision:
    selected: bool
    reason: str
    command_norm: tuple[float, float, float]
    command_rad: tuple[float, float, float]
    primitive_variant_id: str
    primitive_id: str
    controller_id: str
    governor_mode: str
    start_state_family: str
    current_state_class: str
    candidate_count: int
    viable_count: int
    expected_energy_residual_m: float
    expected_updraft_gain_proxy_m: float
    expected_lift_dwell_time_s: float
    memory_enabled: bool
    memory_cell_count: int
    decision_time_s: float
    packet_bytes: bytes


class FrozenFlightController:
    """Self-contained deployment copy of the frozen R8/R10 controller interface."""

    def __init__(self, config: FlightRuntimeConfig) -> None:
        self.config = config
        self.representatives = _load_representatives(config.library_manifest_path)
        self.outcomes = _load_outcome_rows(config.outcome_table_path)
        self.controllers = _load_controller_payloads(config.controller_bundle_path)
        self.governor_config = _load_governor_config(config.governor_config_path)
        self._command_fifo_by_controller_id: dict[str, list[np.ndarray]] = {}
        self._active_payload: dict[str, Any] | None = None
        self._active_variant_id = ""
        self._last_command_norm = np.zeros(3, dtype=float)
        self.sequence = 0
        self.memory_state = RealFlightMemoryState(enabled=bool(config.experiment_memory_enabled))

    def decide(self, state_vector: np.ndarray, *, primitive_step_index: int) -> FlightControllerDecision:
        started = time.perf_counter()
        state = as_state_vector(state_vector)
        route = _validation_route_for_primitive_step(primitive_step_index, state=state)
        governor_mode = _governor_mode_for_route(route)
        context = _live_context_row(
            state,
            library_tier=self.config.library_tier,
            primitive_step_index=primitive_step_index,
            route=route,
            memory_enabled=bool(self.memory_state.enabled),
            memory_launch_index=int(self.memory_state.launch_index),
        )
        selected, candidate_rows = select_compact_representative(
            representatives=self.representatives,
            outcome_rows_by_variant_id=self.outcomes,
            context=context,
            governor_mode=governor_mode,
            policy_id="real_flight_case_local_memory" if self.memory_state.enabled else "real_flight_no_cross_case_memory",
            belief_features=None,
            candidate_belief_features=(
                lambda representative, outcome: self.memory_state.candidate_features(
                    representative,
                    outcome,
                    current_state=state,
                )
            )
            if self.memory_state.enabled
            else None,
            adaptive_memory_active=bool(self.memory_state.enabled),
            governor_config=self.governor_config,
            candidate_row_mode="controller",
        )
        if selected is None:
            return self._neutral_decision(
                reason="no_viable_primitive",
                started=started,
                route=route,
                governor_mode=governor_mode,
                candidate_count=len(candidate_rows),
                viable_count=0,
            )
        variant_id = str(selected.get("primitive_variant_id", ""))
        payload = self.controllers.get(variant_id)
        if payload is None:
            return self._neutral_decision(
                reason=f"missing_frozen_controller_payload:{variant_id}",
                started=started,
                route=route,
                governor_mode=governor_mode,
                candidate_count=len(candidate_rows),
                viable_count=sum(1 for row in candidate_rows if bool(row.get("viable", False))),
            )
        self._active_payload = payload
        self._active_variant_id = variant_id
        command_norm, command_rad = self._command_for_payload(payload, state)
        packet = encode_arduino_command_packet(command_norm, sequence=self.sequence)
        self.sequence += 1
        self._last_command_norm = np.asarray(packet.aggregate_command_norm, dtype=float)
        return FlightControllerDecision(
            selected=True,
            reason="selected",
            command_norm=tuple(float(value) for value in packet.aggregate_command_norm),
            command_rad=tuple(float(value) for value in command_rad),
            primitive_variant_id=variant_id,
            primitive_id=str(selected.get("primitive_id", payload.get("primitive_id", ""))),
            controller_id=str(payload.get("controller_id", "")),
            governor_mode=governor_mode,
            start_state_family=str(route["start_state_family"]),
            current_state_class=str(route["current_state_class"]),
            candidate_count=len(candidate_rows),
            viable_count=sum(1 for row in candidate_rows if bool(row.get("viable", False))),
            expected_energy_residual_m=_safe_float(selected.get("expected_energy_residual_m", 0.0)),
            expected_updraft_gain_proxy_m=_safe_float(selected.get("expected_updraft_gain_proxy_m", 0.0)),
            expected_lift_dwell_time_s=_safe_float(selected.get("expected_lift_dwell_time_s", 0.0)),
            memory_enabled=bool(self.memory_state.enabled),
            memory_cell_count=int(self.memory_state.cell_count()),
            decision_time_s=time.perf_counter() - started,
            packet_bytes=packet.packet_bytes,
        )

    def last_command_norm(self) -> np.ndarray:
        return self._last_command_norm.copy()

    def neutral_packet(self) -> bytes:
        packet = encode_arduino_command_packet(np.zeros(3), sequence=self.sequence)
        self.sequence += 1
        self._last_command_norm = np.zeros(3, dtype=float)
        return packet.packet_bytes

    def packet_for_last_command(self) -> bytes:
        packet = encode_arduino_command_packet(self._last_command_norm, sequence=self.sequence)
        self.sequence += 1
        return packet.packet_bytes

    def packet_for_active_slot_command(self, state_vector: np.ndarray) -> bytes:
        """Return a 20 ms slot command for the currently selected primitive."""

        if self._active_payload is None:
            return self.neutral_packet()
        command_norm, _ = self._command_for_payload(self._active_payload, as_state_vector(state_vector))
        packet = encode_arduino_command_packet(command_norm, sequence=self.sequence)
        self.sequence += 1
        self._last_command_norm = np.asarray(packet.aggregate_command_norm, dtype=float)
        return packet.packet_bytes

    def update_memory_from_decision_records(self, records: list[dict[str, object]]) -> MemoryUpdateSummary:
        return self.memory_state.update_from_decision_records(records)

    def memory_summary(self) -> dict[str, object]:
        return {
            "memory_enabled": bool(self.memory_state.enabled),
            "memory_launch_index": int(self.memory_state.launch_index),
            "memory_update_count": int(self.memory_state.belief.update_count),
            "memory_cell_count": int(self.memory_state.cell_count()),
            "memory_policy": "case_local_real_flight_specific_energy_residual_map"
            if self.memory_state.enabled
            else "disabled",
        }

    def _neutral_decision(
        self,
        *,
        reason: str,
        started: float,
        route: dict[str, object],
        governor_mode: str,
        candidate_count: int,
        viable_count: int,
    ) -> FlightControllerDecision:
        self._active_payload = None
        self._active_variant_id = ""
        packet = encode_arduino_command_packet(np.zeros(3), sequence=self.sequence)
        self.sequence += 1
        self._last_command_norm = np.zeros(3, dtype=float)
        return FlightControllerDecision(
            selected=False,
            reason=reason,
            command_norm=(0.0, 0.0, 0.0),
            command_rad=(0.0, 0.0, 0.0),
            primitive_variant_id="",
            primitive_id="",
            controller_id="",
            governor_mode=governor_mode,
            start_state_family=str(route.get("start_state_family", "")),
            current_state_class=str(route.get("current_state_class", "")),
            candidate_count=int(candidate_count),
            viable_count=int(viable_count),
            expected_energy_residual_m=0.0,
            expected_updraft_gain_proxy_m=0.0,
            expected_lift_dwell_time_s=0.0,
            memory_enabled=bool(self.memory_state.enabled),
            memory_cell_count=int(self.memory_state.cell_count()),
            decision_time_s=time.perf_counter() - started,
            packet_bytes=packet.packet_bytes,
        )

    def _command_for_payload(self, payload: dict[str, Any], state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.asarray(payload["reference_state_vector"], dtype=float).reshape(STATE_SIZE)
        u_ref = np.asarray(payload["reference_command_vector"], dtype=float).reshape(3)
        x_error_reduced = state[list(reduced_state_indices())] - x_ref[list(reduced_state_indices())]
        x_error_reduced = _passive_longitudinal_speed_error_reduced(x_error_reduced)
        predictor_a = _matrix_from_json(payload.get("predictor_A_reduced_json", ""))
        if predictor_a is not None:
            for _ in range(max(0, int(float(payload.get("predictor_horizon_steps", 0))))):
                x_error_reduced = predictor_a @ x_error_reduced
            x_error_reduced = _passive_longitudinal_speed_error_reduced(x_error_reduced)

        delay_steps = max(0, int(float(payload.get("command_delay_steps", 0))))
        augmented_gain = _matrix_from_json(payload.get("augmented_gain_matrix_json", ""))
        if delay_steps and augmented_gain is not None:
            controller_id = str(payload.get("controller_id", ""))
            fifo = self._command_fifo_by_controller_id.setdefault(
                controller_id,
                [u_ref.copy() for _ in range(delay_steps)],
            )
            fifo_array = _normalised_fifo_rad(fifo, fifo_steps=delay_steps, reference_command_rad=u_ref)
            fifo_error = (fifo_array - u_ref.reshape(1, 3)).reshape(-1)
            augmented_error = np.concatenate([x_error_reduced, fifo_error])
            raw_rad = u_ref - augmented_gain @ augmented_error
        else:
            full_gain = np.asarray(payload["k_gain_matrix"], dtype=float).reshape(3, STATE_SIZE)
            raw_rad = u_ref - full_gain @ _passive_longitudinal_speed_error_full(state - x_ref)

        raw_norm = _surface_rad_to_unclipped_norm(raw_rad)
        clipped_norm = clip_normalised_command(raw_norm)
        quantised_norm = quantise_normalised_command_vector(clipped_norm)
        command_rad = _normalised_command_to_surface_rad(quantised_norm)
        self._push_command_fifo(str(payload.get("controller_id", "")), command_rad, delay_steps)
        return quantised_norm, command_rad

    def _push_command_fifo(self, controller_id: str, command_rad: np.ndarray, delay_steps: int) -> None:
        if delay_steps <= 0 or not controller_id:
            return
        fifo = self._command_fifo_by_controller_id.setdefault(controller_id, [])
        # Old-to-new order matches the augmented LQR delay-line state: index 0
        # is the delayed/applied command and the newest request is at the tail.
        fifo.append(np.asarray(command_rad, dtype=float).reshape(3))
        del fifo[:-delay_steps]


def _load_representatives(path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    return [dict(row) for row in payload.get("representatives", [])]


def _load_outcome_rows(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with Path(path).open("r", newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            _add_outcome_key(rows, row, str(row.get("compact_library_id", "")))
            _add_outcome_key(rows, row, str(row.get("primitive_variant_id", "")))
            _add_outcome_key(rows, row, str(row.get("transition_object_id", "")))
            _add_outcome_key(rows, row, f"{row.get('library_size_case_id', '')}|{row.get('compact_library_id', '')}")
            _add_outcome_key(
                rows,
                row,
                f"{row.get('library_size_case_id', '')}|{row.get('transition_object_id', '')}|{row.get('compact_library_id', '')}",
            )
            _add_outcome_key(
                rows,
                row,
                f"{row.get('library_size_case_id', '')}|{row.get('primitive_variant_id', '')}|{row.get('compact_library_id', '')}",
            )
    return rows


def _add_outcome_key(rows: dict[str, dict[str, object]], row: dict[str, object], key: str) -> None:
    if key and key not in rows:
        rows[key] = dict(row)


def _load_controller_payloads(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    controllers: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        if str(record.get("bundle_status", "")) != "ready":
            continue
        controller_payload = dict(record.get("controller_payload", {}))
        controller_payload["k_gain_matrix"] = _json_array(controller_payload.get("k_gain_matrix"))
        controller_payload["reference_state_vector"] = _json_array(controller_payload.get("reference_state_vector"))
        controller_payload["reference_command_vector"] = _json_array(controller_payload.get("reference_command_vector"))
        controllers[str(record.get("primitive_variant_id", ""))] = controller_payload
    return controllers


def _load_governor_config(path: Path) -> GovernorConfig:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    return governor_config_from_row(dict(payload.get("governor_config", payload)))


def _json_array(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _validation_route_for_primitive_step(primitive_step_index: int, *, state: np.ndarray) -> dict[str, object]:
    if int(primitive_step_index) == 0:
        state_class = "launch_gate"
        start_family = "launch_gate"
        reason = "first_0p10s_launch_window"
    else:
        state_class = classify_state(
            state,
            primitive_step_index=int(primitive_step_index),
            allow_post_launch_degraded=int(primitive_step_index) == 1,
        )
        start_family = start_family_for_state_class(state_class)
        reason = f"live_state_routed_{state_class}"
    required_classes = entry_classes_for_state_class(state_class)
    return {
        "current_state_class": state_class,
        "start_state_family": start_family,
        "launch_sequence_phase": "first_0p10s_launch_entry" if int(primitive_step_index) == 0 else "post_launch_inflight",
        "route_required_entry_role": required_entry_role_for_state_class(state_class) or "transition_object",
        "route_required_entry_class": required_classes[0] if required_classes else "",
        "route_reason": reason,
    }


def _governor_mode_for_route(route: dict[str, object]) -> str:
    if str(route.get("route_required_entry_class", "")) in {"boundary_near", "recoverable_degraded"}:
        return "terminal_episode_mode"
    return "continuation_mode"


def _live_context_row(
    state: np.ndarray,
    *,
    library_tier: str,
    primitive_step_index: int,
    route: dict[str, object],
    memory_enabled: bool,
    memory_launch_index: int,
) -> dict[str, object]:
    position = state[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]]
    margins = position_margin_m(position, TRUE_SAFE_BOUNDS)
    heading_margins = heading_aligned_wall_margins_m(position, state[STATE_INDEX["psi"]], TRUE_SAFE_BOUNDS)
    speed = float(np.linalg.norm(state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]))
    return {
        "context_id": f"real_flight_live_ctx{int(primitive_step_index):03d}",
        "W_layer": "Reality",
        "environment_mode": "unknown_real_flight",
        "environment_instance_id": "real_flight_vicon",
        "environment_block_id": "real_flight",
        "outer_case_type": "real_flight",
        "fan_layout_policy": "unknown_real_flight",
        "scheduled_active_fan_count": "",
        "actual_active_fan_count": "",
        "active_fan_count_policy": "unknown_real_flight",
        "fan_position_policy": "unknown_real_flight",
        "environment_seed": -1,
        "start_state_family": str(route["start_state_family"]),
        "primitive_step_index": int(primitive_step_index),
        "launch_sequence_policy": "real_flight_receding_horizon_0p10s",
        "launch_sequence_phase": str(route["launch_sequence_phase"]),
        "route_required_entry_role": str(route["route_required_entry_role"]),
        "route_required_entry_class": str(route["route_required_entry_class"]),
        "route_reason": str(route["route_reason"]),
        "current_state_class": str(route["current_state_class"]),
        "transition_current_state_class": str(route["current_state_class"]),
        "latency_case": "nominal",
        "plant_W_layer": "Reality",
        "implementation_W_layer": "Reality",
        "current_x_w_m": float(position[0]),
        "current_y_w_m": float(position[1]),
        "current_z_w_m": float(position[2]),
        "current_speed_m_s": speed,
        "current_local_lqr_speed_bin_id": lqr_speed_bin_id(speed),
        "mission_x_min_w_m": float(TRUE_SAFE_BOUNDS.x_w_m[0]),
        "front_wall_target_x_w_m": float(TRUE_SAFE_BOUNDS.x_w_m[1]),
        "mission_terminal_y_min_m": float(TRUE_SAFE_BOUNDS.y_w_m[0]),
        "mission_terminal_y_max_m": float(TRUE_SAFE_BOUNDS.y_w_m[1]),
        "mission_terminal_z_min_m": float(TRUE_SAFE_BOUNDS.z_w_m[0]),
        "mission_terminal_z_max_m": float(TRUE_SAFE_BOUNDS.z_w_m[1]),
        "mission_terminal_specific_energy_reference_m": 1.0,
        "wall_margin_m": float(margins["min_wall_margin_m"]),
        "all_wall_margin_m": float(heading_margins["all_wall_margin_m"]),
        "front_wall_margin_m": float(heading_margins["front_wall_margin_m"]),
        "left_wall_margin_m": float(heading_margins["left_wall_margin_m"]),
        "right_wall_margin_m": float(heading_margins["right_wall_margin_m"]),
        "rear_wall_margin_m": float(heading_margins["rear_wall_margin_m"]),
        "governor_wall_margin_m": float(heading_margins["governor_wall_margin_m"]),
        "floor_margin_m": float(margins["floor_margin_m"]),
        "ceiling_margin_m": float(margins["ceiling_margin_m"]),
        "w_wing_mean_m_s": 0.0,
        "w_local_uncertainty_m_s": 1.0,
        "lift_score": 0.0,
        "fan_count": 0,
        "updraft_model_id": "real_flight_unknown_flow",
        "library_size_case_id": str(library_tier),
        "history_length": int(memory_launch_index) if memory_enabled else 0,
        "adaptation_launch_index": int(memory_launch_index) if memory_enabled else 0,
        "policy_id": "real_flight_case_local_memory" if memory_enabled else "real_flight_no_cross_case_memory",
    }


def _matrix_from_json(value: object) -> np.ndarray | None:
    if not value:
        return None
    try:
        return np.asarray(json.loads(str(value)), dtype=float)
    except Exception:
        return None


def _passive_longitudinal_speed_error_reduced(error: np.ndarray) -> np.ndarray:
    result = np.asarray(error, dtype=float).reshape(len(LQR_STATE_MASK)).copy()
    if "u" in LQR_STATE_MASK:
        result[LQR_STATE_MASK.index("u")] = 0.0
    return result


def _passive_longitudinal_speed_error_full(error: np.ndarray) -> np.ndarray:
    result = np.asarray(error, dtype=float).reshape(STATE_SIZE).copy()
    result[STATE_INDEX["u"]] = 0.0
    return result


def _normalised_fifo_rad(
    command_fifo_rad: list[np.ndarray],
    *,
    fifo_steps: int,
    reference_command_rad: np.ndarray,
) -> np.ndarray:
    fifo = np.asarray(command_fifo_rad, dtype=float)
    if fifo.size == 0:
        fifo = np.tile(np.asarray(reference_command_rad, dtype=float).reshape(1, 3), (int(fifo_steps), 1))
    if fifo.ndim == 1:
        fifo = fifo.reshape(1, 3)
    if fifo.shape[0] < int(fifo_steps):
        pad = np.tile(fifo[-1].reshape(1, 3), (int(fifo_steps) - fifo.shape[0], 1))
        fifo = np.vstack([fifo, pad])
    return fifo[: int(fifo_steps), :]


def _surface_rad_to_unclipped_norm(command_rad: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _angle_to_command_norm_unclipped(value, AGGREGATE_LIMITS[name])
            for value, name in zip(command_rad, ("delta_a", "delta_e", "delta_r"), strict=True)
        ],
        dtype=float,
    )


def _normalised_command_to_surface_rad(command_norm: np.ndarray) -> np.ndarray:
    values = []
    for value, name in zip(command_norm, ("delta_a", "delta_e", "delta_r"), strict=True):
        limit = AGGREGATE_LIMITS[name]
        angle_deg = float(value) * (float(limit.positive_deg) if float(value) >= 0.0 else abs(float(limit.negative_deg)))
        values.append(float(np.deg2rad(angle_deg)))
    return np.asarray(values, dtype=float)


def _angle_to_command_norm_unclipped(angle_rad: float, limit: SurfaceLimit) -> float:
    angle_deg = float(np.rad2deg(angle_rad))
    if abs(angle_deg) <= 1e-12:
        return 0.0
    candidates: list[float] = []
    if abs(float(limit.positive_deg)) > 1e-12:
        positive_ratio = angle_deg / float(limit.positive_deg)
        if positive_ratio >= 0.0:
            candidates.append(float(positive_ratio))
    if abs(float(limit.negative_deg)) > 1e-12:
        negative_ratio = angle_deg / float(limit.negative_deg)
        if negative_ratio >= 0.0:
            candidates.append(float(-negative_ratio))
    if candidates:
        return min(candidates, key=lambda value: abs(abs(value) - 1.0) if abs(value) > 1.0 else abs(value))
    return 1.0 if abs(angle_deg - float(limit.positive_deg)) <= abs(angle_deg - float(limit.negative_deg)) else -1.0
