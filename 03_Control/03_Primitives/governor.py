from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from arena import ArenaConfig, safety_margins
from feedback import speed_alpha_beta
from linearisation import STATE_INDEX
from primitive import FlightPrimitive, PrimitiveContext


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Governor dataclasses
# 2) Viability governor
# 3) Candidate metric helper
# =============================================================================

# =============================================================================
# 1) Governor Dataclasses
# =============================================================================
# Governor records separate static, rollout, and fallback decisions for rejection-log audits.
@dataclass(frozen=True)
class GovernorLimits:
    min_altitude_m: float = 0.75
    speed_bounds_m_s: tuple[float, float] = (2.5, 9.5)
    max_abs_phi_rad: float = np.deg2rad(75.0)
    max_abs_theta_rad: float = np.deg2rad(50.0)
    max_abs_alpha_rad: float = np.deg2rad(25.0)


@dataclass(frozen=True)
class GovernorDecision:
    accepted: bool
    reasons: tuple[str, ...]
    selected_primitive_name: str | None = None
    candidate_table: tuple["CandidateEvaluation", ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    primitive_name: str
    entry_set_pass: bool
    predicted_safe: bool
    rejection_reason: str | None
    score: float | None
    predicted_min_wall_distance_m: float | None
    predicted_max_alpha_deg: float | None
    predicted_terminal_speed_m_s: float | None
    selected: bool = False


# =============================================================================
# 2) Viability Governor
# =============================================================================
class ViabilityGovernor:
    def __init__(
        self,
        limits: GovernorLimits | None = None,
        arena_config: ArenaConfig | None = None,
    ):
        self.limits = limits or GovernorLimits()
        self.arena_config = arena_config or ArenaConfig()
        self.rejection_rows: list[dict[str, str | float]] = []

    def evaluate(
        self,
        scenario_id: str,
        primitive: FlightPrimitive,
        x0: np.ndarray,
        context: PrimitiveContext,
    ) -> GovernorDecision:
        x = np.asarray(x0, dtype=float).reshape(15)
        # Static checks are retained for the original single-primitive API
        unique_reasons, telemetry = self._static_reasons(primitive, x, context)
        accepted = not unique_reasons
        if not accepted:
            self._append_rejection(scenario_id, primitive.name, x, telemetry, unique_reasons)
        return GovernorDecision(accepted=accepted, reasons=unique_reasons)

    def select_primitive(
        self,
        scenario_id: str,
        primitives: Sequence[FlightPrimitive],
        x0: np.ndarray,
        context: PrimitiveContext,
        rollout_callable: Callable[[FlightPrimitive], object],
        recovery_name: str = "recovery",
    ) -> GovernorDecision:
        x = np.asarray(x0, dtype=float).reshape(15)
        evaluations: list[CandidateEvaluation] = []
        for primitive in primitives:
            # Entry and static viability checks run before rollout prediction
            reasons, telemetry = self._static_reasons(primitive, x, context)
            if reasons:
                self._append_rejection(scenario_id, primitive.name, x, telemetry, reasons)
                evaluations.append(
                    CandidateEvaluation(
                        primitive_name=primitive.name,
                        entry_set_pass=False,
                        predicted_safe=False,
                        rejection_reason="; ".join(reasons),
                        score=None,
                        predicted_min_wall_distance_m=None,
                        predicted_max_alpha_deg=None,
                        predicted_terminal_speed_m_s=None,
                    )
                )
                continue

            try:
                # Rollout is injected to keep the governor independent of scenario runners
                rollout = rollout_callable(primitive)
                metrics = dict(getattr(rollout, "metrics", rollout))
                states = getattr(rollout, "states", None)
                finite_states = True if states is None else bool(np.all(np.isfinite(states)))
            except Exception as exc:  # pragma: no cover - exercised by integration failures
                finite_states = False
                metrics = {"termination_reason": f"rollout exception: {exc}"}

            predicted_safe = bool(metrics.get("success", False)) and finite_states
            rejection_reason = None
            score = None
            if predicted_safe:
                score = self._score_candidate(metrics, context)
            else:
                rejection_reason = str(metrics.get("termination_reason") or "rollout unsafe")
                if not finite_states:
                    rejection_reason = f"{rejection_reason}; non-finite rollout state"
                self._append_rejection(
                    scenario_id,
                    primitive.name,
                    x,
                    telemetry,
                    (f"governor rollout: {rejection_reason}",),
                )

            evaluations.append(
                CandidateEvaluation(
                    primitive_name=primitive.name,
                    entry_set_pass=True,
                    predicted_safe=predicted_safe,
                    rejection_reason=rejection_reason,
                    score=score,
                    predicted_min_wall_distance_m=_optional_float(
                        metrics.get("min_wall_distance_m")
                    ),
                    predicted_max_alpha_deg=_optional_float(metrics.get("max_alpha_deg")),
                    predicted_terminal_speed_m_s=_optional_float(
                        metrics.get("terminal_speed_m_s")
                    ),
                )
            )

        selected_idx: int | None = None
        safe_candidates = [
            (idx, candidate)
            for idx, candidate in enumerate(evaluations)
            if candidate.predicted_safe and candidate.score is not None
        ]
        if safe_candidates:
            # Higher score favours wall margin and lower alpha/speed deviation
            selected_idx = max(safe_candidates, key=lambda item: (item[1].score, -item[0]))[0]
            fallback_reason = None
        else:
            # Recovery fallback is logged explicitly when no candidate is predicted safe
            selected_idx = next(
                (
                    idx
                    for idx, candidate in enumerate(evaluations)
                    if candidate.primitive_name == recovery_name
                ),
                None,
            )
            fallback_reason = (
                "no safe rollout candidate; recovery fallback selected"
                if selected_idx is not None
                else "no safe rollout candidate and no recovery fallback"
            )

        if selected_idx is not None:
            evaluations[selected_idx] = replace(evaluations[selected_idx], selected=True)
        selected_name = None if selected_idx is None else evaluations[selected_idx].primitive_name
        decision_reasons = tuple(
            candidate.rejection_reason
            for candidate in evaluations
            if candidate.rejection_reason
        )
        return GovernorDecision(
            accepted=selected_name is not None,
            reasons=decision_reasons,
            selected_primitive_name=selected_name,
            candidate_table=tuple(evaluations),
            fallback_reason=fallback_reason,
        )

    def write_rejection_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "scenario_id",
            "primitive",
            "speed_m_s",
            "altitude_m",
            "phi_deg",
            "theta_deg",
            "alpha_deg",
            "reason",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rejection_rows)

    @staticmethod
    def write_candidate_table(
        path: Path,
        candidate_table: Sequence[CandidateEvaluation],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "primitive_name",
            "entry_set_pass",
            "predicted_safe",
            "rejection_reason",
            "score",
            "predicted_min_wall_distance_m",
            "predicted_max_alpha_deg",
            "predicted_terminal_speed_m_s",
            "selected",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidate_table:
                writer.writerow(asdict(candidate))

    def _static_reasons(
        self,
        primitive: FlightPrimitive,
        x: np.ndarray,
        context: PrimitiveContext,
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        reasons = list(primitive.entry_conditions(x, context).reasons)
        if not np.all(np.isfinite(x)):
            reasons.append("governor: state contains non-finite values")

        speed, alpha, _beta = speed_alpha_beta(x)
        low, high = self.limits.speed_bounds_m_s
        if speed < low or speed > high:
            reasons.append(f"governor: speed out of bounds {speed:.3f} m/s")
        altitude = float(x[STATE_INDEX["z_w"]])
        if altitude < self.limits.min_altitude_m:
            reasons.append(f"governor: altitude below floor {altitude:.3f} m")
        if abs(float(x[STATE_INDEX["phi"]])) > self.limits.max_abs_phi_rad:
            reasons.append("governor: bank angle outside viability envelope")
        if abs(float(x[STATE_INDEX["theta"]])) > self.limits.max_abs_theta_rad:
            reasons.append("governor: pitch angle outside viability envelope")
        if abs(alpha) > self.limits.max_abs_alpha_rad:
            reasons.append("governor: angle of attack outside viability envelope")

        margins = safety_margins(x, self.arena_config)
        if not bool(margins["inside_safe_volume"]):
            reasons.append("governor: outside configured safe volume")

        telemetry = {
            "speed_m_s": float(speed),
            "altitude_m": altitude,
            "alpha_rad": float(alpha),
        }
        return tuple(dict.fromkeys(reasons)), telemetry

    def _append_rejection(
        self,
        scenario_id: str,
        primitive_name: str,
        x: np.ndarray,
        telemetry: dict[str, float],
        reasons: tuple[str, ...],
    ) -> None:
        self.rejection_rows.append(
            {
                "scenario_id": scenario_id,
                "primitive": primitive_name,
                "speed_m_s": float(telemetry["speed_m_s"]),
                "altitude_m": float(telemetry["altitude_m"]),
                "phi_deg": float(np.rad2deg(x[STATE_INDEX["phi"]])),
                "theta_deg": float(np.rad2deg(x[STATE_INDEX["theta"]])),
                "alpha_deg": float(np.rad2deg(telemetry["alpha_rad"])),
                "reason": "; ".join(reasons),
            }
        )

    @staticmethod
    def _score_candidate(
        metrics: dict[str, object],
        context: PrimitiveContext,
    ) -> float:
        min_wall = _optional_float(metrics.get("min_wall_distance_m")) or -1.0
        max_alpha = _optional_float(metrics.get("max_alpha_deg")) or 90.0
        terminal_speed = _optional_float(metrics.get("terminal_speed_m_s"))
        speed_penalty = (
            0.0
            if terminal_speed is None
            else abs(float(terminal_speed) - float(context.speed_trim_m_s))
        )
        return float(min_wall - 0.01 * max_alpha - 0.02 * speed_penalty)


# =============================================================================
# 3) Candidate Metric Helper
# =============================================================================
# Candidate metrics treat missing or non-finite values as unavailable, not safe defaults.
def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float
