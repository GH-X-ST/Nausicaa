from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from env_ctx import EnvironmentContext, context_feature_vector
from prim_cat import PrimitiveDefinition
from prim_features import (
    PRIMITIVE_FEATURE_SCHEMA_VERSION,
    primitive_feature_record,
    primitive_feature_vector_from_row,
)
from prim_roll import OUTCOME_CLASSES


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Model dataclasses
# 2) Fitting and prediction
# 3) Serialisation helpers
# =============================================================================


# =============================================================================
# 1) Model Dataclasses
# =============================================================================
DEFAULT_TRAINING_EVIDENCE_ROLES = ("feedback_rollout_candidate",)
DIAGNOSTIC_EVIDENCE_ROLES = ("diagnostic_model_rollout",)


@dataclass(frozen=True)
class PrimitiveModelRecord:
    primitive_id: str
    evidence_role: str
    context_features: tuple[float, ...]
    feature_schema_version: str
    outcome_class: str
    continuation_status: str
    episode_terminal_status: str
    episode_utility_label: str
    terminal_use_trainable: bool
    energy_residual_m: float
    lift_dwell_time_s: float
    minimum_wall_margin_m: float
    termination_cause: str


@dataclass(frozen=True)
class PrimitiveOutcomePrediction:
    primitive_id: str
    probability_accepted: float
    probability_weak: float
    probability_failed: float
    probability_rejected: float
    probability_boundary_terminal: float
    probability_blocked: float
    probability_continuation_success: float
    probability_terminal_useful: float
    predicted_energy_residual_m: float
    predicted_lift_dwell_time_s: float
    predicted_minimum_wall_margin_m: float
    predicted_continuation_margin_m: float
    predicted_termination_cause: str
    uncertainty: float
    neighbour_distance: float
    training_row_count: int
    training_evidence_roles: str
    feature_schema_version: str
    model_backend: str = "auditable_knn_table"


@dataclass(frozen=True)
class PrimitiveOutcomeModel:
    records: tuple[PrimitiveModelRecord, ...]
    k_neighbours: int = 5
    training_evidence_roles: tuple[str, ...] = DEFAULT_TRAINING_EVIDENCE_ROLES

    @property
    def fitted_row_count(self) -> int:
        return len(self.records)


# =============================================================================
# 2) Fitting and Prediction
# =============================================================================
def fit_primitive_outcome_model(
    rows: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    k_neighbours: int = 5,
    include_diagnostic: bool = False,
    allowed_evidence_roles: tuple[str, ...] | None = None,
) -> PrimitiveOutcomeModel:
    """Fit a compact table-backed predictor from rollout evidence rows."""

    evidence_roles = (
        allowed_evidence_roles
        if allowed_evidence_roles is not None
        else DEFAULT_TRAINING_EVIDENCE_ROLES
    )
    if include_diagnostic:
        evidence_roles = tuple(dict.fromkeys((*evidence_roles, *DIAGNOSTIC_EVIDENCE_ROLES)))
    records: list[PrimitiveModelRecord] = []
    for row in rows:
        evidence_role = str(row.get("evidence_role", "feedback_rollout_candidate"))
        if evidence_role not in evidence_roles:
            continue
        outcome_class = str(row.get("outcome_class", "blocked"))
        if outcome_class not in OUTCOME_CLASSES:
            continue
        features = primitive_feature_vector_from_row(row)
        if not features:
            continue
        records.append(
            PrimitiveModelRecord(
                primitive_id=str(row.get("primitive_id", "")),
                evidence_role=evidence_role,
                context_features=features,
                feature_schema_version=str(
                    row.get("feature_schema_version", PRIMITIVE_FEATURE_SCHEMA_VERSION)
                ),
                outcome_class=outcome_class,
                continuation_status=str(row.get("continuation_status", "unknown")),
                episode_terminal_status=str(row.get("episode_terminal_status", "not_terminal")),
                episode_utility_label=str(row.get("episode_utility_label", "unknown")),
                terminal_use_trainable=_parse_bool(row.get("terminal_use_trainable", False)),
                energy_residual_m=float(row.get("energy_residual_m", 0.0)),
                lift_dwell_time_s=float(row.get("lift_dwell_time_s", 0.0)),
                minimum_wall_margin_m=float(row.get("minimum_wall_margin_m", 0.0)),
                termination_cause=str(row.get("termination_cause", "unknown")),
            )
        )
    return PrimitiveOutcomeModel(
        records=tuple(records),
        k_neighbours=max(1, int(k_neighbours)),
        training_evidence_roles=tuple(evidence_roles),
    )


def predict_primitive_outcome(
    model: PrimitiveOutcomeModel,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    *,
    state: np.ndarray | None = None,
    governor_mode: str = "continuation",
) -> PrimitiveOutcomePrediction:
    """Predict primitive outcome from context features without environment branching."""

    candidates = [
        record for record in model.records if record.primitive_id == primitive.primitive_id
    ]
    if not candidates:
        candidates = list(model.records)
    if not candidates:
        return _prior_prediction(primitive.primitive_id)
    if len(candidates[0].context_features) == len(context_feature_vector(context)):
        query = np.asarray(context_feature_vector(context), dtype=float)
    else:
        query = np.asarray(
            primitive_feature_record(
                state=_state_proxy_from_context(context) if state is None else state,
                context=context,
                primitive=primitive,
                governor_mode=governor_mode,
            ).feature_vector,
            dtype=float,
        )

    distances = np.asarray(
        [
            _feature_distance(query, np.asarray(record.context_features, dtype=float))
            for record in candidates
        ],
        dtype=float,
    )
    order = np.argsort(distances)[: min(int(model.k_neighbours), distances.size)]
    neighbours = [candidates[int(index)] for index in order]
    neighbour_distances = distances[order]
    weights = 1.0 / (1.0 + neighbour_distances)
    weights = weights / np.sum(weights)
    probabilities = {label: 0.0 for label in OUTCOME_CLASSES}
    probability_continuation_success = 0.0
    probability_terminal_useful = 0.0
    for weight, record in zip(weights, neighbours, strict=True):
        probabilities[record.outcome_class] += float(weight)
        if record.continuation_status in {"continuation_success", "continuation_weak"}:
            probability_continuation_success += float(weight)
        if (
            record.terminal_use_trainable
            and record.episode_utility_label == "terminal_useful"
        ):
            probability_terminal_useful += float(weight)
    termination = _weighted_mode(
        [record.termination_cause for record in neighbours],
        weights,
    )
    return PrimitiveOutcomePrediction(
        primitive_id=primitive.primitive_id,
        probability_accepted=float(probabilities["accepted"]),
        probability_weak=float(probabilities["weak"]),
        probability_failed=float(probabilities["failed"]),
        probability_rejected=float(probabilities["rejected"]),
        probability_boundary_terminal=float(probabilities["boundary_terminal"]),
        probability_blocked=float(probabilities["blocked"]),
        probability_continuation_success=float(probability_continuation_success),
        probability_terminal_useful=float(probability_terminal_useful),
        predicted_energy_residual_m=_weighted_mean(
            [record.energy_residual_m for record in neighbours],
            weights,
        ),
        predicted_lift_dwell_time_s=_weighted_mean(
            [record.lift_dwell_time_s for record in neighbours],
            weights,
        ),
        predicted_minimum_wall_margin_m=_weighted_mean(
            [record.minimum_wall_margin_m for record in neighbours],
            weights,
        ),
        predicted_continuation_margin_m=_weighted_mean(
            [
                record.minimum_wall_margin_m
                if record.continuation_status in {"continuation_success", "continuation_weak"}
                else min(record.minimum_wall_margin_m, 0.0)
                for record in neighbours
            ],
            weights,
        ),
        predicted_termination_cause=termination,
        uncertainty=float(np.mean(neighbour_distances)),
        neighbour_distance=float(neighbour_distances[0]),
        training_row_count=model.fitted_row_count,
        training_evidence_roles=";".join(model.training_evidence_roles),
        feature_schema_version=_weighted_mode(
            [record.feature_schema_version for record in neighbours],
            weights,
        ),
    )


def _prior_prediction(primitive_id: str) -> PrimitiveOutcomePrediction:
    return PrimitiveOutcomePrediction(
        primitive_id=str(primitive_id),
        probability_accepted=0.0,
        probability_weak=0.25,
        probability_failed=0.25,
        probability_rejected=0.25,
        probability_boundary_terminal=0.0,
        probability_blocked=0.25,
        probability_continuation_success=0.25,
        probability_terminal_useful=0.0,
        predicted_energy_residual_m=0.0,
        predicted_lift_dwell_time_s=0.0,
        predicted_minimum_wall_margin_m=0.0,
        predicted_continuation_margin_m=0.0,
        predicted_termination_cause="unfitted_model_prior",
        uncertainty=float("inf"),
        neighbour_distance=float("inf"),
        training_row_count=0,
        training_evidence_roles="feedback_rollout_candidate",
        feature_schema_version=PRIMITIVE_FEATURE_SCHEMA_VERSION,
    )


def _parse_feature_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    vector = np.asarray(parsed, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        return ()
    return tuple(float(item) for item in vector)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _state_proxy_from_context(context: EnvironmentContext) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = max(1.2, 1.2 + float(context.wall_margin_m))
    state[1] = 2.2
    state[2] = 0.4 + max(float(context.floor_margin_m), 0.0)
    state[6] = 3.0 + float(context.speed_margin_m_s)
    return state


def _feature_distance(query: np.ndarray, record: np.ndarray) -> float:
    size = min(query.size, record.size)
    if size == 0:
        return float("inf")
    delta = query[:size] - record[:size]
    return float(np.linalg.norm(delta) / np.sqrt(float(size)))


def _weighted_mean(values: list[float], weights: np.ndarray) -> float:
    return float(np.dot(np.asarray(values, dtype=float), weights))


def _weighted_mode(values: list[str], weights: np.ndarray) -> str:
    scores: dict[str, float] = {}
    for value, weight in zip(values, weights, strict=True):
        scores[str(value)] = scores.get(str(value), 0.0) + float(weight)
    return max(scores, key=scores.get)


# =============================================================================
# 3) Serialisation Helpers
# =============================================================================
def primitive_prediction_row(prediction: PrimitiveOutcomePrediction) -> dict[str, object]:
    """Return one CSV-ready model prediction row."""

    return asdict(prediction)
