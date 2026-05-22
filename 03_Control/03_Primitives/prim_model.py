from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from env_ctx import EnvironmentContext, context_feature_vector
from prim_cat import PrimitiveDefinition
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
@dataclass(frozen=True)
class PrimitiveModelRecord:
    primitive_id: str
    context_features: tuple[float, ...]
    outcome_class: str
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
    probability_blocked: float
    predicted_energy_residual_m: float
    predicted_lift_dwell_time_s: float
    predicted_minimum_wall_margin_m: float
    predicted_termination_cause: str
    uncertainty: float
    neighbour_distance: float
    model_backend: str = "auditable_knn_table"


@dataclass(frozen=True)
class PrimitiveOutcomeModel:
    records: tuple[PrimitiveModelRecord, ...]
    k_neighbours: int = 5

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
) -> PrimitiveOutcomeModel:
    """Fit a compact table-backed predictor from rollout evidence rows."""

    records: list[PrimitiveModelRecord] = []
    for row in rows:
        outcome_class = str(row.get("outcome_class", "blocked"))
        if outcome_class not in OUTCOME_CLASSES:
            continue
        features = _parse_feature_vector(row.get("context_feature_vector", "[]"))
        if not features:
            continue
        records.append(
            PrimitiveModelRecord(
                primitive_id=str(row.get("primitive_id", "")),
                context_features=features,
                outcome_class=outcome_class,
                energy_residual_m=float(row.get("energy_residual_m", 0.0)),
                lift_dwell_time_s=float(row.get("lift_dwell_time_s", 0.0)),
                minimum_wall_margin_m=float(row.get("minimum_wall_margin_m", 0.0)),
                termination_cause=str(row.get("termination_cause", "unknown")),
            )
        )
    return PrimitiveOutcomeModel(
        records=tuple(records),
        k_neighbours=max(1, int(k_neighbours)),
    )


def predict_primitive_outcome(
    model: PrimitiveOutcomeModel,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
) -> PrimitiveOutcomePrediction:
    """Predict primitive outcome from context features without environment branching."""

    query = np.asarray(context_feature_vector(context), dtype=float)
    candidates = [
        record for record in model.records if record.primitive_id == primitive.primitive_id
    ]
    if not candidates:
        candidates = list(model.records)
    if not candidates:
        return _prior_prediction(primitive.primitive_id)

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
    for weight, record in zip(weights, neighbours, strict=True):
        probabilities[record.outcome_class] += float(weight)
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
        probability_blocked=float(probabilities["blocked"]),
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
        predicted_termination_cause=termination,
        uncertainty=float(np.mean(neighbour_distances)),
        neighbour_distance=float(neighbour_distances[0]),
    )


def _prior_prediction(primitive_id: str) -> PrimitiveOutcomePrediction:
    return PrimitiveOutcomePrediction(
        primitive_id=str(primitive_id),
        probability_accepted=0.0,
        probability_weak=0.25,
        probability_failed=0.25,
        probability_rejected=0.25,
        probability_blocked=0.25,
        predicted_energy_residual_m=0.0,
        predicted_lift_dwell_time_s=0.0,
        predicted_minimum_wall_margin_m=0.0,
        predicted_termination_cause="unfitted_model_prior",
        uncertainty=float("inf"),
        neighbour_distance=float("inf"),
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
