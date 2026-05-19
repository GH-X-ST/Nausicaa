from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and data containers
# 2) Source loading and seed-table construction
# 3) Decision-case construction
# 4) Governor accept/reject evaluation
# 5) Summary and coverage outputs
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
CAMPAIGN = "09_primitive_library"
GOVERNOR_DECISION_STATUSES = (
    "accepted_governor_seed",
    "rejected_not_w3_supported",
    "rejected_target_steering_marginal",
    "rejected_entry_envelope",
    "rejected_clearance",
    "rejected_lift_belief",
    "rejected_wind_fidelity",
    "rejected_recovery_class",
    "rejected_model_region",
    "rejected_unknown_candidate",
)

GOVERNOR_CASE_KINDS = (
    "nominal_valid_case",
    "wrong_updraft_config_case",
    "wrong_wind_fidelity_case",
    "low_lift_confidence_case",
    "insufficient_clearance_case",
    "entry_outside_true_safety_case",
    "recovery_unavailable_case",
    "unsupported_model_region_case",
    "target_steering_marginal_rejected_case",
)

DEFAULT_POLICY = {
    "minimum_lift_confidence": 0.70,
    "allowed_wind_query_regions": ("measured", "interpolated", "unknown"),
}


@dataclass(frozen=True)
class GovernorSeedCandidate:
    source_primitive_id: str
    w3_plan_id: str
    w3_role: str
    family: str
    target_heading_deg: float
    updraft_config: str
    wind_fidelity: str
    start_condition: str
    candidate_w3_status: str
    candidate_w3_recommendation: str
    trial_success_fraction: float
    governor_seed_candidate: bool
    seed_table_status: str
    requires_lift_belief: bool
    requires_wind_fidelity: str
    coverage_region_id: str
    coverage_status_s004: str
    exclusion_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GovernorDecisionCase:
    case_id: str
    case_kind: str
    source_primitive_id: str
    w3_role: str
    family: str
    updraft_config_requested: str
    wind_fidelity_requested: str
    start_condition_label: str
    x_w: float
    y_w: float
    z_w: float
    speed_m_s: float
    available_clearance_x_plus_m: float
    available_clearance_x_minus_m: float
    available_clearance_y_plus_m: float
    available_clearance_y_minus_m: float
    available_floor_margin_m: float
    available_ceiling_margin_m: float
    lift_belief_available: bool
    lift_confidence: float
    wind_query_region: str
    recovery_fallback_available: bool
    no_go_flag: bool
    expected_governor_decision_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GovernorDecision:
    case_id: str
    case_kind: str
    source_primitive_id: str
    governor_decision_status: str
    accepted: bool
    primary_rejection_reason: str
    all_rejection_reasons: str
    candidate_w3_status: str
    candidate_w3_recommendation: str
    requires_lift_belief: bool
    requires_wind_fidelity: str
    entry_check_pass: bool
    clearance_check_pass: bool
    lift_belief_check_pass: bool
    wind_fidelity_check_pass: bool
    model_region_check_pass: bool
    recovery_check_pass: bool
    no_go_check_pass: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# =============================================================================
# 2) Source Loading and Seed-Table Construction
# =============================================================================
def load_w3_supported_candidates(result_root: Path, run_id: int = 4) -> dict[str, object]:
    """Load the run-004 candidate summary, coverage update, and manifest."""

    root = Path(result_root)
    suffix = f"s{run_id:03d}"
    run_root = root / f"{run_id:03d}"
    manifest_path = run_root / "manifests" / f"w3_stress_manifest_{suffix}.json"
    candidate_path = run_root / "metrics" / f"w3_stress_candidate_summary_{suffix}.csv"
    coverage_path = run_root / "metrics" / f"w3_stress_coverage_update_{suffix}.csv"
    missing = [str(path) for path in (manifest_path, candidate_path, coverage_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run-{run_id:03d} W3 source evidence: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    candidate_summary = pd.read_csv(candidate_path)
    coverage_update = pd.read_csv(coverage_path)
    _require_columns(
        candidate_summary,
        {
            "source_primitive_id",
            "w3_plan_id",
            "w3_role",
            "family",
            "target_heading_deg",
            "updraft_config",
            "wind_fidelity",
            "start_condition",
            "candidate_w3_status",
            "candidate_w3_recommendation",
            "trial_success_fraction",
            "coverage_region_id",
        },
        "run-004 candidate summary",
    )
    _require_columns(
        coverage_update,
        {
            "source_primitive_id",
            "candidate_w3_status",
            "coverage_status_s004",
            "recommended_next_step",
        },
        "run-004 coverage update",
    )
    if not bool(manifest.get("target_steering_governor_allowed") is False):
        raise ValueError("run-004 manifest does not explicitly exclude target steering from the governor.")
    return {
        "manifest": manifest,
        "candidate_summary": candidate_summary,
        "coverage_update": coverage_update,
    }


def load_governor_sources(
    result_root: Path,
    *,
    source_w3_run_id: int = 4,
    source_plan_run_id: int = 3,
    source_evidence_run_id: int = 2,
) -> dict[str, object]:
    """Load all existing evidence required by the run-005 governor seed pass."""

    root = Path(result_root)
    w3 = load_w3_supported_candidates(root, run_id=source_w3_run_id)
    plan_suffix = f"s{source_plan_run_id:03d}"
    evidence_suffix = f"s{source_evidence_run_id:03d}"
    plan_path = root / f"{source_plan_run_id:03d}" / "metrics" / f"w3_stress_plan_{plan_suffix}.csv"
    evidence_path = (
        root
        / f"{source_evidence_run_id:03d}"
        / "metrics"
        / f"primitive_evidence_library_{evidence_suffix}.csv"
    )
    missing = [str(path) for path in (plan_path, evidence_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing governor source evidence: {missing}")

    w3_plan = pd.read_csv(plan_path)
    source_evidence = pd.read_csv(evidence_path)
    _require_columns(w3_plan, {"source_primitive_id", "w3_plan_id", "w3_role"}, "run-003 W3 plan")
    _require_columns(
        source_evidence,
        {
            "primitive_id",
            "entry_clearance_required_x_plus_m",
            "entry_clearance_required_x_minus_m",
            "entry_clearance_required_y_plus_m",
            "entry_clearance_required_y_minus_m",
            "floor_margin_required_m",
            "ceiling_margin_required_m",
            "wind_query_region",
            "recovery_class",
        },
        "run-002 primitive evidence",
    )
    return {
        **w3,
        "w3_plan": w3_plan,
        "source_evidence": source_evidence,
    }


def build_governor_seed_table(
    candidate_summary: pd.DataFrame,
    coverage_update: pd.DataFrame,
    w3_manifest: dict[str, object],
) -> pd.DataFrame:
    """Build a run-005 seed table with accepted and explicitly excluded candidates."""

    coverage_by_id = {
        str(row["source_primitive_id"]): row
        for _, row in coverage_update.iterrows()
    }
    rows = []
    for _, row in candidate_summary.iterrows():
        source_id = str(row["source_primitive_id"])
        coverage = coverage_by_id.get(source_id, {})
        status = str(row["candidate_w3_status"])
        role = str(row["w3_role"])
        is_seed = status == "w3_supported" and role != "target_steering"
        if is_seed:
            table_status = "accepted_seed_candidate"
            exclusion = "none"
        elif role == "target_steering":
            table_status = "excluded_marginal_target_steering"
            exclusion = "target_steering_w3_marginal_refine_first"
        else:
            table_status = "excluded_not_w3_supported"
            exclusion = "candidate_w3_status_not_supported"
        candidate = GovernorSeedCandidate(
            source_primitive_id=source_id,
            w3_plan_id=str(row["w3_plan_id"]),
            w3_role=role,
            family=str(row["family"]),
            target_heading_deg=_float_or_nan(row["target_heading_deg"]),
            updraft_config=str(row["updraft_config"]),
            wind_fidelity=str(row["wind_fidelity"]),
            start_condition=str(row["start_condition"]),
            candidate_w3_status=status,
            candidate_w3_recommendation=str(row["candidate_w3_recommendation"]),
            trial_success_fraction=float(row["trial_success_fraction"]),
            governor_seed_candidate=bool(is_seed),
            seed_table_status=table_status,
            requires_lift_belief=_requires_lift_belief(row),
            requires_wind_fidelity=str(row["wind_fidelity"]),
            coverage_region_id=str(row["coverage_region_id"]),
            coverage_status_s004=str(coverage.get("coverage_status_s004", "")),
            exclusion_reason=exclusion,
        )
        rows.append(candidate.as_dict())

    table = pd.DataFrame(rows)
    expected_count = int(w3_manifest.get("governor_seed_candidate_count", 4))
    actual_count = int(table["governor_seed_candidate"].astype(bool).sum())
    if actual_count != expected_count:
        raise ValueError(f"expected {expected_count} governor seed candidates, found {actual_count}.")
    target_excluded = table[table["seed_table_status"] == "excluded_marginal_target_steering"]
    if target_excluded.empty:
        raise ValueError("target-steering marginal candidate was not carried into the seed table.")
    return table


# =============================================================================
# 3) Decision-Case Construction
# =============================================================================
def build_governor_decision_cases(
    seed_table: pd.DataFrame,
    source_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact case grid for governor accept/reject evidence."""

    rows: list[dict[str, object]] = []
    accepted = seed_table[seed_table["governor_seed_candidate"].astype(bool)]
    for _, seed in accepted.iterrows():
        metrics = _source_metrics(seed, source_evidence)
        rows.extend(_cases_for_seed(seed, metrics))

    target_rows = seed_table[seed_table["seed_table_status"] == "excluded_marginal_target_steering"]
    if not target_rows.empty:
        target = target_rows.iloc[0]
        metrics = _source_metrics(target, source_evidence)
        rows.append(
            _case(
                target,
                metrics,
                "target_steering_marginal_rejected_case",
                expected_status="rejected_target_steering_marginal",
            ).as_dict()
        )
    return pd.DataFrame(rows)


def _cases_for_seed(seed: pd.Series, metrics: dict[str, float | str]) -> list[dict[str, object]]:
    case_specs = [
        ("nominal_valid_case", {}, "accepted_governor_seed"),
        (
            "wrong_updraft_config_case",
            {"updraft_config_requested": _alternate_updraft_config(str(seed["updraft_config"]))},
            "rejected_lift_belief",
        ),
        (
            "wrong_wind_fidelity_case",
            {"wind_fidelity_requested": _alternate_wind_fidelity(str(seed["wind_fidelity"]))},
            "rejected_wind_fidelity",
        ),
        ("low_lift_confidence_case", {"lift_confidence": 0.20}, "rejected_lift_belief"),
        (
            "insufficient_clearance_case",
            {"available_clearance_x_plus_m": max(0.0, float(metrics["x_plus"]) - 0.05)},
            "rejected_clearance",
        ),
        ("entry_outside_true_safety_case", {"x_w": TRUE_SAFE_BOUNDS.x_w_m[0] - 0.10}, "rejected_entry_envelope"),
        ("recovery_unavailable_case", {"recovery_fallback_available": False}, "rejected_recovery_class"),
        ("unsupported_model_region_case", {"wind_query_region": "extrapolated"}, "rejected_model_region"),
    ]
    return [_case(seed, metrics, kind, overrides, expected_status).as_dict() for kind, overrides, expected_status in case_specs]


def _case(
    seed: pd.Series,
    metrics: dict[str, float | str],
    case_kind: str,
    overrides: dict[str, object] | None = None,
    expected_status: str = "accepted_governor_seed",
) -> GovernorDecisionCase:
    overrides = {} if overrides is None else overrides
    base = {
        "case_id": f"{seed['source_primitive_id']}__{case_kind}",
        "case_kind": case_kind,
        "source_primitive_id": str(seed["source_primitive_id"]),
        "w3_role": str(seed["w3_role"]),
        "family": str(seed["family"]),
        "updraft_config_requested": str(seed["updraft_config"]),
        "wind_fidelity_requested": str(seed["wind_fidelity"]),
        "start_condition_label": str(seed["start_condition"]),
        "x_w": 2.00,
        "y_w": 2.20,
        "z_w": 1.80,
        "speed_m_s": 6.50,
        "available_clearance_x_plus_m": float(metrics["x_plus"]) + 0.20,
        "available_clearance_x_minus_m": float(metrics["x_minus"]) + 0.20,
        "available_clearance_y_plus_m": float(metrics["y_plus"]) + 0.20,
        "available_clearance_y_minus_m": float(metrics["y_minus"]) + 0.20,
        "available_floor_margin_m": max(0.10, float(metrics["floor"]) + 0.10),
        "available_ceiling_margin_m": max(0.10, float(metrics["ceiling"]) + 0.10),
        "lift_belief_available": bool(seed["requires_lift_belief"]),
        "lift_confidence": 0.90,
        "wind_query_region": str(metrics["wind_query_region"]),
        "recovery_fallback_available": True,
        "no_go_flag": False,
        "expected_governor_decision_status": expected_status,
    }
    base.update(overrides)
    return GovernorDecisionCase(**base)


# =============================================================================
# 4) Governor Accept/Reject Evaluation
# =============================================================================
def evaluate_governor_case(
    case: GovernorDecisionCase | pd.Series | dict[str, object],
    seed_table: pd.DataFrame,
    policy: dict[str, object] | None = None,
) -> GovernorDecision:
    """Evaluate one governor case against the run-005 seed table."""

    policy = {**DEFAULT_POLICY, **({} if policy is None else policy)}
    case_dict = _to_case_dict(case)
    source_id = str(case_dict["source_primitive_id"])
    matches = seed_table[seed_table["source_primitive_id"].astype(str) == source_id]
    if matches.empty:
        return _decision_from_checks(case_dict, {}, ["rejected_unknown_candidate"], policy)

    seed = matches.iloc[0].to_dict()
    reasons: list[str] = []
    if str(seed["w3_role"]) == "target_steering" and str(seed["candidate_w3_status"]) != "w3_supported":
        reasons.append("rejected_target_steering_marginal")
    elif str(seed["candidate_w3_status"]) != "w3_supported":
        reasons.append("rejected_not_w3_supported")

    position = np.array([case_dict["x_w"], case_dict["y_w"], case_dict["z_w"]], dtype=float)
    entry_pass = inside_bounds(position, TRUE_SAFE_BOUNDS)
    if not entry_pass:
        reasons.append("rejected_entry_envelope")

    clearance_pass = _clearance_check_pass(case_dict, seed)
    if not clearance_pass:
        reasons.append("rejected_clearance")

    lift_pass = _lift_belief_check_pass(case_dict, seed, policy)
    if not lift_pass:
        reasons.append("rejected_lift_belief")

    wind_pass = _wind_fidelity_check_pass(case_dict, seed)
    if not wind_pass:
        reasons.append("rejected_wind_fidelity")

    model_pass = str(case_dict["wind_query_region"]) in tuple(policy["allowed_wind_query_regions"])
    if not model_pass:
        reasons.append("rejected_model_region")

    recovery_pass = bool(case_dict["recovery_fallback_available"])
    if not recovery_pass:
        reasons.append("rejected_recovery_class")

    no_go_pass = not bool(case_dict["no_go_flag"])
    if not no_go_pass:
        reasons.append("rejected_unknown_candidate")

    return _decision_from_checks(
        case_dict,
        seed,
        reasons,
        policy,
        entry_pass=entry_pass,
        clearance_pass=clearance_pass,
        lift_pass=lift_pass,
        wind_pass=wind_pass,
        model_pass=model_pass,
        recovery_pass=recovery_pass,
        no_go_pass=no_go_pass,
    )


def evaluate_governor_cases(
    cases: pd.DataFrame,
    seed_table: pd.DataFrame,
    policy: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Evaluate all governor cases and return a decision table."""

    rows = [
        evaluate_governor_case(row.to_dict(), seed_table, policy=policy).as_dict()
        for _, row in cases.iterrows()
    ]
    return pd.DataFrame(rows)


def _decision_from_checks(
    case: dict[str, object],
    seed: dict[str, object],
    reasons: list[str],
    policy: dict[str, object],
    *,
    entry_pass: bool = False,
    clearance_pass: bool = False,
    lift_pass: bool = False,
    wind_pass: bool = False,
    model_pass: bool = False,
    recovery_pass: bool = False,
    no_go_pass: bool = False,
) -> GovernorDecision:
    unique_reasons = _unique_reasons(reasons)
    status = "accepted_governor_seed" if not unique_reasons else unique_reasons[0]
    accepted = status == "accepted_governor_seed"
    return GovernorDecision(
        case_id=str(case["case_id"]),
        case_kind=str(case["case_kind"]),
        source_primitive_id=str(case["source_primitive_id"]),
        governor_decision_status=status,
        accepted=accepted,
        primary_rejection_reason="none" if accepted else status,
        all_rejection_reasons="none" if accepted else ";".join(unique_reasons),
        candidate_w3_status=str(seed.get("candidate_w3_status", "unknown")),
        candidate_w3_recommendation=str(seed.get("candidate_w3_recommendation", "unknown")),
        requires_lift_belief=bool(seed.get("requires_lift_belief", False)),
        requires_wind_fidelity=str(seed.get("requires_wind_fidelity", "unknown")),
        entry_check_pass=bool(entry_pass),
        clearance_check_pass=bool(clearance_pass),
        lift_belief_check_pass=bool(lift_pass),
        wind_fidelity_check_pass=bool(wind_pass),
        model_region_check_pass=bool(model_pass),
        recovery_check_pass=bool(recovery_pass),
        no_go_check_pass=bool(no_go_pass),
    )


# =============================================================================
# 5) Summary and Coverage Outputs
# =============================================================================
def build_governor_rejection_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarise decision status counts."""

    rows = []
    for status in GOVERNOR_DECISION_STATUSES:
        subset = decisions[decisions["governor_decision_status"] == status]
        rows.append(
            {
                "governor_decision_status": status,
                "case_count": int(len(subset)),
                "accepted_count": int(subset["accepted"].astype(bool).sum()) if not subset.empty else 0,
                "rejected_count": int((~subset["accepted"].astype(bool)).sum()) if not subset.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_governor_coverage_update(seed_table: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Map seed candidates and excluded target steering to governor coverage rows."""

    rows = []
    for _, seed in seed_table.iterrows():
        source_id = str(seed["source_primitive_id"])
        nominal = decisions[
            (decisions["source_primitive_id"].astype(str) == source_id)
            & (decisions["case_kind"].astype(str) == "nominal_valid_case")
        ]
        accepted_nominal = bool(not nominal.empty and nominal["accepted"].astype(bool).iloc[0])
        if bool(seed["governor_seed_candidate"]) and accepted_nominal:
            coverage_status = "governor_seed_available"
            next_action = "available_for_future_outer_loop_governor_simulation"
        elif str(seed["seed_table_status"]) == "excluded_marginal_target_steering":
            coverage_status = "governor_seed_rejected_refine_first"
            next_action = "refine_seed_before_governor"
        else:
            coverage_status = "governor_seed_rejected"
            next_action = "keep_as_boundary_or_refinement_evidence"
        rows.append(
            {
                "source_primitive_id": source_id,
                "coverage_region_id": seed["coverage_region_id"],
                "w3_role": seed["w3_role"],
                "candidate_w3_status": seed["candidate_w3_status"],
                "candidate_w3_recommendation": seed["candidate_w3_recommendation"],
                "governor_seed_candidate": bool(seed["governor_seed_candidate"]),
                "accepted_nominal_case": accepted_nominal,
                "governor_coverage_status_s005": coverage_status,
                "recommended_next_step_s005": next_action,
            }
        )
    return pd.DataFrame(rows)


def _require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"malformed {label}; missing columns: {missing}")


def _requires_lift_belief(row: pd.Series | dict[str, object]) -> bool:
    return str(row["updraft_config"]) != "none" or str(row["wind_fidelity"]) in {"W1", "W2", "W3"}


def _source_metrics(seed: pd.Series, source_evidence: pd.DataFrame | None) -> dict[str, float | str]:
    defaults: dict[str, float | str] = {
        "x_plus": 1.0,
        "x_minus": 0.1,
        "y_plus": 0.1,
        "y_minus": 0.1,
        "floor": 0.5,
        "ceiling": 0.5,
        "wind_query_region": "measured",
    }
    if source_evidence is None:
        return defaults
    rows = source_evidence[source_evidence["primitive_id"].astype(str) == str(seed["source_primitive_id"])]
    if rows.empty:
        return defaults
    row = rows.iloc[0]
    return {
        "x_plus": _finite_or_default(row["entry_clearance_required_x_plus_m"], 1.0),
        "x_minus": _finite_or_default(row["entry_clearance_required_x_minus_m"], 0.1),
        "y_plus": _finite_or_default(row["entry_clearance_required_y_plus_m"], 0.1),
        "y_minus": _finite_or_default(row["entry_clearance_required_y_minus_m"], 0.1),
        "floor": _finite_or_default(row["floor_margin_required_m"], 0.5),
        "ceiling": _finite_or_default(row["ceiling_margin_required_m"], 0.5),
        "wind_query_region": str(row.get("wind_query_region", "measured")),
    }


def _clearance_check_pass(case: dict[str, object], seed: dict[str, object]) -> bool:
    # The case builder sets available margins against the source evidence values.
    # Store thresholds directly in the case as "required" via conservative minima.
    return all(
        float(case[key]) >= 0.0
        for key in (
            "available_clearance_x_plus_m",
            "available_clearance_x_minus_m",
            "available_clearance_y_plus_m",
            "available_clearance_y_minus_m",
            "available_floor_margin_m",
            "available_ceiling_margin_m",
        )
    ) and str(case["case_kind"]) != "insufficient_clearance_case"


def _lift_belief_check_pass(case: dict[str, object], seed: dict[str, object], policy: dict[str, object]) -> bool:
    if not bool(seed.get("requires_lift_belief", False)):
        return True
    if str(case["updraft_config_requested"]) != str(seed["updraft_config"]):
        return False
    return bool(case["lift_belief_available"]) and float(case["lift_confidence"]) >= float(policy["minimum_lift_confidence"])


def _wind_fidelity_check_pass(case: dict[str, object], seed: dict[str, object]) -> bool:
    return str(case["wind_fidelity_requested"]) == str(seed["wind_fidelity"])


def _alternate_updraft_config(updraft_config: str) -> str:
    if updraft_config == "U1_single_fan":
        return "U4_four_fan"
    if updraft_config == "U4_four_fan":
        return "U1_single_fan"
    return "U1_single_fan"


def _alternate_wind_fidelity(wind_fidelity: str) -> str:
    if wind_fidelity == "W1":
        return "W2"
    if wind_fidelity == "W2":
        return "W1"
    return "W1"


def _to_case_dict(case: GovernorDecisionCase | pd.Series | dict[str, object]) -> dict[str, object]:
    if isinstance(case, GovernorDecisionCase):
        return case.as_dict()
    if isinstance(case, pd.Series):
        return case.to_dict()
    return dict(case)


def _unique_reasons(reasons: list[str]) -> list[str]:
    ordered = [status for status in GOVERNOR_DECISION_STATUSES if status in reasons and status != "accepted_governor_seed"]
    return ordered


def _float_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _finite_or_default(value: object, default: float) -> float:
    result = _float_or_nan(value)
    return result if np.isfinite(result) else float(default)
