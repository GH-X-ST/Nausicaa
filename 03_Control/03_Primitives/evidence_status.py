from __future__ import annotations

from pathlib import Path


EVIDENCE_STATUS_VALUES = (
    "complete",
    "accepted_fallback",
    "smoke_incomplete",
    "blocked",
    "retired_not_active",
)

REGISTRY_CLAIM_STATUS_BY_STATUS = {
    "complete": "simulation_only_registry_complete",
    "accepted_fallback": "simulation_only_registry_accepted_fallback",
    "smoke_incomplete": "simulation_only_smoke_incomplete",
    "blocked": "simulation_only_blocked",
    "retired_not_active": "retired_not_active",
}

THESIS_ELIGIBLE_EVIDENCE_STATUSES = ("complete", "accepted_fallback")

REGISTRY_BACKED_CONTROLLER_SELECTION_STATUSES = (
    "W0_W1_registry_selected",
    "W2_verified_registry_replay",
    "W3_verified_registry_replay",
)

MISSING_CONTROLLER_SELECTION_STATUSES = (
    "missing_selected_registry_entry",
    "missing_explicit_lqr_controller",
    "missing_or_invalid_source_controller_registry",
)


def registry_claim_status_for(registry_status: str) -> str:
    status = str(registry_status)
    if status not in REGISTRY_CLAIM_STATUS_BY_STATUS:
        return "simulation_only_blocked"
    return REGISTRY_CLAIM_STATUS_BY_STATUS[status]


def is_thesis_eligible_status(status: object) -> bool:
    return str(status) in THESIS_ELIGIBLE_EVIDENCE_STATUSES


def is_registry_backed_selection_status(status: object) -> bool:
    return str(status) in REGISTRY_BACKED_CONTROLLER_SELECTION_STATUSES


def is_missing_controller_status(status: object) -> bool:
    text = str(status)
    return text in MISSING_CONTROLLER_SELECTION_STATUSES or "missing" in text


def source_is_retired(path: Path | str | None) -> bool:
    if path is None:
        return False
    normalised = Path(path).as_posix()
    return "03_Control/99_Archive" in normalised or "retired_pd_contextual" in normalised
