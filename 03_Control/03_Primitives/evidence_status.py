from __future__ import annotations

from pathlib import Path


EVIDENCE_STATUS_VALUES = (
    "complete",
    "weak",
    "failed",
    "rejected",
    "blocked",
    "retired_not_active",
)

CLAIM_STATUS_BY_STATUS = {
    "complete": "simulation_only_w01_evidence_complete",
    "weak": "simulation_only_w01_evidence_weak",
    "failed": "simulation_only_w01_evidence_failed",
    "rejected": "simulation_only_w01_entry_rejected",
    "blocked": "simulation_only_blocked",
    "retired_not_active": "retired_not_active",
}

FIXED_LQR_REPLAY_STATUSES = (
    "W01_variant_registry_candidate",
    "W2_fixed_lqr_survival_replay",
    "W3_fixed_lqr_survival_replay",
)

MISSING_CONTROLLER_STATUSES = (
    "missing_explicit_lqr_controller",
    "missing_or_invalid_primitive_variant",
)


def claim_status_for(status: str) -> str:
    return CLAIM_STATUS_BY_STATUS.get(str(status), "simulation_only_blocked")


def is_active_w01_evidence_status(status: object) -> bool:
    return str(status) in EVIDENCE_STATUS_VALUES and str(status) != "retired_not_active"


def is_fixed_lqr_replay_status(status: object) -> bool:
    return str(status) in FIXED_LQR_REPLAY_STATUSES


def is_missing_controller_status(status: object) -> bool:
    text = str(status)
    return text in MISSING_CONTROLLER_STATUSES or "missing" in text


def source_is_retired(path: Path | str | None) -> bool:
    if path is None:
        return False
    normalised = Path(path).as_posix()
    return "03_Control/99_Archive" in normalised or "retired_pd_contextual" in normalised
