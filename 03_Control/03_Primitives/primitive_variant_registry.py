from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from dense_archive_table_io import filesystem_path
from lqr_controller import LQRController
from prim_cat import PrimitiveDefinition


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Variant schema and entry-role policy
# 2) Variant construction and validation
# 3) Registry serialisation helpers
# =============================================================================


# =============================================================================
# 1) Variant Schema and Entry-Role Policy
# =============================================================================
VARIANT_REGISTRY_VERSION = "w01_primitive_controller_variant_registry_v1"
EXIT_CHECK_VERSION = "w01_exit_checks_v1"
ENTRY_ROLES = ("launch_capable", "inflight_only", "terminal_or_recovery")
ENTRY_ROLE_BY_PRIMITIVE_ID = {
    "glide": "launch_capable",
    "lift_entry": "inflight_only",
    "lift_dwell_arc": "inflight_only",
    "mild_turn_left": "inflight_only",
    "mild_turn_right": "inflight_only",
    "energy_retaining_bank": "inflight_only",
    "recovery": "terminal_or_recovery",
    "safe_exit_or_recovery_handoff": "terminal_or_recovery",
}
ENTRY_ROLE_REJECTION_STATUS = "entry_role_incompatible_start"
ENTRY_ROLE_REJECTION_LABEL = "entry_role_not_launch_capable"


@dataclass(frozen=True)
class PrimitiveControllerVariant:
    primitive_variant_id: str
    primitive_id: str
    primitive_family: str
    entry_role: str
    controller_id: str
    controller_family: str
    reference_state_vector: str
    reference_command_vector: str
    finite_horizon_s: float
    linearisation_id: str
    linearisation_source: str
    Q_weight_json: str
    R_weight_json: str
    K_gain_matrix_json: str
    K_gain_checksum: str
    closed_loop_eigenvalue_summary: str
    sampled_data_check_status: str
    lqr_synthesis_status: str
    lqr_blocked_reason: str
    exit_checks: str
    metrics: str
    failure_labels: str
    claim_status: str
    variant_registry_version: str = VARIANT_REGISTRY_VERSION
    exit_check_version: str = EXIT_CHECK_VERSION


# =============================================================================
# 2) Variant Construction and Validation
# =============================================================================
def entry_role_for_primitive_id(primitive_id: str) -> str:
    try:
        return ENTRY_ROLE_BY_PRIMITIVE_ID[str(primitive_id)]
    except KeyError as exc:
        raise KeyError(f"unknown W01 primitive entry role: {primitive_id}") from exc


def start_family_is_compatible(*, entry_role: str, start_state_family: str) -> bool:
    role = str(entry_role)
    family = str(start_state_family)
    if role == "launch_capable":
        return True
    if role == "inflight_only":
        return family != "launch_gate"
    if role == "terminal_or_recovery":
        return family in {"inflight_boundary_near", "inflight_recovery_edge"}
    raise ValueError(f"unknown entry_role: {entry_role}")


def entry_role_rejection_fields(*, entry_role: str, start_state_family: str) -> dict[str, object]:
    if start_family_is_compatible(entry_role=entry_role, start_state_family=start_state_family):
        return {
            "entry_role_compatible": True,
            "entry_check_status": "entry_role_compatible",
            "entry_rejection_class": "",
            "outcome_class": "",
            "failure_label": "",
        }
    return {
        "entry_role_compatible": False,
        "entry_check_status": ENTRY_ROLE_REJECTION_STATUS,
        "entry_rejection_class": ENTRY_ROLE_REJECTION_LABEL,
        "outcome_class": "rejected",
        "failure_label": ENTRY_ROLE_REJECTION_LABEL,
    }


def primitive_controller_variant(
    *,
    primitive: PrimitiveDefinition,
    controller: LQRController,
    entry_role: str | None = None,
) -> PrimitiveControllerVariant:
    role = str(entry_role or entry_role_for_primitive_id(primitive.primitive_id))
    if role not in ENTRY_ROLES:
        raise ValueError(f"entry_role must be one of {ENTRY_ROLES}")
    k_json = json.dumps(controller.k_gain_matrix, separators=(",", ":"))
    reference_state_json = json.dumps(list(controller.reference_state_vector), separators=(",", ":"))
    reference_command_json = json.dumps(list(controller.reference_command_vector), separators=(",", ":"))
    checksum = controller.lqr_gain_checksum
    variant_id = _variant_id(
        primitive_id=primitive.primitive_id,
        entry_role=role,
        finite_horizon_s=primitive.finite_horizon_s,
        reference_state_vector=reference_state_json,
        reference_command_vector=reference_command_json,
        controller_id=controller.controller_id,
        linearisation_id=controller.linearisation_id,
        q_json=controller.lqr_Q_weights_json,
        r_json=controller.lqr_R_weights_json,
        gain_checksum=checksum,
        exit_check_version=EXIT_CHECK_VERSION,
    )
    return PrimitiveControllerVariant(
        primitive_variant_id=variant_id,
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        entry_role=role,
        controller_id=controller.controller_id,
        controller_family=controller.controller_family,
        reference_state_vector=reference_state_json,
        reference_command_vector=reference_command_json,
        finite_horizon_s=float(primitive.finite_horizon_s),
        linearisation_id=controller.linearisation_id,
        linearisation_source=controller.linearisation_source,
        Q_weight_json=controller.lqr_Q_weights_json,
        R_weight_json=controller.lqr_R_weights_json,
        K_gain_matrix_json=k_json,
        K_gain_checksum=checksum,
        closed_loop_eigenvalue_summary=controller.lqr_closed_loop_eigenvalue_summary,
        sampled_data_check_status=controller.sampled_data_check_status,
        lqr_synthesis_status=controller.lqr_synthesis_status,
        lqr_blocked_reason=controller.lqr_blocked_reason,
        exit_checks=";".join(primitive.exit_checks),
        metrics=";".join(primitive.metrics_to_record),
        failure_labels=";".join(primitive.failure_labels),
        claim_status=primitive.claim_status,
    )


def variants_for_controllers(
    primitive_controller_pairs: Iterable[tuple[PrimitiveDefinition, LQRController]],
) -> tuple[PrimitiveControllerVariant, ...]:
    return tuple(
        primitive_controller_variant(primitive=primitive, controller=controller)
        for primitive, controller in primitive_controller_pairs
    )


def validate_variant_controller_match(
    variant: PrimitiveControllerVariant,
    controller: LQRController,
) -> None:
    if variant.controller_id != controller.controller_id:
        raise ValueError("variant/controller controller_id mismatch")
    if variant.K_gain_checksum != controller.lqr_gain_checksum:
        raise ValueError("variant/controller gain checksum mismatch")
    if variant.linearisation_id != controller.linearisation_id:
        raise ValueError("variant/controller linearisation_id mismatch")


# =============================================================================
# 3) Registry Serialisation Helpers
# =============================================================================
def variant_row(variant: PrimitiveControllerVariant) -> dict[str, object]:
    return asdict(variant)


def write_variant_registry(
    *,
    variants: Iterable[PrimitiveControllerVariant],
    csv_path: Path,
    json_path: Path,
) -> None:
    rows = [variant_row(variant) for variant in variants]
    filesystem_path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(json_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filesystem_path(csv_path), index=False)
    filesystem_path(json_path).write_text(
        json.dumps(
            {
                "registry_version": VARIANT_REGISTRY_VERSION,
                "variant_count": len(rows),
                "primitive_count": len({str(row["primitive_id"]) for row in rows}),
                "entry_roles": ENTRY_ROLE_BY_PRIMITIVE_ID,
                "variants": rows,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="ascii",
    )


def load_variant_registry(path: Path | str) -> dict[str, PrimitiveControllerVariant]:
    registry_path = filesystem_path(Path(path))
    if registry_path.suffix.lower() == ".json":
        payload = json.loads(registry_path.read_text(encoding="ascii"))
        rows = payload.get("variants", [])
    else:
        rows = pd.read_csv(registry_path).to_dict(orient="records")
    variants = [PrimitiveControllerVariant(**_normalise_row(row)) for row in rows]
    return {variant.primitive_variant_id: variant for variant in variants}


def _normalise_row(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    out["finite_horizon_s"] = float(out["finite_horizon_s"])
    return out


def _variant_id(
    *,
    primitive_id: str,
    entry_role: str,
    finite_horizon_s: float,
    reference_state_vector: str,
    reference_command_vector: str,
    controller_id: str,
    linearisation_id: str,
    q_json: str,
    r_json: str,
    gain_checksum: str,
    exit_check_version: str,
) -> str:
    payload = {
        "primitive_id": primitive_id,
        "entry_role": entry_role,
        "finite_horizon_s": float(finite_horizon_s),
        "reference_state_vector": reference_state_vector,
        "reference_command_vector": reference_command_vector,
        "controller_id": controller_id,
        "linearisation_id": linearisation_id,
        "Q_weight_json": q_json,
        "R_weight_json": r_json,
        "K_gain_checksum": gain_checksum,
        "exit_check_version": exit_check_version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()[:12]
    return f"primvar_{primitive_id}_{entry_role}_{digest}"
