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
from primitive_timing_contract import (
    CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE,
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    PRIMITIVE_TIMING_CONTRACT_VERSION,
    assert_primitive_timing_contract,
)


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
VARIANT_REGISTRY_VERSION = "w01_primitive_controller_variant_registry_v411"
EXIT_CHECK_VERSION = "w01_exit_checks_v1"
ENTRY_ROLES = ("launch_capable", "inflight_only", "terminal_or_recovery")
ENTRY_ROLE_START_FAMILY_SEQUENCE = {
    "launch_capable": ("launch_gate",),
    "inflight_only": (
        "inflight_nominal",
        "inflight_nominal",
        "inflight_nominal",
        "inflight_nominal",
        "inflight_nominal",
        "inflight_lift_region",
        "inflight_lift_region",
        "inflight_lift_region",
    ),
    "terminal_or_recovery": ("inflight_boundary_near", "inflight_recovery_edge"),
}
ENTRY_ROLE_BY_PRIMITIVE_ID = {
    "glide": "inflight_only",
    "lift_entry": "inflight_only",
    "lift_dwell_arc": "inflight_only",
    "mild_turn_left": "inflight_only",
    "mild_turn_right": "inflight_only",
    "energy_retaining_bank": "inflight_only",
    "recovery": "terminal_or_recovery",
    "safe_exit_or_recovery_handoff": "terminal_or_recovery",
    "launch_capture_glide_stabilise": "launch_capable",
    "launch_capture_lift_seek": "launch_capable",
    "launch_capture_energy_build": "launch_capable",
    "launch_capture_shallow_left": "launch_capable",
    "launch_capture_shallow_right": "launch_capable",
    "launch_capture_safe_handoff": "launch_capable",
}
ENTRY_ROLE_REJECTION_STATUS = "entry_role_incompatible_start"
ENTRY_ROLE_REJECTION_LABEL = "entry_role_incompatible_start_family"


@dataclass(frozen=True)
class PrimitiveControllerVariant:
    primitive_variant_id: str
    primitive_id: str
    primitive_family: str
    entry_role: str
    candidate_index: int | str
    candidate_weight_label: str
    controller_id: str
    controller_family: str
    reference_state_vector: str
    reference_command_vector: str
    finite_horizon_s: float
    controller_input_slots_per_primitive: int
    controller_input_update_period_s: float
    primitive_timing_contract_version: str
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
    controller_design_role: str
    timing_augmentation_type: str
    timing_design_version: str
    sample_time_s: float
    latency_case: str
    state_feedback_delay_s: float
    command_delay_s: float
    command_delay_steps: int
    actuator_tau_s: str
    actuator_state_count: int
    command_delay_state_count: int
    predictor_horizon_steps: int
    augmented_state_size: int
    augmented_input_size: int
    augmented_A_checksum: str
    augmented_B_checksum: str
    augmented_Q_json: str
    augmented_R_json: str
    augmented_gain_checksum: str
    augmented_closed_loop_spectral_radius: float
    timing_lqr_blocked_reason: str
    timing_aware_synthesis_level: str
    timing_effects_in_synthesis: str
    timing_effects_in_rollout: str
    sampled_data_timing_audit_status: str
    delayed_state_lqr_augmentation_status: str
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
    return family in allowed_start_families_for_entry_role(role)


def start_family_for_entry_role_index(*, entry_role: str, index: int) -> str:
    sequence = ENTRY_ROLE_START_FAMILY_SEQUENCE.get(str(entry_role))
    if sequence is None:
        raise ValueError(f"unknown entry_role: {entry_role}")
    return str(sequence[int(index) % len(sequence)])


def allowed_start_families_for_entry_role(entry_role: str) -> tuple[str, ...]:
    sequence = ENTRY_ROLE_START_FAMILY_SEQUENCE.get(str(entry_role))
    if sequence is None:
        raise ValueError(f"unknown entry_role: {entry_role}")
    return tuple(dict.fromkeys(sequence))


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
    candidate_index: int | str = "",
    candidate_weight_label: str | None = None,
) -> PrimitiveControllerVariant:
    role = str(entry_role or entry_role_for_primitive_id(primitive.primitive_id))
    if role not in ENTRY_ROLES:
        raise ValueError(f"entry_role must be one of {ENTRY_ROLES}")
    assert_primitive_timing_contract(
        finite_horizon_s=primitive.finite_horizon_s,
        controller_input_slots_per_primitive=primitive.controller_input_slots_per_primitive,
        controller_input_update_period_s=primitive.controller_input_update_period_s,
        primitive_timing_contract_version=primitive.primitive_timing_contract_version,
    )
    k_json = json.dumps(controller.k_gain_matrix, separators=(",", ":"))
    reference_state_json = json.dumps(list(controller.reference_state_vector), separators=(",", ":"))
    reference_command_json = json.dumps(list(controller.reference_command_vector), separators=(",", ":"))
    checksum = controller.lqr_gain_checksum
    variant_id = _variant_id(
        primitive_id=primitive.primitive_id,
        entry_role=role,
        finite_horizon_s=primitive.finite_horizon_s,
        controller_input_slots_per_primitive=primitive.controller_input_slots_per_primitive,
        controller_input_update_period_s=primitive.controller_input_update_period_s,
        primitive_timing_contract_version=primitive.primitive_timing_contract_version,
        reference_state_vector=reference_state_json,
        reference_command_vector=reference_command_json,
        controller_id=controller.controller_id,
        linearisation_id=controller.linearisation_id,
        q_json=controller.lqr_Q_weights_json,
        r_json=controller.lqr_R_weights_json,
        gain_checksum=checksum,
        timing_augmentation_type=controller.timing_augmentation_type,
        timing_design_version=controller.timing_design_version,
        sample_time_s=controller.sample_time_s,
        latency_case=controller.latency_case,
        command_delay_steps=controller.command_delay_steps,
        predictor_horizon_steps=controller.predictor_horizon_steps,
        augmented_a_checksum=controller.augmented_A_checksum,
        augmented_b_checksum=controller.augmented_B_checksum,
        augmented_q_json=controller.augmented_Q_json,
        augmented_r_json=controller.augmented_R_json,
        augmented_gain_checksum=controller.augmented_gain_checksum,
        exit_check_version=EXIT_CHECK_VERSION,
    )
    return PrimitiveControllerVariant(
        primitive_variant_id=variant_id,
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        entry_role=role,
        candidate_index=candidate_index,
        candidate_weight_label=str(candidate_weight_label or controller.tuning_stage),
        controller_id=controller.controller_id,
        controller_family=controller.controller_family,
        reference_state_vector=reference_state_json,
        reference_command_vector=reference_command_json,
        finite_horizon_s=float(primitive.finite_horizon_s),
        controller_input_slots_per_primitive=int(primitive.controller_input_slots_per_primitive),
        controller_input_update_period_s=float(primitive.controller_input_update_period_s),
        primitive_timing_contract_version=str(primitive.primitive_timing_contract_version),
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
        controller_design_role=controller.controller_design_role,
        timing_augmentation_type=controller.timing_augmentation_type,
        timing_design_version=controller.timing_design_version,
        sample_time_s=float(controller.sample_time_s),
        latency_case=controller.latency_case,
        state_feedback_delay_s=float(controller.state_feedback_delay_s),
        command_delay_s=float(controller.command_delay_s),
        command_delay_steps=int(controller.command_delay_steps),
        actuator_tau_s=json.dumps(list(controller.actuator_tau_s), separators=(",", ":")),
        actuator_state_count=int(controller.actuator_state_count),
        command_delay_state_count=int(controller.command_delay_state_count),
        predictor_horizon_steps=int(controller.predictor_horizon_steps),
        augmented_state_size=int(controller.augmented_state_size),
        augmented_input_size=int(controller.augmented_input_size),
        augmented_A_checksum=controller.augmented_A_checksum,
        augmented_B_checksum=controller.augmented_B_checksum,
        augmented_Q_json=controller.augmented_Q_json,
        augmented_R_json=controller.augmented_R_json,
        augmented_gain_checksum=controller.augmented_gain_checksum,
        augmented_closed_loop_spectral_radius=float(controller.augmented_closed_loop_spectral_radius),
        timing_lqr_blocked_reason=controller.timing_lqr_blocked_reason,
        timing_aware_synthesis_level=controller.timing_aware_synthesis_level,
        timing_effects_in_synthesis=controller.timing_effects_in_synthesis,
        timing_effects_in_rollout=controller.timing_effects_in_rollout,
        sampled_data_timing_audit_status=controller.sampled_data_timing_audit_status,
        delayed_state_lqr_augmentation_status=controller.delayed_state_lqr_augmentation_status,
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
    out.setdefault("controller_input_slots_per_primitive", 0)
    out.setdefault("controller_input_update_period_s", 0.0)
    out.setdefault("primitive_timing_contract_version", "legacy_not_recorded")
    out.setdefault("candidate_index", "")
    out.setdefault("candidate_weight_label", "")
    out.setdefault("timing_aware_synthesis_level", "trim_local_reduced_order_lqr_no_delay_augmentation")
    out.setdefault("timing_effects_in_synthesis", "sampled_data_stability_and_nominal_latency_actuator_smoke_only")
    out.setdefault("timing_effects_in_rollout", "feedback_delay_command_timing_actuator_lag_applied_in_w01_rollout")
    out.setdefault("sampled_data_timing_audit_status", "legacy_registry_not_recorded")
    out.setdefault("delayed_state_lqr_augmentation_status", "not_implemented_state_delay_simulated_in_rollout")
    out.setdefault("controller_design_role", "legacy_or_superseded_controller")
    out.setdefault("timing_augmentation_type", "legacy_not_recorded")
    out.setdefault("timing_design_version", "legacy_not_recorded")
    out.setdefault("sample_time_s", 0.0)
    out.setdefault("latency_case", "legacy_not_recorded")
    out.setdefault("state_feedback_delay_s", 0.0)
    out.setdefault("command_delay_s", 0.0)
    out.setdefault("command_delay_steps", 0)
    out.setdefault("actuator_tau_s", "[]")
    out.setdefault("actuator_state_count", 0)
    out.setdefault("command_delay_state_count", 0)
    out.setdefault("predictor_horizon_steps", 0)
    out.setdefault("augmented_state_size", 0)
    out.setdefault("augmented_input_size", 0)
    out.setdefault("augmented_A_checksum", "")
    out.setdefault("augmented_B_checksum", "")
    out.setdefault("augmented_Q_json", "")
    out.setdefault("augmented_R_json", "")
    out.setdefault("augmented_gain_checksum", "")
    out.setdefault("augmented_closed_loop_spectral_radius", float("inf"))
    out.setdefault("timing_lqr_blocked_reason", "")
    for key in (
        "sample_time_s",
        "state_feedback_delay_s",
        "command_delay_s",
        "augmented_closed_loop_spectral_radius",
        "controller_input_update_period_s",
    ):
        out[key] = float(out[key])
    for key in (
        "controller_input_slots_per_primitive",
        "command_delay_steps",
        "actuator_state_count",
        "command_delay_state_count",
        "predictor_horizon_steps",
        "augmented_state_size",
        "augmented_input_size",
    ):
        out[key] = int(out[key])
    return out


def _variant_id(
    *,
    primitive_id: str,
    entry_role: str,
    finite_horizon_s: float,
    controller_input_slots_per_primitive: int,
    controller_input_update_period_s: float,
    primitive_timing_contract_version: str,
    reference_state_vector: str,
    reference_command_vector: str,
    controller_id: str,
    linearisation_id: str,
    q_json: str,
    r_json: str,
    gain_checksum: str,
    timing_augmentation_type: str,
    timing_design_version: str,
    sample_time_s: float,
    latency_case: str,
    command_delay_steps: int,
    predictor_horizon_steps: int,
    augmented_a_checksum: str,
    augmented_b_checksum: str,
    augmented_q_json: str,
    augmented_r_json: str,
    augmented_gain_checksum: str,
    exit_check_version: str,
) -> str:
    payload = {
        "primitive_id": primitive_id,
        "entry_role": entry_role,
        "finite_horizon_s": float(finite_horizon_s),
        "controller_input_slots_per_primitive": int(controller_input_slots_per_primitive),
        "controller_input_update_period_s": float(controller_input_update_period_s),
        "primitive_timing_contract_version": str(primitive_timing_contract_version),
        "reference_state_vector": reference_state_vector,
        "reference_command_vector": reference_command_vector,
        "controller_id": controller_id,
        "linearisation_id": linearisation_id,
        "Q_weight_json": q_json,
        "R_weight_json": r_json,
        "K_gain_checksum": gain_checksum,
        "timing_augmentation_type": timing_augmentation_type,
        "timing_design_version": timing_design_version,
        "sample_time_s": float(sample_time_s),
        "latency_case": str(latency_case),
        "command_delay_steps": int(command_delay_steps),
        "predictor_horizon_steps": int(predictor_horizon_steps),
        "augmented_A_checksum": augmented_a_checksum,
        "augmented_B_checksum": augmented_b_checksum,
        "augmented_Q_json": augmented_q_json,
        "augmented_R_json": augmented_r_json,
        "augmented_gain_checksum": augmented_gain_checksum,
        "exit_check_version": exit_check_version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()[:12]
    return f"primvar_{primitive_id}_{entry_role}_{digest}"
