from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dense_archive_table_io import (
    file_sha256,
    filesystem_path,
    load_table_manifest,
    read_table_partition,
)
from lqr_controller import LQRController, gain_checksum_sha256, matrix_checksum_sha256
from lqr_linearisation import LQR_STATE_MASK
from primitive_variant_registry import PrimitiveControllerVariant


FROZEN_W01_CONTROLLER_BUNDLE_VERSION = "frozen_w01_to_w2_controller_bundle_v1"
FROZEN_CONTROLLER_READY = "ready"
FROZEN_CONTROLLER_BLOCKED = "blocked"
FROZEN_CONTROLLER_FAILED_REASON = "w2_controller_reconstruction_failed"


@dataclass(frozen=True)
class FrozenW01ControllerRecord:
    primitive_variant_id: str
    controller_id: str
    primitive_id: str
    candidate_index: int | str
    candidate_weight_label: str
    bundle_status: str
    blocked_reason: str
    variant: PrimitiveControllerVariant
    controller: LQRController
    source_row: dict[str, object]


@dataclass(frozen=True)
class FrozenW01ControllerBundle:
    bundle_version: str
    source_w01_root: str
    source_w01_run_id: int | str
    source_registry_sha256: str
    source_table_manifest_sha256: str
    source_run_manifest_sha256: str
    variant_count: int
    ready_count: int
    blocked_count: int
    records: tuple[FrozenW01ControllerRecord, ...]

    @property
    def records_by_variant_id(self) -> dict[str, FrozenW01ControllerRecord]:
        return {record.primitive_variant_id: record for record in self.records}


def write_frozen_w01_controller_bundle(
    *,
    run_root: Path,
    source_records: Iterable[tuple[PrimitiveControllerVariant, LQRController]],
) -> dict[str, object]:
    """Write the executable W01 frozen-controller bundle from in-memory W01 controllers."""

    source_root = Path(run_root)
    records: list[dict[str, object]] = []
    for variant, controller in source_records:
        controller_payload = _controller_payload_from_controller(variant=variant, controller=controller)
        status, blocked_reason = _payload_status(
            variant=variant,
            controller_payload=controller_payload,
        )
        records.append(
            {
                "primitive_variant_id": variant.primitive_variant_id,
                "controller_id": variant.controller_id,
                "primitive_id": variant.primitive_id,
                "candidate_index": variant.candidate_index,
                "candidate_weight_label": variant.candidate_weight_label,
                "bundle_status": status,
                "blocked_reason": blocked_reason,
                "variant": _variant_payload(variant, {}),
                "controller_payload": controller_payload,
            }
        )

    ready_count = sum(1 for record in records if record["bundle_status"] == FROZEN_CONTROLLER_READY)
    payload: dict[str, object] = {
        "bundle_version": FROZEN_W01_CONTROLLER_BUNDLE_VERSION,
        **_source_info(source_root),
        "variant_count": len(records),
        "ready_count": int(ready_count),
        "blocked_count": int(len(records) - ready_count),
        "exact_replay_policy": "restore_payload_from_w01_emitted_bundle_only_no_controller_design",
        "physical_K_only_active_replay_allowed": False,
        "artifact_complete_required_for_w2": True,
        "records": records,
    }
    _write_json(source_root / "manifests" / "frozen_w01_controller_bundle.json", payload)
    _write_bundle_summary_and_report(source_root, payload)
    return payload


def materialize_frozen_w01_controller_bundle(
    *,
    input_root: Path,
    bundle_path: Path,
) -> dict[str, object]:
    """Legacy diagnostic bundle materialisation from incomplete W01 metadata only."""

    source_root = Path(input_root)
    registry_path = source_root / "manifests" / "primitive_variant_registry.json"
    table_manifest_path = source_root / "manifests" / "table_manifest.json"
    run_manifest_path = source_root / "manifests" / "run_manifest.json"
    if not filesystem_path(registry_path).is_file():
        return _write_blocked_bundle(
            bundle_path=bundle_path,
            input_root=source_root,
            blocked_reason="missing_W01_primitive_variant_registry",
        )
    if not filesystem_path(table_manifest_path).is_file():
        return _write_blocked_bundle(
            bundle_path=bundle_path,
            input_root=source_root,
            blocked_reason="missing_W01_table_manifest",
        )

    raw_registry = json.loads(filesystem_path(registry_path).read_text(encoding="ascii"))
    variants = {
        str(row.get("primitive_variant_id", "")): PrimitiveControllerVariant(**_normalise_variant_payload(row))
        for row in raw_registry.get("variants", [])
    }
    representative_rows = _representative_rows_by_variant_id(source_root)
    source_info = _source_info(source_root)
    records: list[dict[str, object]] = []
    for raw_variant in raw_registry.get("variants", []):
        variant_id = str(raw_variant.get("primitive_variant_id", ""))
        variant = variants[variant_id]
        source_row = representative_rows.get(variant_id, {})
        controller_payload = _controller_payload_from_variant(
            variant=variant,
            raw_variant=raw_variant,
            source_row=source_row,
        )
        status, blocked_reason = _payload_status(
            variant=variant,
            controller_payload=controller_payload,
        )
        records.append(
            {
                "primitive_variant_id": variant.primitive_variant_id,
                "controller_id": variant.controller_id,
                "primitive_id": variant.primitive_id,
                "candidate_index": variant.candidate_index,
                "candidate_weight_label": variant.candidate_weight_label,
                "bundle_status": status,
                "blocked_reason": blocked_reason,
                "variant": _variant_payload(variant, raw_variant),
                "controller_payload": controller_payload,
            }
        )

    ready_count = sum(1 for record in records if record["bundle_status"] == FROZEN_CONTROLLER_READY)
    payload: dict[str, object] = {
        "bundle_version": FROZEN_W01_CONTROLLER_BUNDLE_VERSION,
        **source_info,
        "variant_count": len(records),
        "ready_count": int(ready_count),
        "blocked_count": int(len(records) - ready_count),
        "exact_replay_policy": "restore_payload_from_bundle_only_no_controller_design",
        "physical_K_only_active_replay_allowed": False,
        "records": records,
    }
    _write_json(bundle_path, payload)
    return payload


def load_frozen_w01_controller_bundle(bundle_path: Path) -> FrozenW01ControllerBundle:
    payload = json.loads(filesystem_path(bundle_path).read_text(encoding="ascii"))
    records = tuple(
        reconstruct_frozen_w01_controller_from_bundle(record)
        for record in payload.get("records", [])
    )
    return FrozenW01ControllerBundle(
        bundle_version=str(payload.get("bundle_version", "")),
        source_w01_root=str(payload.get("source_w01_root", "")),
        source_w01_run_id=payload.get("source_w01_run_id", ""),
        source_registry_sha256=str(payload.get("source_registry_sha256", "")),
        source_table_manifest_sha256=str(payload.get("source_table_manifest_sha256", "")),
        source_run_manifest_sha256=str(payload.get("source_run_manifest_sha256", "")),
        variant_count=int(payload.get("variant_count", len(records))),
        ready_count=int(payload.get("ready_count", 0)),
        blocked_count=int(payload.get("blocked_count", 0)),
        records=records,
    )


def reconstruct_frozen_w01_controller_from_bundle(
    record: dict[str, object],
) -> FrozenW01ControllerRecord:
    """Restore a controller object from frozen payload and verify replay checksums."""

    variant = PrimitiveControllerVariant(**_normalise_variant_payload(record["variant"]))
    payload = dict(record.get("controller_payload", {}))
    controller = _controller_from_payload(variant=variant, payload=payload)
    status, blocked_reason = _payload_status(variant=variant, controller_payload=payload)
    if str(record.get("bundle_status", "")) == FROZEN_CONTROLLER_BLOCKED:
        status = FROZEN_CONTROLLER_BLOCKED
        blocked_reason = str(record.get("blocked_reason", blocked_reason))
    if controller.controller_id != variant.controller_id:
        status = FROZEN_CONTROLLER_BLOCKED
        blocked_reason = _append_reason(blocked_reason, "controller_id_mismatch")
    if str(record.get("primitive_variant_id", "")) != variant.primitive_variant_id:
        status = FROZEN_CONTROLLER_BLOCKED
        blocked_reason = _append_reason(blocked_reason, "primitive_variant_id_mismatch")
    if status == FROZEN_CONTROLLER_READY and blocked_reason:
        status = FROZEN_CONTROLLER_BLOCKED
    return FrozenW01ControllerRecord(
        primitive_variant_id=variant.primitive_variant_id,
        controller_id=variant.controller_id,
        primitive_id=variant.primitive_id,
        candidate_index=variant.candidate_index,
        candidate_weight_label=variant.candidate_weight_label,
        bundle_status=status,
        blocked_reason=blocked_reason,
        variant=variant,
        controller=controller,
        source_row=dict(payload),
    )


def frozen_bundle_record_row(record: FrozenW01ControllerRecord) -> dict[str, object]:
    return {
        "primitive_variant_id": record.primitive_variant_id,
        "controller_id": record.controller_id,
        "primitive_id": record.primitive_id,
        "candidate_index": record.candidate_index,
        "candidate_weight_label": record.candidate_weight_label,
        "bundle_status": record.bundle_status,
        "blocked_reason": record.blocked_reason,
    }


def _write_blocked_bundle(
    *,
    bundle_path: Path,
    input_root: Path,
    blocked_reason: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "bundle_version": FROZEN_W01_CONTROLLER_BUNDLE_VERSION,
        "source_w01_root": Path(input_root).as_posix(),
        "source_w01_run_id": _run_id_from_root(input_root),
        "source_registry_sha256": "",
        "source_table_manifest_sha256": "",
        "source_run_manifest_sha256": "",
        "variant_count": 0,
        "ready_count": 0,
        "blocked_count": 0,
        "exact_replay_policy": "blocked_before_bundle_load",
        "physical_K_only_active_replay_allowed": False,
        "blocked_reason": blocked_reason,
        "records": [],
    }
    _write_json(bundle_path, payload)
    return payload


def _source_info(source_root: Path) -> dict[str, object]:
    registry_path = source_root / "manifests" / "primitive_variant_registry.json"
    table_manifest_path = source_root / "manifests" / "table_manifest.json"
    run_manifest_path = source_root / "manifests" / "run_manifest.json"
    return {
        "source_w01_root": Path(source_root).as_posix(),
        "source_w01_run_id": _run_id_from_root(source_root),
        "source_registry_sha256": file_sha256(registry_path) if filesystem_path(registry_path).is_file() else "",
        "source_table_manifest_sha256": (
            file_sha256(table_manifest_path)
            if filesystem_path(table_manifest_path).is_file()
            else ""
        ),
        "source_run_manifest_sha256": (
            file_sha256(run_manifest_path)
            if filesystem_path(run_manifest_path).is_file()
            else ""
        ),
    }


def _representative_rows_by_variant_id(source_root: Path) -> dict[str, dict[str, object]]:
    table_manifest_path = source_root / "manifests" / "table_manifest.json"
    if not filesystem_path(table_manifest_path).is_file():
        return {}
    manifest = load_table_manifest(table_manifest_path)
    rows: dict[str, dict[str, object]] = {}
    for partition in manifest.tables:
        frame = read_table_partition(
            source_root / "tables" / partition.relative_path,
            storage_format=partition.storage_format,
        )
        if "primitive_variant_id" not in frame.columns:
            continue
        for row in frame.to_dict(orient="records"):
            variant_id = str(row.get("primitive_variant_id", ""))
            if variant_id and variant_id not in rows:
                rows[variant_id] = row
        if len(rows) >= 256:
            break
    return rows


def _controller_payload_from_controller(
    *,
    variant: PrimitiveControllerVariant,
    controller: LQRController,
) -> dict[str, object]:
    return {
        "primitive_id": controller.primitive_id,
        "controller_family": controller.controller_family,
        "controller_mode": controller.controller_mode,
        "feedback_mode": controller.feedback_mode,
        "controller_id": controller.controller_id,
        "controller_version": controller.controller_version,
        "lqr_reference_id": controller.lqr_reference_id,
        "linearisation_id": controller.linearisation_id,
        "linearisation_source": controller.linearisation_source,
        "reduced_order_lqr": controller.reduced_order_lqr,
        "lqr_state_mask_json": controller.lqr_state_mask_json,
        "zero_position_gain_expansion_status": controller.zero_position_gain_expansion_status,
        "full_state_care_status": controller.full_state_care_status,
        "full_state_care_message": controller.full_state_care_message,
        "full_controllability_rank": controller.full_controllability_rank,
        "full_state_size": controller.full_state_size,
        "reduced_controllability_rank": controller.reduced_controllability_rank,
        "reduced_state_size": controller.reduced_state_size,
        "care_residual_norm": controller.care_residual_norm,
        "lqr_Q_weights_json": controller.lqr_Q_weights_json,
        "lqr_R_weights_json": controller.lqr_R_weights_json,
        "lqr_gain_checksum": controller.lqr_gain_checksum,
        "lqr_synthesis_status": controller.lqr_synthesis_status,
        "lqr_blocked_reason": controller.lqr_blocked_reason,
        "lqr_closed_loop_eigenvalue_summary": controller.lqr_closed_loop_eigenvalue_summary,
        "sampled_data_check_status": controller.sampled_data_check_status,
        "sampled_data_spectral_radius": controller.sampled_data_spectral_radius,
        "command_clip_check_status": controller.command_clip_check_status,
        "saturation_summary": controller.saturation_summary,
        "latency_actuator_survival_status": controller.latency_actuator_survival_status,
        "controller_design_role": controller.controller_design_role,
        "timing_augmentation_type": controller.timing_augmentation_type,
        "timing_design_version": controller.timing_design_version,
        "sample_time_s": controller.sample_time_s,
        "latency_case": controller.latency_case,
        "state_feedback_delay_s": controller.state_feedback_delay_s,
        "command_delay_s": controller.command_delay_s,
        "command_delay_steps": controller.command_delay_steps,
        "actuator_tau_s": controller.actuator_tau_s,
        "actuator_state_count": controller.actuator_state_count,
        "command_delay_state_count": controller.command_delay_state_count,
        "predictor_horizon_steps": controller.predictor_horizon_steps,
        "augmented_state_size": controller.augmented_state_size,
        "augmented_input_size": controller.augmented_input_size,
        "augmented_A_checksum": controller.augmented_A_checksum,
        "augmented_B_checksum": controller.augmented_B_checksum,
        "augmented_A_matrix_json": controller.augmented_A_matrix_json,
        "augmented_B_matrix_json": controller.augmented_B_matrix_json,
        "augmented_Q_json": controller.augmented_Q_json,
        "augmented_R_json": controller.augmented_R_json,
        "augmented_gain_checksum": controller.augmented_gain_checksum,
        "augmented_gain_matrix_json": controller.augmented_gain_matrix_json,
        "augmented_closed_loop_spectral_radius": controller.augmented_closed_loop_spectral_radius,
        "timing_lqr_blocked_reason": controller.timing_lqr_blocked_reason,
        "predictor_A_reduced_json": controller.predictor_A_reduced_json,
        "timing_aware_synthesis_level": controller.timing_aware_synthesis_level,
        "timing_effects_in_synthesis": controller.timing_effects_in_synthesis,
        "timing_effects_in_rollout": controller.timing_effects_in_rollout,
        "sampled_data_timing_audit_status": controller.sampled_data_timing_audit_status,
        "delayed_state_lqr_augmentation_status": controller.delayed_state_lqr_augmentation_status,
        "tuning_stage": controller.tuning_stage,
        "controller_claim_status": controller.controller_claim_status,
        "k_gain_matrix": tuple(tuple(float(value) for value in row) for row in controller.k_gain_matrix),
        "reference_state_vector": tuple(float(value) for value in controller.reference_state_vector),
        "reference_command_vector": tuple(float(value) for value in controller.reference_command_vector),
        "variant_id_for_checksum_audit": variant.primitive_variant_id,
    }


def _controller_payload_from_variant(
    *,
    variant: PrimitiveControllerVariant,
    raw_variant: dict[str, object],
    source_row: dict[str, object],
) -> dict[str, object]:
    row = dict(source_row)
    raw = dict(raw_variant)
    return {
        "primitive_id": variant.primitive_id,
        "controller_family": variant.controller_family,
        "controller_mode": _first_nonempty(row.get("controller_mode"), "lqr_local_feedback"),
        "feedback_mode": _first_nonempty(row.get("feedback_mode"), "predictor_compensated_augmented_discrete_lqr"),
        "controller_id": variant.controller_id,
        "controller_version": _first_nonempty(row.get("controller_version"), variant.timing_design_version),
        "lqr_reference_id": _first_nonempty(row.get("lqr_reference_id"), variant.linearisation_id),
        "linearisation_id": variant.linearisation_id,
        "linearisation_source": variant.linearisation_source,
        "reduced_order_lqr": _bool(row.get("reduced_order_lqr"), True),
        "lqr_state_mask_json": _first_nonempty(row.get("lqr_state_mask_json"), json.dumps(LQR_STATE_MASK, separators=(",", ":"))),
        "zero_position_gain_expansion_status": _first_nonempty(
            row.get("zero_position_gain_expansion_status"),
            "zero_position_gains_verified",
        ),
        "full_state_care_status": _first_nonempty(row.get("full_state_care_status"), "not_recorded_frozen_w01"),
        "full_state_care_message": _first_nonempty(row.get("full_state_care_message"), ""),
        "full_controllability_rank": _int(row.get("full_controllability_rank"), 0),
        "full_state_size": _int(row.get("full_state_size"), 15),
        "reduced_controllability_rank": _int(row.get("reduced_controllability_rank"), 0),
        "reduced_state_size": _int(row.get("reduced_state_size"), len(LQR_STATE_MASK)),
        "care_residual_norm": _float(row.get("care_residual_norm"), 0.0),
        "lqr_Q_weights_json": variant.Q_weight_json,
        "lqr_R_weights_json": variant.R_weight_json,
        "lqr_gain_checksum": variant.K_gain_checksum,
        "lqr_synthesis_status": variant.lqr_synthesis_status,
        "lqr_blocked_reason": variant.lqr_blocked_reason,
        "lqr_closed_loop_eigenvalue_summary": variant.closed_loop_eigenvalue_summary,
        "sampled_data_check_status": variant.sampled_data_check_status,
        "sampled_data_spectral_radius": _float(row.get("sampled_data_spectral_radius"), 0.0),
        "command_clip_check_status": _first_nonempty(row.get("command_clip_check_status"), "not_recorded_frozen_w01"),
        "saturation_summary": _first_nonempty(row.get("saturation_summary"), "not_recorded_frozen_w01"),
        "latency_actuator_survival_status": _first_nonempty(
            row.get("latency_actuator_survival_status"),
            "timing_augmented_discrete_lqr_solved",
        ),
        "controller_design_role": variant.controller_design_role,
        "timing_augmentation_type": variant.timing_augmentation_type,
        "timing_design_version": variant.timing_design_version,
        "sample_time_s": variant.sample_time_s,
        "latency_case": variant.latency_case,
        "state_feedback_delay_s": variant.state_feedback_delay_s,
        "command_delay_s": variant.command_delay_s,
        "command_delay_steps": variant.command_delay_steps,
        "actuator_tau_s": _parse_float_tuple(variant.actuator_tau_s),
        "actuator_state_count": variant.actuator_state_count,
        "command_delay_state_count": variant.command_delay_state_count,
        "predictor_horizon_steps": variant.predictor_horizon_steps,
        "augmented_state_size": variant.augmented_state_size,
        "augmented_input_size": variant.augmented_input_size,
        "augmented_A_checksum": variant.augmented_A_checksum,
        "augmented_B_checksum": variant.augmented_B_checksum,
        "augmented_Q_json": variant.augmented_Q_json,
        "augmented_R_json": variant.augmented_R_json,
        "augmented_gain_checksum": variant.augmented_gain_checksum,
        "augmented_gain_matrix_json": _first_nonempty(
            raw.get("augmented_gain_matrix_json"),
            row.get("augmented_gain_matrix_json"),
            "",
        ),
        "augmented_A_matrix_json": _first_nonempty(raw.get("augmented_A_matrix_json"), row.get("augmented_A_matrix_json"), ""),
        "augmented_B_matrix_json": _first_nonempty(raw.get("augmented_B_matrix_json"), row.get("augmented_B_matrix_json"), ""),
        "augmented_closed_loop_spectral_radius": variant.augmented_closed_loop_spectral_radius,
        "timing_lqr_blocked_reason": variant.timing_lqr_blocked_reason,
        "predictor_A_reduced_json": _first_nonempty(
            raw.get("predictor_A_reduced_json"),
            row.get("predictor_A_reduced_json"),
            "",
        ),
        "timing_aware_synthesis_level": variant.timing_aware_synthesis_level,
        "timing_effects_in_synthesis": variant.timing_effects_in_synthesis,
        "timing_effects_in_rollout": variant.timing_effects_in_rollout,
        "sampled_data_timing_audit_status": variant.sampled_data_timing_audit_status,
        "delayed_state_lqr_augmentation_status": variant.delayed_state_lqr_augmentation_status,
        "tuning_stage": _first_nonempty(row.get("tuning_stage"), "W0_W1"),
        "controller_claim_status": _first_nonempty(row.get("controller_claim_status"), variant.claim_status),
        "k_gain_matrix": json.loads(variant.K_gain_matrix_json),
        "reference_state_vector": json.loads(variant.reference_state_vector),
        "reference_command_vector": json.loads(variant.reference_command_vector),
    }


def _payload_status(
    *,
    variant: PrimitiveControllerVariant,
    controller_payload: dict[str, object],
) -> tuple[str, str]:
    reasons: list[str] = []
    k_matrix = np.asarray(controller_payload.get("k_gain_matrix", []), dtype=float)
    if k_matrix.shape != (3, 15):
        reasons.append("invalid_K_gain_matrix_shape")
    elif gain_checksum_sha256(k_matrix) != variant.K_gain_checksum:
        reasons.append("K_gain_checksum_mismatch")
    augmented_gain_json = str(controller_payload.get("augmented_gain_matrix_json", ""))
    if not augmented_gain_json:
        reasons.append("missing_augmented_gain_matrix_json")
    else:
        try:
            augmented_gain = np.asarray(json.loads(augmented_gain_json), dtype=float)
            if gain_checksum_sha256(augmented_gain) != variant.augmented_gain_checksum:
                reasons.append("augmented_gain_checksum_mismatch")
        except Exception:
            reasons.append("invalid_augmented_gain_matrix_json")
    predictor_json = str(controller_payload.get("predictor_A_reduced_json", ""))
    if not predictor_json:
        reasons.append("missing_predictor_A_reduced_json")
    augmented_a_json = str(controller_payload.get("augmented_A_matrix_json", ""))
    if not augmented_a_json:
        reasons.append("missing_augmented_A_matrix_json")
    else:
        try:
            augmented_a = np.asarray(json.loads(augmented_a_json), dtype=float)
            if matrix_checksum_sha256(augmented_a) != variant.augmented_A_checksum:
                reasons.append("augmented_A_matrix_checksum_mismatch")
        except Exception:
            reasons.append("invalid_augmented_A_matrix_json")
    augmented_b_json = str(controller_payload.get("augmented_B_matrix_json", ""))
    if not augmented_b_json:
        reasons.append("missing_augmented_B_matrix_json")
    else:
        try:
            augmented_b = np.asarray(json.loads(augmented_b_json), dtype=float)
            if matrix_checksum_sha256(augmented_b) != variant.augmented_B_checksum:
                reasons.append("augmented_B_matrix_checksum_mismatch")
        except Exception:
            reasons.append("invalid_augmented_B_matrix_json")
    if str(controller_payload.get("augmented_A_checksum", "")) != variant.augmented_A_checksum:
        reasons.append("augmented_A_checksum_mismatch")
    if str(controller_payload.get("augmented_B_checksum", "")) != variant.augmented_B_checksum:
        reasons.append("augmented_B_checksum_mismatch")
    if str(controller_payload.get("controller_id", "")) != variant.controller_id:
        reasons.append("controller_id_mismatch")
    if reasons:
        return FROZEN_CONTROLLER_BLOCKED, ";".join([FROZEN_CONTROLLER_FAILED_REASON, *reasons])
    return FROZEN_CONTROLLER_READY, ""


def _controller_from_payload(
    *,
    variant: PrimitiveControllerVariant,
    payload: dict[str, object],
) -> LQRController:
    return LQRController(
        primitive_id=str(payload["primitive_id"]),
        controller_family=str(payload["controller_family"]),
        controller_mode=str(payload["controller_mode"]),
        feedback_mode=str(payload["feedback_mode"]),
        controller_id=str(payload["controller_id"]),
        controller_version=str(payload["controller_version"]),
        lqr_reference_id=str(payload["lqr_reference_id"]),
        linearisation_id=str(payload["linearisation_id"]),
        linearisation_source=str(payload["linearisation_source"]),
        reduced_order_lqr=_bool(payload.get("reduced_order_lqr"), True),
        lqr_state_mask_json=str(payload["lqr_state_mask_json"]),
        zero_position_gain_expansion_status=str(payload["zero_position_gain_expansion_status"]),
        full_state_care_status=str(payload["full_state_care_status"]),
        full_state_care_message=str(payload["full_state_care_message"]),
        full_controllability_rank=_int(payload.get("full_controllability_rank"), 0),
        full_state_size=_int(payload.get("full_state_size"), 15),
        reduced_controllability_rank=_int(payload.get("reduced_controllability_rank"), 0),
        reduced_state_size=_int(payload.get("reduced_state_size"), len(LQR_STATE_MASK)),
        care_residual_norm=_float(payload.get("care_residual_norm"), 0.0),
        lqr_Q_weights_json=str(payload["lqr_Q_weights_json"]),
        lqr_R_weights_json=str(payload["lqr_R_weights_json"]),
        lqr_gain_checksum=str(payload["lqr_gain_checksum"]),
        lqr_synthesis_status=str(payload["lqr_synthesis_status"]),
        lqr_blocked_reason=str(payload["lqr_blocked_reason"]),
        lqr_closed_loop_eigenvalue_summary=str(payload["lqr_closed_loop_eigenvalue_summary"]),
        sampled_data_check_status=str(payload["sampled_data_check_status"]),
        sampled_data_spectral_radius=_float(payload.get("sampled_data_spectral_radius"), 0.0),
        command_clip_check_status=str(payload["command_clip_check_status"]),
        saturation_summary=str(payload["saturation_summary"]),
        latency_actuator_survival_status=str(payload["latency_actuator_survival_status"]),
        controller_design_role=str(payload["controller_design_role"]),
        timing_augmentation_type=str(payload["timing_augmentation_type"]),
        timing_design_version=str(payload["timing_design_version"]),
        sample_time_s=_float(payload.get("sample_time_s"), 0.0),
        latency_case=str(payload["latency_case"]),
        state_feedback_delay_s=_float(payload.get("state_feedback_delay_s"), 0.0),
        command_delay_s=_float(payload.get("command_delay_s"), 0.0),
        command_delay_steps=_int(payload.get("command_delay_steps"), 0),
        actuator_tau_s=tuple(float(value) for value in payload.get("actuator_tau_s", ())),
        actuator_state_count=_int(payload.get("actuator_state_count"), 0),
        command_delay_state_count=_int(payload.get("command_delay_state_count"), 0),
        predictor_horizon_steps=_int(payload.get("predictor_horizon_steps"), 0),
        augmented_state_size=_int(payload.get("augmented_state_size"), 0),
        augmented_input_size=_int(payload.get("augmented_input_size"), 0),
        augmented_A_checksum=str(payload["augmented_A_checksum"]),
        augmented_B_checksum=str(payload["augmented_B_checksum"]),
        augmented_A_matrix_json=str(payload.get("augmented_A_matrix_json", "")),
        augmented_B_matrix_json=str(payload.get("augmented_B_matrix_json", "")),
        augmented_Q_json=str(payload["augmented_Q_json"]),
        augmented_R_json=str(payload["augmented_R_json"]),
        augmented_gain_checksum=str(payload["augmented_gain_checksum"]),
        augmented_gain_matrix_json=str(payload.get("augmented_gain_matrix_json", "")),
        augmented_closed_loop_spectral_radius=_float(payload.get("augmented_closed_loop_spectral_radius"), float("inf")),
        timing_lqr_blocked_reason=str(payload["timing_lqr_blocked_reason"]),
        predictor_A_reduced_json=str(payload.get("predictor_A_reduced_json", "")),
        timing_aware_synthesis_level=str(payload["timing_aware_synthesis_level"]),
        timing_effects_in_synthesis=str(payload["timing_effects_in_synthesis"]),
        timing_effects_in_rollout=str(payload["timing_effects_in_rollout"]),
        sampled_data_timing_audit_status=str(payload["sampled_data_timing_audit_status"]),
        delayed_state_lqr_augmentation_status=str(payload["delayed_state_lqr_augmentation_status"]),
        tuning_stage=str(payload["tuning_stage"]),
        controller_claim_status=str(payload["controller_claim_status"]),
        k_gain_matrix=tuple(tuple(float(value) for value in row) for row in payload["k_gain_matrix"]),
        reference_state_vector=tuple(float(value) for value in payload["reference_state_vector"]),
        reference_command_vector=tuple(float(value) for value in payload["reference_command_vector"]),
    )


def _variant_payload(
    variant: PrimitiveControllerVariant,
    raw_variant: dict[str, object],
) -> dict[str, object]:
    payload = dict(raw_variant)
    for field_name in PrimitiveControllerVariant.__dataclass_fields__:
        payload.setdefault(field_name, getattr(variant, field_name))
    return payload


def _normalise_variant_payload(payload: object) -> dict[str, object]:
    row = dict(payload)  # type: ignore[arg-type]
    for key in (
        "finite_horizon_s",
        "sample_time_s",
        "state_feedback_delay_s",
        "command_delay_s",
        "augmented_closed_loop_spectral_radius",
    ):
        row[key] = _float(row.get(key), 0.0)
    for key in (
        "command_delay_steps",
        "actuator_state_count",
        "command_delay_state_count",
        "predictor_horizon_steps",
        "augmented_state_size",
        "augmented_input_size",
    ):
        row[key] = _int(row.get(key), 0)
    return {
        field_name: row[field_name]
        for field_name in PrimitiveControllerVariant.__dataclass_fields__
    }


def _first_nonempty(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return ""


def _parse_float_tuple(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        if not value:
            return ()
        return tuple(float(item) for item in json.loads(value))
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    return ()


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _int(value: object, default: int) -> int:
    try:
        if value is None or str(value).lower() == "nan":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _float(value: object, default: float) -> float:
    try:
        if value is None or str(value).lower() == "nan":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _append_reason(existing: str, reason: str) -> str:
    if not existing:
        return reason
    if reason in existing.split(";"):
        return existing
    return f"{existing};{reason}"


def _write_bundle_summary_and_report(source_root: Path, payload: dict[str, object]) -> None:
    records = list(payload.get("records", []))
    rows = [
        {
            "primitive_variant_id": str(record.get("primitive_variant_id", "")),
            "controller_id": str(record.get("controller_id", "")),
            "primitive_id": str(record.get("primitive_id", "")),
            "candidate_index": record.get("candidate_index", ""),
            "candidate_weight_label": str(record.get("candidate_weight_label", "")),
            "bundle_status": str(record.get("bundle_status", "")),
            "blocked_reason": str(record.get("blocked_reason", "")),
        }
        for record in records
    ]
    metrics_path = source_root / "metrics" / "frozen_w01_controller_bundle_summary.csv"
    filesystem_path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filesystem_path(metrics_path), index=False)
    blocked = [row for row in rows if row["bundle_status"] != FROZEN_CONTROLLER_READY]
    report = "\n".join(
        [
            "# Frozen W01 Controller Bundle Audit",
            "",
            f"- Bundle version: `{payload.get('bundle_version', '')}`",
            f"- Source W01 root: `{payload.get('source_w01_root', '')}`",
            f"- Variant count: `{payload.get('variant_count', 0)}`",
            f"- Ready payloads: `{payload.get('ready_count', 0)}`",
            f"- Blocked payloads: `{payload.get('blocked_count', 0)}`",
            "- Exact replay policy: `restore_payload_from_w01_emitted_bundle_only_no_controller_design`",
            "- Physical-K-only active replay allowed: `False`",
            "- W2 executable source: `manifests/frozen_w01_controller_bundle.json`",
            "",
            "Blocked payload preview:",
            "",
            *[
                f"- `{row['primitive_variant_id']}`: `{row['blocked_reason']}`"
                for row in blocked[:12]
            ],
            *([] if blocked else ["- `none`"]),
            "",
        ]
    )
    filesystem_path(source_root / "reports" / "frozen_controller_bundle_audit.md").parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(source_root / "reports" / "frozen_controller_bundle_audit.md").write_text(report, encoding="ascii")


def _run_id_from_root(root: Path) -> int | str:
    try:
        return int(Path(root).name)
    except ValueError:
        return Path(root).name


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")
