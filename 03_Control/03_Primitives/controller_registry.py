from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from dense_archive_table_io import filesystem_path
from evidence_status import (
    REGISTRY_BACKED_CONTROLLER_SELECTION_STATUSES,
    registry_claim_status_for,
    source_is_retired,
)
from lqr_controller import LQR_SYNTHESIS_SOLVED, LQRController, LQRWeightSpec, lqr_controller_metadata_row, synthesize_lqr_controller
from prim_cat import primitive_by_id


SELECTED_CONTROLLER_STATUS = "selected"
REJECTED_CONTROLLER_STATUS = "rejected"
BLOCKED_CONTROLLER_STATUS = "blocked"
SMOKE_SELECTED_CONTROLLER_STATUS = "smoke_selected_not_thesis_evidence"


@dataclass(frozen=True)
class ControllerRegistryEntry:
    primitive_id: str
    controller_id: str
    selected_controller_status: str
    selected_controller_reason: str
    candidate_index: int
    candidate_weight_label: str
    controller_family: str
    lqr_synthesis_status: str
    sampled_data_check_status: str
    zero_position_gain_expansion_status: str
    controller_claim_status: str
    linearisation_id: str
    lqr_gain_checksum: str
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    registry_status: str
    registry_claim_status: str
    registry_path: str
    controller_selection_status: str = "W0_W1_registry_selected"


@dataclass(frozen=True)
class SelectedControllerRecord:
    controller: LQRController
    primitive_id: str
    controller_id: str
    selected_controller_status: str
    selected_controller_reason: str
    candidate_index: int
    candidate_weight_label: str
    registry_status: str
    registry_claim_status: str
    registry_path: str
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    linearisation_id: str
    controller_selection_status: str = "W0_W1_registry_selected"

    def row_metadata(self) -> dict[str, object]:
        return {
            "selected_controller_status": self.selected_controller_status,
            "selected_controller_reason": self.selected_controller_reason,
            "candidate_index": int(self.candidate_index),
            "candidate_weight_label": self.candidate_weight_label,
            "registry_status": self.registry_status,
            "registry_claim_status": self.registry_claim_status,
            "registry_path": self.registry_path,
            "controller_selection_status": self.controller_selection_status,
        }


def controller_registry_row(
    controller: LQRController,
    *,
    selected_controller_status: str,
    selected_controller_reason: str,
    candidate_index: int,
    candidate_weight_label: str,
    registry_status: str = "smoke_incomplete",
    registry_claim_status: str | None = None,
    registry_path: str = "",
) -> dict[str, object]:
    resolved_registry_claim_status = registry_claim_status or registry_claim_status_for(registry_status)
    metadata = lqr_controller_metadata_row(controller)
    metadata.update(
        asdict(
            ControllerRegistryEntry(
                primitive_id=controller.primitive_id,
                controller_id=controller.controller_id,
                selected_controller_status=str(selected_controller_status),
                selected_controller_reason=str(selected_controller_reason),
                candidate_index=int(candidate_index),
                candidate_weight_label=str(candidate_weight_label),
                controller_family=controller.controller_family,
                lqr_synthesis_status=controller.lqr_synthesis_status,
                sampled_data_check_status=controller.sampled_data_check_status,
                zero_position_gain_expansion_status=controller.zero_position_gain_expansion_status,
                controller_claim_status=controller.controller_claim_status,
                linearisation_id=controller.linearisation_id,
                lqr_gain_checksum=controller.lqr_gain_checksum,
                lqr_Q_weights_json=controller.lqr_Q_weights_json,
                lqr_R_weights_json=controller.lqr_R_weights_json,
                registry_status=str(registry_status),
                registry_claim_status=str(resolved_registry_claim_status),
                registry_path=str(registry_path),
            )
        )
    )
    return metadata


def write_selected_controller_registry(
    *,
    rows: list[dict[str, object]],
    csv_path: Path,
    json_path: Path,
) -> None:
    rows = [_normalised_registry_row(row, registry_path=csv_path.as_posix()) for row in rows]
    selected = [
        row
        for row in rows
        if row.get("selected_controller_status") in {SELECTED_CONTROLLER_STATUS, SMOKE_SELECTED_CONTROLLER_STATUS}
    ]
    filesystem_path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(json_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filesystem_path(csv_path), index=False)
    filesystem_path(json_path).write_text(
        json.dumps(
            {
                "registry_version": "selected_lqr_controllers_v1",
                "registry_status": _registry_status_for_rows(rows),
                "registry_claim_status": registry_claim_status_for(_registry_status_for_rows(rows)),
                "selected_controller_count": len(selected),
                "primitive_count": len({str(row.get("primitive_id", "")) for row in rows}),
                "controllers": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def load_selected_controller_registry(path: Path | str | None) -> dict[str, LQRController]:
    return {
        primitive_id: record.controller
        for primitive_id, record in load_selected_controller_records(path).items()
    }


def load_selected_controller_records(path: Path | str | None) -> dict[str, SelectedControllerRecord]:
    if path is None or str(path) == "":
        return {}
    registry_path = filesystem_path(Path(path))
    if registry_path.suffix.lower() == ".json":
        payload = json.loads(registry_path.read_text(encoding="ascii"))
        rows = payload.get("controllers", [])
    else:
        rows = pd.read_csv(registry_path).to_dict(orient="records")
    registry: dict[str, SelectedControllerRecord] = {}
    for row in rows:
        status = str(row.get("selected_controller_status", ""))
        if status not in {SELECTED_CONTROLLER_STATUS, SMOKE_SELECTED_CONTROLLER_STATUS}:
            continue
        row = _normalised_registry_row(row, registry_path=Path(path).as_posix())
        controller = controller_from_registry_row(row)
        if controller.controller_id != str(row.get("controller_id", "")):
            raise ValueError(
                "selected controller registry checksum/id mismatch for "
                f"{row.get('primitive_id', '')}: {controller.controller_id} != {row.get('controller_id', '')}"
            )
        registry[controller.primitive_id] = SelectedControllerRecord(
            controller=controller,
            primitive_id=controller.primitive_id,
            controller_id=controller.controller_id,
            selected_controller_status=status,
            selected_controller_reason=str(row.get("selected_controller_reason", "")),
            candidate_index=int(float(row.get("candidate_index", 0))),
            candidate_weight_label=str(row.get("candidate_weight_label", "")),
            registry_status=str(row.get("registry_status", "blocked")),
            registry_claim_status=str(row.get("registry_claim_status", "simulation_only_blocked")),
            registry_path=str(row.get("registry_path", Path(path).as_posix())),
            lqr_Q_weights_json=str(row.get("lqr_Q_weights_json", controller.lqr_Q_weights_json)),
            lqr_R_weights_json=str(row.get("lqr_R_weights_json", controller.lqr_R_weights_json)),
            lqr_gain_checksum=str(row.get("lqr_gain_checksum", controller.lqr_gain_checksum)),
            linearisation_id=str(row.get("linearisation_id", controller.linearisation_id)),
        )
    return registry


def controller_from_registry_row(row: Mapping[str, object]) -> LQRController:
    primitive_id = str(row.get("primitive_id", ""))
    if not primitive_id:
        raise ValueError("controller registry row is missing primitive_id")
    q_payload = _json_mapping(row.get("lqr_Q_weights_json", "{}"))
    r_payload = _json_mapping(row.get("lqr_R_weights_json", "{}"))
    weight_label = str(
        row.get("candidate_weight_label", "")
        or row.get("weight_label", "")
        or "nominal"
    )
    weight_spec = LQRWeightSpec(
        q_attitude=float(q_payload.get("q_attitude", 4.0)),
        q_velocity=float(q_payload.get("q_velocity", 2.0)),
        q_rates=float(q_payload.get("q_rates", 1.6)),
        q_surfaces=float(q_payload.get("q_surfaces", 0.15)),
        r_aileron=float(r_payload.get("r_aileron", 1.0)),
        r_elevator=float(r_payload.get("r_elevator", 0.9)),
        r_rudder=float(r_payload.get("r_rudder", 1.1)),
        tuning_stage=str(row.get("tuning_stage", "W0_W1")),
        weight_label=weight_label,
    )
    return synthesize_lqr_controller(primitive_by_id(primitive_id), weight_spec=weight_spec)


def controller_from_evidence_row(row: Mapping[str, object]) -> LQRController:
    selection_status = str(row.get("controller_selection_status", ""))
    if selection_status not in REGISTRY_BACKED_CONTROLLER_SELECTION_STATUSES:
        raise ValueError(
            "evidence row is not backed by the selected-controller registry: "
            f"{selection_status or 'missing_controller_selection_status'}"
        )
    if not _has_text(row.get("candidate_weight_label", "")):
        # Legacy rows before the selected registry did not carry enough identity
        # to reproduce non-nominal candidate hashes safely.
        raise ValueError("evidence row is missing selected controller candidate_weight_label")
    controller = controller_from_registry_row(row)
    expected_id = str(row.get("controller_id", ""))
    if expected_id and controller.controller_id != expected_id:
        raise ValueError(f"evidence controller_id mismatch: {controller.controller_id} != {expected_id}")
    expected_checksum = str(row.get("lqr_gain_checksum", ""))
    if expected_checksum and controller.lqr_gain_checksum != expected_checksum:
        raise ValueError(
            "evidence lqr_gain_checksum mismatch: "
            f"{controller.lqr_gain_checksum} != {expected_checksum}"
        )
    return controller


def controller_is_executable_lqr(controller: LQRController) -> tuple[bool, str]:
    if controller.controller_family != "lqr":
        return False, "controller_family_not_lqr"
    if controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
        return False, "lqr_synthesis_not_solved"
    if controller.sampled_data_check_status != "sampled_stable":
        return False, "sampled_data_not_stable"
    if controller.zero_position_gain_expansion_status != "zero_position_gains_verified":
        return False, "zero_position_gains_not_verified"
    if controller.controller_claim_status != "simulation_only":
        return False, "controller_claim_status_not_allowed"
    return True, ""


def _json_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return dict(json.loads(str(value)))


def _normalised_registry_row(row: Mapping[str, object], *, registry_path: str) -> dict[str, object]:
    result = dict(row)
    registry_status = str(result.get("registry_status", ""))
    if not registry_status:
        registry_status = "retired_not_active" if source_is_retired(registry_path) else "smoke_incomplete"
    result["registry_status"] = registry_status
    result["registry_claim_status"] = str(
        result.get("registry_claim_status", "") or registry_claim_status_for(registry_status)
    )
    result["registry_path"] = str(result.get("registry_path", "") or registry_path)
    result["controller_selection_status"] = str(
        result.get("controller_selection_status", "") or "W0_W1_registry_selected"
    )
    return result


def _registry_status_for_rows(rows: list[dict[str, object]]) -> str:
    statuses = {str(row.get("registry_status", "blocked")) for row in rows}
    if "retired_not_active" in statuses:
        return "retired_not_active"
    if "complete" in statuses:
        return "complete"
    if "accepted_fallback" in statuses:
        return "accepted_fallback"
    if "smoke_incomplete" in statuses:
        return "smoke_incomplete"
    return "blocked"


def _has_text(value: object) -> bool:
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "null"})
