from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from dense_archive_table_io import filesystem_path
from lqr_controller import LQR_SYNTHESIS_SOLVED, LQRController, LQRWeightSpec, lqr_controller_metadata_row, synthesize_lqr_controller
from prim_cat import primitive_by_id


SELECTED_CONTROLLER_STATUS = "selected"
REJECTED_CONTROLLER_STATUS = "rejected"
BLOCKED_CONTROLLER_STATUS = "blocked"


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
    controller_selection_status: str = "W0_W1_registry_selected"


def controller_registry_row(
    controller: LQRController,
    *,
    selected_controller_status: str,
    selected_controller_reason: str,
    candidate_index: int,
    candidate_weight_label: str,
) -> dict[str, object]:
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
    selected = [row for row in rows if row.get("selected_controller_status") == SELECTED_CONTROLLER_STATUS]
    filesystem_path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(json_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filesystem_path(csv_path), index=False)
    filesystem_path(json_path).write_text(
        json.dumps(
            {
                "registry_version": "selected_lqr_controllers_v1",
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
    if path is None or str(path) == "":
        return {}
    registry_path = filesystem_path(Path(path))
    if registry_path.suffix.lower() == ".json":
        payload = json.loads(registry_path.read_text(encoding="ascii"))
        rows = payload.get("controllers", [])
    else:
        rows = pd.read_csv(registry_path).to_dict(orient="records")
    registry: dict[str, LQRController] = {}
    for row in rows:
        if str(row.get("selected_controller_status", "")) != SELECTED_CONTROLLER_STATUS:
            continue
        controller = controller_from_registry_row(row)
        if controller.controller_id != str(row.get("controller_id", "")):
            raise ValueError(
                "selected controller registry checksum/id mismatch for "
                f"{row.get('primitive_id', '')}: {controller.controller_id} != {row.get('controller_id', '')}"
            )
        registry[controller.primitive_id] = controller
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
    if selection_status != "W0_W1_registry_selected":
        raise ValueError(
            "evidence row is not backed by the W0/W1 selected-controller registry: "
            f"{selection_status or 'missing_controller_selection_status'}"
        )
    if not row.get("candidate_weight_label", ""):
        # Legacy rows before the selected registry did not carry enough identity
        # to reproduce non-nominal candidate hashes safely.
        raise ValueError("evidence row is missing selected controller candidate_weight_label")
    controller = controller_from_registry_row(row)
    expected_id = str(row.get("controller_id", ""))
    if expected_id and controller.controller_id != expected_id:
        raise ValueError(f"evidence controller_id mismatch: {controller.controller_id} != {expected_id}")
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
