from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class PrimitiveRole:
    primitive_family: str
    mission_role: str
    fixed_gate_use: str
    final_mission_required: bool
    hardware_readiness_status: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


OPERATIONAL_PRIMITIVES = (
    PrimitiveRole("glide", "glide", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Baseline transit and energy-loss evidence."),
    PrimitiveRole("recovery", "recovery", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Safe handoff after poor entry or low margin."),
    PrimitiveRole("lift_entry", "lift_entry", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Enter useful lift from fixed-gate reachable states."),
    PrimitiveRole("lift_dwell_arc", "lift_dwell_arc", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Finite-horizon dwell, not indefinite soaring."),
    PrimitiveRole("mild_coordinated_turn_left", "mild_turn", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Low-risk left turn candidate."),
    PrimitiveRole("mild_coordinated_turn_right", "mild_turn", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Low-risk right turn candidate."),
    PrimitiveRole("energy_retaining_bank", "energy_retaining_bank", "active_mission_primitive", True, "requires_W2_W3_and_real_safety_evidence", "Bank candidate retained only when governor margins pass."),
)

DIAGNOSTIC_PRIMITIVES = (
    PrimitiveRole("canyon_steep_bank", "boundary_evidence", "diagnostic_boundary_evidence", False, "not_hardware_ready_without_W2_W3_and_real_safety_evidence", "High-angle expansion is not required for final mission."),
    PrimitiveRole("wingover_lite", "boundary_evidence", "diagnostic_boundary_evidence", False, "not_hardware_ready_without_W2_W3_and_real_safety_evidence", "Boundary-only unless explicitly promoted later."),
    PrimitiveRole("bank_yaw_energy_retaining", "boundary_evidence", "diagnostic_boundary_evidence", False, "not_hardware_ready_without_W2_W3_and_real_safety_evidence", "Retain as archive evidence, not final mission dependency."),
    PrimitiveRole("large_agile_reversal", "boundary_evidence", "diagnostic_boundary_evidence", False, "not_hardware_ready_without_W2_W3_and_real_safety_evidence", "Trajectory-optimised/high-angle reference only."),
)


def primitive_role_table() -> pd.DataFrame:
    return pd.DataFrame([record.as_dict() for record in (*OPERATIONAL_PRIMITIVES, *DIAGNOSTIC_PRIMITIVES)])


def primitive_role_record(primitive_family: str) -> dict[str, object]:
    matches = primitive_role_table()
    rows = matches[matches["primitive_family"].astype(str) == str(primitive_family)]
    if rows.empty:
        return {
            "primitive_family": str(primitive_family),
            "mission_role": "unknown",
            "fixed_gate_use": "not_in_fixed_gate_shortlist",
            "final_mission_required": False,
            "hardware_readiness_status": "not_assessed",
            "note": "Unknown primitive family is not part of the fixed-gate mission shortlist.",
        }
    return rows.iloc[0].to_dict()


def active_mission_primitive_families() -> tuple[str, ...]:
    return tuple(record.primitive_family for record in OPERATIONAL_PRIMITIVES)


def diagnostic_boundary_primitive_families() -> tuple[str, ...]:
    return tuple(record.primitive_family for record in DIAGNOSTIC_PRIMITIVES)
