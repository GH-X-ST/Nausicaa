from __future__ import annotations

import math
from dataclasses import dataclass, asdict


PRIMITIVE_FINITE_HORIZON_S = 0.100
CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE = 5
CONTROLLER_INPUT_UPDATE_PERIOD_S = 0.020
PRIMITIVE_TIMING_CONTRACT_VERSION = "v411_0p10s_5slot_20ms"
LAUNCH_HANDOFF_DURATION_S = 0.040
LAUNCH_HANDOFF_POLICY_VERSION = "launch_gate_neutral_handoff_0p040s_v1"


@dataclass(frozen=True)
class PrimitiveTimingContract:
    finite_horizon_s: float = PRIMITIVE_FINITE_HORIZON_S
    controller_input_slots_per_primitive: int = CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE
    controller_input_update_period_s: float = CONTROLLER_INPUT_UPDATE_PERIOD_S
    primitive_timing_contract_version: str = PRIMITIVE_TIMING_CONTRACT_VERSION


V411_PRIMITIVE_TIMING_CONTRACT = PrimitiveTimingContract()


def primitive_timing_contract_row() -> dict[str, object]:
    """Return the v4.11 timing contract as CSV/JSON-ready metadata."""

    row = asdict(V411_PRIMITIVE_TIMING_CONTRACT)
    row.update(
        {
            "launch_handoff_duration_s": LAUNCH_HANDOFF_DURATION_S,
            "launch_handoff_policy_version": LAUNCH_HANDOFF_POLICY_VERSION,
        }
    )
    return row


def primitive_step_count(
    *,
    finite_horizon_s: float = PRIMITIVE_FINITE_HORIZON_S,
    controller_input_update_period_s: float = CONTROLLER_INPUT_UPDATE_PERIOD_S,
) -> int:
    """Return the exact number of controller updates in one primitive."""

    if float(controller_input_update_period_s) <= 0.0:
        raise ValueError("controller_input_update_period_s must be positive")
    ratio = float(finite_horizon_s) / float(controller_input_update_period_s)
    nearest = int(round(ratio))
    if nearest <= 0 or not math.isclose(ratio, float(nearest), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "primitive finite_horizon_s must be an integer multiple of "
            "controller_input_update_period_s"
        )
    return nearest


def assert_primitive_timing_contract(
    *,
    finite_horizon_s: float,
    controller_input_slots_per_primitive: int,
    controller_input_update_period_s: float,
    primitive_timing_contract_version: str,
) -> None:
    """Raise if metadata does not match the v4.11 primitive timing contract."""

    if not math.isclose(float(finite_horizon_s), PRIMITIVE_FINITE_HORIZON_S, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("finite_horizon_s_not_v411_0p100s")
    if int(controller_input_slots_per_primitive) != CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE:
        raise ValueError("controller_input_slots_per_primitive_not_5")
    if not math.isclose(
        float(controller_input_update_period_s),
        CONTROLLER_INPUT_UPDATE_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("controller_input_update_period_s_not_0p020s")
    if str(primitive_timing_contract_version) != PRIMITIVE_TIMING_CONTRACT_VERSION:
        raise ValueError("primitive_timing_contract_version_not_v411")
    if primitive_step_count(
        finite_horizon_s=finite_horizon_s,
        controller_input_update_period_s=controller_input_update_period_s,
    ) != CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE:
        raise ValueError("primitive_timing_contract_step_count_not_5")


def primitive_timing_contract_status(
    *,
    finite_horizon_s: object,
    controller_input_slots_per_primitive: object,
    controller_input_update_period_s: object,
    primitive_timing_contract_version: object,
) -> tuple[str, str]:
    """Return (status, reason) for manifest and audit rows."""

    try:
        assert_primitive_timing_contract(
            finite_horizon_s=float(finite_horizon_s),
            controller_input_slots_per_primitive=int(controller_input_slots_per_primitive),
            controller_input_update_period_s=float(controller_input_update_period_s),
            primitive_timing_contract_version=str(primitive_timing_contract_version),
        )
    except Exception as exc:
        return "noncompliant", str(exc)
    return "compliant", ""
