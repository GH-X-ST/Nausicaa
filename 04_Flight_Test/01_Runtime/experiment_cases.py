from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    case_name: str
    layout_id: str
    controller_mode: str
    memory_enabled: bool
    expected_visible_fan_min: int
    expected_visible_fan_max: int
    target_valid_throws: int
    target_session_repeats: int
    evidence_role: str


def _case(
    case_id: str,
    case_name: str,
    *,
    layout_id: str,
    controller_mode: str = "closed_loop",
    memory_enabled: bool,
    expected_visible_fan_min: int,
    expected_visible_fan_max: int,
    target_valid_throws: int,
    target_session_repeats: int = 1,
    evidence_role: str = "real_flight_evidence",
) -> ExperimentCase:
    if target_valid_throws <= 0:
        raise ValueError("target_valid_throws must be positive.")
    if target_session_repeats <= 0:
        raise ValueError("target_session_repeats must be positive.")
    return ExperimentCase(
        case_id=case_id,
        case_name=case_name,
        layout_id=layout_id,
        controller_mode=controller_mode,
        memory_enabled=memory_enabled,
        expected_visible_fan_min=expected_visible_fan_min,
        expected_visible_fan_max=expected_visible_fan_max,
        target_valid_throws=target_valid_throws,
        target_session_repeats=target_session_repeats,
        evidence_role=evidence_role,
    )


EXPERIMENT_CASES: dict[str, ExperimentCase] = {
    "E0.1": _case(
        "E0.1",
        "Dry-air shakedown before E1",
        layout_id="E0_dry_air_shakedown",
        memory_enabled=False,
        expected_visible_fan_min=0,
        expected_visible_fan_max=0,
        target_valid_throws=5,
        evidence_role="shakedown_only",
    ),
    "E0.2": _case(
        "E0.2",
        "Four-fan tracked-layout shakedown before E1",
        layout_id="E0_four_fan_shakedown",
        memory_enabled=True,
        expected_visible_fan_min=4,
        expected_visible_fan_max=4,
        target_valid_throws=5,
        evidence_role="shakedown_only",
    ),
    "E1.0": _case(
        "E1.0",
        "Dry air, open-loop neutral",
        layout_id="E1_dry_air",
        controller_mode="open_loop_neutral",
        memory_enabled=False,
        expected_visible_fan_min=0,
        expected_visible_fan_max=0,
        target_valid_throws=10,
        evidence_role="open_loop_real_flight_baseline",
    ),
    "E1.1": _case("E1.1", "Dry air, no memory", layout_id="E1_dry_air", memory_enabled=False, expected_visible_fan_min=0, expected_visible_fan_max=0, target_valid_throws=30),
    "E1.2": _case("E1.2", "Dry air, memory enabled", layout_id="E1_dry_air", memory_enabled=True, expected_visible_fan_min=0, expected_visible_fan_max=0, target_valid_throws=30),
    "E2.0": _case(
        "E2.0",
        "Single fan fixed, open-loop neutral",
        layout_id="E2_single_fan_fixed",
        controller_mode="open_loop_neutral",
        memory_enabled=False,
        expected_visible_fan_min=1,
        expected_visible_fan_max=1,
        target_valid_throws=10,
        evidence_role="open_loop_real_flight_baseline",
    ),
    "E2.1": _case("E2.1", "Single fan fixed, no memory", layout_id="E2_single_fan_fixed", memory_enabled=False, expected_visible_fan_min=1, expected_visible_fan_max=1, target_valid_throws=30),
    "E2.2": _case("E2.2", "Single fan fixed, memory enabled", layout_id="E2_single_fan_fixed", memory_enabled=True, expected_visible_fan_min=1, expected_visible_fan_max=1, target_valid_throws=30),
    "E3.0": _case(
        "E3.0",
        "Four fan fixed, open-loop neutral",
        layout_id="E3_four_fan_fixed",
        controller_mode="open_loop_neutral",
        memory_enabled=False,
        expected_visible_fan_min=4,
        expected_visible_fan_max=4,
        target_valid_throws=10,
        evidence_role="open_loop_real_flight_baseline",
    ),
    "E3.1": _case("E3.1", "Four fan fixed, no memory", layout_id="E3_four_fan_fixed", memory_enabled=False, expected_visible_fan_min=4, expected_visible_fan_max=4, target_valid_throws=30),
    "E3.2": _case("E3.2", "Four fan fixed, memory enabled", layout_id="E3_four_fan_fixed", memory_enabled=True, expected_visible_fan_min=4, expected_visible_fan_max=4, target_valid_throws=30),
    "E4a.0": _case(
        "E4a.0",
        "Random fan layout 1, open-loop neutral",
        layout_id="E4a_random_layout_1",
        controller_mode="open_loop_neutral",
        memory_enabled=False,
        expected_visible_fan_min=1,
        expected_visible_fan_max=4,
        target_valid_throws=10,
        evidence_role="open_loop_real_flight_baseline",
    ),
    "E4a.1": _case("E4a.1", "Random fan layout 1, no memory", layout_id="E4a_random_layout_1", memory_enabled=False, expected_visible_fan_min=1, expected_visible_fan_max=4, target_valid_throws=30),
    "E4a.2": _case("E4a.2", "Random fan layout 1, memory enabled", layout_id="E4a_random_layout_1", memory_enabled=True, expected_visible_fan_min=1, expected_visible_fan_max=4, target_valid_throws=30),
    "E4b.0": _case(
        "E4b.0",
        "Random fan layout 2, open-loop neutral",
        layout_id="E4b_random_layout_2",
        controller_mode="open_loop_neutral",
        memory_enabled=False,
        expected_visible_fan_min=1,
        expected_visible_fan_max=4,
        target_valid_throws=10,
        evidence_role="open_loop_real_flight_baseline",
    ),
    "E4b.1": _case("E4b.1", "Random fan layout 2, no memory", layout_id="E4b_random_layout_2", memory_enabled=False, expected_visible_fan_min=1, expected_visible_fan_max=4, target_valid_throws=30),
    "E4b.2": _case("E4b.2", "Random fan layout 2, memory enabled", layout_id="E4b_random_layout_2", memory_enabled=True, expected_visible_fan_min=1, expected_visible_fan_max=4, target_valid_throws=30),
}


def get_experiment_case(case_id: str) -> ExperimentCase:
    key = str(case_id)
    try:
        return EXPERIMENT_CASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(EXPERIMENT_CASES))
        raise ValueError(f"unknown experiment case {key!r}. Known cases: {known}") from exc


def experiment_case_manifest() -> list[dict[str, object]]:
    return [
        {
            "case_id": case.case_id,
            "case_name": case.case_name,
            "layout_id": case.layout_id,
            "controller_mode": case.controller_mode,
            "memory_enabled": case.memory_enabled,
            "expected_visible_fan_min": case.expected_visible_fan_min,
            "expected_visible_fan_max": case.expected_visible_fan_max,
            "target_valid_throws": case.target_valid_throws,
            "target_session_repeats": case.target_session_repeats,
            "total_protocol_valid_throws": case.target_valid_throws * case.target_session_repeats,
            "evidence_role": case.evidence_role,
        }
        for case in EXPERIMENT_CASES.values()
    ]
