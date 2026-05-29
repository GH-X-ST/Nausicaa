from __future__ import annotations

from dataclasses import asdict, dataclass

from env_instance import EnvironmentRandomisationConfig


ROBUST_EVIDENCE_DISTRIBUTION_VERSION = "anchor_plus_r10_uncertainty_family_v1"
ROBUST_ACTIVE_FAN_COUNT_SEQUENCE = (0, 1, 2, 3, 4)
ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M = ((0.0, 8.0), (0.0, 4.8))
ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M = 0.5
FIXED_RANDOM_SCALE_RANGE = (1.0, 1.0)
DEFAULT_RANDOM_POWER_SCALE_RANGE = (0.85, 1.15)
DEFAULT_RANDOM_WIDTH_SCALE_RANGE = (0.85, 1.15)
DEFAULT_RANDOM_UNCERTAINTY_SCALE_RANGE = (1.0, 1.5)


@dataclass(frozen=True)
class EvidenceBlockSpec:
    block_id: str
    human_label: str
    W_layer: str
    environment_mode: str
    uncertainty_tier: str
    stage_role: str
    active_fan_count_policy: str
    fan_position_policy: str
    fixed_active_fan_count: int | None = None
    active_fan_count_sequence: tuple[int, ...] = ()
    fan_power_scale_range: tuple[float, float] = DEFAULT_RANDOM_POWER_SCALE_RANGE
    width_scale_range: tuple[float, float] = DEFAULT_RANDOM_WIDTH_SCALE_RANGE
    uncertainty_scale_range: tuple[float, float] = DEFAULT_RANDOM_UNCERTAINTY_SCALE_RANGE
    fan_position_shift_range_m: tuple[float, float] = (-0.20, 0.20)
    fan_position_xy_bounds_m: tuple[tuple[float, float], tuple[float, float]] = ARENA_WIDE_FAN_POSITION_XY_BOUNDS_M
    fan_position_safety_radius_m: float = ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M


def _fixed_block_config() -> dict[str, object]:
    return {
        "fan_position_policy": "fixed_base_positions",
        "fan_power_scale_range": FIXED_RANDOM_SCALE_RANGE,
        "width_scale_range": FIXED_RANDOM_SCALE_RANGE,
        "uncertainty_scale_range": FIXED_RANDOM_SCALE_RANGE,
        "fan_position_shift_range_m": (0.0, 0.0),
    }


R5_EVIDENCE_BLOCKS: tuple[EvidenceBlockSpec, ...] = (
    EvidenceBlockSpec(
        block_id="r5_anchor_dry_air",
        human_label="R5 dry-air anchor",
        W_layer="W0",
        environment_mode="dry_air",
        uncertainty_tier="anchor_dry_air",
        stage_role="control_anchor",
        active_fan_count_policy="fixed_zero_active_fans",
        fan_position_policy="no_fan_positions",
        fixed_active_fan_count=0,
        fan_power_scale_range=FIXED_RANDOM_SCALE_RANGE,
        width_scale_range=FIXED_RANDOM_SCALE_RANGE,
        uncertainty_scale_range=FIXED_RANDOM_SCALE_RANGE,
    ),
    EvidenceBlockSpec(
        block_id="r5_anchor_single_fan_fixed",
        human_label="R5 fixed single-fan anchor",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_single",
        uncertainty_tier="anchor_fixed_single_fan",
        stage_role="control_anchor",
        active_fan_count_policy="fixed_one_active_fan",
        fixed_active_fan_count=1,
        **_fixed_block_config(),
    ),
    EvidenceBlockSpec(
        block_id="r5_anchor_four_fan_fixed",
        human_label="R5 fixed four-fan anchor",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_four",
        uncertainty_tier="anchor_fixed_four_fan",
        stage_role="control_anchor",
        active_fan_count_policy="fixed_four_active_fans",
        fixed_active_fan_count=4,
        **_fixed_block_config(),
    ),
    EvidenceBlockSpec(
        block_id="r5_random_single_fan_local",
        human_label="R5 randomized single-fan local uncertainty",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_single",
        uncertainty_tier="single_fan_local_uncertainty",
        stage_role="uncertainty_family_training",
        active_fan_count_policy="fixed_one_active_fan",
        fan_position_policy="common_shift",
        fixed_active_fan_count=1,
    ),
    EvidenceBlockSpec(
        block_id="r5_random_four_fan_parameter",
        human_label="R5 four-fan parameter uncertainty",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_four",
        uncertainty_tier="four_fan_parameter_uncertainty",
        stage_role="uncertainty_family_training",
        active_fan_count_policy="fixed_four_active_fans",
        fan_position_policy="fixed_base_positions",
        fixed_active_fan_count=4,
    ),
    EvidenceBlockSpec(
        block_id="r5_random_active_fan_count",
        human_label="R5 active fan-count uncertainty",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_four",
        uncertainty_tier="active_fan_count_uncertainty",
        stage_role="uncertainty_family_training",
        active_fan_count_policy="balanced_0_1_2_3_4_for_active_fan_number_variation",
        fan_position_policy="fixed_base_positions",
        active_fan_count_sequence=ROBUST_ACTIVE_FAN_COUNT_SEQUENCE,
    ),
    EvidenceBlockSpec(
        block_id="r5_random_local_fan_position",
        human_label="R5 local fan-position uncertainty",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_four",
        uncertainty_tier="local_fan_position_uncertainty",
        stage_role="uncertainty_family_training",
        active_fan_count_policy="fixed_four_active_fans",
        fan_position_policy="common_shift",
        fixed_active_fan_count=4,
    ),
    EvidenceBlockSpec(
        block_id="r5_random_arena_wide",
        human_label="R5 arena-wide full randomisation",
        W_layer="W1",
        environment_mode="w1_annular_gp_randomised_four",
        uncertainty_tier="arena_wide_full_randomisation",
        stage_role="uncertainty_family_training",
        active_fan_count_policy="balanced_0_1_2_3_4_with_arena_wide_fan_position_generalisation",
        fan_position_policy="independent_uniform_xy_bounds",
        active_fan_count_sequence=ROBUST_ACTIVE_FAN_COUNT_SEQUENCE,
    ),
)


R7_EVIDENCE_BLOCKS: tuple[EvidenceBlockSpec, ...] = (
    EvidenceBlockSpec(
        block_id="r7_anchor_dry_air",
        human_label="R7 dry-air anchor",
        W_layer="W0",
        environment_mode="dry_air",
        uncertainty_tier="anchor_dry_air",
        stage_role="heldout_control_anchor",
        active_fan_count_policy="fixed_zero_active_fans",
        fan_position_policy="no_fan_positions",
        fixed_active_fan_count=0,
        fan_power_scale_range=FIXED_RANDOM_SCALE_RANGE,
        width_scale_range=FIXED_RANDOM_SCALE_RANGE,
        uncertainty_scale_range=FIXED_RANDOM_SCALE_RANGE,
    ),
    EvidenceBlockSpec(
        block_id="r7_anchor_single_fan_fixed",
        human_label="R7 fixed single-fan anchor",
        W_layer="W3",
        environment_mode="w3_randomised_single",
        uncertainty_tier="anchor_fixed_single_fan",
        stage_role="heldout_control_anchor",
        active_fan_count_policy="fixed_one_active_fan",
        fixed_active_fan_count=1,
        **_fixed_block_config(),
    ),
    EvidenceBlockSpec(
        block_id="r7_anchor_four_fan_fixed",
        human_label="R7 fixed four-fan anchor",
        W_layer="W3",
        environment_mode="w3_randomised_four",
        uncertainty_tier="anchor_fixed_four_fan",
        stage_role="heldout_control_anchor",
        active_fan_count_policy="fixed_four_active_fans",
        fixed_active_fan_count=4,
        **_fixed_block_config(),
    ),
    EvidenceBlockSpec(
        block_id="r7_random_single_fan_local",
        human_label="R7 randomized single-fan local uncertainty",
        W_layer="W3",
        environment_mode="w3_randomised_single",
        uncertainty_tier="single_fan_local_uncertainty",
        stage_role="heldout_uncertainty_validation",
        active_fan_count_policy="fixed_one_active_fan",
        fan_position_policy="common_shift",
        fixed_active_fan_count=1,
    ),
    EvidenceBlockSpec(
        block_id="r7_random_four_fan_parameter",
        human_label="R7 four-fan parameter uncertainty",
        W_layer="W3",
        environment_mode="w3_randomised_four",
        uncertainty_tier="four_fan_parameter_uncertainty",
        stage_role="heldout_uncertainty_validation",
        active_fan_count_policy="fixed_four_active_fans",
        fan_position_policy="fixed_base_positions",
        fixed_active_fan_count=4,
    ),
    EvidenceBlockSpec(
        block_id="r7_random_active_fan_count",
        human_label="R7 active fan-count uncertainty",
        W_layer="W3",
        environment_mode="w3_randomised_four",
        uncertainty_tier="active_fan_count_uncertainty",
        stage_role="heldout_uncertainty_validation",
        active_fan_count_policy="balanced_0_1_2_3_4_for_active_fan_number_variation",
        fan_position_policy="fixed_base_positions",
        active_fan_count_sequence=ROBUST_ACTIVE_FAN_COUNT_SEQUENCE,
    ),
    EvidenceBlockSpec(
        block_id="r7_random_local_fan_position",
        human_label="R7 local fan-position uncertainty",
        W_layer="W3",
        environment_mode="w3_randomised_four",
        uncertainty_tier="local_fan_position_uncertainty",
        stage_role="heldout_uncertainty_validation",
        active_fan_count_policy="fixed_four_active_fans",
        fan_position_policy="common_shift",
        fixed_active_fan_count=4,
    ),
    EvidenceBlockSpec(
        block_id="r7_random_arena_wide",
        human_label="R7 arena-wide full randomisation",
        W_layer="W3",
        environment_mode="w3_randomised_four",
        uncertainty_tier="arena_wide_full_randomisation",
        stage_role="heldout_uncertainty_validation",
        active_fan_count_policy="balanced_0_1_2_3_4_with_arena_wide_fan_position_generalisation",
        fan_position_policy="independent_uniform_xy_bounds",
        active_fan_count_sequence=ROBUST_ACTIVE_FAN_COUNT_SEQUENCE,
    ),
)


def evidence_block_ids(blocks: tuple[EvidenceBlockSpec, ...]) -> tuple[str, ...]:
    return tuple(block.block_id for block in blocks)


def unique_environment_cases(blocks: tuple[EvidenceBlockSpec, ...]) -> tuple[tuple[str, str], ...]:
    seen: set[tuple[str, str]] = set()
    cases: list[tuple[str, str]] = []
    for block in blocks:
        case = (str(block.W_layer), str(block.environment_mode))
        if case not in seen:
            seen.add(case)
            cases.append(case)
    return tuple(cases)


def unique_environment_modes(blocks: tuple[EvidenceBlockSpec, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    modes: list[str] = []
    for block in blocks:
        mode = str(block.environment_mode)
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return tuple(modes)


def evidence_block_for_index(blocks: tuple[EvidenceBlockSpec, ...], index: int) -> EvidenceBlockSpec:
    if not blocks:
        raise ValueError("evidence block list must not be empty")
    return blocks[int(index) % len(blocks)]


def evidence_block_by_id(blocks: tuple[EvidenceBlockSpec, ...], block_id: str) -> EvidenceBlockSpec:
    for block in blocks:
        if block.block_id == str(block_id):
            return block
    raise KeyError(f"unknown evidence block id: {block_id}")


def scheduled_active_fan_count(block: EvidenceBlockSpec, paired_start_index: int) -> int:
    if block.fixed_active_fan_count is not None:
        return int(block.fixed_active_fan_count)
    if block.active_fan_count_sequence:
        sequence = tuple(int(value) for value in block.active_fan_count_sequence)
        return int(sequence[int(paired_start_index) % len(sequence)])
    if str(block.W_layer).upper() == "W0" or str(block.environment_mode) == "dry_air":
        return 0
    if "single" in str(block.environment_mode):
        return 1
    return 4


def randomisation_config_for_block(block: EvidenceBlockSpec, paired_start_index: int) -> EnvironmentRandomisationConfig | None:
    if str(block.W_layer).upper() == "W0" or str(block.environment_mode) == "dry_air":
        return None
    return EnvironmentRandomisationConfig(
        fan_position_shift_range_m=tuple(float(value) for value in block.fan_position_shift_range_m),
        fan_position_policy=str(block.fan_position_policy),
        fan_position_xy_bounds_m=tuple(
            tuple(float(axis_value) for axis_value in axis)
            for axis in block.fan_position_xy_bounds_m
        ),
        fan_position_safety_radius_m=float(block.fan_position_safety_radius_m),
        fan_power_scale_range=tuple(float(value) for value in block.fan_power_scale_range),
        active_fan_count=scheduled_active_fan_count(block, paired_start_index),
        width_scale_range=tuple(float(value) for value in block.width_scale_range),
        uncertainty_scale_range=tuple(float(value) for value in block.uncertainty_scale_range),
    )


def evidence_block_manifest_rows(blocks: tuple[EvidenceBlockSpec, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order_index, block in enumerate(blocks):
        row = asdict(block)
        row["order_index"] = int(order_index)
        row["active_fan_count_sequence"] = list(block.active_fan_count_sequence)
        row["fan_position_xy_bounds_m"] = [list(axis) for axis in block.fan_position_xy_bounds_m]
        rows.append(row)
    return rows
