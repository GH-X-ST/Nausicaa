from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from lqr_controller import (
    LQR_SYNTHESIS_SOLVED,
    lqr_command_for_state,
    lqr_controller_for_primitive_id,
    lqr_controller_metadata_row,
    synthesis_audit_row,
)
from lqr_linearisation import LQR_STATE_MASK, ZERO_POSITION_GAIN_STATES
from prim_cat import ACTIVE_PRIMITIVE_IDS, primitive_by_id
from state_contract import STATE_INDEX


def test_lqr_controller_audit_contract_for_all_primitives() -> None:
    assert ACTIVE_PRIMITIVE_IDS == (
        "glide",
        "recovery",
        "lift_entry",
        "lift_dwell_arc",
        "mild_turn_left",
        "mild_turn_right",
        "energy_retaining_bank",
        "safe_exit_or_recovery_handoff",
    )

    for primitive_id in ACTIVE_PRIMITIVE_IDS:
        controller = lqr_controller_for_primitive_id(primitive_id)
        metadata = lqr_controller_metadata_row(controller)
        audit = synthesis_audit_row(primitive_by_id(primitive_id))

        assert controller.controller_family == "lqr"
        assert controller.controller_id.startswith(f"lqr_{primitive_id}_")
        assert controller.lqr_synthesis_status == LQR_SYNTHESIS_SOLVED
        assert controller.reduced_order_lqr is True
        assert tuple(json.loads(metadata["lqr_state_mask_json"])) == LQR_STATE_MASK
        assert metadata["zero_position_gain_expansion_status"] == "zero_position_gains_verified"
        assert metadata["sampled_data_check_status"] == "sampled_stable"
        assert metadata["latency_actuator_survival_status"] in {
            "survives_nominal_latency_actuator_lag",
            "latency_margin_warning",
        }
        assert audit["primitive_id"] == primitive_id
        assert audit["full_state_care_status"] in {"solved", "unsuitable_use_reduced_order"}
        assert math.isfinite(float(metadata["care_residual_norm"]))
        assert float(metadata["care_residual_norm"]) < 1.0e-8
        assert float(metadata["sampled_data_spectral_radius"]) < 1.0

        gain = np.asarray(controller.k_gain_matrix, dtype=float)
        assert gain.shape == (3, 15)
        for state_name in ZERO_POSITION_GAIN_STATES:
            assert np.allclose(gain[:, STATE_INDEX[state_name]], 0.0)


def test_blocked_lqr_controller_cannot_emit_executable_zero_command() -> None:
    controller = replace(
        lqr_controller_for_primitive_id("glide"),
        lqr_synthesis_status="blocked_lqr_synthesis",
        lqr_blocked_reason="unit_test_blocked",
    )

    with pytest.raises(RuntimeError, match="blocked LQR controller"):
        lqr_command_for_state(controller=controller, state_vector=np.zeros(15))
