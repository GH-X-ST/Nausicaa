from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
INNER_LOOP = ROOT / "03_Control" / "02_Inner_Loop"
if str(INNER_LOOP) not in sys.path:
    sys.path.insert(0, str(INNER_LOOP))

import run_fit_neutral_aero_residual_calibration as sysid  # noqa: E402
from A_model_parameters import mass_properties_estimate  # noqa: E402


def test_default_neutral_sysid_is_longitudinal_primary_with_lateral_diagnostic() -> None:
    assert sysid.DEFAULT_FIT_ATTACHED_LATERAL_COUPLING is False
    assert sysid.DEFAULT_FIT_TRANSITION_LATERAL_COUPLING is False
    assert sysid.DEFAULT_FIT_LATERAL_SURFACES is False
    assert sysid.DEFAULT_FIT_POST_STALL_SURFACES is False
    assert sysid.DEFAULT_FIT_SECONDARY_LATERAL_DIAGNOSTIC is True


def test_launch_confidence_uses_only_lateral_launch_contamination() -> None:
    state = np.zeros(12, dtype=float)
    base_score = sysid.launch_quality_score_from_state(state)
    assert base_score == pytest.approx(0.0)

    longitudinally_messy = state.copy()
    longitudinally_messy[4] = math.radians(15.0)  # theta
    longitudinally_messy[6] = 7.5  # u
    longitudinally_messy[8] = 0.9  # w
    longitudinally_messy[10] = 1.2  # q
    assert sysid.launch_quality_score_from_state(longitudinally_messy) == pytest.approx(base_score)

    laterally_messy = state.copy()
    laterally_messy[3] = math.radians(20.0)  # phi
    laterally_messy[5] = math.radians(20.0)  # psi
    laterally_messy[7] = 1.5  # v
    laterally_messy[9] = 1.2  # p
    laterally_messy[11] = 1.8  # r
    assert sysid.launch_quality_score_from_state(laterally_messy) == pytest.approx(1.0)
    assert sysid.launch_confidence_weight_from_state(laterally_messy) < 1.0


def test_lateral_candidate_application_is_limited_to_minimal_terms() -> None:
    base = sysid.active_parameter_dict()
    coeffs = sysid.zero_coefficients()
    coeffs.update(
        {
            "side_force_bias_coeff": 9.0,
            "side_force_beta_coeff": 0.1,
            "roll_moment_beta_coeff": 9.0,
            "roll_moment_p_hat_coeff": 0.2,
            "yaw_moment_beta_coeff": 9.0,
            "yaw_moment_r_hat_coeff": 0.3,
            "post_stall_lift_residual_coeff": 9.0,
        }
    )
    candidate = sysid.candidate_from_fit(
        base,
        {"coefficients": coeffs},
        apply_attached_cm_bias=False,
        fit_post_stall_surfaces=False,
        fit_post_stall_damping=False,
        fit_attached_lateral_coupling=True,
        fit_transition_lateral_coupling=False,
        fit_lateral_surfaces=False,
        group_scales={
            "attached_lateral": 1.0,
            "post_stall_longitudinal": 0.0,
            "transition_blender": 0.0,
        },
    )

    assert candidate["side_force_beta_coeff"] == pytest.approx(base["side_force_beta_coeff"] + 0.1)
    assert candidate["roll_moment_p_hat_coeff"] == pytest.approx(base["roll_moment_p_hat_coeff"] + 0.2)
    assert candidate["yaw_moment_r_hat_coeff"] == pytest.approx(base["yaw_moment_r_hat_coeff"] + 0.3)
    assert candidate["side_force_bias_coeff"] == pytest.approx(base["side_force_bias_coeff"])
    assert candidate["roll_moment_beta_coeff"] == pytest.approx(base["roll_moment_beta_coeff"])
    assert candidate["yaw_moment_beta_coeff"] == pytest.approx(base["yaw_moment_beta_coeff"])
    assert candidate["post_stall_lift_residual_coeff"] == pytest.approx(base["post_stall_lift_residual_coeff"])


def test_mass_properties_match_ballasted_glider_measurement() -> None:
    assert mass_properties_estimate.MASS_KG == pytest.approx(0.14856)
    assert mass_properties_estimate.R_CG_BUILD_M[0] == pytest.approx(0.1055)
    assert mass_properties_estimate.R_CG_BUILD_M[1] == pytest.approx(0.0)
    inertia = np.asarray(mass_properties_estimate.INERTIA_B, dtype=float)
    assert inertia.shape == (3, 3)
    assert np.allclose(inertia, inertia.T)
