from __future__ import annotations

import inspect

import numpy as np

from primitive_library_generators import generate_command_profile
from primitive_library_schema import PrimitiveCandidateSpec


TARGET_LADDER_DEG = (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
AGGRESSIVE_FAMILIES = (
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)


def test_generate_command_profile_public_signature_is_preserved() -> None:
    signature = inspect.signature(generate_command_profile)

    assert tuple(signature.parameters) == ("spec", "time_s")


def test_aggressive_target_profiles_are_distinguishable_and_bounded() -> None:
    time_s = np.linspace(0.0, 0.95, 64)

    for family in AGGRESSIVE_FAMILIES:
        profiles = [
            generate_command_profile(_spec(family, target, direction_sign=1), time_s)[0]
            for target in TARGET_LADDER_DEG
        ]
        for profile in profiles:
            assert profile.shape == (time_s.size, 3)
            assert np.all(np.isfinite(profile))
            assert np.max(profile) <= 1.0
            assert np.min(profile) >= -1.0
        for lower, upper in zip(profiles[:-1], profiles[1:]):
            assert not np.allclose(lower, upper)


def test_target_profile_intensity_is_monotonic_and_clipped() -> None:
    time_s = np.linspace(0.0, 0.95, 64)

    for family in AGGRESSIVE_FAMILIES:
        peaks = [
            _lateral_peak(
                generate_command_profile(_spec(family, target, direction_sign=1), time_s)[0]
            )
            for target in TARGET_LADDER_DEG
        ]
        assert peaks == sorted(peaks)
        assert np.allclose(
            generate_command_profile(_spec(family, 5.0, direction_sign=1), time_s)[0],
            generate_command_profile(_spec(family, 15.0, direction_sign=1), time_s)[0],
        )
        assert np.allclose(
            generate_command_profile(_spec(family, 200.0, direction_sign=1), time_s)[0],
            generate_command_profile(_spec(family, 180.0, direction_sign=1), time_s)[0],
        )


def test_direction_sign_flips_lateral_and_yaw_channels_only() -> None:
    time_s = np.linspace(0.0, 0.95, 64)

    for family in AGGRESSIVE_FAMILIES:
        positive = generate_command_profile(
            _spec(family, 90.0, direction_sign=1),
            time_s,
        )[0]
        negative = generate_command_profile(
            _spec(family, 90.0, direction_sign=-1),
            time_s,
        )[0]

        assert np.allclose(positive[:, 0], -negative[:, 0])
        assert np.allclose(positive[:, 2], -negative[:, 2])
        assert np.allclose(positive[:, 1], negative[:, 1])


def test_glide_and_recovery_profiles_ignore_target_heading() -> None:
    time_s = np.linspace(0.0, 0.95, 64)

    for family in ("glide", "recovery"):
        low = generate_command_profile(_spec(family, 15.0, direction_sign=1), time_s)[0]
        high = generate_command_profile(_spec(family, 180.0, direction_sign=1), time_s)[0]

        assert np.allclose(low, high)


def _lateral_peak(profile: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(profile, dtype=float)[:, 0])))


def _spec(
    family: str,
    target_heading_deg: float | None,
    *,
    direction_sign: int,
) -> PrimitiveCandidateSpec:
    return PrimitiveCandidateSpec(
        primitive_id=f"{family}_test",
        parent_primitive_id=family,
        variant_id=f"{family}_test",
        family=family,
        target_heading_deg=target_heading_deg,
        updraft_config="U1_single_fan",
        wind_fidelity="W1",
        start_condition="favourable",
        direction_sign=int(direction_sign),
        horizon_s=0.95,
    )
