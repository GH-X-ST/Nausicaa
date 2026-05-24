from __future__ import annotations

from collections import Counter

from state_sampling import start_state_family_for_row


def test_w01_mixed_start_sampler_distribution_is_exact_over_twenty_rows() -> None:
    counts = Counter(start_state_family_for_row(index) for index in range(20))

    assert counts == {
        "launch_gate": 8,
        "inflight_nominal": 5,
        "inflight_lift_region": 3,
        "inflight_boundary_near": 2,
        "inflight_recovery_edge": 2,
    }
