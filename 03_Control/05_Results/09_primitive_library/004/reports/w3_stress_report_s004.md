# Selected W3 Stress Report

This run executes selected W3 simulation stress for the five run-003 primitive-library candidates.
It is not a governor, outer-loop, OCP, TVLQR, hardware, real-flight, or high-incidence validation pass.

## Source W3 Plan

- Selected candidates: `5`
- Trial rows: `125`
- Trial successes: `112`

## Candidate-Level W3 Results

| source_primitive_id | role | status | success_fraction | recommendation |
|---|---|---:|---:|---|
| bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1 | target_steering | w3_marginal | 0.480 | refine_seed_before_governor |
| glide_none_favourable_U4_four_fan_W2_dp1 | glide_transit | w3_supported | 1.000 | send_to_governor_seed |
| recovery_none_favourable_U4_four_fan_W1_dp1 | recovery_fallback | w3_supported | 1.000 | send_to_governor_seed |
| mild_bank_none_favourable_U1_single_fan_W2_dp1 | mild_bank_updraft_encounter | w3_supported | 1.000 | send_to_governor_seed |
| glide_none_favourable_U1_single_fan_W2_dp1 | environment_comparison | w3_supported | 1.000 | send_to_governor_seed |

## Coverage Update

- `bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1`: `w3_marginal_needs_refinement` -> `refine_seed_before_governor`
- `glide_none_favourable_U4_four_fan_W2_dp1`: `w3_supported_pending_governor` -> `send_to_governor_seed`
- `recovery_none_favourable_U4_four_fan_W1_dp1`: `w3_supported_pending_governor` -> `send_to_governor_seed`
- `mild_bank_none_favourable_U1_single_fan_W2_dp1`: `w3_supported_pending_governor` -> `send_to_governor_seed`
- `glide_none_favourable_U1_single_fan_W2_dp1`: `w3_supported_pending_governor` -> `send_to_governor_seed`

## Dominant Failure Mechanisms

- `none`: `112`
- `turn_authority_limited`: `13`

## Recommended Next Step Toward Governor

- `proceed_to_governor_seed`

## No-Overclaiming Statement

- W3 stress simulation evidence: `true`
- Governor implemented: `false`
- Outer-loop implemented: `false`
- OCP/TVLQR implemented: `false`
- Hardware or real-flight validation claim: `false`
- High-incidence validation claim: `false`
