# Primitive Library Governor Seed Report

This run converts selected W3 stress evidence into an offline governor accept/reject seed layer.
It does not replay dynamics, implement an outer loop, implement OCP/TVLQR, touch hardware, or claim real-flight readiness.
Clearance is checked numerically as available margin minus the primitive-specific requirement copied from run-002 evidence; case labels do not force clearance rejection.

## Accepted Governor Seeds

- `glide_none_favourable_U4_four_fan_W2_dp1`: role `glide_transit`, status `w3_supported`, updraft `U4_four_fan`, wind `W2`
- `recovery_none_favourable_U4_four_fan_W1_dp1`: role `recovery_fallback`, status `w3_supported`, updraft `U4_four_fan`, wind `W1`
- `mild_bank_none_favourable_U1_single_fan_W2_dp1`: role `mild_bank_updraft_encounter`, status `w3_supported`, updraft `U1_single_fan`, wind `W2`
- `glide_none_favourable_U1_single_fan_W2_dp1`: role `environment_comparison`, status `w3_supported`, updraft `U1_single_fan`, wind `W2`

Accepted seed candidate count: `4`

## Excluded Target Steering

- `bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1` remains `excluded_marginal_target_steering` with W3 status `w3_marginal` and next action `refine_seed_before_governor`.

The target-steering candidate is not governor-allowed in run-005.

## Decision Cases

- Decision cases: `33`
- Accepted decisions: `4`
- Rejected decisions: `29`

| decision_status | case_count | accepted_count | rejected_count |
|---|---:|---:|---:|
| accepted_governor_seed | 4 | 4 | 0 |
| rejected_target_steering_marginal | 1 | 0 | 1 |
| rejected_entry_envelope | 4 | 0 | 4 |
| rejected_clearance | 4 | 0 | 4 |
| rejected_lift_belief | 8 | 0 | 8 |
| rejected_wind_fidelity | 4 | 0 | 4 |
| rejected_recovery_class | 4 | 0 | 4 |
| rejected_model_region | 4 | 0 | 4 |

## Coverage Update

- `bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1` -> `governor_seed_rejected_refine_first` (refine_seed_before_governor)
- `glide_none_favourable_U4_four_fan_W2_dp1` -> `governor_seed_available` (available_for_future_outer_loop_governor_simulation)
- `recovery_none_favourable_U4_four_fan_W1_dp1` -> `governor_seed_available` (available_for_future_outer_loop_governor_simulation)
- `mild_bank_none_favourable_U1_single_fan_W2_dp1` -> `governor_seed_available` (available_for_future_outer_loop_governor_simulation)
- `glide_none_favourable_U1_single_fan_W2_dp1` -> `governor_seed_available` (available_for_future_outer_loop_governor_simulation)

## No-Overclaiming Statement

- Offline governor seed/query layer implemented: `true`
- Online flight-ready governor: `false`
- Outer-loop mission simulation: `false`
- OCP/TVLQR: `false`
- Hardware or real-flight validation claim: `false`
- High-incidence validation claim: `false`

## Next Step

Use the accepted seed table as the input contract for a later outer-loop governor simulation.
Target steering must be refined before it can be considered by that governor.
