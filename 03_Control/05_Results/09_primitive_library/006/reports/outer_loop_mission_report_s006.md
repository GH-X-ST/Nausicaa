# Outer-Loop Mission Simulation Report

Run-006 uses the run-005 offline governor seed layer to select or reject existing primitive seeds.
It does not add primitives, implement OCP/TVLQR, touch hardware, or claim real-flight readiness.

## Governor Seeds

- `glide_none_favourable_U4_four_fan_W2_dp1`: role `glide_transit`, updraft `U4_four_fan`, wind `W2`
- `recovery_none_favourable_U4_four_fan_W1_dp1`: role `recovery_fallback`, updraft `U4_four_fan`, wind `W1`
- `mild_bank_none_favourable_U1_single_fan_W2_dp1`: role `mild_bank_updraft_encounter`, updraft `U1_single_fan`, wind `W2`
- `glide_none_favourable_U1_single_fan_W2_dp1`: role `environment_comparison`, updraft `U1_single_fan`, wind `W2`

## Target Steering

`bank_yaw_energy_retaining_015_favourable_U1_single_fan_W1_dp1` remains excluded with status `excluded_marginal_target_steering` and is not selectable in run-006.

## Mission Outcomes

- `U1_lift_sector_governed_transit`: `completed_with_governor_evidence`, stop `no_candidate_accepted_by_governor_clearance`, accepted `1` steps, energy delta `-0.046 m`
- `U4_lift_sector_governed_transit`: `completed_with_governor_evidence`, stop `no_candidate_accepted_by_governor_clearance`, accepted `1` steps, energy delta `-0.064 m`
- `low_lift_confidence_rejection`: `no_go_coverage_gap`, stop `no_candidate_accepted_by_governor_lift_belief`, accepted `0` steps, energy delta `0.000 m`
- `clearance_limited_no_go`: `no_go_coverage_gap`, stop `no_candidate_accepted_by_governor_clearance`, accepted `0` steps, energy delta `0.000 m`

## Lift Dwell

- `U1_lift_sector_governed_transit`: dwell fraction `1.000`, mean energy residual `-0.023 m`
- `U4_lift_sector_governed_transit`: dwell fraction `1.000`, mean energy residual `-0.032 m`
- `low_lift_confidence_rejection`: dwell fraction `0.000`, mean energy residual `0.000 m`
- `clearance_limited_no_go`: dwell fraction `1.000`, mean energy residual `0.000 m`

## Coverage Gaps

- `U1_lift_sector_governed_transit`: `short_mission_supported_without_target_steering`, next `proceed_to_ablation`, higher target `not_requested_current_library_sufficient_for_test`
- `U4_lift_sector_governed_transit`: `short_mission_supported_without_target_steering`, next `proceed_to_ablation`, higher target `not_requested_current_library_sufficient_for_test`
- `clearance_limited_no_go`: `clearance_limited`, next `entry_envelope_or_start_state_restriction`, higher target `not_requested_clearance_limited`
- `low_lift_confidence_rejection`: `lift_belief_limited`, next `improve_lift_belief_or_recovery_policy`, higher target `not_requested_lift_belief_limited`

## No-Overclaiming

- Outer-loop mission simulation implemented: `true`
- Online flight-ready governor: `false`
- Target steering used: `false`
- Higher-target primitives added: `false`
- Hardware or real-flight validation claim: `false`
