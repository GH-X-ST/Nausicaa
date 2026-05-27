# R9 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `210`
- Expected history launches: `5550`
- Gate profile: `internal_reduced_fixed_case_preflight_for_r10_initialisation`
- Safety thresholds: hard failure <= `0.2`, no-viable <= `0.3`, safe success >= `0.2`, full safe success >= `None`, terminal/lift >= `0.3`.
- Launch sequence policy: `first_0p10s_launch_capture_then_inflight_then_recovery_safe_exit`
- Recovery route: `inflight_boundary_near` below `0.25` m safe margin, `inflight_recovery_edge` for degraded attitude, rate, or boundary contact.
- Launch score: `r9_r10_r11_updraft_gain_multiplicative_launch_score_v3`; rewards safe valid flight time and updraft-gain proxy, while net/gross energy drift remains audit-only.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `210` required `210`
- `history_launch_count`: `True` observed `5550` required `5550`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `14` required `14`
- `pairing_audit`: `True` observed `3` required `3`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `10.0` required `>=0.1`
- `launch_inflight_recovery_sequence_policy`: `True` observed `5760` required `5760`
- `hard_failure_rate_within_stage_profile`: `False` observed `0.8` required `0.2`
- `floor_or_ceiling_violation_rate_zero`: `False` observed `0.3333333333333333` required `0.0`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.0` required `0.3`
- `safe_success_rate_within_stage_profile`: `True` observed `0.2` required `0.2`
- `terminal_or_lift_capture_within_stage_profile`: `True` observed `0.6666666666666666` required `0.3`
- `selected_primitive_family_count_ge_5`: `True` observed `5` required `5`
- `selected_variant_count_ge_10`: `True` observed `11` required `10`
