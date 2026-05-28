# R9 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `30`
- Expected history launches: `300`
- Gate profile: `internal_reduced_fixed_case_preflight_for_r10_initialisation`
- Safety thresholds: hard failure <= `0.2`, no-viable <= `0.3`, safe success >= `0.2`, full safe success >= `None`, terminal/lift >= `0.3`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, updraft gain, flight time, and residual memory.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_updraft_gain_multiplicative_launch_score_v4`; rewards safe valid flight time and updraft-gain proxy, while net/gross energy drift remains audit-only.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `30` required `30`
- `history_launch_count`: `True` observed `300` required `300`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `2` required `2`
- `pairing_audit`: `True` observed `3` required `3`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `10.0` required `>=0.1`
- `launch_inflight_recovery_sequence_policy`: `True` observed `330` required `330`
- `hard_failure_rate_within_stage_profile`: `False` observed `0.7333333333333333` required `0.2`
- `floor_or_ceiling_violation_rate_zero`: `False` observed `0.3333333333333333` required `0.0`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.0` required `0.3`
- `safe_success_rate_within_stage_profile`: `True` observed `0.26666666666666666` required `0.2`
- `terminal_or_lift_capture_within_stage_profile`: `True` observed `0.6666666666666666` required `0.3`
- `selected_primitive_family_count_ge_5`: `False` observed `4` required `5`
- `selected_variant_count_ge_10`: `True` observed `12` required `10`
