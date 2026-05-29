# R10 Repeated-Launch Validation

- Status: `smoke_run`
- Pass gate: `False`
- Expected final held-out launches: `1000`
- Expected history launches: `10750`
- Gate profile: `relaxed_changed_case_viability_governor_learning_not_final_validation`
- Safety thresholds: hard failure <= `0.2`, no-viable <= `0.3`, safe success >= `0.2`, full safe success >= `None`, terminal/lift >= `0.3`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, updraft gain, flight time, and residual memory.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_updraft_gain_multiplicative_launch_score_v4`; rewards safe valid flight time and updraft-gain proxy, while net/gross energy drift remains audit-only.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `False` observed `20` required `1000`
- `history_launch_count`: `False` observed `215` required `10750`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `4` required `4`
- `pairing_audit`: `True` observed `1` required `1`
- `primitive_count_cap_disabled_for_full_validation`: `False` observed `1` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `20.0` required `>=0.1`
- `no_glider_latency_variation_audit`: `True` observed `1` required `1`
- `launch_inflight_recovery_sequence_policy`: `True` observed `235` required `235`
- `hard_failure_rate_within_stage_profile`: `True` observed `0.0` required `0.2`
- `floor_or_ceiling_violation_rate_zero`: `True` observed `0.0` required `0.0`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.0` required `0.3`
- `safe_success_rate_within_stage_profile`: `True` observed `1.0` required `0.2`
- `terminal_or_lift_capture_within_stage_profile`: `False` observed `0.0` required `0.3`
- `selected_primitive_family_count_ge_5`: `False` observed `1` required `5`
- `selected_variant_count_ge_10`: `False` observed `4` required `10`
