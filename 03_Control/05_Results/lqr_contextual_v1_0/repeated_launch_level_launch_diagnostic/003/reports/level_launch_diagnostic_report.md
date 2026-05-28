# R9_LEVEL_DIAGNOSTIC Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `3`
- Expected history launches: `0`
- Gate profile: `single_case_level_launch_diagnostic_expected_all_safe_terminal`
- Safety thresholds: hard failure <= `0.0`, no-viable <= `0.0`, safe success >= `1.0`, full safe success >= `None`, terminal/lift >= `1.0`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, updraft gain, flight time, and residual memory.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_updraft_gain_multiplicative_launch_score_v4`; rewards safe valid flight time and updraft-gain proxy, while net/gross energy drift remains audit-only.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `3` required `3`
- `history_launch_count`: `True` observed `0` required `0`
- `library_size_case_count`: `True` observed `1` required `1`
- `policy_history_condition_count`: `True` observed `1` required `1`
- `pairing_audit`: `True` observed `3` required `3`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `10.0` required `>=0.1`
- `reduced_diagnostic_not_target_R10`: `False` observed `reduced_diagnostic_not_target_R10` required `full_R10_validation_required`
- `launch_inflight_recovery_sequence_policy`: `True` observed `3` required `3`
- `hard_failure_rate_within_stage_profile`: `True` observed `0.0` required `0.0`
- `floor_or_ceiling_violation_rate_zero`: `True` observed `0.0` required `0.0`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.0` required `0.0`
- `safe_success_rate_within_stage_profile`: `True` observed `1.0` required `1.0`
- `terminal_or_lift_capture_within_stage_profile`: `True` observed `1.0` required `1.0`
- `selected_primitive_family_count_ge_5`: `False` observed `3` required `5`
- `selected_variant_count_ge_10`: `False` observed `6` required `10`
