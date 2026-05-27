# R9 Repeated-Launch Validation

- Status: `smoke_run`
- Pass gate: `False`
- Expected final held-out launches: `4200`
- Expected history launches: `111000`
- Launch sequence policy: `first_0p10s_launch_capture_then_inflight_then_recovery_safe_exit`
- Recovery route: `inflight_boundary_near` below `0.25` m safe margin, `inflight_recovery_edge` for degraded speed, attitude, rate, or boundary contact.
- Launch score: `r9_r10_specific_energy_multiplicative_launch_score_v1`; paired score deltas are audit evidence, not pass-gate substitutes.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `False` observed `210` required `4200`
- `history_launch_count`: `False` observed `5550` required `111000`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `14` required `14`
- `pairing_audit`: `True` observed `3` required `3`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `0.3` required `>=0.1`
- `launch_inflight_recovery_sequence_policy`: `False` observed `4070` required `5760`
- `hard_failure_rate_le_1pct`: `False` observed `0.3333333333333333` required `0.01`
- `floor_or_ceiling_violation_rate_zero`: `True` observed `0.0` required `0.0`
- `no_viable_primitive_rate_le_2pct`: `False` observed `0.3333333333333333` required `0.02`
- `safe_success_rate_near_100pct`: `False` observed `0.3333333333333333` required `0.99`
- `terminal_or_lift_capture_ge_90pct`: `False` observed `0.0` required `0.9`
- `selected_primitive_family_count_ge_5`: `False` observed `3` required `5`
- `selected_variant_count_ge_10`: `False` observed `3` required `10`
