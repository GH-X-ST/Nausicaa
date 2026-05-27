# R10 Repeated-Launch Validation

- Status: `dry_run_schedule`
- Pass gate: `False`
- Expected final held-out launches: `8400`
- Expected history launches: `222000`
- Launch sequence policy: `first_0p10s_launch_capture_then_inflight_then_recovery_safe_exit`
- Recovery route: `inflight_boundary_near` below `0.25` m safe margin, `inflight_recovery_edge` for degraded speed, attitude, rate, or boundary contact.
- Launch score: `r9_r10_specific_energy_multiplicative_launch_score_v1`; paired score deltas are audit evidence, not pass-gate substitutes.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `8400` required `8400`
- `history_launch_count`: `True` observed `222000` required `222000`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `14` required `14`
- `pairing_audit`: `True` observed `120` required `120`
- `max_primitives_per_launch_full_validation`: `True` observed `12` required `>=4`
- `no_glider_latency_variation_audit`: `True` observed `120` required `120`
- `final_rollout_rows_present`: `False` observed `0` required `8400`
