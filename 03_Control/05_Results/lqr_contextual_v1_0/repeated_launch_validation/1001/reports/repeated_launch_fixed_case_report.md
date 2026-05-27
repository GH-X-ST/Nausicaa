# R9 Repeated-Launch Validation

- Status: `dry_run_schedule`
- Pass gate: `False`
- Expected final held-out launches: `4200`
- Expected history launches: `111000`
- Launch sequence policy: `first_0p10s_launch_capture_then_inflight_then_recovery_safe_exit`
- Recovery route: `inflight_boundary_near` below `0.25` m safe margin, `inflight_recovery_edge` for degraded speed, attitude, rate, or boundary contact.
- Launch score: `r9_r10_specific_energy_multiplicative_launch_score_v1`; paired score deltas are audit evidence, not pass-gate substitutes.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `4200` required `4200`
- `history_launch_count`: `True` observed `111000` required `111000`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `14` required `14`
- `pairing_audit`: `True` observed `60` required `60`
- `max_primitives_per_launch_full_validation`: `True` observed `12` required `>=4`
- `final_rollout_rows_present`: `False` observed `0` required `4200`
