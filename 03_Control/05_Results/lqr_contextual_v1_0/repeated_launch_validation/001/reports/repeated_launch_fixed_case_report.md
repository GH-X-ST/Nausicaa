# R9 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `3360`
- Expected history launches: `88800`
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `3360` required `3360`
- `history_launch_count`: `True` observed `88800` required `88800`
- `library_size_case_count`: `True` observed `4` required `4`
- `policy_history_condition_count`: `True` observed `14` required `14`
- `pairing_audit`: `True` observed `60` required `60`
- `hard_failure_rate_le_1pct`: `False` observed `0.08333333333333333` required `0.01`
- `floor_or_ceiling_violation_rate_zero`: `True` observed `0.0` required `0.0`
- `no_viable_primitive_rate_le_2pct`: `False` observed `0.8041666666666667` required `0.02`
- `safe_success_rate_near_100pct`: `False` observed `0.1125` required `0.99`
- `terminal_or_lift_capture_ge_90pct`: `False` observed `0.058333333333333334` required `0.9`
- `selected_primitive_family_count_ge_5`: `False` observed `2` required `5`
- `selected_variant_count_ge_10`: `False` observed `2` required `10`
