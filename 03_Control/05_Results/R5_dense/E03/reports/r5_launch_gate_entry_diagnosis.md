# R5 Launch-Entry Transition Diagnosis

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Status: `complete`
- Rows written: `102400`
- Expected dense rows: `102400`
- Expected launch-gate rows per active primitive: `5120`
- R5 transition-aware decision: `R5_TRANSITION_AWARE_DENSE_PASSED_FOR_REVIEW`
- Regime labels: `launch_entry_evidence_for_8_families`, `inflight_entry_evidence_for_8_families`, `boundary_or_recovery_entry_evidence_for_8_families`.
- Launch, in-flight, boundary, and recovery evidence are separate transition entries for the same eight active primitive families.

Launch-entry transition summary:

- `glide: rows=2385, accepted=1169, weak=1216, continuation_valid=2385, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `recovery: rows=2360, accepted=1163, weak=1197, continuation_valid=2360, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_entry: rows=2369, accepted=1161, weak=1208, continuation_valid=2369, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_dwell_arc: rows=2371, accepted=1172, weak=1199, continuation_valid=2371, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_left: rows=2377, accepted=1189, weak=1188, continuation_valid=2377, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_right: rows=2407, accepted=1200, weak=1207, continuation_valid=2407, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `energy_retaining_bank: rows=2358, accepted=1178, weak=1180, continuation_valid=2358, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `safe_exit_or_recovery_handoff: rows=2399, accepted=1193, weak=1206, continuation_valid=2399, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
