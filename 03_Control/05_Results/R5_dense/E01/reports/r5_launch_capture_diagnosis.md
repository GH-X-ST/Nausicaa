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

- `glide: rows=5041, accepted=2372, weak=2669, continuation_valid=5041, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `recovery: rows=5031, accepted=2365, weak=2666, continuation_valid=5031, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_entry: rows=5044, accepted=2375, weak=2669, continuation_valid=5044, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_dwell_arc: rows=5028, accepted=2371, weak=2657, continuation_valid=5028, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_left: rows=5032, accepted=2372, weak=2660, continuation_valid=5032, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_right: rows=5024, accepted=2355, weak=2669, continuation_valid=5024, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `energy_retaining_bank: rows=5017, accepted=2352, weak=2665, continuation_valid=5017, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `safe_exit_or_recovery_handoff: rows=5025, accepted=2367, weak=2658, continuation_valid=5025, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
