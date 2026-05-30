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

- `glide: rows=2207, accepted=1118, weak=1089, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `recovery: rows=2207, accepted=1116, weak=1091, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_entry: rows=2205, accepted=1117, weak=1088, continuation_valid=2205, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_dwell_arc: rows=2207, accepted=1120, weak=1087, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_left: rows=2207, accepted=1114, weak=1093, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_right: rows=2207, accepted=1119, weak=1088, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `energy_retaining_bank: rows=2207, accepted=1116, weak=1091, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `safe_exit_or_recovery_handoff: rows=2207, accepted=1118, weak=1089, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
