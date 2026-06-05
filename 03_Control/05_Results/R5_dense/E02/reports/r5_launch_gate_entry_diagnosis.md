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

- `glide: rows=2330, accepted=1175, weak=1155, continuation_valid=2330, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `recovery: rows=2359, accepted=1185, weak=1174, continuation_valid=2359, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_entry: rows=2356, accepted=1182, weak=1174, continuation_valid=2356, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_dwell_arc: rows=2375, accepted=1196, weak=1179, continuation_valid=2375, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_left: rows=2349, accepted=1196, weak=1153, continuation_valid=2349, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_right: rows=2339, accepted=1182, weak=1157, continuation_valid=2339, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `energy_retaining_bank: rows=2347, accepted=1193, weak=1154, continuation_valid=2347, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `safe_exit_or_recovery_handoff: rows=2353, accepted=1183, weak=1170, continuation_valid=2353, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
