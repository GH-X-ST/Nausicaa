# R5 Launch-Entry Transition Diagnosis

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Status: `complete`
- Rows written: `76800`
- Expected dense rows: `76800`
- Expected launch-gate rows per active primitive: `3840`
- R5 transition-aware decision: `R5_TRANSITION_AWARE_DENSE_PASSED_FOR_REVIEW`
- Regime labels: `launch_entry_evidence_for_8_families`, `inflight_entry_evidence_for_8_families`, `boundary_or_recovery_entry_evidence_for_8_families`.
- Launch, in-flight, boundary, and recovery evidence are separate transition entries for the same eight active primitive families.

Launch-entry transition summary:

- `glide: rows=1774, accepted=680, weak=1094, continuation_valid=1774, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `recovery: rows=1775, accepted=682, weak=1093, continuation_valid=1775, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_entry: rows=1775, accepted=681, weak=1094, continuation_valid=1775, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `lift_dwell_arc: rows=1774, accepted=680, weak=1094, continuation_valid=1774, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_left: rows=1772, accepted=680, weak=1092, continuation_valid=1772, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `mild_turn_right: rows=1773, accepted=681, weak=1092, continuation_valid=1773, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `energy_retaining_bank: rows=1774, accepted=682, weak=1092, continuation_valid=1774, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `safe_exit_or_recovery_handoff: rows=1775, accepted=681, weak=1094, continuation_valid=1775, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
