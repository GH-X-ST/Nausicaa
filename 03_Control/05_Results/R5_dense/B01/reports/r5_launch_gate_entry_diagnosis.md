# R5 Launch-Entry Transition Diagnosis

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Status: `complete`
- Rows written: `102400`
- Expected dense rows: `102400`
- Expected launch-gate rows per active primitive: `10240`
- R5 transition-aware decision: `R5_TRANSITION_AWARE_DENSE_BLOCKED_FIX_REQUIRED`
- Regime labels: `launch_entry_evidence_for_8_families`, `inflight_entry_evidence_for_8_families`, `boundary_or_recovery_entry_evidence_for_8_families`.
- Launch, in-flight, boundary, and recovery evidence are separate transition entries for the same eight active primitive families.

Launch-entry transition summary:

- `glide: rows=2207, accepted=1118, weak=1089, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `recovery: rows=2207, accepted=1116, weak=1091, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `lift_entry: rows=2205, accepted=1117, weak=1088, continuation_valid=2205, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `lift_dwell_arc: rows=2207, accepted=1120, weak=1087, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `mild_turn_left: rows=2207, accepted=1114, weak=1093, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `mild_turn_right: rows=2207, accepted=1119, weak=1088, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `energy_retaining_bank: rows=2207, accepted=1116, weak=1091, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`
- `safe_exit_or_recovery_handoff: rows=2207, accepted=1118, weak=1089, continuation_valid=2207, terminal_useful=0, hard_failure=0, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=False`

Blockers:

- `glide:launch_gate_row_count_5120_expected_10240`
- `recovery:launch_gate_row_count_5120_expected_10240`
- `lift_entry:launch_gate_row_count_5120_expected_10240`
- `lift_dwell_arc:launch_gate_row_count_5120_expected_10240`
- `mild_turn_left:launch_gate_row_count_5120_expected_10240`
- `mild_turn_right:launch_gate_row_count_5120_expected_10240`
- `energy_retaining_bank:launch_gate_row_count_5120_expected_10240`
- `safe_exit_or_recovery_handoff:launch_gate_row_count_5120_expected_10240`
- `missing_selected_inflight_stable_transition_object_for_energy_retaining_bank`
- `missing_selected_inflight_stable_transition_object_for_glide`
- `missing_selected_inflight_stable_transition_object_for_lift_dwell_arc`
- `missing_selected_inflight_stable_transition_object_for_lift_entry`
- `missing_selected_inflight_stable_transition_object_for_mild_turn_left`
- `missing_selected_inflight_stable_transition_object_for_mild_turn_right`
- `missing_selected_inflight_stable_transition_object_for_recovery`
- `missing_selected_inflight_stable_transition_object_for_safe_exit_or_recovery_handoff`
- `missing_selected_launch_gate_transition_object_for_energy_retaining_bank`
- `missing_selected_launch_gate_transition_object_for_glide`
- `missing_selected_launch_gate_transition_object_for_lift_dwell_arc`
- `missing_selected_launch_gate_transition_object_for_lift_entry`
- `missing_selected_launch_gate_transition_object_for_mild_turn_left`
- `missing_selected_launch_gate_transition_object_for_mild_turn_right`
- `missing_selected_launch_gate_transition_object_for_recovery`
- `missing_selected_launch_gate_transition_object_for_safe_exit_or_recovery_handoff`
- `r5_transition_selected_for_r7_empty`
- `r5_transition_training_not_passed`
- `frozen_w01_controller_bundle_empty`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
