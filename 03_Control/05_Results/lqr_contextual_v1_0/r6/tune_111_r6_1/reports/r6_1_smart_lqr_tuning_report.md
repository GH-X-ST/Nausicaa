# R6.1 Smart LQR Tuning Report

Interpreter contract: .venv/Scripts/python.exe
Run ID: 111
Run root: 03_Control/05_Results/lqr_contextual_v1_0/r6/tune_111_r6_1
Strategy: r6_1_staged
Diagnostic source: available
Candidates generated: 256
Selected or fallback controller records: 0
Rejected controller records: 256
Blocked controller records: 0
Stage E required primitives: 0
Stage E no-help closure records: 8
Registry status: blocked
Registry claim status: simulation_only_blocked

Method boundary: time-invariant LQR Q/R tuning only for W0/W1 selected-controller evidence.
W2/W3 remain replay-only. No forbidden non-LQR controller, planner-chain, online layout branch, hardware-readiness, transfer, robustness, or mission-success claim is made.


Stage E closure reasons:
- energy_retaining_bank: stage_e_not_applicable_no_candidate_above_minimum_gate
- glide: stage_e_not_applicable_no_candidate_above_minimum_gate
- lift_dwell_arc: stage_e_not_applicable_no_candidate_above_minimum_gate
- lift_entry: stage_e_not_applicable_no_candidate_above_minimum_gate
- mild_turn_left: stage_e_not_applicable_no_candidate_above_minimum_gate
- mild_turn_right: stage_e_not_applicable_no_candidate_above_minimum_gate
- recovery: stage_e_not_applicable_no_candidate_above_minimum_gate
- safe_exit_or_recovery_handoff: stage_e_not_applicable_no_candidate_above_minimum_gate

R7 blocked: missing_selected_or_accepted_fallback_controller:energy_retaining_bank,glide,lift_dwell_arc,lift_entry,mild_turn_left,mild_turn_right,recovery,safe_exit_or_recovery_handoff
