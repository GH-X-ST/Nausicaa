# L6 Move-On Check

- Status: `complete`
- Run class: `rich_side_l6_candidate`
- Rows written: `102400`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Cross-layer smoke status: `cross_layer_smoke_start_family_complete`
- Start-family counts: `{"inflight_boundary_near": 10240, "inflight_lift_region": 15360, "inflight_nominal": 25600, "inflight_recovery_edge": 10240, "launch_gate": 40960}`
- Start-family mix exact or blocked: `True`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 12800, "incompatible": 0}, "glide": {"compatible": 12800, "incompatible": 0}, "lift_dwell_arc": {"compatible": 12800, "incompatible": 0}, "lift_entry": {"compatible": 12800, "incompatible": 0}, "mild_turn_left": {"compatible": 12800, "incompatible": 0}, "mild_turn_right": {"compatible": 12800, "incompatible": 0}, "recovery": {"compatible": 12800, "incompatible": 0}, "safe_exit_or_recovery_handoff": {"compatible": 12800, "incompatible": 0}}`
- History-backed FIFO count: `102400`
- Ready frozen controller count: `0`
- R5 transition training status: `blocked`
- Selected transition-object bundle cleared for R7 planning: `False`

Blockers before heavy W01:

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

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
