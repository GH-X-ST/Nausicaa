# L6 Move-On Check

- Status: `complete`
- Run class: `rich_side_l6_candidate`
- Rows written: `134400`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Cross-layer smoke status: `cross_layer_smoke_start_family_complete`
- Start-family counts: `{"inflight_boundary_near": 9600, "inflight_lift_region": 21600, "inflight_nominal": 36000, "inflight_recovery_edge": 9600, "launch_gate": 57600}`
- Start-family mix exact or blocked: `True`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 9600, "incompatible": 0}, "glide": {"compatible": 9600, "incompatible": 0}, "launch_capture_energy_build": {"compatible": 9600, "incompatible": 0}, "launch_capture_glide_stabilise": {"compatible": 9600, "incompatible": 0}, "launch_capture_lift_seek": {"compatible": 9600, "incompatible": 0}, "launch_capture_safe_handoff": {"compatible": 9600, "incompatible": 0}, "launch_capture_shallow_left": {"compatible": 9600, "incompatible": 0}, "launch_capture_shallow_right": {"compatible": 9600, "incompatible": 0}, "lift_dwell_arc": {"compatible": 9600, "incompatible": 0}, "lift_entry": {"compatible": 9600, "incompatible": 0}, "mild_turn_left": {"compatible": 9600, "incompatible": 0}, "mild_turn_right": {"compatible": 9600, "incompatible": 0}, "recovery": {"compatible": 9600, "incompatible": 0}, "safe_exit_or_recovery_handoff": {"compatible": 9600, "incompatible": 0}}`
- History-backed FIFO count: `134400`
- Ready frozen controller count: `448`
- Rich-side W01 fixed-library cleared for W2 planning: `True`

Blockers before heavy W01:

- `none`

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
