# L6 Move-On Check

- Status: `complete`
- Run class: `rich_side_l6_candidate`
- Rows written: `76800`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Cross-layer smoke status: `cross_layer_smoke_start_family_complete`
- Start-family counts: `{"inflight_boundary_near": 7680, "inflight_lift_region": 11520, "inflight_nominal": 19200, "inflight_recovery_edge": 7680, "launch_gate": 30720}`
- Start-family mix exact or blocked: `True`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 9600, "incompatible": 0}, "glide": {"compatible": 9600, "incompatible": 0}, "lift_dwell_arc": {"compatible": 9600, "incompatible": 0}, "lift_entry": {"compatible": 9600, "incompatible": 0}, "mild_turn_left": {"compatible": 9600, "incompatible": 0}, "mild_turn_right": {"compatible": 9600, "incompatible": 0}, "recovery": {"compatible": 9600, "incompatible": 0}, "safe_exit_or_recovery_handoff": {"compatible": 9600, "incompatible": 0}}`
- History-backed FIFO count: `76800`
- Ready frozen controller count: `1106`
- R5 transition training status: `passed`
- Selected transition-object bundle cleared for R7 planning: `True`

Blockers before heavy W01:

- `none`

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
