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
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 5760, "incompatible": 3840}, "glide": {"compatible": 9600, "incompatible": 0}, "lift_dwell_arc": {"compatible": 5760, "incompatible": 3840}, "lift_entry": {"compatible": 5760, "incompatible": 3840}, "mild_turn_left": {"compatible": 5760, "incompatible": 3840}, "mild_turn_right": {"compatible": 5760, "incompatible": 3840}, "recovery": {"compatible": 1920, "incompatible": 7680}, "safe_exit_or_recovery_handoff": {"compatible": 1920, "incompatible": 7680}}`
- History-backed FIFO count: `42240`
- Ready frozen controller count: `256`
- Rich-side W01 fixed-library cleared for W2 planning: `True`

Blockers before heavy W01:

- `none`

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
