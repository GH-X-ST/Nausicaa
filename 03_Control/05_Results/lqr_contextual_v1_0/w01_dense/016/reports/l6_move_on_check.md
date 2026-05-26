# L6 Move-On Check

- Status: `complete`
- Run class: `preflight`
- Rows written: `80`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Cross-layer smoke status: `cross_layer_smoke_start_family_complete`
- Start-family counts: `{"inflight_boundary_near": 8, "inflight_lift_region": 12, "inflight_nominal": 20, "inflight_recovery_edge": 8, "launch_gate": 32}`
- Start-family mix exact or blocked: `True`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 6, "incompatible": 4}, "glide": {"compatible": 10, "incompatible": 0}, "lift_dwell_arc": {"compatible": 6, "incompatible": 4}, "lift_entry": {"compatible": 6, "incompatible": 4}, "mild_turn_left": {"compatible": 6, "incompatible": 4}, "mild_turn_right": {"compatible": 6, "incompatible": 4}, "recovery": {"compatible": 2, "incompatible": 8}, "safe_exit_or_recovery_handoff": {"compatible": 2, "incompatible": 8}}`
- History-backed FIFO count: `44`
- Ready frozen controller count: `16`
- Rich-side W01 fixed-library cleared for W2 planning: `False`

Blockers before heavy W01:

- `below_19200_fallback_scale_threshold`

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
