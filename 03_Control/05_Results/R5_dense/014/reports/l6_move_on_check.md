# L6 Move-On Check

- Status: `complete`
- Run class: `preflight`
- Rows written: `960`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Cross-layer smoke status: `cross_layer_smoke_start_family_complete`
- Start-family counts: `{"inflight_boundary_near": 96, "inflight_lift_region": 144, "inflight_nominal": 240, "inflight_recovery_edge": 96, "launch_gate": 384}`
- Start-family mix exact or blocked: `True`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 72, "incompatible": 48}, "glide": {"compatible": 120, "incompatible": 0}, "lift_dwell_arc": {"compatible": 72, "incompatible": 48}, "lift_entry": {"compatible": 72, "incompatible": 48}, "mild_turn_left": {"compatible": 72, "incompatible": 48}, "mild_turn_right": {"compatible": 72, "incompatible": 48}, "recovery": {"compatible": 24, "incompatible": 96}, "safe_exit_or_recovery_handoff": {"compatible": 24, "incompatible": 96}}`
- History-backed FIFO count: `528`
- Ready frozen controller count: `16`
- Rich-side W01 fixed-library cleared for W2 planning: `False`

Blockers before heavy W01:

- `below_19200_fallback_scale_threshold`

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
