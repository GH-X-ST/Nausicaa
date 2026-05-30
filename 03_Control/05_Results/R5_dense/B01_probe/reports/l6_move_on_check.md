# L6 Move-On Check

- Status: `dry_run_schedule`
- Run class: `dry_run_schedule`
- Rows written: `0`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Cross-layer smoke status: `cross_layer_smoke_start_family_complete`
- Start-family counts: `{"inflight_boundary_near": 10240, "inflight_lift_region": 15360, "inflight_nominal": 25600, "inflight_recovery_edge": 10240, "launch_gate": 40960}`
- Start-family mix exact or blocked: `True`
- Entry-role compatibility by primitive: `{}`
- History-backed FIFO count: `0`
- Ready frozen controller count: `3840`
- R5 transition training status: `incomplete`
- Selected transition-object bundle cleared for R7 planning: `False`

Blockers before heavy W01:

- `no_rollout_evidence_written`

Blocked claims remain final W0/W1 dense completion, W2 survival execution, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
