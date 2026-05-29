# L6 Move-On Check

- Status: `dry_run_schedule`
- Run class: `dry_run_schedule`
- Rows written: `0`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Heavy W01 move-on allowed: `False`

Blockers before heavy W01:

- `no_rollout_evidence_written`

Blocked claims remain W0/W1 dense completion, W2 survival, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
