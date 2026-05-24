# L6 Move-On Check

- Status: `complete`
- Run class: `preflight`
- Rows written: `2000`
- Controller synthesis boundary: `predictor_compensated_augmented_discrete_lqr_v1`
- Timing design: `actuator_surface_state_command_fifo_predictor_compensated`
- Rollout boundary: `panel_wind_feedback_delay_command_timing_actuator_lag_with_plant_and_implementation_instances`
- Deferred validation: `no_true_full_delayed_state_feedback_validation_claim`
- Heavy W01 move-on allowed: `True`

Blockers before heavy W01:

- `none`

Blocked claims remain W0/W1 dense completion, W2 survival, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, and mission success.
