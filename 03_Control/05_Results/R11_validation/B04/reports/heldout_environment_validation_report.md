# R11 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `1600`
- Expected history launches: `17200`
- Gate profile: `strict_final_heldout_validation`
- Safety thresholds: hard failure <= `0.01`, no-viable <= `0.02`, safe success >= `0.99`, full safe success >= `0.99`, terminal/lift >= `0.9`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, front-wall progress, front-wall terminal proxy, progress-gated terminal total specific-energy proxy, updraft gain, lift dwell, and residual memory.
- Residual memory policy: `outer_loop_baseline_shielded_recency_safe_exploration_memory_v1_5` applies capped, recency-weighted, specific-energy-dominant candidate-path residual memory over seven path probes plus shielded uncertainty exploration after viability filtering.
- Governor learning strategy: `case_local_online_memory_plus_r10_global_deterministic_calibration_v1` keeps online memory `case_local_reset_per_final_schedule_row`; R10 calibration scope is `aggregate_all_r10_final_heldout_rows_and_selector_opportunity_diagnostics` and R11 uses `single_frozen_r10_governor_config_used_for_r11_validation`.
- Calibration search policy: `deterministic_bounded_rule_update_no_profile_ladder_no_black_box_search`.
- Memory opportunity audit: `memory_opportunity_summary.csv` and `memory_opportunity_decision_log.csv` report baseline-vs-memory candidate gaps, correction deltas, shield status, and accepted/rejected switch reasons.
- The adaptive selector uses one baseline shield at every launch; there is no branch that treats a held-out final launch as a known final mission.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_front_wall_mission_updraft_terminal_energy_score_v2`; rewards front-wall terminal mission completion plus capped updraft/lift evidence and terminal total specific energy reserve. Airborne time and generic net/gross energy drift remain audit-only.
- Dry-air or scheduled-zero-fan low-launch-speed floor stops keep the raw primitive `floor_violation` audit label, but are interpreted as expected energy depletion rather than governor or memory failure.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `1600` required `1600`
- `history_launch_count`: `True` observed `17200` required `17200`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `4` required `4`
- `pairing_audit`: `True` observed `80` required `80`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `20.0` required `>=0.1`
- `no_glider_latency_variation_audit`: `True` observed `80` required `80`
- `launch_inflight_recovery_sequence_policy`: `True` observed `18800` required `18800`
- `claim_bearing_final_launch_count_positive`: `True` observed `1180` required `>0_after_expected_low_energy_dry_air_sink_exclusion`
- `expected_low_energy_dry_air_sink_labeled`: `True` observed `420` required `diagnostic_not_claim_bearing_governor_failure`
- `hard_failure_rate_within_stage_profile`: `False` observed `0.03474576271186441` required `0.01`
- `floor_or_ceiling_violation_rate_within_stage_profile`: `False` observed `0.010169491525423728` required `0.0`
- `no_viable_primitive_rate_within_stage_profile`: `False` observed `0.08135593220338982` required `0.02`
- `safe_success_rate_within_stage_profile`: `False` observed `0.8838983050847458` required `0.99`
- `terminal_or_lift_capture_within_stage_profile`: `False` observed `0.8838983050847458` required `0.9`
- `front_wall_mission_success_rate_diagnostic`: `True` observed `0.8194915254237288` required `diagnostic_only_score_target_not_current_pass_gate`
- `wrong_wall_exit_rate_diagnostic`: `True` observed `0.08898305084745763` required `diagnostic_only_penalised_by_launch_score`
- `selected_primitive_family_count_diagnostic`: `True` observed `8` required `diagnostic_only_not_a_governor_pass_gate`
- `selected_variant_count_diagnostic`: `True` observed `253` required `diagnostic_only_not_a_governor_pass_gate`
- `lift_dwell_arc_selected_diagnostic`: `True` observed `True` required `diagnostic_only_expected_when_viable_lift_dwell_evidence_wins`
- `full_safe_success_rate_within_stage_profile`: `False` observed `0.0` required `0.99`
