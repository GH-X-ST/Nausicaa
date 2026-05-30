# R10 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `40`
- Expected history launches: `430`
- Gate profile: `targeted_memory_mechanism_diagnostic_not_full_r10`
- Safety thresholds: hard failure <= `0.2`, no-viable <= `0.3`, safe success >= `0.2`, full safe success >= `None`, terminal/lift >= `0.0`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, front-wall progress, front-wall terminal proxy, progress-gated terminal total specific-energy proxy, updraft gain, lift dwell, and residual memory.
- Memory policy: `outer_loop_spatial_flow_belief_aggressive_safe_objective_memory_v2_3` maintains a case-local 0.1 m 3D updraft-utility belief map; each flown primitive writes dense executed-segment residual samples at 0.1 m spacing with launch-index recency decay, and each candidate path queries the accumulated map through a 0.2 m neighbourhood over seven probes plus a bounded 0.8 m / 35 deg azimuth / 20 deg elevation sparse 3D reachable-flow attraction cone capped at 0.25 m. Residual path utility and reachable-flow attraction are confidence-gated, capped primary memory-objective terms among already-viable front-progress-compatible candidates, then accepted only through the baseline shield after viability filtering.
- Governor learning strategy: `case_local_online_memory_plus_r10_global_deterministic_calibration_v1` keeps online memory `case_local_reset_per_final_schedule_row`; R10 calibration scope is `aggregate_all_r10_final_heldout_rows_and_selector_opportunity_diagnostics` and R11 uses `single_frozen_r10_governor_config_used_for_r11_validation`.
- Calibration search policy: `deterministic_bounded_rule_update_no_profile_ladder_no_black_box_search`.
- Memory opportunity audit: `memory_opportunity_summary.csv` and `memory_opportunity_decision_log.csv` report baseline-vs-memory candidate gaps, correction deltas, shield status, and accepted/rejected switch reasons.
- The adaptive selector uses one baseline shield at every launch; there is no branch that treats a held-out final launch as a known final mission.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_front_wall_mission_updraft_terminal_energy_score_v2`; rewards front-wall terminal mission completion plus capped updraft/lift evidence and terminal total specific energy reserve. Airborne time and generic net/gross energy drift remain audit-only.
- Dry-air or scheduled-zero-fan low-launch-speed floor stops keep the raw primitive `floor_violation` audit label, but are interpreted as expected energy depletion rather than governor or memory failure.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `40` required `40`
- `history_launch_count`: `True` observed `430` required `430`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `4` required `4`
- `pairing_audit`: `True` observed `2` required `2`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `4.0` required `>=0.1`
- `no_glider_latency_variation_audit`: `True` observed `2` required `2`
- `reduced_diagnostic_not_target_R10`: `False` observed `reduced_diagnostic_not_target_R10` required `full_R10_validation_required`
- `launch_inflight_recovery_sequence_policy`: `True` observed `470` required `470`
- `claim_bearing_final_launch_count_positive`: `True` observed `40` required `>0_after_expected_low_energy_dry_air_sink_exclusion`
- `expected_low_energy_dry_air_sink_labeled`: `True` observed `0` required `diagnostic_not_claim_bearing_governor_failure`
- `hard_failure_rate_within_stage_profile`: `True` observed `0.0` required `0.2`
- `floor_or_ceiling_violation_rate_within_stage_profile`: `True` observed `0.0` required `0.2`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.0` required `0.3`
- `safe_success_rate_within_stage_profile`: `True` observed `1.0` required `0.2`
- `terminal_or_lift_capture_within_stage_profile`: `True` observed `1.0` required `0.0`
- `front_wall_mission_success_rate_diagnostic`: `True` observed `1.0` required `diagnostic_only_score_target_not_current_pass_gate`
- `wrong_wall_exit_rate_diagnostic`: `True` observed `0.0` required `diagnostic_only_penalised_by_launch_score`
- `selected_primitive_family_count_diagnostic`: `True` observed `8` required `diagnostic_only_not_a_governor_pass_gate`
- `selected_variant_count_diagnostic`: `True` observed `29` required `diagnostic_only_not_a_governor_pass_gate`
- `lift_dwell_arc_selected_diagnostic`: `True` observed `True` required `diagnostic_only_expected_when_viable_lift_dwell_evidence_wins`
- `r10_learning_stage_uses_final_reject_rate_not_candidate_reject_rate`: `True` observed `0.0` required `bounded_final_no_viable_rate_plus_tuning_handoff_diagnostics`
- `r10_memory_improvement_is_tuning_signal_not_final_claim_gate`: `True` observed `governor_config_selection.csv_and_memory_opportunity_summary.csv` required `R11_or_reality_required_for_memory_improvement_claim`
