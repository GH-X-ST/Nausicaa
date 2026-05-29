# R10 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `1000`
- Expected history launches: `10750`
- Gate profile: `relaxed_changed_case_viability_governor_learning_not_final_validation`
- Safety thresholds: hard failure <= `0.2`, no-viable <= `0.3`, safe success >= `0.2`, full safe success >= `None`, terminal/lift >= `0.3`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, front-wall progress, front-wall terminal proxy, progress-gated terminal total specific-energy proxy, updraft gain, lift dwell, and residual memory.
- Residual memory policy: `outer_loop_baseline_shielded_recency_safe_exploration_memory_v1_5` applies capped, recency-weighted, specific-energy-dominant candidate-path residual memory over seven path probes plus shielded uncertainty exploration after viability filtering.
- Memory opportunity audit: `memory_opportunity_summary.csv` and `memory_opportunity_decision_log.csv` report baseline-vs-memory candidate gaps, correction deltas, shield status, and accepted/rejected switch reasons.
- The adaptive selector uses one baseline shield at every launch; there is no branch that treats a held-out final launch as a known final mission.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_front_wall_mission_updraft_terminal_energy_score_v2`; rewards front-wall terminal mission completion plus capped updraft/lift evidence and terminal total specific energy reserve. Airborne time and generic net/gross energy drift remain audit-only.
- Dry-air or scheduled-zero-fan low-launch-speed floor stops keep the raw primitive `floor_violation` audit label, but are interpreted as expected energy depletion rather than governor or memory failure.
- Claim boundary: simulation-only; no hardware, real-flight transfer, mission, autonomy, or memory-improvement claim.

Gate summary:

- `final_heldout_launch_count`: `True` observed `1000` required `1000`
- `history_launch_count`: `True` observed `10750` required `10750`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `4` required `4`
- `pairing_audit`: `True` observed `50` required `50`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `20.0` required `>=0.1`
- `no_glider_latency_variation_audit`: `True` observed `50` required `50`
- `launch_inflight_recovery_sequence_policy`: `True` observed `11750` required `11750`
- `claim_bearing_final_launch_count_positive`: `True` observed `1000` required `>0_after_expected_low_energy_dry_air_sink_exclusion`
- `expected_low_energy_dry_air_sink_labeled`: `True` observed `0` required `diagnostic_not_claim_bearing_governor_failure`
- `hard_failure_rate_within_stage_profile`: `False` observed `0.203` required `0.2`
- `floor_or_ceiling_violation_rate_within_stage_profile`: `True` observed `0.16` required `0.2`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.056` required `0.3`
- `safe_success_rate_within_stage_profile`: `True` observed `0.741` required `0.2`
- `terminal_or_lift_capture_within_stage_profile`: `True` observed `0.741` required `0.3`
- `front_wall_mission_success_rate_diagnostic`: `True` observed `0.713` required `diagnostic_only_score_target_not_current_pass_gate`
- `wrong_wall_exit_rate_diagnostic`: `True` observed `0.071` required `diagnostic_only_penalised_by_launch_score`
- `selected_primitive_family_count_diagnostic`: `True` observed `8` required `diagnostic_only_not_a_governor_pass_gate`
- `selected_variant_count_diagnostic`: `True` observed `213` required `diagnostic_only_not_a_governor_pass_gate`
- `lift_dwell_arc_selected_diagnostic`: `True` observed `True` required `diagnostic_only_expected_when_viable_lift_dwell_evidence_wins`
