# R10 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `False`
- Expected final held-out launches: `200`
- Expected history launches: `2150`
- Gate profile: `strict_final_validation`
- Safety thresholds: hard failure <= `0.01`, no-viable <= `0.02`, safe success >= `0.99`, full safe success >= `None`, terminal/lift >= `0.9`.
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

- `final_heldout_launch_count`: `True` observed `200` required `200`
- `history_launch_count`: `True` observed `2150` required `2150`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `4` required `4`
- `pairing_audit`: `True` observed `10` required `10`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `20.0` required `>=0.1`
- `no_glider_latency_variation_audit`: `True` observed `10` required `10`
- `reduced_diagnostic_not_target_R10`: `False` observed `reduced_diagnostic_not_target_R10` required `full_R10_validation_required`
- `launch_inflight_recovery_sequence_policy`: `True` observed `2350` required `2350`
- `claim_bearing_final_launch_count_positive`: `True` observed `192` required `>0_after_expected_low_energy_dry_air_sink_exclusion`
- `expected_low_energy_dry_air_sink_labeled`: `True` observed `8` required `diagnostic_not_claim_bearing_governor_failure`
- `hard_failure_rate_within_stage_profile`: `False` observed `0.020833333333333332` required `0.01`
- `floor_or_ceiling_violation_rate_within_stage_profile`: `False` observed `0.020833333333333332` required `0.0`
- `no_viable_primitive_rate_within_stage_profile`: `False` observed `0.14583333333333334` required `0.02`
- `safe_success_rate_within_stage_profile`: `False` observed `0.8333333333333334` required `0.99`
- `terminal_or_lift_capture_within_stage_profile`: `False` observed `0.8333333333333334` required `0.9`
- `front_wall_mission_success_rate_diagnostic`: `True` observed `0.8333333333333334` required `diagnostic_only_score_target_not_current_pass_gate`
- `wrong_wall_exit_rate_diagnostic`: `True` observed `0.0` required `diagnostic_only_penalised_by_launch_score`
- `selected_primitive_family_count_diagnostic`: `True` observed `8` required `diagnostic_only_not_a_governor_pass_gate`
- `selected_variant_count_diagnostic`: `True` observed `85` required `diagnostic_only_not_a_governor_pass_gate`
- `lift_dwell_arc_selected_diagnostic`: `True` observed `True` required `diagnostic_only_expected_when_viable_lift_dwell_evidence_wins`
