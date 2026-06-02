# R10 Repeated-Launch Validation

- Status: `complete`
- Pass gate: `True`
- Expected final held-out launches: `1000`
- Expected history launches: `10750`
- Gate profile: `relaxed_changed_case_viability_governor_learning_not_final_validation`
- Safety thresholds: hard failure <= `0.2`, no-viable <= `0.3`, safe success >= `0.2`, full safe success >= `None`, terminal/lift >= `0.3`.
- Launch sequence policy: `state_class_transition_entry_governor_no_launch_specific_family`
- Governor route: classify current transition state, filter matching primitive entry class, then score transition viability, front-wall progress, front-wall terminal proxy, progress-gated terminal total specific-energy proxy, updraft gain, lift dwell, and residual memory.
- Memory policy: `outer_loop_cost_benefit_spatial_flow_memory_v4_1` maintains a case-local 0.1 m 3D updraft-utility belief map; each flown primitive writes dense executed-segment residual samples at 0.1 m spacing with launch-index recency decay. The in-flight controller and full diagnostics query the accumulated map through the same 0.2 m neighbourhood over seven probes. The timed in-flight boundary uses a compact controller-row selector fast path before the 0.100 s boundary, while table flushing, full candidate-row expansion, and post-hoc candidate/memory diagnostics stay outside that boundary. Both use bounded current-to-exit, reachable-cone, and short-horizon route-flow probes from the candidate exit. The selector collapses those map queries into one cost-benefit memory value: remembered flow benefit plus small information value minus frozen mission-score, front-progress, risk, and path-margin costs. The value acts only among already-viable candidates and is accepted only through the baseline shield after viability filtering.
- Governor learning strategy: `case_local_online_memory_plus_r10_global_deterministic_calibration_v1` keeps online memory `case_local_reset_per_final_schedule_row`; R10 calibration scope is `aggregate_all_r10_final_heldout_rows_and_selector_opportunity_diagnostics` and R11 uses `single_frozen_r10_governor_config_used_for_r11_validation`.
- Calibration search policy: `deterministic_bounded_rule_update_no_profile_ladder_no_black_box_search`.
- Memory opportunity audit: `memory_opportunity_summary.csv` and `memory_opportunity_decision_log.csv` report baseline-vs-memory candidate gaps, correction deltas, shield status, and accepted/rejected switch reasons; large decision logs are partitioned under `tables/memory_opportunity_decision_log/` with the metrics CSV kept as a small index.
- The adaptive selector uses one baseline shield at every launch; there is no branch that treats a held-out final launch as a known final mission.
- Boundary-near is a route state, not automatic failure; hard_failure is the failure class.
- Launch score: `r10_r11_front_wall_mission_updraft_terminal_energy_score_v2`; rewards front-wall terminal mission completion plus capped updraft/lift evidence and terminal total specific energy reserve. Airborne time and generic net/gross energy drift remain audit-only.
- Dry-air or scheduled-zero-fan low-launch-speed floor stops keep the raw primitive `floor_violation` audit label, but are interpreted as expected energy depletion rather than governor or memory failure.
- Start-energy audit: `speed_bin_policy_ladder_summary.csv` reports mission/safety rates by environment ladder, library tier, policy, repeated-launch history length, and initial-speed bin. `start_energy_group_policy_ladder_summary.csv` separates low-start-energy launches from high-start-energy launches using `5.0 m/s` as a fixed post-hoc reporting threshold; paired score-delta summaries are split the same way.
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
- `claim_bearing_final_launch_count_positive`: `True` observed `787` required `>0_after_expected_low_energy_dry_air_sink_exclusion`
- `expected_low_energy_dry_air_sink_labeled`: `True` observed `213` required `diagnostic_not_claim_bearing_governor_failure`
- `hard_failure_rate_within_stage_profile`: `True` observed `0.060991105463786534` required `0.2`
- `floor_or_ceiling_violation_rate_within_stage_profile`: `True` observed `0.020330368487928845` required `0.2`
- `no_viable_primitive_rate_within_stage_profile`: `True` observed `0.030495552731893267` required `0.3`
- `safe_success_rate_within_stage_profile`: `True` observed `0.9085133418043202` required `0.2`
- `terminal_or_lift_capture_within_stage_profile`: `True` observed `0.9085133418043202` required `0.3`
- `front_wall_mission_success_rate_diagnostic`: `True` observed `0.8945362134688691` required `diagnostic_only_score_target_not_current_pass_gate`
- `wrong_wall_exit_rate_diagnostic`: `True` observed `0.029224904701397714` required `diagnostic_only_penalised_by_launch_score`
- `selected_primitive_family_count_diagnostic`: `True` observed `8` required `diagnostic_only_not_a_governor_pass_gate`
- `selected_variant_count_diagnostic`: `True` observed `184` required `diagnostic_only_not_a_governor_pass_gate`
- `lift_dwell_arc_selected_diagnostic`: `True` observed `True` required `diagnostic_only_expected_when_viable_lift_dwell_evidence_wins`
- `r10_learning_stage_uses_final_reject_rate_not_candidate_reject_rate`: `True` observed `0.030495552731893267` required `bounded_final_no_viable_rate_plus_tuning_handoff_diagnostics`
- `r10_memory_improvement_is_tuning_signal_not_final_claim_gate`: `True` observed `governor_config_selection.csv_and_memory_opportunity_summary.csv` required `R11_or_reality_required_for_memory_improvement_claim`
