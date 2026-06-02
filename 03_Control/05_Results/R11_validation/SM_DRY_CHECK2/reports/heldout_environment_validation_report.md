# R11 Repeated-Launch Validation

- Status: `dry_run_schedule`
- Pass gate: `False`
- Expected final held-out launches: `320`
- Expected history launches: `3440`
- Gate profile: `strict_final_heldout_validation`
- Safety thresholds: hard failure <= `0.01`, no-viable <= `0.02`, safe success >= `0.99`, full safe success >= `0.99`, terminal/lift >= `0.9`.
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

- `final_heldout_launch_count`: `True` observed `320` required `320`
- `history_launch_count`: `True` observed `3440` required `3440`
- `library_size_case_count`: `True` observed `5` required `5`
- `policy_history_condition_count`: `True` observed `4` required `4`
- `pairing_audit`: `True` observed `16` required `16`
- `primitive_count_cap_disabled_for_full_validation`: `True` observed `0` required `0_or_negative_disabled`
- `max_episode_time_budget_positive`: `True` observed `20.0` required `>=0.1`
- `no_glider_latency_variation_audit`: `True` observed `16` required `16`
- `final_rollout_rows_present`: `False` observed `0` required `320`
