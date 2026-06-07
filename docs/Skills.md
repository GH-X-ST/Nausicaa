# Nausicaa Project Skills

<!-- R9_LAUNCH_GATE_ALIGNMENT_START -->

## Active Transition-Aware Thesis Workflow

The active thesis workflow is `R5 -> R7 -> R8 -> R10 -> R11 -> Reality`. R9 remains internal preflight only and is not thesis-facing evidence. R10 tunes the viability governor with residual updraft adaptation, and R11 is the held-out validation gate.

Launch is an entry regime, not a separate controller family. The active primitive catalogue has exactly eight manoeuvre families: `glide`, `recovery`, `lift_entry`, `lift_dwell_arc`, `mild_turn_left`, `mild_turn_right`, `energy_retaining_bank`, and `safe_exit_or_recovery_handoff`. Retired `launch_capture_*` IDs are archive aliases only and must not appear in active evidence. `safe_exit_or_recovery_handoff` remains active as an evidence-tested recovery / controlled-terminal primitive; it must be demoted only if R8/R10 evidence shows no unique transition coverage, not removed by assumption.

Every primitive is treated as a transition object. R5 is robust transition-aware primitive / transition-object Q/R plus primitive attitude/bank reference-bias training across five entry start families with exact dense percentage mix per primitive/candidate/evidence block: 40% `launch_gate`, 25% `inflight_nominal`, 15% `inflight_lift_region`, 10% `inflight_boundary_near`, and 10% `inflight_recovery_edge`. The dense evaluation target is now `8 * 32 * 8 * 50 = 102,400` rows: eight active primitives, 32 candidate designs, eight evidence blocks, and 50 paired tests per candidate per block. The evidence-block ladder keeps dry-air, fixed single-fan, and fixed four-fan anchors, then adds randomized single-fan, four-fan parameter, active fan-count 0/1/2/3/4, local fan-position, and arena-wide full-randomisation blocks so R5/R7 share the same uncertainty family as R10/R11. Row count alone is not a pass condition. Candidate 0 is nominal, candidates 1-7 are named physical anchors with small interpretable attitude/bank reference biases, and candidates 8-31 are deterministic Latin-hypercube log multipliers over the seven grouped LQR weights plus bounded pitch and bank reference biases. In-flight start-state velocity envelopes cover most of the local-speed scheduling grid: nominal `u=3.0--8.2`, lift-region `u=3.2--8.0`, boundary-near `u=3.0--8.0`, and recovery-edge `u=2.2--5.2` m/s, with wider lateral/vertical body-velocity perturbations logged by the sampler. Speed-bin selection is for local model scheduling only; active primitives must not chase speed as a hard reference. R5 writes `r5_transition_candidate_training_summary.csv`, `r5_transition_selected_for_r7.csv`, `r5_transition_pareto_front.csv`, and `r5_transition_training_manifest.json`, then freezes only selected transition objects for R7 while keeping the full candidate bundle as audit evidence.

R7 is held-out transition validation of the frozen R5-selected transition objects. No primitive may pass R7 solely on local rollout success; no primitive may pass R5 or R7 from dense row count or aggregate primitive success across entry classes. R7 replays the selected transition objects over the same eight-block anchor plus uncertainty-family ladder used by R5, but with W3 held-out plant/randomisation and no retuning: dry-air, fixed single-fan, fixed four-fan, randomized single-fan, four-fan parameter, active fan-count 0/1/2/3/4, local fan-position, and arena-wide full-randomisation blocks. R7 uses entry-class-specific labels: `survived` is reserved for strict high-probability `inflight_stable` evidence, `route_usable` keeps `launch_gate` evidence when transition probability is at least 0.40 with near-zero hard failure and keeps `boundary_near` evidence when transition probability is at least 0.40 with hard failure below limit, and `recovery_route_usable` keeps `recoverable_degraded` evidence when it has nonzero recovery progress in dry-air, single-fan, and four-fan R7 modes with low hard failure. For recovery starts, `recoverable_degraded -> recoverable_degraded` remains a conditional route pass when attitude/rate risk improves, front/side boundary time margin does not collapse, floor margin does not collapse, and hard-failure risk remains low; `recoverable_degraded -> boundary_near` is reported as route/weak evidence, not a full pass. A controller can be strict-surviving for one `primitive_id + entry_class`, route-usable for another, and fail for another. R8 compresses R7/W3-eligible transition objects (`survived`, `route_usable`, and `recovery_route_usable`) grouped by `primitive_id` and `transition_entry_class` using coverage-aware medoid selection without averaging Q/R, K, references, or controller IDs. R8 must preserve distinct W3-eligible local LQR speed-bin coverage and R7 evidence-block, uncertainty-tier, active-fan-policy, and fan-position-policy coverage within each primitive/entry-class group up to the effective entry-class budget; `launch_gate` uses wider step-0 budgets (`heavy=2`, `balanced=5`, `light=8`, `super_light=12`, `no_cluster=all`) while non-launch budgets remain `1/3/6/12/all`; speed-bin collapse is a library-coverage failure, not an LQR-principle failure, and uncertainty-block collapse is reported the same way.

The governor classifies the current state, filters representatives by validated `transition_entry_class`, rejects high hard-failure risk, and scores already-admissible candidates using transition/terminal probability, hard-failure risk, candidate-path front-wall progress, a front-wall terminal proxy, progress-gated terminal total specific-energy proxy, wrong-boundary penalty, context-conditioned updraft gain, lift-dwell time, candidate-specific spatial flow-belief memory correction, and calibrated regime-mismatch risk from the active calibration residual-blend AoA boundary (currently 14--18 deg from `neutral_dry_air_calibration.py`, with candidate and selector logs recording the source calibration ID plus boundary values). The regime term is a bounded score penalty and diagnostic split, not a hard anti-lift gate: high-AoA candidates remain selectable only when their mission value justifies model-mismatch exposure. Airborne or flight time is audit-only and must not be a selector or launch-score reward. Step 0 has `current_state_class = launch_gate`, so it selects only transition objects validated for `entry_class = launch_gate`; there is no launch-specific primitive family route. Because repeated-launch episodes must end somewhere, a finite controlled x-y arena exit with positive floor/ceiling margin is a `safe_terminal` outcome, not a hard failure. For final R9/R10/R11 launch scoring, `mission_success` is stricter than generic `safe_terminal`: the held-out rollout must exit through the front wall at `x_w = 6.6 m` with y/z inside the true safe bounds, while wrong-wall exits, floor/ceiling impact, invalid state, uncontrolled attitude/rates, and no-viable tails remain penalties or failures.

The inner LQR remains a stabilising tracker around the primitive-defined local reference. It must not become the manoeuvre planner: primitive transition objects define manoeuvre intent and R5/R7/R8 decide which transition objects are valid, while the governor selects among those frozen transition objects. Signed turn intent is diagnostic only: R5/R7 may record signed bank, signed roll-rate, and lateral turn tendency for audit, but active selection must not reward turn-expression strength and `mild_turn_left` / `mild_turn_right` must not receive sign-constrained bank or roll-rate reference forcing. `mild_turn_left` and `mild_turn_right` remain separate directional primitive IDs because the arena, local flow, and reachable transitions are not guaranteed symmetric; R8 may compress or downweight their representatives by evidence, but active code must not merge the IDs into one aggregate score. `energy_retaining_bank` remains non-directional because it is an energy/posture primitive, not an explicit left/right command. Same-start trajectory plots are diagnostic-only sanity checks for controller behaviour and do not replace R5/R7 transition evidence.

Spatial flow-belief memory is a bounded case-local safe-region modifier: `predicted_mission_utility = frozen_prediction + spatial_flow_belief_correction`. The active R9/R10/R11 outer-loop policy is candidate-specific, recency-weighted, baseline-shielded, and mission-aligned: each already-viable primitive receives candidate-path front-wall/energy utility plus a capped specific-energy-dominant correction from a lightweight 0.1 m x 0.1 m x 0.1 m 3D updraft-utility belief map. Each flown primitive writes dense executed-segment residual samples into the map at 0.1 m spacing with launch-index recency decay, and applies those samples in one batch per executed primitive so h10/h30 accumulate a fuller arena belief than h3 without rebuilding or logging the map once per sample; older launches become weak prior evidence. Candidate paths do not define the memory; the active controller and full diagnostics use the same selector contract as simulation: a cheap `geometry_only` candidate-path pass over transition-compatible candidates, one residual-belief lookup context per governor decision, and full 0.2 m spatial flow queries only for the selector shortlist before final bounded memory ranking. The timed in-flight boundary keeps table flushing, full candidate-row expansion, and post-hoc diagnostics outside the 0.100 s flight-control boundary, while open-loop and no-memory paths still bypass spatial-memory feature evaluation entirely. Full memory scoring still uses seven current-to-exit probes plus bounded reachable-cone and short-horizon route-flow probes from the candidate exit. Candidate map queries are collapsed into one cost-benefit memory value: remembered flow benefit plus a small information value minus frozen mission-score, front-progress, risk, and path-margin costs. The baseline shield then accepts a memory-selected candidate only when it improves the cost-benefit-adjusted score without transition-success, hard-failure, calibrated regime-mismatch risk, or path-exit-margin regression; a higher-risk remembered-flow switch is logged as `rejected_calibrated_regime_mismatch_risk_regression` rather than being accepted for a small memory benefit. Pure uncertainty exploration is no longer a separate selection block; under-observed map regions contribute only through the small information value inside the same cost-benefit term, with no fan-layout-specific or final-run-only logic. Memory uses launch-index half-life recency, so the most recent few history launches dominate while older launches remain weak prior evidence. The same shield is applied at every repeated-launch decision; the code does not branch on a launch being a known final mission. It must not override state classification or entry/exit compatibility. Memory is reinitialised per final test row. The learning strategy is two-level: online spatial flow-belief memory is case-local and reset per final test row, while R10 performs deterministic global calibration from claim-bearing full R10 final held-out safety rates plus all-final diagnostics and selector-opportunity diagnostics, then freezes exactly one governor config for R11. It uses bounded rule updates only: no profile ladder, Bayesian optimisation, neural tuning, or black-box search. R10 may tune memory sensitivity, cost-benefit memory weight/cap/cost terms, shield margins, exploration thresholds, residual caps, confidence observations, and recency half-life from selector-opportunity diagnostics; R11 treats the frozen R10 handoff as validation input. R9/R10/R11 write `memory_opportunity_summary.csv` plus `memory_opportunity_decision_log.csv` for small runs, or a small `memory_opportunity_decision_log.csv` index plus partitioned row logs under `tables/memory_opportunity_decision_log/` for large R10/R11 runs, to show baseline-vs-memory score gaps, memory-objective gates, flow-belief correction deltas, route-flow and information-gain scores, calibrated alpha/regime/risk fields, shield status, and accepted/rejected switch reasons without producing a single GitHub-incompatible CSV. Episode summaries use `memory_changed_selection` only for accepted memory-shield switches inside the episode; baseline-vs-policy selection differences are logged separately so memory audits do not depend on that field for a different meaning. Final scoring is computed only from the final held-out rollout path: R9/R10/R11 launch score rewards front-wall mission completion, capped updraft-gain and lift-dwell evidence, and terminal total specific-energy reserve after front-wall success. Airborne time and generic net/gross energy drift remain audit-only. There is no hidden speed gate, no energy-loss hard failure, no PD/PID, no TVLQR, and no fan-layout-specific controller logic.

The repeated-launch outer-loop scheduler must not assume free real-time computation. R9/R10/R11 profile context construction, a cheap candidate-path geometry pre-pass, shortlisted spatial flow-belief query, and compact-library selection after a real-time compatibility shortlist keyed by transition entry class and current nearest local LQR speed bin, with nearest populated-bin fallback if the exact speed bin has no candidates. Every primitive decision is profiled against a preferred 20 ms controller-slot budget and a hard 0.100 s primitive-boundary budget while recording total library candidates, evaluated shortlist candidates, and skipped candidates. The required real-flight timing scope is `heavy_cluster` and `balanced_cluster`, which must satisfy the 0.100 s hard in-flight boundary budget; `light_cluster` and `super_light_cluster` are optional extended-library diagnostics where limited violations are reported but do not block real-flight use; `no_cluster_no_merge` is unrestricted offline comparison/stress evidence. The first real-flight experiment adopts `balanced_cluster` as the single active deployment tier because E03 is now the active frozen evidence set: R5/R8/R10 executable inputs are frozen under the E03 evidence-sensitive calibration conventions, R10 E03 passes the cluster-scoped training gate with balanced/light claim profiles, and the R11 E03.1 through E03.4 repeat set provides the current held-out validation boundary. That repeat set is not a strict gate pass; it is accepted only as physically explainable evidence where low-start-energy dry-air and L7 full-domain cases dominate failures, closed-loop is not broadly worse than open-loop, spatial memory remains bounded, and the heavy/balanced in-flight timing audit still has limited 0.100 s violations. The aggregate E03.1 through E03.4 appendix table is `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_4.tex` and includes matched true-neutral/open-loop diagnostics for every repeat. `heavy_cluster` remains a compact fallback when runtime or library size becomes the limiting deployment concern. This choice is a deployment tradeoff and does not claim that one compact library dominates every speed bin, environment ladder, or repeated-launch policy. Environment/context caches are warmed before profiled primitive decisions. Step 0 is prepared from the approved launch-gate state during the fixed 0.040 s neutral handoff, and active command emission starts from the latest post-handoff state at the handoff boundary. Later real-flight steps use the hybrid scheduler implemented in `run_real_flight.py` and `frozen_flight_controller.py`: during primitive A, a one-thread background prepare path predicts the A/B boundary state, selects primitive B without emitting a packet, then commits B at the boundary using the latest Vicon state for the first command packet; if preparation is late, the runtime logs `continuation_decision_late` and continues streaming the active primitive until the prepared decision is available. Runtime command FIFO order is old-to-new, while cross-primitive FIFO continuity remains a separate audit item. The timing boundary measures controller compute only: context construction, candidate-path geometry pre-pass, shortlisted spatial-belief query, and compact controller-row library selector computation; table flushing, full candidate-row expansion, and post-hoc diagnostic row construction are outside the flight-control boundary. If step-0 memory selection misses the fixed 0.040 s launch handoff, the runtime logs `first_launch_decision_missed_handoff_budget`, keeps the launch-gate approval in the record, and marks the attempt non-valid for controlled evidence so it does not update memory or consume a target valid throw. Real-flight active logging uses `buffer_active_rows_flush_after_active_record`: time-critical governor commit and 50 Hz packet emission precede active metric flush, fan positions are logged only once as a prelaunch fan snapshot before active flight, with no fan polling during launch handoff or post-exit, and the active runtime wakes 2 ms ahead of scheduled events to reduce scheduler-lag risk. The latest targeted C16 sanity check for this boundary used 40 final launches and 430 history launches, with 0 hard failures, 0 no-viable events, 13/13 accepted memory switches, and required heavy/balanced in-flight decisions at 144/144 under 0.100 s with max 0.0937 s; this is targeted diagnostic evidence only, not a full R10/R11 validation or broad memory-improvement claim. The post-v4.4 real-flight memory timing probe used 159 synthetic memory cells and 30 launch decisions, with max first-decision time 0.0239 s and zero calls above 0.040 s; the completed E1.2 dry-air memory null run then recorded max step-0 first-decision time 0.0386 s, max decision time 0.0570 s, and zero decisions above 0.100 s. These are engineering timing and workflow checks, not new R10/R11 or memory-improvement claims.

R9/R10/R11 and real-flight reporting now include compact posthoc score audits in addition to success-rate summaries. Simulation runs write `metrics/posthoc_final.csv` for one-row-per-final-launch `LAUNCH_SCORE_VERSION` outcomes, `metrics/posthoc_exec.csv` for the accumulated selected primitive scores and score components that were actually executed, and `metrics/posthoc_delta.csv` for paired deltas versus no-memory and open-loop baselines. Real-flight throws write `posthoc_throw.csv`, session runs write `posthoc_session.csv`, and closed-loop `controller_decisions.csv` carries selected score component fields; open-loop neutral still has zero controller decisions but an explicit posthoc row. These tables are report-only, use short GitHub-safe filenames, and stay outside the real-time controller compute boundary. Real-flight simulation-replay figures now live under `04_Flight_Test/A_figures/real_flight_sim_replay/` and are generated by `04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py`. Each valid throw is plotted against two simulation replies: one where the simulator re-selects primitives with the frozen governor, and one where the simulator reuses the logged real `primitive_variant_id` sequence when present. Open-loop neutral throws without `controller_decisions.csv` remain valid replay inputs; their `sim_real_decisions` trace has an empty selected-primitive sequence and stays neutral after the measured handoff splice. Replay version `real_flight_sim_replay_measured_fan_updraft_v2` builds the per-throw replay environment from `fan_positions.csv`: W2 single-fan cases use the measured x/y fan snapshot with the fixed annular-GP updraft model, nominal fan power/width, and the active mask, while no-visible-fan dry-air cases use the zero-wind replay environment. Both replays copy the measured 0.040 s handoff state trace, start simulation from the measured post-handoff state, apply runtime `surface_command_delay_s` before actuator lag, pass the replay wind field into the dynamics, and write `real_flight_sim_replay_summary.csv`, per-throw replay traces, `first_0p04_state_replay_error_summary.csv`, `execution_timing_audit.csv`, `replay_environment_summary.csv`, and a Markdown report. Figures include a bottom-left launch-condition block with launch state, speed, attitude, body velocity/rates, and handoff duration; fan cases also draw R11-style 3D updraft slices and isosurfaces. This is a replay and timing diagnostic layer, not an additional SysID fit, R10/R11 validation claim, fan-flow validation, simulation-side mission-score recomputation, or mission-success claim.

The current E1 random-sample replay audit is stored under `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/`. It plots E1.0 redo `20260607_190445` throws 005 and 010, E1.1 no-memory `20260606_230007` throws 005 and 011, and E1.2 memory `20260607_122640` throws 002 and 029, all with `balanced_cluster`, logged real-decision timing, and replay version `real_flight_sim_replay_measured_fan_updraft_v2`. The first-0.040 s state-splice audit reports zero maximum residual across the sample set, so the comparison starts from the agreed measured-handoff alignment. Because E1 is dry-air and no fan is visible, `replay_environment_summary.csv` records W0 zero-wind replay with `updraft_max_m_s = 0.0` for every E1 sample; the simulator-own reply remains the closed-loop counterfactual, while open-loop E1.0 `sim_real_decisions` stays neutral. The replay figures show real-vs-simulation model mismatch and closed-loop robustness diagnostics; they do not prove that real flight scores higher than simulation because simulation replay scores are not recomputed there. The real selected-score audit remains runtime/posthoc-only: E1.1 session mean accumulated selected score is -1.2406162083333327, E1.2 is 0.8705906510995837, and E1.0 is 0.0 because open-loop has no selected controller decisions.

The completed dry-air real-flight workflow now has three current records under active calibration profile hash `c4fca6c930dd4a9c9836c53fd3bb796ac14973d08c32796a1ecf0d155edd2d2f`: redo E1.0 `20260607_190445` is the open-loop neutral baseline with 10 valid throws, 6 rejected starts, zero controller decisions, speed range 5.016--6.653 m/s, mean final observable specific energy 1.634 m, 3 front-wall exits, and 7 floor exits; E1.1 `20260606_230007` is closed-loop no-memory with 30 valid throws, 16 rejected/timeout starts, 30/30 active controlled throws, 10--12 controller decisions per valid throw, 50--60 20 ms slot updates per valid throw, max decision time 0.00432 s, speed range 5.295--6.841 m/s, mean final observable specific energy 1.747 m, 28 front-wall exits, and 2 floor exits; E1.2 `20260607_122640` is the dry-air memory null test with 30 valid throws, 10 rejected starts, 30/30 active controlled throws, 10--12 controller decisions per valid throw, max first-decision time 0.0386 s, max decision time 0.0570 s, speed range 5.287--6.774 m/s, mean final observable specific energy 1.771 m, 21 front-wall exits, 9 floor exits, 297 final memory cells, and 339 memory updates over history buckets h0/h1-3/h4-10/h11-30 = 1/3/7/19. Dry-air memory is interpreted as a bounded null/safety test: it should not invent a fan-updraft improvement claim, and rejected starts do not update memory or count as valid throws.

The completed E2 single-fan workflow now has open-loop, no-memory, and memory-enabled records under the same active calibration profile. E2.0 `20260607_163303` is the fixed single-fan open-loop neutral baseline with 10 valid throws, 8 rejected starts, one visible fan snapshot per throw, speed range 5.626--6.260 m/s, mean speed 5.977 m/s, mean final observable specific energy 1.506 m, 6 front-wall exits, and 4 floor exits. E2.1 `20260607_165533` is fixed single-fan closed-loop no-memory with 30 valid throws, 10 rejected starts, controller decisions on every valid controlled throw, max decision time 0.03197 s, speed range 5.053--6.412 m/s, mean speed 5.936 m/s, mean final observable specific energy 1.635 m, 24 front-wall exits, and 6 floor exits. E2.2 is collected as two 30-valid-throw memory sessions, `20260607_173345` and `20260607_175359`, with 60 valid rows total, 19 rejected starts, max decision times 0.08695 s and 0.07169 s, mean speed 5.758 m/s, mean final observable specific energy 1.625 m, 44 front-wall exits, 14 floor exits, and two blank terminal rows, one of which is the logged non-controlled launch-handoff abort `launch_handoff_abort:vicon_invalid:vicon_subject_occluded`. The non-abort E2.2 memory rows keep positive selected and memory score components, with h4-10 history giving 13/14 front-wall exits, mean speed 5.876 m/s, and mean final observable specific energy 1.681 m. E2 supports single-fan transfer, timing, speed, and memory diagnostics; the raw E2.2-versus-E2.1 comparison is speed-confounded and should not be presented as a broad standalone memory-improvement proof.

R10 and R11 have separate roles. R10 is governor/spatial-memory tuning on one hard training distribution, `r10_l7_full_domain_randomisation_arena_wide_training`, with 50 outer cases per condition: four-fan geometry, active fan count 0/1/2/3/4, fan parameter uncertainty, arena-wide non-overlapping fan positions, and W3 plant/implementation perturbations. Because R10 is a randomised tuning stage rather than a final claim gate, its blocking pass profile is cluster-scoped: any one of `heavy_cluster`, `balanced_cluster`, or `light_cluster` may claim success when its claim-bearing final rows satisfy the stage profile (hard failure <= 0.20, floor/ceiling violation <= 0.20, final no-viable <= 0.30, safe success >= 0.20, and terminal-or-lift capture >= 0.30). Aggregate all-cluster rates remain diagnostics, are still reported exactly, and continue to drive conservative governor tuning. R10 also tunes mission/risk weights plus memory sensitivity, cost-benefit memory weight/cap/cost terms, shield margins, exploration thresholds, residual caps, confidence observations, and recency half-life through the same deterministic bounded rule update while writing improvement diagnostics for the frozen R11 handoff. Arena-wide fan positions are rejection-sampled inside the tracker footprint with a 0.5 m safety radius around each fan, so fan-centre distances must be at least 1.0 m and safety circles do not overlap. Within each R10 outer case, fan number, fan positions, and plant/implementation are fixed across history and final launches; launch state varies on every throw, and only nominal +/-15 percent static fan power and width plus measured-scatter uncertainty-scale noise may vary between launches. R11 is held-out validation on an eight-block fidelity ladder with 20 outer cases per block, 160 total, paired by stratified local launch-start indices spread across the 0--49 launch grid and reused across L0--L7; this clean full R11 unit schedules 4,000 final held-out launches plus 34,400 history launches, and richer evidence should come from repeated full-R11 seeds. The code retains `--smoke-outer-cases-per-block` only for engineering readiness checks; smoke runs are labelled non-claim-bearing and must not replace this full R11 unit: L0 dry-air fixed, L1 single-fan fixed nominal, L2 four-fan fixed nominal, L3 fan-parameter uncertainty, L4 local fan-position uncertainty, L5 active-fan-count uncertainty 0/1/2/3/4, L6 environment-only full uncertainty, and L7 full-domain arena-wide randomisation. R11 uses the same memory-episode rule: L0-L2 are fixed nominal environments; L3 keeps fan count/positions fixed while nominal static fan parameters and measured-scatter uncertainty vary; L4 chooses one locally shifted layout per outer case and holds it fixed; L5 chooses one active fan count per outer case and holds it fixed; L6 fixes one local layout and active fan count while nominal static fan parameters and measured-scatter uncertainty vary; L7 fixes one arena-wide layout, active fan count, and plant/implementation per outer case while nominal static fan parameters and measured-scatter uncertainty vary. R11 additionally writes a comparison-only, non-claim-bearing open-loop zero-command baseline for every final held-out case; the claim-bearing core comparison remains no memory versus recency-weighted spatial flow-belief memory histories h3, h10, and h30 with built-in cost-benefit spatial flow memory. h100 and standalone safe-explore policy variants are optional appendix ablations. R10/R11 post-analysis is speed-conditioned: `speed_bin_policy_ladder_summary.csv` groups success, safety, score, updraft, and terminal-energy metrics by environment ladder, launch-speed bin (`<4.0`, `4.0--5.0`, `5.0--6.0`, `6.0--7.0`, and `>=7.0` m/s), library-size case, policy, and history length; `start_energy_group_policy_ladder_summary.csv` groups the same evidence into low/high start-energy starts split at 5.0 m/s. Paired memory-score deltas are also split by speed bin and start-energy group. These are reporting layers only: they do not change selection, scoring, simulation physics, or any pass gate, but they support an explicit launch-energy operating-envelope interpretation for thesis and journal presentation. The appendix generator `03_Control/01_Plotting/run_r11_appendix_speed_tables.py` converts the R11 speed-conditioned summary into 16.2 cm `longtblr` tables with thesis-facing metric definitions. Hardware readiness, real-flight transfer, mission success, autonomy, and memory-improvement claims require R11 and later real-flight evidence.

<!-- R9_LAUNCH_GATE_ALIGNMENT_END -->

## Purpose

This file defines stable working rules for coding, writing, experiments, slides, schedules, and project decisions. It should not contain daily targets, run IDs, exact launch gates, or short-lived implementation plans. Those belong in the project plan and daily schedule.

The goal is to make every output useful, auditable, easy to explain, and suitable for a robotics-style thesis or paper.

---

## 0. Source priority

When sources conflict, use this order:

1. Latest explicit user instruction in the current conversation.
2. Latest project plan or project contract.
3. Latest daily schedule for execution timing.
4. Latest thesis draft for submitted structure and wording constraints.
5. Current repository state.
6. Older plans, slides, logbooks, and code attempts.
7. Literature or web sources.

Do not silently merge conflicting versions. State which source controls the decision when it matters.

---

## 1. Stable project centre

The stable centre is:

> Primitive-level sim-to-real transfer of feedback-stabilised fixed-wing manoeuvre primitives for a small glider operating in measured, uncertain indoor updrafts.

The preferred method is transition-aware primitive selection: each 0.100 s LQR primitive is treated as an entry-class -> exit-class object, and the active governor selects only primitives whose entry class matches the current state class and whose predicted exit is chain-compatible. The active thesis workflow is R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9, safe-explore policies, and large history-matrix studies are internal audit or ablation material only. Low speed is not an active governor rejection reason, recovery-route trigger, score factor, or audit gate; the physical rollout decides whether the simulated flight actually hits floor, ceiling, wall, or an unrecoverable terminal condition. Energy loss is also not a hard-failure reason; R9/R10/R11 score useful updraft extraction and terminal total specific-energy reserve separately from whole-flight net energy drift and gross energy loss. Model-backed rollout evidence must integrate positive wing-panel vertical wind along the simulated primitive trajectory; primitive-start local `w_wing` exposure is only a smoke/legacy fallback. The post-W3 representative outcome table is only a base estimate; R9/R10/R11 must pass it through a robust context-conditioned adapter before governor scoring. That adapter may only downgrade probabilities, cap soft rewards, or increase hard-failure risk based on current transition state class, environment class, local lift, local uncertainty, environment block, active fan count, and numeric positive `sample_count` coverage. Governor soft scoring uses context-conditioned `expected_updraft_gain_proxy_m` with active weights named `updraft_gain_weight` and `terminal_updraft_gain_weight`, plus mission terms for candidate-path front-wall progress, front-wall terminal proximity, progress-gated total specific-energy reserve, and wrong-boundary avoidance. Dry/no-local-lift contexts must not inherit positive updraft score from aggregate representative evidence. Spatial flow-belief memory stores both `updraft_gain_residual_m` and total `specific_energy_residual_m` in 0.1 m 3D cells; the active candidate-path memory utility is specific-energy dominant, with updraft residual retained as an auxiliary lift-usefulness signal. Missing compact-library outcome evidence is its own rejection, not a zero-probability proxy. `safe_success` remains a broad sequence-compliance safety metric, while `mission_success` is stricter and requires the final held-out rollout to terminate through the front wall at `x_w = 6.6 m` with y/z inside the true safe bounds. Terminal-useful safe exits must be reported separately from `full_safe_success`, and front-wall mission success must be reported separately from generic safe terminal outcomes. Memory is reinitialised per final test row and may only use that row's own history launches; it must not carry between outer cases, library-size cases, policies, or validation blocks. Final scoring is computed only from the final held-out rollout path under controlled pairing: front-wall mission completion, capped updraft-gain/lift-dwell evidence, and terminal total specific-energy reserve after front-wall success are rewarded; airborne time, net energy drift, and gross energy loss are audit-only.

Treat the LQR stabiliser as part of the primitive. The active control/evidence unit is a primitive-controller variant, not a free-standing controller bank. R5 dry-air plus annular-GP dense generation tunes a primitive-local LQR for every generated variant and preserves the rich library; dry-air rows use no updraft but still use W3-style plant and implementation perturbations. Axisymmetric Gaussian plume evidence is diagnostic-only. W2 is optional diagnostic evidence only, while W3 replays fixed variants under held-out randomisation to eliminate or downgrade cases that fail. Clustering and merging occur only after W3, and late validation freezes the post-W3 library-size condition, governor, selector, and memory logic. Hidden retuning inside W2, W3, clustering, or validation is not allowed. The active model is a simplified as-built rigid-body strip-aerodynamics model with deterministic mass/inertia constants, local-speed scheduled LQR linearisation, and explicit plant/implementation perturbation metadata; do not write as if it is CFD, aeroelastic, or flight-identified.

The LQR Q/R and attitude/bank reference-bias sweep is not a global optimal-control result. It
is a deterministic design-of-experiments lookup surface, followed by R5
transition-evidence selection and R7 held-out validation. Use wording such as
`R5-selected`, `empirically robust`, or `transition-validated`; avoid saying the
controller is optimal unless a later proof or validation gate explicitly supports
that narrower claim.

Do not turn the work into:

- a generic autopilot project;
- a generic path-following project;
- direct reinforcement-learning surface control;
- a broad nonlinear MPC project;
- a platform-only design project;
- a presentation of attractive trajectories without transfer evidence.

Use **primitive**, not skill, in project-specific writing and code comments.

---

## 2. Technical invariants

Preserve unless explicitly changed:

- Public thesis world frame: `z` up.
- Body frame: `x` forward, `y` starboard, `z` down.
- State order: `[x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r]`.
- Command order: `[delta_a_cmd, delta_e_cmd, delta_r_cmd]`.
- Positive aileron: left trailing edge up, right trailing edge down; negative roll moment.
- Positive elevator: positive pitch moment, nose up.
- Positive rudder: positive yaw moment, nose right.
- Angles in code: radians unless a field name explicitly says `_deg`.
- Distances: metres. Times: seconds.
- Wind modes: no wind, centre-of-gravity wind, panelwise wind.
- Latency evidence must distinguish ideal timing, actuator lag, command delay, state delay, nominal latency, and conservative timing.
- W0/W1 primitive-controller evidence must already use the panel-wise glider model, feedback latency, command timing, and actuator lag when those effects affect LQR tuning.
- LQR synthesis must use explicit local speed bins and must not depend on optimizer CSV output or an implicit global 6.5 m/s trim target. The speed bin schedules the local model; the active command law must not chase longitudinal speed as a hard target.
- LQR tuning must be described as deterministic candidate generation plus R5/R7 evidence selection, not online optimisation and not a mathematically global optimum claim.

---

## 3. Robotics-journal orientation

The writing and experiments should target a top robotics-journal style rather than a theory-first control-journal style.

Emphasise:

- real system behaviour;
- measured environment and instrumentation;
- baselines and ablations;
- failure cases and boundaries;
- traceable logs and reproducibility;
- compact equations tied to implementation;
- figures that explain what the robot did and why it mattered.

Do not over-emphasise:

- formal proof without experiment;
- elegant theory that does not change robot behaviour;
- broad claims beyond the measured system;
- simulation-only results as if they were hardware validation.

---

## 4. Coding skill

Code should be clean, short enough, easy to understand, efficient, and auditable.

| Requirement | Meaning |
|---|---|
| Clean | clear module boundary, explicit units, no hidden frame/sign conversion |
| Short enough | no speculative framework, but do not remove useful comments or audit fields |
| Understandable | conventional flight-dynamics notation, small functions, explicit assumptions |
| Efficient | avoids unnecessary allocation and slow repeated work |
| Auditable | produces tests, manifests, metrics, or reports matched to the claim |

Rich comments are encouraged. They must explain scientific or engineering meaning, not obvious syntax.

Use comments for:

- frame, units, and sign conventions;
- context-feature definitions;
- equations and approximations;
- why a safety gate exists;
- why a metric supports or does not support a claim;
- why a branch is simulation-only, hardware-shakedown, or blocked;
- assumptions behind sampling, post-W3 library-size cross-study, and latency labels.

For nontrivial modules, use a section map when it improves auditability.

Prefer:

- type hints;
- dataclasses for structured records;
- `Path` for file paths;
- explicit unit suffixes;
- deterministic seeds;
- small pure helpers for physics and metrics;
- manifests for evidence-generating scripts.

Avoid:

- hidden global state;
- broad mutable dictionaries as the main interface;
- silent unit conversion;
- unlabelled arrays;
- unused extension ports;
- adding dependencies without reason.

---

## 5. Runtime, storage, and file-size skill

Dense/archive/thesis-scale simulation must use the approved runtime strategy:

```text
chunked execution
resumable chunk manifests
compressed table partitions
worker count selected by profiling and memory guardrails
checksums and table manifests
no giant in-memory final table
```

For R9/R10/R11 repeated-launch validation, worker parallelism is across independent final held-out schedule rows. Each worker must run that row's history launches sequentially before its final launch; do not parallelise within a memory-history chain. R10 is a single hard full-domain arena-wide tuning block with 50 outer cases per condition, active fan count 0/1/2/3/4, non-overlapping arena-wide fan positions, fan-parameter randomisation, and W3 plant/implementation perturbation sampled once per outer case. Within each R10 outer case, fan count, fan layout, and plant/implementation are fixed across history/final launches; launch state varies, and only nominal +/-15 percent static fan power and width plus measured-scatter uncertainty-scale noise may vary between launches. Its tuning pass profile is cluster-scoped rather than aggregate-blocking: any one of `heavy_cluster`, `balanced_cluster`, or `light_cluster` may claim success when its claim-bearing final rows satisfy hard failure <= 0.20, floor/ceiling violation <= 0.20, final no-viable <= 0.30, safe success >= 0.20, and terminal-or-lift capture >= 0.30; aggregate all-cluster rates remain reported diagnostics and conservative tuning inputs. The frozen governor config written by R10 is the handoff input for R11; this is a single handoff config, not a profile ladder or black-box search result. R11 is the eight-block held-out fidelity ladder with 20 outer cases per block, 160 total, paired by stratified local launch-start indices spread across the 0--49 launch grid and reused across L0--L7; this clean full R11 unit schedules 4,000 final held-out launches plus 34,400 history launches, and richer evidence should come from repeated full-R11 seeds. The code retains `--smoke-outer-cases-per-block` only for engineering readiness checks; smoke runs are labelled non-claim-bearing and must not replace this full R11 unit: dry air fixed, single fan fixed, four fan fixed, fan-parameter uncertainty, local fan-position uncertainty, active-fan-count uncertainty, environment-only full uncertainty, and full-domain arena-wide randomisation. R11 uses a fixed-memory-episode rule: L0-L2 are fixed nominal environments; L3 keeps fan count/positions fixed while nominal static fan parameters and measured-scatter uncertainty vary; L4 chooses one locally shifted layout per outer case and holds it fixed; L5 chooses one active fan count per outer case and holds it fixed; L6 fixes one local layout and active fan count while nominal static fan parameters and measured-scatter uncertainty vary; L7 fixes one arena-wide layout, active fan count, and plant/implementation per outer case while nominal static fan parameters and measured-scatter uncertainty vary. R9 is a quick internal preflight with a 10 s episode budget, all five library-size cases, the three fixed blocks no-updraft/single-fan/four-fan, and `no_memory_baseline` plus `spatial_flow_belief_memory_h3/h10/h30`, giving 60 final held-out launches and 645 history launches. Any primitive-count cap is diagnostic only. Spatial flow-belief memory uses `outer_loop_cost_benefit_spatial_flow_memory_v4_1`: each flown primitive writes dense executed-segment residual samples into a 0.1 m 3D updraft-utility map at 0.1 m spacing with launch-index recency decay. The active controller and diagnostics share the same selector contract as simulation: a cheap `geometry_only` candidate-path pass over transition-compatible candidates, one residual-belief lookup context per governor decision, and full 0.2 m spatial flow queries only for the selector shortlist before final bounded memory ranking. Full memory scoring still uses the seven current-to-exit probes plus bounded reachable-cone and short-horizon route-flow probes from the candidate exit, while table flushing, full candidate-row expansion, and post-hoc diagnostics stay outside the 0.100 s flight-control boundary. Residual path utility, reachable-flow attraction, and route-flow exploitation use known useful flow, while under-observed path/cone/route probes add bounded information gain for safe map building among already-viable front-progress-compatible candidates, with launch-index half-life recency, confidence/effective-count gating, baseline-shield acceptance checks, candidate-path exit-margin non-regression, and unchanged safety filtering. The same candidate-path geometry supplies mission utility for no-memory and memory policies, while reachable-flow attraction is memory-only: forward progress toward `x_w = 6.6 m`, front-wall terminal proxy, progress-gated terminal total specific-energy proxy, and wrong-boundary avoidance. `memory_opportunity_summary.csv` explains aggregate memory opportunities; `memory_opportunity_decision_log.csv` is the row-level log for small runs and a small partition index for large R10/R11 runs, with full row-level evidence under `tables/memory_opportunity_decision_log/`, so baseline-vs-memory score gaps, memory-objective gates, residual correction deltas, route-flow and information-gain scores, and accepted/rejected switch reasons are preserved without producing a single GitHub-incompatible CSV. Selected primitive-family count, selected variant count, and `lift_dwell_arc` selection are diagnostics only, not hard governor pass gates. Final held-out launch scoring uses the front-wall mission score: `mission_success` requires front-wall exit at `x_w = 6.6 m` with y/z inside the true safe bounds, updraft-gain and lift-dwell bonuses are capped, terminal total specific-energy reserve is rewarded only after front-wall success, and airborne time/net/gross energy drift are audit-only. Full R10/R11 use `history_log_mode=plot_summary` by default: every history launch retains episode summaries, selected primitive execution rows, `history_plot_trace`, `history_memory_trace`, and `history_selector_summary`, while verbose per-decision candidate/selector/memory/belief debug tables are final-only unless R9, reduced diagnostics, or explicit sampled/full debug mode is requested. Low-launch-speed dry-air or scheduled-zero-fan floor stops keep the raw primitive `floor_violation` audit label, but episode summaries label them `expected_low_energy_dry_air_sink` and exclude them from governor/memory claim-bearing gates. R10/R11 post-analysis writes speed-conditioned and start-energy-conditioned summaries: `speed_bin_policy_ladder_summary.csv`, `start_energy_group_policy_ladder_summary.csv`, `paired_score_delta_by_speed_bin_summary.csv`, and `paired_score_delta_by_start_energy_group_summary.csv`. Speed bins are `<4.0`, `4.0--5.0`, `5.0--6.0`, `6.0--7.0`, and `>=7.0` m/s; the low/high start-energy split is 5.0 m/s and is a reporting boundary only, not a hidden selector or scoring gate. Repeated-launch loaders resolve stale legacy `lqr_contextual_v1_0` embedded source paths to the renamed direct `05_Results` stage folders at read time and record the resolved frozen-controller source root in new manifests; old A01 result manifests are not edited in place.

Default local dense-run policy:

```text
workers = 16
max_workers = 16
storage_format = auto, resolving to parquet if available, otherwise csv_gz
compression_level = 1 for csv_gz
resume = true
```

Every generated file should stay below 100 MB. Prefer below 75 MB. If a file may exceed 100 MB, split it into partitions before writing. Large local-only evidence must be explicitly labelled and must not be pushed without approval.

---

## 6. Writing skill

Writing should be formal, natural, and traceable. It should not look like AI-generated text.

Use:

- one paragraph, one purpose;
- claim, evidence, limitation order;
- common technical words before rare words;
- no inflated novelty statements;
- no repeated claim under different wording;
- no mechanical transition stacking;
- no unsupported generalisation;
- no unnecessary bold, italics, or quotation marks in thesis text;
- no em dash as a default connector.

Before finalising important text, ask:

- What is the claim?
- What evidence supports it?
- What does it not prove?
- Could this sentence appear in any generic UAV paper?

---

## 7. Literature skill

Literature should justify the project position, not decorate the thesis.

Use references to establish:

- what problem class already exists;
- what assumptions those works make;
- what evidence they provide;
- what is different in this project;
- what limitation remains.

Perching literature may be useful as contrast, but it must not define the final objective.

---

## 8. Plotting and slides

Figures and slides should carry evidence, not just labels.

Good figures usually show:

- measured environment or context map;
- primitive outcome envelope;
- accepted / weak / failed / rejected regions;
- repeated-launch traces;
- belief evolution;
- baseline comparison;
- sim-real pairing;
- failure labels.

Every thesis-body figure should answer one question that a reader can understand without code.

---

## 9. Output standard

For every substantial answer, state:

- files or sources used;
- assumptions made;
- changes made or proposed;
- what remains uncertain;
- what should be tested next.

Use concise answers unless detail improves correctness, auditability, or thesis quality.
