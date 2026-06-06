# Repository Housekeeping, Naming, Runtime, and Storage Rules

<!-- R9_LAUNCH_GATE_ALIGNMENT_START -->

## Active Transition-Aware Thesis Workflow

The active thesis workflow is `R5 -> R7 -> R8 -> R10 -> R11 -> Reality`. R9 remains internal preflight only and is not thesis-facing evidence. R10 tunes the viability governor with residual updraft adaptation, and R11 is the held-out validation gate.

Launch is an entry regime, not a separate controller family. The active primitive catalogue has exactly eight manoeuvre families: `glide`, `recovery`, `lift_entry`, `lift_dwell_arc`, `mild_turn_left`, `mild_turn_right`, `energy_retaining_bank`, and `safe_exit_or_recovery_handoff`. Retired `launch_capture_*` IDs are archive aliases only and must not appear in active evidence. `safe_exit_or_recovery_handoff` remains active as an evidence-tested recovery / controlled-terminal primitive; it must be demoted only if R8/R10 evidence shows no unique transition coverage, not removed by assumption.

Every primitive is treated as a transition object. R5 is robust transition-aware primitive / transition-object Q/R plus primitive attitude/bank reference-bias training across five entry start families with exact dense percentage mix per primitive/candidate/evidence block: 40% `launch_gate`, 25% `inflight_nominal`, 15% `inflight_lift_region`, 10% `inflight_boundary_near`, and 10% `inflight_recovery_edge`. The dense evaluation target is now `8 * 32 * 8 * 50 = 102,400` rows: eight active primitives, 32 candidate designs, eight evidence blocks, and 50 paired tests per candidate per block. The evidence-block ladder keeps dry-air, fixed single-fan, and fixed four-fan anchors, then adds randomized single-fan, four-fan parameter, active fan-count 0/1/2/3/4, local fan-position, and arena-wide full-randomisation blocks so R5/R7 share the same uncertainty family as R10/R11. Row count alone is not a pass condition. Candidate 0 is nominal, candidates 1-7 are named physical anchors with small interpretable attitude/bank reference biases, and candidates 8-31 are deterministic Latin-hypercube log multipliers over the seven grouped LQR weights plus bounded pitch and bank reference biases. In-flight start-state velocity envelopes cover most of the local-speed scheduling grid: nominal `u=3.0--8.2`, lift-region `u=3.2--8.0`, boundary-near `u=3.0--8.0`, and recovery-edge `u=2.2--5.2` m/s, with wider lateral/vertical body-velocity perturbations logged by the sampler. Speed-bin selection is for local model scheduling only; active primitives must not chase speed as a hard reference. R5 writes `r5_transition_candidate_training_summary.csv`, `r5_transition_selected_for_r7.csv`, `r5_transition_pareto_front.csv`, and `r5_transition_training_manifest.json`, then freezes only selected transition objects for R7 while keeping the full candidate bundle as audit evidence.

R7 is held-out transition validation of the frozen R5-selected transition objects. No primitive may pass R7 solely on local rollout success; no primitive may pass R5 or R7 from dense row count or aggregate primitive success across entry classes. R7 replays the selected transition objects over the same eight-block anchor plus uncertainty-family ladder used by R5, but with W3 held-out plant/randomisation and no retuning: dry-air, fixed single-fan, fixed four-fan, randomized single-fan, four-fan parameter, active fan-count 0/1/2/3/4, local fan-position, and arena-wide full-randomisation blocks. R7 uses entry-class-specific labels: `survived` is reserved for strict high-probability `inflight_stable` evidence, `route_usable` keeps `launch_gate` evidence when transition probability is at least 0.40 with near-zero hard failure and keeps `boundary_near` evidence when transition probability is at least 0.40 with hard failure below limit, and `recovery_route_usable` keeps `recoverable_degraded` evidence when it has nonzero recovery progress in dry-air, single-fan, and four-fan R7 modes with low hard failure. For recovery starts, `recoverable_degraded -> recoverable_degraded` remains a conditional route pass when attitude/rate risk improves, front/side boundary time margin does not collapse, floor margin does not collapse, and hard-failure risk remains low; `recoverable_degraded -> boundary_near` is reported as route/weak evidence, not a full pass. A controller can be strict-surviving for one `primitive_id + entry_class`, route-usable for another, and fail for another. R8 compresses R7/W3-eligible transition objects (`survived`, `route_usable`, and `recovery_route_usable`) grouped by `primitive_id` and `transition_entry_class` using coverage-aware medoid selection without averaging Q/R, K, references, or controller IDs. R8 must preserve distinct W3-eligible local LQR speed-bin coverage and R7 evidence-block, uncertainty-tier, active-fan-policy, and fan-position-policy coverage within each primitive/entry-class group up to the effective entry-class budget; `launch_gate` uses wider step-0 budgets (`heavy=2`, `balanced=5`, `light=8`, `super_light=12`, `no_cluster=all`) while non-launch budgets remain `1/3/6/12/all`; speed-bin collapse is a library-coverage failure, not an LQR-principle failure, and uncertainty-block collapse is reported the same way.

The governor classifies the current state, filters representatives by validated `transition_entry_class`, rejects high hard-failure risk, and scores already-admissible candidates using transition/terminal probability, hard-failure risk, candidate-path front-wall progress, a front-wall terminal proxy, progress-gated terminal total specific-energy proxy, wrong-boundary penalty, context-conditioned updraft gain, lift-dwell time, candidate-specific spatial flow-belief memory correction, and calibrated regime-mismatch risk from the active normal / transition / post-stall AoA boundary. The regime term is a bounded score penalty and diagnostic split, not a hard anti-lift gate: high-AoA candidates remain selectable only when their mission value justifies model-mismatch exposure. Airborne or flight time is audit-only and must not be a selector or launch-score reward. Step 0 has `current_state_class = launch_gate`, so it selects only transition objects validated for `entry_class = launch_gate`; there is no launch-specific primitive family route. Because repeated-launch episodes must end somewhere, a finite controlled x-y arena exit with positive floor/ceiling margin is a `safe_terminal` outcome, not a hard failure. For final R9/R10/R11 launch scoring, `mission_success` is stricter than generic `safe_terminal`: the held-out rollout must exit through the front wall at `x_w = 6.6 m` with y/z inside the true safe bounds, while wrong-wall exits, floor/ceiling impact, invalid state, uncontrolled attitude/rates, and no-viable tails remain penalties or failures.

The inner LQR remains a stabilising tracker around the primitive-defined local reference. It must not become the manoeuvre planner: primitive transition objects define manoeuvre intent and R5/R7/R8 decide which transition objects are valid, while the governor selects among those frozen transition objects. Signed turn intent is diagnostic only: R5/R7 may record signed bank, signed roll-rate, and lateral turn tendency for audit, but active selection must not reward turn-expression strength and `mild_turn_left` / `mild_turn_right` must not receive sign-constrained bank or roll-rate reference forcing. `mild_turn_left` and `mild_turn_right` remain separate directional primitive IDs because the arena, local flow, and reachable transitions are not guaranteed symmetric; R8 may compress or downweight their representatives by evidence, but active code must not merge the IDs into one aggregate score. `energy_retaining_bank` remains non-directional because it is an energy/posture primitive, not an explicit left/right command. Same-start trajectory plots are diagnostic-only sanity checks for controller behaviour and do not replace R5/R7 transition evidence.

The active real-flight calibration path treats hand launch variation as part of the measured operating envelope rather than as an unmodelled nuisance. Vicon frame calibration writes the active position offset/profile into `04_Flight_Test/01_Runtime/calibration_profile.py`; direct armed closed-loop flight refuses to run without an explicit calibrated profile or offset, and also refuses to use calibration data unless the deployment evidence manifest confirms regenerated R5/R7/R8/R10/R11 inputs for the active profile hash. The physical and synthetic launch gate is stated in body-axis terms: `4.0 <= u <= 8.0 m/s`, `|v| <= 1.5 m/s`, and `|w| <= 0.9 m/s`, with the existing attitude and body-rate bounds. Simulation and real-flight runtime now share `launch_gate_neutral_handoff_0p040s_v1`: after launch-gate approval, both paths hold neutral/open-loop for exactly 0.040 s (two 20 ms controller slots), prepare the first launch-gate primitive from the approved state, and start the unchanged 0.100 s / 5-slot / 20 ms active primitive using the latest post-handoff state, while the launch-gate acceptance checks remain unchanged; step-0 launch rollout evidence therefore reports 0.140 s physical duration while active primitive metadata remains 0.100 s. Dry-air glider SysID remains neutral-first and now uses 0.040 s first-motion aligned replay, aligned with the synchronized launch-handoff boundary, plus randomised session-stratified held-out splits, lateral-only launch-confidence weighting, and 8-worker replay support. The replay-aligned post-analysis sanity filter uses `3.0 <= u <= 8.0 m/s`, `|v| <= 1.5 m/s`, and `|w| <= 0.9 m/s`; this relaxed lower `u` bound does not change the real launch gate because replay starts after the 0.040 s neutral handoff and is a post-alignment SysID acceptance filter only. The conservative trajectory-residual path uses `cm_regime_staged` for staged longitudinal fitting and `compact_joint_sweep` for from-active compact trade-off exploration: both keep the same compact model family, fitting attached Cm, transition Cm, post-stall Cm/Cmq, transition blend timing, and optional compact post-stall CL/CD cleanup instead of one global Cm offset. Outside the selected compact coupling terms, it keeps richer transition lateral deltas, post-stall lateral surfaces, and rich alpha-RBF longitudinal surfaces disabled unless explicitly requested, downweights replay-aligned edge-of-gate lateral starts relative to starts closer to `phi0=psi0=v0=p0=r0=0`, and uses excitation-aware weighting for compact lateral/coupling diagnostics. Staged runs write `neutral_aero_residual_cm_stage_history.csv` plus `neutral_aero_residual_stage_replay_errors.csv`; compact joint-sweep runs additionally write `neutral_aero_residual_joint_sweep_candidates.csv`, `neutral_aero_residual_joint_sweep_pareto.csv`, and `neutral_aero_residual_joint_sweep_selected.csv`. 0.040 s joint Pareto audits now have small and heavy profiles. The small profile keeps the earlier `n30_joint_pareto_040_audit` diagnostic path. The heavy bounded profile writes `n30_joint_pareto_040_heavy_v1` metrics/reports/manifests, keeps the same compact parameter family, evaluates top longitudinal bases plus scaled single/pair/capped triple lateral bundles on the same held-out 0.040 s split, and compares each lateral bundle against its own longitudinal-only base while also reporting the global best longitudinal reference. The corrected heavy run generated 210 candidates, 41 accepted rows, and 6 selected Pareto survivors; it localises the promising region to transition-blend 14/18 longitudinal proposals plus yaw-beta and post-stall Cl_r corrections. These rows are diagnostic and should feed a dense local Pareto refinement before any active coefficient promotion. The promoted compact coupling terms are treated as replay-alignment terms rather than accurate lateral SysID; the optional secondary lateral diagnostic freezes the compact candidate, uses excitation-aware lateral weighting, fits only `CY_beta`, `Cl_p`, and `Cn_r`, and is accepted only if held-out dy, roll, and yaw improve without degrading dx, altitude loss, sink, or pitch. Legacy 0.100 s alignment is retired from the default path; sensitivity alignment windows are optional replay-only diagnostics and do not change fitted coefficients or acceptance gates. The active checked-in constants are now `neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_elevator_rudder_effectiveness_tiny_cnbeta_heavy_sweep_v1`: the selected compact residual-calibrated replay model promoted after held-out replay, with attached Cm, transition Cm, post-stall Cm/Cmq, a delayed 12--22 deg residual blend, the selected compact coupling terms (`CY_beta`, transition `CY_r`, transition `Cn_p`, and post-stall `Cn_p` at 20 deg), and conservative elevator and rudder aerodynamic effectiveness scales (`delta_e=0.60`, `delta_r=0.531`). The old pure theory/geometry baseline is retained only as `03_Control/02_Inner_Loop/A_model_parameters/neutral_dry_air_theory_baseline_comparison.json` and is not imported by simulation or real-flight runtime code. The current mass model includes the 15 g nose ballast case: total mass is 148.56 g and x_CG is 10.55 cm aft of the wing LE with regenerated inertia. Pulse data are sustained single-axis control-effect evidence; the conservative elevator and rudder effectiveness scales are promoted, while aileron effectiveness, lateral/cross-axis derivatives, and alpha-regime surface schedules remain diagnostic because the held-out trade-offs are mixed and launch-condition contaminated. R5/R7/R10/R11 robustness runs perturb surface aero authority as simulation-only plant uncertainty by scaling the active `control_mix` columns with W3 multipliers `0.50--1.00` on all three axes around the active scales (`delta_a=1.00`, `delta_e=0.60`, `delta_r=0.531`); implementation-side achieved-surface effectiveness also stays within `0.50--1.00` on all three axes. This is not another promoted SysID claim. If the physical airframe is changed, pre-fix and post-fix throws must be separated or explicitly session/profile/build tagged before fitting.

Spatial flow-belief memory is a bounded case-local safe-region modifier: `predicted_mission_utility = frozen_prediction + spatial_flow_belief_correction`. The active R9/R10/R11 outer-loop policy is candidate-specific, recency-weighted, baseline-shielded, and mission-aligned: each already-viable primitive receives candidate-path front-wall/energy utility plus a capped specific-energy-dominant correction from a lightweight 0.1 m x 0.1 m x 0.1 m 3D updraft-utility belief map. Each flown primitive writes dense executed-segment residual samples into the map at 0.1 m spacing with launch-index recency decay, and applies those samples in one batch per executed primitive so h10/h30 accumulate a fuller arena belief than h3 without rebuilding or logging the map once per sample; older launches become weak prior evidence. Candidate paths do not define the memory; the in-flight controller and full diagnostics query the accumulated map through the same 0.2 m spatial neighbourhood over seven current-to-exit probes. The timed in-flight boundary uses a compact controller-row selector fast path before the 0.100 s boundary, while table flushing, full candidate-row expansion, and post-hoc diagnostics stay outside that boundary. Both use bounded current-to-exit, reachable-cone, and short-horizon route-flow probes from the candidate exit. Candidate map queries are collapsed into one cost-benefit memory value: remembered flow benefit plus a small information value minus frozen mission-score, front-progress, risk, and path-margin costs. The baseline shield then accepts a memory-selected candidate only when it improves the cost-benefit-adjusted score without transition-success, hard-failure, calibrated regime-mismatch risk, or path-exit-margin regression; a higher-risk remembered-flow switch is logged as `rejected_calibrated_regime_mismatch_risk_regression` rather than being accepted for a small memory benefit. Pure uncertainty exploration is no longer a separate selection block; under-observed map regions contribute only through the small information value inside the same cost-benefit term, with no fan-layout-specific or final-run-only logic. Memory uses launch-index half-life recency, so the most recent few history launches dominate while older launches remain weak prior evidence. The same shield is applied at every repeated-launch decision; the code does not branch on a launch being a known final mission. It must not override state classification or entry/exit compatibility. Memory is reinitialised per final test row. The learning strategy is two-level: online spatial flow-belief memory is case-local and reset per final test row, while R10 performs deterministic global calibration from all full R10 final held-out rows and selector-opportunity diagnostics, then freezes exactly one governor config for R11. It uses bounded rule updates only: no profile ladder, Bayesian optimisation, neural tuning, or black-box search. R10 may tune memory sensitivity, cost-benefit memory weight/cap/cost terms, shield margins, exploration thresholds, residual caps, confidence observations, and recency half-life from selector-opportunity diagnostics; R11 treats the frozen R10 handoff as validation input. R9/R10/R11 write `memory_opportunity_summary.csv` plus `memory_opportunity_decision_log.csv` for small runs, or a small `memory_opportunity_decision_log.csv` index plus partitioned row logs under `tables/memory_opportunity_decision_log/` for large R10/R11 runs, to show baseline-vs-memory score gaps, memory-objective gates, flow-belief correction deltas, route-flow and information-gain scores, calibrated alpha/regime/risk fields, shield status, and accepted/rejected switch reasons without producing a single GitHub-incompatible CSV. Episode summaries use `memory_changed_selection` only for accepted memory-shield switches inside the episode; baseline-vs-policy selection differences are logged separately so memory audits do not depend on that field for a different meaning. Final scoring is computed only from the final held-out rollout path: R9/R10/R11 launch score rewards front-wall mission completion, capped updraft-gain and lift-dwell evidence, and terminal total specific-energy reserve after front-wall success. Airborne time and generic net/gross energy drift remain audit-only. There is no hidden speed gate, no energy-loss hard failure, no PD/PID, no TVLQR, and no fan-layout-specific controller logic.

The repeated-launch outer-loop scheduler must not assume free real-time computation. R9/R10/R11 profile context construction, spatial flow-belief query, and compact-library selection after a cheap real-time compatibility shortlist keyed by transition entry class and current nearest local LQR speed bin, with nearest populated-bin fallback if the exact speed bin has no candidates. Every primitive decision is profiled against a preferred 20 ms controller-slot budget and a hard 0.100 s primitive-boundary budget while recording total library candidates, evaluated shortlist candidates, and skipped candidates. The required real-flight timing scope is `heavy_cluster` and `balanced_cluster`, which must satisfy the 0.100 s hard in-flight boundary budget; `light_cluster` and `super_light_cluster` are optional extended-library diagnostics where limited violations are reported but do not block real-flight use; `no_cluster_no_merge` is unrestricted offline comparison/stress evidence. The first real-flight experiment adopts `balanced_cluster` as the single active deployment tier because E01 real-flight-aligned validation favours broader transition diversity after the launch-rate, actuator-limit, and boundary-safety updates while remaining defensible for high-energy starts with bounded memory behaviour; the R11 E01 result is defensible rather than strict-pass and shows that starts below 5.0 m/s remain the dominant failure mode; `heavy_cluster` remains a compact fallback when runtime or library size becomes the limiting deployment concern. This choice is a deployment tradeoff and does not claim that one compact library dominates every speed bin, environment ladder, or repeated-launch policy. Environment/context caches are warmed before profiled primitive decisions. Step 0 is prepared from the approved launch-gate state during the fixed 0.040 s neutral handoff, and active command emission starts from the latest post-handoff state at the handoff boundary. Later steps use a prepared next-decision path so the next primitive is committed at the boundary; runtime command FIFO order is now old-to-new, while cross-primitive FIFO continuity remains a separate audit item. The timing boundary measures controller compute only: context, full spatial-belief query, and compact controller-row library selector computation; table flushing, full candidate-row expansion, and post-hoc diagnostic row construction are outside the flight-control boundary. The latest targeted C16 sanity check for this boundary used 40 final launches and 430 history launches, with 0 hard failures, 0 no-viable events, 13/13 accepted memory switches, and required heavy/balanced in-flight decisions at 144/144 under 0.100 s with max 0.0937 s; this is targeted diagnostic evidence only, not a full R10/R11 validation or broad memory-improvement claim.

R10 and R11 have separate roles. R10 is governor/spatial-memory tuning on one hard training distribution, `r10_l7_full_domain_randomisation_arena_wide_training`, with 50 outer cases per condition: four-fan geometry, active fan count 0/1/2/3/4, fan parameter uncertainty, arena-wide non-overlapping fan positions, and W3 plant/implementation perturbations. Because R10 is a randomised tuning stage rather than a final claim gate, its pass profile allows bounded claim-bearing hard failures and bounded floor/ceiling violations, both capped at 0.20, while still reporting the exact rates and using them to tune the governor conservatively. R10 also tunes mission/risk weights plus memory sensitivity, cost-benefit memory weight/cap/cost terms, shield margins, exploration thresholds, residual caps, confidence observations, and recency half-life through the same deterministic bounded rule update; its pass profile focuses on bounded safety and final no-viable reject rate while writing improvement diagnostics for the frozen R11 handoff. Arena-wide fan positions are rejection-sampled inside the tracker footprint with a 0.5 m safety radius around each fan, so fan-centre distances must be at least 1.0 m and safety circles do not overlap. Within each R10 outer case, fan number, fan positions, and plant/implementation are fixed across history and final launches; launch state varies on every throw, and only nominal +/-15 percent static fan power and width plus measured-scatter uncertainty-scale noise may vary between launches. R11 is held-out validation on an eight-block fidelity ladder with 20 outer cases per block, 160 total, paired by stratified local launch-start indices spread across the 0--49 launch grid and reused across L0--L7; this clean full R11 unit schedules 4,000 final held-out launches plus 34,400 history launches, and richer evidence should come from repeated seeds rather than a separate R11 smoke wrapper, which is retired: L0 dry-air fixed, L1 single-fan fixed nominal, L2 four-fan fixed nominal, L3 fan-parameter uncertainty, L4 local fan-position uncertainty, L5 active-fan-count uncertainty 0/1/2/3/4, L6 environment-only full uncertainty, and L7 full-domain arena-wide randomisation. R11 uses the same memory-episode rule: L0-L2 are fixed nominal environments; L3 keeps fan count/positions fixed while nominal static fan parameters and measured-scatter uncertainty vary; L4 chooses one locally shifted layout per outer case and holds it fixed; L5 chooses one active fan count per outer case and holds it fixed; L6 fixes one local layout and active fan count while nominal static fan parameters and measured-scatter uncertainty vary; L7 fixes one arena-wide layout, active fan count, and plant/implementation per outer case while nominal static fan parameters and measured-scatter uncertainty vary. R11 additionally writes a comparison-only, non-claim-bearing open-loop zero-command baseline for every final held-out case; the claim-bearing core comparison remains no memory versus recency-weighted spatial flow-belief memory histories h3, h10, and h30 with built-in cost-benefit spatial flow memory. h100 and standalone safe-explore policy variants are optional appendix ablations. Hardware readiness, real-flight transfer, mission success, autonomy, and memory-improvement claims require R11 and later real-flight evidence.

<!-- R9_LAUNCH_GATE_ALIGNMENT_END -->

## 1. Result folder contract

Generated control results normally live under:

```text
03_Control/05_Results/<NN_group>/<NN_case>/<run_id>/
```
Pipeline-level evidence roots may instead use:

```text
03_Control/05_Results/<evidence_family>/<stage>/<stage_run_id>/
```
For example, the active LQR evidence roots live directly under
`03_Control/05_Results`, with short stage folders such as
`R5_dense`, `R6_archived`, `R7_survival`, `R8_library_size_study`,
`R8_outcome`, `R9_test`, `R10_learn`, and `R11_validation`. The stage run folder is the atomic evidence unit.
It must contain the manifest and the compact analysis files needed to audit the
run. Primitive-controller variant registries, W2/W3 survival summaries, and post-W3 merge summaries are compact audit artefacts; raw dense partitions still belong under `tables/`.

Preferred result groups:

```text
00_smoke
01_latency
02_env_model
03_primitive
04_context_archive
05_outcome_model
06_policy_eval
07_real_flight
08_simreal
09_figures
12_reproducibility
99_misc
```

Historical generated result folders are local-only unless the user explicitly requests preservation. Approved evidence roots may be tracked when every file passes the 100 MB audit and path-length audit. During ordinary validation, keep `03_Control/05_Results/.gitkeep` as the only result placeholder unless an approved local evidence root is explicitly allowed.

Scratch and preflight roots are local only. They must not be pushed unless the user explicitly requests preservation.

Real-flight calibration data uses a separate short-path contract to avoid
Windows/Git path failures under long OneDrive repository roots:

```text
04_Flight_Test/05_Results/cal/<block_storage_id>/<session>/<case_storage_id>/v001/
04_Flight_Test/05_Results/cal/<block_storage_id>/<session>/<case_storage_id>/bad/i001/
```

The active block storage IDs are:

```text
neutral_30 -> n30
pulse_ladder_elevator_30 -> pe30
pulse_ladder_aileron_30 -> pa30
pulse_ladder_rudder_30 -> pr30
```

Full human-readable block IDs, case IDs, command values, calibration-profile
hashes, and physical-build/session labels belong in manifests and CSV rows, not
in deep folder names. Calibration files may be tracked only after the same
100 MB and path-length audit used for simulation evidence.

---

## 2. Naming rules

Use concise numbered lower-snake-case names.

Good examples:

```text
04_context_archive/01_w01_lqr_annular_gp/001
04_context_archive/02_w2_survival_annular_gp/001
04_context_archive/03_w3_survival_rand_annular_gp/001
04_context_archive/04_post_w3_cluster_merge/001
05_outcome_model/01_lqr_terminal_targets/001
06_policy_eval/01_terminal_episode_smoke/001
08_simreal/01_rf_replay/001
```

Use these abbreviations consistently:

```text
ctx   environment context
prim  primitive
pol   policy
ep    episode
rf    real flight
sr    sim-real replay
w0    dry air
w1    active annular-GP randomized training layer; Gaussian plume diagnostic-only
w2    GP-corrected annular-Gaussian survival layer
w3    randomised GP-corrected annular-Gaussian survival layer
w01   combined W0/W1 rich primitive-library generation
post_w3  post-W3 library-size cross-study
nom   nominal
rand  randomised
sum   summary
```

Avoid:

```text
codex
tmp
new
final
test_output
very_long_descriptive_file_names
```

Target limits:

```text
filename stem <= 64 characters
repository-relative path <= 140 characters
generated file <= 100 MB
```

These limits reduce Git and OneDrive path/file problems.

---

## 3. Atomic result contents

Evidence-generating folders should contain:

```text
manifests/
metrics/
reports/
tables/
figures/          optional
chunk_manifests/  dense runs only
```

Those subfolders must remain inside the run folder, not directly under `03_Control/05_Results`.

Required compact files for most evidence runs:

```text
manifests/run_manifest.json
reports/run_report.md
metrics/runtime_summary.csv
metrics/outcome_summary.csv
metrics/file_size_audit.csv
```

---

## 4. Dense runtime contract

A run is dense if any condition is true:

```text
planned rollout rows >= 4,000
planned candidate rows >= 5,000
expected runtime > 30 minutes
expected uncompressed table size > 250 MB
used for thesis evidence, primitive-controller variant evidence, W0/W1 dense generation, W2/W3 survival replay, post-W3 library-size cross-study, envelope maps, outcome models, or selector/governor reports
```

Dense runs must not use a single-process full-memory runner that builds the entire rollout table before writing.

Dense runners must support:

```text
--workers
--max-workers
--candidate-chunk-size or --chunk-size
--storage-format auto|parquet|csv_gz|csv
--compression-level
--resume
--repair-incomplete
--dry-run-schedule
--stop-after-chunks
--continue-on-chunk-failure
```

Default local dense-run policy:

```text
workers = 8
max_workers = 8
storage_format = auto, resolving to parquet if supported, otherwise csv_gz
compression_level = 1 for csv_gz
resume = true
```

If memory pressure occurs, reduce chunk size or worker count first. Do not fall back to a full-memory single-process runner.

---

## 5. Storage and 100 MB file-size contract

Every generated file should target:

```text
preferred size <= 75 MB
hard project limit <= 100 MB
```

A file above 100 MB is allowed only as explicitly approved local-only evidence. It must not be pushed unless the user approves another storage method.

Dense tables must be written as compressed partitions:

```text
tables/<table_name>/c00000.csv.gz
tables/<table_name>/c00001.csv.gz
```

or parquet if available.

Contextual archive partitions may include short audit tokens, for example
`tables/contextual_rows/c00012_W1_annular-gp-single.csv.gz`. Do not use nested
`run_*/context_id=*/environment_id=*/chunk_index=*/part-00000.*` paths for new
dense evidence; those paths are too long for reliable Git and Windows tooling.

Every partition must be recorded in a table manifest with:

```text
table_name
relative_path
storage_format
row_count
byte_count
columns
checksum_sha256
```

Root-level or `metrics/` CSV files should be compact summaries only. Do not write full rollout tables as uncompressed metrics CSVs for dense runs.

Every evidence run must write `metrics/file_size_audit.csv` with at least:

```text
relative_path
byte_count
size_mb
above_75mb
above_100mb
push_allowed
```

Archive, optional W2 diagnostic, R7/W3 survival, post-W3 library-size cross-study, outcome-model, and episode-smoke outputs must keep continuation-valid targets separate from episode-terminal-useful targets. Controlled x-y arena exits with positive floor and ceiling margin are retained as terminal episode evidence; they must not be relabelled as continuation success or hard failure. R5 W0/W1 outputs must not be compressed into a small retained shortlist before R7/W3.

---

## 6. Chunk progress and resume rules

Each dense-run chunk must record:

```text
chunk_index
chunk_count
status
row_count
storage_format
partition_paths
checksum_sha256
worker_count
start_time
end_time
failure_message if any
```

`--resume` skips complete chunks only when manifests and checksums agree.

`--repair-incomplete` may rewrite corrupt chunks. It must not overwrite a complete official run unless explicitly authorised.

---

## 7. Cleanup rules

Remove these whenever they appear:

```text
.pytest_cache/
__pycache__/
_tmp_*/
03_Control/05_Results/logs/
03_Control/05_Results/metrics/
03_Control/05_Results/__pycache__/
```

Do not delete numbered historical result folders unless the user asks for that specific evidence to be removed.

---

## 8. Documentation rules

Project documentation belongs under `docs/`. Repository ignore rules must not hide the whole `docs/` directory.

Private local instruction and context files should be ignored by precise path rather than by broad directory patterns.

Every new dense-run script must document:

```text
which chunked runtime is reused
which compressed writer is reused
where worker_count_decision is recorded
where resume/repair is implemented
where partition checksums are written
how the 100 MB file limit is enforced
why a new runner is needed rather than wrapping an old one
```

