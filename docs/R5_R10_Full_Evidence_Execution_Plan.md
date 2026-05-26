# R5–R10 Full Evidence Execution Plan

## Status and authority

This file is a hard project-control document for the Nausicaa glider control pipeline. It should be placed at:

```text
docs/R5_R10_Full_Evidence_Execution_Plan.md
```

CODEX must treat this document, `Glider_Control_Project_Plan.md`, `Daily_Schedule.txt`, `Skills.md`, `Python Coding Instruction.txt`, `Python Coding to CODEX.txt`, `MATLAB Coding.txt`, `housekeeping_and_naming_rules.md`, and `PR.txt` as the controlling sources for the next implementation pass.

If this document conflicts with older generated plans, old result roots, stale pipeline scripts, old bounded-PD dense runs, old clustering outputs, or old policy-evaluation outputs, this document controls.

The goal is not to create another prepare-only scaffold. The goal is to implement or repair the orchestrated evidence pipeline and then run the full staged simulation chain in the same CODEX execution pass, stopping only when a stage fails its gate.

---

## 1. Non-negotiable method constraints

The active method is:

```text
environment-conditioned primitive selection
with primitive-local time-invariant LQR
and viability-filtered safe exploration/exploitation
over repeated-launch directional 3D residual memory
```

CODEX must preserve:

```text
finite_horizon_s = 0.100
controller_input_slots_per_primitive = 5
controller_input_update_period_s = 0.020
primitive_timing_contract_version = v411_0p10s_5slot_20ms
```

The active evidence chain is:

```text
R5 W0/W1 dense LQR primitive-library generation
→ R6 W2 fixed-LQR annular-GP replay
→ R7 W3 fixed-LQR randomised annular-GP replay
→ R8 post-W3 library-size cross-study and outcome model
→ R9 fixed-case repeated-launch validation
→ R10 environment-only changed-case validation
```

The pipeline must not introduce or reintroduce:

```text
PD, PID, bounded-PD, hand-tuned fallback controller
TVLQR or time-varying LQR as the active workflow
new controller families
direct reinforcement learning
nonlinear MPC as the active method
fan-layout-specific controller logic
W-layer-specific online controller branches
reachable-state chaining as a required success gate
clustering before W3
W2 or W3 Q/R, K, reference, horizon, entry-role, controller-ID, or primitive-variant-ID mutation
stale v4.10/v4.9/v4.8 validation evidence as active evidence
full-memory dense runners
silent reduction of R9/R10 case counts
hardware-readiness claims before R9/R10 pass
```

LQR synthesis failure must be recorded as blocked evidence. It must not trigger fallback to PD/PID or any other controller family.

X/y terminal boundary evidence must be retained as terminal episode evidence when appropriate. It is not continuation-valid evidence, but it must not be deleted merely because a launch reaches a lateral or wall-related boundary.

---

## 2. Required CODEX implementation objective

CODEX must implement or repair:

```text
03_Control/04_Scenarios/run_r5_r10_pipeline.py
```

This runner is the strict staged evidence orchestrator.

The orchestrator must:

1. read and hash all controlling `/docs` files;
2. run the source audit;
3. inspect active/retired code boundaries;
4. run stage preflight checks;
5. start full R5 dense generation in the same execution pass after preflight passes;
6. continue automatically to R6 only if R5 passes;
7. continue automatically to R7 only if R6 passes;
8. continue automatically to R8 only if R7 passes;
9. continue automatically to R9 only if R8 passes;
10. continue automatically to R10 only if R9 passes;
11. terminate with a blocked report if any gate fails;
12. never weaken the protocol to continue.

This is not a prepare-only run. Dry schedules and smoke checks are allowed as preflight checks, but they cannot replace the full R5 dense run and cannot unblock R6.

---

## 3. Pipeline root and artefacts

The pipeline root must be:

```text
03_Control/05_Results/lqr_contextual_v1_0/r5_r10_pipeline/<pipeline_run_id>/
```

Minimum required pipeline artefacts:

```text
manifests/pipeline_manifest.json
manifests/stage_state.json
metrics/stage_summary.csv
metrics/decision_log.csv
metrics/preflight_checks.csv
metrics/post_stage_checks.csv
metrics/repair_log.csv
metrics/file_size_audit.csv
reports/pipeline_report.md
```

Each stage still writes to its own atomic evidence root:

```text
03_Control/05_Results/lqr_contextual_v1_0/w01_dense/<run_id>/
03_Control/05_Results/lqr_contextual_v1_0/w2_survival/<run_id>/
03_Control/05_Results/lqr_contextual_v1_0/w3_survival/<run_id>/
03_Control/05_Results/lqr_contextual_v1_0/post_w3_library_size_study/<run_id>/
03_Control/05_Results/lqr_contextual_v1_0/outcome_model/<run_id>/
03_Control/05_Results/lqr_contextual_v1_0/repeated_launch_validation/<run_id>/
03_Control/05_Results/lqr_contextual_v1_0/changed_case_validation/<run_id>/
```

Do not overwrite unrelated result roots. Use the next safe unused numeric run ID unless a run ID is explicitly supplied.

---

## 4. Pipeline CLI contract

The orchestrator must provide at least:

```text
--run-id
--start-stage
--stop-after-stage
--resume
--repair-incomplete
--workers
--max-workers
--storage-format
--compression-level
--candidate-chunk-size
--allow-stage-smoke
--dry-run-schedule
--continue-on-stage-failure false
--r10-mode full | reduced_diagnostic_50
```

Default policy:

```text
start-stage = R5
stop-after-stage = R10
resume = true
repair-incomplete = true
workers = 8
max-workers = 8
storage-format = auto
compression-level = 1
candidate-chunk-size = 800
allow-stage-smoke = false
continue-on-stage-failure = false
r10-mode = full
```

The pipeline may run dry schedules and smokes as preflight checks, but after preflight passes it must start full R5 unless explicitly invoked in dry-run mode by the user.

---

## 5. Stage decision model

Every stage must produce one or more decision-log rows:

```text
pipeline_run_id
stage_id
stage_name
stage_run_id
source_root
output_root
preflight_status
execution_status
postcheck_status
decision
blocked_reason
repair_attempted
repair_result
resume_used
worker_count
storage_format
largest_file_size_mb
claim_status
```

Allowed `decision` values:

```text
continue
repair_and_rerun
terminate_blocked
```

Permitted automatic repairs:

```text
regenerate compact reports from valid tables
regenerate file-size audit
regenerate table manifest if partitions and checksums are intact
repair incomplete chunk manifests if partition files are present and valid
resume incomplete stage from existing valid chunks
```

Forbidden repairs:

```text
using smoke/preflight roots as method evidence
using old v4.10/v4.9/v4.8 full-loop roots as active evidence
using fixture roots as method evidence
changing controller gains after W0/W1
retuning in W2/W3
changing primitive horizon or slot count to make a run pass
dropping failed rows
deleting x/y terminal evidence
falling back to simpler physics or old controller logic
reducing R9/R10 target counts without diagnostic label
```

---

## 6. R5 — full W0/W1 dense LQR primitive-library generation

### Purpose

R5 generates the rich primitive-controller variant library. This is the first full evidence stage after the repair cycle.

### Required R5 scale

```text
primitive_count = 8
candidate_count = 32
W0/W1 environment cases = 3
paired_tests_per_candidate = 100
target_rows = 8 * 32 * 3 * 100 = 76,800
```

Environment cases:

```text
W0 dry_air
W1 gaussian_single
W1 gaussian_four
```

R5 must randomise launch and in-flight primitive-start states according to the active mixed start distribution. It must preserve every solved, blocked, weak, failed, rejected, continuation-valid, and x/y terminal-useful row.

### R5 execution command pattern

The orchestrator should call the Python API directly where practical, but the equivalent command is:

```powershell
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_w01_dense_chunked.py `
  --run-id <next_w01_run_id> `
  --rows 76800 `
  --candidate-count 32 `
  --paired-tests-per-candidate 100 `
  --candidate-chunk-size 800 `
  --workers 8 `
  --max-workers 8 `
  --storage-format auto `
  --compression-level 1 `
  --resume `
  --repair-incomplete
```

### R5 pass gate

R5 passes only if the R5 manifest and compact checks show:

```text
method_evidence_level = w01_dense_evidence_complete
w01_dense_evidence_complete = true
actual_row_count >= 76800
candidate_count >= 32
paired_tests_per_candidate >= 100
primitive_family_coverage_complete = true
start_family_mix_complete = true
cross_layer_smoke_status = cross_layer_smoke_start_family_complete
finite_horizon_s = 0.100 for all active variants
controller_input_slots_per_primitive = 5 for all active variants
controller_input_update_period_s = 0.020 for all active variants
primitive_timing_contract_status = compliant
frozen_w01_controller_bundle exists
ready frozen controller count > 0
table_manifest exists
chunk manifests complete
file_size_audit passes
no generated file > 100 MB
no PD/PID/bounded fallback used
simulation-only claim boundary preserved
```

If R5 does not pass, the pipeline must stop before R6.

---

## 7. R6 — rich-side W2 fixed-LQR annular-GP replay

### Purpose

R6 replays the fixed R5 W0/W1 primitive-controller variants under W2 GP-corrected annular-Gaussian conditions. It is a rich-side dense survival stage, not a smoke, fixture, diagnostic, or fallback run.

### Required R6 source

R6 may consume only the accepted R5 root whose manifest says:

```text
method_evidence_level = w01_dense_evidence_complete
w01_dense_evidence_complete = true
```

R6 must reject:

```text
smoke roots
preflight roots
dry schedules
old W01 roots
diagnostic_not_passed roots
fixture roots
roots missing the timing contract
```

### Required R6 scale

Use the rich R5 library. For the planned 256 R5 primitive-controller variants:

```text
W2 environment cases = 2
paired_tests_per_variant = 100
target_rows = 256 * 2 * 100 = 51,200
```

Environment cases:

```text
annular_gp_single
annular_gp_four
```

### R6 execution command pattern

```powershell
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_w2_survival.py `
  --run-id <next_w2_run_id> `
  --input-root <passed_r5_w01_root> `
  --paired-tests-per-variant 100 `
  --candidate-chunk-size 800 `
  --workers 8 `
  --max-workers 8 `
  --storage-format auto `
  --compression-level 1 `
  --resume `
  --repair-incomplete
```

### R6 pass gate

R6 passes only if:

```text
w2_dense_survival_evidence_complete = true
status in {w2_dense_survival_pass, survived_variants_available}
survivor_count > 0
w2_survivor_registry exists
fixed_lqr_replay_only = true
no Q/R mutation
no K mutation
no reference mutation
no horizon mutation
no entry-role mutation
no controller-ID mutation
no primitive-variant-ID mutation
timing fields compliant
downgraded/eliminated/blocked rows retained
x/y terminal boundary evidence retained where applicable
file_size_audit passes
```

If R6 does not pass, the pipeline must stop before R7.

---

## 8. R7 — rich-side W3 fixed-LQR randomised annular-GP replay

### Purpose

R7 replays accepted W2 survivors under W3 randomised annular-GP conditions. It is a rich-side dense survival stage, not a smoke, fixture, diagnostic, or fallback run.

### Required R7 source

R7 may consume only the accepted R6 root whose manifest and survivor registry prove:

```text
w2_dense_survival_evidence_complete = true
survivor_count > 0
fixed_lqr_replay_only = true
```

R7 must reject:

```text
W2 smoke roots
W2 fixture roots
old W2 roots
diagnostic roots
roots missing timing compliance
roots without real survivors
```

### Required R7 scale

Use all accepted W2 survivors.

Minimum rich-side target:

```text
W3 environment cases = 2
paired_tests_per_variant >= 20
target_rows = W2_survivor_count * 2 * 20
```

If runtime allows, CODEX may use a stronger R7 run with:

```text
paired_tests_per_variant = 50 or 100
```

but the chosen value must be recorded in the manifest and decision log.

Environment cases:

```text
w3_randomised_single
w3_randomised_four
```

W3 randomisation may include:

```text
updraft amplitude
updraft width
updraft centre
fan position
fan number
fan power or active fan subset
glider model parameters
feedback latency
command timing
actuator delay
```

but W3 still must not retune LQR controllers.

### R7 pass gate

R7 passes only if:

```text
w3_dense_survival_evidence_complete = true
w3_survivor_registry exists
survivor_count > 0
all survivors timing compliant
fixed_lqr_replay_only = true
no Q/R, K, reference, horizon, entry role, controller ID, or primitive variant ID mutation
randomisation manifest exists
downgraded/eliminated/blocked rows retained
x/y terminal boundary evidence retained where applicable
file_size_audit passes
```

If R7 does not pass, the pipeline must stop before R8.

---

## 9. R8 — post-W3 four-case library-size cross-study and outcome model

### Purpose

R8 measures the effect of reducing the post-W3 library. It must not assume a single final compact-library size.

### Required R8 source

R8 may consume only the accepted R7 W3 survivor root.

### Required R8 cases

Machine IDs:

```text
heavy_cluster
balanced_cluster
light_cluster
no_cluster_no_merge
```

Human labels:

```text
heavy clustering and merging
balanced clustering and merging
light clustering and merging
no-clustering/no-merging
```

`no_cluster_no_merge` must include all W3 survivors directly, with no clustering or merging.

### R8 execution pattern

```powershell
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_post_w3_library_size_study.py `
  --run-id <next_r8_run_id> `
  --input-root <passed_r7_w3_root>
```

Then:

```powershell
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_outcome_model_build.py `
  --run-id <next_outcome_run_id> `
  --compact-library <r8_root>/manifests/post_w3_library_size_study_manifest.json
```

### R8 pass gate

R8 passes only if:

```text
all four library-size case manifests exist
all four representative tables exist
library_size_case_summary.csv exists
post_w3_representative_library_all_cases.csv exists
no_cluster_no_merge representative_count == W3 survivor_count
all representatives trace source W01, W2, and W3 roots
all representatives retain timing fields
no Q/R, K, reference, horizon, entry role, controller ID, or primitive variant ID mutation
outcome_model_table exists
outcome model covers all four case IDs
outcome rows retain timing fields
file_size_audit passes
```

If R8 does not pass, the pipeline must stop before R9.

---

## 10. R9 — full fixed-case repeated-launch validation

### Purpose

R9 tests library-size cases, outcome model, viability governor, selector, directional 3D residual memory, and safe exploration/exploitation under controlled fixed-case repeated-launch simulation.

R9 must be true repeated-launch rollout validation. It must not be selector-only scoring over an outcome table.

### Required R9 library cases

All four R8 cases:

```text
heavy_cluster
balanced_cluster
light_cluster
no_cluster_no_merge
```

### Required R9 policy/history conditions

There are 14 policy/history conditions:

```text
no_memory_baseline
static_map_baseline

directional_3d_residual_memory_h0
directional_3d_residual_memory_h5
directional_3d_residual_memory_h10
directional_3d_residual_memory_h20
directional_3d_residual_memory_h50
directional_3d_residual_memory_h100

safe_explore_then_exploit_h0
safe_explore_then_exploit_h5
safe_explore_then_exploit_h10
safe_explore_then_exploit_h20
safe_explore_then_exploit_h50
safe_explore_then_exploit_h100
```

The safe-explore group is a separate ablation. Exploration-change metrics must also be logged for every selection where exploration can affect ranking.

### Required R9 outer cases

Use exactly:

```text
60 outer test cases per policy/history condition per library-size case
```

Split per condition and library case:

```text
20 no-updraft cases
20 single-fan cases
20 four-fan cases
```

### Required R9 counts

Final held-out launches:

```text
4 library cases * 14 policy/history conditions * 60 outer cases = 3,360
```

History lengths for memory family:

```text
0 + 5 + 10 + 20 + 50 + 100 = 185
```

History lengths for safe-explore family:

```text
0 + 5 + 10 + 20 + 50 + 100 = 185
```

History launches:

```text
4 library cases * 60 outer cases * (185 + 185) = 88,800
```

Approximate total R9 launch episodes:

```text
3,360 final held-out launches + 88,800 history launches = 92,160
```

The orchestrator must verify these expected counts unless a run is explicitly marked diagnostic. No reduced R9 target is allowed to pass.

### R9 controlled comparison rule

For each outer test case:

```text
the final evaluation launch must use the same launch point
and the same environment instance
across every policy/history condition
and every library-size case
```

For policies using history launches:

```text
history launches may be randomised within the allowed launch distribution
history seeds should be paired across comparable policies where possible
the no-memory and static-map baselines must face the same final held-out launch as memory policies
```

### R9 fixed-case rule

Inside each R9 outer test case, hold fixed:

```text
environment instance
glider plant instance
latency/actuator instance
controller library
policy settings
final held-out launch comparison protocol
```

### R9 required outputs

```text
manifests/repeated_launch_fixed_case_manifest.json
metrics/outer_case_schedule.csv
metrics/history_launch_schedule.csv
metrics/final_heldout_launch_schedule.csv
metrics/episode_summary.csv
metrics/primitive_execution_log.csv
metrics/candidate_score_log.csv
metrics/selector_decision_log.csv
metrics/memory_residual_update_log.csv
metrics/belief_snapshot_log.csv
metrics/library_size_case_comparison.csv
metrics/policy_history_comparison.csv
metrics/termination_summary.csv
metrics/pass_fail_gate_summary.csv
metrics/pairing_audit.csv
metrics/file_size_audit.csv
reports/repeated_launch_fixed_case_report.md
```

### R9 required metrics

At minimum:

```text
safe_success_rate
hard_failure_rate
floor_or_ceiling_violation_rate
no_viable_primitive_rate
terminal_useful_rate
lift_capture_rate
mean_lift_dwell_time_s
mean_energy_residual_m
mean_min_wall_margin_m
selected_primitive_family_count
selected_variant_count
memory_changed_selection_rate
exploration_changed_selection_rate
governor_rejection_count
belief_observation_count
belief_uncertainty
history_length
library_size_case_id
policy_id
outer_case_type
```

### R9 pass gate

R9 passes only if:

```text
all four library-size cases present
all 14 policy/history conditions present
all 60 outer cases per condition per library case present
20/20/20 no-updraft/single-fan/four-fan split present
final held-out launch pairing audit passes
hard_failure_rate <= 1%
floor_or_ceiling_violation_rate == 0
no_viable_primitive_rate <= 2%
safe_success_rate near 100% inside declared envelope
terminal_useful_rate or lift_capture_rate >= 90% in declared fixed-case validation
selected_primitive_family_count >= 5 across validation
selected_variant_count >= 10 across validation
memory_changed_selection_rate > 0 during early histories
exploration_changed_selection_rate > 0 for safe-explore policies
file_size_audit passes
no hardware-readiness or memory-improvement claim is made unless all gates pass
```

If R9 does not pass, the pipeline must stop before R10.

---

## 11. R10 — full environment-only changed-case validation

### Purpose

R10 tests environment-only changes after R9 fixed-case validation passes. It must not vary glider, latency, actuator lag, surface calibration, mass, CG, or inertia inside R10.

### Required R10 source

R10 may run only if R9 passes.

### Required R10 library cases and policies

Use the same four library-size cases and the same 14 policy/history conditions as R9.

### Required R10 outer cases

Use exactly:

```text
100 outer test cases per policy/history condition per library-size case
```

Split into five blocks:

```text
20 nominal single-fan perturbation cases
20 nominal four-fan perturbation cases
20 shifted single-fan-position cases
20 shifted four-fan-position cases
20 active-fan-number variation cases
```

### R10 allowed environment-only changes

Allowed:

```text
updraft amplitude
updraft width
updraft centre
residual uncertainty
fan position within physically reasonable limits
active fan number / active fan subset
```

Not allowed inside R10:

```text
glider mass variation
CG variation
inertia variation
surface calibration variation
latency variation
actuator lag variation
controller-library change
policy-setting change
launch comparison protocol change
```

Those variations belong to W3 robustness replay or a later explicitly approved study, not R10.

### Required R10 counts

Final held-out launches:

```text
4 library cases * 14 policy/history conditions * 100 outer cases = 5,600
```

History launches:

```text
4 library cases * 100 outer cases * (185 + 185) = 148,000
```

Approximate total R10 launch episodes:

```text
5,600 final held-out launches + 148,000 history launches = 153,600
```

The orchestrator must verify these expected counts for full R10.

### R10 reduced diagnostic mode

A reduced mode may exist only as:

```text
R10_reduced_50
```

with label:

```text
reduced_diagnostic_not_target_R10
```

Reduced mode structure:

```text
10 cases per environment block
50 outer cases total
```

Reduced mode cannot satisfy the R10 pass gate and cannot support a full R10 success claim.

### R10 controlled comparison rule

For each outer test case, all policies, histories, and library-size cases must share:

```text
same final held-out launch point
same final environment instance
same glider model
same latency model
same actuator model
same controller library
same policy settings
same comparison protocol
paired random seeds
```

Only history launches may differ. History launches must remain within the allowed launch distribution and use paired seeds where possible.

### R10 required outputs

```text
manifests/environment_changed_case_manifest.json
metrics/environment_block_schedule.csv
metrics/outer_case_schedule.csv
metrics/history_launch_schedule.csv
metrics/final_heldout_launch_schedule.csv
metrics/episode_summary.csv
metrics/primitive_execution_log.csv
metrics/candidate_score_log.csv
metrics/selector_decision_log.csv
metrics/memory_residual_update_log.csv
metrics/belief_snapshot_log.csv
metrics/library_size_case_comparison.csv
metrics/policy_history_comparison.csv
metrics/environment_block_comparison.csv
metrics/termination_summary.csv
metrics/pass_fail_gate_summary.csv
metrics/pairing_audit.csv
metrics/no_glider_latency_variation_audit.csv
metrics/file_size_audit.csv
reports/environment_changed_case_report.md
```

### R10 pass gate

R10 passes only if:

```text
R9 passed first
all four library-size cases present
all 14 policy/history conditions present
all 100 outer cases per condition per library case present
all five 20-case environment blocks present
controlled final-launch pairing audit passes
no glider/latency/actuator/surface/mass/CG/inertia variation audit passes
hard_failure_rate <= 1%
floor_or_ceiling_violation_rate == 0
no_viable_primitive_rate <= 2%
safe_success_rate near 100% inside declared envelope
terminal_useful_rate or lift_capture_rate >= 90% in declared changed-case validation
selected_primitive_family_count >= 5 across validation
selected_variant_count >= 10 across validation
memory_changed_selection_rate > 0 during early histories
exploration_changed_selection_rate > 0 for safe-explore policies
no major degradation outside declared tolerance under environment-only changes
file_size_audit passes
```

Only after R9 and R10 pass may the project discuss hardware-readiness preparation. Real-flight transfer claims still require matched real logs and sim-real replay.

---

## 12. Runtime, storage, and robustness requirements

Every dense or validation-scale stage must be:

```text
chunked
resumable
compressed
worker-enabled
checksum-manifested
file-size audited
```

Use:

```text
workers <= 8
max_workers <= 8
compression_level = 1
storage_format = auto unless explicitly overridden
```

No generated file may exceed:

```text
100 MB
```

Prefer:

```text
75 MB or below
```

Dense table data must be partitioned under:

```text
tables/
```

Compact summaries go under:

```text
metrics/
```

Chunk-level manifests go under:

```text
chunk_manifests/
```

A full-memory final rollout table is not allowed.

---

## 13. Tests required before and after implementation

CODEX must add or repair tests for:

```text
docs gate reads all required control files
source audit ready state
active primitive timing contract
all active primitives use 0.100 s horizon
all active primitives use 5 slots and 20 ms update period
no active PD/PID/bounded fallback
no active TVLQR workflow
directional 3D residual memory blocks missing canonical coordinates
safe exploration applied only after viability filtering
R5 dense gate rejects smoke/preflight roots
R5 dense gate accepts only w01_dense_evidence_complete roots
R6 consumes only rich dense R5 roots
R7 consumes only rich dense W2 survivor roots
R8 contains all four library-size cases
R8 no_cluster_no_merge uses all W3 survivors
outcome model covers all four library-size cases
R9 exact matrix dimensions: 4 * 14 * 60 final launches
R9 exact history-launch count: 4 * 60 * 370
R10 exact matrix dimensions: 4 * 14 * 100 final launches
R10 exact history-launch count: 4 * 100 * 370
R10 contains all five 20-case environment blocks
R9/R10 final held-out launch pairing is enforced
R10 no glider/latency/actuator/surface/mass/CG/inertia variation audit passes
file-size audit blocks files above 100 MB
pipeline terminates on failed gates
pipeline writes blocked report with exact reason
```

Minimum command set:

```powershell
.\.venv\Scripts\python.exe -m py_compile 03_Control\03_Primitives\*.py 03_Control\04_Scenarios\*.py
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v411_source_audit.py
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_r5_r10_pipeline.py --run-id <next_pipeline_id> --start-stage R5 --stop-after-stage R10 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --allow-stage-smoke false
```

The final command above is the full staged execution command, not a prepare-only command.

---

## 14. Required final CODEX report

At the end of the CODEX run, report:

```text
docs read and hashes
latest repo title/version used
source audit result
pipeline root
stage roots attempted
R5 status and row count
R6 status and survivor count if reached
R7 status and survivor count if reached
R8 library-size case counts if reached
R9 matrix counts and pass/fail summary if reached
R10 matrix counts and pass/fail summary if reached
largest generated file size
worker settings
storage format
resume/repair status
blocked stage, if any
blocked reason, if any
next exact command, if any
claims explicitly not made
```

If the pipeline stops early, explain which gate stopped it and why. Do not describe the run as successful unless the relevant stage gates actually passed.

---

## 15. Claim boundaries

Allowed after implementation if true:

```text
R5–R10 orchestrator implemented
source audit passed
R5 full dense started
R5 full dense completed
R6 reached only after R5 pass
R7 reached only after R6 pass
R8 reached only after R7 pass
R9 reached only after R8 pass
R10 reached only after R9 pass
simulation-only evidence
```

Blocked unless actually supported:

```text
W2 survival complete
W3 robustness complete
post-W3 library-size case accepted
repeated-launch memory improvement
environment-only changed-case robustness
hardware readiness
real-flight transfer
mission success
full autonomy
formal ROA/funnel guarantee
```

Do not claim hardware readiness, real-flight transfer, mission success, full autonomy, or memory improvement unless full R9/R10 gates pass and the claim is explicitly supported by the generated evidence.

---

## 16. Short instruction to CODEX

Read `docs/R5_R10_Full_Evidence_Execution_Plan.md` before editing. Implement or repair the pipeline exactly according to it. This is not a prepare-only task. Run the full staged evidence chain in the same execution pass, with R6–R10 gated by the previous stage. Stop only when a gate fails or R10 completes. Never weaken the protocol to continue.
