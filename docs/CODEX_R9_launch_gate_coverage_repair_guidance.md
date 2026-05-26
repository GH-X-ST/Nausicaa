# CODEX Guidance — R9 Launch-Gate Coverage and Outcome-Lookup Repair

## Purpose

Repair the current Nausicaa R9 repeated-launch validation blocker without changing the project direction.

The current pipeline successfully reached R9 after R5–R8, then terminated because R9 failed its pass gate. The failure is not acceptable as a move-on result. The likely physical cause is inadequate launch-gate primitive diversity, and there is also a concrete R9 outcome-lookup bug that makes compressed library-size cases appear worse than they are.

This repair must preserve the active project plan:

```text
primitive-local time-invariant LQR
0.100 s primitive horizon
5 controller-input slots
20 ms controller update period
directional 3D residual memory
safe exploration/exploitation only after viability filtering
W0/W1 rich dense generation
W2/W3 fixed-LQR no-retune replay
post-W3 four-case library-size cross-study
R9/R10 repeated-launch validation
```

Do not introduce PD/PID, bounded feedback fallback, TVLQR, new controller families, fan-layout-specific controller logic, W2/W3 retuning, early clustering, stale v4.10 evidence, or a different research direction.

## Controlling references

Before editing, CODEX must read and hash:

```text
docs/R5_R10_Full_Evidence_Execution_Plan.md
docs/Glider_Control_Project_Plan.md
docs/Daily_Schedule.txt
docs/Skills.md
docs/Python Coding Instruction.txt
docs/Python Coding to CODEX.txt
docs/MATLAB Coding.txt
docs/housekeeping_and_naming_rules.md
docs/PR.txt
```

The repeated docs-alignment guard in `run_r5_r10_pipeline.py` must remain active. If any controlling doc changes, stop with `docs_changed_reaudit_required`.

## Current evidence summary

The latest pipeline reached R9 and stopped:

```text
R5 passed: w01_dense/019
R6 passed: w2_survival/016
R7 passed: w3_survival/014
R8 passed: post_w3_library_size_study/001 and outcome_model/003
R9 blocked: repeated_launch_validation/001 failed pass_gate
R10 not run
```

R9 quantitative structure was correct:

```text
final_heldout_launch_count = 3360 / 3360
history_launch_count = 88800 / 88800
library_size_case_count = 4 / 4
policy_history_condition_count = 14 / 14
pairing_audit = passed
```

But R9 failed the actual control gates:

```text
hard_failure_rate = 0.0833 > 0.01
no_viable_primitive_rate = 0.8042 > 0.02
safe_success_rate = 0.1125 < 0.99
terminal_or_lift_capture = 0.0583 < 0.90
selected_primitive_family_count = 2 < 5
selected_variant_count = 2 < 10
```

## Diagnosis

### A. Confirmed physical/library coverage problem

The current W0/W1 entry-role map has only one launch-capable family:

```text
glide -> launch_capable
energy_retaining_bank -> inflight_only
lift_entry -> inflight_only
lift_dwell_arc -> inflight_only
mild_turn_left -> inflight_only
mild_turn_right -> inflight_only
recovery -> terminal_or_recovery
safe_exit_or_recovery_handoff -> terminal_or_recovery
```

For a launch-gate start, all `inflight_only` and `terminal_or_recovery` candidates are entry-role incompatible. W01 did sample many launch-gate rows:

```text
launch_gate rows = 30,720
```

but only 3,840 of those launch-gate rows belong to the single launch-capable `glide` family. The remaining launch-gate rows are structurally rejected by entry role:

```text
inflight_only at launch_gate: 19,200 / 19,200 rejected
terminal_or_recovery at launch_gate: 7,680 / 7,680 rejected
launch_capable at launch_gate: 0 / 3,840 rejected
```

Therefore, at the first launch decision the governor has only glide-family candidates available by entry role:

```text
heavy_cluster: 8 total reps, 1 launch-compatible glide candidate
balanced_cluster: 24 total reps, 3 launch-compatible glide candidates
light_cluster: 48 total reps, 6 launch-compatible glide candidates
no_cluster_no_merge: 256 total reps, 32 launch-compatible glide candidates
```

This is insufficient for real exploration/exploitation immediately after launch. Memory and exploration cannot choose between lift-entry, lift-dwell, energy-retaining, or turn behaviours if those behaviours are all classified as `inflight_only` and the validator starts every launch at `launch_gate`.

### B. Concrete R9 compressed-library outcome lookup bug

`run_repeated_launch_learning_curve.py` currently loads outcome rows with a dictionary keyed only by `primitive_variant_id`:

```python
return {str(row["primitive_variant_id"]): row for row in frame.to_dict(orient="records")}
```

But the same primitive variant appears in multiple library-size cases with different `compact_library_id` and `library_size_case_id`. As a result, later rows overwrite earlier rows. Then the R9 code filters by `library_size_case_id`, which can leave `case_outcomes` empty for heavy/balanced/light cases.

Observed symptom:

```text
heavy_cluster no_viable_primitive_rate = 1.0
balanced_cluster no_viable_primitive_rate = 1.0
light_cluster no_viable_primitive_rate = 1.0
no_cluster_no_merge no_viable_primitive_rate = 0.2167
```

This is not purely a physical failure; it is also an outcome-table indexing bug.

### C. R9 currently tests only one primitive per launch

The validator has `max_primitives_per_launch`, but `_run_one_launch` currently performs a single primitive selection and rollout. The result is dominated by the first launch-gate decision. This makes launch-capture coverage essential. If the first primitive family is only `glide`, the validator cannot transition into richer in-flight primitive families within the same launch.

### D. Memory/exploration metrics are not properly computed

The compact comparison table currently forces:

```text
memory_changed_selection_rate = 0.0
exploration_changed_selection_rate = 0.0
```

rather than computing those values from selected variants or candidate rank changes. This makes memory/exploration pass/fail evidence unreliable even after the launch coverage problem is fixed.

## Required repair strategy

Implement a project-aligned launch-capture repair.

The goal is not to distort existing in-flight primitive families. Keep the existing in-flight families intact. Add a separate launch-gate-capable subset that handles the first transition from launch into usable flight.

Preferred implementation:

```text
Add explicit launch_capture primitive family or launch-gate entry-role subset.
```

Acceptable variants:

```text
launch_capture_glide_stabilise
launch_capture_lift_seek
launch_capture_energy_build
launch_capture_shallow_left
launch_capture_shallow_right
launch_capture_safe_handoff
```

Keep names concise and project-consistent. The exact number may be adjusted, but the launch-capture set must be rich enough that R9 first decision can select, explore, and exploit among several viable launch-capable behaviours.

Each launch-capture primitive must:

```text
use primitive-local time-invariant LQR
use finite_horizon_s = 0.100
use controller_input_slots_per_primitive = 5
use controller_input_update_period_s = 0.020
have entry_role = launch_capable
have stable primitive_id and primitive_variant_id
have LQR Q/R/K evidence from W0/W1
have exit checks and failure labels
avoid fan-layout-specific logic
```

Do not relabel all existing in-flight primitives as launch-capable just to bypass the gate. Only create physically meaningful launch-capture variants.

## Files likely to change

Inspect and update as needed:

```text
03_Control/03_Primitives/prim_cat.py
03_Control/03_Primitives/primitive_variant_registry.py
03_Control/03_Primitives/lqr_controller.py
03_Control/03_Primitives/prim_roll.py
03_Control/04_Scenarios/run_lqr_w01_dense_chunked.py
03_Control/04_Scenarios/run_w2_survival.py
03_Control/04_Scenarios/run_w3_survival.py
03_Control/04_Scenarios/run_post_w3_library_size_study.py
03_Control/04_Scenarios/run_outcome_model_build.py
03_Control/04_Scenarios/run_repeated_launch_learning_curve.py
03_Control/04_Scenarios/run_changed_case_validation.py
03_Control/04_Scenarios/run_r5_r10_pipeline.py
03_Control/tests/
```

Do not edit unrelated hardware code, thesis text, or plotting code unless needed for tests or manifest labels.

## Required code fixes

### 1. Add launch-capture primitive family or subset

Add launch-capture primitive definitions with `entry_role = launch_capable`.

Update:

```text
ACTIVE_PRIMITIVE_IDS
ENTRY_ROLE_BY_PRIMITIVE_ID
tests for primitive catalogue and entry roles
```

Expected move-on check:

```text
launch_capable primitive family count >= 2
launch_capable variant count after W0/W1 >= 64 preferred
first-decision launch-compatible candidate count:
    heavy_cluster >= 2
    balanced_cluster >= 6
    light_cluster >= 12
    no_cluster_no_merge >= 64 preferred if two launch families * 32 candidates
```

If a smaller launch-capture set is chosen, justify it in the manifest and ensure it still passes diversity gates.

### 2. Fix outcome lookup to be library-case-aware

Do not key outcome rows only by `primitive_variant_id`.

Use a composite key such as:

```text
(library_size_case_id, compact_library_id)
```

or:

```text
(library_size_case_id, primitive_variant_id, compact_library_id)
```

Representatives should also be matched using `compact_library_id`, not only `primitive_variant_id`, because the same variant may appear in multiple library-size cases.

Required test:

```text
heavy_cluster, balanced_cluster, light_cluster, and no_cluster_no_merge each produce non-empty case_outcomes.
```

### 3. Add launch-gate viability audit outputs

For W01, W2, W3, R8, and R9, add compact metrics:

```text
launch_gate_entry_role_audit.csv
launch_gate_outcome_audit.csv
launch_gate_candidate_availability.csv
first_decision_candidate_summary.csv
first_decision_governor_rejection_summary.csv
```

Minimum columns:

```text
stage_id
library_size_case_id
policy_id
history_length
outer_case_type
primitive_id
primitive_family
entry_role
start_state_family
candidate_count
entry_compatible_count
viable_count
selected_count
rejection_reason
row_count
accepted_count
weak_count
failed_count
rejected_count
blocked_count
continuation_valid_count
episode_terminal_useful_count
hard_failure_count
```

### 4. Compute memory/exploration selection-change metrics

Replace hardcoded zero values with real metrics.

For each matched final launch:

```text
memory_changed_selection = selected_variant(memory policy) != selected_variant(no_memory_baseline)
exploration_changed_selection = selected_variant(safe_explore policy) != selected_variant(direction_memory policy at same history)
```

Also compute rank-based diagnostics from candidate logs:

```text
rank_change_due_to_memory
rank_change_due_to_exploration
memory_score_component
exploration_score_component
score_margin_to_selected
```

### 5. Decide whether R9 should allow short multi-primitive launch rollout

The current R9 validator executes only one primitive per launch. This is useful for isolating first-decision coverage, but it is too restrictive for full repeated-launch evaluation if the launch-capture primitive is supposed to hand off to in-flight primitives.

Add an explicit configuration:

```text
max_primitives_per_launch
```

and support at least:

```text
R9 first-decision audit mode: max_primitives_per_launch = 1
R9 full validation mode: max_primitives_per_launch >= 4
```

The full pass gate should use the full validation mode. The one-step audit should be kept as a diagnostic table.

Do not let multi-primitive rollout hide launch-gate failure. First-decision candidate availability must still pass.

## Required result regeneration

After code repair, regenerate from the correct point.

Because the primitive catalogue and entry roles change, the correct regeneration point is R5:

```text
R5 new W0/W1 dense run
→ R6 W2 fixed replay
→ R7 W3 fixed replay
→ R8 library-size study and outcome model
→ R9 repeated-launch validation
→ R10 only if R9 passes
```

Do not patch R9 using old W01/W2/W3/R8 roots after adding launch-capture primitives. That would mix libraries and invalidate the evidence chain.

## Required stage gates after repair

### R5 launch-capture gates

```text
w01_dense_evidence_complete = true
launch_gate rows present for every launch-capable family
launch_capture family rows present
all launch-capture variants timing compliant
no PD/PID fallback
entry_role_rejection_summary shows launch_capture rows not rejected at launch_gate
launch_gate_outcome_audit written
```

### R8 library-size gates

```text
all four library-size cases present
each case retains at least one launch-capture representative
no_cluster_no_merge retains all W3 survivors
outcome rows keyed by compact_library_id and library_size_case_id
```

### R9 gates

Existing gates still apply, plus:

```text
first_decision_candidate_summary present
first_decision viable_count > 0 for >= 98% of final held-out launches
launch-compatible primitive family count >= 2 in every library-size case
selected launch primitive family count >= 2
memory/exploration metrics computed from actual matched selections
```

The original pass gates remain:

```text
hard_failure_rate <= 1%
floor_or_ceiling_violation_rate == 0
no_viable_primitive_rate <= 2%
safe_success_rate >= 99% inside declared envelope
terminal_useful or lift_capture >= 90%
selected_primitive_family_count >= 5
selected_variant_count >= 10
```

## Tests to add or update

Add tests for:

```text
launch_capture primitives exist
launch_capture primitives are launch_capable
all launch_capture primitives use 0.100 s / 5-slot / 20 ms timing
existing in-flight primitives remain inflight_only
terminal/recovery primitives remain terminal_or_recovery
W01 launch_gate coverage includes multiple launch-capable families
entry_role_rejection_summary rejects inflight_only at launch_gate but not launch_capture
outcome lookup distinguishes duplicated primitive_variant_id across library-size cases
R9 heavy/balanced/light case_outcomes are non-empty
first_decision_candidate_summary counts entry-compatible and viable candidates
memory_changed_selection_rate is not hardcoded
exploration_changed_selection_rate is not hardcoded
max_primitives_per_launch loop works and logs each primitive_step_index
pipeline restarts from R5 after primitive catalogue change
```

Minimum command set:

```powershell
.\.venv\Scripts\python.exe -m py_compile 03_Control\03_Primitives\*.py 03_Control\04_Scenarios\*.py
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v411_source_audit.py
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_r5_r10_pipeline.py --run-id <next> --start-stage R5 --stop-after-stage R10 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --allow-stage-smoke false
```

## Claim boundary

Do not claim:

```text
R9 pass
R10 pass
memory improvement
hardware readiness
real-flight transfer
mission success
```

until the regenerated pipeline passes the relevant gates.

Allowed after implementation but before regenerated pass:

```text
launch-gate coverage repair implemented
R9 outcome-lookup bug repaired
new W0/W1 dense rerun started or completed
simulation-only evidence
```

## Final CODEX report

Report:

```text
whether launch-capture primitives were added
number of launch-capable primitive families
number of launch-capable variants in W01
first-decision candidate counts by library-size case
first-decision viable counts by library-size case
R9 no-viable rate before and after repair
R9 selected primitive family count before and after repair
whether outcome lookup uses library-size case and compact_library_id
whether memory/exploration selection-change metrics are computed from actual selections
new R5/R6/R7/R8/R9/R10 roots
which gate stops the pipeline, if any
claims explicitly not made
```
