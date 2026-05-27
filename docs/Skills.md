# Nausicaa Project Skills

<!-- R9_LAUNCH_GATE_ALIGNMENT_START -->

## Active Transition-Aware Thesis Workflow

The active thesis workflow is `R5 -> R7 -> R8 -> R10 -> R11 -> Reality`. R9 remains internal preflight only and is not thesis-facing evidence. R10 tunes the viability governor with residual updraft adaptation, and R11 is the held-out validation gate.

Launch is an entry regime, not a separate controller family. The active primitive catalogue has exactly eight manoeuvre families: `glide`, `recovery`, `lift_entry`, `lift_dwell_arc`, `mild_turn_left`, `mild_turn_right`, `energy_retaining_bank`, and `safe_exit_or_recovery_handoff`. Retired `launch_capture_*` IDs are archive aliases only and must not appear in active evidence.

Every primitive is treated as a transition object. R5 learns transition-aware primitive variants across five entry start families with exact dense proportions per primitive/candidate/environment: 40 `launch_gate`, 25 `inflight_nominal`, 15 `inflight_lift_region`, 10 `inflight_boundary_near`, and 10 `inflight_recovery_edge`. The dense target is `8 * 32 * 3 * 100 = 76,800` rows.

R7 is the hard transition gate. No primitive may pass R7 solely on local rollout success. A controller can survive for one `primitive_id + entry_class` and fail for another. R8 compresses transition objects grouped by `primitive_id` and `transition_entry_class` using coverage-aware medoid selection without averaging Q/R, K, references, or controller IDs.

The governor classifies the current state, filters representatives by validated `transition_entry_class`, rejects high hard-failure risk, scores transition probability plus updraft gain plus flight time plus residual-memory correction, executes the best transition object, and updates case-local residual memory. Step 0 has `current_state_class = launch_gate`, so it selects only transition objects validated for `entry_class = launch_gate`; there is no launch-specific primitive family route.

Residual memory is a small case-local modifier: `predicted_updraft_gain = library_prediction + residual_memory_correction`. It must not override state classification or entry/exit compatibility. Memory is reinitialised per final test row. Final scoring is computed only from the final held-out rollout path. There is no hidden speed gate, no energy-loss hard failure, no PD/PID, no TVLQR, and no fan-layout-specific controller logic.

Core comparison is no memory versus residual-memory histories such as h5, h20, and h100, with safe-explore only as optional ablation. Hardware readiness, real-flight transfer, mission success, autonomy, and memory-improvement claims require R11 and later real-flight evidence.

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

The preferred method is transition-aware primitive selection: each 0.100 s LQR primitive is treated as an entry-class -> exit-class object, and the active governor selects only primitives whose entry class matches the current state class and whose predicted exit is chain-compatible. The active thesis workflow is R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9, safe-explore policies, and large history-matrix studies are internal audit or ablation material only. Low speed is not an active governor rejection reason, recovery-route trigger, score factor, or audit gate; the physical rollout decides whether the simulated flight actually hits floor, ceiling, wall, or an unrecoverable terminal condition. Energy loss is also not a hard-failure reason; R10/R11 score useful updraft extraction separately from whole-flight net energy drift and gross energy loss. Model-backed rollout evidence must integrate positive wing-panel vertical wind along the simulated primitive trajectory; primitive-start local `w_wing` exposure is only a smoke/legacy fallback. The post-W3 representative outcome table is only a base estimate; R10/R11 must pass it through a robust context-conditioned adapter before governor scoring. That adapter may only downgrade probabilities, cap soft rewards, or increase hard-failure risk based on current transition state class, environment class, local lift, local uncertainty, environment block, active fan count, and numeric positive `sample_count` coverage. Governor soft scoring uses context-conditioned `expected_updraft_gain_proxy_m` with active weights named `updraft_gain_weight` and `terminal_updraft_gain_weight`; dry/no-local-lift contexts must not inherit positive updraft score from aggregate representative evidence. Directional memory stores `updraft_gain_residual_m`; legacy `energy_residual` fields are compatibility aliases only, not total-energy memory. Missing compact-library outcome evidence is its own rejection, not a zero-probability proxy. `safe_success` requires a role-compliant launch -> in-flight/recovery transition sequence ending in continuation-valid, terminal-useful, or explicit time-budget completion, with no physical hard failure, floor/ceiling violation, or no-viable primitive tail. Terminal-useful safe exits must be reported separately from `full_safe_success`, and strict R11 must pass a `full_safe_success` gate after consuming the frozen governor config written by R10. Memory is reinitialised per final test row and may only use that row's own history launches; it must not carry between outer cases, library-size cases, policies, or validation blocks. Final scoring is computed only from the final held-out rollout path under controlled pairing, so compared launch-history lengths share the same final launch point and environment instance.

Treat the LQR stabiliser as part of the primitive. The active control/evidence unit is a primitive-controller variant, not a free-standing controller bank. R5 dry-air plus annular-GP dense generation tunes a primitive-local LQR for every generated variant and preserves the rich library; dry-air rows use no updraft but still use W3-style plant and implementation perturbations. Axisymmetric Gaussian plume evidence is diagnostic-only. W2 is optional diagnostic evidence only, while W3 replays fixed variants under held-out randomisation to eliminate or downgrade cases that fail. Clustering and merging occur only after W3, and late validation freezes the post-W3 library-size condition, governor, selector, and memory logic. Hidden retuning inside W2, W3, clustering, or validation is not allowed.

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
- Positive aileron: positive roll moment, right wing down.
- Positive elevator: positive pitch moment, nose up.
- Positive rudder: positive yaw moment, nose right.
- Angles in code: radians unless a field name explicitly says `_deg`.
- Distances: metres. Times: seconds.
- Wind modes: no wind, centre-of-gravity wind, panelwise wind.
- Latency evidence must distinguish ideal timing, actuator lag, command delay, state delay, nominal latency, and conservative timing.
- W0/W1 primitive-controller evidence must already use the panel-wise glider model, feedback latency, command timing, and actuator lag when those effects affect LQR tuning.

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

For R9/R10/R11 repeated-launch validation, worker parallelism is across independent final held-out schedule rows. Each worker must run that row's history launches sequentially before its final launch; do not parallelise within a memory-history chain. R10/R11 full validation uses a 20 s per-episode simulation safety budget, not a primitive-count cap; R9 is a reduced internal preflight with a 10 s episode budget. Any primitive-count cap is diagnostic only. History and final launch selected-primitive trajectory evidence must be retained for thesis plotting; exhaustive all-candidate score rows should be compacted by default.

Default local dense-run policy:

```text
workers = 8
max_workers = 8
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
