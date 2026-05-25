# Nausicaa Project Skills

## Purpose

This file defines stable working rules for coding, writing, experiments, slides, schedules, and project decisions. It should not contain daily targets, run IDs, exact launch gates, or short-lived implementation plans. Those belong in the project plan and daily schedule.

The goal is to make every output useful, auditable, easy to explain, and suitable for a robotics-style thesis or paper.

Current repair-cycle checks also include history lengths `0, 5, 10, 20, 50, and 100` and require rejected v4.10-style evidence to be labelled `diagnostic_not_passed` before new W0/W1 evidence is generated.

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


Current controlling project overwrite:

```text
Latest user instruction requires 0.10 s primitives with 5 controller-input slots at a 20 ms controller update period, directional 3D residual memory, safe exploration/exploitation, a four-case post-W3 library-size cross-study, archival of rejected v4.10-style results, and a new W0/W1 dense rerun before further validation.
```

If this conflicts with older project-plan or schedule wording, this overwrite controls until explicitly replaced.

---

## 1. Stable project centre

The stable centre is:

> Primitive-level sim-to-real transfer of feedback-stabilised fixed-wing manoeuvre primitives for a small glider operating in measured, uncertain indoor updrafts.

The preferred method is environment-conditioned primitive selection: use glider state and local flow-context features to select 0.10 s primitives through a viability governor, safely explore and exploit viable lift-field choices, then update a directional 3D residual lift belief across launches.

Treat the LQR stabiliser as part of the primitive. The active control/evidence unit is a primitive-controller variant, not a free-standing controller bank. W0/W1 dense generation tunes a primitive-local LQR for every generated variant and preserves the rich library. W2 and W3 replay fixed variants to eliminate or downgrade cases that fail under higher-fidelity or randomised conditions. Post-W3 compression still occurs only after W3, but it must now be studied across heavy, balanced, light, and no-clustering/no-merging cases before a validation library-size condition is accepted. Late validation freezes the selected library-size condition, governor, selector, and memory logic. Hidden retuning inside W2, W3, clustering, or validation is not allowed.

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


- Each active primitive-controller variant must use `finite_horizon_s = 0.100`.
- Each active primitive must support `controller_input_slots_per_primitive = 5` and `controller_input_update_period_s = 0.020`.
- Legacy longer primitive horizons are diagnostic only unless a later explicit decision restores them.
- The active belief is directional, 3D, residual-based, and non-fan-layout-specific.
- Safe exploration may modify ranking only after viability filtering; it must not bypass safety gates.

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
- assumptions behind sampling, the post-W3 library-size cross-study, and latency labels.

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
- directional 3D residual belief evolution;
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
