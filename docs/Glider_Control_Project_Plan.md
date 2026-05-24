# Glider Control Project Plan

## LQR-stabilised environment-conditioned primitive library after model-only restart

This plan integrates the full environment-conditioned primitive strategy with the model-only restart assumption. It is the active project contract for the Nausicaa glider control work.

The project uses a compact library of short LQR-stabilised **primitives** to exploit local vertical flow across repeated fixed-gate launches. The online method is independent of a particular fan layout. It uses the measured glider state and local flow-context features to predict primitive outcomes, rejects unsafe choices through a viability governor, executes one short LQR-stabilised primitive, records the episode result, and updates an episodic lift belief for later launches.

The active inner-loop controller for each primitive is a time-invariant Linear Quadratic Regulator (LQR) controller synthesised from the control-oriented glider model. The earlier dense run based on simple bounded PD-like feedback is retired. It must be archived or deleted before the next dense run, and it must not be retained as a fallback, baseline, or final-method comparison unless a later explicit project decision reverses this.

Submitted thesis title:

```text
Viability-Guided Sim-to-Real Transfer for a Small Fixed-Wing Glider in Uncertain Indoor Updrafts
```

The title remains suitable because the thesis is still about primitive-level sim-to-real transfer in uncertain indoor updrafts.

---

## 0. Active repository status after restart

This revision assumes a controlled model-only restart from the retained technical foundation:

```text
glider model
latency model
updraft / local-flow model
runtime and storage utilities
model tests and project documentation
```

Retired fan-layout-specific archive code, chain-formation logic, previous bounded-PD/contextual-feedback dense-run outputs, stale clustering packages, stale policy-evaluation artefacts, and stale Codex-generated plans are no longer part of the active method.

This reset is an implementation and project-management cleanup. It does not change the thesis title, the physical platform, or the high-level theme of viability-guided sim-to-real transfer.

The active repository should therefore be treated as if it contains only:

```text
validated model foundations
current project documentation
current coding and housekeeping rules
new LQR-based environment-conditioned primitive modules as they are built
```

Do not reuse historical archive outputs, chain-formation outputs, or the previous bounded-PD dense-run outputs as method evidence. They may be mentioned only as discarded implementation attempts if useful for logbook context.

---

## 1. North star

Build a simple, reproducible, robotics-style sim-to-real control programme showing whether a small fixed-wing glider can select safe and useful primitives from local flow context, and whether this selection transfers from simulation to repeated real launches.

Primary research question:

> Can environment-conditioned primitive selection improve repeated-launch lift capture and finite-horizon dwell for a small glider in a measured uncertain updraft, while preserving safety through a viability governor?

Equivalent implementation question:

```text
Can a compact primitive library transfer from simulation to repeated real launches
when primitive selection is conditioned on local flow context rather than on a
specific fan layout or arena-specific chain of actions?
```

Controller-level implementation question:

```text
Can each active primitive be stabilised by its own compact time-invariant LQR
controller, tuned as part of that primitive's local reference/entry-set design,
then evaluated and shrunk through contextual archive evidence without hidden
validation retuning?
```

The expected contribution is not one attractive trajectory. It is an auditable method linking measured environment, primitive outcome evidence, viability filtering, repeated-launch belief update, and sim-real replay.

The project remains a robotics-style experimental contribution:

```text
measured environment
real hardware and instrumentation
fixed-gate launch protocol
short closed-loop primitives
explicit baselines and ablations
failure labels
runtime-safe archive generation
sim-real replay pairing
claim boundaries
```

---

## 2. What the project is not

The final method is not:

```text
same-flight recapture
perching
indefinite soaring
all-arena exploration
fan-layout-specific controller logic
fixed W0/W1 chain construction
reachable-state extraction as a required success gate
high-angle agile turning as a required final behaviour
generic path following
generic autopilot tuning
direct surface-command reinforcement learning
full nonlinear MPC as the main controller
PD, PID, or hand-tuned bounded feedback as the active primitive controller
TVLQR or time-varying LQR as the active controller in the current workflow
LQR-tree, funnel-library, or formally verified region-of-attraction claims
```

The project may still use high-angle or aggressive manoeuvres as future boundary evidence, but they are not required for final mission success unless direct closed-loop evidence supports their use. The active workflow is an original LQR-stabilised, environment-conditioned primitive method; it must not be branded through another method family.

A launch ending by safety-volume exit, wall-margin stop, low speed, tracking loss, or manual abort is not automatically a failed dataset. It is an episode outcome and must be logged as such.

---

## 3. Method in one chain

```text
measured state
+ local flow context
+ LQR-stabilised primitive catalogue
    -> primitive outcome model
    -> viability governor
    -> selected primitive
    -> episode outcome log
    -> lift-belief update
    -> next launch
```

Use **primitive** as the project term. Do not rename the method to skills unless the project direction explicitly changes.

A primitive is a short closed-loop manoeuvre with:

```text
entry checks
local LQR feedback controller
finite duration
exit checks
metrics
failure labels
claim status
```

The active local controller is time-invariant LQR. The LQR controller is part of the primitive definition, not a free-standing bank chosen independently after the primitive is defined. The active evidence unit is a **primitive-controller variant**: one primitive entry set, reference state, nominal command, finite horizon, local LQR Q/R weights, gain matrix, exit checks, metrics, failure labels, and claim status. Each primitive-controller variant must expose a stable `primitive_variant_id`, `primitive_id`, `controller_id`, nominal reference state, nominal surface command, linearisation metadata, Q/R weights, LQR gain matrix, gain checksum, and synthesis/audit status. A primitive whose LQR synthesis fails must be marked `blocked` or `not_supported`; it must not silently fall back to the old PD-like bounded controller. TVLQR is outside the active workflow because the primitives are short and the additional implementation burden is not justified at this stage.

The online controller must not branch on a named fan setup. Fan count, fan position, and fan power are environment parameters used to create the flow field. The controller receives only:

```text
state
local flow-context features
primitive parameters
uncertainty information
belief map
safety limits
```

Single-fan, four-fan, fan-shift, and fan-power cases are environment instances used for training, validation, and reporting. They are not separate online algorithms.

Compressed-schedule learning and validation separation:

```text
R6  develops primitive-local LQR variants: Q/R weights and limited reference
    parameters are tuned inside each primitive/entry role, producing a rich but
    bounded set of fixed primitive-controller variants.
R7  runs the broad contextual archive, clustering, representative-case
    selection, and library shrinking using those fixed primitive-controller
    variants. R7 may choose among variants, but it must not mutate Q/R weights
    or gains while generating archive evidence.
W2/W3 learning sweeps may replay and shrink the surviving fixed variants under
    hardware-aware and randomised conditions. If a failure motivates retuning,
    the retuned design receives a new primitive_variant_id/controller_id and
    returns to W0/W1/R6 before being replayed again.
Final validation uses frozen variants on held-out randomised simulation and
    real fixed-gate launches. No hidden retuning is allowed in validation.
```

---

## 4. Retained foundation

After repository cleanup, the active codebase should retain and audit the following foundations.

### 4.1 Glider model

Retain:

```text
rigid-body dynamics
aerodynamic model
geometry / mass / inertia / CG
surface limits and sign conventions
state and command contracts
trim and RK4 smoke capability
control-oriented model audit utilities
```

Audit:

```text
state order and units
body/world frame definitions
surface sign conventions
mass, CG, inertia values
aerodynamic coefficient assumptions
trim convergence
RK4 integration sanity
surface saturation and actuator-state handling
comment and docstring consistency with the current project plan
```

### 4.2 Latency model

Retain:

```text
state feedback delay
command onset / transport delay
actuator first-order lag
nominal and conservative timing cases
latency metadata and audit fields
```

Audit:

```text
state delay is not confused with actuator lag
command delay is not confused with surface response time
nominal and conservative timing cases remain distinct
latency status is recorded in every evidence row
instant-feedback evidence is not promoted to true delayed-state-feedback evidence
```

### 4.3 Updraft and local-flow model

Retain:

```text
measured or fitted updraft surrogate
general fan-field / environment representation
wing-scale wind descriptors
local uncertainty descriptors
physically meaningful randomisation of fan position, fan power, amplitude, centre, width, and residual field
```

Audit:

```text
world-frame convention for vertical wind
measured surrogate environment instances
support for fan-position and fan-power variation
wing-mean, left-right, and spanwise-gradient descriptors
model-source labels
fallback handling when a measured model is unavailable
```

Surrogate roles for implementation:

```text
Gaussian plume surrogate                         W1 nominal measured-updraft replay only
GP-corrected annular-Gaussian surrogate          W2 hardware-aware measured-updraft replay only
randomised GP-corrected annular-Gaussian         W3 environment-randomised robustness replay only
GP residual / empirical uncertainty              local uncertainty descriptor for W2/W3 and conservative scoring
```

Do not introduce any additional updraft-surrogate family into the active validation ladder. Earlier fitted surrogates may remain background model-development history, but they are not active W-layer surrogates and must not become implementation branches. These are environment-model choices, not online controller branches. The controller should see only local flow-context features, model-source labels, and uncertainty descriptors.

### 4.4 Runtime and storage utilities

Retain:

```text
chunked execution
8-worker local policy with memory guardrails
compressed table partitions
checksums
table manifests
resume / repair
file-size audit
no-overwrite result handling
```

Audit:

```text
every dense run uses chunked execution
worker-count decision is recorded
all large tables are compressed partitions
no full-memory dense runner is used
no generated file exceeds 100 MB without explicit local-only approval
```

---

## 5. Primitive catalogue

The operational primitive catalogue should stay compact:

```text
glide
recovery
lift_entry
lift_dwell_arc
mild_turn_left
mild_turn_right
energy_retaining_bank
safe_exit_or_recovery_handoff
```

Each primitive has:

```text
primitive_id
primitive_family
primitive_parameters
entry_set
local_lqr_controller
finite_horizon_s
exit_checks
safety_metrics
failure_labels
claim_status
```

The first active primitive set should define:

```text
parameters
entry conditions
LQR reference state and nominal command
Q/R weight metadata and LQR gain matrix
finite horizon
exit checks
metrics
failure labels
```

High-incidence or aggressive manoeuvres may remain as diagnostic boundary evidence, but they are not required for final mission success unless direct closed-loop evidence supports their use. Do not build the main thesis around 45--180 deg agile turns.

Each active primitive-controller variant should also declare an entry role for R6/R7 use:

```text
launch_capable        may be evaluated from the fixed launch gate or launch-acquisition entry set
inflight_only         evaluated from in-flight primitive-entry states, not rejected solely by launch-gate failure
terminal_or_recovery  evaluated for safe recovery, handoff, or useful terminal episode evidence
```

The catalogue may contain more primitive-controller variants than are finally used online. R7 clustering and outcome modelling are expected to shrink the usable library to a smaller real-time set.


### 5.1 LQR primitive synthesis contract

The active primitive implementation is LQR-only. The previous bounded PD-like controller is retired and should be removed from the active controller path after an archive/delete manifest is written.

Each active primitive controller must define:

```text
controller_family              lqr
primitive_variant_id           stable unique ID for the primitive entry role, reference, and controller package
controller_id                  stable unique ID for the primitive-local LQR controller
primitive_id                   active primitive ID
reference_state_vector         x_ref in the canonical 15-state order
reference_command_vector       u_ref in command/surface order
linearisation_id               deterministic identifier for A, B, x_ref, u_ref
linearisation_source           trim / local operating point / short nominal primitive reference
Q_weight_json                  state penalty weights, with units/comments
R_weight_json                  command penalty weights, with units/comments
K_gain_matrix                  LQR feedback gain
K_gain_checksum                checksum for reproducibility
closed_loop_eigenvalue_summary stability audit summary
lqr_synthesis_status           solved / blocked / approximate
controller_claim_status        simulation_only / hardware_shakedown / real_flight_evidence / not_tested
```

Do not tune the raw gain matrix directly. Tune the LQR weight parameterisation, normally log-scaled diagonal entries of `Q` and `R`, plus only a small number of primitive-level reference or entry-set parameters where required. Tuning is part of primitive synthesis: it produces primitive-controller variants rather than a separate controller bank. The first implementation should keep the search auditable but rich enough to give R7 a useful library: start from physically interpretable baseline Q/R weights, generate several role-aware variants per primitive, then let R7 evidence and clustering shrink the library.

Tuning policy:

```text
W0  dry-air LQR synthesis and sanity checks for primitive-local variants
W1  nominal measured-updraft LQR weight/reference tuning and candidate selection
W2  hardware-aware survival replay for fixed variants; any retuned design becomes a new variant and returns to W0/W1
W3  environment-randomised replay for fixed variants; any retuned design becomes a new variant and returns to W0/W1
```

If W2 or W3 exposes a failure mode that requires retuning, create a new `primitive_variant_id` and `controller_id`, return the candidate to W0/W1, and then replay W2/W3 again. Do not mutate an already reported variant in place.

---

## 6. Environment context

The core abstraction is a local environment-context vector. It converts any wind field into comparable features.

Required context features:

```text
w_cg_m_s                         centre-of-gravity vertical wind
w_wing_mean_m_s                  wing-mean vertical wind
delta_w_lr_m_s                   left-right wing vertical-wind difference
spanwise_w_gradient_m_s_per_m    spanwise vertical-wind gradient
w_local_uncertainty_m_s          local flow uncertainty
lift_score                       normalised nearby useful-lift score
lift_direction_xy                direction to locally useful lift
wall_margin_m                    minimum wall margin
floor_margin_m                   floor margin
ceiling_margin_m                 ceiling margin
speed_margin_m_s                 margin to low-speed limit
attitude_margin_rad              margin to roll/pitch admission bound
latency_case                     ideal, actuator-lag-only, nominal, conservative
actuator_case                    nominal or conservative actuator response
```

Environment metadata may include:

```text
environment_id
fan_count
fan_positions_m
fan_power_scales
updraft_model_id
updraft_amplitude_scale
updraft_width_scale
updraft_centre_shift_m
residual_field_id
randomisation_seed
```

These metadata fields support audit and generalisation tests. They must not become separate online controller branches.

The online algorithm must not care whether lift comes from:

```text
one fan
four fans
shifted fan positions
different fan powers
a different measured local-flow field
```

It only uses the local context vector and the uncertainty descriptors.

---

## 7. Episode definition

An episode is one simulated or physical launch. The physical evaluation uses a fixed nominal release gate for repeatability:

```text
x_w   in [1.2, 1.4] m
y_w   in [1.8, 2.2] m
z_w   in [1.5, 1.9] m
psi   in [-30, 30] deg
phi   in [-45, 45] deg
theta in [-45, 45] deg
V     in [3, 8] m/s
```

The measured initial state must be retained:

```text
x0 = [x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r]
```

Do not replace the measured state with the gate centre. Manual launch variation is part of the sim-to-real condition.

Episode termination may occur by:

```text
safety-volume exit
wall-margin stop
floor / ceiling boundary
low speed
tracking loss
controller rejection
manual abort
controlled finish
```

Termination is an outcome label, not automatic mission failure.

For archive generation before clustering, x-y wall or lateral safety-volume exit should be retained as a terminal outcome rather than used as a row-deletion rule. These terminal rows may be weak, failed, or boundary-terminal evidence, but they are still useful for learning where primitives stop being viable in a repeated-launch task. Floor and ceiling violations remain safety-critical z-boundary failures and must be labelled separately.

Implementation must distinguish two non-z-boundary primitive uses:

```text
continuation_valid
    The primitive finishes with enough x-y wall margin, speed margin, attitude margin, and state validity to seed another primitive in the same simulated episode.

episode_terminal_useful
    The primitive may terminate at an x-y wall limit or lateral safety-volume edge, but it still provides useful lift capture, finite dwell, energy retention, or boundary evidence for the repeated-launch objective. It is retained for archive learning and repeated-launch evaluation, but not used as a downstream continuation state.
```

This distinction prevents the governor from rejecting every useful primitive merely because every launch eventually reaches the lateral boundary in the small arena. The archive records both labels; the selector decides later whether the current mode is continuation or terminal episode use.

Recommended physical start trigger:

```text
Vicon rigid body valid for several consecutive frames
speed above start threshold
state inside gate or accepted tolerance shell
controller has valid pose and command path ready
```

### 7.1 Primitive-start distribution for archive and replay

The fixed launch gate is the physical episode start, but it is not the only primitive-start state used for primitive outcome evidence. A primitive may be attempted at launch, or it may be attempted after a previous primitive has ended. Archive generation should therefore use a mixed primitive-start distribution, with launch-gate states treated as one important subcase of the broader primitive-start distribution rather than as a separate algorithm or separate result family.

Do not use this distribution to rebuild fixed primitive chains or reachable-state chains. The archive should sample independent primitive-attempt states and ask what happens if one primitive is attempted from that state and context. Single-primitive exit states may be resampled into an in-flight start-state pool, but they must be logged as independent primitive-start samples rather than as part of an optimised chain or arena-specific action sequence.

A first default archive mixture is:

```text
launch_gate                 40%   real episode starts and first primitive choices
inflight_nominal            25%   ordinary in-flight states after clean primitive exits
inflight_lift_region        15%   in-flight states near useful local updraft
inflight_boundary_near      10%   near-wall states for terminal-boundary and governor learning
inflight_recovery_edge      10%   low-speed, high-attitude, or recovery-margin states
```

Every sampled primitive-start state should retain the full canonical state, including position, attitude, body velocity, body rates, and surface states:

```text
x0 = [x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r]
```

The archive should also record primitive-start provenance fields:

```text
start_state_family          launch_gate / inflight_nominal / inflight_lift_region / inflight_boundary_near / inflight_recovery_edge
state_sample_source         measured_log / synthetic_launch_gate / synthetic_inflight / rollout_exit_resampled / stress_sample
state_envelope_label        approved_launch_gate / local_primitive_envelope / lift_region / boundary_near / recovery_edge
paired_start_key            key for W0/W1/W2/W3 paired comparisons where applicable
previous_primitive_status   launch_start / clean_exit / weak_exit / boundary_terminal / recovery_edge / unknown
synthetic_previous_primitive_id
synthetic_time_since_launch_s
state_sampling_seed
```

Launch-gate rows should obey the physical release gate above. In-flight rows should cover plausible primitive-entry states inside the safety volume without encoding a named sequence of previous actions as a controller branch. This gives the outcome model evidence for both first-launch primitive attempts and mid-flight primitive attempts while preserving the project decision not to make chain construction a success gate.

---

## 8. Primitive outcome evidence

Every archive row should answer:

```text
At this state, in this local flow context, under this timing case, what happens if this primitive is attempted?
```

Minimum row fields:

```text
rollout_id
episode_id optional
environment_id
W_layer
initial_state_vector
context_feature_vector
primitive_id
primitive_family
primitive_parameters
controller_mode
feedback_mode
controller_family              lqr
controller_id
lqr_reference_id
linearisation_id
lqr_Q_weights_json
lqr_R_weights_json
lqr_gain_checksum
lqr_synthesis_status
lqr_closed_loop_audit
tuning_stage                   W0_synthesis / W1_tuning / W2_survival / W3_survival / real_flight
latency_case
accepted
outcome_class               accepted / weak / failed / rejected / blocked
energy_residual_m
lift_dwell_time_s
minimum_wall_margin_m
minimum_speed_m_s
exit_state_vector
termination_cause
continuation_valid
episode_terminal_useful
boundary_use_class          continuation_valid / episode_terminal_useful / hard_failure / blocked
failure_label
claim_status
```

Additional primitive-start provenance fields should be recorded when the archive uses the mixed launch/in-flight start-state distribution:

```text
start_state_family
state_sample_source
state_envelope_label
paired_start_key
previous_primitive_status
synthetic_previous_primitive_id
synthetic_time_since_launch_s
state_sampling_seed
```

Accepted rows, weak rows, failed rows, and rejected rows are all evidence. Do not erase failures through clustering or averaging.

Do not discard rollout rows before clustering only because the simulated primitive reaches an x-y wall limit or lateral safety-volume edge. In the archive, this is a terminal outcome and a source of boundary evidence. Penalising or rejecting such a primitive for real execution belongs to the viability governor or later selector, not to the archive row-generation filter. A z-boundary violation, nonfinite state, or physically invalid rollout must remain a hard failure label.

Boundary-use labels must be assigned before clustering:

```text
continuation_valid
    exit state can be used to seed another primitive in the same simulated episode
episode_terminal_useful
    x-y boundary or wall-limit termination is retained as useful repeated-launch terminal evidence
hard_failure
    floor/ceiling violation, nonfinite state, corrupt integration, or physically impossible state
blocked
    unsupported surrogate, feedback, latency, or entry condition before rollout
```

The archive should train the outcome model on terminal-useful cases as terminal evidence, but must not let those rows masquerade as continuation-success rows.

Archive evidence should be organised as:

```text
state features
+ local flow context
+ primitive family
+ primitive parameters
+ controller_id
+ LQR synthesis metadata
+ latency case
+ uncertainty descriptors
    -> accept / reject / weak / fail
    -> minimum margin
    -> energy residual
    -> finite dwell time
    -> exit-state summary
    -> termination cause
    -> failure label
```

Do not organise the method as a fan-layout-specific or validation-layer-specific chain of actions.

---

## 9. Primitive outcome model

The primitive outcome model maps:

```text
state + environment_context + primitive_parameters + controller_id + LQR metadata -> predicted outcome
```

The state distribution used for model fitting should mix launch-gate and in-flight primitive-start states in the same training table. Controller identity must be retained as part of the evidence, because the archive now evaluates LQR-stabilised primitive attempts rather than generic primitive labels alone. The launch-gate subset is not a separate policy; it is one labelled subset of the primitive-start distribution. Validation splits should therefore report performance by `start_state_family`, `state_envelope_label`, `environment_instance_id`, primitive, W-layer, latency case, and seed where the data volume permits.

First acceptable implementations:

```text
nearest-neighbour / kNN lookup
binning table
medoid lookup
table-based score model
calibrated logistic or tree model
small transparent regressor/classifier
```

Avoid large black-box learning unless simpler outcome models fail and there is enough data to justify the extra complexity.

The model should predict or estimate outcomes conditional on the selected LQR controller. It must not merge rows from retired PD-like controllers into active LQR evidence.

The model should predict or estimate:

```text
acceptance probability
weak/failure probability
energy residual
lift dwell time
minimum safety margin
termination label
continuation_valid probability or label
episode_terminal_useful probability or label
```


The outcome model should keep continuation and terminal-use labels separate. A primitive that is useful as an episode-ending lift-capture action should not be counted as a continuation-valid primitive unless its exit state also satisfies the margin and state-validity checks needed to start another primitive.

The model must remain explainable enough to support a robotics-journal method section.

---

## 10. Viability governor

The governor is the safety filter. It rejects primitives before execution if predicted, checked, or simulated behaviour violates:

```text
safety volume
minimum wall / floor / ceiling margin
minimum speed
attitude or incidence limits
surface limits
actuator saturation
valid LQR synthesis and closed-loop audit
primitive entry set
exit check
recovery margin
uncertainty margin
supported latency / LQR feedback mode
```

Primitive scoring is allowed only after viability filtering. The governor must reject a primitive if its LQR controller is missing, unstable under the recorded audit, outside its supported reference envelope, or marked blocked/approximate beyond the allowed claim status. It must not substitute an archived PD-like controller.

The governor is where x-y wall risk is converted into a selection penalty or rejection for real execution. Archive generation should preserve boundary-terminal evidence first, then the governor learns or applies the conservative decision boundary from those labelled outcomes.

The governor must therefore support two explicit operating modes:

```text
continuation_mode
    reject primitives predicted to hit the x-y wall before their finite horizon or to exit without enough margin for another primitive

terminal_episode_mode
    allow an x-y boundary-terminal primitive if it is predicted to provide useful lift capture, finite dwell, or energy retention before a controlled terminal outcome, while still rejecting z-boundary violation, low-speed unrecoverability, nonfinite trajectories, and unsupported feedback or surrogate cases
```

The selected mode must be logged. This keeps the repeated-launch mission feasible without pretending that lateral-boundary termination is safe continuation evidence.

Suggested utility:

```text
utility = predicted_energy_residual
        + lift_dwell_weight * predicted_lift_dwell_time
        - wall_risk_weight * predicted_wall_risk
        - uncertainty_weight * local_uncertainty
        - saturation_weight * predicted_saturation
```

Keep the score simple enough to explain in one slide or algorithm box.

The governor must log both accepted and rejected candidates. Rejection logs are part of the evidence, not implementation clutter.

---

## 11. Repeated-launch belief

The belief layer estimates where useful lift was observed and how stale that information is. A first auditable update is:

```text
b_{k+1}(r) = lambda b_k(r) + (1 - lambda) w_hat_k(r)
```

Evaluate:

```text
lambda in {0.0, 0.5, 0.8, 0.95}
```

where `lambda = 0` is the no-memory baseline.

The belief proposes objectives or lift-seeking context. The viability governor still decides whether a primitive is allowed.

The repeated-launch objective is:

```text
improve lift capture, finite dwell, and energy retention across launches
while preserving safety margins and rejection logs
```

Evaluation policies:

```text
no_memory_baseline
static_map_baseline
environment_conditioned_selector_without_memory
environment_conditioned_selector_with_episodic_lift_belief
oracle_simulation_upper_bound_optional
```

---

## 12. Validation ladder

W0--W3 are validation layers, not online controller branches.

```text
W0  dry-air baseline
W1  nominal measured-updraft replay
W2  hardware-aware replay
W3  environment-randomised robustness replay
real flight  repeated-launch validation
```

Updraft surrogate use across the validation ladder:

```text
W0  no updraft; use dry air only
W1  nominal measured-updraft replay; use the Gaussian plume surrogate only
W2  hardware-aware replay; use panelwise wind and the GP-corrected annular-Gaussian surrogate only, carrying its residual or empirical uncertainty into the context features
W3  environment-randomised replay; use the randomised GP-corrected annular-Gaussian surrogate only, randomising fan position, fan power, amplitude, centre, width, residual field, and GP/residual uncertainty
```


Primitive-controller variant use across the validation ladder:

```text
W0  synthesise and sanity-check time-invariant LQR variants for each active primitive/entry role
W1  tune LQR Q/R weights and primitive reference parameters under nominal measured-updraft replay
W2  replay fixed W1-supported variants under hardware-aware assumptions; shrink or downgrade the library
W3  replay fixed W2-supported and informative variants under environment/implementation/plant randomisation; shrink or downgrade the library
real flight  use only variants whose claim status permits hardware shakedown or real-flight evidence collection
```

W0/W1 are synthesis and tuning layers. W2/W3 have two explicitly separated uses: learning/shrinking sweeps may identify failures and motivate new primitive-controller variants, while held-out validation replays frozen variants without hidden retuning. Any W2/W3-motivated retuning must create a new `primitive_variant_id` and `controller_id`, return to W0/W1, and then be replayed again before support is claimed.

For the compressed project schedule, R6 should be treated as a primitive-local LQR variant generation stage rather than a thesis-scale coverage archive. It should produce a rich but bounded set of fixed, auditable primitive-controller variants to unlock R7. R7 is the first broad dense archive used for validation, clustering, representative-case selection, and library shrinking. R6 may include a small training-randomised band if needed for controller robustness, but those rows are training/tuning evidence, not held-out W2/W3 validation.

The implementation must record `updraft_model_id`, model source, residual / uncertainty descriptor, and environment metadata for every row. The online primitive selector should not branch on Gaussian versus GP-corrected model family; it should use the resulting local context features and uncertainty labels. If the requested W-layer surrogate is unavailable, the run must write a blocked manifest rather than silently falling back to another surrogate.

### 12.1 W0 — dry-air baseline

Dry-air baseline. Quantifies what each primitive does without updraft assistance.

Use W0 for:

```text
LQR trim/local-reference linearisation smoke
controllability and closed-loop eigenvalue audit
model and runtime debugging
dry-air energy-loss baseline
checking whether a primitive only works because of lift
paired comparison with measured-updraft runs
```

### 12.2 W1 — nominal measured-updraft replay

Nominal measured-updraft replay using the Gaussian plume surrogate only. Tests primitive outcomes in a low-order measured-updraft baseline field.

Rules:

```text
W1 is the main LQR Q/R tuning and candidate-selection layer.
W1 is not filtered by W0 winners.
W0 and W1 may use paired start-state keys where useful.
W0 failure may be useful ablation evidence, not automatic W1 rejection.
Fan layout is environment metadata, not an online branch.
```

### 12.3 W2 — hardware-aware replay

Hardware-aware replay using higher-fidelity timing and safety assumptions:

```text
panelwise wind
state-delay model
command delay or FIFO/onset delay
actuator lag
Vicon-style start trigger where practical
real safety-volume termination
measured launch-state distribution when available
```

W2 uses the GP-corrected annular-Gaussian surrogate and reduces the gap between cheap nominal simulation and hardware. W2 must not retune LQR weights in place. W2 should replay a representative selection of W1 accepted, weak, boundary-terminal, and informative failed cases; it must not be limited to W1 winners only.

### 12.4 W3 — environment-randomised robustness replay

Environment-randomised robustness test. Randomise physically meaningful parameters:

```text
fan position
fan power
active fan subset
updraft amplitude
updraft centre
updraft width
residual vertical field
local uncertainty
launch state
latency and actuator response
mass / CG / inertia
surface calibration scale
```

W3 should be heavier than W2. In addition to environment randomisation, W3 should include implementation and glider-plant randomisation as first-class logged inputs rather than hidden changes inside the dynamics code. Required W3 randomisation groups are:

```text
environment randomisation
    fan position
    fan power
    active fan subset where the surrogate can represent it honestly
    updraft amplitude
    updraft centre
    updraft width
    residual vertical field
    local uncertainty scale

implementation randomisation
    state-feedback delay
    command onset / transport delay
    actuator lag
    latency jitter
    command quantisation
    surface neutral bias
    surface limit scale
    surface effectiveness scale
    left/right aileron asymmetry

glider plant randomisation
    mass scale
    centre-of-gravity offset in x/y/z
    Ixx / Iyy / Izz scale
    cross-inertia perturbation if supported by the model
    aerodynamic coefficient scale where justified
    surface calibration scale
```

Every W3 replay row should log a deterministic randomisation seed and the active instance identifiers:

```text
environment_instance_id
implementation_instance_id
plant_instance_id
plant_randomisation_seed
actuator_randomisation_seed
```

If a randomisation component cannot be represented honestly by the available surrogate or dynamics model, the row or component must be labelled as approximate or blocked rather than silently treated as exact.

W3 uses the randomised GP-corrected annular-Gaussian surrogate and is the main simulation test that the method is not tied to one fan layout. W3 must not retune LQR weights in place. W3 output should be a pass/fail/weak label under uncertainty, not just a trajectory plot. W3 should stress W2-supported and W2-informative cases; it must not reintroduce extra updraft-surrogate-family branching.

### 12.5 Real flight

Real flight is fixed-gate repeated-launch validation with measured initial states. Real data must be paired with matched simulation before any transfer claim is made.

---

## 13. Simulation budgets after restart

Use small, meaningful runs first. Dense runs are allowed only after the foundation smoke tests pass.

| Stage | Preferred target | Fallback | Purpose |
|---|---:|---:|---|
| model audit smoke | 100--500 rows | 50 | check model, latency, wind, storage |
| LQR synthesis smoke | 1--2 controllers per primitive | 1 primitive family | check trim/local reference, linearisation, Riccati solve, gain audit |
| contextual primitive smoke | 1k--3k rows | 500 | check LQR primitive interfaces and labels |
| W0/W1 primitive-local LQR variant tuning | rich but bounded entry-role-aware primitive-controller variant set; preferred 12--24 selected/accepted-fallback variant IDs plus rejected/blocked records, emergency minimum 8--12 if time-constrained | minimum launch/recovery, glide/continuation, lift/context, and terminal/recovery coverage if physically viable | tune Q/R weights and limited primitive/reference entry-role parameters enough to give R7 a useful library |
| contextual LQR archive | 40k--80k rows | 20k | validate fixed R6 primitive-controller variants, train outcome model, cluster evidence, and shrink library for real-time selection |
| hardware-aware LQR survival | 5k--15k rows | 2k | W2 survival under latency, actuator, panelwise wind, realistic termination |
| environment generalisation | 10k--30k rows | 5k | W3 fan position/power/updraft/implementation/plant randomisation |
| repeated-launch simulation | 100--300 episodes | 50 | compare policies using W2/W3-supported LQR controllers |
| hardware shortlist | 5--10 candidates | 3 | choose real-flight LQR primitive candidates |

For contextual LQR archive and environment-generalisation stages, the default primitive-start mixture should combine launch-gate and in-flight primitive-entry states in a single table. Primitive-local LQR variant tuning should be completed before the main dense archive; the dense archive should evaluate declared `primitive_variant_id` and `controller_id` values rather than continue searching over Q/R weights. A useful first target is 40% launch-gate states and 60% in-flight states, with in-flight rows divided among nominal, lift-region, boundary-near, and recovery-edge cases. R7 may cluster and shrink the fixed primitive-controller library, but it must not retune Q/R weights in place. This is not a return to primitive-chain construction; it is a way to evaluate each primitive-controller variant on realistic launch and mid-flight entry states while keeping each row as one independent primitive attempt.

Do not run large all-arena archives. Large all-arena sweeps are not the main result and must not displace real-flight preparation, analysis, or writing time.

---

## 14. Baselines and ablations

Minimum baselines:

```text
no_memory_baseline
static_measured_map_baseline
state_only_primitive_selector
state_plus_flow_context_selector
episodic_memory_selector
oracle_simulation_upper_bound_optional
```

Minimum ablations:

```text
without wing-scale wind features
without uncertainty penalty
without episodic memory
without viability governor, simulation only
without environment randomisation
without LQR weight tuning, using nominal LQR weights only
```

Unsafe ablations remain simulation-only. Do not preserve the retired PD-like controller as a baseline, fallback, or ablation in the active workflow.

Baseline and ablation outputs must include:

```text
sample count or episode count
capture rate
finite dwell time
energy residual
minimum wall margin
termination causes
governor rejection count
claim status
```

---

## 15. Real-flight evidence

Real evaluation remains fixed-gate because it makes sim-real comparison interpretable.

Minimum useful real evidence:

```text
dry-air or fan-off baseline
measured-updraft static-map baseline
measured-updraft environment-conditioned primitive selection
episodic-memory repeated launches
changed fan power or fan subset if safe
matched sim-real replay for every usable flight
```

Metrics:

```text
lift capture rate
finite dwell time
energy residual
height loss
minimum wall margin
termination cause
governor rejection count
prediction error versus matched simulation
```

Minimum real-flight comparison:

```text
dry-air / fan-off baseline
static-map updraft baseline
environment-conditioned selector without memory
environment-conditioned selector with episodic belief
matched sim-real replay
```

Every usable real flight should produce:

```text
episode_id
measured initial state
environment_id
primitive attempted
accepted / rejected status
termination cause
energy residual
finite dwell time
minimum margin
belief before and after
matched simulation result
```

---

## 16. Archive compression and clustering

Clustering is for compression, explanation, representative examples, hardware shortlist, and figures. It is not the online controller. In the active workflow, clustering compresses LQR controller/outcome evidence and must not mix retired PD-like evidence into active clusters.

Cluster within physically meaningful strata:

```text
W_layer
environment_family
primitive_family
controller_family
controller_id
linearisation_id
latency_case
outcome_class
evidence_role
```

When mixed primitive-start evidence is used, reports should also stratify or summarise by:

```text
start_state_family
state_envelope_label
previous_primitive_status
```

Use medoids or representative real rows where possible. A medoid remains replayable and auditable.

Outputs:

```text
representative accepted cases
representative weak cases
representative failures
hardware-candidate shortlist
failure-label summary
figure-source manifest
```

Do not hide failures through clustering. Failed and rejected cases are part of the result because they define the boundary of transfer.

Boundary-terminal rows, including x-y wall-limit or lateral safety-volume exits, should form explicit clusters or failure summaries rather than being removed before clustering. They are not automatically hardware candidates, but they are essential for learning the context boundary where the governor should become conservative. Clustering must not select only clean accepted rows for W2/W3; representative weak, boundary-terminal, and informative failed rows are needed so the outcome model and governor learn the operating boundary rather than only the easy region.

Cluster reports should keep `continuation_valid` and `episode_terminal_useful` cases visibly separate. Terminal-useful clusters may support repeated-launch lift-capture decisions, but they must not be promoted to downstream-continuation evidence.

---

## 17. Runtime, storage, and file-size contract

This section is mandatory for all future dense/archive/thesis-scale simulation.

A run is dense if any condition is true:

```text
planned rollout rows >= 10,000
planned candidate rows >= 5,000
expected runtime > 30 minutes
expected uncompressed table size > 250 MB
used for thesis evidence, environment generalisation, envelope maps, clustering, W2/W3 replay, outcome models, or governor packages
```

Dense runs must be:

```text
chunked
resumable
compressed
environment-partitioned where applicable
worker-enabled
checksum-manifested
file-size-audited
```

Required runner options:

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

Default local policy:

```text
workers = 8
max_workers = 8
storage_format = auto, resolving to parquet if available, otherwise csv_gz
compression_level = 1 for csv_gz
resume = true
```

This 8-worker and compressed-partition policy is a hard project requirement for dense/archive/thesis-scale runs. New evidence runners must reuse the retained runtime and table-I/O utilities instead of introducing a full-memory single-process path.

Do not use a single-process full-memory runner for dense runs.

### 17.1 Output file-size rule

Every generated file should target:

```text
preferred size <= 75 MB
hard project limit <= 100 MB
```

If an artifact is expected to exceed 100 MB, split it before writing. Dense tables must be partitioned. Figures should be simplified or split if needed. A file above 100 MB is allowed only as explicitly approved local-only evidence and must not be pushed without an approved storage method.

No generated result file may exceed 100 MB in committed project outputs.

---

## 18. Result naming

Use concise, numbered, lower-snake-case names.

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
99_misc
```

Preferred abbreviations:

```text
ctx  environment context
prim primitive
pol  policy
ep   episode
sim  simulation
rf   real flight
sr   sim-real replay
lqr  linear quadratic regulator
w0   dry air
w1   measured updraft
w2   hardware-aware replay
w3   environment randomisation
```

Avoid long filenames that can stall OneDrive or exceed Windows path limits. Target:

```text
filename stem <= 64 characters
repository-relative path <= 140 characters
generated file <= 100 MB
```

Every output should remain identifiable from:

```text
result group
case slug
run ID
manifest content
compact filename stem
```

Do not encode the entire experiment description into the filename.

---

## 19. Claim boundaries

Do not claim the following unless direct evidence supports it:

```text
real-flight transfer
mission success
hardware readiness
same-flight recapture
perching
all-environment validity
full W3 robustness
true delayed-state-feedback validation
full autonomy
environment generalisation
formal LQR-tree, funnel, or verified region-of-attraction guarantees
```

Use claim labels:

```text
simulation_only
hardware_shakedown
real_flight_evidence
partial_transfer
negative_transfer
instrumentation_limited
not_tested
```

Allowed final claim forms:

```text
simulation-supported
real-flight-supported
partial transfer
negative transfer
instrumentation-limited
not tested
```

A result is not a transfer result until a real flight has a matched simulation replay and the metric comparison is reported.

---

## 20. Immediate restart deliverables

Before any new dense run:

```text
1. archive/delete manifest for the previous bounded-PD dense run and active-code references
2. model-only repository cleanup report
3. glider/latency/updraft model audit
4. LQR trim/local-reference, linearisation, Riccati, and closed-loop audit
5. runtime/storage/file-size audit
6. contextual LQR primitive module skeletons
7. smoke archive proving the new LQR data schema
```

Priority modules:

```text
environment_context.py
primitive_catalog.py
lqr_linearisation.py
lqr_controller.py
lqr_tuning.py
contextual_lqr_archive.py
primitive_outcome_model.py
viability_primitive_selector.py
episodic_lift_belief.py
run_lqr_contextual_archive_chunked.py
run_lqr_w2_replay.py
run_lqr_w3_generalisation_eval.py
run_repeated_launch_contextual_policy_eval.py
```

Historical archives, including the first bounded-PD dense run, should not be reused as the main method path. They may be used only as modelling lessons after explicit conversion into the current context-feature schema and with an explicit `retired_not_active` label.

The first new implementation should prove:

```text
model foundation imports cleanly
context features can be computed for dry air and measured updraft
primitive catalogue contains the compact active set
at least one LQR controller can be synthesised or explicitly blocked with reason
one primitive rollout row contains state, context, primitive, controller_id, LQR metadata, outcome, and claim status
chunked storage obeys the 100 MB file-size rule
```

---

## 21. Thesis integration note

The submitted thesis structure can remain broadly compatible. The control chapter should be reframed around:

```text
environment context
LQR primitive synthesis
primitive outcome evidence
viability-governed primitive selection
episodic lift belief
W0--W3 validation layers
sim-real replay pairing
```

Avoid framing Chapter 6 around fan-layout cases, chain construction, PD-style controller tuning, or high-angle agile primitives as required final behaviour.

The results chapter should be structured around:

```text
model audit
LQR synthesis and tuning evidence
contextual LQR primitive evidence
outcome-model / lookup performance
environment generalisation
repeated-launch simulation
real-flight comparison
claim boundaries
```

---

## 22. Decision rule for future changes

A proposed method change should be accepted only if it improves at least one of:

```text
simplicity of explanation
reproducibility
real-flight readiness
sim-real traceability
environment generalisation
safety and claim discipline
runtime/storage reliability
```

Reject changes that mainly add project-specific complexity, fan-layout-specific logic, hidden dependencies on a particular environment layout, TVLQR implementation burden, or reintroduction of PD-like active controller paths.
