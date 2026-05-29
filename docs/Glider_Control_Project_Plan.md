# Glider Control Project Plan

<!-- R9_LAUNCH_GATE_ALIGNMENT_START -->

## Active Transition-Aware Thesis Workflow

The active thesis workflow is `R5 -> R7 -> R8 -> R10 -> R11 -> Reality`. R9 remains internal preflight only and is not thesis-facing evidence. R10 tunes the viability governor with residual updraft adaptation, and R11 is the held-out validation gate.

Launch is an entry regime, not a separate controller family. The active primitive catalogue has exactly eight manoeuvre families: `glide`, `recovery`, `lift_entry`, `lift_dwell_arc`, `mild_turn_left`, `mild_turn_right`, `energy_retaining_bank`, and `safe_exit_or_recovery_handoff`. Retired `launch_capture_*` IDs are archive aliases only and must not appear in active evidence. `safe_exit_or_recovery_handoff` remains active as an evidence-tested recovery / controlled-terminal primitive; it must be demoted only if R8/R10 evidence shows no unique transition coverage, not removed by assumption.

Every primitive is treated as a transition object. R5 is robust transition-aware primitive / transition-object Q/R plus primitive attitude/bank reference-bias training across five entry start families with exact dense proportions per primitive/candidate/environment: 40 `launch_gate`, 25 `inflight_nominal`, 15 `inflight_lift_region`, 10 `inflight_boundary_near`, and 10 `inflight_recovery_edge`. The dense evaluation target remains `8 * 32 * 3 * 100 = 76,800` rows, but row count alone is not a pass condition. Candidate 0 is nominal, candidates 1-7 are named physical anchors with small interpretable attitude/bank reference biases, and candidates 8-31 are deterministic Latin-hypercube log multipliers over the seven grouped LQR weights plus bounded pitch and bank reference biases. In-flight start-state velocity envelopes cover most of the local-speed scheduling grid: nominal `u=3.0--8.2`, lift-region `u=3.2--8.0`, boundary-near `u=3.0--8.0`, and recovery-edge `u=2.2--5.2` m/s, with wider lateral/vertical body-velocity perturbations logged by the sampler. Speed-bin selection is for local model scheduling only; active primitives must not chase speed as a hard reference. R5 writes `r5_transition_candidate_training_summary.csv`, `r5_transition_selected_for_r7.csv`, `r5_transition_pareto_front.csv`, and `r5_transition_training_manifest.json`, then freezes only selected transition objects for R7 while keeping the full candidate bundle as audit evidence.

R7 is held-out transition validation of the frozen R5-selected transition objects. No primitive may pass R7 solely on local rollout success; no primitive may pass R5 or R7 from dense row count or aggregate primitive success across entry classes. R7 uses entry-class-specific labels: `survived` is reserved for strict high-probability `inflight_stable` evidence, `route_usable` keeps `launch_gate` evidence when transition probability is at least 0.40 with near-zero hard failure and keeps `boundary_near` evidence when transition probability is at least 0.40 with hard failure below limit, and `recovery_route_usable` keeps `recoverable_degraded` evidence when it has nonzero recovery progress in dry-air, single-fan, and four-fan R7 modes with low hard failure. For recovery starts, `recoverable_degraded -> recoverable_degraded` remains a conditional route pass when attitude/rate risk improves, front/side boundary time margin does not collapse, floor margin does not collapse, and hard-failure risk remains low; `recoverable_degraded -> boundary_near` is reported as route/weak evidence, not a full pass. A controller can be strict-surviving for one `primitive_id + entry_class`, route-usable for another, and fail for another. R8 compresses R7/W3-eligible transition objects (`survived`, `route_usable`, and `recovery_route_usable`) grouped by `primitive_id` and `transition_entry_class` using coverage-aware medoid selection without averaging Q/R, K, references, or controller IDs. R8 must also preserve distinct W3-eligible local LQR speed-bin coverage within each primitive/entry-class group up to the case representative budget; speed-bin collapse is a library-coverage failure, not an LQR-principle failure.

The governor classifies the current state, filters representatives by validated `transition_entry_class`, rejects high hard-failure risk, scores transition probability plus updraft gain plus flight time plus residual-memory correction, executes the best transition object, and updates case-local residual memory. Step 0 has `current_state_class = launch_gate`, so it selects only transition objects validated for `entry_class = launch_gate`; there is no launch-specific primitive family route. Because repeated-launch episodes must end somewhere, a finite controlled x-y arena exit with positive floor/ceiling margin is a `safe_terminal` outcome, not a hard failure. Floor/ceiling impact, invalid state, uncontrolled attitude/rates, and uncontrolled boundary exit remain hard failures.

The inner LQR remains a stabilising tracker around the primitive-defined local reference. It must not become the manoeuvre planner: primitive transition objects define manoeuvre intent and R5/R7/R8 decide which transition objects are valid, while the governor selects among those frozen transition objects. Signed turn intent is diagnostic only: R5/R7 may record signed bank, signed roll-rate, and lateral turn tendency for audit, but active selection must not reward turn-expression strength and `mild_turn_left` / `mild_turn_right` must not receive sign-constrained bank or roll-rate reference forcing. `mild_turn_left` and `mild_turn_right` remain separate directional primitive IDs because the arena, local flow, and reachable transitions are not guaranteed symmetric; R8 may compress or downweight their representatives by evidence, but active code must not merge the IDs into one aggregate score. `energy_retaining_bank` remains non-directional because it is an energy/posture primitive, not an explicit left/right command. Same-start trajectory plots are diagnostic-only sanity checks for controller behaviour and do not replace R5/R7 transition evidence.

Residual memory is a small case-local modifier: `predicted_updraft_gain = library_prediction + residual_memory_correction`. It must not override state classification or entry/exit compatibility. Memory is reinitialised per final test row. Final scoring is computed only from the final held-out rollout path. There is no hidden speed gate, no energy-loss hard failure, no PD/PID, no TVLQR, and no fan-layout-specific controller logic.

The repeated-launch outer-loop scheduler must not assume free real-time computation. R9/R10/R11 profile context construction, residual-memory query, and compact-library selection for every primitive decision against a preferred 20 ms controller-slot budget and a hard 0.100 s primitive-boundary budget. Step 0 may be prepared before release; later steps use a prepared next-decision path so the next primitive is committed at the boundary while actuator/command FIFO history remains continuous. This is an offline wall-clock profile only, not a hardware real-time-readiness claim.

R10 and R11 have separate roles. R10 is governor/residual-memory tuning on one hard training distribution, `r10_l7_full_domain_randomisation_arena_wide_training`, with 140 outer cases per condition: four-fan geometry, active fan count 0/1/2/3/4, fan parameter uncertainty, arena-wide non-overlapping fan positions, and W3 plant/implementation perturbations. Arena-wide fan positions are rejection-sampled inside the tracker footprint with a 0.5 m safety radius around each fan, so fan-centre distances must be at least 1.0 m and safety circles do not overlap. Between R10 history launches, launch state and environment/fan realisations are resampled, but glider physical parameters, latency, and actuator implementation are sampled once per outer case and held fixed to keep memory learning interpretable. R11 is held-out validation on an eight-block fidelity ladder: L0 dry-air fixed, L1 single-fan fixed nominal, L2 four-fan fixed nominal, L3 fan-parameter uncertainty, L4 local fan-position uncertainty, L5 active-fan-count uncertainty 0/1/2/3/4, L6 environment-only full uncertainty, and L7 full-domain arena-wide randomisation. R11 L0-L2 keep the environment fixed between history episodes; L3-L7 resample environment/fan/launch histories, with plant/implementation perturbations only in L7 and still fixed per outer case. Core comparison is no memory versus residual-memory histories h3, h10, and h30, with h100 and safe-explore only as optional appendix ablations. Hardware readiness, real-flight transfer, mission success, autonomy, and memory-improvement claims require R11 and later real-flight evidence.

<!-- R9_LAUNCH_GATE_ALIGNMENT_END -->

## LQR-stabilised environment-conditioned primitive library after model-only restart

This plan integrates the full environment-conditioned primitive strategy with the model-only restart assumption. It is the active project contract for the Nausicaa glider control work.

The project first learns a rich library of short LQR-stabilised **primitive-controller variants** and only later compresses that library for real-time use. R5 W0/W1 is now the robust randomized synthesis/training stage: it randomises launch and in-flight primitive-start states, uses dry air plus annular-GP randomized training blocks for active pass evidence, applies W3-style implementation/plant variation to both dry-air and annular-GP rows, and tunes primitive-local LQR references and Q/R weights for generated variants. Axisymmetric Gaussian plume rows are retired from active R5/R7 acceptance evidence. W2 is retained only as an optional diagnostic or legacy comparison. Frozen R7 holdout replay then validates the R5 frozen bundle under dry-air plus held-out randomised annular-GP single/four conditions without retuning. Clustering and merging occur only after W3, producing the accepted post-W3 library-size condition used by the viability governor in real-time repeated-launch operation.

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

Build a simple, reproducible, robotics-style sim-to-real control programme for the following final question: can feedback-stabilised glider manoeuvre primitives be selected and transferred consistently across repeated launches when vehicle dynamics, implementation timing, and aerodynamic forcing differ between simulation and reality?

Primary research question:

> Can feedback-stabilised glider manoeuvre primitives be selected and transferred consistently across repeated launches when vehicle dynamics, implementation timing, and aerodynamic forcing differ between simulation and reality?

Equivalent implementation question:

```text
Can a compact primitive library transfer from simulation to repeated real launches
when primitive selection is conditioned on local flow context rather than on a
specific fan layout or arena-specific chain of actions?
```

Controller-level implementation question:

```text
Can each active primitive be developed as a primitive-controller variant whose
local time-invariant LQR stabiliser is tuned inside that primitive's own
entry-set/reference design during robust randomized R5 W0/W1 synthesis, then
validated by frozen held-out W3 replay before post-W3 library-size cross-study
and frozen full-loop validation?
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

A launch ending by safety-volume exit, wall-margin stop, tracking loss, or manual abort is not automatically a failed dataset. It is an episode outcome and must be logged as such.

---

## 3. Method in one chain

```text
measured state
+ local flow context
+ LQR-stabilised primitive catalogue
    -> primitive outcome model
    -> transition viability governor
    -> selected 0.10 s primitive
    -> entry_class -> exit_class transition log
    -> episode outcome log
    -> directional 3D residual updraft-gain update
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

The LQR stabiliser is part of the primitive. The active library item is a **primitive-controller variant**:

```text
primitive-controller variant
    = entry set
    + reference state/command
    + finite horizon
    + local time-invariant LQR Q/R/K
    + exit checks
    + failure labels
    + claim status
```

The primitive should not choose an external LQR controller from a free-standing controller bank. Each primitive develops its own local LQR variants during robust R5 W0/W1 synthesis. R5 preserves the generated primitive library as a rich evidence set rather than selecting a 12--24 variant shortlist. W2 may be run as an optional diagnostic, but it is not the accepted move-on gate. Frozen R7 dry-air plus W3 annular-GP holdout replay shrinks the library only by transition evidence and must not retune. Post-W3 library-size cross-study then compresses the surviving library toward a small set that can be evaluated efficiently in late simulation and, only after the required gates pass, considered as future real-flight candidates.


The active local controller is time-invariant LQR. The LQR stabiliser is part of the primitive, not an external controller bank. Each primitive-controller variant must expose a stable `primitive_variant_id`, `primitive_id`, `transition_entry_class` where evidence is entry-specific, `controller_id`, nominal reference state, nominal surface command, finite horizon, linearisation metadata, Q/R weights, LQR gain matrix, gain checksum, exit checks, metrics, failure labels, and synthesis/audit status. A primitive whose LQR synthesis fails must be marked `blocked` or `not_supported`; it must not silently fall back to the old PD-like bounded controller. TVLQR is outside the active workflow because the primitives are short and the additional implementation burden is not justified at this stage.

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

Active code reality:

```text
baseline model file       03_Control/02_Inner_Loop/glider.py
dynamics file             03_Control/02_Inner_Loop/flight_dynamics.py
mass-property constants   03_Control/02_Inner_Loop/A_model_parameters/mass_properties_estimate.py
runtime plant variation   03_Control/03_Primitives/plant_instance.py
```

The active baseline glider is a simplified rigid-body strip-aerodynamics model,
not CFD, not aeroelastic, and not a flight-identified aerodynamic database.  The
state is the canonical 15-state vector:

```text
x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r
```

The baseline as-built constants used by code are:

```text
mass_kg                 0.13356
R_CG_BUILD_M            [0.126, 0.0, -0.008940123518632263] m
INERTIA_B               [[0.0027016173, 0, 0.0000476285],
                         [0, 0.0024588056, 0],
                         [0.0000476285, 0, 0.0050785298]] kg m^2
```

The active aerodynamic model sums strip forces and moments from a finite-wing
section model: finite-wing lift slope, parabolic induced drag, hard-coded
surface `cd0` / `alpha0` / efficiency assumptions, thin-airfoil flap
effectiveness, a smooth post-stall blend around 12 deg, and lumped fuselage drag
at the CG.  These are analytical or engineering-code assumptions.  They are not
loaded from measured aerodynamic correction tables.

Plant randomisation, where enabled by the W layer, is applied by
`plant_instance.py` to the already-adapted aircraft model.  W3 plant instances
currently apply mass scale, diagonal inertia scaling, CG offset by shifting strip
moment arms, strip `cd0` scaling, and flap-scale surface-calibration scaling.
Cross-inertia terms are retained from the baseline and are not perturbed.

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

Active timing constants are held in `03_Control/03_Primitives/latency.py` under
`measured_vicon_one_pole_command_response_v1`:

```text
command_dt_s                  0.020
actuator onset latency        0.073 s
nominal half response         0.108 s
conservative actuator bound   0.151 s
Vicon nominal latency         0.0149 s
Vicon p95 latency             0.0169 s
Vicon filter delay            0.0080 s
Vicon filter model            one_pole, 20 Hz
```

The plant integrates first-order actuator surface states.  Direction-specific
surface limits are retained in the command/latency contract; saturation must be
audited from the unclipped raw command rather than hidden by command clipping.
The active aggregate surface limits are:

```text
delta_a   +22 / -26 deg
delta_e   +22 / -30 deg
delta_r   +28 / -35 deg
```

Audit:

```text
state delay is not confused with actuator lag
command delay is not confused with surface response time
nominal and conservative timing cases remain distinct
latency status is recorded in every evidence row
instant-feedback evidence is not promoted to true delayed-state-feedback evidence
```

The panel-wise glider model, state-feedback latency, command timing model, and actuator lag model are active from W0/W1 dense generation. They must not be postponed to W2, because they affect the primitive-local LQR tuning and the meaning of each `controller_id`.


### 4.3 Updraft and local-flow model

Retain:

```text
measured or fitted updraft surrogate
general fan-field / environment representation
wing-scale wind descriptors
local uncertainty descriptors
physically meaningful single-layer randomisation of fan position, active fan count, per-fan power, width/spread, and local uncertainty
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
dry-air zero-wind surrogate                      W0 no-updraft baseline inside R5 synthesis
randomised GP-corrected annular-Gaussian         active W1 annular-GP training blocks and W3 held-out validation
GP-corrected annular-Gaussian surrogate          optional W2 hardware-aware measured-updraft diagnostic replay only
axisymmetric Gaussian plume surrogate            retired diagnostic / historical comparison only, not active R5/R7 pass evidence
GP residual / empirical uncertainty              local uncertainty descriptor for annular-GP training, W2/W3, and conservative scoring
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

The operational primitive catalogue is deliberately compact. Launch is an
entry regime for these same manoeuvre families, not a separate primitive
family:

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

The retired `launch_capture_*` IDs are archive aliases only. They must not be
used in active R5/R7/R8/R10/R11 evidence.

Every active primitive is tested as a transition object across launch,
in-flight, boundary-near, and recovery-edge entry regimes. R7 decides which
`primitive_id + entry_class -> exit_class` transitions survive. Do not create a
launch-specific controller family to repair a gate.

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


### 5.1 LQR primitive synthesis contract

The active primitive implementation is LQR-only. The previous bounded PD-like controller is retired and should be removed from the active controller path after an archive/delete manifest is written.

Each active primitive controller must define:

```text
controller_family              lqr
primitive_variant_id           stable unique ID for the primitive reference, horizon, and controller package
controller_id                  stable unique ID for the primitive-local LQR controller
primitive_id                   active primitive ID
reference_state_vector         x_ref in the canonical 15-state order
reference_command_vector       u_ref in command/surface order
linearisation_id               deterministic identifier for A, B, x_ref, u_ref
linearisation_source           explicit trim or local speed operating point
Q_weight_json                  state penalty weights, with units/comments
R_weight_json                  command penalty weights, with units/comments
K_gain_matrix                  LQR feedback gain
K_gain_checksum                checksum for reproducibility
closed_loop_eigenvalue_summary stability audit summary
lqr_synthesis_status           solved / blocked / approximate
controller_claim_status        simulation_only / hardware_shakedown / real_flight_evidence / not_tested
```

Do not tune the raw gain matrix directly. Tune the LQR weight parameterisation using the active R5 structured log-space Q/R method: candidate 0 is nominal, candidates 1--7 are interpretable physical anchors, and candidates 8--31 are deterministic Latin-hypercube log multipliers over `q_attitude`, `q_velocity`, `q_rates`, `q_surfaces`, `r_aileron`, `r_elevator`, and `r_rudder`. `q_velocity` is a soft damping/stability term, not a speed-hold objective; the command law treats longitudinal speed error as passive so speed can vary naturally. Tuning is part of R5 primitive synthesis and produces transition objects, not a separate LQR bank. R5 must select candidates by entry-specific transition quality before R7; dense row count, local accepted rows, or aggregate success across entry classes are not sufficient.

This is an empirical robust-selection procedure, not a proof of global LQR
optimality for the nonlinear glider task.  Write about the selected controllers
as R5-trained or R5-selected primitive-local LQR variants.  Do not claim that
the Q/R table is mathematically optimal; it is a deterministic design-of-
experiments lookup surface that becomes credible only after transition evidence
and held-out R7 validation.

Active LQR synthesis must use local-speed scheduled operating-point
linearisation.  Bare `linearise_trim()` calls and optimizer-CSV trim seeds are
not active-method inputs.  The current local operating speed grid is:

```text
2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0 m/s
```

The feasible straight-trim seed grid is:

```text
4.8, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0 m/s
```

Operating points below the feasible trim range are local model points generated
from the nearest feasible trim and velocity-scaled to the requested local speed;
they are not claimed to be steady trim conditions.

Primitive-library learning policy:

```text
R5 W0/W1 robust randomized synthesis
    Generate and tune a rich primitive-controller library from mixed primitive-start
    states and W3-style randomized training conditions. W0 includes dry/near-dry
    baselines. W1 includes supported randomized single-fan, four-fan, and active
    fan-count cases. Tune each primitive-local LQR inside its own transition
    object using the structured 32-candidate Q/R plus primitive attitude/bank reference-bias method. Summarise every
    primitive_id + candidate_index + transition_entry_class by Wilson transition
    success and hard-failure bounds, worst-environment success, updraft-gain
    proxy, lift dwell, rollout duration, and saturation. Keep the full candidate
    bundle as audit evidence, but freeze only the selected transition objects for
    R7. Use
    the panel-wise glider model, feedback-latency lag, command timing model,
    actuator lag model, left/right aileron asymmetry, and CG-offset model from
    this stage where enabled. Preserve all variants, blocked cases, weak cases,
    and x-y terminal outcomes; do not select by aggregate local rollout success.

Optional W2 diagnostic sweep
    Replay the fixed R5 library under the single-fan and four-fan GP-corrected
    annular-Gaussian surrogate. Do not regenerate or retune LQR controllers.
    Remove, downgrade, or label variants for diagnostic reports only; W2 does
    not provide accepted move-on evidence.

R7 dry-air plus W3 frozen holdout sweep
    Replay exactly the frozen R5-selected transition objects under held-out domain randomisation of the
    updraft surrogate, fan positions and fan number, glider model, and
    timing/latency/actuator model. Do not regenerate or retune LQR controllers.
    Remove, downgrade, or label primitive_id + transition_entry_class objects
    that fail transition compatibility under randomisation.

Post-W3 library-size cross-study
    Cluster and merge only the W3-eligible library to obtain the final compact
    robust library for the governor and realistic in-flight computation.
```

If W3 exposes a failure mode, the current library item is eliminated, downgraded, or labelled in that survival pass. If optional W2 exposes a failure mode, the result is diagnostic unless the user explicitly promotes a new repaired R5 synthesis cycle. A redesigned controller is a separate future R5 W0/W1 learning cycle and must not mutate an already reported primitive-controller variant in place.

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
wall_margin_m                    all-wall minimum margin for hard-boundary audit
all_wall_margin_m                explicit duplicate of all-wall audit margin
front_wall_margin_m              heading-aligned forward wall clearance
left_wall_margin_m               heading-aligned left wall clearance
right_wall_margin_m              heading-aligned right wall clearance
rear_wall_margin_m               heading-aligned rear wall clearance
governor_wall_margin_m           governor admission margin, min(front, left, right)
floor_margin_m                   floor margin
ceiling_margin_m                 ceiling margin
attitude_margin_rad              margin to roll/pitch admission bound
latency_case                     ideal, actuator-lag-only, nominal, conservative
actuator_case                    nominal or conservative actuator response
```

The viability governor uses `governor_wall_margin_m`, not the all-wall
minimum, for preselection wall admission whenever heading-aware margins are
available. This prevents a forward launch from being rejected only because the
rear wall is close. Rollout simulation and post-stage safety audits still use
true all-wall/floor/ceiling boundary checks, so any actual wall crossing remains
a hard failure.

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
For active annular-GP R5/R7 evidence, `fan_power_scales` is the active vertical-strength perturbation, `fan_positions_m` is the active spatial/layout perturbation, and `updraft_width_scale` is the active spread perturbation. `updraft_amplitude_scale` is fixed at 1.0 unless an explicitly labelled residual-calibration diagnostic is requested, `updraft_centre_shift_m` is fixed at zero to avoid duplicating fan-position shift, and the extra `RandomisedWindField` wrapper is disabled for pass-gated annular-GP evidence.

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
psi   in [-20, 20] deg
phi   in [-20, 20] deg
theta in [-10, 20] deg
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
tracking loss
controller rejection
manual abort
controlled finish
```

Termination is an outcome label, not automatic mission failure.

For archive generation before post-W3 library-size cross-study, x-y wall or lateral safety-volume exit should be retained as a terminal outcome rather than used as a row-deletion rule. A finite, controlled x-y arena exit with positive floor and ceiling margin is `controlled_xy_boundary_terminal` and should map to `episode_terminal_useful` / `safe_terminal`, even when the no-updraft flight loses energy. Uncontrolled x-y exits are labelled `uncontrolled_xy_boundary_exit`; floor and ceiling violations remain safety-critical z-boundary failures and must be labelled separately.

Implementation must distinguish two non-z-boundary primitive uses:

```text
continuation_valid
    The primitive finishes in a chain-compatible exit class that can seed another primitive in the same simulated episode.

episode_terminal_useful
    The primitive may terminate at an x-y wall limit or lateral safety-volume edge while still controlled and airborne. It is retained for archive learning and repeated-launch evaluation, but not used as a downstream continuation state.
```

This distinction prevents the governor from rejecting every useful primitive merely because every launch eventually reaches the lateral boundary in the small arena. The archive records both labels; the selector decides later whether the current mode is continuation or terminal episode use.

Recommended hardware-only physical start trigger:

```text
Vicon rigid body valid for several consecutive frames
speed readiness logged by the launch system if required for hardware arming
state inside gate or accepted tolerance shell
controller has valid pose and command path ready
```

### 7.1 Primitive-start distribution for archive and replay

The fixed launch gate is the physical episode start, but it is not the only primitive-start state used for primitive outcome evidence. A primitive may be attempted at launch, or it may be attempted after a previous primitive has ended. Archive generation should therefore use a mixed primitive-start distribution and must label every rollout as a transition object, with an explicit `entry_class -> exit_class` pair.

Do not use this distribution to rebuild fixed primitive chains or reachable-state chains. The archive should sample independent primitive-attempt states and ask what happens if one primitive is attempted from that state and context. Single-primitive exit states may be resampled into an in-flight start-state pool, but they must be logged as independent primitive-start samples rather than as part of an optimised chain or arena-specific action sequence.

A first default archive mixture is:

```text
launch_gate                 40%   real episode starts and first primitive choices
inflight_nominal            25%   ordinary in-flight states after clean primitive exits
inflight_lift_region        15%   in-flight states near useful local updraft
inflight_boundary_near      10%   near-wall states for terminal-boundary and governor learning
inflight_recovery_edge      10%   high-attitude, high-rate, or recovery-margin states
```

The active in-flight velocity envelopes deliberately cover most of the
local-speed LQR scheduling grid instead of staying near one nominal trim.
Current synthetic start-state bounds are:

```text
inflight_nominal            u 3.0--8.2 m/s, v +/-0.35 m/s, w +/-0.25 m/s
inflight_lift_region        u 3.2--8.0 m/s, v +/-0.30 m/s, w +/-0.22 m/s
inflight_boundary_near      u 3.0--8.0 m/s, v +/-0.35 m/s, w +/-0.25 m/s
inflight_recovery_edge      u 2.2--5.2 m/s, v +/-0.45 m/s, w +/-0.35 m/s
```

These ranges are still start-state randomisation, not speed-command targets.
The LQR speed bin schedules the local model; the primitive controller does not
force the glider to hold a global trim speed.

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

The global dense sweep remains one auditable evidence run, but acceptance is
separated by transition entry class. Every active primitive is scheduled across:

```text
40 launch_gate
25 inflight_nominal
15 inflight_lift_region
10 inflight_boundary_near
10 inflight_recovery_edge
```

Launch-gate rows, in-flight rows, boundary-near rows, and recovery-edge rows
are reported as separate `entry_class -> exit_class` transition evidence. A
primitive may pass for `inflight_stable` and fail for `launch_gate`, or the
reverse. Accepted rows from one entry class cannot compensate for weak handoff
evidence in another entry class.

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
minimum_speed_m_s              audit field only; not a standalone hard-failure gate
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

Do not discard rollout rows before post-W3 library-size cross-study only because the simulated primitive reaches an x-y wall limit or lateral safety-volume edge. In the archive, this is a terminal outcome and a source of boundary evidence. Penalising or rejecting such a primitive for real execution belongs to the viability governor or later selector, not to the archive row-generation filter. A z-boundary violation, nonfinite state, or physically invalid rollout must remain a hard failure label.

Boundary-use labels must be assigned before post-W3 library-size cross-study:

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
entry_class
exit_class
transition_success_probability
hard_failure_probability
updraft_gain_proxy
flight_time
continuation_valid probability or label
episode_terminal_useful probability or label
```


The outcome model should keep continuation, terminal-use, and transition labels separate. A primitive that is locally accepted but exits to the wrong state class is not a chain-compatible primitive for its declared role.

The model must remain explainable enough to support a robotics-journal method section.

---

## 10. Viability governor

The governor is the transition and safety filter. It rejects primitives before execution if predicted, checked, or simulated behaviour violates:

```text
safety volume
floor / ceiling safety or hard arena boundary
attitude or incidence limits
surface limits
actuator saturation
valid LQR synthesis and closed-loop audit
primitive entry class
required exit class
transition compatibility
uncertainty margin
supported latency / LQR feedback mode
```

Primitive scoring is allowed only after viability filtering. The governor must reject a primitive if its LQR controller is missing, unstable under the recorded audit, outside its supported reference envelope, or marked blocked/approximate beyond the allowed claim status. It must not substitute an archived PD-like controller.

Low airspeed is not a governor rejection reason, recovery-route trigger, or
score/audit feature in the active validation logic. The rollout uses the
physical glider model and true arena bounds to decide whether the simulated
flight actually causes floor, ceiling, wall, nonfinite, or unrecoverable
terminal failure. Raw `minimum_speed_m_s` may remain in legacy rollout telemetry
for backward compatibility, but speed margin must not block, terminate, route,
or score a primitive.

The governor is where x-y boundary proximity is routed into `boundary_near` or recovery/safe-terminal transitions. Archive generation should preserve `episode_terminal_useful` x-y boundary evidence first, then the governor learns or applies conservative transition decisions from those labelled outcomes.

Governor admission is transition-entry based. The governor classifies the
current state, filters compact representatives by validated
`transition_entry_class`, rejects high hard-failure risk, scores transition
probability plus updraft gain plus flight time plus residual-memory correction,
executes the best transition object, and updates case-local residual memory.
Wall margin fields remain audit and rollout-boundary telemetry, but
`boundary_near` is a route state, not automatic failure. The active soft gain
terms are `updraft_gain_weight` and `terminal_updraft_gain_weight`; legacy
`energy_weight` names may be read from old frozen configs but must not be
emitted by new governor manifests. If a compact representative has no supported
outcome evidence, the rejection reason is `missing_outcome_evidence_for_candidate`
before any zero-probability test is applied.

Recovery transitions use a progress-aware route rule. A `recoverable_degraded`
state does not need to become fully stable within one 0.100 s primitive, but a
`recoverable_degraded -> recoverable_degraded` self-transition is chain-compatible
only when the rollout reduces attitude/rate recovery risk, does not collapse the
front/side boundary time margin, does not collapse floor margin, and keeps hard
failure risk low. `recoverable_degraded -> boundary_near` is recorded as route or
weak evidence for later safe-terminal handling, not as a full pass.

The governor must therefore support two explicit operating modes:

```text
continuation_mode
    reject primitives predicted to hit the x-y wall before their finite horizon or to exit without enough margin for another primitive

terminal_episode_mode
    allow an x-y terminal-useful primitive if it is predicted to provide useful lift capture, finite dwell, or energy retention before a controlled terminal outcome, while still rejecting z-boundary violation, nonfinite trajectories, and unsupported feedback or surrogate cases
```

The selected mode must be logged. This keeps the repeated-launch mission feasible without pretending that lateral-boundary termination is safe continuation evidence.

Repeated-launch routing is deterministic at the episode level:

```text
primitive_step_index = 0
    current_state_class = launch_gate
    route only to representatives validated for transition_entry_class = launch_gate
    start_state_family = launch_gate

primitive_step_index >= 1
    use the current simulated state to choose the route
    route by current state class and validated transition_entry_class
```

The governor does not need a separate "is this launch?" detector beyond `primitive_step_index = 0`. Later recovery routing is state-based: the system does not know whether the next primitive is the final allowed primitive, so recovery/safe-exit candidates become viable when the current state itself indicates degraded margin or boundary-near conditions. This prevents recovery primitives from being ignored until after it is too late.

Active utility:

```text
utility = transition_viability
        + updraft_gain_weight * predicted_updraft_gain_proxy
        + flight_time_weight * predicted_flight_time
        + residual_memory_weight * residual_updraft_correction
        - hard_failure_weight * hard_failure_probability
```

Keep the score simple enough to explain in one slide or algorithm box.

The governor must log both accepted and rejected candidates. Rejection logs are part of the evidence, not implementation clutter.

---

## 11. Repeated-launch belief

The belief layer estimates residual updraft gain for repeated launches inside one paired validation case. It is a small score modifier only; it cannot alter state classification, entry-class rules, or exit-class compatibility. A first auditable update is:

```text
b_{k+1}(r) = lambda b_k(r) + (1 - lambda) w_hat_k(r)
```

Thesis-core comparison:

```text
no_memory_baseline
directional_3d_residual_memory_h3
directional_3d_residual_memory_h10
directional_3d_residual_memory_h30
```

Safe-explore policies and older 14-condition matrices are ablation/appendix material only.

The belief proposes objectives or lift-seeking context. The viability governor still decides whether a primitive is allowed.

The repeated-launch objective is:

```text
improve lift capture, finite dwell, and energy retention across launches
while preserving safety margins and rejection logs
```

Do not present R9 as thesis-facing evidence. The thesis-facing repeated-launch stages are:

```text
R10 governor tuning
R11 held-out validation
```

---

## 12. Fidelity ladder, learning sequence, and late validation

W0--W3 are fidelity layers for primitive-library learning and later full-loop simulation validation. They are not online controller branches. The accepted learning order is now a two-stage evidence flow with W2 retained only as an optional diagnostic:

```text
R5 robust W0/W1 synthesis   rich primitive-controller library with per-variant LQR tuning under dry-air plus annular-GP W3-style randomisation
optional W2 diagnostic      fixed-LQR replay under hardware-aware annular-GP conditions, diagnostic only
R7 frozen dry-air plus W3 holdout        fixed-LQR replay under held-out domain randomisation, no retuning
post-W3 library-size cross-study/merging  final compact robust library for the governor
late validation             thesis-facing R10 governor tuning, R11 held-out validation, then Reality; R9 remains internal preflight only
```

Updraft surrogate use across the validation ladder:

```text
W0  no updraft; use dry air only
W1  randomized measured-updraft synthesis; active pass-gated R5 evidence uses annular-GP randomized single-fan and four-fan training blocks with active-fan-count variation where supported; Gaussian plume is diagnostic-only
W2  optional hardware-aware diagnostic replay; use panelwise wind and the GP-corrected annular-Gaussian surrogate only, carrying its residual or empirical uncertainty into the context features; do not treat it as a required pass gate
W3  frozen held-out environment-randomised replay; use the randomised GP-corrected annular-Gaussian surrogate only, randomising fan position, active fan count, per-fan power, width/spread, and local uncertainty on held-out seeds; do not add a second residual-field, amplitude, or centre randomisation wrapper
```


Primitive-controller variant learning across the fidelity ladder:

```text
W0/W1  generate and tune a rich primitive-controller library using mixed start states
       and dry-air plus annular-GP W3-style randomized training conditions; preserve all generated
       variants, blocked rows, weak rows, failures, and x-y terminal outcomes;
       do not compress to a 12--24 variant shortlist
W2     optional diagnostic replay of the fixed R5 library under hardware-aware
       annular-GP conditions; no LQR retuning or regeneration; no required
       move-on authority
W3     replay the frozen R5 library under held-out domain randomisation of
       updraft, fan layout/count, glider plant, latency, command timing, and
       actuator delay; no LQR retuning
post-W3 library-size cross-study/merge  compress only the W3-eligible library into the final compact set
future hardware shakedown  use only frozen primitive-controller variants whose claim status permits hardware-facing evidence collection
```

R5 W0/W1 synthesis and R7 dry-air plus W3 holdout form the accepted learning and library-shrinking ladder. W2 can still be useful for debugging the annular-GP bridge, but it cannot be the evidence stage that makes a weak launch library acceptable. Final full-loop validation is later and uses frozen variants, governor, selector, and memory logic. A redesign after W3 failure is a separate explicitly authorised R5 W0/W1 synthesis cycle, not an in-place W3/R8/R9/R10/R11 retune.

The implementation must record `updraft_model_id`, model source, residual / uncertainty descriptor, and environment metadata for every row. The online primitive selector should not branch on Gaussian versus GP-corrected model family; active accepted R5/R7 evidence is annular-GP or dry-air only, and Gaussian rows are diagnostic-only. The selector should use the resulting local context features and uncertainty labels. If the requested W-layer surrogate is unavailable, the run must write a blocked manifest rather than silently falling back to another surrogate.

### 12.1 W0 ??dry-air and perturbed launch baseline inside R5 synthesis

Dry-air and near-dry perturbed baseline inside R5 robust synthesis. It quantifies what each primitive does without updraft assistance while using the same panel-wise glider model, feedback-latency lag, command timing, actuator lag, launch-state perturbation, implementation-randomisation, and plant-randomisation assumptions as the rest of the accepted synthesis/holdout ladder.

Use W0 for:

```text
LQR trim/local-reference linearisation smoke
controllability and closed-loop eigenvalue audit
model and runtime debugging
dry-air energy-loss baseline
checking whether a primitive only works because of lift
paired comparison with measured-updraft and randomized-updraft runs
rich dry-air primitive evidence, not shortlist selection
launch_gate transition handoff-quality stress before W3 holdout
```

### 12.2 W1 ??randomized measured-updraft synthesis

Randomized measured-updraft synthesis using the annular-GP surrogate for active pass-gated evidence. W1 is paired with W0 as the second half of the robust R5 dense synthesis stage and is run for single-fan, four-fan, and active-fan-count variation where the surrogate can represent it honestly. Axisymmetric Gaussian plume remains available only as diagnostic or historical comparison evidence and cannot satisfy an R5/R7 acceptance gate.

Rules:

```text
W1 helps tune primitive-local LQR Q/R, reference, horizon, and entry-set parameters.
W1 is not filtered by W0 winners.
W0 and W1 may use paired start-state keys where useful.
W0 failure may be useful ablation evidence, not automatic W1 rejection.
W1 output is a rich primitive-controller library, not a 12--24 variant shortlist.
All solved, blocked, weak, failed, rejected, continuation-valid, and x-y terminal-useful rows are retained for the frozen W3 holdout and optional W2 diagnostic.
Fan layout is environment metadata, not an online branch.
Training seeds used in W1 must be separated from W3 holdout seeds.
W1 annular-GP training and W3 annular-GP holdout use the same randomisation dimensions but disjoint seeds/environment instances.
```

### 12.3 W2 ??optional hardware-aware diagnostic replay

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

W2 uses the GP-corrected annular-Gaussian surrogate, called annular-GP in the repository, for both single-fan and four-fan environment instances. It reduces the gap between cheap nominal simulation and hardware by replaying the fixed rich R5 primitive library with no LQR regeneration, no Q/R mutation, and no hidden reference retuning. Under the current robust-synthesis plan, W2 is optional diagnostic/legacy comparison evidence only. It may eliminate, downgrade, or label variants for debugging reports, but it must not be a required pass gate and must not be the accepted source for R8/R9/R10/R11. It must include accepted, weak, `episode_terminal_useful` x-y boundary, and informative failed cases; it must not be limited to W1 winners only.

### 12.4 W3 ??frozen held-out environment-randomised robustness replay

Environment-randomised robustness test. Randomise physically meaningful parameters:

```text
fan position
per-fan power
active fan subset
updraft width/spread
local uncertainty / residual descriptor
launch state
latency and actuator response
mass scale / CG-offset moment-arm shift / diagonal inertia scale
strip cd0 scale
flap-scale surface calibration
```

W3 is the accepted frozen holdout validation stage. It consumes the frozen R5 robust-synthesis library directly, or an explicitly traceable copy of that library, and uses held-out seeds/environment instances that were not used to tune Q/R or references. In addition to environment randomisation, W3 should include implementation and glider-plant randomisation as first-class logged inputs rather than hidden changes inside the dynamics code. Required W3 randomisation groups are:

```text
environment randomisation
    fan position
    per-fan power
    active fan subset through composed single-fan annular-GP kernels
    updraft width/spread
    local uncertainty / residual descriptor scale

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
    global inertia scale plus Ixx / Iyy / Izz diagonal scale
    strip cd0 aerodynamic coefficient scale
    flap-scale surface calibration scale
    cross-inertia status logged as not_perturbed
```

For four-fan W3 randomisation, active fan count must be explicitly scheduled and logged across 1, 2, 3, and 4 active fans rather than left to an implicit random mask. Single-fan W3 rows use one active fan. The active fan count audit must be written so fan-number coverage is visible before post-W3 compression.

Current implementation notes that must remain true:

```text
left/right aileron asymmetry is applied to the per-strip control mix
centre-of-gravity offset is applied by shifting aerodynamic moment arms
plant aero coefficient scale currently scales strip cd0, not a full aero database
surface calibration scale currently scales flap effectiveness
cross-inertia terms are not perturbed by W3 plant_instance
W3 still does not retune Q/R, K, reference, horizon, transition entry, controller ID, or primitive variant ID
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

W3 uses the randomised GP-corrected annular-Gaussian surrogate and is the main simulation test that the method is not tied to one fan layout. W3 must not regenerate, retune, or mutate LQR weights, references, gains, horizons, or entry sets in place. W3 output should be a pass/fail/weak label under uncertainty, not just a trajectory plot. W3 stresses frozen R5 variants under held-out randomised updraft parameters, fan positions, fan number, glider mass scale, CG offset, diagonal inertia scaling, strip `cd0` scale, flap-scale surface calibration, latency jitter, command timing, and actuator delay. The W3 survivors are then passed to post-W3 library-size cross-study.

### 12.5 Real flight

Real flight is fixed-gate repeated-launch validation with measured initial states. Real data must be paired with matched simulation before any transfer claim is made.

---

## 13. Simulation budgets after restart

Use small, meaningful runs first. Dense runs are allowed only after the foundation smoke tests pass.

| Stage | Preferred target | Fallback | Purpose |
|---|---:|---:|---|
| model audit smoke | 100--500 rows | 50 | check glider model, panel-wise wind, latency, command timing, actuator lag, wind, and storage |
| LQR synthesis smoke | 1--2 controllers per primitive | 1 primitive family | check trim/local reference, linearisation, Riccati solve, gain audit, and blocked labels |
| contextual primitive smoke | 1k--3k rows | 500 | check LQR primitive interfaces, mixed start-state sampling, x-y terminal labels, and W0/W1 row schema |
| R5 W0/W1 robust randomized primitive synthesis | active primitive catalogue x 32--128 variants x 100--300 paired tests, with W3-style randomized training blocks | active primitive catalogue x 16 variants x 50 tests | tune a primitive-local LQR for every generated variant and preserve the rich frozen library for W3 holdout; the active simplified catalogue has 8 primitive families, so the current 32-candidate/3-environment/100-test target is 76,800 rows |
| optional W2 annular-GP diagnostic sweep | 20k--60k rows if requested | 10k | replay the fixed R5 library in single-fan and four-fan annular-GP with panel-wise wind and timing realism; diagnostic labels only, no required move-on gate, no retuning |
| R7 dry-air plus W3 frozen held-out domain-randomised survival sweep | 30k--100k rows | 15k | replay frozen R5 variants under dry-air plus held-out updraft, fan-layout/count, plant, latency, command-timing, and actuator-delay randomisation; eliminate/downgrade failures without retuning |
| post-W3 library-size cross-study | all W3 survivors plus labelled weak/failure evidence | compact representative subset | compare heavy, balanced, light, super-light, and no-clustering/no-merging library-size conditions before accepting any online validation library |
| full-loop simulation validation | 100--300 episodes per policy | 50 | compare frozen governor/selector/memory policies across W0--W3 using the post-W3 library-size condition |
| future hardware shortlist | 5--10 candidates | 3 | choose candidate LQR primitives from the post-W3 selected library-size condition only after the full simulation gates permit hardware-facing work |

For R5 W0/W1 robust primitive synthesis and R7 dry-air plus W3 holdout, the default primitive-start mixture should combine launch-gate and in-flight primitive-entry states in a single table while preserving entry-role and start-state-regime separation in diagnostics and gates. LQR tuning is completed only inside R5 for each primitive-controller variant. W3 evaluates declared `controller_id` values rather than continuing the controller search. The required mixture is 40% launch-gate states and 60% in-flight states, with in-flight rows divided into nominal, lift-region, boundary-near, and recovery-edge cases unless a later source-audited schedule replaces those proportions. This is not a return to primitive-chain construction; it is a way to train each primitive on realistic launch and mid-flight entry states while keeping each row as one independent primitive attempt.

Do not run large all-arena archives. Large all-arena sweeps are not the main result and must not displace validation analysis, reporting, or any later hardware preparation that is explicitly permitted by passed gates.

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
without directional 3D residual memory
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

Clustering and merging occur only after frozen W3 survival replay. They are for compression, explanation, representative examples, future hardware-candidate shortlists, and figures. Clustering is not the online controller and must not be used after R5 W0/W1 to preselect a small library before W3. In the active workflow, clustering compresses W3-eligible LQR controller/outcome evidence and must not mix retired PD-like evidence, optional W2-only diagnostic evidence, or entry-role-incompatible evidence into active clusters.

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
transition_entry_class
regime_label
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

Episode-terminal-useful x-y boundary rows, including x-y wall-limit or lateral safety-volume exits, should form explicit clusters or failure summaries rather than being removed before clustering. They are not automatically hardware candidates, but they are essential for learning the context boundary where the governor should become conservative. Post-W3 library-size cross-study must not select only clean accepted rows; representative weak, episode-terminal-useful, and informative failed rows are needed so the outcome model and governor learn the operating boundary rather than only the easy region.

Cluster reports should keep `continuation_valid` and `episode_terminal_useful` cases visibly separate. Terminal-useful clusters may support repeated-launch lift-capture decisions, but they must not be promoted to downstream-continuation evidence.

---

## 17. Runtime, storage, and file-size contract

Housekeeping precedence: when result folders, naming, runtime, storage, cleanup, or file-size details conflict, `housekeeping_and_naming_rules.md` is the controlling document. This section keeps the project-plan method requirement that dense evidence must be chunked, resumable, compressed, worker-enabled, checksum-manifested, and file-size-audited.

This section is mandatory for all future dense/archive/thesis-scale simulation.

A run is dense if any condition is true:

```text
planned rollout rows >= 10,000
planned candidate rows >= 5,000
expected runtime > 30 minutes
expected uncompressed table size > 250 MB
used for thesis evidence, environment generalisation, envelope maps, clustering, optional W2 diagnostics, W3 replay, outcome models, or governor packages
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

R9/R10/R11 repeated-launch validators must also be worker-enabled. Parallelism is across independent final held-out schedule rows; each worker runs the policy/history launches for that row sequentially so directional memory causality is preserved. The parent process owns table partition writing, chunk manifests, file-size audits, and final pass/fail summaries. The validation runner must not impose a fixed primitive-count cap in R10/R11 full validation; both use a high simulation safety budget of 20 s per episode instead. R10 is a single hard full-domain arena-wide governor-tuning block with 140 outer cases per condition, active fan count 0/1/2/3/4, non-overlapping arena-wide fan positions, fan-parameter randomisation, and W3 plant/implementation perturbation sampled once per outer case. R11 is the eight-block held-out fidelity ladder with 160 outer cases per condition: dry air fixed, single fan fixed, four fan fixed, fan-parameter uncertainty, local fan-position uncertainty, active-fan-count uncertainty, environment-only full uncertainty, and full-domain arena-wide randomisation. R11 L0-L2 keep the environment fixed between history episodes; R11 L3-L7 resample environment/fan/launch histories, with plant/implementation perturbations only in L7 and fixed per outer case. Any `max_primitives_per_launch > 0` setting is a diagnostic cap and cannot satisfy the R10/R11 full gate. Short smoke runs may lower `max_episode_time_s` explicitly, but full R10/R11 should not. R9 is a quick internal preflight only: all five library-size cases, three fixed blocks, `no_memory_baseline` plus `directional_3d_residual_memory_h3/h10/h30`, 60 final held-out launches, 645 history launches, and a 10 s episode budget. The default worker backend for R9/R10/R11 is process-based for speed. The row-level rollout evidence keeps history and final launch episode summaries, selected-primitive execution rows, selector decisions, memory updates, and belief snapshots for plotting and thesis audit. The exhaustive all-candidate score table is compact by default: it retains the selected candidate, top-k viable candidates, transition-entry representatives, and rejection-reason representatives rather than every rejected candidate at every primitive step.

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
12_reproducibility
99_misc
```

Preferred abbreviations:

```text
ctx   environment context
prim  primitive
pol   policy
ep    episode
sim   simulation
rf    real flight
sr    sim-real replay
lqr   linear quadratic regulator
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
w01_dense_primitive_library.py
w2_annular_gp_survival.py
w3_domain_randomised_survival.py
post_w3_primitive_clustering.py
primitive_outcome_model.py
viability_primitive_selector.py
episodic_lift_belief.py
run_lqr_w01_dense_chunked.py
run_lqr_w2_survival_chunked.py
run_lqr_w3_survival_chunked.py
run_post_w3_cluster_merge.py
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
directional 3D residual lift belief
W0/W1 generation, W2/W3 survival filtering, post-W3 library-size cross-study, and full-loop validation
sim-real replay pairing
```

Avoid framing Chapter 6 around fan-layout cases, chain construction, PD-style controller tuning, or high-angle agile primitives as required final behaviour.

The results chapter should be structured around:

```text
model audit
W0/W1 rich LQR primitive-library evidence
W2 and W3 survival filtering evidence
post-W3 library-size cross-study evidence
outcome-model / lookup performance
full-loop simulation validation
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
