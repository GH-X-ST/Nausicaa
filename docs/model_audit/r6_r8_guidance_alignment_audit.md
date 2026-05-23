# R6-R8 Guidance Alignment Audit

## Scope

This audit covers active code paths and preserved support guidance for the feedback-backed contextual primitive evidence pass.

## Current Alignment

| Item | Status | Note |
|---|---|---|
| Strict W0-W3 surrogate ladder | fixed | W0 is dry air only; W1 Gaussian plume only; W2 GP-corrected annular-Gaussian only; W3 randomised GP-corrected annular-Gaussian only. Invalid requests produce blocked bindings or blocked rows, not fallback. |
| Feedback rollout evidence roles | fixed | Feedback, command-template diagnostic, smoke, and blocked feedback rows are explicitly labelled. |
| Boundary-terminal evidence | fixed | X/y wall or lateral boundary exits are retained as `boundary_terminal` episode evidence and are not continuation successes. |
| Hard failure labels | fixed | Floor/ceiling, nonfinite, corrupt, low-speed, and physically impossible initial states remain failed, rejected, or blocked evidence. |
| Outcome-model targets | fixed | Default training uses `lqr_rollout_candidate` rows and separates continuation success from terminal usefulness. Explicit `blocked_lqr_synthesis` rows are excluded by default. |
| Selector modes | fixed | Selector exposes `continuation` and `terminal_episode` modes with different gates. |
| Restored support docs | fixed | MATLAB, plotting, daily schedule, coding, and housekeeping guidance now describe W labels as validation layers, fan cases as environment instances, and boundary exits as retained terminal episode evidence. |
| Unrelated restored docs | intentionally_preserved_non_contract_support | Non-contract restored support material is preserved unless it breaks imports, active contracts, file-size rules, or active control-method gates. |
| Runtime/storage contract | fixed | Dense runs remain 8-worker capable, chunked, compressed, resumable, checksum-manifested, and file-size audited against the 100 MB project limit. |
| Environment instances and wind effects | fixed | Dry-air, Gaussian single/four, fan-shift, fan-power, active-mask, amplitude, width, and centre-shift cases produce real wind/context differences or blocked rows. |
| Archive state sampling | fixed | The archive scaffold now uses deterministic launch and envelope state samples with state-source, paired-start, and envelope labels. |
| Latency mechanics | fixed | Rollouts apply state delay, command delay, and actuator lag according to the selected latency case and log mechanism flags. |
| Feature schema and uncertainty | fixed | Primitive model rows include state, context, primitive, latency, and uncertainty features with conservative nonzero uncertainty fallback where needed. |
| Belief and report scaffolds | fixed | Temp-only lift-belief, selector-report, W2 replay, and W3 generalisation scaffolds emit manifests without claiming completed R6/R7/R8 evidence. |
| Mixed primitive-start sampling | fixed | Primitive starts now follow the 40% launch-gate / 60% in-flight mixture, with launch-gate compliance and in-flight provenance logged explicitly. |
| Canonical entry-state logging | fixed | Samples, rollout rows, selector rows, W2 replay rows, and W3 scaffold rows log the full canonical entry state for replay and audit. |
| Raw-coordinate anti-branch guard | fixed | Active selector, report, replay, model-summary, and primitive-naming paths are audited so raw x/y/z coordinates are not used as primitive names, branches, evidence group names, or primary coordinate bins. |
| W3 implementation and plant randomisation | fixed | W3 scaffolding includes explicit implementation/actuator and plant instances; supported mass, inertia, actuator, and surface perturbations are applied, while unsupported components are labelled blocked or approximate. |
| v1.3 stage isolation | fixed | The overnight driver records R6, R7, R8, and R9 independently as complete, fallback, partial, blocked, or deferred; later-stage blocks do not invalidate earlier completed evidence. |
| R6 W0/W1 boundary | fixed | R6 archive execution is constrained to W0/W1 only. W2 and W3 are reserved for replay/generalisation stages and are not mixed into the R6 contextual archive. |
| R8/R9 actual replay requirement | fixed | R8 and R9 completion require model-backed primitive rollout rows. Copied source labels and scaffold case tables cannot mark replay stages complete. |
| Stage evidence outputs | fixed | Dense stages write compressed partitions, table manifests, checksums, runtime/outcome/coverage/file-size summaries, blocked/approximate ratio summaries, and claim-boundary reports. |
| v1.4 run hardening | fixed | The preferred stage driver adds fail-fast preflight, dry-run schedules, stop-after-stage, resume/repair flags, stage-specific first-chunk projections, and status-manifest updates after every stage without launching the full overnight run during implementation validation. |

## Claim Boundary

This pass does not claim controller performance, W2/W3 robustness, real-flight transfer, mission success, hardware readiness, or repeated-launch improvement.
