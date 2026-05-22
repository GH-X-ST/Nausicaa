# R6-R8 Pipeline Upgrade Build Note

## Status

This pass keeps the selected Path A policy: code-plus-smoke validation only. Official R6/R7/R8 evidence runs are deferred and `03_Control/05_Results` remains a placeholder-only result root.

Implemented pipeline additions:

- strict W0-W3 surrogate binding;
- deterministic environment instances for dry air, Gaussian single/four, fan shift, fan power, active fan mask, amplitude, width, and centre-shift cases;
- environment-context rows carrying surrogate provenance and conservative nonzero uncertainty fallback;
- deterministic launch/envelope state samples carrying `state_sample_source`, `paired_start_key`, and `state_envelope_label`;
- actual state-delay, command-delay, and actuator-lag rollout mechanisms;
- bounded feedback primitive command evidence;
- feedback rollout rows with separate continuation-valid and episode-terminal-useful labels, saturation metrics, and boundary-use class;
- state/context/primitive/latency/uncertainty feature schema for the auditable primitive outcome model;
- ordered viability-filtered primitive selection with checked recovery fallback and `continuation` / `terminal_episode` modes;
- episodic lift-belief smoke support with lambda values `{0.0, 0.5, 0.8, 0.95}`;
- selector, W2 replay, and W3 generalisation smoke/report scaffolds with blocked claim manifests;
- worker-enabled contextual archive preflight.

The selected evidence policy remains temp-only. Tests may write archive outputs under pytest temporary directories, while the active result root remains `03_Control/05_Results/.gitkeep` only.

## Git State Caveat

The user noted that unrelated restored files outside `03_Control` are intentional. Restored support documents are preserved unless they break imports, active contracts, file-size rules, or active method gates. MATLAB and plotting guidance are retained after wording alignment.

## Interfaces Added

- `env_instance.py`: `EnvironmentInstance`, `EnvironmentRandomisationConfig`, `environment_instance_for_mode(...)`, `sample_environment_randomisation(...)`, and `environment_metadata_from_instance(...)`.
- `state_sampling.py`: deterministic archive state samples and measured-log compatibility helpers.
- `env_surrogate.py`: `SurrogateBinding`, `resolve_surrogate_binding(...)`, `validate_surrogate_ladder(...)`, and `wind_field_for_binding(...)`.
- `prim_ctrl.py`: bounded feedback commands for the eight active primitives.
- `prim_features.py`: `primitive_feature_record(...)`, feature rows, and feature schema version.
- `prim_model.py`: table/kNN-style primitive outcome prediction with continuation and terminal-use targets.
- `prim_select.py`: viability-filtered primitive selection with accepted and rejected candidate logging for `continuation` and `terminal_episode` modes.
- `episodic_lift_belief.py`: compact grid belief, update/query helpers, and belief snapshot rows.
- `run_ctx_episode_smoke.py`: temp-only repeated-launch smoke loop with belief before/after labels and no improvement claim.
- `run_primitive_selector_report.py`, `run_w2_replay.py`, and `run_w3_generalisation.py`: temp/smoke report and replay scaffolds.

`env_ctx.py`, `updraft_models.py`, `prim_roll.py`, and `run_ctx_archive.py` now propagate real environment-instance wind effects, surrogate status, rollout backend, evidence role, trajectory-integrity status, latency mechanism flags, continuation status, episode-terminal status, boundary-use class, saturation metrics, floor/ceiling/wall margins, compressed chunk partitions, process-worker execution, checksums, runtime summaries, outcome summaries, and file-size audits.

## Evidence Boundary

This pass may support only the claim that a temp-validated, strict-surrogate, feedback-backed contextual primitive pipeline scaffold exists. It does not claim R6 archive completion, R7 selector-report completion, R8 W2 replay completion, controller performance, W2/W3 robustness, real-flight transfer, hardware readiness, mission success, environment generalisation, or repeated-launch improvement.

## Path A Acceptance Status

- Pipeline ready: achieved when the retained tests pass, because the code paths exist, temp/smoke outputs are manifest-backed, and the active result root stays placeholder-only.
- R6 archive complete: deferred. This still requires a later approved local archive with at least 20k rows, W0/W1 coverage, environment instances, state-sample metadata, compressed partitions, checksums, runtime summary, outcome summary, and file-size audit.
- R7 selector report complete: deferred. This still requires a later report fitted from the R6 archive with validation splits, candidate logs, governor rejection reasons, calibration-style summaries, and simulation-only claim status.
- R8 W2 replay complete: deferred. This still requires a later W2 replay over representative R6 rows using GP-corrected annular-Gaussian wind plus active state delay, command delay, and actuator lag.

## Temp-Only Output Policy

Official local archive runs are deferred. The later local commands are:

```powershell
python 03_Control/04_Scenarios/run_ctx_archive.py --run-id 60 --rows 20000 --seed 60 --w-layers W0,W1,W2,W3 --env-modes dry_air,gaussian_single,gaussian_four,fan_shift,power_scale --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --rollout-backend model_backed_feedback --output-root 03_Control/05_Results/context_archive/r6_feedback_20k
python 03_Control/04_Scenarios/run_ctx_archive.py --run-id 61 --rows 40000 --seed 61 --w-layers W0,W1,W2,W3 --env-modes dry_air,gaussian_single,gaussian_four,fan_shift,power_scale --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --rollout-backend model_backed_feedback --output-root 03_Control/05_Results/context_archive/r6_feedback_40k
```

Those runs must preserve chunked execution, resumable chunk manifests, compressed table partitions, worker execution, checksums, table manifests, runtime summaries, outcome summaries, and the 100 MB generated-file limit.
