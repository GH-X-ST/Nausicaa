# R6-R8 Feedback-Backed Contextual Primitive Build Note

## Status

This pass moves the R2-R7 contextual primitive scaffold into the first feedback-backed evidence path:

- strict W0-W3 surrogate binding;
- environment-context rows carrying surrogate provenance;
- bounded feedback primitive command evidence;
- feedback rollout rows with separate continuation-valid and episode-terminal-useful labels;
- first auditable primitive outcome model;
- viability-filtered primitive selection scaffold with `continuation` and `terminal_episode` modes;
- worker-enabled contextual archive preflight.

The selected evidence policy remains temp-only. Tests may write archive outputs under pytest temporary directories, while the active result root remains `03_Control/05_Results/.gitkeep` only.

## Git State Caveat

The user noted that unrelated restored files outside `03_Control` are intentional. Restored support documents are preserved unless they break imports, active contracts, file-size rules, or active method gates. MATLAB and plotting guidance are retained after wording alignment.

## Interfaces Added

- `env_surrogate.py`: `SurrogateBinding`, `resolve_surrogate_binding(...)`, `validate_surrogate_ladder(...)`, and `wind_field_for_binding(...)`.
- `prim_ctrl.py`: bounded feedback commands for the eight active primitives.
- `prim_model.py`: table/kNN-style primitive outcome prediction with continuation and terminal-use targets.
- `prim_select.py`: viability-filtered primitive selection with accepted and rejected candidate logging for `continuation` and `terminal_episode` modes.
- `run_ctx_episode_smoke.py`: temp-only repeated-launch smoke loop with placeholder memory labels.

`env_ctx.py`, `prim_roll.py`, and `run_ctx_archive.py` now propagate surrogate status, rollout backend, evidence role, trajectory-integrity status, continuation status, episode-terminal status, floor/ceiling/wall margins, compressed chunk partitions, process-worker execution, checksums, runtime summaries, outcome summaries, and file-size audits.

## Evidence Boundary

This pass may support only the claim that a strict surrogate feedback-backed contextual primitive evidence interface and preflight scaffold exist. It does not claim controller performance, W2/W3 robustness, real-flight transfer, hardware readiness, mission success, or repeated-launch improvement.

## Temp-Only Output Policy

Official local archive runs are deferred. The later local commands are:

```powershell
python 03_Control/04_Scenarios/run_ctx_archive.py --run-id 60 --rows 20000 --seed 60 --w-layers W0,W1,W2,W3 --env-modes dry_air,gaussian_single,gaussian_four,fan_shift,power_scale --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --rollout-backend model_backed_feedback --output-root 03_Control/05_Results/context_archive/r6_feedback_20k
python 03_Control/04_Scenarios/run_ctx_archive.py --run-id 61 --rows 40000 --seed 61 --w-layers W0,W1,W2,W3 --env-modes dry_air,gaussian_single,gaussian_four,fan_shift,power_scale --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --rollout-backend model_backed_feedback --output-root 03_Control/05_Results/context_archive/r6_feedback_40k
```

Those runs must preserve chunked execution, resumable chunk manifests, compressed table partitions, worker execution, checksums, table manifests, runtime summaries, outcome summaries, and the 100 MB generated-file limit.
