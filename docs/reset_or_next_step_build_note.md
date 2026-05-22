# R6-R7 Strict Surrogate Model-Backed Build Note

## Status

This pass moves the R2-R5 contextual primitive scaffold into the first model-backed path:

- strict W0-W3 surrogate binding;
- environment-context rows carrying surrogate provenance;
- model-backed primitive rollout evidence;
- first auditable primitive outcome model;
- viability-filtered primitive selection scaffold;
- worker-enabled contextual archive preflight.

The selected evidence policy remains temp-only. Tests may write archive outputs under pytest temporary directories, while the active result root remains `03_Control/05_Results/.gitkeep` only.

## Git State Caveat

At the start of this pass, `git status --short` showed a modified active project plan and two stale untracked non-contract docs. The stale docs were removed because they were outside the active allowlist. The modified project plan was not reverted.

The user noted that unrelated restored files outside `03_Control` are intentional. This implementation leaves them untouched unless a future active import or housekeeping gate requires a focused change.

## Interfaces Added

- `env_surrogate.py`: `SurrogateBinding`, `resolve_surrogate_binding(...)`, `validate_surrogate_ladder(...)`, and `wind_field_for_binding(...)`.
- `prim_model.py`: table/kNN-style primitive outcome prediction with uncertainty and neighbour distance.
- `prim_select.py`: viability-filtered primitive selection with accepted and rejected candidate logging.

`env_ctx.py`, `prim_roll.py`, and `run_ctx_archive.py` now propagate surrogate status, rollout backend, trajectory-integrity status, floor/ceiling margins, compressed chunk partitions, worker execution, checksums, runtime summaries, outcome summaries, and file-size audits.

## Evidence Boundary

This pass may support only the claim that a strict surrogate model-backed contextual primitive interface and preflight scaffold exist. It does not claim controller performance, W2/W3 robustness, real-flight transfer, hardware readiness, mission success, or repeated-launch improvement.

## Temp-Only Output Policy

Official local archive runs are deferred. The later local commands are:

```powershell
python 03_Control/04_Scenarios/run_ctx_archive.py --run-id 60 --rows 20000 --seed 60 --w-layers W0,W1 --env-modes dry_air,measured_updraft --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --output-root 03_Control/05_Results/context_archive/r6_model_backed_20k
python 03_Control/04_Scenarios/run_ctx_archive.py --run-id 61 --rows 40000 --seed 61 --w-layers W0,W1 --env-modes dry_air,measured_updraft --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --output-root 03_Control/05_Results/context_archive/r6_model_backed_40k
```

Those runs must preserve chunked execution, resumable chunk manifests, compressed table partitions, worker execution, checksums, table manifests, runtime summaries, outcome summaries, and the 100 MB generated-file limit.
