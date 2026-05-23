# LQR Dense Evidence Restart Build Note

## Status

This note supersedes the old reset-era build note. Active control is now
time-invariant LQR only, and new evidence uses the active LQR runner family.

Implemented active additions:

- strict W0-W3 surrogate binding;
- deterministic environment instances for dry air, Gaussian plume, GP-corrected
  annular-Gaussian, and randomised validation layers;
- project start-state families with `paired_start_key` coverage;
- actual state-delay, command-delay, and actuator-lag rollout mechanisms;
- LQR primitive command evidence with controller IDs, gain checksums, CARE
  residuals, sampled-data checks, and latency/actuator-lag survival labels;
- rollout rows with continuation-valid and episode-terminal-useful labels,
  saturation metrics, boundary-use class, and full entry-state logging;
- state/context/primitive/latency/uncertainty feature schema for the auditable
  primitive outcome model;
- ordered viability-filtered primitive selection with accepted and rejected
  candidate logging for `continuation` and `terminal_episode` modes;
- episodic lift-belief smoke support with lambda values `{0.0, 0.5, 0.8, 0.95}`;
- selector, W2 replay, and W3 generalisation smoke/report scaffolds with
  simulation-only claim manifests;
- worker-enabled LQR archive and replay preflight gates.

The archived legacy evidence is documented in `docs/lqr_restart_archive_manifest.md`.
Current synthesis and smoke evidence is documented in `docs/lqr_foundation_audit.md`.

## Interfaces Added

- `lqr_linearisation.py`: reduced-state mask, full/reduced linearisation rows,
  controllability ranks, and primitive reference construction.
- `lqr_controller.py`: full-state CARE attempt records, reduced-order synthesis,
  zero-position-gain expansion, controller IDs, gain checksums, command clipping,
  sampled-data checks, and latency survival labels.
- `lqr_tuning.py`: grouped log-scaled diagonal Q/R candidate schedules, hard
  gates, and soft objective terms.
- `run_lqr_tuning_sweep.py`: W0/W1 tuning smoke and Q/R ranking tables.
- `run_lqr_contextual_archive.py`: active LQR contextual archive entrypoint.
- `run_lqr_w2_replay.py`: W2 replay wrapper for active LQR rows.
- `run_lqr_w3_generalisation.py`: W3 replay/generalisation wrapper for active
  LQR rows.

Existing `env_ctx.py`, `updraft_models.py`, `prim_roll.py`, and
`run_ctx_archive.py` now propagate real environment-instance wind effects,
surrogate status, rollout backend, evidence role, trajectory-integrity status,
latency mechanism flags, continuation status, episode-terminal status,
boundary-use class, saturation metrics, floor/ceiling/wall margins, compressed
chunk partitions, process-worker execution, checksums, runtime summaries,
outcome summaries, and file-size audits.

## Evidence Boundary

This implementation may support only simulation-backed thesis evidence claims.
It does not claim real-flight transfer, hardware readiness, mission success,
formal region-of-attraction guarantees, or W2/W3 robustness unless the
corresponding replay table, manifest, and claim-boundary notes exist.

## Active Local Commands

Use these commands for the pre-dense go/no-go sequence:

```powershell
python -m compileall 03_Control
python -m pytest 03_Control -q
python 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
python 03_Control/04_Scenarios/run_lqr_contextual_archive.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
python 03_Control/04_Scenarios/run_lqr_w2_replay.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
python 03_Control/04_Scenarios/run_lqr_w3_generalisation.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
python 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --rows 500 --seed 1 --candidate-count 16 --paired-tests-per-candidate 50 --candidate-chunk-size 125 --workers 1 --max-workers 1 --storage-format csv_gz --compression-level 1 --stop-after-chunks 4 --repair-incomplete
```

When local smoke evidence remains under `03_Control/05_Results/lqr_contextual_v1_0`,
set `NAUSICAA_ALLOW_LOCAL_EVIDENCE_ROOT=03_Control/05_Results/lqr_contextual_v1_0`
for housekeeping tests.

## Dense Evidence Targets

R6 W0/W1 tuning targets are 8 primitives x 16-32 grouped Q/R candidates x
50-100 paired W0/W1 start keys, with the documented fallback of 8 x 8 x 25.
W2 and W3 are replay-only survival stages. Retuning after W2/W3 requires a new
controller ID and a return to W0/W1 tuning evidence.

Dense tables must remain chunked, resumable, compressed, checksum-manifested,
and under the 100 MB generated-file limit unless the user explicitly approves a
local-only exception.

