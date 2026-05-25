# LQR Dense Evidence Restart Build Note

Status: retained next-step orientation note. This file is useful for quick
handoff context, but it is not the controlling contract. Use
`docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`,
`docs/Skills.md`, `docs/Python Coding Instruction.txt`,
`docs/MATLAB Coding.txt`, `docs/housekeeping_and_naming_rules.md`, and
`docs/local_validation_environment.md` for current requirements.

## Current Status

The v4.10-style outer-loop/governor result is not accepted as a move-on result.
It is diagnostic only. The current cycle must archive or relabel rejected
results as `diagnostic_not_passed`, repair any code paths that assume old
horizons or pure exploitation, and rerun W0/W1 dense generation before W2, W3,
post-W3 library-size studies, or repeated-launch validation are treated as
passing evidence.

Active control remains time-invariant LQR-stabilised primitives, but the active
evidence unit is now a short primitive-controller variant:

```text
finite_horizon_s = 0.100
controller_input_slots_per_primitive = 5
controller_input_update_period_s = 0.020
```

W0/W1 must preserve the rich primitive-controller library. W2 and W3 replay
fixed variants only. Post-W3 processing is a four-case library-size cross-study:
heavy clustering, balanced clustering, light clustering, and no
clustering/merging. Late validation freezes the selected library-size condition,
governor, selector, and directional 3D residual memory policy.

## Retained Foundation Work

The following foundations remain useful unless a current audit marks a specific
path retired:

- strict W0-W3 surrogate binding;
- deterministic environment instances for dry air, variable Gaussian plume,
  GP-corrected annular-Gaussian, and randomised annular-Gaussian layers;
- project start-state families with `paired_start_key` coverage;
- panel-wise wind, state-feedback latency, command timing, and actuator-lag
  rollout mechanisms;
- primitive-controller variant rows with controller IDs, gain checksums,
  Q/R metadata, timing metadata, exit checks, metrics, and failure labels;
- rollout rows that keep continuation-valid and episode-terminal-useful labels
  separate;
- boundary-use labels that preserve x-y terminal-useful evidence rather than
  deleting it before post-W3 analysis;
- compressed chunk partitions, checksums, runtime summaries, outcome summaries,
  file-size audits, and path-length audits.

The archived legacy evidence is documented in
`docs/lqr_restart_archive_manifest.md`. The first LQR foundation cleanup is
documented in `docs/lqr_foundation_audit.md`.

## Current Sequence

The next accepted evidence chain is:

```text
R0  archive rejected v4.10-style evidence as diagnostic_not_passed
R1  audit and rewrite misaligned code paths
R2  enforce 0.10 s primitive schema with 5 controller-input slots
R3  repair directional 3D residual memory and belief logging
R4  repair safe exploration/exploitation in the governor
R5  rerun W0/W1 dense primitive-library generation
R6  replay fixed variants through W2
R7  replay W2 survivors through W3
R8  run the four-case post-W3 library-size cross-study
R9  run repeated-launch fixed-case validation at histories 0, 5, 10, 20, 50, 100
R10 run changed-environment validation only after R9 passes
```

Missing R5 blocks W2, W3, post-W3, and late-validation claims. Missing W2 or W3
blocks post-W3 library-size processing. Missing four-case post-W3 evidence
blocks a final library-size choice. Missing repeated-launch learning curves
blocks hardware-readiness work.

## Active Local Commands

Use these checks as the baseline before or after a code change:

```powershell
.\.venv\Scripts\python.exe -m py_compile 03_Control/02_Inner_Loop/*.py 03_Control/03_Primitives/*.py 03_Control/04_Scenarios/*.py
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_w01_w2_w3_contract_audit.py
git diff --check
```

Stage-specific dense, source-audit, post-W3, governor, and repeated-launch
commands must follow `docs/Daily_Schedule.txt` and the current implementation
plan, not old R6/R8/R9 runner examples.

When local smoke or dense evidence remains under
`03_Control/05_Results/lqr_contextual_v1_0`, it must be explicitly allowed for
local validation and must not be pushed unless every file passes the 100 MB
file-size audit and path-length audit.

## Evidence Boundary

This implementation may support only simulation-backed thesis evidence claims.
It does not claim real-flight transfer, hardware readiness, mission success,
formal region-of-attraction guarantees, W3 robustness proof, memory improvement,
or governor performance improvement unless the corresponding current-stage
tables, manifests, reports, and claim-boundary notes exist.
