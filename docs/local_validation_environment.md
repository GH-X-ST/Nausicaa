# Local Validation Environment

Date: 2026-05-25

Status: active environment note. The workflow details are controlled by
`docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`,
`docs/Skills.md`, `docs/Python Coding Instruction.txt`,
`docs/MATLAB Coding.txt`, and `docs/housekeeping_and_naming_rules.md`.

This repository uses one project-owned virtual environment for active Python
work:

```text
.\.venv\Scripts\python.exe
```

Do not use the old `Paul_Li_FYP` Conda environment for active validation or new
development work. Historical audit notes may mention it as a past fallback, but
it is not the active project environment.

The current `.venv` was created from the Miniforge base interpreter:

```text
C:\ProgramData\miniforge3\python.exe
```

Active control validation dependencies are installed from:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-control-dev.txt
```

Whole-repository development should still use the same `.venv`. If design-side
code is being run, install the aggregate development dependencies into the same
environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Do not create or switch to a second named environment for `02_Glider_Design` or
`03_Control`. The dependency files define which packages are needed; the active
interpreter stays the same.

Required active validation baseline:

```powershell
$files = Get-ChildItem -Path 03_Control/02_Inner_Loop,03_Control/03_Primitives,03_Control/04_Scenarios -Filter *.py -File | ForEach-Object { $_.FullName }
.\.venv\Scripts\python.exe -m py_compile @files
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v411_source_audit.py --dry-run --no-write-archive
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v53_algorithm_contract_audit.py
git diff --check
```

The default pytest command is now the fast regression tier. Slow
pipeline/archive integration tests are marked `slow` and skipped unless
explicitly requested. Run them only before dense evidence regeneration or when
touching archive/replay orchestration:

```powershell
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --run-slow -m slow --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
```

`run_active_contract_audit.py` and W01/W2/W3-only audit names are retained only
as compatibility references. New instructions should name the active source
audit and the current transition-aware algorithm contract audit directly. The current evidence
workflow is controlled by `docs/Glider_Control_Project_Plan.md`: R5 is robust
primitive synthesis, R6/W2 is archived diagnostic-only, R7 is frozen W3
validation, R8 is the five-case coverage-aware medoid post-W3 library-size
study, R9 is internal quick fixed-case preflight/ablation only and is not
thesis-facing evidence, R10 is single-block full-domain arena-wide governor
tuning, and R11 is strict held-out eight-block fidelity-ladder validation. R10
tunes mission/risk weights plus memory sensitivity, shield margins, exploration
thresholds, residual caps, confidence observations, and recency half-life; R11
freezes that handoff for validation. R9 defaults to all five
library-size cases, no-updraft/single-fan/four-fan fixed blocks, no-memory plus
h3/h10/h30 recency-weighted spatial flow-belief memory, 60 final held-out launches, and 645 history launches.
The R9/R10/R11 governor uses the same candidate-path geometry for no-memory and
memory policies: forward progress to `x_w = 6.6 m`, front-wall terminal proxy,
progress-gated terminal total specific-energy proxy, wrong-boundary avoidance,
and then updraft/lift plus optional spatial-memory correction after unchanged
safety and transition-entry filters. The current governor also applies calibrated
normal / transition / post-stall regime-mismatch risk as a bounded score penalty
and memory-shield non-regression diagnostic. The risk boundary is read from the
active `neutral_dry_air_calibration.py` residual-blend limits, currently
14--18 deg for the promoted
`neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal`
model, and candidate/selector logs record the source calibration ID plus boundary
values. High-AoA lift exploitation remains possible only when it earns enough
mission value to justify the active-model mismatch exposure.
R9/R10/R11 final launch scoring uses the current front-wall mission score:
front-wall terminal success at `x_w = 6.6 m` with y/z inside the true safe
bounds is the main success component, updraft-gain and lift-dwell terms remain
capped lift-usefulness evidence, terminal total specific-energy reserve is
rewarded only after front-wall success, and airborne time/net/gross energy drift
remain audit-only fields.
R9/R10/R11 and real-flight reporting now include compact posthoc score audits in
addition to success-rate summaries. Simulation runs write
`metrics/posthoc_final.csv` for one-row-per-final-launch `LAUNCH_SCORE_VERSION`
outcomes, `metrics/posthoc_exec.csv` for the accumulated selected primitive
scores and score components that were actually executed, and
`metrics/posthoc_delta.csv` for paired deltas versus no-memory and open-loop
baselines. Real-flight throws write `posthoc_throw.csv`, session runs write
`posthoc_session.csv`, and closed-loop `controller_decisions.csv` carries
selected score component fields; open-loop neutral still has zero controller
decisions but an explicit posthoc row. These tables are report-only, use short
GitHub-safe filenames, and stay outside the real-time controller compute
boundary. Completed dry-air real-flight workflow records now include redo E1.0
`20260607_190445`, the open-loop neutral baseline with 10 valid throws, 6
launch-gate rejected starts, zero controller decisions, speed range
5.016--6.653 m/s, mean final observable specific energy 1.634 m, 3 front-wall
exits, and 7 floor exits; E1.1 `20260606_230007`, the closed-loop no-memory
baseline with 30 valid throws, 16 launch-gate rejected/timeout starts, active
controller decisions on all valid throws, speed range 5.295--6.841 m/s, mean
final observable specific energy 1.747 m, 28 front-wall exits, and 2 floor
exits; and E1.2 `20260607_122640`, the dry-air memory null test with 30 valid
throws, 10 launch-gate rejected starts, active controller decisions on all valid
throws, speed range 5.287--6.774 m/s, mean final observable specific energy
1.771 m, 21 front-wall exits, 9 floor exits, 297 final memory cells, and 339
memory updates. These are workflow and posthoc audit records; E1.2 is
interpreted as a bounded dry-air memory null/safety test, not a fan-updraft
memory-improvement claim.

Completed fixed single-fan E2 records now include E2.0 `20260607_163303`, the
open-loop neutral baseline with 10 valid throws, 8 launch-gate rejected starts,
speed range 5.626--6.260 m/s, mean final observable specific energy 1.506 m, 6
front-wall exits, and 4 floor exits; E2.1 `20260607_165533`, the closed-loop
no-memory run with 30 valid throws, 10 launch-gate rejected starts, active
controller decisions on all valid throws, max decision time 0.03197 s, speed
range 5.053--6.412 m/s, mean final observable specific energy 1.635 m, 24
front-wall exits, and 6 floor exits; and E2.2 memory sessions
`20260607_173345` plus `20260607_175359`, collected as two 30-valid-throw runs
with 60 valid rows total, 19 rejected starts, max decision times 0.08695 s and
0.07169 s, mean final observable specific energy 1.625 m, 44 front-wall exits,
14 floor exits, and two blank terminal rows, one of which is the logged
non-controlled launch-handoff abort
`launch_handoff_abort:vicon_invalid:vicon_subject_occluded`. E2.2 memory rows
retain positive selected and memory score components, but the raw
E2.2-versus-E2.1 comparison is speed-confounded and remains a transfer and
diagnostic record rather than a broad standalone memory-improvement proof.

Current replay diagnostics are stored under
`04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/` and
`04_Flight_Test/A_figures/real_flight_sim_replay_E2_random_samples/`. The E1
sample set plots E1.0 redo `20260607_190445` throws 005 and 010, E1.1
no-memory `20260606_230007` throws 005 and 011, and E1.2 memory
`20260607_122640` throws 002 and 029. The E2 sample set plots E2.0
`20260607_163303` throws 001 and 002, E2.1 `20260607_165533` throws 016 and
027, and two throws from each completed E2.2 memory session. The plotting code
accepts open-loop neutral throws without `controller_decisions.csv`; for those
cases, the `sim_real_decisions` replay has an empty selected-primitive sequence
and remains neutral after the measured 0.040 s handoff splice. The first-0.040 s
state-splice residual audit has maximum `max_abs = 0` across the sample sets.
Replay version `real_flight_sim_replay_measured_fan_updraft_v2` uses W0 zero
wind for no-visible-fan E1 samples and measured-fan W2 annular-GP updraft for
E2 samples. Replay figures are model-mismatch, timing, and decision-consistency
diagnostics; they do not recompute a simulation-side mission score. The real
selected-score audit remains runtime/posthoc-only: E1.1 mean accumulated
selected score is `-1.2406162083333327`, E1.2 is `0.8705906510995837`, and
E1.0 is `0.0` because open-loop has no selected decisions.
R9/R10/R11 also write a repeated-launch real-time scheduler profile: context
construction, the cheap `geometry_only` candidate-path pre-pass, shortlisted
spatial-belief queries, and compact-library selection are measured against
a preferred 20 ms controller-slot budget and a hard 0.100 s primitive-boundary
budget. Launch-gate step 0 is the physical-duration exception: simulation and
real flight share `launch_gate_neutral_handoff_0p040s_v1`, so both hold
neutral/open-loop for exactly 0.040 s (two 20 ms slots), prepare the first
primitive from the approved launch-gate state, then start the unchanged 0.100 s
active primitive from the latest post-handoff state; step-0 evidence reports
0.140 s physical duration. Real-flight Vicon state packing estimates aggregate
surface states from the command history using the same measured nominal
command-onset delay and one-pole actuator lag envelope as the simulation nominal
latency case; packet output itself remains the hardware command path, not a
simulated delay. Later real-flight in-flight decisions use the same hybrid
scheduler as the deployment runtime: a predicted boundary state selects the
next primitive before commit, the latest Vicon state emits the first command
packet at commit, and late preparation is logged instead of becoming a new
flight blocker. This is an offline wall-clock audit, not a hardware real-time
claim. The current compact controller-row selector timing boundary matches the
active code path: context construction, the cheap `geometry_only`
candidate-path pre-pass, shortlisted spatial-belief queries, and compact
controller-row library selection are in the timed controller path, while full
candidate-row expansion, table flushing, and post-hoc diagnostics are outside
it. If step-0 memory selection misses the fixed 0.040 s launch handoff, the
runtime logs `first_launch_decision_missed_handoff_budget`, keeps the
launch-gate approval in the record, and marks the attempt non-valid for
controlled evidence. Latest targeted C16 sanity check: 40 final launches, 430
history launches, 0 hard failures, 0 no-viable events, 13/13 accepted memory
switches, and required heavy/balanced in-flight decisions at 144/144 under
0.100 s with max 0.0937 s; this is targeted diagnostic evidence only, not full
R10/R11 validation. The post-v4.4 real-flight memory timing probe used 159
synthetic memory cells and 30 launch decisions, with max first-decision time
0.0239 s and zero calls above 0.040 s; the completed E1.2 dry-air memory null
run then recorded max step-0 first-decision time 0.0386 s, max decision time
0.0570 s, and zero decisions above 0.100 s. These are engineering timing and
workflow checks, not new R10/R11 or memory-improvement claims.

Use the repo-local pytest temp/cache paths above so validation does not depend
on the Windows user temp directory. Local `.venv` and `.codex_run_logs`
contents are ignored and must not be committed. Generated evidence roots must
also follow the 100 MB file-size rule, path-length audit, and local-only result
handling in `docs/housekeeping_and_naming_rules.md`.

`aerosandbox` is not part of active `03_Control` validation. It remains isolated
in `requirements-design.txt` for glider-design-side work and is installed only
when whole-repository or design-side validation is needed. The active control
route requires `casadi`, `pytest`, and `openpyxl` through
`requirements-control-dev.txt`.
