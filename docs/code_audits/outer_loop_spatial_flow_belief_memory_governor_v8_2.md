# Outer-Loop Spatial Flow-Belief Memory Governor v8.2

Date: 2026-06-06

## Scope

This note records the post-v3.9 reporting and real-flight estimator-alignment
update for the outer-loop spatial-flow-belief memory governor. The controller,
primitive timing, memory shield, launch gate, and calibrated simulation model
are unchanged.

The change is reporting-only: R11 simulation and real-flight outputs now expose
both the final outcome score and the accumulated selected primitive scores that
were actually executed.

The real-flight Vicon state adapter also now estimates aggregate surface states
from command history using the same measured nominal command-onset delay and
one-pole actuator-lag envelope used by the simulation nominal latency case.
This affects the controller-facing estimated surface states and logs only; the
actual packet output remains the hardware command path.

## Simulation Reporting

Repeated-launch validation now writes compact posthoc score tables:

- `metrics/posthoc_final.csv`: one row per final held-out launch, using the
  existing `LAUNCH_SCORE_VERSION` launch-score fields.
- `metrics/posthoc_exec.csv`: one row per launch/policy, summing selected
  selector scores and score components for primitives actually executed.
- `metrics/posthoc_delta.csv`: paired deltas versus `no_memory_baseline` and
  the open-loop zero-command comparison baseline when a matched launch key
  exists.

Selector decision rows now keep the selected candidate score components needed
for the executed-score audit: base-library, mission, exploration, memory,
calibrated-regime mismatch, and total selected score.

These tables are compact metric files with short names. They are report-only
and excluded from the 0.100 s primitive-boundary controller compute budget.

## Real-Flight Reporting

Closed-loop real-flight `controller_decisions.csv` now records the same selected
score and component fields as the simulation selector log. Each throw writes a
compact `posthoc_throw.csv` row with:

- experiment case, controller mode, memory enabled flag, throw validity, and
  termination labels;
- launch speed and final observable specific energy when a final Vicon state is
  available;
- selected decision count and accumulated selected score components;
- real-flight memory-history bucket: `open_loop`, `no_memory`, `h0`, `h1_3`,
  `h4_10`, or `h11_30_plus`.

Experiment sequences additionally write `posthoc_session.csv` and add the same
summary to the session manifest. Open-loop neutral runs still emit zero
controller decisions, but they now produce explicit posthoc rows so open-loop,
no-memory, and memory-enabled cases can be audited with the same tables.

## Runtime Estimator Alignment

`FlightRuntimeConfig` now derives the real-flight surface-state estimator
defaults from `latency_case_config("nominal")`:

- surface command delay: measured nominal command onset plus transport delay;
- actuator tau: measured nominal one-pole command-response tau;
- timing model version: shared latency contract metadata.

`NausicaaViconStateAdapter` keeps a compact command history, samples the delayed
command with zero-order hold, and then applies the same first-order surface lag.
The same adapter change is mirrored in the `03_Control` copy used by
simulation-side tests/tools.

## Perturbation Consistency Follow-Up

The post-v3.9 perturbation audit keeps linear aerodynamic surface-authority
uncertainty on the plant side only. The active W3 plant perturbation multiplies
the alpha-regime scheduled surface authority with a global `0.75--1.25`
multiplier plus axis-specific `0.85--1.15` multipliers on
aileron/elevator/rudder.

Implementation-side surface-effectiveness scaling is retired and fixed at
`1.0` on all three aggregate axes so it cannot duplicate the plant-side
scheduled-authority uncertainty. Implementation perturbations still cover
timing, actuator lag, neutral bias, surface-limit clipping, command
quantisation, and left/right aileron asymmetry. Surface-limit clipping is
treated as actuator saturation, not as a second linear effectiveness multiplier.

The roadmap wording was updated accordingly: `control_mix` remains the
geometry/sign mapping, `flap_scale_strip`/surface calibration is retained
nominal for W3 plant instances, and W3 authority uncertainty is described as
global-plus-axis scheduled surface-authority uncertainty rather than old
control-mix effectiveness scaling.

## Documentation Audit

The active `.txt` and `.md` workflow docs now state that R11 and real-flight
reports should include final outcome scores and accumulated executed primitive
scores, not only success-rate summaries. The docs also state that these posthoc
tables are report-only, GitHub-safe, and outside the runtime control budget.
They also state that real-flight surface-state estimation uses measured nominal
command-onset delay plus one-pole actuator lag, and that implementation-side
surface-effectiveness scaling is retired to avoid duplicating plant-side
surface-authority perturbation.

Historical v3.6/v3.7/v3.8/v3.9 SysID audit files remain historical records and
are not rewritten as active v8.2 evidence notes.

## Claim Boundary

This update does not change governor scoring weights, primitive selection,
spatial-memory update logic, launch-gate behaviour, calibration constants,
surface-effectiveness schedules, or real-flight packet command output.

No R10/R11 validation, memory-improvement, hardware-readiness, real-flight
transfer, mission-success, full-autonomy, or new SysID claim is made by this
change.

## Checks

- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py 03_Control/tests/test_v411_repair_cycle.py --basetemp .pytest_tmp_posthoc_control -p no:cacheprovider`
- `python -m pytest 04_Flight_Test/04_Tests/test_flight_runtime_contract.py --basetemp .pytest_tmp_posthoc_flight -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py::test_v53_selector_decision_logs_score_components_for_posthoc_audit 03_Control/tests/test_v53_algorithm_contract.py::test_v53_posthoc_tables_report_final_and_executed_scores_with_paired_deltas --basetemp .pytest_tmp_posthoc_fast_control -p no:cacheprovider`
- `python -m pytest 04_Flight_Test/04_Tests/test_flight_runtime_contract.py::test_closed_loop_dry_run_holds_neutral_during_launch_handoff 04_Flight_Test/04_Tests/test_flight_runtime_contract.py::test_open_loop_neutral_dry_run_records_state_without_controller_or_memory --basetemp .pytest_tmp_posthoc_fast_flight -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_prim_roll.py --basetemp .pytest_tmp_posthoc_fast_roll -p no:cacheprovider`
- `python -m py_compile 03_Control/04_Scenarios/episode_selector.py 03_Control/04_Scenarios/run_repeated_launch_learning_curve.py`
- `python -m py_compile 04_Flight_Test/01_Runtime/frozen_flight_controller.py 04_Flight_Test/01_Runtime/run_real_flight.py 04_Flight_Test/01_Runtime/run_experiment_sequence.py`
- `python -m pytest 03_Control/tests/test_real_flight_io.py 03_Control/tests/test_latency_chain.py 03_Control/tests/test_latency_step_response.py --basetemp .pytest_tmp_latency_control -p no:cacheprovider`
- `python -m pytest 04_Flight_Test/04_Tests/test_flight_runtime_contract.py::test_real_flight_surface_state_estimator_matches_nominal_latency_contract 04_Flight_Test/04_Tests/test_flight_runtime_contract.py::test_vicon_rigid_body_adapter_uses_command_history_surfaces --basetemp .pytest_tmp_latency_flight -p no:cacheprovider`
- `python -m py_compile 03_Control/03_Primitives/real_flight_io.py 04_Flight_Test/02_Controller/real_flight_io.py 04_Flight_Test/01_Runtime/flight_config.py 04_Flight_Test/01_Runtime/run_real_flight.py 04_Flight_Test/01_Runtime/run_glider_calibration_sequence.py`
- `git diff --check`
