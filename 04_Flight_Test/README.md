# Nausicaa Real-Flight Runtime

This folder is the self-contained real-flight deployment copy. It does not call
`03_Control` or `C_Overall_Latency` during flight.

## Runtime Path

`Vicon rigid body Nausicaa -> canonical 15-state -> frozen governor/library controller -> 20 percent command lattice -> Nano33 IoT packet -> transmitter/receiver/servos`

The flight runtime tracks only the rigid body named `Nausicaa`. It does not use
control-surface Vicon markers, latency-identification events, response-fraction
analysis, or bench deflection diagnostics inside the in-flight loop.

## Layout

- `01_Runtime`: live Python runtime, Vicon reader, serial transmitter, logger, and safety monitor.
- `02_Controller`: duplicated online controller contracts from the simulation pipeline.
- `03_Frozen_Inputs`: duplicated frozen R8 library/outcome evidence, R5 controller bundle, and R10 governor config.
- `Nano33IoT_Transmitter`: copied Arduino transmitter firmware.
- `05_Results/<run_label>`: local experiment manifests, metrics, and reports.

## Commands

Install the serial dependency:

```powershell
C:\ProgramData\miniforge3\python.exe -m pip install -r 04_Flight_Test\requirements-flight.txt
```

Run a hardware-free dry run:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode dry-run --duration-s 2 --run-label F_dry_run
```

Run a Vicon-only smoke test:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode vicon-smoke --calibration-profile active --duration-s 5 --run-label F_vicon_smoke
```

Run a serial packet smoke test:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode packet-smoke --serial-port COM11 --run-label F_packet_smoke
```

Run armed closed-loop flight only after the Vicon and serial smoke tests pass:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode armed --calibration-profile active --serial-port COM11 --duration-s 20 --run-label F01
```

Direct armed mode refuses to run without either `--calibration-profile active`
or an explicit `--vicon-offset-m X Y Z`. Armed closed-loop mode also refuses to
run unless `03_Frozen_Inputs\deployment_evidence_manifest.json` confirms that
R5/R7/R8/R10/R11 were regenerated and frozen for the active calibration profile
hash. Open-loop neutral and calibration-data collection are not blocked by this
closed-loop evidence guard.

Before closed-loop flight, verify servo sign and receiver-channel order:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_surface_sign_check.py --mode armed --serial-port COM11
```

The surface check sweeps aileron, elevator, and rudder through the full 20
percent command lattice `[-1.0, -0.8, ..., +0.8, +1.0]` with live console
progress while packets are repeated to the Nano33 IoT transmitter.

For dry-air glider model calibration before regenerating controller evidence,
use the neutral and single-axis pulse workflow described in
`GLIDER_CALIBRATION_PLAN.txt`:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_vicon_frame_calibration.py
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_glider_calibration_sequence.py --block neutral_30
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_glider_calibration_sequence.py --block pulse_ladder_30
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_glider_calibration_sequence.py --block pulse_supplement_aileron_rudder_high
```

Frame calibration updates `01_Runtime\calibration_profile.py`; start the glider
calibration sequence as a separate new process so it reloads the updated
profile. It does not auto-start calibration throws.

This calibration workflow uses the same Vicon state adapter, launch gate,
Arduino packet path, and 20 percent command lattice as the real-flight runtime,
but deliberately does not call the governor, memory, primitive selector, or LQR
controller.

The current `03_Control` neutral SysID entry point is
`run_fit_neutral_dry_air_calibration.py`. It uses only neutral open-loop real
throws, starts sim-real replay from the measured state after a `0.10 s`
first-motion alignment window, selects held-out throws with a randomised
session-stratified split, runs with 8 workers, and can use a staged search over
simple loss scales plus Cm0/Cl0/Cn0-style aerodynamic moment-bias diagnostics.
Neutral control-surface trims stay disabled by default. The active checked-in
constants are still `neutral_dry_air_aligned_0p20_N07`; the richer
`n30_staged_coupled_moment_bias_rich_v2` run did not pass the promotion gate,
so pulse data are not fitted until the neutral open-loop dry-air mismatch is
resolved.

For repeated experiment blocks, edit `CURRENT_EXPERIMENT_CASE` at the top of
`01_Runtime\run_experiment_sequence.py`, then run:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_experiment_sequence.py
```

The sequence launcher stores results under
`05_Results/<case_id>/<session_label>/throw_001`, waits 5 s after each valid
throw or failed launch attempt, and automatically re-arms. Failed launch-gate
attempts are stored under `invalid_attempts`, do not count as throws, and do not
update memory. A launch-plane crossing faster than 2 m/s that fails the formal
launch gate is treated as a finished invalid attempt rather than waiting for the
full launch-timeout period. The full operator checklist is in
`REAL_FLIGHT_EXPERIMENT_INSTRUCTIONS.txt`.

In armed mode, the controller first waits in an armed-ready state and sends
neutral commands. Active control and the flight record start only after the
launch trigger is approved for `2` consecutive frames with sufficient SO(3)
body-rate confidence. The normal trigger is a measured state inside the
complete R5 launch window:

- `x_w in [1.2, 1.4] m`
- `y_w in [1.8, 2.2] m`
- `z_w in [1.3, 1.8] m`
- roll within `+-20 deg`
- pitch within `[-10, +20] deg`
- yaw within `+-20 deg`
- body-speed magnitude in `[3.0, 8.0] m/s`
- roll rate `p in [-1.2, +1.2] rad/s`
- pitch rate `q in [-1.2, +1.2] rad/s`
- yaw rate `r in [-1.8, +1.8] rad/s`
- SO(3) body-rate observer confidence at least `0.65`
- `2` consecutive approved frames

The runtime also evaluates the interpolated `x_w = 1.3 m` launch-plane crossing
with the same y/z, attitude, speed, body-rate, and confidence checks. Under the
current `2`-frame debounce, practical approval normally comes from full-window
samples, while the interpolated plane remains a fallback and audit trail for
fast crossings.

If this gate is not detected before `--launch-wait-timeout-s`, the flight record
is cancelled and no active controller-decision log is written.

The active flight record terminates at the first exit from the validated
operational region:

- `x_w in [1.2, 6.6] m`
- `y_w in [0.0, 4.4] m`
- `z_w in [0.4, 3.5] m`

After an exit, the controller does not attempt a separate recovery or level-trim
mode. It sends neutral commands for a short post-exit tail and closes the active
record.

If the Vicon origin or yaw alignment changes, override the arena transform:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_vicon_frame_calibration.py
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode vicon-smoke --calibration-profile active --duration-s 5 --run-label F_vicon_frame_check
```

## Flight Defaults

- Vicon host: `192.168.0.100:801`
- Vicon subject: `Nausicaa`
- Serial port: `COM11`
- Serial baud: `1000000`
- Vicon tracking poll period: `0.005 s` / `200 Hz`
- Vicon velocity filter: one-pole `8 Hz`
- Vicon body-rate observer: corrected SO(3) rotation-window observer
- Launch body-rate observer confidence: `>=0.65`
- Controller period: `0.100 s`
- Serial packet period: `0.020 s`
- Launch wait timeout: `8.0 s`
- Launch gate debounce: `2` consecutive approved frames
- Post-exit neutral tail: `0.30 s`
- Default library tier: `balanced_cluster`
- Selectable real-flight fallback tier: `heavy_cluster`
- Servo command authority: full `[-1.0, 1.0]`; the old `0.70` cap is not used here.
- Measured full-authority endpoints in the duplicated controller contract:
  aggregate `delta_a +19.3/-21.5 deg`, `delta_e +23.7/-32.0 deg`,
  and `delta_r +/-33.0 deg`.

## Deployment Library Tier

The first real-flight experiment uses `balanced_cluster` as the single active
deployment tier. R11 already compares all five library tiers, so the flight test
does not repeat the full clustering ladder. `balanced_cluster` is selected
because E01 real-flight-aligned validation favours the broader transition
diversity needed after the launch-rate, actuator-limit, and boundary-safety
updates, while remaining defensible for high-energy starts with bounded memory
behaviour. The R11 E01 result is defensible rather than strict-pass, and shows
that starts below 5.0 m/s remain the dominant failure mode.

`heavy_cluster` remains a compact fallback if runtime or library size becomes
the limiting deployment concern. This selection is a deployment tradeoff, not a
claim that one compact library dominates every speed bin, environment ladder,
or repeated-launch policy.

Memory-enabled experiment cases keep one case-local 0.1 m flow-belief map
across valid throws only. No-memory cases keep `adaptive_memory_active=False`.
Open-loop `.0` experiment cases are true neutral baselines: they still use the
same launch gate, Vicon state logger, fan tracker, serial packet path, exit
gate, and result folders, but after launch approval they send only neutral
commands, do not call the governor, and do not update memory.
Fan Vicon rigid bodies named `Fan_1` to `Fan_4` are logged for evidence and
sanity checking, but fan positions do not create fan-layout-specific controller
branches.

Experiment case suffixes are:

- `.0`: open-loop neutral baseline.
- `.1`: closed-loop balanced-cluster controller with no memory.
- `.2`: closed-loop balanced-cluster controller with case-local memory enabled.

## Vicon Arena Frame

The controller world frame uses the operational region
`x_w in [1.2, 6.6] m`, `y_w in [0.0, 4.4] m`, and
`z_w in [0.4, 3.5] m`. The real-flight default is the active calibrated Vicon
profile, not the old arena-centre placeholder. The current active offset is
`(3.912561157643868, 2.430125153483199, 0.0380633081971309) m`, with
`attitude_signs = (1, -1, -1)`.

The default axis convention is still `+X` forward, `+Y` left, and `+Z` up. If
the Vicon axes are yaw-rotated relative to the arena, use `--vicon-yaw-deg`.
The current `Nausicaa` rigid-body orientation check showed pitch and yaw signs
reversed relative to the controller convention, so the runtime applies
`attitude_signs = (1, -1, -1)` for `(phi, theta, psi)`: roll is unchanged,
physical nose-up becomes positive `theta`, and physical nose-right becomes
positive `psi`. Re-run `run_vicon_orientation_check.py` after any Vicon
rigid-body re-registration. The checker now verifies both pose signs and the
SO(3) observer rate signs: pitch-up motion should produce positive `q`,
right-roll motion positive `p`, and nose-right yaw motion positive `r`. It also
checks the measured Vicon sample rate and fails the preflight check if the
effective stream rate drops well below the requested `200 Hz`.

Every real-flight and glider-calibration manifest records the calibration
profile hash. This hash is the link between the Vicon transform used during
data collection and the frozen evidence chain used for closed-loop deployment.

Angular rates are estimated only after this attitude correction has been
applied. The runtime builds a corrected body-to-world rotation matrix and feeds
that into a small SO(3) rotation-window observer. The launch gate therefore
requires both the normal R5 launch bounds and a body-rate observer confidence of
at least `0.65` for `2` consecutive approved frames before active control starts.
One-frame Vicon rotation spikes are downweighted by local consistency, while
sustained high angular rates are kept as real motion and judged by the launch
rate bounds.
