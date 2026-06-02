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
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode vicon-smoke --duration-s 5 --run-label F_vicon_smoke
```

Run a serial packet smoke test:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode packet-smoke --serial-port COM11 --run-label F_packet_smoke
```

Run armed closed-loop flight only after the Vicon and serial smoke tests pass:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode armed --serial-port COM11 --duration-s 20 --run-label F01
```

Before closed-loop flight, verify servo sign and receiver-channel order:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_surface_sign_check.py --mode armed --serial-port COM11
```

For repeated experiment blocks, edit `CURRENT_EXPERIMENT_CASE` at the top of
`01_Runtime\run_experiment_sequence.py`, then run:

```powershell
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_experiment_sequence.py
```

The sequence launcher stores results under
`05_Results/<case_id>/<session_label>/throw_001`, waits 20 s after each valid
throw, and automatically re-arms. Failed launch-gate attempts are stored under
`invalid_attempts`, do not count as throws, and do not update memory. The full
operator checklist is in `REAL_FLIGHT_EXPERIMENT_INSTRUCTIONS.txt`.

In armed mode, the controller first waits in an armed-ready state and sends
neutral commands. Active control and the flight record start only when the
first measured state satisfies the complete R5 launch window:

- `x_w in [1.2, 1.4] m`
- `y_w in [1.8, 2.2] m`
- `z_w in [1.4, 1.9] m`
- roll within `+-20 deg`
- pitch within `[-10, +20] deg`
- yaw within `+-20 deg`
- body-speed magnitude in `[3.0, 8.0] m/s`

The `x_w = 1.3 m` launch-plane crossing is still computed as a fallback and
diagnostic, but the real-flight start trigger is the first full-window-valid
sample. This avoids rejecting a good throw merely because the exact boundary
interpolation is slightly too fast or too high.

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
C:\ProgramData\miniforge3\python.exe 04_Flight_Test\01_Runtime\run_real_flight.py --mode vicon-smoke --vicon-offset-m 3.9 2.2 1.95 --vicon-yaw-deg 0 --duration-s 5 --run-label F_vicon_frame_check
```

## Flight Defaults

- Vicon host: `192.168.0.100:801`
- Vicon subject: `Nausicaa`
- Serial port: `COM11`
- Serial baud: `1000000`
- Vicon tracking poll period: `0.005 s` / `200 Hz`
- Vicon derivative/state filter: one-pole `8 Hz`
- Controller period: `0.100 s`
- Serial packet period: `0.020 s`
- Launch wait timeout: `8.0 s`
- Launch gate debounce: `3` consecutive approved frames
- Post-exit neutral tail: `0.30 s`
- Default library tier: `heavy_cluster`
- Selectable real-flight fallback tier: `balanced_cluster`
- Servo command authority: full `[-1.0, 1.0]`; the old `0.70` cap is not used here.
- Measured full-authority endpoints in the duplicated controller contract:
  aggregate `delta_a +19.3/-21.5 deg`, `delta_e +23.7/-32.0 deg`,
  and `delta_r +/-33.0 deg`.

## Deployment Library Tier

The first real-flight experiment uses `heavy_cluster` as the single active
deployment tier. R11 already compares all five library tiers, so the flight test
does not repeat the full clustering ladder. `heavy_cluster` is selected because
E01 real-flight-aligned validation preserves a compact deployment library with
defensible high-energy performance and bounded memory behaviour. The R11 E01
result is defensible rather than strict-pass, and shows that starts below
5.0 m/s remain the dominant failure mode.

`balanced_cluster` remains a fallback if additional primitive diversity is
needed during smoke testing. This selection is a deployment tradeoff, not a
claim that one compact library dominates every speed bin, environment ladder,
or repeated-launch policy.

Memory-enabled experiment cases keep one case-local 0.1 m flow-belief map
across valid throws only. No-memory cases keep `adaptive_memory_active=False`.
Fan Vicon rigid bodies named `Fan_1` to `Fan_4` are logged for evidence and
sanity checking, but fan positions do not create fan-layout-specific controller
branches.

## Vicon Arena Frame

The controller world frame uses the operational region
`x_w in [1.2, 6.6] m`, `y_w in [0.0, 4.4] m`, and
`z_w in [0.4, 3.5] m`. The real-flight default assumes the raw Vicon origin is
at the centre of that region, so raw Vicon `(0, 0, 0)` maps to controller world
`(3.9, 2.2, 1.95) m`.

The default axis convention is still `+X` forward, `+Y` left, and `+Z` up. If
the Vicon axes are yaw-rotated relative to the arena, use `--vicon-yaw-deg`.
The current `Nausicaa` rigid-body orientation check showed pitch and yaw signs
reversed relative to the controller convention, so the runtime applies
`attitude_signs = (1, -1, -1)` for `(phi, theta, psi)`: roll is unchanged,
physical nose-up becomes positive `theta`, and physical nose-right becomes
positive `psi`. Re-run `run_vicon_orientation_check.py` after any Vicon
rigid-body re-registration.
