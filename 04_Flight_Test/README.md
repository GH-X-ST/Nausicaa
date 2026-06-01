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

## Flight Defaults

- Vicon host: `192.168.0.100:801`
- Vicon subject: `Nausicaa`
- Serial port: `COM11`
- Serial baud: `1000000`
- Controller period: `0.100 s`
- Serial packet period: `0.020 s`
- Default library tier: `balanced_cluster`
- Selectable real-flight tier: `heavy_cluster`
- Servo command authority: full `[-1.0, 1.0]`; the old `0.70` cap is not used here.
