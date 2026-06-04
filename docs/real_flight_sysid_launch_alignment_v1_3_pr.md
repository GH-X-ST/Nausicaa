Real-Flight SysID Launch Alignment v1.3

```text
Python control-simulation and flight-test calibration pipeline

- Aligns real and synthetic launch handling around the body-axis gate: `4.0 <= u <= 8.0 m/s`, `|v| <= 1.5 m/s`, and `|w| <= 0.9 m/s`, while preserving the existing attitude, body-rate, and 2-frame approval checks.
- Keeps live calibration output operator-readable by reporting `u`, `v`, and `w` instead of only total speed.
- Extends neutral dry-air SysID replay to the current staged 0.10 s workflow with 8-worker execution, session-stratified held-out splits, optional coupled longitudinal search, and Cm0/Cl0/Cn0-style aerodynamic moment-bias diagnostics while keeping neutral surface trims disabled by default.
- Fixes the staged candidate search so each stage can retain the current best parameter set instead of being forced to move when the current value is not in a later coordinate grid.
- Runs the rich n30 neutral replay over 80 valid real throws with 16 held-out throws; the best coupled moment-bias run still leaves pitch, sink, altitude, and dx residuals above the acceptance scale, so the active dry-air calibration constants are deliberately not updated.
- Keeps the sustained single-axis control-effect ladder as the next data-collection design: 0.15 s onset, long command window until wall/floor exit, elevator/aileron/rudder at +/-0.2 through +/-1.0, and 3 valid throws per command case.
- Aligns flight-facing docs and project docs around the current launch gate, 0.10 s neutral replay, staged fitting workflow, failed-rich-fit claim boundary, and sustained command-effect tests.
- Checks run: rich staged no-trim fit, trim diagnostic fit, full moment-bias fit, coupled moment-bias fit, focused py_compile, focused state/control/flight-runtime pytest suite (`56 passed` with short Windows temp root), stale-doc rg check, and `git diff --check`.
- No new active neutral dry-air calibration, pulse-derived control-effectiveness fit, hardware-readiness, real-flight transfer, mission-success, or full-autonomy claim is made by this change.
```
