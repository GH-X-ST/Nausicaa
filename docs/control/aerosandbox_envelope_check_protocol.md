# AeroSandbox Envelope Check Protocol

Date: 2026-05-17

## Purpose

This workflow is a low-alpha attached-flow sanity check for the reset
`03_Control` Nausicaa model. It compares the current strip/panel model against a
minimal AeroSandbox model when AeroSandbox is installed, and otherwise produces a
local-model-only audit.

This is not high-incidence validation and does not validate agile-reversal
fidelity. High-incidence evidence must come later from OCP replay and real Vicon
flight logs.

## Command

```powershell
python 03_Control/02_Inner_Loop/run_aerosandbox_envelope_check.py `
  --output-root 03_Control/05_Results/00_model_audit/aerosandbox_envelope/001
```

Optional arguments:

```text
--alpha-min-deg
--alpha-max-deg
--alpha-step-deg
--speeds 4.5 5.5 6.5 7.5 8.5
--betas -6 0 6
```

## Envelope

Default alpha values are `-8` to `12` degrees in `2` degree steps. Default beta
values are `-6`, `0`, and `6` degrees. Default speeds are `4.5`, `5.5`, `6.5`,
`7.5`, and `8.5` m/s.

Controls include the neutral clean case plus one-axis perturbations of `-10`
and `+10` degrees for elevator, aileron, and rudder. All evaluations are
zero-wind.

## Outputs

Default output directory:

```text
03_Control/05_Results/00_model_audit/aerosandbox_envelope/001/
```

Expected files:

```text
local_envelope_coefficients.csv
aerosandbox_envelope_coefficients.csv
pointwise_comparison.csv
comparison_summary.csv
manifest.json
report.md
figures/cl_vs_alpha.png
figures/cd_vs_alpha.png
figures/cm_vs_alpha.png
figures/drag_polar.png
figures/cl_alpha_by_speed.png
figures/local_control_derivatives.png
```

`aerosandbox_envelope_coefficients.csv` is written only when AeroSandbox is
available and an analysis can run.

## Interfaces

The script exposes these public functions:

```python
build_verification_grid()
local_model_coefficients(grid)
aerosandbox_coefficients(grid)
estimate_slope(df, x_col, y_col, selector=None)
compare_envelope(local, aerosandbox)
run_aerosandbox_envelope_check(...)
```

The local model is evaluated only through `build_nausicaa_glider()`,
`adapt_glider()`, and `evaluate_state()`. No plant, trim, linearisation,
primitive, controller, governor, mission, Vicon, or hardware code is changed by
this workflow.

## Conventions

The velocity convention is:

```text
u = V cos(alpha) cos(beta)
v = V sin(beta)
w = V sin(alpha) cos(beta)
```

This matches the active model convention `alpha = atan2(w, u)`. Coefficients use
the current `03_Control` reference area, span, chord, density, and zero-wind
aerodynamic force/moment output.

The comparison scope field is always:

```text
low_alpha_attached_flow_sanity_only
```

The high-incidence validation claim field is always:

```text
false
```
