# Longitudinal Moment Audit Protocol

## Purpose

This workflow audits the low-alpha longitudinal pitching-moment behaviour of the reset `03_Control` model before controller work resumes. It decomposes aerodynamic strip forces and moments by lifting surface and records whether the current positive `Cm_alpha` trend is explainable from local-model surface contributions.

## Command

```powershell
python 03_Control/02_Inner_Loop/run_longitudinal_moment_audit.py --output-root 03_Control/05_Results/00_model_audit/longitudinal_moment/001
```

Optional CLI inputs are `--alpha-min-deg`, `--alpha-max-deg`, `--alpha-step-deg`, and `--speed-values-m-s`.

## Model Interface

The audit evaluates only:

- `build_nausicaa_glider()`
- `adapt_glider()`
- `evaluate_state()`

It groups returned strip loads using `glider.surface_code`:

- `0`: wing
- `1`: horizontal tail
- `2`: vertical tail

The state convention is the same as the envelope check: `u = V cos(alpha)`, `v = 0`, and `w = V sin(alpha)`, matching the current `alpha = atan2(w, u)` implementation.

## Outputs

The default output directory is `03_Control/05_Results/00_model_audit/longitudinal_moment/001/`.

Generated files:

- `surface_force_moment_breakdown.csv`
- `surface_slope_summary.csv`
- `geometry_reference.csv`
- `aerosandbox_geometry_reference.csv`
- `static_margin_proxy.csv`
- `manifest.json`
- `report.md`
- `figures/cm_alpha_surface_breakdown.png`
- `figures/cl_alpha_surface_breakdown.png`
- `figures/geometry_side_view.png`

## Scope Labels

- `comparison_scope`: `low_alpha_attached_flow_sanity_only`
- `high_incidence_validation_claim`: `false`
- `audit_scope`: `low_alpha_longitudinal_moment_strip_breakdown_only`

## Limitations

This audit does not import AeroSandbox, tune the aerodynamic model, validate high-incidence agile reversal, or authorize OCP/TVLQR/controller claims. It is diagnostic evidence for the low-alpha pitching-moment mismatch only.
