# Longitudinal Moment Audit Report

This audit decomposes low-alpha local strip forces and moments by lifting surface.
It does not tune the model and it does not validate high-incidence agile reversal.

## Scope

- Comparison scope: `low_alpha_attached_flow_sanity_only`
- High-incidence validation claim: `false`
- AeroSandbox is not imported by this audit.
- Forces and moments are aerodynamic strip loads only, grouped by surface code.

## Status

- Interpretation status: `pass`
- Total CL-alpha per rad: `4.88369`
- Total Cm-alpha per rad: `0.199509`
- Wing Cm-alpha per rad: `1.99804`
- Horizontal-tail Cm-alpha per rad: `-1.7985`
- x_CG/MAC from current mass properties: `0.763636`

## Finding

The corrected strip force point leaves a small positive total Cm-alpha.
The wing positive contribution and horizontal-tail negative contribution
mostly offset each other, so the audit no longer shows the earlier
erroneous wing-dominated pitching moment.

## Next Action

Keep this result as low-alpha attached-flow sanity evidence only.
High-incidence controller claims still need the separate validation path.
