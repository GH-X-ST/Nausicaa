# Longitudinal Moment Audit Report

This audit decomposes low-alpha local strip forces and moments by lifting surface.
It does not tune the model and it does not validate high-incidence agile reversal.

## Scope

- Comparison scope: `low_alpha_attached_flow_sanity_only`
- High-incidence validation claim: `false`
- AeroSandbox is not imported by this audit.
- Forces and moments are aerodynamic strip loads only, grouped by surface code.

## Status

- Interpretation status: `pass_with_pitch_moment_review`
- Total CL-alpha per rad: `4.88369`
- Total Cm-alpha per rad: `2.42905`
- Wing Cm-alpha per rad: `3.95447`
- Horizontal-tail Cm-alpha per rad: `-1.5254`
- x_CG/MAC from current mass properties: `0.763636`

## Finding

The current positive low-alpha Cm-alpha is dominated by the wing strip contribution.
The horizontal tail contributes negative Cm-alpha, so the mismatch is not hidden
inside the tail model. This should be reviewed before OCP or TVLQR claims resume.

## Next Action

Review the wing pitching-moment reference, strip force application point, and
body-axis sign convention before using this model for longitudinal controller claims.
