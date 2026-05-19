# Primitive Library Shortlist and W3 Planning Report

This planning-only pass reads run `s002` evidence and writes a derived run `s003` shortlist.
It does not replay candidates, execute W3, implement a governor, implement OCP/TVLQR, or make a real-flight claim.

## Shortlist Summary

- `boundary_only`: `15`
- `governor_reject_entry_envelope`: `45`
- `needs_seed_refinement`: `14`
- `selected_for_governor_seed`: `3`
- `selected_for_w3_stress`: `13`

## Coverage Decisions

- `boundary_keep_for_discussion`: `5`
- `covered_keep`: `1`
- `covered_send_to_w3`: `5`
- `entry_envelope_reject`: `15`
- `generator_refinement_needed`: `4`

- W3 stress candidates planned: `5`
- W3 rows are planning rows only; `not_implemented_in_this_pass=True`.

## Higher-Target Requests

- `45 deg`: `defer_boundary_only` - 30_deg_uncovered_region_is_boundary_or_entry_envelope_limited_not_library_growth
- `60 deg`: `defer_boundary_only` - 30_deg_uncovered_region_is_boundary_or_entry_envelope_limited_not_library_growth
- `90 deg`: `not_requested_boundary_only` - targets_above_60_deg_are_not_next_step_without_specific_coverage_evidence
- `120 deg`: `not_requested_boundary_only` - targets_above_60_deg_are_not_next_step_without_specific_coverage_evidence
- `150 deg`: `not_requested_boundary_only` - targets_above_60_deg_are_not_next_step_without_specific_coverage_evidence
- `180 deg`: `not_requested_boundary_only` - targets_above_60_deg_are_not_next_step_without_specific_coverage_evidence

The higher-target table is coverage-driven. A failed or uncovered 30 deg row is
not enough by itself to request 45/60 deg. Boundary or entry-envelope
limitations defer higher-target work unless a separate mission-critical
coverage row proves the larger heading is required and plausible.

## Unimplemented Scope

- W3 selected stress: `False`
- Governor: `False`
- Outer loop: `False`
- OCP/TVLQR: `False`
- Real-flight validation: `False`
- High-incidence validation claim: `False`
