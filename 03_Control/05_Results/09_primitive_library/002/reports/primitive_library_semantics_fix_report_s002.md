# Primitive Library Semantics Fix Report

Run 002 fixes primitive-library evidence semantics without changing the
plant, command bridge, safety volume, or updraft models. The run-002
primitive-library rows are first-pass deterministic seed-library evidence
unless `evidence_source` explicitly says otherwise.

Run 001 created the scaffold but over-classified W1/W2 dry-recoverable
rows as boundary evidence. Run 001 also over-triggered
`library_growth_trigger` for rows that should remain entry-envelope or
candidate-refinement evidence.

This pass does not implement W3, clustering, governor, outer loop, OCP,
TVLQR, or real-flight validation.

## Run 001 Baseline Diagnosis

- Baseline available: `True`
- Candidate-class counts: `{'boundary_evidence': 87, 'w0_standalone_commandable': 3}`
- Envelope-status counts: `{'not_present_in_001': 90}`
- Coverage-status counts: `{'not_present_in_001': 90}`
- Library-growth trigger count: `87`
- W1/W2 dry-recoverable boundary rows: `24`

## Run 002 Candidate-Class Counts

- `boundary_evidence`: `74`
- `updraft_assisted_commandable`: `13`
- `w0_standalone_commandable`: `3`

## Run 002 Envelope-Status Counts

- `candidate_family_boundary`: `15`
- `candidate_family_needs_refinement`: `14`
- `outside_entry_envelope_governor_reject`: `45`
- `widening_existing_envelope`: `16`

## Run 002 Coverage-Status Counts

- `covered_by_existing_envelope`: `16`
- `uncovered_boundary`: `15`
- `uncovered_governor_reject`: `45`
- `uncovered_needs_refinement`: `14`

## Envelope Group Status

- `candidate_family_boundary`: `15`
- `candidate_family_needs_refinement`: `14`
- `outside_entry_envelope_governor_reject`: `45`
- `widening_existing_envelope`: `16`

## Coverage Region Status

- `covered_by_existing_envelope`: `6`
- `uncovered_boundary`: `5`
- `uncovered_governor_reject`: `15`
- `uncovered_needs_refinement`: `4`

- Run-002 library-growth trigger count: `0`
- W1/W2 dry-recoverable rows are no longer boundary evidence by default.
- Mid-arena entry-envelope failures are governor rejections, not immediate library growth.

The archived high-alpha/perch-like branch remains boundary evidence only.
