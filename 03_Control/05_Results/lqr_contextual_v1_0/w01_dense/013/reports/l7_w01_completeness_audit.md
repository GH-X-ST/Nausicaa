# L7 W01 Completeness Audit

- Status: `dry_run_schedule`
- Run class: `dry_run_schedule`
- Rows written: `0`
- Worker count: `1`
- Chunk count: `4`
- Chunk size: `240`
- Storage format: `csv_gz`
- Candidate count requested: `2`
- Paired tests per candidate: `20`
- Largest file: `manifests/frozen_w01_controller_bundle.json` at `0.2826061248779297` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- W1 single/four mixed in one root: `False`
- Fixed W01 library cleared for future W2 fixed-LQR replay: `False`

Coverage summaries:

- Primitives: `{}`
- Candidate indices present: `0`
- Start families: `{}`
- Environments: `{}`
- Boundary use: `{}`

Timing-aware synthesis preview:

- `empty`

Blockers:

- `no_rollout_evidence_written`

Blocked claims remain W2 execution, W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
