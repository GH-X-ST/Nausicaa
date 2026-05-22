# Fixed-Gate W0/W1 Primitive Rollout Archive

Active mission path: `fixed launch gate -> primitive rollout archive over launch-gate and reachable downstream states -> W0/W1 fixed-gate archive -> W2/W3 focused replay -> primitive-envelope clustering -> governor candidate package -> repeated-launch episode simulation -> real-flight ingest and matched replay`

- Sample rows: `30`
- Candidate rows: `400`
- Rollout rows: `800`
- Mission-candidate rows: `0`
- Partial-feedback rows: `260`
- Accepted W0 partial-feedback rows: `1`
- Accepted W1 partial-feedback rows: `1`
- Blocked-partial rows: `140`
- Diagnostic open-loop rows: `400`
- W0 rows by branch: `{'four_fan_branch': 200, 'single_fan_branch': 200}`
- W1 rows by branch: `{'four_fan_branch': 200, 'single_fan_branch': 200}`
- Non-dry W1 measured-updraft rows: `400`
- W1 measured-updraft rows by branch: `{'four_fan_branch': 200, 'single_fan_branch': 200}`
- Code-ready status: `ready`
- Archive-prepared status: `ready`
- Mission-evidence-ready status: `blocked_no_mission_or_partial_feedback_rows_for_both_branches`
- Feedback path status: `partial_feedback_instant_state_no_delayed_state_feedback`
- W1 remains scheduled independently of W0 success.

Open-loop rollout rows are ablation diagnostics only and are not governor-package or mission-facing evidence.

No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.
