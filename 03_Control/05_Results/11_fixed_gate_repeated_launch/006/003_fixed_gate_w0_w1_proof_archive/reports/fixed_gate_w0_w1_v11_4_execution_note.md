# v11.4 Fixed-Gate W0/W1 Execution Note

- Run ID: `6`
- Run scope: `official_archive_run`
- Worker count selected: `8`
- Chunk size: `2500`
- Storage/compression: `csv_gz`, level `1`
- Chunks completed/skipped/failed/corrupt: `0` / `96` / `0` / `0`
- Candidate rows: `240000`
- Rollout rows: `480000`
- W0 rows by branch: `{'single_fan_branch': 120000, 'four_fan_branch': 120000}`
- W1 rows by branch: `{'single_fan_branch': 120000, 'four_fan_branch': 120000}`
- W1 measured rows by branch: `{'single_fan_branch': 120000, 'four_fan_branch': 120000}`
- W1 scheduled independently of W0 success: `True`
- Branch coverage conclusion: `ready_for_downstream_non_diagnostic_evidence`
- Downstream status: `not_run_by_archive_runner`

No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, hardware-ready agile-turn, true delayed-state-feedback, full W2/W3 robustness, or real repeated-launch validation claim is made.
