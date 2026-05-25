# L7 W01 Completeness Audit

- Status: `complete`
- Run class: `preflight`
- Rows written: `240`
- Worker count: `1`
- Chunk count: `3`
- Chunk size: `80`
- Storage format: `csv_gz`
- Candidate count requested: `2`
- Paired tests per candidate: `5`
- Largest file: `manifests/frozen_w01_controller_bundle.json` at `0.2826061248779297` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- W1 single/four mixed in one root: `True`
- Fixed W01 library cleared for future W2 fixed-LQR replay: `False`

Coverage summaries:

- Primitives: `{"energy_retaining_bank": 30, "glide": 30, "lift_dwell_arc": 30, "lift_entry": 30, "mild_turn_left": 30, "mild_turn_right": 30, "recovery": 30, "safe_exit_or_recovery_handoff": 30}`
- Candidate indices present: `2`
- Start families: `{"launch_gate": 240}`
- Environments: `{"dry_air": 80, "gaussian_four": 80, "gaussian_single": 80}`
- Boundary use: `{"blocked": 210, "continuation_valid": 2, "hard_failure": 28}`

Timing-aware synthesis preview:

- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":15}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":15}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"glide","row_count":15}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"glide","row_count":15}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_dwell_arc","row_count":15}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_dwell_arc","row_count":15}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_entry","row_count":15}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_entry","row_count":15}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_left","row_count":15}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_left","row_count":15}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_right","row_count":15}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_right","row_count":15}`

Blockers:

- `below_19200_fallback_scale_threshold`

Blocked claims remain W2 execution, W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
