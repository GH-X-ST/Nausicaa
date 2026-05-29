# L7 W01 Completeness Audit

- Status: `complete`
- Run class: `preflight`
- Rows written: `80`
- Worker count: `1`
- Chunk count: `2`
- Chunk size: `40`
- Storage format: `csv_gz`
- Candidate count requested: `2`
- Paired tests per candidate: `1`
- Largest file: `manifests/frozen_w01_controller_bundle.json` at `0.2852859497070312` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- W1 single/four mixed in one root: `True`
- Fixed W01 library cleared for future W2 fixed-LQR replay: `False`

Coverage summaries:

- Primitives: `{"energy_retaining_bank": 10, "glide": 10, "lift_dwell_arc": 10, "lift_entry": 10, "mild_turn_left": 10, "mild_turn_right": 10, "recovery": 10, "safe_exit_or_recovery_handoff": 10}`
- Candidate indices present: `2`
- Start families: `{"inflight_boundary_near": 8, "inflight_lift_region": 12, "inflight_nominal": 20, "inflight_recovery_edge": 8, "launch_gate": 32}`
- Environments: `{"dry_air": 32, "gaussian_four": 24, "gaussian_single": 24}`
- Boundary use: `{"blocked": 36, "continuation_valid": 29, "hard_failure": 15}`

Timing-aware synthesis preview:

- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":5}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":5}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"glide","row_count":5}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"glide","row_count":5}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_dwell_arc","row_count":5}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_dwell_arc","row_count":5}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_entry","row_count":5}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_entry","row_count":5}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_left","row_count":5}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_left","row_count":5}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_right","row_count":5}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_right","row_count":5}`

Blockers:

- `below_19200_fallback_scale_threshold`

Blocked claims remain W2 execution, W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
