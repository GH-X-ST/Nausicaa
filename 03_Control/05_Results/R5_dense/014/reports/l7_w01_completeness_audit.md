# L7 W01 Completeness Audit

- Status: `complete`
- Run class: `preflight`
- Rows written: `960`
- Worker count: `2`
- Chunk count: `4`
- Chunk size: `240`
- Storage format: `csv_gz`
- Candidate count requested: `2`
- Paired tests per candidate: `20`
- Largest file: `tables/w01_primitive_rows/c00002.csv.gz` at `0.3419408798217773` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- W1 single/four mixed in one root: `True`
- Fixed W01 library cleared for future W2 fixed-LQR replay: `False`

Coverage summaries:

- Primitives: `{"energy_retaining_bank": 120, "glide": 120, "lift_dwell_arc": 120, "lift_entry": 120, "mild_turn_left": 120, "mild_turn_right": 120, "recovery": 120, "safe_exit_or_recovery_handoff": 120}`
- Candidate indices present: `2`
- Start families: `{"inflight_boundary_near": 96, "inflight_lift_region": 144, "inflight_nominal": 240, "inflight_recovery_edge": 96, "launch_gate": 384}`
- Environments: `{"dry_air": 320, "gaussian_four": 320, "gaussian_single": 320}`
- Boundary use: `{"blocked": 432, "continuation_valid": 42, "episode_terminal_useful": 204, "hard_failure": 282}`

Timing-aware synthesis preview:

- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":60}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":60}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"glide","row_count":60}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"glide","row_count":60}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_dwell_arc","row_count":60}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_dwell_arc","row_count":60}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_entry","row_count":60}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"lift_entry","row_count":60}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_left","row_count":60}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_left","row_count":60}`
- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_right","row_count":60}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"mild_turn_right","row_count":60}`

Blockers:

- `below_19200_fallback_scale_threshold`

Blocked claims remain W2 execution, W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
