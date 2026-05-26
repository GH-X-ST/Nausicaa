# L7 W01 Completeness Audit

- Status: `complete`
- Run class: `rich_side_l6_candidate`
- Rows written: `134400`
- Worker count: `8`
- Chunk count: `168`
- Chunk size: `800`
- Storage format: `csv_gz`
- Candidate count requested: `32`
- Paired tests per candidate: `100`
- Largest file: `manifests/frozen_w01_controller_bundle.json` at `8.008462905883789` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- W1 randomised single/four mixed in one root: `True`
- Fixed R5 library cleared for future W3 frozen held-out replay: `True`

Coverage summaries:

- Primitives: `{"energy_retaining_bank": 9600, "glide": 9600, "launch_capture_energy_build": 9600, "launch_capture_glide_stabilise": 9600, "launch_capture_lift_seek": 9600, "launch_capture_safe_handoff": 9600, "launch_capture_shallow_left": 9600, "launch_capture_shallow_right": 9600, "lift_dwell_arc": 9600, "lift_entry": 9600, "mild_turn_left": 9600, "mild_turn_right": 9600, "recovery": 9600, "safe_exit_or_recovery_handoff": 9600}`
- Candidate indices present: `32`
- Start families: `{"inflight_boundary_near": 9600, "inflight_lift_region": 21600, "inflight_nominal": 36000, "inflight_recovery_edge": 9600, "launch_gate": 57600}`
- Environments: `{"dry_air": 44800, "w1_randomised_four": 44800, "w1_randomised_single": 44800}`
- Boundary use: `{"continuation_valid": 81939, "episode_terminal_useful": 7428, "hard_failure": 45033}`

Timing-aware synthesis preview:

- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":10,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":11,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":12,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":13,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":14,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":15,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":16,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":17,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":18,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`
- `{"candidate_index":19,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":300}`

Blockers:

- `none`

Blocked claims remain W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
