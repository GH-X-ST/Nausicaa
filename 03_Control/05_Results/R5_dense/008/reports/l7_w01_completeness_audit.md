# L7 W01 Completeness Audit

- Status: `complete`
- Run class: `rich_side_l6_candidate`
- Rows written: `76800`
- Worker count: `8`
- Chunk count: `96`
- Chunk size: `800`
- Storage format: `csv_gz`
- Candidate count requested: `32`
- Paired tests per candidate: `100`
- Largest file: `manifests/primitive_variant_registry.json` at `1.3162765502929688` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- W1 single/four mixed in one root: `True`
- Fixed W01 library cleared for future W2 fixed-LQR replay: `True`

Coverage summaries:

- Primitives: `{"energy_retaining_bank": 9600, "glide": 9600, "lift_dwell_arc": 9600, "lift_entry": 9600, "mild_turn_left": 9600, "mild_turn_right": 9600, "recovery": 9600, "safe_exit_or_recovery_handoff": 9600}`
- Candidate indices present: `32`
- Start families: `{"inflight_boundary_near": 7680, "inflight_lift_region": 11520, "inflight_nominal": 19200, "inflight_recovery_edge": 7680, "launch_gate": 30720}`
- Environments: `{"dry_air": 25600, "gaussian_four": 25600, "gaussian_single": 25600}`
- Boundary use: `{"blocked": 34560, "continuation_valid": 2990, "episode_terminal_useful": 18202, "hard_failure": 21048}`

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

Blocked claims remain W2 execution, W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
