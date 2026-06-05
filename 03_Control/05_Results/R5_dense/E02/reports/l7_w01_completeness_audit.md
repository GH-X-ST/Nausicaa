# L7 W01 Completeness Audit

- Status: `complete`
- Run class: `rich_side_l6_candidate`
- Rows written: `102400`
- Worker count: `8`
- Chunk count: `128`
- Chunk size: `800`
- Storage format: `csv_gz`
- Candidate count requested: `32`
- Paired tests per candidate: `50`
- Largest file: `manifests/frozen_w01_controller_bundle_all_candidates.json` at `73.98793125152588` MB
- Above 75 MB present: `False`
- Above 100 MB present: `False`
- R5 anchor plus uncertainty-family evidence blocks present: `False`
- Fixed R5 library cleared for future W3 frozen held-out replay: `True`

Coverage summaries:

- Primitives: `{"energy_retaining_bank": 12800, "glide": 12800, "lift_dwell_arc": 12800, "lift_entry": 12800, "mild_turn_left": 12800, "mild_turn_right": 12800, "recovery": 12800, "safe_exit_or_recovery_handoff": 12800}`
- Candidate indices present: `32`
- Start families: `{"inflight_boundary_near": 10240, "inflight_lift_region": 15360, "inflight_nominal": 25600, "inflight_recovery_edge": 10240, "launch_gate": 40960}`
- Environments: `{"dry_air": 12800, "w1_annular_gp_randomised_four": 64000, "w1_annular_gp_randomised_single": 25600}`
- Evidence blocks: `{}`
- Boundary use: `{"continuation_valid": 96115, "episode_terminal_useful": 6282, "hard_failure": 3}`

Timing-aware synthesis preview:

- `{"candidate_index":0,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":1,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":10,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":11,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":12,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":13,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":14,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":15,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":16,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":17,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":18,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`
- `{"candidate_index":19,"controller_design_role":"active_timing_aware_w01","lqr_synthesis_status":"solved","primitive_id":"energy_retaining_bank","row_count":400}`

Blockers:

- `none`

Blocked claims remain W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal LQR-tree/funnel/region-of-attraction guarantees.
