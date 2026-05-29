# R5 Run 037 Mild Turn Left/Right Matched Audit

Matched rows use the same `paired_start_key`, start family, environment, W layer, candidate index, and local LQR speed bin. R5 stores start and exit states, not full state histories, so the plots are 0.100 s start-to-exit segments.

- Matched pair count: `9600`
- Selected plotted pairs: `12`
- Mean absolute lateral displacement separation: `0.0007 m`
- Median absolute lateral displacement separation: `0.0002 m`
- Mean absolute yaw separation: `0.178 deg`
- Median absolute yaw separation: `0.092 deg`
- Mean absolute bank separation: `0.693 deg`
- Median absolute bank separation: `0.270 deg`

Plots:
- `plots/mild_turn_left_right_matched_segments_3d.png`
- `plots/mild_turn_left_right_dy_compare.png`
- `plots/mild_turn_left_right_dpsi_compare.png`
- `plots/mild_turn_left_right_dphi_compare.png`
- `plots/mild_turn_left_right_aileron_compare.png`
- `plots/mild_turn_left_right_rudder_compare.png`

Claim boundary: this is an R5 primitive-identity audit only, not R7 validation or full-flight evidence.
