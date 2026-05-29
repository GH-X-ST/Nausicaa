# R5 Run 037 Sampled Primitive Segment Audit

This audit samples recorded R5 dense evidence rows and plots 3D start-to-exit segments for each 0.100 s primitive. The R5 table does not store full time-resolved state histories, so these figures are not full trajectory traces; they are recorded initial-state to exit-state primitive segments with start/end markers.

- R5 root: `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/037`
- Random seed / deterministic sample id: `52337`
- Accepted samples: `24`
- Weak samples: `24`
- Rejected samples: `0`
- Outcome counts: `{"accepted": 26278, "failed": 1381, "weak": 49141}`

Plots:
- accepted: `03_Control/05_Results/lqr_contextual_v1_0/lqr_controller_trajectory_audit/008/plots/r5_037_accepted_3_per_family_segments_3d.png` (24 segments)
- weak: `03_Control/05_Results/lqr_contextual_v1_0/lqr_controller_trajectory_audit/008/plots/r5_037_weak_3_per_family_segments_3d.png` (24 segments)
- rejected: no `outcome_class = rejected` rows were recorded in R5 run 037, so no rejected plot was generated.

Claim boundary: this is an R5 visual audit only. It does not replace R7 held-out validation, R8 compression, R10 governor tuning, R11 validation, or real-flight evidence.
