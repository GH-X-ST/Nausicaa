# R9 Preflight 3D Figures

This run visualises fixed-case R9 behaviour from `03_Control/05_Results/R10_learn/C15_full_memory_fastquery` using the shared four-fan 3D plotting baseline.
The updraft context uses low-resolution 3D centre slices and transparent isosurfaces, following the `01_Thermal/four_fan_gp_3D.py` visual style.

- Input root: `03_Control/05_Results/R10_learn/C15_full_memory_fastquery`
- Figure count: 6
- Library case: `balanced_cluster`
- History policy: `spatial_flow_belief_memory_h10`
- Plot frame: `true_safe` x=[1.2, 6.6] m, y=[0.0, 4.4] m, z=[0.4, 3.5] m
- Updraft grid: 48 x 32 x 26
- Updraft surface method: `marching_cubes`
- Primitive markers: `o` endpoint circles
- Claim status: R9 preflight visualisation only; no memory-improvement claim.

## Figures

- `03_Control/A_figures/04_r10_c15_balanced_cluster_paths/figures/r10_c15_targeted_four_fan_case0000_history_paths_3d.png`: history_paths / targeted_memory_opportunity_arena_wide_four_fan / case 0
- `03_Control/A_figures/04_r10_c15_balanced_cluster_paths/figures/r10_c15_targeted_four_fan_case0000_history_h30_paths_3d.png`: history_paths / targeted_memory_opportunity_arena_wide_four_fan / case 0
- `03_Control/A_figures/04_r10_c15_balanced_cluster_paths/figures/r10_c15_targeted_four_fan_case0000_final_paired_paths_3d.png`: final_paired_paths / targeted_memory_opportunity_arena_wide_four_fan / case 0
- `03_Control/A_figures/04_r10_c15_balanced_cluster_paths/figures/r10_c15_targeted_four_fan_case0001_history_paths_3d.png`: history_paths / targeted_memory_opportunity_arena_wide_four_fan / case 1
- `03_Control/A_figures/04_r10_c15_balanced_cluster_paths/figures/r10_c15_targeted_four_fan_case0001_history_h30_paths_3d.png`: history_paths / targeted_memory_opportunity_arena_wide_four_fan / case 1
- `03_Control/A_figures/04_r10_c15_balanced_cluster_paths/figures/r10_c15_targeted_four_fan_case0001_final_paired_paths_3d.png`: final_paired_paths / targeted_memory_opportunity_arena_wide_four_fan / case 1
