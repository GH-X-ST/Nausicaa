# R9 Preflight 3D Figures

This run visualises fixed-case R9 behaviour from `03_Control/05_Results/R9_test/A06` using the shared four-fan 3D plotting baseline.
The updraft context uses low-resolution 3D centre slices and transparent isosurfaces, following the `01_Thermal/four_fan_gp_3D.py` visual style.

- Input root: `03_Control/05_Results/R9_test/A06`
- Figure count: 9
- Library case: `balanced_cluster`
- History policy: `directional_3d_residual_memory_h10`
- Plot frame: `true_safe` x=[1.2, 6.6] m, y=[0.0, 4.4] m, z=[0.4, 3.5] m
- Updraft grid: 56 x 36 x 30
- Updraft surface method: `marching_cubes`
- Primitive markers: `o` endpoint circles
- Claim status: R9 preflight visualisation only; no memory-improvement claim.

## Figures

- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_no_updraft_history_paths_3d.png`: history_paths / no_updraft
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_no_updraft_history_h30_paths_3d.png`: history_paths / no_updraft
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_no_updraft_final_paired_paths_3d.png`: final_paired_paths / no_updraft
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_single_fan_history_paths_3d.png`: history_paths / single_fan
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_single_fan_history_h30_paths_3d.png`: history_paths / single_fan
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_single_fan_final_paired_paths_3d.png`: final_paired_paths / single_fan
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_four_fan_history_paths_3d.png`: history_paths / four_fan
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_four_fan_history_h30_paths_3d.png`: history_paths / four_fan
- `03_Control/A_figures/01_r9_a06_preflight_paths/figures/r9_a06_four_fan_final_paired_paths_3d.png`: final_paired_paths / four_fan
