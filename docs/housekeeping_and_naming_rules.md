# Repository Housekeeping and Naming Rules

## Result Folder Contract

All generated control results live under:

```text
03_Control/05_Results/<NN_result_group>/<NN_case_slug>/<seed_or_run_id>/
```

The seed or run folder is the atomic result unit. It must contain the manifest and
the analysis files needed to audit that run. Canonical plotted scenario cases keep
CSV analysis data and figures together in this folder.

Allowed result groups are:

```text
00_smoke
01_latency_interface
03_primitives
04_scenario_matrix
06_updraft_models
07_annular_gp_models
11_governor
12_reproducibility
99_misc
```

## Naming Rules

- Use numbered lower-snake-case folders, for example
  `03_primitives/05_agile_tvlqr_reversal_left/001`.
- Do not use tool, person, or scratch labels in committed/generated result paths.
- Do not use `codex`, `tmp`, `new`, `final`, or `test_output` in result paths.
- Use three-digit seed folders for deterministic runs: `001`, `002`, `003`.
- Keep result artifact filenames compact. Deep result bundles should target a
  repository-relative path length below 140 characters so Git can index them on
  Windows checkouts with long project roots.
- Keep temporary runner files only under a local `_run` folder inside the relevant
  run folder, and delete that folder before the command completes.

Compact result artifact stems should encode the target and seed without repeating
the full scenario name:

```text
summary_s001.csv
030_s001.csv
030_s001_governor_candidates.csv
030_s001_governor_rejections.csv
030_s001_sample000.csv
```

The full scenario identifier remains inside CSV rows and manifests for audit
traceability.

## Atomic Result Contents

Canonical plotted scenario folders contain:

```text
actual_rollout.csv
actual_metrics.csv
manifest.json
A_trajectory_command_actuator.png
B_flight_rates.png
C_flight_state_alpha_beta.png
D_envelope_variables.png
E_2d_trajectory_geometry.png
```

Search, optimisation, and audit bundles may contain local subfolders:

```text
logs/
metrics/
manifests/
trajectories/
```

Those subfolders must remain inside the atomic run folder, not directly under
`03_Control/05_Results`.

## Cleanup Rules

Remove these whenever they appear:

```text
.pytest_cache/
__pycache__/
_tmp_*/
03_Control/05_Results/logs/
03_Control/05_Results/metrics/
03_Control/05_Results/__pycache__/
```

Do not delete existing numbered historical result folders unless the user asks for
that specific historical evidence to be removed.

## Documentation Rules

Project documentation belongs under `docs/`. The repository ignore rules must not
hide the whole `docs/` directory.

Private local instruction and context files shared only between the project owner
and the coding assistant are not publication artifacts. Ignore them by precise
path instead of ignoring the whole documentation tree. Current private paths are:

```text
CODEX_*.md
docs/CODEX_*.md
Nausicaa_Five_Figure_Plotting_Guidance.md
docs/Nausicaa_Five_Figure_Plotting_Guidance.md
Glider_Control_Project_Plan.md
docs/control/glider_control_project_plan.md
docs/MATLAB Coding.txt
docs/Python Coding Instruction.txt
docs/Python Plotting Guidance.txt
docs/Skills.md
```
