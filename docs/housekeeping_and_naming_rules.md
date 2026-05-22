# Repository Housekeeping, Naming, Runtime, and Storage Rules

## Result Folder Contract
All generated control results live under:

```text
03_Control/05_Results/<NN_result_group>/<NN_case_slug>/<seed_or_run_id>/
```

The seed or run folder is the atomic result unit. It must contain the manifest and the analysis files needed to audit that run. Canonical plotted scenario cases keep CSV analysis data and figures together in this folder.

Allowed result groups are:

```text
00_smoke
01_latency_interface
03_primitives
04_scenario_matrix
06_updraft_models
07_annular_gp_models
11_governor
11_fixed_gate_repeated_launch
12_reproducibility
99_misc
```

The fixed-gate W0/W1 archive may use the dedicated control-campaign folder already present in the codebase:

```text
03_Control/05_Results/11_fixed_gate_repeated_launch/<run_id>/
```

If this path is used, it must still obey the atomic-run, manifest, naming, cleanup, chunking, compression, and no-overwrite rules below.

The fixed-gate chunked-runtime scratch root is legal only for local preflight runs and is not an official evidence campaign path:

```text
03_Control/05_Results/11_fixed_gate_repeated_launch_chunked_preflight/<run_id>/
```

Preflight roots must not be pushed unless the user explicitly asks to preserve the scratch evidence.

## Naming Rules
Use numbered lower-snake-case folders, for example:

```text
03_primitives/05_agile_tvlqr_reversal_left/001
```

Do not use tool, person, or scratch labels in committed/generated result paths.
Do not use `codex`, `tmp`, `new`, `final`, or `test_output` in result paths.
Use three-digit seed or run folders for deterministic runs: `001`, `002`, `003`.

Keep result artifact filenames compact. Deep result bundles should target a repository-relative path length below 140 characters so Git can index them on Windows checkouts with long project roots.

Keep temporary runner files only under a local `_run` folder inside the relevant run folder, and delete that folder before the command completes.

Compact result artifact stems should encode the target and seed without repeating the full scenario name:

```text
summary_s001.csv
030_s001.csv
030_s001_governor_candidates.csv
030_s001_governor_rejections.csv
030_s001_sample000.csv
```

The full scenario identifier remains inside CSV rows and manifests for audit traceability.

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

Search, optimisation, archive, and audit bundles may contain local subfolders:

```text
logs/
metrics/
manifests/
reports/
tables/
trajectories/
chunk_manifests/
```

Those subfolders must remain inside the atomic run folder, not directly under `03_Control/05_Results`.

## Dense Runtime Contract
Dense, archive-scale, or thesis-scale runs must use chunked, resumable, compressed execution.

A run is dense if any of the following is true:

```text
planned rollout rows >= 10,000
planned candidate rows >= 5,000
expected runtime > 30 minutes
expected uncompressed table size > 250 MB
used for thesis evidence, envelope maps, clustering, W2/W3 replay, or governor packages
```

Dense runs must not use a single-process full-memory runner that builds the entire rollout table before writing. They must support:

```text
--workers
--max-workers
--candidate-chunk-size or --chunk-size
--storage-format auto|parquet|csv_gz|csv
--compression-level
--resume
--repair-incomplete
--dry-run-schedule
--stop-after-chunks
--continue-on-chunk-failure
```

Default local dense-run policy:

```text
workers = 8
max_workers = 8
storage_format = auto, resolving to parquet if supported, otherwise csv_gz
compression_level = 1 for csv_gz
resume = true
```

If memory pressure occurs, reduce chunk size or worker count first. Do not fall back to a full-memory single-process runner.

## Dense Storage Contract
Dense runs must write full tables as compressed partitions under `tables/`, not as one giant uncompressed CSV.

Required pattern:

```text
tables/<table_name>/part-00000.csv.gz
tables/<table_name>/part-00001.csv.gz
```

or parquet if available.

Every partition must be recorded in a table manifest with:

```text
table_name
relative_path
storage_format
row_count
byte_count
columns
checksum_sha256
```

Root-level or `metrics/` CSV files should be compact summaries only. For dense runs, do not write full rollout tables as uncompressed `metrics/*.csv` unless an explicit debug flag is used and the row count is below the dense threshold.

Required compact dense-run outputs:

```text
manifests/run_manifest.json
manifests/table_manifest.json
reports/run_report.md
metrics/runtime_summary.csv
metrics/branch_coverage_summary.csv
metrics/outcome_summary.csv
metrics/pairing_audit.csv
```

## Chunk Progress and Resume Rules
Each dense-run chunk must record:

```text
chunk_index
chunk_count
status complete|failed|skipped|corrupt
input row count
output row count
partition paths
checksums
worker id or process id if available
wall-clock runtime
failure type and message if failed
```

Resume behavior:

```text
--resume skips complete chunks
--repair-incomplete rewrites corrupt or incomplete chunks
without --resume, an existing non-empty official run folder must stop loudly
```

Do not overwrite existing numbered historical result folders unless the user asks for that specific evidence to be removed.

## Fixed-Gate W0/W1 Archive Rule
The fixed-gate W0/W1 smoke run may remain small and simple. Any rich fixed-gate W0/W1 archive must obey the dense runtime and storage contract.

For fixed-gate W0/W1:

```text
W0 = dry-air baseline, wind_mode=none
W1 = measured-updraft replay, wind_mode=panel
W1 single-fan = single_gaussian_var
W1 four-fan = four_gaussian_var
W1 is not filtered by W0 success
branches remain separate
```

The fixed-gate rich archive must write branch-coverage summaries and must not use diagnostic rows as mission evidence.

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

Do not delete existing numbered historical result folders unless the user asks for that specific historical evidence to be removed.

Temporary local result roots used for preflight should be clearly outside official numbered evidence runs, for example:

```text
03_Control/05_Results/11_fixed_gate_repeated_launch_chunked_preflight/
```

Preflight folders should not be pushed unless the user explicitly wants the preflight evidence preserved.

## Documentation Rules
Project documentation belongs under `docs/`. The repository ignore rules must not hide the whole `docs/` directory.

Private local instruction and context files shared only between the project owner and the coding assistant are not publication artifacts. Ignore them by precise path instead of ignoring the whole documentation tree. Current private paths are:

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

Do not ignore the full `docs/` directory.

## Push and Archive Safety
Before any Git push involving generated results, check:

```text
no accidental __pycache__ or pytest cache
no local _run folder remains
no giant uncompressed dense rollout CSV is included
partition manifests exist for dense result tables
repository-relative paths remain short enough for Windows indexing
claim-boundary manifest exists
```

If a dense result package is too large for normal Git use, keep only compact manifests, summaries, reports, and selected plots in the repository, and record where the full local partitioned archive is stored.
