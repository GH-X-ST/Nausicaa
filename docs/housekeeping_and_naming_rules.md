# Repository Housekeeping, Naming, Runtime, and Storage Rules

## 1. Result folder contract

Generated control results normally live under:

```text
03_Control/05_Results/<NN_group>/<NN_case>/<run_id>/
```

Pipeline-level evidence roots may instead use:

```text
03_Control/05_Results/<evidence_family>/<stage>/<stage_run_id>/
```

For example, the active LQR evidence root is
`03_Control/05_Results/lqr_contextual_v1_0`, with short stage folders
`r6`, `r7`, `r8`, and `r9`. The stage run folder is the atomic evidence unit.
It must contain the manifest and the compact analysis files needed to audit the
run.

Preferred result groups:

```text
00_smoke
01_latency
02_env_model
03_primitive
04_context_archive
05_outcome_model
06_policy_eval
07_real_flight
08_simreal
09_figures
12_reproducibility
99_misc
```

Historical generated result folders are local-only unless the user explicitly requests preservation. Approved evidence roots may be tracked when every file passes the 100 MB audit and path-length audit. During ordinary validation, keep `03_Control/05_Results/.gitkeep` as the only result placeholder unless an approved local evidence root is explicitly allowed. Historical generated roots kept for context must be labelled `retired_not_active`.

Scratch and preflight roots are local only. They must not be pushed unless the user explicitly requests preservation.

---

## 2. Naming rules

Use concise numbered lower-snake-case names.

Good examples:

```text
04_context_archive/01_r6_lqr_w1_gaussian/001
05_outcome_model/01_lqr_terminal_targets/001
06_policy_eval/01_terminal_episode_smoke/001
08_simreal/01_rf_replay/001
```

Use these abbreviations consistently:

```text
ctx   environment context
prim  primitive
pol   policy
ep    episode
rf    real flight
sr    sim-real replay
w0    dry air
w1    Gaussian plume validation layer
w2    GP-corrected annular-Gaussian validation layer
w3    randomised GP-corrected annular-Gaussian validation layer
nom   nominal
rand  randomised
sum   summary
```

Avoid:

```text
codex
tmp
new
final
test_output
very_long_descriptive_file_names
```

Target limits:

```text
filename stem <= 64 characters
repository-relative path <= 140 characters
generated file <= 100 MB
```

These limits reduce Git and OneDrive path/file problems.

---

## 3. Atomic result contents

Evidence-generating folders should contain:

```text
manifests/
metrics/
reports/
tables/
figures/          optional
chunk_manifests/  dense runs only
```

Those subfolders must remain inside the run folder, not directly under `03_Control/05_Results`.

Required compact files for most evidence runs:

```text
manifests/run_manifest.json
reports/run_report.md
metrics/runtime_summary.csv
metrics/outcome_summary.csv
metrics/file_size_audit.csv
```

---

## 4. Dense runtime contract

A run is dense if any condition is true:

```text
planned rollout rows >= 10,000
planned candidate rows >= 5,000
expected runtime > 30 minutes
expected uncompressed table size > 250 MB
used for thesis evidence, envelope maps, W2/W3 replay, outcome models, or selector/governor reports
```

Dense runs must not use a single-process full-memory runner that builds the entire rollout table before writing.

Dense runners must support:

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

---

## 5. Storage and 100 MB file-size contract

Every generated file should target:

```text
preferred size <= 75 MB
hard project limit <= 100 MB
```

A file above 100 MB is allowed only as explicitly approved local-only evidence. It must not be pushed unless the user approves another storage method.

Dense tables must be written as compressed partitions:

```text
tables/<table_name>/c00000.csv.gz
tables/<table_name>/c00001.csv.gz
```

or parquet if available.

Contextual archive partitions may include short audit tokens, for example
`tables/contextual_rows/c00012_W1_gaussian-single.csv.gz`. Do not use nested
`run_*/context_id=*/environment_id=*/chunk_index=*/part-00000.*` paths for new
dense evidence; those paths are too long for reliable Git and Windows tooling.

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

Root-level or `metrics/` CSV files should be compact summaries only. Do not write full rollout tables as uncompressed metrics CSVs for dense runs.

Every evidence run must write `metrics/file_size_audit.csv` with at least:

```text
relative_path
byte_count
size_mb
above_75mb
above_100mb
push_allowed
```

Archive, outcome-model, and episode-smoke outputs must keep continuation-valid targets separate from episode-terminal-useful targets. X/y boundary exits may be retained as terminal episode evidence; they must not be relabelled as continuation success.

---

## 6. Chunk progress and resume rules

Each dense-run chunk must record:

```text
chunk_index
chunk_count
status
row_count
storage_format
partition_paths
checksum_sha256
worker_count
start_time
end_time
failure_message if any
```

`--resume` skips complete chunks only when manifests and checksums agree.

`--repair-incomplete` may rewrite corrupt chunks. It must not overwrite a complete official run unless explicitly authorised.

---

## 7. Cleanup rules

Remove these whenever they appear:

```text
.pytest_cache/
__pycache__/
_tmp_*/
03_Control/05_Results/logs/
03_Control/05_Results/metrics/
03_Control/05_Results/__pycache__/
```

Do not delete numbered historical result folders unless the user asks for that specific evidence to be removed.

---

## 8. Documentation rules

Project documentation belongs under `docs/`. Repository ignore rules must not hide the whole `docs/` directory.

Private local instruction and context files should be ignored by precise path rather than by broad directory patterns.

Every new dense-run script must document:

```text
which chunked runtime is reused
which compressed writer is reused
where worker_count_decision is recorded
where resume/repair is implemented
where partition checksums are written
how the 100 MB file limit is enforced
why a new runner is needed rather than wrapping an old one
```
