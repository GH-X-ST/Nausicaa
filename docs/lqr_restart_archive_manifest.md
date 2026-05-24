# LQR Restart Archive Manifest

Status: retained historical retirement manifest. This file records what was
moved out of active use. It is not a naming template for new evidence runs; use
`docs/housekeeping_and_naming_rules.md` for current result folders and file-size
rules.

## Retired Artefact

- Retired artefact path: `03_Control/05_Results/feedback_contextual_v1_4/`
- Archive path: `03_Control/99_Archive/retired_pd_contextual_v1_4/results/`
- Action: moved into archive, not deleted.
- Retirement reason: the active method is now time-invariant LQR-stabilised primitives. The old dense run came from the retired contextual feedback workflow and is not active evidence, fallback, baseline, or ablation material.

## Row Counts

The `r6`, `r7`, `r8`, and `r9` labels below are archived run identifiers from
the retired workflow.

| Manifest | Tables | Rows | Storage |
|---|---:|---:|---|
| `proj/r6/r642/manifests/table_manifest.json` | 1 | 1000 | `csv_gz` |
| `proj/r8/w2_648/manifests/table_manifest.json` | 1 | 1000 | `csv_gz` |
| `proj/r9/w3_649/manifests/table_manifest.json` | 1 | 1000 | `csv_gz` |
| `r6/r646/manifests/table_manifest.json` | 80 | 80000 | `csv_gz` |
| `r8/w2_648/manifests/table_manifest.json` | 15 | 15000 | `csv_gz` |
| `r9/w3_649/manifests/table_manifest.json` | 30 | 30000 | `csv_gz` |

## File-Size Audit

- Retained file count: 345.
- Retained byte count: 142326142.
- Retained size: 135.73 MB.
- Largest retained file: `r7/sel_647/tables/selector_candidate_log.csv`, 22.54 MB.
- No retained file is above the 100 MB project limit.

## Active-Evidence Boundary

The archive is `retired_not_active`. It must not be imported by active code, used as LQR evidence, used as a fallback controller source, used as a thesis baseline, or mixed into active LQR clustering. It may be cited only as a retired implementation record.

Current W labels remain fidelity/evidence layers only: W0 dry air, W1 variable
Gaussian plume, W2 fixed-LQR annular-GP survival replay, W3 fixed-LQR
randomised annular-GP survival replay, followed by post-W3 clustering and
merging.

The active restart must use:

```text
03_Control/03_Primitives/lqr_linearisation.py
03_Control/03_Primitives/lqr_controller.py
03_Control/03_Primitives/lqr_tuning.py
03_Control/04_Scenarios/run_lqr_tuning_sweep.py
03_Control/04_Scenarios/run_lqr_contextual_archive.py
03_Control/04_Scenarios/run_lqr_w2_replay.py
03_Control/04_Scenarios/run_lqr_w3_generalisation.py
```
