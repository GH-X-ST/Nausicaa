# LQR Restart Archive Manifest

Status: retained historical retirement manifest. This file records the old
feedback-contextual result retirement. It is not a naming template, active plan,
baseline, or move-on report. For current result folders, path-length handling,
and file-size rules, use `docs/housekeeping_and_naming_rules.md`.

## Retired Artefact

- Retired artefact path: `03_Control/05_Results/feedback_contextual_v1_4/`
- Archive path: `03_Control/99_Archive/retired_pd_contextual_v1_4/results/`
- Action: moved into archive, not deleted.
- Retirement reason: the active method is LQR-stabilised
  primitive-controller variants. The old dense run came from a retired
  contextual feedback workflow and is not active evidence, fallback, baseline,
  ablation material, or governor-training input.

## Row Counts

The `r6`, `r7`, `r8`, and `r9` labels below are archived run identifiers from
the retired workflow. They must not be reused as active stage names.

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

## Current Boundary

The archive is `retired_not_active`. It must not be imported by active code,
used as LQR evidence, used as a fallback controller source, used as a thesis
baseline, or mixed into W0/W1 dense generation, W2/W3 survival replay, post-W3
library-size studies, outcome models, governor reports, or real-flight claims.
It may be cited only as a retired implementation record.

The current controlling plan supersedes earlier wording that treated the
v4.10-style outer-loop result as accepted. Rejected v4.10-style full-loop and
governor-calibration outputs must be labelled `diagnostic_not_passed`, retained
or archived with compact manifests, and excluded from move-on evidence.

Current W labels are fidelity/evidence layers only:

```text
W0      dry-air primitive-library generation
W1      active annular-GP randomized primitive-library training; Gaussian plume diagnostic-only
W2      optional fixed-LQR GP-corrected annular-Gaussian diagnostic replay
W3      fixed-LQR randomised GP-corrected annular-Gaussian held-out survival replay
post_w3 four-case library-size cross-study
late    repeated-launch validation with frozen library/governor/selector/memory
```

The next active restart is not the old R6/R8/R9 runner family. It must rerun
R5 dry-air plus annular-GP randomized dense generation with 0.10 s primitives,
5 controller-input slots, and a 20 ms controller update period, then replay
fixed variants through held-out W3, then run the post-W3 library-size
cross-study and repeated-launch validation only if the earlier gates pass.
