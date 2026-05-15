# Aggressive Reversal Boundary Report

ECT layer: Exploration fix pass

Output root: `C:/Users/GH-X-ST/AppData/Local/Temp/nausicaa_aggressive_fix`

Active blocker: `high_alpha_boundary`

| Target deg | Heading change deg | Source success | Fallback | Label | Failure reason | First bad reason |
|---:|---:|---|---|---|---|---|
| 30 | 0.000 | False | True | solver_failure | alpha_abort | alpha_abort |
| 90 | -19.209 | False | False | high_alpha_boundary | high_alpha_boundary |  |
| 180 | 0.000 | False | True | solver_failure | alpha_abort | alpha_abort |

Rows distinguish source trajectory failure, fallback evidence, finite under-turning evidence, terminal recovery limits, and TVLQR gating status when present.
The aggressive high-incidence reversal results are simulation-surrogate/boundary evidence only.
They are not real-flight claims until separate Transfer-layer gates pass.
