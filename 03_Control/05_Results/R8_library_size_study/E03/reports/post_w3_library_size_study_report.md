# v5.20 Post-W3 Library-Size Study

- Status: `complete`
- Library-size cases: `heavy_cluster,balanced_cluster,light_cluster,super_light_cluster,no_cluster_no_merge`
- Human label retained for no_cluster_no_merge: `no-clustering/no-merging`
- Selection: `coverage-aware behavior/Q-R medoid selection with greedy marginal coverage fill`
- Launch-gate budget: R8 keeps wider step-0 launch-gate representative budgets than non-launch entries: heavy=2, balanced=5, light=8, super-light=12, no-cluster=all.
- Speed-bin coverage: R8 preserves distinct W3-eligible local LQR speed bins within each primitive/entry-class group up to the effective entry-class budget and writes `speed_bin_coverage_audit.csv`.
- Hard safety filter: within each primitive/entry-role group, prefer survivors with `hard_failure_rate < 0.75`; if all candidates exceed that threshold the group is retained for explicit downstream blocking/audit.
- Medoids are existing W3-eligible variants; no Q/R, K, reference, horizon, entry-role, controller-ID, or primitive-variant-ID mutation is performed.
- Claim boundary: simulation-only; no hardware-readiness, transfer, mission, or memory-improvement claim.
