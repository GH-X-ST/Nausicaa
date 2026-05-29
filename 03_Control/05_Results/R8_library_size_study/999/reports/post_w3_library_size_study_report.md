# v5.3 Post-W3 Library-Size Study

- Status: `complete`
- Library-size cases: `heavy_cluster,balanced_cluster,light_cluster,super_light_cluster,no_cluster_no_merge`
- Human label retained for no_cluster_no_merge: `no-clustering/no-merging`
- Selection: `coverage-aware behavior/Q-R medoid selection with greedy marginal coverage fill`
- Hard safety filter: within each primitive/entry-role group, prefer survivors with `hard_failure_rate < 0.75`; if all candidates exceed that threshold the group is retained for explicit downstream blocking/audit.
- Medoids are existing W3-surviving variants; no Q/R, K, reference, horizon, entry-role, controller-ID, or primitive-variant-ID mutation is performed.
- Claim boundary: simulation-only; no hardware-readiness, transfer, mission, or memory-improvement claim.
