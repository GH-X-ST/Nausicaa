# Aggressive Reversal Rewrite Plan

ECT layer sequence: Cleanup -> Exploration -> Candidate

The active aggressive branch is a high-incidence pitch-brake/yaw-roll reversal exploration path for 30, 90, and 180 deg fixed-start simulation attempts. It replaces the retired regular-turn/OCP30/Phase-2 TVLQR branch.

Implementation boundaries:

- Use the frozen 15-state order and 3-command order.
- Use full calibrated command authority with clipping and saturation metrics.
- Treat terminal recovery as a metric during Exploration, not a transfer gate.
- Add only one clean TVLQR wrapper after the old TVLQR code has been purged.
- Do not add Vicon, real-flight, W0-W3 stress, governor promotion, dashboards, or multiple feedback variants in this task.

Evidence target:

- finite trajectory arrays or classified solver failures for 30, 90, and 180 deg;
- compact CSV/JSON artifacts under the requested result roots;
- a boundary report stating that high-incidence results are simulation-surrogate evidence only.

