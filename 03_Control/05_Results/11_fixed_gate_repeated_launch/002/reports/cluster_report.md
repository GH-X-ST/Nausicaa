# Fixed-Gate Primitive-Envelope Cluster Report

Active mission path: `fixed launch gate -> primitive rollout archive over launch-gate and reachable downstream states -> W0/W1 fixed-gate archive -> W2/W3 focused replay -> primitive-envelope clustering -> governor candidate package -> repeated-launch episode simulation -> real-flight ingest and matched replay`

Primitive rollout rows were clustered before any repeated-launch episode clustering.
Whole-episode clustering is deferred until repeated-launch policy logs exist.

- Input rows: `800`
- Medoids: `248`
- Mission medoids: `0`
- Partial-feedback medoids: `72`
- Diagnostic medoids: `120`
- Governor candidate rows: `2`
- Open-loop and command-template medoids are retained as diagnostics only and cannot enter the governor candidate package.

No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.
