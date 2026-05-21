# Paired W0/W1 Archive Aggregation Report

Paired W0/W1 proof aggregation only; W1 is evaluated independently of W0 success, single_fan_branch and four_fan_branch remain branch-local, and no W2/W3/W4/W5, mission, hardware, or sim-to-real claim is made.

- Run id: `14`
- Planning run id: `13`
- Trial count total: `10000`
- Trial count by environment: `{'W0_four_fan_branch': 2500, 'W0_single_fan_branch': 2500, 'W1_four_fan': 2500, 'W1_single_fan': 2500}`
- Selected worker count: `8`
- Worker fallback reason: `none`
- W1 acceptance rule: `latency_case=nominal and latency_pass_label=nominal_pass`
- GPU assessment: GPU acceleration is deferred in this pass because it would require a batched dynamics refactor, new dependencies, and numerical-equivalence validation. CPU chunk parallelism preserves RK4, state_derivative, plant dynamics, latency constants, command semantics, and acceptance metrics.
