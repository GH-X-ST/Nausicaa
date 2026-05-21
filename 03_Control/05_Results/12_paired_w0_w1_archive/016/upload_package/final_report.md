# Paired W0/W1 Archive Aggregation Report

D1a thesis-scale paired W0/W1 aggregation simulation evidence only; W1 is evaluated independently of W0 success, single_fan_branch and four_fan_branch remain branch-local, and no production-floor completion, W2/W3/W4/W5 completion, mission success, hardware readiness, or sim-to-real completion claim is made.

- Run id: `16`
- Planning run id: `15`
- Trial count total: `250000`
- Trial count by environment: `{'W0_four_fan_branch': 25000, 'W0_single_fan_branch': 25000, 'W1_four_fan': 100000, 'W1_single_fan': 100000}`
- Selected worker count: `8`
- Worker fallback reason: `none`
- W1 acceptance rule: `latency_case=nominal and latency_pass_label=nominal_pass`
- GPU assessment: GPU acceleration is deferred in this pass because it would require a batched dynamics refactor, new dependencies, and numerical-equivalence validation. CPU chunk parallelism preserves RK4, state_derivative, plant dynamics, latency constants, command semantics, and acceptance metrics.
