# W2 Fixed-LQR Survival Replay

- Status: `w2_dense_survival_pass`
- Source W01 root: `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/024`
- Rows written/planned: `89600`
- Variant count: `448`
- Ready frozen controllers: `448`
- Blocked frozen controllers: `0`
- W2 environments: `annular_gp_single`, `annular_gp_four`
- Start-family counts: `{"inflight_boundary_near": 6400, "inflight_lift_region": 13824, "inflight_nominal": 24576, "inflight_recovery_edge": 6400, "launch_gate": 38400}`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 6400, "incompatible": 0}, "glide": {"compatible": 6400, "incompatible": 0}, "launch_capture_energy_build": {"compatible": 6400, "incompatible": 0}, "launch_capture_glide_stabilise": {"compatible": 6400, "incompatible": 0}, "launch_capture_lift_seek": {"compatible": 6400, "incompatible": 0}, "launch_capture_safe_handoff": {"compatible": 6400, "incompatible": 0}, "launch_capture_shallow_left": {"compatible": 6400, "incompatible": 0}, "launch_capture_shallow_right": {"compatible": 6400, "incompatible": 0}, "lift_dwell_arc": {"compatible": 6400, "incompatible": 0}, "lift_entry": {"compatible": 6400, "incompatible": 0}, "mild_turn_left": {"compatible": 6400, "incompatible": 0}, "mild_turn_right": {"compatible": 6400, "incompatible": 0}, "recovery": {"compatible": 6400, "incompatible": 0}, "safe_exit_or_recovery_handoff": {"compatible": 6400, "incompatible": 0}}`
- History-backed FIFO count: `89600`
- Smoke/dense status labels are separate: `w2_dense_survival_pass`
- Fixed replay only: `True`
- Q/R, K, reference, horizon, entry role, controller ID, and variant ID mutation: `False`
- Boundary terminal-useful evidence is retained.

Blocked claims remain W3 robustness, post-W3 library-size readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal ROA guarantees.
