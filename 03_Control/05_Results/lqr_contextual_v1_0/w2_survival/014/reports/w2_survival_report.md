# W2 Fixed-LQR Survival Replay

- Status: `w2_artifact_smoke_pass`
- Source W01 root: `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/014`
- Rows written/planned: `640`
- Variant count: `16`
- Ready frozen controllers: `16`
- Blocked frozen controllers: `0`
- W2 environments: `annular_gp_single`, `annular_gp_four`
- Start-family counts: `{"inflight_boundary_near": 64, "inflight_lift_region": 96, "inflight_nominal": 160, "inflight_recovery_edge": 64, "launch_gate": 256}`
- Entry-role compatibility by primitive: `{"energy_retaining_bank": {"compatible": 48, "incompatible": 32}, "glide": {"compatible": 80, "incompatible": 0}, "lift_dwell_arc": {"compatible": 48, "incompatible": 32}, "lift_entry": {"compatible": 48, "incompatible": 32}, "mild_turn_left": {"compatible": 48, "incompatible": 32}, "mild_turn_right": {"compatible": 48, "incompatible": 32}, "recovery": {"compatible": 16, "incompatible": 64}, "safe_exit_or_recovery_handoff": {"compatible": 16, "incompatible": 64}}`
- History-backed FIFO count: `352`
- Smoke/dense status labels are separate: `w2_artifact_smoke_pass`
- Fixed replay only: `True`
- Q/R, K, reference, horizon, entry role, controller ID, and variant ID mutation: `False`
- Boundary terminal-useful evidence is retained.

Blocked claims remain W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, and formal ROA guarantees.
