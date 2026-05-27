# LQR Current-Code Audit Fix v3.4

Date: 2026-05-23

Status: historical audit report. This file is retained for traceability only.
It is not an active project plan, evidence target, or stage-naming contract.
For current method control, use `docs/Glider_Control_Project_Plan.md`.
Older W0/W1/W2/W3 and selected-controller wording below is superseded where it
conflicts with the current R5 -> R7 -> R8 -> R10 -> R11 workflow, with R9 retained only as internal preflight/ablation.

## Fixed mismatches

- W0/W1 tuning hard gates now receive the configured minimum speed explicitly instead of reading a missing rollout-evidence field.
- W0/W1 tuning rows now write as chunked table partitions with chunk manifests, checksums, a table manifest, runtime/coverage/file-size metrics, and selected-controller registry outputs derived from emitted rollout rows.
- Active stale generated files under `03_Control/05_Results/lqr_contextual_v1_0/` were removed. `03_Control/05_Results/.gitkeep` remains the active result placeholder.
- Rollout rows now include `entry_rejection_class` so physical hard failures, speed-gate blocks, controller blocks, and surrogate blocks are not collapsed into one blocked label.
- Blocked-row termination causes are explicit and no longer default to `surrogate_binding_blocked` for controller/registry or physical entry-state failures.
- Boundary-near state sampling now uses canonical `previous_primitive_status = boundary_terminal`; old terminal-use wording is retained only in `state_sample_detail`.
- W2/W3 blocked replay rows preserve replay-only provenance and label surrogate blocks separately from controller-registry verification failures.
- Active guidance text was narrowed so `boundary_terminal` is only previous/source/detail provenance, not an active rollout `outcome_class`.
- The active-contract audit now checks source priority, canonical evidence fields, W-layer surrogate ladder, LQR-only active scope, selected-controller registry provenance, W2/W3 replay-only status, dense-runtime markers, and stale active generated evidence.

## Checks run

- `python -m compileall 03_Control` - blocked before execution: WindowsApps `python.exe` failed with "specified logon session does not exist".
- `python -m pytest -q 03_Control/tests` - blocked before execution by the same `python.exe` launcher failure.
- `python 03_Control/04_Scenarios/run_active_contract_audit.py` - blocked before execution by the same `python.exe` launcher failure.
- `python 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --run-id 934 --rows 32 --candidate-count 2 --paired-tests-per-candidate 1 --candidate-chunk-size 8 --workers 1 --max-workers 1 --output-root .codex_run_logs/v3_4_tuning_smoke` - blocked before execution by the same `python.exe` launcher failure.
- `git diff --check` - passed; Git reported line-ending normalization warnings only.
- Active forbidden-method text scan excluding tests, active audit code, generated results, and retired archives - no matches.
- Active generated-result scan under `03_Control/05_Results` - only `.gitkeep` remains.

## Remaining blocked items

- Python validation requires a working interpreter on PATH or an explicit project environment. `pytest>=7.0` is now listed in both `requirements.txt` and `pyproject.toml`.
- No local smoke output was generated because every Python command failed before entering project code.

## Claim boundary

No dense archive, hardware-readiness, real-flight transfer, W3 robustness, environment-generalisation, mission-success, PD/PID fallback, TVLQR/MPC/LQR-tree, reachable-chain, online fan-layout branch, or claim beyond `simulation_only` was introduced by this pass.
