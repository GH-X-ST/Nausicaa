# Model-Only Foundation Audit

## Glider Model

- State order is `[x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r]`.
- Public world frame is z-up. Body frame is x-forward, y-starboard, z-down.
- Command order is `[delta_a_cmd, delta_e_cmd, delta_r_cmd]`.
- Command signs are preserved by the model and contract tests: positive aileron gives positive roll moment with right wing down, positive elevator gives positive pitch moment nose up, and positive rudder gives positive yaw moment nose right.
- Mass, inertia, and CG are loaded from `A_model_parameters/mass_properties_estimate.py`.
- Trim and finite derivative checks remain part of the retained foundation test set.
- The reset does not certify high-incidence manoeuvre fidelity or any controller performance.

## Latency Model

- Retained latency cases distinguish no latency, actuator lag only, nominal timing, and conservative timing.
- The retained tests cover command limit conversion, step response, labels, and separated mechanism fields.
- Actuator time constants remain explicit in seconds.
- The reset does not claim hardware timing readiness.

## Updraft And Local Flow

- `updraft_models.py` loads measured/fitted updraft model inputs from `01_Thermal/B_results`.
- Wind output uses public z-up world coordinates and m/s units.
- Randomisation shifts and scales the environment instance. It does not create separate online algorithms.
- `wing_wind_descriptors.py` records wing-scale vertical-flow features with signed span convention and unit-labelled columns.

## Runtime And Storage

- Runtime/storage is retained only for chunked, resumable, compressed, worker-enabled, checksum-manifested execution.
- The hard generated-file limit is 100 MB.
- The active repository must not track generated result evidence under `03_Control/05_Results`.
