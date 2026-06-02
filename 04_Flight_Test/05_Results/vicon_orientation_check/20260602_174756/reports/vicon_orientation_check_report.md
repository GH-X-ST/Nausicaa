# Vicon Orientation Check Report
- Status: `passed`
- Result root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/04_Flight_Test/05_Results/vicon_orientation_check/20260602_174756`

Expected convention:
- move forward -> x_w increases
- move left -> y_w increases
- move up -> z_w increases
- nose up -> theta positive
- right wing down / left wing up -> phi positive
- nose right -> psi positive
- nose-up motion -> q positive
- right-roll motion -> p positive
- nose-right motion -> r positive

| Step | Signal | Observed | Rate | Confidence | Measured Hz | Pass |
|---|---:|---:|---:|---:|---:|---:|
| `move_forward` | `x_w` | 0.824 m | - | 0.95 | 167.3 | `True` |
| `move_left` | `y_w` | 0.741 m | - | 0.95 | 168.6 | `True` |
| `move_up` | `z_w` | 0.781 m | - | 0.95 | 166.3 | `True` |
| `pitch_up` | `theta` | 14.08 deg | 23.66 deg/s | 0.95 | 165.9 | `True` |
| `pitch_down` | `theta` | -23.61 deg | -38.43 deg/s | 0.95 | 167.9 | `True` |
| `roll_right` | `phi` | 26.26 deg | 40.38 deg/s | 0.95 | 166.0 | `True` |
| `roll_left` | `phi` | -53.58 deg | -54.77 deg/s | 0.95 | 167.6 | `True` |
| `yaw_right` | `psi` | 25.07 deg | 37.33 deg/s | 0.95 | 164.0 | `True` |
| `yaw_left` | `psi` | -42.95 deg | -66.15 deg/s | 0.95 | 168.0 | `True` |
