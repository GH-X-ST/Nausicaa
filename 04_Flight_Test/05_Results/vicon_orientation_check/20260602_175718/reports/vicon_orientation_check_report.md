# Vicon Orientation Check Report
- Status: `failed`
- Result root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/04_Flight_Test/05_Results/vicon_orientation_check/20260602_175718`

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
| `move_forward` | `x_w` | 0.896 m | - | 0.95 | 168.0 | `True` |
| `move_left` | `y_w` | 1.025 m | - | 0.95 | 148.3 | `False` |
| `move_up` | `z_w` | 0.658 m | - | 0.95 | 166.3 | `True` |
| `pitch_up` | `theta` | 41.38 deg | 60.19 deg/s | 0.95 | 169.6 | `True` |
| `pitch_down` | `theta` | -31.75 deg | -57.01 deg/s | 0.95 | 168.9 | `True` |
| `roll_right` | `phi` | 89.26 deg | 99.89 deg/s | 0.95 | 165.7 | `True` |
| `roll_left` | `phi` | -65.88 deg | -97.19 deg/s | 0.95 | 168.6 | `True` |
| `yaw_right` | `psi` | 52.37 deg | 80.83 deg/s | 0.95 | 169.7 | `True` |
| `yaw_left` | `psi` | -48.58 deg | -86.69 deg/s | 0.95 | 169.0 | `True` |
