# Vicon Orientation Check Report
- Status: `failed`
- Result root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/04_Flight_Test/05_Results/vicon_orientation_check/20260602_180537`

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
| `move_forward` | `x_w` | 1.082 m | - | 0.95 | 168.9 | `True` |
| `move_left` | `y_w` | 0.910 m | - | 0.95 | 165.6 | `True` |
| `move_up` | `z_w` | 0.761 m | - | 0.95 | 169.0 | `True` |
| `pitch_up` | `theta` | 38.36 deg | 72.72 deg/s | 0.95 | 168.0 | `True` |
| `pitch_down` | `theta` | -11.77 deg | -70.30 deg/s | 0.95 | 169.6 | `True` |
| `roll_right` | `phi` | 45.02 deg | 96.90 deg/s | 0.95 | 169.3 | `True` |
| `roll_left` | `phi` | -71.94 deg | -110.19 deg/s | 0.95 | 170.3 | `True` |
| `yaw_right` | `psi` | 41.00 deg | 64.94 deg/s | 0.95 | 168.6 | `True` |
| `yaw_left` | `psi` | -61.36 deg | -87.03 deg/s | 0.95 | 148.0 | `False` |
