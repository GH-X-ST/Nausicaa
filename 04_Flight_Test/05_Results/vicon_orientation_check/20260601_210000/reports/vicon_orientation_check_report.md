# Vicon Orientation Check Report
- Status: `passed`
- Result root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/04_Flight_Test/05_Results/vicon_orientation_check/20260601_210000`

Expected convention:
- move forward -> x_w increases
- move left -> y_w increases
- move up -> z_w increases
- nose up -> theta positive
- right wing down / left wing up -> phi positive
- nose right -> psi positive

| Step | Signal | Observed | Expected | Pass |
|---|---:|---:|---:|---:|
| `move_forward` | `x_w` | 0.908 m | `+` | `True` |
| `move_left` | `y_w` | 0.936 m | `+` | `True` |
| `move_up` | `z_w` | 0.599 m | `+` | `True` |
| `pitch_up` | `theta` | 42.60 deg | `+` | `True` |
| `pitch_down` | `theta` | -32.53 deg | `-` | `True` |
| `roll_right` | `phi` | 48.33 deg | `+` | `True` |
| `roll_left` | `phi` | -37.29 deg | `-` | `True` |
| `yaw_right` | `psi` | 42.45 deg | `+` | `True` |
| `yaw_left` | `psi` | -26.31 deg | `-` | `True` |
