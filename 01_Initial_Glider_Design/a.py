### Import
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import aerosandbox.tools.units as u
import pandas as pd
import copy
import subprocess
import os


### Gaussian plume model
# 2×2 fan thermal model
# partly hard coded for simplicity
def vertical_velocity_field(Q_v, r_th0, k, r, z, z0, fan_spacing):
    # Q_v         - Vertical volume flux (m^3/s)
    # r_th0       - Core radius at z0 (m)
    # k           - Empirical spreading rate
    # z0          - referemce height for r_th0 (m)
    # fan_spacing - spacing between each fan (m)

    # core radius as function of height
    r_th = r_th0 + k * (z - z0)
    r_th = np.maximum(r_th, 1e-6) # avoid negative radius 

    # peak vertical velocity
    w_th = Q_v / (np.pi * r_th ** 2)

    # fan centres
    fan_centers = [
        (-fan_spacing / 2, -fan_spacing / 2),
        ( fan_spacing / 2, -fan_spacing / 2),
        (-fan_spacing / 2,  fan_spacing / 2),
        ( fan_spacing / 2,  fan_spacing / 2),
    ]

    # total w at a single point (x, y) from all four fans
    def w_at_xy(x, y):
        w_total = 0.0
        for (xc, yc) in fan_centers:
            r_i = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            w_i = w_th * np.exp(-(r_i / r_th) ** 2)
            w_i = np.where(z < z0, 0.0, w_i)
            w_total = w_total + w_i
        return w_total
    
    # sample 4 azimuth angles around the orbit
    thetas = np.array([
        np.pi / 4,
        3 * np.pi / 4,
        5 * np.pi / 4,
        7 * np.pi / 4,
    ])

    # total vertical velocity
    w_sum = 0.0
    for theta in thetas:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        w_sum = w_sum + w_at_xy(x, y)
    
    # take average as assumption for glider operation
    w_avg = w_sum / len(thetas)
    return w_avg

### Setup
# CAMAX30 fan parameters
Q_v      = 1.69
x_center = 4.0
y_center = 2.5

# plume parameters
r_th0 = 0.381 # assume core radius equal to fan radius
k     = 0.10  # typical turbulent plume spreading rate
z0    = 0.50  # reference height at fan centre
fan_spacing = 2 * r_th0 + 0.7

r_target = 1.5

z = 1

# compute average w(r, z)
w = vertical_velocity_field(Q_v = Q_v, r_th0 = r_th0, k = k, r = r_target, z = z, z0 = z0, fan_spacing = fan_spacing,)

print(w)