"""
BlueROV2 Heavy physical parameters.

All values are for the 4-DOF (surge, sway, heave, yaw) formulation.
"""

# ---------- Mass & Inertia ----------
m = 11.4        # BlueROV2 mass (kg)
g = 9.82        # gravitational field strength (m/s^2)

F_bouy = 1026 * 0.0115 * g   # Buoyancy force (N)

# ---------- Added Mass ----------
X_ud = -2.6     # Added mass in x (surge) direction (kg)
Y_vd = -18.5    # Added mass in y (sway) direction (kg)
Z_wd = -13.3    # Added mass in z (heave) direction (kg)
K_pd = -0.054   # Added mass for rotation about x (kg)
M_qd = -0.0173  # Added mass for rotation about y (kg)
N_rd = -0.28    # Added mass for rotation about z (yaw) (kg)

# ---------- Moments of Inertia ----------
I_xx = 0.21     # (kg·m²)
I_yy = 0.245    # (kg·m²)
I_zz = 0.245    # (kg·m²)

# ---------- Linear Damping ----------
X_u  = -0.09    # Surge (N·s/m)
Y_v  = -0.26    # Sway (N·s/m)
Z_w  = -0.19    # Heave (N·s/m)
K_p  = -0.895   # Roll (N·s/rad)
M_q  = -0.287   # Pitch (N·s/rad)
N_r  = -4.64    # Yaw (N·s/rad)

# ---------- Quadratic Damping ----------
X_uc = -34.96   # Surge (N·s²/m²)
Y_vc = -103.25  # Sway (N·s²/m²)
Z_wc = -74.23   # Heave (N·s²/m²)
K_pc = -0.084   # Roll (N·s²/rad²)
M_qc = -0.028   # Pitch (N·s²/rad²)
N_rc = -0.43    # Yaw (N·s²/rad²)

# ---------- Geometry ----------
z_b = -0.1      # Distance between CB and CG along the z-axis (m)

# ---------- BlueROV2 Heavy Thruster Configuration ----------
# 8 thrusters: 4 horizontal (vectored), 4 vertical
# Positions are relative to the center of gravity [x, y, z] in meters.
# Orientations are unit thrust direction vectors [fx, fy, fz].
THRUSTER_CONFIG = {
    # Horizontal thrusters (vectored configuration at 45°)
    0: {"position": [ 0.156,  0.111, -0.01], "orientation": [ 0.707,  0.707, 0.0]},
    1: {"position": [ 0.156, -0.111, -0.01], "orientation": [ 0.707, -0.707, 0.0]},
    2: {"position": [-0.156,  0.111, -0.01], "orientation": [-0.707,  0.707, 0.0]},
    3: {"position": [-0.156, -0.111, -0.01], "orientation": [-0.707, -0.707, 0.0]},
    # Vertical thrusters
    4: {"position": [ 0.120,  0.218,  0.0],  "orientation": [0.0, 0.0, 1.0]},
    5: {"position": [ 0.120, -0.218,  0.0],  "orientation": [0.0, 0.0, 1.0]},
    6: {"position": [-0.120,  0.218,  0.0],  "orientation": [0.0, 0.0, 1.0]},
    7: {"position": [-0.120, -0.218,  0.0],  "orientation": [0.0, 0.0, 1.0]},
}
