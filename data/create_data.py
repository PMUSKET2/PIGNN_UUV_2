"""
Generate trajectory datasets for training / validation / testing.

Usage:
    python data/create_data.py

Outputs are saved as .pt files in subdirectories of the current working
directory (training_set/, dev_set/, etc.).
"""

import os
import sys
import numpy as np
import torch
import control as ct
from scipy.stats.qmc import LatinHypercube

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bluerov import bluerov
from data.data_utility import random_input, random_x0

np.random.seed(0)

# Build the nonlinear I/O system once
rov_sys = ct.NonlinearIOSystem(
    bluerov, None,
    inputs=("X", "Y", "Z", "M_z"),
    outputs=("x", "y", "z", "psi", "u", "v", "w", "r"),
    states=("x", "y", "z", "psi", "u", "v", "w", "r"),
    name="bluerov_system",
)


def create_data(
    N_traj, input_type, params=None,
    T_tot=5.2, dt=0.08, N_x=9, N_u=4,
    N_coll=0, fixed_coll_points=None, intervals=None,
):
    if intervals is None:
        intervals = [0.0] * 8
    if fixed_coll_points is None:
        fixed_coll_points = [dt]

    N = int(T_tot / dt)
    N_coll_tot = N_coll + len(fixed_coll_points)
    t = np.linspace(0, T_tot, N, dtype=np.float32)

    U      = np.zeros((N_traj, N, N_u), dtype=np.float32)
    X      = np.zeros((N_traj, N, N_x), dtype=np.float32)
    t_coll = np.zeros((N_traj, N, N_coll_tot), dtype=np.float32)

    if N_coll > 0:
        lhs = LatinHypercube(1, seed=0)

    for n in range(N_traj):
        print(f"  trajectory {n + 1}/{N_traj}", end="\r")
        x_0    = random_x0(intervals)
        U[n, :] = random_input(t, N_u, input_type, params=params)

        if input_type not in ("line", "circle", "figure8"):
            U[n, :, 1] *= 0.1
            U[n, :, 2]  = 5 * np.abs(U[n, :, 2])
            U[n, :, -1] *= 0.05

        _, x = ct.input_output_response(rov_sys, t, U[n, :].T, x_0)
        x = x.T

        X[n, :, :3] = x[:, :3]
        X[n, :, 3]  = np.cos(x[:, 3])
        X[n, :, 4]  = np.sin(x[:, 3])
        X[n, :, 5:] = x[:, 4:]

        for k in range(N):
            if N_coll > 0:
                rand_t = dt * lhs.random(n=N_coll).flatten()
                coll_t = np.sort(np.concatenate((rand_t, fixed_coll_points)))
            else:
                coll_t = np.array(fixed_coll_points)
            t_coll[n, k, :] = coll_t

    print()
    return X, U, t, t_coll


def main():
    dt    = 0.08
    T_tot = 5.2
    N_x, N_u, N_coll = 9, 4, 0

    paths    = ["training_set", "dev_set", "test_set_interp", "test_set_extrap"]
    no_trajs = [400, 1000, 1000, 1000]
    dts      = [dt, dt, dt - 0.02, dt + 0.02]
    T_tots   = [T_tot, T_tot, 3.9, 6.5]

    intervals_dict = {
        "training_set":    [1.0, 1.0, 1.0, np.pi, 1.0, 0.0, 0.1, 0.0],
        "dev_set":         [0.0]*8,
        "test_set_interp": [0.0]*8,
        "test_set_extrap": [0.0]*8,
    }
    input_type_dict = {
        "training_set":    "noise",
        "dev_set":         "sine",
        "test_set_interp": "sine",
        "test_set_extrap": "sine",
    }

    for path, n_traj, dt_, T_tot_ in zip(paths, no_trajs, dts, T_tots):
        print(f"\n=== {path} ===")
        os.makedirs(path, exist_ok=True)
        intervals  = intervals_dict.get(path, [0.0]*8)
        input_type = input_type_dict.get(path, "sine")

        X, U, t, t_coll = create_data(
            N_traj=n_traj, input_type=input_type,
            T_tot=T_tot_, dt=dt_, N_x=N_x, N_u=N_u, N_coll=N_coll,
            fixed_coll_points=[dt_], intervals=intervals,
        )
        torch.save(torch.from_numpy(t),      os.path.join(path, "t.pt"))
        torch.save(torch.from_numpy(U),      os.path.join(path, "U.pt"))
        torch.save(torch.from_numpy(X),      os.path.join(path, "X.pt"))
        torch.save(torch.from_numpy(t_coll), os.path.join(path, "t_coll.pt"))


if __name__ == "__main__":
    main()
