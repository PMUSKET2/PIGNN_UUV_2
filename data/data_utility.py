"""
Data utilities: trajectory dataset loader and input-signal generators.

The state representation uses cos/sin for yaw:
    X = [x, y, z, cos(ψ), sin(ψ), u, v, w, r]   (dim = 9)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """
    Loads pre-generated trajectory data from .pt files.

    Expected files in *data_dir*:
        X.pt       — (N_traj, N_seq, 9)
        U.pt       — (N_traj, N_seq, 4)
        t_coll.pt  — (N_traj, N_seq, N_coll)
        t.pt       — (N_seq,)
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.X      = torch.load(os.path.join(data_dir, "X.pt"), weights_only=True)
        self.U      = torch.load(os.path.join(data_dir, "U.pt"), weights_only=True)
        self.t_coll = torch.load(os.path.join(data_dir, "t_coll.pt"), weights_only=True)
        time        = torch.load(os.path.join(data_dir, "t.pt"), weights_only=True)

        N_traj, N_seq = self.X.shape[0], self.X.shape[1]
        self.time = time.unsqueeze(0).unsqueeze(-1).expand(N_traj, N_seq, 1)

        assert self.X.shape[0] == self.U.shape[0] == self.t_coll.shape[0]
        assert self.X.shape[1] == self.U.shape[1] == self.t_coll.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.U[idx], self.t_coll[idx], self.time[idx]


# ---------------------------------------------------------------------------
# Input-signal generation
# ---------------------------------------------------------------------------
def random_input(t, N_u, input_type="noise", params=None):
    """Generate control-input signals for trajectory data generation."""
    N = len(t)
    U = np.zeros((N, N_u), dtype=np.float32)
    if params is None:
        params = {}

    if input_type in ("noise", "noise_x"):
        u_signs   = np.random.choice([-1, 1], size=N_u)
        u_offsets = np.random.normal(0, 0.5, size=N_u)
        m_idx     = np.arange(N)
        half_N    = N // 2
        ramp_up   = u_signs * m_idx[:half_N, None] / half_N - 0.5 * u_signs + u_offsets
        ramp_down = u_signs * (N - m_idx[half_N:, None]) / half_N - 0.5 * u_signs + u_offsets
        U_noise   = np.vstack((ramp_up, ramp_down)).astype(np.float32)
        if input_type == "noise":
            U += U_noise
        else:
            U[:, 0] += U_noise[:, 0]

    elif input_type in ("sine", "sine_x"):
        freq  = np.random.uniform(0.01, 0.2, N_u)
        phase = np.random.uniform(0, 2 * np.pi, N_u)
        amp   = 3.0
        U_sine = (amp * np.sin(2 * np.pi * freq * t[:, None] + phase)).astype(np.float32)
        if input_type == "sine":
            U += U_sine
        else:
            U[:, 0] += U_sine[:, 0]

    elif input_type == "line":
        U[:, 0] = params.get("forward_thrust", 5.0)

    elif input_type == "circle":
        U[:, 0] = params.get("forward_thrust", 5.0)
        U[:, 3] = params.get("yaw_moment", 0.5)

    elif input_type == "figure8":
        U[:, 0] = params.get("forward_thrust", 5.0)
        yaw_amp  = params.get("yaw_amplitude", 1.0)
        yaw_freq = params.get("yaw_frequency", 0.2)
        U[:, 3]  = yaw_amp * np.sin(2 * np.pi * yaw_freq * t)

    else:
        raise ValueError(f"Unknown input_type '{input_type}'")

    return U


def random_x0(intervals):
    """Sample a random initial state [x, y, z, ψ, u, v, w, r] (8-dim)."""
    intervals = np.asarray(intervals)
    assert intervals.shape[0] == 8
    p   = np.random.uniform(-intervals[:3], intervals[:3])
    psi = np.random.uniform(-intervals[3], intervals[3])
    v   = np.random.uniform(-intervals[4:7], intervals[4:7])
    v[2] = np.abs(v[2])
    w_r = np.random.uniform(-intervals[7], intervals[7])
    return np.concatenate((p, [psi], v, [w_r]))
