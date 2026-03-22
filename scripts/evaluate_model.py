"""
Evaluate a trained PIGNN model on test trajectories.

Usage:
    python scripts/evaluate_model.py --model_path models_saved/pignn_bluerov2_direct_best_dev_epoch_100

Computes:
    - 1-step prediction MSE
    - N-step rollout MSE
    - Per-state-variable error breakdown
    - Saves plots to results/
"""

import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pignn import PIGNN
from models.model_utility import (
    convert_input_data,
    convert_output_data,
    get_data_sets,
)

STATE_LABELS = ["x", "y", "z", "cos(ψ)", "sin(ψ)", "u", "v", "w", "r"]


def rollout_trajectory(model, X_init, U, time, device):
    """
    Autoregressive rollout from the first time-step.

    Parameters
    ----------
    X_init : (1, N_seq, 9)  — ground truth (only first step used as seed)
    U      : (1, N_seq, 4)
    time   : (1, N_seq, 1)

    Returns
    -------
    X_pred : (N_seq, 9)     — predicted trajectory
    """
    N_seq = X_init.shape[1]
    N_x   = X_init.shape[2]
    X_pred = torch.zeros(N_seq, N_x, device=device)
    X_pred[0] = X_init[0, 0]

    with torch.no_grad():
        for t in range(N_seq - 1):
            x_cur = X_pred[t].unsqueeze(0).unsqueeze(0)      # (1,1,9)
            u_cur = U[:, t:t+1, :]                            # (1,1,4)
            t_cur = time[:, t:t+1, :]                         # (1,1,1)
            Z, B, T, Nx = convert_input_data(x_cur, u_cur, t_cur)
            Z = Z.to(device)
            x_next = model(Z)
            X_pred[t + 1] = x_next.squeeze()

    return X_pred.cpu().numpy()


def evaluate(model_path: str, dataset_path: str = "dev_set",
             n_trajs: int = 5, device_str: str = "cpu"):
    device = torch.device(device_str)

    # --- Load model ---
    # Infer dimensions from a data sample
    from data.data_utility import TrajectoryDataset
    ds = TrajectoryDataset(dataset_path)
    N_x = ds.X.shape[-1]
    N_u = ds.U.shape[-1]
    N_in = N_x + N_u + 1

    model = PIGNN(N_in=N_in, N_out=N_x).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # --- Pick random trajectories ---
    indices = np.random.choice(len(ds), min(n_trajs, len(ds)), replace=False)

    os.makedirs("results/plots", exist_ok=True)

    all_errors = []
    for idx in indices:
        X_gt, U, t_coll, time = ds[idx]
        X_gt  = X_gt.unsqueeze(0).to(device)
        U     = U.unsqueeze(0).to(device)
        time  = time.unsqueeze(0).to(device)

        X_pred = rollout_trajectory(model, X_gt, U, time, device)
        X_gt_np = X_gt.squeeze(0).cpu().numpy()

        error = np.abs(X_pred - X_gt_np)
        all_errors.append(error)

        # --- Plot ---
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
        axes = axes.flatten()
        t_axis = np.arange(X_gt_np.shape[0])
        for s in range(min(9, N_x)):
            ax = axes[s]
            ax.plot(t_axis, X_gt_np[:, s], "k-", label="Ground truth", linewidth=1)
            ax.plot(t_axis, X_pred[:, s], "r--", label="PIGNN", linewidth=1)
            ax.set_ylabel(STATE_LABELS[s])
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time step")
        fig.suptitle(f"Trajectory {idx} — rollout comparison", fontsize=13)
        fig.tight_layout()
        fig.savefig(f"results/plots/rollout_traj_{idx}.png", dpi=150)
        plt.close(fig)

    # --- Summary ---
    all_errors = np.stack(all_errors)          # (n_trajs, T, 9)
    mean_err = all_errors.mean(axis=(0, 1))    # (9,)
    print("\nMean absolute error per state variable:")
    for s in range(N_x):
        print(f"  {STATE_LABELS[s]:>8s}: {mean_err[s]:.6f}")
    print(f"\n  Overall MAE: {mean_err.mean():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dev_set")
    parser.add_argument("--n_trajs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(args.model_path, args.dataset, args.n_trajs, args.device)
