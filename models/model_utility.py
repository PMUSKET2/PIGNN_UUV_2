"""
Training utilities for the Physics-Informed GNN.

Provides:
    - Input / output conversion helpers (same interface as original PINC)
    - Loss functions: data, physics-residual, initial-condition, rollout
    - Training & dev-set evaluation loops
    - Dataset loading
"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.func import jvp

from data.data_utility import TrajectoryDataset
from src.bluerov_torch import bluerov_compute


# ===================================================================
# Input / output helpers — identical interface to the original PINC
# ===================================================================

def convert_input_data(X: Tensor, U: Tensor, time: Tensor):
    """
    Flatten (B, T, *) trajectory tensors into (B*T, N_in) for the model.

    Returns: Z, N_batch, N_seq, N_x
    """
    N_batch, N_seq, N_x = X.shape
    Z = torch.cat((X, U, time), dim=2)
    N_in = Z.shape[2]
    Z = Z.view(-1, N_in)
    Z.requires_grad_()
    return Z, N_batch, N_seq, N_x


def convert_input_collocation(X: Tensor, U: Tensor, t_coll: Tensor):
    N_coll = t_coll.shape[2]
    X_coll = X.unsqueeze(2).expand(-1, -1, N_coll, -1)
    U_coll = U.unsqueeze(2).expand(-1, -1, N_coll, -1).contiguous()
    Z = torch.cat((X_coll, U_coll, t_coll.unsqueeze(3)), dim=3)
    N_in = Z.shape[3]
    Z = Z.view(-1, N_in)
    Z.requires_grad_()
    return Z, U_coll


def convert_output_data(X_hat, N_batch, N_seq, N_out):
    return X_hat.view(N_batch, N_seq, N_out)


# ===================================================================
# Time-derivative computation (via forward-mode AD)
# ===================================================================

def compute_time_derivatives(Z, N_in, model):
    """
    Compute dX̂/dt using forward-mode AD (jvp) w.r.t. the time input.
    """
    v = torch.zeros_like(Z)
    v[:, N_in - 1] = 1.0
    X_hat, dX_hat_dt = jvp(model, (Z,), (v,))
    return X_hat, dX_hat_dt


# ===================================================================
# Loss functions
# ===================================================================

def compute_physics_loss(X_hat_flat, dX_hat_dt_flat, U_flat):
    """MSE between predicted ẋ and Fossen-equation ẋ."""
    x_dot_physics = bluerov_compute(0, X_hat_flat, U_flat)
    return ((dX_hat_dt_flat - x_dot_physics) ** 2).mean()


def data_loss_fn(model, X, U, time, device, noise_level=0.0):
    """One-step-ahead prediction loss."""
    X_noisy = X[:, :-1, :] + torch.normal(
        0, noise_level, X[:, :-1, :].shape, device=device
    )
    Z, B, T, N_x = convert_input_data(X_noisy, U[:, :-1, :], time[:, :-1, :])
    Z = Z.to(device)
    X_hat = model(Z)
    X_hat = convert_output_data(X_hat, B, T, N_x)
    return mse_loss(X_hat, X[:, 1:])


def initial_condition_loss(model, X, U, time, device):
    """Self-consistency loss at t = 0."""
    Z, B, T, N_x = convert_input_data(X, U, torch.zeros_like(time))
    Z = Z.to(device)
    X_hat = model(Z)
    X_hat = convert_output_data(X_hat, B, T, N_x)
    return mse_loss(X_hat, X)


def physics_loss_fn(model, X, U, t_coll, device, noise_level=0.0):
    """Physics-residual loss evaluated at collocation points."""
    N_x  = X.shape[2]
    N_u  = U.shape[-1]
    N_in = N_x + N_u + 1

    X_noisy = X + torch.normal(0, noise_level, X.shape, device=device)
    Z_coll, U_coll = convert_input_collocation(X_noisy, U, t_coll)
    Z_coll = Z_coll.to(device)
    U_flat = U_coll.to(device).view(-1, N_u)
    X_hat_flat, dX_hat_dt_flat = compute_time_derivatives(Z_coll, N_in, model)
    return compute_physics_loss(X_hat_flat, dX_hat_dt_flat, U_flat)


def rollout_loss_fn(model, X, U, time, N_roll, device, t_coll, pinn,
                    noise_level=0.0):
    """Multi-step rollout loss (+ optional physics loss over rollout)."""
    N_seq = X.shape[1]
    N_seq_slice = N_seq - N_roll
    X_hat = X[:, :N_seq_slice, :]
    l_roll = 0.0
    l_phy  = 0.0

    for i in range(N_roll):
        X_in = X_hat + torch.normal(0, noise_level, X_hat.shape, device=device)
        Z, B, T, N_x = convert_input_data(
            X_in, U[:, i:i + N_seq_slice, :], time[:, i:i + N_seq_slice, :]
        )
        X_hat = model(Z)
        X_hat = convert_output_data(X_hat, B, T, N_x)
        l_roll += mse_loss(X_hat, X[:, i + 1:i + 1 + N_seq_slice])
        if pinn:
            l_phy += physics_loss_fn(
                model, X_hat,
                U[:, i:i + N_seq_slice, :],
                t_coll[:, i:i + N_seq_slice, :],
                device,
            )

    return l_roll / N_roll, l_phy / N_roll


# ===================================================================
# Dataset loading
# ===================================================================

def get_data_sets(N_batch=32,
                  train_path="training_set", dev_path="dev_set",
                  test_1_path="test_set_interp", test_2_path="test_set_extrap"):
    train_dl = DataLoader(TrajectoryDataset(train_path),
                          batch_size=N_batch, shuffle=True)
    dev_dl   = DataLoader(TrajectoryDataset(dev_path),
                          batch_size=N_batch, shuffle=True)
    t1_dl    = DataLoader(TrajectoryDataset(test_1_path),
                          batch_size=N_batch, shuffle=True)
    t2_dl    = DataLoader(TrajectoryDataset(test_2_path),
                          batch_size=N_batch, shuffle=True)
    return train_dl, dev_dl, t1_dl, t2_dl


# ===================================================================
# Training loop (one epoch)
# ===================================================================

def train(
    model: Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    epoch: int,
    device,
    writer: SummaryWriter,
    pinn: bool,
    rollout: bool,
    noise_level: float = 0.0,
    gradient_method: str = "direct",
    n_roll: int = 20,
):
    """
    Train for one epoch.

    gradient_method:
        'direct'    — weighted sum of losses, single backward pass.
        'normalize' — gradient normalisation (scale physics / rollout
                      gradient norms to match data-loss gradient norm).
    n_roll:
        Number of rollout steps (only used when rollout=True).
    """
    model.train()

    epoch_loss_data     = 0.0
    epoch_loss_phys     = 0.0
    epoch_loss_ic       = 0.0
    epoch_loss_roll     = 0.0
    epoch_loss_roll_phy = 0.0
    num_batches         = 0

    for X, U, t_coll, time in train_loader:
        num_batches += 1
        X      = X.to(device)
        U      = U.to(device)
        time   = time.to(device)
        t_coll = t_coll.to(device)

        # --- losses ---
        l_data = data_loss_fn(model, X, U, time, device, noise_level)
        epoch_loss_data += l_data.item()

        l_ic = initial_condition_loss(model, X, U, time, device)
        epoch_loss_ic += l_ic.item()

        l_phy = torch.tensor(0.0, device=device)
        if pinn:
            l_phy = physics_loss_fn(model, X, U, t_coll, device, noise_level)
            epoch_loss_phys += l_phy.item()

        l_roll     = torch.tensor(0.0, device=device)
        l_roll_phy = torch.tensor(0.0, device=device)
        if rollout:
            l_roll, l_roll_phy = rollout_loss_fn(
                model, X, U, time, N_roll=n_roll, device=device,
                t_coll=t_coll, pinn=pinn, noise_level=noise_level,
            )
            epoch_loss_roll += l_roll.item()
            if pinn:
                epoch_loss_roll_phy += l_roll_phy.item()

        # --- gradient step ---
        optimizer.zero_grad()

        if gradient_method == "direct":
            l_total = 1.0 * l_data + 1.0 * l_ic
            if pinn:
                l_total += 0.5 * l_phy
            if rollout:
                l_total += 1.0 * l_roll
                if pinn:
                    l_total += 0.5 * l_roll_phy
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            writer.add_scalar("Loss_train/total", l_total.item(), epoch)

        elif gradient_method == "normalize":
            # Gradient-normalisation approach
            grads = []

            l_data.backward(retain_graph=True)
            grads.append(_get_grad_vec(model))

            if pinn:
                optimizer.zero_grad()
                l_phy.backward(retain_graph=True)
                grads.append(_get_grad_vec(model))

            if rollout:
                optimizer.zero_grad()
                l_roll.backward(retain_graph=True)
                grads.append(_get_grad_vec(model))
                if pinn:
                    optimizer.zero_grad()
                    l_roll_phy.backward(retain_graph=True)
                    grads.append(_get_grad_vec(model))

            combined = _combine_grads_normalised(grads, pinn, rollout)
            _apply_grad_vec(model, combined)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        else:
            raise ValueError(f"Unknown gradient_method: {gradient_method}")

    # --- logging ---
    n = max(num_batches, 1)
    avg_data = epoch_loss_data / n
    writer.add_scalar("Loss_train/data", avg_data, epoch)
    writer.add_scalar("Loss_train/data_log",
                      torch.log10(torch.tensor(avg_data + 1e-12)).item(), epoch)
    writer.add_scalar("Loss_train/ic", epoch_loss_ic / n, epoch)
    if pinn:
        writer.add_scalar("Loss_train/phys", epoch_loss_phys / n, epoch)
    if rollout:
        writer.add_scalar("Loss_train/roll", epoch_loss_roll / n, epoch)
        if pinn:
            writer.add_scalar("Loss_train/roll_phy",
                              epoch_loss_roll_phy / n, epoch)
    writer.flush()
    return avg_data


# ===================================================================
# Validation / dev-set evaluation
# ===================================================================

def test_dev_set(model: Module, loader: DataLoader, epoch: int,
                 device, writer: SummaryWriter):
    model.eval()
    total_data = 0.0
    total_roll = 0.0
    n = 0

    with torch.no_grad():
        for X, U, t_coll, time in loader:
            n += 1
            X, U = X.to(device), U.to(device)
            t_coll, time = t_coll.to(device), time.to(device)

            Z, B, T, N_x = convert_input_data(X, U, time)
            Z = Z.to(device)
            X_hat = model(Z)
            X_hat = convert_output_data(X_hat, B, T, N_x)
            total_data += mse_loss(X_hat[:, :-1], X[:, 1:]).item()

            if epoch % 25 == 0:
                l_roll, _ = rollout_loss_fn(
                    model, X, U, time, N_roll=10, device=device,
                    t_coll=t_coll, pinn=False,
                )
                total_roll += l_roll.item()

    n = max(n, 1)
    avg = total_data / n
    writer.add_scalar("Loss_dev/data", avg, epoch)
    writer.add_scalar("Loss_dev/data_log",
                      torch.log10(torch.tensor(avg + 1e-12)).item(), epoch)
    if epoch % 25 == 0 and total_roll > 0:
        writer.add_scalar("Loss_dev/roll", total_roll / n, epoch)
    writer.flush()
    return avg


# ===================================================================
# Gradient helpers (for 'normalize' method)
# ===================================================================

def _get_grad_vec(model: Module) -> Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros_like(p).flatten())
    return torch.cat(grads)


def _apply_grad_vec(model: Module, vec: Tensor):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        if p.grad is None:
            p.grad = vec[idx:idx + numel].view_as(p).clone()
        else:
            p.grad.copy_(vec[idx:idx + numel].view_as(p))
        idx += numel


def _combine_grads_normalised(grads, pinn, rollout):
    g0_norm = torch.norm(grads[0])
    if pinn and rollout and len(grads) >= 4:
        s1 = g0_norm / (torch.norm(grads[1]) + 1e-12)
        s2 = g0_norm / (torch.norm(grads[2]) + 1e-12)
        s3 = g0_norm / (torch.norm(grads[3]) + 1e-12)
        combined = grads[0] + 0.5 * s1 * grads[1] + s2 * grads[2] + 0.5 * s3 * grads[3]
    elif rollout and len(grads) >= 2:
        s1 = g0_norm / (torch.norm(grads[1]) + 1e-12)
        combined = grads[0] + s1 * grads[1]
    elif pinn and len(grads) >= 2:
        s1 = g0_norm / (torch.norm(grads[1]) + 1e-12)
        combined = grads[0] + 0.5 * s1 * grads[1]
    else:
        combined = grads[0]
    # Re-normalise to data-loss scale
    combined = g0_norm / (torch.norm(combined) + 1e-12) * combined
    return combined
